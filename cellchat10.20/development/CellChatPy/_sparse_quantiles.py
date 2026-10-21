"""Reusable sparse order statistics for label-permutation inference.

Values are sorted once per gene. A changed group assignment only changes which
ordered subsequence each cell contributes to; it does not require sorting the
expression again. Implicit zeros occupy a virtual interval between negative
and positive values. Optional Numba compilation accelerates the exact same
linear traversal, without fast-math or reordered floating-point reductions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from numba import njit
except ImportError:  # Correctness does not depend on the optional compiler.
    njit = None


def _ordered_sparse_statistics(values, cells, indptr, codes, group_sizes, median):
    n_genes = len(indptr) - 1
    n_groups = len(group_sizes)
    output = np.zeros((n_genes, n_groups), dtype=np.float64)
    for gene in range(n_genes):
        start, stop = indptr[gene], indptr[gene + 1]
        stored = np.zeros(n_groups, dtype=np.int64)
        nonnan = np.zeros(n_groups, dtype=np.int64)
        for offset in range(start, stop):
            group = codes[cells[offset]]
            if group < 0:
                continue
            value = values[offset]
            stored[group] += 1
            if not np.isnan(value):
                nonnan[group] += 1

        sizes = group_sizes - stored + nonnan
        zeros = group_sizes - stored
        ranks = np.full((n_groups, 6), -1, dtype=np.int64)
        weights = np.zeros((n_groups, 3), dtype=np.float64)
        endpoints = np.zeros((n_groups, 6), dtype=np.float64)
        for group in range(n_groups):
            size = sizes[group]
            if size <= 0:
                if median:
                    output[gene, group] = np.nan
                continue
            if median:
                ranks[group, 0] = (size - 1) // 2
                ranks[group, 1] = size // 2
            else:
                for quantile in range(3):
                    position = (size - 1) * ((quantile + 1) * 0.25)
                    lower = int(np.floor(position))
                    upper = min(lower + 1, size - 1)
                    ranks[group, 2 * quantile] = lower
                    ranks[group, 2 * quantile + 1] = upper
                    weights[group, quantile] = position - lower

        seen = np.zeros(n_groups, dtype=np.int64)
        for offset in range(start, stop):
            value = values[offset]
            if np.isnan(value):
                continue
            group = codes[cells[offset]]
            if group < 0:
                continue
            rank = seen[group]
            if value >= 0:
                rank += zeros[group]
            seen[group] += 1
            number_of_ranks = 2 if median else 6
            for slot in range(number_of_ranks):
                if rank == ranks[group, slot]:
                    endpoints[group, slot] = value

        for group in range(n_groups):
            if sizes[group] <= 0:
                continue
            if median:
                # np.nanmedian uses np.mean on one/two selected middle values.
                # An odd middle value must not be doubled (overflow matters).
                if sizes[group] % 2:
                    output[gene, group] = endpoints[group, 0]
                else:
                    output[gene, group] = (endpoints[group, 0] + endpoints[group, 1]) / 2.0
            else:
                quantiles = np.zeros(3, dtype=np.float64)
                for quantile in range(3):
                    lower = endpoints[group, 2 * quantile]
                    upper = endpoints[group, 2 * quantile + 1]
                    weight = weights[group, quantile]
                    difference = upper - lower
                    # This branch is NumPy's _lerp, not just an algebraically
                    # equivalent interpolation; permutation ties are strict.
                    if weight >= 0.5:
                        quantiles[quantile] = upper - difference * (1.0 - weight)
                    else:
                        quantiles[quantile] = lower + difference * weight
                output[gene, group] = (quantiles[0] + 2.0 * quantiles[1] + quantiles[2]) / 4.0
    return output


_compiled_sparse_statistics = (
    njit(cache=True, fastmath=False)(_ordered_sparse_statistics)
    if njit is not None else None
)


class SparseQuantilePlan:
    """Pre-sort a genes-by-cells sparse matrix for reusable exact summaries.

    Parameters
    ----------
    data
        A SciPy sparse matrix in genes-by-cells orientation. Duplicate indices
        are summed, stored zeros removed, and the input is never modified.
        Explicit NaNs are omitted per gene/group, matching upstream reducers.
    use_numba
        Compile the linear traversal when Numba is available. The first call
        may include JIT compilation time; report that separately in benchmarks.

    Notes
    -----
    Preprocessing costs one expression sort per gene. Every subsequent average
    uses O(nnz + genes * groups + cells) work and O(genes * groups + cells)
    additional memory, with a constant of six ranks for triMean. The persistent
    plan stores O(nnz + genes) values and cell indices.
    """

    def __init__(self, data, *, use_numba=True):
        if not sparse.issparse(data):
            raise TypeError("SparseQuantilePlan requires a SciPy sparse matrix")
        matrix = data.astype(np.float64, copy=True).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        self.shape = matrix.shape
        self.indptr = np.asarray(matrix.indptr, dtype=np.int64).copy()
        self.values = np.empty(matrix.nnz, dtype=np.float64)
        self.cells = np.empty(matrix.nnz, dtype=np.int64)
        for gene in range(matrix.shape[0]):
            start, stop = matrix.indptr[gene:gene + 2]
            order = np.argsort(matrix.data[start:stop], kind="stable")
            self.values[start:stop] = matrix.data[start:stop][order]
            self.cells[start:stop] = matrix.indices[start:stop][order]
        self._kernel = (_compiled_sparse_statistics if use_numba and
                        _compiled_sparse_statistics is not None else
                        _ordered_sparse_statistics)
        self.backend = "numba" if self._kernel is _compiled_sparse_statistics else "python"
        self._categories = None
        self._category_index = None

    def average(self, labels, categories, type_method="triMean"):
        if type_method not in {"triMean", "median"}:
            raise ValueError("SparseQuantilePlan supports only triMean and median")
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != self.shape[1]:
            raise ValueError("labels must contain one group per expression column")
        categories = tuple(categories)
        if categories != self._categories:
            index = pd.Index(categories)
            if not index.is_unique:
                raise ValueError("categories must be unique")
            self._categories = categories
            self._category_index = index
        codes = np.asarray(self._category_index.get_indexer(labels), dtype=np.int64)
        # The reference ignores missing labels in each group's summary but
        # keeps their cells in the complete label-permutation population.
        missing = codes < 0
        if np.any(missing):
            if not np.all(pd.isna(labels[missing])):
                raise ValueError("every label must occur in categories")
            counted_codes = codes[~missing]
        else:
            counted_codes = codes
        group_sizes = np.bincount(counted_codes, minlength=len(categories)).astype(np.int64)
        return self._kernel(self.values, self.cells, self.indptr, codes,
                            group_sizes, type_method == "median")
