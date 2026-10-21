"""Exact tests of the reusable sparse order-statistic plan."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "development"))

from CellChatPy._sparse_quantiles import SparseQuantilePlan


def reference(matrix, labels, categories, method):
    output = np.zeros((matrix.shape[0], len(categories)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for gene, row in enumerate(matrix):
            for group, category in enumerate(categories):
                selected = row[labels == category]
                selected = selected[~np.isnan(selected)]
                if method == "median":
                    output[gene, group] = np.median(selected)
                elif len(selected):
                    q1 = np.quantile(selected, 0.25)
                    q2 = np.quantile(selected, 0.5)
                    q3 = np.quantile(selected, 0.75)
                    output[gene, group] = (q1 + 2 * q2 + q3) / 4
    return output


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("method", ["triMean", "median"])
@pytest.mark.parametrize("density", [0.05, 0.3, 1.0])
@pytest.mark.parametrize("seed", [3, 47, 1729])
def test_permuted_sparse_quantiles_exact(use_numba, method, density, seed):
    rng = np.random.RandomState(seed)
    matrix = rng.normal(size=(9, 41))
    matrix[rng.uniform(size=matrix.shape) > density] = 0.0
    matrix[rng.uniform(size=matrix.shape) < 0.08] = np.nan
    matrix[0] = 0.0
    matrix[1] = np.nan
    plan = SparseQuantilePlan(sparse.csr_matrix(matrix), use_numba=use_numba)
    labels = np.array(["B"] * 15 + ["A"] * 26)
    categories = ["A", "unused", "B"]
    for iteration in range(5):
        permuted_labels = rng.permutation(labels)
        actual = plan.average(permuted_labels, categories, method)
        expected = reference(matrix, permuted_labels, categories, method)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_cells", range(1, 33))
@pytest.mark.parametrize("method", ["triMean", "median"])
def test_every_small_interpolation_weight(n_cells, method):
    rng = np.random.RandomState(n_cells)
    matrix = rng.lognormal(-0.7, 3, (6, n_cells))
    matrix[rng.uniform(size=matrix.shape) < 0.4] = 0
    labels = np.zeros(n_cells, dtype=int)
    plan = SparseQuantilePlan(sparse.csr_matrix(matrix))
    np.testing.assert_array_equal(plan.average(labels, [0], method),
                                  reference(matrix, labels, [0], method))


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("method", ["triMean", "median"])
def test_duplicate_entries_and_explicit_zeros_do_not_mutate_input(use_numba, method):
    matrix = sparse.csr_matrix((np.array([0.0, 0.2, 0.3, -0.1, 0.4, 0.0, 0.9]),
                                np.array([0, 1, 1, 3, 3, 0, 4]),
                                np.array([0, 5, 7])), shape=(2, 6))
    original = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
    labels = np.array([0, 0, 0, 1, 1, 1])
    plan = SparseQuantilePlan(matrix, use_numba=use_numba)
    np.testing.assert_array_equal(plan.average(labels, [0, 1], method),
                                  reference(matrix.toarray(), labels, [0, 1], method))
    for left, right in zip(original, (matrix.data, matrix.indices, matrix.indptr)):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("method", ["triMean", "median"])
def test_infinities_and_overflow_follow_numpy(use_numba, method):
    matrix = np.array([[1, 2, 3, np.inf], [-np.inf, 0, 1, 2],
                       [-np.inf, -np.inf, np.inf, np.inf],
                       [1.6e308, 1.7e308, 1.7e308, 1.7e308],
                       [np.nan, 0, 0, np.inf]])
    labels = np.array([0, 0, 0, 0])
    plan = SparseQuantilePlan(sparse.csr_matrix(matrix), use_numba=use_numba)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = plan.average(labels, [0], method)
    np.testing.assert_array_equal(actual, reference(matrix, labels, [0], method))


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(TypeError, match="sparse"):
        SparseQuantilePlan(np.ones((2, 3)))
    plan = SparseQuantilePlan(sparse.csr_matrix(np.ones((2, 3))))
    with pytest.raises(ValueError, match="triMean and median"):
        plan.average([0, 0, 1], [0, 1], "truncatedMean")
    with pytest.raises(ValueError, match="one group"):
        plan.average([0, 0], [0])
    with pytest.raises(ValueError, match="every label"):
        plan.average([0, 0, 1], [0])
    with pytest.raises(ValueError, match="unique"):
        plan.average([0, 0, 1], [0, 0, 1])


@pytest.mark.parametrize("use_numba", [False, True])
@pytest.mark.parametrize("method", ["triMean", "median"])
def test_missing_group_labels_are_excluded_without_dropping_cells(use_numba, method):
    matrix = np.array([[0, 1, 1000, 2, 0, 2000, 4, 5],
                       [3, 0, 0, 7, np.nan, 3000, 0, 9]], dtype=float)
    labels = np.array([0, 0, np.nan, 1, 1, np.nan, 0, 1])
    plan = SparseQuantilePlan(sparse.csr_matrix(matrix), use_numba=use_numba)
    assignments = [labels, labels[[4, 3, 2, 5, 0, 1, 6, 7]], np.full(8, np.nan)]
    for assignment in assignments:
        actual = plan.average(assignment, [0, 1, 2], method)
        expected = reference(matrix, assignment, [0, 1, 2], method)
        np.testing.assert_array_equal(actual, expected)
