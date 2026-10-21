"""Memory-bounded execution of the published CellChatPy group-level model.

The statistic, permutation distribution, strict exceedance test and operation
ordering are unchanged. Only execution and storage are reorganized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats


def expression_matrix(cellchat, raw_use=True):
    """Return genes x cells without the upstream sparse->pandas->dense path."""
    layer = 'signaling' if raw_use else 'smoothed'
    matrix = getattr(cellchat, layer)
    if matrix is None:
        raise ValueError(f"Expression layer {layer!r} is not available.")
    if matrix.shape != cellchat.shape:
        raise ValueError(f"Expression layer {layer!r} must have cells x genes shape {cellchat.shape}; got {matrix.shape}.")
    mask = np.ones(cellchat.n_vars, dtype=bool)
    if raw_use and 'is_signaling' in cellchat.var:
        mask = cellchat.var['is_signaling'].fillna(False).to_numpy(dtype=bool)
    names = list(cellchat.var_names[mask].astype(str))
    matrix = matrix[:, mask]
    if sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix.T, dtype=np.float64)
        matrix.sum_duplicates()
        matrix.sort_indices()
    else:
        matrix = np.asarray(matrix, dtype=np.float64).T
    return matrix, names


def required_gene_indices(names, pairs, complexes, cofactors):
    """Static dependency closure, applied AFTER the global normalization max."""
    index = {g: i for i, g in enumerate(names)}
    required = set()
    subcols = [c for c in complexes.columns if c.startswith('subunit')]
    cocols = [c for c in cofactors.columns if 'cofactor' in c.lower()]
    for column in ('ligand', 'receptor'):
        for name in pairs[column].astype(str):
            if name in index:
                required.add(index[name])
            elif name in complexes.index:
                required.update(index[g] for g in complexes.loc[name, subcols]
                                if isinstance(g, str) and g in index)
    for column in ('co_A_receptor', 'co_I_receptor', 'agonist', 'antagonist'):
        if column not in pairs:
            continue
        for name in pairs[column]:
            if pd.notna(name) and name != '' and name in cofactors.index:
                required.update(index[g] for g in cofactors.loc[name, cocols]
                                if isinstance(g, str) and g in index)
    return np.array(sorted(required), dtype=np.intp)


class GroupAverager:
    """Reuse immutable sparse order statistics across label permutations."""
    def __init__(self, data, categories, type_method, trim=0.1, gene_batch_size=64):
        self.data = data
        self.categories = list(categories)
        self.type_method = type_method
        self.trim = trim
        self.gene_batch_size = gene_batch_size
        self.plan = None
        if sparse.issparse(data) and type_method not in {'truncatedMean', 'thresholdedMean'}:
            from ._sparse_quantiles import SparseQuantilePlan
            self.plan = SparseQuantilePlan(data)

    def __call__(self, labels):
        if self.plan is not None:
            method = 'median' if self.type_method == 'median' else 'triMean'
            return self.plan.average(labels, self.categories, type_method=method)
        return _batched_group_average(self.data, labels, self.categories,
                                      self.type_method, self.trim, self.gene_batch_size)


def group_average(data, labels, clusters, type_method, trim=0.1, gene_batch_size=64):
    return GroupAverager(data, clusters, type_method, trim, gene_batch_size)(labels)


def _batched_group_average(data, labels, clusters, type_method, trim, gene_batch_size):
    """Bound dense scratch space; preserve scalar reduction for mean methods."""
    labels = np.asarray(labels)
    out = np.zeros((data.shape[0], len(clusters)), dtype=np.float64)
    for ci, cluster in enumerate(clusters):
        positions = np.flatnonzero(labels == cluster)
        if not len(positions):
            if type_method == 'median':
                out[:, ci] = np.nan
            continue
        for start in range(0, data.shape[0], gene_batch_size):
            stop = min(start + gene_batch_size, data.shape[0])
            values = data[start:stop, :][:, positions]
            if sparse.issparse(values):
                values = values.toarray()
            values = np.ascontiguousarray(values, dtype=np.float64)
            if type_method in {'truncatedMean', 'thresholdedMean'}:
                # The scalar partition/reduction order matters at P_boot=P_obs.
                for j, row in enumerate(values):
                    row = row[~np.isnan(row)]
                    if len(row) == 0:
                        continue
                    if type_method == 'truncatedMean':
                        out[start + j, ci] = stats.trim_mean(row, trim)
                    elif np.count_nonzero(row) / len(row) >= trim:
                        out[start + j, ci] = np.mean(row)
            elif type_method == 'median':
                out[start:stop, ci] = np.nanmedian(values, axis=1)
            else:
                quantiles = np.nanquantile(values, (0.25, 0.5, 0.75), axis=1)
                result = (quantiles[0] + 2 * quantiles[1] + quantiles[2]) / 4.0
                result[np.all(np.isnan(values), axis=1)] = 0.0
                out[start:stop, ci] = result
    return out


class InteractionPlan:
    """Compile DB lookups once and reuse unique complex/cofactor expressions."""
    def __init__(self, names, pairs, complexes, cofactors, kh, n, use_agan):
        index = {g: i for i, g in enumerate(names)}
        subcols = [c for c in complexes.columns if c.startswith('subunit')]
        cocols = [c for c in cofactors.columns if 'cofactor' in c.lower()]
        unique = list(dict.fromkeys(list(pairs['ligand'].astype(str)) + list(pairs['receptor'].astype(str))))
        uindex = {g: i for i, g in enumerate(unique)}
        self.entities = []
        for name in unique:
            if name in index:
                self.entities.append(('gene', [index[name]]))
            elif name in complexes.index:
                ids = [index[g] for g in complexes.loc[name, subcols]
                       if isinstance(g, str) and g != '' and g in index]
                self.entities.append(('complex', ids))
            else:
                self.entities.append(('missing', []))
        self.ligands = np.array([uindex[g] for g in pairs['ligand'].astype(str)], dtype=np.intp)
        self.receptors = np.array([uindex[g] for g in pairs['receptor'].astype(str)], dtype=np.intp)
        self.cofactors = {}
        self.co_maps = {}
        for column in ('co_A_receptor', 'co_I_receptor', 'agonist', 'antagonist'):
            mapping = []
            for name in pairs.get(column, pd.Series([''] * len(pairs))):
                ids = ()
                if pd.notna(name) and name != '' and name in cofactors.index:
                    ids = tuple(index[g] for g in cofactors.loc[name, cocols]
                                if isinstance(g, str) and g != '' and g in index)
                mapping.append(ids)
            self.co_maps[column] = mapping
        self.kh, self.n, self.use_agan = kh, n, use_agan

    def resolve(self, avg):
        k = avg.shape[1]
        values = np.zeros((len(self.entities), k))
        for i, (kind, ids) in enumerate(self.entities):
            if kind == 'gene':
                values[i] = avg[ids[0]]
            elif ids:
                x = avg[ids]
                with np.errstate(divide='ignore', invalid='ignore'):
                    logs = np.where(np.isnan(x), np.nan, np.log(x))
                    values[i] = np.exp(np.nanmean(logs, axis=0))
        ligand = values[self.ligands]
        receptor = values[self.receptors]
        effects = {}
        for column, mapping in self.co_maps.items():
            result = np.ones((len(mapping), k))
            if column in {'agonist', 'antagonist'} and not self.use_agan:
                effects[column] = result
                continue
            cache = {}
            for i, ids in enumerate(mapping):
                if not ids:
                    continue
                if ids not in cache:
                    product = np.ones(k)
                    for j in ids:
                        row = avg[j]
                        if column == 'agonist':
                            value = 1.0 + row**self.n / (self.kh**self.n + row**self.n)
                        elif column == 'antagonist':
                            value = self.kh**self.n / (self.kh**self.n + row**self.n)
                        else:
                            value = 1 + row
                        product = product * value
                    cache[ids] = product
                result[i] = cache[ids]
            effects[column] = result
        inhib = effects['co_I_receptor']
        inhib[inhib == 0] = 1.0
        receptor = receptor * effects['co_A_receptor'] / inhib
        return ligand, receptor, effects['agonist'], effects['antagonist']


def _probability_block(resolved, start, stop, kh, n, population, spatial, contact, contact_mask):
    ligand, receptor, agonist, antagonist = [x[start:stop] for x in resolved]
    lr = ligand[:, :, None] * receptor[:, None, :]
    p1 = lr**n / (kh**n + lr**n)
    p2 = agonist[:, :, None] * agonist[:, None, :]
    p3 = antagonist[:, :, None] * antagonist[:, None, :]
    weight = np.where(contact_mask[start:stop, None, None], (spatial * contact)[None], spatial[None])
    # Preserve the published left-associative multiplication sequence.
    probability = p1 * p2 * p3 * population * weight
    active = np.sum(p1 * weight, axis=(1, 2)) != 0
    return probability, active


def infer_streaming(data, averager, labels, categories, pairs, complexes, cofactors,
                    names, observed_average, *, n_boot, seed, permutation_use,
                    kh, n, use_agan, population_size, spatial, contact,
                    contact_mask, lr_batch_size):
    """Accumulate only exceedance counts; no N*B or B*genes*groups storage."""
    nc, k, m = data.shape[1], len(categories), len(pairs)
    supplied = None
    if permutation_use is not None:
        supplied = np.asarray(permutation_use)
        if supplied.ndim != 2 or supplied.shape[0] != nc or supplied.shape[1] == 0:
            raise ValueError('permutation_use must have shape (n_cells, n_boot), with n_boot > 0')
        n_boot = supplied.shape[1]
    elif not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError('n_boot must be a positive integer')
    plan = InteractionPlan(names, pairs, complexes, cofactors, kh, n, use_agan)
    fractions = np.array([np.sum(labels == c) / nc for c in categories])
    population = np.outer(fractions, fractions) if population_size else np.ones((k, k))
    observed = np.zeros((m, k, k))
    exceedances = np.zeros((m, k, k), dtype=np.int64)
    active = np.ones(m, dtype=bool)
    resolved = plan.resolve(observed_average)
    for start in range(0, m, lr_batch_size):
        stop = min(start + lr_batch_size, m)
        value, enabled = _probability_block(resolved, start, stop, kh, n, population, spatial, contact, contact_mask)
        value[~enabled] = 0.0
        observed[start:stop] = value
        active[start:stop] = enabled
    # Deliberately use the same global RNG as upstream: subsequent tutorial
    # computations may depend on its post-inference state as well as the draws.
    for b in range(n_boot):
        if supplied is None:
            order = np.random.permutation(nc)
        else:
            column = supplied[:, b]
            if np.issubdtype(column.dtype, np.integer):
                order = column.astype(np.intp, copy=False)
            else:
                # Validate one column at a time, including integer-valued
                # float inputs, without N*n_boot conversion or boolean masks.
                if not np.all(np.isfinite(column)):
                    raise ValueError('permutation_use must contain integer indices')
                order = column.astype(np.intp)
                if not np.all(column == order):
                    raise ValueError('permutation_use must contain integer indices')
            if np.any(order < 0) or np.any(order >= nc) or np.unique(order).size != nc:
                raise ValueError('every permutation_use column must be a permutation of all cell indices')
        avg = averager(labels[order])
        resolved = plan.resolve(avg)
        for start in range(0, m, lr_batch_size):
            stop = min(start + lr_batch_size, m)
            value, _ = _probability_block(resolved, start, stop, kh, n, population, spatial, contact, contact_mask)
            exceedances[start:stop] += (value - observed[start:stop]) > 0
    pval = exceedances / n_boot
    pval[~active] = 1.0
    pval[observed == 0] = 1.0
    return np.moveaxis(observed, 0, 2), np.moveaxis(pval, 0, 2), n_boot
