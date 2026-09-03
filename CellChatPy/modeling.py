#!/usr/bin/env python3
"""Cell-cell communication inference functions for CellChatPy."""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import cKDTree
from collections.abc import Sequence
from scipy.spatial.distance import pdist, squareform
from typing import Union, Optional, Dict, List, Tuple
import warnings

from .network_storage import (
    matrix_dict_from_array,
    network_names,
    stack_network_field,
    zero_group_axes,
)
from .analysis import _feature_by_cell_frame


# ---------------------------------------------------------------------------
# Helper: triMean  (Tukey trimean: mean of Q1, Q2, Q2, Q3)
# ---------------------------------------------------------------------------
def _tri_mean(x: np.ndarray) -> float:
    """Tukey's trimean: (Q1 + 2*median + Q3) / 4"""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    q1 = np.quantile(x, 0.25)
    q2 = np.quantile(x, 0.50)
    q3 = np.quantile(x, 0.75)
    return (q1 + 2 * q2 + q3) / 4.0


def _truncated_mean(x: np.ndarray, trim: float = 0.1) -> float:
    """Truncated mean (same as R mean(x, trim=trim))"""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    return stats.trim_mean(x, trim)


def _thresholded_mean(x: np.ndarray, trim: float = 0.1) -> float:
    """Return mean only if fraction of expressing cells >= trim, else 0"""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    percent = np.count_nonzero(x) / len(x)
    if percent < trim:
        return 0.0
    return float(np.mean(x))


def _geometric_mean(x: np.ndarray) -> float:
    """Geometric mean matching R: exp(mean(log(x), na.rm=TRUE)).
    R does NOT pre-replace zeros; log(0)=-Inf propagates through mean,
    so any zero subunit collapses the complex expression to 0.
    Only NA values are excluded (na.rm=TRUE).
    """
    x = x.astype(float)
    # Replace NaN with np.nan so nanmean skips them (mirrors R na.rm=TRUE)
    # Do NOT replace zeros; R keeps zeros and log(0)=-Inf drives result to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_vals = np.where(np.isnan(x), np.nan, np.log(x))
    if np.all(np.isnan(log_vals)):
        return 0.0
    return float(np.exp(np.nanmean(log_vals)))


def _geometric_mean_matrix_cols(mat: np.ndarray) -> np.ndarray:
    """Geometric mean per column (axis=0), matching R geometricMean for matrix input.
    R: exp(apply(log(x), 2, mean, na.rm=TRUE))
    Zeros produce log(0)=-Inf; the column mean becomes -Inf; exp(-Inf)=0.
    """
    mat = mat.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_mat = np.where(np.isnan(mat), np.nan, np.log(mat))
    return np.exp(np.nanmean(log_mat, axis=0))


# ---------------------------------------------------------------------------
# computeExpr_LR  - mirrors R computeExpr_LR
# ---------------------------------------------------------------------------
def _compute_expr_lr(
    gene_lr: list,          # list of gene/complex names, length nLR
    data_avg: np.ndarray,   # shape (n_genes, n_clusters); rows indexed by gene_names
    gene_names: list,       # row labels of data_avg
    complex_input: pd.DataFrame  # index = complex names, columns = subunit_*
) -> np.ndarray:
    """
    For each entry in gene_lr:
      - If it is a single gene present in data_avg -> use that row
      - If it is a complex name -> geometric mean of its subunits
      - Otherwise -> row of zeros

    Returns array of shape (nLR, n_clusters).
    """
    nLR = len(gene_lr)
    n_clusters = data_avg.shape[1]
    result = np.zeros((nLR, n_clusters))

    gene_index = {g: i for i, g in enumerate(gene_names)}

    for i, gene in enumerate(gene_lr):
        if gene in gene_index:
            result[i, :] = data_avg[gene_index[gene], :]
        elif len(complex_input) > 0 and gene in complex_input.index:
            # Get subunit columns
            sub_cols = [c for c in complex_input.columns if c.startswith('subunit')]
            subunits = [v for v in complex_input.loc[gene, sub_cols] if isinstance(v, str) and v != '']
            # Geometric mean of subunit expression per cluster
            sub_data = []
            for s in subunits:
                if s in gene_index:
                    sub_data.append(data_avg[gene_index[s], :])
            if len(sub_data) > 0:
                sub_mat = np.vstack(sub_data)  # shape (n_subunits, n_clusters)
                result[i, :] = _geometric_mean_matrix_cols(sub_mat)
            # else remains 0
        # else remains 0

    return result


# ---------------------------------------------------------------------------
# computeExpr_coreceptor  - mirrors R computeExpr_coreceptor
# ---------------------------------------------------------------------------
def _compute_expr_coreceptor(
    cofactor_input: pd.DataFrame,
    data_avg: np.ndarray,
    gene_names: list,
    pair_lr_sig: pd.DataFrame,
    cotype: str  # "A" or "I"
) -> np.ndarray:
    """
    Returns array of shape (nLR, n_clusters) representing co-receptor effect.
    type="A" -> co-activation (multiply receptor expression by (1 + expr))
    type="I" -> co-inhibition (divide receptor expression by (1 + expr))
    Default value (no coreceptor) is 1.
    """
    nLR = len(pair_lr_sig)
    n_clusters = data_avg.shape[1]
    result = np.ones((nLR, n_clusters))

    if len(cofactor_input) == 0:
        return result

    if cotype == "A":
        col = 'co_A_receptor'
    else:
        col = 'co_I_receptor'

    if col not in pair_lr_sig.columns:
        return result

    gene_index = {g: i for i, g in enumerate(gene_names)}
    cofactor_cols = [c for c in cofactor_input.columns if 'cofactor' in c.lower()]

    for i, (_, row) in enumerate(pair_lr_sig.iterrows()):
        coreceptor_name = row.get(col, '')
        if pd.isna(coreceptor_name) or coreceptor_name == '':
            continue
        if coreceptor_name not in cofactor_input.index:
            continue
        cofactor_genes = [v for v in cofactor_input.loc[coreceptor_name, cofactor_cols]
                          if isinstance(v, str) and v != '']
        cofactor_genes = [g for g in cofactor_genes if g in gene_index]
        if len(cofactor_genes) == 1:
            expr = data_avg[gene_index[cofactor_genes[0]], :]
            result[i, :] = 1 + expr
        elif len(cofactor_genes) > 1:
            product = np.ones(n_clusters)
            for g in cofactor_genes:
                product = product * (1 + data_avg[gene_index[g], :])
            result[i, :] = product
        # else stays 1

    return result


# ---------------------------------------------------------------------------
# computeExpr_agonist / antagonist  - mirrors R computeExpr_agonist/antagonist
# ---------------------------------------------------------------------------
def _compute_expr_agonist(
    data_avg: np.ndarray,
    gene_names: list,
    pair_lr_sig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_agonist: int,
    kh: float,
    n: float
) -> np.ndarray:
    """Returns 1-D array of length n_clusters (one value per cluster)"""
    n_clusters = data_avg.shape[1]
    gene_index = {g: i for i, g in enumerate(gene_names)}
    cofactor_cols = [c for c in cofactor_input.columns if 'cofactor' in c.lower()]

    agonist_name = pair_lr_sig.iloc[index_agonist].get('agonist', '')
    if pd.isna(agonist_name) or agonist_name == '' or agonist_name not in cofactor_input.index:
        return np.ones(n_clusters)

    agonist_genes = [v for v in cofactor_input.loc[agonist_name, cofactor_cols]
                     if isinstance(v, str) and v != '']
    agonist_genes = [g for g in agonist_genes if g in gene_index]

    if len(agonist_genes) == 1:
        avg = data_avg[gene_index[agonist_genes[0]], :]
        return 1.0 + avg**n / (kh**n + avg**n)
    elif len(agonist_genes) > 1:
        product = np.ones(n_clusters)
        for g in agonist_genes:
            avg = data_avg[gene_index[g], :]
            product = product * (1.0 + avg**n / (kh**n + avg**n))
        return product
    return np.ones(n_clusters)


def _compute_expr_antagonist(
    data_avg: np.ndarray,
    gene_names: list,
    pair_lr_sig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_antagonist: int,
    kh: float,
    n: float
) -> np.ndarray:
    n_clusters = data_avg.shape[1]
    gene_index = {g: i for i, g in enumerate(gene_names)}
    cofactor_cols = [c for c in cofactor_input.columns if 'cofactor' in c.lower()]

    antagonist_name = pair_lr_sig.iloc[index_antagonist].get('antagonist', '')
    if pd.isna(antagonist_name) or antagonist_name == '' or antagonist_name not in cofactor_input.index:
        return np.ones(n_clusters)

    antagonist_genes = [v for v in cofactor_input.loc[antagonist_name, cofactor_cols]
                        if isinstance(v, str) and v != '']
    antagonist_genes = [g for g in antagonist_genes if g in gene_index]

    if len(antagonist_genes) == 1:
        avg = data_avg[gene_index[antagonist_genes[0]], :]
        return kh**n / (kh**n + avg**n)
    elif len(antagonist_genes) > 1:
        product = np.ones(n_clusters)
        for g in antagonist_genes:
            avg = data_avg[gene_index[g], :]
            product = product * (kh**n / (kh**n + avg**n))
        return product
    return np.ones(n_clusters)


# ---------------------------------------------------------------------------
# Main: compute_communication_probability
# ---------------------------------------------------------------------------
def compute_communication_probability(
    cellchat,
    type_method: str = "triMean",
    trim: float = 0.1,
    lr_use=None,
    raw_use: bool = True,
    population_size: bool = False,
    distance_use: bool = True,
    interaction_range: float = 250,
    scale_distance: float = 0.01,
    k_min: int = 10,
    contact_dependent: bool = True,
    contact_range=10.0,
    contact_knn_k=None,
    contact_dependent_forced: bool = False,
    do_symmetric: bool = True,
    n_boot: int = 100,
    seed_use: int = 1,
    kh: float = 0.5,
    n_param: float = 1,
    use_agan: bool = True,
    tol: Optional[float] = None,
    permutation_use=None,
    **kwargs
):
    """
    Compute group-level ligand-receptor communication probabilities.

    Parameters (additional)
    -----------------------
    permutation_use : np.ndarray, optional
        Pre-computed permutation matrix of shape (n_cells, n_boot), 0-based integer indices.
        When provided, overrides seed_use and n_boot (n_boot is inferred from columns).
        Pass R-exported permutations (converted to 0-based) for exact cross-language reproducibility.
    """
    if 'type' in kwargs:
        type_method = kwargs.pop('type')
    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise TypeError(f"compute_communication_probability() got unexpected keyword argument(s): {unknown}")

    print(f"{type_method} is used for calculating the average gene expression per cell group.")

    np.random.seed(seed_use)

    # ----- Select averaging function -----
    if type_method == "triMean":
        fun_mean = _tri_mean
    elif type_method == "truncatedMean":
        fun_mean = lambda x: _truncated_mean(x, trim)
    elif type_method == "thresholdedMean":
        fun_mean = lambda x: _thresholded_mean(x, trim)
    elif type_method == "median":
        fun_mean = lambda x: float(np.nanmedian(x))
    else:
        fun_mean = _tri_mean

    # ----- Get expression data -----
    if raw_use:
        data = _feature_by_cell_frame(cellchat, 'signaling')
    else:
        data = _feature_by_cell_frame(cellchat, 'smoothed')

    if data is None:
        raise ValueError("No signaling expression data available. Run subset_signaling_data first.")

    if any(isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes):
        if any(
            isinstance(dtype, pd.SparseDtype)
            and pd.isna(dtype.fill_value)
            for dtype in data.dtypes
        ):
            data = data.fillna(0)
        data = data.sparse.to_dense()

    gene_names = list(data.index)
    n_genes = len(gene_names)

    # ----- LR pairs -----
    if lr_use is None:
        if ('significant' in cellchat.lr_pairs
                and cellchat.lr_pairs['significant'] is not None
                and len(cellchat.lr_pairs['significant']) > 0):
            pair_lr_sig = cellchat.lr_pairs['significant'].copy().reset_index(drop=True)
        else:
            raise ValueError("No LR pairs. Run identify_overexpressed_interactions first.")
    else:
        pair_lr_sig = lr_use.copy().reset_index(drop=True)

    # ----- DB accessors -----
    database = cellchat.database
    complex_input = database.get('complex', pd.DataFrame())
    cofactor_input = database.get('cofactor', pd.DataFrame())

    # ----- Cell groups -----
    groups = cellchat.groups
    unique_clusters = list(groups.categories)
    n_clusters = len(unique_clusters)
    n_C = data.shape[1]   # total cells

    # ----- Normalize data: data.use <- data / max(data)  (R line 111) -----
    data_matrix = data.values.astype(float)
    max_val = data_matrix.max()
    if max_val > 0:
        data_use = data_matrix / max_val
    else:
        data_use = data_matrix.copy()

    # ----- Compute average expression per group -----
    # R: data.use.avg <- aggregate(t(data.use), list(group), FUN = FunMean)
    # Result shape: (n_genes, n_clusters)
    data_use_avg = np.zeros((n_genes, n_clusters))
    group_values = np.array(groups)
    for ci, cluster in enumerate(unique_clusters):
        mask = (group_values == cluster)
        cluster_data = data_use[:, mask]   # (n_genes, n_cells_in_cluster)
        for gi in range(n_genes):
            data_use_avg[gi, ci] = fun_mean(cluster_data[gi, :])

    # ----- Compute ligand / receptor average expression -----
    gene_L = list(pair_lr_sig['ligand'].astype(str))
    gene_R = list(pair_lr_sig['receptor'].astype(str))
    n_LR = len(pair_lr_sig)

    data_L_avg = _compute_expr_lr(gene_L, data_use_avg, gene_names, complex_input)  # (nLR, nCluster)
    data_R_avg = _compute_expr_lr(gene_R, data_use_avg, gene_names, complex_input)

    # ----- Co-receptor effects -----
    data_R_co_A = _compute_expr_coreceptor(cofactor_input, data_use_avg, gene_names, pair_lr_sig, 'A')
    data_R_co_I = _compute_expr_coreceptor(cofactor_input, data_use_avg, gene_names, pair_lr_sig, 'I')
    # Avoid division by zero
    data_R_co_I[data_R_co_I == 0] = 1.0
    data_R_avg = data_R_avg * data_R_co_A / data_R_co_I

    # ----- Population size factors -----
    # dataLavg2: fraction of cells in each cluster
    cluster_fractions = np.array([np.sum(groups == cl) / n_C for cl in unique_clusters])
    # shape (nLR, nCluster); same value repeated for each LR pair
    data_L_avg2 = np.tile(cluster_fractions, (n_LR, 1))
    data_R_avg2 = data_L_avg2.copy()

    # ----- Agonist / antagonist indices -----
    index_agonist = []
    index_antagonist = []
    if 'agonist' in pair_lr_sig.columns:
        for i, v in enumerate(pair_lr_sig['agonist']):
            if not (pd.isna(v) or v == ''):
                index_agonist.append(i)
    if 'antagonist' in pair_lr_sig.columns:
        for i, v in enumerate(pair_lr_sig['antagonist']):
            if not (pd.isna(v) or v == ''):
                index_antagonist.append(i)

    # ----- Spatial constraint -----
    datatype = cellchat.settings.get('datatype', 'RNA')
    P_spatial = np.ones((n_clusters, n_clusters))
    adj_contact = np.ones((n_clusters, n_clusters))

    if str(datatype).lower() != 'rna':
        coordinates = cellchat.obsm.get('spatial')
        if coordinates is None:
            raise ValueError(
                "Spatial coordinates are missing. Create the object with coordinates."
            )

        spatial_factors = cellchat.spatial.get('spatial_factors')
        if not spatial_factors:
            raise ValueError(
                "Spatial factors are missing. Provide 'ratio' and 'tol' when "
                "creating the CellChat object."
            )

        datasets = pd.Categorical(cellchat.obs['cellchat_dataset'])
        dataset_names = list(datasets.categories)

        def _factor_values(name):
            values = tol if name == 'tol' and tol is not None else spatial_factors.get(name)
            if values is None:
                raise ValueError(f"spatial_factors is missing {name!r}")
            values = np.atleast_1d(values).astype(float).tolist()
            if len(values) == 1:
                values *= len(dataset_names)
            if len(values) != len(dataset_names):
                raise ValueError(
                    f"spatial_factors[{name!r}] must contain one value per cellchat_dataset"
                )
            return values

        spatial_metadata = pd.DataFrame({
            'cellchat_group': pd.Categorical(
                np.asarray(groups), categories=unique_clusters, ordered=True
            ),
            'cellchat_dataset': pd.Categorical(
                np.asarray(datasets), categories=dataset_names
            ),
        }, index=cellchat.obs_names)
        spatial_result = compute_region_distance(
            coordinates=np.asarray(coordinates),
            metadata=spatial_metadata,
            interaction_range=interaction_range,
            ratio=_factor_values('ratio'),
            tol=_factor_values('tol'),
            k_min=k_min,
            contact_dependent=contact_dependent,
            contact_range=contact_range,
            contact_knn_k=contact_knn_k,
            do_symmetric=do_symmetric,
        )
        d_spatial = spatial_result['d_spatial']
        adj_contact = spatial_result['adj_contact']

        if distance_use:
            scaled_distance = d_spatial * scale_distance
            np.fill_diagonal(scaled_distance, np.nan)
            finite_scaled = scaled_distance[np.isfinite(scaled_distance)]
            if finite_scaled.size == 0:
                raise ValueError(
                    "No spatially proximal cell-group pairs were found. Check "
                    "interaction_range, ratio, tol, and k_min."
                )
            min_scaled = float(finite_scaled.min())
            if min_scaled < 1:
                suggested = scale_distance / min_scaled
                raise ValueError(
                    "The minimum scaled spatial distance must be at least 1. "
                    f"Calculated {min_scaled:.4g}; increase scale_distance to a "
                    f"value slightly below {suggested:.4g}."
                )
            P_spatial = np.zeros_like(scaled_distance, dtype=float)
            valid = np.isfinite(scaled_distance) & (scaled_distance != 0)
            P_spatial[valid] = 1.0 / scaled_distance[valid]
            np.fill_diagonal(P_spatial, float(P_spatial.max()))
        else:
            P_spatial = np.ones_like(d_spatial, dtype=float)
            P_spatial[~np.isfinite(d_spatial)] = 0.0
            np.fill_diagonal(P_spatial, 1.0)

        stored_distance = np.array(d_spatial, copy=True)
        if distance_use:
            np.fill_diagonal(stored_distance, np.nan)
        spatial = cellchat.spatial
        spatial['distance'] = stored_distance
        cellchat.spatial = spatial

    annotations = (
        pair_lr_sig['annotation'].fillna('').astype(str).to_numpy()
        if 'annotation' in pair_lr_sig.columns
        else np.full(n_LR, '', dtype=object)
    )
    # SpatialCellChat treats Secreted/ECM/Non-protein interactions as
    # diffusible and Cell-Cell Contact as short-range.  Do not rely on input
    # row ordering (older code assumed contact pairs were last).
    diffusible = {'Secreted Signaling', 'ECM-Receptor', 'Non-protein Signaling'}
    contact_mask = np.array([a not in diffusible for a in annotations], dtype=bool)
    if not use_agan:
        index_agonist = []; index_antagonist = []
    if contact_dependent_forced:
        contact_mask[:] = True
    elif not contact_dependent:
        contact_mask[:] = False

    # ----- Bootstrap permutations -----
    # R: set.seed(seed.use); permutation <- replicate(nboot, sample.int(nC, size = nC))
    if permutation_use is not None:
        permutation = np.asarray(permutation_use, dtype=int)
        if permutation.shape[0] != n_C:
            raise ValueError(
                f"permutation_use has {permutation.shape[0]} rows but n_cells={n_C}"
            )
        n_boot = permutation.shape[1]
        print(f"Using externally supplied permutation matrix ({n_C} x {n_boot}).")
    else:
        permutation = np.zeros((n_C, n_boot), dtype=int)
        for b in range(n_boot):
            permutation[:, b] = np.random.permutation(n_C)

    # Pre-compute bootstrapped averaged expressions
    group_values = np.array(groups)   # categorical group labels

    def _avg_boot(perm_idx):
        """Compute data_use_avg for a permuted group assignment."""
        shuffled = group_values[perm_idx]
        avg_b = np.zeros((n_genes, n_clusters))
        for ci, cluster in enumerate(unique_clusters):
            mask = (shuffled == cluster)
            if not np.any(mask):
                continue
            cluster_data = data_use[:, mask]
            for gi in range(n_genes):
                avg_b[gi, ci] = fun_mean(cluster_data[gi, :])
        return avg_b

    print(f"Pre-computing {n_boot} bootstrap averaged expressions...")
    data_use_avg_boot = [_avg_boot(permutation[:, b]) for b in range(n_boot)]

    # ----- Main inference loop -----
    Prob = np.zeros((n_clusters, n_clusters, n_LR))
    Pval = np.zeros((n_clusters, n_clusters, n_LR))

    print(f"Computing communication probabilities for {n_LR} LR pairs...")

    for i in range(n_LR):
        P_spatial_i = P_spatial * adj_contact if contact_mask[i] else P_spatial
        # Ligand x Receptor outer product -> (nCluster, nCluster) matrix
        dataLR = np.outer(data_L_avg[i, :], data_R_avg[i, :])   # [source, target]

        # Hill function
        P1 = dataLR**n_param / (kh**n_param + dataLR**n_param)
        P1_Pspatial = P1 * P_spatial_i

        if np.sum(P1_Pspatial) == 0:
            Prob[:, :, i] = 0.0
            Pval[:, :, i] = 1.0
            continue

        # Agonist / antagonist
        if i in index_agonist:
            data_agonist = _compute_expr_agonist(
                data_use_avg, gene_names, pair_lr_sig, cofactor_input, i, kh, n_param)
            P2 = np.outer(data_agonist, data_agonist)
        else:
            P2 = np.ones((n_clusters, n_clusters))

        if i in index_antagonist:
            data_antagonist = _compute_expr_antagonist(
                data_use_avg, gene_names, pair_lr_sig, cofactor_input, i, kh, n_param)
            P3 = np.outer(data_antagonist, data_antagonist)
        else:
            P3 = np.ones((n_clusters, n_clusters))

        # Population size
        if population_size:
            P4 = np.outer(data_L_avg2[i, :], data_R_avg2[i, :])
        else:
            P4 = np.ones((n_clusters, n_clusters))

        Pnull = P1 * P2 * P3 * P4 * P_spatial_i
        Prob[:, :, i] = Pnull

        # ----- Bootstrap p-values -----
        Pnull_vec = Pnull.ravel()  # length n_clusters^2

        Pboot_mat = np.zeros((n_clusters * n_clusters, n_boot))
        for b in range(n_boot):
            avg_b = data_use_avg_boot[b]
            dL_b = _compute_expr_lr([gene_L[i]], avg_b, gene_names, complex_input)  # (1, nC)
            dR_b = _compute_expr_lr([gene_R[i]], avg_b, gene_names, complex_input)

            # co-receptor
            dR_coA_b = _compute_expr_coreceptor(cofactor_input, avg_b, gene_names, pair_lr_sig.iloc[[i]].reset_index(drop=True), 'A')
            dR_coI_b = _compute_expr_coreceptor(cofactor_input, avg_b, gene_names, pair_lr_sig.iloc[[i]].reset_index(drop=True), 'I')
            dR_coI_b[dR_coI_b == 0] = 1.0
            dR_b = dR_b * dR_coA_b / dR_coI_b

            dLR_b = np.outer(dL_b[0, :], dR_b[0, :])
            P1_b = dLR_b**n_param / (kh**n_param + dLR_b**n_param)

            if i in index_agonist:
                da_b = _compute_expr_agonist(avg_b, gene_names, pair_lr_sig, cofactor_input, i, kh, n_param)
                P2_b = np.outer(da_b, da_b)
            else:
                P2_b = np.ones((n_clusters, n_clusters))

            if i in index_antagonist:
                dan_b = _compute_expr_antagonist(avg_b, gene_names, pair_lr_sig, cofactor_input, i, kh, n_param)
                P3_b = np.outer(dan_b, dan_b)
            else:
                P3_b = np.ones((n_clusters, n_clusters))

            if population_size:
                shuffled = group_values[permutation[:, b]]
                frac_b = np.array([np.sum(shuffled == cl) / n_C for cl in unique_clusters])
                P4_b = np.outer(frac_b, frac_b)
            else:
                P4_b = np.ones((n_clusters, n_clusters))

            Pboot_b = P1_b * P2_b * P3_b * P4_b * P_spatial_i
            Pboot_mat[:, b] = Pboot_b.ravel()

        # nReject = rowSums(Pboot - Pnull > 0)
        n_reject = np.sum(Pboot_mat - Pnull_vec[:, np.newaxis] > 0, axis=1)
        p = n_reject / n_boot
        Pval[:, :, i] = p.reshape(n_clusters, n_clusters)

    # R: Pval[Prob == 0] <- 1
    Pval[Prob == 0] = 1.0

    # ----- Build long-format communication table -----
    results = []
    for lr_idx in range(n_LR):
        ligand = gene_L[lr_idx]
        receptor = gene_R[lr_idx]

        # Get metadata from pair_lr_sig
        if lr_idx < len(pair_lr_sig):
            row = pair_lr_sig.iloc[lr_idx]
            interaction_name = row.get('interaction_name', f'{ligand}_{receptor}')
            pathway_name = row.get('pathway_name', 'Unknown')
            annotation = row.get('annotation', 'Secreted Signaling')
        else:
            interaction_name = f'{ligand}_{receptor}'
            pathway_name = 'Unknown'
            annotation = 'Secreted Signaling'

        for src_idx in range(n_clusters):
            for tgt_idx in range(n_clusters):
                prob_val = float(Prob[src_idx, tgt_idx, lr_idx])
                pval_val = float(Pval[src_idx, tgt_idx, lr_idx])

                # Store all entries (filtering happens in downstream functions)
                results.append({
                    'source': unique_clusters[src_idx],
                    'target': unique_clusters[tgt_idx],
                    'ligand_complex': ligand,
                    'receptor_complex': receptor,
                    'interaction_name': interaction_name,
                    'pathway_name': pathway_name,
                    'annotation': annotation,
                    'prob': prob_val,
                    'pval': pval_val,
                    'ligand_mean_expr': float(data_L_avg[lr_idx, src_idx]),
                    'receptor_mean_expr': float(data_R_avg[lr_idx, tgt_idx])
                })

    communication = pd.DataFrame(results)

    lr_names = list(pair_lr_sig.index.astype(str)) if pair_lr_sig.index.dtype != object else list(pair_lr_sig.index)
    if 'interaction_name' in pair_lr_sig.columns:
        lr_names = list(pair_lr_sig['interaction_name'])

    cellchat.network = {
        'prob': matrix_dict_from_array(Prob, lr_names, sparse_output=True),
        'pval': matrix_dict_from_array(Pval, lr_names),
        'groups': unique_clusters,
        'interactions': pair_lr_sig.reset_index(drop=True)
    }

    is_spatial = str(datatype).lower() != 'rna'
    cellchat.settings['parameter'] = {
        'type_mean': type_method,
        'trim': trim,
        'raw_use': raw_use,
        'population_size': population_size,
        'n_boot': n_boot,
        'seed_use': seed_use,
        'kh': kh,
        'n': n_param,
        'distance_use': distance_use if is_spatial else None,
        'interaction_range': interaction_range if is_spatial else None,
        'scale_distance': scale_distance if is_spatial else None,
        'k_min': k_min if is_spatial else None,
        'contact_dependent': bool(contact_dependent and is_spatial),
        'contact_range': contact_range if is_spatial else None,
        'contact_knn_k': contact_knn_k if is_spatial else None,
        'contact_dependent_forced': contact_dependent_forced,
    }

    print(">>> CellChat inference is done.")
    print("Results stored in .network.")
    return cellchat


# ---------------------------------------------------------------------------
# compute_pathway_probability
# ---------------------------------------------------------------------------
def compute_pathway_probability(
    cellchat,
    lr_sig=None,
    thresh: float = 0.05
):
    """
    Aggregate the canonical L-R network to pathway level.
    """
    # Use the filtered ligand-receptor network as the source for pathway values.
    if isinstance(cellchat.network, dict) and 'prob' in cellchat.network:

        lr_names = network_names(cellchat.network)
        prob = stack_network_field(cellchat.network, 'prob', names=lr_names, fill_value=0.0)
        pval = stack_network_field(cellchat.network, 'pval', names=lr_names, fill_value=1.0)
        unique_clusters = cellchat.network['groups']
        n_clusters = len(unique_clusters)

        # The probability tensor is indexed by the interaction table captured
        # when compute_communication_probability ran. Prefer that aligned metadata over the
        # mutable object-level LR selection.
        if lr_sig is None:
            lr_sig = cellchat.network.get('interactions')
            if lr_sig is None:
                lr_sig = cellchat.lr_pairs.get('significant')
        if lr_sig is None:
            raise ValueError("No LR pairs available.")
        lr_sig = lr_sig.reset_index(drop=True)
        if len(lr_sig) != len(lr_names):
            raise ValueError(
                "LR metadata does not match the communication probability tensor. "
                "Pass the interaction table used for compute_communication_probability."
            )

        # R-equivalent project convention: retain only pval < thresh.
        prob[pval >= thresh] = 0.0

        pathways = lr_sig.get('pathway_name', pd.Series([f'Pair_{i}' for i in range(len(lr_sig))]))
        unique_pathways = list(pathways.unique())

        # R: group <- factor(pairLR.use$pathway_name, levels = pathways)
        # R: prob.pathways <- aperm(apply(prob, c(1, 2), by, group, sum), c(2, 3, 1))
        n_paths = len(unique_pathways)
        prob_pathways = np.zeros((n_clusters, n_clusters, n_paths))

        for pi, pathway in enumerate(unique_pathways):
            idx = np.where(pathways == pathway)[0]
            # sum over matching LR pairs
            prob_pathways[:, :, pi] = np.sum(prob[:, :, idx], axis=2)

        # R: pathways.sig <- pathways[apply(prob.pathways, 3, sum) != 0]
        path_sums = np.sum(prob_pathways, axis=(0, 1))
        sig_mask = path_sums != 0
        pathways_sig = [unique_pathways[pi] for pi in range(n_paths) if sig_mask[pi]]
        prob_pathways_sig = prob_pathways[:, :, sig_mask]
        pval_pathways_sig = np.zeros_like(prob_pathways_sig)

        # R: order by decreasing total probability
        if prob_pathways_sig.shape[2] > 0:
            total_probs = np.sum(prob_pathways_sig, axis=(0, 1))
            order_idx = np.argsort(-total_probs)
            pathways_sig = [pathways_sig[i] for i in order_idx]
            prob_pathways_sig = prob_pathways_sig[:, :, order_idx]
            pval_pathways_sig = pval_pathways_sig[:, :, order_idx]

        # Persist pathway networks in the same sparse dictionary form as LR networks.
        pathway_network = dict(cellchat.pathway_network)
        pathway_network.update({
            'prob': matrix_dict_from_array(
                prob_pathways_sig, pathways_sig, sparse_output=True
            ),
            'pval': matrix_dict_from_array(pval_pathways_sig, pathways_sig),
            'groups': unique_clusters,
        })
        cellchat.pathway_network = pathway_network

        print("Pathway-level results stored in .pathway_network.")

    else:
        raise ValueError("Run compute_communication_probability first.")

    return cellchat


# ---------------------------------------------------------------------------
# aggregate_network
# ---------------------------------------------------------------------------
def aggregate_signaling_matrix(
    cellchat,
    signaling: Optional[Union[str, List[str]]] = None,
    thresh: float = 0.05,
) -> np.ndarray:
    """Aggregate significant pathway probabilities into a group-by-group matrix.

    Parameters
    ----------
    cellchat
        A CellChat object with a canonical ``pathway_network``.
    signaling
        One pathway name, multiple pathway names, or ``None`` to aggregate all
        available pathways.
    thresh
        Retain pathway edges with ``pval < thresh``, matching the package-wide
        CellChatPy/R filtering convention.

    Returns
    -------
    numpy.ndarray
        A dense sender-by-receiver communication-strength matrix. The group
        order is ``cellchat.pathway_network['groups']``.
    """
    if not np.isfinite(thresh) or thresh < 0:
        raise ValueError("thresh must be a finite non-negative number.")

    network = cellchat.pathway_network
    if not isinstance(network, dict) or "prob" not in network:
        raise ValueError("No pathway probability data found in cellchat.pathway_network.")

    pathways = network_names(network)
    if not pathways:
        raise ValueError("cellchat.pathway_network contains no signaling pathways.")

    if signaling is None:
        selected = pathways
    elif isinstance(signaling, str):
        selected = [signaling]
    else:
        selected = [str(pathway) for pathway in signaling]
        if not selected:
            raise ValueError("signaling must name at least one pathway.")

    missing = [pathway for pathway in selected if pathway not in pathways]
    if missing:
        raise ValueError(f"Signaling pathway(s) not found in cellchat.pathway_network: {missing}")

    probability = stack_network_field(
        network, "prob", names=selected, fill_value=0.0
    )
    pvalue = stack_network_field(
        network, "pval", names=selected, fill_value=0.0
    )
    if probability.ndim != 3 or pvalue.shape != probability.shape:
        raise ValueError(
            "Pathway probability and p-value data must be aligned C x C x pathway tensors."
        )

    probability = probability.copy()
    probability[pvalue >= thresh] = 0.0
    return np.sum(probability, axis=2)


def aggregate_network(
    cellchat,
    slot_name: str = "network",
    thresh: float = 0.05
):
    """
    Aggregate communication probabilities into group-by-group matrices.

    The resulting ``count`` and ``weight`` arrays are stored in the canonical
    network dictionary (normally ``uns['network']``). ``groups`` records the row
    and column order of both matrices.
    """
    if slot_name != "network":
        raise ValueError("aggregate_network currently supports slot_name='network' only")

    net = cellchat.network
    if isinstance(net, dict) and 'prob' in net:
        names = network_names(net)
        prob = stack_network_field(net, 'prob', names=names, fill_value=0.0)
        pval = stack_network_field(net, 'pval', names=names, fill_value=1.0)
        if prob.ndim != 3 or pval.shape != prob.shape:
            raise ValueError(
                "Temporary network probability and p-value tensors must have matching "
                "C x C x L shapes"
            )

        pval[prob == 0] = 1.0
        # Retain only interactions with pval < thresh.
        prob[pval >= thresh] = 0.0
        count = np.sum(prob > 0, axis=2)
        weight = np.nansum(prob, axis=2)

        net['count'] = count
        net['weight'] = weight
        if 'groups' not in net:
            groups = cellchat.groups
            if groups is None:
                raise ValueError("cellchat.groups must be set before aggregating the network.")
            net['groups'] = list(groups.categories)
        cellchat.network = net

    else:
        raise ValueError("Run compute_communication_probability first.")

    return cellchat


# ---------------------------------------------------------------------------
# compute_network_centrality
# ---------------------------------------------------------------------------
def _compute_centrality_local(net0):
    """
    Compute centrality measures for a single (n x n) communication matrix.
    Mirrors R computeCentralityLocal() using networkx instead of igraph/sna.

    R uses:
      outdeg    = igraph::strength(G, mode="out")   -> weighted out-degree
      indeg     = igraph::strength(G, mode="in")    -> weighted in-degree
      hub       = igraph::hub_score(G)$vector       -> HITS hub score
      authority = igraph::authority_score(G)$vector -> HITS authority score
      eigen     = igraph::eigen_centrality(G)$vector
      page_rank = igraph::page_rank(G)$vector
      betweenness = igraph::betweenness(G, weights=1/weight)  (inverted weights)
      flowbet   = sna::flowbet(net)                 -> max-flow betweenness (approx)
      info      = sna::infocent(net)                -> information centrality (approx)
    """
    import networkx as nx
    n = net0.shape[0]
    zeros = np.zeros(n)

    outdeg = np.sum(net0, axis=1)
    indeg  = np.sum(net0, axis=0)
    # Keep edge counts alongside weighted strength.  The dual role plot uses
    # these values for its "Number of interactions" panel.
    outdeg_unweighted = np.sum(net0 > 0, axis=1, dtype=float)
    indeg_unweighted = np.sum(net0 > 0, axis=0, dtype=float)

    # Build directed weighted graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            w = float(net0[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        return {
            'outdeg': outdeg, 'indeg': indeg,
            'outdeg_unweighted': outdeg_unweighted,
            'indeg_unweighted': indeg_unweighted,
            'hub': zeros.copy(), 'authority': zeros.copy(),
            'eigen': zeros.copy(), 'page_rank': zeros.copy(),
            'betweenness': zeros.copy(), 'flowbet': zeros.copy(), 'info': zeros.copy()
        }

    # HITS: hub and authority scores
    try:
        hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
        hub_arr   = np.array([hubs.get(i, 0.0)       for i in range(n)])
        auth_arr  = np.array([authorities.get(i, 0.0) for i in range(n)])
    except Exception:
        hub_arr = auth_arr = zeros.copy()

    # Eigenvector centrality (weighted)
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        eigen_arr = np.array([ec.get(i, 0.0) for i in range(n)])
    except Exception:
        eigen_arr = zeros.copy()

    # PageRank (weighted)
    try:
        pr = nx.pagerank(G, weight='weight')
        pr_arr = np.array([pr.get(i, 0.0) for i in range(n)])
    except Exception:
        pr_arr = zeros.copy()

    # Betweenness: R inverts edge weights before computing betweenness
    try:
        G_inv = G.copy()
        for u, v, d in G_inv.edges(data=True):
            d['weight'] = 1.0 / d['weight'] if d['weight'] > 0 else 1e10
        bet = nx.betweenness_centrality(G_inv, weight='weight', normalized=True)
        bet_arr = np.array([bet.get(i, 0.0) for i in range(n)])
    except Exception:
        bet_arr = zeros.copy()

    # flowbet: R uses sna::flowbet(net), i.e. max-flow betweenness.
    # Approximate it by the drop in maximum s->t flow after removing each node.
    try:
        flowbet_arr = np.zeros(n, dtype=float)
        for v in range(n):
            for s in range(n):
                if s == v:
                    continue
                for t in range(n):
                    if t in (s, v):
                        continue
                    try:
                        base_flow = nx.maximum_flow_value(G, s, t, capacity='weight')
                        if base_flow <= 0:
                            continue
                        G_removed = G.copy()
                        G_removed.remove_node(v)
                        without_v = nx.maximum_flow_value(G_removed, s, t, capacity='weight') if (
                            s in G_removed and t in G_removed
                        ) else 0.0
                        flowbet_arr[v] += max(0.0, base_flow - without_v)
                    except Exception:
                        continue
    except Exception:
        flowbet_arr = zeros.copy()

    # info: R calls sna::infocent(net, diag=TRUE, rescale=TRUE, cmode="lower").
    # The cmode="lower" conversion keeps the lower triangle before treating the
    # graph as undirected; then information centrality is current-flow closeness.
    try:
        lower_net = np.tril(np.asarray(net0, dtype=float), k=-1)
        G_info = nx.Graph()
        G_info.add_nodes_from(range(n))
        rows, cols = np.where(lower_net > 0)
        for i, j in zip(rows, cols, strict=False):
            w = float(lower_net[i, j])
            G_info.add_edge(i, j, weight=w)

        info_arr = np.zeros(n, dtype=float)
        for comp in nx.connected_components(G_info):
            if len(comp) < 2:
                continue
            H = G_info.subgraph(comp).copy()
            # current-flow algorithms operate on connected components only.
            info_raw = nx.current_flow_closeness_centrality(H, weight='weight')
            for node, val in info_raw.items():
                info_arr[node] = float(val)
        mx = info_arr.max()
        if mx > 0:
            info_arr = info_arr / mx
    except Exception:
        info_arr = zeros.copy()

    return {
        'outdeg':      outdeg,
        'indeg':       indeg,
        'outdeg_unweighted': outdeg_unweighted,
        'indeg_unweighted': indeg_unweighted,
        'hub':         hub_arr,
        'authority':   auth_arr,
        'eigen':       eigen_arr,
        'page_rank':   pr_arr,
        'betweenness': bet_arr,
        'flowbet':     flowbet_arr,
        'info':        info_arr,
    }


def compute_network_centrality(
    cellchat,
    slot_name: str = "pathway_network",
    thresh: float = 0.05
):
    """
    Compute centrality for each pathway.
    Compute weighted and unweighted signaling centrality measures.

    Stores the per-pathway measures in ``pathway_network['centrality']`` and the
    equivalent long table in ``uns['cellchat_centrality']``.
    """
    if slot_name not in {"network", "pathway_network"}:
        raise ValueError("slot_name must be 'network' or 'pathway_network'.")
    net_data = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if not isinstance(net_data, dict) or "prob" not in net_data:
        raise ValueError(f"No probability data found in {slot_name}. Run the communication inference first.")
    names = network_names(net_data)
    prob = stack_network_field(net_data, "prob", names=names, fill_value=0.0)
    pval = stack_network_field(net_data, "pval", names=names, fill_value=1.0)
    groups = list(net_data.get("groups", []))
    if not groups:
        raise ValueError(f"{slot_name} must define its group order.")
    pathway_names = names
    n_paths = len(pathway_names)

    # Apply threshold
    pval = pval.copy()
    pval[prob == 0] = 1.0
    prob = prob.copy()
    # CellChatPy uses the strict significance rule pval < thresh.
    prob[pval >= thresh] = 0.0

    # Compute centrality for each pathway
    centrality_rows = []
    measures = [
        'outdeg', 'indeg', 'outdeg_unweighted', 'indeg_unweighted',
        'hub', 'authority', 'eigen', 'page_rank', 'betweenness', 'flowbet', 'info'
    ]

    for pi in range(n_paths):
        net0 = prob[:, :, pi]
        centr_dict = _compute_centrality_local(net0)

        # Convert to long format
        for measure in measures:
            values = centr_dict.get(measure, np.zeros(len(groups)))
            for gi, group in enumerate(groups):
                centrality_rows.append({
                    'group': group,
                    'pathway_name': pathway_names[pi],
                    'measure': measure,
                    'value': float(values[gi])
                })

    # Store in new format
    df_centrality = pd.DataFrame(centrality_rows)
    if not hasattr(cellchat, 'uns'):
        cellchat.uns = {}
    cellchat.uns['cellchat_centrality'] = df_centrality

    centrality_all = {}
    for pi in range(n_paths):
        net0 = prob[:, :, pi]
        centrality_all[pathway_names[pi]] = _compute_centrality_local(net0)

    if slot_name == "pathway_network":
        cellchat.pathway_network['centrality'] = centrality_all
    else:
        cellchat.network['centrality'] = centrality_all

    return cellchat


# ---------------------------------------------------------------------------
# filter_communication
# ---------------------------------------------------------------------------
def filter_communication(
    cellchat,
    min_cells: int = 10,
    slot_name: str = "network"
):
    """
    Filter out cell-cell communication if there are only few cells in certain
    cell groups - port of R filterCommunication.
    """
    if slot_name not in {"network", "pathway_network"}:
        raise ValueError("slot_name must be 'network' or 'pathway_network'.")
    groups = cellchat.groups
    if groups is None or not hasattr(groups, "categories"):
        raise ValueError("cellchat.groups must be a categorical vector before filtering.")
    group_counts = {cl: int(np.sum(groups == cl)) for cl in groups.categories}

    network = cellchat.network if slot_name == "network" else cellchat.pathway_network
    # Observation categories and network matrices can use different orders.
    # Resolve excluded group names against the network's explicit axes.
    excluded_names = {
        str(cl) for cl in groups.categories if group_counts[cl] <= min_cells
    }
    network_groups = [str(group) for group in network.get("groups", [])]
    if excluded_names and not network_groups:
        raise ValueError(f"{slot_name} must define its group order before filtering.")
    network_excludes = [
        index for index, group in enumerate(network_groups)
        if group in excluded_names
    ]

    if excluded_names:
        print(f"Excluding cell groups due to few cells: {sorted(excluded_names)}")

    if 'prob' in network and network_excludes:
        network['prob'] = zero_group_axes(network['prob'], network_excludes)
        if 'pval' in network:
            # Keep p-values dense and mark every zero-probability edge as
            # non-significant, as filterCommunication does in R.
            filtered_pval = {}
            for name, pval in network['pval'].items():
                filtered = np.array(pval, copy=True)
                probability = network['prob'][name]
                probability_array = probability.toarray() if hasattr(probability, "toarray") else np.asarray(probability)
                filtered[probability_array == 0] = 1.0
                filtered_pval[name] = filtered
            network['pval'] = filtered_pval

    if slot_name == "network":
        cellchat.network = network
    else:
        cellchat.pathway_network = network
    return cellchat


# ---------------------------------------------------------------------------
# subset_communication
# ---------------------------------------------------------------------------
class _MergedDatasetShim:
    """Read-only adapter exposing a single dataset of a merged object so that
    single-object subset_communication can run on it."""
    def __init__(self, network, pathway_network, parent):
        self.network = network
        self.pathway_network = pathway_network if pathway_network is not None else {}
        groups = list(network.get('groups', []))
        self.groups = pd.Categorical(groups, categories=groups, ordered=True)
        self.database = parent.database


def subset_communication(
    cellchat,
    slot_name: str = "network",
    thresh_pval: float = 0.05,
    thresh_prob: float = 0.0,
    sources_use=None,
    targets_use=None,
    signaling=None,
    pair_lr_use=None,
    network_table=None,
    dataset_names=None,
    ligand_log_fc=None,
    receptor_log_fc=None,
):
    """
    Extract communication results as a tidy DataFrame.

    ``network_table`` accepts a precomputed communication table (for example
    from ``map_network_deg``) and applies the same source, target, signaling,
    dataset, and logFC filters.
      dataset_names: keep only rows of these datasets (merged objects).
      ligand_log_fc / receptor_log_fc : keep rows whose ligand/receptor logFC is
                     >= the threshold (None = no filter on that side).
    """
    # DEG-table filtering path.
    if network_table is not None and isinstance(network_table, pd.DataFrame):
        df = network_table.copy()
        if dataset_names is not None and 'cellchat_dataset' in df.columns:
            ds = [dataset_names] if isinstance(dataset_names, str) else list(dataset_names)
            df = df[df['cellchat_dataset'].isin(ds)]
        if ligand_log_fc is not None and 'ligand_log_fc' in df.columns:
            if ligand_log_fc >= 0:
                df = df[df['ligand_log_fc'].fillna(-np.inf) >= ligand_log_fc]
            else:
                df = df[df['ligand_log_fc'].fillna(np.inf) <= ligand_log_fc]
        if receptor_log_fc is not None and 'receptor_log_fc' in df.columns:
            if receptor_log_fc >= 0:
                df = df[df['receptor_log_fc'].fillna(-np.inf) >= receptor_log_fc]
            else:
                df = df[df['receptor_log_fc'].fillna(np.inf) <= receptor_log_fc]
        if sources_use is not None:
            df = df[df['source'].isin(sources_use)]
        if targets_use is not None:
            df = df[df['target'].isin(targets_use)]
        if signaling is not None:
            sig = [signaling] if isinstance(signaling, str) else signaling
            df = df[df['pathway_name'].isin(sig)]
        return df.reset_index(drop=True)

    # Merged object without an explicit net table: return per-dataset dict
    network_slot = cellchat.network if slot_name == "network" else cellchat.pathway_network
    if isinstance(network_slot, dict) and not ('prob' in network_slot):
        ds_keys = [k for k, v in network_slot.items()
                   if isinstance(v, dict) and 'prob' in v]
        result = {}
        for name in ds_keys:
            pathway_network_for_dataset = None
            if isinstance(cellchat.pathway_network, dict):
                pathway_network_for_dataset = cellchat.pathway_network.get(name)
            shim = _MergedDatasetShim(
                network_slot[name], pathway_network_for_dataset, cellchat
            )
            result[name] = subset_communication(
                shim, slot_name=slot_name, thresh_pval=thresh_pval,
                thresh_prob=thresh_prob, sources_use=sources_use,
                targets_use=targets_use, signaling=signaling, pair_lr_use=pair_lr_use)
        return result

    if slot_name == "network":
        net_data = cellchat.network
    elif slot_name == "pathway_network":
        net_data = cellchat.pathway_network
    else:
        raise ValueError("slot_name must be 'network' or 'pathway_network'")

    names = network_names(net_data)
    prob_array = stack_network_field(
        net_data, 'prob', names=names, fill_value=0.0
    )
    pval_array = stack_network_field(
        net_data, 'pval', names=names, fill_value=1.0
    )
    groups = net_data.get('groups', [f'group_{i}' for i in range(prob_array.shape[0])])

    # CellChatPy uses the strict significance rule pval < thresh_pval.
    prob_array[pval_array >= thresh_pval] = 0.0

    n_source, n_target, n_pairs = prob_array.shape

    # For pathway_network: pathways are the "pairs"
    if slot_name == "pathway_network":
        pathway_names = names
        results = []
        for pi in range(n_pairs):
            pathway_name = pathway_names[pi] if pi < len(pathway_names) else f'pathway_{pi}'
            for si in range(n_source):
                for ti in range(n_target):
                    prob = float(prob_array[si, ti, pi])
                    pval = float(pval_array[si, ti, pi]) if pval_array.shape == prob_array.shape else 1.0
                    if prob > thresh_prob:
                        results.append({
                            'source': groups[si] if si < len(groups) else f'group_{si}',
                            'target': groups[ti] if ti < len(groups) else f'group_{ti}',
                            'pathway_name': pathway_name,
                            'prob': prob,
                            'pval': pval,
                        })
        df = pd.DataFrame(results)
        if len(df) > 0:
            if sources_use is not None:
                df = df[df['source'].isin(sources_use)]
            if targets_use is not None:
                df = df[df['target'].isin(targets_use)]
            if signaling is not None:
                if isinstance(signaling, str):
                    signaling = [signaling]
                df = df[df['pathway_name'].isin(signaling)]
        return df.reset_index(drop=True)

    # Ligand-receptor network entries
    # Get interaction metadata
    interactions = net_data.get('interactions', None)
    lr_names = names

    results = []
    for pair_idx in range(n_pairs):
        if interactions is not None and pair_idx < len(interactions):
            row = interactions.iloc[pair_idx]
            ligand = str(row.get('ligand', f'ligand_{pair_idx}'))
            receptor = str(row.get('receptor', f'receptor_{pair_idx}'))
            interaction_name = str(row.get('interaction_name', f'{ligand}_{receptor}'))
            interaction_name_2 = str(row.get('interaction_name_2', f'{ligand} - {receptor}'))
            pathway_name = str(row.get('pathway_name', f'pathway_{pair_idx}'))
            annotation = str(row.get('annotation', 'Secreted Signaling'))
            evidence = str(row.get('evidence', ''))
        else:
            ligand = f'ligand_{pair_idx}'
            receptor = f'receptor_{pair_idx}'
            interaction_name = lr_names[pair_idx] if pair_idx < len(lr_names) else f'interaction_{pair_idx}'
            interaction_name_2 = f'{ligand} - {receptor}'
            pathway_name = f'pathway_{pair_idx}'
            annotation = 'Secreted Signaling'
            evidence = ''

        for source_idx in range(n_source):
            for target_idx in range(n_target):
                prob = prob_array[source_idx, target_idx, pair_idx]
                pval = pval_array[source_idx, target_idx, pair_idx] if pval_array.shape[2] > pair_idx else 1.0

                # R: subset(net, prob > 0); only keep strictly positive prob
                if prob > thresh_prob:
                    results.append({
                        'source': groups[source_idx],
                        'target': groups[target_idx],
                        'ligand': ligand,
                        'receptor': receptor,
                        'prob': prob,
                        'pval': pval,
                        'interaction_name': interaction_name,
                        'interaction_name_2': interaction_name_2,
                        'pathway_name': pathway_name,
                        'annotation': annotation,
                        'evidence': evidence
                    })

    df = pd.DataFrame(results)
    if len(df) > 0:
        # Apply source/target filtering
        if sources_use is not None:
            df = df[df['source'].isin(sources_use)]
        if targets_use is not None:
            df = df[df['target'].isin(targets_use)]
        if signaling is not None:
            if isinstance(signaling, str):
                signaling = [signaling]
            df = df[df['pathway_name'].isin(signaling)]
        if pair_lr_use is not None:
            if 'interaction_name' in pair_lr_use.columns:
                df = df[df['interaction_name'].isin(pair_lr_use['interaction_name'])]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# identify_enriched_interactions  (R: identify_enriched_interactions)
# ---------------------------------------------------------------------------
def identify_enriched_interactions(
    cellchat,
    from_,
    to,
    bidirection: bool = False,
    pair_only: bool = True,
    pair_lr_use0=None,
    thresh: float = 0.05
) -> pd.DataFrame:
    """
    Return enriched L-R interactions between specific source-target cell groups.

    Mirrors R identify_enriched_interactions. Requires rank_network_pairwise to have
    been run first (stores results in cellchat.network['pairwise_rank']).

    Parameters
    ----------
    cellchat : CellChat object
    from_ : list of str or int
        Source cell group names or numeric indices.
    to : list of str or int
        Target cell group names or numeric indices.
    bidirection : bool
        If True, also check the reverse direction.
    pair_only : bool
        If True, return only [ligand, receptor] columns.
    pair_lr_use0 : DataFrame, optional
        Pre-filtered L-R pairs.
    thresh : float
        P-value threshold.

    Returns
    -------
    DataFrame of enriched L-R pairs.
    """
    pairwise_lr = cellchat.network.get('pairwise_rank', None)
    if pairwise_lr is None:
        raise ValueError(
            "The interactions between pairwise cell groups have not been extracted! "
            "Please first run `rank_network_pairwise(cellchat)`"
        )

    group_names_all = list(pairwise_lr.keys())

    if not isinstance(from_[0], (int, np.integer)):
        from_ = [group_names_all.index(f) for f in from_ if f in group_names_all]
        if len(from_) == 0:
            raise ValueError("No valid 'from' cell group names found.")
    if not isinstance(to[0], (int, np.integer)):
        to = [group_names_all.index(t) for t in to if t in group_names_all]
        if len(to) == 0:
            raise ValueError("No valid 'to' cell group names found.")

    if len(from_) != len(to):
        raise ValueError("The length of 'from' and 'to' must be the same!")

    if bidirection:
        from_ = list(from_) + list(to)
        to = list(to) + list(from_)

    if pair_lr_use0 is None:
        pair_lr_use0_list = []
        for i in range(len(from_)):
            fi = from_[i]
            ti = to[i]
            if fi >= len(group_names_all) or ti >= len(group_names_all):
                continue
            pairwise_lr_ij = pairwise_lr[group_names_all[fi]][group_names_all[ti]]
            mask = pairwise_lr_ij['pval'] < thresh
            if mask.sum() > 0:
                pair_lr_use0_list.append(pairwise_lr_ij[mask].copy())
        if len(pair_lr_use0_list) == 0:
            return pd.DataFrame()
        pair_lr_use0 = pd.concat(pair_lr_use0_list)

    n_pairs = len(pair_lr_use0)
    pval_mat = np.zeros((n_pairs, len(from_)))
    prob_mat = np.zeros((n_pairs, len(from_)))

    for k_idx in range(len(from_)):
        fi = from_[k_idx]
        ti = to[k_idx]
        pairwise_lr_ij = pairwise_lr[group_names_all[fi]][group_names_all[ti]]
        # Align by index
        common_idx = pair_lr_use0.index.intersection(pairwise_lr_ij.index)
        pairwise_lr_ij_aligned = pairwise_lr_ij.loc[common_idx]
        aligned_use = pair_lr_use0.loc[common_idx]

        if len(pairwise_lr_ij_aligned) == 0:
            continue

        pval_ij = pairwise_lr_ij_aligned['pval'].values.copy()
        prob_ij = pairwise_lr_ij_aligned['prob'].values.copy()

        pval_code = np.ones_like(pval_ij)
        pval_code[(pval_ij > 0.01) & (pval_ij <= 0.05)] = 2
        pval_code[pval_ij <= 0.01] = 3
        prob_ij[pval_code == 1] = 0

        for row_idx, orig_idx in enumerate(common_idx):
            if orig_idx in pair_lr_use0.index:
                pos = list(pair_lr_use0.index).index(orig_idx)
                pval_mat[pos, k_idx] = pval_code[row_idx]
                prob_mat[pos, k_idx] = prob_ij[row_idx]

    prob_mat[prob_mat == 0] = np.nan

    keep_rows = ~np.all(np.isnan(prob_mat), axis=1)
    pair_lr_use0 = pair_lr_use0.loc[keep_rows]

    if pair_only:
        keep_cols = [c for c in ['ligand', 'receptor'] if c in pair_lr_use0.columns]
        return pair_lr_use0[keep_cols].reset_index(drop=True)

    return pair_lr_use0.reset_index(drop=True)


# ---------------------------------------------------------------------------
# compute_region_distance  (R: compute_region_distance)
# ---------------------------------------------------------------------------
def compute_region_distance(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    interaction_range: Optional[float] = None,
    ratio: Optional[List[float]] = None,
    tol: Optional[List[float]] = None,
    k_min: int = 10,
    contact_dependent: bool = True,
    contact_range: Optional[float] = None,
    contact_knn_k: Optional[int] = None,
    do_symmetric: bool = True,
) -> Dict:
    """
    Compute pairwise region distances and adjacency matrices from spatial coordinates.

    Mirrors R compute_region_distance.

    Parameters
    ----------
    coordinates : ndarray (n_cells, 2)
        Spatial coordinates of each cell/spot.
    metadata : DataFrame
        Must contain 'cellchat_group' and 'cellchat_dataset' columns.
    interaction_range : float
        Maximum interaction/diffusion range in microns.
    ratio : list of float
        Conversion factor from pixels to microns (one per sample).
    tol : list of float
        Tolerance (half cell/spot size in microns).
    k_min : int
        Minimum number of interacting cell pairs for adjacency.
    contact_dependent : bool
        Whether to determine spatially proximal groups.
    contact_range : float
        Interaction range for contact-dependent signaling (microns).
    contact_knn_k : int
        Number of KNN neighbors for contact-based adjacency.
    do_symmetric : bool
        Make adjacency matrices symmetric.

    Returns
    -------
    dict with keys: 'd_spatial', 'adj_contact'
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("scikit-learn is required for compute_region_distance")

    required_columns = {'cellchat_group', 'cellchat_dataset'}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"metadata is missing required columns: {sorted(missing_columns)}"
        )
    group_arr = np.array(metadata['cellchat_group'])
    datasets_arr = np.array(metadata['cellchat_dataset'])

    unique_groups = list(metadata['cellchat_group'].unique())
    if hasattr(metadata['cellchat_group'], 'cat'):
        unique_groups = list(metadata['cellchat_group'].cat.categories)
    unique_datasets = list(metadata['cellchat_dataset'].unique())
    if hasattr(metadata['cellchat_dataset'], 'cat'):
        unique_datasets = list(metadata['cellchat_dataset'].cat.categories)

    num_cluster = len(unique_groups)
    n_datasets = len(unique_datasets)

    d_spatial = np.full((num_cluster, num_cluster, n_datasets), np.nan)
    adj_spatial = np.zeros((num_cluster, num_cluster, n_datasets))
    adj_contact = np.zeros((num_cluster, num_cluster, n_datasets))
    adj_contact_knn = np.zeros((num_cluster, num_cluster, n_datasets))

    if contact_dependent and contact_knn_k is not None:
        nn_ranked = np.full((coordinates.shape[0], contact_knn_k), -1, dtype=int)
        for k_idx, dataset_name in enumerate(unique_datasets):
            idx_k = np.where(datasets_arr == dataset_name)[0]
            if len(idx_k) <= contact_knn_k:
                nn_ranked[idx_k, :] = np.tile(idx_k.reshape(-1, 1), (1, contact_knn_k))
            else:
                nbrs = NearestNeighbors(n_neighbors=min(contact_knn_k, len(idx_k)), algorithm='auto')
                nbrs.fit(coordinates[idx_k, :])
                _, indices = nbrs.kneighbors(coordinates[idx_k, :])
                nn_ranked[idx_k, :] = idx_k[indices]
        k_min_contact = k_min
    else:
        nn_ranked = np.ones((coordinates.shape[0], 1), dtype=int)
        k_min_contact = -1

    if ratio is None:
        ratio = [1.0] * n_datasets
    if tol is None:
        tol = [10.0] * n_datasets
    if interaction_range is None:
        interaction_range = 250.0
    if contact_dependent and contact_range is None and contact_knn_k is None:
        raise ValueError(
            "Provide either contact_range or contact_knn_k when "
            "contact_dependent=True."
        )
    contact_range_val = contact_range if contact_dependent and contact_range else 10000.0

    def _fun_mean(x):
        values = np.sort(np.asarray(x, dtype=float))
        values = values[np.isfinite(values)]
        trim_count = int(np.floor(0.1 * len(values)))
        if trim_count:
            values = values[trim_count:-trim_count]
        return float(np.mean(values)) if len(values) else np.nan

    for k_idx, dataset_name in enumerate(unique_datasets):
        idx_k = np.where(datasets_arr == dataset_name)[0]
        for i in range(num_cluster):
            for j in range(num_cluster):
                idx_i = np.where((group_arr == unique_groups[i]) & (datasets_arr == dataset_name))[0]
                idx_j = np.where((group_arr == unique_groups[j]) & (datasets_arr == dataset_name))[0]

                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue

                coords_i = coordinates[idx_i, :]
                coords_j = coordinates[idx_j, :]

                nbrs = NearestNeighbors(n_neighbors=min(1, len(idx_j)), algorithm='auto')
                nbrs.fit(coords_j)
                distances, indices_q = nbrs.kneighbors(coords_i)
                distances = distances.ravel() * ratio[k_idx]

                idx_within = (distances - interaction_range) < tol[k_idx]
                adj_spatial[i, j, k_idx] = (
                    len(np.unique(indices_q.ravel()[idx_within])) >= k_min
                ) * 1

                idx_contact = (distances - contact_range_val) < tol[k_idx]
                adj_contact[i, j, k_idx] = (
                    len(np.unique(indices_q.ravel()[idx_contact])) >= k_min
                ) * 1

                knn_i = np.unique(nn_ranked[idx_i, :].ravel())
                adj_contact_knn[i, j, k_idx] = (
                    len(set(knn_i) & set(idx_j)) >= k_min_contact
                ) * 1

                d_spatial[i, j, k_idx] = _fun_mean(distances)

    d_spatial_avg = np.nanmean(d_spatial, axis=2)
    adj_spatial_avg = np.mean(adj_spatial, axis=2)
    adj_contact_avg = np.mean(adj_contact, axis=2)
    adj_contact_knn_avg = np.mean(adj_contact_knn, axis=2)

    adj_spatial_avg[adj_spatial_avg > 0] = 1
    adj_contact_avg[adj_contact_avg > 0] = 1
    adj_contact_knn_avg[adj_contact_knn_avg > 0] = 1

    if do_symmetric:
        adj_spatial_avg = adj_spatial_avg * adj_spatial_avg.T
        adj_contact_avg = adj_contact_avg * adj_contact_avg.T
        adj_contact_knn_avg = adj_contact_knn_avg * adj_contact_knn_avg.T

    d_spatial_avg = (d_spatial_avg + d_spatial_avg.T) / 2

    adj_spatial_avg[adj_spatial_avg == 0] = np.nan
    d_spatial_avg = d_spatial_avg * adj_spatial_avg

    if contact_knn_k is not None:
        adj_contact_avg = adj_contact_knn_avg

    return {
        'd_spatial': d_spatial_avg,
        'adj_contact': adj_contact_avg,
    }


def communication_dataframe_to_network(communication, groups=None):
    """
    Convert long-form communication results to canonical network storage.

    Parameters
    ----------
    communication : DataFrame with columns: source, target, interaction_name, prob, pval, etc.
    groups : list of str, optional
        Group names. If None, inferred from DataFrame.

    Returns
    -------
    dict
        Network with CSR ``prob`` matrices and dense ``pval`` matrices keyed
        by L-R name, plus group and interaction metadata. The ``prob`` and
        ``pval`` dictionary keys define the L-R names and their order.
    """
    if groups is None:
        groups = sorted(
            set(
                communication['source'].unique().tolist()
                + communication['target'].unique().tolist()
            )
        )

    lr_names = communication['interaction_name'].unique().tolist()

    n_clusters = len(groups)
    n_LR = len(lr_names)

    prob = np.zeros((n_clusters, n_clusters, n_LR))
    pval = np.ones((n_clusters, n_clusters, n_LR))

    group_idx = {g: i for i, g in enumerate(groups)}
    lr_idx = {lr: i for i, lr in enumerate(lr_names)}

    for _, row in communication.iterrows():
        si = group_idx.get(row['source'])
        ti = group_idx.get(row['target'])
        li = lr_idx.get(row['interaction_name'])

        if si is not None and ti is not None and li is not None:
            prob[si, ti, li] = row['prob']
            pval[si, ti, li] = row.get('pval', 1.0)

    # Build interactions DataFrame
    interactions = communication[['ligand_complex', 'receptor_complex', 'interaction_name',
                                  'pathway_name', 'annotation']].drop_duplicates('interaction_name')
    interactions = interactions.rename(columns={'ligand_complex': 'ligand', 'receptor_complex': 'receptor'})

    return {
        'prob': matrix_dict_from_array(prob, lr_names, sparse_output=True),
        'pval': matrix_dict_from_array(pval, lr_names),
        'groups': groups,
        'interactions': interactions.reset_index(drop=True)
    }


# ---------------------------------------------------------------------------
# Spot-level spatial communication
# ---------------------------------------------------------------------------
def _spatial_inputs(cellchat):
    if str(cellchat.settings.get("datatype", "")).lower() != "spatial":
        raise ValueError("Spot-level communication requires a spatial CellChat object.")
    coordinates = np.asarray(cellchat.obsm.get("spatial"), dtype=float)
    if coordinates.shape != (cellchat.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("cellchat.obsm['spatial'] must be a finite n_spots x 2 matrix.")
    factors = cellchat.spatial.get("spatial_factors", {})
    if not {"ratio", "tol"}.issubset(factors):
        raise ValueError("cellchat.spatial['spatial_factors'] must contain ratio and tol.")
    ratios = np.atleast_1d(factors["ratio"]).astype(float)
    tolerances = np.atleast_1d(factors["tol"]).astype(float)
    if len(ratios) != len(tolerances) or np.any(ratios <= 0) or np.any(tolerances < 0):
        raise ValueError("Spatial ratio and tol must be aligned positive/non-negative vectors.")
    if "cellchat_dataset" in cellchat.obs:
        datasets = cellchat.obs["cellchat_dataset"].astype(str).to_numpy()
        levels = list(pd.unique(datasets))
    else:
        datasets = np.repeat("sample1", cellchat.n_obs)
        levels = ["sample1"]
    if len(ratios) == 1 and len(levels) > 1:
        ratios = np.repeat(ratios, len(levels))
        tolerances = np.repeat(tolerances, len(levels))
    if len(ratios) != len(levels):
        raise ValueError("Spatial ratio/tol must contain one value per cellchat_dataset.")
    return coordinates, datasets, levels, ratios, tolerances


def compute_spot_distances(cellchat, interaction_range: float = 250.0, contact_range: float = 10.0):
    """Compute within-sample spot distances and contact adjacency in micrometers."""
    if not np.isfinite(interaction_range) or interaction_range <= 0:
        raise ValueError("interaction_range must be a positive finite number.")
    if not np.isfinite(contact_range) or contact_range < 0:
        raise ValueError("contact_range must be a finite non-negative number.")
    coordinates, datasets, levels, ratios, tolerances = _spatial_inputs(cellchat)
    rows, cols, values, contact_rows, contact_cols = [], [], [], [], []
    for level, ratio, tolerance in zip(levels, ratios, tolerances):
        global_indices = np.flatnonzero(datasets == level)
        if not len(global_indices):
            continue
        local_coordinates = coordinates[global_indices]
        pairs = cKDTree(local_coordinates).query_pairs(
            (interaction_range + tolerance) / ratio, output_type="ndarray"
        )
        if pairs.size:
            source, target = global_indices[pairs[:, 0]], global_indices[pairs[:, 1]]
            distance_um = np.linalg.norm(
                local_coordinates[pairs[:, 0]] - local_coordinates[pairs[:, 1]], axis=1
            ) * ratio
            rows.extend(np.concatenate([source, target]).tolist())
            cols.extend(np.concatenate([target, source]).tolist())
            values.extend(np.concatenate([distance_um, distance_um]).tolist())
            contact_mask = distance_um <= contact_range + tolerance
            if np.any(contact_mask):
                contact_source, contact_target = source[contact_mask], target[contact_mask]
                contact_rows.extend(np.concatenate([contact_source, contact_target]).tolist())
                contact_cols.extend(np.concatenate([contact_target, contact_source]).tolist())
    shape = (cellchat.n_obs, cellchat.n_obs)
    distance = sparse.csr_matrix((values, (rows, cols)), shape=shape, dtype=float)
    contact = sparse.csr_matrix((np.ones(len(contact_rows)), (contact_rows, contact_cols)), shape=shape)
    contact.setdiag(1.0)
    contact.eliminate_zeros()
    return distance, contact


def _spatial_weight(distance, distance_use: bool, scale_distance: float):
    if not np.isfinite(scale_distance) or scale_distance <= 0:
        raise ValueError("scale_distance must be a positive finite number.")
    weight = distance.copy().tocsr()
    if weight.nnz:
        scaled = weight.data * scale_distance
        if distance_use and scaled.min() < 1:
            suggested = 1.0 / weight.data.min()
            raise ValueError(
                "The minimum scaled spatial distance must be at least 1. "
                f"Increase scale_distance to approximately {suggested:.4g}."
            )
        weight.data = 1.0 / scaled if distance_use else np.ones_like(scaled)
        diagonal_value = float(weight.data.max()) if distance_use else 1.0
    else:
        diagonal_value = 1.0
    weight.setdiag(diagonal_value)
    weight.eliminate_zeros()
    return weight


def _interaction_table(cellchat, lr_use):
    if lr_use is None:
        lr_use = cellchat.lr_pairs.get("significant")
    if not isinstance(lr_use, pd.DataFrame) or lr_use.empty:
        raise ValueError("No L-R pairs are available. Run identify_overexpressed_interactions first.")
    required = {"interaction_name", "ligand", "receptor"}
    missing = required.difference(lr_use.columns)
    if missing:
        raise ValueError(f"L-R interaction metadata is missing columns: {sorted(missing)}")
    interactions = lr_use.copy().reset_index(drop=True)
    interactions["interaction_name"] = interactions["interaction_name"].astype(str)
    if interactions["interaction_name"].duplicated().any():
        raise ValueError("interaction_name must be unique for spot-level communication.")
    return interactions


def _spot_expression(cellchat, interactions, raw_use):
    layer = cellchat.signaling if raw_use else cellchat.smoothed
    if layer is None:
        raise ValueError("The requested signaling expression layer is unavailable.")
    if sparse.issparse(layer):
        nonzero_genes = np.asarray(layer.getnnz(axis=0)).ravel() > 0
        data = layer[:, nonzero_genes].T.toarray().astype(float, copy=False)
    else:
        values = np.asarray(layer, dtype=float)
        nonzero_genes = np.any(values != 0, axis=0)
        data = values[:, nonzero_genes].T
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Signaling expression must contain finite non-negative values.")
    maximum = float(data.max(initial=0.0))
    if maximum > 0:
        data = data / maximum
    genes = cellchat.var_names.astype(str)[nonzero_genes].tolist()
    database = cellchat.database
    complex_input, cofactor_input = database.get("complex", pd.DataFrame()), database.get("cofactor", pd.DataFrame())
    ligand = _compute_expr_lr(interactions["ligand"].astype(str).tolist(), data, genes, complex_input)
    receptor = _compute_expr_lr(interactions["receptor"].astype(str).tolist(), data, genes, complex_input)
    receptor *= _compute_expr_coreceptor(cofactor_input, data, genes, interactions, "A")
    receptor /= _compute_expr_coreceptor(cofactor_input, data, genes, interactions, "I")
    return data, genes, ligand, receptor, cofactor_input


def _edge_probability(source_expression, target_expression, weight, kh, hill_n, modifier=None):
    coo = weight.tocoo()
    product = source_expression[coo.row] * target_expression[coo.col]
    powered = np.power(product, hill_n)
    values = powered / (np.power(kh, hill_n) + powered) * coo.data
    if modifier is not None:
        values *= modifier[coo.row] * modifier[coo.col]
    matrix = sparse.csr_matrix((values, (coo.row, coo.col)), shape=weight.shape)
    matrix.eliminate_zeros()
    return matrix


def compute_spot_communication_probability(
    cellchat, lr_use: pd.DataFrame | None = None, raw_use: bool = True, kh: float = 0.5,
    hill_n: float = 1.0, distance_use: bool = True, interaction_range: float = 250.0,
    scale_distance: float = 0.01, use_agonist_antagonist: bool = True,
    contact_dependent: bool = True, contact_range: float = 10.0,
    contact_dependent_forced: bool = False,
):
    """Infer one sparse spot-by-spot network per ligand-receptor interaction."""
    if not np.isfinite(kh) or kh <= 0 or not np.isfinite(hill_n) or hill_n <= 0:
        raise ValueError("kh and hill_n must be positive finite numbers.")
    interactions = _interaction_table(cellchat, lr_use)
    distance, contact = compute_spot_distances(cellchat, interaction_range, contact_range)
    spatial_weight = _spatial_weight(distance, distance_use, scale_distance)
    data, genes, ligand, receptor, cofactor_input = _spot_expression(cellchat, interactions, raw_use)
    probabilities = {}
    for index, row in interactions.iterrows():
        is_contact = contact_dependent_forced or (
            contact_dependent and str(row.get("annotation", "")) == "Cell-Cell Contact"
        )
        edge_weight = spatial_weight.multiply(contact) if is_contact else spatial_weight
        modifier = None
        if use_agonist_antagonist:
            agonist = _compute_expr_agonist(data, genes, interactions, cofactor_input, index, kh, hill_n)
            antagonist = _compute_expr_antagonist(data, genes, interactions, cofactor_input, index, kh, hill_n)
            modifier = np.asarray(agonist).ravel() * np.asarray(antagonist).ravel()
        probabilities[str(row["interaction_name"])] = _edge_probability(
            ligand[index], receptor[index], edge_weight, kh, hill_n, modifier
        )
    cellchat.spot_network = {
        "spots": cellchat.obs_names.astype(str).tolist(), "prob": probabilities,
        "interactions": interactions,
        "parameters": {
            "raw_use": bool(raw_use), "kh": float(kh), "hill_n": float(hill_n),
            "distance_use": bool(distance_use), "interaction_range": float(interaction_range),
            "scale_distance": float(scale_distance), "use_agonist_antagonist": bool(use_agonist_antagonist),
            "contact_dependent": bool(contact_dependent), "contact_range": float(contact_range),
            "contact_dependent_forced": bool(contact_dependent_forced),
        },
        "distance": distance, "spatial_weight": spatial_weight, "contact_adjacency": contact,
        "ligand_expression": ligand, "receptor_expression": receptor,
    }
    return cellchat


def filter_spot_probability(cellchat, n_boot: int = 100, seed_use: int = 666, thresh: float = 0.05):
    """Apply the SpatialCellChat empirical probability-quantile filter."""
    network = cellchat.spot_network
    if not network:
        raise ValueError("Run compute_spot_communication_probability first.")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer.")
    if not np.isfinite(thresh) or not 0 <= thresh <= 1:
        raise ValueError("thresh must be between 0 and 1.")
    if thresh == 1:
        return cellchat
    rng = np.random.default_rng(seed_use)
    n_spots = len(network["spots"])
    if n_boot > n_spots:
        raise ValueError("n_boot cannot exceed the number of spots when sampling without replacement.")
    interactions, filtered = network["interactions"], {}
    permutations = [rng.choice(n_spots, size=n_boot, replace=False) for _ in range(len(network["prob"]))]
    for index, (name, probability) in enumerate(network["prob"].items()):
        row = interactions.iloc[index]
        is_contact = network["parameters"]["contact_dependent_forced"] or (
            network["parameters"]["contact_dependent"] and str(row.get("annotation", "")) == "Cell-Cell Contact"
        )
        topology = network["contact_adjacency"] if is_contact else network["distance"].copy().tocsr()
        if not is_contact:
            topology.setdiag(1.0)
            topology.eliminate_zeros()
        values = []
        for source in permutations[index]:
            targets = topology.indices[topology.indptr[source]:topology.indptr[source + 1]]
            if len(targets):
                values.append(np.asarray(probability[source, targets].toarray()).ravel())
        cutoff = float(np.quantile(np.concatenate(values) if values else np.array([0.0]), 1.0 - thresh))
        matrix = probability.copy()
        if cutoff > 0:
            matrix.data[matrix.data < cutoff] = 0.0
            matrix.eliminate_zeros()
        filtered[name] = matrix
    updated = dict(network)
    updated["prob"] = filtered
    updated["parameters"] = dict(network["parameters"], probability_filter={"n_boot": int(n_boot), "seed_use": int(seed_use), "thresh": float(thresh)})
    cellchat.spot_network = updated
    return cellchat


def filter_spot_communication(cellchat, min_links: int | None = 5, min_spots: int | None = 5):
    """Remove spot L-R networks with too few links, senders, or receivers."""
    network = cellchat.spot_network
    if not network:
        raise ValueError("Run compute_spot_communication_probability first.")
    for name, value in (("min_links", min_links), ("min_spots", min_spots)):
        if value is not None and (not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be None or a positive integer.")
    filtered, n_spots = {}, len(network["spots"])
    for name, probability in network["prob"].items():
        keep = min_links is None or probability.nnz >= min_links
        if keep and min_spots is not None:
            keep = (np.count_nonzero(np.diff(probability.indptr)) >= min_spots and
                    np.count_nonzero(np.asarray((probability != 0).sum(axis=0)).ravel()) >= min_spots)
        filtered[name] = probability.copy() if keep else sparse.csr_matrix((n_spots, n_spots))
    updated = dict(network)
    updated["prob"] = filtered
    updated["parameters"] = dict(network["parameters"], communication_filter={"min_links": min_links, "min_spots": min_spots})
    cellchat.spot_network = updated
    return cellchat


def aggregate_visium_communication(
    cellchat, cell_type_decomposition: pd.DataFrame, average_type: str = "average",
    do_permutation: bool = True, n_boot: int = 100, seed_use: int = 1,
):
    """Aggregate spot networks to cell-type networks using Visium proportions."""
    network = cellchat.spot_network
    if not network:
        raise ValueError("Run compute_spot_communication_probability first.")
    if average_type not in {"average", "sum"}:
        raise ValueError("average_type must be 'average' or 'sum'.")
    if not isinstance(cell_type_decomposition, pd.DataFrame):
        raise TypeError("cell_type_decomposition must be a pandas DataFrame.")
    decomposition = cell_type_decomposition.copy()
    decomposition.index = decomposition.index.astype(str)
    if decomposition.index.tolist() != network["spots"]:
        raise ValueError("Decomposition rows must match spot names in the same order.")
    if decomposition.columns.duplicated().any():
        raise ValueError("Decomposition cell-type names must be unique.")
    values = decomposition.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Decomposition values must be finite and non-negative.")
    groups, one_hot = decomposition.columns.astype(str).tolist(), (values > 0).astype(float)

    def aggregate(probability, proportions, binary):
        numerator = proportions.T @ (probability @ proportions)
        if average_type == "sum":
            return np.asarray(numerator)
        adjacency = probability.copy()
        adjacency.data = np.ones_like(adjacency.data)
        denominator = binary.T @ (adjacency @ binary)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)

    names = list(network["prob"])
    aggregated = {name: aggregate(matrix, values, one_hot) for name, matrix in network["prob"].items()}
    if do_permutation:
        if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
            raise ValueError("n_boot must be a positive integer.")
        rng, pvalues = np.random.default_rng(seed_use), {}
        permutations = [rng.permutation(len(groups)) for _ in range(n_boot)]
        for name, probability in network["prob"].items():
            observed, exceed = aggregated[name], np.zeros_like(aggregated[name], dtype=float)
            for order in permutations:
                exceed += aggregate(probability, values[:, order], one_hot[:, order]) > observed
            pvalue = exceed / n_boot
            pvalue[observed == 0] = 1.0
            pvalues[name] = pvalue
    else:
        pvalues = {name: np.zeros_like(matrix) for name, matrix in aggregated.items()}
    cellchat.network = {
        "groups": groups, "prob": {name: sparse.csr_matrix(aggregated[name]) for name in names},
        "pval": pvalues, "interactions": network["interactions"].copy(),
    }
    cellchat.settings["spot_aggregation"] = {
        "average_type": average_type, "do_permutation": bool(do_permutation),
        "n_boot": int(n_boot), "seed_use": int(seed_use),
    }
    return cellchat


def compute_spot_pathway_probability(cellchat):
    """Sum spot L-R matrices into spot-level pathway matrices."""
    network = cellchat.spot_network
    if not network:
        raise ValueError("Run compute_spot_communication_probability first.")
    interactions = network["interactions"]
    if "pathway_name" not in interactions:
        raise ValueError("Spot interaction metadata must contain pathway_name.")
    pathways = list(pd.unique(interactions["pathway_name"].astype(str)))
    matrices = {}
    for pathway in pathways:
        names = interactions.loc[interactions["pathway_name"].astype(str) == pathway, "interaction_name"].astype(str)
        matrix = sum((network["prob"][name] for name in names), start=sparse.csr_matrix((len(network["spots"]), len(network["spots"]))))
        matrix.eliminate_zeros()
        if matrix.nnz:
            matrices[pathway] = matrix
    matrices = dict(sorted(matrices.items(), key=lambda item: -float(item[1].sum())))
    cellchat.spot_pathway_network = {
        "spots": list(network["spots"]), "prob": matrices,
        "interactions": pd.DataFrame({"interaction_name": list(matrices), "pathway_name": list(matrices)}),
        "parameters": {"source": "spot_network", "aggregation": "sum"},
    }
    return cellchat


def compute_spot_network_centrality(cellchat, slot_name: str = "spot_pathway_network"):
    """Compute weighted and unweighted in/out degree for every spot network."""
    if slot_name not in {"spot_network", "spot_pathway_network"}:
        raise ValueError("slot_name must be 'spot_network' or 'spot_pathway_network'.")
    network = getattr(cellchat, slot_name)
    if not network:
        raise ValueError(f"{slot_name} is empty.")
    names, spots = list(network["prob"]), network["spots"]
    measures = {key: np.zeros((len(spots), len(names)), dtype=float) for key in ("outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg")}
    for column, name in enumerate(names):
        matrix = network["prob"][name]
        measures["outdeg_unweighted"][:, column] = np.diff(matrix.indptr)
        measures["indeg_unweighted"][:, column] = np.asarray((matrix != 0).sum(axis=0)).ravel()
        measures["outdeg"][:, column] = np.asarray(matrix.sum(axis=1)).ravel()
        measures["indeg"][:, column] = np.asarray(matrix.sum(axis=0)).ravel()
    updated = dict(network)
    updated["centrality"] = {measure: pd.DataFrame(values, index=spots, columns=names) for measure, values in measures.items()}
    setattr(cellchat, slot_name, updated)
    return cellchat


