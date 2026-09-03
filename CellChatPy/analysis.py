#!/usr/bin/env python3
"""
Analysis functions for CellChat
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import umap
import warnings
from typing import Union, Optional, Dict, List, Tuple, Any
import logging
from collections.abc import Mapping, Sequence

from .network_storage import (
    is_matrix_dict,
    network_names,
    stack_network_field,
)

try:
    import geosketch
    geosketch_available = True
except ImportError:
    geosketch_available = False


def _zero_filled_sparse_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Restore scipy sparse zero semantics after pandas conversion."""
    if any(
        isinstance(dtype, pd.SparseDtype)
        and pd.isna(dtype.fill_value)
        for dtype in data.dtypes
    ):
        return data.fillna(0)
    return data


def _feature_by_cell_frame(cellchat, layer: str | None = "signaling") -> pd.DataFrame:
    """Create a temporary genes-by-cells view from canonical AnnData storage."""
    if layer is None or layer == "x":
        matrix = cellchat.X
    elif layer in {"raw", "signaling", "scaled", "smoothed"}:
        matrix = getattr(cellchat, layer)
    else:
        raise ValueError(
            "layer must be one of None, 'x', 'raw', 'signaling', 'scaled', or 'smoothed'."
        )
    if matrix is None:
        raise ValueError(f"Expression layer {layer!r} is not available.")
    if matrix.shape != cellchat.shape:
        raise ValueError(
            f"Expression layer {layer!r} must have cells x genes shape "
            f"{cellchat.shape}; got {matrix.shape}."
        )

    gene_mask = np.ones(cellchat.n_vars, dtype=bool)
    if layer == "signaling" and "is_signaling" in cellchat.var:
        gene_mask = cellchat.var["is_signaling"].fillna(False).to_numpy(dtype=bool)

    matrix = matrix[:, gene_mask]
    gene_names = pd.Index(cellchat.var_names[gene_mask].astype(str))
    cell_names = pd.Index(cellchat.obs_names.astype(str))
    if sparse.issparse(matrix):
        return pd.DataFrame.sparse.from_spmatrix(
            matrix.T.tocsr(),
            index=gene_names,
            columns=cell_names,
        )
    return pd.DataFrame(
        np.asarray(matrix).T,
        index=gene_names,
        columns=cell_names,
    )


def _get_pathway_arrays(cellchat, slot_name: str = "pathway_network"):
    """Return calculation arrays from one canonical network slot."""
    if slot_name not in {"pathway_network", "network"}:
        raise ValueError("slot_name must be 'pathway_network' or 'network'.")
    source_net = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if not isinstance(source_net, Mapping) or 'prob' not in source_net:
        raise ValueError(
            f"No probability data found in cellchat.{slot_name}. "
            "Run the corresponding probability computation first."
        )
    names = network_names(source_net)
    prob = stack_network_field(source_net, 'prob', names=names, fill_value=0.0)
    pval = stack_network_field(source_net, 'pval', names=names, fill_value=1.0)
    if not names:
        names = [f'network_{index}' for index in range(prob.shape[2])]
    groups = list(source_net.get('groups', [f'C{index}' for index in range(prob.shape[0])]))
    return prob, pval, names, groups


def _get_net_group_names(cellchat, net_data):
    groups = net_data.get('groups') if isinstance(net_data, Mapping) else None
    if groups is not None:
        return list(groups)
    try:
        return list(cellchat.groups.categories)
    except Exception:
        prob = net_data.get('prob') if isinstance(net_data, Mapping) else None
        if is_matrix_dict(prob):
            first = next(iter(prob.values()), None)
            n = first.shape[0] if first is not None else 0
        else:
            n = prob.shape[0] if prob is not None else 0
        return [f'group_{i}' for i in range(n)]


def _comparison_name(comparison: Optional[Union[List[int], Tuple[int, ...]]]) -> str:
    if comparison is None:
        return 'single'
    return "-".join(str(c) for c in comparison)


def _merged_dataset_keys(target: Dict[str, Any]) -> List[Any]:
    """Return dataset keys from a canonical merged network."""
    if not isinstance(target, dict):
        return []
    if {'groups', 'prob', 'pval'}.intersection(target):
        return []
    return [key for key, value in target.items() if isinstance(value, Mapping) and value]


def _network_similarity_data(cellchat, slot_name: str, similarity_type: str) -> Dict[str, Any]:
    """Return the similarity result for one slot and analysis type."""
    if slot_name not in {"network", "pathway_network"}:
        raise ValueError("slot_name must be 'pathway_network' or 'network'.")
    if similarity_type not in {"functional", "structural"}:
        raise ValueError("type must be 'functional' or 'structural'.")
    merged = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if isinstance(merged, Mapping) and not {'groups', 'prob', 'pval'}.intersection(merged):
        return cellchat.network_similarity.setdefault(slot_name, {}).setdefault(similarity_type, {})
    return merged.setdefault('similarity', {}).setdefault(similarity_type, {})


def _pathways_for_similarity(sim_data: Dict[str, Any], key: str, n_pts: int) -> List[str]:
    pathways = sim_data.get('pathways')
    if isinstance(pathways, dict):
        pathways = pathways.get(key)
    if pathways is None:
        pathways = sim_data.get('pathways_all')
    pathways = list(pathways) if pathways is not None else [f'P{i}' for i in range(n_pts)]
    if len(pathways) != n_pts:
        pathways = (pathways + [f'P{i}' for i in range(len(pathways), n_pts)])[:n_pts]
    return pathways


def _set_similarity_pathways(sim_data: Dict[str, Any], key: str, pathways: List[str]) -> None:
    if key == 'single':
        sim_data['pathways'] = list(pathways)
        return
    current = sim_data.get('pathways')
    if not isinstance(current, dict):
        current = {}
    current[key] = list(pathways)
    sim_data['pathways'] = current


def _build_pattern_matrix(
    cellchat,
    slot_name: str,
    pattern: str,
    thresh: Optional[float] = None,
):
    """Build the CellChat pattern matrix used by selectK/identifyCommunicationPatterns."""
    prob, pval, pathway_names, cluster_names = _get_pathway_arrays(cellchat, slot_name)
    prob_f = np.array(prob, dtype=float, copy=True)

    # R uses the stored probability tensor directly. Thresholding is opt-in.
    if thresh is not None and pval is not None and getattr(pval, 'shape', None) == prob_f.shape:
        prob_f[pval >= thresh] = 0.0

    n_clusters = len(cluster_names)
    n_pathways = len(pathway_names)
    mat = np.zeros((n_clusters, n_pathways), dtype=float)
    if prob_f.ndim == 3:
        if pattern == "outgoing":
            mat[:, :] = np.sum(prob_f, axis=1)
        elif pattern == "incoming":
            mat[:, :] = np.sum(prob_f, axis=0)
        else:
            raise ValueError("pattern must be 'outgoing' or 'incoming'")

    mat[mat < 0] = 0.0
    col_max = np.nanmax(mat, axis=0) if mat.size else np.array([])
    col_max[~np.isfinite(col_max) | (col_max == 0)] = 1.0
    data0 = mat / col_max
    data0[~np.isfinite(data0)] = 0.0

    row_keep = data0.sum(axis=1) != 0
    data_nmf = data0[row_keep, :]
    cluster_names_nmf = [cluster_names[i] for i, keep in enumerate(row_keep) if keep]
    return data0, data_nmf, cluster_names, cluster_names_nmf, pathway_names


def _nndsvd_init(mat: np.ndarray, n_components: int, eps: float = 1e-12):
    """NNDSVD initialization used before Lee-Seung NMF updates."""
    from scipy.linalg import svd

    X = np.asarray(mat, dtype=float)
    n_rows, n_cols = X.shape
    W = np.zeros((n_rows, n_components), dtype=float)
    H = np.zeros((n_components, n_cols), dtype=float)
    if X.size == 0 or n_components == 0:
        return W, H

    U, S, Vt = svd(X, full_matrices=False)
    rank = min(n_components, len(S))
    if rank == 0:
        return W, H

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

    for j in range(1, rank):
        x = U[:, j]
        y = Vt[j, :]
        xp = np.maximum(x, 0)
        xn = np.maximum(-x, 0)
        yp = np.maximum(y, 0)
        yn = np.maximum(-y, 0)

        xpn = np.linalg.norm(xp)
        ypn = np.linalg.norm(yp)
        xnn = np.linalg.norm(xn)
        ynn = np.linalg.norm(yn)
        mp = xpn * ypn
        mn = xnn * ynn

        if mp > mn:
            u = xp / (xpn + eps)
            v = yp / (ypn + eps)
            sigma = mp
        else:
            u = xn / (xnn + eps)
            v = yn / (ynn + eps)
            sigma = mn

        scale = np.sqrt(S[j] * sigma)
        W[:, j] = scale * u
        H[j, :] = scale * v

    # Lee-Seung multiplicative updates cannot move exact zeros.  R's NMF
    # implementation avoids the frozen-zero sklearn behaviour, so replace zeros
    # by a tiny data-scaled floor before updating.
    floor = max(float(X.mean()) * 1e-6, eps)
    W[W <= 0] = floor
    H[H <= 0] = floor
    return W, H


def _lee_nmf(
    mat: np.ndarray,
    n_components: int,
    seed_use: int = 1,
    init: str = "nndsvd",
    max_iter: int = 10000,
    tol: float = 1e-6,
    update_order: str = "HW",
):
    """Factorize mat ~= W @ H with Lee-Seung multiplicative updates."""
    X = np.asarray(mat, dtype=float)
    X = np.maximum(X, 0)
    eps = np.finfo(float).eps

    if init == "random":
        rng = np.random.default_rng(seed_use)
        avg = np.sqrt(max(X.mean(), eps) / max(n_components, 1))
        W = rng.random((X.shape[0], n_components)) * avg + eps
        H = rng.random((n_components, X.shape[1])) * avg + eps
    else:
        W, H = _nndsvd_init(X, n_components, eps=eps)

    prev_err = None
    if update_order not in {"HW", "WH"}:
        raise ValueError("update_order must be 'HW' or 'WH'")

    for _ in range(max_iter):
        if update_order == "WH":
            numerator = X @ H.T
            denominator = (W @ H @ H.T) + eps
            W *= numerator / denominator
            W = np.maximum(W, eps)

            numerator = W.T @ X
            denominator = (W.T @ W @ H) + eps
            H *= numerator / denominator
            H = np.maximum(H, eps)
        else:
            numerator = W.T @ X
            denominator = (W.T @ W @ H) + eps
            H *= numerator / denominator
            H = np.maximum(H, eps)

            numerator = X @ H.T
            denominator = (W @ H @ H.T) + eps
            W *= numerator / denominator
            W = np.maximum(W, eps)

        # Keep the arbitrary NMF component scale stable.  This does not change
        # W @ H, but matches the stable Lee-Seung trajectory used by R NMF more
        # closely and avoids pattern assignment drift in W.
        component_scale = W.sum(axis=0)
        component_scale[component_scale <= eps] = 1.0
        W = W / component_scale
        H = H * component_scale[:, np.newaxis]

        if tol is not None and tol > 0:
            err = np.linalg.norm(X - W @ H, ord='fro')
            if prev_err is not None:
                denom = max(prev_err, eps)
                if abs(prev_err - err) / denom < tol:
                    break
            prev_err = err

    return W, H


def normalize_data(
    data_raw: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
    scale_factor: float = 10000,
    do_log: bool = True,
    do_sparse: bool = True
) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Normalize a cells-by-genes expression matrix by each cell's library size.

    Parameters
    ----------
    data_raw : array-like
        Input raw data
    scale_factor : float
        Scaling factor for each cell
    do_log : bool
        Whether to apply log transformation
    do_sparse : bool
        Whether to return sparse matrix

    Returns
    -------
    array-like
        Normalized data
    """
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be a positive finite number.")
    if not hasattr(data_raw, "shape") or len(data_raw.shape) != 2:
        raise ValueError("data_raw must be a two-dimensional cells-by-genes matrix.")

    if sparse.issparse(data_raw):
        matrix = data_raw.tocsr().astype(float, copy=True)
        if matrix.nnz and (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)):
            raise ValueError("data_raw must contain finite non-negative counts.")
        library_size = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        matrix = np.asarray(data_raw, dtype=float)
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("data_raw must contain finite non-negative counts.")
        library_size = matrix.sum(axis=1)
    if np.any(library_size <= 0):
        raise ValueError("Every cell must have a positive library size.")

    if sparse.issparse(matrix):
        expr = sparse.diags(scale_factor / library_size) @ matrix
        expr = expr.tocsr()
        if do_log:
            expr.data = np.log1p(expr.data)
        return expr if do_sparse else expr.toarray()

    expr = matrix * (scale_factor / library_size)[:, None]
    if do_log:
        expr = np.log1p(expr)
    return sparse.csr_matrix(expr) if do_sparse else expr


def preprocess_signaling_data(
    cellchat: 'CellChat',
    n_components: Optional[int] = None,
    random_state: int = 1,
    clip_negative: bool = True,
) -> 'CellChat':
    """Apply a low-rank denoising pass to the signaling expression layer.

    The spatial workflow can apply a denoising pass after selecting signaling
    genes.  This implementation uses a randomized truncated-SVD reconstruction
    and replaces ``cellchat.signaling`` while keeping the original cells and
    genes unchanged.

    Parameters are deliberately Pythonic and do not mirror R's dotted names.
    ``n_components`` defaults to a conservative rank of at most 50; callers
    can lower it for large datasets or increase it when preserving more detail
    is important.
    """
    signaling = cellchat.signaling
    if signaling is None or not hasattr(signaling, "shape"):
        raise ValueError("CellChat object has no signaling expression layer. Run subset_signaling_data first.")
    n_obs, n_genes = signaling.shape
    if n_obs < 2 or n_genes < 2:
        return cellchat
    if sparse.issparse(signaling):
        matrix = signaling.tocsr().astype(float, copy=False)
        dense = matrix.toarray()
    else:
        dense = np.asarray(signaling, dtype=float)
    if not np.isfinite(dense).all() or np.any(dense < 0):
        raise ValueError("Signaling expression must contain finite non-negative values.")
    max_rank = min(n_obs, n_genes) - 1
    rank = max_rank if n_components is None else int(n_components)
    if rank < 1 or rank > max_rank:
        raise ValueError(f"n_components must be between 1 and {max_rank}.")
    model = TruncatedSVD(n_components=rank, n_iter=7, random_state=int(random_state))
    reconstructed = model.fit_transform(dense) @ model.components_
    if clip_negative:
        reconstructed[reconstructed < 0] = 0.0
    cellchat.signaling = sparse.csr_matrix(reconstructed)
    cellchat.settings.setdefault("preprocessing", {})["signaling_low_rank"] = {
        "method": "truncated_svd", "n_components": rank,
        "random_state": int(random_state), "clip_negative": bool(clip_negative),
    }
    return cellchat


def scale_data(
    data_use: Union[np.ndarray, sparse.spmatrix],
    do_center: bool = True
) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Scale the data using StandardScaler

    Parameters
    ----------
    data_use : array-like
        Input data
    do_center : bool
        Whether to center the data

    Returns
    -------
    array-like
        Scaled data
    """
    scaler = StandardScaler(with_mean=do_center, with_std=True)
    if sparse.issparse(data_use):
        data_use = data_use.toarray()

    return scaler.fit_transform(data_use.T).T


def scale_matrix(
    x: Union[np.ndarray, pd.DataFrame],
    scale: str,
    na_rm: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Scale a data matrix by rows or columns

    Parameters
    ----------
    x : array-like
        Data matrix
    scale : str
        Scaling method: 'none', 'row', 'column', 'r1', 'c1'
    na_rm : bool
        Whether to remove NA values

    Returns
    -------
    array-like
        Scaled matrix
    """
    valid_scales = ['none', 'row', 'column', 'r1', 'c1']
    if scale not in valid_scales:
        raise ValueError(f"scale must be one of {valid_scales}")

    if scale == 'none':
        return x

    x_array = np.array(x)

    if scale == 'row':
        # Row-wise scaling
        row_means = np.nanmean(x_array, axis=1, keepdims=True)
        x_centered = x_array - row_means
        row_sds = np.nanstd(x_centered, axis=1, keepdims=True)
        x_scaled = x_centered / row_sds

    elif scale == 'column':
        # Column-wise scaling
        col_means = np.nanmean(x_array, axis=0, keepdims=True)
        x_centered = x_array - col_means
        col_sds = np.nanstd(x_centered, axis=0, keepdims=True)
        x_scaled = x_centered / col_sds

    elif scale == 'r1':
        # Row-wise sum normalization
        row_sums = np.nansum(x_array, axis=1, keepdims=True)
        x_scaled = x_array / row_sums

    elif scale == 'c1':
        # Column-wise sum normalization
        col_sums = np.nansum(x_array, axis=0, keepdims=True)
        x_scaled = x_array / col_sums

    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    return x_scaled


def sketch_data(
    data: Union[np.ndarray, pd.DataFrame],
    percent: float,
    do_pca: bool = True,
    dim_pc: int = 30
) -> List[str]:
    """
    Downsample single cell data using geometric sketching algorithm

    Parameters
    ----------
    data : array-like
        Input data (cells in rows, features in columns)
    percent : float
        Percentage of data to sketch
    do_pca : bool
        Whether to perform PCA on input data
    dim_pc : int
        Number of principal components to use

    Returns
    -------
    list
        List of cell names to use for downsampling
    """
    if not geosketch_available:
        raise ImportError("geosketch package required. Install with: pip install geosketch")

    if do_pca:
        pca_result = run_pca(data, do_fast=True, dim_pc=dim_pc)
        X_pcs = pca_result
    else:
        X_pcs = data

    sketch_size = int(percent * X_pcs.shape[0])
    sketch_indices = geosketch.gs(X_pcs, sketch_size)

    if hasattr(data, 'index'):
        cell_names = data.index.tolist()
    else:
        cell_names = [f"cell_{i}" for i in range(data.shape[0])]

    sketch_cells = [cell_names[i] for i in sketch_indices]
    return sketch_cells


def add_metadata(
    cellchat: 'CellChat',
    metadata: Union[pd.DataFrame, Dict],
    metadata_name: Optional[str] = None
) -> 'CellChat':
    """
    Add cell metadata to CellChat object

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    metadata : DataFrame or dict
        Cell metadata to add
    metadata_name : str, optional
        Name for the metadata column

    Returns
    -------
    CellChat
        Updated CellChat object
    """
    if isinstance(metadata, dict):
        if metadata_name is None:
            raise ValueError("'metadata_name' must be provided for dict metadata types")
        metadata_frame = pd.DataFrame([metadata])
        metadata_frame.columns = [metadata_name]
    else:
        metadata_frame = pd.DataFrame(metadata)

    if metadata_name is not None:
        metadata_frame.columns = [metadata_name]

    cellchat.obs = metadata_frame
    return cellchat


def add_reduction(
    cellchat: 'CellChat',
    dr: Optional[pd.DataFrame] = None,
    dr_name: Optional[str] = None,
    seu_obj: Optional[Any] = None,
    dr_use: Optional[str] = None,
    force_add: bool = False
) -> 'CellChat':
    """
    Add reduced dimensional space to CellChat object

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    dr : DataFrame, optional
        Reduced dimensional coordinates
    dr_name : str, optional
        Name for the reduction
    seu_obj : object, optional
        Seurat object containing reductions
    dr_use : str, optional
        Specific reduction to use from Seurat object
    force_add : bool
        Whether to force adding new reduction

    Returns
    -------
    CellChat
        Updated CellChat object
    """
    if len(cellchat.obsm) > 0 and not force_add:
        existing_reductions = list(cellchat.obsm.keys())
        raise ValueError(f"object.obsm already contains: {existing_reductions}. Set force_add=True to add new reduction.")

    if dr is not None:
        if dr_name is None:
            raise ValueError("When providing dr, dr_name must also be provided")

        dr_df = pd.DataFrame(dr)
        cell_names = cellchat.X.columns.tolist()

        if not all(name in dr_df.index for name in cell_names):
            raise ValueError("Some cell barcodes missing from dr. Ensure all cells are included.")

        cellchat.obsm[dr_name] = dr_df.loc[cell_names]

    elif seu_obj is not None:
        # Handle Seurat object (simplified)
        if dr_use is not None:
            if hasattr(seu_obj, dr_use):
                reduction_data = getattr(seu_obj, dr_use)
                cellchat.obsm[dr_use] = pd.DataFrame(reduction_data)
        else:
            # Add all available reductions (simplified)
            pass

    return cellchat


def update_cluster_labels(
    cellchat: 'CellChat',
    old_cluster_name: Optional[List[str]] = None,
    new_cluster_name: Optional[List[str]] = None,
    new_order: Optional[List[str]] = None,
    new_cluster_metadata_name: str = "new_labels"
) -> 'CellChat':
    """
    Update and reorder cluster labels

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    old_cluster_name : list, optional
        Old cluster names
    new_cluster_name : list, optional
        New cluster names
    new_order : list, optional
        New order for clusters
    new_cluster_metadata_name : str
        Column name for new labels in cellchat metadata

    Returns
    -------
    CellChat
        Updated CellChat object
    """
    from .cellchat_class import set_identity

    if old_cluster_name is None:
        old_cluster_name = list(cellchat.groups.cat.categories)

    if new_cluster_metadata_name in cellchat.obs.columns:
        raise ValueError(
            f"Column '{new_cluster_metadata_name}' already exists in metadata"
        )

    if new_cluster_name is not None:
        # Create mapping from old to new names
        mapping = dict(zip(old_cluster_name, new_cluster_name))
        new_labels = [mapping.get(label, label) for label in cellchat.groups]
        cellchat.obs[new_cluster_metadata_name] = new_labels
        cellchat = set_identity(
            cellchat, group_by=new_cluster_metadata_name, display_warning=False
        )
    else:
        new_cluster_metadata_name = None
        print("Only reordering clusters, not renaming")

    if new_order is not None:
        if new_cluster_metadata_name is None:
            group_by = cellchat.groups.name or "cellchat_group"
        else:
            group_by = new_cluster_metadata_name
        cellchat = set_identity(cellchat, group_by=group_by,
                           levels=new_order, display_warning=False)

    print("Re-running downstream analyses...")
    # Note: These would need to be implemented
    # cellchat = compute_pathway_probability(cellchat)
    # cellchat = aggregate_network(cellchat)
    # cellchat = compute_network_centrality(cellchat)

    return cellchat


def subset_signaling_data(
    cellchat: 'CellChat',
    features: Optional[List[str]] = None
) -> 'CellChat':
    """
    Subset expression data to signaling genes.
    Subset expression data to genes required by the selected signaling database.
    """
    if features is None:
        database = cellchat.database
        if 'interaction' in database:
            interaction_data = database['interaction']

            datatype = cellchat.settings.get('datatype', 'RNA')
            if datatype != 'RNA':
                if 'annotation' not in interaction_data.columns:
                    warnings.warn("Adding 'annotation' column for spatial data.")
                    interaction_data['annotation'] = 'Secreted Signaling'

            if 'annotation' in interaction_data.columns:
                unique_annotations = interaction_data['annotation'].unique()
                if len(unique_annotations) > 1:
                    annotation_order = ['Secreted Signaling', 'ECM-Receptor', 'Non-protein Signaling', 'Cell-Cell Contact']
                    interaction_data['annotation'] = pd.Categorical(
                        interaction_data['annotation'], categories=annotation_order, ordered=True)
                    interaction_data = interaction_data.sort_values('annotation')
                    interaction_data['annotation'] = interaction_data['annotation'].astype(str)
                    cellchat.database['interaction'] = interaction_data

            # --- extract_genes logic: get actual gene symbols (expanding complexes & cofactors) ---
            complex_input = database.get('complex', pd.DataFrame())
            cofactor_input = database.get('cofactor', pd.DataFrame())
            gene_info = database.get('gene_info', pd.DataFrame())

            gene_symbols = set(gene_info['Symbol'].dropna()) if len(gene_info) > 0 and 'Symbol' in gene_info.columns else None

            def _expand_gene_set(gene_set):
                """Expand gene names: single genes are kept, complex names are replaced by subunits."""
                result = set()
                for g in gene_set:
                    if g == '' or pd.isna(g):
                        continue
                    # If it's a plain gene symbol -> keep as is
                    if gene_symbols is None or g in gene_symbols:
                        result.add(g)
                    elif len(complex_input) > 0 and g in complex_input.index:
                        # Complex: add all non-empty subunit values
                        sub_cols = [c for c in complex_input.columns if c.startswith('subunit')]
                        for sv in complex_input.loc[g, sub_cols]:
                            if isinstance(sv, str) and sv != '':
                                result.add(sv)
                    else:
                        # Unknown: keep anyway (may be in data)
                        result.add(g)
                return result

            # Ligands & receptors (expand complexes)
            ligands = list(interaction_data['ligand'].dropna().astype(str))
            receptors = list(interaction_data['receptor'].dropna().astype(str))
            gene_use = _expand_gene_set(ligands) | _expand_gene_set(receptors)

            # Cofactors (agonist, antagonist, co_A_receptor, co_I_receptor)
            if len(cofactor_input) > 0:
                cofactor_cols = [c for c in cofactor_input.columns if 'cofactor' in c.lower()]
                for col in ['agonist', 'antagonist', 'co_A_receptor', 'co_I_receptor']:
                    if col in interaction_data.columns:
                        for name in interaction_data[col].dropna():
                            if isinstance(name, str) and name != '' and name in cofactor_input.index:
                                for sv in cofactor_input.loc[name, cofactor_cols]:
                                    if isinstance(sv, str) and sv != '':
                                        gene_use.add(sv)

            # Intersect with available data genes
            # Handle both DataFrame and sparse matrix
            if hasattr(cellchat.X, 'index'):
                data_genes = set(cellchat.X.index)
            elif hasattr(cellchat, 'var_names'):
                data_genes = set(cellchat.var_names)
            else:
                raise ValueError("Cannot determine gene names from cellchat.X")
            features = list(gene_use & data_genes)
        else:
            # Get all genes
            if hasattr(cellchat.X, 'index'):
                features = list(cellchat.X.index)
            elif hasattr(cellchat, 'var_names'):
                features = list(cellchat.var_names)
            else:
                raise ValueError("Cannot determine gene names from cellchat.X")

    feature_names = {str(feature) for feature in features}
    gene_mask = np.asarray(
        pd.Index(cellchat.var_names.astype(str)).isin(feature_names),
        dtype=bool,
    )
    if sparse.issparse(cellchat.X):
        signaling = cellchat.X.multiply(gene_mask[np.newaxis, :]).tocsr()
        signaling.eliminate_zeros()
    else:
        signaling = np.array(cellchat.X, copy=True)
        signaling[:, ~gene_mask] = 0

    cellchat.signaling = signaling
    cellchat.var['is_signaling'] = gene_mask
    return cellchat


def _identify_over_expressed_genes_dataset(
    cellchat, group_dataset="cellchat_dataset", pos_dataset=None,
    features_name="features", only_pos=False,
    thresh_pc=0.1, thresh_fc=0.05, thresh_p=0.05, return_object=True,
    do_fast=True, group_de_combined=False,
):
    """Cross-dataset differential expression on a merged CellChat object.

    With ``group_de_combined=False``, each joint cell type is analysed
    separately: ``pos_dataset`` is compared with the other dataset(s) within
    that cell type.  With ``group_de_combined=True``, cell type labels are
    ignored during the test, so all cells from ``pos_dataset`` are compared
    with all cells from the other dataset(s); the resulting dataset-level
    markers are then associated with every joint cell type, matching the R
    ``group.DE.combined`` behavior.

    With ``do_fast=True`` this mirrors the R ``presto::wilcoxauc`` path:
    ``log_fc`` is the difference of the two group means on the supplied expression
    matrix, not the log of back-transformed means used by R's non-fast path.
    """
    data_matrix = _feature_by_cell_frame(cellchat, layer='signaling')
    if data_matrix is None or getattr(data_matrix, 'shape', (0,))[0] < 1:
        raise ValueError("Merged signaling layer is missing. Re-run merge_cellchat().")
    data_matrix = _zero_filled_sparse_frame(data_matrix)

    metadata = cellchat.obs
    if group_dataset not in metadata.columns:
        raise ValueError(f"'{group_dataset}' column not found in cellchat.obs.")
    dataset_labels = metadata[group_dataset].astype(str).values

    # joint cell-type labels per cell
    groups = cellchat.groups
    if groups is None or not hasattr(groups, 'categories'):
        raise ValueError("Merged object must define categorical cellchat.groups.")
    cell_types = np.asarray(groups)
    type_levels = list(groups.categories)

    datasets = list(dict.fromkeys(dataset_labels))
    if pos_dataset is None:
        pos_dataset = datasets[-1]
    if pos_dataset not in datasets:
        raise ValueError(
            "Please set pos_dataset to be one of the following dataset names: "
            f"{datasets}"
        )

    other_datasets = [d for d in datasets if d != pos_dataset]
    neg_dataset = ", ".join(other_datasets) if other_datasets else "other"
    # R rebuilds labels.dataset with factor levels c(pos.dataset, other), so
    # presto and the following sort use the requested comparison direction.
    dataset_order = [pos_dataset] + ([neg_dataset] if other_datasets else [])

    features_use = list(data_matrix.index)
    data_mat = data_matrix.values  # genes x cells

    def presto_pvalues(values, group_mask):
        """Match presto's Gaussian approximation of the Wilcoxon U test."""
        n_group = int(group_mask.sum())
        n_total = values.shape[1]
        n_other = n_total - n_group
        n1n2 = n_group * n_other

        ranks = stats.rankdata(values, axis=1, method="average")
        ustat = ranks[:, group_mask].sum(axis=1) - n_group * (n_group + 1) / 2
        z = ustat - 0.5 * n1n2
        z -= np.sign(z) * 0.5

        # presto's tie correction is calculated feature by feature after
        # ranking.  This is deliberately its normal approximation rather
        # than scipy's Mann-Whitney implementation.
        tie_term = np.zeros(values.shape[0], dtype=float)
        for i, row in enumerate(values):
            _, counts = np.unique(row, return_counts=True)
            tie_term[i] = np.sum(counts ** 3 - counts)
        variance_factor = (
            (n_total ** 3 - n_total - tie_term)
            / (12.0 * (n_total ** 2 - n_total))
        )
        sigma = np.sqrt(n1n2 * variance_factor)
        pvalues = np.ones(values.shape[0], dtype=float)
        valid = sigma > 0
        pvalues[valid] = 2.0 * stats.norm.sf(np.abs(z[valid] / sigma[valid]))
        return pvalues

    def bh_adjust(pvalues):
        """R stats::p.adjust(method='BH') for one presto result table."""
        n = len(pvalues)
        if n == 0:
            return pvalues
        order = np.argsort(pvalues)[::-1]
        ranked = pvalues[order] * n / np.arange(n, 0, -1)
        adjusted = np.minimum.accumulate(ranked)
        result = np.empty(n, dtype=float)
        result[order] = np.minimum(adjusted, 1.0)
        return result

    rows = []
    comparison_levels = type_levels if not group_de_combined else [None]
    for ct in comparison_levels:
        # ``group_de_combined`` deliberately removes cell-type information
        # from the statistical comparison. Results are expanded to each joint
        # cell type when rows are created below.
        ct_mask = np.ones(cell_types.shape[0], dtype=bool) if group_de_combined else cell_types == ct
        pos_mask = ct_mask & (dataset_labels == pos_dataset)
        neg_mask = ct_mask & (dataset_labels != pos_dataset)
        n1, n2 = int(pos_mask.sum()), int(neg_mask.sum())
        if n1 < 2 or n2 < 2:
            continue

        d1 = data_mat[:, pos_mask]
        d2 = data_mat[:, neg_mask]

        if do_fast:
            # R: presto::wilcoxauc(data.use.i, labels.i).  It emits one row
            # per feature for *each* group, so preserve both directions and
            # let netMappingDEG's first-match behavior use the ordered
            # pos_dataset row. For group.DE.combined=TRUE, R keeps the first
            # half of the combined presto result, i.e. the pos_dataset rows.
            values = np.concatenate([d1, d2], axis=1)
            pos_group_mask = np.r_[
                np.ones(n1, dtype=bool),
                np.zeros(n2, dtype=bool),
            ]
            neg_group_mask = np.r_[
                np.zeros(n1, dtype=bool),
                np.ones(n2, dtype=bool),
            ]
            group_specs = [(pos_dataset, pos_group_mask)]
            if not group_de_combined:
                group_specs.append((neg_dataset, neg_group_mask))
            for dataset, group_mask in group_specs:
                in_values = values[:, group_mask]
                out_values = values[:, ~group_mask]
                pct1 = 100.0 * np.mean(in_values > 0, axis=1)
                pct2 = 100.0 * np.mean(out_values > 0, axis=1)
                fc = np.mean(in_values, axis=1) - np.mean(out_values, axis=1)
                pvals = presto_pvalues(values, group_mask)
                pval_adj = bh_adjust(pvals)
                keep = (
                    (pvals < thresh_p)
                    & (np.abs(fc) >= thresh_fc)
                    & (np.maximum(pct1, pct2) > thresh_pc * 100.0)
                )
                if only_pos:
                    keep &= fc > 0
                for gi in np.flatnonzero(keep):
                    result_clusters = type_levels if group_de_combined else [ct]
                    for result_cluster in result_clusters:
                        rows.append({
                            'clusters': result_cluster,
                            'features': features_use[gi],
                            'pvalues': pvals[gi],
                            'log_fc': fc[gi],
                            'pct_1': pct1[gi],
                            'pct_2': pct2[gi],
                            'pvalues_adj': pval_adj[gi],
                            'cellchat_dataset': dataset,
                        })
        else:
            # R's non-fast path tests LS against all other datasets once and
            # assigns the dataset label from the sign of that contrast.
            pct1 = np.round(np.sum(d1 > 0, axis=1) / n1, 3)
            pct2 = np.round(np.sum(d2 > 0, axis=1) / n2, 3)
            keep = np.maximum(pct1, pct2) > thresh_pc
            if not keep.any():
                continue
            idx = np.flatnonzero(keep)
            m1 = np.log(np.mean(np.expm1(d1[idx]), axis=1) + 1)
            m2 = np.log(np.mean(np.expm1(d2[idx]), axis=1) + 1)
            fc = m1 - m2
            keep_fc = fc > thresh_fc if only_pos else np.abs(fc) > thresh_fc
            idx, fc = idx[keep_fc], fc[keep_fc]
            if len(idx) == 0:
                continue
            pvals = np.array([
                stats.mannwhitneyu(d1[gi], d2[gi], alternative='two-sided', use_continuity=True).pvalue
                for gi in idx
            ])
            pval_adj = np.minimum(pvals * len(features_use), 1.0)
            for kk, gi in enumerate(idx):
                if pvals[kk] < thresh_p:
                    result_clusters = type_levels if group_de_combined else [ct]
                    for result_cluster in result_clusters:
                        rows.append({
                            'clusters': result_cluster,
                            'features': features_use[gi],
                            'pvalues': pvals[kk],
                            'log_fc': fc[kk],
                            'pct_1': pct1[gi],
                            'pct_2': pct2[gi],
                            'pvalues_adj': pval_adj[kk],
                            'cellchat_dataset': pos_dataset if fc[kk] > 0 else neg_dataset,
                        })

    if rows:
        markers_all = pd.DataFrame(rows)
        markers_all['cellchat_dataset'] = pd.Categorical(
            markers_all['cellchat_dataset'], categories=dataset_order, ordered=True,
        )
        markers_all = markers_all.sort_values(
            ['cellchat_dataset', 'pvalues', 'log_fc'], ascending=[True, True, False]
        ).reset_index(drop=True)
        markers_all['cellchat_dataset'] = markers_all['cellchat_dataset'].astype(str)
    else:
        markers_all = pd.DataFrame(
            columns=['clusters', 'features', 'pvalues', 'log_fc', 'pct_1', 'pct_2',
                     'pvalues_adj', 'cellchat_dataset'])

    features_sig = list(markers_all['features'].unique()) if len(markers_all) > 0 else []
    cellchat.feature_results[features_name] = features_sig
    cellchat.feature_results[f"{features_name}_info"] = markers_all

    return cellchat if return_object else markers_all


def identify_overexpressed_genes(
    cellchat: 'CellChat',
    data_use: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    group_by: Optional[str] = None,
    groups_use: Optional[List[str]] = None,
    invert: bool = False,
    group_dataset: Optional[str] = None,
    pos_dataset: Optional[str] = None,
    group_de_combined: bool = False,
    features_name: str = "features",
    only_pos: bool = True,
    features: Optional[List[str]] = None,
    return_object: bool = True,
    thresh_pc: float = 0.0,
    thresh_fc: float = 0.0,
    thresh_p: float = 0.05,
    do_de: bool = True,
    do_fast: bool = True,
    min_cells: int = 10
) -> Union['CellChat', pd.DataFrame]:
    """
    Identify over-expressed signaling genes per cell group.
    The default matches R identifyOverExpressedGenes(do.fast=TRUE).

    When ``group_dataset`` is provided for a merged object,
    ``group_de_combined=False`` compares datasets within each joint cell
    type.  Setting it to ``True`` ignores cell type during the DE test and
    associates the combined dataset-level result with every joint cell type,
    matching R's ``group.DE.combined`` behavior.
    """
    if not isinstance(cellchat.feature_results, Mapping):
        raise ValueError("Please update CellChat object first")

    # Cross-dataset DE on a merged object (R: group.dataset="datasets")
    if group_dataset is not None:
        return _identify_over_expressed_genes_dataset(
            cellchat, group_dataset=group_dataset, pos_dataset=pos_dataset,
            features_name=features_name, only_pos=only_pos,
            thresh_pc=thresh_pc, thresh_fc=thresh_fc, thresh_p=thresh_p,
            return_object=return_object, do_fast=do_fast,
            group_de_combined=group_de_combined,
        )

    if data_use is None:
        data_matrix = _feature_by_cell_frame(cellchat, layer='signaling')
        if data_matrix is None or getattr(data_matrix, 'shape', (0,))[0] < 3:
            raise ValueError("The signaling layer is missing or too small. Run subset_signaling_data first.")
    else:
        data_matrix = pd.DataFrame(data_use)

    if features is None:
        features_use = list(data_matrix.index)
    else:
        features_use = list(set(features).intersection(set(data_matrix.index)))

    data_subset = _zero_filled_sparse_frame(data_matrix.loc[features_use])

    if do_de:
        # Get cell group labels
        if group_by is None:
            labels = cellchat.groups
        else:
            labels = cellchat.obs[group_by]

        labels = pd.Categorical(labels)
        level_use = [lv for lv in labels.categories if lv in labels]

        if groups_use is not None:
            if invert:
                level_use = [lv for lv in level_use if lv not in groups_use]
            else:
                level_use = [lv for lv in level_use if lv in groups_use]

        labels_arr = np.array(labels)

        # R's mean function for logFC: log(mean(expm1(x)) + 1)
        def mean_fxn(x):
            return np.log(np.mean(np.expm1(x.astype(float))) + 1)

        markers_list = []

        for cluster in level_use:
            cell_use1 = np.where(labels_arr == cluster)[0]
            cell_use2 = np.setdiff1d(np.arange(len(labels_arr)), cell_use1)

            if len(cell_use1) < 2 or len(cell_use2) < 2:
                continue

            data_mat = data_subset.values  # (n_features, n_cells)

            # Percentage filters (thresh.min = 0 in R)
            pct_1 = np.round(np.sum(data_mat[:, cell_use1] > 0, axis=1) / len(cell_use1), 3)
            pct_2 = np.round(np.sum(data_mat[:, cell_use2] > 0, axis=1) / len(cell_use2), 3)
            alpha_min = np.maximum(pct_1, pct_2)
            features_pass_pc = [features_use[i] for i in range(len(features_use)) if alpha_min[i] > thresh_pc]

            if len(features_pass_pc) == 0:
                continue

            fi = [features_use.index(f) for f in features_pass_pc]
            data1_pass = data_mat[fi, :][:, cell_use1]
            data2_pass = data_mat[fi, :][:, cell_use2]

            # logFC: R uses log(mean(expm1(x)) + 1) difference
            data_1_mean = np.array([mean_fxn(data1_pass[j, :]) for j in range(len(features_pass_pc))])
            data_2_mean = np.array([mean_fxn(data2_pass[j, :]) for j in range(len(features_pass_pc))])
            fc = data_1_mean - data_2_mean

            if only_pos:
                features_diff = [features_pass_pc[j] for j in range(len(features_pass_pc)) if fc[j] > thresh_fc]
            else:
                features_diff = [features_pass_pc[j] for j in range(len(features_pass_pc)) if abs(fc[j]) > thresh_fc]

            features_final = [f for f in features_pass_pc if f in features_diff]
            if len(features_final) == 0:
                continue

            fi2 = [features_use.index(f) for f in features_final]
            data1_final = data_mat[fi2, :][:, cell_use1]
            data2_final = data_mat[fi2, :][:, cell_use2]
            fc_final = fc[[features_pass_pc.index(f) for f in features_final]]
            pct1_final = pct_1[[features_use.index(f) for f in features_final]]
            pct2_final = pct_2[[features_use.index(f) for f in features_final]]

            # Wilcoxon rank-sum test (R: wilcox.test two-sided)
            pvals = np.ones(len(features_final))
            for k in range(len(features_final)):
                try:
                    _, pval = stats.mannwhitneyu(
                        data1_final[k, :], data2_final[k, :],
                        alternative='two-sided', use_continuity=True
                    )
                    pvals[k] = pval
                except Exception:
                    pvals[k] = 1.0

            # Bonferroni correction (R: p.adjust(method="bonferroni", n=nrow(X)))
            n_total = len(features_use)
            pval_adj = np.minimum(pvals * n_total, 1.0)

            cluster_df = pd.DataFrame({
                'clusters': cluster,
                'features': features_final,
                'pvalues': pvals,
                'log_fc': fc_final,
                'pct_1': pct1_final,
                'pct_2': pct2_final,
                'pvalues_adj': pval_adj
            })
            cluster_df = cluster_df[cluster_df['pvalues'] < thresh_p]
            if only_pos:
                cluster_df = cluster_df[cluster_df['log_fc'] > 0]

            if len(cluster_df) > 0:
                cluster_df = cluster_df.sort_values(['pvalues', 'log_fc'], ascending=[True, False])
                markers_list.append(cluster_df)

        if markers_list:
            markers_all = pd.concat(markers_list, ignore_index=True)
        else:
            markers_all = pd.DataFrame(
                columns=['clusters', 'features', 'pvalues', 'log_fc', 'pct_1', 'pct_2', 'pvalues_adj'])

    else:
        # do_de = False: select genes expressed in at least min_cells cells
        n_cells_expr = np.sum(data_subset.values > 0, axis=1)
        features_arr = np.array(features_use)
        mask = n_cells_expr >= min_cells
        markers_all = pd.DataFrame({
            'features': features_arr[mask],
            'n_cells': n_cells_expr[mask]
        })

    # Store results
    features_sig = list(markers_all['features'].unique()) if len(markers_all) > 0 else []
    cellchat.feature_results[features_name] = features_sig
    cellchat.feature_results[f"{features_name}_info"] = markers_all

    if return_object:
        return cellchat
    else:
        return markers_all


def identify_overexpressed_ligand_receptor(
    cellchat: 'CellChat',
    features_name: str = "features",
    features: Optional[List[str]] = None,
    return_object: bool = True
) -> Union['CellChat', pd.DataFrame]:
    """
    Identify over-expressed ligands and receptors

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    features_name : str
        Name of features to use
    features : list, optional
        Specific features to use
    return_object : bool
        Whether to return CellChat object

    Returns
    -------
    CellChat or DataFrame
        Updated CellChat object or results DataFrame
    """
    features_name_lr = f"{features_name}_lr_pairs"
    features_name_info = f"{features_name}_info"

    database = cellchat.database
    interaction_input = database.get('interaction', pd.DataFrame())
    complex_input = database.get('complex', pd.DataFrame())

    if features is None:
        if isinstance(cellchat.feature_results, Mapping) and features_name_info in cellchat.feature_results:
            markers_all = cellchat.feature_results[features_name_info]
        else:
            raise ValueError("Please update CellChat object first")
    else:
        # Filter markers by provided features
        markers_all = cellchat.feature_results[features_name_info]
        markers_all = markers_all[markers_all['features'].isin(features)]

    # Process ligand-receptor pairs
    if len(interaction_input) > 0:
        pair_lr = interaction_input[['ligand', 'receptor']].dropna()
        lr_use = set(pair_lr['ligand']).union(set(pair_lr['receptor']))
    else:
        lr_use = set()

    # Handle complex subunits
    markers_all_new = []
    for _, marker in markers_all.iterrows():
        feature = marker['features']
        if feature in lr_use:
            markers_all_new.append(marker)
        elif len(complex_input) > 0:
            # Check if feature is part of a complex
            complex_subunits = complex_input.filter(like='subunit').dropna(how='all')
            for _, complex_row in complex_subunits.iterrows():
                subunits = [s for s in complex_row if pd.notna(s) and s != '']
                if feature in subunits:
                    # Add the complex as a feature
                    marker_copy = marker.copy()
                    marker_copy['features'] = complex_row.name  # Complex name
                    markers_all_new.append(marker_copy)

    if markers_all_new:
        markers_all_new = pd.DataFrame(markers_all_new)
    else:
        markers_all_new = pd.DataFrame()

    cellchat.feature_results[features_name_lr] = markers_all_new

    if return_object:
        return cellchat
    else:
        return markers_all_new


def run_umap(
    data_use: Union[np.ndarray, pd.DataFrame],
    n_neighbors: int = 30,
    n_components: int = 2,
    metric: str = "correlation",
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    min_dist: float = 0.3,
    spread: float = 1.0,
    seed_use: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction

    Parameters
    ----------
    data_use : array-like
        Input data
    n_neighbors : int
        Number of neighbors
    n_components : int
        Number of components
    metric : str
        Distance metric
    n_epochs : int, optional
        Number of epochs
    learning_rate : float
        Learning rate
    min_dist : float
        Minimum distance
    spread : float
        Spread parameter
    seed_use : int
        Random seed

    Returns
    -------
    array-like
        UMAP coordinates
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")

    np.random.seed(seed_use)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        min_dist=min_dist,
        spread=spread,
        random_state=seed_use
    )

    umap_result = reducer.fit_transform(data_use)

    # Add row and column names if input was DataFrame
    if isinstance(data_use, pd.DataFrame):
        umap_result = pd.DataFrame(
            umap_result,
            index=data_use.index,
            columns=[f'UMAP{i+1}' for i in range(umap_result.shape[1])]
        )

    return umap_result


def run_pca(
    data_use: Union[np.ndarray, pd.DataFrame],
    do_fast: bool = True,
    dim_pc: int = 50,
    seed_use: int = 42,
    weight_by_var: bool = True
) -> np.ndarray:
    """
    Run PCA on data

    Parameters
    ----------
    data_use : array-like
        Input data
    do_fast : bool
        Whether to use fast PCA implementation
    dim_pc : int
        Number of principal components
    seed_use : int
        Random seed
    weight_by_var : bool
        Whether to weight by variance

    Returns
    -------
    array-like
        PCA coordinates
    """
    np.random.seed(seed_use)

    if do_fast:
        from sklearn.decomposition import TruncatedSVD
        dim_pc = min(dim_pc, data_use.shape[1] - 1)
        pca = TruncatedSVD(n_components=dim_pc, random_state=seed_use)
        pc_scores = pca.fit_transform(data_use)

        if weight_by_var:
            pc_scores = pc_scores * pca.singular_values_
    else:
        from sklearn.decomposition import PCA
        dim_pc = min(dim_pc, data_use.shape[1] - 1)
        pca = PCA(n_components=dim_pc, random_state=seed_use)
        pc_scores = pca.fit_transform(data_use)

        if weight_by_var:
            pc_scores = pc_scores * pca.explained_variance_

    # Add row and column names if input was DataFrame
    if isinstance(data_use, pd.DataFrame):
        pc_scores = pd.DataFrame(
            pc_scores,
            index=data_use.index,
            columns=[f'PC{i+1}' for i in range(pc_scores.shape[1])]
        )

    return pc_scores


def smooth_data(
    cellchat: 'CellChat',
    method: str = "netSmooth",
    adj: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    normalize_adj_matrix: str = 'rows'
) -> 'CellChat':
    """
    Smooth gene expression data using network projection

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    method : str
        Smoothing method ('netSmooth')
    adj : array-like, optional
        Adjacency matrix of protein-protein interaction network
    alpha : float
        Smoothing parameter (0-1)
    normalize_adj_matrix : str
        How to normalize adjacency matrix ('rows' or 'columns')

    Returns
    -------
    CellChat
        Updated CellChat object with smoothed data
    """
    if cellchat.smoothed is None:
        raise ValueError("CellChat object has no smooth expression layer. Update object first.")

    if method == "netSmooth":
        if adj is None:
            raise ValueError("Adjacency matrix required for netSmooth method")

        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")

        if np.any(np.sum(adj, axis=0) == 0) or np.any(np.sum(adj, axis=1) == 0):
            raise ValueError("PPI network cannot have zero rows/columns")

        data_df = _feature_by_cell_frame(cellchat, layer='signaling')
        data = data_df.sparse.to_dense().to_numpy() if any(
            isinstance(dtype, pd.SparseDtype) for dtype in data_df.dtypes
        ) else data_df.to_numpy()

        # Project and recombine data
        data_projected = _project_and_recombine(data, adj, alpha, normalize_adj_matrix)
        cellchat.smoothed = data_projected.T

    return cellchat


def _project_and_recombine(
    gene_expression: np.ndarray,
    adj_matrix: np.ndarray,
    alpha: float,
    normalize_adj_matrix: str
) -> np.ndarray:
    """Helper function for network projection"""
    # Normalize adjacency matrix
    if normalize_adj_matrix == 'rows':
        adj_norm = adj_matrix / np.sum(adj_matrix, axis=1, keepdims=True)
    else:
        adj_norm = adj_matrix / np.sum(adj_matrix, axis=0, keepdims=True)

    # Create projection matrix
    eye = np.eye(adj_matrix.shape[0])
    aa = eye - alpha * adj_norm
    bb = (1 - alpha) * gene_expression

    # Solve linear system
    try:
        result = np.linalg.solve(aa, bb)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if matrix is singular
        result = np.linalg.lstsq(aa, bb, rcond=None)[0]

    return result

def identify_overexpressed_interactions(
    cellchat: 'CellChat',
    features_name: str = "features",
    variable_both: bool = True,
    features: Optional[List[str]] = None,
    return_object: bool = True
) -> Union['CellChat', pd.DataFrame]:
    """
    Identify over-expressed ligand-receptor interactions.
    Identify ligand-receptor interactions supported by over-expressed genes.
    """
    signaling_data = _feature_by_cell_frame(cellchat, layer='signaling')
    gene_use = list(signaling_data.index)
    database = cellchat.database

    if features is None:
        if isinstance(cellchat.feature_results, Mapping):
            features_sig = cellchat.feature_results.get(features_name, [])
        else:
            raise ValueError("Please update CellChat object first")
    else:
        features_sig = list(features)

    interaction_input = database.get('interaction', pd.DataFrame())
    complex_input = database.get('complex', pd.DataFrame())

    if len(interaction_input) == 0:
        cellchat.lr_pairs['significant'] = pd.DataFrame()
        print("No interaction data available")
        return cellchat if return_object else pd.DataFrame()

    # Build complex subunit lookup
    sub_cols = [c for c in complex_input.columns if c.startswith('subunit')] if len(complex_input) > 0 else []

    def _get_complex_subunits(complex_name):
        if len(complex_input) == 0 or complex_name not in complex_input.index:
            return []
        return [v for v in complex_input.loc[complex_name, sub_cols] if isinstance(v, str) and v != '']

    # For a given gene name (single gene or complex), check:
    #   1. Whether it is significant (in features_sig or is a complex with a sig subunit)
    #      AND all its subunits are present in gene_use
    # Build: complexSubunits.sig = complexes where:
    #   - at least one subunit in features_sig
    #   - all subunits in gene_use
    complex_sig_names = set()
    complex_use_names = set()   # all complexes with all subunits in gene_use

    if len(complex_input) > 0:
        for cname in complex_input.index:
            subunits = _get_complex_subunits(cname)
            if len(subunits) == 0:
                continue
            all_in_gene_use = all(s in gene_use for s in subunits)
            if all_in_gene_use:
                complex_use_names.add(cname)
                has_sig = any(s in features_sig for s in subunits)
                if has_sig:
                    complex_sig_names.add(cname)

    # Now filter interaction_input
    # variable_both=True: BOTH ligand and receptor must be in (features_sig | complexSubunits.sig)
    # variable_both=False: at least one must be significant, AND both must be available
    pair_lr = interaction_input[['ligand', 'receptor']].copy()
    sig_and_complex = set(features_sig) | complex_sig_names
    avail_and_complex = set(gene_use) | complex_use_names

    index_sig = []
    for i in range(len(pair_lr)):
        ligand = str(pair_lr.iloc[i]['ligand'])
        receptor = str(pair_lr.iloc[i]['receptor'])
        if variable_both:
            # Both ligand and receptor in (features_sig + complexSubunits.sig)
            if ligand in sig_and_complex and receptor in sig_and_complex:
                index_sig.append(i)
        else:
            # Both available, at least one significant
            both_avail = ligand in avail_and_complex and receptor in avail_and_complex
            one_sig = ligand in sig_and_complex or receptor in sig_and_complex
            if both_avail and one_sig:
                index_sig.append(i)

    pair_lr_sig = interaction_input.iloc[index_sig].copy()
    cellchat.lr_pairs['significant'] = pair_lr_sig
    print(f"The number of highly variable ligand-receptor pairs used for signaling inference is {len(pair_lr_sig)}")

    if return_object:
        return cellchat
    else:
        return pair_lr_sig


# ---------------------------------------------------------------------------
# extractEnrichedLR
# ---------------------------------------------------------------------------
def extract_enriched_lr(
    cellchat: 'CellChat',
    signaling: Union[str, List[str]],
    gene_lr_return: bool = False,
    enriched_only: bool = True,
    thresh: float = 0.05,
    slot_name: str = 'network',
) -> Union[pd.DataFrame, Dict[str, Union[pd.DataFrame, List[str]]]]:
    """
    Extract ligand-receptor pairs and genes for one or more signaling pathways.

    This follows R ``extractEnrichedLR()``: candidates come from
    ``lr_pairs["significant"]``. ``slot_name="network"`` selects active
    group-level interactions, while ``slot_name="spot_network"`` selects
    active individual-spot interactions;
    with ``enriched_only=True`` only candidates with a non-zero, thresholded
    communication probability are returned.  Set ``gene_lr_return=True`` to
    additionally return ligand and receptor genes, with complexes expanded to
    their subunits.
    """
    if isinstance(signaling, str):
        signaling = [signaling]
    else:
        signaling = list(signaling)
    if not signaling:
        raise ValueError("signaling must contain at least one pathway name.")
    if slot_name not in {'network', 'spot_network'}:
        raise ValueError("slot_name must be 'network' or 'spot_network'.")

    def _ordered_unique(values):
        return list(dict.fromkeys(values))

    database = cellchat.database
    complex_input = database.get('complex', pd.DataFrame())
    gene_info = database.get('gene_info', pd.DataFrame())
    gene_symbols = (
        set(gene_info['Symbol'].dropna().astype(str))
        if 'Symbol' in gene_info.columns else set()
    )
    subunit_cols = (
        [col for col in complex_input.columns if col.startswith('subunit')]
        if len(complex_input) > 0 else []
    )

    def _extract_gene_subset(gene_values):
        """Extract the requested genes while preserving database row order."""
        result = []
        for value in _ordered_unique(str(gene) for gene in gene_values if pd.notna(gene)):
            if value in gene_symbols:
                result.append(value)
            elif len(complex_input) > 0 and value in complex_input.index:
                for subunit in complex_input.loc[value, subunit_cols]:
                    if isinstance(subunit, str) and subunit != '':
                        result.append(subunit)
        return _ordered_unique(result)

    def _single_dataset_result(net_data, lr_data, pathway):
        if not isinstance(lr_data, Mapping):
            raise ValueError("L-R data must contain a 'significant' DataFrame.")
        lr_sig = lr_data.get('significant')
        if not isinstance(lr_sig, pd.DataFrame):
            raise ValueError(
                "No LR pairs found. Run identify_overexpressed_interactions first."
            )
        required_columns = {'interaction_name', 'pathway_name', 'ligand', 'receptor'}
        missing_columns = required_columns.difference(lr_sig.columns)
        if missing_columns:
            raise ValueError(
                "significant L-R pairs are missing required columns: "
                f"{sorted(missing_columns)}"
            )

        candidates = lr_sig.loc[
            lr_sig['pathway_name'].astype(str) == str(pathway)
        ].copy()
        candidates = candidates.drop_duplicates('interaction_name', keep='first')
        if not enriched_only:
            selected = candidates
        else:
            if not isinstance(net_data, Mapping) or not isinstance(net_data.get('prob'), Mapping):
                raise ValueError("Run compute_communication_probability first.")
            prob_by_name = net_data['prob']
            pval_by_name = net_data.get('pval', {})
            selected_names = []
            for interaction_name in candidates['interaction_name'].astype(str):
                matrix = prob_by_name.get(interaction_name)
                if matrix is None:
                    continue
                prob_matrix = (
                    matrix.toarray() if sparse.issparse(matrix)
                    else np.asarray(matrix, dtype=float)
                ).copy()
                pval_matrix = (
                    pval_by_name.get(interaction_name)
                    if isinstance(pval_by_name, Mapping) else None
                )
                if pval_matrix is not None:
                    pval_matrix = np.asarray(pval_matrix, dtype=float)
                    if pval_matrix.shape != prob_matrix.shape:
                        raise ValueError(
                            f"pval matrix for {interaction_name!r} does not match prob."
                        )
                    # Keep only edges with pval strictly below the threshold.
                    prob_matrix[pval_matrix >= thresh] = 0.0
                if np.nansum(prob_matrix) != 0:
                    selected_names.append(interaction_name)
            selected = candidates.loc[
                candidates['interaction_name'].astype(str).isin(selected_names)
            ].copy()

        pair_names = selected['interaction_name'].astype(str).tolist()
        genes = _extract_gene_subset(selected['ligand'].tolist())
        genes.extend(_extract_gene_subset(selected['receptor'].tolist()))
        return _ordered_unique(genes), pair_names

    net_slot = getattr(cellchat, slot_name)
    lr_slot = cellchat.lr_pairs
    if isinstance(net_slot, Mapping) and isinstance(net_slot.get('prob'), Mapping):
        datasets = [(net_slot, lr_slot)]
    else:
        dataset_keys = _merged_dataset_keys(net_slot)
        if not dataset_keys:
            raise ValueError("Run compute_communication_probability first.")
        datasets = [(net_slot[key], lr_slot.get(key, {})) for key in dataset_keys]

    pair_names_all = []
    gene_names_all = []
    for pathway in signaling:
        pathway_pairs = []
        pathway_genes = []
        for net_data, lr_data in datasets:
            genes, pair_names = _single_dataset_result(net_data, lr_data, pathway)
            pathway_genes = _ordered_unique(pathway_genes + genes)
            pathway_pairs = _ordered_unique(pathway_pairs + pair_names)
        gene_names_all.extend(pathway_genes)
        pair_names_all.extend(pathway_pairs)

    pair_lr = pd.DataFrame({'interaction_name': pair_names_all})
    if gene_lr_return:
        return {'pair_lr': pair_lr, 'gene_lr': gene_names_all}
    return pair_lr


# ---------------------------------------------------------------------------
# identifyCommunicationPatterns  (NMF-based pattern analysis)
# ---------------------------------------------------------------------------
def identify_communication_patterns(
    cellchat: 'CellChat',
    pattern: str = "outgoing",
    k: int = 5,
    slot_name: str = "pathway_network",
    thresh: Optional[float] = None,
    seed_use: int = 1
) -> 'CellChat':
    """
    Identify global communication patterns using NMF.
    Mirrors R identifyCommunicationPatterns().

    Outgoing and incoming patterns factorize the same column-normalized
    cell-group x pathway matrix used by the R implementation.
    """
    data0, mat_nmf, cluster_names, cluster_names_nmf, pathway_names = _build_pattern_matrix(
        cellchat, slot_name, pattern, thresh=thresh
    )

    if mat_nmf.shape[0] == 0:
        warnings.warn("All rows are zero after normalization; skipping pattern analysis.")
        return cellchat

    k = min(k, min(mat_nmf.shape))
    if k < 1:
        warnings.warn("Too few features for pattern analysis.")
        return cellchat

    # R: NMF::nmf(data, rank=k, method='lee', seed='nndsvd').  Use a
    # local Lee-Seung implementation rather than sklearn's MU solver because
    # sklearn keeps NNDSVD zeros fixed, which distorts the cell-pattern W matrix.
    W_raw, H_raw = _lee_nmf(
        mat_nmf,
        n_components=k,
        seed_use=seed_use,
        init='nndsvd',
        max_iter=10000,
        tol=1e-6,
    )

    # R scaleMat 'r1': divide each row by its sum (W).
    W_row_sums = W_raw.sum(axis=1, keepdims=True)
    W_row_sums[W_row_sums == 0] = 1.0
    W_norm = W_raw / W_row_sums

    # R scaleMat 'c1': divide each column by its sum (H).
    H_col_sums = H_raw.sum(axis=0, keepdims=True)
    H_col_sums[H_col_sums == 0] = 1.0
    H_norm = H_raw / H_col_sums

    pattern_labels = [f'Pattern {i+1}' for i in range(k)]

    data_cell_rows = []
    for ci, cname in enumerate(cluster_names_nmf):
        for pi, pname in enumerate(pattern_labels):
            data_cell_rows.append({
                'CellGroup': cname,
                'Pattern': pname,
                'Contribution': float(W_norm[ci, pi]),
            })
    data_cell = pd.DataFrame(data_cell_rows)
    data_cell['CellGroup'] = pd.Categorical(
        data_cell['CellGroup'], categories=cluster_names_nmf, ordered=True)
    data_cell['Pattern'] = pd.Categorical(
        data_cell['Pattern'], categories=pattern_labels, ordered=True)

    data_sig_rows = []
    for pi, pname in enumerate(pattern_labels):
        for si, sname in enumerate(pathway_names):
            data_sig_rows.append({
                'Pattern': pname,
                'Signaling': sname,
                'Contribution': float(H_norm[pi, si]),
            })
    data_sig = pd.DataFrame(data_sig_rows)
    data_sig['Pattern'] = pd.Categorical(
        data_sig['Pattern'], categories=pattern_labels, ordered=True)
    data_sig['Signaling'] = pd.Categorical(
        data_sig['Signaling'], categories=pathway_names, ordered=True)

    if not hasattr(cellchat, 'uns'):
        cellchat.uns = {}

    pattern_key = f'cellchat_patterns_{pattern}'
    cellchat.uns[pattern_key] = {
        'cell': data_cell,
        'signaling': data_sig,
        'pattern': {'cell': data_cell, 'signaling': data_sig},
        'k': k,
        'W': W_raw,
        'H': H_raw,
        'data': data0,
        'data_nmf': mat_nmf,
        'data_names': {
            'cell': cluster_names,
            'cell_nmf': cluster_names_nmf,
            'signaling': pathway_names,
        },
    }

    if slot_name == "pathway_network":
        if 'pattern' not in cellchat.pathway_network:
            cellchat.pathway_network['pattern'] = {}
        cellchat.pathway_network['pattern'][pattern] = cellchat.uns[pattern_key]
    else:
        if 'pattern' not in cellchat.network:
            cellchat.network['pattern'] = {}
        cellchat.network['pattern'][pattern] = cellchat.uns[pattern_key]

    return cellchat

# ---------------------------------------------------------------------------
# select_k - choose number of NMF patterns
# ---------------------------------------------------------------------------
def select_k(
    cellchat: 'CellChat',
    pattern: str = "outgoing",
    slot_name: str = "pathway_network",
    k_range: Optional[List[int]] = None,
    nrun: int = 30,
    thresh: Optional[float] = None,
    seed_use: int = 10
) -> Dict[str, Any]:
    """
    Evaluate Cophenetic and Silhouette scores over a range of k.
    Mirrors R selectK() / NMF::nmfEstimateRank().

    The R NMF package builds consensus over matrix columns (samples), so for
    CellChat this clusters signaling pathways using the coefficient matrix H.
    """
    try:
        from scipy.cluster.hierarchy import linkage, cophenet, fcluster
        from scipy.spatial.distance import squareform
        from sklearn.metrics import silhouette_score
    except ImportError:
        raise ImportError("scikit-learn and scipy required.")

    _, mat, _, _, _ = _build_pattern_matrix(cellchat, slot_name, pattern, thresh=thresh)

    if mat.shape[0] < 2 or mat.shape[1] < 2:
        warnings.warn("Too few non-zero rows or pathways for pattern selection.")
        return {'k_range': k_range or [], 'cophenetic': [], 'silhouette': []}

    if k_range is None:
        k_range = list(range(2, min(10, min(mat.shape)) + 1))

    rng = np.random.default_rng(seed_use)
    cophenetic_scores = []
    silhouette_scores = []
    n_samples = mat.shape[1]

    for k in k_range:
        C_sum = np.zeros((n_samples, n_samples), dtype=np.float64)
        n_ok = 0
        for _ in range(nrun):
            seed_r = int(rng.integers(0, 2**31))
            try:
                # R calls NMF::nmfEstimateRank(..., method='lee'), whose
                # consensus is built from repeated Lee-Seung NMF fits.  Using
                # sklearn's MU solver here changes the update path enough to
                # shift the cophenetic/silhouette curves, so reuse the local
                # Lee-Seung implementation used by identifyCommunicationPatterns.
                _, H_r = _lee_nmf(
                    mat,
                    n_components=k,
                    seed_use=seed_r,
                    init='random',
                    max_iter=2000,
                    tol=1e-5,
                    update_order='WH',
                )
                labels_r = np.argmax(H_r, axis=0)
                C_sum += (labels_r[:, None] == labels_r[None, :]).astype(float)
                n_ok += 1
            except Exception:
                continue

        if n_ok == 0:
            cophenetic_scores.append(np.nan)
            silhouette_scores.append(np.nan)
            continue

        C = C_sum / n_ok
        dist_mat = 1.0 - C
        np.fill_diagonal(dist_mat, 0.0)
        dist_vec = squareform(dist_mat, checks=False)

        try:
            Z = linkage(dist_vec, method='average')
            cophenetic_scores.append(float(cophenet(Z, dist_vec)[0]))
        except Exception:
            Z = None
            cophenetic_scores.append(np.nan)

        try:
            if Z is None:
                Z = linkage(dist_vec, method='average')
            cluster_labels = fcluster(Z, k, criterion='maxclust')
            if len(np.unique(cluster_labels)) >= 2:
                s = silhouette_score(dist_mat, cluster_labels, metric='precomputed')
                silhouette_scores.append(float(s))
            else:
                silhouette_scores.append(np.nan)
        except Exception:
            silhouette_scores.append(np.nan)

    return {
        'k_range': k_range,
        'cophenetic': cophenetic_scores,
        'silhouette': silhouette_scores,
    }

# ---------------------------------------------------------------------------
# compute_network_similarity
# ---------------------------------------------------------------------------
def _build_snn_from_ranked_neighbors(nn_ranked, prune: float = 1.0 / 15):
    """Build an SNN matrix from zero-based ranked neighbor indices."""
    nn_ranked = np.asarray(nn_ranked, dtype=int)
    if nn_ranked.ndim != 2:
        raise ValueError("nn_ranked must be a 2D matrix of zero-based indices.")

    n_samples, k = nn_ranked.shape
    if n_samples == 0:
        return sparse.csr_matrix((0, 0), dtype=float)
    if np.any(nn_ranked < 0) or np.any(nn_ranked >= n_samples):
        raise ValueError("nn_ranked must contain zero-based indices within the sample count.")

    rows = np.repeat(np.arange(n_samples), k)
    cols = nn_ranked.reshape(-1)
    indicator = sparse.csr_matrix(
        (np.ones(rows.size, dtype=float), (rows, cols)),
        shape=(n_samples, n_samples),
    )
    snn = (indicator @ indicator.T).tolil()

    for i, j in zip(*snn.nonzero()):
        overlap = snn[i, j]
        value = overlap / (k + (k - overlap))
        snn[i, j] = value if value >= prune else 0.0

    snn = snn.tocsr()
    snn.eliminate_zeros()
    return snn


def _build_snn(sim_mat: np.ndarray, k: int, prune_snn: float = 1.0/15) -> np.ndarray:
    """
    Build Shared Nearest Neighbor (SNN) matrix from similarity matrix.
    Mirrors R buildSNN(): FNN k-nearest neighbors -> Jaccard overlap -> prune.

    Input sim_mat is (n x n); R treats it as a feature matrix where
    columns are samples, so we transpose: each pathway is a sample described
    by n features (its similarity to every other pathway).
    """
    from sklearn.neighbors import NearestNeighbors
    n = sim_mat.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)

    k_actual = max(1, min(int(k), n))
    neighbor_cols = max(k_actual - 1, 0)
    knn_query_k = min(max(10 * k_actual, k_actual), n)

    # features = rows of sim_mat (each pathway described by its similarity vector)
    features = np.asarray(sim_mat, dtype=float).T
    nbrs = NearestNeighbors(n_neighbors=knn_query_k, metric='euclidean').fit(features)
    raw_indices = nbrs.kneighbors(features, return_distance=False)

    # sklearn versions differ in how clearly they document self-neighbor
    # handling. R/FNN::get.knn returns non-self neighbors, then CellChat prepends
    # self explicitly, so remove self here in a version-independent way.
    indices = np.empty((n, neighbor_cols), dtype=int)
    for i, row in enumerate(raw_indices):
        non_self = [int(j) for j in row if int(j) != i]
        if len(non_self) < neighbor_cols:
            dists = np.linalg.norm(features - features[i], axis=1)
            fallback = [
                int(j) for j in np.argsort(dists, kind='mergesort')
                if int(j) != i and int(j) not in non_self
            ]
            non_self.extend(fallback)
        indices[i, :] = non_self[:neighbor_cols]

    # Build a zero-based ranked neighbour list including self at position 0.
    nn_ranked = np.hstack([np.arange(n).reshape(-1, 1), indices])  # (n, k_actual)
    snn = _build_snn_from_ranked_neighbors(nn_ranked, prune=prune_snn).toarray()
    np.fill_diagonal(snn, 1.0)
    return snn


def compute_network_similarity(
    cellchat: 'CellChat',
    type: str = "functional",
    slot_name: str = "pathway_network",
    thresh: float = None,
    k: int = None,
) -> 'CellChat':
    """
    Compute similarity between signaling pathway networks.
    Mirrors R's network-similarity procedure:
      - functional: binarize prob matrices, pairwise Jaccard, then SNN smooth
      - structural: R computeNetD_structure distance, then SNN smooth

    Results are stored in ``pathway_network['similarity'][type]`` (or ``network`` when
    ``slot_name='network'``). Single-dataset results use the ``'single'`` key,
    matching the comparison-keyed layout used for merged datasets.
    """
    prob, _, pathway_names, _ = _get_pathway_arrays(cellchat, slot_name)
    n_paths = len(pathway_names)

    # R: if (is.null(k)) k <- ceiling(sqrt(n_pathways)) [+1 if >25]
    if k is None:
        k = int(np.ceil(np.sqrt(n_paths)))
        if n_paths > 25:
            k += 1
    k = max(2, min(k, n_paths - 1))

    # R: if (!is.null(thresh)) prob[prob < quantile(prob[prob!=0], thresh)] <- 0
    prob_f = prob.copy()
    if thresh is not None:
        nonzero_vals = prob_f[prob_f != 0]
        if len(nonzero_vals) > 0:
            cutoff = np.quantile(nonzero_vals, thresh)
            prob_f[prob_f < cutoff] = 0.0

    # Compute pairwise similarity
    sim_mat = np.zeros((n_paths, n_paths), dtype=float)

    if type == "functional":
        # R: Gi = (prob[,,i] > 0)*1; S3[i,j] = Jaccard(Gi, Gj)
        for i in range(n_paths - 1):
            Gi = (prob_f[:, :, i] > 0).astype(float)
            for j in range(i + 1, n_paths):
                Gj = (prob_f[:, :, j] > 0).astype(float)
                inter = np.sum(Gi * Gj)
                union = np.sum(Gi + Gj - Gi * Gj)
                sim_mat[i, j] = inter / union if union > 0 else 0.0
        # R: S3 = S3 + t(S3); diag(S3) = 1
        sim_mat = sim_mat + sim_mat.T
        np.fill_diagonal(sim_mat, 1.0)

    else:  # structural
        # R: D_signalings[i,j] <- computeNetD_structure(Gi, Gj)
        #     D_signalings <- D_signalings + t(D_signalings); S_signalings <- 1 - D
        dist_mat = np.zeros((n_paths, n_paths), dtype=float)
        for i in range(n_paths - 1):
            Gi = (prob_f[:, :, i] > 0).astype(float)
            for j in range(i + 1, n_paths):
                Gj = (prob_f[:, :, j] > 0).astype(float)
                dist_mat[i, j] = _compute_net_d_structure(Gi, Gj)
        dist_mat[~np.isfinite(dist_mat)] = 0.0
        dist_mat = dist_mat + dist_mat.T
        sim_mat = 1.0 - dist_mat
        np.fill_diagonal(sim_mat, 1.0)
    sim_mat = np.clip(sim_mat, 0, 1)

    # R: SNN <- buildSNN(S_signalings, k=k, prune.SNN=1/15)
    #    Similarity <- S_signalings * SNN
    snn = _build_snn(sim_mat, k=k, prune_snn=1.0/15)
    sim_smoothed = sim_mat * snn
    np.fill_diagonal(sim_smoothed, 1.0)

    target = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if _merged_dataset_keys(target):
        raise ValueError("compute_network_similarity requires one canonical network, not a merged slot.")
    sim_data = _network_similarity_data(cellchat, slot_name, type)
    sim_data.setdefault('matrix', {})['single'] = sim_smoothed
    sim_data['pathways'] = pathway_names
    sim_data['snn_k'] = k

    return cellchat


# ---------------------------------------------------------------------------
# embed_network
# ---------------------------------------------------------------------------
def embed_network(
    cellchat: 'CellChat',
    type: str = "functional",
    slot_name: str = "pathway_network",
    n_components: int = 2,
    min_dist: float = 0.3,
    n_neighbors: int = None,
    seed_use: int = 42,
    umap_init: Optional[str] = None,
    n_epochs: Optional[int] = None,
    comparison: Optional[Union[List[int], Tuple[int, ...]]] = None
) -> 'CellChat':
    """
    Manifold learning (UMAP) on pathway similarity matrix.
    Mirrors R's network-embedding procedure:
      - Input is the SNN-smoothed similarity matrix directly (NOT converted to distance)
      - Removes isolated pathways (colSum == 1, i.e. only self-similarity)
      - n_neighbors = ceiling(sqrt(n)) + 1
      - metric = "correlation" (R runUMAP default)
      - runUMAP() seeds Python/NumPy through reticulate::py_set_seed(), then
        lets umap-learn use its own defaults for init, n_epochs, and
        random_state unless the user explicitly passes them through.

    Reads and stores results in ``pathway_network['similarity'][type]`` (or ``network``).
    """
    target = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    pairwise_sim = _network_similarity_data(cellchat, slot_name, type)

    # The R implementation defaults to every dataset in a merged CellChat object.
    # Keep the same default here instead of accidentally looking for a
    # single-dataset similarity matrix.
    if comparison is None:
        dataset_keys = _merged_dataset_keys(target)
        if len(dataset_keys) > 1:
            comparison = list(range(len(dataset_keys)))
    comparison_key = _comparison_name(comparison)

    matrix_store = pairwise_sim.get('matrix', {})
    if comparison_key not in matrix_store:
        if comparison is None:
            raise ValueError(f"Run compute_network_similarity(type='{type}') first.")
        raise ValueError(f"Run compute_pairwise_network_similarity(type='{type}', comparison={comparison}) first.")
    sim_data = pairwise_sim
    sim_mat = np.array(matrix_store[comparison_key])
    pathway_names_all = _pathways_for_similarity(sim_data, comparison_key, sim_mat.shape[0])

    # R: pathway.remove <- rownames(Similarity)[which(colSums(Similarity) == 1)]
    # colSums == 1 means only the diagonal is non-zero -> isolated pathway
    col_sums = sim_mat.sum(axis=0)
    keep_mask = col_sums != 1.0
    if not np.all(keep_mask):
        sim_mat = sim_mat[np.ix_(keep_mask, keep_mask)]
        pathway_names_all = [pathway_names_all[i] for i in range(len(pathway_names_all)) if keep_mask[i]]

    n_pts = sim_mat.shape[0]

    # R: n_neighbors = ceiling(sqrt(n)) + 1
    if n_neighbors is None:
        n_neighbors = int(np.ceil(np.sqrt(n_pts))) + 1
    n_neighbors = max(2, min(n_neighbors, n_pts - 1))

    # R: runUMAP(Similarity, ...) calls reticulate::py_set_seed(seed.use) and
    # runs fit_transform(t(data.use)).  umap-learn also maintains a compiled
    # random state, so resetting Python/NumPy globals alone is insufficient
    # after another UMAP run.  Pass the same seed explicitly to keep the
    # CellChat result reproducible across calls.
    try:
        import random
        import umap as umap_lib
        random.seed(seed_use)
        np.random.seed(seed_use)
        umap_kwargs = dict(
            n_neighbors=int(n_neighbors),
            n_components=int(n_components),
            metric='correlation',
            n_epochs=n_epochs,
            learning_rate=1.0,
            min_dist=min_dist,
            spread=1.0,
            set_op_mix_ratio=1.0,
            local_connectivity=1,
            repulsion_strength=1,
            negative_sample_rate=5,
            a=None,
            b=None,
            metric_kwds=None,
            angular_rp_forest=False,
            verbose=False,
            random_state=seed_use,
        )
        if umap_init is not None:
            umap_kwargs['init'] = umap_init
        reducer = umap_lib.UMAP(**umap_kwargs)
        embedding = reducer.fit_transform(sim_mat.T)
        embedding_method = 'umap-learn'
        embedding_error = None
    except Exception as exc:
        from sklearn.manifold import MDS
        dist_mat = np.clip(1 - sim_mat, 0, 1)
        mds = MDS(n_components=n_components, dissimilarity='precomputed',
                  random_state=seed_use, normalized_stress='auto')
        embedding = mds.fit_transform(dist_mat)
        embedding_method = 'mds-fallback'
        embedding_error = f'{type(exc).__name__}: {exc}'

    sim_data.setdefault('dr', {})[comparison_key] = embedding
    _set_similarity_pathways(sim_data, comparison_key, pathway_names_all)
    sim_data['embedding_method'] = embedding_method
    sim_data['embedding_init'] = umap_init if embedding_method == 'umap-learn' else None
    sim_data['embedding_n_epochs'] = n_epochs if embedding_method == 'umap-learn' else None
    if embedding_error is not None:
        sim_data['embedding_error'] = embedding_error
    else:
        sim_data.pop('embedding_error', None)

    return cellchat


# ---------------------------------------------------------------------------
# cluster_network
# ---------------------------------------------------------------------------
def _hartigan_wong_once(
    data: np.ndarray,
    init_centers: np.ndarray,
    max_iter: int = 10,
) -> Tuple[np.ndarray, float]:
    """Approximate R stats::kmeans Hartigan-Wong transfer steps."""
    X = np.asarray(data, dtype=float)
    centers = np.asarray(init_centers, dtype=float).copy()
    n_clusters = centers.shape[0]

    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1).astype(int)

    for _ in range(max_iter):
        sizes = np.array([np.sum(labels == k) for k in range(n_clusters)], dtype=int)
        if np.any(sizes == 0):
            raise ValueError("empty cluster")
        centers = np.vstack([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        moved = False

        for i, point in enumerate(X):
            current = int(labels[i])
            if sizes[current] <= 1:
                continue

            dist = np.sum((centers - point) ** 2, axis=1)
            remove_cost = sizes[current] / (sizes[current] - 1) * dist[current]
            best_delta = 0.0
            best_cluster = current

            for candidate in range(n_clusters):
                if candidate == current:
                    continue
                add_cost = sizes[candidate] / (sizes[candidate] + 1) * dist[candidate]
                delta = add_cost - remove_cost
                if delta < best_delta:
                    best_delta = delta
                    best_cluster = candidate

            if best_cluster != current:
                centers[current] = (centers[current] * sizes[current] - point) / (sizes[current] - 1)
                sizes[current] -= 1
                centers[best_cluster] = (centers[best_cluster] * sizes[best_cluster] + point) / (sizes[best_cluster] + 1)
                sizes[best_cluster] += 1
                labels[i] = best_cluster
                moved = True

        if not moved:
            break

    sizes = np.array([np.sum(labels == k) for k in range(n_clusters)], dtype=int)
    if np.any(sizes == 0):
        raise ValueError("empty cluster")
    centers = np.vstack([X[labels == k].mean(axis=0) for k in range(n_clusters)])
    withinss = float(sum(np.sum((X[labels == k] - centers[k]) ** 2) for k in range(n_clusters)))
    return labels, withinss


def _fit_r_kmeans(
    data: np.ndarray,
    n_clusters: int,
    seed_use: int = 42,
    nstart: int = 10,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Fit k-means with R stats::kmeans integer-centers semantics."""
    data = np.asarray(data, dtype=float)
    n_clusters = int(n_clusters)
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("data must be a non-empty 2D array.")
    if n_clusters > data.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of observations.")
    if n_clusters == 1:
        return np.zeros(data.shape[0], dtype=int)

    unique_points = np.unique(data, axis=0)
    if unique_points.shape[0] < n_clusters:
        raise ValueError("more cluster centers than distinct data points")

    if rng is None:
        rng = np.random.RandomState(seed_use)
    best_labels = None
    best_withinss = np.inf

    # R samples initial centers from unique data rows for nstart >= 2.
    for _ in range(max(1, int(nstart))):
        center_idx = rng.choice(unique_points.shape[0], size=n_clusters, replace=False)
        init_centers = unique_points[center_idx]
        try:
            labels, withinss = _hartigan_wong_once(data, init_centers, max_iter=10)
        except ValueError:
            continue
        if withinss < best_withinss:
            best_labels = labels
            best_withinss = withinss

    if best_labels is None:
        raise ValueError("empty cluster: try a better set of initial centers")
    return best_labels


def cluster_network(
    cellchat: 'CellChat',
    type: str = "functional",
    slot_name: str = "pathway_network",
    k: Optional[int] = None,
    seed_use: int = 42,
    comparison: Optional[Union[List[int], Tuple[int, ...]]] = None
) -> 'CellChat':
    """
    Cluster signaling pathways based on UMAP embedding.
    Mirrors R's network-clustering procedure: K-means on UMAP coordinates.
    If k is None, infer the number of signaling groups via the R consensus
    eigengap procedure instead of forcing a fixed cluster count.
    """
    target = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network

    # The R implementation uses all datasets by default for a merged object.
    if comparison is None:
        dataset_keys = _merged_dataset_keys(target)
        if len(dataset_keys) > 1:
            comparison = list(range(len(dataset_keys)))
    comparison_key = _comparison_name(comparison)

    sim_data = _network_similarity_data(cellchat, slot_name, type)
    embedding = sim_data.get('dr', {}).get(comparison_key)

    # R: Y <- slot(object, slot.name)$similarity[[type]]$dr[[comparison.name]]
    #    data.use <- Y; kmeans is run on UMAP coordinates, not similarity.
    if embedding is None:
        raise ValueError(f"Run embed_network(type='{type}') first.")

    data_use = np.asarray(embedding, dtype=float)
    if data_use.ndim != 2 or data_use.shape[0] == 0:
        raise ValueError("Embedding must be a non-empty 2D array.")

    n_items = data_use.shape[0]
    rng = np.random.RandomState(seed_use)

    def _fit_kmeans(n_clusters: int) -> np.ndarray:
        return _fit_r_kmeans(data_use, n_clusters=n_clusters, seed_use=seed_use, nstart=10, rng=rng)

    if k is None:
        # The R implementation runs kmeans over k=2:min(N-1,10), averages
        # co-clustering matrices, then infer the cluster count with eigengap.
        if n_items <= 2:
            k_actual = 1
            eigengap_res = None
        else:
            k_range = list(range(2, min(n_items - 1, 10) + 1))
            consensus = np.zeros((n_items, n_items), dtype=float)
            for kk in k_range:
                labels_tmp = _fit_kmeans(kk)
                consensus += (labels_tmp[:, None] == labels_tmp[None, :]).astype(float)
            consensus /= len(k_range)
            eigengap_res = compute_eigengap(consensus)
            k_actual = int(eigengap_res['upper_bound'])
    else:
        k_actual = int(k)
        eigengap_res = None

    k_actual = max(1, min(k_actual, n_items))
    labels = _fit_kmeans(k_actual)

    # Store the zero-based cluster labels returned by the Python K-means code.
    group_labels = [int(label) for label in labels]

    sim_data.setdefault('group', {})[comparison_key] = group_labels
    sim_data['cluster_k'] = k_actual
    if eigengap_res is not None:
        sim_data.setdefault('eigengap', {})[comparison_key] = eigengap_res

    return cellchat


# ---------------------------------------------------------------------------
# net_analysis_signaling_role_scatter
# ---------------------------------------------------------------------------
def net_analysis_signaling_role_scatter(
    cellchat: 'CellChat',
    signaling=None,
    slot_name: str = "pathway_network",
    color_use=None,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: str = "Outgoing interaction strength",
    ylabel: str = "Incoming interaction strength",
    title: str = None,
    dot_size=(2, 6),
    label_size: float = 3,
    dot_alpha: float = 0.6,
    do_label: bool = True,
    show_legend: bool = True,
    weight_min_max=None,
    fig_size=(8, 6),
    return_fig: bool = False
):
    """
    2D scatter plot of dominant senders and receivers.
    Mirrors R net_analysis_signaling_role_scatter().
    x-axis = total outgoing strength, y-axis = total incoming strength.
    Dot size proportional to number of inferred links.

    weight_min_max : (min, max) tuple to fix the link-count -> dot-size scaling
        across datasets (mirrors R weight.MinMax). None = auto-scale per object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    if slot_name == "pathway_network":
        net_data = cellchat.pathway_network
    else:
        net_data = cellchat.network

    centrality = net_data.get('centrality', {})
    if not centrality:
        raise ValueError("Run compute_network_centrality first.")

    cluster_names = _get_net_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)

    # Build outgoing/incoming matrices: rows=clusters, cols=pathways
    pathway_names = list(centrality.keys())
    if signaling is not None:
        if isinstance(signaling, str):
            signaling = [signaling]
        pathway_names = [p for p in pathway_names if p in signaling]

    # R fills the default scatter coordinates from per-pathway outdeg/indeg.
    # For these two default measures this is exactly the row/column sum of
    # pathway_network probability, so use the tensor directly to avoid stale or mis-ordered
    # centrality arrays in previously saved objects.
    out_cells = None
    in_cells = None
    prob = net_data.get('prob')
    if prob is not None and x_measure == 'outdeg' and y_measure == 'indeg':
        full_pathways = network_names(net_data)
        prob_arr = stack_network_field(net_data, 'prob', full_pathways, 0.0)
        pval_arr = stack_network_field(net_data, 'pval', full_pathways, 1.0)

        prob_groups = list(net_data.get('groups', cluster_names))
        if set(prob_groups) == set(cluster_names):
            order_idx = [prob_groups.index(g) for g in cluster_names]
            prob_arr = prob_arr[np.ix_(order_idx, order_idx, np.arange(prob_arr.shape[2]))]
            pval_arr = pval_arr[np.ix_(order_idx, order_idx, np.arange(pval_arr.shape[2]))]

        pathway_idx = [full_pathways.index(p) for p in pathway_names if p in full_pathways]
        if pathway_idx:
            prob_arr = prob_arr[:, :, pathway_idx].copy()
            pval_arr = pval_arr[:, :, pathway_idx]
            prob_arr[pval_arr >= 0.05] = 0.0
            out_cells = prob_arr.sum(axis=(1, 2))
            in_cells = prob_arr.sum(axis=(0, 2))

    if out_cells is None or in_cells is None:
        outgoing = np.zeros((n_clusters, len(pathway_names)))
        incoming = np.zeros((n_clusters, len(pathway_names)))
        for pi, pname in enumerate(pathway_names):
            c = centrality[pname]
            outdeg = c.get(x_measure, c.get('outdeg', np.zeros(n_clusters)))
            indeg = c.get(y_measure, c.get('indeg', np.zeros(n_clusters)))
            outgoing[:, pi] = outdeg
            incoming[:, pi] = indeg
        out_cells = outgoing.sum(axis=1)
        in_cells = incoming.sum(axis=1)

    def _count_lr_links():
        """Count significant ligand-receptor edges for visualization."""
        count_mat = np.zeros((n_clusters, n_clusters), dtype=float)
        net_lr = cellchat.network
        prob_lr = net_lr.get('prob') if isinstance(net_lr, dict) else None
        if prob_lr is None:
            return count_mat
        lr_names = network_names(net_lr)
        prob_lr = stack_network_field(net_lr, 'prob', lr_names, 0.0)
        pval_lr = stack_network_field(net_lr, 'pval', lr_names, 1.0)
        lr_idx = np.arange(len(lr_names))
        interactions = net_lr.get('interactions')
        if signaling is not None and isinstance(interactions, pd.DataFrame) and 'pathway_name' in interactions.columns:
            lr_idx = np.array([
                i for i, p in enumerate(interactions['pathway_name'].astype(str).tolist())
                if p in signaling and i < prob_lr.shape[2]
            ], dtype=int)
        lr_groups = list(net_lr.get('groups', cluster_names))
        if set(lr_groups) == set(cluster_names):
            order_idx = [lr_groups.index(g) for g in cluster_names]
            prob_lr = prob_lr[np.ix_(order_idx, order_idx, lr_idx)]
            pval_lr = pval_lr[np.ix_(order_idx, order_idx, lr_idx)]
        else:
            prob_lr = prob_lr[:, :, lr_idx]
            pval_lr = pval_lr[:, :, lr_idx]
        prob_lr[pval_lr >= 0.05] = 0.0
        return np.sum(prob_lr > 0, axis=2)

    count_mat = _count_lr_links()
    num_link = count_mat.sum(axis=1) + count_mat.sum(axis=0) - np.diag(count_mat)

    if color_use is None:
        from .visualization import sc_palette
        color_use = sc_palette(n_clusters)

    fig, ax = plt.subplots(figsize=fig_size)

    vmin_s, vmax_s = dot_size[0], dot_size[1]
    if weight_min_max is not None:
        lo, hi = weight_min_max[0], weight_min_max[1]
        nl = np.clip(num_link, lo, hi)
        if hi > lo:
            sizes = np.interp(nl, (lo, hi), (vmin_s**2 * 5, vmax_s**2 * 5))
        else:
            sizes = np.full(n_clusters, ((vmin_s + vmax_s) / 2) ** 2 * 5)
    elif num_link.max() > num_link.min():
        sizes = np.interp(num_link, (num_link.min(), num_link.max()), (vmin_s**2 * 5, vmax_s**2 * 5))
    else:
        sizes = np.full(n_clusters, ((vmin_s + vmax_s) / 2) ** 2 * 5)

    for i, name in enumerate(cluster_names):
        c = color_use[i] if i < len(color_use) else '#999999'
        ax.scatter(out_cells[i], in_cells[i], s=sizes[i],
                   c=[to_rgba(c, alpha=dot_alpha)],
                   edgecolors=c, linewidths=0.8, zorder=3)
        if do_label:
            ax.annotate(name, (out_cells[i], in_cells[i]),
                        fontsize=label_size * 2.5, color=c, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, linestyle='--', color='grey', linewidth=0.5, alpha=0.7)
    ax.axvline(0, linestyle='--', color='grey', linewidth=0.5, alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, ha='center')
    ax.spines[['top', 'right']].set_visible(False)

    if show_legend:
        hi = float(weight_min_max[1]) if weight_min_max is not None else float(np.nanmax(num_link))
        if not np.isfinite(hi):
            hi = 0.0
        if hi <= 10:
            breaks = [0, 2, 4, 6]
        elif hi <= 125:
            breaks = [25, 50, 75, 100]
        else:
            top = int(np.ceil(hi / 25.0) * 25)
            breaks = list(range(25, top + 1, max(25, top // 4)))

        def _size_for_count(value):
            if weight_min_max is not None:
                lo, hi_local = weight_min_max[0], weight_min_max[1]
                value = np.clip(value, lo, hi_local)
            else:
                lo, hi_local = float(np.nanmin(num_link)), float(np.nanmax(num_link))
            if hi_local > lo:
                return float(np.interp(value, (lo, hi_local), (vmin_s**2 * 5, vmax_s**2 * 5)))
            return float(((vmin_s + vmax_s) / 2) ** 2 * 5)

        handles = [
            ax.scatter([], [], s=_size_for_count(b), c='black', edgecolors='black')
            for b in breaks
        ]
        ax.legend(handles, [str(int(b)) for b in breaks], title="Count",
                  fontsize=8, title_fontsize=10, frameon=False,
                  bbox_to_anchor=(1.02, 0.5), loc='center left', scatterpoints=1)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# net_analysis_signaling_role_heatmap
# ---------------------------------------------------------------------------
def _role_heatmap_strength(row_sums: np.ndarray) -> np.ndarray:
    """Return the R CellChat row-annotation Strength values.

    This mirrors ``net_analysis_signaling_role_heatmap()`` in ``R/analysis.R``:
    it transforms a pathway's raw total with ``-1 / log(total)`` and replaces
    infinite or negative transformed values according to their rank in the
    original totals.  In particular, a missing pathway has a raw total of zero
    and remains a zero-width bar.
    """
    raw_strength = np.asarray(row_sums, dtype=float).reshape(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        strength = -1.0 / np.log(raw_strength)

    # R uses ``pSum[is.na(pSum)] <- 0`` before identifying values to replace.
    # Keep infinities here because they are handled by the next R-equivalent
    # branch rather than being silently converted to zero.
    strength[np.isnan(strength)] = 0.0
    invalid = np.isinf(strength) | (strength < 0)
    if not np.any(invalid):
        return strength

    # R: values.assign <- seq(max(pSum) * 1.1, max(pSum) * 1.5, ...)
    # followed by an assignment ordered by pSum.original.  Do not special-case
    # pathway names or replace the raw totals with a different scale.
    values_assign = np.linspace(
        float(np.max(strength)) * 1.1,
        float(np.max(strength)) * 1.5,
        int(np.sum(invalid)),
    )
    original_order = np.argsort(raw_strength[invalid], kind='stable')
    replacements = np.empty(int(np.sum(invalid)), dtype=float)
    replacements[original_order] = values_assign
    strength[invalid] = replacements
    return strength


def net_analysis_signaling_role_heatmap(
    cellchat: 'CellChat',
    signaling=None,
    pattern: str = "outgoing",
    slot_name: str = "pathway_network",
    color_use=None,
    color_heatmap: str = "BuGn",
    title: str = None,
    font_size: int = 8,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    fig_size=(10, 6),
    return_fig: bool = False
):
    """
    Heatmap of outgoing/incoming/overall signaling strength per cell group.
    Mirrors R net_analysis_signaling_role_heatmap().
    Rows = signaling pathways, Cols = cell groups.
    Color = row-scaled relative strength.
    Top bar = total strength per cell group.
    Right bar = total strength per pathway.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    if slot_name == "pathway_network":
        net_data = cellchat.pathway_network
    else:
        net_data = cellchat.network

    centrality = net_data.get('centrality', {})
    if not centrality:
        raise ValueError("Run compute_network_centrality first.")

    cluster_names = _get_net_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)
    pathway_names_all = list(centrality.keys())

    # R reconstructs a matrix whose rows exactly match `signaling`, filling
    # pathways absent from this dataset with zeroes. This keeps the row set and
    # order aligned when heatmaps from multiple datasets are placed side by side.
    if signaling is not None:
        pathway_names_plot = [signaling] if isinstance(signaling, str) else list(signaling)
    else:
        pathway_names_plot = pathway_names_all

    # Build matrices, retaining requested pathways that are absent from centr.
    outgoing = np.zeros((len(pathway_names_plot), n_clusters))
    incoming = np.zeros((len(pathway_names_plot), n_clusters))
    for pi, pname in enumerate(pathway_names_plot):
        if pname not in centrality:
            continue
        c = centrality[pname]
        outgoing[pi, :] = c.get('outdeg', np.zeros(n_clusters))
        incoming[pi, :] = c.get('indeg', np.zeros(n_clusters))

    if pattern == "outgoing":
        mat_ori = outgoing
        legend_name = "Outgoing"
    elif pattern == "incoming":
        mat_ori = incoming
        legend_name = "Incoming"
    else:
        mat_ori = outgoing + incoming
        legend_name = "Overall"

    if title is None:
        title = f"{legend_name} signaling patterns"

    if len(pathway_names_plot) == 0:
        warnings.warn("No signaling pathways to plot.")
        return None

    # Row-scale. All-zero rows intentionally remain NaN/white, matching R's
    # sweep(..., max) followed by mat[mat == 0] <- NA.
    row_max = mat_ori.max(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = mat_ori / row_max
    mat[mat == 0] = np.nan

    # Optional clustering
    if cluster_rows and len(pathway_names_plot) > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        order = leaves_list(linkage(np.nan_to_num(mat), method='average'))
        mat = mat[order, :]
        mat_ori = mat_ori[order, :]
        pathway_names_plot = [pathway_names_plot[i] for i in order]
    if cluster_cols and n_clusters > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        order = leaves_list(linkage(np.nan_to_num(mat).T, method='average'))
        mat = mat[:, order]
        mat_ori = mat_ori[:, order]
        cluster_names = [cluster_names[i] for i in order]
        if color_use is not None:
            color_use = [color_use[i] for i in order]

    if color_use is None:
        from .visualization import sc_palette
        color_use = sc_palette(n_clusters)

    # Colormap
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(color_heatmap).copy()
    except Exception:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("hm", ['#ffffff', '#1a9641'])
    cmap.set_bad('white')

    nR, nC = mat.shape
    col_sums = mat_ori.sum(axis=0)
    pSum = _role_heatmap_strength(mat_ori.sum(axis=1))

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[0.12, 0.88],
                           width_ratios=[0.88, 0.06, 0.06],
                           hspace=0.02, wspace=0.02)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])
    fig.add_subplot(gs[0, 1]).axis('off')
    fig.add_subplot(gs[0, 2]).axis('off')

    # Heatmap
    im = ax_heat.imshow(mat, cmap=cmap, aspect='auto',
                        vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_xticks(np.arange(nC))
    ax_heat.set_xticklabels(cluster_names, rotation=90, ha='center', fontsize=font_size)
    ax_heat.set_yticks(np.arange(nR))
    ax_heat.set_yticklabels(pathway_names_plot, fontsize=font_size)
    ax_heat.set_xlabel('Cell groups', fontsize=font_size + 1)

    # Grid lines
    ax_heat.set_xticks(np.arange(nC + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(nR + 1) - 0.5, minor=True)
    ax_heat.grid(which='minor', color='white', linewidth=0.5)
    ax_heat.tick_params(which='minor', bottom=False, left=False)

    # Top bar (total per cell group)
    bar_colors = color_use[:nC] if len(color_use) >= nC else color_use + ['#999999'] * (nC - len(color_use))
    ax_top.bar(np.arange(nC), col_sums, color=bar_colors, edgecolor='none', width=0.7)
    ax_top.set_xlim(-0.5, nC - 0.5)
    ax_top.set_xticks([])
    ax_top.set_title(title, fontsize=font_size + 2, fontweight='bold', pad=4)
    ax_top.spines[:].set_visible(False)
    ax_top.tick_params(left=False, labelleft=False)

    # Right bar (pathway strength)
    ax_right.barh(np.arange(nR), pSum, color='grey', edgecolor='none', height=0.7)
    ax_right.set_ylim(-0.5, nR - 0.5)
    ax_right.invert_yaxis()
    ax_right.set_yticks([])
    ax_right.spines[:].set_visible(False)
    ax_right.tick_params(bottom=False, labelbottom=False)

    # Colorbar
    plt.colorbar(im, cax=ax_cbar)
    ax_cbar.set_ylabel('Relative strength', fontsize=font_size - 1)
    ax_cbar.tick_params(labelsize=font_size - 2)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# net_analysis_signaling_role_network
# ---------------------------------------------------------------------------
def net_analysis_signaling_role_network(
    cellchat: 'CellChat',
    signaling,
    slot_name: str = "pathway_network",
    measure=("outdeg", "indeg", "flowbet", "info"),
    measure_name=("Sender", "Receiver", "Mediator", "Influencer"),
    color_use=None,
    color_heatmap: str = "BuGn",
    font_size: int = 8,
    fig_size=(8, 2),
    return_fig: bool = False
):
    """
    Heatmap of centrality scores for specific signaling pathways.
    Mirrors R net_analysis_signaling_role_network().
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec

    if slot_name == "pathway_network":
        net_data = cellchat.pathway_network
    else:
        net_data = cellchat.network

    centrality = net_data.get('centrality', {})
    if not centrality:
        raise ValueError("Run compute_network_centrality first.")

    if isinstance(signaling, str):
        signaling = [signaling]

    cluster_names = _get_net_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)

    if color_use is None:
        from .visualization import sc_palette
        color_use = sc_palette(n_clusters)

    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(color_heatmap)
    except Exception:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("hm", ['#ffffff', '#1a9641'])

    figs = []
    for sig in signaling:
        if sig not in centrality:
            warnings.warn(f"Signaling '{sig}' not found in centrality data.")
            continue
        c = centrality[sig]
        measure_keys = list(measure)
        measure_labels = list(measure_name)

        rows = []
        for mk in measure_keys:
            vals = c.get(mk, np.zeros(n_clusters))
            rows.append(vals)
        mat = np.array(rows)  # (n_measures, n_clusters)

        # Row-normalize
        row_max = mat.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        mat_norm = mat / row_max

        fig, ax = plt.subplots(figsize=fig_size)
        im = ax.imshow(mat_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(n_clusters))
        ax.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=font_size)
        ax.set_yticks(np.arange(len(measure_labels)))
        ax.set_yticklabels(measure_labels, fontsize=font_size)
        ax.set_title(f"{sig} signaling pathway network", fontsize=font_size + 2)

        # Color annotation bar at bottom
        for ci, col in enumerate(color_use[:n_clusters]):
            ax.add_patch(plt.Rectangle((ci - 0.5, len(measure_labels) - 0.5),
                                       1, 0.3, color=col, clip_on=False))

        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.04,
                     label='Importance')
        plt.tight_layout()
        figs.append(fig)

    if return_fig:
        return figs[0] if len(figs) == 1 else figs
    for f in figs:
        plt.figure(f.number)
        plt.show()
    return None


# ---------------------------------------------------------------------------
# rank_net  -  rank signaling pathways by communication strength
# ---------------------------------------------------------------------------
def rank_net(
    cellchat: 'CellChat',
    mode: str = "single",
    slot_name: str = "pathway_network",
    measure: str = "weight",
    sources_use=None,
    targets_use=None,
    signaling=None,
    thresh: float = 0.05,
    color_use=None,
    title: str = None,
    font_size: int = 10,
    fig_size=(6, 8),
    comparison=(0, 1),
    stacked: bool = False,
    do_stat: bool = False,
    paired_test: bool = True,
    cutoff_pvalue: float = 0.05,
    tol: float = 0.05,
    show_raw: bool = False,
    show_legend: bool = True,
    return_fig: bool = False,
    return_data: bool = False,
    signaling_type=None,
):
    """
    Rank signaling pathways by their communication strength.
    Mirrors R rank_net().

    mode="single"      : rank pathways for one (single) CellChat object.
    mode="comparison"  : compare two datasets of a merged object. Each pathway
                         gets one bar per dataset (paired/stacked). When
                         do_stat=True a Wilcoxon test across source-target
                         pairs adds significance markers (mirrors R do.stat).
    """
    import matplotlib.pyplot as plt

    if mode == "comparison":
        if signaling_type is not None:
            raise ValueError("signaling_type is supported only for rank_net(mode='single').")
        return _rank_net_comparison(
            cellchat, slot_name=slot_name, measure=measure,
            sources_use=sources_use, targets_use=targets_use, signaling=signaling,
            thresh=thresh, color_use=color_use, title=title, font_size=font_size,
            fig_size=fig_size, comparison=comparison, stacked=stacked,
            do_stat=do_stat, paired_test=paired_test,
            cutoff_pvalue=cutoff_pvalue, tol=tol, show_raw=show_raw,
            show_legend=show_legend,
            return_fig=return_fig, return_data=return_data,
        )

    source_net = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    net_data = source_net
    pathway_names = network_names(source_net)
    prob = stack_network_field(source_net, 'prob', pathway_names, 0.0)
    pval = stack_network_field(source_net, 'pval', pathway_names, 1.0)

    if prob is None:
        raise ValueError("Run compute_pathway_probability first.")

    prob_f = prob.copy()
    if pval is not None:
        prob_f[pval >= thresh] = 0.0

    cluster_names = list(cellchat.groups.categories)

    # Filter sources/targets
    if sources_use is not None:
        src_idx = [i for i, n in enumerate(cluster_names) if n in sources_use]
        mask = np.zeros(prob_f.shape[0], dtype=bool)
        mask[src_idx] = True
        prob_f[~mask, :, :] = 0.0
    if targets_use is not None:
        tgt_idx = [i for i, n in enumerate(cluster_names) if n in targets_use]
        mask = np.zeros(prob_f.shape[1], dtype=bool)
        mask[tgt_idx] = True
        prob_f[:, ~mask, :] = 0.0

    # Compute scores per pathway
    if measure == "count":
        scores = np.sum(prob_f > 0, axis=(0, 1))
    else:
        scores = np.sum(prob_f, axis=(0, 1))

    df = pd.DataFrame({'pathway': pathway_names, 'score': scores})
    df = df[df['score'] > 0]

    if signaling_type is not None:
        types = [signaling_type] if isinstance(signaling_type, str) else list(signaling_type)
        if not types:
            raise ValueError("signaling_type must contain at least one annotation.")
        types = {str(value) for value in types}
        interaction_table = cellchat.database.get("interaction", pd.DataFrame())
        if not {"annotation", "pathway_name"}.issubset(interaction_table.columns):
            raise ValueError(
                "The database interaction table must contain annotation and pathway_name."
            )
        if slot_name == "pathway_network":
            allowed_names = set(
                interaction_table.loc[
                    interaction_table["annotation"].astype(str).isin(types),
                    "pathway_name",
                ].astype(str)
            )
        else:
            allowed_names = set(
                interaction_table.loc[
                    interaction_table["annotation"].astype(str).isin(types),
                    "interaction_name",
                ].astype(str)
            ) if "interaction_name" in interaction_table.columns else set()
        df = df[df["pathway"].astype(str).isin(allowed_names)]

    if signaling is not None:
        if isinstance(signaling, str):
            signaling = [signaling]
        df = df[df['pathway'].isin(signaling)]

    df = df.sort_values('score', ascending=True).reset_index(drop=True)

    if return_data:
        return df

    if color_use is None:
        bar_color = '#4DAF4A'
    else:
        bar_color = color_use[0] if isinstance(color_use, list) else color_use

    fig, ax = plt.subplots(figsize=fig_size)
    ax.barh(df['pathway'], df['score'], color=bar_color, edgecolor='white')
    ax.set_xlabel('Communication strength' if measure == 'weight' else 'Number of interactions',
                  fontsize=font_size)
    ax.set_title(title or 'Signaling pathway ranking', fontsize=font_size + 1, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=font_size - 1)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# rank_net comparison mode helper (mirrors R rank_net mode="comparison")
# ---------------------------------------------------------------------------
def _rank_net_comparison(
    cellchat, slot_name="pathway_network", measure="weight",
    sources_use=None, targets_use=None, signaling=None, thresh=0.05,
    color_use=None, title=None, font_size=10, fig_size=(6, 8),
    comparison=(0, 1), stacked=False, do_stat=False, show_legend=True,
    paired_test=True, cutoff_pvalue=0.05, tol=0.05, show_raw=False,
    return_fig=False, return_data=False,
):
    """Compare information flow of two merged datasets following R ``rank_net``."""
    import matplotlib.pyplot as plt
    from scipy import stats as _stats

    net_all = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if not isinstance(net_all, dict):
        raise ValueError("rank_net(mode='comparison') requires a merged CellChat object.")

    ds_keys = [k for k, v in net_all.items()
               if isinstance(v, dict) and 'prob' in v]
    if len(comparison) != 2:
        raise ValueError("rank_net(mode='comparison') currently requires exactly two datasets.")
    if any(index < 0 or index >= len(ds_keys) for index in comparison):
        raise ValueError(f"Merged object has datasets {ds_keys}; comparison={comparison} out of range.")
    name1 = ds_keys[comparison[0]]
    name2 = ds_keys[comparison[1]]

    def _dataset_groups(dataset_name, nd):
        groups = list(nd.get('groups', []))
        if groups:
            return groups
        raise ValueError(f"Merged network for {dataset_name!r} is missing its group order.")

    def _selected_indices(groups, selected):
        if selected is None:
            return None
        selected = [selected] if isinstance(selected, (str, int, np.integer)) else list(selected)
        if not groups:
            raise ValueError("sources_use/targets_use requires group names in the merged object.")
        if all(isinstance(value, str) for value in selected):
            unknown = set(selected).difference(groups)
            if unknown:
                raise ValueError(f"Unknown cell groups: {sorted(unknown)}")
            return [groups.index(value) for value in selected]
        indices = [int(value) for value in selected]
        if any(value < 0 or value >= len(groups) for value in indices):
            raise ValueError("Numerical sources_use/targets_use indices are zero-based group positions.")
        return indices

    def _pathway_scores(dataset_name, nd):
        """Return raw pathway totals and source-target vectors after R filtering."""
        if not isinstance(nd, Mapping) or 'prob' not in nd:
            raise ValueError("Run compute_pathway_probability before rank_net.")
        pathways = network_names(nd)
        prob = stack_network_field(nd, 'prob', pathways, 0.0)
        pval = stack_network_field(nd, 'pval', pathways, 1.0)
        groups = _dataset_groups(dataset_name, nd)
        prob_f = prob.copy().astype(float)
        if pval is not None:
            prob_f[pval >= thresh] = 0.0
        if measure == "count":
            prob_f = (prob_f > 0).astype(float)

        source_indices = _selected_indices(groups, sources_use)
        target_indices = _selected_indices(groups, targets_use)
        if source_indices is not None:
            m = np.zeros(prob_f.shape[0], dtype=bool)
            m[source_indices] = True
            prob_f[~m, :, :] = 0.0
        if target_indices is not None:
            m = np.zeros(prob_f.shape[1], dtype=bool)
            m[target_indices] = True
            prob_f[:, ~m, :] = 0.0

        if np.sum(prob_f) == 0:
            raise ValueError("No inferred communications for the input.")

        totals = {}
        vectors = {}
        for pi, pw in enumerate(pathways):
            sl = prob_f[:, :, pi]
            totals[pw] = float(np.sum(sl))
            vectors[pw] = sl.ravel(order='F')
        return pathways, totals, vectors, prob_f.shape

    pathways1, raw1, vectors1, shape1 = _pathway_scores(name1, net_all[name1])
    pathways2, raw2, vectors2, shape2 = _pathway_scores(name2, net_all[name2])
    all_pw = list(dict.fromkeys(pathways1 + pathways2))
    if signaling is not None:
        sig = [signaling] if isinstance(signaling, str) else signaling
        all_pw = [p for p in all_pw if p in sig]

    def _scaled_values(raw):
        if measure == "count":
            return np.asarray(raw, dtype=float), np.zeros(len(raw), dtype=bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = -1.0 / np.log(np.asarray(raw, dtype=float))
        scaled[np.isnan(scaled)] = 0.0
        invalid = np.isinf(scaled) | (scaled < 0)
        return scaled, invalid

    raw_values1 = np.asarray([raw1.get(pw, 0.0) for pw in all_pw], dtype=float)
    raw_values2 = np.asarray([raw2.get(pw, 0.0) for pw in all_pw], dtype=float)
    scaled1, invalid1 = _scaled_values(raw_values1)
    scaled2, invalid2 = _scaled_values(raw_values2)

    # R replaces invalid -1/log(flow) values globally, ranked by raw flow.
    if measure == "weight":
        invalid_raw = np.concatenate((raw_values1[invalid1], raw_values2[invalid2]))
        if invalid_raw.size:
            finite_scaled = np.concatenate((scaled1[~invalid1], scaled2[~invalid2]))
            max_scaled = float(np.max(finite_scaled)) if finite_scaled.size else 0.0
            assigned = np.linspace(max_scaled * 1.1, max_scaled * 1.5, invalid_raw.size)
            rank = np.empty(invalid_raw.size, dtype=int)
            rank[np.argsort(invalid_raw, kind='stable')] = np.arange(invalid_raw.size)
            replacements = assigned[rank]
            count1 = int(np.sum(invalid1))
            scaled1[invalid1] = replacements[:count1]
            scaled2[invalid2] = replacements[count1:]

    with np.errstate(divide='ignore', invalid='ignore'):
        relative = raw_values2 / raw_values1
    relative[np.isnan(relative)] = 0.0
    relative = np.asarray([
        float(f"{value:.1g}") if np.isfinite(value) else value for value in relative
    ], dtype=float)

    pvalues = np.zeros(len(all_pw), dtype=float)
    if do_stat:
        if paired_test and shape1[:2] != shape2[:2]:
            raise ValueError(
                "Paired test is not applicable to datasets with different cellular "
                "compositions. Set do_stat=False or paired_test=False."
            )
        for i, pw in enumerate(all_pw):
            vec1 = vectors1.get(pw)
            vec2 = vectors2.get(pw)
            if vec1 is None:
                vec1 = np.full(shape1[0] * shape1[1], np.nan)
            if vec2 is None:
                vec2 = np.full(shape2[0] * shape2[1], np.nan)
            if vec1.shape != vec2.shape:
                if not paired_test:
                    values1 = vec1[np.isfinite(vec1)]
                    values2 = vec2[np.isfinite(vec2)]
                    if values1.size > 3 and values2.size > 3:
                        pvalues[i] = _stats.ranksums(values1, values2).pvalue
                continue
            values = np.column_stack((vec1, vec2))
            values = values[np.nansum(values, axis=1) != 0]
            if values.shape[0] > 3 and not np.isnan(values).any():
                try:
                    if paired_test:
                        pvalues[i] = _stats.wilcoxon(values[:, 0], values[:, 1]).pvalue
                    else:
                        pvalues[i] = _stats.ranksums(values[:, 0], values[:, 1]).pvalue
                except ValueError:
                    pvalues[i] = 0.0

    rows = []
    for i, pw in enumerate(all_pw):
        if raw_values1[i] + raw_values2[i] == 0:
            continue
        rows.append({
            'pathway': pw,
            name1: raw_values1[i],
            name2: raw_values2[i],
            f'{name1}_scaled': scaled1[i],
            f'{name2}_scaled': scaled2[i],
            'contribution_relative': relative[i],
            'pvalue': pvalues[i],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        warnings.warn("No significant pathways for rank_net comparison.")
        return None

    df = df.assign(_ratio=-df['contribution_relative'], _second=-df[name2])
    df = df.sort_values(['_ratio', name1, '_second'], kind='mergesort').drop(
        columns=['_ratio', '_second']
    ).reset_index(drop=True)

    if color_use is None:
        # R reverses ggPalette() before coord_flip(), so the final plot maps
        # the first compared dataset to coral and the second to cyan.
        color_use = ['#F8766D', '#00BFC4']
    if len(color_use) < 2:
        raise ValueError("color_use must provide one color for each compared dataset.")
    c1, c2 = color_use[0], color_use[1]

    label_colors = []
    for _, row in df.iterrows():
        significant = (not do_stat) or row['pvalue'] < cutoff_pvalue
        if significant and row['contribution_relative'] < 1 - tol:
            label_colors.append(c1)
        elif significant and row['contribution_relative'] > 1 + tol:
            label_colors.append(c2)
        else:
            label_colors.append('black')
    df['label_color'] = label_colors

    if stacked:
        totals = df[name1] + df[name2]
        df['display_' + name1] = df[name1] / totals
        df['display_' + name2] = df[name2] / totals
    elif show_raw or measure == 'count':
        df['display_' + name1] = df[name1]
        df['display_' + name2] = df[name2]
    else:
        df['display_' + name1] = df[f'{name1}_scaled']
        df['display_' + name2] = df[f'{name2}_scaled']

    if return_data:
        return df

    fig, ax = plt.subplots(figsize=fig_size)
    y = np.arange(len(df))
    labels = df['pathway'].tolist()
    display1 = df['display_' + name1]
    display2 = df['display_' + name2]

    if stacked:
        ax.barh(y, display1, color=c1, edgecolor='white', label=name1)
        ax.barh(y, display2, left=display1, color=c2, edgecolor='white', label=name2)
        ax.axvline(0.5, linestyle='--', color='grey', linewidth=0.8)
    else:
        h = 0.375
        ax.barh(y - h / 2, display1, height=h, color=c1, edgecolor='white', label=name1)
        ax.barh(y + h / 2, display2, height=h, color=c2, edgecolor='white', label=name2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=font_size - 2)
    for tick, color in zip(ax.get_yticklabels(), df['label_color']):
        tick.set_color(color)
    if stacked:
        xlabel = 'Relative information flow' if measure == 'weight' else 'Relative number of interactions'
    else:
        xlabel = 'Information flow' if measure == 'weight' else 'Number of interactions'
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_title(title or 'Compare signaling information flow',
                 fontsize=font_size + 1, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    if show_legend:
        ax.legend(fontsize=font_size - 1, frameon=False, loc='lower right')

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# plot_gene_expression  -  violin/dot/bar plot of gene expression
# ---------------------------------------------------------------------------
def plot_gene_expression(
    cellchat: 'CellChat',
    signaling: str = None,
    features=None,
    enriched_only: bool = True,
    thresh: float = 0.05,
    plot_type: str = "violin",
    color_use=None,
    split_by: str = None,
    fig_size=None,
    return_fig: bool = False
):
    """
    Plot gene expression for signaling genes.
    Mirrors R plot_gene_expression().

    split_by : metadata column name (e.g. "cellchat_dataset"). When given, violins are split
        by that grouping side-by-side within each cell type (mirrors R split.by).
    """
    import matplotlib.pyplot as plt

    valid_plot_types = {"violin", "dot", "bar"}
    if plot_type not in valid_plot_types:
        raise ValueError(
            f"plot_type must be one of {sorted(valid_plot_types)}; got {plot_type!r}."
        )

    if features is not None and signaling is not None:
        warnings.warn(
            "Both features and signaling were provided; features will be used."
        )

    # For a merged object, select one dataset's canonical networks to resolve
    # the enriched L-R genes for the requested signaling pathway.
    lr_source = cellchat
    pathway_network = cellchat.pathway_network
    if isinstance(pathway_network, dict) and 'prob' not in pathway_network:
        ds_keys = [k for k, v in pathway_network.items()
                   if isinstance(v, dict) and 'prob' in v]
        if ds_keys:
            class _Shim:
                pass
            sh = _Shim()
            # prefer a dataset that actually contains the pathway
            chosen = ds_keys[0]
            if signaling is not None:
                want = signaling if isinstance(signaling, str) else (signaling[0] if signaling else None)
                for k in ds_keys:
                    if want in network_names(pathway_network[k]):
                        chosen = k
                        break
            sh.network = cellchat.network.get(chosen, {})
            sh.pathway_network = pathway_network[chosen]
            sh.lr_pairs = cellchat.lr_pairs.get(chosen, {})
            sh.groups = cellchat.groups
            sh.database = cellchat.database
            lr_source = sh

    # Resolve the genes represented by the requested signaling pathway.
    if features is None and signaling is not None:
        res = extract_enriched_lr(lr_source, signaling=signaling,
                                   gene_lr_return=True,
                                   enriched_only=enriched_only,
                                   thresh=thresh)
        if isinstance(res, dict):
            gene_lr = res.get('gene_lr', [])
        elif isinstance(res, pd.DataFrame):
            # Extract genes from ligand/receptor columns
            gene_lr = []
            if 'ligand' in res.columns:
                gene_lr.extend(res['ligand'].dropna().tolist())
            if 'receptor' in res.columns:
                gene_lr.extend(res['receptor'].dropna().tolist())
            gene_lr = list(set(gene_lr))
        else:
            gene_lr = []
        features = gene_lr

    if isinstance(features, str):
        features = [features]
    elif features is not None:
        features = list(features)
    features = list(dict.fromkeys(str(feature) for feature in (features or [])))
    if not features:
        detail = (
            f" for signaling pathway {signaling!r} with "
            f"enriched_only={enriched_only} and thresh={thresh}"
            if signaling is not None else ""
        )
        raise ValueError(f"No genes are available to plot{detail}.")

    available_mask = np.ones(cellchat.n_vars, dtype=bool)
    if "is_signaling" in cellchat.var:
        available_mask = cellchat.var["is_signaling"].fillna(False).to_numpy(dtype=bool)
    available_names = set(cellchat.var_names[available_mask].astype(str))
    features = [feature for feature in features if feature in available_names]
    if not features:
        raise ValueError(
            "None of the requested genes were found in the signaling expression layer."
        )

    from .visualization import plot_bar, plot_dot, plot_stacked_violin

    plotter = {
        "violin": plot_stacked_violin,
        "dot": plot_dot,
        "bar": plot_bar,
    }[plot_type]
    plot_kwargs = {
        "cellchat": cellchat,
        "features": features,
        "slot_data": "signaling",
        "color_use": color_use,
        "fig_size": fig_size,
        "return_fig": True,
    }
    if plot_type == "violin":
        plot_kwargs["split_by"] = split_by
    figure = plotter(**plot_kwargs)
    figure.suptitle(
        f"{signaling} signaling" if signaling else "Gene Expression",
        fontsize=10,
        fontweight="bold",
    )
    if return_fig:
        return figure
    plt.show()
    return None


# ---------------------------------------------------------------------------
# Network structural metrics (mirrors R entropia, node_distance, nnd,
# alpha_centrality, computeNetD_structure)
# ---------------------------------------------------------------------------

def _entropia(a: np.ndarray) -> float:
    """Shannon entropy of positive elements (mirrors R entropia)."""
    a = a[a > 0]
    if len(a) == 0:
        return 0.0
    return float(-np.sum(a * np.log(a)))


def _node_distance(adj_matrix: np.ndarray) -> np.ndarray:
    """Node distance PDF from unweighted shortest paths (mirrors R node_distance)."""
    import networkx as nx
    n = adj_matrix.shape[0]
    if n == 1:
        return np.ones((1, 1))

    G = nx.DiGraph(adj_matrix)
    try:
        m = nx.floyd_warshall_numpy(G)
    except Exception:
        G_undir = nx.Graph(adj_matrix)
        m = nx.floyd_warshall_numpy(G_undir)
    m[np.isinf(m)] = n
    uniq_vals = sorted(set(m.ravel()) - {0})

    a = np.zeros((n, n))
    for dval in uniq_vals:
        col = int(dval) - 1
        if 0 <= col < n:
            # R counts ``which(m == d)`` by matrix column after dividing the
            # 1-based column-major indices by n.  This is equivalent to the
            # number of source nodes at distance d for each target node.
            a[:, col] = np.sum(m == dval, axis=0)
    return a / (n - 1)


def _nnd(adj_matrix: np.ndarray) -> np.ndarray:
    """NND vector for a network (mirrors R nnd)."""
    N = adj_matrix.shape[0]
    nd = _node_distance(adj_matrix)
    pdfm = nd.mean(axis=0)
    nz = np.count_nonzero(pdfm[:N - 1]) + 1
    norm = max(np.log(max(2, nz)), 1e-10)
    ent_avg = _entropia(pdfm)
    ent_full = _entropia(nd.ravel()) / N
    return np.append(pdfm, max(0.0, ent_avg - ent_full) / norm)


def _alpha_centrality(adj_matrix: np.ndarray) -> np.ndarray:
    """Alpha centrality vector (mirrors R alpha_centrality)."""
    import networkx as nx
    N = adj_matrix.shape[0]
    G = nx.DiGraph(adj_matrix)
    deg = np.array([G.degree(i) for i in range(N)], dtype=float)
    exo = deg / max(N - 1, 1)
    alpha = 1.0 / N
    try:
        beta = {i: float(exo[i]) for i in range(N)}
        ac = nx.katz_centrality(
            G,
            alpha=alpha,
            beta=beta,
            normalized=False,
            weight=None,
            max_iter=10000,
            tol=1e-12,
        )
        ac_raw = np.array([ac.get(i, 0.0) for i in range(N)], dtype=float)
        r = np.sort(ac_raw) / (N * N)
    except Exception:
        A = np.asarray(adj_matrix, dtype=float)
        try:
            ac_raw = np.linalg.solve(np.eye(N) - alpha * A.T, exo)
        except np.linalg.LinAlgError:
            ac_raw = np.linalg.lstsq(np.eye(N) - alpha * A.T, exo, rcond=None)[0]
        ac_raw = np.nan_to_num(ac_raw, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.sort(ac_raw) / (N * N)
    return np.append(r, max(0.0, 1.0 - r.sum()))


def _compute_net_d_structure(g_adj: np.ndarray, h_adj: np.ndarray,
                             w1: float = 0.45, w2: float = 0.45,
                             w3: float = 0.1) -> float:
    """Structural distance between two directed networks (mirrors R computeNetD_structure)."""
    import networkx as nx
    N = g_adj.shape[0]
    M = h_adj.shape[0]

    first = second = third = 0.0

    if w1 + w2 > 0:
        pg = _nnd(g_adj)
        PM = np.zeros(max(N, M))
        PM[:N - 1] = pg[:N - 1]
        PM[-1] = pg[N - 1]
        ph = _nnd(h_adj)
        PM[:M - 1] += ph[:M - 1]
        PM[-1] += ph[M - 1]
        PM /= 2.0

        e_pg = _entropia(pg[:N])
        e_ph = _entropia(ph[:M])
        e_pm = _entropia(PM)
        denom = max(np.log(2), 1e-10)
        first = np.sqrt(max((e_pm - (e_pg + e_ph) / 2.0) / denom, 0))
        second = abs(np.sqrt(pg[N]) - np.sqrt(ph[M]))

    if w3 > 0:
        pg_a = _alpha_centrality(g_adj)
        ph_a = _alpha_centrality(h_adj)
        m = max(len(pg_a), len(ph_a))
        Pg = np.zeros(m)
        Ph = np.zeros(m)
        Pg[m - len(pg_a):] = pg_a
        Ph[m - len(ph_a):] = ph_a
        third += np.sqrt((_entropia((Pg + Ph) / 2.0) -
                          (_entropia(pg_a) + _entropia(ph_a)) / 2.0) / denom) / 2.0

        # R igraph::graph.complementer excludes loops; keep the diagonal zero
        # even when the original pathway network has autocrine/self edges.
        gc_adj = 1.0 - (g_adj > 0).astype(float)
        hc_adj = 1.0 - (h_adj > 0).astype(float)
        np.fill_diagonal(gc_adj, 0.0)
        np.fill_diagonal(hc_adj, 0.0)
        pg_a_c = _alpha_centrality(gc_adj)
        ph_a_c = _alpha_centrality(hc_adj)
        m2 = max(len(pg_a_c), len(ph_a_c))
        Pg2 = np.zeros(m2)
        Ph2 = np.zeros(m2)
        Pg2[m2 - len(pg_a_c):] = pg_a_c
        Ph2[m2 - len(ph_a_c):] = ph_a_c
        third += np.sqrt((_entropia((Pg2 + Ph2) / 2.0) -
                          (_entropia(pg_a_c) + _entropia(ph_a_c)) / 2.0) / denom) / 2.0

    return w1 * first + w2 * second + w3 * third


# ---------------------------------------------------------------------------
# compute_pairwise_network_similarity  (mirrors R computeNetSimilarityPairwise)
# ---------------------------------------------------------------------------
def compute_pairwise_network_similarity(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    type: str = "functional",
    comparison: Optional[List[int]] = None,
    k: Optional[int] = None,
    thresh: Optional[float] = None
) -> 'CellChat':
    """
    Compute signaling network similarity across paired datasets.
    Mirrors R computeNetSimilarityPairwise().
    """
    if comparison is None:
        metadata = cellchat.obs
        if 'cellchat_dataset' in metadata.columns:
            n_datasets = metadata['cellchat_dataset'].nunique()
            comparison = list(range(n_datasets))
        else:
            raise ValueError(
                "No comparison indices provided and no 'cellchat_dataset' column in cellchat.obs."
            )

    comparison_name = "-".join(str(c) for c in comparison)
    print(f"Compute signaling network similarity for datasets {comparison}")

    if slot_name == "pathway_network":
        all_net = cellchat.pathway_network
    else:
        all_net = cellchat.network

    net_list = []
    signaling_all = []
    net_name_all = []

    dataset_keys = _merged_dataset_keys(all_net)
    is_merged = len(dataset_keys) > 0

    if is_merged:
        for i in range(len(comparison)):
            idx = comparison[i]
            if idx < 0 or idx >= len(dataset_keys):
                raise ValueError(f"Comparison index {comparison[i]} out of range.")
            key = dataset_keys[idx]
            source_net = all_net[key]
            if 'prob' not in source_net:
                raise ValueError(f"No prob in {slot_name}[{key}]")
            pnames = network_names(source_net)
            prob = stack_network_field(source_net, 'prob', pnames, 0.0)
            net_list.append(prob)
            signaling_all.extend([f"{p}--{key}" for p in pnames])
            net_name_all.append(key)
    else:
        if not isinstance(all_net, Mapping) or 'prob' not in all_net:
            raise ValueError(f"No prob in {slot_name}")
        pnames = network_names(all_net)
        prob = stack_network_field(all_net, 'prob', pnames, 0.0)
        net_list.append(prob)
        signaling_all = list(pnames)
        net_name_all = ["Dataset_1"]

    net_dim = [net.shape[2] for net in net_list]
    nnet = sum(net_dim)
    position = np.cumsum([0] + net_dim)

    if k is None:
        k = int(np.ceil(np.sqrt(nnet))) + (1 if nnet > 25 else 0)
    if nnet <= 1:
        k = 1
    else:
        k = max(2, min(k, nnet - 1))

    if thresh is not None:
        for i in range(len(net_list)):
            vals = net_list[i][net_list[i] != 0]
            if len(vals) > 0:
                cutoff = np.quantile(vals, thresh)
                net_list[i][net_list[i] < cutoff] = 0.0

    if type == "functional":
        S3 = np.zeros((nnet, nnet))
        for i in range(nnet):
            idx_i = np.searchsorted(position, i + 1)
            net_i = net_list[idx_i - 1]
            Gi = (net_i[:, :, i - position[idx_i - 1]] > 0).astype(float)
            for j in range(nnet):
                idx_j = np.searchsorted(position, j + 1)
                net_j = net_list[idx_j - 1]
                Gj = (net_j[:, :, j - position[idx_j - 1]] > 0).astype(float)
                inter = np.sum(Gi * Gj)
                union = np.sum(Gi + Gj - Gi * Gj)
                S3[i, j] = inter / union if union > 0 else 0.0
        S3[np.isnan(S3)] = 0.0
        np.fill_diagonal(S3, 1.0)
        S_signalings = S3
    else:
        D_signalings = np.zeros((nnet, nnet))
        for i in range(nnet):
            idx_i = np.searchsorted(position, i + 1)
            net_i = net_list[idx_i - 1]
            Gi = (net_i[:, :, i - position[idx_i - 1]] > 0).astype(float)
            for j in range(nnet):
                idx_j = np.searchsorted(position, j + 1)
                net_j = net_list[idx_j - 1]
                Gj = (net_j[:, :, j - position[idx_j - 1]] > 0).astype(float)
                D_signalings[i, j] = _compute_net_d_structure(Gi, Gj)
        D_signalings[~np.isfinite(D_signalings)] = 0.0
        D_signalings[np.isnan(D_signalings)] = 0.0
        S_signalings = 1.0 - D_signalings

    SNN = _build_snn(S_signalings, k=k, prune_snn=1.0 / 15)
    Similarity = S_signalings * SNN
    np.fill_diagonal(Similarity, 1.0)

    sim_data = _network_similarity_data(cellchat, slot_name, type)
    sim_data.setdefault('matrix', {})[comparison_name] = Similarity
    _set_similarity_pathways(sim_data, comparison_name, signaling_all)
    sim_data['pathways_all'] = signaling_all
    sim_data['dataset_names'] = list(net_name_all)
    sim_data['comparison'] = list(comparison)
    sim_data['comparison_name'] = comparison_name
    # Python-only diagnostics: keep the SNN neighbour count distinct from the
    # k-means cluster count recorded by cluster_network().
    sim_data['snn_k'] = int(k)
    return cellchat


# ---------------------------------------------------------------------------
# rankSimilarity  (mirrors R rankSimilarity)
# ---------------------------------------------------------------------------
def rank_similarity(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    type: str = "functional",
    comparison1: Optional[List[int]] = None,
    comparison2: List[int] = None,
    font_size: int = 8,
    color_use: Optional[str] = None,
    title: Optional[str] = None,
    return_fig: bool = False
):
    """
    Rank shared pathway similarity between two datasets.
    Mirrors R rankSimilarity().
    """
    import matplotlib.pyplot as plt

    if comparison2 is None:
        comparison2 = [0, 1]

    if comparison1 is None:
        metadata = cellchat.obs
        if 'cellchat_dataset' in metadata.columns:
            n_datasets = metadata['cellchat_dataset'].nunique()
            comparison1 = list(range(n_datasets))
        else:
            comparison1 = list(range(2))

    comparison_name = "-".join(str(c) for c in comparison1)

    target = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    sim = _network_similarity_data(cellchat, slot_name, type)
    dr_all = sim.get('dr', {}).get(comparison_name)
    if dr_all is None:
        raise ValueError(f"No embedding for {comparison_name}. Run embed_network first.")

    all_keys = list(sim.get('dataset_names') or _merged_dataset_keys(target))

    if all(0 <= index < len(all_keys) for index in comparison2):
        name1 = all_keys[comparison2[0]]
        name2 = all_keys[comparison2[1]]
    else:
        name1 = f"Dataset_{comparison2[0] + 1}"
        name2 = f"Dataset_{comparison2[1] + 1}"

    dr_all = np.asarray(dr_all)
    n_pts = dr_all.shape[0]

    # embed_network removes isolated pathways before fitting UMAP. Pairwise
    # similarity keeps the original ``pathways_all`` labels, so prefer the
    # filtered labels stored by embed_network when available; otherwise recreate
    # the same keep mask from the similarity matrix.
    pathways_all = _pathways_for_similarity(sim, comparison_name, n_pts)
    if len(pathways_all) != n_pts:
        raw_pathways = list(sim.get('pathways_all', []) or [])
        matrix_all = sim.get('matrix', {}).get(comparison_name)
        if raw_pathways and matrix_all is not None:
            matrix_all = np.asarray(matrix_all)
            keep_mask = matrix_all.sum(axis=0) != 1.0
            if len(raw_pathways) == len(keep_mask) and int(np.sum(keep_mask)) == n_pts:
                pathways_all = [p for p, keep in zip(raw_pathways, keep_mask) if keep]
            else:
                pathways_all = raw_pathways
        else:
            pathways_all = raw_pathways
    if len(pathways_all) != n_pts:
        pathways_all = (pathways_all + [f'P{i}' for i in range(len(pathways_all), n_pts)])[:n_pts]

    labels = [p.replace(f'--{name1}', '').replace(f'--{name2}', '') for p in pathways_all]
    groups = []
    for p in pathways_all:
        if f'--{name1}' in p:
            groups.append(name1)
        elif f'--{name2}' in p:
            groups.append(name2)
        else:
            groups.append('other')

    data1 = dr_all[np.array(groups) == name1]
    data2 = dr_all[np.array(groups) == name2]
    labels1 = [labels[i] for i in range(len(labels)) if groups[i] == name1]
    labels2 = [labels[i] for i in range(len(labels)) if groups[i] == name2]

    common = sorted(set(labels1) & set(labels2))
    idx1 = [labels1.index(p) for p in common]
    idx2 = [labels2.index(p) for p in common]
    d1 = data1[idx1]
    d2 = data2[idx2]
    dists = np.sqrt(np.sum((d1 - d2) ** 2, axis=1))

    df = pd.DataFrame({'name': common, 'dist': dists})
    df = df.sort_values('dist', ascending=True).reset_index(drop=True)
    df['name'] = pd.Categorical(df['name'], categories=df['name'], ordered=True)

    fig, ax = plt.subplots(figsize=(5, max(4, len(df) * 0.25)))
    ax.barh(df['name'], df['dist'], color=color_use or '#4DAF4A', height=0.7)
    ax.set_xlabel('Pathway distance')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=font_size)
    if title:
        ax.set_title(title, fontsize=font_size + 2)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# compareInteractions  (mirrors R compareInteractions)
# ---------------------------------------------------------------------------
def compare_interactions(
    object_list: List['CellChat'],
    measure: str = "count",
    color_use: Optional[List[str]] = None,
    group: Optional[List] = None,
    group_levels: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    width: float = 0.6,
    size_text: int = 10,
    return_fig: bool = False
):
    """
    Bar plot comparing interactions count/strength across datasets.
    Mirrors R compareInteractions().
    """
    import matplotlib.pyplot as plt

    vals = []
    dataset_names = []
    for obj in object_list:
        net = obj.network if hasattr(obj, 'network') else {}
        if measure == "count":
            v = np.sum(net.get('count', 0))
        else:
            v = np.sum(net.get('weight', 0))
        vals.append(float(v))
        dataset_names.append(f"Dataset_{len(dataset_names) + 1}")

    df = pd.DataFrame({'dataset': dataset_names, 'value': vals})
    if group is None:
        group = ['Group 1'] * len(vals)
    df['group'] = group
    if group_levels is not None:
        df['group'] = pd.Categorical(df['group'], categories=group_levels)
    else:
        df['group'] = pd.Categorical(df['group'])

    if ylabel is None:
        ylabel = "Number of inferred interactions" if measure == "count" else "Interaction strength"

    if color_use is None:
        from .visualization import gg_palette
        color_use = gg_palette(df['group'].nunique())

    fig, ax = plt.subplots(figsize=(max(4, len(dataset_names) * 1.2), 4))
    groups = df['group'].cat.categories
    bar_width = width / max(len(groups), 1)
    x = np.arange(len(dataset_names))
    for gi, grp in enumerate(groups):
        mask = df['group'] == grp
        offset = (gi - len(groups) / 2 + 0.5) * bar_width
        ax.bar(x[mask] + offset, df.loc[mask, 'value'], bar_width,
               color=color_use[gi % len(color_use)], label=grp)
        for xi, vi in zip(x[mask] + offset, df.loc[mask, 'value']):
            ax.text(xi, vi, f'{vi:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=size_text)
    ax.set_ylabel(ylabel, fontsize=size_text)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size_text)
    if title_name:
        ax.set_title(title_name, fontsize=size_text + 2)
    ax.spines[['top', 'right']].set_visible(False)
    if len(groups) > 1:
        ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# rank_network_pairwise
# ---------------------------------------------------------------------------
def rank_network_pairwise(
    cellchat: 'CellChat',
    lr_use: Optional[pd.DataFrame] = None
) -> 'CellChat':
    """
    Rank L-R interactions per source-target pair.
    Mirrors R's pairwise network-ranking procedure.
    """
    if lr_use is None:
        if 'significant' in cellchat.lr_pairs and cellchat.lr_pairs['significant'] is not None:
            pair_lr_use = cellchat.lr_pairs['significant']
        else:
            raise ValueError("No LR pairs found. Run identify_overexpressed_interactions first.")
    else:
        pair_lr_use = lr_use

    net = cellchat.network
    if not isinstance(net, Mapping) or 'prob' not in net:
        raise ValueError("Run compute_communication_probability first.")
    names = network_names(net)
    prob = stack_network_field(net, 'prob', names, 0.0)
    pval = stack_network_field(net, 'pval', names, 1.0)

    num_cluster = prob.shape[0]
    cluster_names = list(net.get('groups', [f'C{i}' for i in range(num_cluster)]))
    pairwise_lr = []
    for i_source in range(num_cluster):
        temp = []
        for i_target in range(num_cluster):
            pval_ij = np.asarray(pval[i_source, i_target, :]).ravel()
            prob_ij = np.asarray(prob[i_source, i_target, :]).ravel()
            data = pair_lr_use.copy()
            data['prob'] = prob_ij
            data['pval'] = pval_ij
            data = data.sort_values(['pval', 'prob'], ascending=[True, False]).reset_index(drop=True)
            temp.append(data)
        pairwise_lr.append({cluster_names[j]: temp[j] for j in range(num_cluster)})

    pairwise_lr = {cluster_names[i]: pairwise_lr[i] for i in range(num_cluster)}
    net['pairwise_rank'] = pairwise_lr
    cellchat.network = net
    return cellchat


# ---------------------------------------------------------------------------
# getMaxWeight  (mirrors R getMaxWeight)
# ---------------------------------------------------------------------------
def get_max_weight(
    object_list: List['CellChat'],
    slot_name: Union[str, List[str]] = "groups",
    attribute: Union[str, List[str]] = "groups"
) -> Dict[str, float]:
    """
    Get max values across objects for scaling visualization limits.
    Mirrors R getMaxWeight().
    """
    if isinstance(slot_name, str):
        slot_name = [slot_name]
    if isinstance(attribute, str):
        attribute = [attribute]

    weight = []
    for i in range(len(slot_name)):
        if slot_name[i] == "groups":
            weight_all = []
            for obj in object_list:
                groups = obj.groups
                if isinstance(groups, pd.Categorical):
                    w = max(pd.Series(groups).value_counts().values)
                    weight_all.append(w)
                else:
                    weight_all.append(len(groups))
            weight.append(max(weight_all))
        elif slot_name[i] == "network":
            if attribute[i] in ("count", "weight"):
                weight_all = []
                for obj in object_list:
                    net = obj.network
                    if attribute[i] in net:
                        weight_all.append(float(np.max(net[attribute[i]])))
                    else:
                        weight_all.append(0.0)
                weight.append(max(weight_all) if weight_all else 0.0)
            else:
                weight_all = []
                for obj in object_list:
                    prob = obj.network.get('prob')
                    if isinstance(prob, Mapping) and attribute[i] in prob:
                        matrix = prob[attribute[i]]
                        maximum = matrix.max() if sparse.issparse(matrix) else np.max(matrix)
                        weight_all.append(float(maximum))
                weight.append(max(weight_all) if weight_all else 0.0)
        elif slot_name[i] == "pathway_network":
            weight_all = []
            for obj in object_list:
                if hasattr(obj, 'pathway_network') and obj.pathway_network and 'prob' in obj.pathway_network:
                    prob = obj.pathway_network['prob']
                    if attribute[i] in prob:
                        matrix = prob[attribute[i]]
                        weight_all.append(float(matrix.max()))
            weight.append(max(weight_all) if weight_all else 0.0)

    return dict(zip(attribute, weight))


# ---------------------------------------------------------------------------
# mergeInteractions  (mirrors R mergeInteractions)
# ---------------------------------------------------------------------------
def merge_interactions(
    cellchat: 'CellChat',
    group_new: Union[List[str], pd.Categorical]
) -> 'CellChat':
    """
    Merge interactions by collapsing cell groups into coarser categories.
    Mirrors R mergeInteractions().
    """
    if not isinstance(group_new, pd.Categorical):
        group_new = pd.Categorical(group_new)

    net = cellchat.network
    count = net.get('count')
    weight = net.get('weight')
    if count is None:
        raise ValueError("Run aggregate_network first.")

    orig_groups = list(net.get('groups', [f'C{i}' for i in range(count.shape[0])]))
    if len(group_new) != len(orig_groups):
        raise ValueError(
            "group_new must contain one label for each group in cellchat.network."
        )
    if weight is None:
        weight = np.zeros_like(count, dtype=float)
    if np.shape(weight) != np.shape(count):
        raise ValueError(
            f"network['weight'] has shape {np.shape(weight)}; "
            f"expected {np.shape(count)}"
        )
    new_groups = list(group_new.categories)
    n_new = len(new_groups)

    merged_count = np.zeros((n_new, n_new), dtype=float)
    merged_weight = np.zeros((n_new, n_new), dtype=float)

    for i_new, ni in enumerate(new_groups):
        mask_i = np.array([g == ni for g in group_new])
        for j_new, nj in enumerate(new_groups):
            mask_j = np.array([g == nj for g in group_new])
            merged_count[i_new, j_new] = np.sum(count[np.ix_(mask_i, mask_j)])
            merged_weight[i_new, j_new] = np.sum(weight[np.ix_(mask_i, mask_j)])

    group_mapping = dict(zip(orig_groups, list(group_new)))
    observed_groups = np.asarray(cellchat.groups).astype(str)
    unknown_groups = sorted(set(observed_groups).difference(group_mapping))
    if unknown_groups:
        raise ValueError(
            "cellchat.groups contains labels missing from cellchat.network['groups']: "
            f"{unknown_groups}"
        )

    result = cellchat.copy()
    result.groups = pd.Categorical(
        [group_mapping[group] for group in observed_groups],
        categories=new_groups,
    )
    result.network = {
        'groups': new_groups,
        'prob': {},
        'pval': {},
        'count': merged_count,
        'weight': merged_weight,
    }
    return result


# ---------------------------------------------------------------------------
# netMappingDEG  (mirrors R netMappingDEG)
# ---------------------------------------------------------------------------
def map_network_deg(
    cellchat: 'CellChat',
    features_name: str,
    variable_all: bool = True,
    thresh: float = 0.05
) -> pd.DataFrame:
    """
    Map differential expression results onto inferred communications.
    Mirrors R netMappingDEG().
    """
    features_name_info = f"{features_name}_info"
    if features_name_info not in (cellchat.feature_results or {}):
        raise ValueError(
            f"features '{features_name_info}' not in var.features. Run identify_overexpressed_genes first."
        )

    DEG = cellchat.feature_results[features_name_info]
    database = cellchat.database
    gene_info = database.get('gene_info', pd.DataFrame())
    complex_input = database.get('complex', pd.DataFrame())

    # Get communication data
    from .modeling import subset_communication
    df_net = subset_communication(cellchat, thresh_pval=thresh)
    if not isinstance(df_net, pd.DataFrame):
        frames = []
        for name, dataset_net in df_net.items():
            # Every per-dataset table starts its pandas index at zero.  The
            # assignments below address rows by index, so retain a unique
            # merged row index just as R's vectorized cbind assignment does.
            frames.append(dataset_net.assign(cellchat_dataset=name))
        net = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        net = df_net.copy().reset_index(drop=True)

    net['source_ligand'] = net['source'].astype(str) + '.' + net['ligand'].astype(str)
    net['target_receptor'] = net['target'].astype(str) + '.' + net['receptor'].astype(str)
    DEG = DEG.copy()
    DEG['cluster_feature'] = DEG['clusters'].astype(str) + '.' + DEG['features'].astype(str)

    for prefix in ['ligand', 'receptor']:
        for suffix in ['pvalues', 'log_fc', 'pct_1', 'pct_2']:
            net[f'{prefix}_{suffix}'] = np.nan

    gene_symbols = set(gene_info['Symbol'].dropna()) if 'Symbol' in gene_info.columns else set()

    # Ligand mapping
    for _, row in net.iterrows():
        ligand = str(row['ligand'])
        source_ligand = row['source_ligand']

        if ligand in gene_symbols:
            idx = DEG[DEG['cluster_feature'] == source_ligand].index
            if len(idx) > 0:
                for suffix in ['pvalues', 'log_fc', 'pct_1', 'pct_2']:
                    net.at[row.name, f'ligand_{suffix}'] = DEG.at[idx[0], suffix]
        elif len(complex_input) > 0 and ligand in complex_input.index:
            sub_cols = [c for c in complex_input.columns if c.startswith('subunit')]
            subunits = [v for v in complex_input.loc[ligand, sub_cols] if isinstance(v, str) and v != '']
            source_ligand_complex = [f"{row['source']}.{s}" for s in subunits]
            idx_matches = DEG[DEG['cluster_feature'].isin(source_ligand_complex)].index
            if len(idx_matches) > 0:
                sub_df = DEG.loc[idx_matches, ['pvalues', 'log_fc', 'pct_1', 'pct_2']]
                if variable_all:
                    vals = sub_df.mean(axis=0)
                else:
                    vals = pd.Series({
                        'pvalues': sub_df['pvalues'].min(),
                        'log_fc': sub_df['log_fc'].max(),
                        'pct_1': sub_df['pct_1'].max(),
                        'pct_2': sub_df['pct_2'].max()
                    })
                for suffix in ['pvalues', 'log_fc', 'pct_1', 'pct_2']:
                    net.at[row.name, f'ligand_{suffix}'] = vals.get(suffix, np.nan)

    # Receptor mapping
    for _, row in net.iterrows():
        receptor = str(row['receptor'])
        target_receptor = row['target_receptor']

        if receptor in gene_symbols:
            idx = DEG[DEG['cluster_feature'] == target_receptor].index
            if len(idx) > 0:
                for suffix in ['pvalues', 'log_fc', 'pct_1', 'pct_2']:
                    net.at[row.name, f'receptor_{suffix}'] = DEG.at[idx[0], suffix]
        elif len(complex_input) > 0 and receptor in complex_input.index:
            sub_cols = [c for c in complex_input.columns if c.startswith('subunit')]
            subunits = [v for v in complex_input.loc[receptor, sub_cols] if isinstance(v, str) and v != '']
            target_receptor_complex = [f"{row['target']}.{s}" for s in subunits]
            idx_matches = DEG[DEG['cluster_feature'].isin(target_receptor_complex)].index
            if len(idx_matches) > 0:
                sub_df = DEG.loc[idx_matches, ['pvalues', 'log_fc', 'pct_1', 'pct_2']]
                if variable_all:
                    vals = sub_df.mean(axis=0)
                else:
                    vals = pd.Series({
                        'pvalues': sub_df['pvalues'].min(),
                        'log_fc': sub_df['log_fc'].max(),
                        'pct_1': sub_df['pct_1'].max(),
                        'pct_2': sub_df['pct_2'].max()
                    })
                for suffix in ['pvalues', 'log_fc', 'pct_1', 'pct_2']:
                    net.at[row.name, f'receptor_{suffix}'] = vals.get(suffix, np.nan)

    return net


# ---------------------------------------------------------------------------
# computeEnrichmentScore  (mirrors R computeEnrichmentScore)
# ---------------------------------------------------------------------------
def extract_gene_subset_from_pair(
    pair_lr: pd.DataFrame,
    cellchat: 'CellChat',
) -> List[str]:
    """Return the unique ligand/receptor genes involved in a set of LR pairs.

    Mirrors R extract_gene_subset_from_pair(): expands complexes via the database's
    complex table so that multi-subunit ligands/receptors contribute all subunits.
    """
    if pair_lr is None or len(pair_lr) == 0:
        return []

    database = cellchat.database
    complex_input = database.get('complex', pd.DataFrame())

    def _expand(name):
        name = str(name)
        if len(complex_input) > 0 and name in complex_input.index:
            sub_cols = [c for c in complex_input.columns if c.lower().startswith('subunit')]
            return [v for v in complex_input.loc[name, sub_cols]
                    if isinstance(v, str) and v.strip() != '']
        return [name]

    genes = set()
    for col in ('ligand', 'receptor'):
        if col in pair_lr.columns:
            for v in pair_lr[col].dropna():
                genes.update(_expand(v))

    # Fall back to parsing interaction_name when ligand/receptor columns absent
    if not genes and 'interaction_name' in pair_lr.columns:
        for v in pair_lr['interaction_name'].dropna():
            for part in str(v).split('_'):
                genes.update(_expand(part))

    return sorted(genes)


def compute_enrichment_score(
    df: pd.DataFrame,
    measure: str = "ligand",
    variable_both: bool = True,
    species: str = "human",
    database: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute DEG-based enrichment scores from mapped communication data.

    Scores can be summarized by ligand, signaling pathway, or L-R pair.
    Mirrors R computeEnrichmentScore() (without wordcloud dependency).
    """
    valid_measures = {"ligand", "signaling", "LR-pair"}
    if measure not in valid_measures:
        raise ValueError(
            "measure must be one of 'ligand', 'signaling', or 'LR-pair'; "
            f"got {measure!r}."
        )

    if 'interaction_name' not in df.columns:
        raise ValueError("DataFrame must have 'interaction_name' column. Run netMappingDEG first.")

    lr_pairs = df['interaction_name'].unique().tolist()
    ES = np.zeros(len(lr_pairs))
    for i, lrp in enumerate(lr_pairs):
        df_i = df[df['interaction_name'] == lrp].copy()
        if variable_both:
            df_i = df_i.dropna(subset=['ligand_log_fc', 'receptor_log_fc',
                                        'ligand_pct_1', 'ligand_pct_2',
                                        'receptor_pct_1', 'receptor_pct_2'])
        if len(df_i) == 0:
            ES[i] = np.nan
            continue
        lig_fc = np.abs(df_i['ligand_log_fc'].values)
        rec_fc = np.abs(df_i['receptor_log_fc'].values)
        lig_pct_diff = np.abs(df_i['ligand_pct_2'].values - df_i['ligand_pct_1'].values)
        rec_pct_diff = np.abs(df_i['receptor_pct_2'].values - df_i['receptor_pct_1'].values)
        ES[i] = np.nanmean(lig_fc * rec_fc * lig_pct_diff * rec_pct_diff)

    valid = ~np.isnan(ES)
    ES = ES[valid]
    lr_pairs = [lr_pairs[i] for i in range(len(lr_pairs)) if valid[i]]

    if len(ES) == 0:
        raise ValueError("No enriched signaling found.")

    if database is None:
        from .database import load_database
        database = load_database(species)

    interaction_db = database.get('interaction', pd.DataFrame())
    df_es = pd.DataFrame({'score': ES}, index=lr_pairs)
    if len(interaction_db) > 0:
        for col in ['ligand', 'receptor', 'pathway_name']:
            if col in interaction_db.columns:
                df_es[col] = [interaction_db.at[p, col] if p in interaction_db.index else ''
                              for p in lr_pairs]

    if measure == "LR-pair":
        df_ensemble = (
            df_es.rename_axis('interaction_name')
            .reset_index()
            .sort_values('score', ascending=False)
            .reset_index(drop=True)
        )
    else:
        group_col = 'ligand' if measure == "ligand" else 'pathway_name'
        if group_col not in df_es.columns:
            raise ValueError(
                f"The interaction database must contain {group_col!r} "
                f"to summarize measure={measure!r}."
            )
        df_ensemble = (
            df_es.groupby(group_col)['score']
            .sum()
            .sort_values(ascending=False)
            .reset_index(name='total')
        )

    return df_ensemble


# ---------------------------------------------------------------------------
# findEnrichedSignaling  (mirrors R findEnrichedSignaling)
# ---------------------------------------------------------------------------
def find_enriched_signaling(
    cellchat: 'CellChat',
    features: List[str],
    groups_use: Optional[List[str]] = None,
    pattern: str = "both",
    thresh: float = 0.05
) -> pd.DataFrame:
    """
    Find enriched signaling involving given genes and cell groups.
    Mirrors R findEnrichedSignaling().
    """
    if pattern not in ("both", "outgoing", "incoming"):
        raise ValueError("pattern must be 'both', 'outgoing', or 'incoming'")

    from .modeling import subset_communication
    df_net = subset_communication(cellchat, thresh_pval=thresh)

    if groups_use is not None:
        if pattern == "both":
            idx = df_net['source'].isin(groups_use) | df_net['target'].isin(groups_use)
        elif pattern == "outgoing":
            idx = df_net['source'].isin(groups_use)
        else:
            idx = df_net['target'].isin(groups_use)
        idx_feature = df_net['ligand'].isin(features) | df_net['receptor'].isin(features)
        df_sub = df_net[idx & idx_feature]
    else:
        if pattern == "both":
            idx_feature = df_net['ligand'].isin(features) | df_net['receptor'].isin(features)
        elif pattern == "outgoing":
            idx_feature = df_net['ligand'].isin(features)
        else:
            idx_feature = df_net['receptor'].isin(features)
        df_sub = df_net[idx_feature]

    return df_sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# computeLaplacian / computeEigengap  (mirrors R computeLaplacian / computeEigengap)
# ---------------------------------------------------------------------------
def compute_laplacian(
    communication_matrix: np.ndarray,
    tol: float = 0.01
) -> Dict[str, Any]:
    """
    Compute normalized graph Laplacian eigenvalues.
    Mirrors R computeLaplacian().
    """
    communication_matrix = np.asarray(communication_matrix, dtype=float)
    if communication_matrix.ndim != 2 or communication_matrix.shape[0] != communication_matrix.shape[1]:
        raise ValueError("communication_matrix must be a square two-dimensional matrix.")
    col_sums = communication_matrix.sum(axis=0)
    Dsq = np.sqrt(np.maximum(col_sums, 1e-15))
    laplacian = -communication_matrix / (Dsq[:, np.newaxis] * Dsq[np.newaxis, :])
    np.fill_diagonal(laplacian, 1 + np.diag(laplacian))

    # R uses eigenvalues of the dense normalized Laplacian.  The matrices here
    # are small pathway consensus matrices, so a dense symmetric eigensolver is
    # more stable than sparse eigsh(k=N) and avoids import-order issues.
    eigs = np.abs(np.real(np.linalg.eigvalsh(laplacian)))
    eigs = np.sort(eigs)[:min(100, len(eigs))]
    n_zeros = int(np.sum(eigs <= tol))
    return {'val': eigs, 'n_zeros': n_zeros}


def compute_eigengap(
    communication_matrix: np.ndarray,
    tau: Optional[float] = None,
    tol: float = 0.01
) -> Dict[str, Any]:
    """
    Determine optimal cluster number from consensus matrix eigengap.
    Mirrors R computeEigengap().
    """
    communication_matrix = np.asarray(communication_matrix, dtype=float)
    K_init = compute_laplacian(communication_matrix, tol=tol)['n_zeros']
    if tau is None:
        if K_init <= 5:
            tau = 0.3
        elif K_init <= 10:
            tau = 0.4
        else:
            tau = 0.5

    truncated_matrix = communication_matrix.copy()
    truncated_matrix[truncated_matrix <= tau] = 0.0
    truncated_matrix = (truncated_matrix + truncated_matrix.T) / 2.0

    eigs = compute_laplacian(truncated_matrix, tol=tol)
    gaps = np.diff(eigs['val'])
    upper_bound = int(np.argmax(gaps) + 1)
    lower_bound = eigs['n_zeros']

    return {
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'eigs': eigs
    }


# ---------------------------------------------------------------------------
# net_analysis_diff_signaling_role_scatter  (mirrors R)
# ---------------------------------------------------------------------------
def net_analysis_diff_signaling_role_scatter(
    cellchat: 'CellChat',
    color_use: Optional[List[str]] = None,
    comparison: List[int] = None,
    signaling: Optional[List[str]] = None,
    signaling_exclude: Optional[List[str]] = None,
    groups_exclude: Optional[List[str]] = None,
    slot_name: str = "pathway_network",
    group: Optional[List] = None,
    dot_size: float = 2.5,
    label_size: float = 3.0,
    dot_alpha: float = 0.6,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    tight_axes: bool = True,
    axis_pad: float = 0.08,
    font_size: int = 10,
    do_label: bool = True,
    show_legend: bool = True,
    return_fig: bool = False
):
    """
    Scatter plot of differential outgoing vs incoming signaling between two datasets.
    Mirrors R net_analysis_diff_signaling_role_scatter().
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    if comparison is None:
        comparison = [0, 1]

    if slot_name == "pathway_network":
        all_net = cellchat.pathway_network
    else:
        all_net = cellchat.network

    dataset_keys = _merged_dataset_keys(all_net)

    if len(dataset_keys) < 2:
        raise ValueError("Need merged CellChat object with at least 2 datasets.")

    if len(comparison) != 2 or any(index < 0 or index >= len(dataset_keys) for index in comparison):
        raise ValueError("comparison must contain two zero-based dataset indices.")
    name1 = dataset_keys[comparison[0]]
    name2 = dataset_keys[comparison[1]]
    print(f"Visualizing differential signaling from {name1} to {name2}")
    if title is None:
        title = f"Signaling changes ({name1} vs. {name2})"

    groups = cellchat.groups
    if groups is None or not hasattr(groups, 'categories'):
        raise ValueError("Merged object must define categorical cellchat.groups.")
    cell_levels = list(groups.categories)

    if xlabel is None:
        xlabel = "Differential outgoing interaction strength"
    if ylabel is None:
        ylabel = "Differential incoming interaction strength"

    if signaling is None:
        sig1 = network_names(all_net[dataset_keys[comparison[0]]])
        sig2 = network_names(all_net[dataset_keys[comparison[1]]])
        signaling = list(set(sig1) | set(sig2))
    if signaling_exclude:
        signaling = [s for s in signaling if s not in signaling_exclude]

    mat_all_merged = []
    for ii in range(len(comparison)):
        net_data = all_net[dataset_keys[comparison[ii]]]
        centrality = net_data.get('centrality', {})
        if not centrality:
            raise ValueError("Run compute_network_centrality first.")

        outgoing = np.zeros((len(cell_levels), len(centrality)))
        incoming = np.zeros((len(cell_levels), len(centrality)))
        for i, (pname, cvals) in enumerate(centrality.items()):
            outgoing[:, i] = cvals.get(x_measure, np.zeros(len(cell_levels)))
            incoming[:, i] = cvals.get(y_measure, np.zeros(len(cell_levels)))

        mat_out = outgoing.T
        mat_in = incoming.T

        mat_all = np.zeros((len(signaling), len(cell_levels), 2))
        for mat_idx, (mat, mat_name) in enumerate([(mat_out, 'outgoing'), (mat_in, 'incoming')]):
            for pi, sig_path in enumerate(signaling):
                if sig_path in centrality:
                    pii = list(centrality.keys()).index(sig_path)
                    if pii < mat.shape[0]:
                        mat_all[pi, :, int(mat_name == 'incoming')] = mat[pii, :]
        mat_all_merged.append(mat_all)

    mat_diff = mat_all_merged[1] - mat_all_merged[0]
    out_diff = mat_diff[:, :, 0].sum(axis=0)
    in_diff = mat_diff[:, :, 1].sum(axis=0)

    df = pd.DataFrame({'x': out_diff, 'y': in_diff, 'cellchat_group': cell_levels})
    df['cellchat_group'] = pd.Categorical(df['cellchat_group'], categories=cell_levels)

    if groups_exclude:
        df = df[~df['cellchat_group'].isin(groups_exclude)]
        cell_levels = [c for c in cell_levels if c not in groups_exclude]

    if color_use is None:
        color_use = sc_palette(len(cell_levels))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(cell_levels):
        row = df[df['cellchat_group'] == name]
        if len(row) == 0:
            continue
        c = color_use[i] if i < len(color_use) else '#999999'
        shape = 'o'
        if group is not None and i < len(group):
            shapes = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            shape = shapes[group[i] % len(shapes)] if isinstance(group[i], int) else 'o'
        ax.scatter(row['x'].values[0], row['y'].values[0],
                   s=dot_size * 30, c=[to_rgba(c, dot_alpha)],
                   edgecolors=c, linewidths=0.8, marker=shape, zorder=3)
        if do_label:
            ax.annotate(name, (row['x'].values[0], row['y'].values[0]),
                       fontsize=label_size * 2.5, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, linestyle='dashed', color='grey', linewidth=0.5)
    ax.axvline(0, linestyle='dashed', color='grey', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size, fontweight='normal')
    ax.spines[['top', 'right']].set_visible(False)

    def _axis_limits(values, explicit_limits):
        if explicit_limits is not None:
            return explicit_limits
        if not tight_axes:
            return None
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo = min(float(vals.min()), 0.0)
        hi = max(float(vals.max()), 0.0)
        span = hi - lo
        if span < 1e-12:
            pad = max(abs(hi), 1.0) * axis_pad
        else:
            pad = span * axis_pad
        return lo - pad, hi + pad

    xlim_use = _axis_limits(df['x'].values, xlims)
    ylim_use = _axis_limits(df['y'].values, ylims)
    if xlim_use is not None:
        ax.set_xlim(xlim_use)
    if ylim_use is not None:
        ax.set_ylim(ylim_use)

    if show_legend:
        handles = [mpatches.Patch(color=color_use[i] if i < len(color_use) else '#999999',
                                  label=cell_levels[i]) for i in range(len(cell_levels))]
        ax.legend(handles=handles, fontsize=7, frameon=False,
                 bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# net_analysis_signaling_changes_scatter  (mirrors R)
# ---------------------------------------------------------------------------
def net_analysis_signaling_changes_scatter(
    cellchat: 'CellChat',
    group_use: str,
    color_use: Optional[List[str]] = None,
    comparison: List[int] = None,
    signaling: Optional[List[str]] = None,
    signaling_label: Optional[List[str]] = None,
    top_label: float = 1.0,
    signaling_exclude: Optional[List[str]] = None,
    slot_name: str = "pathway_network",
    dot_size: float = 2.5,
    label_size: float = 3.0,
    dot_alpha: float = 0.6,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    tight_axes: bool = True,
    axis_pad: float = 0.08,
    font_size: int = 10,
    do_label: bool = True,
    show_legend: bool = True,
    return_fig: bool = False
):
    """
    Scatter plot of pathway-level differential signaling for a specific cell group.
    Mirrors R net_analysis_signaling_changes_scatter().
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    if comparison is None:
        comparison = [0, 1]

    if slot_name == "pathway_network":
        all_net = cellchat.pathway_network
    else:
        all_net = cellchat.network

    dataset_keys = _merged_dataset_keys(all_net)

    if len(dataset_keys) < 2:
        raise ValueError("Need merged CellChat object with at least 2 datasets.")

    if len(comparison) != 2 or any(index < 0 or index >= len(dataset_keys) for index in comparison):
        raise ValueError("comparison must contain two zero-based dataset indices.")
    name1 = dataset_keys[comparison[0]]
    name2 = dataset_keys[comparison[1]]
    print(f"Visualizing signaling changes for {group_use} from {name1} to {name2}")

    if title is None:
        title = f"Signaling changes of {group_use} ({name1} vs. {name2})"

    groups = cellchat.groups
    if groups is None or not hasattr(groups, 'categories'):
        raise ValueError("Merged object must define categorical cellchat.groups.")
    cell_levels = list(groups.categories)

    if group_use not in cell_levels:
        raise ValueError(f"Cell group '{group_use}' not found.")

    if xlabel is None:
        xlabel = "Differential outgoing interaction strength"
    if ylabel is None:
        ylabel = "Differential incoming interaction strength"

    if signaling is None:
        sig1 = network_names(all_net[dataset_keys[comparison[0]]])
        sig2 = network_names(all_net[dataset_keys[comparison[1]]])
        signaling = list(set(sig1) | set(sig2))
    if signaling_exclude:
        signaling = [s for s in signaling if s not in signaling_exclude]

    mat_all_merged = []
    for ii in range(len(comparison)):
        net_data = all_net[dataset_keys[comparison[ii]]]
        centrality = net_data.get('centrality', {})
        if not centrality:
            raise ValueError("Run compute_network_centrality first.")

        outgoing = np.zeros((len(cell_levels), len(centrality)))
        incoming = np.zeros((len(cell_levels), len(centrality)))
        for i, (pname, cvals) in enumerate(centrality.items()):
            outgoing[:, i] = cvals.get(x_measure, np.zeros(len(cell_levels)))
            incoming[:, i] = cvals.get(y_measure, np.zeros(len(cell_levels)))

        mat_all = np.zeros((len(signaling), len(cell_levels), 2))
        for pi, sig_path in enumerate(signaling):
            if sig_path in centrality:
                pii = list(centrality.keys()).index(sig_path)
                if pii < outgoing.shape[1]:
                    mat_all[pi, :, 0] = outgoing[:, pii]
                    mat_all[pi, :, 1] = incoming[:, pii]
        mat_all_merged.append(mat_all)

    ci = cell_levels.index(group_use)
    mat_use = [m[:, ci, :] for m in mat_all_merged]

    idx_specific = mat_use[0] * mat_use[1]
    mat_sum = mat_use[1] + mat_use[0]
    out_specific = np.where((mat_sum[:, 0] != 0) & (idx_specific[:, 0] == 0))[0]
    in_specific = np.where((mat_sum[:, 1] != 0) & (idx_specific[:, 1] == 0))[0]

    mat_diff = mat_use[1] - mat_use[0]
    keep = np.abs(mat_diff).sum(axis=1) != 0
    mat_diff = mat_diff[keep]
    pathways_keep = [signaling[i] for i, k in enumerate(keep) if k]

    out_spec_mask = np.isin(pathways_keep, [signaling[i] for i in out_specific])
    in_spec_mask = np.isin(pathways_keep, [signaling[i] for i in in_specific])
    both_spec = out_spec_mask & in_spec_mask

    specificity_out_in = np.zeros(len(pathways_keep), dtype=int)
    specificity_out_in[both_spec] = 2
    specificity_out_in[out_spec_mask & ~both_spec] = 1
    specificity_out_in[in_spec_mask & ~both_spec] = -1

    specificity = np.zeros(len(pathways_keep), dtype=int)
    specificity[(specificity_out_in != 0) & (mat_diff.min(axis=1) >= 0)] = 1
    specificity[(specificity_out_in != 0) & (mat_diff.max(axis=1) <= 0)] = -1

    cat_out_in = {-1: "Incoming specific", 0: "Shared", 1: "Outgoing specific",
                   2: "Incoming & Outgoing specific"}
    cat_spec = {-1: f"{name1} specific", 0: "Shared", 1: f"{name2} specific"}

    df = pd.DataFrame({
        'outgoing': mat_diff[:, 0],
        'incoming': mat_diff[:, 1],
        'label': pathways_keep,
        'specificity_out_in': [cat_out_in.get(v, "Shared") for v in specificity_out_in],
        'specificity': [cat_spec.get(v, "Shared") for v in specificity]
    })

    if color_use is None:
        color_use = ["#1a1a1a", "#F8766D", "#00BFC4"]

    uniq_out_in = sorted(set(df['specificity_out_in']))
    uniq_spec = sorted(set(df['specificity']))
    shapes_map = {'o': 0, 's': 1, 'D': 2, '^': 3}
    shapes = ['o', 's', 'D', '^']
    color_map = dict(zip(uniq_spec, color_use[:len(uniq_spec)]))

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in df.iterrows():
        c = color_map.get(row['specificity'], '#999999')
        sh = shapes[uniq_out_in.index(row['specificity_out_in'])] if row['specificity_out_in'] in uniq_out_in else 'o'
        ax.scatter(row['outgoing'], row['incoming'], s=dot_size * 30,
                   c=[to_rgba(c, dot_alpha)], edgecolors=c,
                   linewidths=0.8, marker=sh)

    if do_label:
        if signaling_label:
            label_df = df[df['label'].isin(signaling_label)]
        else:
            thresh_val = np.quantile(np.abs(df[['outgoing', 'incoming']].values), 1.0 - top_label)
            label_df = df[(np.abs(df['outgoing']) > thresh_val) |
                         (np.abs(df['incoming']) > thresh_val)]
        for _, row in label_df.iterrows():
            ax.annotate(row['label'], (row['outgoing'], row['incoming']),
                       fontsize=label_size * 2.5, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, linestyle='dashed', color='grey', linewidth=0.5)
    ax.axvline(0, linestyle='dashed', color='grey', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size, fontweight='normal')
    ax.spines[['top', 'right']].set_visible(False)

    def _axis_limits(values, explicit_limits):
        if explicit_limits is not None:
            return explicit_limits
        if not tight_axes:
            return None
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo = min(float(vals.min()), 0.0)
        hi = max(float(vals.max()), 0.0)
        span = hi - lo
        if span < 1e-12:
            pad = max(abs(hi), 1.0) * axis_pad
        else:
            pad = span * axis_pad
        return lo - pad, hi + pad

    xlim_use = _axis_limits(df['outgoing'].values, xlims)
    ylim_use = _axis_limits(df['incoming'].values, ylims)
    if xlim_use is not None:
        ax.set_xlim(xlim_use)
    if ylim_use is not None:
        ax.set_ylim(ylim_use)

    if show_legend:
        handles = [mpatches.Patch(color=color_map[s], label=s) for s in uniq_spec]
        ax.legend(handles=handles, fontsize=7, frameon=False,
                 bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# colorRamp3  (mirrors R colorRamp3 from utilities.R)
# ---------------------------------------------------------------------------
def color_ramp(
    values: np.ndarray,
    colors: List[str],
    na_color: str = "grey"
) -> np.ndarray:
    """
    Map numeric values to interpolated colors (mirrors R colorRamp3).
    Returns array of hex color strings.
    """
    values = np.asarray(values, dtype=float)
    n_colors = len(colors)
    if n_colors < 2:
        return np.array([colors[0]] * len(values))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("cr3", colors, N=256)

    mask = np.isnan(values)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin < 1e-15:
        normed = np.full(values.shape, 0.5)
    else:
        normed = (values - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0, 1)

    from matplotlib.colors import to_hex
    result = np.array([to_hex(cmap(v)) for v in normed])
    result[mask] = na_color
    return result


# ---------------------------------------------------------------------------
# Spatial statistics and spot communication topics
# ---------------------------------------------------------------------------
def _spatial_coordinates_array(cellchat) -> np.ndarray:
    coordinates = np.asarray(cellchat.obsm.get("spatial"), dtype=float)
    if coordinates.shape != (cellchat.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("cellchat.obsm['spatial'] must be a finite n_spots x 2 matrix.")
    return coordinates


def _knn_weights(coordinates: np.ndarray, n_neighbors: int, include_self: bool) -> sparse.csr_matrix:
    n_spots = len(coordinates)
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if n_spots < 2:
        raise ValueError("Spatial statistics require at least two spots.")
    if n_neighbors >= n_spots:
        raise ValueError("n_neighbors must be smaller than the number of spots.")
    neighbours = cKDTree(coordinates).query(coordinates, k=n_neighbors + 1)[1][:, 1:]
    rows = np.repeat(np.arange(n_spots), n_neighbors)
    weights = sparse.csr_matrix((np.ones(len(rows)), (rows, neighbours.reshape(-1))), shape=(n_spots, n_spots))
    weights.setdiag(1.0 if include_self else 0.0)
    weights.eliminate_zeros()
    return weights


def _delaunay_weights(coordinates: np.ndarray) -> sparse.csr_matrix:
    coordinates = np.asarray(coordinates, dtype=float)
    n_spots = len(coordinates)
    if n_spots < 3:
        raise ValueError("Delaunay spatial statistics require at least three spots.")
    adjusted, seen = coordinates.copy(), {}
    for index, point in enumerate(adjusted):
        key = (float(point[0]), float(point[1]))
        occurrence = seen.get(key, 0)
        if occurrence:
            adjusted[index] += 1e-6 * occurrence
        seen[key] = occurrence + 1
    try:
        triangulation = Delaunay(adjusted)
    except QhullError as error:
        raise ValueError("Spatial coordinates must contain at least three non-collinear points for the MERINGUE Delaunay graph.") from error
    edges = set()
    for simplex in triangulation.simplices:
        for first, second in ((simplex[0], simplex[1]), (simplex[0], simplex[2]), (simplex[1], simplex[2])):
            first, second = int(first), int(second)
            if first != second:
                edges.add((min(first, second), max(first, second)))
    if not edges:
        raise ValueError("The Delaunay spatial graph contains no links.")
    edge_array = np.asarray(sorted(edges), dtype=int)
    rows, cols = np.concatenate([edge_array[:, 0], edge_array[:, 1]]), np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_spots, n_spots))


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    values, result = np.asarray(pvalues, dtype=float), np.ones_like(pvalues, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    selected = values[finite]
    order = np.argsort(selected)
    adjusted = selected[order] * len(selected) / np.arange(1, len(selected) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def identify_spatially_variable_genes(
    cellchat, n_neighbors: int = 12, thresh_p: float = 0.05,
    features_name: str = "features", min_spots: int = 10,
):
    """Select spatially variable signaling genes with MERINGUE Moran's I."""
    if not isinstance(min_spots, (int, np.integer)) or min_spots < 2:
        raise ValueError("min_spots must be an integer of at least 2.")
    if not np.isfinite(thresh_p) or not 0 <= thresh_p <= 1:
        raise ValueError("thresh_p must be between 0 and 1.")
    coordinates, expression = _spatial_coordinates_array(cellchat), cellchat.signaling
    if expression is None:
        raise ValueError("Run subset_signaling_data before spatial feature selection.")
    mask = cellchat.var["is_signaling"].fillna(False).to_numpy(dtype=bool) if "is_signaling" in cellchat.var else np.ones(cellchat.n_vars, dtype=bool)
    genes, values = cellchat.var_names.astype(str)[mask], expression[:, mask]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Signaling expression must contain finite non-negative values.")
    detected = np.count_nonzero(values > 0, axis=0)
    keep = (detected >= min_spots) & (np.ptp(values, axis=0) > 0)
    values, genes, detected = values[:, keep], genes[keep], detected[keep]
    if not len(genes):
        raise ValueError("No signaling genes pass min_spots and non-constant filtering.")
    if len(coordinates) < 4:
        raise ValueError("MERINGUE Moran's I requires at least four spots.")
    weights = _delaunay_weights(coordinates)
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    weights = (sparse.diags(np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)) @ weights).tocsr()
    s0 = float(weights.sum())
    if s0 == 0:
        raise ValueError("The spatial neighbour graph contains no links.")
    centered, denominator = values - values.mean(axis=0, keepdims=True), np.square(values - values.mean(axis=0, keepdims=True)).sum(axis=0)
    observed = len(coordinates) * np.sum(centered * (weights @ centered), axis=0) / (s0 * denominator)
    n_spots, expectation = len(coordinates), -1.0 / (len(coordinates) - 1)
    symmetric_sum = weights + weights.T
    s1 = 0.5 * float(symmetric_sum.multiply(symmetric_sum).sum())
    row_col_sum = np.asarray(weights.sum(axis=1)).ravel() + np.asarray(weights.sum(axis=0)).ravel()
    s2 = float(np.square(row_col_sum).sum())
    s3 = np.sum(centered**4, axis=0) / n_spots
    s3 = np.divide(s3, (denominator / n_spots) ** 2, out=np.full_like(s3, np.nan), where=denominator > 0)
    s4 = (n_spots**2 - 3 * n_spots + 3) * s1 - n_spots * s2 + 3 * s0**2
    s5 = (n_spots**2 - n_spots) * s1 - 2 * n_spots * s2 + 6 * s0**2
    expected_square = (n_spots * s4 - s3 * s5) / ((n_spots - 1) * (n_spots - 2) * (n_spots - 3) * s0**2)
    variance = np.where(expected_square - expectation**2 >= 0, expected_square - expectation**2, np.nan)
    z_score = np.divide(observed - expectation, np.sqrt(variance), out=np.full_like(observed, np.nan), where=np.isfinite(variance) & (variance > 0))
    pvalue = stats.norm.sf(z_score)
    result = pd.DataFrame({"features": genes, "observed": observed, "expected": expectation, "z_score": z_score, "pvalues": pvalue, "pvalues_adj": _benjamini_hochberg(pvalue), "n_spots": detected}).sort_values(["pvalues_adj", "observed"], ascending=[True, False], ignore_index=True)
    cellchat.feature_results[features_name] = result.loc[(result["pvalues_adj"] < thresh_p) & np.isfinite(result["observed"]), "features"].tolist()
    cellchat.feature_results[f"{features_name}_info"] = result
    cellchat.settings["spatial_variable_features"] = {"method": "moran_meringue", "neighbor_graph": "delaunay", "thresh_p": float(thresh_p), "min_spots": int(min_spots)}
    return cellchat


def get_spot_signaling_scores(cellchat, signaling: str | Sequence[str] | None = None, measure: str = "indeg", slot_name: str = "spot_pathway_network", binary: bool = False) -> pd.Series:
    """Return one score per spot by summing selected centrality columns."""
    if slot_name not in {"spot_network", "spot_pathway_network"}:
        raise ValueError("slot_name must be 'spot_network' or 'spot_pathway_network'.")
    if measure not in {"indeg", "outdeg", "indeg_unweighted", "outdeg_unweighted"}:
        raise ValueError("Unsupported spot centrality measure.")
    network = getattr(cellchat, slot_name)
    centrality = network.get("centrality", {}) if network else {}
    if measure not in centrality:
        raise ValueError(f"Run compute_spot_network_centrality for {slot_name} first.")
    table = centrality[measure]
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table, index=network["spots"], columns=list(network["prob"]))
    selected = table.columns.tolist() if signaling is None else ([signaling] if isinstance(signaling, str) else [str(value) for value in signaling])
    missing = pd.Index(selected).difference(table.columns)
    if len(missing):
        raise ValueError(f"Unknown signaling names in {slot_name}: {missing.tolist()}")
    values = table.loc[:, selected].to_numpy(dtype=float)
    return pd.Series(np.asarray((values > 0 if binary else values).sum(axis=1), dtype=float), index=table.index.astype(str), name=measure)


def compute_getis_ord_gi(values, coordinates, n_neighbors: int = 12, gstar: bool = False) -> pd.DataFrame:
    """Compute local Getis-Ord Gi or Gi* z scores with binary KNN weights."""
    values, coordinates = np.asarray(values, dtype=float).reshape(-1), np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a finite n_spots x 2 matrix aligned with values.")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite.")
    weights = _knn_weights(coordinates, n_neighbors, include_self=gstar)
    weight_sum, weight_square_sum, weighted_sum, n_spots = np.asarray(weights.sum(axis=1)).ravel(), np.asarray(weights.multiply(weights).sum(axis=1)).ravel(), np.asarray(weights @ values).ravel(), len(values)
    if gstar:
        mean, variance, sample_n = np.repeat(values.mean(), n_spots), np.repeat(np.mean(np.square(values)) - values.mean() ** 2, n_spots), np.repeat(n_spots, n_spots)
    else:
        sample_n = np.repeat(n_spots - 1, n_spots)
        mean = (values.sum() - values) / sample_n
        variance = (np.square(values).sum() - np.square(values)) / sample_n - np.square(mean)
    spatial_term = (sample_n * weight_square_sum - np.square(weight_sum)) / np.maximum(sample_n - 1, 1)
    statistic = np.divide(weighted_sum - mean * weight_sum, np.sqrt(np.maximum(variance, 0) * np.maximum(spatial_term, 0)), out=np.zeros_like(weighted_sum), where=np.maximum(variance, 0) * np.maximum(spatial_term, 0) > 0)
    return pd.DataFrame({"score": values, "gi": statistic, "pvalue": 2.0 * stats.norm.sf(np.abs(statistic))})


def compute_spatial_gi(cellchat, signaling: str | Sequence[str] | None = None, measure: str = "indeg", slot_name: str = "spot_pathway_network", binary: bool = True, n_neighbors: int = 12, gstar: bool = False, result_name: str | None = None):
    """Compute and store Gi/Gi* statistics for spot communication scores."""
    score = get_spot_signaling_scores(cellchat, signaling, measure, slot_name, binary)
    result = compute_getis_ord_gi(score.to_numpy(), _spatial_coordinates_array(cellchat), n_neighbors, gstar)
    result.index = score.index
    result.insert(0, "spot", result.index)
    if cellchat.groups is not None:
        result["cellchat_group"] = np.asarray(cellchat.groups).astype(str)
    if result_name is None:
        selected = "all" if signaling is None else (signaling if isinstance(signaling, str) else "+".join(map(str, signaling)))
        result_name = f"{slot_name}:{selected}:{measure}:{'gstar' if gstar else 'gi'}"
    statistics, gi_results = dict(cellchat.spatial_statistics), dict(cellchat.spatial_statistics.get("gi", {}))
    gi_results[str(result_name)] = result
    statistics["gi"] = gi_results
    cellchat.spatial_statistics = statistics
    return cellchat


def _spatial_variable_matrix(cellchat, value, name: str) -> pd.DataFrame:
    if isinstance(value, str):
        if value not in cellchat.obs:
            raise ValueError(f"{name}={value!r} is not a column in cellchat.obs.")
        value = cellchat.obs[value]
    if isinstance(value, pd.Series):
        series = value.reindex(cellchat.obs_names)
        if series.isna().any():
            raise ValueError(f"{name} is not aligned with cellchat.obs_names.")
        if pd.api.types.is_numeric_dtype(series):
            frame = series.astype(float).to_frame(series.name or name)
        else:
            levels = list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype) else list(pd.unique(series.astype(str)))
            frame = pd.get_dummies(pd.Categorical(series.astype(str), categories=levels), dtype=float)
            frame.index, frame.columns = cellchat.obs_names, [str(level) for level in levels]
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.index = frame.index.astype(str)
        if not frame.index.equals(pd.Index(cellchat.obs_names.astype(str))):
            raise ValueError(f"{name} rows must match cellchat.obs_names in the same order.")
    else:
        array = np.asarray(value)
        if array.ndim == 1:
            if len(array) != cellchat.n_obs:
                raise ValueError(f"{name} must have one value per spot.")
            if np.issubdtype(array.dtype, np.number):
                frame = pd.DataFrame({name: array.astype(float)}, index=cellchat.obs_names)
            else:
                return _spatial_variable_matrix(cellchat, pd.Series(array, index=cellchat.obs_names, name=name), name)
        elif array.ndim == 2 and array.shape[0] == cellchat.n_obs:
            frame = pd.DataFrame(array, index=cellchat.obs_names, columns=[f"{name}_{index + 1}" for index in range(array.shape[1])])
        else:
            raise ValueError(f"{name} must be a spot vector or spot-by-feature matrix.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite values.")
    return frame.astype(float)


def _lee_statistic(x: np.ndarray, y: np.ndarray, weights: sparse.csr_matrix) -> float:
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.square(x_centered).sum() * np.square(y_centered).sum())
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    weight_scale = np.square(row_sums).sum()
    if denominator == 0 or weight_scale == 0:
        return np.nan
    return float(len(x) * np.dot(np.asarray(weights @ x_centered).ravel(), np.asarray(weights @ y_centered).ravel()) / (weight_scale * denominator))


def compute_spatial_lee(cellchat, x, y=None, weight_type: str = "identity", interaction_range: float = 250.0, contact_range: float = 10.0, result_name: str | None = None):
    """Compute and store bivariate Lee statistics between spatial features."""
    if y is None:
        if cellchat.groups is None:
            raise ValueError("y is required when cellchat.groups is unavailable.")
        y = pd.Series(np.asarray(cellchat.groups), index=cellchat.obs_names, name="cellchat_group")
    x_frame, y_frame = _spatial_variable_matrix(cellchat, x, "x"), _spatial_variable_matrix(cellchat, y, "y")
    if weight_type == "identity":
        weights = sparse.eye(cellchat.n_obs, format="csr")
    elif weight_type in {"contact", "interaction"}:
        from .modeling import compute_spot_distances
        distance, contact = compute_spot_distances(cellchat, interaction_range, contact_range)
        if weight_type == "contact":
            weights = contact.astype(float)
        else:
            weights = distance.copy().astype(float)
            weights.data = 1.0 / weights.data
            weights.setdiag(1.0)
            row_sum = np.asarray(weights.sum(axis=1)).ravel()
            weights = (sparse.diags(np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)) @ weights).tocsr()
    else:
        raise ValueError("weight_type must be 'identity', 'contact', or 'interaction'.")
    values = np.empty((y_frame.shape[1], x_frame.shape[1]), dtype=float)
    for row in range(y_frame.shape[1]):
        for column in range(x_frame.shape[1]):
            values[row, column] = _lee_statistic(x_frame.iloc[:, column].to_numpy(), y_frame.iloc[:, row].to_numpy(), weights)
    result = pd.DataFrame(values, index=y_frame.columns.astype(str), columns=x_frame.columns.astype(str))
    if result_name is None:
        result_name = f"{weight_type}:{'+'.join(result.columns)}:{'+'.join(result.index)}"
    statistics, lee_results = dict(cellchat.spatial_statistics), dict(cellchat.spatial_statistics.get("lee", {}))
    lee_results[str(result_name)] = result
    statistics["lee"] = lee_results
    cellchat.spatial_statistics = statistics
    return cellchat


def identify_cell_topics(cellchat, n_topics: int, pattern: str = "incoming", slot_name: str = "spot_network", scale: bool = False, seed_use: int = 666, tol: float = 1e-6, max_iter: int = 600):
    """Factor spot communication scores into spot and signaling topics."""
    if pattern not in {"incoming", "outgoing"}:
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if slot_name not in {"spot_network", "spot_pathway_network"}:
        raise ValueError("slot_name must be 'spot_network' or 'spot_pathway_network'.")
    if not isinstance(n_topics, (int, np.integer)) or n_topics < 1:
        raise ValueError("n_topics must be a positive integer.")
    network, measure = getattr(cellchat, slot_name), ("indeg" if pattern == "incoming" else "outdeg")
    centrality = network.get("centrality", {}) if network else {}
    if measure not in centrality:
        raise ValueError(f"Run compute_spot_network_centrality for {slot_name} first.")
    scores = centrality[measure]
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(scores, index=network["spots"], columns=list(network["prob"]))
    data = scores.to_numpy(dtype=float)
    active_signaling, active_spots = np.any(data > 0, axis=0), np.any(data > 0, axis=1)
    if np.count_nonzero(active_signaling) < n_topics or np.count_nonzero(active_spots) < n_topics:
        raise ValueError("n_topics exceeds the number of active spots or signaling features.")
    fitted = data[:, active_signaling]
    if scale:
        standard_deviation = fitted.std(axis=0, ddof=0)
        fitted = np.divide(fitted, standard_deviation, out=np.zeros_like(fitted), where=standard_deviation > 0)
    model = NMF(n_components=int(n_topics), init="nndsvda", solver="cd", tol=float(tol), max_iter=int(max_iter), random_state=int(seed_use))
    cell_active, signaling_active = model.fit_transform(fitted), model.components_.T
    topic_names = [f"Topic_{index + 1}" for index in range(n_topics)]
    cell_values = np.zeros((len(scores), n_topics), dtype=float)
    cell_values[:, :] = cell_active
    signaling_values = np.zeros((scores.shape[1], n_topics), dtype=float)
    signaling_values[active_signaling] = signaling_active
    cell_frame = pd.DataFrame(cell_values, index=scores.index.astype(str), columns=topic_names)
    signaling_frame = pd.DataFrame(signaling_values, index=scores.columns.astype(str), columns=topic_names)
    topics, slot_topics = dict(cellchat.cell_topics), dict(cellchat.cell_topics.get(slot_name, {}))
    slot_topics[pattern] = {
        "cell": cell_frame, "signaling": signaling_frame,
        "assignment": pd.Series(cell_frame.idxmax(axis=1), index=cell_frame.index, name=f"topic_{pattern}"),
        "reconstruction_error": float(model.reconstruction_err_),
        "parameters": {"n_topics": int(n_topics), "scale": bool(scale), "seed_use": int(seed_use), "tol": float(tol), "max_iter": int(max_iter)},
    }
    slot_topics[pattern]["cell"].index.name, slot_topics[pattern]["signaling"].index.name = "spot", "signaling"
    topics[slot_name] = slot_topics
    cellchat.cell_topics = topics
    return cellchat


