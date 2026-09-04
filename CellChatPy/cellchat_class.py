#!/usr/bin/env python3
"""
CellChat Class - Main class for cell-cell communication analysis
This class implements the CellChat R analysis workflow in Python.

Refactored to inherit from AnnData for native integration with scanpy ecosystem.
"""
from .network_storage import (
    expand_dense_group_axes,
    expand_group_axes,
    is_matrix_dict,
    normalize_network_similarity,
    normalize_network_slot,
    normalize_spot_network,
    reorder_group_axes,
)
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Optional, Dict, List, Any, Tuple
import warnings
import logging
import os

# AnnData is now a REQUIRED dependency
from anndata import AnnData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# CellChat class -inherits from AnnData
# ===========================================================================
class CellChat(AnnData):
    """
    The CellChat object for analyzing intercellular communication from
    single-cell transcriptomics data.

    Inherits from AnnData for full compatibility with the scanpy/squidpy
    single-cell Python ecosystem. Expression matrices always use the AnnData
    cells-by-genes orientation. CellChat-specific results are stored in named
    ``uns`` entries and accessed through snake_case properties.

    Parameters
    ----------
    adata : AnnData, optional
        An existing AnnData object to initialize from.
        If None, an empty AnnData is created.
    **kwargs
        Additional arguments passed to AnnData constructor.
    """

    def __init__(self, adata: Optional[AnnData] = None, **kwargs):
        if adata is not None:
            # Initialize from existing AnnData
            super().__init__(
                X=adata.X,
                obs=adata.obs.copy() if adata.obs is not None else None,
                var=adata.var.copy() if adata.var is not None else None,
                uns=adata.uns.copy() if adata.uns else {},
                obsm=adata.obsm.copy() if adata.obsm else {},
                varm=adata.varm.copy() if hasattr(adata, 'varm') and adata.varm else {},
                layers=adata.layers.copy() if adata.layers else {},
                obsp=adata.obsp.copy() if hasattr(adata, 'obsp') and adata.obsp else {},
                dtype=adata.X.dtype if adata.X is not None else np.float32
            )
        else:
            # Create new empty object
            super().__init__(**kwargs)

        # Initialize CellChat-specific uns keys
        self._init_cellchat_keys()

    def _init_cellchat_keys(self):
        """Initialize all CellChat-required uns dictionary keys."""
        defaults = {
            'cellchat_network': {},
            'cellchat_pathway_network': {},
            'cellchat_network_similarity': {},
            'cellchat_lr_pairs': {},
            'cellchat_database': {},
            'cellchat_feature_results': {},
            'cellchat_settings': {'mode': 'single', 'datatype': 'RNA'},
            'cellchat_spatial': {},
            'cellchat_spot_network': {},
            'cellchat_spot_pathway_network': {},
            'cellchat_spatial_statistics': {},
            'cellchat_cell_topics': {},
        }
        for key, default in defaults.items():
            self.uns.setdefault(key, default)
        self.uns['cellchat_network'] = normalize_network_slot(
            self.uns['cellchat_network'], 'network'
        )
        self.uns['cellchat_pathway_network'] = normalize_network_slot(
            self.uns['cellchat_pathway_network'], 'pathway_network'
        )
        self.uns['cellchat_network_similarity'] = normalize_network_similarity(
            self.uns['cellchat_network_similarity']
        )
        self.uns['cellchat_spot_network'] = normalize_spot_network(
            self.uns['cellchat_spot_network'], 'spot_network'
        )
        self.uns['cellchat_spot_pathway_network'] = normalize_spot_network(
            self.uns['cellchat_spot_pathway_network'], 'spot_pathway_network'
        )

    def _validate_layer(self, value, name: str):
        """Validate a cells-by-genes layer without inferring an orientation."""
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(self.obs_names) or not value.columns.equals(self.var_names):
                raise ValueError(
                    f"{name} DataFrame must use obs_names as its index and var_names as its columns."
                )
            value = value.to_numpy()
        if not hasattr(value, 'shape') or tuple(value.shape) != self.shape:
            raise ValueError(f"{name} must have cells x genes shape {self.shape}; got {getattr(value, 'shape', None)}.")
        return value.tocsr() if sparse.issparse(value) else np.asarray(value)

    def _get_layer(self, name: str):
        return self.layers.get(name)

    def _set_layer(self, name: str, value) -> None:
        if value is None:
            self.layers.pop(name, None)
        else:
            self.layers[name] = self._validate_layer(value, name)

    @property
    def raw(self):
        """Optional raw expression layer with cells x genes orientation."""
        return self._get_layer('raw')

    @raw.setter
    def raw(self, value):
        self._set_layer('raw', value)

    @property
    def signaling(self):
        """Optional signaling-expression layer with cells x genes orientation."""
        return self._get_layer('signaling')

    @signaling.setter
    def signaling(self, value):
        self._set_layer('signaling', value)

    @property
    def scaled(self):
        return self._get_layer('scale')

    @scaled.setter
    def scaled(self, value):
        self._set_layer('scale', value)

    @property
    def smoothed(self):
        return self._get_layer('smooth')

    @smoothed.setter
    def smoothed(self, value):
        self._set_layer('smooth', value)

    @property
    def groups(self):
        """Categorical cell identities stored in ``obs['cellchat_group']``."""
        if 'cellchat_group' not in self.obs:
            return None
        return pd.Categorical(self.obs['cellchat_group'])

    @groups.setter
    def groups(self, value) -> None:
        if value is None:
            self.obs.pop('cellchat_group', None)
            return
        if len(value) != self.n_obs:
            raise ValueError(f"groups must contain one value for each of the {self.n_obs} observations.")
        self.obs['cellchat_group'] = pd.Categorical(value)

    @property
    def network(self) -> dict:
        return self.uns['cellchat_network']

    @network.setter
    def network(self, value: dict) -> None:
        self.uns['cellchat_network'] = normalize_network_slot(value, 'network')

    @property
    def pathway_network(self) -> dict:
        return self.uns['cellchat_pathway_network']

    @pathway_network.setter
    def pathway_network(self, value: dict) -> None:
        self.uns['cellchat_pathway_network'] = normalize_network_slot(value, 'pathway_network')

    @property
    def network_similarity(self) -> dict:
        return self.uns['cellchat_network_similarity']

    @network_similarity.setter
    def network_similarity(self, value: dict) -> None:
        self.uns['cellchat_network_similarity'] = normalize_network_similarity(value)

    def _get_uns_mapping(self, key: str) -> dict:
        value = self.uns[key]
        if not isinstance(value, dict):
            raise TypeError(f"uns[{key!r}] must be a dictionary.")
        return value

    def _set_uns_mapping(self, key: str, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"{key.removeprefix('cellchat_')} must be a dictionary.")
        self.uns[key] = value

    @property
    def lr_pairs(self) -> dict:
        return self._get_uns_mapping('cellchat_lr_pairs')

    @lr_pairs.setter
    def lr_pairs(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_lr_pairs', value)

    @property
    def database(self) -> dict:
        return self._get_uns_mapping('cellchat_database')

    @database.setter
    def database(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_database', value)

    @property
    def feature_results(self) -> dict:
        return self._get_uns_mapping('cellchat_feature_results')

    @feature_results.setter
    def feature_results(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_feature_results', value)

    @property
    def settings(self) -> dict:
        return self._get_uns_mapping('cellchat_settings')

    @settings.setter
    def settings(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_settings', value)

    @property
    def spatial(self) -> dict:
        return self._get_uns_mapping('cellchat_spatial')

    @spatial.setter
    def spatial(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_spatial', value)

    @property
    def spot_network(self) -> dict:
        return self.uns['cellchat_spot_network']

    @spot_network.setter
    def spot_network(self, value: dict) -> None:
        self.uns['cellchat_spot_network'] = normalize_spot_network(value, 'spot_network')

    @property
    def spot_pathway_network(self) -> dict:
        return self.uns['cellchat_spot_pathway_network']

    @spot_pathway_network.setter
    def spot_pathway_network(self, value: dict) -> None:
        self.uns['cellchat_spot_pathway_network'] = normalize_spot_network(
            value, 'spot_pathway_network'
        )

    @property
    def spatial_statistics(self) -> dict:
        return self._get_uns_mapping('cellchat_spatial_statistics')

    @spatial_statistics.setter
    def spatial_statistics(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_spatial_statistics', value)

    @property
    def cell_topics(self) -> dict:
        return self._get_uns_mapping('cellchat_cell_topics')

    @cell_topics.setter
    def cell_topics(self, value: dict) -> None:
        self._set_uns_mapping('cellchat_cell_topics', value)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """String representation of the CellChat object."""
        mode = self.settings.get('mode', 'single')
        if mode == 'single':
            output = f"An object of class {self.__class__.__name__} created from a single dataset\n"
            # Use AnnData shape directly (cellsxgenes)
            if self.X is not None:
                n_cells, n_genes = self.X.shape
                output += f"{n_genes} genes, {n_cells} cells\n"
        else:
            output = f"An object of class {self.__class__.__name__} created from a merged object with multiple datasets\n"
            if self.X is not None:
                n_cells, n_genes = self.X.shape
                output += f"{n_genes} genes, {n_cells} cells\n"

        datatype = self.settings.get('datatype', 'RNA')
        if datatype == 'RNA':
            output += "CellChat analysis of single cell RNA-seq data!\n"
        else:
            output += f"CellChat analysis of {datatype} data!\n"
            imgs = self.spatial
            if 'coordinates' in imgs:
                output += "The input spatial locations are:\n"
                coords = imgs['coordinates']
                if hasattr(coords, 'head'):
                    output += str(coords.head())
                else:
                    output += str(coords[:5])

        return output

    def __repr__(self) -> str:
        return self.__str__()

    def show(self) -> None:
        """Display CellChat object information."""
        print(self.__str__())

    # ------------------------------------------------------------------
    # Copy support
    # ------------------------------------------------------------------
    def copy(self, filename=None):
        """Return an independent copy that preserves the CellChat type.

        AnnData performs the full copy, including CellChat data stored in
        ``uns``. Wrapping that result restores the CellChat API instead of
        returning a plain AnnData object.
        """
        copied = super().copy(filename=filename)
        return CellChat(copied)


# ===========================================================================
# Factory functions
# ===========================================================================

def create_cellchat(
    object: Union[np.ndarray, sparse.spmatrix, pd.DataFrame, AnnData],
    metadata: Optional[pd.DataFrame] = None,
    group_by: Optional[str] = None,
    datatype: str = "RNA",
    coordinates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    spatial_factors: Optional[Dict[str, float]] = None,
    assay: Optional[str] = None,
    do_sparse: bool = True,
    genes: Optional[list] = None,
    cells: Optional[list] = None
) -> CellChat:
    """
    Create a new CellChat object from various input types.

    Parameters
    ----------
    object : array-like or AnnData
        Input data -can be a normalized data matrix (cells x genes),
        AnnData object, or other single-cell data container.
    metadata : pd.DataFrame, optional
        Cell metadata with cells as rows. If None and input is AnnData,
        uses ``adata.obs``.
    group_by : str, optional
        Column name in metadata defining cell groups.
    datatype : str
        ``"RNA"`` for scRNA-seq or ``"spatial"`` for spatial transcriptomics.
    coordinates : array-like, optional
        Spatial coordinates (required for spatial data).
    spatial_factors : dict, optional
        Dictionary with ``'ratio'`` and ``'tol'`` for spatial data.
    assay : str, optional
        Assay name for AnnData objects (default: use ``adata.X``).
    do_sparse : bool
        Whether to convert data to sparse matrix format.
    genes : list, optional
        Gene names when input is scipy sparse matrix or numpy array without
        column names. If not provided, defaults to ``gene_0, gene_1, ...``.
    cells : list, optional
        Cell names when input is scipy sparse matrix or numpy array without
        row names. If not provided, defaults to ``cell_0, cell_1, ...``.

    Returns
    -------
    CellChat
        A new CellChat object (inherits from AnnData).
    """
    datatype = datatype.lower()
    if datatype not in ["rna", "spatial"]:
        raise ValueError("datatype must be 'RNA' or 'spatial'")

    gene_names = None
    cell_names = None
    data_matrix = None

    # ------------------------------------------------------------------
    # 1. Extract data from input
    # ------------------------------------------------------------------
    if isinstance(object, AnnData):
        print("Create a CellChat object from an AnnData object")
        adata_input = object

        # Extract data matrix (cells x genes)
        if assay is None:
            data_cxg = adata_input.X  # cells x genes (AnnData native)
        else:
            if assay in adata_input.layers:
                data_cxg = adata_input.layers[assay]
            else:
                raise ValueError(f"Assay '{assay}' not found in AnnData object")

        gene_names = adata_input.var_names.tolist() if len(adata_input.var_names) > 0 else None
        cell_names = adata_input.obs_names.tolist() if len(adata_input.obs_names) > 0 else None

        # Extract metadata
        if metadata is None:
            print("Using AnnData.obs as cell metadata")
            metadata = adata_input.obs.copy()

        # Extract coordinates if spatial
        if datatype == "spatial" and coordinates is None:
            if 'spatial' in adata_input.obsm:
                coordinates = adata_input.obsm['spatial']
                print("Using adata.obsm['spatial'] for coordinates")

    elif isinstance(object, (pd.DataFrame, np.ndarray, sparse.spmatrix)):
        print("Create a CellChat object from a data matrix")
        # Input is cells x genes, matching AnnData directly.
        if isinstance(object, pd.DataFrame):
            # Check for sparse DataFrame and convert to dense to avoid NaN bug
            if hasattr(object, 'sparse') and hasattr(object.dtypes, 'apply'):
                if object.dtypes.apply(lambda x: hasattr(x, 'subtype')).any():
                    print("  Warning: sparse DataFrame detected. Converting to dense to avoid NaN issues.")
                    object = object.sparse.to_dense()
            cell_names = object.index.astype(str).tolist()
            gene_names = object.columns.astype(str).tolist()
            data_cxg = object.to_numpy()
        elif isinstance(object, (np.ndarray, sparse.spmatrix)):
            if genes is not None:
                gene_names = list(genes)
                if len(gene_names) != object.shape[1]:
                    raise ValueError(f"genes length ({len(gene_names)}) != matrix columns ({object.shape[1]})")
            else:
                gene_names = [f"gene_{i}" for i in range(object.shape[1])]
            if cells is not None:
                cell_names = list(cells)
                if len(cell_names) != object.shape[0]:
                    raise ValueError(f"cells length ({len(cell_names)}) != matrix rows ({object.shape[0]})")
            else:
                cell_names = [f"cell_{i}" for i in range(object.shape[0])]
            data_cxg = object

    else:
        raise TypeError(f"Unsupported input type: {type(object)}")

    # ------------------------------------------------------------------
    # 2. Convert to sparse if requested
    # ------------------------------------------------------------------
    if do_sparse and not sparse.issparse(data_cxg):
        data_cxg = sparse.csr_matrix(data_cxg)

    # ------------------------------------------------------------------
    # 3. Validate data
    # ------------------------------------------------------------------
    if sparse.issparse(data_cxg):
        data_min = data_cxg.data.min() if data_cxg.nnz > 0 else 0
    else:
        data_min = np.min(data_cxg)

    if data_min < 0:
        raise ValueError(
            "Data matrix contains negative values. "
            "Please ensure normalized data is used."
        )

    # ------------------------------------------------------------------
    # 4. Build AnnData
    # ------------------------------------------------------------------
    obs_df = pd.DataFrame(index=cell_names) if cell_names is not None else pd.DataFrame()
    var_df = pd.DataFrame(index=gene_names) if gene_names is not None else pd.DataFrame()

    adata = AnnData(X=data_cxg, obs=obs_df, var=var_df)

    # ------------------------------------------------------------------
    # 5. Create CellChat (now inherits from AnnData)
    # ------------------------------------------------------------------
    cellchat = CellChat(adata)

    # ------------------------------------------------------------------
    # 6. Handle metadata
    # ------------------------------------------------------------------
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            metadata = pd.DataFrame(metadata)

        metadata.index = metadata.index.astype(str)
        expected_cells = pd.Index(cell_names, dtype=str)
        if metadata.index.has_duplicates or not metadata.index.equals(expected_cells):
            raise ValueError("metadata must be indexed by cell names in the same order as the expression matrix.")
        cellchat.obs = metadata.copy()

        # All inputs use the same dataset label, including spatial datasets.
        if 'cellchat_dataset' not in metadata.columns:
            warnings.warn(
                "Adding 'cellchat_dataset' column to metadata. "
                "All cells assigned to 'sample1'"
            )
            cellchat.obs['cellchat_dataset'] = 'sample1'
            cellchat.obs['cellchat_dataset'] = pd.Categorical(cellchat.obs['cellchat_dataset'])
        elif not isinstance(cellchat.obs['cellchat_dataset'].dtype, pd.CategoricalDtype):
            warnings.warn("Converting 'cellchat_dataset' to categorical")
            cellchat.obs['cellchat_dataset'] = pd.Categorical(cellchat.obs['cellchat_dataset'])

        # Set cell identities
        if group_by is None:
            group_by = 'cellchat_group' if 'cellchat_group' in metadata.columns else None

        if group_by and group_by in metadata.columns:
            cellchat = set_identity(cellchat, group_by=group_by)
            if cellchat.groups is not None:
                print(f"Cell groups used for CellChat analysis: "
                      f"{list(cellchat.groups.categories)}")

    # ------------------------------------------------------------------
    # 7. Handle spatial data
    # ------------------------------------------------------------------
    if datatype == "spatial":
        if coordinates is None:
            raise ValueError("Coordinates must be provided for spatial data")

        if spatial_factors is None or 'ratio' not in spatial_factors or 'tol' not in spatial_factors:
            raise ValueError(
                "spatial_factors with 'ratio' and 'tol' must be provided "
                "for spatial data"
            )

        if not isinstance(coordinates, pd.DataFrame):
            coordinates = pd.DataFrame(coordinates, index=cell_names)
        coordinates.index = coordinates.index.astype(str)
        if not coordinates.index.equals(pd.Index(cell_names, dtype=str)):
            raise ValueError("coordinates must be indexed by cell names in expression-matrix order.")
        coordinates = coordinates.to_numpy()
        if coordinates.shape[1] != 2:
            raise ValueError("Coordinates must have exactly 2 columns (x, y)")

        cellchat.spatial = {
            'coordinates': pd.DataFrame(coordinates, index=cell_names) if cell_names is not None else coordinates,
            'spatial_factors': spatial_factors
        }
        cellchat.obsm['spatial'] = coordinates
        print("Create a CellChat object from spatial transcriptomics data...")

    cellchat.settings['datatype'] = datatype.upper()

    # ------------------------------------------------------------------
    # 8. Initialize the signaling layer with the full expression matrix.
    # ------------------------------------------------------------------
    if cellchat.signaling is None:
        cellchat.signaling = cellchat.X

    return cellchat

def _sync_network_groups_after_set_identity(cellchat: CellChat, display_warning: bool) -> None:
    """Synchronize existing LR-level networks after cell identities change."""
    network = cellchat.network
    if not isinstance(network, dict) or "prob" not in network:
        return

    groups = cellchat.groups
    if groups is None or not hasattr(groups, "categories"):
        return

    new_groups = [str(group) for group in groups.categories]
    stored_groups = [str(group) for group in network.get("groups", [])]

    prob = network.get("prob")
    if not is_matrix_dict(prob):
        warnings.warn(
            "Existing network['prob'] is not matrix-dictionary storage; "
            "cannot synchronize cell group order. Re-run compute_communication_probability.",
            UserWarning,
            stacklevel=3,
        )
        return

    first_matrix = next(iter(prob.values()), None)
    if first_matrix is None:
        network["groups"] = new_groups
        return

    if first_matrix.shape[0] != first_matrix.shape[1]:
        raise ValueError("Network matrices must be square group-by-group matrices")

    n_groups = first_matrix.shape[0]

    if stored_groups and set(stored_groups) == set(new_groups) and len(stored_groups) == len(new_groups):
        order_indices = [stored_groups.index(group) for group in new_groups]

        if order_indices != list(range(len(stored_groups))):
            print("Reorder cell groups!")
            print(f"The cell group order before reordering is {stored_groups}")

            network["prob"] = reorder_group_axes(network["prob"], order_indices)

            if "pval" in network and is_matrix_dict(network["pval"]):
                network["pval"] = reorder_group_axes(network["pval"], order_indices)

            print(f"The cell group order after reordering is {new_groups}")

        network["groups"] = new_groups

    elif len(new_groups) == n_groups:
        print("Rename cell groups but do not change the order!")
        print(f"The cell group order before renaming is {stored_groups or list(range(n_groups))}")
        network["groups"] = new_groups
        print(f"The cell group order after renaming is {new_groups}")

    else:
        warnings.warn(
            "The new cell group levels do not match the existing network "
            f"shape ({n_groups} groups). Existing network['prob'] was not reordered. "
            "Re-run compute_communication_probability.",
            UserWarning,
            stacklevel=3,
        )
        return

    if display_warning:
        warnings.warn(
            "All calculations after compute_communication_probability should be re-run. "
            "These include compute_pathway_probability, aggregate_network, and "
            "compute_network_centrality.",
            UserWarning,
            stacklevel=3,
        )


def _expand_square_group_matrix(
    matrix, source_to_target, n_source_groups, n_target_groups, fill_value=0.0
):
    """Expand a group-by-group dense or sparse matrix to a new group order."""
    source_to_target = np.asarray(source_to_target, dtype=int)
    if matrix.shape != (n_source_groups, n_source_groups):
        raise ValueError(
            "Network matrix has shape "
            f"{matrix.shape}; expected {(n_source_groups, n_source_groups)}"
        )

    if sparse.issparse(matrix):
        source = matrix.tocoo()
        expanded = sparse.coo_matrix(
            (
                source.data,
                (source_to_target[source.row], source_to_target[source.col]),
            ),
            shape=(n_target_groups, n_target_groups),
        ).tocsr()
        expanded.eliminate_zeros()
        return expanded

    source = np.asarray(matrix)
    expanded = np.full(
        (n_target_groups, n_target_groups), fill_value, dtype=source.dtype
    )
    expanded[np.ix_(source_to_target, source_to_target)] = source
    return expanded


def _first_network_group_count(network_slot):
    """Return the matrix axis length for a network slot, if available."""
    for key in ['prob', 'pval']:
        value = network_slot.get(key)
        if is_matrix_dict(value):
            first_matrix = next(iter(value.values()), None)
            if first_matrix is not None:
                if first_matrix.shape[0] != first_matrix.shape[1]:
                    raise ValueError(
                        f"Network {key!r} matrices must be square group-by-group matrices"
                    )
                return first_matrix.shape[0]

    for key in ['count', 'weight']:
        matrix = network_slot.get(key)
        if matrix is not None:
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(
                    f"Network {key!r} matrix must be square group-by-group matrix"
                )
            return matrix.shape[0]

    return None


def _lift_network_slot(
    network_slot, group_max, context, pval_fill=1.0, source_groups=None
):
    """Expand a CellChat network slot to a shared set of cell groups."""
    if not isinstance(network_slot, dict) or not network_slot:
        return network_slot

    n_existing = _first_network_group_count(network_slot)
    stored_groups = network_slot.get('groups')
    if stored_groups:
        source_groups = list(stored_groups)
    elif n_existing == len(group_max):
        source_groups = list(group_max)
    else:
        source_groups = list(source_groups or [])
    if not source_groups:
        return network_slot

    missing_groups = [group for group in source_groups if group not in group_max]
    if missing_groups:
        raise ValueError(
            f"group_new is missing existing network groups for {context}: "
            f"{missing_groups}"
        )

    source_to_target = [group_max.index(group) for group in source_groups]
    n_source_groups = len(source_groups)
    result = dict(network_slot)

    if 'prob' in result:
        if not is_matrix_dict(result['prob']):
            raise TypeError(
                f"{context}['prob'] must be a dictionary of name -> "
                "two-dimensional matrix"
            )
        result['prob'] = expand_group_axes(
            result['prob'], source_to_target, len(group_max)
        )

    if 'pval' in result:
        if not is_matrix_dict(result['pval']):
            raise TypeError(
                f"{context}['pval'] must be a dictionary of name -> "
                "two-dimensional matrix"
            )
        result['pval'] = expand_dense_group_axes(
            result['pval'],
            source_to_target,
            len(group_max),
            fill_value=pval_fill,
        )

    for key in ['count', 'weight']:
        if key in result:
            result[key] = _expand_square_group_matrix(
                result[key],
                source_to_target,
                n_source_groups,
                len(group_max),
                fill_value=0.0,
            )

    result['groups'] = list(group_max)
    return result

def set_identity(
    cellchat: CellChat,
    group_by: str,
    levels: Optional[List[str]] = None,
    display_warning: bool = True
) -> CellChat:
    """
    Set the default identity of cells.

    Parameters
    ----------
    cellchat : CellChat
        CellChat object.
    group_by : str
        Metadata column to use for cell identities.
    levels : list of str, optional
        Levels for the categorical variable.
    display_warning : bool
        Whether to display warning messages.

    Returns
    -------
    CellChat
        Updated CellChat object.
    """
    if group_by not in cellchat.obs.columns:
        raise KeyError(f"Metadata column {group_by!r} does not exist.")
    cellchat.groups = pd.Categorical(cellchat.obs[group_by])

    if levels is not None:
        cellchat.groups = pd.Categorical(cellchat.groups, categories=levels)

    groups = cellchat.groups
    if groups is not None and hasattr(groups, 'categories'):
        if '0' in groups.categories:
            raise ValueError("Cell labels cannot contain '0'!")
    _sync_network_groups_after_set_identity(
        cellchat,
        display_warning=display_warning,
    )
    return cellchat


def merge_cellchat(
    object_list: List[CellChat],
    add_names: Optional[List[str]] = None,
    merge_data: bool = False,
    cell_prefix: bool = False
) -> CellChat:
    """
    Merge multiple CellChat objects.

    Parameters
    ----------
    object_list : list of CellChat
        List of CellChat objects to merge.
    add_names : list of str, optional
        Names for each dataset.  If None, uses ``"Dataset_1"``, etc.
    merge_data : bool
        Whether to merge all gene data or just signaling genes.
    cell_prefix : bool
        Whether to prefix cell names with dataset names.

    Returns
    -------
    CellChat
        Merged CellChat object.
    """
    if len(object_list) < 2:
        raise ValueError("At least 2 CellChat objects required for merging")

    if add_names is None:
        add_names = [f"Dataset_{i}" for i in range(1, len(object_list) + 1)]
    if len(add_names) != len(object_list):
        raise ValueError("Length of add_names must match object_list")
    # Validate compatible data types
    datatypes = set(obj.settings.get('datatype', 'RNA') for obj in object_list)
    if len(datatypes) > 1:
        print(f"Data types in objects: {list(datatypes)}")
        raise ValueError("Cannot merge objects with different data types")

    if len(set(add_names)) != len(add_names):
        raise ValueError("add_names must contain unique dataset names")
    for obj in object_list:
        if obj.groups is None:
            raise ValueError("Every object must define obs['cellchat_group'] before merging")

    network_by_dataset = {
        name: obj.network for name, obj in zip(add_names, object_list)
    }
    pathway_network_by_dataset = {
        name: obj.pathway_network for name, obj in zip(add_names, object_list)
    }
    lr_pairs_by_dataset = {
        name: obj.lr_pairs for name, obj in zip(add_names, object_list)
    }
    feature_results_by_dataset = {
        name: obj.feature_results for name, obj in zip(add_names, object_list)
    }
    spatial_by_dataset = {
        name: obj.spatial for name, obj in zip(add_names, object_list)
    }

    # ------------------------------------------------------------------
    # Handle cell names
    # ------------------------------------------------------------------
    if cell_prefix:
        warnings.warn("Prefixing cell names with dataset names")
        for i, obj in enumerate(object_list):
            # Update obs_names directly
            new_obs_names = [f"{name}_{add_names[i]}" for name in obj.obs_names]
            obj.obs_names = new_obs_names
    else:
        all_cell_names = []
        for obj in object_list:
            all_cell_names.extend(obj.obs_names.tolist())

        if len(set(all_cell_names)) != len(all_cell_names):
            raise ValueError(
                "Duplicate cell names detected across datasets. "
                "Set cell_prefix=True"
            )

    # ------------------------------------------------------------------
    # Merge metadata
    # ------------------------------------------------------------------
    meta_use_cols = set(object_list[0].obs.columns.tolist())
    for obj in object_list[1:]:
        meta_use_cols &= set(obj.obs.columns.tolist())
    meta_use_cols = list(meta_use_cols)

    meta_combined_list = []
    all_cell_names = []

    for i, obj in enumerate(object_list):
        dataset_name = add_names[i]
        current_cell_names = obj.obs_names.tolist()
        all_cell_names.extend(current_cell_names)

        obj_meta = obj.obs[meta_use_cols].copy()
        obj_meta['cellchat_dataset'] = dataset_name
        meta_combined_list.append(obj_meta)

    meta_combined = pd.concat(meta_combined_list)

    # Align if needed
    if not meta_combined.index.equals(pd.Index(all_cell_names)):
        warnings.warn("Aligning metadata index with cell names")
        meta_combined.index = all_cell_names

    # ------------------------------------------------------------------
    # Find common genes
    # ------------------------------------------------------------------
    common_genes = set(object_list[0].var_names)
    for obj in object_list[1:]:
        common_genes &= set(obj.var_names)
    common_genes = [
    gene for gene in object_list[0].var_names
    if all(gene in obj.var_names for obj in object_list[1:])
    ]

    # ------------------------------------------------------------------
    # Merge data matrices (cells x genes internally, so we vstack)
    # ------------------------------------------------------------------
    if merge_data:
        print("Merging expression, metadata, spatial data, and dataset networks")
        # Extract cellsxgenes slices and vertical-stack
        mats = []
        for obj in object_list:
            m = obj[:, common_genes].X
            mats.append(m)
        data_cxg_combined = sparse.vstack(mats) if sparse.issparse(mats[0]) else np.vstack(mats)
        merged_var_names = common_genes
    else:
        print("Merging signaling expression, metadata, spatial data, and dataset networks")
        signaling_genes = [
            gene for gene in common_genes
            if all(
                'is_signaling' not in obj.var.columns
                or bool(obj.var.loc[gene, 'is_signaling'])
                for obj in object_list
            )
        ]
        if not signaling_genes:
            signaling_genes = common_genes

        mats = []
        for obj in object_list:
            m = obj[:, signaling_genes].X
            mats.append(m)
        data_cxg_combined = sparse.vstack(mats) if sparse.issparse(mats[0]) else np.vstack(mats)
        merged_var_names = signaling_genes

    all_group_levels = []
    for obj in object_list:
        if hasattr(obj.groups, 'categories'):
            for group in obj.groups.categories:
                if group not in all_group_levels:
                    all_group_levels.append(group)
    meta_combined['cellchat_group'] = pd.Categorical(
        meta_combined['cellchat_group'], categories=all_group_levels
    )
    meta_combined['cellchat_dataset'] = pd.Categorical(
        meta_combined['cellchat_dataset'], categories=add_names, ordered=True
    )

    # ------------------------------------------------------------------
    # Create merged AnnData after fixing canonical categorical columns.
    # ------------------------------------------------------------------
    merged_adata = AnnData(
        X=data_cxg_combined,
        obs=meta_combined,
        var=pd.DataFrame(index=merged_var_names)
    )

    # ------------------------------------------------------------------
    # Create merged CellChat
    # ------------------------------------------------------------------
    merged_object = CellChat(merged_adata)

    merged_object.signaling = data_cxg_combined

    merged_object.spatial = spatial_by_dataset
    merged_object.network = network_by_dataset
    merged_object.pathway_network = pathway_network_by_dataset
    merged_object.feature_results = feature_results_by_dataset
    merged_object.lr_pairs = lr_pairs_by_dataset
    merged_object.database = object_list[0].database.copy()

    merged_object.settings['mode'] = 'merged'
    merged_object.settings['datatype'] = object_list[0].settings.get('datatype', 'RNA')

    return merged_object


def lift_cellchat(
    cellchat: CellChat,
    group_new: Optional[List[str]] = None
) -> CellChat:
    """
    Update CellChat object by lifting cell groups to same labels across datasets.

    Useful when comparing inferred communications across different datasets
    with different cellular compositions.

    Parameters
    ----------
    cellchat : CellChat
        CellChat object (single or merged).
    group_new : list of str, optional
        New cell group labels to use. If None and merged object, uses the
        ordered union of labels across datasets.

    Returns
    -------
    CellChat
        Updated CellChat object with unified cell labels.
    """
    def _validate_group_new_contains_existing(groups, group_max, context):
        existing_groups = list(pd.Categorical(groups).categories)
        missing_groups = [group for group in existing_groups if group not in group_max]
        if missing_groups:
            raise ValueError(
                f"group_new is missing existing cell groups for {context}: "
                f"{missing_groups}"
            )

    if cellchat.settings.get('mode') == 'merged':
        if 'cellchat_dataset' not in cellchat.obs.columns:
            raise ValueError("Merged objects require obs['cellchat_dataset']")
        dataset_names = list(cellchat.obs['cellchat_dataset'].cat.categories)
        groups_by_dataset = [
            pd.Categorical(
                cellchat.obs.loc[cellchat.obs['cellchat_dataset'] == dataset_name,
                                 'cellchat_group']
            )
            for dataset_name in dataset_names
        ]

        if group_new is None:
            group_max = []
            for dataset_groups in groups_by_dataset:
                for group in dataset_groups.categories:
                    if group not in group_max:
                        group_max.append(group)
        else:
            group_max = list(group_new)

        print(f"Lifting CellChat object using cell labels: "
              f"{', '.join(group_max)}")

        network_by_dataset = dict(cellchat.network)
        pathway_network_by_dataset = dict(cellchat.pathway_network)
        # Pandas categoricals cannot accept values with a different category
        # set during subset assignment.  Normalize once after all datasets
        # have been processed instead of assigning categoricals per subset.
        cellchat.obs['cellchat_group'] = cellchat.obs['cellchat_group'].astype(object)
        for dataset_name, dataset_groups in zip(dataset_names, groups_by_dataset):
            _validate_group_new_contains_existing(dataset_groups, group_max, dataset_name)
            print(f"Updating dataset {dataset_name}")
            cell_mask = cellchat.obs['cellchat_dataset'].astype(str).eq(str(dataset_name))
            if dataset_name in network_by_dataset:
                network_by_dataset[dataset_name] = _lift_network_slot(
                    network_by_dataset[dataset_name],
                    group_max,
                    f"network[{dataset_name!r}]",
                    source_groups=list(dataset_groups.categories),
                )
            if dataset_name in pathway_network_by_dataset:
                pathway_network_by_dataset[dataset_name] = _lift_network_slot(
                    pathway_network_by_dataset[dataset_name],
                    group_max,
                    f"pathway_network[{dataset_name!r}]",
                    pval_fill=0.0,
                    source_groups=list(dataset_groups.categories),
                )

        cellchat.obs['cellchat_group'] = pd.Categorical(
            cellchat.obs['cellchat_group'], categories=group_max
        )
        cellchat.network = network_by_dataset
        cellchat.pathway_network = pathway_network_by_dataset
    else:
        if group_new is None:
            raise ValueError(
                "group_new must be specified for single CellChat objects"
            )

        group_max = list(group_new)
        print(f"Lifting CellChat object using cell labels: "
              f"{', '.join(group_max)}")
        groups = pd.Categorical(cellchat.groups)
        _validate_group_new_contains_existing(groups, group_max, 'single object')
        cellchat.groups = pd.Categorical(groups, categories=group_max)
        cellchat.network = _lift_network_slot(
            cellchat.network,
            group_max,
            'network',
            source_groups=list(groups.categories),
        )
        cellchat.pathway_network = _lift_network_slot(
            cellchat.pathway_network,
            group_max,
            'pathway_network',
            pval_fill=0.0,
            source_groups=list(groups.categories),
        )

    return cellchat


def _as_indexed_categorical_series(labels, index, name="labels") -> pd.Series:
    """Return labels as a categorical Series aligned to cell names."""
    if labels is None:
        raise ValueError(f"{name} are not set")

    index = pd.Index(index)
    if isinstance(labels, pd.Series):
        series = labels.copy()
        if series.index.is_unique and index.isin(series.index).all():
            series = series.reindex(index)
        elif len(series) == len(index):
            series = pd.Series(series.to_numpy(), index=index, name=series.name)
        else:
            raise ValueError(
                f"{name} length ({len(series)}) does not match cell count ({len(index)})"
            )
    else:
        if len(labels) != len(index):
            raise ValueError(
                f"{name} length ({len(labels)}) does not match cell count ({len(index)})"
            )
        series = pd.Series(labels, index=index, name=name)

    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")
    return series


def _copy_subset_spatial(spatial: dict, cells_use: List[str], cell_indices: List[int]) -> dict:
    """Copy spatial image metadata and subset per-cell coordinates when present."""
    subset_spatial = {}
    for key, value in spatial.items():
        if key == 'coordinates':
            if isinstance(value, pd.DataFrame):
                subset_spatial[key] = value.reindex(cells_use).copy()
            else:
                subset_spatial[key] = np.asarray(value)[cell_indices].copy()
        elif hasattr(value, 'copy'):
            subset_spatial[key] = value.copy()
        else:
            subset_spatial[key] = value
    return subset_spatial


def subset_cellchat(
    cellchat: CellChat,
    cells_use: Optional[List[str]] = None,
    groups_use: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    invert: bool = False,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> CellChat:
    """
    Subset CellChat object using a portion of cells.

    Parameters
    ----------
    cellchat : CellChat
        CellChat object to subset.
    cells_use : list of str, optional
        List of cell barcodes to subset.
    groups_use : list of str, optional
        List of cell identity labels to subset.
    group_by : str, optional
        Column name to use for grouping.
    invert : bool
        Whether to invert the selection.
    metadata_filter : dict, optional
        Metadata criteria used to select cells. Keys must be columns in
        ``cellchat.obs``. A scalar value selects equal values; a list, tuple,
        set, or pandas index selects values contained in that collection.
        Multiple criteria are combined with AND. This option is mutually
        exclusive with ``cells_use``, ``groups_use``, ``group_by``, and
        ``invert``.

    Returns
    -------
    CellChat
        Subsetted CellChat object.
    """
    has_metadata_filter = metadata_filter is not None
    if has_metadata_filter:
        if not isinstance(metadata_filter, dict) or not metadata_filter:
            raise ValueError("metadata_filter must be a non-empty dictionary")
        if any(value is not None for value in (cells_use, groups_use, group_by)) or invert:
            raise ValueError(
                "metadata_filter cannot be combined with cells_use, groups_use, "
                "group_by, or invert"
            )
    elif cells_use is None and groups_use is None:
        raise ValueError("Must specify either cells_use, groups_use, or metadata_filter")
    elif cells_use is not None and groups_use is not None:
        raise ValueError("Specify only one of cells_use or groups_use")

    cell_index = pd.Index(cellchat.obs_names)
    if has_metadata_filter:
        selection = pd.Series(True, index=cell_index, dtype=bool)
        for column, criterion in metadata_filter.items():
            if column not in cellchat.obs.columns:
                raise ValueError(f"Column {column!r} not found in cellchat.obs")
            values = cellchat.obs.loc[cell_index, column]
            if isinstance(criterion, (list, tuple, set, frozenset, pd.Index, np.ndarray)):
                if isinstance(criterion, np.ndarray) and criterion.ndim != 1:
                    raise ValueError(
                        f"metadata_filter criterion for {column!r} must be one-dimensional"
                    )
                allowed = list(criterion)
                if any(not np.isscalar(value) and value is not None for value in allowed):
                    raise ValueError(
                        f"metadata_filter criterion for {column!r} must contain scalar values"
                    )
                if any(pd.isna(value) for value in allowed):
                    criterion_mask = values.isin(allowed) | values.isna()
                else:
                    criterion_mask = values.isin(allowed)
            elif not np.isscalar(criterion) and criterion is not None:
                raise ValueError(
                    f"metadata_filter criterion for {column!r} must be a scalar or collection"
                )
            elif pd.isna(criterion):
                criterion_mask = values.isna()
            else:
                criterion_mask = values == criterion
            selection &= criterion_mask.fillna(False).to_numpy()

        cell_indices = np.flatnonzero(selection.to_numpy()).tolist()
        cells_use = cell_index[selection.to_numpy()].tolist()
        labels = _as_indexed_categorical_series(
            cellchat.groups, cell_index, name="cell identities"
        )
        selected_labels = labels.loc[cells_use]
        observed_levels = set(selected_labels.dropna())
        level_use = [
            level for level in labels.cat.categories
            if level in observed_levels
        ]
    else:
        if group_by is None:
            labels = cellchat.groups
        else:
            if group_by not in cellchat.obs.columns:
                raise ValueError(f"Column {group_by!r} not found in cellchat.obs")
            labels = cellchat.obs[group_by]

        labels = _as_indexed_categorical_series(labels, cell_index, name="cell identities")

        if groups_use is not None:
            requested_levels = list(groups_use)
            missing_levels = [
                level for level in requested_levels
                if level not in labels.cat.categories
            ]
            if missing_levels:
                raise ValueError(f"Identity labels not found: {missing_levels}")

            observed_levels = set(labels.dropna())
            available_levels = [
                level for level in labels.cat.categories
                if level in observed_levels
            ]
            requested = set(requested_levels)
            if invert:
                level_use = [level for level in available_levels if level not in requested]
            else:
                level_use = [level for level in available_levels if level in requested]

            mask = labels.isin(level_use).to_numpy()
            cell_indices = np.flatnonzero(mask).tolist()
            cells_use = cell_index[mask].tolist()
        else:
            requested_cells = pd.Index(cells_use)
            if requested_cells.has_duplicates:
                duplicates = requested_cells[requested_cells.duplicated()].unique().tolist()
                raise ValueError(f"Duplicate cells requested: {duplicates}")
            missing_cells = requested_cells.difference(cell_index)
            if len(missing_cells) > 0:
                raise ValueError(f"Cells not found: {missing_cells.tolist()}")

            cell_indices = cell_index.get_indexer(requested_cells).tolist()
            cells_use = requested_cells.tolist()

            selected_labels = labels.loc[cells_use]
            observed_levels = set(selected_labels.dropna())
            level_use = [
                level for level in labels.cat.categories
                if level in observed_levels
            ]

    if not cell_indices:
        raise ValueError("No cells selected")

    print(f"Cell groups used for analysis: {level_use}")

    sub_adata = cellchat[cell_indices, :].copy()
    subset_object = CellChat(sub_adata)

    subset_object.groups = pd.Categorical(labels.loc[cells_use], categories=level_use)

    subset_object.database = cellchat.database.copy()
    subset_object.lr_pairs = cellchat.lr_pairs.copy()
    subset_object.feature_results = cellchat.feature_results.copy()
    subset_object.settings = cellchat.settings.copy()
    subset_object.network = {}
    subset_object.pathway_network = {}

    if cellchat.spatial:
        subset_object.spatial = _copy_subset_spatial(cellchat.spatial, cells_use, cell_indices)

    return subset_object


def intersection(lst1: List, lst2: List) -> List:
    """Return intersection of two lists."""
    return list(set(lst1) & set(lst2))


