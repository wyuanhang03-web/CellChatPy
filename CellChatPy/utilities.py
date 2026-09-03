#!/usr/bin/env python3
"""
Utility functions for CellChat
"""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist
from typing import Union, Optional, Dict, List, Any
import warnings

from .cellchat_class import CellChat
from .network_storage import (
    matrix_dict_from_array,
    network_names,
    normalize_network_similarity,
    normalize_network_slot,
)


# ---------------------------------------------------------------------------
# General spatial input/output
# ---------------------------------------------------------------------------
spatial_h5_schema = "cellchat-spatial-h5"
spatial_h5_schema_version = "2.0"


def _decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _write_strings(group: h5py.Group, name: str, values) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([str(value) for value in values], dtype=object), dtype=dtype)


def _read_frame(group: h5py.Group) -> pd.DataFrame:
    index = _decode(group["index"][:])
    columns = _decode(group["columns"][:])
    data = {}
    for column in columns:
        values = group[column][:]
        data[column] = _decode(values) if values.dtype.kind in "SUO" else values
    return pd.DataFrame(data, index=index)


def _write_frame(parent: h5py.Group, name: str, frame: pd.DataFrame) -> None:
    group = parent.create_group(name)
    _write_strings(group, "index", frame.index.astype(str))
    _write_strings(group, "columns", frame.columns.astype(str))
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            group.create_dataset(str(column), data=values.to_numpy())
        else:
            _write_strings(group, str(column), values.fillna("").astype(str))


def _write_matrix_group(parent: h5py.Group, name: str, matrix, features: Sequence[str], cells: Sequence[str]) -> None:
    group = parent.create_group(name)
    matrix = matrix.tocsc() if sparse.issparse(matrix) else sparse.csc_matrix(np.asarray(matrix))
    if matrix.shape != (len(cells), len(features)):
        raise ValueError(f"Matrix shape does not match its cell and feature names: {matrix.shape}")
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)
    group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
    _write_strings(group, "features", features)
    _write_strings(group, "cells", cells)


def _normalise_modalities(modalities, cells: Sequence[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name, value in (modalities or {}).items():
        name = str(name)
        if not name or name == "expression" or "/" in name:
            raise ValueError("Modality names must be non-empty and cannot be 'expression' or contain '/'.")
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Each modality must be a pandas DataFrame with cells as rows and features as columns.")
        frame = value.astype(float).copy()
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"Modality {name!r} must have unique cell and feature names.")
        if not frame.index.equals(pd.Index(cells, dtype=str)):
            raise ValueError(f"Modality {name!r} must use the expression cells in the same order.")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Modality {name!r} must contain finite non-negative values.")
        result[name] = frame
    return result


def _as_spatial_csc(expression, genes: Sequence[str], cells: Sequence[str]) -> sparse.csc_matrix:
    matrix = expression.tocsc() if sparse.issparse(expression) else sparse.csc_matrix(np.asarray(expression))
    expected_shape = (len(cells), len(genes))
    if matrix.shape != expected_shape:
        raise ValueError(f"Expression matrix must be cells x genes {expected_shape}; got {matrix.shape}.")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Expression matrix must contain finite non-negative values.")
    return matrix


def _normalise_spatial_metadata(metadata, cells: Sequence[str], sample: str | None = None) -> pd.DataFrame:
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame indexed by cell/spot ID.")
    result = metadata.copy()
    result.index = result.index.astype(str)
    cells = [str(cell) for cell in cells]
    if result.index.has_duplicates:
        raise ValueError("metadata index must contain unique cell/spot IDs.")
    missing = pd.Index(cells).difference(result.index)
    if len(missing):
        raise ValueError(f"metadata is missing {len(missing)} expression cells/spots.")
    result = result.loc[cells].copy()
    if "cellchat_group" not in result.columns:
        raise ValueError("metadata must contain a 'cellchat_group' column for CellChat grouping.")
    result["cellchat_group"] = result["cellchat_group"].astype(str)
    if sample is not None:
        result["cellchat_dataset"] = str(sample)
    elif "cellchat_dataset" not in result.columns:
        result["cellchat_dataset"] = "sample1"
    result["cellchat_dataset"] = result["cellchat_dataset"].astype(str)
    return result


def _read_prediction_frame(group: h5py.Group, cells: Sequence[str]) -> pd.DataFrame | None:
    if "values" not in group:
        return None
    values = np.asarray(group["values"][:])
    cells = [str(cell) for cell in cells]
    if "index" in group and "columns" in group:
        frame = pd.DataFrame(values, index=_decode(group["index"][:]), columns=_decode(group["columns"][:]))
    elif "cells" in group and "labels" in group:
        stored_cells, labels = _decode(group["cells"][:]), _decode(group["labels"][:])
        if values.shape == (len(stored_cells), len(labels)):
            frame = pd.DataFrame(values, index=stored_cells, columns=labels)
        elif values.shape == (len(labels), len(stored_cells)):
            frame = pd.DataFrame(values.T, index=stored_cells, columns=labels)
        else:
            raise ValueError("Prediction matrix dimensions do not match its cell and label names.")
    else:
        raise ValueError("predictions must store either index/columns or cells/labels.")
    frame.index = frame.index.astype(str)
    if not set(cells).issubset(frame.index):
        raise ValueError("Prediction rows are not aligned with expression cells/spots.")
    return frame.loc[cells].copy()


def _spatial_factors_from_handle(handle: h5py.File) -> dict[str, list[float]]:
    if "spatial_factors" in handle:
        factors = handle["spatial_factors"]
        ratio, tol = factors["ratio"][:], factors["tol"][:]
    elif "ratio_um_per_pixel" in handle.attrs and "tol_um" in handle.attrs:
        ratio, tol = [handle.attrs["ratio_um_per_pixel"]], [handle.attrs["tol_um"]]
    else:
        raise ValueError("Spatial H5 file must contain ratio and tol spatial factors.")
    ratio = np.atleast_1d(ratio).astype(float).tolist()
    tol = np.atleast_1d(tol).astype(float).tolist()
    if not ratio or len(ratio) != len(tol) or not np.isfinite(ratio).all() or not np.isfinite(tol).all():
        raise ValueError("Spatial factors must contain matching finite ratio and tol values.")
    return {"ratio": ratio, "tol": tol}


def write_spatial_h5(path, expression, genes: Sequence[str], cells: Sequence[str], metadata: pd.DataFrame,
                     coordinates, spatial_factors: Mapping[str, Any], *, sample: str | None = None,
                     predictions: pd.DataFrame | None = None, coordinates_raw=None, image_hires=None,
                     scale_factors: pd.DataFrame | None = None,
                     modalities: Mapping[str, pd.DataFrame] | None = None,
                     attrs: Mapping[str, Any] | None = None) -> None:
    """Write the canonical CellChatPy spatial H5 input format."""
    path = os.fspath(path)
    genes, cells = [str(gene) for gene in genes], [str(cell) for cell in cells]
    if len(set(genes)) != len(genes) or len(set(cells)) != len(cells):
        raise ValueError("genes and cells must each be unique.")
    matrix = _as_spatial_csc(expression, genes, cells)
    modalities = _normalise_modalities(modalities, cells)
    metadata = _normalise_spatial_metadata(metadata, cells, sample=sample)
    coordinates = pd.DataFrame(coordinates).copy()
    coordinates.index = coordinates.index.astype(str)
    if not set(cells).issubset(coordinates.index) or coordinates.shape[1] < 2:
        raise ValueError("coordinates must contain at least two columns aligned with expression cells/spots.")
    coordinates = coordinates.loc[cells]
    if not np.isfinite(coordinates.iloc[:, :2].to_numpy(dtype=float)).all():
        raise ValueError("Spatial coordinates must be finite numeric values.")
    if not {"ratio", "tol"}.issubset(spatial_factors):
        raise ValueError("spatial_factors must contain 'ratio' and 'tol'.")
    ratio = np.atleast_1d(spatial_factors["ratio"]).astype(float)
    tol = np.atleast_1d(spatial_factors["tol"]).astype(float)
    if len(ratio) != len(tol) or not len(ratio) or not np.isfinite(ratio).all() or not np.isfinite(tol).all():
        raise ValueError("spatial_factors ratio and tol must be matching non-empty finite vectors.")

    predictions = None if predictions is None else pd.DataFrame(predictions).copy()
    if predictions is not None:
        predictions.index = predictions.index.astype(str)
        if not set(cells).issubset(predictions.index):
            raise ValueError("Prediction rows must be aligned with expression cells/spots.")
        predictions = predictions.loc[cells]
    raw_frame = None if coordinates_raw is None else pd.DataFrame(coordinates_raw).copy()
    if raw_frame is not None:
        raw_frame.index = raw_frame.index.astype(str)
        if not set(cells).issubset(raw_frame.index):
            raise ValueError("coordinates_raw must be aligned with expression cells/spots.")
        raw_frame = raw_frame.loc[cells]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = spatial_h5_schema
        handle.attrs["schema_version"] = spatial_h5_schema_version
        if sample is not None:
            handle.attrs["sample"] = str(sample)
        for key, value in (attrs or {}).items():
            if isinstance(value, (str, int, float, np.number, bool)):
                handle.attrs[str(key)] = value
        group = handle.create_group("expression")
        group.create_dataset("data", data=matrix.data)
        group.create_dataset("indices", data=matrix.indices)
        group.create_dataset("indptr", data=matrix.indptr)
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        _write_strings(group, "genes", genes)
        _write_strings(group, "cells", cells)
        if modalities:
            modalities_group = handle.create_group("modalities")
            for name, frame in modalities.items():
                _write_matrix_group(modalities_group, name, frame.to_numpy(), frame.columns, cells)
        _write_frame(handle, "metadata", metadata)
        group = handle.create_group("spatial")
        group.create_dataset("coordinates", data=coordinates.to_numpy(dtype=float))
        _write_strings(group, "cells", cells)
        _write_strings(group, "columns", coordinates.columns.astype(str))
        if raw_frame is not None:
            group.create_dataset("coordinates_raw", data=raw_frame.to_numpy(dtype=float))
            _write_strings(group, "raw_columns", raw_frame.columns.astype(str))
        if image_hires is not None:
            group.create_dataset("image_hires", data=np.asarray(image_hires), compression="gzip")
        if scale_factors is not None:
            _write_frame(group, "scale_factors", pd.DataFrame(scale_factors))
        group = handle.create_group("spatial_factors")
        group.create_dataset("ratio", data=ratio)
        group.create_dataset("tol", data=tol)
        if predictions is not None:
            group = handle.create_group("predictions")
            group.create_dataset("values", data=predictions.to_numpy(dtype=float))
            _write_strings(group, "index", predictions.index.astype(str))
            _write_strings(group, "columns", predictions.columns.astype(str))


def read_spatial_h5(path, *, sample: str | None = None) -> dict[str, Any]:
    """Read a versioned CellChatPy spatial H5 file."""
    path = os.fspath(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != spatial_h5_schema:
            raise ValueError("This is not a CellChatPy spatial H5 file.")
        if str(handle.attrs.get("schema_version")) != spatial_h5_schema_version:
            raise ValueError(
                f"Unsupported spatial H5 schema version {handle.attrs.get('schema_version')!r}; "
                f"expected {spatial_h5_schema_version}."
            )
        if not {"expression", "metadata", "spatial"}.issubset(handle):
            raise ValueError("CellChatPy spatial H5 is missing expression, metadata, or spatial data.")
        group = handle["expression"]
        shape = tuple(group["shape"][:].astype(int))
        expression = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
        )
        genes, cells = _decode(group["genes"][:]), _decode(group["cells"][:])
        if expression.shape != (len(cells), len(genes)):
            raise ValueError("Expression matrix shape does not match gene/cell names.")
        metadata = _read_frame(handle["metadata"])
        metadata.index = metadata.index.astype(str)
        if not set(cells).issubset(metadata.index):
            raise ValueError("Metadata rows are not aligned with expression cells/spots.")
        metadata = metadata.loc[cells].copy()
        group = handle["spatial"]
        spatial_cells = _decode(group["cells"][:])
        coordinates = pd.DataFrame(
            group["coordinates"][:], index=spatial_cells, columns=_decode(group["columns"][:])
        ).loc[cells]
        if coordinates.shape[1] < 2 or not np.isfinite(coordinates.iloc[:, :2].to_numpy(dtype=float)).all():
            raise ValueError("Spatial coordinates must contain at least two finite numeric columns.")
        result: dict[str, Any] = {
            "path": path,
            "sample": str(sample or handle.attrs.get("sample") or "sample1"),
            "expression": expression, "genes": genes, "cells": cells, "metadata": metadata,
            "coordinates": coordinates, "spatial_factors": _spatial_factors_from_handle(handle),
            "attrs": dict(handle.attrs),
        }
        if "predictions" in handle:
            predictions = _read_prediction_frame(handle["predictions"], cells)
            if predictions is not None:
                result["predictions"] = predictions
        if "coordinates_raw" in group:
            columns = _decode(group["raw_columns"][:]) if "raw_columns" in group else ["x", "y"]
            result["coordinates_raw"] = pd.DataFrame(
                group["coordinates_raw"][:], index=spatial_cells, columns=columns
            ).loc[cells]
        if "image_hires" in group:
            result["image_hires"] = group["image_hires"][:]
        if "scale_factors" in group:
            result["scale_factors"] = _read_frame(group["scale_factors"])
        result["modalities"] = {}
        if "modalities" in handle:
            for name, modality_group in handle["modalities"].items():
                matrix_shape = tuple(modality_group["shape"][:].astype(int))
                modality = sparse.csc_matrix(
                    (modality_group["data"][:], modality_group["indices"][:], modality_group["indptr"][:]),
                    shape=matrix_shape,
                )
                features = _decode(modality_group["features"][:])
                modality_cells = _decode(modality_group["cells"][:])
                if modality_cells != cells:
                    raise ValueError(f"Modality {name!r} is not aligned with the expression cells.")
                result["modalities"][name] = pd.DataFrame(modality.toarray(), index=cells, columns=features)
    labels = metadata["cellchat_group"] if "cellchat_group" in metadata else None
    if labels is None and "predictions" in result:
        prediction_columns = [column for column in result["predictions"] if str(column).lower() != "max"]
        if not prediction_columns:
            raise ValueError("predictions contains no label columns after excluding summary columns.")
        labels = result["predictions"].loc[:, prediction_columns].idxmax(axis=1)
    if labels is None:
        raise ValueError("metadata must contain 'cellchat_group' when no predictions are available.")
    metadata = metadata.copy()
    metadata["cellchat_group"] = pd.Categorical(labels.astype(str))
    metadata["cellchat_dataset"] = pd.Categorical(
        [result["sample"]] * len(metadata), categories=[result["sample"]], ordered=True
    )
    result["metadata"] = metadata
    return result


def merge_spatial_inputs(inputs: Sequence[Mapping[str, Any]], sample_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Merge spatial inputs on common genes and shared named modalities."""
    if not inputs:
        raise ValueError("inputs must contain at least one spatial dataset.")
    if sample_names is None:
        sample_names = [str(item.get("sample", f"sample{i + 1}")) for i, item in enumerate(inputs)]
    if len(sample_names) != len(inputs) or len(set(sample_names)) != len(sample_names):
        raise ValueError("sample_names must have one unique name per input dataset.")
    common_genes = [str(gene) for gene in inputs[0]["genes"]]
    for item in inputs[1:]:
        gene_set = set(map(str, item["genes"]))
        common_genes = [gene for gene in common_genes if gene in gene_set]
    if not common_genes:
        raise ValueError("No common genes were found across the spatial inputs.")
    expression_parts, metadata_parts, coordinate_parts = [], [], []
    spatial_factors = {"ratio": [], "tol": []}
    cells = []
    for sample_name, item in zip(sample_names, inputs, strict=True):
        original_cells = [str(cell) for cell in item["cells"]]
        prefixed_cells = [f"{sample_name}_{cell}" for cell in original_cells]
        gene_pos = pd.Index(item["genes"]).astype(str).get_indexer(common_genes)
        expression_parts.append(item["expression"][:, gene_pos].tocsc())
        metadata = _normalise_spatial_metadata(item["metadata"], original_cells, sample=str(sample_name))
        metadata.index = prefixed_cells
        coordinates = pd.DataFrame(item["coordinates"]).copy()
        coordinates.index = coordinates.index.astype(str)
        coordinates = coordinates.loc[original_cells]
        coordinates.index = prefixed_cells
        metadata_parts.append(metadata)
        coordinate_parts.append(coordinates)
        cells.extend(prefixed_cells)
        factors = item["spatial_factors"]
        spatial_factors["ratio"].extend(np.atleast_1d(factors["ratio"]).astype(float).tolist())
        spatial_factors["tol"].extend(np.atleast_1d(factors["tol"]).astype(float).tolist())
    metadata = pd.concat(metadata_parts)
    group_categories = list(pd.unique(metadata["cellchat_group"].astype(str)))
    metadata["cellchat_group"] = pd.Categorical(metadata["cellchat_group"].astype(str), categories=group_categories, ordered=True)
    metadata["cellchat_dataset"] = pd.Categorical(metadata["cellchat_dataset"].astype(str), categories=list(sample_names), ordered=True)
    modality_names = set(inputs[0].get("modalities", {}))
    for item in inputs[1:]:
        modality_names.intersection_update(item.get("modalities", {}))
    modalities = {}
    for name in sorted(modality_names):
        common_features = list(map(str, inputs[0]["modalities"][name].columns))
        for item in inputs[1:]:
            feature_set = set(map(str, item["modalities"][name].columns))
            common_features = [feature for feature in common_features if feature in feature_set]
        if not common_features:
            continue
        parts = []
        for sample_name, item in zip(sample_names, inputs, strict=True):
            original_cells = [str(cell) for cell in item["cells"]]
            frame = item["modalities"][name].copy()
            frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
            frame = frame.loc[original_cells, common_features]
            frame.index = [f"{sample_name}_{cell}" for cell in original_cells]
            parts.append(frame)
        modalities[name] = pd.concat(parts)
    return {
        "expression": sparse.vstack(expression_parts, format="csc"), "genes": common_genes,
        "cells": cells, "metadata": metadata, "coordinates": pd.concat(coordinate_parts),
        "spatial_factors": spatial_factors, "sample_names": list(sample_names), "modalities": modalities,
    }


def read_visium_spatial_info(spatial_dir, spot_ids, spot_size_um: float = 65.0) -> dict[str, Any]:
    """Read standard 10x Visium coordinates and scale factors."""
    spatial_dir = Path(spatial_dir)
    positions_path = next((spatial_dir / name for name in ("tissue_positions.csv", "tissue_positions_list.csv") if (spatial_dir / name).exists()), None)
    if positions_path is None:
        raise FileNotFoundError(f"No tissue_positions.csv or tissue_positions_list.csv in {spatial_dir}")
    scale_path = spatial_dir / "scalefactors_json.json"
    if not scale_path.exists():
        raise FileNotFoundError(scale_path)
    raw = pd.read_csv(positions_path, header=None)
    if raw.shape[1] < 6:
        raise ValueError(f"Unexpected Visium coordinate file layout: {positions_path}")
    if str(raw.iloc[0, 0]).lower() in {"barcode", "barcodes"}:
        raw = raw.iloc[1:].copy()
    raw.iloc[:, 0] = raw.iloc[:, 0].astype(str)
    raw = raw.set_index(raw.columns[0])
    spot_ids = [str(spot) for spot in spot_ids]
    missing = pd.Index(spot_ids).difference(raw.index)
    if len(missing):
        raise KeyError(f"Spatial coordinates are missing for {len(missing)} spots.")
    coordinates = raw.loc[spot_ids, [raw.columns[-1], raw.columns[-2]]].astype(float)
    coordinates.columns = ["x", "y"]
    scalefactors = json.loads(scale_path.read_text(encoding="utf-8"))
    if "spot_diameter_fullres" not in scalefactors:
        raise ValueError("Visium scale factors are missing 'spot_diameter_fullres'.")
    spot_diameter_px = float(scalefactors["spot_diameter_fullres"])
    if spot_size_um <= 0 or spot_diameter_px <= 0:
        raise ValueError("spot_size_um and spot_diameter_fullres must be positive.")
    pixel_to_um = float(spot_size_um) / spot_diameter_px
    nearest = float(pdist(coordinates.to_numpy()).min() * pixel_to_um) if len(coordinates) > 1 else np.nan
    return {"coordinates": coordinates, "scalefactors": scalefactors, "spatial_factors": {"ratio": pixel_to_um, "tol": float(spot_size_um) / 2.0}, "pixel_to_um": pixel_to_um, "spot_diameter_px": spot_diameter_px, "nearest_center_distance_um": nearest}


def update_communication_score(
    cellchat: 'CellChat',
    network_table: pd.DataFrame
) -> 'CellChat':
    """
    Update cell-cell communication scores from custom data

    Parameters
    ----------
    cellchat : CellChat
        CellChat object
    network_table : DataFrame
        Custom communication scores

    Returns
    -------
    CellChat
        Updated CellChat object
    """

    required_cols = ['source', 'target', 'ligand', 'receptor', 'score']
    if not isinstance(network_table, pd.DataFrame):
        raise TypeError("network_table must be a pandas DataFrame.")
    if not all(col in network_table.columns for col in required_cols):
        raise ValueError(f"network_table must contain columns: {required_cols}")
    if cellchat.groups is None:
        raise ValueError("cellchat.groups must be set before updating communication scores.")
    network_table = network_table.copy()

    # Add missing columns
    if 'interaction_name' not in network_table.columns:
        network_table['interaction_name'] = (
            network_table['ligand'].astype(str).str.upper()
            + "_"
            + network_table['receptor'].astype(str).str.upper()
        )

    if 'interaction_name_2' not in network_table.columns:
        network_table['interaction_name_2'] = (
            network_table['ligand'].astype(str) + " - " + network_table['receptor'].astype(str)
        )

    if 'pval' not in network_table.columns:
        network_table['pval'] = 0.0

    network_table['prob'] = pd.to_numeric(network_table['score'], errors='raise')
    network_table['pval'] = pd.to_numeric(network_table['pval'], errors='raise')

    # Create communication arrays
    lr_names = network_table['interaction_name'].astype(str).drop_duplicates().tolist()
    cell_levels = list(cellchat.groups.categories)
    n_clusters = len(cell_levels)
    group_index = {name: index for index, name in enumerate(cell_levels)}
    unknown_groups = sorted(
        set(network_table['source'].astype(str))
        .union(network_table['target'].astype(str))
        .difference(group_index)
    )
    if unknown_groups:
        raise ValueError(f"network_table contains groups absent from cellchat.groups: {unknown_groups}")

    prob_array = np.zeros((n_clusters, n_clusters, len(lr_names)))
    pval_array = np.zeros((n_clusters, n_clusters, len(lr_names)))

    for i, lr_name in enumerate(lr_names):
        lr_data = network_table[network_table['interaction_name'].astype(str) == lr_name]

        for _, row in lr_data.iterrows():
            source_idx = group_index[str(row['source'])]
            target_idx = group_index[str(row['target'])]

            prob_array[source_idx, target_idx, i] = row['prob']
            pval_array[source_idx, target_idx, i] = row['pval']

    # Update CellChat object
    cellchat.network = {
        'prob': matrix_dict_from_array(prob_array, lr_names, sparse_output=True),
        'pval': matrix_dict_from_array(pval_array, lr_names),
        'groups': cell_levels,
        'interactions': network_table.drop_duplicates('interaction_name').reset_index(drop=True),
    }

    return cellchat


def preprocess_multiomics(
    data_list: List[Union[np.ndarray, pd.DataFrame]],
    database: Dict,
    cutoff: float = 0.5,
    do_sparse: bool = True
) -> Dict[str, Any]:
    """
    Preprocess multi-omics data for CellChat analysis

    Parameters
    ----------
    data_list : list
        RNA and ADT matrices in canonical cells/spots x features orientation.
        DataFrame row labels are cell/spot identifiers and column labels are
        gene or antibody identifiers. Array-like inputs follow the same axis
        order and receive generated labels.
    database : dict
        CellChatDB database
    cutoff : float
        Cutoff for low protein expression
    do_sparse : bool
        Whether to use sparse format

    Returns
    -------
    dict
        ``expression`` in cells/spots x features orientation and the updated
        database.
    """

    if len(data_list) < 2:
        raise ValueError("data_list must contain RNA and ADT matrices")

    rna_input, adt_input = data_list[0], data_list[1]

    def _matrix_to_frame(matrix, prefix):
        if isinstance(matrix, pd.DataFrame):
            return matrix.astype(float).copy()
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=float)
        rows = [f"cell_{i}" for i in range(matrix.shape[0])]
        cols = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix, index=rows, columns=cols)

    data_rna = _matrix_to_frame(rna_input, "gene")
    data_adt = _matrix_to_frame(adt_input, "protein")
    if not data_rna.index.equals(data_adt.index):
        raise ValueError("RNA and ADT matrices must have the same cell/spot rows")

    def _divide_by_global_max(frame):
        max_value = float(np.nanmax(frame.to_numpy(dtype=float))) if frame.size else 0.0
        if not np.isfinite(max_value) or max_value <= 0:
            return frame * 0.0
        return frame / max_value

    data_rna = _divide_by_global_max(data_rna)
    data_adt = _divide_by_global_max(data_adt)

    adt_values = data_adt.to_numpy(dtype=float, copy=True)
    feature_min = np.nanmin(adt_values, axis=0, keepdims=True)
    feature_max = np.nanmax(adt_values, axis=0, keepdims=True)
    feature_range = feature_max - feature_min
    adt_scaled = np.divide(
        adt_values - feature_min,
        feature_range,
        out=np.zeros_like(adt_values, dtype=float),
        where=feature_range != 0,
    )
    adt_values[adt_scaled < cutoff] = 0
    data_adt = pd.DataFrame(adt_values, index=data_adt.index, columns=data_adt.columns)

    proteins = pd.Index(data_adt.columns.astype(str))
    gene_info = database.get('gene_info', pd.DataFrame()).copy()
    if 'AntibodyName' in gene_info.columns:
        antibody_values = gene_info['AntibodyName'].astype('string')
        gene_info_subset = gene_info[antibody_values.isin(proteins)].copy()
        mapped_symbols = gene_info_subset.set_index('AntibodyName')['Symbol'].astype(str).to_dict()
        proteins_nonmapping = set(proteins) - set(mapped_symbols)
        if proteins_nonmapping:
            warnings.warn(f"Antibodies not found in database: {sorted(proteins_nonmapping)}")
    else:
        mapped_symbols = {protein: protein for protein in proteins}
        gene_info_subset = pd.DataFrame({'Symbol': list(proteins), 'AntibodyName': list(proteins)})
        proteins_nonmapping = set()

    symbol_to_antibody = {symbol: antibody for antibody, symbol in mapped_symbols.items()}
    measured_symbols = set(data_rna.columns.astype(str)) | set(mapped_symbols.values())
    measured_antibodies = set(proteins)

    database_use = {
        key: value.copy() if hasattr(value, 'copy') else value
        for key, value in database.items()
    }
    interaction = database_use.get('interaction', pd.DataFrame()).copy()
    if not interaction.empty and {'ligand', 'receptor'}.issubset(interaction.columns):
        ligand = interaction['ligand'].astype(str)
        receptor = interaction['receptor'].astype(str)
        has_measured_protein = ligand.isin(symbol_to_antibody) | receptor.isin(symbol_to_antibody)
        counterpart_available = ligand.isin(measured_symbols) | ligand.isin(measured_antibodies)
        counterpart_available &= receptor.isin(measured_symbols) | receptor.isin(measured_antibodies)
        interaction = interaction[has_measured_protein & counterpart_available].copy()
        interaction['ligand'] = interaction['ligand'].replace(symbol_to_antibody)
        interaction['receptor'] = interaction['receptor'].replace(symbol_to_antibody)
        if 'interaction_name' in interaction.columns:
            interaction['interaction_name'] = (
                interaction['ligand'].astype(str).str.upper()
                + '_'
                + interaction['receptor'].astype(str).str.upper()
            )
        if 'interaction_name_2' in interaction.columns:
            interaction['interaction_name_2'] = (
                interaction['ligand'].astype(str) + ' - ' + interaction['receptor'].astype(str)
            )
        database_use['interaction'] = interaction

    antibody_gene_info = pd.DataFrame({'Symbol': list(proteins)})
    if 'AntibodyName' in gene_info_subset.columns:
        antibody_gene_info['AntibodyName'] = list(proteins)
    database_use['gene_info'] = pd.concat(
        [database_use.get('gene_info', pd.DataFrame()), antibody_gene_info],
        ignore_index=True,
    ).drop_duplicates(subset=['Symbol'] if 'Symbol' in antibody_gene_info.columns else None)

    data_combined = pd.concat([data_rna, data_adt], axis=1)
    if not isinstance(rna_input, pd.DataFrame) and not isinstance(adt_input, pd.DataFrame):
        data_values = data_combined.to_numpy(dtype=float)
        data_combined = sparse.csr_matrix(data_values) if do_sparse else data_values

    return {
        'expression': data_combined,
        'database': database_use,
    }


def compute_cell_distance(
    coordinates: Union[np.ndarray, pd.DataFrame],
    spatial_factors: Optional[Dict[str, float]] = None,
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    ratio: Optional[float] = None,
    tol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute cell center-to-center distances for spatial data

    Parameters
    ----------
    coordinates : array-like
        Cell coordinates
    spatial_factors : dict, optional
        Spatial factors for unit conversion

    Returns
    -------
    dict
        Distance statistics and matrices
    """

    coords = coordinates.values if isinstance(coordinates, pd.DataFrame) else coordinates
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite n x 2 numeric matrix.")
    if not np.isfinite(interaction_range) or interaction_range <= 0:
        raise ValueError("interaction_range must be a positive finite number.")
    if not np.isfinite(contact_range) or contact_range < 0:
        raise ValueError("contact_range must be a finite non-negative number.")
    factors = spatial_factors or {}
    ratio = factors.get('ratio', 1.0) if ratio is None else ratio
    tol = factors.get('tol', 5.0) if tol is None else tol
    ratio = float(np.asarray(ratio).ravel()[0]); tol = float(np.asarray(tol).ravel()[0])
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("ratio must be positive and tol must be non-negative.")

    # Apply thresholds in source-coordinate units, then convert retained
    # distances to micrometres.
    from scipy.spatial.distance import pdist, squareform
    raw_distances = squareform(pdist(coords))
    keep = (raw_distances > 0) & (raw_distances <= (interaction_range + tol) / ratio)
    distances = np.where(keep, raw_distances * ratio, 0.0)
    contact_mask = keep & (distances <= contact_range + tol)
    np.fill_diagonal(contact_mask, True)
    d_spatial = sparse.csr_matrix(distances)
    adj_contact = sparse.csr_matrix(contact_mask.astype(float))
    finite = distances[distances > 0]
    min_distance = float(np.min(finite)) if finite.size else np.nan
    mean_distance = float(np.mean(finite)) if finite.size else np.nan
    median_distance = float(np.median(finite)) if finite.size else np.nan

    return {
        'distance_matrix': distances,
        'd.spatial': d_spatial,
        'adj.contact': adj_contact,
        'adj_contact': adj_contact,
        'min_distance': min_distance,
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'suggested_tol': min_distance / 2  # Suggested tolerance
    }


def validate_cellchat(cellchat: 'CellChat') -> Dict[str, Any]:
    """
    Validate CellChat object structure and content.

    CellChat inherits from AnnData, so expression matrices are expected to use
    the AnnData-native cells x genes layout.

    Parameters
    ----------
    cellchat : CellChat
        CellChat object to validate

    Returns
    -------
    dict
        Validation results
    """

    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    if not isinstance(cellchat, CellChat):
        validation_results['errors'].append(
            f"Expected a CellChat object, got {type(cellchat).__name__}"
        )
        validation_results['is_valid'] = False
        return validation_results

    data = cellchat.X
    if data is None:
        validation_results['errors'].append("Missing expression matrix: data")
        validation_results['is_valid'] = False
    elif not hasattr(data, 'shape') or len(data.shape) != 2:
        validation_results['errors'].append("Expression matrix must be two-dimensional")
        validation_results['is_valid'] = False
    else:
        n_cells, n_genes = data.shape
        validation_results['info']['n_cells'] = n_cells
        validation_results['info']['n_genes'] = n_genes

        if n_cells == 0:
            validation_results['errors'].append("Expression matrix has zero cells")
            validation_results['is_valid'] = False
        if n_genes == 0:
            validation_results['errors'].append("Expression matrix has zero genes")
            validation_results['is_valid'] = False

        if cellchat.n_obs != n_cells:
            validation_results['errors'].append(
                "AnnData obs length doesn't match expression matrix rows"
            )
            validation_results['is_valid'] = False
        if cellchat.n_vars != n_genes:
            validation_results['errors'].append(
                "AnnData var length doesn't match expression matrix columns"
            )
            validation_results['is_valid'] = False

        metadata = cellchat.obs
        if metadata is None or len(metadata) != n_cells:
            validation_results['errors'].append(
                "Cell metadata length doesn't match number of cells"
            )
            validation_results['is_valid'] = False

        groups = cellchat.groups
        if groups is None:
            validation_results['errors'].append("Missing categorical cell groups: obs['cellchat_group']")
            validation_results['is_valid'] = False
        elif len(groups) != n_cells or not hasattr(groups, 'categories'):
            validation_results['errors'].append("obs['cellchat_group'] must be categorical and aligned to cells")
            validation_results['is_valid'] = False
        else:
            validation_results['info']['groups'] = list(groups.categories)

        for context, value in (
            ('network', cellchat.network),
            ('pathway_network', cellchat.pathway_network),
        ):
            try:
                normalize_network_slot(value, context)
            except (TypeError, ValueError) as exc:
                validation_results['errors'].append(str(exc))
        try:
            normalize_network_similarity(cellchat.network_similarity)
        except (TypeError, ValueError) as exc:
            validation_results['errors'].append(str(exc))

        if cellchat.settings.get('mode') == 'merged':
            datasets = cellchat.obs.get('cellchat_dataset')
            if datasets is None or not pd.api.types.is_categorical_dtype(datasets):
                validation_results['errors'].append("Merged objects require categorical obs['cellchat_dataset']")
                validation_results['is_valid'] = False

    opts = cellchat.settings
    if not isinstance(opts, dict):
        validation_results['errors'].append("settings must be a dictionary")
        validation_results['is_valid'] = False
    else:
        required_options = ['mode', 'datatype']
        for opt in required_options:
            if opt not in opts:
                validation_results['warnings'].append(f"Missing option: {opt}")

        mode = opts.get('mode')
        if mode is not None and mode not in {'single', 'merged'}:
            validation_results['warnings'].append(
                f"Unexpected analysis mode: {mode}"
            )

        datatype = opts.get('datatype')
        if datatype is not None and str(datatype).upper() not in {'RNA', 'SPATIAL'}:
            validation_results['warnings'].append(
                f"Unexpected datatype: {datatype}"
            )

    return validation_results


