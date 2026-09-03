"""Canonical dictionary storage helpers for communication networks."""

from collections.abc import Mapping

import numpy as np
from scipy import sparse


_NETWORK_FIELDS = {
    "groups",
    "prob",
    "pval",
    "interactions",
    "count",
    "weight",
    "centrality",
    "similarity",
    "pattern",
    "pairwise_rank",
}
_REQUIRED_NETWORK_FIELDS = {"groups", "prob", "pval"}
_NETWORK_SLOT_NAMES = {"network", "pathway_network"}
_SIMILARITY_TYPES = {"functional", "structural"}

_SPOT_NETWORK_FIELDS = {
    "spots",
    "prob",
    "interactions",
    "parameters",
    "distance",
    "spatial_weight",
    "contact_adjacency",
    "ligand_expression",
    "receptor_expression",
    "centrality",
}
_REQUIRED_SPOT_NETWORK_FIELDS = {"spots", "prob", "interactions", "parameters"}


def _unique_names(names):
    result = []
    counts = {}
    for index, name in enumerate(names):
        base = str(name) if name is not None else f"network_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        result.append(base if count == 1 else f"{base}__{count}")
    return result


def is_matrix_dict(value):
    """Whether value maps names to two-dimensional matrices."""
    if not isinstance(value, Mapping):
        return False
    return all(
        sparse.issparse(matrix) or np.asarray(matrix).ndim == 2
        for matrix in value.values()
    )


def network_names(net, key="prob"):
    """Return the ordered communication names stored by one network."""
    value = net.get(key) if isinstance(net, Mapping) else None
    if isinstance(value, Mapping):
        return list(value.keys())
    return []


def matrix_dict_from_array(array, names=None, sparse_output=False):
    """Create name-to-matrix storage from a C x C x N calculation array."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError("Network arrays must have shape C x C x N")
    if names is None:
        names = [f"network_{index}" for index in range(array.shape[2])]
    if len(names) != array.shape[2]:
        raise ValueError("Number of names must match the array's third axis")
    names = _unique_names(names)
    if sparse_output:
        return {name: sparse.csr_matrix(array[:, :, index])
                for index, name in enumerate(names)}
    return {name: np.array(array[:, :, index], copy=True)
            for index, name in enumerate(names)}


def stack_network_field(net, field, names=None, fill_value=0.0):
    """Stack one canonical matrix-dictionary field for an in-memory calculation."""
    if not isinstance(net, Mapping):
        raise TypeError("Network must be a dictionary")
    values = net.get(field)
    names = network_names(net) if names is None else list(names)
    groups = list(net.get("groups", []))
    shape = (len(groups), len(groups))
    if values is None:
        return np.full((*shape, len(names)), fill_value, dtype=float)
    if not isinstance(values, Mapping):
        raise TypeError(f"network[{field!r}] must be a name-to-matrix dictionary")
    result = np.full((*shape, len(names)), fill_value, dtype=float)
    for index, name in enumerate(names):
        if name not in values:
            continue
        matrix = values[name]
        array = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        if array.shape != shape:
            raise ValueError(
                f"network[{field!r}][{name!r}] has shape {array.shape}; expected {shape}"
            )
        result[:, :, index] = array
    return result


def _csr_dict(value, field, context):
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{context}[{field!r}] must be a dictionary of name -> CSR matrix; "
            f"received {type(value).__name__}"
        )
    result = {}
    for name, matrix in value.items():
        if not sparse.isspmatrix_csr(matrix):
            raise TypeError(f"{context}[{field!r}][{name!r}] must be a CSR matrix")
        if matrix.ndim != 2:
            raise TypeError(f"{context}[{field!r}][{name!r}] must be two-dimensional")
        result[str(name)] = matrix.copy()
    return result


def _dense_dict(value, field, context):
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{context}[{field!r}] must be a dictionary of name -> dense matrix; "
            f"received {type(value).__name__}"
        )
    result = {}
    for name, matrix in value.items():
        if sparse.issparse(matrix):
            raise TypeError(f"{context}[{field!r}][{name!r}] must be a dense matrix")
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise TypeError(f"{context}[{field!r}][{name!r}] must be two-dimensional")
        result[str(name)] = np.array(array, copy=True)
    return result


def normalize_network(net, context="network"):
    """Validate and copy an aligned communication-network dictionary."""
    if not isinstance(net, Mapping):
        raise TypeError(f"{context} must be a dictionary")
    if not net:
        return {}

    unknown = set(net).difference(_NETWORK_FIELDS)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {sorted(unknown)}")
    missing = _REQUIRED_NETWORK_FIELDS.difference(net)
    if missing:
        raise ValueError(
            f"{context} must contain groups, prob, and pval; missing {sorted(missing)}"
        )

    result = dict(net)
    result["prob"] = _csr_dict(result["prob"], "prob", context)
    names = list(result["prob"])
    pval_dict = _dense_dict(result["pval"], "pval", context)
    if set(pval_dict) != set(names):
        raise ValueError(f"{context}['prob'] and {context}['pval'] names must match exactly")

    groups = list(result["groups"])
    if len(set(map(str, groups))) != len(groups):
        raise ValueError(f"{context}['groups'] must contain unique names")
    expected_shape = (len(groups), len(groups))

    for field in ("count", "weight"):
        if field not in result:
            continue
        matrix = np.asarray(result[field])
        if matrix.ndim != 2 or matrix.shape != expected_shape:
            raise ValueError(
                f"{context}[{field!r}] has shape {matrix.shape}; "
                f"expected {expected_shape}"
            )
        result[field] = np.array(matrix, copy=True)

    for name, matrix in result["prob"].items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{context}['prob'][{name!r}] has shape {matrix.shape}; "
                f"expected {expected_shape}"
            )

    result["pval"] = {}
    for name, matrix in result["prob"].items():
        pval_matrix = pval_dict[name]
        if pval_matrix.shape != expected_shape:
            raise ValueError(
                f"{context}['pval'][{name!r}] has shape {pval_matrix.shape}; "
                f"expected {expected_shape}"
            )
        result["pval"][name] = pval_matrix
    result["groups"] = groups
    return result


def normalize_network_slot(value, context):
    """Validate one canonical network or a named collection of such networks."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a dictionary")
    if not value:
        return {}

    single_markers = _NETWORK_FIELDS.intersection(value)
    if single_markers:
        return normalize_network(value, context)

    result = {}
    for key, item in value.items():
        if not isinstance(item, Mapping):
            raise TypeError(f"{context}[{key!r}] must be a network dictionary")
        if not item:
            result[str(key)] = {}
            continue
        result[str(key)] = normalize_network(item, f"{context}[{key!r}]")
    return result


def normalize_network_similarity(value, context="network_similarity"):
    """Validate object-level results produced from merged network comparisons."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a dictionary")
    unknown_slots = set(value).difference(_NETWORK_SLOT_NAMES)
    if unknown_slots:
        raise ValueError(f"{context} contains unknown slots: {sorted(unknown_slots)}")
    result = {}
    for slot_name, analyses in value.items():
        if not isinstance(analyses, Mapping):
            raise TypeError(f"{context}[{slot_name!r}] must be a dictionary")
        unknown_types = set(analyses).difference(_SIMILARITY_TYPES)
        if unknown_types:
            raise ValueError(
                f"{context}[{slot_name!r}] contains unknown similarity types: "
                f"{sorted(unknown_types)}"
            )
        result[slot_name] = {}
        for similarity_type, analysis in analyses.items():
            if not isinstance(analysis, Mapping):
                raise TypeError(
                    f"{context}[{slot_name!r}][{similarity_type!r}] must be a dictionary"
                )
            result[slot_name][similarity_type] = dict(analysis)
    return result


def normalize_spot_network(value, context="spot_network"):
    """Validate one spot-level sparse communication-network dictionary.

    Spot networks are deliberately separate from group networks. Every entry
    in ``prob`` is an aligned spot-by-spot CSR matrix, while group networks
    additionally carry dense p-value matrices and group-by-group summaries.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a dictionary")
    if not value:
        return {}

    unknown = set(value).difference(_SPOT_NETWORK_FIELDS)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {sorted(unknown)}")
    missing = _REQUIRED_SPOT_NETWORK_FIELDS.difference(value)
    if missing:
        raise ValueError(f"{context} is missing required fields: {sorted(missing)}")

    spots = [str(spot) for spot in value["spots"]]
    if len(set(spots)) != len(spots):
        raise ValueError(f"{context}['spots'] must contain unique names")
    probabilities = _csr_dict(value["prob"], "prob", context)
    expected_shape = (len(spots), len(spots))
    for name, matrix in probabilities.items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{context}['prob'][{name!r}] has shape {matrix.shape}; "
                f"expected {expected_shape}"
            )
        if matrix.nnz and (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)):
            raise ValueError(f"{context}['prob'][{name!r}] must be finite and non-negative")

    interactions = value["interactions"]
    if not hasattr(interactions, "columns") or "interaction_name" not in interactions.columns:
        raise TypeError(f"{context}['interactions'] must be a DataFrame with interaction_name")
    interaction_names = interactions["interaction_name"].astype(str).tolist()
    if interaction_names != list(probabilities):
        raise ValueError(
            f"{context} interaction rows and probability names must have the same order"
        )
    if not isinstance(value["parameters"], Mapping):
        raise TypeError(f"{context}['parameters'] must be a dictionary")

    result = dict(value)
    result["spots"] = spots
    result["prob"] = probabilities
    result["interactions"] = interactions.copy()
    result["parameters"] = dict(value["parameters"])

    for field in ("distance", "spatial_weight", "contact_adjacency"):
        if field not in value:
            continue
        matrix = value[field]
        if not sparse.isspmatrix_csr(matrix) or matrix.shape != expected_shape:
            raise TypeError(f"{context}['{field}'] must be a spot-by-spot CSR matrix")
        result[field] = matrix.copy()

    for field in ("ligand_expression", "receptor_expression"):
        if field not in value:
            continue
        array = np.asarray(value[field])
        if array.shape != (len(probabilities), len(spots)):
            raise ValueError(
                f"{context}['{field}'] must have shape "
                f"({len(probabilities)}, {len(spots)})"
            )
        result[field] = np.array(array, copy=True)

    if "centrality" in value:
        centrality = value["centrality"]
        if not isinstance(centrality, Mapping):
            raise TypeError(f"{context}['centrality'] must be a dictionary")
        expected_measures = {
            "outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg"
        }
        unknown_measures = set(centrality).difference(expected_measures)
        if unknown_measures:
            raise ValueError(
                f"{context}['centrality'] contains unknown measures: "
                f"{sorted(unknown_measures)}"
            )
        result["centrality"] = {}
        expected_index = np.asarray(spots, dtype=str)
        expected_columns = np.asarray(list(probabilities), dtype=str)
        for measure, table in centrality.items():
            if not hasattr(table, "index") or not hasattr(table, "columns"):
                raise TypeError(
                    f"{context}['centrality'][{measure!r}] must be a pandas DataFrame"
                )
            if not np.array_equal(np.asarray(table.index, dtype=str), expected_index):
                raise ValueError(
                    f"{context}['centrality'][{measure!r}] rows must match spots"
                )
            if not np.array_equal(np.asarray(table.columns, dtype=str), expected_columns):
                raise ValueError(
                    f"{context}['centrality'][{measure!r}] columns must match prob names"
                )
            table_values = table.to_numpy(dtype=float)
            if not np.isfinite(table_values).all() or np.any(table_values < 0):
                raise ValueError(
                    f"{context}['centrality'][{measure!r}] must be finite and non-negative"
                )
            result["centrality"][measure] = table.copy()
    return result


def zero_group_axes(matrix_dict, group_indices):
    """Set selected source and target rows/columns to zero in every matrix."""
    if not group_indices:
        return dict(matrix_dict)
    result = {}
    for name, matrix in matrix_dict.items():
        if sparse.issparse(matrix):
            edited = matrix.tolil(copy=True)
            edited[group_indices, :] = 0.0
            edited[:, group_indices] = 0.0
            result[name] = edited.tocsr()
        else:
            edited = np.array(matrix, copy=True)
            edited[group_indices, :] = 0.0
            edited[:, group_indices] = 0.0
            result[name] = edited
    return result


def reorder_group_axes(matrix_dict, order_indices):
    """Reorder source and target rows/columns in every communication matrix."""
    order_indices = np.asarray(order_indices, dtype=int)
    result = {}
    for name, matrix in matrix_dict.items():
        if sparse.issparse(matrix):
            result[name] = matrix.tocsr()[order_indices, :][:, order_indices].tocsr()
        else:
            array = np.asarray(matrix)
            result[name] = np.array(array[np.ix_(order_indices, order_indices)], copy=True)
    return result


def expand_group_axes(matrix_dict, source_to_target, n_target):
    """Expand every sparse communication matrix to a new group order."""
    source_to_target = np.asarray(source_to_target, dtype=int)
    result = {}
    for name, matrix in matrix_dict.items():
        source = matrix.tocoo()
        expanded = sparse.coo_matrix(
            (
                source.data,
                (source_to_target[source.row], source_to_target[source.col]),
            ),
            shape=(n_target, n_target),
        ).tocsr()
        expanded.eliminate_zeros()
        result[name] = expanded
    return result


def expand_dense_group_axes(
    matrix_dict, source_to_target, n_target, fill_value=0.0
):
    """Expand every dense communication matrix using a field-specific fill value."""
    source_to_target = np.asarray(source_to_target, dtype=int)
    result = {}
    for name, matrix in matrix_dict.items():
        source = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        expanded = np.full((n_target, n_target), fill_value, dtype=source.dtype)
        expanded[np.ix_(source_to_target, source_to_target)] = source
        result[name] = expanded
    return result


