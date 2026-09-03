"""Small access helpers for CellChatPy's canonical network storage."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .network_storage import network_names


_slot_names = {"network", "pathway_network"}


def _validate_slot_name(slot_name: str) -> None:
    if slot_name not in _slot_names:
        raise ValueError("slot_name must be 'network' or 'pathway_network'.")


def _get_cluster_names_from_cellchat(cellchat: "CellChat") -> List[str]:
    """Return the canonical categorical cell-group order."""
    groups = cellchat.groups
    return list(groups.categories) if isinstance(groups, pd.Categorical) else []


def _network_group_names(cellchat: "CellChat", network: Dict[str, Any]) -> List[str]:
    """Return the group order that indexes one stored network.

    Network matrices are allowed to use an order different from the metadata
    categorical levels.  The stored network axis is therefore authoritative;
    the metadata order is only a fallback for objects created before explicit
    network group names were required.
    """
    groups = network.get("groups")
    if groups is None:
        groups = _get_cluster_names_from_cellchat(cellchat)
    result = [str(group) for group in groups]
    if not result or len(set(result)) != len(result):
        raise ValueError("Each network must define a non-empty unique groups order.")
    return result


def _get_aggregated_network(cellchat: "CellChat", measure: str = "weight") -> Tuple[Optional[np.ndarray], List[str]]:
    """Return one stored aggregate matrix from the L-R network."""
    if measure not in {"count", "weight"}:
        raise ValueError("measure must be 'count' or 'weight'.")
    network = cellchat.network
    groups = list(network.get("groups", _get_cluster_names_from_cellchat(cellchat)))
    value = network.get(measure)
    if value is None:
        return None, groups
    matrix = value.to_numpy() if isinstance(value, pd.DataFrame) else np.asarray(value)
    if matrix.ndim != 2 or matrix.shape != (len(groups), len(groups)):
        raise ValueError(f"network[{measure!r}] must have shape ({len(groups)}, {len(groups)}).")
    return matrix, groups


def _validate_single_network(network: Dict[str, Any], slot_name: str) -> Dict[str, Any]:
    """Validate and return one canonical network without changing its structure."""
    _validate_slot_name(slot_name)
    if not isinstance(network, dict) or "prob" not in network:
        raise ValueError(f"cellchat.{slot_name} has no probability network.")
    return network


def _network_dataset_items(cellchat: "CellChat", slot_name: str = "network") -> List[Tuple[str, Dict[str, Any]]]:
    """Return a single network or the named child networks of a merge."""
    _validate_slot_name(slot_name)
    raw = cellchat.pathway_network if slot_name == "pathway_network" else cellchat.network
    if not isinstance(raw, dict):
        return []
    if "prob" in raw:
        return [("dataset", _validate_single_network(raw, slot_name))]
    return [
        (str(name), _validate_single_network(value, slot_name))
        for name, value in raw.items()
        if isinstance(value, dict) and "prob" in value
    ]


def _comparison_networks(cellchat: "CellChat", slot_name: str, comparison: Optional[Tuple[int, int]] = None) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[Tuple[int, int]]]:
    """Resolve a one- or two-dataset plotting view without level fallback."""
    items = _network_dataset_items(cellchat, slot_name)
    if len(items) <= 1:
        if comparison is not None:
            raise ValueError("comparison requires a merged object with at least two datasets.")
        return items, None
    if comparison is None:
        if len(items) != 2:
            raise ValueError("comparison is required when more than two datasets are present.")
        comparison = (0, 1)
    if len(comparison) != 2 or any(not isinstance(index, (int, np.integer)) for index in comparison):
        raise ValueError("comparison must contain two zero-based dataset indices.")
    first, second = map(int, comparison)
    if first == second or min(first, second) < 0 or max(first, second) >= len(items):
        raise ValueError("comparison must contain two distinct valid dataset indices.")
    return [items[first], items[second]], (first, second)


def _get_pathway_names(cellchat: "CellChat") -> List[str]:
    """Return pathway names from the canonical pathway network."""
    network = cellchat.pathway_network
    if not isinstance(network, dict) or "prob" not in network:
        return []
    return list(network["prob"])


def _get_centrality_data(cellchat: "CellChat", pathway_name: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
    """Read long-format centrality records and align them to network groups."""
    data = cellchat.uns.get("cellchat_centrality")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None
    required = {"group", "pathway_name", "measure", "value"}
    if not required.issubset(data.columns):
        raise ValueError("cellchat_centrality must contain group, pathway_name, measure, and value columns.")
    if pathway_name is not None:
        data = data[data["pathway_name"].astype(str) == str(pathway_name)]
    if data.empty:
        return None
    groups = list(cellchat.pathway_network.get("groups", _get_cluster_names_from_cellchat(cellchat)))
    result: Dict[str, np.ndarray] = {}
    for measure, frame in data.groupby("measure", sort=False):
        values = frame.drop_duplicates("group").set_index("group")["value"]
        result[str(measure)] = values.reindex(groups, fill_value=0.0).to_numpy(dtype=float)
    return result or None


def _network_view_for_visualization(cellchat: "CellChat", slot_name: str = "network") -> Dict[str, Any]:
    """Return one canonical network or a merged collection of canonical networks."""
    _validate_slot_name(slot_name)
    items = _network_dataset_items(cellchat, slot_name)
    if len(items) == 1:
        return items[0][1]
    if len(items) > 1:
        return {"datasets": [name for name, _ in items], "networks": [network for _, network in items]}
    raise ValueError(f"cellchat.{slot_name} has no probability network.")
