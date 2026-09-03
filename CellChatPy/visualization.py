#!/usr/bin/env python3
"""
Visualization functions for CellChatPy
Mirrors R CellChat visualization.R functions.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections.abc import Mapping
from matplotlib import colors as mcolors
from typing import Union, Optional, Dict, List, Any, Tuple
import warnings
import networkx as nx
from scipy import sparse

# Canonical network views used only while plotting.
from .viz_helpers import (
    _network_view_for_visualization,
    _get_cluster_names_from_cellchat,
    _network_group_names,
    _get_aggregated_network,
    _get_pathway_names,
    _get_centrality_data,
    _comparison_networks,
)
from .network_storage import network_names, stack_network_field


def _network_arrays(network: Dict[str, Any]):
    """Read the canonical matrix dictionaries as temporary calculation arrays."""
    names = network_names(network)
    if not names:
        raise ValueError("Network probability data is empty")
    pval_values = network.get("pval")
    if not isinstance(pval_values, dict) or list(pval_values) != names:
        raise ValueError("Network probability and p-value names must be aligned")
    prob = stack_network_field(network, "prob", names=names, fill_value=0.0)
    pval = stack_network_field(network, "pval", names=names)
    if pval.shape != prob.shape:
        raise ValueError("Network probability and p-value matrices must be aligned")
    return names, prob, pval


def _network_matrix(network: Dict[str, Any], field: str, name: str) -> np.ndarray:
    """Return one named canonical network matrix as a dense calculation array."""
    values = network.get(field)
    if not isinstance(values, dict) or name not in values:
        raise ValueError(f"Network field {field!r} has no entry {name!r}")
    matrix = values[name]
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)


def _network_strengths(network: Dict[str, Any], names: List[str]) -> np.ndarray:
    """Return total probability for each requested communication name."""
    values = network.get("prob", {})
    totals = []
    for name in names:
        matrix = values.get(name)
        if matrix is None:
            totals.append(0.0)
        elif sparse.issparse(matrix):
            totals.append(float(matrix.sum()))
        else:
            totals.append(float(np.asarray(matrix, dtype=float).sum()))
    return np.asarray(totals, dtype=float)


def _pairwise_similarity_data(cellchat: 'CellChat', slot_name: str, emb_type: str) -> Dict[str, Any]:
    """Return object-level similarity results for a merged-network comparison."""
    slot = cellchat.network_similarity.get(slot_name, {})
    if not isinstance(slot, dict):
        raise TypeError(f"network_similarity[{slot_name!r}] must be a dictionary")
    result = slot.get(emb_type, {})
    if not isinstance(result, dict):
        raise TypeError(
            f"network_similarity[{slot_name!r}][{emb_type!r}] must be a dictionary"
        )
    return result


# ---------------------------------------------------------------------------
# Color palette helpers - mirrors scPalette() and ggPalette() in R
# ---------------------------------------------------------------------------

def _smart_round(value: float) -> float:
    """Round a value to 1-2 significant figures for axis tick labels."""
    if value == 0:
        return 0.0
    import math
    mag = math.floor(math.log10(abs(value)))
    scale = 10 ** (mag - 1)
    rounded = round(value / scale) * scale
    # If result ends in .0, return as float with 1 decimal
    return float(f"{rounded:.6g}")


def _ordered_levels(values) -> List[Any]:
    """Return factor/category order when available; otherwise first-seen order."""
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().tolist())
        return [v for v in values.cat.categories.tolist() if v in present]
    ser = pd.Series(values).dropna()
    return pd.unique(ser).tolist()


sc_palette_colors = [
    '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#F29403', '#F781BF',
    '#BC9DCC', '#A65628', '#54B0E4', '#222F75', '#1B9E77', '#B2DF8A',
    '#E3BE00', '#FB9A99', '#E7298A', '#910241', '#00CDD1', '#A6CEE3',
    '#CE1261', '#5E4FA2', '#8CA77B', '#00441B', '#DEDC00', '#DCF0B9',
    '#8DD3C7', '#999999'
]


def sc_palette(n: int) -> List[str]:
    """Generate n colors from CellChat scPalette (mirrors R scPalette)."""
    if n <= len(sc_palette_colors):
        return sc_palette_colors[:n]
    from matplotlib.colors import to_hex
    from matplotlib.cm import get_cmap
    cmap = LinearSegmentedColormap.from_list("sc", sc_palette_colors, N=n)
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def gg_palette(n: int) -> List[str]:
    """Generate n HCL-spaced colors (mirrors R ggPalette)."""
    import colorsys
    from matplotlib.colors import to_hex
    hues = np.linspace(15, 375, n + 1)[:n]
    colors = []
    for h in hues:
        # HCL -> approximate via HSL: l=0.65, c~0.78 in HCL maps to s~0.9 HSL
        r, g, b = colorsys.hls_to_rgb(h / 360.0, 0.65, 0.78)
        colors.append(to_hex((np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1))))
    return colors


def _resolve_colors(n: int, color_use: Optional[List[str]] = None) -> List[str]:
    """Return exactly ``n`` colors, cycling a supplied palette if necessary."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if color_use is not None:
        colors = list(color_use)
        if not colors:
            raise ValueError("color_use must contain at least one color")
        return [colors[i % len(colors)] for i in range(n)]
    return sc_palette(n)


def _validate_dot_size(dot_size: Tuple[float, float]) -> Tuple[float, float]:
    """Validate the R ``dot.size`` equivalent used by embedding plots."""
    try:
        values = tuple(float(value) for value in dot_size)
    except (TypeError, ValueError):
        raise ValueError("dot_size must contain two finite non-negative values")
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("dot_size must contain exactly two finite values")
    if values[0] < 0 or values[1] < 0 or values[0] > values[1]:
        raise ValueError("dot_size must satisfy 0 <= dot_size[0] <= dot_size[1]")
    return values


def _expression_feature_dataframe(
    cellchat: 'CellChat',
    slot_data: str,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a genes-by-cells view containing only requested expression features."""
    layer_names = {"raw", "signaling", "scaled", "smoothed"}
    normalized_slot = str(slot_data).lower()
    if normalized_slot == "x":
        data_mat = cellchat.X
    elif normalized_slot in layer_names:
        data_mat = getattr(cellchat, normalized_slot)
    else:
        raise ValueError(
            "slot_data must be one of 'x', 'raw', 'signaling', 'scaled', or 'smoothed'."
        )
    if data_mat is None:
        raise ValueError(f"Expression layer '{normalized_slot}' is not available.")
    if not hasattr(data_mat, "shape") or len(data_mat.shape) != 2 or data_mat.shape != cellchat.shape:
        raise ValueError(
            f"Expression layer '{normalized_slot}' must have cells x genes shape "
            f"{cellchat.shape}; got {getattr(data_mat, 'shape', None)}."
        )

    gene_names = pd.Index(cellchat.var_names.astype(str))
    available = np.ones(cellchat.n_vars, dtype=bool)
    if normalized_slot == "signaling" and "is_signaling" in cellchat.var:
        available = cellchat.var["is_signaling"].fillna(False).to_numpy(dtype=bool)

    if features is None:
        positions = np.flatnonzero(available)
    else:
        requested = list(dict.fromkeys(str(feature) for feature in features))
        if not gene_names.is_unique:
            raise ValueError("Expression feature names must be unique.")
        position_by_name = {
            name: position for position, name in enumerate(gene_names) if available[position]
        }
        positions = np.asarray(
            [position_by_name[name] for name in requested if name in position_by_name],
            dtype=int,
        )

    selected = data_mat[:, positions]
    array = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    return pd.DataFrame(
        array.T,
        index=gene_names[positions],
        columns=pd.Index(cellchat.obs_names.astype(str)),
        dtype=float,
    )


def _resolve_expression_groups(
    cellchat: 'CellChat', cell_names: List[str], group_by: Optional[str]
) -> Tuple[pd.Series, List[Any]]:
    """Resolve expression plot groups and align them to ``cell_names``."""
    if group_by is not None:
        if group_by not in cellchat.obs.columns:
            raise ValueError(f"Metadata column '{group_by}' was not found.")
        values = cellchat.obs[group_by]
    else:
        values = cellchat.groups
        if values is None:
            raise ValueError("cellchat.groups is None.")

    if isinstance(values, pd.Series):
        series = values.copy()
        if len(series) == len(cell_names) and not series.index.equals(pd.Index(cell_names)):
            series.index = cell_names
        else:
            series = series.reindex(cell_names)
    elif isinstance(values, pd.Categorical):
        series = pd.Series(values, index=cell_names)
    else:
        values = list(values)
        if len(values) != len(cell_names):
            raise ValueError("Group labels and expression cells have different lengths.")
        series = pd.Series(values, index=cell_names)

    if series.isna().all():
        raise ValueError("No cell has a valid expression group label.")
    levels = _ordered_levels(series)
    if not levels:
        raise ValueError("No expression groups are available for plotting.")
    return series, levels


def _spatial_coordinates(cellchat: 'CellChat') -> pd.DataFrame:
    """Return canonical spatial coordinates aligned to ``obs_names``."""
    coordinates = cellchat.spatial.get("coordinates")
    if coordinates is None:
        raise ValueError("No spatial coordinates. Set cellchat.spatial['coordinates'].")
    frame = pd.DataFrame(coordinates).copy()
    frame.index = frame.index.astype(str)
    cells = pd.Index(cellchat.obs_names.astype(str))
    if not cells.isin(frame.index).all():
        raise ValueError("Spatial coordinates must contain every cell in cellchat.obs_names.")
    frame = frame.loc[cells]
    if frame.shape[1] < 2:
        raise ValueError("Spatial coordinates must contain at least two columns.")
    values = frame.iloc[:, :2].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Spatial coordinates must contain finite numeric values.")
    return frame


def _validate_expression_plot_inputs(
    features: List[str], ncol: Optional[int] = None,
    dot_scale: Optional[float] = None, pt_size: Optional[float] = None,
) -> List[str]:
    """Validate common feature-plot arguments."""
    features = list(features) if features is not None else []
    if not features:
        raise ValueError("features must contain at least one feature.")
    if ncol is not None and ncol <= 0:
        raise ValueError("ncol must be a positive integer.")
    if dot_scale is not None and dot_scale < 0:
        raise ValueError("dot_scale must be non-negative.")
    if pt_size is not None and pt_size < 0:
        raise ValueError("pt_size must be non-negative.")
    return features


def _pairwise_probability_norm(
    cellchat: 'CellChat', slot_name: str, comparison: Optional[Tuple[int, int]],
    pathway_names: List[str], point_counts: List[int]
) -> List[np.ndarray]:
    """Return per-dataset probability scales for canonical pairwise points."""
    try:
        networks, _ = _comparison_networks(cellchat, slot_name, comparison)
    except ValueError:
        networks = []
    result = []
    for di, point_count in enumerate(point_counts):
        network_name, network = networks[di] if di < len(networks) else ('', {})
        prob = network.get('prob') if isinstance(network, dict) else None
        names = network_names(network)
        names_for_dataset = [name for name in pathway_names
                             if str(name).endswith(f'--{network_name}')]
        if not names_for_dataset:
            result.append(np.ones(point_count))
            continue
        if not isinstance(prob, dict):
            result.append(np.ones(point_count))
            continue
        sums = {
            name: float(matrix.sum()) if sparse.issparse(matrix)
            else float(np.asarray(matrix, dtype=float).sum())
            for name, matrix in prob.items()
        }
        selected = []
        for name in names_for_dataset:
            if not str(name).endswith(f'--{network_name}'):
                continue
            base = str(name).rsplit('--', 1)[0]
            selected.append(sums.get(base, 0.0))
        selected = np.asarray(selected, dtype=float)
        normalized = selected / selected.max() if selected.size and selected.max() > 0 else np.ones(len(selected))
        result.append(normalized if len(normalized) == point_count else np.ones(point_count))
    return result


def _format_edge_label(value: float) -> str:
    """Format an edge weight like R's round(..., digits = 1)."""
    return f"{round(float(value), 1):g}"


def _curved_edge_label_position(
    start: Tuple[float, float], end: Tuple[float, float], curvature: float,
) -> Tuple[float, float]:
    """Return a readable label position near the midpoint of a curved edge."""
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    distance = float(np.hypot(dx, dy))
    if distance == 0:
        return x0, y0
    offset = float(curvature) * distance * 0.55
    return (
        (x0 + x1) / 2.0 - dy / distance * offset,
        (y0 + y1) / 2.0 + dx / distance * offset,
    )


def _draw_circular_self_loop(
    ax: plt.Axes,
    position: Tuple[float, float],
    angle: float,
    *,
    color,
    linewidth: float,
    alpha: float,
    mutation_scale: float,
) -> Tuple[FancyArrowPatch, Tuple[float, float]]:
    """Draw a non-degenerate directed self-loop outside a circular node."""
    x, y = position
    radial = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    tangent = np.array([-radial[1], radial[0]], dtype=float)
    node = np.array([x, y], dtype=float)

    start = node + 0.035 * radial + 0.045 * tangent
    control_1 = node + 0.10 * radial + 0.16 * tangent
    control_2 = node + 0.27 * radial + 0.15 * tangent
    outer = node + 0.29 * radial
    control_3 = node + 0.27 * radial - 0.15 * tangent
    control_4 = node + 0.10 * radial - 0.16 * tangent
    end = node + 0.035 * radial - 0.045 * tangent
    vertices = np.vstack([
        start, control_1, control_2, outer,
        control_3, control_4, end,
    ])
    path = Path(
        vertices,
        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
         Path.CURVE4, Path.CURVE4, Path.CURVE4],
    )
    loop = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    loop.set_gid("network-self-loop")
    ax.add_patch(loop)
    label_position = tuple(node + 0.33 * radial)
    return loop, label_position


# ---------------------------------------------------------------------------
# 1.  plot_network_circle  - circular network plot
# ---------------------------------------------------------------------------

def plot_network_circle(
    cellchat: 'CellChat',
    slot_name: str = "network",
    net_matrix: Optional[np.ndarray] = None,
    group_names: Optional[List[str]] = None,
    signaling: Optional[str] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    color_use: Optional[List[str]] = None,
    top: float = 1.0,
    weight_scale: bool = True,
    vertex_weight: Optional[Union[float, List[float]]] = None,
    vertex_size_max: float = 15.0,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 4.0,
    alpha_edge: float = 0.6,
    label_edge: bool = False,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.8,
    edge_curved: float = 0.2,
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    margin: float = 0.2,
    thresh: float = 0.05,
    remove_isolate: bool = False,
    title_name: Optional[str] = None,
    vertex_label_cex: float = 1.0,
    vertex_label_color: str = "black",
    fig_size: Tuple[int, int] = (7, 7),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Circular network plot for cell-cell communication.
    Mirrors R plot_network_circle().

    Labels are placed outside nodes (clockwise), node size scales with
    vertex_weight, edge width proportional to weight, edge colour = sender colour.
    Set ``label_edge=True`` to show edge weights rounded to one decimal place.
    By default this aggregates the L-R-level ``cellchat.network`` slot. Pass
    ``slot_name="pathway_network"`` to aggregate the pathway-level slot instead.
    """
    if group_names is None and net_matrix is None:
        network_for_groups = _network_view_for_visualization(cellchat, slot_name)
        group_names = _network_group_names(cellchat, network_for_groups)
    cluster_names = (
        list(group_names)
        if group_names is not None
        else _get_cluster_names_from_cellchat(cellchat)
    )
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    # --- extract weight matrix ---
    if net_matrix is not None:
        net_mat = np.array(net_matrix, dtype=float).copy()
    else:
        # Materialize a temporary view of canonical matrix-dictionary storage.
        net_data = _network_view_for_visualization(cellchat, slot_name)

        if 'prob' not in net_data:
            raise ValueError("No probability data found")

        if signaling is not None and slot_name == "pathway_network":
            if signaling not in net_data['prob']:
                raise ValueError(f"Signaling pathway '{signaling}' not found")
            net_mat = _network_matrix(net_data, 'prob', signaling).copy()
            pval_matrix = _network_matrix(net_data, 'pval', signaling)
            net_mat[pval_matrix >= thresh] = 0.0
        elif 'weight' in net_data:
            net_mat = np.asarray(net_data['weight'], dtype=float).copy()
        else:
            _, prob_array, pval_array = _network_arrays(net_data)
            prob_array[pval_array >= thresh] = 0.0
            net_mat = np.sum(prob_array, axis=2)

    if net_mat.ndim != 2 or net_mat.shape != (n_clusters, n_clusters):
        raise ValueError(
            "net_matrix must be a square matrix whose dimensions match group_names."
        )

    # top fraction filter
    if top < 1.0 and (net_mat > 0).any():
        thresh_val = np.quantile(net_mat[net_mat > 0], 1 - top)
        net_mat[net_mat < thresh_val] = 0

    if sources_use is not None:
        mask_r = [i for i, n in enumerate(cluster_names) if n not in sources_use]
        net_mat[mask_r, :] = 0
    if targets_use is not None:
        mask_c = [i for i, n in enumerate(cluster_names) if n not in targets_use]
        net_mat[:, mask_c] = 0

    if remove_isolate:
        keep = np.where((net_mat.sum(axis=1) + net_mat.sum(axis=0)) > 0)[0]
        net_mat = net_mat[np.ix_(keep, keep)]
        cluster_names = [cluster_names[i] for i in keep]
        colors = [colors[i] for i in keep]
        color_map = dict(zip(cluster_names, colors))
        n_clusters = len(cluster_names)

    if n_clusters == 0:
        warnings.warn("No clusters to plot")
        return None

    # --- compute circular positions (clockwise from top) ---
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_clusters, endpoint=False)
    R = 1.0  # circle radius
    pos = {name: (R * np.cos(a), R * np.sin(a))
           for name, a in zip(cluster_names, angles)}

    # node sizes: proportional to vertex_weight
    if vertex_weight is None:
        # try to use cell counts
        try:
            counts = np.array(list(pd.Series(cellchat.groups).value_counts().reindex(cluster_names).fillna(1).values), dtype=float)
        except Exception:
            counts = np.ones(n_clusters)
        if counts.max() > counts.min():
            node_radii = np.interp(counts, (counts.min(), counts.max()), (2, min(vertex_size_max, 6)))
        else:
            node_radii = np.full(n_clusters, 4.0)
    elif isinstance(vertex_weight, (int, float)):
        node_radii = np.full(n_clusters, float(vertex_weight))
    else:
        vw = np.array(vertex_weight, dtype=float)
        if vw.max() > vw.min():
            node_radii = np.interp(vw, (vw.min(), vw.max()), (2, min(vertex_size_max, 6)))
        else:
            node_radii = np.full(n_clusters, 4.0)
    # convert radius -> scatter size (points^2)
    node_sizes = (node_radii * 20) ** 2 / 400

    # edge widths
    all_weights = [net_mat[i, j] for i in range(n_clusters) for j in range(n_clusters)
                   if net_mat[i, j] > 0]
    if not all_weights:
        warnings.warn("No significant interactions found for circle plot")
        return None
    if edge_weight_max is None:
        edge_weight_max = max(all_weights)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- draw edges (curved arcs) ---
    for i, src in enumerate(cluster_names):
        for j, tgt in enumerate(cluster_names):
            w = net_mat[i, j]
            if w <= 0:
                continue
            if weight_scale:
                lw = 0.3 + w / edge_weight_max * edge_width_max
            else:
                lw = 0.3 + edge_width_max * w
            lw = min(lw, edge_width_max)

            c = to_rgba(color_map[src], alpha=alpha_edge)
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]

            if src == tgt:
                _, label_position = _draw_circular_self_loop(
                    ax, (x0, y0), angles[i], color=c,
                    linewidth=lw, alpha=alpha_edge,
                    mutation_scale=6 * arrow_size / 0.2 * arrow_width,
                )
            else:
                rad = edge_curved
                mutation = 6 * arrow_size / 0.2 * arrow_width
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c,
                        lw=lw,
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=mutation,
                    )
                )
                label_position = _curved_edge_label_position(
                    (x0, y0), (x1, y1), rad
                )

            if label_edge:
                edge_text = ax.text(
                    *label_position, _format_edge_label(w),
                    color=edge_label_color,
                    fontsize=max(1.0, edge_label_cex * 10),
                    ha="center", va="center", zorder=7,
                )
                edge_text.set_gid("network-edge-label")

    # --- draw nodes ---
    for idx, name in enumerate(cluster_names):
        x, y = pos[name]
        ax.scatter(x, y, s=node_sizes[idx] * 15, c=[color_map[name]],
                   edgecolors=[color_map[name]], linewidths=1.0, zorder=5)

    # --- labels outside the circle ---
    label_r = R + 0.18
    for name, angle in zip(cluster_names, angles):
        lx = label_r * np.cos(angle)
        ly = label_r * np.sin(angle)
        ha = 'left' if np.cos(angle) >= 0 else 'right'
        va = 'bottom' if np.sin(angle) >= 0 else 'top'
        ax.text(lx, ly, name,
                ha=ha, va=va,
                fontsize=max(7, vertex_label_cex * 10),
                color=vertex_label_color,
                fontweight='normal')

    pad = margin + 0.3
    ax.set_xlim(-R - pad, R + pad)
    ax.set_ylim(-R - pad, R + pad)

    if title_name:
        ax.set_title(title_name, fontsize=11, fontweight='normal',
                     ha='center', pad=6)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None
# ---------------------------------------------------------------------------
# 2.  plot_network_heatmap  - heatmap of interaction count or weight
# ---------------------------------------------------------------------------

def plot_network_heatmap(
    cellchat: 'CellChat',
    measure: str = "weight",
    signaling: Optional[str] = None,
    slot_name: str = "pathway_network",
    color_use: Optional[List[str]] = None,
    color_heatmap: Optional[Union[str, List[str]]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    remove_isolate: bool = False,
    thresh: float = 0.05,
    font_size: int = 8,
    font_size_title: int = 10,
    width: Optional[float] = None,
    height: Optional[float] = None,
    ylim_top: Optional[float] = None,
    ylim_right: Optional[float] = None,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
    comparison: Optional[Tuple[int, int]] = None,
) -> Optional[plt.Figure]:
    """
    Heatmap of cell-cell communication strength / count.
    Mirrors R plot_network_heatmap() (ComplexHeatmap style).

    Layout:
    - Top bar: column sums (colored by target cell)
    - Left thin strip: row cell colors
    - Right bar: row sums (colored by source cell)
    - Vertical colorbar on the far right
    - X-axis labels: 90-degree vertical
    - Title: top center
    """
    selected_networks, resolved_comparison = _comparison_networks(
        cellchat, slot_name, comparison
    )
    if not selected_networks:
        raise ValueError(f"No network data found in {slot_name} slot")
    is_merged_comparison = resolved_comparison is not None
    net_data = selected_networks[0][1]
    cluster_names = _network_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)

    def _matrix_in_group_order(network, value):
        """Read a matrix and reorder it to the first network's group names."""
        source_groups = _network_group_names(cellchat, network)
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (len(source_groups), len(source_groups)):
            raise ValueError("Network measure dimensions do not match its groups order.")
        if set(source_groups) != set(cluster_names):
            raise ValueError("Compared networks must contain the same cell groups.")
        order = [source_groups.index(group) for group in cluster_names]
        return matrix[np.ix_(order, order)]

    if signaling is not None:
        if signaling not in net_data['prob']:
            raise ValueError(f"Signaling pathway '{signaling}' not found")
        mat = _matrix_in_group_order(net_data, _network_matrix(net_data, 'prob', signaling)).copy()
        if is_merged_comparison:
            other = selected_networks[1][1]
            if signaling not in other['prob']:
                raise ValueError(f"Signaling pathway '{signaling}' not found in both compared datasets")
            mat = _matrix_in_group_order(other, _network_matrix(other, 'prob', signaling)) - mat
        if title_name is None:
            title_name = (f"Differential {signaling} signaling network"
                          if is_merged_comparison else f"{signaling} signaling network")
        legend_name = "Communication Prob."
        is_differential = is_merged_comparison
    else:
        def _aggregate(network):
            if measure in network:
                return _matrix_in_group_order(network, network[measure]).copy()
            _, prob_array, pval_array = _network_arrays(network)
            prob_c = prob_array.copy()
            prob_c[pval_array >= thresh] = 0.0
            result = (np.sum(prob_c > 0, axis=2).astype(float)
                    if measure == 'count' else np.sum(prob_c, axis=2))
            return _matrix_in_group_order(network, result)

        mat = _aggregate(net_data)
        if is_merged_comparison:
            mat = _aggregate(selected_networks[1][1]) - mat
        if title_name is None:
            base_title = "Number of interactions" if measure == "count" else "Interaction strength"
            title_name = f"Differential {base_title}" if is_merged_comparison else base_title
        legend_name = title_name
        is_differential = is_merged_comparison

    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        raise ValueError("The selected network contains no values to plot")
    row_names = list(cluster_names)
    col_names = list(cluster_names)
    row_colors = list(colors)
    col_colors = list(colors)

    if sources_use is not None:
        keep_r = [i for i, n in enumerate(row_names) if n in sources_use]
        mat = mat[keep_r, :]
        row_names = [row_names[i] for i in keep_r]
        row_colors = [row_colors[i] for i in keep_r]
    if targets_use is not None:
        keep_c = [i for i, n in enumerate(col_names) if n in targets_use]
        mat = mat[:, keep_c]
        col_names = [col_names[i] for i in keep_c]
        col_colors = [col_colors[i] for i in keep_c]

    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError("sources_use and targets_use must retain at least one cell group")

    if remove_isolate:
        keep_r = np.where(np.abs(mat).sum(axis=1) > 0)[0]
        keep_c = np.where(np.abs(mat).sum(axis=0) > 0)[0]
        mat = mat[np.ix_(keep_r, keep_c)]
        row_names = [row_names[i] for i in keep_r]
        col_names = [col_names[i] for i in keep_c]
        row_colors = [row_colors[i] for i in keep_r]
        col_colors = [col_colors[i] for i in keep_c]

    if mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError("No non-isolated cell groups remain after filtering")

    if cluster_rows and len(row_names) > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        order = leaves_list(linkage(mat, method='average'))
        mat = mat[order, :]
        row_names = [row_names[i] for i in order]
        row_colors = [row_colors[i] for i in order]
    if cluster_cols and len(col_names) > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        order = leaves_list(linkage(mat.T, method='average'))
        mat = mat[:, order]
        col_names = [col_names[i] for i in order]
        col_colors = [col_colors[i] for i in order]

    # colormap
    mat_min, mat_max = float(np.nanmin(mat)), float(np.nanmax(mat))
    is_differential = is_differential and mat_min < 0 < mat_max
    if color_heatmap is None:
        color_heatmap = ['#2166ac', '#b2182b'] if is_differential else 'Reds'

    if isinstance(color_heatmap, list):
        if is_differential:
            cmap = LinearSegmentedColormap.from_list(
                "div", [color_heatmap[0], '#f7f7f7', color_heatmap[1]])
            vmin, vmax = mat_min, mat_max
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            cmap = LinearSegmentedColormap.from_list("hm", color_heatmap)
            norm = None
    else:
        cmap = plt.get_cmap(color_heatmap)
        norm = None

    nR, nC = mat.shape
    row_sums = np.abs(mat).sum(axis=1)
    col_sums = np.abs(mat).sum(axis=0)

    pos_vals = mat[mat > 0]
    vmin_hm = pos_vals.min() if (norm is None and len(pos_vals) > 0) else None
    vmax_hm = mat.max()      if (norm is None) else None

    # -- GridSpec layout --------------------------------------------------------
    # Layout (rows x cols):
    #
    #  Col:  [row_labels | left_strip | heatmap    | right_strip | right_bar | cbar ]
    #  Row0: [  off      |    off     | top_bar    |    off      |   off     |  off ]
    #  Row1: [  off      |    off     | top_strip  |    off      |   off     |  off ]
    #  Row2: [  labels   | left_strip | heatmap    | right_strip | right_bar | cbar ]
    #  Row3: [  off      |    off     | bot_strip  |    off      |   off     |  off ]
    #  Row4: [  off      |    off     | x_labels   |    off      |   off     |  off ]
    #
    # Design rules:
    #   top_strip  : same colors as bottom_strip (col colors), aligned to heatmap cols
    #   right_strip: same colors as left_strip (row colors), aligned to heatmap rows
    #   top bar    : bars grow upward; height ~ col_sum; zero -> hairline (0.02 * ymax)
    #                y-ticks on RIGHT side (avoids overlap with strip); top spine only
    #   right bar  : bars grow rightward; height ~ row_sum; zero -> hairline
    #                x-ticks on TOP side; right spine only
    #   colorbar   : small vertical legend, centered vertically in cbar column

    fig_w, fig_h = fig_size
    if width  is not None: fig_w = width
    if height is not None: fig_h = height
    if fig_w <= 0 or fig_h <= 0:
        raise ValueError("width and height must be positive")

    w_labels  = 1.0            # row label area (inches)
    w_lstrip  = 0.12           # left color strip
    w_rstrip  = 0.12           # right color strip (mirrors left)
    w_gap     = 0.08           # small gap between right strip and bar
    w_rbar    = 0.65           # right bar chart
    w_cbar    = 0.65           # colorbar column
    w_heat    = max(fig_w - w_labels - w_lstrip - w_rstrip - w_gap - w_rbar - w_cbar, 0.5)

    h_top      = 0.75          # top bar chart
    h_tstrip   = 0.12          # top color strip (mirrors bottom)
    h_bstrip   = 0.12          # bottom color strip
    h_xlabels  = 1.1           # x-axis labels
    h_heat     = fig_h - h_top - h_tstrip - h_bstrip - h_xlabels

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        5, 7,
        height_ratios=[h_top, h_tstrip, max(h_heat, 0.5), h_bstrip, h_xlabels],
        width_ratios=[w_labels, w_lstrip, w_heat, w_rstrip, w_gap, w_rbar, w_cbar],
        hspace=0.0,
        wspace=0.0,
        left=0.02, right=0.98, top=0.91, bottom=0.02,
    )

    # Row 0 - top bar
    for c in [0, 1, 3, 4, 5, 6]:
        fig.add_subplot(gs[0, c]).axis('off')
    ax_top = fig.add_subplot(gs[0, 2])

    # Row 1 - top color strip
    for c in [0, 1, 3, 4, 5, 6]:
        fig.add_subplot(gs[1, c]).axis('off')
    ax_strip_t = fig.add_subplot(gs[1, 2])

    # Row 2 - main row
    ax_labels      = fig.add_subplot(gs[2, 0])
    ax_strip_l     = fig.add_subplot(gs[2, 1])
    ax_heat        = fig.add_subplot(gs[2, 2])
    ax_strip_r     = fig.add_subplot(gs[2, 3])
    ax_gap         = fig.add_subplot(gs[2, 4]); ax_gap.axis('off'); ax_gap.set_visible(False)
    ax_right       = fig.add_subplot(gs[2, 5])
    ax_cbar_holder = fig.add_subplot(gs[2, 6])

    # Row 3 - bottom color strip
    for c in [0, 1, 3, 4, 5, 6]:
        fig.add_subplot(gs[3, c]).axis('off')
    ax_strip_b = fig.add_subplot(gs[3, 2])

    # Row 4 - x-axis labels
    for c in [0, 1, 3, 4, 5, 6]:
        fig.add_subplot(gs[4, c]).axis('off')
    ax_xlbl = fig.add_subplot(gs[4, 2])

    # -- heatmap ---------------------------------------------------------------
    mat_plot = mat.copy().astype(float)
    mat_plot[mat_plot == 0] = np.nan

    im = ax_heat.imshow(
        mat_plot, cmap=cmap, norm=norm, aspect='auto',
        vmin=vmin_hm, vmax=vmax_hm, interpolation='nearest',
    )
    ax_heat.set_xticks([]); ax_heat.set_yticks([])
    ax_heat.set_xticks(np.arange(nC + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(nR + 1) - 0.5, minor=True)
    ax_heat.grid(which='minor', color='white', linewidth=0.5)
    ax_heat.tick_params(which='minor', bottom=False, left=False)
    ax_heat.spines[:].set_visible(False)

    # -- row labels ------------------------------------------------------------
    ax_labels.set_xlim(0, 1); ax_labels.set_ylim(-0.5, nR - 0.5)
    ax_labels.axis('off')
    for idx, name in enumerate(row_names):
        ax_labels.text(0.98, nR - 1 - idx, name,
                       ha='right', va='center', fontsize=font_size)
    mid_y = (nR - 1) / 2.0
    ax_labels.text(-0.12, mid_y, 'Sources (Sender)',
                   ha='center', va='center', fontsize=font_size_title,
                   rotation=90, transform=ax_labels.transData)

    # -- left color strip ------------------------------------------------------
    ax_strip_l.set_xlim(0, 1); ax_strip_l.set_ylim(-0.5, nR - 0.5)
    ax_strip_l.axis('off')
    for idx, c in enumerate(row_colors):
        ax_strip_l.add_patch(mpatches.Rectangle(
            (0, nR - 1 - idx - 0.5), 1, 1, color=c, linewidth=0))

    # -- right color strip: hidden (no top/right color strips) ----------------
    ax_strip_r.axis('off')

    # -- top color strip: hidden -----------------------------------------------
    ax_strip_t.axis('off')

    # -- top bar chart (bars grow upward) --------------------------------------
    # y-ticks on LEFT side: 0, and rounded max values
    ymax_top = ylim_top if ylim_top is not None else (col_sums.max() * 1.25 if col_sums.max() > 0 else 1)
    for i_col, (col_h, col_c) in enumerate(zip(col_sums, col_colors)):
        if col_h > 0:
            ax_top.bar(i_col, col_h, color=col_c, edgecolor='none', width=0.8)
        else:
            # hairline: draw as a horizontal line at y=0
            ax_top.plot([i_col - 0.4, i_col + 0.4], [0, 0],
                        color=col_c, linewidth=1.5, solid_capstyle='butt')
    ax_top.set_xlim(-0.5, nC - 0.5)
    ax_top.set_ylim(0, ymax_top)
    ax_top.set_xticks([])
    # y-ticks on LEFT side (matching screenshot: 0, 0.1, 0.2 on left spine)
    ytick_max = _smart_round(col_sums.max()) if col_sums.max() > 0 else 0
    ytick_mid = _smart_round(col_sums.max() / 2.0) if col_sums.max() > 0 else 0
    yticks = sorted(set([0] + [t for t in [ytick_mid, ytick_max] if t > 0]))
    ax_top.set_yticks(yticks)
    ax_top.yaxis.set_label_position('left')
    ax_top.yaxis.tick_left()
    ax_top.tick_params(axis='y', labelsize=font_size - 1, length=2, pad=2)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['left'].set_linewidth(0.6)
    ax_top.set_title(title_name, fontsize=font_size_title,
                     fontweight='normal', pad=5, ha='center')

    # -- bottom color strip ----------------------------------------------------
    ax_strip_b.set_xlim(-0.5, nC - 0.5); ax_strip_b.set_ylim(0, 1)
    ax_strip_b.axis('off')
    for idx, c in enumerate(col_colors):
        ax_strip_b.add_patch(mpatches.Rectangle(
            (idx - 0.5, 0), 1, 1, color=c, linewidth=0))

    # -- x-axis labels (below bottom strip) -----------------------------------
    ax_xlbl.set_xlim(-0.5, nC - 0.5); ax_xlbl.set_ylim(0, 1)
    ax_xlbl.axis('off')
    for idx, name in enumerate(col_names):
        ax_xlbl.text(idx, 0.98, name,
                     ha='right', va='top', fontsize=font_size,
                     rotation=90, transform=ax_xlbl.transData)
    ax_xlbl.text((nC - 1) / 2.0, -0.15, 'Targets (Receiver)',
                 ha='center', va='top', fontsize=font_size_title,
                 transform=ax_xlbl.transData)

    # -- right bar chart (bars grow rightward) ---------------------------------
    # Zero-strength rows -> hairline; non-zero -> proportional width
    # x-ticks on BOTTOM (left side of bar area), y-ticks (row labels) suppressed
    # Matching screenshot: x-ticks at bottom: 0, 0.1, 0.2, 0.3
    xmax_r = ylim_right if ylim_right is not None else (row_sums.max() * 1.25 if row_sums.max() > 0 else 1)

    for i_row, (row_w, row_c) in enumerate(zip(row_sums, row_colors)):
        bar_idx = nR - 1 - i_row   # barh y-position (reversed to match heatmap order)
        if row_w > 0:
            ax_right.barh(bar_idx, row_w, color=row_c, edgecolor='none', height=0.8)
        else:
            # hairline: draw as a vertical line at x=0
            ax_right.plot([0, 0], [bar_idx - 0.4, bar_idx + 0.4],
                          color=row_c, linewidth=1.5, solid_capstyle='butt')
    ax_right.set_ylim(-0.5, nR - 0.5)
    ax_right.set_xlim(0, xmax_r)
    ax_right.set_yticks([])
    xtick_max = _smart_round(row_sums.max()) if row_sums.max() > 0 else 0
    xtick_mid = _smart_round(row_sums.max() / 2.0) if row_sums.max() > 0 else 0
    # x-ticks on BOTTOM side: 0, mid, max (matching screenshot)
    xticks = sorted(set([0, xtick_mid, xtick_max]))
    ax_right.set_xticks(xticks)
    ax_right.xaxis.set_label_position('bottom')
    ax_right.xaxis.tick_bottom()
    ax_right.tick_params(axis='x', labelsize=font_size - 1, length=2, pad=2,
                         rotation=90)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_linewidth(0.6)

    # -- colorbar: small vertical legend, centered in cbar column -------------
    ax_cbar_holder.axis('off')
    # center the inset: [left, bottom, width, height] in axes fraction
    cbar_inset = ax_cbar_holder.inset_axes([0.05, 0.25, 0.4, 0.5])
    cb = plt.colorbar(im, cax=cbar_inset, orientation='vertical')
    cb.set_label(legend_name, fontsize=font_size - 1, labelpad=3)
    cb.ax.tick_params(labelsize=font_size - 1, length=2)
    if norm is None and len(pos_vals) > 0:
        # Show 4 evenly-spaced ticks: 0, 1/3, 2/3, max (matching screenshot)
        t0 = 0.0
        t1 = _smart_round(vmax_hm / 3.0)
        t2 = _smart_round(vmax_hm * 2.0 / 3.0)
        t3 = _smart_round(vmax_hm)
        cb_ticks = sorted(set([t for t in [t0, t1, t2, t3] if t <= vmax_hm]))
        cb.set_ticks(cb_ticks)
    cbar_inset.spines[:].set_visible(False)

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 3.  plot_network_bubble  - bubble / dot plot for L-R pairs
# ---------------------------------------------------------------------------

def plot_network_bubble(
    cellchat: 'CellChat',
    sources_use=None,
    targets_use=None,
    signaling=None,
    pair_lr_use=None,
    thresh: float = 0.05,
    color_heatmap: str = "Spectral",
    remove_isolate: bool = False,
    line_on: bool = True,
    line_size: float = 0.2,
    color_grid: str = "grey90",
    title_name: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 10,
    dot_size_min: Optional[float] = None,
    dot_size_max: Optional[float] = None,
    angle_x: int = 90,
    vjust_x: Optional[float] = None,
    hjust_x: Optional[float] = None,
    sort_by_source: bool = False,
    sort_by_target: bool = False,
    sort_by_source_priority: bool = True,
    fig_size: Tuple[int, int] = (12, 8),
    return_fig: bool = False,
    comparison: Optional[Tuple[int, int]] = None,
    max_dataset: Optional[int] = None,
    return_data: bool = False,
) -> Union[Optional[plt.Figure], Dict[str, Any]]:
    """
    Bubble plot showing significant L-R pair interactions.
    Mirrors R plot_network_bubble().

    Dot size   = p-value significance level (3 categories):
                   1 = p > 0.05 (smallest)
                   2 = 0.01 < p <= 0.05
                   3 = p <= 0.01  (largest)
    Dot colour = -1/log(prob) communication probability (Spectral reversed)
    X-axis     = "source -> target" labels, 90-degree vertical
    Legend     = size legend (p-value) + narrow vertical colorbar

    sort_by_source : bool
        Sort x-axis by the order of sources_use (mirrors R sort.by.source).
    sort_by_target : bool
        Sort x-axis by the order of targets_use (mirrors R sort.by.target).
    sort_by_source_priority : bool
        When both sort_by_source and sort_by_target are True, True means
        group by source first then target; False means group by target first
        then source (mirrors R sort.by.source.priority).
    """
    selected_networks, resolved_comparison = _comparison_networks(
        cellchat, 'network', comparison
    )
    if not selected_networks or 'prob' not in selected_networks[0][1]:
        raise ValueError("No probability data in the network slot.")
    comparison_mode = resolved_comparison is not None
    net_data = selected_networks[0][1]
    lr_names_first, prob_first, pval_first = _network_arrays(net_data)

    # Use the group order stored with the network to keep matrix axes aligned.
    cluster_names = list(net_data.get('groups') or _get_cluster_names_from_cellchat(cellchat))

    # Build lookup dictionaries across all compared datasets. An interaction
    # can be significant in only one dataset, so the first network's metadata
    # is not necessarily complete for a comparison.
    lr_to_display: Dict[str, str] = {}
    lr_to_pathway: Dict[str, str] = {}
    for _, dataset_network in selected_networks:
        interactions_df = dataset_network.get('interactions', None)
        if not isinstance(interactions_df, pd.DataFrame) or 'interaction_name' not in interactions_df.columns:
            continue
        for _, irow in interactions_df.iterrows():
            iname = str(irow['interaction_name'])
            if 'interaction_name_2' in interactions_df.columns and pd.notna(irow['interaction_name_2']):
                lr_to_display.setdefault(iname, str(irow['interaction_name_2']))
            if 'pathway_name' in interactions_df.columns and pd.notna(irow['pathway_name']):
                lr_to_pathway.setdefault(iname, str(irow['pathway_name']))

    # pair_lr_use: filter Y-axis to specific L-R pairs (mirrors R pairLR.use)
    # Accepts DataFrame (from extractEnrichedLR) or list of interaction names
    pair_lr_filter: Optional[set] = None
    pair_lr_pathway_filter: Optional[set] = None
    if pair_lr_use is not None:
        if isinstance(pair_lr_use, pd.DataFrame):
            if 'interaction_name' in pair_lr_use.columns:
                pair_lr_filter = set(pair_lr_use['interaction_name'].astype(str))
            elif 'pathway_name' in pair_lr_use.columns:
                pair_lr_pathway_filter = set(pair_lr_use['pathway_name'].astype(str))
            else:
                pair_lr_filter = set(pair_lr_use.iloc[:, 0].astype(str))
        elif isinstance(pair_lr_use, (list, tuple, set)):
            pair_lr_filter = set(str(x) for x in pair_lr_use)

    if signaling is not None and isinstance(signaling, str):
        signaling = [signaling]

    # sources_use / targets_use: accept zero-based integer indices or strings.
    def _resolve_cell_indices(val, cluster_names):
        if val is None:
            return cluster_names
        if isinstance(val, (int, np.integer)):
            val = [val]
        if isinstance(val, (list, tuple)):
            resolved = []
            for v in val:
                if isinstance(v, (int, np.integer)):
                    idx = int(v)
                    if 0 <= idx < len(cluster_names):
                        resolved.append(cluster_names[idx])
                else:
                    resolved.append(str(v))
            return resolved
        return val

    sources_use = _resolve_cell_indices(sources_use, cluster_names)
    targets_use = _resolve_cell_indices(targets_use, cluster_names)

    src_idx = [cluster_names.index(s) for s in sources_use if s in cluster_names]
    tgt_idx = [cluster_names.index(t) for t in targets_use if t in cluster_names]

    dataset_arrays = [(selected_networks[0][0], lr_names_first, prob_first, pval_first)]
    if comparison_mode:
        other_name, other_data = selected_networks[1]
        other_groups = list(other_data.get('groups') or cluster_names)
        if other_groups != cluster_names:
            raise ValueError("Compared datasets must use the same cell-group order")
        other_lr_names, other_prob_raw, other_pval_raw = _network_arrays(other_data)
        dataset_arrays.append((other_name, other_lr_names, other_prob_raw, other_pval_raw))

        # R CellChat compares the per-dataset communication tables by L-R name.
        # Preserve the first dataset's order and append pairs unique to later
        # datasets, then align temporary arrays without changing the CellChat
        # object's canonical per-dataset networks.
        lr_names_full = list(lr_names_first)
        seen_lr = set(lr_names_full)
        for lr_name in other_lr_names:
            if lr_name not in seen_lr:
                lr_names_full.append(lr_name)
                seen_lr.add(lr_name)

        aligned_arrays = []
        for dataset_name, lr_names, dataset_prob, dataset_pval in dataset_arrays:
            name_to_index = {name: index for index, name in enumerate(lr_names)}
            aligned_prob = np.zeros((*dataset_prob.shape[:2], len(lr_names_full)), dtype=float)
            aligned_pval = np.ones((*dataset_pval.shape[:2], len(lr_names_full)), dtype=float)
            for aligned_index, lr_name in enumerate(lr_names_full):
                source_index = name_to_index.get(lr_name)
                if source_index is not None:
                    aligned_prob[:, :, aligned_index] = dataset_prob[:, :, source_index]
                    aligned_pval[:, :, aligned_index] = dataset_pval[:, :, source_index]
            aligned_arrays.append((dataset_name, aligned_prob, aligned_pval))

        prob_array = aligned_arrays[0][1]
        pval_array = aligned_arrays[0][2]
        other_prob = aligned_arrays[1][1]
        other_pval = aligned_arrays[1][2]
        n_lr = len(lr_names_full)
        if max_dataset is not None:
            if max_dataset not in (1, 2):
                raise ValueError("max_dataset must be 1 (first dataset) or 2 (second dataset)")
            first_scores = prob_array[np.ix_(src_idx, tgt_idx, range(n_lr))].sum(axis=(0, 1))
            second_scores = other_prob[np.ix_(src_idx, tgt_idx, range(n_lr))].sum(axis=(0, 1))
            winner_scores = first_scores if max_dataset == 1 else second_scores
            other_scores = second_scores if max_dataset == 1 else first_scores
            winner_lrs = {lr_names_full[k] for k in range(n_lr)
                          if winner_scores[k] >= other_scores[k] or np.isclose(winner_scores[k], other_scores[k])}
        else:
            winner_lrs = set(lr_names_full)
    else:
        lr_names_full = list(lr_names_first)
        prob_array = prob_first
        pval_array = pval_first
        n_lr = len(lr_names_full)
        other_name, other_prob, other_pval = None, None, None
        winner_lrs = set(lr_names_full)

    rows = []
    datasets_to_draw = [(selected_networks[0][0], prob_array, pval_array)]
    if comparison_mode:
        datasets_to_draw.append((other_name, other_prob, other_pval))
    for k, lr in enumerate(lr_names_full):
        if lr not in winner_lrs:
            continue
        lr_pathway = lr_to_pathway.get(lr, '')
        if pair_lr_filter is not None and lr not in pair_lr_filter:
            continue
        if pair_lr_pathway_filter is not None and lr_pathway not in pair_lr_pathway_filter:
            continue
        if signaling is not None and lr_pathway not in signaling:
            continue
        display_name = lr_to_display.get(lr, lr)
        for dataset_name, dataset_prob, dataset_pval in datasets_to_draw:
            for i in src_idx:
                for j in tgt_idx:
                    probability = dataset_prob[i, j, k]
                    pvalue = dataset_pval[i, j, k]
                    if probability <= 0 or pvalue >= thresh:
                        continue
                    source_target = f"{cluster_names[i]} -> {cluster_names[j]}"
                    if comparison_mode:
                        source_target = f"{source_target} ({dataset_name})"
                    rows.append({
                        'dataset': dataset_name,
                        'source': cluster_names[i],
                        'target': cluster_names[j],
                        'interaction_name': lr,
                        'interaction_name_2': display_name,
                        'lr': display_name,
                        'prob': float(probability),
                        'pval': float(pvalue),
                        'source_target': source_target,
                    })

    if not rows:
        warnings.warn("No significant interactions found for bubble plot")
        return None

    df = pd.DataFrame(rows)

    # R adds isolate source-target pairs with pval=1 and prob=NA when
    # remove.isolate=FALSE. They are visually white but keep the full x-axis and
    # make the p > 0.05 legend level appear.
    all_sources = sources_use if sources_use is not None else cluster_names
    all_targets = targets_use if targets_use is not None else cluster_names
    all_st_pairs = [f"{s} -> {t}" for s in all_sources for t in all_targets
                    if s in cluster_names and t in cluster_names]
    if comparison_mode and not remove_isolate:
        # Keep zero-interaction source-target pairs visible for each dataset,
        # matching the paired comparison layout in CellChat R.
        present_st = set(row['source_target'] for row in rows)
        first_lr = next((row['lr'] for row in rows), None)
        first_name = next((row['interaction_name'] for row in rows), None)
        if first_lr is not None:
            for dataset_name, _, _ in datasets_to_draw:
                for source in all_sources:
                    for target in all_targets:
                        source_target = f"{source} -> {target} ({dataset_name})"
                        if source_target not in present_st:
                            rows.append({
                                'dataset': dataset_name, 'source': source, 'target': target,
                                'interaction_name': first_name, 'interaction_name_2': first_lr,
                                'lr': first_lr, 'prob': np.nan, 'pval': 1.0,
                                'source_target': source_target,
                            })
            df = pd.DataFrame(rows)
    elif not comparison_mode and not remove_isolate:
        present_st = set(df['source_target'].astype(str))
        first_lr = str(df['lr'].iloc[0])
        isolate_rows = []
        for st in all_st_pairs:
            if st not in present_st:
                src, tgt = st.split(' -> ', 1)
                isolate_rows.append({
                    'dataset': selected_networks[0][0],
                    'source': src,
                    'target': tgt,
                    'interaction_name': first_lr,
                    'interaction_name_2': first_lr,
                    'lr': first_lr,
                    'prob': np.nan,
                    'pval': 1.0,
                    'source_target': st,
                })
        if isolate_rows:
            df = pd.concat([df, pd.DataFrame(isolate_rows)], ignore_index=True)

    # p-value -> 3 discrete size levels (mirrors R)
    def _pval_level(p):
        if p <= 0.01:
            return 3
        elif p <= 0.05:
            return 2
        else:
            return 1

    df['pval_level'] = df['pval'].apply(_pval_level)

    # colour value: -1/log(prob)  (mirrors R: df.network$prob <- -1/log(df.network$prob))
    df['prob_transformed'] = np.nan
    prob_mask = df['prob'].notna() & (df['prob'] > 0)
    df.loc[prob_mask, 'prob_transformed'] = -1.0 / np.log(
        df.loc[prob_mask, 'prob'].clip(1e-300, 1 - 1e-10)
    )
    finite_mask = np.isfinite(df['prob_transformed']) & (df['prob_transformed'] > 0)
    if finite_mask.any():
        max_finite = df.loc[finite_mask, 'prob_transformed'].max()
        bad_mask = prob_mask & ~finite_mask
        df.loc[bad_mask, 'prob_transformed'] = max_finite * 1.3
        finite_mask = np.isfinite(df['prob_transformed']) & (df['prob_transformed'] > 0)
    else:
        warnings.warn("No finite communication probabilities found for bubble plot")
        return None
    df.loc[prob_mask, 'prob_transformed'] = df.loc[prob_mask, 'prob_transformed'].clip(lower=0)

    if df.empty:
        warnings.warn("No significant interactions found for bubble plot")
        return None

    # R: remove.isolate = TRUE also disables grid lines
    if remove_isolate:
        line_on = False

    # Y-axis ordering: if pair_lr_use provided, use its order; otherwise alphabetical
    if pair_lr_use is not None and isinstance(pair_lr_use, pd.DataFrame):
        if 'interaction_name_2' in pair_lr_use.columns:
            pair_lr_y_order = [str(x) for x in pair_lr_use['interaction_name_2'] if str(x) in df['lr'].values]
        elif 'interaction_name' in pair_lr_use.columns:
            pair_lr_y_order = [lr_to_display.get(str(x), str(x)) for x in pair_lr_use['interaction_name']
                              if lr_to_display.get(str(x), str(x)) in df['lr'].values]
        else:
            pair_lr_y_order = None
        if pair_lr_y_order:
            remaining = sorted(set(df['lr'].unique()) - set(pair_lr_y_order))
            lr_order = pair_lr_y_order + remaining
        else:
            lr_order = sorted(df['lr'].unique())
    else:
        lr_order = sorted(df['lr'].unique())

    # X-axis ordering: mirrors R sort.by.source / sort.by.target logic
    def _split_source_target(st):
        source, target = st.split(' -> ', 1)
        if comparison_mode:
            target = target.rsplit(' (', 1)[0]
        return source, target

    def _src_rank(st):
        s, _ = _split_source_target(st)
        return all_sources.index(s) if s in all_sources else 999

    def _tgt_rank(st):
        _, t = _split_source_target(st)
        return all_targets.index(t) if t in all_targets else 999

    def _default_rank(st):
        s, t = _split_source_target(st)
        si = cluster_names.index(s) if s in cluster_names else 999
        ti = cluster_names.index(t) if t in cluster_names else 999
        return (si, ti)

    if sort_by_source and sort_by_target:
        if sort_by_source_priority:
            key_fn = lambda st: (_src_rank(st), _tgt_rank(st))
        else:
            key_fn = lambda st: (_tgt_rank(st), _src_rank(st))
    elif sort_by_source:
        key_fn = lambda st: (_src_rank(st), _default_rank(st)[1])
    elif sort_by_target:
        key_fn = lambda st: (_tgt_rank(st), _default_rank(st)[0])
    else:
        key_fn = _default_rank

    # remove_isolate: True = only show source-target pairs with dots;
    #                 False = show ALL requested pairs on x-axis (R default)
    if remove_isolate:
        st_order = sorted(df.loc[df['prob'].notna(), 'source_target'].unique(), key=key_fn)
    else:
        if comparison_mode:
            all_st_pairs = [f"{s} -> {t} ({dataset_name})"
                            for dataset_name, _, _ in datasets_to_draw
                            for s in all_sources for t in all_targets]
        st_order = sorted(set(all_st_pairs) | set(df['source_target'].unique()), key=key_fn)

    df['lr'] = pd.Categorical(df['lr'], categories=lr_order)
    df['source_target'] = pd.Categorical(df['source_target'], categories=st_order)

    # colormap: Spectral reversed (direction=-1)
    try:
        base_cmap = plt.get_cmap(color_heatmap)
        cmap = base_cmap.reversed()
    except Exception:
        cmap = plt.cm.RdYlBu

    vmin = df.loc[finite_mask, 'prob_transformed'].min()
    vmax = df.loc[finite_mask, 'prob_transformed'].max()
    if vmin == vmax:
        vmin = max(0, vmax - 0.001)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # size mapping: 3 levels -> dot_size_min .. dot_size_max
    _dsmin = dot_size_min if dot_size_min is not None else 1.5
    _dsmax = dot_size_max if dot_size_max is not None else 4.5
    size_map = {
        1: _dsmin,
        2: (_dsmin + _dsmax) / 2,
        3: _dsmax,
    }

    fig, ax = plt.subplots(figsize=fig_size)

    for _, row in df.iterrows():
        x = st_order.index(row['source_target'])
        y = lr_order.index(row['lr'])
        s = size_map[row['pval_level']] ** 2 * 15  # area in points^2
        if pd.isna(row['prob_transformed']):
            # R keeps isolate pairs as prob=NA; scale_colour_gradientn renders
            # them with na.value="white" while preserving the p > 0.05 legend.
            color = 'white'
        else:
            color = cmap(norm(row['prob_transformed']))
        ax.scatter(x, y, s=s, c=[color], zorder=3,
                   edgecolors='none', alpha=1.0)

    # -- grid lines (theme_linedraw style, very thin) ---------------------------
    # Convert R-style grey names (e.g. "grey90") to hex for matplotlib
    import re as _re
    def _r_grey(c):
        m = _re.fullmatch(r'gr[ae]y(\d+)', c.strip().lower())
        if m:
            v = int(m.group(1)) / 100.0
            return (v, v, v)
        return c
    _color_grid = _r_grey(color_grid)
    if line_on:
        for xv in np.arange(0.5, len(st_order) - 0.5):
            ax.axvline(xv, color=_color_grid, lw=line_size, zorder=1)
        for yv in np.arange(0.5, len(lr_order) - 0.5):
            ax.axhline(yv, color=_color_grid, lw=line_size, zorder=1)
    if comparison_mode and len(datasets_to_draw) == 2:
        # Visually separate the two dataset blocks while retaining one shared
        # communication table for downstream comparison/filtering.
        first_count = sum(1 for value in st_order if str(value).endswith(f"({datasets_to_draw[0][0]})"))
        if first_count:
            ax.axvline(first_count - 0.5, color='black', lw=0.8,
                       linestyle='--', zorder=2)

    # -- x-axis labels (angle_x, vjust_x, hjust_x mirrors R params) -----------
    def _alignment(value, axis):
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isclose(numeric, 0):
                return 'left' if axis == 'x' else 'bottom'
            if np.isclose(numeric, 0.5):
                return 'center'
            if np.isclose(numeric, 1):
                return 'right' if axis == 'x' else 'top'
            raise ValueError(f"{axis}-axis alignment must be 0, 0.5, 1, or a Matplotlib alignment string")
        return value

    _ha = _alignment(hjust_x, 'x') or ('center' if angle_x == 90 else 'right')
    _va = _alignment(vjust_x, 'y') or ('top' if angle_x != 0 else 'center')
    ax.set_xticks(range(len(st_order)))
    ax.set_xticklabels(st_order, rotation=angle_x, ha=_ha, va=_va,
                        fontsize=max(font_size - 2, 7))
    ax.set_yticks(range(len(lr_order)))
    ax.set_yticklabels(lr_order, fontsize=font_size - 1)

    ax.set_xlim(-0.5, len(st_order) - 0.5)
    ax.set_ylim(-0.5, len(lr_order) - 0.5)

    # theme_linedraw: keep all 4 borders
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_visible(True)
    ax.tick_params(axis='both', which='both', length=0)

    # -- legends ---------------------------------------------------------------
    # 1) p-value size legend (only show levels present in data, like R)
    pval_labels = {1: 'p > 0.05', 2: '0.01 < p <= 0.05', 3: 'p <= 0.01'}
    present_levels = sorted(df['pval_level'].unique())
    legend_handles = []
    for level in present_levels:
        h = ax.scatter([], [], s=size_map[level] ** 2 * 15,
                       c='grey', label=pval_labels[level], edgecolors='none')
        legend_handles.append(h)
    leg = ax.legend(handles=legend_handles, title='p-value',
                    title_fontsize=font_size - 1,
                    fontsize=font_size - 2,
                    loc='upper left', bbox_to_anchor=(1.01, 1.0),
                    frameon=False, borderpad=0.5)
    ax.add_artist(leg)

    # 2) colorbar (narrow vertical strip, barwidth=0.5 style)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.3, pad=0.18, aspect=20,
                         anchor=(0, 0.0))
    cbar.set_label('Commun. Prob.', fontsize=font_size - 1, labelpad=3)
    cbar.ax.tick_params(labelsize=font_size - 2)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(['min', 'max'])

    if title_name:
        ax.set_title(title_name, fontsize=font_size_title, fontweight='normal',
                     pad=6, ha='center')

    plt.tight_layout()
    if return_data:
        return {'communication': df, 'figure': fig}
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 4.  plot_network_barplot  - bar chart of interaction count or strength
# ---------------------------------------------------------------------------

def plot_network_barplot(
    cellchat: 'CellChat',
    measure: str = "weight",
    signaling: Optional[str] = None,
    slot_name: str = "pathway_network",
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    color_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    x_lab_rot: bool = True,
    thresh: float = 0.05,
    fig_size: Tuple[int, int] = (8, 5),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Bar plot of number of interactions or interaction strength per target cell.
    Mirrors R plot_network_barplot().
    """
    if measure not in {'count', 'weight'}:
        raise ValueError("measure must be 'count' or 'weight'")
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    net_data = _network_view_for_visualization(cellchat, slot_name)

    if signaling is not None:
        if signaling not in net_data.get('prob', {}):
            raise ValueError(f"Signaling pathway '{signaling}' not found")
        mat = _network_matrix(net_data, 'prob', signaling).copy()
        pval_matrix = _network_matrix(net_data, 'pval', signaling)
        mat[pval_matrix >= thresh] = 0.0
        if title_name is None:
            title_name = f"{signaling} signaling network"
    else:
        if measure not in net_data:
            _, prob, pval = _network_arrays(net_data)
            prob_c = prob.copy()
            prob_c[pval >= thresh] = 0.0
            mat = np.sum(prob_c > 0, axis=2).astype(float) if measure == "count" else np.sum(prob_c, axis=2)
        else:
            mat = np.asarray(net_data[measure], dtype=float).copy()
            if mat.ndim != 2:
                raise ValueError(f"{measure} data must be a two-dimensional matrix")
        if title_name is None:
            title_name = "Number of interactions" if measure == "count" else "Interaction strength"

    # filter
    row_names = list(cluster_names)
    col_names = list(cluster_names)
    if sources_use is not None:
        keep_r = [i for i, n in enumerate(row_names) if n in sources_use]
        mat = mat[keep_r, :]
        row_names = [row_names[i] for i in keep_r]
    if targets_use is not None:
        keep_c = [i for i, n in enumerate(col_names) if n in targets_use]
        mat = mat[:, keep_c]
        col_names = [col_names[i] for i in keep_c]

    if not row_names or not col_names:
        raise ValueError("sources_use and targets_use must retain at least one cell group")

    # sum over sources to get per-target strength
    target_scores = mat.sum(axis=0)
    bar_colors = [color_map.get(n, '#999999') for n in col_names]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(np.arange(len(col_names)), target_scores, color=bar_colors, edgecolor='white')
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45 if x_lab_rot else 0, ha='right', fontsize=9)
    ax.set_ylabel("Communication Score", fontsize=9)
    ax.set_title(title_name, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 5.  plot_analysis_dot  - dot plot for pattern associations
# ---------------------------------------------------------------------------

def plot_analysis_dot(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    pattern: str = "outgoing",
    cutoff: Optional[float] = None,
    color_use: Optional[List[str]] = None,
    pathway_show: Optional[List[str]] = None,
    group_show: Optional[List[str]] = None,
    shape: int = 21,
    dot_size: Tuple[float, float] = (1, 3),
    dot_alpha: float = 1.0,
    main_title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Dot plot showing cell group <-> signaling pathway associations.
    Mirrors R plot_analysis_dot().

    Dot size   = W * H contribution score, scaled to dot_size range (R: scale_size_continuous)
    Dot colour = cell group colour (sc_palette)
    Y-axis     = cell groups reversed (R: scale_y_discrete(limits=rev(...)))
    X-axis     = signaling pathways, rotated 45 deg
    Theme      = theme_linedraw: black border, white panel, light grey grid

    The contribution score for (CellGroup, Signaling) is computed as:
      W_ij * H_jk  (pattern j contribution from cell i times pattern j strength for signal k)
      summed/max across patterns, then zeroed if the underlying probability is zero
      (mirrors R's data12 / data312 intersection check).
    """
    marker_map = {21: 'o', 22: 's'}
    if shape not in marker_map:
        raise ValueError("shape must be 21 (circle) or 22 (square)")
    net_data = _network_view_for_visualization(cellchat, slot_name)

    if 'pattern' not in net_data:
        raise ValueError(f"No pattern data in {slot_name}. Run identify_communication_patterns first.")

    pattern_data = net_data['pattern'].get(pattern)
    if pattern_data is None:
        raise ValueError(f"Pattern '{pattern}' not found")
    if 'pattern' in pattern_data:
        pattern_data = {**pattern_data, **pattern_data['pattern']}

    data_cell = pattern_data.get('cell').copy()    # CellGroup, Pattern, Contribution
    data_sig  = pattern_data.get('signaling').copy()  # Pattern, Signaling, Contribution

    # R default cutoff: 1/n_patterns (e.g. 1/6 <=0.167 for k=6)
    n_patterns = data_cell['Pattern'].nunique()
    if cutoff is None:
        cutoff = 1.0 / n_patterns

    if main_title is None:
        main_title = ("Outgoing communication patterns of secreting cells"
                      if pattern == "outgoing"
                      else "Incoming communication patterns of target cells")

    # -- colour map -------------------------------------------------------------
    try:
        groups = cellchat.groups
        group_order = list(groups.categories) if isinstance(groups, pd.Categorical) else []
    except Exception:
        group_order = []

    palette_size = len(group_order) if group_order else data_cell['CellGroup'].nunique()
    all_palette = _resolve_colors(palette_size, color_use)
    if group_order:
        color_map = {g: all_palette[i] for i, g in enumerate(group_order)}
    else:
        uniq = data_cell['CellGroup'].unique().tolist()
        color_map = {g: all_palette[i % len(all_palette)] for i, g in enumerate(uniq)}

    # -- apply cutoff (R: data1$Contribution[< cutoff] <- 0, same for data2) ---
    data_cell.loc[data_cell['Contribution'] < cutoff, 'Contribution'] = 0.0
    data_sig.loc[data_sig['Contribution']   < cutoff, 'Contribution'] = 0.0

    # -- build raw probability matrix to mask zero-interaction pairs ------------
    # R: data <- as.X.frame(as.table(data0))  (data0 is the col-max normalized mat)
    # Then only (CellGroup, Signaling) pairs that appear in data0 with non-zero
    # values are shown.  We replicate using the stored probability.
    raw_data = pattern_data.get('data')   # (n_cells, n_pathways) matrix or None
    valid_pairs: Optional[set] = None
    if raw_data is not None:
        # raw_data rows = cell groups (those with non-zero rows), cols = pathways
        # We need cell-group and pathway names for indexing
        data_names = pattern_data.get('data_names', {})
        all_cells_in_data = data_names.get('cell') or _ordered_levels(data_cell['CellGroup'])
        all_sigs_in_data = data_names.get('signaling') or _ordered_levels(data_sig['Signaling'])
        # row index maps to cell group in the order they appear
        n_r, n_c = raw_data.shape if hasattr(raw_data, 'shape') else (0, 0)
        if n_r > 0 and n_c > 0:
            row_idx, col_idx = np.nonzero(np.asarray(raw_data))
            valid_pairs = {
                (all_cells_in_data[ri], all_sigs_in_data[ci])
                for ri, ci in zip(row_idx, col_idx)
                if ri < len(all_cells_in_data) and ci < len(all_sigs_in_data)
            }

    # -- merge W x H contributions ---------------------------------------------
    # R: merge(data1, data2, by.x="Pattern", by.y="Pattern")
    #    data3$Contribution <- data3$Contribution.x * data3$Contribution.y
    #    data3 <- data3[, c("CellGroup","Signaling","Contribution")]
    #    data3 <- data3 %>% group_by(id) %>% top_n(1, Contribution)
    merged = pd.merge(data_cell, data_sig, on='Pattern', suffixes=('_cell', '_sig'))
    merged['Contribution'] = merged['Contribution_cell'] * merged['Contribution_sig']
    merged = merged[['CellGroup', 'Signaling', 'Contribution']]

    # Deduplicate: keep highest contribution per (CellGroup, Signaling) pair
    merged = merged.groupby(
        ['CellGroup', 'Signaling'], as_index=False, observed=False
    )['Contribution'].max()

    # Zero out pairs not present in the raw data (R's data12 / data312 check)
    if valid_pairs is not None:
        mask = pd.MultiIndex.from_frame(
            merged[['CellGroup', 'Signaling']]
        ).isin(pd.MultiIndex.from_tuples(valid_pairs))
        merged.loc[~mask, 'Contribution'] = 0.0

    # Set zero contributions to NaN (R: data3$Contribution[== 0] <- NA)
    merged.loc[merged['Contribution'] == 0, 'Contribution'] = np.nan

    if pathway_show is not None:
        merged = merged[merged['Signaling'].isin(pathway_show)]
    if group_show is not None:
        merged = merged[merged['CellGroup'].isin(group_show)]

    # Drop all-NaN rows
    merged = merged.dropna(subset=['Contribution'])
    if merged.empty:
        warnings.warn(f"No data above cutoff for dot plot (pattern='{pattern}')")
        return None

    # -- axis ordering ---------------------------------------------------------
    available_groups  = _ordered_levels(merged['CellGroup'])
    available_signals = _ordered_levels(merged['Signaling'])

    if group_show is not None:
        all_groups = [g for g in group_show if g in available_groups]
    else:
        group_order = _ordered_levels(data_cell['CellGroup'])
        all_groups = [g for g in group_order if g in available_groups]
        all_groups += [g for g in available_groups if g not in all_groups]

    if pathway_show is not None:
        all_signals = [s for s in pathway_show if s in available_signals]
    else:
        signal_order = _ordered_levels(data_sig['Signaling'])
        all_signals = [s for s in signal_order if s in available_signals]
        all_signals += [s for s in available_signals if s not in all_signals]

    n_groups  = len(all_groups)
    n_signals = len(all_signals)

    if fig_size is None:
        w = max(6, n_signals * 0.55 + 2.5)
        h = max(4, n_groups  * 0.45 + 1.5)
        fig_size = (w, h)

    # -- scale dot sizes to match R scale_size_continuous(range=dot_size) ------
    contrib_vals = merged['Contribution'].dropna()
    c_min = float(contrib_vals.min()) if len(contrib_vals) > 0 else 0.0
    c_max = float(contrib_vals.max()) if len(contrib_vals) > 0 else 1.0
    if c_max == c_min:
        c_min = c_max * 0.5 if c_max > 0 else 0.0

    # R uses geom_point with size mapped linearly; matplotlib scatter uses area (pt^2).
    # Calibrate: dot_size values are "point radius" in R ggplot units.
    # A typical ggplot size=3 <=60 pt^2 in matplotlib scatter.
    _scale = 50.0   # pt^2 per unit of dot_size
    def _s(contribution):
        r = np.interp(contribution, (c_min, c_max),
                      (dot_size[0], dot_size[1]))
        return r * r * _scale  # area = r^2 * scale

    # -- draw ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    sig_idx  = {s: i for i, s in enumerate(all_signals)}
    grp_idx  = {g: i for i, g in enumerate(all_groups)}

    for _, row in merged.iterrows():
        if pd.isna(row['Contribution']): continue
        g = row['CellGroup']; s = row['Signaling']
        if g not in grp_idx or s not in sig_idx: continue
        x = sig_idx[s];  y = grp_idx[g]
        c = color_map.get(g, '#999999')
        sz = _s(row['Contribution'])
        # R shapes 21/22 are filled circle/square with a matching outline.
        ax.scatter(x, y, s=sz,
                   facecolors=to_rgba(c, alpha=dot_alpha),
                   edgecolors=c, linewidths=0.4, marker=marker_map[shape], zorder=3)

    # -- axes ------------------------------------------------------------------
    ax.set_xticks(range(n_signals))
    ax.set_xticklabels(all_signals, rotation=45, ha='right', fontsize=font_size - 1)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(all_groups, fontsize=font_size - 1)
    ax.set_xlim(-0.5, n_signals - 0.5)
    ax.set_ylim(-0.5, n_groups  - 0.5)
    ax.invert_yaxis()   # R: scale_y_discrete(limits = rev(levels(...)))

    # -- theme_linedraw --------------------------------------------------------
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color('black'); sp.set_visible(True)
    ax.tick_params(axis='both', which='both', length=2, width=0.6)
    # light grey major grid (R: panel.grid.major = element_line(colour="grey90"))
    ax.grid(True, color='#e5e5e5', linewidth=0.6, zorder=0)

    # -- size legend (R: scale_size_continuous, guides(fill="none")) -----------
    # Show 3 representative sizes at min / mid / max contribution
    levels = [c_min, (c_min + c_max) / 2, c_max]
    handles = []
    for lv in levels:
        h = ax.scatter([], [], s=_s(lv),
                       facecolors='grey', edgecolors='grey', linewidths=0.4,
                       label=f'{lv:.2f}')
        handles.append(h)
    ax.legend(handles=handles, title='Contribution',
              title_fontsize=font_size - 1, fontsize=font_size - 2,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              frameon=False, handletextpad=1.2)

    ax.set_title(main_title, fontsize=font_size_title, fontweight='normal',
                 ha='center', pad=8)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 6.  plot_analysis_river  - alluvial / river plot for patterns
# ---------------------------------------------------------------------------

def plot_analysis_river(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    pattern: str = "outgoing",
    cutoff: float = 0.5,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    signaling: Optional[List[str]] = None,
    color_use: Optional[List[str]] = None,
    color_use_pattern: Optional[List[str]] = None,
    color_use_signaling: str = "grey50",
    do_order: bool = False,
    main_title: Optional[str] = None,
    font_size: float = 8,
    font_size_title: float = 12,
    fig_size: Tuple[int, int] = (10, 5),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Alluvial / river plot showing cell groups <-> patterns <-> signaling pathways.
    Mirrors R plot_analysis_river() with ggalluvial.

    Layout: two sub-panels on one figure (or one panel when no signaling data).
      Left panel:  Cell groups -> Patterns
      Right panel: Patterns -> Signaling

    Each panel has two stacked-bar columns connected by S-curve Bezier ribbons.
    Strata are separated by a small gap. Labels sit inside strata (or beside
    them when too narrow). Ribbons tile strata exactly (no overlap, no gap).
    """
    net_data = _network_view_for_visualization(cellchat, slot_name)

    if 'pattern' not in net_data:
        raise ValueError(
            f"No pattern data in {slot_name}. Run identify_communication_patterns first.")

    pattern_data = net_data['pattern'].get(pattern)
    if pattern_data is None:
        raise ValueError(f"Pattern '{pattern}' not found")
    if 'pattern' in pattern_data:
        pattern_data = {**pattern_data, **pattern_data['pattern']}

    data_cell = pattern_data.get('cell').copy()
    data_sig  = pattern_data.get('signaling')
    if data_sig is not None:
        data_sig = data_sig.copy()

    if main_title is None:
        main_title = ("Outgoing communication patterns of secreting cells"
                      if pattern == "outgoing"
                      else "Incoming communication patterns of target cells")

    # -- apply cutoff and optional source/target filter (mirrors R behaviour) ---
    # R: data1$Contribution[data1$Contribution < cutoff] <- 0
    # R does NOT drop rows with 0 contribution -cell groups with all-zero
    # contributions still appear (with width 0) in ggalluvial strata.
    # We keep all cell groups that have at least one non-zero contribution,
    # matching R's effective behaviour (ggalluvial silently drops pure-zero strata).
    data_cell.loc[data_cell['Contribution'] < cutoff, 'Contribution'] = 0.0
    if sources_use is not None:
        data_cell = data_cell[data_cell['CellGroup'].isin(sources_use)]
    if targets_use is not None:
        data_cell = data_cell[data_cell['CellGroup'].isin(targets_use)]
    # keep all cell groups that have any non-zero contribution
    active_groups = _ordered_levels(data_cell.loc[data_cell['Contribution'] > 0, 'CellGroup'])
    data_cell = data_cell[data_cell['CellGroup'].isin(active_groups)].copy()

    if data_cell.empty:
        warnings.warn("No data above cutoff for river plot")
        return None

    # -- Cell-group order -> stable cell-group colours --------------------------
    try:
        groups = cellchat.groups
        group_order = list(groups.categories) if isinstance(groups, pd.Categorical) else []
    except Exception:
        group_order = []

    raw_cell_groups = _ordered_levels(data_cell['CellGroup'])
    if group_order:
        cell_groups  = [g for g in group_order if g in raw_cell_groups]
        cell_groups += [g for g in raw_cell_groups if g not in cell_groups]
    else:
        cell_groups = raw_cell_groups

    # Natural sort of pattern labels: Pattern 1, Pattern 2 ... or Pattern_1 ...
    def _pat_sort_key(s):
        import re
        m = re.search(r'(\d+)$', s)
        return int(m.group(1)) if m else s
    patterns_list = _ordered_levels(data_cell['Pattern'])
    patterns_list = sorted(patterns_list, key=_pat_sort_key)
    n_patterns    = len(patterns_list)

    def _cluster_order(frame: pd.DataFrame, item_col: str, feature_col: str,
                       base_order: List[str]) -> List[str]:
        """Order active strata by average-linkage clustering of contributions."""
        if not do_order or len(base_order) < 2:
            return base_order
        matrix = frame.pivot_table(
            index=item_col, columns=feature_col, values='Contribution',
            aggfunc='sum', fill_value=0.0, observed=False
        ).reindex(base_order, fill_value=0.0)
        if matrix.shape[0] < 2 or matrix.shape[1] == 0:
            return base_order
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            return [base_order[i] for i in leaves_list(linkage(matrix.to_numpy(), method='average'))]
        except ValueError:
            return base_order

    cell_groups = _cluster_order(data_cell, 'CellGroup', 'Pattern', cell_groups)

    # -- colours ----------------------------------------------------------------
    if color_use is None and group_order:
        all_pal     = sc_palette(len(group_order))
        colors_cell = [all_pal[group_order.index(g)] if g in group_order
                       else '#999999' for g in cell_groups]
    else:
        colors_cell = _resolve_colors(len(cell_groups), color_use)
    color_cell_map = dict(zip(cell_groups, colors_cell))

    if color_use_pattern is None:
        pal = gg_palette(n_patterns * 2)
        color_use_pattern = (pal[::2][:n_patterns] if pattern == "outgoing"
                             else pal[1::2][:n_patterns])
    else:
        color_use_pattern = _resolve_colors(n_patterns, color_use_pattern)
    color_pat_map = dict(zip(patterns_list, color_use_pattern))

    has_signaling = (data_sig is not None)

    # -- prepare signaling data -------------------------------------------------
    sig_names     = []
    sig_color_map = {}
    ps_links      = {}
    pat_totals_r  = {}
    sig_totals    = {}

    if has_signaling:
        data_sig = data_sig.copy()
        data_sig.loc[data_sig['Contribution'] < cutoff, 'Contribution'] = 0.0
        if signaling is not None:
            data_sig = data_sig[data_sig['Signaling'].isin(signaling)]
        data_sig = data_sig[data_sig['Contribution'] > 0].copy()

        if data_sig.empty:
            has_signaling = False
        else:
            # patterns in the signaling data may differ from cell-cutoff patterns
            patterns_sig = _ordered_levels(data_sig['Pattern'])
            patterns_sig = sorted(patterns_sig, key=_pat_sort_key)
            sig_names = _ordered_levels(data_sig['Signaling'])
            sig_names = _cluster_order(data_sig, 'Signaling', 'Pattern', sig_names)
            n_sigs    = len(sig_names)
            # extend color_pat_map to any new patterns in sig
            for p in patterns_sig:
                if p not in color_pat_map:
                    idx = len(color_pat_map) % len(color_use_pattern)
                    color_pat_map[p] = color_use_pattern[idx]
            if isinstance(color_use_signaling, str):
                # map R colour name / hex
                try:
                    from matplotlib.colors import to_hex
                    _sc = to_hex(color_use_signaling)
                except Exception:
                    _sc = '#808080'
                sig_color_map = {s: _sc for s in sig_names}
            else:
                sig_color_map = dict(zip(sig_names,
                                         list(color_use_signaling) +
                                         ['#808080'] * n_sigs))
            pat_totals_r = {p: data_sig[data_sig['Pattern'] == p]['Contribution'].sum()
                            for p in patterns_sig}
            sig_totals   = {s: data_sig[data_sig['Signaling'] == s]['Contribution'].sum()
                            for s in sig_names}
            for _, row in data_sig.iterrows():
                key = (row['Pattern'], row['Signaling'])
                ps_links[key] = ps_links.get(key, 0.0) + row['Contribution']

    # -------------------------------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------------------------------

    # Layout constants (in axis-coordinate units, x in [0,1])
    stratum_width = 0.12
    stratum_gap = 0.008
    horizontal_padding = 0.18

    def _bezier_ribbon(ax, x_l0, x_l1, x_r0, x_r1,
                       yl_bot, yl_top, yr_bot, yr_top, color):
        """
        Draw an S-curve filled ribbon connecting two strata.
        (x_l0..x_l1) is the x-span of the left stratum,
        (x_r0..x_r1) is the x-span of the right stratum.
        yl_bot/yl_top = y range of this flow on the left stratum edge (x_l1).
        yr_bot/yr_top = y range of this flow on the right stratum edge (x_r0).
        """
        # Control-point x: midway between right edge of left col and left edge of right col
        cx = (x_l1 + x_r0) / 2.0
        verts = [
            (x_l1, yl_bot),   # MOVETO  -bottom-left anchor
            (cx,   yl_bot),   # CURVE4 ctrl1
            (cx,   yr_bot),   # CURVE4 ctrl2
            (x_r0, yr_bot),   # CURVE4  -bottom-right anchor
            (x_r0, yr_top),   # LINETO  -up right edge
            (cx,   yr_top),   # CURVE4 ctrl1
            (cx,   yl_top),   # CURVE4 ctrl2
            (x_l1, yl_top),   # CURVE4  -top-left anchor
            (x_l1, yl_bot),   # CLOSEPOLY
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(Path(verts, codes),
                               facecolor=color, edgecolor='none',
                               alpha=0.8, zorder=1))

    def _draw_alluvial_panel(ax, left_items, right_items,
                             left_totals, right_totals, links,
                             left_colors, right_colors,
                             x_left=0.2, x_right=0.8,
                             left_label="", right_label="",
                             flow_colors=None):
        """
        Draw one alluvial panel.
        Strata are stacked bottom-to-top with small gaps between them.
        Ribbons connect left strata to right strata proportionally.
        flow_colors: optional dict {left_item: color} for ribbon fill;
                     defaults to left_colors with reduced alpha.
        """
        # filter to items that have positive totals
        left_items  = [i for i in left_items  if left_totals.get(i, 0) > 0]
        right_items = [i for i in right_items if right_totals.get(i, 0) > 0]
        if not left_items or not right_items:
            return

        total_l = sum(left_totals[i] for i in left_items)
        total_r = sum(right_totals[i] for i in right_items)
        if total_l == 0 or total_r == 0:
            return

        n_l = len(left_items);  n_r = len(right_items)

        # Total gap space consumed (n-1 gaps between n strata)
        gap_total_l = stratum_gap * (n_l - 1)
        gap_total_r = stratum_gap * (n_r - 1)
        usable_l    = 1.0 - gap_total_l
        usable_r    = 1.0 - gap_total_r

        # Build strata: y_bot, y_top for each item (bottom-to-top)
        def _strata(items, totals, total, gap_total, usable):
            boxes = {}
            y = 0.0
            # ggalluvial displays the first factor level at the top; matplotlib
            # y coordinates grow upward, so stack in reverse order.
            for item in reversed(items):
                h = (totals[item] / total) * usable
                boxes[item] = (y, y + h)
                y += h + stratum_gap
            return boxes

        boxes_l = _strata(left_items,  left_totals,  total_l, gap_total_l, usable_l)
        boxes_r = _strata(right_items, right_totals, total_r, gap_total_r, usable_r)

        x_l0 = x_left;        x_l1 = x_left + stratum_width
        x_r0 = x_right - stratum_width; x_r1 = x_right

        # -- draw ribbons first (zorder=1, under strata) -----------------------
        # cursor tracks filled height within each stratum
        cur_l = {item: boxes_l[item][0] for item in left_items}
        cur_r = {item: boxes_r[item][0] for item in right_items}

        def _lpos(i): return left_items.index(i)  if i in left_items  else 999
        def _rpos(i): return right_items.index(i) if i in right_items else 999

        for (li, ri), val in sorted(links.items(),
                                    key=lambda kv: (_lpos(kv[0][0]), _rpos(kv[0][1]))):
            if val <= 0: continue
            if li not in boxes_l or ri not in boxes_r: continue

            l_h = boxes_l[li][1] - boxes_l[li][0]   # stratum height on left
            r_h = boxes_r[ri][1] - boxes_r[ri][0]   # stratum height on right
            l_tot = left_totals.get(li, 0)
            r_tot = right_totals.get(ri, 0)
            if l_tot == 0 or r_tot == 0: continue

            fh_l = (val / l_tot) * l_h
            fh_r = (val / r_tot) * r_h

            yl_bot = cur_l[li];  yl_top = yl_bot + fh_l
            yr_bot = cur_r[ri];  yr_top = yr_bot + fh_r
            cur_l[li] = yl_top
            cur_r[ri] = yr_top

            fc = (flow_colors or left_colors).get(li, '#aaaaaa')
            _bezier_ribbon(ax, x_l0, x_l1, x_r0, x_r1,
                           yl_bot, yl_top, yr_bot, yr_top, fc)

        # -- draw strata on top (zorder=3) --------------------------------------
        def _draw_strata(items, boxes, colors_map, x0, label_side='left'):
            for item in items:
                yb, yt = boxes[item]
                h = yt - yb
                c = colors_map.get(item, '#999999')
                ax.add_patch(mpatches.Rectangle(
                    (x0, yb), stratum_width, h,
                    linewidth=0.5, edgecolor='black',
                    facecolor=c, alpha=0.85, zorder=3))
                # label: inside if tall enough, otherwise outside
                mid_y = yb + h / 2
                if h >= 0.04:
                    ax.text(x0 + stratum_width / 2, mid_y, item,
                            ha='center', va='center',
                            fontsize=font_size, clip_on=False, zorder=5)
                else:
                    # place label to the outside of this column
                    if label_side == 'left':
                        ax.text(x0 - 0.02, mid_y, item,
                                ha='right', va='center',
                                fontsize=font_size, clip_on=False, zorder=5)
                    else:
                        ax.text(x0 + stratum_width + 0.02, mid_y, item,
                                ha='left', va='center',
                                fontsize=font_size, clip_on=False, zorder=5)

        _draw_strata(left_items,  boxes_l, left_colors,  x_l0, label_side='left')
        _draw_strata(right_items, boxes_r, right_colors, x_r0, label_side='right')

        # -- column axis labels -------------------------------------------------
        ax.text((x_l0 + x_l1) / 2, -0.06, left_label,
                ha='center', va='top', fontsize=font_size + 1)
        ax.text((x_r0 + x_r1) / 2, -0.06, right_label,
                ha='center', va='top', fontsize=font_size + 1)

    # -------------------------------------------------------------------------
    # Figure layout: one wide Axes containing both sub-panels
    # R uses cowplot::plot_grid(gg1, gg2, align="h", nrow=1).
    # We replicate this with a single figure and two invisible sub-axes, OR
    # simply draw both panels side-by-side on a single axis using an x range
    # of [0, 2] so the left panel occupies [0,1] and the right [1,2].
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(-horizontal_padding, (2.0 if has_signaling else 1.0) + horizontal_padding)
    ax.set_ylim(-0.12, 1.08)
    ax.axis('off')

    # cell_totals and pat_totals for left panel
    cell_totals  = {g: data_cell[data_cell['CellGroup'] == g]['Contribution'].sum()
                    for g in cell_groups}
    pat_totals_l = {p: data_cell[data_cell['Pattern']   == p]['Contribution'].sum()
                    for p in patterns_list}

    cp_links = {}
    for _, row in data_cell.iterrows():
        key = (row['CellGroup'], row['Pattern'])
        cp_links[key] = cp_links.get(key, 0.0) + row['Contribution']

    if has_signaling:
        # Left panel: x in [0.05, 0.95]; Right panel: x in [1.05, 1.95]
        _draw_alluvial_panel(ax,
                             left_items=cell_groups,  right_items=patterns_list,
                             left_totals=cell_totals, right_totals=pat_totals_l,
                             links=cp_links,
                             left_colors=color_cell_map, right_colors=color_pat_map,
                             x_left=0.05, x_right=0.95,
                             left_label="Cell groups", right_label="Patterns",
                             flow_colors=color_cell_map)

        _draw_alluvial_panel(ax,
                             left_items=patterns_sig, right_items=sig_names,
                             left_totals=pat_totals_r, right_totals=sig_totals,
                             links=ps_links,
                             left_colors=color_pat_map, right_colors=sig_color_map,
                             x_left=1.05, x_right=1.95,
                             left_label="Patterns", right_label="Signaling",
                             flow_colors=color_pat_map)
    else:
        _draw_alluvial_panel(ax,
                             left_items=cell_groups,  right_items=patterns_list,
                             left_totals=cell_totals, right_totals=pat_totals_l,
                             links=cp_links,
                             left_colors=color_cell_map, right_colors=color_pat_map,
                             x_left=0.1, x_right=0.9,
                             left_label="Cell groups", right_label="Patterns",
                             flow_colors=color_cell_map)

    ax.set_title(main_title, fontsize=font_size_title, fontweight='normal', pad=10)
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None

def plot_network_embedding(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    emb_type: str = "functional",
    color_use: Optional[List[str]] = None,
    pathway_labeled: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 6),
    label_size: float = 2.0,
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
    do_label: bool = True,
    show_legend: bool = True,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    2D manifold / embedding visualisation of signaling pathways.
    Mirrors R plot_network_embedding().

    Labels are repelled away from their dot and from each other (force-directed,
    pure-numpy, no extra dependency).
    Axes: bottom + left lines only, no tick marks, no tick labels.

    Requires that cellchat.pathway_network['similarity'][type]['dr'] has been computed.
    """
    net_data = _network_view_for_visualization(cellchat, slot_name)

    sim    = net_data.get('similarity', {}).get(emb_type, {})
    dr     = sim.get('dr', {}).get('single', None)
    groups = sim.get('group', {}).get('single', None)
    prob = net_data.get('prob')

    if dr is None:
        raise ValueError(
            f"No embedding found for type='{emb_type}'. "
            "Run compute_network_similarity(), then embed_network() first. "
            "Use plot_network_embedding_pairwise() for merged comparisons."
        )

    dr = np.asarray(dr, dtype=float)
    if dr.ndim != 2 or dr.shape[1] < 2:
        raise ValueError("Embedding coordinates must have shape (n_pathways, >=2)")
    n_pts = dr.shape[0]
    full_pathways = network_names(net_data)
    pathway_names_list = list(sim.get('pathways', full_pathways[:n_pts]))
    if len(pathway_names_list) != n_pts:
        pathway_names_list = (pathway_names_list + [f"P{i}" for i in range(len(pathway_names_list), n_pts)])[:n_pts]

    if groups is None:
        groups = ['unknown'] * n_pts
    groups = list(groups)
    if len(groups) != n_pts:
        raise ValueError(
            f"Embedding groups must contain one label per pathway ({n_pts}); "
            f"received {len(groups)}"
        )

    unique_groups  = sorted(set(groups))
    gg_colors      = gg_palette(len(unique_groups))
    resolved       = _resolve_colors(len(unique_groups), color_use)
    group_color_map = dict(zip(unique_groups,
                               gg_colors if color_use is None else resolved))

    # dot size proportional to total communication probability
    if isinstance(prob, dict):
        prob_sum = np.array([
            float(prob[pathway].sum()) if sparse.issparse(prob[pathway])
            else float(np.asarray(prob[pathway]).sum())
            if pathway in prob else 0.0
            for pathway in pathway_names_list
        ])
        prob_norm = prob_sum / prob_sum.max() if prob_sum.max() > 0 else np.ones(n_pts)
    else:
        prob_norm = np.ones(n_pts)

    fig, ax = plt.subplots(figsize=fig_size)

    for g in unique_groups:
        idx   = [i for i, grp in enumerate(groups) if grp == g]
        xs    = dr[idx, 0]
        ys    = dr[idx, 1]
        pn    = prob_norm[idx]
        sizes = np.interp(pn, (0, 1), dot_size) ** 2 * 5
        c     = group_color_map[g]
        ax.scatter(xs, ys, s=sizes,
                   c=[to_rgba(c, alpha=dot_alpha)] * len(xs),
                   edgecolors=c, linewidths=0.5, label=g, zorder=3)

    # -- repelled labels --------------------------------------------------------
    if do_label:
        labels_to_show = (set(pathway_labeled) if pathway_labeled
                          else set(pathway_names_list))

        # collect (point_xy, label) pairs
        label_pts  = [(dr[i], nm)
                      for i, nm in enumerate(pathway_names_list)
                      if nm in labels_to_show]

        if label_pts:
            # data ->display coordinates for repulsion computation
            fig.canvas.draw()
            trans  = ax.transData
            inv    = ax.transData.inverted()

            # initial label positions = point position
            lxy = np.array([p for p, _ in label_pts], dtype=float)  # (N, 2) data coords
            lxy_disp = trans.transform(lxy)                           # display coords

            pt_disp  = trans.transform(np.array([p for p, _ in label_pts]))
            offset   = np.full_like(lxy_disp, [6.0, 6.0])            # initial offset px

            # simple force-directed repulsion in display space
            n_lab = len(label_pts)
            for _ in range(120):
                for a in range(n_lab):
                    pos_a = lxy_disp[a] + offset[a]
                    force = np.zeros(2)
                    # repel from dots
                    for b in range(n_lab):
                        diff = pos_a - pt_disp[b]
                        d    = np.linalg.norm(diff)
                        if d < 1e-3:
                            d = 1e-3
                        if d < 30:
                            force += diff / (d ** 2) * 200
                    # repel from other labels
                    for b in range(n_lab):
                        if b == a:
                            continue
                        pos_b = lxy_disp[b] + offset[b]
                        diff  = pos_a - pos_b
                        d     = np.linalg.norm(diff)
                        if d < 1e-3:
                            d = 1e-3
                        if d < 40:
                            force += diff / (d ** 2) * 300
                    # attract back towards own dot
                    spring = pt_disp[a] - pos_a
                    d_s    = np.linalg.norm(spring)
                    force += spring * 0.04 * max(d_s - 8, 0)

                    step   = force * 0.15
                    step   = np.clip(step, -4, 4)
                    offset[a] += step

            for (pt_xy, nm), off in zip(label_pts, offset):
                px, py = trans.transform(pt_xy)
                lx, ly = inv.transform([px + off[0], py + off[1]])
                ax.annotate(
                    nm,
                    xy=pt_xy,
                    xytext=(lx, ly),
                    fontsize=max(label_size * 4, 6),
                    alpha=0.85,
                    arrowprops=None,
                    zorder=5,
                )

    # -- axis style: bottom + left lines only, no ticks ------------------------
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title or f"Signaling Network Embedding ({emb_type})",
                 fontsize=font_size_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    # no tick marks, no tick labels
    ax.tick_params(axis='both', which='both',
                   length=0, labelbottom=False, labelleft=False)

    if show_legend:
        ax.legend(title='Group', fontsize=font_size - 2, title_fontsize=font_size - 1,
                  markerscale=0.8, frameon=False)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 8.  plot_network_rank  - ranked bar charts (pathway / cell)
# ---------------------------------------------------------------------------

def plot_network_rank(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    mode: str = "all",
    stacked: bool = False,
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    fig_size: Tuple[int, int] = (12, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Ranked bar charts for pathways and/or cell types.
    Mirrors R plot_analysis_river / ranked bar idea in R.

    mode : 'pathway' | 'cell' | 'all'
    """
    if mode not in {'pathway', 'cell', 'all'}:
        raise ValueError("mode must be 'pathway', 'cell', or 'all'")
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)

    net_data = _network_view_for_visualization(cellchat, slot_name)

    if 'prob' not in net_data:
        raise ValueError("No probability data found")

    pathway_names, prob_array, pval_array = _network_arrays(net_data)
    prob_f = prob_array.copy()
    prob_f[pval_array >= thresh] = 0.0

    if mode == "pathway":
        scores = np.sum(prob_f, axis=(0, 1))
        df = pd.DataFrame({'pathway': pathway_names, 'score': scores})
        df = df.sort_values('score', ascending=True)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.barh(df['pathway'], df['score'], color='#4DAF4A', edgecolor='white')
        ax.set_xlabel('Total Communication Score', fontsize=10)
        ax.set_title('Ranked Communication Pathways', fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    elif mode == "cell":
        out_degree = np.sum(prob_f, axis=(1, 2))
        in_degree = np.sum(prob_f, axis=(0, 2))
        df = pd.DataFrame({'cell_type': cluster_names,
                           'outgoing': out_degree, 'incoming': in_degree})
        df = df.sort_values('outgoing', ascending=True)
        cell_colors = _resolve_colors(n_clusters, color_use)
        c_map = dict(zip(cluster_names, cell_colors))

        fig, ax = plt.subplots(figsize=fig_size)
        bar_c = [c_map.get(ct, '#999999') for ct in df['cell_type']]
        if stacked:
            ax.barh(df['cell_type'], df['outgoing'], color=bar_c, label='Outgoing')
            ax.barh(df['cell_type'], df['incoming'], left=df['outgoing'],
                    color=[to_rgba(c, 0.5) for c in bar_c], label='Incoming')
        else:
            x = np.arange(len(df))
            ax.barh(x - 0.2, df['outgoing'], height=0.4, color=bar_c, label='Outgoing')
            ax.barh(x + 0.2, df['incoming'], height=0.4,
                    color=[to_rgba(c, 0.55) for c in bar_c], label='Incoming')
            ax.set_yticks(x)
            ax.set_yticklabels(df['cell_type'], fontsize=9)
        ax.set_xlabel('Communication Score', fontsize=10)
        ax.set_title('Cell Type Communication Ranking', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    else:  # all
        pathway_scores = np.sum(prob_f, axis=(0, 1))
        out_degree = np.sum(prob_f, axis=(1, 2))

        pathway_df = pd.DataFrame({'pathway': pathway_names, 'score': pathway_scores})
        pathway_df = pathway_df.sort_values('score', ascending=True)

        cell_df = pd.DataFrame({'cell_type': cluster_names, 'score': out_degree})
        cell_df = cell_df.sort_values('score', ascending=True)
        cell_colors_sorted = [colors[cluster_names.index(ct)] for ct in cell_df['cell_type']]

        fig, axes = plt.subplots(1, 2, figsize=fig_size)

        axes[0].barh(pathway_df['pathway'], pathway_df['score'],
                     color='#4DAF4A', edgecolor='white')
        axes[0].set_xlabel('Total Communication Score', fontsize=9)
        axes[0].set_title('Pathway Ranking', fontsize=10, fontweight='bold')
        axes[0].spines[['top', 'right']].set_visible(False)

        axes[1].barh(cell_df['cell_type'], cell_df['score'],
                     color=cell_colors_sorted, edgecolor='white')
        axes[1].set_xlabel('Total Communication Score', fontsize=9)
        axes[1].set_title('Cell Type Ranking (Outgoing)', fontsize=10, fontweight='bold')
        axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 9.  plot_network  - general spring/circular network
# ---------------------------------------------------------------------------

def plot_network_layout(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    layout: str = "spring",
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    fig_size: Tuple[int, int] = (10, 8),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    General network layout plot (spring / circular / random).
    Mirrors R plot_network_aggregate() with non-circle layout.
    """
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    net_data = _network_view_for_visualization(cellchat, slot_name)
    if 'prob' not in net_data:
        raise ValueError("No probability data found")

    _, prob_array, pval_array = _network_arrays(net_data)
    prob_f = prob_array.copy()
    prob_f[pval_array >= thresh] = 0.0

    agg = np.sum(prob_f, axis=2) if prob_f.ndim == 3 else prob_f

    G = nx.DiGraph()
    for name in cluster_names:
        G.add_node(name)
    for i, src in enumerate(cluster_names):
        for j, tgt in enumerate(cluster_names):
            if i != j and agg[i, j] > 0:
                G.add_edge(src, tgt, weight=float(agg[i, j]))

    if G.number_of_edges() == 0:
        warnings.warn("No interactions found")
        return None

    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)

    node_weights = []
    for node in G.nodes():
        total = (sum(G[u][v]['weight'] for u, v in G.out_edges(node)) +
                 sum(G[u][v]['weight'] for u, v in G.in_edges(node)))
        node_weights.append(total)

    if max(node_weights) > min(node_weights):
        node_sizes = list(np.interp(node_weights, (min(node_weights), max(node_weights)), (300, 1500)))
    else:
        node_sizes = [600] * len(node_weights)

    edge_ws = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = list(np.interp(edge_ws, (min(edge_ws), max(edge_ws)), (0.5, 5)))
    edge_colors = [to_rgba(color_map[u], 0.6) for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=fig_size)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=[color_map[n] for n in G.nodes()],
                           edgecolors=[color_map[n] for n in G.nodes()],
                           linewidths=1, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                           arrows=True, arrowsize=15,
                           connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title(f'Cell-Cell Communication Network ({layout.title()} Layout)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 10.  plot_network_aggregate - mirrors R plot_network_aggregate (hierarchy/circle/chord)
# ---------------------------------------------------------------------------

def plot_network_aggregate(
    cellchat: 'CellChat',
    signaling: Optional[Union[str, List[str]]] = None,
    layout: str = "hierarchy",
    vertex_receiver: Optional[List[int]] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    top: float = 1.0,
    title_name: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
    vertex_weight: float = 1.0,
    vertex_size_max: float = 15.0,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    vertex_label_cex: float = 0.8,
) -> Optional[plt.Figure]:
    """
    Aggregate visualization of cell-cell communication at signaling pathway level.
    Mirrors R plot_network_aggregate() with layouts: hierarchy, circle, chord.
    """
    # All matrices below are indexed by the selected network's explicit group
    # order.  It is independent from the order of cellchat.groups.categories.
    cluster_names = None
    n_clusters = 0
    signaling_names = None if signaling is None else (
        [signaling] if isinstance(signaling, str) else list(signaling)
    )

    # R selects from the L-R layer before aggregation when pathway annotations
    # are available.
    lr_data = _network_view_for_visualization(cellchat, "network")
    interactions = lr_data.get('interactions')
    use_lr_data = (
        signaling_names is not None
        and isinstance(lr_data.get('prob'), dict)
        and isinstance(interactions, pd.DataFrame)
        and 'pathway_name' in interactions.columns
    )

    if use_lr_data:
        cluster_names = list(lr_data.get("groups", []))
        pair_names, prob_array, pval_array = _network_arrays(lr_data)
        if 'interaction_name' in interactions.columns:
            pathway_by_pair = dict(zip(
                interactions['interaction_name'].astype(str),
                interactions['pathway_name'].astype(str),
            ))
        elif len(interactions) == len(pair_names):
            pathway_by_pair = {
                pair_names[index]: str(interactions.iloc[index]['pathway_name'])
                for index in range(len(pair_names))
            }
        else:
            raise ValueError("Network interactions are not aligned with the ligand-receptor matrices")
        selected_indices = [
            index for index, pair_name in enumerate(pair_names)
            if pathway_by_pair.get(str(pair_name)) in set(map(str, signaling_names))
        ]
        if not selected_indices:
            raise ValueError("None of the requested signaling pathways were found in net")
    else:
        net_data = _network_view_for_visualization(cellchat, "pathway_network")
        cluster_names = list(net_data.get("groups", []))
        if 'prob' not in net_data:
            raise ValueError("No probability data found in pathway_network")
        pathways, prob_array, pval_array = _network_arrays(net_data)
        if signaling_names is None:
            selected_indices = list(range(prob_array.shape[2]))
        else:
            selected_indices = [index for index, name in enumerate(pathways) if name in signaling_names]
        if not selected_indices:
            raise ValueError("None of the requested signaling pathways were found")

    if not cluster_names:
        raise ValueError("The selected network must define its group order")
    n_clusters = len(cluster_names)

    if pval_array.shape != prob_array.shape:
        raise ValueError("Probability and p-value tensors must have the same shape")
    prob_selected = prob_array[:, :, selected_indices].copy()
    # Match R netVisual_aggregate: only p-values strictly greater than the
    # threshold are removed, so an edge with pval == thresh is retained.
    prob_selected[pval_array[:, :, selected_indices] > thresh] = 0.0
    net_mat = np.sum(prob_selected, axis=2)
    net_mat = _filter_hierarchy_matrix(
        net_mat, cluster_names, sources_use, targets_use, top
    )

    if layout == "circle":
        return plot_network_circle(
            cellchat, net_matrix=net_mat, group_names=cluster_names,
            title_name=title_name,
            color_use=color_use, edge_weight_max=edge_weight_max,
            edge_width_max=edge_width_max, vertex_weight=vertex_weight,
            vertex_size_max=vertex_size_max, vertex_label_cex=vertex_label_cex,
            fig_size=fig_size, return_fig=return_fig
        )
    elif layout == "chord":
        # Use chord function if available, otherwise fall back to circle
        try:
            return plot_network_chord_cell(
                cellchat, net_matrix=net_mat, group_names=cluster_names,
                title_name=title_name,
                color_use=color_use, fig_size=fig_size, return_fig=return_fig
            )
        except:
            warnings.warn("Chord layout not fully implemented, using circle layout")
            return plot_network_circle(
                cellchat, net_matrix=net_mat, group_names=cluster_names,
                title_name=title_name,
                color_use=color_use, fig_size=fig_size, return_fig=return_fig
            )
    elif layout == "hierarchy":
        if vertex_receiver is None:
            raise ValueError("vertex_receiver is required for hierarchy layout")
        receivers = _validate_hierarchy_receivers(vertex_receiver, n_clusters)
        if not np.any(net_mat > 0):
            warnings.warn("No significant hierarchy interactions found")
            return None
        figure = _draw_hierarchy_grid(
            [net_mat], ["Aggregate"], cluster_names, receivers, color_use,
            vertex_weight, vertex_size_max, edge_width_max,
            None, edge_weight_max, vertex_label_cex, fig_size[1], title_name,
        )
        if return_fig:
            return figure
        plt.show()
        return None
    else:
        raise ValueError("layout must be 'circle', 'chord', or 'hierarchy'")


# ---------------------------------------------------------------------------
# 11.  plot_network_individual - for individual L-R pairs
# ---------------------------------------------------------------------------

def plot_network_individual(
    cellchat: 'CellChat',
    signaling: str,
    pair_lr_use: Optional[List[str]] = None,
    layout: str = "hierarchy",
    vertex_receiver: Optional[List[int]] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    title_name: Optional[str] = None,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Visualize cell-cell communication for individual L-R pairs.
    Mirrors R plot_network_individual().
    """
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    net_data = _network_view_for_visualization(cellchat, "network")

    if 'prob' not in net_data:
        raise ValueError("No probability data found in cellchat.network.")

    lr_names, prob_array, pval_array = _network_arrays(net_data)

    # R plot_network_individual() first restricts the L-R axis to the requested
    # pathway, then applies its p-value filter to each retained L-R pair.
    if pair_lr_use is not None and len(pair_lr_use) > 0:
        lr_name = pair_lr_use[0] if isinstance(pair_lr_use, list) else pair_lr_use
        if lr_name not in lr_names:
            raise ValueError(f"L-R pair '{lr_name}' not found")
        lr_indices = [lr_names.index(lr_name)]
    else:
        interactions_df = net_data.get('interactions')
        if not isinstance(interactions_df, pd.DataFrame) or 'pathway_name' not in interactions_df.columns:
            raise ValueError(
                "Network interactions must contain pathway_name to select L-R pairs by signaling."
            )

        if 'interaction_name' in interactions_df.columns:
            pathway_by_lr = dict(zip(
                interactions_df['interaction_name'].astype(str),
                interactions_df['pathway_name'].astype(str),
            ))
        elif len(interactions_df) == len(lr_names):
            pathway_by_lr = {
                lr_name: str(interactions_df.iloc[index]['pathway_name'])
                for index, lr_name in enumerate(lr_names)
            }
        else:
            raise ValueError(
                "Network interactions must contain interaction_name or align with the L-R axis."
            )

        lr_indices = [
            index for index, lr_name in enumerate(lr_names)
            if pathway_by_lr.get(str(lr_name)) == signaling
        ]
        if not lr_indices:
            raise ValueError(f"No L-R pairs found for signaling pathway '{signaling}'")

    prob_selected = prob_array[:, :, lr_indices].copy()
    pval_selected = pval_array[:, :, lr_indices]
    prob_selected[pval_selected >= thresh] = 0.0
    net_mat = np.sum(prob_selected, axis=2)

    # Filter sources/targets
    if sources_use is not None:
        mask_r = [i for i, n in enumerate(cluster_names) if n not in sources_use]
        net_mat[mask_r, :] = 0
    if targets_use is not None:
        mask_c = [i for i, n in enumerate(cluster_names) if n not in targets_use]
        net_mat[:, mask_c] = 0

    if layout == "circle":
        return plot_network_circle(
            cellchat, net_matrix=net_mat, title_name=title_name,
            color_use=color_use, fig_size=fig_size, return_fig=return_fig
        )
    elif layout == "chord":
        try:
            return plot_network_chord_cell(
                cellchat, net_matrix=net_mat, title_name=title_name,
                color_use=color_use, fig_size=fig_size, return_fig=return_fig
            )
        except:
            warnings.warn("Chord layout not fully implemented, using circle layout")
            return plot_network_circle(
                cellchat, net_matrix=net_mat, title_name=title_name,
                color_use=color_use, fig_size=fig_size, return_fig=return_fig
            )
    else:  # hierarchy
        n_clusters = len(cluster_names)
        colors = _resolve_colors(n_clusters, color_use)

        if vertex_receiver is not None:
            # Reorder for hierarchy layout
            all_indices = list(range(n_clusters))
            sender_indices = [i for i in all_indices if i not in vertex_receiver]
            new_order = vertex_receiver + sender_indices
            net_mat = net_mat[np.ix_(new_order, new_order)]
            new_cluster_names = [cluster_names[i] for i in new_order]
            new_colors = [colors[i] for i in new_order]
            color_map = dict(zip(new_cluster_names, new_colors))
        else:
            color_map = dict(zip(cluster_names, colors))
            new_cluster_names = cluster_names
            new_colors = colors

        return plot_network_circle(
            cellchat, net_matrix=net_mat, title_name=title_name,
            color_use=new_colors, fig_size=fig_size, return_fig=return_fig
        )


# ---------------------------------------------------------------------------
# Chord diagram drawing engine (shared by plot_network_chord_cell and
# plot_network_chord_gene).
#
# A "chord diagram" has:
#   -An annular ring broken into one arc per sector.  Arc length is
#     proportional to that sector's total weight (outgoing + incoming,
#     minus self-loop which is counted once).
#   -Filled quadratic-Bezier ribbons connecting pairs of sectors.
#     Each ribbon touches sector i at angular span [a_i0 .. a_i0 + da_ij]
#     and sector j at [a_j0 .. a_j0 + da_ji], where da_ij / arc_i is
#     proportional to mat[i,j] / sector_weight[i].
#   -For directed flow mat[i,j] != mat[j,i] the ribbon is coloured by
#     the heavier-flow side.
#   -Labels placed radially just outside the ring.
# ---------------------------------------------------------------------------

def _chord_arc(ax, r_inner, r_outer, theta0, theta1, color, alpha=0.9, zorder=4):
    """Draw a filled arc (annular sector) between theta0 and theta1 (radians)."""
    n  = max(3, int(abs(theta1 - theta0) / (2 * np.pi) * 360) + 2)
    th = np.linspace(theta0, theta1, n)
    # outer edge then inner edge (reversed) ->closed polygon
    ox = r_outer * np.cos(th);  oy = r_outer * np.sin(th)
    ix = r_inner * np.cos(th);  iy = r_inner * np.sin(th)
    xs = np.concatenate([ox, ix[::-1]])
    ys = np.concatenate([oy, iy[::-1]])
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=zorder, linewidth=0)


def _chord_ribbon(ax, r_src, t0i, t1i, t0j, t1j, color,
                  r_tgt=None, alpha=0.4, zorder=2,
                  diff_height=0.04):
    """
    Directional arrow: a curved line from source arc midpoint to target arc
    midpoint (quadratic Bezier through origin), with an arrowhead at the target.
    """
    from matplotlib.colors import to_rgba as _to_rgba

    if r_tgt is None:
        r_tgt = r_src + diff_height

    n = 60
    t_src = (t0i + t1i) / 2
    t_tgt = (t0j + t1j) / 2

    x0, y0 = r_src * np.cos(t_src), r_src * np.sin(t_src)
    x1, y1 = r_tgt * np.cos(t_tgt), r_tgt * np.sin(t_tgt)

    # quadratic Bezier, control point = origin
    tt = np.linspace(0, 1, n)
    bx = (1 - tt)**2 * x0 + tt**2 * x1
    by = (1 - tt)**2 * y0 + tt**2 * y1

    rgba = _to_rgba(color, alpha=min(alpha * 2, 0.9))
    ax.plot(bx, by, color=rgba, linewidth=1.2, zorder=zorder, solid_capstyle='round')

    ax.annotate("", xy=(x1, y1), xytext=(bx[-4], by[-4]),
                arrowprops=dict(arrowstyle="-|>",
                                color=rgba,
                                lw=1.2,
                                mutation_scale=10),
                zorder=zorder + 1)


def _normalize_matrix_bistochastic(mat: np.ndarray,
                                    standard_sum: float = 100.0,
                                    max_iter: int = 15,
                                    tol: float = 1e-4) -> np.ndarray:
    """
    Iteratively normalise *mat* so every non-zero row-sum and column-sum
    converge to *standard_sum*.  This gives each cell-type equal arc width in
    the chord diagram, mirroring omicverse / CellChat R behaviour.
    """
    m = mat.copy().astype(float)
    for _ in range(max_iter):
        rs = m.sum(axis=1)
        for i in range(len(rs)):
            if rs[i] > tol:
                m[i, :] *= standard_sum / rs[i]
        cs = m.sum(axis=0)
        for j in range(len(cs)):
            if cs[j] > tol:
                m[:, j] *= standard_sum / cs[j]
        nz_r = rs[rs > tol]; nz_c = cs[cs > tol]
        if len(nz_r) and len(nz_c):
            if np.std(nz_r) < tol and np.std(nz_c) < tol:
                break
    return m




# ---------------------------------------------------------------------------
# 12.  plot_network_chord_cell - chord diagram for cell-cell communication
# ---------------------------------------------------------------------------

def plot_network_chord_cell(
    cellchat: 'CellChat',
    signaling: Optional[str] = None,
    net_matrix: Optional[np.ndarray] = None,
    group: Optional[Dict[str, str]] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    gap: float = 0.03,
    use_gradient: bool = True,
    directed: bool = True,
    sort: Optional[str] = "size",
    font_size: int = 12,
    fig_size: Tuple[int, int] = (8, 8),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Chord diagram for cell-cell communication using mpl_chord_diagram.

    Produces filled ribbon chords with optional gradient colouring and
    directional arrows -matching R CellChat plot_network_chord_cell output.

    Chord widths are proportional to communication probability (strength).
    Cell types with no interactions are kept as visible sectors with no
    connections, matching R's remove.isolate=FALSE behaviour.

    Parameters
    ----------
    signaling : pathway name to show (None = aggregate all)
    net_matrix : pre-built NxN matrix (overrides signaling)
    group : dict mapping cell-type ->group name for aggregation
    sources_use / targets_use : restrict which senders / receivers appear
    thresh : p-value cutoff
    gap : gap between arc segments (mpl_chord_diagram parameter)
    use_gradient : colour ribbon with source->target gradient
    directed : show arrowheads at target end
    sort : arc ordering -"size", "distance", or None
    """
    from mpl_chord_diagram import chord_diagram

    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters    = len(cluster_names)
    colors        = _resolve_colors(n_clusters, color_use)

    # -- build raw matrix ------------------------------------------------------
    if net_matrix is not None:
        mat = np.array(net_matrix, dtype=float).copy()
    else:
        net_data = _network_view_for_visualization(cellchat, "pathway_network" if signaling else "network")
        if 'prob' not in net_data:
            raise ValueError("No probability data found")
        if signaling:
            if signaling not in net_data['prob']:
                raise ValueError(f"Signaling pathway '{signaling}' not found")
            mat = _network_matrix(net_data, 'prob', signaling).copy()
            pval_matrix = _network_matrix(net_data, 'pval', signaling)
            mat[pval_matrix >= thresh] = 0.0
        else:
            _, prob_array, pval_array = _network_arrays(net_data)
            prob_array[pval_array >= thresh] = 0.0
            mat = prob_array.sum(axis=2)

    # -- source / target filter ------------------------------------------------
    if sources_use is not None:
        mask = [i for i, n in enumerate(cluster_names) if n not in sources_use]
        mat[mask, :] = 0.0
    if targets_use is not None:
        mask = [i for i, n in enumerate(cluster_names) if n not in targets_use]
        mat[:, mask] = 0.0

    # -- group aggregation -----------------------------------------------------
    if group is not None:
        seen: dict = {}
        for v in group.values():
            seen.setdefault(v, None)
        group_names  = list(seen.keys())
        n_g          = len(group_names)
        group_mat    = np.zeros((n_g, n_g))
        group_colors = _resolve_colors(n_g, color_use)
        cidx         = {name: group_names.index(group[name])
                        for name in cluster_names if name in group}
        for i in range(n_clusters):
            for j in range(n_clusters):
                ni, nj = cluster_names[i], cluster_names[j]
                if ni in cidx and nj in cidx:
                    group_mat[cidx[ni], cidx[nj]] += mat[i, j]
        mat           = group_mat
        cluster_names = group_names
        colors        = group_colors

    if mat.sum() == 0:
        warnings.warn("No interactions to draw in chord diagram")
        return None

    # Identify isolated cells (no interactions) -give them a self-loop so
    # they appear as visible sectors, then hide the self-loop chord via
    # monkey-patching chord_arc (matching R's link.visible=FALSE).
    isolated_indices = list(np.where((mat.sum(axis=1) + mat.sum(axis=0)) == 0)[0])
    if len(isolated_indices) > 0:
        degrees = mat.sum(axis=1) + mat.sum(axis=0)
        connected_degrees = degrees[degrees > 0]
        # Self-loop contributes to both row and col sum (degree = 2*value),
        # so use half the median connected degree to get comparable sector size
        median_deg = float(np.median(connected_degrees)) if len(connected_degrees) > 0 else 1.0
        fill_value = median_deg / 2.0
        for i in isolated_indices:
            mat[i, i] = fill_value

    display_names = list(cluster_names)

    # -- draw ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    # Identify which chord_arc calls correspond to isolated-cell self-loops
    # so we can hide them via a temporary monkey-patch of mpl_chord_diagram's
    # internal chord_arc function. This emulates R's link.visible=FALSE.
    skip_call_indices = set()
    if isolated_indices:
        zero_set    = set(isolated_indices)
        num_nodes   = len(mat)
        call_idx    = 0
        for i in range(num_nodes):
            targets = range(num_nodes) if directed else range(i)
            for j in targets:
                if mat[i, j] > 0 or (not directed and mat[j, i] > 0):
                    if i == j and i in zero_set:
                        skip_call_indices.add(call_idx)
                    call_idx += 1

    import sys as _sys
    _cdm_module      = _sys.modules.get('mpl_chord_diagram.chord_diagram')
    _do_patch        = bool(skip_call_indices) and _cdm_module is not None
    _original_arc    = None

    if _do_patch:
        _original_arc = _cdm_module.chord_arc
        _counter      = [0]

        def _patched_chord_arc(*args, **kwargs):
            idx = _counter[0]
            _counter[0] += 1
            if idx in skip_call_indices:
                return None
            return _original_arc(*args, **kwargs)

        _cdm_module.chord_arc = _patched_chord_arc

    try:
        chord_diagram(
            mat, display_names,
            ax=ax,
            colors=colors,
            gap=gap,
            use_gradient=use_gradient,
            directed=directed,
            sort=sort,
            fontsize=font_size,
            fontcolor="black",
            rotate_names=False,
        )
    finally:
        if _do_patch:
            _cdm_module.chord_arc = _original_arc

    if title_name:
        ax.set_title(title_name, fontsize=font_size + 4, pad=20)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 13.  plot_network_chord_gene - chord diagram for gene-level communication
# ---------------------------------------------------------------------------

def plot_network_chord_gene(
    cellchat: 'CellChat',
    sources_use: Optional[List[Union[int, str]]] = None,
    targets_use: Optional[List[Union[int, str]]] = None,
    signaling: Optional[List[str]] = None,
    slot_name: str = "network",
    network_table: Optional[pd.DataFrame] = None,
    thresh: float = 0.05,
    title_name: Optional[str] = None,
    color_use: Optional[List[str]] = None,
    gap: float = 0.03,
    use_gradient: bool = True,
    directed: bool = True,
    sort: Optional[str] = "size",
    font_size: int = 8,
    fig_size: Tuple[int, int] = (10, 10),
    show_legend: bool = True,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Chord diagram for ligand-receptor / pathway level communication.

    Uses mpl_chord_diagram for filled ribbon chords with gradient colouring
    and directional arrows.  Each sector = one (cell_group x LR_pair).

    Parameters
    ----------
    slot_name : "network" ->L-R pair level; "pathway_network" ->signaling pathway level.
    show_legend : draw a Cell State colour legend.
    gap / use_gradient / directed / sort : passed to chord_diagram().
    """
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters    = len(cluster_names)

    # resolve sources / targets (accept int index or str name)
    def _resolve_cells(use_list):
        if use_list is None:
            return list(range(n_clusters))
        if isinstance(use_list, (int, np.integer, str)):
            use_list = [use_list]
        out = []
        for v in use_list:
            if isinstance(v, (int, np.integer)):
                if 0 <= int(v) < n_clusters:
                    out.append(int(v))
            elif isinstance(v, str) and v in cluster_names:
                out.append(cluster_names.index(v))
        return out

    src_idx = _resolve_cells(sources_use)
    tgt_idx = _resolve_cells(targets_use)

    if network_table is not None:
        if not isinstance(network_table, pd.DataFrame):
            raise TypeError("network_table must be a pandas DataFrame when supplied")
        required = {'source', 'target', 'ligand', 'receptor', 'prob'}
        missing = required.difference(network_table.columns)
        if missing:
            raise ValueError(f"network_table is missing required columns: {sorted(missing)}")
        explicit = network_table.copy()
        explicit['interaction_name'] = (
            explicit.get('interaction_name',
                         explicit['ligand'].astype(str) + '_' + explicit['receptor'].astype(str))
            .astype(str)
        )
        explicit['pathway_name'] = explicit.get('pathway_name', '')
        explicit['pval'] = explicit.get('pval', 0.0)
        lr_names = explicit['interaction_name'].drop_duplicates().tolist()
        prob = np.zeros((n_clusters, n_clusters, len(lr_names)), dtype=float)
        pval = np.ones_like(prob)
        group_index = {str(name): i for i, name in enumerate(cluster_names)}
        lr_index = {name: i for i, name in enumerate(lr_names)}
        for row in explicit.itertuples(index=False):
            source, target = str(row.source), str(row.target)
            if source in group_index and target in group_index:
                i, j, k = group_index[source], group_index[target], lr_index[str(row.interaction_name)]
                prob[i, j, k] = float(row.prob)
                pval[i, j, k] = float(row.pval)
        net_data = {
            'groups': list(cluster_names),
            'prob': {
                name: sparse.csr_matrix(prob[:, :, index])
                for index, name in enumerate(lr_names)
            },
            'pval': {
                name: pval[:, :, index].copy()
                for index, name in enumerate(lr_names)
            },
            'interactions': explicit,
        }
    else:
        net_data = _network_view_for_visualization(cellchat, slot_name)
    if 'prob' not in net_data:
        raise ValueError("No probability data found")

    lr_names_full, prob_array, pval_array = _network_arrays(net_data)
    interactions_df = net_data.get('interactions', None)

    prob_f = prob_array.copy()
    prob_f[pval_array >= thresh] = 0.0

    # -- build lookup maps for ligand, receptor, pathway -----------------------
    # interactions_df is integer-indexed (positionally aligned with lr_names_full)
    ligand_map   = {}   # lr_name -> ligand gene name
    receptor_map = {}   # lr_name -> receptor gene name
    pathway_map  = {}   # lr_name -> pathway name

    if interactions_df is not None and len(interactions_df) > 0:
        # use positional lookup when the df is integer-indexed and same length
        is_positional = (len(interactions_df) == len(lr_names_full) and
                         pd.api.types.is_integer_dtype(interactions_df.index.dtype))
        for k, lr in enumerate(lr_names_full):
            if is_positional and k < len(interactions_df):
                row = interactions_df.iloc[k]
            elif lr in interactions_df.index:
                row = interactions_df.loc[lr]
            else:
                row = None
            if row is not None:
                ligand_map[lr]   = str(row.get('ligand',       lr))
                receptor_map[lr] = str(row.get('receptor',     lr))
                pathway_map[lr]  = str(row.get('pathway_name', ''))
            else:
                ligand_map[lr]   = lr
                receptor_map[lr] = lr
                pathway_map[lr]  = ''

    # -- filter to requested signaling pathways --------------------------------
    if signaling is not None:
        if slot_name == "pathway_network":
            # In pathway_network, the selected names are pathway names.
            keep_lr = [k for k, lr in enumerate(lr_names_full) if lr in signaling]
        else:
            keep_lr = [k for k, lr in enumerate(lr_names_full)
                       if pathway_map.get(lr, '') in signaling]
    else:
        keep_lr = list(range(len(lr_names_full)))

    # -- collect raw flows: (ligand_label, receptor_label, src_cell_idx, tgt_cell_idx, w) --
    flows_raw = []
    for k in keep_lr:
        lr = lr_names_full[k]
        if slot_name == "pathway_network":
            # use pathway name as sector label directly
            lig_label = lr
            rec_label = " "
        else:
            lig_label = ligand_map.get(lr, lr)
            rec_label = receptor_map.get(lr, lr)
        for i in src_idx:
            for j in tgt_idx:
                w = prob_f[i, j, k]
                if w > 0:
                    flows_raw.append((lig_label, rec_label, i, j, float(w)))

    if not flows_raw:
        warnings.warn("No significant L-R flows found for chord gene diagram")
        return None

    # -- build colour helper ---------------------------------------------------
    try:
        groups      = cellchat.groups
        group_order = list(groups.categories) if isinstance(groups, pd.Categorical) else []
    except Exception:
        ident_order = []

    all_palette = list(color_use) if color_use is not None else sc_palette(
        len(group_order) if group_order else n_clusters)

    def _cell_color(cell_idx):
        name = cluster_names[cell_idx]
        if group_order and name in group_order:
            idx = group_order.index(name)
            return all_palette[idx] if idx < len(all_palette) else '#999999'
        return all_palette[cell_idx % len(all_palette)]

    # -- deduplicate sector names (mirrors R: add trailing spaces) -------------
    # src sector key: (ligand_label, src_cell_idx)
    # tgt sector key: (receptor_label, tgt_cell_idx)
    src_sector_map = {}   # (lig, i) -> unique sector name
    for lig, rec, i, j, w in flows_raw:
        key = (lig, i)
        if key not in src_sector_map:
            base = lig
            while base in src_sector_map.values():
                base = base + ' '
            src_sector_map[key] = base

    tgt_sector_map = {}   # (rec, j) -> unique sector name
    for lig, rec, i, j, w in flows_raw:
        key = (rec, j)
        if key not in tgt_sector_map:
            base = rec
            while base in tgt_sector_map.values():
                base = base + ' '
            tgt_sector_map[key] = base

    # -- order sectors: by cell group order, then alphabetically within each cell --
    src_sectors_ordered = []
    src_cell_of = {}   # sector_name -> cell_idx
    for i in src_idx:
        cell_pairs = [(k, v) for k, v in src_sector_map.items() if k[1] == i]
        cell_pairs.sort(key=lambda x: x[1].rstrip())   # alpha by display name
        for key, sname in cell_pairs:
            src_sectors_ordered.append(sname)
            src_cell_of[sname] = i

    tgt_sectors_ordered = []
    tgt_cell_of = {}   # sector_name -> cell_idx
    for j in tgt_idx:
        cell_pairs = [(k, v) for k, v in tgt_sector_map.items() if k[1] == j]
        cell_pairs.sort(key=lambda x: x[1].rstrip())
        for key, sname in cell_pairs:
            tgt_sectors_ordered.append(sname)
            tgt_cell_of[sname] = j

    if not src_sectors_ordered or not tgt_sectors_ordered:
        warnings.warn("No sectors to draw in chord gene diagram")
        return None

    # -- map flows to sector indices -------------------------------------------
    src_name_to_idx = {v: idx for idx, v in enumerate(src_sectors_ordered)}
    tgt_name_to_idx = {v: idx for idx, v in enumerate(tgt_sectors_ordered)}

    flows_indexed = []
    for lig, rec, i, j, w in flows_raw:
        si = src_name_to_idx[src_sector_map[(lig, i)]]
        ti = tgt_name_to_idx[tgt_sector_map[(rec, j)]]
        flows_indexed.append((si, ti, w))

    # -- sector colours --------------------------------------------------------
    src_colors = [_cell_color(src_cell_of[s]) for s in src_sectors_ordered]
    tgt_colors = [_cell_color(tgt_cell_of[s]) for s in tgt_sectors_ordered]
    all_colors = src_colors + tgt_colors

    # -- build flow matrix for mpl_chord_diagram ------------------------------
    # All sectors (src then tgt) become rows/cols; flow only goes src->tgt.
    all_names  = [s.rstrip() for s in src_sectors_ordered] + \
                 [s.rstrip() for s in tgt_sectors_ordered]
    n_src_s    = len(src_sectors_ordered)
    n_all      = len(all_names)
    flow_mat   = np.zeros((n_all, n_all))
    for si, ti, w in flows_indexed:
        flow_mat[si, n_src_s + ti] += w

    # -- draw with mpl_chord_diagram -------------------------------------------
    from mpl_chord_diagram import chord_diagram

    fig, ax = plt.subplots(figsize=fig_size)
    chord_diagram(
        flow_mat, all_names,
        ax=ax,
        colors=all_colors,
        gap=gap,
        use_gradient=use_gradient,
        directed=directed,
        sort=sort,
        fontsize=font_size,
        fontcolor="black",
        rotate_names=False,
    )

    # -- legend ----------------------------------------------------------------
    if show_legend:
        seen_cells: dict = {}
        for s in src_sectors_ordered:
            ci = src_cell_of[s]
            seen_cells.setdefault(cluster_names[ci], _cell_color(ci))
        for s in tgt_sectors_ordered:
            ci = tgt_cell_of[s]
            seen_cells.setdefault(cluster_names[ci], _cell_color(ci))
        legend_patches = [mpatches.Patch(color=col, label=name)
                          for name, col in seen_cells.items()]
        ax.legend(handles=legend_patches, title="Cell State",
                  loc='lower right', bbox_to_anchor=(1.35, 0.0),
                  fontsize=max(font_size - 1, 6),
                  title_fontsize=max(font_size, 7),
                  frameon=True, framealpha=0.9)

    _title = title_name or ("L-R communication: " + ", ".join(signaling)
                            if signaling else "L-R communication")
    ax.set_title(_title, fontsize=font_size + 4, pad=20)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 14.  plot_analysis_contribution - contribution plot for L-R pairs
# ---------------------------------------------------------------------------

def plot_analysis_contribution(
    cellchat: 'CellChat',
    signaling: str,
    signaling_name: Optional[str] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    width: float = 0.1,
    vertex_receiver: Optional[List[int]] = None,
    thresh: float = 0.05,
    return_data: bool = False,
    x_rotation: float = 0,
    title: Optional[str] = "Contribution of each L-R pair",
    font_size: int = 10,
    font_size_title: int = 10,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Bar plot showing relative contribution of each L-R pair to a signaling pathway.
    Mirrors R plot_analysis_contribution() with all parameters.

    Parameters
    ----------
    signaling : str
        Signaling pathway name.
    signaling_name : str, optional
        Alternative name to display on plot title (default: same as signaling).
    sources_use : list of str, optional
        Source cell group names to include.
    targets_use : list of str, optional
        Target cell group names to include.
    width : float
        Relative bar width (0-). Default 0.1 matches R default.
        R internally uses geom_bar(width=0.7) for the actual rendered bar,
        but exposes `width` as a scaling hint. Here width controls height of
        each bar in the horizontal plot (height = width * bar_unit).
    vertex_receiver : list of int, optional
        If provided, generate 3-panel hierarchy plot (All / Hierarchy1 / Hierarchy2)
        instead of the single panel. Indices of receiver cell groups.
    thresh : float
        P-value threshold; interactions with pval >= thresh are zeroed.
    return_data : bool
        If True, return (fig, df) tuple where df contains LR pair contributions.
    x_rotation : float
        Rotation angle of y-axis tick labels (mirrors R x.rotation parameter).
    title : str or None
        Plot title. Set to None to suppress.
    font_size : int
        Axis tick label font size.
    font_size_title : int
        Title font size.
    fig_size : tuple, optional
        Figure size (width, height) in inches. Auto-sized if None.
    return_fig : bool
        Return matplotlib Figure object instead of calling plt.show().
    """
    net_data = _network_view_for_visualization(cellchat, "network")

    if 'prob' not in net_data:
        raise ValueError("No probability data found. Run compute_communication_probability first.")

    if signaling_name is None:
        signaling_name = signaling

    lr_names_all, prob_array, pval_array = _network_arrays(net_data)
    interactions_df = net_data.get('interactions', None)
    cluster_names = _get_cluster_names_from_cellchat(cellchat)

    # Keep only edges with p-value strictly below the threshold.
    prob_array[pval_array >= thresh] = 0.0

    # Filter sources/targets
    if sources_use is not None:
        src_excl = [i for i, n in enumerate(cluster_names) if n not in sources_use]
        prob_array[src_excl, :, :] = 0.0
    if targets_use is not None:
        tgt_excl = [i for i, n in enumerate(cluster_names) if n not in targets_use]
        prob_array[:, tgt_excl, :] = 0.0

    # Build lr_name ->pathway_name and display_name lookup
    # interactions_df may be integer-indexed (positionally aligned) or string-indexed
    lr_pathway_map = {}
    lr_display_map = {}
    if interactions_df is not None and len(interactions_df) > 0 and 'pathway_name' in interactions_df.columns:
        is_positional = (
            len(interactions_df) == len(lr_names_all) and
            not isinstance(interactions_df.index.dtype, object)
        )
        if is_positional:
            for i, lr_name in enumerate(lr_names_all):
                row = interactions_df.iloc[i]
                lr_pathway_map[lr_name] = str(row.get('pathway_name', ''))
                lr_display_map[lr_name] = str(row.get('interaction_name_2', lr_name)) \
                    if 'interaction_name_2' in interactions_df.columns else lr_name
        else:
            name_to_row = {}
            for col in ('interaction_name', 'interaction_name_2'):
                if col in interactions_df.columns:
                    for idx, val in interactions_df[col].items():
                        name_to_row.setdefault(str(val), idx)
            for idx in interactions_df.index:
                name_to_row.setdefault(str(idx), idx)
            for lr_name in lr_names_all:
                row_idx = name_to_row.get(lr_name)
                if row_idx is not None:
                    row = interactions_df.loc[row_idx]
                    lr_pathway_map[lr_name] = str(row.get('pathway_name', ''))
                    lr_display_map[lr_name] = str(row.get('interaction_name_2', lr_name)) \
                        if 'interaction_name_2' in interactions_df.columns else lr_name
                else:
                    lr_pathway_map[lr_name] = ''
                    lr_display_map[lr_name] = lr_name

    # Restrict to the canonical significant L-R pairs for this pathway.
    lr_sig = cellchat.lr_pairs.get('significant', None)
    lr_sig_names = None
    if lr_sig is not None and len(lr_sig) > 0:
        if hasattr(lr_sig, 'index'):
            lr_sig_names = set(lr_sig.index.tolist())
        elif isinstance(lr_sig, pd.DataFrame) and 'interaction_name' in lr_sig.columns:
            lr_sig_names = set(lr_sig['interaction_name'].tolist())

    lr_indices = []
    for i, lr_name in enumerate(lr_names_all):
        if lr_pathway_map.get(lr_name, '') != signaling:
            continue
        if lr_sig_names is not None and lr_name not in lr_sig_names:
            continue
        lr_indices.append(i)

    if not lr_indices:
        warnings.warn(f"No significant L-R pairs found for pathway '{signaling_name}'")
        return (None, pd.DataFrame()) if return_data else None

    # Keep only LR pairs with non-zero total probability (after filtering)
    lr_indices = [i for i in lr_indices
                  if prob_array[:, :, i].sum() != 0]

    if not lr_indices:
        warnings.warn(f"No significant communication of {signaling_name}")
        return (None, pd.DataFrame()) if return_data else None

    # Slice and normalize: R does (prob - min(prob)) / (max(prob) - min(prob))
    prob_sub = prob_array[:, :, lr_indices]   # (C, C, n_lr)
    p_min = prob_sub.min()
    p_max = prob_sub.max()
    if p_max > p_min:
        prob_norm = (prob_sub - p_min) / (p_max - p_min)
    else:
        prob_norm = prob_sub.copy()

    p_total = prob_norm.sum()
    if p_total == 0:
        warnings.warn(f"No significant communication of {signaling_name}")
        return (None, pd.DataFrame()) if return_data else None

    # Display names for the kept LR indices
    display_names = [lr_display_map.get(lr_names_all[i], lr_names_all[i]) for i in lr_indices]

    def _make_contribution_series(prob_slice_3d):
        """Sum each LR slice over all cell pairs, divide by grand total."""
        p_sum = np.array([prob_slice_3d[:, :, k].sum() for k in range(prob_slice_3d.shape[2])])
        grand = prob_slice_3d.sum()
        if grand > 0:
            p_sum = p_sum / grand
        p_sum[np.isnan(p_sum)] = 0.0
        return p_sum

    if vertex_receiver is None:
        # -- Single panel (standard mode) --------------------------------------
        pSum = _make_contribution_series(prob_norm)
        y_lim = float(pSum.max()) if pSum.max() > 0 else 1.0

        # Build data: real pairs + padding to min 10 rows (mirrors R)
        df1 = pd.DataFrame({'name': display_names, 'contribution': pSum})
        df1 = df1[df1['contribution'] > 0]
        n_real = len(df1)

        if n_real == 0:
            warnings.warn(f"No positive contributions for pathway '{signaling_name}'")
            return (None, pd.DataFrame()) if return_data else None

        # Pad to 10 rows with invisible zero-contribution entries (R behaviour)
        n_pad = max(0, 10 - n_real)
        if n_pad > 0:
            pad = pd.DataFrame({
                'name': [str(j) for j in range(1, n_pad + 1)],
                'contribution': [0.0] * n_pad
            })
            df_plot = pd.concat([df1, pad], ignore_index=True)
        else:
            df_plot = df1.copy()

        # Sort descending so highest bar is at top after coord_flip
        df_plot = df_plot.sort_values('contribution', ascending=False).reset_index(drop=True)
        df1     = df1.sort_values('contribution', ascending=False).reset_index(drop=True)

        # y-axis order: R uses scale_x_discrete(limits=rev(levels(df$name)))
        # ->highest at top means we reverse for matplotlib (bottom=index 0)
        # Plot order: real names (descending) then padding (ascending from bottom)
        real_names_ordered = list(df1['name'])
        pad_names = [n for n in df_plot['name'] if n not in set(real_names_ordered)]
        plot_names = pad_names + list(reversed(real_names_ordered))   # pad at bottom
        plot_contribs = []
        contrib_map = dict(zip(df_plot['name'], df_plot['contribution']))
        for nm in plot_names:
            plot_contribs.append(contrib_map.get(nm, 0.0))

        n_rows = len(plot_names)

        # Bar height: R uses geom_bar(width=0.7). We scale by the user `width`
        # parameter but keep bars narrow. R's width=0.7 in a 10-row plot gives
        # each bar ~70% of its slot. Python height=0.35 gives a similar thin look.
        bar_height = np.clip(width * 3.5, 0.05, 0.85)

        if fig_size is None:
            fig_h = max(2.5, 0.35 * n_rows + 1.2)
            _fig_size = (5.0, fig_h)
        else:
            _fig_size = fig_size

        fig, ax = plt.subplots(figsize=_fig_size)

        y_pos = np.arange(n_rows)
        bars = ax.barh(y_pos, plot_contribs, height=bar_height,
                       color='#333333', edgecolor='none')

        # y-axis: show real names; hide padding labels (R: labels = c(rep("", pad), rev(real)))
        tick_labels = ['' if nm in set(pad_names) else nm for nm in plot_names]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            tick_labels,
            fontsize=font_size,
            color='black',
            rotation=x_rotation,
            ha='right' if x_rotation != 0 else 'right',
        )

        # x-axis: data values; hide tick marks (R: axis.ticks = element_blank())
        ax.set_xlim(0, y_lim * 1.05)
        ax.tick_params(axis='both', which='both', length=0)
        ax.tick_params(axis='x', labelsize=font_size)
        ax.set_xlabel('Relative contribution', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)

        # theme_classic: only bottom + left spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        if title:
            ax.set_title(title, fontsize=font_size_title, ha='center')

        plt.tight_layout()

        if return_data:
            df_out = df1[df1['contribution'] > 0][['name', 'contribution']].copy()
            return (fig, df_out) if return_fig else (None, df_out)

        if return_fig:
            return fig
        plt.show()
        return None

    else:
        # -- Three-panel hierarchy mode -----------------------------------------
        pSum_all = _make_contribution_series(prob_norm)
        p_total_hier = prob_norm.sum()
        y_lim = float(pSum_all.max()) if pSum_all.max() > 0 else 1.0

        n_lr = len(lr_indices)

        def _hier_sum(receiver_indices):
            if n_lr > 1:
                return np.array([
                    prob_norm[:, receiver_indices, k].sum()
                    for k in range(n_lr)
                ]) / p_total_hier
            else:
                return np.array([prob_norm[:, receiver_indices, 0].sum()]) / p_total_hier

        n_cells = prob_norm.shape[0]
        non_recv = [i for i in range(n_cells) if i not in vertex_receiver]
        pSum_h1 = _hier_sum(vertex_receiver)
        pSum_h2 = _hier_sum(non_recv)

        for arr in (pSum_all, pSum_h1, pSum_h2):
            arr[np.isnan(arr)] = 0.0

        bar_width_hier = np.clip(width * 2.0, 0.05, 0.50)

        if fig_size is None:
            _fig_size = (12, max(3, 0.3 * n_lr + 1.5))
        else:
            _fig_size = fig_size

        fig, axes = plt.subplots(1, 3, figsize=_fig_size)
        x_pos = np.arange(n_lr)

        panel_data = [
            ('All',        pSum_all),
            ('Hierarchy1', pSum_h1),
            ('Hierarchy2', pSum_h2),
        ]

        for ax_i, (panel_title, pdata) in zip(axes, panel_data):
            ax_i.bar(x_pos, pdata, width=bar_width_hier,
                     color='#333333', edgecolor='none')
            ax_i.set_xticks(x_pos)
            ax_i.set_xticklabels(
                display_names,
                rotation=x_rotation if x_rotation != 0 else 45,
                ha='right', fontsize=font_size - 1
            )
            ax_i.set_ylim(0, y_lim * 1.05)
            ax_i.set_ylabel('Relative contribution', fontsize=font_size)
            ax_i.set_title(panel_title, fontsize=font_size, ha='center')
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            ax_i.tick_params(axis='both', which='both', length=0)

        super_title = f"Contribution of each signaling in {signaling_name} pathway"
        fig.suptitle(super_title, fontsize=font_size_title, fontweight='bold', y=1.01)
        plt.tight_layout()

        if return_data:
            df_out = pd.DataFrame({'name': display_names, 'contribution': pSum_all})
            df_out = df_out[df_out['contribution'] > 0]
            return (fig, df_out) if return_fig else (None, df_out)

        if return_fig:
            return fig
        plt.show()
        return None




# ---------------------------------------------------------------------------
# 15. plot_network_circle_grid -grid of per-source circular plots
# ---------------------------------------------------------------------------

def plot_network_circle_grid(
    cellchat: 'CellChat',
    sources_use: Optional[List[str]] = None,
    slot_name: str = "pathway_network",
    thresh: float = 0.05,
    color_use: Optional[List[str]] = None,
    edge_width_max: float = 6.0,
    ncol: int = 4,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Grid of circular plots -one panel per source cell group.
    All panels use the same fixed circular layout and node colours.
    Mirrors the R 3x4 grid produced by looping plot_network_circle with
    sources.use=[i] for each cell group.

    Parameters
    ----------
    sources_use : list of str, optional
        Cell groups to show as panels (default: all groups).
    ncol : int
        Number of columns in the grid (default 4 ->nrow = ceil(n/4)).
    """
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    if sources_use is None:
        sources_use = cluster_names

    # --- extract aggregated weight matrix ---
    net_data = _network_view_for_visualization(cellchat, slot_name)
    if 'prob' not in net_data:
        raise ValueError("No probability data found")
    _, prob_array, pval_array = _network_arrays(net_data)
    prob_f = prob_array.copy()
    prob_f[pval_array >= thresh] = 0.0
    full_mat = prob_f.sum(axis=2) if prob_f.ndim == 3 else prob_f

    # global edge_weight_max for consistent scaling across panels
    all_weights = full_mat[full_mat > 0]
    if len(all_weights) == 0:
        warnings.warn("No interactions to plot in circle grid")
        return None
    edge_weight_max_global = float(all_weights.max())

    # fixed circular positions -same for every panel
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_clusters, endpoint=False)
    R = 1.0
    pos = {name: (R * np.cos(a), R * np.sin(a))
           for name, a in zip(cluster_names, angles)}

    # fixed node sizes from global cell counts
    try:
        counts = np.array(list(pd.Series(cellchat.groups).value_counts()
                               .reindex(cluster_names).fillna(1).values), dtype=float)
    except Exception:
        counts = np.ones(n_clusters)
    if counts.max() > counts.min():
        node_radii = np.interp(counts, (counts.min(), counts.max()), (2, 7))
    else:
        node_radii = np.full(n_clusters, 4.0)

    n_sources = len(sources_use)
    nrow = int(np.ceil(n_sources / ncol))

    if fig_size is None:
        fig_size = (ncol * 3.5, nrow * 3.5)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    for panel_idx, src_name in enumerate(sources_use):
        row, col = divmod(panel_idx, ncol)
        ax = axes[row, col]
        ax.set_aspect('equal')
        ax.axis('off')
        pad = 0.45
        ax.set_xlim(-R - pad, R + pad)
        ax.set_ylim(-R - pad, R + pad)

        # build per-source matrix
        if src_name not in cluster_names:
            ax.set_title(src_name, fontsize=8)
            continue
        src_i = cluster_names.index(src_name)
        panel_mat = np.zeros_like(full_mat)
        panel_mat[src_i, :] = full_mat[src_i, :]

        # draw edges
        for j, tgt in enumerate(cluster_names):
            w = panel_mat[src_i, j]
            if w <= 0:
                continue
            lw = 0.3 + w / edge_weight_max_global * edge_width_max
            lw = min(lw, edge_width_max)
            c = to_rgba(color_map[src_name], alpha=0.7)
            x0, y0 = pos[src_name]
            x1, y1 = pos[tgt]
            if src_name == tgt:
                theta = angles[src_i]
                loop_r = 0.06
                cx = (R + loop_r * 0.85) * np.cos(theta)
                cy = (R + loop_r * 0.85) * np.sin(theta)
                loop_patch = plt.Circle((cx, cy), loop_r, fill=False,
                                        color=c, linewidth=lw * 0.6)
                ax.add_patch(loop_patch)
            else:
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=c,
                        lw=lw,
                        connectionstyle="arc3,rad=0.12",
                        mutation_scale=8,
                    )
                )

        # draw nodes
        for idx, name in enumerate(cluster_names):
            x, y = pos[name]
            ax.scatter(x, y, s=(node_radii[idx] * 15) ** 2 / 400,
                       c=[color_map[name]], edgecolors=[color_map[name]],
                       linewidths=1.0, zorder=5)

        # labels
        label_r = R + 0.16
        for name, angle in zip(cluster_names, angles):
            lx = label_r * np.cos(angle)
            ly = label_r * np.sin(angle)
            ha = 'left' if np.cos(angle) >= 0 else 'right'
            va = 'bottom' if np.sin(angle) >= 0 else 'top'
            ax.text(lx, ly, name, ha=ha, va=va, fontsize=6, clip_on=True)

        ax.set_title(src_name, fontsize=8, fontweight='normal', pad=4)

    # hide unused panels
    for panel_idx in range(n_sources, nrow * ncol):
        row, col = divmod(panel_idx, ncol)
        axes[row, col].axis('off')

    plt.tight_layout(pad=0.5)
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 16. plot_select_k -line plot of NMF k selection scores
# ---------------------------------------------------------------------------

def plot_select_k(
    k_result: Dict[str, Any],
    pattern: str = "outgoing",
    title_name: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 4),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Line plot of Cophenetic and Silhouette scores vs. number of NMF patterns k.
    Mirrors R selectK(): two faceted panels with independent y-axes, shared bold
    title at top centre, axis labels and frame around each panel.

    Parameters
    ----------
    k_result : dict
        Output of cc.select_k() with keys 'k_range', 'cophenetic', 'silhouette'.
    pattern : str
        'outgoing' or 'incoming', used for default title.
    title_name : str, optional
        Override for the figure title.
    fig_size : tuple
        Figure size.  Default (10, 4) gives two side-by-side panels.
    """
    k_vals = k_result.get('k_range', k_result.get('k', []))
    cop = k_result['cophenetic']
    sil = k_result['silhouette']

    if title_name is None:
        title_name = f"{pattern} signaling"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    fig.suptitle(title_name, fontsize=11, fontweight='bold', ha='center', y=1.01)

    cop_color = '#66C2A5'  # R Set2 green
    sil_color = '#FC8D62'  # R Set2 orange

    def _style_panel(ax, label):
        """Apply R theme_classic + facet-strip styling."""
        ax.set_xlabel('Number of patterns', fontsize=10)
        ax.set_ylabel('Measure score', fontsize=10)
        ax.set_xticks(k_vals)
        ax.set_xticklabels([str(k) for k in k_vals], fontsize=9)
        # facet strip spanning the whole panel, matching ggplot facet_wrap.
        strip = mpatches.Rectangle((0, 1.0), 1, 0.11, transform=ax.transAxes,
                                   facecolor='white', edgecolor='black',
                                   linewidth=0.8, clip_on=False, zorder=4)
        ax.add_patch(strip)
        ax.text(0.5, 1.055, label, transform=ax.transAxes, ha='center',
                va='center', fontsize=10, zorder=5)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
        ax.tick_params(which='both', direction='out', length=3, width=0.8)

    # Cophenetic panel
    cop_arr = np.array(cop, dtype=float)
    mask = np.isfinite(cop_arr)
    if mask.any():
        ax1.plot(np.array(k_vals)[mask], cop_arr[mask], '-o',
                 color=cop_color, linewidth=1.5, markersize=5)
    _style_panel(ax1, 'Cophenetic')

    # Silhouette panel
    sil_arr = np.array(sil, dtype=float)
    mask2 = np.isfinite(sil_arr)
    if mask2.any():
        ax2.plot(np.array(k_vals)[mask2], sil_arr[mask2], '-o',
                 color=sil_color, linewidth=1.5, markersize=5)
    _style_panel(ax2, 'Silhouette')

    legend_handles = [
        plt.Line2D([0], [0], color=cop_color, marker='o', linewidth=1.5,
                   markersize=5, label='Cophenetic'),
        plt.Line2D([0], [0], color=sil_color, marker='o', linewidth=1.5,
                   markersize=5, label='Silhouette'),
    ]
    fig.legend(handles=legend_handles, title='Measure type', loc='center left',
               bbox_to_anchor=(0.91, 0.5), frameon=False, fontsize=10,
               title_fontsize=11)

    plt.tight_layout(rect=(0, 0, 0.88, 0.98))
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 17. plot_analysis_pattern_heatmap -NMF W/H matrix dual-panel heatmap
# ---------------------------------------------------------------------------

def plot_analysis_pattern_heatmap(
    cellchat: 'CellChat',
    pattern: str = "outgoing",
    slot_name: str = "pathway_network",
    color_use: Optional[List[str]] = None,
    color_heatmap: str = "Spectral",
    title_legend: str = "Contributions",
    font_size: int = 8,
    cluster_rows: bool = True,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Dual-panel heatmap of NMF pattern contributions.
    Left panel  = Cell patterns  (W matrix: cell group x Pattern).
    Right panel = Communication patterns (H matrix: signaling x Pattern).

    Each panel has:
      - A dendrogram on the far left (from hierarchical clustering of rows).
      - A thin coloured annotation strip next to the dendrogram.
      - The heatmap body with row labels on the left.
      - X-axis: pattern labels, rotated 45 deg.

    Shared colour bar on the far right.  Matches R ComplexHeatmap output.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

    net_data = _network_view_for_visualization(cellchat, slot_name)

    if 'pattern' not in net_data:
        raise ValueError(f"No pattern data in {slot_name}. Run identify_communication_patterns first.")

    pattern_data = net_data['pattern'].get(pattern)
    if pattern_data is None:
        raise ValueError(f"Pattern '{pattern}' not found")

    data_cell = pattern_data.get('cell')
    data_sig  = pattern_data.get('signaling')
    if data_cell is None or data_sig is None:
        raise ValueError("Pattern data missing 'cell' or 'signaling'")

    # -- build W (cell groups x patterns) --------------------------------------
    def _pat_sort_key(s):
        import re; m = re.search(r'(\d+)$', s); return int(m.group(1)) if m else s

    cell_groups = data_cell['CellGroup'].unique().tolist()
    try:
        ident_order = list(cellchat.groups.categories)
        cell_groups = [g for g in ident_order if g in cell_groups] + \
                      [g for g in cell_groups if g not in ident_order]
    except Exception:
        pass

    patterns = sorted(data_cell['Pattern'].unique().tolist(), key=_pat_sort_key)
    n_pat    = len(patterns)

    W = np.zeros((len(cell_groups), n_pat))
    for _, row in data_cell.iterrows():
        if row['CellGroup'] in cell_groups and row['Pattern'] in patterns:
            W[cell_groups.index(row['CellGroup']), patterns.index(row['Pattern'])] = row['Contribution']

    # -- build H (signaling pathways x patterns) -------------------------------
    sig_names = sorted(data_sig['Signaling'].unique().tolist())
    H = np.zeros((len(sig_names), n_pat))
    for _, row in data_sig.iterrows():
        if row['Signaling'] in sig_names and row['Pattern'] in patterns:
            H[sig_names.index(row['Signaling']), patterns.index(row['Pattern'])] = row['Contribution']

    # -- hierarchical clustering order (R: cluster_rows=T, method="average") ---
    def _cluster_order(mat):
        if mat.shape[0] < 2:
            return np.arange(mat.shape[0]), None
        try:
            Z = linkage(mat, method='average', metric='euclidean')
            return leaves_list(Z), Z
        except Exception:
            return np.arange(mat.shape[0]), None

    if cluster_rows:
        w_order, w_Z = _cluster_order(W)
        h_order, h_Z = _cluster_order(H)
    else:
        w_order, w_Z = np.arange(len(cell_groups)), None
        h_order, h_Z = np.arange(len(sig_names)),  None

    W_plot     = W[w_order, :]
    H_plot     = H[h_order, :]
    cell_labels = [cell_groups[i] for i in w_order]
    sig_labels  = [sig_names[i]   for i in h_order]

    # -- colours ----------------------------------------------------------------
    if color_use is not None:
        all_pal = list(color_use)
    else:
        all_pal = sc_palette(len(cell_groups))
    try:
        ident_order2 = list(cellchat.groups.categories)
        cell_colors = [all_pal[ident_order2.index(g)] if g in ident_order2
                       else '#999999' for g in cell_labels]
    except Exception:
        cell_colors = [all_pal[i % len(all_pal)] for i in range(len(cell_labels))]

    try:
        cmap = plt.get_cmap(color_heatmap).reversed()
    except Exception:
        cmap = plt.cm.RdBu_r

    # -- figure layout ---------------------------------------------------------
    # Columns per panel: [dendrogram | color-strip | heatmap]
    # Between panels: gap.  Far right: colorbar.
    n_cell = len(cell_labels);  n_sig = len(sig_labels)

    if fig_size is None:
        # Scale height to the taller panel
        h = max(3.5, max(n_cell, n_sig) * 0.3 + 1.5)
        fig_size = (14, h)

    fig = plt.figure(figsize=fig_size)

    # Width ratios: dend_w(left), strip_w, heat_w, gap, dend_h, strip_w, heat_h, cbar
    _dend  = 0.8   # inches
    _strip = 0.15  # inches  (colour annotation strip)
    _gap   = 0.5   # inches
    _cbar  = 0.2   # inches
    _heat  = (fig_size[0] - 2 * _dend - 2 * _strip - _gap - _cbar) / 2

    wr = [_dend, _strip, _heat, _gap, _dend, _strip, _heat, _cbar]
    gs = gridspec.GridSpec(1, 8, width_ratios=wr,
                           wspace=0.02,
                           left=0.02, right=0.97, top=0.90, bottom=0.12)

    ax_dend_w  = fig.add_subplot(gs[0, 0])
    ax_strip_w = fig.add_subplot(gs[0, 1])
    ax_w       = fig.add_subplot(gs[0, 2])
    fig.add_subplot(gs[0, 3]).axis('off')   # gap
    ax_dend_h  = fig.add_subplot(gs[0, 4])
    ax_strip_h = fig.add_subplot(gs[0, 5])
    ax_h       = fig.add_subplot(gs[0, 6])
    ax_cbar    = fig.add_subplot(gs[0, 7])

    def _draw_dendrogram(ax, linkage_matrix, n_rows):
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, n_rows - 0.5); ax.axis('off')
        if linkage_matrix is None:
            return
        # Draw dendrogram rotated 90 deg (leaves on y-axis)
        ddata = dendrogram(linkage_matrix, orientation='left', no_plot=True)
        # ddata['icoord'] and 'dcoord' are in leaf-index space x 10
        y_scale = (n_rows - 1) / max((n_rows - 1) * 10 or 1, 1)
        max_d = max(max(d) for d in ddata['dcoord']) if ddata['dcoord'] else 1.0
        for xs, ys in zip(ddata['icoord'], ddata['dcoord']):
            # xs are y positions (0..n*10), ys are distances
            y_pts = [(x / 10.0 - 0.5) for x in xs]
            x_pts = [1.0 - (y / max_d) * 0.95 for y in ys]
            ax.plot(x_pts, y_pts, 'k-', linewidth=0.6)

    def _draw_strip(ax, colors, n_rows):
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, n_rows - 0.5); ax.axis('off')
        for i, c in enumerate(colors):
            ax.add_patch(mpatches.Rectangle((0, i - 0.5), 1, 1,
                                            color=c, linewidth=0))

    def _draw_heatmap(ax, mat, row_labels, col_labels, title_str):
        n_r, n_c = mat.shape
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_xticks(np.arange(n_c))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=font_size)
        # row labels on the LEFT of the heatmap
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels(row_labels, fontsize=font_size)
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()
        ax.tick_params(which='both', length=0)
        ax.spines[:].set_visible(False)
        # minor grid lines (white cell borders)
        ax.set_xticks(np.arange(n_c + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_r + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        ax.tick_params(which='minor', left=False, bottom=False)
        ax.set_title(title_str, fontsize=font_size + 1, pad=5)
        return im

    # -- left panel: W (cell groups x patterns) --------------------------------
    _draw_dendrogram(ax_dend_w, w_Z, n_cell)
    _draw_strip(ax_strip_w, cell_colors, n_cell)
    im = _draw_heatmap(ax_w, W_plot, cell_labels, patterns, "Cell patterns")

    # -- right panel: H (signaling x patterns) ---------------------------------
    # signaling annotation strip: use grey (R default color.use.signaling="grey50")
    sig_strip_colors = ['#808080'] * n_sig
    _draw_dendrogram(ax_dend_h, h_Z, n_sig)
    _draw_strip(ax_strip_h, sig_strip_colors, n_sig)
    _draw_heatmap(ax_h, H_plot, sig_labels, patterns, "Communication patterns")

    # -- shared colorbar --------------------------------------------------------
    cb = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
    cb.set_label(title_legend, fontsize=font_size, rotation=270, labelpad=12)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['0', '1'], fontsize=font_size - 1)
    cb.ax.tick_params(length=2)

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 18. plot_analysis_signaling_role_scatter_dual -dual-panel version (count+weight)
# ---------------------------------------------------------------------------

def plot_analysis_signaling_role_scatter_dual(
    cellchat: 'CellChat',
    signaling=None,
    slot_name: str = "pathway_network",
    color_use: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 6),
    label_size: float = 3,
    dot_alpha: float = 0.6,
    do_label: bool = True,
    show_legend: bool = True,
    fig_size: Tuple[int, int] = (14, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Two-panel scatter of signaling roles:
      Left  = based on count (number of links, large circles)
      Right = based on weight (interaction strength, smaller circles)
    Mirrors R 485_48.png dual-panel layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    # we need to draw each panel onto a specific axes -delegate to a helper
    def _draw_one(ax, measure_label, x_measure, y_measure):
        """Draw a single role-scatter panel on ax."""
        net_data = _network_view_for_visualization(cellchat, slot_name)
        centrality = net_data.get('centrality', {})
        if not centrality:
            ax.set_title("No centrality data", fontsize=9)
            return

        cluster_names = _network_group_names(cellchat, net_data)
        n_clusters = len(cluster_names)

        pathway_names = list(centrality.keys())
        if signaling is not None:
            s = [signaling] if isinstance(signaling, str) else signaling
            pathway_names = [p for p in pathway_names if p in s]

        out_mat = np.zeros((n_clusters, len(pathway_names)))
        in_mat  = np.zeros((n_clusters, len(pathway_names)))
        for pi, pname in enumerate(pathway_names):
            c = centrality[pname]
            out_values = c.get(x_measure)
            in_values = c.get(y_measure)
            if out_values is None or in_values is None:
                prob = net_data.get('prob')
                pval = net_data.get('pval')
                pathways = network_names(net_data)
                if (x_measure == 'outdeg_unweighted' and y_measure == 'indeg_unweighted'
                        and isinstance(prob, dict) and pname in prob):
                    pair_mat = _network_matrix(net_data, 'prob', pname).copy()
                    if pval is not None:
                        pair_pval = _network_matrix(net_data, 'pval', pname)
                        pair_mat[pair_pval >= 0.05] = 0.0
                    out_values = np.sum(pair_mat > 0, axis=1, dtype=float)
                    in_values = np.sum(pair_mat > 0, axis=0, dtype=float)
                else:
                    out_values = c.get('outdeg', np.zeros(n_clusters))
                    in_values = c.get('indeg', np.zeros(n_clusters))
            out_mat[:, pi] = out_values
            in_mat[:, pi] = in_values

        out_cells = out_mat.sum(axis=1)
        in_cells  = in_mat.sum(axis=1)

        prob = net_data.get('prob')
        if isinstance(prob, dict):
            _, prob_f, pval_f = _network_arrays(net_data)
            prob_f[pval_f >= 0.05] = 0.0
            count_mat = np.sum(prob_f > 0, axis=2)
            num_link = count_mat.sum(axis=1) + count_mat.sum(axis=0) - np.diag(count_mat)
        else:
            num_link = np.ones(n_clusters)

        _colors = color_use if color_use else sc_palette(n_clusters)

        vmin_s, vmax_s = dot_size
        if num_link.max() > num_link.min():
            sizes = np.interp(num_link, (num_link.min(), num_link.max()),
                              (vmin_s ** 2 * 5, vmax_s ** 2 * 5))
        else:
            sizes = np.full(n_clusters, ((vmin_s + vmax_s) / 2) ** 2 * 5)

        for i, name in enumerate(cluster_names):
            c = _colors[i] if i < len(_colors) else '#999999'
            ax.scatter(out_cells[i], in_cells[i], s=sizes[i],
                       c=[to_rgba(c, alpha=dot_alpha)],
                       edgecolors=c, linewidths=0.8, zorder=3)
            if do_label:
                ax.annotate(name, (out_cells[i], in_cells[i]),
                            fontsize=label_size * 2.5, ha='left', va='bottom',
                            xytext=(3, 3), textcoords='offset points')

        ax.axhline(0, linestyle='--', color='grey', linewidth=0.5, alpha=0.7)
        ax.axvline(0, linestyle='--', color='grey', linewidth=0.5, alpha=0.7)
        axis_suffix = "number" if "unweighted" in x_measure else "strength"
        ax.set_xlabel(f"Outgoing interaction {axis_suffix}", fontsize=9)
        ax.set_ylabel(f"Incoming interaction {axis_suffix}", fontsize=9)
        ax.set_title(measure_label, fontsize=10, ha='center')
        ax.spines[['top', 'right']].set_visible(False)

        if show_legend:
            handles = [mpatches.Patch(color=_colors[i] if i < len(_colors) else '#999999',
                                      label=cluster_names[i]) for i in range(n_clusters)]
            ax.legend(handles=handles, fontsize=7, frameon=False,
                      bbox_to_anchor=(1.01, 1), loc='upper left')

    _draw_one(axes[0], "Number of interactions", "outdeg_unweighted", "indeg_unweighted")
    _draw_one(axes[1], "Interaction strength",   "outdeg", "indeg")

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 19. plot_analysis_signaling_role_heatmap_combined -outgoing+incoming dual panel
# ---------------------------------------------------------------------------

def plot_analysis_signaling_role_heatmap_combined(
    cellchat: 'CellChat',
    signaling=None,
    slot_name: str = "pathway_network",
    color_use: Optional[List[str]] = None,
    color_heatmap: str = "BuGn",
    font_size: int = 8,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    fig_size: Tuple[int, int] = (16, 6),
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Combined dual-panel heatmap: Outgoing (left) and Incoming (right) on the
    same canvas, sharing row/column ordering.  Mirrors R 486_48.png.
    """
    net_data = _network_view_for_visualization(cellchat, slot_name)
    centrality = net_data.get('centrality', {})
    if not centrality:
        raise ValueError("Run compute_network_centrality first.")

    cluster_names = _network_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)
    pathway_names_all = list(centrality.keys())

    # Filter pathways
    if signaling is not None:
        sig_list = [signaling] if isinstance(signaling, str) else list(signaling)
        pathway_names_all = [p for p in pathway_names_all if p in sig_list]

    n_pathways = len(pathway_names_all)

    # Build outgoing and incoming matrices (pathway x cell)
    out_mat = np.zeros((n_pathways, n_clusters))
    in_mat  = np.zeros((n_pathways, n_clusters))
    for pi, pname in enumerate(pathway_names_all):
        c = centrality[pname]
        out_mat[pi, :] = c.get('outdeg', np.zeros(n_clusters))
        in_mat[pi, :]  = c.get('indeg',  np.zeros(n_clusters))

    # Remove all-zero rows (union of both)
    keep = (out_mat.sum(axis=1) + in_mat.sum(axis=1)) > 0
    out_mat = out_mat[keep, :]
    in_mat  = in_mat[keep, :]
    pathway_labels = [pathway_names_all[i] for i, k in enumerate(keep) if k]
    n_pathways = len(pathway_labels)

    if n_pathways == 0:
        warnings.warn("No non-zero pathways to plot.")
        return None

    # Row-scale each panel independently
    def _row_scale(mat):
        m = mat.copy()
        rmax = m.max(axis=1, keepdims=True)
        rmax[rmax == 0] = 1.0
        m = m / rmax
        m[m == 0] = np.nan
        return m

    out_scaled = _row_scale(out_mat)
    in_scaled  = _row_scale(in_mat)

    # R's heatmaps use one optional row/column ordering for both panels, so a
    # pathway or cell group remains aligned between outgoing and incoming data.
    def _cluster_order(mat):
        if mat.shape[0] < 2:
            return np.arange(mat.shape[0])
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            return leaves_list(linkage(mat, method='average', metric='euclidean'))
        except ValueError:
            return np.arange(mat.shape[0])

    combined_for_order = np.concatenate([out_mat, in_mat], axis=1)
    row_order = _cluster_order(combined_for_order) if cluster_rows else np.arange(n_pathways)
    # Rows are pathways. To cluster cell groups, stack the outgoing and
    # incoming pathway profiles vertically, leaving cell groups as columns.
    combined_cell_profiles = np.concatenate([out_mat, in_mat], axis=0)
    col_order = _cluster_order(combined_cell_profiles.T) if cluster_cols else np.arange(n_clusters)

    out_mat = out_mat[np.ix_(row_order, col_order)]
    in_mat = in_mat[np.ix_(row_order, col_order)]
    out_scaled = out_scaled[np.ix_(row_order, col_order)]
    in_scaled = in_scaled[np.ix_(row_order, col_order)]
    pathway_labels = [pathway_labels[i] for i in row_order]
    cluster_names = [cluster_names[i] for i in col_order]

    if color_use is None:
        color_use = sc_palette(n_clusters)

    try:
        cmap = plt.get_cmap(color_heatmap)
    except Exception:
        cmap = plt.cm.YlGn

    # Shared layout: two heatmaps side by side with a colorbar on the right
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[0.12, 0.88],
        width_ratios=[0.48, 0.48, 0.04],
        hspace=0.05, wspace=0.06,
        left=0.12, right=0.96, top=0.92, bottom=0.06
    )

    ax_top_out = fig.add_subplot(gs[0, 0])
    ax_top_in  = fig.add_subplot(gs[0, 1])
    ax_out     = fig.add_subplot(gs[1, 0])
    ax_in      = fig.add_subplot(gs[1, 1])
    ax_cbar    = fig.add_subplot(gs[:, 2])

    def _draw_panel(ax, ax_top, mat, raw_mat, title_label):
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_xticks(np.arange(n_clusters))
        ax.set_xticklabels(cluster_names, rotation=90, ha='center', fontsize=font_size)
        ax.set_yticks(np.arange(n_pathways))
        ax.set_yticklabels(pathway_labels, fontsize=font_size)
        ax.set_xticks(np.arange(n_clusters + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_pathways + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        ax.tick_params(which='minor', left=False, bottom=False)
        ax.tick_params(which='both', length=0)
        ax.spines[:].set_visible(False)

        # R uses colSums(mat.ori): retain raw strength rather than the
        # row-normalized heatmap values shown below it.
        col_sums = np.sum(raw_mat, axis=0)
        ax_top.bar(np.arange(n_clusters), col_sums,
                   color=color_use[:n_clusters], edgecolor='none', width=0.7)
        ax_top.set_xlim(-0.5, n_clusters - 0.5)
        ax_top.set_xticks([])
        ax_top.set_title(title_label, fontsize=font_size + 1, fontweight='normal', pad=4)
        ax_top.spines[:].set_visible(False)
        ax_top.tick_params(left=False, labelleft=False)
        return im

    _draw_panel(ax_out, ax_top_out, out_scaled, out_mat, "Outgoing")
    im = _draw_panel(ax_in, ax_top_in, in_scaled, in_mat, "Incoming")

    # right y-axis labels only on the left panel
    ax_in.set_yticks([])

    cb = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
    cb.set_label("Relative strength", fontsize=font_size, labelpad=2)
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=font_size - 1)

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 20. plot_network_embedding_by_group -per-group faceted embedding scatter
# ---------------------------------------------------------------------------

def plot_network_embedding_by_group(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    emb_type: str = "functional",
    color_use: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 8),
    dot_alpha: float = 0.6,
    ncol: int = 2,
    font_size: int = 9,
    do_label: bool = True,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Per-group faceted scatter plot of pathway embedding.
    Each panel shows only the pathways belonging to that group, other pathways hidden.
    Mirrors R 497_48.png 2x2 per-group layout.

    Requires compute_network_similarity + embed_network + cluster_network to have been run.
    """
    net_data = _network_view_for_visualization(cellchat, slot_name)

    sim    = net_data.get('similarity', {}).get(emb_type, {})
    dr     = sim.get('dr', {}).get('single', None)
    groups = sim.get('group', {}).get('single', None)
    prob   = net_data.get('prob')

    if dr is None:
        raise ValueError(
            f"No embedding found for type='{emb_type}'. "
            "Run compute_network_similarity, embed_network, and cluster_network first."
        )

    dr = np.array(dr)
    n_pts = dr.shape[0]
    full_pathways = network_names(net_data)
    pathway_names_list = list(sim.get('pathways', full_pathways[:n_pts]))
    if len(pathway_names_list) != n_pts:
        pathway_names_list = (pathway_names_list + [f"P{i}" for i in range(len(pathway_names_list), n_pts)])[:n_pts]

    if groups is None:
        groups = ['G1'] * n_pts
    groups = list(groups)[:n_pts]

    unique_groups  = sorted(set(groups))
    n_groups       = len(unique_groups)
    nrow           = int(np.ceil(n_groups / ncol))

    gg_colors  = gg_palette(n_groups)
    resolved   = _resolve_colors(n_groups, color_use)
    group_color_map = dict(zip(unique_groups,
                               gg_colors if color_use is None else resolved))

    if isinstance(prob, dict):
        prob_sum = _network_strengths(net_data, pathway_names_list)
        prob_norm = prob_sum / prob_sum.max() if prob_sum.max() > 0 else np.ones(n_pts)
    else:
        prob_norm = np.ones(n_pts)

    if fig_size is None:
        fig_size = (ncol * 4, nrow * 4)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    # global axis limits
    x_all, y_all = dr[:, 0], dr[:, 1]
    x_pad = (x_all.max() - x_all.min()) * 0.1 + 0.1
    y_pad = (y_all.max() - y_all.min()) * 0.1 + 0.1
    xlim = (x_all.min() - x_pad, x_all.max() + x_pad)
    ylim = (y_all.min() - y_pad, y_all.max() + y_pad)

    for panel_idx, g in enumerate(unique_groups):
        row, col = divmod(panel_idx, ncol)
        ax = axes[row, col]

        # Background: all points in light grey
        ax.scatter(x_all, y_all, s=15, c='#dddddd',
                   edgecolors='none', alpha=0.4, zorder=1)

        # Foreground: only this group
        idx_g  = [i for i, grp in enumerate(groups) if grp == g]
        xs     = dr[idx_g, 0]
        ys     = dr[idx_g, 1]
        pn     = prob_norm[idx_g]
        sizes  = np.interp(pn, (0, 1), dot_size) ** 2 * 5
        c      = group_color_map[g]
        ax.scatter(xs, ys, s=sizes,
                   c=[to_rgba(c, alpha=dot_alpha)] * len(xs),
                   edgecolors=c, linewidths=0.5, zorder=3)

        if do_label:
            for i_local, i_global in enumerate(idx_g):
                nm = pathway_names_list[i_global]
                ax.text(dr[i_global, 0], dr[i_global, 1], nm,
                        fontsize=max(font_size - 3, 5), ha='center', va='bottom',
                        alpha=0.85, zorder=4)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Group {g}", fontsize=font_size, color=c, fontweight='bold')
        ax.set_xlabel("Dim 1", fontsize=font_size - 1)
        ax.set_ylabel("Dim 2", fontsize=font_size - 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)

    # hide unused panels
    for panel_idx in range(n_groups, nrow * ncol):
        row, col = divmod(panel_idx, ncol)
        axes[row, col].axis('off')

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 18.  cellchat_theme_options  - matplotlib rcParams for clean white-bg plots
# ---------------------------------------------------------------------------

def cellchat_theme_options() -> Dict[str, Any]:
    """Return a dict of matplotlib rcParams mirroring R cellchat_theme_options().

    Style: classic theme, white backgrounds, no panel borders, clean grid.
    Apply with ``plt.rcParams.update(cellchat_theme_options())`` or merge
    into a ``with plt.rc_context(cellchat_theme_options()):`` block.
    """
    return {
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'grid.color': '#e5e5e5',
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 1.0,
        'patch.edgecolor': 'black',
        'patch.linewidth': 0.5,
    }


def _validate_hierarchy_receivers(vertex_receiver: List[int], n_clusters: int) -> List[int]:
    """Validate the Python zero-based receiver indices used by hierarchy plots."""
    if not isinstance(vertex_receiver, (list, tuple, np.ndarray)):
        raise TypeError("vertex_receiver must be a sequence of zero-based cell-group indices")
    receivers = list(vertex_receiver)
    if any(not isinstance(index, (int, np.integer)) for index in receivers):
        raise TypeError("vertex_receiver must contain only integer zero-based indices")
    receivers = [int(index) for index in receivers]
    if any(index < 0 or index >= n_clusters for index in receivers):
        raise IndexError(
            f"vertex_receiver uses zero-based indices in [0, {n_clusters - 1}]"
        )
    if len(set(receivers)) != len(receivers):
        raise ValueError("vertex_receiver must not contain duplicate indices")
    if not receivers or len(receivers) == n_clusters:
        raise ValueError("vertex_receiver must contain at least one, but not all, cell groups")
    return receivers


def _filter_hierarchy_matrix(
    net_mat: np.ndarray,
    cluster_names: List[str],
    sources_use: Optional[List[str]],
    targets_use: Optional[List[str]],
    top: float,
) -> np.ndarray:
    """Apply the R hierarchy source, target, and top-edge filters."""
    filtered = np.asarray(net_mat, dtype=float).copy()
    if sources_use is not None:
        filtered[[i for i, name in enumerate(cluster_names) if name not in sources_use], :] = 0.0
    if targets_use is not None:
        filtered[:, [i for i, name in enumerate(cluster_names) if name not in targets_use]] = 0.0
    if not 0 < top <= 1:
        raise ValueError("top must be in the interval (0, 1]")
    positive = filtered[filtered > 0]
    if top < 1 and positive.size:
        filtered[filtered < np.quantile(positive, 1 - top)] = 0.0
    return filtered


def _draw_hierarchy_grid(
    matrices: List[np.ndarray],
    labels: List[str],
    cluster_names: List[str],
    vertex_receiver: List[int],
    color_use: Optional[List[str]],
    vertex_weight: float,
    vertex_size_max: float,
    edge_width_max: float,
    edge_weight_max_individual: Optional[float],
    edge_weight_max_aggregate: Optional[float],
    vertex_label_cex: float,
    height: float,
    title: Optional[str],
) -> plt.Figure:
    """Draw R plot_network's paired hierarchy panels on one Matplotlib figure."""
    fig, axes = plt.subplots(len(matrices), 2, figsize=(14, max(height, 3) * len(matrices)), squeeze=False)
    for row, (matrix, label) in enumerate(zip(matrices, labels)):
        edge_max = edge_weight_max_aggregate if label == "Aggregate" else edge_weight_max_individual
        panel_title = None if label == "Aggregate" else label
        plot_network_hierarchy_1(
            matrix, cluster_names, vertex_receiver, color_use=color_use,
            title_name=panel_title, vertex_weight=vertex_weight,
            vertex_size_max=vertex_size_max, edge_weight_max=edge_max,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            ax=axes[row, 0],
        )
        # R passes the complement to hierarchy2, whose left-side nodes are its
        # vertex_receiver argument.
        complement = [index for index in range(len(cluster_names)) if index not in vertex_receiver]
        plot_network_hierarchy_2(
            matrix, cluster_names, complement, color_use=color_use,
            title_name=panel_title, vertex_weight=vertex_weight,
            vertex_size_max=vertex_size_max, edge_weight_max=edge_max,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            ax=axes[row, 1],
        )
    if title:
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    return fig


# ---------------------------------------------------------------------------
# 19.  plot_network  - generic wrapper dispatching to layout-specific functions
# ---------------------------------------------------------------------------

def plot_network(
    cellchat: 'CellChat',
    signaling: Optional[str] = None,
    signaling_name: Optional[str] = None,
    color_use: Optional[List[str]] = None,
    vertex_receiver: Optional[List[int]] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    top: float = 1.0,
    remove_isolate: bool = False,
    vertex_weight: float = 1.0,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    weight_scale: bool = True,
    edge_weight_max_individual: Optional[float] = None,
    edge_weight_max_aggregate: Optional[float] = None,
    edge_width_max: float = 8.0,
    layout: str = "circle",
    height: float = 5.0,
    thresh: float = 0.05,
    pt_title: float = 12.0,
    title_space: float = 6.0,
    vertex_label_cex: float = 0.8,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
    **kwargs,
) -> Optional[plt.Figure]:
    """Generic network visualization dispatcher -mirrors R plot_network().

    Routes to ``plot_network_circle``, ``plot_network_chord_cell``, hierarchy
    plots (``plot_network_hierarchy_1`` / ``plot_network_hierarchy_2``), or a
    manual spatial layout depending on the *layout* parameter.

    Parameters
    ----------
    signaling : str
        Signaling pathway name to visualise.
    layout : {"circle", "hierarchy", "chord", "spatial"}
    vertex_receiver : list of int
        Indices of cell groups designated as receivers (hierarchy layout).
    """
    # --- resolve signalling pathway ---
    cluster_names = _get_cluster_names_from_cellchat(cellchat)
    try:
        net_data = _network_view_for_visualization(cellchat, "pathway_network")
    except ValueError:
        net_data = _network_view_for_visualization(cellchat, "network")

    if layout == "spatial":
        # Spatial layout reads the canonical spatial mapping and observation table.
        coords = cellchat.spatial.get('coordinates')
        meta_df = cellchat.obs
        return plot_network_spatial(
            cellchat, signaling=signaling,
            coordinates=coords, metadata=meta_df,
            color_use=color_use, thresh=thresh,
            title_name=signaling_name,
            vertex_size_max=vertex_size_max,
            edge_width_max=edge_width_max,
            fig_size=fig_size, return_fig=return_fig,
        )

    if layout == "chord":
        return plot_network_chord_cell(
            cellchat, signaling=signaling,
            sources_use=sources_use,
            targets_use=targets_use,
            color_use=color_use, thresh=thresh,
            title_name=signaling_name,
            font_size=int(vertex_label_cex * 12),
            fig_size=fig_size or (8, 8),
            return_fig=return_fig,
        )

    if layout == "hierarchy":
        if vertex_receiver is None:
            # Default: treat all as receivers, fall through to circle
            return plot_network_circle(
                cellchat, signaling=signaling, sources_use=sources_use,
                targets_use=targets_use, color_use=color_use, thresh=thresh,
                top=top, remove_isolate=remove_isolate,
                vertex_weight=vertex_weight,
                vertex_size_max=vertex_size_max or 15.0,
                edge_width_max=edge_width_max, title_name=signaling_name,
                vertex_label_cex=vertex_label_cex,
                fig_size=fig_size or (7, 7), return_fig=return_fig,
            )
        vertex_receiver = _validate_hierarchy_receivers(vertex_receiver, len(cluster_names))

        # R plot_network selects the L-R layer first, then draws each selected
        # pair and a final aggregate. Hierarchy is therefore an L-R-only view.
        lr_data = _network_view_for_visualization(cellchat, "network")
        pair_names, prob_array, pval_array = _network_arrays(lr_data)
        interactions = lr_data.get('interactions')

        if signaling is not None:
            if not isinstance(interactions, pd.DataFrame) or 'pathway_name' not in interactions.columns:
                raise ValueError("Network interactions must contain pathway_name to filter signaling")
            if 'interaction_name' in interactions.columns:
                pathway_by_pair = dict(zip(
                    interactions['interaction_name'].astype(str),
                    interactions['pathway_name'].astype(str),
                ))
            elif len(interactions) == len(pair_names):
                pathway_by_pair = {
                    pair_names[index]: str(interactions.iloc[index]['pathway_name'])
                    for index in range(len(pair_names))
                }
            else:
                raise ValueError("Network interactions are not aligned with the ligand-receptor matrices")
            pair_indices = [
                index for index, pair in enumerate(pair_names)
                if pathway_by_pair.get(str(pair)) == str(signaling)
            ]
            if not pair_indices:
                raise ValueError(f"No L-R pairs found for signaling pathway '{signaling}'")
        else:
            pair_indices = list(range(prob_array.shape[2]))

        matrices = []
        labels = []
        for index in pair_indices:
            matrix = prob_array[:, :, index].copy()
            matrix[pval_array[:, :, index] >= thresh] = 0.0
            matrix = _filter_hierarchy_matrix(matrix, cluster_names, sources_use, targets_use, top)
            if np.any(matrix > 0):
                matrices.append(matrix)
                labels.append(str(pair_names[index]))
        aggregate = np.sum(matrices, axis=0) if matrices else np.zeros(prob_array.shape[:2])
        aggregate = _filter_hierarchy_matrix(aggregate, cluster_names, sources_use, targets_use, top)
        if np.any(aggregate > 0):
            matrices.append(aggregate)
            labels.append("Aggregate")
        if not matrices:
            warnings.warn("No significant hierarchy interactions found")
            return None

        fig = _draw_hierarchy_grid(
            matrices, labels, cluster_names, vertex_receiver, color_use,
            vertex_weight, vertex_size_max or 15.0, edge_width_max,
            edge_weight_max_individual, edge_weight_max_aggregate,
            vertex_label_cex, height, signaling_name or signaling,
        )
        if return_fig:
            return fig
        plt.show()
        return None

    # default: circle
    return plot_network_circle(
        cellchat, slot_name="pathway_network", signaling=signaling, sources_use=sources_use,
        targets_use=targets_use, color_use=color_use, thresh=thresh,
        top=top, remove_isolate=remove_isolate,
        vertex_weight=vertex_weight,
        vertex_size_max=vertex_size_max or 15.0,
        edge_width_max=edge_width_max, title_name=signaling_name,
        vertex_label_cex=vertex_label_cex,
        fig_size=fig_size or (7, 7), return_fig=return_fig,
    )


# ---------------------------------------------------------------------------
# 20.  plot_network_hierarchy_1  - hierarchy plot: signaling TO vertex.receiver
# ---------------------------------------------------------------------------

def plot_network_hierarchy_1(
    net_mat: np.ndarray,
    cluster_names: List[str],
    vertex_receiver: List[int],
    color_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    thresh: float = 0.05,
    vertex_weight: float = 1.0,
    vertex_size_max: float = 15.0,
    edge_width_max: float = 8.0,
    vertex_label_cex: float = 0.8,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
    ax=None,
    edge_weight_max: Optional[float] = None,
) -> Optional[plt.Figure]:
    """Hierarchy plot showing signaling TO ``vertex_receiver`` cell groups.

    Source nodes (senders NOT in vertex_receiver) are placed on the left;
    target nodes (vertex_receiver) are placed on the right.  Directed edges
    are drawn as arrows from sources to targets.

    Mirrors R ``plot_network_hierarchy_1()``.
    """
    n = len(cluster_names)
    receiver_set = set(vertex_receiver)
    all_idx = list(range(n))
    source_idx = [i for i in all_idx if i not in receiver_set]
    target_idx = list(vertex_receiver)

    if len(source_idx) == 0 or len(target_idx) == 0:
        warnings.warn("Hierarchy plot requires both senders and receivers.")
        return None

    colors = _resolve_colors(n, color_use)
    color_map = dict(zip(cluster_names, colors))

    # --- node positions: sources left, targets right ---
    pos = {}
    y_spacing = max(1.0, 8.0 / max(1, n))
    x_src = -2.0
    x_tgt = 2.0

    for rank, si in enumerate(source_idx):
        y = rank * y_spacing - (len(source_idx) - 1) * y_spacing / 2
        pos[si] = (x_src, y)
    for rank, ti in enumerate(target_idx):
        y = rank * y_spacing - (len(target_idx) - 1) * y_spacing / 2
        pos[ti] = (x_tgt, y)

    # --- compute node sizes from row/col sums ---
    row_sum = np.sum(net_mat, axis=1)
    col_sum = np.sum(net_mat, axis=0)
    total_per_node = row_sum + col_sum
    t_min = total_per_node[total_per_node > 0].min() if (total_per_node > 0).any() else 0.0
    t_max = total_per_node.max()
    if t_max > t_min:
        sizes = np.interp(total_per_node, (t_min, t_max), (3.0, vertex_size_max))
    else:
        sizes = np.full(n, vertex_size_max / 2.0)

    sizes = sizes * vertex_weight

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # --- draw edges as arrows ---
    max_w = 0.0
    for si in source_idx:
        for ti in target_idx:
            w = net_mat[si, ti]
            if w > 0:
                max_w = max(max_w, w)
    if edge_weight_max is not None:
        if edge_weight_max <= 0:
            raise ValueError("edge_weight_max must be positive when provided")
        max_w = edge_weight_max

    for si in source_idx:
        for ti in target_idx:
            w = net_mat[si, ti]
            if w <= 0:
                continue
            x0, y0 = pos[si]
            x1, y1 = pos[ti]
            width = edge_width_max * (w / max_w) if max_w > 0 else 1.0
            c = color_map[cluster_names[si]]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='-|>', color=c, lw=width,
                    alpha=0.7, shrinkA=8, shrinkB=8,
                    connectionstyle='arc3,rad=0.1',
                ),
            )

    # --- draw nodes ---
    for i in range(n):
        x, y = pos[i]
        c = color_map[cluster_names[i]]
        ax.scatter(x, y, s=sizes[i] ** 2, c=c, edgecolors='black',
                   linewidths=0.8, zorder=5)
        offset_y = -0.35 - sizes[i] * 0.02
        ax.text(x, y + offset_y, cluster_names[i],
                ha='center', va='top', fontsize=vertex_label_cex * 10,
                color='black')

    # --- divider line ---
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', zorder=1)

    # --- axis limits ---
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    mx = max(abs(min(all_x)), abs(max(all_x))) + 1.0
    my = max(abs(min(all_y)), abs(max(all_y))) + 1.0
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-my, my)

    if title_name:
        ax.set_title(title_name, fontsize=12, pad=10)

    if own_figure:
        plt.tight_layout()
    else:
        return ax
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 21.  plot_network_hierarchy_2  - hierarchy plot: signaling NOT TO vertex.receiver
# ---------------------------------------------------------------------------

def plot_network_hierarchy_2(
    net_mat: np.ndarray,
    cluster_names: List[str],
    vertex_receiver: List[int],
    color_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    thresh: float = 0.05,
    vertex_weight: float = 1.0,
    vertex_size_max: float = 15.0,
    edge_width_max: float = 8.0,
    vertex_label_cex: float = 0.8,
    fig_size: Tuple[int, int] = (8, 6),
    return_fig: bool = False,
    ax=None,
    edge_weight_max: Optional[float] = None,
) -> Optional[plt.Figure]:
    """Hierarchy plot showing signaling TO cell groups NOT in ``vertex_receiver``.

    Source nodes (senders in vertex_receiver) are placed on the left;
    target nodes (NOT in vertex_receiver) are placed on the right.

    Mirrors R ``plot_network_hierarchy_2()`` -setdiff logic.
    """
    n = len(cluster_names)
    receiver_set = set(vertex_receiver)
    all_idx = list(range(n))
    source_idx = list(vertex_receiver)          # receivers are now senders
    target_idx = [i for i in all_idx if i not in receiver_set]  # non-receivers are targets

    if len(source_idx) == 0 or len(target_idx) == 0:
        warnings.warn("Hierarchy plot requires both senders and receivers.")
        return None

    colors = _resolve_colors(n, color_use)
    color_map = dict(zip(cluster_names, colors))

    pos = {}
    y_spacing = max(1.0, 8.0 / max(1, n))
    x_src = -2.0
    x_tgt = 2.0

    for rank, si in enumerate(source_idx):
        y = rank * y_spacing - (len(source_idx) - 1) * y_spacing / 2
        pos[si] = (x_src, y)
    for rank, ti in enumerate(target_idx):
        y = rank * y_spacing - (len(target_idx) - 1) * y_spacing / 2
        pos[ti] = (x_tgt, y)

    row_sum = np.sum(net_mat, axis=1)
    col_sum = np.sum(net_mat, axis=0)
    total_per_node = row_sum + col_sum
    t_min = total_per_node[total_per_node > 0].min() if (total_per_node > 0).any() else 0.0
    t_max = total_per_node.max()
    if t_max > t_min:
        sizes = np.interp(total_per_node, (t_min, t_max), (3.0, vertex_size_max))
    else:
        sizes = np.full(n, vertex_size_max / 2.0)
    sizes = sizes * vertex_weight

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    max_w = 0.0
    for si in source_idx:
        for ti in target_idx:
            w = net_mat[si, ti]
            if w > 0:
                max_w = max(max_w, w)
    if edge_weight_max is not None:
        if edge_weight_max <= 0:
            raise ValueError("edge_weight_max must be positive when provided")
        max_w = edge_weight_max

    for si in source_idx:
        for ti in target_idx:
            w = net_mat[si, ti]
            if w <= 0:
                continue
            x0, y0 = pos[si]
            x1, y1 = pos[ti]
            width = edge_width_max * (w / max_w) if max_w > 0 else 1.0
            c = color_map[cluster_names[si]]
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='-|>', color=c, lw=width,
                    alpha=0.7, shrinkA=8, shrinkB=8,
                    connectionstyle='arc3,rad=0.1',
                ),
            )

    for i in range(n):
        x, y = pos[i]
        c = color_map[cluster_names[i]]
        ax.scatter(x, y, s=sizes[i] ** 2, c=c, edgecolors='black',
                   linewidths=0.8, zorder=5)
        offset_y = -0.35 - sizes[i] * 0.02
        ax.text(x, y + offset_y, cluster_names[i],
                ha='center', va='top', fontsize=vertex_label_cex * 10,
                color='black')

    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', zorder=1)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    mx = max(abs(min(all_x)), abs(max(all_x))) + 1.0
    my = max(abs(min(all_y)), abs(max(all_y))) + 1.0
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-my, my)

    if title_name:
        ax.set_title(title_name, fontsize=12, pad=10)

    if own_figure:
        plt.tight_layout()
    else:
        return ax
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 22.  plot_network_spatial  - spatial plot of cell-cell communication
# ---------------------------------------------------------------------------

def plot_network_spatial(
    cellchat: 'CellChat',
    signaling: Optional[str] = None,
    coordinates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    metadata: Optional[pd.DataFrame] = None,
    color_use: Optional[List[str]] = None,
    thresh: float = 0.05,
    title_name: Optional[str] = None,
    vertex_size_max: float = 15.0,
    edge_width_max: float = 8.0,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Spatial plot showing cell group centroids and communication edges.

    Computes median coordinates per cell group, then draws edges between
    group centroids proportional to communication probability.

    Mirrors R ``plot_network_spatial()`` (ggplot2 + geom_curve + geom_point).
    """
    # --- resolve coordinates ---
    if coordinates is None:
        coordinate_frame = _spatial_coordinates(cellchat)
    else:
        coordinate_frame = pd.DataFrame(coordinates).copy()
        coordinate_frame.index = coordinate_frame.index.astype(str)
        cells = pd.Index(cellchat.obs_names.astype(str))
        if not cells.isin(coordinate_frame.index).all():
            raise ValueError("coordinates must contain every cell in cellchat.obs_names.")
        coordinate_frame = coordinate_frame.loc[cells]
    coord_arr = coordinate_frame.iloc[:, :2].to_numpy(dtype=float)
    if coord_arr.shape[1] < 2 or not np.isfinite(coord_arr).all():
        raise ValueError("coordinates must contain at least two finite numeric columns.")

    # --- resolve cell identities ---
    groups = cellchat.groups
    if groups is None:
        raise ValueError("cellchat.groups must be set before spatial network plotting.")
    cell_labels = [str(label) for label in groups]

    # The matrix axis, rather than metadata category order, defines the
    # network group order used below.
    net_data = _network_view_for_visualization(cellchat, "pathway_network")
    cluster_names = _network_group_names(cellchat, net_data)
    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    # --- compute group centroids (median) ---
    centroids = {}
    for ci, cn in enumerate(cluster_names):
        mask = np.asarray([lbl == cn for lbl in cell_labels], dtype=bool)
        if mask.any():
            centroids[ci] = np.median(coord_arr[mask], axis=0)
        else:
            centroids[ci] = np.array([np.nan, np.nan])

    # --- build pathway-level weight matrix ---
    if 'prob' not in net_data:
        raise ValueError("No probability data found.")

    if signaling is not None:
        if signaling not in net_data['prob']:
            raise ValueError(f"Signaling pathway '{signaling}' not found.")
        net_mat = _network_matrix(net_data, 'prob', signaling).copy()
        pval_matrix = _network_matrix(net_data, 'pval', signaling)
        net_mat[pval_matrix >= thresh] = 0.0
    else:
        _, prob_array, pval_array = _network_arrays(net_data)
        prob_array[pval_array >= thresh] = 0.0
        net_mat = np.sum(prob_array, axis=2)

    # --- draw ---
    if fig_size is None:
        fig_size = (7, 7)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)

    # Plot all cells as small background dots
    all_x = coord_arr[:, 0]
    all_y = coord_arr[:, 1]
    ax.scatter(all_x, all_y, c='lightgrey', s=3, alpha=0.5, zorder=1)

    # Plot edges between centroids
    max_w = 0.0
    for si in range(n_clusters):
        for ti in range(n_clusters):
            if si == ti:
                continue
            w = net_mat[si, ti] if si < net_mat.shape[0] and ti < net_mat.shape[1] else 0.0
            if w > max_w:
                max_w = w

    for si in range(n_clusters):
        for ti in range(n_clusters):
            if si >= net_mat.shape[0] or ti >= net_mat.shape[1]:
                continue
            w = net_mat[si, ti]
            if w <= 0 or si == ti:
                continue
            c0 = centroids.get(si)
            c1 = centroids.get(ti)
            if c0 is None or c1 is None or np.isnan(c0[0]) or np.isnan(c1[0]):
                continue
            width = edge_width_max * (w / max_w) if max_w > 0 else 1.0
            ec = color_map[cluster_names[si]]
            ax.annotate(
                '', xy=(c1[0], c1[1]), xytext=(c0[0], c0[1]),
                arrowprops=dict(
                    arrowstyle='-|>', color=ec, lw=width,
                    alpha=0.6, shrinkA=5, shrinkB=5,
                    connectionstyle='arc3,rad=0.15',
                ),
                zorder=2,
            )

    # Plot centroids on top
    for ci in range(n_clusters):
        c = centroids.get(ci)
        if c is None or np.isnan(c[0]):
            continue
        ax.scatter(*c, s=vertex_size_max ** 2, c=color_map[cluster_names[ci]],
                   edgecolors='black', linewidths=1.0, zorder=5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=color_map[cn], label=cn) for cn in cluster_names
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=False,
              title='Cell group', fontsize=8)

    ax.set_xlabel('Spatial 1', fontsize=10)
    ax.set_ylabel('Spatial 2', fontsize=10)
    if title_name:
        ax.set_title(title_name, fontsize=12, pad=8)
    else:
        ax.set_title(f'Spatial communication: {signaling or "all"}', fontsize=12, pad=8)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 23.  plot_network_diff_interaction  - differential interaction circle plot
# ---------------------------------------------------------------------------

def plot_network_diff_interaction(
    cellchat: 'CellChat',
    comparison: Tuple[int, int] = (0, 1),
    measure: str = "count",
    color_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[List[str]] = None,
    targets_use: Optional[List[str]] = None,
    remove_isolate: bool = False,
    edge_width_max: float = 8.0,
    vertex_size_max: float = 15.0,
    vertex_label_cex: float = 0.8,
    label_edge: bool = False,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.8,
    edge_curved: float = 0.15,
    alpha_edge: float = 0.6,
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Circle plot of differential interactions between two datasets.

    Red edges = increased (dataset2 > dataset1), blue edges = decreased.
    Edge width = magnitude of difference.

    Mirrors R ``plot_network_diff_interaction()``.
    """
    networks, resolved_comparison = _comparison_networks(
        cellchat, "network", comparison
    )
    if len(networks) != 2 or resolved_comparison is None:
        raise ValueError(
            "plot_network_diff_interaction requires a merged CellChat object "
            "with at least two per-dataset networks."
        )
    (name1, net1), (name2, net2) = networks
    idx1, idx2 = resolved_comparison

    mat1 = net1.get(measure, net1.get('prob'))
    mat2 = net2.get(measure, net2.get('prob'))
    if mat1 is None or mat2 is None:
        raise ValueError(f"Measure '{measure}' was not found in the selected network.")

    mat1 = np.array(mat1, dtype=float)
    mat2 = np.array(mat2, dtype=float)

    # Ensure 2D
    if mat1.ndim == 3:
        mat1 = mat1.sum(axis=2)
    if mat2.ndim == 3:
        mat2 = mat2.sum(axis=2)

    groups1 = list(net1.get('groups', [f'g{i}' for i in range(mat1.shape[0])]))
    groups2 = list(net2.get('groups', [f'g{i}' for i in range(mat2.shape[0])]))
    if set(groups1) != set(groups2) or len(groups1) != mat1.shape[0] or len(groups2) != mat2.shape[0]:
        raise ValueError("Compared networks must contain the same groups with matching dimensions.")
    if mat1.shape != (len(groups1), len(groups1)) or mat2.shape != (len(groups2), len(groups2)):
        raise ValueError("Compared network measures must be square group-by-group matrices.")
    order2 = [groups2.index(group) for group in groups1]
    mat2 = mat2[np.ix_(order2, order2)]
    diff_mat = mat2 - mat1

    groups = groups1
    n = len(groups)
    colors = _resolve_colors(n, color_use)
    color_map = dict(zip(groups, colors))

    # Filter sources / targets
    if sources_use is not None:
        mask_r = [i for i, g in enumerate(groups) if g not in sources_use]
        diff_mat[mask_r, :] = 0.0
    if targets_use is not None:
        mask_c = [i for i, g in enumerate(groups) if g not in targets_use]
        diff_mat[:, mask_c] = 0.0

    if remove_isolate:
        keep = np.where((np.abs(diff_mat).sum(axis=1) + np.abs(diff_mat).sum(axis=0)) > 0)[0]
        if len(keep) == 0:
            warnings.warn("No differential interactions to plot.")
            return None
        diff_mat = diff_mat[np.ix_(keep, keep)]
        groups = [groups[i] for i in keep]
        colors = [colors[i] for i in keep]
        color_map = dict(zip(groups, colors))
        n = len(groups)

    # --- circular layout ---
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    R = 1.0
    pos = {g: (R * np.cos(a), R * np.sin(a)) for g, a in zip(groups, angles)}

    # --- node sizes from absolute total interaction ---
    abs_total = np.abs(diff_mat).sum(axis=1) + np.abs(diff_mat).sum(axis=0)
    a_min = abs_total[abs_total > 0].min() if (abs_total > 0).any() else 0.0
    a_max = abs_total.max()
    if a_max > a_min:
        node_sizes = np.interp(abs_total, (a_min, a_max), (3.0, vertex_size_max))
    else:
        node_sizes = np.full(n, vertex_size_max / 2.0)

    if fig_size is None:
        fig_size = (8, 8)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # --- edge max ---
    max_abs = np.max(np.abs(diff_mat)) if np.any(diff_mat) else 1.0

    for si in range(n):
        for ti in range(n):
            d = diff_mat[si, ti]
            if d == 0:
                continue
            x0, y0 = pos[groups[si]]
            x1, y1 = pos[groups[ti]]
            width = edge_width_max * (abs(d) / max_abs)
            edge_color = '#e41a1c' if d > 0 else '#377eb8'  # red up, blue down
            if si == ti:
                _, label_position = _draw_circular_self_loop(
                    ax, (x0, y0), angles[si], color=edge_color,
                    linewidth=width, alpha=alpha_edge,
                    mutation_scale=6 * arrow_size / 0.2 * arrow_width,
                )
            else:
                mutation = 6 * arrow_size / 0.2 * arrow_width
                ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle='-|>', color=edge_color, lw=width,
                        alpha=alpha_edge, shrinkA=10, shrinkB=10,
                        connectionstyle=f'arc3,rad={edge_curved}',
                    ),
                )
                label_position = _curved_edge_label_position(
                    (x0, y0), (x1, y1), edge_curved
                )

            if label_edge:
                edge_text = ax.text(
                    *label_position, _format_edge_label(d),
                    color=edge_label_color,
                    fontsize=max(1.0, edge_label_cex * 10),
                    ha='center', va='center', zorder=7,
                )
                edge_text.set_gid('network-edge-label')

    # --- nodes ---
    for i, g in enumerate(groups):
        x, y = pos[g]
        ax.scatter(x, y, s=node_sizes[i] ** 2, c=color_map[g],
                   edgecolors='black', linewidths=0.8, zorder=5)
        # label outside circle
        label_angle = angles[i]
        rad_offset = R + 0.15
        lx = rad_offset * np.cos(label_angle)
        ly = rad_offset * np.sin(label_angle)
        ha = 'left' if np.cos(label_angle) >= 0 else 'right'
        ax.text(lx, ly, g, ha=ha, va='center',
                fontsize=vertex_label_cex * 10, color='black')

    # Legend
    legend_handles = [
        mpatches.Patch(color='#e41a1c', label='Increased'),
        mpatches.Patch(color='#377eb8', label='Decreased'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    if title_name:
        ax.set_title(title_name, fontsize=12, pad=10)
    else:
        ax.set_title(f'Differential interactions: {name2} vs {name1}',
                     fontsize=12, pad=10)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 24.  plot_network_embedding_zoom_in  - faceted embedding scatter by cluster
# ---------------------------------------------------------------------------

def plot_network_embedding_zoom_in(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    emb_type: str = "functional",
    color_use: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 6),
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
    do_label: bool = True,
    ncol: int = 3,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Faceted scatter showing pathway embeddings zoomed into each cluster.

    Each subplot shows only pathways belonging to one cluster, with all
    other points grayed out.  Reads from ``cellchat.pathway_network['similarity'][type]``.

    Mirrors R ``plot_network_embedding_zoom_in()``.
    """
    dot_size = _validate_dot_size(dot_size)
    if ncol <= 0:
        raise ValueError("ncol must be a positive integer.")
    net_data = _network_view_for_visualization(cellchat, slot_name)
    similarity_slot = cellchat.pathway_network if slot_name == 'pathway_network' else cellchat.network
    sim = similarity_slot.get('similarity', {}).get(emb_type, {})
    dr = sim.get('dr', {}).get('single', None)
    groups = sim.get('group', {}).get('single', None)

    if dr is None:
        raise ValueError(
            f"No embedding found for type='{emb_type}'. "
            "Run compute_pairwise_network_similarity first."
        )

    dr = np.array(dr)
    n_pts = dr.shape[0]
    pathway_names = list(sim.get('pathways', network_names(net_data)[:n_pts]))
    if len(pathway_names) != n_pts:
        raise ValueError("Embedding coordinates and pathway labels must have matching rows.")

    if groups is None:
        groups = ['unknown'] * n_pts

    unique_groups = sorted(set(groups))
    gg_cols = gg_palette(len(unique_groups))
    resolved = _resolve_colors(len(unique_groups), color_use)
    group_color_map = dict(zip(
        unique_groups,
        resolved if color_use is not None else gg_cols,
    ))

    # dot size proportional to total communication probability
    prob = net_data.get('prob')
    if isinstance(prob, dict):
        prob_sum = _network_strengths(net_data, pathway_names)
        prob_norm = prob_sum / prob_sum.max() if prob_sum.max() > 0 else np.ones(n_pts)
    else:
        prob_norm = np.ones(n_pts)

    n_groups = len(unique_groups)
    nrow = int(np.ceil(n_groups / ncol))

    if fig_size is None:
        fig_size = (ncol * 4.0, nrow * 3.5)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    for pi, grp in enumerate(unique_groups):
        r, c = divmod(pi, ncol)
        ax = axes[r, c]

        idx = [i for i, gg in enumerate(groups) if gg == grp]
        xs = dr[idx, 0]
        ys = dr[idx, 1]
        sizes = np.interp(prob_norm[idx], (0, 1), dot_size) ** 2 * 5
        col = group_color_map[grp]
        ax.scatter(xs, ys, s=sizes,
                   c=[to_rgba(col, alpha=dot_alpha)] * len(xs),
                   edgecolors=col, linewidths=0.5, zorder=3)

        # Labels for highlighted group
        if do_label:
            idx_grp = [i for i, gg in enumerate(groups) if gg == grp]
            for i in idx_grp:
                ax.text(dr[i, 0], dr[i, 1], pathway_names[i],
                        fontsize=font_size - 2, ha='center', va='bottom',
                        color='black', alpha=0.7)

        ax.set_title(grp, fontsize=font_size_title - 1)
        ax.set_xlabel(xlabel, fontsize=font_size - 2)
        ax.set_ylabel(ylabel, fontsize=font_size - 2)
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
        ax.tick_params(labelsize=font_size - 3)

    # Hide unused axes
    for pi in range(n_groups, nrow * ncol):
        r, c = divmod(pi, ncol)
        if r < axes.shape[0] and c < axes.shape[1]:
            axes[r, c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=font_size_title + 2, fontweight='normal')

    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.93)

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 25.  plot_network_embedding_pairwise  - joint manifold from two datasets
# ---------------------------------------------------------------------------

def _pairwise_embedding_parts(sim, net_data, comparison):
    """Split a canonical pairwise embedding into dataset-specific points."""
    dr_store = sim.get('dr', {})
    group_store = sim.get('group', {})
    comparison_key = sim.get('comparison_name')
    if comparison is not None:
        if len(comparison) < 2 or any(int(index) < 0 for index in comparison[:2]):
            raise ValueError("comparison must contain two valid zero-based dataset indices.")
        comparison_key = '-'.join(str(int(index)) for index in comparison[:2])
    elif comparison_key not in dr_store:
        keys = list(dr_store)
        if len(keys) == 1:
            comparison_key = keys[0]

    embedding = dr_store.get(comparison_key)
    group_labels = group_store.get(comparison_key)
    pathways_store = sim.get('pathways', {})
    pathway_names = (list(pathways_store.get(comparison_key, []))
                     if isinstance(pathways_store, dict)
                     else list(pathways_store))

    if embedding is None or group_labels is None:
        raise ValueError(
            "No canonical pairwise embedding was found. Run "
            "compute_pairwise_network_similarity() and embed_network() first."
        )

    embedding = np.asarray(embedding)
    group_labels = np.asarray(group_labels, dtype=object)
    if embedding.ndim != 2 or embedding.shape[0] != len(group_labels):
        raise ValueError("Pairwise embedding and cluster labels must have matching rows.")
    if len(pathway_names) != embedding.shape[0]:
        raise ValueError(
            "The canonical pairwise embedding has no matching pathway labels. "
            "Recompute the pairwise similarity and embedding."
        )

    dataset_names = list(sim.get('dataset_names', []))
    dataset_indices = list(sim.get('comparison', []))
    if not dataset_names or len(dataset_indices) != len(dataset_names):
        raise ValueError(
            "Canonical pairwise similarity metadata must contain matching "
            "dataset_names and comparison entries."
        )

    all_dr, all_groups, labels = [], [], []
    for local_index, dataset_name in enumerate(dataset_names):
        suffix = f'--{dataset_name}'
        mask = np.array([str(name).endswith(suffix) for name in pathway_names])
        if np.any(mask):
            all_dr.append(embedding[mask])
            all_groups.append(group_labels[mask].tolist())
            labels.append(f'Dataset {dataset_indices[local_index] + 1}')
    if not all_dr:
        raise ValueError(
            "Canonical pairwise pathway labels do not match the selected datasets. "
            "Recompute the pairwise similarity and embedding."
        )
    return all_dr, all_groups, labels, pathway_names


def plot_network_embedding_pairwise(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    emb_type: str = "functional",
    comparison: Optional[Tuple[int, int]] = None,
    color_use: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 6),
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
    do_label: bool = True,
    pathway_labeled: Optional[List[str]] = None,
    show_legend: bool = True,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Joint manifold learning scatter from two datasets.

    Points from dataset 1 are circles, dataset 2 are triangles.
    Groups (clusters) are distinguished by colour.

    Reads the comparison-keyed embedding in
    ``cellchat.network_similarity[slot_name][type]['dr']``. ``comparison`` uses
    Python zero-based dataset indices.

    Mirrors R ``plot_network_embedding_pairwise()``.
    """
    dot_size = _validate_dot_size(dot_size)
    net_data = _network_view_for_visualization(cellchat, slot_name)
    sim = _pairwise_similarity_data(cellchat, slot_name, emb_type)
    all_dr, all_groups, labels, pathway_names = _pairwise_embedding_parts(
        sim, net_data, comparison
    )
    prob_norms = _pairwise_probability_norm(
        cellchat, slot_name, comparison, pathway_names, [len(points) for points in all_dr]
    )

    unique_groups = sorted(set(sum((list(g) for g in all_groups), [])))
    gg_cols = gg_palette(len(unique_groups))
    resolved = _resolve_colors(len(unique_groups), color_use)
    group_color_map = dict(zip(
        unique_groups,
        resolved if color_use is not None else gg_cols,
    ))

    shapes = ['o', '^', 's', 'D', 'v']
    if fig_size is None:
        fig_size = (9, 7)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)

    for di, (dr_mat, grps) in enumerate(zip(all_dr, all_groups)):
        dr_mat = np.array(dr_mat)
        marker = shapes[di % len(shapes)]
        for g in unique_groups:
            idx = [i for i, gg in enumerate(grps) if gg == g]
            if not idx:
                continue
            xs = dr_mat[idx, 0]
            ys = dr_mat[idx, 1]
            c = group_color_map[g]
            sizes = np.interp(prob_norms[di][idx], (0, 1), dot_size) ** 2 * 5
            ax.scatter(xs, ys, marker=marker,
                       s=sizes,
                       c=[to_rgba(c, alpha=dot_alpha)] * len(xs),
                       edgecolors=c, linewidths=0.5,
                       label=f'{g} ({labels[di]})' if show_legend else None,
                       zorder=3)

    if do_label:
        if pathway_labeled:
            labels_to_show = set(pathway_labeled)
        else:
            labels_to_show = set(pathway_names)

        dr0 = np.array(all_dr[0])
        for i, nm in enumerate(pathway_names):
            if nm in labels_to_show and i < dr0.shape[0]:
                ax.text(dr0[i, 0], dr0[i, 1], nm,
                        fontsize=font_size - 2, ha='center', va='bottom',
                        color='black', alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(labelsize=font_size - 2)

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  frameon=False, fontsize=font_size - 2)

    if title:
        ax.set_title(title, fontsize=font_size_title, pad=10)
    else:
        ax.set_title(f'Joint embedding ({emb_type})', fontsize=font_size_title, pad=10)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 26.  plot_network_embedding_pairwise_zoom_in  - pairwise zoom to clusters
# ---------------------------------------------------------------------------

def plot_network_embedding_pairwise_zoom_in(
    cellchat: 'CellChat',
    slot_name: str = "pathway_network",
    emb_type: str = "functional",
    comparison: Optional[Tuple[int, int]] = None,
    color_use: Optional[List[str]] = None,
    dot_size: Tuple[float, float] = (2, 6),
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
    do_label: bool = True,
    ncol: int = 3,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Faceted pairwise embedding scatter zoomed into each cluster.

    Each subplot highlights pathways belonging to one cluster from both
    datasets, with other points grayed out.  Shapes distinguish datasets.

    Mirrors R ``plot_network_embedding_pairwise_zoom_in()``.
    """
    dot_size = _validate_dot_size(dot_size)
    if ncol <= 0:
        raise ValueError("ncol must be a positive integer.")
    net_data = _network_view_for_visualization(cellchat, slot_name)
    sim = _pairwise_similarity_data(cellchat, slot_name, emb_type)
    all_dr, all_groups, labels, pathway_names = _pairwise_embedding_parts(
        sim, net_data, comparison
    )
    prob_norms = _pairwise_probability_norm(
        cellchat, slot_name, comparison, pathway_names, [len(points) for points in all_dr]
    )

    unique_groups = sorted(set(sum((list(g) for g in all_groups), [])))
    gg_cols = gg_palette(len(unique_groups))
    resolved = _resolve_colors(len(unique_groups), color_use)
    group_color_map = dict(zip(
        unique_groups,
        resolved if color_use is not None else gg_cols,
    ))

    shapes = ['o', '^', 's', 'D', 'v']
    n_groups = len(unique_groups)
    nrow = int(np.ceil(n_groups / ncol))

    if fig_size is None:
        fig_size = (ncol * 4.5, nrow * 4.0)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    for pi, grp in enumerate(unique_groups):
        r, c = divmod(pi, ncol)
        ax = axes[r, c]

        # Gray background
        for di, (dr_mat, grps) in enumerate(zip(all_dr, all_groups)):
            for g in unique_groups:
                idx = [i for i, gg in enumerate(grps) if gg == g]
                if not idx:
                    continue
                xs = dr_mat[idx, 0]
                ys = dr_mat[idx, 1]
                sizes = np.interp(prob_norms[di][idx], (0, 1), dot_size) ** 2 * 5
                if g == grp:
                    marker = shapes[di % len(shapes)]
                    col = group_color_map[g]
                    ax.scatter(xs, ys, marker=marker,
                               s=sizes,
                               c=[to_rgba(col, alpha=dot_alpha)] * len(xs),
                               edgecolors=col, linewidths=0.5, zorder=3,
                               label=labels[di])
                else:
                    ax.scatter(xs, ys, s=sizes, c='lightgrey', alpha=0.25,
                               edgecolors='none', zorder=1)

        if do_label:
            # Label points from first dataset
            dr0 = all_dr[0]
            grp0 = all_groups[0]
            for i, (nm, g) in enumerate(zip(pathway_names, grp0)):
                if g == grp and i < dr0.shape[0]:
                    ax.text(dr0[i, 0], dr0[i, 1], nm,
                            fontsize=font_size - 3, ha='center', va='bottom',
                            color='black', alpha=0.7)

        ax.set_title(grp, fontsize=font_size_title - 1)
        ax.set_xlabel(xlabel, fontsize=font_size - 2)
        ax.set_ylabel(ylabel, fontsize=font_size - 2)
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
        ax.tick_params(labelsize=font_size - 3)

        if pi == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 0),
                      frameon=False, fontsize=font_size - 3)

    for pi in range(n_groups, nrow * ncol):
        r, c = divmod(pi, ncol)
        if r < axes.shape[0] and c < axes.shape[1]:
            axes[r, c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=font_size_title + 2)
    else:
        fig.suptitle(f'Pairwise zoom-in ({emb_type})', fontsize=font_size_title + 2)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)

    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 27.  plot_spatial_dim  - spatial positions coloured by cell group
# ---------------------------------------------------------------------------

def plot_spatial_dim(
    cellchat: 'CellChat',
    color_use: Optional[List[str]] = None,
    title_name: Optional[str] = None,
    pt_size: float = 3.0,
    pt_alpha: float = 0.8,
    do_raster: bool = False,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot spatial positions coloured by cell group identity.

    Simple scatter plot of spatial coordinates where each cell is coloured
    by its assigned cell group.

    Mirrors R ``plot_spatial_dim()`` (Seurat/ggplot2).
    """
    coordinate_frame = _spatial_coordinates(cellchat)
    coord_arr = coordinate_frame.iloc[:, :2].to_numpy(dtype=float)
    groups = cellchat.groups
    if groups is None:
        raise ValueError("cellchat.groups must be set before spatial plotting.")
    cell_labels = list(groups)
    cluster_names = list(groups.categories)

    n_clusters = len(cluster_names)
    colors = _resolve_colors(n_clusters, color_use)
    color_map = dict(zip(cluster_names, colors))

    point_colors = [color_map.get(lbl, '#999999') for lbl in cell_labels]

    if fig_size is None:
        fig_size = (7, 7)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)

    if do_raster:
        ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=point_colors,
                   s=pt_size ** 2, alpha=pt_alpha, edgecolors='none',
                   rasterized=True)
    else:
        ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=point_colors,
                   s=pt_size ** 2, alpha=pt_alpha, edgecolors='none')

    legend_handles = [
        mpatches.Patch(color=color_map[cn], label=cn) for cn in cluster_names
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=False,
              title='Cell group', fontsize=8)

    ax.set_xlabel('Spatial 1', fontsize=10)
    ax.set_ylabel('Spatial 2', fontsize=10)
    ax.set_aspect('equal')

    if title_name:
        ax.set_title(title_name, fontsize=12, pad=8)
    else:
        ax.set_title('Spatial dim plot', fontsize=12, pad=8)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 28.  plot_spatial_feature  - spatial positions coloured by feature values
# ---------------------------------------------------------------------------

def plot_spatial_feature(
    cellchat: 'CellChat',
    features: Optional[List[str]] = None,
    signaling: Optional[str] = None,
    slot_data: str = "x",
    color_use: Optional[List[str]] = None,
    color_bar: str = "viridis",
    pt_size: float = 2.0,
    pt_alpha: float = 0.9,
    ncol: int = 3,
    do_raster: bool = False,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot spatial positions coloured by gene expression or metadata features.

    For continuous features (gene expression): colour mapped via *color_bar*.
    For binary features (e.g. ligand/receptor expression > 0): red/grey split.

    Mirrors R ``plot_spatial_feature()`` (Seurat).
    """
    coordinate_frame = _spatial_coordinates(cellchat)
    coord_arr = coordinate_frame.iloc[:, :2].to_numpy(dtype=float)
    cell_names = coordinate_frame.index.tolist()
    if coord_arr.ndim != 2 or coord_arr.shape[1] < 2:
        raise ValueError("Spatial coordinates must be a two-dimensional array with at least two columns.")
    if coord_arr.shape[0] == 0:
        raise ValueError("Spatial coordinates contain no cells.")
    if ncol <= 0:
        raise ValueError("ncol must be a positive integer.")
    if pt_size < 0:
        raise ValueError("pt_size must be non-negative.")

    # Determine feature values
    if features is not None:
        features = _validate_expression_plot_inputs(features, ncol=ncol)
        expression = _expression_feature_dataframe(cellchat, slot_data)
        missing = [feature for feature in features if str(feature) not in expression.index]
        if missing:
            raise ValueError(f"Features not found in slot '{slot_data}': {missing}")
        feature_vals = {
            str(feature): expression.loc[str(feature)].reindex(cell_names).to_numpy(dtype=float)
            for feature in features
        }
        binary_features = set()
    elif signaling is not None:
        # Binary mode: ligand/receptor expression for a signaling pathway
        database = cellchat.database
        interaction_db = database.get('interaction', pd.DataFrame())
        if interaction_db.empty:
            raise ValueError("No interaction database found; cannot resolve ligands/receptors.")

        sig_rows = (interaction_db[interaction_db['pathway_name'] == signaling]
                    if 'pathway_name' in interaction_db.columns else interaction_db.iloc[0:0])
        if sig_rows.empty:
            from .database import search_pair
            sig_rows = search_pair(signaling, interaction_db, key='pathway_name', pair_only=True)

        ligands = set(sig_rows.get('ligand', pd.Series(dtype=str)).dropna().unique())
        receptors = set(sig_rows.get('receptor', pd.Series(dtype=str)).dropna().unique())
        ligand_genes = [g for g in ligands if isinstance(g, str)]
        receptor_genes = [g for g in receptors if isinstance(g, str)]
        if not ligand_genes and not receptor_genes:
            raise ValueError(f"No ligand/receptor genes found for signaling '{signaling}'.")
        expression = _expression_feature_dataframe(cellchat, 'x')

        def _binary_for_genes(genes):
            available = [gene for gene in genes if gene in expression.index]
            if not available:
                return None
            values = expression.loc[available].reindex(columns=cell_names).to_numpy(dtype=float)
            return np.any(values > 0, axis=0).astype(float)

        lig_expr = _binary_for_genes(ligand_genes)
        rec_expr = _binary_for_genes(receptor_genes)

        feature_vals = {}
        binary_features = set()
        if lig_expr is not None:
            feature_vals[f'{signaling} (ligand)'] = lig_expr
            binary_features.add(f'{signaling} (ligand)')
        if rec_expr is not None:
            feature_vals[f'{signaling} (receptor)'] = rec_expr
            binary_features.add(f'{signaling} (receptor)')
        if not feature_vals:
            raise ValueError(f"None of the ligand/receptor genes for '{signaling}' are present in cellchat.X.")
    else:
        raise ValueError("Provide 'features' (non-empty list) or 'signaling' (pathway name).")

    feature_names = list(feature_vals.keys())
    n_feats = len(feature_names)
    nrow = int(np.ceil(n_feats / ncol))
    ncol_actual = min(ncol, n_feats)

    if fig_size is None:
        fig_size = (ncol_actual * 3.5, nrow * 3.5)

    fig, axes = plt.subplots(nrow, ncol_actual, figsize=fig_size)
    if nrow == 1 and ncol_actual == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol_actual == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.cm.get_cmap(color_bar)

    for fi, fname in enumerate(feature_names):
        r, c = divmod(fi, ncol_actual)
        ax = axes[r, c]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)

        vals = feature_vals[fname]
        finite_mask = np.isfinite(vals)

        if fname in binary_features:
            # Binary mode
            binary_colors = list(color_use) if color_use is not None else ['#e41a1c', '#d3d3d3']
            if len(binary_colors) == 1:
                binary_colors.append('#d3d3d3')
            colors = np.where(vals > 0, binary_colors[0], binary_colors[1])
            if do_raster:
                ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=colors,
                           s=pt_size ** 2, alpha=pt_alpha, edgecolors='none',
                           rasterized=True)
            else:
                ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=colors,
                           s=pt_size ** 2, alpha=pt_alpha, edgecolors='none')
        else:
            # Continuous mode
            vmin = np.nanmin(vals[finite_mask]) if finite_mask.any() else 0
            vmax = np.nanmax(vals[finite_mask]) if finite_mask.any() else 1
            if vmax == vmin:
                vmax = vmin + 1
            norm = Normalize(vmin, vmax)
            mapped = cmap(norm(vals))
            if do_raster:
                ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=mapped,
                           s=pt_size ** 2, alpha=pt_alpha, edgecolors='none',
                           rasterized=True)
            else:
                ax.scatter(coord_arr[:, 0], coord_arr[:, 1], c=mapped,
                           s=pt_size ** 2, alpha=pt_alpha, edgecolors='none')
            # Colorbar
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20,
                         label=fname)

        ax.set_title(fname, fontsize=10)
        ax.set_xlabel('Spatial 1', fontsize=8)
        ax.set_ylabel('Spatial 2', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Hide unused
    for fi in range(n_feats, nrow * ncol_actual):
        r, c = divmod(fi, ncol_actual)
        if r < axes.shape[0] and c < axes.shape[1]:
            axes[r, c].set_visible(False)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 29.  plot_dot  - dot plot of gene expression
# ---------------------------------------------------------------------------

def plot_dot(
    cellchat: 'CellChat',
    features: List[str],
    group_by: Optional[str] = None,
    slot_data: str = "x",
    scale: bool = True,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    dot_scale: float = 6.0,
    color_use: Optional[List[str]] = None,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Dot plot of gene expression: size = percent expressed, colour = avg expression.

    Mirrors R Seurat ``DotPlot()``.
    """
    features = _validate_expression_plot_inputs(features, dot_scale=dot_scale)
    if scale_min is not None and scale_max is not None and scale_min > scale_max:
        raise ValueError("scale_min must not be greater than scale_max.")
    data_mat = _expression_feature_dataframe(cellchat, slot_data, features)
    missing = [feat for feat in features if str(feat) not in data_mat.index]
    if missing:
        raise ValueError(f"Features not found in slot '{slot_data}': {missing}")
    cell_names = list(data_mat.columns)
    groups, cluster_names = _resolve_expression_groups(cellchat, cell_names, group_by)

    n_groups = len(cluster_names)
    n_feats = len(features)

    # Compute percent expressed and mean expression per group
    pct_exp = np.zeros((n_groups, n_feats))
    avg_exp = np.zeros((n_groups, n_feats))

    for gi, grp in enumerate(cluster_names):
        grp_cells = groups.index[groups == grp].tolist()
        if not grp_cells:
            continue
        for fi, feat in enumerate(features):
            vals = data_mat.loc[str(feat), grp_cells].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positive = vals > 0
            pct_exp[gi, fi] = np.mean(positive) * 100
            avg_exp[gi, fi] = np.mean(vals[positive]) if positive.any() else 0.0

    # Scale
    if scale:
        for fi in range(n_feats):
            col = avg_exp[:, fi]
            nonzero = col > 0
            if nonzero.any():
                c_min = col[nonzero].min()
                c_max = col[nonzero].max()
                if c_max > c_min:
                    scaled = -2.5 + (col - c_min) / (c_max - c_min) * 5.0
                    scaled = np.where(col > 0, scaled, 0.0)
                else:
                    scaled = np.where(col > 0, 0.0, 0.0)
                if scale_min is not None:
                    scaled = np.where(col > 0, np.maximum(scaled, scale_min), 0.0)
                if scale_max is not None:
                    scaled = np.where(col > 0, np.minimum(scaled, scale_max), 0.0)
                avg_exp[:, fi] = scaled

    if fig_size is None:
        fig_size = (max(5, n_feats * 0.7 + 2), max(4, n_groups * 0.45 + 1.5))

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_facecolor('white')

    # Color range for expression, including negative values after scaling.
    expressed = avg_exp[pct_exp > 0]
    vmin = expressed.min() if expressed.size else 0.0
    vmax = expressed.max() if expressed.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1

    # Color for zero-expression dots
    zero_color = '#e0e0e0'

    for gi in range(n_groups):
        for fi in range(n_feats):
            pe = pct_exp[gi, fi]
            ae = avg_exp[gi, fi]
            if pe > 0:
                frac = (ae - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                palette = list(color_use) if color_use is not None else ['#2166ac', '#f7f7f7', '#b2182b']
                if len(palette) == 1:
                    palette = ['#f7f7f7', palette[0]]
                cmap = LinearSegmentedColormap.from_list('dotplot', palette)
                dot_color = cmap(np.clip(frac, 0.0, 1.0))
            else:
                dot_color = to_rgba(zero_color)[:3]

            size = dot_scale * (pe / 100.0) * 40
            ax.scatter(fi, gi, s=size, facecolors=[dot_color],
                       edgecolors='black', linewidths=0.3, zorder=3)

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(cluster_names, fontsize=9)
    ax.set_xlim(-0.5, n_feats - 0.5)
    ax.set_ylim(-0.5, n_groups - 0.5)
    ax.invert_yaxis()

    # Grid
    ax.grid(True, color='#e5e5e5', linewidth=0.5, zorder=0)
    ax.tick_params(which='both', length=2, width=0.5)

    # Size legend
    pct_ticks = [25, 50, 75, 100]
    legend_sizes = [dot_scale * (p / 100.0) * 40 for p in pct_ticks]
    legend_handles = []
    for p, sz in zip(pct_ticks, legend_sizes):
        legend_handles.append(
            ax.scatter([], [], s=sz, facecolors='grey',
                       edgecolors='black', linewidths=0.3,
                       label=f'{p}%')
        )
    ax.legend(handles=legend_handles, title='Percent expressed',
              title_fontsize=9, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=False,
              handletextpad=1.0)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 30.  plot_stacked_violin  - stacked violin plots of gene expression
# ---------------------------------------------------------------------------

def plot_stacked_violin(
    cellchat: 'CellChat',
    features: List[str],
    group_by: Optional[str] = None,
    slot_data: str = "x",
    pt_size: float = 0.05,
    same_y_lims: bool = False,
    color_use: Optional[List[str]] = None,
    split_by: Optional[str] = None,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Stacked violin plots of gene expression per cell group.

    One row per feature, one violin per cell group.  Mirrors R
    ``plot_stacked_violin()`` (Seurat + patchwork).
    """
    features = _validate_expression_plot_inputs(features, pt_size=pt_size)
    data_mat = _expression_feature_dataframe(cellchat, slot_data, features)
    missing = [feat for feat in features if str(feat) not in data_mat.index]
    if missing:
        raise ValueError(f"Features not found in slot '{slot_data}': {missing}")
    groups, cluster_names = _resolve_expression_groups(
        cellchat, list(data_mat.columns), group_by
    )

    n_groups = len(cluster_names)
    n_feats = len(features)
    colors = _resolve_colors(n_groups, color_use)
    group_colors = dict(zip(cluster_names, colors))

    split_groups = None
    split_levels = None
    split_colors = None
    if split_by is not None:
        split_groups, split_levels = _resolve_expression_groups(
            cellchat, list(data_mat.columns), split_by
        )
        split_colors = _resolve_colors(len(split_levels), color_use)

    if fig_size is None:
        fig_size = (max(6, n_groups * 0.7 + 2), n_feats * 2.5)

    # Compute global y limits
    global_ymin, global_ymax = np.inf, -np.inf
    if same_y_lims:
        for feat in features:
            for grp in cluster_names:
                grp_cells = groups.index[groups == grp].tolist()
                if not grp_cells:
                    continue
                vals = data_mat.loc[str(feat), grp_cells].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    global_ymin = min(global_ymin, vals.min())
                    global_ymax = max(global_ymax, vals.max())

    fig, axes = plt.subplots(n_feats, 1, figsize=fig_size, sharex=False)
    if n_feats == 1:
        axes = [axes]

    for fi, feat in enumerate(features):
        ax = axes[fi]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)

        violin_data = []
        positions = []
        vcolors = []
        constant_data = []
        has_data = False
        all_finite = []
        for gi, grp in enumerate(cluster_names):
            if split_levels is None:
                subsets = [(groups == grp, gi + 1, group_colors[grp])]
            else:
                width = 0.8 / len(split_levels)
                subsets = [
                    (
                        (groups == grp) & (split_groups == level),
                        gi + 1 + (si - (len(split_levels) - 1) / 2) * width,
                        split_colors[si],
                    )
                    for si, level in enumerate(split_levels)
                ]

            for cell_mask, position, color in subsets:
                grp_cells = groups.index[cell_mask.fillna(False)].tolist()
                if not grp_cells:
                    continue
                vals = data_mat.loc[str(feat), grp_cells].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                has_data = True
                all_finite.append(vals)
                if vals.size < 2 or np.ptp(vals) <= np.finfo(float).eps:
                    constant_data.append((position, float(vals[0]), color, vals))
                else:
                    violin_data.append(vals)
                    positions.append(position)
                    vcolors.append(color)

        if not has_data:
            ax.text(0.5, 0.5, f'{feat}: no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_ylabel(feat, fontsize=10)
            continue

        if violin_data:
            violin_width = 0.7 if split_levels is None else 0.7 / len(split_levels)
            vp = ax.violinplot(
                violin_data,
                positions=positions,
                showmeans=False,
                showmedians=True,
                showextrema=True,
                widths=violin_width,
            )

            for body, c in zip(vp['bodies'], vcolors):
                body.set_facecolor(c)
                body.set_alpha(0.7)
                body.set_edgecolor('black')
                body.set_linewidth(0.5)

            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(1.0)

        marker_half_width = 0.3 if split_levels is None else 0.3 / len(split_levels)
        for position, value, color, _ in constant_data:
            ax.hlines(
                value, position - marker_half_width, position + marker_half_width,
                color=color, linewidth=4.0, alpha=0.8, zorder=4,
            )
            ax.scatter(
                [position], [value], s=12, c=[color], edgecolors='black',
                linewidths=0.4, zorder=5,
            )

        # Add jittered points
        if pt_size > 0:
            point_sets = [
                (position, values) for position, values in zip(positions, violin_data)
            ] + [
                (position, values) for position, _, _, values in constant_data
            ]
            for position, values in point_sets:
                visible = values[values != 0]
                if visible.size == 0:
                    continue
                if visible.size > 500:
                    sample_index = np.linspace(0, visible.size - 1, 500, dtype=int)
                    visible = visible[sample_index]
                jitter = np.linspace(-0.04, 0.04, visible.size) if visible.size > 1 else 0.0
                ax.scatter(position + jitter, visible,
                           s=max(pt_size * 100, 4.0), c='black', alpha=0.35,
                           edgecolors='none', zorder=5)

        ax.set_xticks([i + 1 for i in range(n_groups)])
        ax.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(str(feat), fontsize=9, fontweight='bold', loc='left')

        feature_values = np.concatenate(all_finite)
        feature_ymin = float(feature_values.min())
        feature_ymax = float(feature_values.max())
        span = feature_ymax - feature_ymin
        padding = span * 0.05 if span > 0 else max(abs(feature_ymin), 1.0) * 0.05

        if same_y_lims and np.isfinite(global_ymin) and np.isfinite(global_ymax):
            span = global_ymax - global_ymin
            padding = span * 0.05 if span > 0 else max(abs(global_ymin), 1.0) * 0.05
            ax.set_ylim(global_ymin - padding, global_ymax + padding)
        else:
            ax.set_ylim(feature_ymin - padding, feature_ymax + padding)

        if split_levels is not None and fi == 0:
            handles = [
                mpatches.Patch(color=split_colors[index], label=str(level))
                for index, level in enumerate(split_levels)
            ]
            ax.legend(
                handles=handles, title=split_by, frameon=False,
                bbox_to_anchor=(1.01, 1.0), loc='upper left', fontsize=8,
            )

        ax.tick_params(labelsize=8)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# 31.  plot_bar  - bar plot of average gene expression per cell group
# ---------------------------------------------------------------------------

def plot_bar(
    cellchat: 'CellChat',
    features: List[str],
    group_by: Optional[str] = None,
    slot_data: str = "x",
    color_use: Optional[List[str]] = None,
    ncol: int = 3,
    fig_size: Optional[Tuple[int, int]] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Bar plot of average gene expression per cell group.

    One subplot per feature.  Bars show mean expression per group with
    standard error whiskers.

    Mirrors R ``plot_bar()`` (ggplot2 + aggregate).
    """
    features = _validate_expression_plot_inputs(features, ncol=ncol)
    data_mat = _expression_feature_dataframe(cellchat, slot_data, features)
    missing = [feat for feat in features if str(feat) not in data_mat.index]
    if missing:
        raise ValueError(f"Features not found in slot '{slot_data}': {missing}")
    groups, cluster_names = _resolve_expression_groups(
        cellchat, list(data_mat.columns), group_by
    )

    n_groups = len(cluster_names)
    n_feats = len(features)
    nrow = int(np.ceil(n_feats / ncol))
    ncol_actual = min(ncol, n_feats)

    colors = _resolve_colors(n_groups, color_use)
    group_colors = dict(zip(cluster_names, colors))

    if fig_size is None:
        fig_size = (ncol_actual * 3.5, nrow * 3.0)

    fig, axes = plt.subplots(nrow, ncol_actual, figsize=fig_size)
    if nrow == 1 and ncol_actual == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes.reshape(1, -1)
    elif ncol_actual == 1:
        axes = axes.reshape(-1, 1)

    for fi, feat in enumerate(features):
        r, c = divmod(fi, ncol_actual)
        ax = axes[r, c]
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)

        means = np.zeros(n_groups)
        stds = np.zeros(n_groups)
        has_data = np.zeros(n_groups, dtype=bool)

        for gi, grp in enumerate(cluster_names):
            grp_cells = groups.index[groups == grp].tolist()
            if not grp_cells:
                continue
            vals = data_mat.loc[str(feat), grp_cells].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                means[gi] = np.mean(vals)
                stds[gi] = np.std(vals) / np.sqrt(len(vals))
                has_data[gi] = True

        x_pos = np.arange(n_groups)
        bar_colors = [group_colors.get(grp, '#999999') for grp in cluster_names]

        bars = ax.bar(x_pos, means, yerr=stds, color=bar_colors,
                      edgecolor='black', linewidth=0.5, capsize=3,
                      error_kw={'linewidth': 0.8})

        ax.set_xticks(x_pos)
        ax.set_xticklabels(cluster_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean expression', fontsize=9)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(labelsize=8)

        # Theme
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused
    for fi in range(n_feats, nrow * ncol_actual):
        r, c = divmod(fi, ncol_actual)
        if r < axes.shape[0] and c < axes.shape[1]:
            axes[r, c].set_visible(False)

    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# Spot-level spatial communication, statistics, and topic plots
# ---------------------------------------------------------------------------
def _spot_coordinates(cellchat, coordinates=None) -> np.ndarray:
    values = cellchat.obsm.get("spatial") if coordinates is None else coordinates
    values = np.asarray(values, dtype=float)
    if values.shape != (cellchat.n_obs, 2) or not np.isfinite(values).all():
        raise ValueError("coordinates must be a finite n_spots x 2 matrix.")
    return values


def _spatial_finish(fig, return_fig):
    fig.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


def _spatial_limits(values, quantile):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Spatial values contain no finite observations.")
    if quantile is None:
        lower, upper = float(finite.min()), float(finite.max())
    else:
        if not 0.5 < quantile <= 1:
            raise ValueError("quantile must be in (0.5, 1].")
        positive = finite[finite > 0]
        lower, upper = min(0.0, float(finite.min())), float(np.quantile(positive if len(positive) else finite, quantile))
    return lower, upper if upper > lower else lower + 1.0


def plot_spatial_values(cellchat, values, *, coordinates=None, titles=None, color_map="viridis", quantile=0.99, point_size=12.0, alpha=0.9, ncol=2, reverse_y=True, background_image=None, fig_size=None, return_fig=False):
    """Plot one or more aligned numeric spot features."""
    if isinstance(values, pd.Series):
        frame = values.to_frame(values.name or "value")
    elif isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError("values must be a spot vector or spot-by-feature matrix.")
        frame = pd.DataFrame(array)
    if frame.shape[0] != cellchat.n_obs:
        raise ValueError("values must contain one row per spot.")
    numeric = frame.to_numpy(dtype=float)
    titles = [str(value) for value in frame.columns] if titles is None else [str(value) for value in titles]
    if len(titles) != frame.shape[1]:
        raise ValueError("titles must contain one name per feature.")
    if not isinstance(ncol, (int, np.integer)) or ncol < 1:
        raise ValueError("ncol must be a positive integer.")
    xy, ncol_actual = _spot_coordinates(cellchat, coordinates), min(ncol, frame.shape[1])
    nrow = int(np.ceil(frame.shape[1] / ncol_actual))
    fig, axes = plt.subplots(nrow, ncol_actual, figsize=fig_size or (4.2 * ncol_actual, 4.0 * nrow), squeeze=False)
    cmap = plt.get_cmap(color_map)
    for index, title in enumerate(titles):
        ax, feature = axes.flat[index], numeric[:, index]
        if background_image is not None:
            ax.imshow(np.asarray(background_image), origin="upper", alpha=0.78)
        lower, upper = _spatial_limits(feature, quantile)
        order = np.argsort(np.nan_to_num(feature, nan=-np.inf))
        points = ax.scatter(xy[order, 0], xy[order, 1], c=feature[order], cmap=cmap, norm=mcolors.Normalize(lower, upper, clip=True), s=point_size, alpha=alpha, linewidths=0)
        fig.colorbar(points, ax=ax, shrink=0.72, pad=0.02)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("Spatial 1"); ax.set_ylabel("Spatial 2")
        if reverse_y and background_image is None:
            ax.invert_yaxis()
    for ax in axes.flat[frame.shape[1]:]:
        ax.set_visible(False)
    return _spatial_finish(fig, return_fig)


def plot_spatial_proportions(cellchat, proportions: pd.DataFrame, *, coordinates=None, min_fraction=0.1, radius=None, color_use=None, reverse_y=True, background_image=None, title="Spot cell-type proportions", fig_size=(7, 7), return_fig=False):
    """Draw a proportional pie at every spot."""
    if not isinstance(proportions, pd.DataFrame):
        raise TypeError("proportions must be a pandas DataFrame.")
    table = proportions.copy(); table.index = table.index.astype(str)
    if table.index.tolist() != cellchat.obs_names.astype(str).tolist():
        raise ValueError("proportions rows must match cellchat.obs_names in the same order.")
    values = table.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("proportions must be finite and non-negative.")
    if not np.isfinite(min_fraction) or not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be between 0 and 1.")
    values[values < min_fraction] = 0.0
    row_sums = values.sum(axis=1, keepdims=True)
    values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    xy = _spot_coordinates(cellchat, coordinates)
    if radius is None:
        from scipy.spatial import cKDTree
        nearest = cKDTree(xy).query(xy, k=2)[0][:, 1]
        radius = 0.34 * float(np.median(nearest[np.isfinite(nearest)]))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite.")
    names = table.columns.astype(str).tolist()
    if isinstance(color_use, Mapping):
        missing = set(names).difference(color_use)
        if missing:
            raise ValueError(f"color_use is missing cell types: {sorted(missing)}")
        colors = [color_use[name] for name in names]
    elif color_use is None:
        colors = sns.color_palette("tab10", len(names)).as_hex()
    else:
        colors = list(color_use)
        if len(colors) < len(names):
            raise ValueError("color_use must provide one color per cell type.")
    fig, ax = plt.subplots(figsize=fig_size)
    if background_image is not None:
        ax.imshow(np.asarray(background_image), origin="upper", alpha=0.78)
    for (x_value, y_value), fractions in zip(xy, values, strict=True):
        angle = 0.0
        for fraction, color in zip(fractions, colors, strict=True):
            if fraction <= 0:
                continue
            next_angle = angle + 360.0 * float(fraction)
            ax.add_patch(mpatches.Wedge((x_value, y_value), radius, angle, next_angle, facecolor=color, edgecolor="none"))
            angle = next_angle
    ax.legend(handles=[mpatches.Patch(facecolor=color, label=name) for name, color in zip(names, colors)], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_title(title); ax.set_aspect("equal"); ax.autoscale_view()
    if reverse_y and background_image is None:
        ax.invert_yaxis()
    return _spatial_finish(fig, return_fig)


def plot_spatial_categories(cellchat, categories, *, coordinates=None, color_use=None, point_size=14.0, alpha=0.9, reverse_y=True, background_image=None, title="Spatial categories", fig_size=(5.5, 4.8), return_fig=False):
    """Plot one categorical value per spot with a stable legend."""
    if isinstance(categories, str):
        if categories not in cellchat.obs:
            raise ValueError(f"categories={categories!r} is not present in cellchat.obs.")
        values = cellchat.obs[categories].copy()
    elif isinstance(categories, pd.Series):
        values = categories.copy(); values.index = values.index.astype(str)
        if not values.index.equals(pd.Index(cellchat.obs_names.astype(str))):
            raise ValueError("categories index must match cellchat.obs_names in the same order.")
    else:
        array = np.asarray(categories)
        if array.ndim != 1 or len(array) != cellchat.n_obs:
            raise ValueError("categories must contain one value per spot.")
        values = pd.Series(array, index=cellchat.obs_names)
    if values.isna().any():
        raise ValueError("categories cannot contain missing values.")
    levels = [str(value) for value in values.cat.categories] if isinstance(values.dtype, pd.CategoricalDtype) else list(pd.unique(values.astype(str)))
    labels = values.astype(str).to_numpy()
    if isinstance(color_use, Mapping):
        missing = set(levels).difference(color_use)
        if missing:
            raise ValueError(f"color_use is missing categories: {sorted(missing)}")
        palette = {level: color_use[level] for level in levels}
    elif color_use is None:
        palette = dict(zip(levels, sns.color_palette("tab10", len(levels)).as_hex()))
    else:
        colors = list(color_use)
        if len(colors) < len(levels):
            raise ValueError("color_use must provide one color per category.")
        palette = dict(zip(levels, colors))
    xy = _spot_coordinates(cellchat, coordinates); fig, ax = plt.subplots(figsize=fig_size)
    if background_image is not None:
        ax.imshow(np.asarray(background_image), origin="upper", alpha=0.78)
    for level in levels:
        selected = labels == level
        ax.scatter(xy[selected, 0], xy[selected, 1], s=point_size, color=palette[level], alpha=alpha, linewidths=0, label=level)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False); ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("Spatial 1"); ax.set_ylabel("Spatial 2")
    if reverse_y and background_image is None:
        ax.invert_yaxis()
    return _spatial_finish(fig, return_fig)


def plot_spot_signaling_scores(cellchat, signaling=None, *, measures=("outdeg", "indeg"), slot_name="spot_pathway_network", binary=False, color_maps=("Blues", "Reds"), quantile=0.99, point_size=12.0, return_fig=False):
    """Plot outgoing and/or incoming communication scores for spots."""
    measures = [str(value) for value in measures]
    if not measures or len(color_maps) < len(measures):
        raise ValueError("measures must be non-empty and color_maps must provide one map per measure.")
    from .analysis import get_spot_signaling_scores
    fig, axes = plt.subplots(1, len(measures), figsize=(4.3 * len(measures), 4), squeeze=False)
    xy = _spot_coordinates(cellchat)
    selected = "all" if signaling is None else (signaling if isinstance(signaling, str) else "+".join(map(str, signaling)))
    for index, measure in enumerate(measures):
        score = get_spot_signaling_scores(cellchat, signaling, measure, slot_name, binary).to_numpy()
        lower, upper = _spatial_limits(score, quantile); order = np.argsort(score)
        points = axes[0, index].scatter(xy[order, 0], xy[order, 1], c=score[order], cmap=color_maps[index], norm=mcolors.Normalize(lower, upper, clip=True), s=point_size, linewidths=0)
        fig.colorbar(points, ax=axes[0, index], shrink=0.72, pad=0.02); axes[0, index].set_title(f"{selected}: {measure}"); axes[0, index].set_aspect("equal"); axes[0, index].invert_yaxis()
    return _spatial_finish(fig, return_fig)


def plot_spatial_gi(cellchat, result_name, *, pvalue=0.05, point_size=14.0, return_fig=False):
    """Plot a stored Getis-Ord Gi/Gi* result."""
    table = cellchat.spatial_statistics.get("gi", {}).get(result_name)
    if not isinstance(table, pd.DataFrame) or not {"gi", "pvalue"}.issubset(table):
        raise ValueError(f"Unknown or invalid Gi result: {result_name!r}.")
    xy, gi = _spot_coordinates(cellchat), table["gi"].to_numpy(dtype=float); bound = max(float(np.nanmax(np.abs(gi))), np.finfo(float).eps)
    significant = np.ones(len(gi), dtype=bool) if pvalue is None else table["pvalue"].to_numpy() < pvalue
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); ax.scatter(xy[~significant, 0], xy[~significant, 1], c="#d9d9d9", s=point_size, linewidths=0, alpha=0.6)
    points = ax.scatter(xy[significant, 0], xy[significant, 1], c=gi[significant], cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), s=point_size, linewidths=0)
    fig.colorbar(points, ax=ax, shrink=0.72, pad=0.02, label="Getis-Ord Gi"); ax.set_title(result_name); ax.set_aspect("equal"); ax.invert_yaxis()
    return _spatial_finish(fig, return_fig)


def plot_spatial_lee(cellchat, result_name, *, cutoff=None, color_map="Reds", annotate=True, fig_size=None, return_fig=False):
    """Plot a stored Lee spatial co-occurrence matrix."""
    table = cellchat.spatial_statistics.get("lee", {}).get(result_name)
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"Unknown Lee result: {result_name!r}.")
    shown = table.where(table > cutoff) if cutoff is not None else table.copy()
    if shown.empty or not np.isfinite(shown.to_numpy(dtype=float)).any():
        raise ValueError(f"Lee result {result_name!r} has no finite values.")
    fig_size = fig_size or (max(4.0, 0.5 * shown.shape[1] + 2), max(3.0, 0.42 * shown.shape[0] + 1.5)); fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(shown, cmap=color_map, mask=shown.isna(), annot=annotate, fmt=".2f", linewidths=0.35, linecolor="white", ax=ax, cbar_kws={"label": "Lee statistic"})
    ax.set_title(result_name); ax.set_xlabel("Feature"); ax.set_ylabel("Cell group / feature")
    return _spatial_finish(fig, return_fig)


def _topic_result(cellchat, slot_name, pattern):
    try:
        return cellchat.cell_topics[slot_name][pattern]
    except KeyError as error:
        raise ValueError(f"Run identify_cell_topics for {slot_name!r}, pattern={pattern!r} first.") from error


def plot_spatial_topics(cellchat, *, slot_name="spot_network", pattern="incoming", color_map="Reds", quantile=0.99, point_size=12.0, ncol=3, return_fig=False):
    """Plot the spatial distribution of fitted communication topics."""
    result = _topic_result(cellchat, slot_name, pattern)
    return plot_spatial_values(cellchat, result["cell"], color_map=color_map, quantile=quantile, point_size=point_size, ncol=ncol, return_fig=return_fig)


def plot_topic_composition(cellchat, *, slot_name="spot_network", pattern="incoming", group_by="cellchat_group", normalize=True, fig_size=None, return_fig=False):
    """Show cell-group composition of every communication topic."""
    result = _topic_result(cellchat, slot_name, pattern)
    if group_by not in cellchat.obs:
        raise ValueError(f"group_by={group_by!r} is not present in cellchat.obs.")
    topic = result["assignment"].reindex(cellchat.obs_names.astype(str)); groups = cellchat.obs[group_by].astype(str).to_numpy(); table = pd.crosstab(topic.to_numpy(), groups)
    if normalize:
        table = table.div(table.sum(axis=1), axis=0).fillna(0)
    fig_size = fig_size or (max(5, 0.7 * len(table.columns) + 2), max(3, 0.45 * len(table) + 1.5)); fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(table, cmap="Blues", annot=True, fmt=".2f" if normalize else "g", ax=ax); ax.set_xlabel(group_by); ax.set_ylabel("Topic"); ax.set_title(f"{pattern.capitalize()} topic composition")
    return _spatial_finish(fig, return_fig)


def plot_topic_signaling(cellchat, *, slot_name="spot_network", pattern="incoming", top_n=15, fig_size=None, return_fig=False):
    """Plot the strongest L-R pairs or pathways loading on each topic."""
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    loadings = _topic_result(cellchat, slot_name, pattern)["signaling"]; selected = set()
    for column in loadings:
        selected.update(loadings[column].nlargest(top_n).index)
    shown = loadings.loc[[name for name in loadings.index if name in selected]]
    fig_size = fig_size or (max(5, 0.75 * shown.shape[1] + 2), max(4, 0.25 * shown.shape[0] + 2)); fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(shown, cmap="Blues", ax=ax, cbar_kws={"label": "NMF loading"}); ax.set_xlabel("Topic"); ax.set_ylabel("Signaling"); ax.set_title(f"{pattern.capitalize()} topic signaling")
    return _spatial_finish(fig, return_fig)


