"""Spatial analysis helpers for CellChatPy.

The core inference lives in :mod:`CellChatPy.modeling`; this module provides
the small spatial utilities that were absent from the original translation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def compute_colocalization(coordinates, group, idents_use=None, symmetric=True,
                           n_boot=100, seed_use=1):
    """Permutation p-values for spatial co-localization of cell groups.

    The observed statistic is the median nearest-group mean distance,
    compared with label permutations.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite n x 2 matrix")
    labels = pd.Series(group)
    if not pd.api.types.is_categorical_dtype(labels):
        labels = labels.astype("category")
    levels = list(labels.cat.categories)
    if idents_use is not None:
        levels = [x for x in levels if x in set(idents_use)]
    if len(levels) == 0 or n_boot < 1:
        raise ValueError("group and n_boot must define at least one group and one permutation")
    rng = np.random.default_rng(seed_use)
    out = np.full((len(levels), len(levels)), np.nan)
    for i, left in enumerate(levels):
        idx_left = np.flatnonzero(labels.to_numpy() == left)
        if not len(idx_left):
            continue
        for j, right in enumerate(levels):
            idx_right = np.flatnonzero(labels.to_numpy() == right)
            if not len(idx_right):
                continue
            observed = np.median(cdist(coords[idx_left], coords[idx_right]).mean(axis=1))
            pool = np.flatnonzero(~labels.isin([left, right]).to_numpy())
            if len(pool) < len(idx_left):
                out[i, j] = np.nan
                continue
            null = np.empty(n_boot)
            target = coords[idx_right]
            for b in range(n_boot):
                sampled = rng.choice(pool, size=len(idx_left), replace=False)
                null[b] = np.median(cdist(coords[sampled], target).mean(axis=1))
            out[i, j] = np.mean(null <= observed)
    if symmetric:
        out = np.minimum(out, out.T)
    return pd.DataFrame(out, index=levels, columns=levels)


def spatial_visual_scoring(cellchat, signaling, measure=("outdeg", "indeg"),
                           measure_name=None, slot_name="spot_pathway_network",
                           do_group=False, merge=False, **kwargs):
    """Plot stored spot/group scores in the style of ``spatialVisual_scoring``.

    For individual-spot scores this delegates to ``plot_spatial_values``;
    group scores are expanded to spots using ``cellchat.groups``.
    """
    from .analysis import get_spot_signaling_scores
    from .visualization import plot_spatial_values
    names = list(measure_name or measure)
    values = []
    for m in measure:
        score = get_spot_signaling_scores(cellchat, signaling, measure=m,
                                          slot_name=slot_name, binary=False)
        values.append(score.rename(names[len(values)] if len(names) > len(values) else m))
    frame = pd.concat(values, axis=1)
    return plot_spatial_values(cellchat, frame, titles=list(frame.columns),
                               return_fig=kwargs.pop("return_fig", False), **kwargs)


def communication_distance_plot(cellchat, enriched_only=True, signaling_type="All",
                                return_fig=False, **kwargs):
    """Visualize diffusible/contact distance distributions from a spatial network."""
    import matplotlib.pyplot as plt
    network = getattr(cellchat, "spot_network", {})
    distance = network.get("distance")
    if distance is None:
        raise ValueError("Run spatial communication inference first.")
    vals = distance.data
    fig, ax = plt.subplots(figsize=kwargs.pop("fig_size", (6, 4)))
    if len(vals):
        ax.hist(vals, bins=kwargs.pop("bins", 30), color="#69b3a2", alpha=0.7)
    ax.set_xlabel("Distance between cell pairs (um)"); ax.set_ylabel("Count")
    ax.set_title("Spatial communication distance")
    return fig if return_fig else ax


def compute_grid_size(coordinates, grid_resolution=2.0, cellsize=None):
    """Return a suggested square grid size from the coordinate spacing."""
    if hasattr(coordinates, "obsm"):
        coordinates = coordinates.obsm.get("spatial")
    xy = np.asarray(coordinates, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coordinates must be n x 2")
    if cellsize is None:
        from scipy.spatial.distance import pdist
        d = pdist(xy)
        base = float(np.nanmedian(d[d > 0])) if np.any(d > 0) else 1.0
    else:
        base = float(np.asarray(cellsize).ravel()[0])
    return {"cellsize": base * float(grid_resolution), "base_cellsize": base,
            "grid_resolution": float(grid_resolution)}


def make_grid_spatial_cellchat(cellchat, cellsize=5.0, data_layer="X"):
    """Aggregate spots into square spatial grids."""
    from .cellchat_class import create_cellchat
    coords = np.asarray(cellchat.obsm.get("spatial"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or cellsize <= 0:
        raise ValueError("cellchat must contain n x 2 spatial coordinates and cellsize must be positive")
    origin = coords.min(axis=0)
    keys = np.floor((coords - origin) / float(cellsize)).astype(int)
    groups = pd.Series([f"Grid{x}_{y}" for x, y in keys], index=cellchat.obs_names)
    unique = list(pd.unique(groups))
    layer = getattr(cellchat, data_layer, None) if data_layer != "X" else cellchat.X
    values = layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)
    expr = np.vstack([values[groups.to_numpy() == key].mean(axis=0) for key in unique])
    meta = pd.DataFrame(index=unique)
    source_meta = cellchat.obs
    if "cellchat_group" in source_meta:
        meta["cellchat_group"] = [source_meta.loc[groups == key, "cellchat_group"].mode().iloc[0] for key in unique]
    meta["spots_count"] = [int((groups == key).sum()) for key in unique]
    coords_new = np.vstack([coords[groups.to_numpy() == key].mean(axis=0) for key in unique])
    result = create_cellchat(pd.DataFrame(expr, index=unique, columns=cellchat.var_names),
                             metadata=meta, group_by="cellchat_group" if "cellchat_group" in meta else None,
                             datatype="spatial", coordinates=coords_new,
                             spatial_factors=cellchat.spatial.get("spatial_factors", {"ratio": 1.0, "tol": cellsize / 2}))
    result.database = cellchat.database
    result.lr_pairs = cellchat.lr_pairs
    return result
