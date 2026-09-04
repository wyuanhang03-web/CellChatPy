# Spatial Communication Topics

This tutorial explains how CellChatPy learns repeatable spatial communication
patterns from a spot-level ligand–receptor network. The executed psoriasis
tutorial identifies these patterns with `identify_cell_topics` and displays
their spatial weights with `plot_spatial_topics`.

The complete runnable workflow is in
[`SpatialCellChat_analysis_of_spatial_transcriptomics_data.ipynb`](SpatialCellChat_analysis_of_spatial_transcriptomics_data),
and the saved output is
`tutorial/figures_spatial_transcriptomics/15_incoming_topics_spatial.png`.

## 1. Prepare a spatial CellChatPy object

The object must already contain spot coordinates and a spot-level
communication network. The advanced tutorial creates those objects in its
first two parts.

```python
import CellChatPy as cc

# cellchat must be a spatial object with coordinates and spot_network.
assert "spatial" in str(cellchat.datatype).lower()
assert "spatial" in cellchat.obsm
```

## 2. Learn communication motifs

`n_topics` is the number of motifs. Twelve reproduces the published example;
for a new dataset, inspect several values and choose a stable solution.

```python
cellchat = cc.identify_cell_topics(
    cellchat,
    n_topics=12,
    pattern="incoming",
    slot_name="spot_network",
    seed_use=666,
)
```

## 3. Generate the spatial communication topic map

```python
fig = cc.plot_spatial_topics(
    cellchat,
    slot_name="spot_network",
    pattern="incoming",
    color_map="Reds",
    quantile=0.99,
    point_size=12,
    ncol=6,
    return_fig=True,
)
fig.savefig("incoming_topics_spatial.png", dpi=300, bbox_inches="tight")
```

Each panel is one motif. Redder spots have larger motif weights; blank or very
pale spots contribute little to that motif. The plot is a spatial distribution
map, not proof that molecules physically move between spots.

```{image} ../_static/spatial/15_incoming_topics_spatial.png
:alt: Spatial distribution of incoming communication motifs
:width: 100%
```

## 4. Inspect motif composition and signaling genes

Use these companion plots to explain what each spatial motif represents:

```python
composition = cc.plot_topic_composition(
    cellchat, slot_name="spot_network", pattern="incoming",
    group_by="cellchat_group", return_fig=True,
)
signaling = cc.plot_topic_signaling(
    cellchat, slot_name="spot_network", pattern="incoming",
    top_n=8, return_fig=True,
)
```

The first compares motif weights across annotated cell groups. The second lists
the strongest ligand–receptor signals per motif. Always read the spatial map
together with these two summaries.

## 5. Interpret the result

Use the spatial topic maps together with the composition and signaling summaries
to describe which cell groups and ligand–receptor signals contribute to each
communication pattern.
