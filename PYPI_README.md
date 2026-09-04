# CellChatPy

CellChatPy is a Python toolkit for inferring and visualizing cell-cell
communication from single-cell and spatial transcriptomics data.

## Installation

```bash
python -m pip install CellChatPy
```

CellChatPy requires Python 3.10 or newer.

## Quick Start

```python
import CellChatPy as cc

cellchat = cc.create_cellchat(
    object=expression_matrix,
    metadata=cell_metadata,
    group_by="cell_type",
)

database = cc.load_database(species="human")
cellchat = cc.set_database(cellchat, database=database)
cellchat = cc.subset_signaling_data(cellchat)
cellchat = cc.identify_overexpressed_genes(cellchat)
cellchat = cc.identify_overexpressed_interactions(cellchat)
cellchat = cc.compute_communication_probability(cellchat)
cellchat = cc.filter_communication(cellchat, min_cells=10)
cellchat = cc.compute_pathway_probability(cellchat)
cellchat = cc.aggregate_network(cellchat)
```

## Main Features

- Human, mouse, and zebrafish ligand-receptor databases
- Cell-cell communication probability and pathway inference
- Network comparison and communication pattern analysis
- Circle, heatmap, bubble, chord, hierarchy, and spatial plots
- Spatial transcriptomics and spatial multiomics workflows

## Documentation and Source Code

- GitHub: https://github.com/wyuanhang03-web/CellChatPy
- Documentation source and tutorials: available in the GitHub repository
- Issue tracker: https://github.com/wyuanhang03-web/CellChatPy/issues

The large input datasets used by the tutorial notebooks are not included in
the PyPI package. See the GitHub repository for tutorial information and data
availability notes.

## Relationship to CellChat R

CellChatPy is a Python reimplementation of the CellChat R package. It
translates the main cell-cell communication analysis workflow into native
Python data structures and plotting tools. CellChatPy is an independent
Python project and is not the official CellChat R package.

See the [CellChat R package](https://github.com/jinworks/CellChat) for the
original R implementation and publications.

## License

CellChatPy is distributed under the GNU General Public License v3.0.
