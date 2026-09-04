# CellChatPy

CellChatPy is a Python toolkit for inferring, analyzing, and visualizing
cell-cell communication from single-cell and spatial transcriptomics data.
The workflow is inspired by the CellChat R package and uses Python-native data
structures and plotting tools.

## What It Provides

- CellChat objects from matrices, metadata, or AnnData
- Human, mouse, and zebrafish ligand-receptor databases
- Overexpressed gene and ligand-receptor analysis
- Communication probability and pathway-level network inference
- Network comparison, centrality, and communication pattern analysis
- Circle, heatmap, bubble, chord, hierarchy, and spatial visualization
- Spatial transcriptomics and spatial multiomics workflows

## Install

Install the released package from PyPI:

```bash
python -m pip install CellChatPy
```

Install the development version from this repository:

```bash
python -m pip install "git+https://github.com/wyuanhang03-web/CellChatPy.git"
```

CellChatPy requires Python 3.10 or newer.

## Quick Example

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

## Tutorials

The `tutorial/` directory contains seven notebooks covering:

- Human skin single-dataset analysis
- Human skin NL/LS comparison
- Spatial transcriptomics
- Spatial multiomics
- Multiple spatial transcriptomics datasets
- Different cellular compositions
- Advanced spatial communication and motif analysis

The large input datasets are not stored in this repository. See
[`data/README.md`](data/README.md) for the required filenames and data policy.
Saved notebook outputs, figures, and tables are included so the examples can
be inspected without downloading the original datasets.

## Documentation

The `docs/` directory contains the Sphinx source for the tutorial website.
It contains the project documentation and tutorial pages.

Full documentation, tutorials, and API reference:
https://cellchatpy.readthedocs.io/en/latest/

The website can be built with Sphinx and nbsphinx. Notebook execution is
disabled during documentation builds, so saved outputs are used.

## Repository Layout

```text
CellChatPy/
|-- CellChatPy/       Python package and CellChatDB
|-- data/              Data instructions; large inputs stay outside GitHub
|-- tutorial/          Seven notebooks and saved outputs
|-- docs/              Documentation website source
|-- pyproject.toml     Package metadata and dependencies
|-- MANIFEST.in        Source distribution file rules
|-- LICENSE             GPL-3.0 license
`-- README.md
```

## Relationship to CellChat

CellChatPy is an Python implementation and public resources of
[CellChat](https://github.com/jinworks/CellChat). It is not the official
CellChat R package. Please cite the original CellChat publications when its
methods or database resources contribute to your work.

## License

This repository is distributed under the GNU General Public License v3.0.
See [`LICENSE`](LICENSE).

## Links

- [PyPI package](https://pypi.org/project/CellChatPy/)
- [Documentation source](docs/index.md)
- [Issue tracker](https://github.com/wyuanhang03-web/CellChatPy/issues)
