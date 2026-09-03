# CellChatPy

CellChatPy is a Python toolkit for inferring, analyzing, and visualizing
cell-cell communication from single-cell and spatially resolved transcriptomics
data. Its workflow and terminology are based on the CellChat R package, while
using Python-native data structures and plotting tools.

## Features

- Create CellChat objects from matrices, metadata, or AnnData
- Load ligand-receptor databases for human, mouse, and zebrafish
- Identify overexpressed genes and ligand-receptor interactions
- Infer communication probabilities and pathway-level networks
- Compare communication networks across datasets
- Analyze centrality, communication patterns, and network similarity
- Visualize circle, heatmap, bubble, chord, hierarchy, and spatial networks
- Read and write structured HDF5 inputs for spatial and multiomics workflows

## Installation

CellChatPy requires Python 3.10 or newer. Install the latest release from PyPI:

```bash
pip install CellChatPy
```

To install the latest development version directly from GitHub:

```bash
pip install "git+https://github.com/wyuanhang03-web/CellChatPy.git"
```

For local development and tutorials:

```bash
git clone https://github.com/wyuanhang03-web/CellChatPy.git
cd CellChatPy
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tutorial]"
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tutorial]"
```

## Quick start

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

Seven executable notebooks and their saved outputs are available in
[`tutorial/`](tutorial/README.md):

- Human skin single-dataset analysis
- Human skin NL/LS comparison
- Spatial transcriptomics analysis
- Spatial multiomics analysis
- Multiple spatial transcriptomics datasets
- Comparison with different cellular compositions
- Advanced SpatialCellChat analysis with hotspot, co-occurrence, and motif maps

The large tutorial input datasets are intentionally not stored in this GitHub
repository. See [`data/README.md`](data/README.md) for the expected filenames
and copy the inputs into `data/` locally before running a notebook. The saved
notebook outputs and documentation pages do not require these inputs to render.

The tutorial website is built with Sphinx, MyST, and nbsphinx. Read the Docs
renders the Markdown, code, tables, section anchors, and saved notebook images
without rerunning the analyses:

```bash
pandoc --version
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Pandoc must be available on `PATH` for local builds. Read the Docs installs it
automatically from `.readthedocs.yaml`.

See [`GITHUB_UPLOAD_GUIDE_CN.md`](GITHUB_UPLOAD_GUIDE_CN.md) for the complete
GitHub, Read the Docs, and PyPI publishing procedure.

The motif-style spatial visualization is documented at
[`Spatial Visualization of Motif`](docs/tutorials/spatial_motif_enrichment.md).

## Repository structure

```text
CellChatPy/
|-- CellChatPy/          # Python package and bundled CellChatDB files
|-- tutorial/            # Seven notebooks, figures, and tabular outputs
|-- data/                # README only; large tutorial inputs stay local
|-- docs/                # Sphinx/nbsphinx documentation source
|-- .readthedocs.yaml    # Read the Docs build configuration
|-- pyproject.toml       # Package metadata and dependencies
|-- MANIFEST.in          # Source distribution data rules
|-- LICENSE              # GPL-3.0 license
`-- README.md
```

## Relationship to CellChat

CellChatPy is an independent Python implementation inspired by the methods,
workflow, and public resources of
[CellChat](https://github.com/jinworks/CellChat). It is not the official CellChat
R package. Please cite the original CellChat publications when its methods or
database resources contribute to your work.

## License

This repository is distributed under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE).

## Contact

Please use [GitHub Issues](https://github.com/wyuanhang03-web/CellChatPy/issues)
for bug reports and feature requests.
