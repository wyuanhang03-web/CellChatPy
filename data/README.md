# Tutorial Data

The large input datasets used by the tutorial notebooks are not included in
this GitHub repository. This keeps the source repository lightweight and
avoids redistributing datasets without their original licenses.

The notebooks can still be read online because their saved figures, tables,
and other outputs are included in the repository. To run a notebook from the
beginning, obtain the corresponding datasets from their original data provider
or from the project maintainer, then place the files in this directory.

## Expected Files

| File | Used by |
| --- | --- |
| `data_humanSkin.h5` | Human skin analysis and NL/LS comparison |
| `mouse_cortex_visium.h5` | Spatial transcriptomics analysis |
| `mouse_spleen_spatial_multiomics.h5` | Spatial multiomics analysis |
| `human_intestine_A1.h5` | Multiple spatial transcriptomics datasets |
| `human_intestine_A2.h5` | Multiple spatial transcriptomics datasets |
| `mouse_skin_e13.h5ad` | Different cellular composition analysis |
| `mouse_skin_e14.h5ad` | Different cellular composition analysis |
| `human_psoriasis.h5` | Advanced SpatialCellChat analysis |

The expected layout is:

```text
CellChatPy/
|-- CellChatPy/
|-- data/
|   |-- data_humanSkin.h5
|   |-- mouse_cortex_visium.h5
|   `-- ...
`-- tutorial/
```

Run Jupyter from the repository root:

```bash
python -m pip install -e ".[tutorial]"
jupyter lab
```

Then open a notebook under `tutorial/`. The notebooks locate the package,
input data, and output directories relative to the repository root.

## Data Availability

The eight input files referenced by the current published notebooks
are approximately 1.071 GiB in total (1,150,368,414 bytes). This is a minimum
for the current tutorial set, not a complete inventory of every data file used
by all historical CellChatPy workflows. Older notebooks, converted formats,
and optional R/Python comparison workflows may require additional files.

The files are intentionally kept outside GitHub. This repository currently does
not redistribute them. Please check the original publication, data provider,
or project release notes for the appropriate download location and usage terms.

## Troubleshooting

If a notebook reports that an input file is missing, check the filename,
extension, and location. From the repository root, the following command lists
the files currently present:

```bash
python -c "from pathlib import Path; print(*sorted(Path('data').glob('*')), sep='\n')"
```

The repository ignores local `.h5` and `.h5ad` files, so they will not be
included by `git add .`.
