# Tutorial data

The large tutorial inputs are intentionally not included in the GitHub
repository. This directory contains only this README. Before running a
notebook locally, place the required files here using the filenames below.
The repository `.gitignore` prevents these files from being committed.

Expected filenames:

| File | Used by |
| --- | --- |
| `data_humanSkin.h5` | Human skin analysis and NL/LS comparison |
| `mouse_cortex_visium.h5` | Spatial transcriptomics analysis |
| `mouse_spleen_spatial_multiomics.h5` | Spatial multiomics analysis |
| `human_intestine_A1.h5` | Multiple spatial datasets analysis |
| `human_intestine_A2.h5` | Multiple spatial datasets analysis |
| `mouse_skin_e13.h5ad` | Different cellular compositions analysis |
| `mouse_skin_e14.h5ad` | Different cellular compositions analysis |

For the current local workspace, the files can be copied from the original
data directory with PowerShell:

```powershell
Copy-Item "E:\Jin\cellchat.python_new\data\*.h5" -Destination . -Force
Copy-Item "E:\Jin\cellchat.python_new\data\*.h5ad" -Destination . -Force
```

The seven datasets total approximately 1.04 GiB and are not needed to build the
Read the Docs website because the published notebooks already contain their
saved outputs. For a long-term scientific release, archive the datasets with
checksums, licenses, and a DOI on Zenodo, Figshare, or OSF, then add the DOI to
this file.

The human skin notebook can optionally use the tracked `r_permutations.csv`
file from the repository root for an R/Python comparison.
