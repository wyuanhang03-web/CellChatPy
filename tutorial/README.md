# Tutorials

This directory contains the seven CellChatPy notebooks prepared for publication.
Run Jupyter from the repository root so that package and data paths resolve
consistently:

```bash
jupyter lab
```

| Notebook | Topic |
| --- | --- |
| `CellChat_humanSkin_LS.ipynb` | Single-dataset human skin workflow |
| `CellChat_comparison_analysis_NL_LS.ipynb` | NL versus LS comparison |
| `CellChat_analysis_of_spatial_transcriptomics_data.ipynb` | Single spatial transcriptomics dataset |
| `CellChat_analysis_of_spatial_multiomics_data.ipynb` | Spatial RNA and protein multiomics |
| `SpatialCellChat_analysis_of_spatial_transcriptomics_data.ipynb` | Advanced spatial communication, hotspots, co-occurrence, and motifs |
| `CellChat_analysis_of_multiple_spatial_transcriptomics_datasets.ipynb` | Multiple spatial transcriptomics datasets |
| `Comparison_analysis_of_multiple_datasets_with_different_cellular_compositions.ipynb` | Datasets with different cell compositions |

See [`../data/README.md`](../data/README.md) for the expected input filenames.
Notebook outputs, the selected `figures_*` directories, and interaction
CSV files are retained so the published examples are inspectable on GitHub.
The Read the Docs source copies of all seven notebooks live in
[`../docs/tutorials/`](../docs/tutorials/index.md).

After editing or rerunning a notebook, synchronize the documentation copy from
the repository root:

```powershell
Copy-Item tutorial\*.ipynb docs\tutorials\ -Force
```

Serialized `.pkl` analysis objects remain excluded. They are optional caches,
not inputs to the website, and should be archived separately when needed.
