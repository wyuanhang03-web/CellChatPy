"""RNA/ADT preprocessing and database-remapping tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CellChatPy.utilities import preprocess_multiomics


def test_preprocess_multiomics_preserves_labels_and_handles_constant_adt() -> None:
    rna = pd.DataFrame(
        [[2.0, 4.0], [1.0, 3.0]],
        index=["spot_1", "spot_2"],
        columns=["Ligand", "Receptor"],
    )
    adt = pd.DataFrame(
        [[0.0, 1.0], [0.0, 5.0]],
        index=rna.index,
        columns=["CD3E", "CD4"],
    )
    database = {"gene_info": pd.DataFrame({"Symbol": ["Ligand", "Receptor"]})}

    combined = preprocess_multiomics([rna, adt], database=database, cutoff=0.5)[
        "expression"
    ]
    assert isinstance(combined, pd.DataFrame)
    assert combined.index.tolist() == ["spot_1", "spot_2"]
    assert combined.columns.tolist() == ["Ligand", "Receptor", "CD3E", "CD4"]
    assert combined["CD3E"].eq(0.0).all()
    assert combined.loc["spot_1", "CD4"] == 0.0
    assert combined.loc["spot_2", "CD4"] > 0.0


def test_preprocess_multiomics_builds_antibody_focused_database() -> None:
    rna = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["spot_1", "spot_2"],
        columns=["Il16", "Cd4"],
    )
    adt = pd.DataFrame([[0.2], [0.8]], index=rna.index, columns=["CD4"])
    database = {
        "interaction": pd.DataFrame(
            {
                "interaction_name": ["IL16_CD4", "OTHER_PAIR"],
                "ligand": ["Il16", "OtherL"],
                "receptor": ["Cd4", "OtherR"],
            }
        ),
        "gene_info": pd.DataFrame(
            {
                "Symbol": ["Il16", "Cd4", "OtherL", "OtherR"],
                "AntibodyName": [pd.NA, "CD4", pd.NA, pd.NA],
            }
        ),
    }

    result = preprocess_multiomics([rna, adt], database=database)
    interaction = result["database"]["interaction"]
    assert interaction["interaction_name"].tolist() == ["IL16_CD4"]
    assert interaction.iloc[0]["ligand"] == "Il16"
    assert interaction.iloc[0]["receptor"] == "CD4"
    assert "CD4" in result["database"]["gene_info"]["Symbol"].values


def test_preprocess_multiomics_rejects_misaligned_rows() -> None:
    rna = pd.DataFrame([[1.0], [2.0]], index=["a", "b"], columns=["G"])
    adt = pd.DataFrame([[1.0], [2.0]], index=["b", "a"], columns=["P"])
    with pytest.raises(ValueError, match="same cell/spot rows"):
        preprocess_multiomics([rna, adt], database={})


def test_preprocess_multiomics_array_inputs_can_return_sparse() -> None:
    rna = np.array([[1.0, 2.0], [2.0, 1.0]])
    adt = np.array([[0.0], [3.0]])
    result = preprocess_multiomics([rna, adt], database={}, do_sparse=True)
    assert sparse.isspmatrix_csr(result["expression"])
    assert result["expression"].shape == (2, 3)
