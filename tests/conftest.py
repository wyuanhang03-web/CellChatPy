"""Small deterministic fixtures shared by the CellChatPy test suite."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import CellChatPy as cc
from CellChatPy.network_storage import matrix_dict_from_array


@pytest.fixture
def toy_expression() -> pd.DataFrame:
    """Return a cells-by-genes matrix with stable names and non-negative values."""
    return pd.DataFrame(
        [
            [4.0, 1.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 1.0],
            [5.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 0.0, 5.0, 1.0],
        ],
        index=[f"cell_{index}" for index in range(6)],
        columns=["L1", "L2", "R1", "R2"],
    )


@pytest.fixture
def toy_metadata(toy_expression: pd.DataFrame) -> pd.DataFrame:
    """Return aligned metadata with ordered groups and two conditions."""
    return pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["A", "A", "A", "B", "B", "B"],
                categories=["A", "B"],
                ordered=True,
            ),
            "condition": ["control", "control", "treated", "control", "treated", "treated"],
            "cellchat_dataset": pd.Categorical(["sample"] * 6),
        },
        index=toy_expression.index,
    )


@pytest.fixture
def toy_cellchat(toy_expression: pd.DataFrame, toy_metadata: pd.DataFrame):
    """Return a minimal CellChat object without external data dependencies."""
    return cc.create_cellchat(
        toy_expression,
        metadata=toy_metadata,
        group_by="cell_type",
    )


@pytest.fixture
def mini_database() -> dict[str, pd.DataFrame]:
    """Return a two-pathway database small enough for unit tests."""
    interaction = pd.DataFrame(
        {
            "interaction_name": ["L1_R1", "L2_Rcomplex", "L2_R2"],
            "pathway_name": ["PATH_A", "PATH_B", "PATH_B"],
            "ligand": ["L1", "L2", "L2"],
            "receptor": ["R1", "Rcomplex", "R2"],
            "annotation": [
                "Secreted Signaling",
                "Cell-Cell Contact",
                "Cell-Cell Contact",
            ],
            "evidence": ["PMID:1", "PMID:2", "PMID:3"],
        }
    )
    complex_table = pd.DataFrame(
        {"subunit_1": ["R1"], "subunit_2": ["R2"]},
        index=["Rcomplex"],
    )
    return {
        "interaction": interaction,
        "complex": complex_table,
        "cofactor": pd.DataFrame(),
        "gene_info": pd.DataFrame(
            {"Symbol": ["L1", "L2", "R1", "R2"]}
        ),
    }


@pytest.fixture
def network_cellchat(toy_cellchat):
    """Attach aligned LR-level and pathway-level networks to a toy object."""
    lr_probability = np.array(
        [
            [[0.0, 0.1], [0.4, 0.2]],
            [[0.3, 0.0], [0.0, 0.5]],
        ]
    )
    lr_pvalue = np.array(
        [
            [[1.0, 0.01], [0.01, 0.20]],
            [[0.04, 1.0], [1.0, 0.01]],
        ]
    )
    lr_names = ["L1_R1", "L2_R2"]
    interactions = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "interaction_name": lr_names,
            "pathway_name": ["PATH_A", "PATH_B"],
            "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
        }
    )
    toy_cellchat.network = {
        "groups": ["A", "B"],
        "prob": matrix_dict_from_array(lr_probability, lr_names, sparse_output=True),
        "pval": matrix_dict_from_array(lr_pvalue, lr_names),
        "interactions": interactions,
        "count": np.array([[1, 1], [1, 1]]),
        "weight": np.array([[0.1, 0.4], [0.3, 0.5]]),
    }
    toy_cellchat.pathway_network = {
        "groups": ["A", "B"],
        "prob": matrix_dict_from_array(
            np.array([[[0.1, 0.0], [0.4, 0.2]], [[0.3, 0.0], [0.0, 0.5]]]),
            ["PATH_A", "PATH_B"],
            sparse_output=True,
        ),
        "pval": matrix_dict_from_array(
            np.zeros((2, 2, 2), dtype=float),
            ["PATH_A", "PATH_B"],
        ),
    }
    return toy_cellchat
