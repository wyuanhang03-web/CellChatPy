"""Small database selection, reporting, and lookup helper tests."""

import pandas as pd

import CellChatPy as cc


def test_extract_genes_returns_sorted_unique_ligands_and_receptors(mini_database) -> None:
    assert cc.extract_genes(mini_database) == ["L1", "L2", "R1", "R2", "Rcomplex"]


def test_filter_database_by_category_does_not_mutate_source(mini_database) -> None:
    filtered = cc.filter_database_by_category(mini_database, ["Cell-Cell Contact"])
    assert filtered["interaction"]["interaction_name"].tolist() == [
        "L2_Rcomplex",
        "L2_R2",
    ]
    assert len(mini_database["interaction"]) == 3


def test_get_pathway_interactions_preserves_database_order(mini_database) -> None:
    selected = cc.get_pathway_interactions(mini_database, ["PATH_B"])
    assert selected["interaction_name"].tolist() == ["L2_Rcomplex", "L2_R2"]


def test_validate_database_reports_counts_and_duplicate_warning() -> None:
    interaction = pd.DataFrame(
        {
            "ligand": ["L", "L"],
            "receptor": ["R", "R"],
            "pathway_name": ["PATH", "PATH"],
        }
    )
    result = cc.validate_database({"interaction": interaction})
    assert result["is_valid"] is True
    assert result["stats"] == {
        "n_interactions": 2,
        "n_unique_ligands": 1,
        "n_unique_receptors": 1,
        "n_pathways": 1,
    }
    assert result["warnings"] == ["1 duplicate interactions found"]


def test_get_pathway_interactions_returns_empty_frame_without_pathway_column() -> None:
    result = cc.get_pathway_interactions(
        {"interaction": pd.DataFrame({"ligand": ["L"], "receptor": ["R"]})},
        ["PATH"],
    )
    assert result.empty
