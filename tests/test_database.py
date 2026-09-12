"""Bundled and synthetic CellChatDB tests."""

import pandas as pd
import pytest

import CellChatPy as cc


@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_bundled_database_loads_with_required_schema(species: str) -> None:
    database = cc.load_database(species)

    assert set(database) == {"interaction", "complex", "cofactor", "gene_info"}
    assert not database["interaction"].empty
    assert {"ligand", "receptor", "pathway_name", "annotation"}.issubset(
        database["interaction"].columns
    )
    validation = cc.validate_database(database)
    assert validation["is_valid"], validation["errors"]


def test_unknown_species_is_rejected() -> None:
    with pytest.raises(ValueError, match="not available"):
        cc.load_database("not-a-species")


def test_subset_database_uses_or_within_column_and_and_between_columns(
    mini_database: dict[str, pd.DataFrame]
) -> None:
    subset = cc.subset_database(
        mini_database,
        search=[["PATH_A", "PATH_B"], ["Cell-Cell Contact"]],
        key=["pathway_name", "annotation"],
    )
    assert subset["interaction"]["interaction_name"].tolist() == [
        "L2_Rcomplex",
        "L2_R2",
    ]


def test_search_pair_supports_case_insensitive_partial_matching(
    mini_database: dict[str, pd.DataFrame]
) -> None:
    result = cc.search_pair(
        "path_b",
        mini_database["interaction"],
        key="pathway_name",
        matching_exact=False,
    )
    assert result["interaction_name"].tolist() == ["L2_Rcomplex", "L2_R2"]


def test_extract_lr_from_genes_includes_interactions_using_a_complex(
    mini_database: dict[str, pd.DataFrame]
) -> None:
    result = cc.extract_lr_from_genes(["R1"], mini_database)
    assert "Rcomplex" in result["complexes"]
    assert set(result["lr_pairs"]["interaction_name"]) == {"L1_R1", "L2_Rcomplex"}


def test_validate_database_reports_missing_required_column() -> None:
    invalid = {"interaction": pd.DataFrame({"ligand": ["L1"]})}
    validation = cc.validate_database(invalid)
    assert validation["is_valid"] is False
    assert "Missing column: 'receptor'" in validation["errors"]
