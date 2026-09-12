"""Comparative-network and small analysis-helper tests."""

import numpy as np
import pandas as pd
import pytest

import CellChatPy as cc


def test_get_max_weight_handles_groups_summaries_and_named_interactions(
    network_cellchat,
) -> None:
    result = cc.get_max_weight(
        [network_cellchat],
        slot_name=["groups", "network", "network"],
        attribute=["groups", "weight", "L1_R1"],
    )
    assert result == {"groups": 3, "weight": 0.5, "L1_R1": 0.4}


def test_merge_interactions_collapses_axes_without_mutating_source(
    network_cellchat,
) -> None:
    result = cc.merge_interactions(network_cellchat, ["combined", "combined"])
    assert result.network["groups"] == ["combined"]
    np.testing.assert_allclose(result.network["count"], [[4.0]])
    np.testing.assert_allclose(result.network["weight"], [[1.3]])
    assert list(result.groups.categories) == ["combined"]
    assert network_cellchat.network["count"].shape == (2, 2)
    assert list(network_cellchat.groups.categories) == ["A", "B"]


def test_merge_interactions_rejects_one_label_per_cell_instead_of_per_group(
    network_cellchat,
) -> None:
    with pytest.raises(ValueError, match="one label for each group"):
        cc.merge_interactions(network_cellchat, ["x"] * network_cellchat.n_obs)


def test_extract_gene_subset_expands_complex_subunits(
    toy_cellchat, mini_database
) -> None:
    toy_cellchat.database = mini_database
    pair = pd.DataFrame({"ligand": ["L2"], "receptor": ["Rcomplex"]})
    assert cc.extract_gene_subset_from_pair(pair, toy_cellchat) == ["L2", "R1", "R2"]


def test_compute_enrichment_score_matches_hand_calculation() -> None:
    mapped = pd.DataFrame(
        {
            "interaction_name": ["L_R"],
            "ligand_log_fc": [2.0],
            "receptor_log_fc": [3.0],
            "ligand_pct_1": [0.25],
            "ligand_pct_2": [0.75],
            "receptor_pct_1": [0.2],
            "receptor_pct_2": [0.7],
        }
    )
    database = {
        "interaction": pd.DataFrame(
            {"ligand": ["L"], "receptor": ["R"], "pathway_name": ["PATH"]},
            index=["L_R"],
        )
    }
    result = cc.compute_enrichment_score(mapped, measure="LR-pair", database=database)
    assert result.loc[0, "interaction_name"] == "L_R"
    assert result.loc[0, "score"] == pytest.approx(1.5)


def test_compute_enrichment_score_rejects_unknown_measure() -> None:
    with pytest.raises(ValueError, match="measure must be one of"):
        cc.compute_enrichment_score(
            pd.DataFrame({"interaction_name": ["L_R"]}), measure="unknown"
        )


def test_color_ramp_maps_endpoints_and_missing_values() -> None:
    result = cc.color_ramp(np.array([0.0, 0.5, 1.0, np.nan]), ["black", "white"])
    assert result[0] == "#000000"
    assert result[2] == "#ffffff"
    assert result[3] == "grey"
