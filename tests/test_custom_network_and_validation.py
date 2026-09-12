"""Custom communication-table ingestion and CellChat validation tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc


def _custom_network_table():
    return pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "score": [0.8, 0.3],
        }
    )


def test_update_communication_score_builds_labeled_sparse_network(toy_cellchat) -> None:
    result = cc.update_communication_score(toy_cellchat, _custom_network_table())
    assert result.network["groups"] == ["A", "B"]
    assert list(result.network["prob"]) == ["L1_R1", "L2_R2"]
    assert all(sparse.isspmatrix_csr(value) for value in result.network["prob"].values())
    assert result.network["prob"]["L1_R1"][0, 1] == pytest.approx(0.8)
    assert result.network["prob"]["L2_R2"][1, 0] == pytest.approx(0.3)
    np.testing.assert_array_equal(result.network["pval"]["L1_R1"], np.zeros((2, 2)))


def test_update_communication_score_preserves_explicit_names_and_pvalues(
    toy_cellchat,
) -> None:
    table = _custom_network_table().iloc[[0]].copy()
    table["interaction_name"] = ["custom_pair"]
    table["pval"] = [0.025]
    result = cc.update_communication_score(toy_cellchat, table)
    assert list(result.network["prob"]) == ["custom_pair"]
    assert result.network["pval"]["custom_pair"][0, 1] == pytest.approx(0.025)


def test_update_communication_score_rejects_missing_columns(toy_cellchat) -> None:
    with pytest.raises(ValueError, match="must contain columns"):
        cc.update_communication_score(toy_cellchat, _custom_network_table().drop(columns="score"))


def test_update_communication_score_rejects_unknown_groups(toy_cellchat) -> None:
    table = _custom_network_table()
    table.loc[0, "target"] = "unknown"
    with pytest.raises(ValueError, match="absent from cellchat.groups"):
        cc.update_communication_score(toy_cellchat, table)


def test_validate_cellchat_reports_shape_groups_and_validity(toy_cellchat) -> None:
    result = cc.validate_cellchat(toy_cellchat)
    assert result["is_valid"] is True
    assert result["errors"] == []
    assert result["info"] == {"n_cells": 6, "n_genes": 4, "groups": ["A", "B"]}


def test_validate_cellchat_rejects_non_cellchat_objects() -> None:
    result = cc.validate_cellchat(np.zeros((2, 2)))
    assert result["is_valid"] is False
    assert result["errors"] == ["Expected a CellChat object, got ndarray"]
