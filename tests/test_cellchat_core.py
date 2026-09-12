"""Core CellChat construction, subsetting, copy, merge, and lift tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc
from CellChatPy.cellchat_class import CellChat
from CellChatPy.network_storage import matrix_dict_from_array, stack_network_field


def test_create_cellchat_preserves_cells_by_genes_contract(
    toy_expression: pd.DataFrame, toy_metadata: pd.DataFrame
) -> None:
    cellchat = cc.create_cellchat(
        toy_expression,
        metadata=toy_metadata,
        group_by="cell_type",
    )

    assert isinstance(cellchat, CellChat)
    assert cellchat.shape == (6, 4)
    assert list(cellchat.obs_names) == list(toy_expression.index)
    assert list(cellchat.var_names) == list(toy_expression.columns)
    assert sparse.isspmatrix_csr(cellchat.X)
    assert list(cellchat.groups.categories) == ["A", "B"]
    assert cellchat.settings["datatype"] == "RNA"


def test_create_cellchat_rejects_negative_expression(
    toy_expression: pd.DataFrame, toy_metadata: pd.DataFrame
) -> None:
    invalid = toy_expression.copy()
    invalid.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative values"):
        cc.create_cellchat(invalid, metadata=toy_metadata, group_by="cell_type")


def test_create_cellchat_rejects_misaligned_metadata(
    toy_expression: pd.DataFrame, toy_metadata: pd.DataFrame
) -> None:
    reversed_metadata = toy_metadata.iloc[::-1].copy()
    with pytest.raises(ValueError, match="same order"):
        cc.create_cellchat(
            toy_expression,
            metadata=reversed_metadata,
            group_by="cell_type",
        )


def test_metadata_subset_preserves_order_and_does_not_mutate_input(toy_cellchat) -> None:
    subset = cc.subset_cellchat(
        toy_cellchat,
        metadata_filter={"condition": "control"},
    )

    assert isinstance(subset, CellChat)
    assert list(subset.obs_names) == ["cell_0", "cell_1", "cell_3"]
    assert list(subset.groups.categories) == ["A", "B"]
    assert list(toy_cellchat.obs_names) == [f"cell_{index}" for index in range(6)]


def test_metadata_subset_combines_columns_with_and(toy_cellchat) -> None:
    subset = cc.subset_cellchat(
        toy_cellchat,
        metadata_filter={"condition": ["treated"], "cell_type": "B"},
    )
    assert list(subset.obs_names) == ["cell_4", "cell_5"]
    assert list(subset.groups.categories) == ["B"]


def test_subset_rejects_missing_cells_and_empty_selections(toy_cellchat) -> None:
    with pytest.raises(ValueError, match="Cells not found"):
        cc.subset_cellchat(toy_cellchat, cells_use=["not-a-cell"])
    with pytest.raises(ValueError, match="No cells selected"):
        cc.subset_cellchat(
            toy_cellchat,
            metadata_filter={"condition": "not-a-condition"},
        )


def test_copy_is_independent_and_keeps_cellchat_type(toy_cellchat) -> None:
    copied = toy_cellchat.copy()
    copied.X[0, 0] = 99.0

    assert isinstance(copied, CellChat)
    assert float(toy_cellchat.X[0, 0]) == 4.0
    assert float(copied.X[0, 0]) == 99.0


def _cellchat_with_groups(groups: list[str]):
    labels = [groups[0], groups[-1], groups[0], groups[-1]]
    expression = pd.DataFrame(
        np.ones((4, 2)),
        index=[f"cell_{index}" for index in range(4)],
        columns=["L", "R"],
    )
    metadata = pd.DataFrame({"group": labels}, index=expression.index)
    obj = cc.create_cellchat(expression, metadata=metadata, group_by="group")
    obj.groups = pd.Categorical(labels, categories=groups, ordered=True)
    matrix = np.arange(len(groups) ** 2, dtype=float).reshape(len(groups), len(groups))
    tensor = matrix[:, :, None]
    obj.network = {
        "groups": groups,
        "prob": matrix_dict_from_array(tensor, ["LR"], sparse_output=True),
        "pval": matrix_dict_from_array(np.zeros_like(tensor), ["LR"]),
        "count": matrix,
        "weight": matrix,
    }
    return obj, matrix


def test_lift_cellchat_pads_missing_group_axes_with_zero() -> None:
    cellchat, original = _cellchat_with_groups(["A", "C"])
    lifted = cc.lift_cellchat(cellchat, group_new=["A", "B", "C"])

    assert lifted.network["groups"] == ["A", "B", "C"]
    np.testing.assert_array_equal(
        lifted.network["count"][np.ix_([0, 2], [0, 2])], original
    )
    assert not lifted.network["count"][1, :].any()
    assert not lifted.network["count"][:, 1].any()
    probability = stack_network_field(lifted.network, "prob")
    assert probability.shape == (3, 3, 1)


def test_merge_cellchat_keeps_first_seen_group_order() -> None:
    first, _ = _cellchat_with_groups(["A", "C"])
    second, _ = _cellchat_with_groups(["A", "B", "C"])

    merged = cc.merge_cellchat(
        [first, second],
        add_names=["first", "second"],
        cell_prefix=True,
    )

    assert list(merged.obs["cellchat_group"].cat.categories) == ["A", "C", "B"]
    assert set(merged.network) == {"first", "second"}
