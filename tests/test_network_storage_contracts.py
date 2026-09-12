"""Canonical group- and spot-network storage contract tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from CellChatPy.network_storage import (
    expand_dense_group_axes,
    expand_group_axes,
    normalize_network,
    normalize_network_similarity,
    normalize_spot_network,
    reorder_group_axes,
    stack_network_field,
    zero_group_axes,
)


def _group_network():
    return {
        "groups": ["A", "B"],
        "prob": {"PAIR": sparse.csr_matrix([[0.0, 0.4], [0.2, 0.0]])},
        "pval": {"PAIR": np.array([[1.0, 0.01], [0.04, 1.0]])},
    }


def _spot_network():
    spots = ["spot_1", "spot_2"]
    return {
        "spots": spots,
        "prob": {"PAIR": sparse.csr_matrix([[0.0, 0.5], [0.2, 0.0]])},
        "interactions": pd.DataFrame({"interaction_name": ["PAIR"]}),
        "parameters": {"distance_use": False},
        "centrality": {
            "outdeg": pd.DataFrame([[0.5], [0.2]], index=spots, columns=["PAIR"])
        },
    }


def test_stack_network_field_respects_requested_order_and_fill_value() -> None:
    network = _group_network()
    stacked = stack_network_field(
        network, "pval", names=["MISSING", "PAIR"], fill_value=1.0
    )
    assert stacked.shape == (2, 2, 2)
    np.testing.assert_array_equal(stacked[:, :, 0], np.ones((2, 2)))
    np.testing.assert_array_equal(stacked[:, :, 1], network["pval"]["PAIR"])


def test_normalize_network_returns_independent_canonical_copies() -> None:
    source = _group_network()
    normalized = normalize_network(source)
    normalized["prob"]["PAIR"].data[0] = 9.0
    normalized["pval"]["PAIR"][0, 1] = 8.0
    assert source["prob"]["PAIR"].data[0] != 9.0
    assert source["pval"]["PAIR"][0, 1] != 8.0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"groups": ["A", "A"]}, "unique names"),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_normalize_network_rejects_ambiguous_structure(update, message: str) -> None:
    network = _group_network()
    network.update(update)
    with pytest.raises(ValueError, match=message):
        normalize_network(network)


def test_normalize_similarity_rejects_unknown_slot_and_analysis_type() -> None:
    with pytest.raises(ValueError, match="unknown slots"):
        normalize_network_similarity({"other": {}})
    with pytest.raises(ValueError, match="unknown similarity types"):
        normalize_network_similarity({"network": {"other": {}}})


def test_normalize_spot_network_copies_sparse_and_tabular_values() -> None:
    source = _spot_network()
    normalized = normalize_spot_network(source)
    normalized["prob"]["PAIR"].data[0] = 7.0
    normalized["centrality"]["outdeg"].iloc[0, 0] = 6.0
    assert source["prob"]["PAIR"].data[0] != 7.0
    assert source["centrality"]["outdeg"].iloc[0, 0] != 6.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(spots=["spot_1", "spot_1"]), "unique names"),
        (
            lambda value: value["prob"].update(
                PAIR=sparse.csr_matrix([[0.0, -0.1], [0.0, 0.0]])
            ),
            "finite and non-negative",
        ),
        (
            lambda value: value.update(
                interactions=pd.DataFrame({"interaction_name": ["OTHER"]})
            ),
            "same order",
        ),
    ],
)
def test_normalize_spot_network_rejects_misaligned_values(mutate, message: str) -> None:
    network = _spot_network()
    mutate(network)
    with pytest.raises(ValueError, match=message):
        normalize_spot_network(network)


def test_group_axis_helpers_preserve_values_and_input_independence() -> None:
    dense = {"PAIR": np.array([[1.0, 2.0], [3.0, 4.0]])}
    sparse_values = {"PAIR": sparse.csr_matrix(dense["PAIR"])}

    zeroed = zero_group_axes(dense, [1])["PAIR"]
    np.testing.assert_array_equal(zeroed, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(dense["PAIR"], [[1.0, 2.0], [3.0, 4.0]])

    reordered = reorder_group_axes(sparse_values, [1, 0])["PAIR"]
    assert sparse.isspmatrix_csr(reordered)
    np.testing.assert_array_equal(reordered.toarray(), [[4.0, 3.0], [2.0, 1.0]])

    expanded_sparse = expand_group_axes(sparse_values, [0, 2], 3)["PAIR"]
    np.testing.assert_array_equal(
        expanded_sparse.toarray(), [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]
    )
    expanded_dense = expand_dense_group_axes(dense, [0, 2], 3, fill_value=1.0)[
        "PAIR"
    ]
    np.testing.assert_array_equal(
        expanded_dense, [[1.0, 1.0, 2.0], [1.0, 1.0, 1.0], [3.0, 1.0, 4.0]]
    )
