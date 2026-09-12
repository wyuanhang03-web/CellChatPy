"""Numerical primitives, canonical network storage, and inference workflow tests."""

import copy
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc
from CellChatPy.modeling import _thresholded_mean, _tri_mean, _truncated_mean
from CellChatPy.network_storage import (
    matrix_dict_from_array,
    normalize_network,
    stack_network_field,
)


def test_matrix_dictionary_round_trip_preserves_names_and_values() -> None:
    tensor = np.arange(12, dtype=float).reshape(2, 2, 3)
    network = {
        "groups": ["A", "B"],
        "prob": matrix_dict_from_array(
            tensor, ["LR", "LR", "OTHER"], sparse_output=True
        ),
        "pval": matrix_dict_from_array(
            np.zeros_like(tensor), ["LR", "LR", "OTHER"]
        ),
    }

    assert list(network["prob"]) == ["LR", "LR__2", "OTHER"]
    assert all(sparse.isspmatrix_csr(value) for value in network["prob"].values())
    np.testing.assert_array_equal(stack_network_field(network, "prob"), tensor)


def test_network_validation_rejects_dense_probability_storage() -> None:
    invalid = {
        "groups": ["A"],
        "prob": {"LR": np.ones((1, 1))},
        "pval": {"LR": np.zeros((1, 1))},
    }
    with pytest.raises(TypeError, match="CSR matrix"):
        normalize_network(invalid)


def test_network_validation_rejects_misaligned_probability_names() -> None:
    invalid = {
        "groups": ["A"],
        "prob": {"LR": sparse.csr_matrix([[1.0]])},
        "pval": {"OTHER": np.zeros((1, 1))},
    }
    with pytest.raises(ValueError, match="names must match"):
        normalize_network(invalid)


@pytest.mark.parametrize(
    "average",
    [
        _tri_mean,
        lambda values: _truncated_mean(values, trim=0.25),
        lambda values: _thresholded_mean(values, trim=0.25),
        np.nanmedian,
    ],
    ids=["triMean", "truncatedMean", "thresholdedMean", "median"],
)
def test_scalar_average_methods_have_known_values(average) -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert average(values) == pytest.approx(2.5)


def test_communication_dataframe_builds_aligned_sparse_network() -> None:
    communication = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "interaction_name": ["LR1", "LR2"],
            "ligand_complex": ["L1", "L2"],
            "receptor_complex": ["R1", "R2"],
            "pathway_name": ["P1", "P2"],
            "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
            "prob": [0.4, 0.3],
            "pval": [0.01, 0.04],
        }
    )
    network = cc.communication_dataframe_to_network(communication, groups=["A", "B"])

    assert network["groups"] == ["A", "B"]
    np.testing.assert_allclose(network["prob"]["LR1"].toarray(), [[0, 0.4], [0, 0]])
    np.testing.assert_allclose(network["prob"]["LR2"].toarray(), [[0, 0], [0.3, 0]])


def test_aggregate_network_applies_strict_pvalue_threshold(network_cellchat) -> None:
    result = cc.aggregate_network(network_cellchat, thresh=0.05)

    np.testing.assert_array_equal(result.network["count"], [[1, 1], [1, 1]])
    np.testing.assert_allclose(result.network["weight"], [[0.1, 0.4], [0.3, 0.5]])


def test_pathway_probability_uses_interaction_metadata(network_cellchat) -> None:
    result = cc.compute_pathway_probability(network_cellchat, thresh=1.01)

    assert list(result.pathway_network["prob"]) == ["PATH_B", "PATH_A"]
    total = cc.aggregate_signaling_matrix(result, thresh=0.05)
    np.testing.assert_allclose(total, [[0.1, 0.6], [0.3, 0.5]])


def _inference_cellchat():
    cells = [f"cell_{index}" for index in range(12)]
    expression = pd.DataFrame(
        {
            "L1": [4, 5, 4, 5, 4, 5, 0, 0, 1, 0, 1, 0],
            "R1": [0, 1, 0, 1, 0, 1, 4, 5, 4, 5, 4, 5],
        },
        index=cells,
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A"] * 6 + ["B"] * 6),
            "cellchat_dataset": pd.Categorical(["sample"] * 12),
        },
        index=cells,
    )
    cellchat = cc.create_cellchat(expression, metadata=metadata, group_by="cell_type")
    cellchat.database = {"complex": pd.DataFrame(), "cofactor": pd.DataFrame()}
    cellchat.lr_pairs["significant"] = pd.DataFrame(
        {
            "ligand": ["L1"],
            "receptor": ["R1"],
            "interaction_name": ["L1_R1"],
            "pathway_name": ["PATH_A"],
            "annotation": ["Secreted Signaling"],
        }
    )
    return cellchat


def test_accelerated_inference_matches_reference_path() -> None:
    if "accelerate" not in inspect.signature(cc.compute_communication_probability).parameters:
        pytest.skip("accelerated inference is not available in this revision")

    cellchat = _inference_cellchat()
    rng = np.random.RandomState(23)
    permutations = np.column_stack([rng.permutation(cellchat.n_obs) for _ in range(7)])

    reference = cc.compute_communication_probability(
        copy.deepcopy(cellchat),
        permutation_use=permutations,
        accelerate=False,
    )
    accelerated = cc.compute_communication_probability(
        copy.deepcopy(cellchat),
        permutation_use=permutations,
        accelerate=True,
        bootstrap_batch_size=1,
    )

    for field in ("prob", "pval"):
        np.testing.assert_allclose(
            stack_network_field(accelerated.network, field),
            stack_network_field(reference.network, field),
            rtol=1e-13,
            atol=1e-15,
        )


def test_small_inference_pipeline_produces_finite_labeled_outputs() -> None:
    cellchat = _inference_cellchat()
    result = cc.compute_communication_probability(cellchat, n_boot=5, seed_use=7)
    result = cc.compute_pathway_probability(result, thresh=1.01)
    result = cc.aggregate_network(result, thresh=1.01)
    table = cc.subset_communication(result, thresh_pval=1.01)

    probability = stack_network_field(result.network, "prob")
    pvalue = stack_network_field(result.network, "pval")
    assert result.network["groups"] == ["A", "B"]
    assert probability.shape == pvalue.shape == (2, 2, 1)
    assert np.isfinite(probability).all() and np.all(probability >= 0)
    assert np.all((pvalue >= 0) & (pvalue <= 1))
    assert {"source", "target", "prob", "pval"}.issubset(table.columns)
