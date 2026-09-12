"""Graph Laplacian, eigengap, centrality, and pathway selection tests."""

import numpy as np
import pytest

import CellChatPy as cc


def test_normalized_laplacian_has_expected_two_node_spectrum() -> None:
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = cc.compute_laplacian(adjacency)
    np.testing.assert_allclose(result["val"], [0.0, 2.0], atol=1e-12)
    assert result["n_zeros"] == 1


def test_compute_laplacian_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        cc.compute_laplacian(np.ones((2, 3)))


def test_eigengap_returns_consistent_bounds_and_eigenvalues() -> None:
    consensus = np.array(
        [
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.0, 0.1],
            [0.1, 0.0, 1.0, 0.8],
            [0.0, 0.1, 0.8, 1.0],
        ]
    )
    result = cc.compute_eigengap(consensus, tau=0.3)
    assert 0 <= result["lower_bound"] <= 4
    assert 1 <= result["upper_bound"] <= 3
    assert np.all(np.diff(result["eigs"]["val"]) >= -1e-12)


def test_pathway_centrality_contains_finite_group_scores(network_cellchat) -> None:
    result = cc.compute_network_centrality(
        network_cellchat, slot_name="pathway_network"
    )
    centrality = result.pathway_network["centrality"]
    assert set(centrality) == {"PATH_A", "PATH_B"}
    for pathway_scores in centrality.values():
        assert {"outdeg", "indeg"}.issubset(pathway_scores)
        assert len(pathway_scores["outdeg"]) == 2
        assert np.isfinite(pathway_scores["outdeg"]).all()


def test_aggregate_signaling_matrix_rejects_unknown_pathway(network_cellchat) -> None:
    with pytest.raises(ValueError, match="not found"):
        cc.aggregate_signaling_matrix(network_cellchat, signaling="UNKNOWN")
