"""Distance, spot-network inference, aggregation, and grid utility tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc


def _spatial_cellchat():
    cells = ["s1_a", "s1_b", "s2_a", "s2_b"]
    expression = pd.DataFrame(
        {
            "Ligand": [4.0, 0.0, 5.0, 0.0],
            "Receptor": [0.0, 4.0, 0.0, 5.0],
        },
        index=cells,
    )
    metadata = pd.DataFrame(
        {
            "cellchat_group": pd.Categorical(["A", "B", "A", "B"]),
            "cellchat_dataset": pd.Categorical(["sample1", "sample1", "sample2", "sample2"]),
        },
        index=cells,
    )
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.0], [1.2, 0.0]])
    cellchat = cc.create_cellchat(
        expression,
        metadata=metadata,
        group_by="cellchat_group",
        datatype="spatial",
        coordinates=coordinates,
        spatial_factors={"ratio": [1.0, 1.0], "tol": [0.0, 0.0]},
    )
    cellchat.database = {"complex": pd.DataFrame(), "cofactor": pd.DataFrame()}
    cellchat.lr_pairs["significant"] = pd.DataFrame(
        {
            "interaction_name": ["Ligand_Receptor"],
            "ligand": ["Ligand"],
            "receptor": ["Receptor"],
            "pathway_name": ["Synthetic"],
            "annotation": ["Secreted Signaling"],
        }
    )
    return cellchat


def test_compute_cell_distance_applies_ratio_and_ranges() -> None:
    result = cc.compute_cell_distance(
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        ratio=2.0,
        tol=0.0,
        interaction_range=10.0,
        contact_range=5.0,
    )
    np.testing.assert_allclose(result["distance_matrix"], [[0.0, 6.0], [6.0, 0.0]])
    np.testing.assert_array_equal(result["adj_contact"].toarray(), np.eye(2))
    assert result["min_distance"] == 6.0


def test_compute_region_distance_uses_trimmed_mean_for_outlier() -> None:
    coordinates = np.array(
        [[float(x), 0.0] for x in range(9)]
        + [[100.0, 0.0]]
        + [[float(x), 10.0] for x in range(10)]
    )
    metadata = pd.DataFrame(
        {
            "cellchat_group": pd.Categorical(["A"] * 10 + ["B"] * 10),
            "cellchat_dataset": pd.Categorical(["sample"] * 20),
        }
    )
    result = cc.compute_region_distance(
        coordinates,
        metadata,
        interaction_range=200.0,
        ratio=[1.0],
        tol=[0.0],
        k_min=1,
        contact_dependent=False,
    )
    nearest = np.sqrt(
        ((coordinates[:10, None, :] - coordinates[None, 10:, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    expected = np.sort(nearest)[1:-1].mean()
    assert result["d_spatial"][0, 1] == pytest.approx(expected)
    assert result["d_spatial"][0, 1] < nearest.mean()


def test_spot_distances_never_connect_different_samples() -> None:
    distance, contact = cc.compute_spot_distances(
        _spatial_cellchat(), interaction_range=2.0, contact_range=0.5
    )
    assert distance[0, 1] == pytest.approx(1.0)
    assert distance[2, 3] == pytest.approx(1.0)
    assert distance[0, 2] == 0.0
    assert distance[1, 3] == 0.0
    np.testing.assert_array_equal(contact.diagonal(), np.ones(4))


def test_spot_probability_is_sparse_nonnegative_and_sample_local() -> None:
    result = cc.compute_spot_communication_probability(
        _spatial_cellchat(),
        distance_use=False,
        interaction_range=2.0,
        contact_dependent=False,
    )
    probability = result.spot_network["prob"]["Ligand_Receptor"]
    assert sparse.isspmatrix_csr(probability)
    assert probability.shape == (4, 4)
    assert np.isfinite(probability.data).all() and np.all(probability.data >= 0)
    assert probability[0, 1] > 0 and probability[2, 3] > 0
    assert probability[0, 3] == 0 and probability[2, 1] == 0


def test_spot_probability_filter_rejects_sample_larger_than_spots() -> None:
    cellchat = cc.compute_spot_communication_probability(
        _spatial_cellchat(), distance_use=False, contact_dependent=False
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        cc.filter_spot_probability(cellchat, n_boot=cellchat.n_obs + 1)


def test_visium_aggregation_creates_group_network() -> None:
    cellchat = cc.compute_spot_communication_probability(
        _spatial_cellchat(), distance_use=False, contact_dependent=False
    )
    decomposition = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        index=cellchat.obs_names,
        columns=["A", "B"],
    )
    result = cc.aggregate_visium_communication(
        cellchat,
        decomposition,
        average_type="average",
        do_permutation=False,
    )
    assert result.network["groups"] == ["A", "B"]
    probability = result.network["prob"]["Ligand_Receptor"]
    assert sparse.isspmatrix_csr(probability)
    assert probability.shape == (2, 2)
    assert probability[0, 1] > 0


def test_compute_grid_size_uses_positive_coordinate_spacing() -> None:
    result = cc.compute_grid_size(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        grid_resolution=2.0,
    )
    assert result["base_cellsize"] == pytest.approx(1.0)
    assert result["cellsize"] == pytest.approx(2.0)


def test_colocalization_is_reproducible_and_symmetric() -> None:
    coordinates = np.array(
        [[0, 0], [0, 1], [3, 0], [3, 1], [6, 0], [6, 1]], dtype=float
    )
    groups = pd.Categorical(["A", "A", "B", "B", "C", "C"])
    first = cc.compute_colocalization(coordinates, groups, n_boot=5, seed_use=19)
    second = cc.compute_colocalization(coordinates, groups, n_boot=5, seed_use=19)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(first.to_numpy(), first.to_numpy().T, equal_nan=True)
