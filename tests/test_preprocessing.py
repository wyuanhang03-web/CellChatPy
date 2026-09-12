"""Expression normalization, scaling, feature selection, and reduction tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc


def test_normalize_data_scales_each_cell_library() -> None:
    counts = np.array([[1.0, 3.0], [2.0, 2.0]])
    normalized = cc.normalize_data(
        counts, scale_factor=100.0, do_log=False, do_sparse=False
    )
    np.testing.assert_allclose(normalized, [[25.0, 75.0], [50.0, 50.0]])
    np.testing.assert_allclose(normalized.sum(axis=1), [100.0, 100.0])


def test_normalize_data_dense_and_sparse_inputs_match() -> None:
    counts = np.array([[1.0, 3.0], [2.0, 2.0]])
    dense = cc.normalize_data(counts, scale_factor=10.0, do_log=True, do_sparse=False)
    sparse_result = cc.normalize_data(
        sparse.csr_matrix(counts), scale_factor=10.0, do_log=True, do_sparse=True
    )
    assert sparse.isspmatrix_csr(sparse_result)
    np.testing.assert_allclose(sparse_result.toarray(), dense)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([[0.0, 0.0], [1.0, 2.0]]), "positive library size"),
        (np.array([[1.0, -1.0], [1.0, 2.0]]), "non-negative counts"),
    ],
)
def test_normalize_data_rejects_invalid_counts(counts, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cc.normalize_data(counts)


def test_scale_matrix_preserves_dataframe_labels() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]],
        index=["gene_a", "gene_b"],
        columns=["cell_a", "cell_b", "cell_c"],
    )
    scaled = cc.scale_matrix(frame, scale="row")
    assert isinstance(scaled, pd.DataFrame)
    assert scaled.index.tolist() == frame.index.tolist()
    assert scaled.columns.tolist() == frame.columns.tolist()
    np.testing.assert_allclose(scaled.mean(axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(scaled.std(axis=1, ddof=0), 1.0)


def test_sum_scaling_normalizes_requested_axis() -> None:
    matrix = np.array([[1.0, 3.0], [1.0, 1.0]])
    np.testing.assert_allclose(cc.scale_matrix(matrix, "r1").sum(axis=1), 1.0)
    np.testing.assert_allclose(cc.scale_matrix(matrix, "c1").sum(axis=0), 1.0)


def test_subset_signaling_data_keeps_anndata_shape_and_marks_features(toy_cellchat) -> None:
    result = cc.subset_signaling_data(toy_cellchat, features=["L1", "R1", "missing"])
    assert result is toy_cellchat
    assert result.signaling.shape == result.shape == (6, 4)
    assert sparse.isspmatrix_csr(result.signaling)
    np.testing.assert_array_equal(
        result.var["is_signaling"].to_numpy(), [True, False, True, False]
    )
    np.testing.assert_array_equal(
        result.signaling.toarray()[:, [1, 3]], np.zeros((6, 2))
    )


def test_low_rank_signaling_preprocessing_is_reproducible(toy_cellchat) -> None:
    first = cc.preprocess_signaling_data(
        toy_cellchat.copy(), n_components=1, random_state=17
    )
    second = cc.preprocess_signaling_data(
        toy_cellchat.copy(), n_components=1, random_state=17
    )
    first_values = first.signaling.toarray()
    second_values = second.signaling.toarray()
    assert first_values.shape == toy_cellchat.shape
    assert np.isfinite(first_values).all() and np.all(first_values >= 0)
    np.testing.assert_allclose(first_values, second_values)
    assert first.settings["preprocessing"]["signaling_low_rank"]["n_components"] == 1


def test_run_pca_preserves_dataframe_index_and_component_names() -> None:
    frame = pd.DataFrame(
        np.arange(20, dtype=float).reshape(5, 4),
        index=[f"cell_{index}" for index in range(5)],
        columns=[f"gene_{index}" for index in range(4)],
    )
    result = cc.run_pca(frame, dim_pc=2, seed_use=7)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 2)
    assert result.index.tolist() == frame.index.tolist()
    assert result.columns.tolist() == ["PC1", "PC2"]
