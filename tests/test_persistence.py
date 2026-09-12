"""Copy, pickle, and AnnData H5AD persistence tests."""

import copy
import pickle

import numpy as np
from anndata import read_h5ad
from scipy import sparse

from CellChatPy.cellchat_class import CellChat


def test_deepcopy_preserves_type_and_nested_network_independence(network_cellchat) -> None:
    copied = copy.deepcopy(network_cellchat)
    copied.network["weight"][0, 0] = 99.0
    copied.network["prob"]["L1_R1"][0, 1] = 88.0

    assert isinstance(copied, CellChat)
    assert network_cellchat.network["weight"][0, 0] != 99.0
    assert network_cellchat.network["prob"]["L1_R1"][0, 1] != 88.0


def test_pickle_round_trip_preserves_cellchat_and_networks(network_cellchat) -> None:
    restored = pickle.loads(pickle.dumps(network_cellchat))

    assert isinstance(restored, CellChat)
    np.testing.assert_array_equal(restored.X.toarray(), network_cellchat.X.toarray())
    assert restored.network["groups"] == ["A", "B"]
    assert sparse.isspmatrix_csr(restored.network["prob"]["L1_R1"])
    np.testing.assert_allclose(
        restored.pathway_network["prob"]["PATH_B"].toarray(),
        network_cellchat.pathway_network["prob"]["PATH_B"].toarray(),
    )


def test_h5ad_round_trip_preserves_canonical_network_storage(
    network_cellchat, tmp_path
) -> None:
    network_cellchat.feature_results["features"] = ["L1", "R1"]
    path = tmp_path / "cellchat.h5ad"
    network_cellchat.write_h5ad(path)
    restored = CellChat(read_h5ad(path))

    assert restored.network["groups"] == ["A", "B"]
    assert list(restored.network["prob"]) == ["L1_R1", "L2_R2"]
    assert all(sparse.isspmatrix_csr(value) for value in restored.network["prob"].values())
    np.testing.assert_allclose(
        restored.network["prob"]["L1_R1"].toarray(),
        network_cellchat.network["prob"]["L1_R1"].toarray(),
    )
    np.testing.assert_allclose(
        restored.network["pval"]["L2_R2"],
        network_cellchat.network["pval"]["L2_R2"],
    )
    assert list(restored.pathway_network["prob"]) == ["PATH_A", "PATH_B"]
    assert list(restored.feature_results["features"]) == ["L1", "R1"]
