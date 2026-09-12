"""Small, self-contained tests for the spatial interchange format."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import CellChatPy as cc


def _spatial_input(sample: str = "sample1") -> dict:
    cells = ["spot_a", "spot_b"]
    return {
        "expression": sparse.csc_matrix([[1.0, 0.0], [0.0, 2.0]]),
        "genes": ["GeneA", "GeneB"],
        "cells": cells,
        "metadata": pd.DataFrame(
            {
                "cellchat_group": ["A", "B"],
                "cellchat_dataset": [sample, sample],
            },
            index=cells,
        ),
        "coordinates": pd.DataFrame(
            {"x": [10.0, 30.0], "y": [20.0, 40.0]}, index=cells
        ),
        "spatial_factors": {"ratio": [0.5], "tol": [32.5]},
    }


def test_spatial_h5_round_trip_preserves_values_and_labels(tmp_path) -> None:
    item = _spatial_input()
    path = tmp_path / "spatial.h5"
    adt = pd.DataFrame(
        [[3.0, 0.0], [1.0, 4.0]],
        index=item["cells"],
        columns=["CD3", "CD4"],
    )
    cc.write_spatial_h5(path, **item, sample="example", modalities={"adt": adt})
    loaded = cc.read_spatial_h5(path)

    assert loaded["sample"] == "example"
    assert loaded["genes"] == item["genes"]
    assert loaded["cells"] == item["cells"]
    np.testing.assert_array_equal(loaded["expression"].toarray(), item["expression"].toarray())
    np.testing.assert_allclose(loaded["coordinates"].to_numpy(), item["coordinates"].to_numpy())
    pd.testing.assert_frame_equal(loaded["modalities"]["adt"], adt)


def test_merge_spatial_inputs_prefixes_cells_and_samples(tmp_path) -> None:
    item = _spatial_input()
    path = tmp_path / "spatial.h5"
    cc.write_spatial_h5(path, **item)
    loaded = cc.read_spatial_h5(path)

    merged = cc.merge_spatial_inputs([loaded, loaded], sample_names=["one", "two"])
    assert merged["expression"].shape == (4, 2)
    assert merged["cells"] == ["one_spot_a", "one_spot_b", "two_spot_a", "two_spot_b"]
    assert merged["metadata"]["cellchat_dataset"].astype(str).tolist() == [
        "one", "one", "two", "two"
    ]


def test_read_visium_spatial_info_uses_image_column_as_x(tmp_path) -> None:
    spatial_dir = tmp_path / "spatial"
    spatial_dir.mkdir()
    (spatial_dir / "tissue_positions.csv").write_text(
        "barcode,in_tissue,array_row,array_col,imagerow,imagecol\n"
        "spot_a,1,2,3,40,50\n"
        "spot_b,1,4,5,60,70\n",
        encoding="utf-8",
    )
    (spatial_dir / "scalefactors_json.json").write_text(
        json.dumps({"spot_diameter_fullres": 100.0}), encoding="utf-8"
    )

    info = cc.read_visium_spatial_info(spatial_dir, ["spot_b", "spot_a"])
    assert info["coordinates"].columns.tolist() == ["x", "y"]
    np.testing.assert_allclose(info["coordinates"].to_numpy(), [[70, 60], [50, 40]])
    assert info["spatial_factors"] == {"ratio": 0.65, "tol": 32.5}


def test_write_spatial_h5_rejects_missing_group_column(tmp_path) -> None:
    item = _spatial_input()
    item["metadata"] = item["metadata"].drop(columns="cellchat_group")
    with pytest.raises(ValueError, match="cellchat_group"):
        cc.write_spatial_h5(tmp_path / "invalid.h5", **item)


def test_write_spatial_h5_rejects_misaligned_modality_rows(tmp_path) -> None:
    item = _spatial_input()
    adt = pd.DataFrame(
        [[1.0], [2.0]],
        index=list(reversed(item["cells"])),
        columns=["CD3"],
    )
    with pytest.raises(ValueError, match="same order"):
        cc.write_spatial_h5(
            tmp_path / "invalid.h5", **item, modalities={"adt": adt}
        )
