"""Independent regression against the frozen GitHub commit, with adversarial ties.

Run from cellchat10.20 with:
    .venv/Scripts/python -m pytest validation/test_streaming_engine.py -q
"""

from __future__ import annotations

import copy
import importlib
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "development"))

import CellChatPy as cc
from CellChatPy import modeling
from CellChatPy._streaming import group_average
from CellChatPy.network_storage import stack_network_field


@pytest.fixture(scope="session")
def upstream():
    """Load the frozen source, not the development 'legacy' dispatch branch."""
    name = "_cellchat_frozen_reference"
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "upstream" / "CellChatPy")]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.modeling")


def build_fixture(storage="csr", constant=False):
    genes = ["L1", "L2", "R1", "R2", "CA1", "CA2", "CI1", "CI2",
             "AG1", "AG2", "AN1", "AN2", "ZERO", "IGNORED"]
    rng = np.random.RandomState(407)
    values = rng.lognormal(-0.4, 0.9, size=(36, len(genes)))
    values[rng.uniform(size=values.shape) < 0.42] = 0.0
    values[:9, 0] += 2.0
    values[9:21, 2] += 1.7
    values[:, 12] = 0.0
    values[:, 13] = 500.0  # Must not affect signaling-layer normalization.
    if constant:
        values[:, :12] = np.arange(1, 13) / 13.0
    labels = pd.Categorical(["B"] * 9 + ["A"] * 12 + ["C"] * 15,
                            categories=["C", "B", "A"], ordered=True)
    expression = pd.DataFrame(values, columns=genes,
                              index=[f"cell_{i}" for i in range(len(values))])
    meta = pd.DataFrame({"cluster": labels, "cellchat_dataset": "sample"},
                        index=expression.index)
    obj = cc.create_cellchat(expression, metadata=meta, group_by="cluster")
    obj.X = sparse.csr_matrix(values) if storage == "csr" else values.copy()
    obj.signaling = obj.X.copy()
    obj.smoothed = obj.X.copy()
    obj.var["is_signaling"] = [True] * 13 + [False]
    interactions = pd.DataFrame({
        "interaction_name": ["simple", "reverse", "complex", "partial_complex",
                             "zero_complex", "missing", "activation", "inhibition",
                             "agonist", "antagonist", "all_factors", "unknown_factor"],
        "ligand": ["L1", "R1", "Lcomplex", "Lpartial", "Lzero", "NOT_MEASURED",
                   "L1", "L1", "L1", "L1", "Lcomplex", "L1"],
        "receptor": ["R1", "L1", "Rcomplex", "R1", "R1", "R1",
                     "R1", "R1", "R1", "R1", "Rcomplex", "R1"],
        "pathway_name": ["PATH_A"] * 6 + ["PATH_B"] * 6,
        "annotation": ["Secreted Signaling", "Cell-Cell Contact", "ECM-Receptor"] * 4,
        "co_A_receptor": [""] * 6 + ["ACT", "", "", "", "ACT", "UNKNOWN"],
        "co_I_receptor": [""] * 7 + ["INH", "", "", "INH", ""],
        "agonist": [""] * 8 + ["AGON", "", "AGON", ""],
        "antagonist": [""] * 9 + ["ANTAG", "ANTAG", ""],
    })
    obj.database = {
        "interaction": interactions.copy(),
        "complex": pd.DataFrame({
            "subunit_1": ["L1", "R1", "L2", "L1"],
            "subunit_2": ["L2", "R2", "ABSENT", "ZERO"],
            "subunit_3": ["", "", "", ""],
        }, index=["Lcomplex", "Rcomplex", "Lpartial", "Lzero"]),
        "cofactor": pd.DataFrame({
            "cofactor1": ["CA1", "CI1", "AG1", "AN1"],
            "cofactor2": ["CA2", "CI2", "AG2", "AN2"],
            "cofactor3": ["ABSENT", "", "", ""],
        }, index=["ACT", "INH", "AGON", "ANTAG"]),
        "gene_info": pd.DataFrame({"Symbol": genes}),
    }
    obj.lr_pairs = {"significant": interactions.copy()}
    return obj


def permutations(n_cells, n_boot=9):
    rng = np.random.RandomState(2001)
    # Identity guarantees that exact observed/bootstrap ties are exercised.
    return np.column_stack([np.arange(n_cells)] +
                           [rng.permutation(n_cells) for _ in range(n_boot - 1)])


def assert_same_result(reference, actual):
    assert list(reference.network["groups"]) == list(actual.network["groups"])
    pd.testing.assert_frame_equal(reference.network["interactions"],
                                  actual.network["interactions"])
    np.testing.assert_allclose(stack_network_field(actual.network, "prob"),
                               stack_network_field(reference.network, "prob"),
                               rtol=2e-14, atol=1e-16, equal_nan=True)
    # A tolerant probability comparison must never conceal a changed P value.
    np.testing.assert_array_equal(stack_network_field(actual.network, "pval"),
                                  stack_network_field(reference.network, "pval"))


@pytest.mark.parametrize("method", ["triMean", "truncatedMean", "thresholdedMean", "median"])
@pytest.mark.parametrize("storage", ["dense", "csr"])
@pytest.mark.parametrize("population_size", [False, True])
def test_same_external_permutations(upstream, method, storage, population_size):
    obj = build_fixture(storage)
    kwargs = dict(type_method=method, trim=0.2, population_size=population_size,
                  permutation_use=permutations(obj.n_obs), n_boot=2,
                  kh=0.37, n_param=1.3)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(
        obj.copy(), engine="streaming", gene_batch_size=3, lr_batch_size=5, **kwargs)
    assert_same_result(expected, actual)
    assert actual.settings["parameter"]["n_boot"] == 9


@pytest.mark.parametrize("gene_batch_size,lr_batch_size", [(1, 1), (4, 3), (64, 64)])
def test_fixed_seed_and_batch_sizes_match_frozen_source(upstream, gene_batch_size, lr_batch_size):
    obj = build_fixture()
    kwargs = dict(n_boot=11, seed_use=817, population_size=True)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(
        obj.copy(), engine="streaming", gene_batch_size=gene_batch_size,
        lr_batch_size=lr_batch_size, **kwargs)
    assert_same_result(expected, actual)


@pytest.mark.parametrize("method", ["triMean", "median", "truncatedMean", "thresholdedMean"])
def test_constant_expression_preserves_strict_ties(upstream, method):
    obj = build_fixture(constant=True)
    kwargs = dict(type_method=method, permutation_use=permutations(obj.n_obs),
                  population_size=True)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)
    if method in {"triMean", "median"}:
        probability = stack_network_field(actual.network, "prob")
        pvalue = stack_network_field(actual.network, "pval")
        assert np.all(pvalue[probability > 0] == 0)
        assert np.all(pvalue[probability == 0] == 1)


def test_identity_permutation_and_repeated_lr_rows(upstream):
    obj = build_fixture()
    rows = obj.lr_pairs["significant"].iloc[[10, 0, 10, 4]].copy()
    rows["interaction_name"] = ["all_again", "single", "all_again_2", "zero"]
    kwargs = dict(lr_use=rows, permutation_use=np.arange(obj.n_obs)[:, None])
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


def test_strict_comparison_does_not_merge_near_ties(upstream):
    expression = pd.DataFrame({"L": [1.0, 1.0, 1.0 + 1e-13, 1.0 + 1e-13],
                               "R": [1.0] * 4}, index=["a", "b", "c", "d"])
    meta = pd.DataFrame({"cluster": ["A", "A", "B", "B"]}, index=expression.index)
    obj = cc.create_cellchat(expression, metadata=meta, group_by="cluster")
    obj.signaling = obj.X.copy()
    obj.var["is_signaling"] = True
    interactions = pd.DataFrame({"ligand": ["L"], "receptor": ["R"],
                                 "interaction_name": ["L_R"], "pathway_name": ["P"]})
    obj.database = {"interaction": interactions, "complex": pd.DataFrame(),
                    "cofactor": pd.DataFrame()}
    obj.lr_pairs = {"significant": interactions}
    kwargs = dict(permutation_use=np.array([[2], [3], [0], [1]]))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)
    pvalue = stack_network_field(actual.network, "pval")
    assert pvalue[0, 0, 0] == 1.0
    assert pvalue[1, 0, 0] == 0.0


@pytest.mark.parametrize("raw_use,use_agan", [(False, False), (False, True), (True, False)])
def test_layer_selection_and_disable_agonist(upstream, raw_use, use_agan):
    obj = build_fixture()
    smooth = obj.X.toarray()
    smooth[:, :12] = np.sqrt(smooth[:, :12])
    obj.smoothed = sparse.csr_matrix(smooth)
    kwargs = dict(raw_use=raw_use, use_agan=use_agan,
                  permutation_use=permutations(obj.n_obs, 5))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


@pytest.mark.parametrize("distance_use,forced", [(True, False), (False, False), (True, True)])
def test_spatial_contact_arithmetic(upstream, monkeypatch, distance_use, forced):
    obj = build_fixture()
    obj.settings["datatype"] = "spatial"
    obj.obsm["spatial"] = np.arange(obj.n_obs * 2).reshape(obj.n_obs, 2).astype(float)
    obj.spatial = {"spatial_factors": {"ratio": 1.0, "tol": 0.0}}
    dist = np.array([[0.0, 150.0, np.inf], [150.0, 0.0, 200.0], [np.inf, 200.0, 0.0]])
    contact = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def fake_distance(**kwargs):
        return {"d_spatial": dist.copy(), "adj_contact": contact.copy()}

    monkeypatch.setattr(upstream, "compute_region_distance", fake_distance)
    monkeypatch.setattr(modeling, "compute_region_distance", fake_distance)
    kwargs = dict(distance_use=distance_use, contact_dependent_forced=forced,
                  permutation_use=permutations(obj.n_obs, 5), scale_distance=0.01)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


def scalar_group_average(upstream, data, labels, clusters, method, trim):
    if method == "truncatedMean":
        reducer = lambda x: upstream._truncated_mean(x, trim)
    elif method == "thresholdedMean":
        reducer = lambda x: upstream._thresholded_mean(x, trim)
    elif method == "median":
        reducer = lambda x: float(np.nanmedian(x))
    else:
        reducer = upstream._tri_mean
    return np.array([[reducer(row[labels == label]) for label in clusters]
                     for row in data])


@pytest.mark.parametrize("method", ["triMean", "truncatedMean", "thresholdedMean", "median", "fallback"])
@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_group_statistics_sparse_zero_negative_nan_semantics(upstream, method, storage):
    data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.1, 0.7, 1.1, 0, 0.3, 0, 0, 0.9, 0],
        [-3, -1, 0, 0.1, 0.7, 1.1, 0, -0.3, 0, 0, 0.9, 0],
        [np.nan, 0, 0, 0.1, 0.7, 1.1, 0, 0.3, np.nan, 0, 0.9, 0],
        [np.nan] * 12,
        [0.123456789, 0.123456781, 0.123456788, 1.123456789, 0, 0,
         0.987654321, 0.111111111, 0, 0, 1.987654321, 0],
    ], dtype=float)
    labels = np.array(["B"] * 5 + ["A"] * 7)
    clusters = ["A", "B"]
    expected = scalar_group_average(upstream, data, labels, clusters, method, 0.2)
    matrix = sparse.csr_matrix(data) if storage == "csr" else data.copy()
    actual = group_average(matrix, labels, clusters, method, trim=0.2, gene_batch_size=2)
    # Exact equality is deliberate: these summaries feed a strict inequality.
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_cells", list(range(1, 18)))
def test_trimean_all_small_quantile_boundaries(upstream, n_cells):
    rng = np.random.RandomState(n_cells)
    data = rng.lognormal(-1.0, 2.0, (7, n_cells))
    data[rng.random_sample(data.shape) < 0.65] = 0.0
    labels = np.zeros(n_cells, dtype=int)
    expected = scalar_group_average(upstream, data, labels, [0], "triMean", 0.1)
    actual = group_average(sparse.csr_matrix(data), labels, [0], "triMean", gene_batch_size=3)
    np.testing.assert_array_equal(actual, expected)


def test_explicit_zeros_and_duplicate_sparse_entries(upstream):
    data = sparse.csr_matrix((np.array([0.0, 0.2, 0.3, -0.1, 0.4, 0.0, 0.9]),
                              np.array([0, 1, 1, 3, 3, 0, 4]),
                              np.array([0, 5, 7])), shape=(2, 6))
    labels = np.array([0, 0, 0, 1, 1, 1])
    original = (data.data.copy(), data.indices.copy(), data.indptr.copy())
    expected = scalar_group_average(upstream, data.toarray(), labels, [0, 1], "triMean", 0.1)
    actual = group_average(data, labels, [0, 1], "triMean", gene_batch_size=1)
    np.testing.assert_array_equal(actual, expected)
    for before, after in zip(original, (data.data, data.indices, data.indptr)):
        np.testing.assert_array_equal(before, after)


def test_downstream_pathway_and_aggregation_preserved(upstream):
    obj = build_fixture()
    kwargs = dict(permutation_use=permutations(obj.n_obs, 13))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    for module, result in ((upstream, expected), (modeling, actual)):
        module.compute_pathway_probability(result)
        module.aggregate_network(result)
    np.testing.assert_array_equal(actual.network["count"], expected.network["count"])
    np.testing.assert_allclose(actual.network["weight"], expected.network["weight"],
                               rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(stack_network_field(actual.pathway_network, "prob"),
                               stack_network_field(expected.pathway_network, "prob"),
                               rtol=2e-14, atol=1e-16)


def test_source_expression_not_mutated():
    obj = build_fixture()
    x = obj.X.copy()
    signaling = obj.signaling.copy()
    database = copy.deepcopy(obj.database)
    modeling.compute_communication_probability(obj, engine="streaming", n_boot=3)
    assert (obj.X != x).nnz == 0
    assert (obj.signaling != signaling).nnz == 0
    for key in database:
        pd.testing.assert_frame_equal(database[key], obj.database[key])


@pytest.mark.parametrize("storage", ["csr", "dense"])
def test_dependency_pruning_preserves_full_signaling_maximum(upstream, monkeypatch, storage):
    from CellChatPy import _streaming

    obj = build_fixture(storage)
    # IGNORED is now selected as signaling but is not an LR/cofactor dependency.
    # Its value of 500 must still be used before dependency pruning.
    obj.var["is_signaling"] = True
    source = obj.signaling.toarray() if sparse.issparse(obj.signaling) else obj.signaling
    original = _streaming.GroupAverager
    captured = []

    def capture(data, *args, **kwargs):
        captured.append(data.toarray() if sparse.issparse(data) else data.copy())
        return original(data, *args, **kwargs)

    monkeypatch.setattr(_streaming, "GroupAverager", capture)
    kwargs = dict(permutation_use=permutations(obj.n_obs, 3))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)
    assert len(captured) == 1
    # Float64 division must match dense NumPy division bit for bit; SciPy's
    # sparse / scalar may multiply by a reciprocal and alter rounding.
    np.testing.assert_array_equal(captured[0], (source / source.max()).T[:13])


@pytest.mark.parametrize("bad", [
    np.arange(36),
    np.zeros((35, 2), dtype=int),
    np.empty((36, 0), dtype=int),
    np.zeros((36, 2), dtype=int),
    np.arange(36, dtype=float)[:, None] + 0.25,
    np.full((36, 1), np.nan),
    np.arange(1, 37)[:, None],
    np.arange(-1, 35)[:, None],
])
def test_invalid_external_permutations_rejected(bad):
    with pytest.raises(ValueError, match="permutation"):
        modeling.compute_communication_probability(build_fixture(), engine="streaming",
                                                    permutation_use=bad)


def test_empty_explicit_lr_table(upstream):
    obj = build_fixture()
    kwargs = dict(lr_use=obj.lr_pairs["significant"].iloc[:0].copy(), n_boot=3)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


def test_sparse_streaming_does_not_densify_or_allocate_bootstrap_tensors(monkeypatch):
    obj = build_fixture()
    n_boot = 7
    forbidden_shapes = {(obj.n_obs, n_boot), (n_boot, obj.n_obs),
                        (n_boot, obj.n_vars, 3), (n_boot, 13, 3),
                        (n_boot, 12, 3, 3)}

    def forbid_densification(*args, **kwargs):
        raise AssertionError("triMean sparse streaming must not call toarray/todense")

    def guard_allocator(original):
        def allocate(shape, *args, **kwargs):
            if isinstance(shape, (tuple, list)) and tuple(shape) in forbidden_shapes:
                raise AssertionError(f"Unexpected full bootstrap allocation {shape}")
            return original(shape, *args, **kwargs)
        return allocate

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbid_densification)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbid_densification)
    monkeypatch.setattr(np, "zeros", guard_allocator(np.zeros))
    monkeypatch.setattr(np, "empty", guard_allocator(np.empty))
    modeling.compute_communication_probability(obj, engine="streaming", n_boot=n_boot,
                                               gene_batch_size=2, lr_batch_size=3)


def test_global_rng_state_matches_frozen_source(upstream):
    obj = build_fixture()
    upstream.compute_communication_probability(obj.copy(), n_boot=5, seed_use=604)
    expected = np.random.random_sample(10)
    modeling.compute_communication_probability(obj.copy(), engine="streaming", n_boot=5, seed_use=604)
    actual = np.random.random_sample(10)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["triMean", "median", "truncatedMean", "thresholdedMean"])
def test_unused_group_category_matches_reference(upstream, method):
    obj = build_fixture()
    obj.groups = pd.Categorical(np.asarray(obj.groups), categories=["C", "B", "A", "unused"])
    kwargs = dict(type_method=method, permutation_use=permutations(obj.n_obs, 3))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


def test_no_measured_lr_dependencies_returns_reference_zero_network(upstream):
    obj = build_fixture()
    rows = obj.lr_pairs["significant"].iloc[:1].copy()
    rows["ligand"] = "UNMEASURED_LIGAND"
    rows["receptor"] = "UNMEASURED_RECEPTOR"
    kwargs = dict(lr_use=rows, n_boot=3)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)
    assert actual.settings["execution"]["evaluated_genes"] == 0


def test_empty_signaling_gene_selection_rejected_by_both_engines(upstream):
    obj = build_fixture()
    obj.var["is_signaling"] = False
    with pytest.raises(ValueError):
        upstream.compute_communication_probability(obj.copy(), n_boot=3)
    with pytest.raises(ValueError):
        modeling.compute_communication_probability(obj.copy(), engine="streaming", n_boot=3)


def test_inferred_streaming_network_survives_h5ad_round_trip(tmp_path):
    from anndata import read_h5ad
    from CellChatPy.cellchat_class import CellChat

    actual = modeling.compute_communication_probability(build_fixture(), engine="streaming", n_boot=3)
    path = tmp_path / "inferred_streaming_network.h5ad"
    actual.write_h5ad(path)
    restored = CellChat(read_h5ad(path))
    assert_same_result(actual, restored)
    assert restored.settings["execution"] == actual.settings["execution"]


@pytest.mark.parametrize("duplicate_name", [False, True])
def test_h5ad_retains_reversed_lr_and_pathway_axes(tmp_path, duplicate_name):
    from anndata import read_h5ad
    from CellChatPy.cellchat_class import CellChat
    from CellChatPy.network_storage import network_names

    obj = build_fixture()
    rows = obj.lr_pairs["significant"].iloc[::-1].reset_index(drop=True)
    if duplicate_name:
        rows.loc[0, "interaction_name"] = rows.loc[1, "interaction_name"]
    actual = modeling.compute_communication_probability(obj, engine="streaming", lr_use=rows, n_boot=3)
    modeling.compute_pathway_probability(actual, thresh=1.01)
    pathway = actual.pathway_network
    reverse_order = sorted(network_names(pathway), reverse=True)
    # Reorder real inferred pathway matrices before persisting; the explicit
    # saved axis must survive HDF5's independently sorted dictionary keys.
    actual.pathway_network = {
        "groups": pathway["groups"],
        "prob": {name: pathway["prob"][name] for name in reverse_order},
        "pval": {name: pathway["pval"][name] for name in reverse_order},
    }
    path = tmp_path / f"reversed_axis_duplicate_{duplicate_name}.h5ad"
    actual.write_h5ad(path)
    restored = CellChat(read_h5ad(path))
    assert_same_result(actual, restored)
    assert network_names(restored.pathway_network) == reverse_order
    for field in ("prob", "pval"):
        np.testing.assert_array_equal(stack_network_field(restored.pathway_network, field),
                                      stack_network_field(actual.pathway_network, field))
    if duplicate_name:
        assert network_names(restored.network)[1].endswith("__2")
    # Downstream pathway recomputation must still associate each LR score with
    # its original metadata, rather than silently using alphabetical key order.
    modeling.compute_pathway_probability(actual, thresh=1.01)
    modeling.compute_pathway_probability(restored, thresh=1.01)
    np.testing.assert_array_equal(stack_network_field(restored.pathway_network, "prob"),
                                  stack_network_field(actual.pathway_network, "prob"))


@pytest.mark.parametrize("method", ["triMean", "median"])
@pytest.mark.parametrize("storage", ["dense", "csr"])
def test_missing_group_labels_preserve_reference_permutation_population(upstream, method, storage):
    obj = build_fixture(storage)
    labels = np.asarray(obj.groups, dtype=object).copy()
    labels[[1, 10, 25]] = [np.nan, None, np.nan]
    obj.groups = pd.Categorical(labels, categories=obj.groups.categories)
    kwargs = dict(type_method=method, population_size=True,
                  permutation_use=permutations(obj.n_obs, 5))
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    assert_same_result(expected, actual)


@pytest.mark.parametrize("method", ["triMean", "median"])
def test_missing_group_labels_preserve_seeded_result_and_global_rng(upstream, method):
    obj = build_fixture("csr")
    labels = np.asarray(obj.groups, dtype=object).copy()
    labels[[0, 13, 35]] = np.nan
    obj.groups = pd.Categorical(labels, categories=obj.groups.categories)
    kwargs = dict(type_method=method, population_size=True, n_boot=5, seed_use=2481)
    expected = upstream.compute_communication_probability(obj.copy(), **kwargs)
    expected_rng = np.random.random_sample(9)
    actual = modeling.compute_communication_probability(obj.copy(), engine="streaming", **kwargs)
    actual_rng = np.random.random_sample(9)
    assert_same_result(expected, actual)
    np.testing.assert_array_equal(actual_rng, expected_rng)


def test_float_external_permutations_use_only_column_sized_validation(upstream, monkeypatch):
    """Catch both full-matrix conversion and full-matrix finite masks."""
    from CellChatPy import _streaming

    obj = build_fixture()
    supplied = permutations(obj.n_obs, 5).astype(float)
    expected = upstream.compute_communication_probability(
        obj.copy(), permutation_use=supplied, population_size=True)
    converted_shapes = []
    finite_shapes = []

    class NoMatrixConversion(np.ndarray):
        def astype(self, dtype, *args, **kwargs):
            converted_shapes.append(self.shape)
            assert self.ndim == 1, "external permutation conversion must be per column"
            return super().astype(dtype, *args, **kwargs)

    guarded = supplied.view(NoMatrixConversion)
    original_asarray = np.asarray
    original_isfinite = np.isfinite

    def guarded_asarray(value, *args, **kwargs):
        if value is supplied:
            return guarded
        return original_asarray(value, *args, **kwargs)

    def guarded_isfinite(value, *args, **kwargs):
        if isinstance(value, NoMatrixConversion):
            finite_shapes.append(value.shape)
            assert value.ndim == 1, "external permutation finite mask must be per column"
        return original_isfinite(value, *args, **kwargs)

    monkeypatch.setattr(_streaming.np, "asarray", guarded_asarray)
    monkeypatch.setattr(_streaming.np, "isfinite", guarded_isfinite)
    actual = modeling.compute_communication_probability(
        obj.copy(), engine="streaming", permutation_use=supplied, population_size=True)
    assert_same_result(expected, actual)
    assert converted_shapes == [(obj.n_obs,)] * supplied.shape[1]
    assert finite_shapes == [(obj.n_obs,)] * supplied.shape[1]


@pytest.mark.parametrize("bad_value", [0.25, np.nan, np.inf])
def test_float_external_permutations_validate_later_columns(bad_value):
    obj = build_fixture()
    supplied = permutations(obj.n_obs, 3).astype(float)
    supplied[0, -1] = bad_value
    with pytest.raises(ValueError, match="integer indices"):
        modeling.compute_communication_probability(
            obj, engine="streaming", permutation_use=supplied)
