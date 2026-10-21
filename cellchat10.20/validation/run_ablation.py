"""Warm single-process ablation; intentionally separate from fresh-process benchmarks."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "development"))
sys.path.insert(0, str(ROOT / "scripts"))

import CellChatPy as cc
from CellChatPy import _streaming
from CellChatPy.network_storage import stack_network_field
from benchmark_scaling import build_object, machine_info


@contextlib.contextmanager
def ablated(variant):
    original_closure = _streaming.required_gene_indices
    original_averager = _streaming.GroupAverager
    if variant == "no_pruning":
        _streaming.required_gene_indices = lambda names, *args: np.arange(len(names), dtype=np.intp)
    elif variant == "no_sparse_plan":
        class DenseFallbackAverager(original_averager):
            def __init__(self, data, categories, type_method, trim=0.1, gene_batch_size=64):
                self.data, self.categories, self.type_method = data, list(categories), type_method
                self.trim, self.gene_batch_size, self.plan = trim, gene_batch_size, None
        _streaming.GroupAverager = DenseFallbackAverager
    elif variant != "all_enabled":
        raise ValueError(variant)
    try:
        yield
    finally:
        _streaming.required_gene_indices = original_closure
        _streaming.GroupAverager = original_averager


def main():
    args = SimpleNamespace(dataset="synthetic", cells=5000, genes=256,
                           groups=10, density=0.05, pairs=100)
    obj, candidates, metadata = build_object(args, cc)
    kwargs = dict(lr_use=candidates, type_method="triMean", n_boot=20, seed_use=1,
                  population_size=False, engine="streaming", gene_batch_size=64,
                  lr_batch_size=64)
    source_files = [ROOT / "development/CellChatPy" / name for name in
                    ("modeling.py", "_streaming.py", "_sparse_quantiles.py", "network_storage.py")]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}

    def run(variant):
        copied = obj.copy()  # Data preparation/copy is excluded from timed inference.
        with ablated(variant), contextlib.redirect_stdout(io.StringIO()):
            started = perf_counter()
            cc.compute_communication_probability(copied, **kwargs)
            elapsed = perf_counter() - started
        return copied, elapsed

    reference, initial_warm_seconds = run("all_enabled")
    reference_arrays = {field: stack_network_field(reference.network, field) for field in ("prob", "pval")}
    variants = ["all_enabled", "no_pruning", "no_sparse_plan"]
    results = {variant: {"seconds": [], "exact_prob": [], "exact_pval": [],
                         "evaluated_genes": []} for variant in variants}
    warmups = {"all_enabled": initial_warm_seconds}
    for variant in variants[1:]:
        _, elapsed = run(variant)
        warmups[variant] = elapsed
    run_order = []
    for repeat in range(3):
        rotated = variants[repeat:] + variants[:repeat]
        for variant in rotated:
            actual, elapsed = run(variant)
            row = results[variant]
            row["seconds"].append(elapsed)
            row["evaluated_genes"].append(actual.settings["execution"]["evaluated_genes"])
            for field in ("prob", "pval"):
                row[f"exact_{field}"].append(bool(np.array_equal(
                    stack_network_field(actual.network, field), reference_arrays[field])))
            run_order.append({"repeat": repeat + 1, "variant": variant})
    base = float(np.median(results["all_enabled"]["seconds"]))
    for row in results.values():
        row["median_seconds"] = float(np.median(row["seconds"]))
        row["median_time_relative_to_all_enabled"] = row["median_seconds"] / base
    report = {
        "scope": "synthetic group-level inference ablation; not a real biological dataset",
        "timing_mode": "one warm process; one untimed warmup per variant; rotated variant order; three timed repeats",
        "excluded_from_timing": ["package import", "object generation", "object copy", "array comparison"],
        "parameters": vars(args), "inference_parameters": {k: v for k, v in kwargs.items() if k != "lr_use"},
        "dataset": metadata, "source_sha256": source_hashes,
        "warmup_seconds_excluded": warmups, "run_order": run_order,
        "results": results, "machine": machine_info(),
        "interpretation": "Ablations remove one execution optimization at a time; effects need not be additive. Sparse fallback remains batched and vectorized. These ratios are not the independent fresh-process upstream benchmark.",
    }
    destination = ROOT / "validation/ablation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "output": str(destination)}, indent=2))
    if not all(all(row[field]) for row in results.values() for field in ("exact_prob", "exact_pval")):
        raise SystemExit("Ablation changed numerical outputs")


if __name__ == "__main__":
    main()
