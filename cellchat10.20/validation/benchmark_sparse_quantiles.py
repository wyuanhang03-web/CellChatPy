"""Small reproducible microbenchmark; not an end-to-end speed claim."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "development"))
from CellChatPy._sparse_quantiles import SparseQuantilePlan
from CellChatPy.modeling import _tri_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=20_000)
    parser.add_argument("--genes", type=int, default=128)
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--density", type=float, default=0.35)
    parser.add_argument("--permutations", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.RandomState(1207)
    matrix = sparse.random(args.genes, args.cells, density=args.density,
                           format="csr", random_state=rng,
                           data_rvs=lambda n: rng.gamma(2, 0.3, n))
    labels = np.arange(args.cells) % args.groups
    categories = list(range(args.groups))
    orders = [labels] + [rng.permutation(labels) for _ in range(args.permutations)]
    start = perf_counter()
    plan = SparseQuantilePlan(matrix)
    setup_seconds = perf_counter() - start
    start = perf_counter()
    plan.average(labels, categories)
    first_call_seconds = perf_counter() - start

    dense = matrix.toarray()  # Deliberately exclude densification from baseline timing.
    reference_times, planned_times = [], []
    exact = True
    for order in orders:
        start = perf_counter()
        expected = np.zeros((args.genes, args.groups))
        for group in categories:
            block = dense[:, order == group]
            for gene in range(args.genes):
                expected[gene, group] = _tri_mean(block[gene])
        reference_times.append(perf_counter() - start)
        start = perf_counter()
        actual = plan.average(order, categories)
        planned_times.append(perf_counter() - start)
        exact &= bool(np.array_equal(actual, expected))
    report = {"scope": "group triMean primitive only; synthetic expression; not end-to-end",
              "parameters": vars(args), "nnz": matrix.nnz, "backend": plan.backend,
              "plan_setup_seconds": setup_seconds,
              "first_call_seconds_including_compile_or_cache_load": first_call_seconds,
              "reference_seconds_per_grouping": reference_times,
              "warm_plan_seconds_per_grouping": planned_times,
              "median_primitive_speedup": float(np.median(reference_times) / np.median(planned_times)),
              "exact_array_equal_for_every_grouping": exact,
              "baseline_densification_included": False}
    output = ROOT / "validation" / "sparse_quantile_microbenchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit("Exact comparison failed")


if __name__ == "__main__":
    main()
