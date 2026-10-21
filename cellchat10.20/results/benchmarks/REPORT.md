# Inference benchmark results

Only completed equal-parameter runs receive a speed ratio. Times are medians across recorded repeats.
Inference includes first-call JIT initialization/cache loading; process wall time also includes imports and setup.
Real PBMC and synthetic scale experiments are separate. Synthetic cells are not a biological atlas.

| Experiment | Cells | Genes | LR pairs | Bootstraps | Upstream inference s | Streaming inference s | Kernel speed ratio | Upstream / streaming process s | Upstream / streaming peak RSS MiB | Exact prob+pval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pbmc68579_b10 | 68,579 | 1,404 | 3,072 | 10 | 95.589 | 5.759 | 16.599 | 124.609 / 29.734 | 3297.758 / 789.430 | yes |
| pbmc68579_b100 | 68,579 | 1,404 | 3,072 | 100 | 678.768 | 8.260 | 82.175 | 710.250 / 32.313 | 3219.996 / 779.227 | yes |
| synthetic1000000_b100 | 1,000,000 | 256 | 100 | 100 | — | 29.903 | — | — / 60.719 | — / 1543.160 | not established |
| synthetic100000_b100 | 100,000 | 256 | 100 | 100 | 180.537 | 1.574 | 114.674 | 204.297 / 24.515 | 1787.887 / 765.461 | yes |
| synthetic5000_b20 | 5,000 | 256 | 100 | 20 | 14.649 | 0.233 | 62.936 | 39.625 / 23.234 | 747.602 / 697.180 | yes |

## Completion and limitations

- pbmc68579_b10: completed repeats {'upstream': 1, 'streaming': 1}; all requested workers completed. [raw summary](pbmc68579_b10/summary.json)
- pbmc68579_b100: completed repeats {'upstream': 1, 'streaming': 1}; all requested workers completed. [raw summary](pbmc68579_b100/summary.json)
- synthetic1000000_b100: completed repeats {'upstream': 0, 'streaming': 1}; upstream repeat 0: private_memory_budget_exceeded [raw summary](synthetic1000000_b100/summary.json)
- synthetic100000_b100: completed repeats {'upstream': 3, 'streaming': 3}; all requested workers completed. [raw summary](synthetic100000_b100/summary.json)
- synthetic5000_b20: completed repeats {'upstream': 3, 'streaming': 3}; all requested workers completed. [raw summary](synthetic5000_b20/summary.json)

Peak process-tree RSS is sampled at 100 ms. Per-inference RSS is separately available in worker JSON.
This host has limited free memory; paging and background activity may affect absolute time. Paging I/O was not measured.
The early synthetic5000 upstream pilot predates full source-hash recording. Every currently reported streaming worker records the final four source file hashes and package version.
Raw data provenance and full-cell clustering are documented in ../../data/pbmc68k_signaling/manifest.json.
Candidate interactions in these kernel benchmarks use complete ligand/receptor availability, without DE screening.
Original tutorial workflows are validated separately; do not treat kernel speed ratios as end-to-end pipeline ratios.

## Final-source refresh and million-cell boundary

After the final missing-label and bounded permutation-validation fixes, eight streaming workers were rerun: PBMC with 10 and 100 permutations, synthetic 5,000 cells with three repeats, and synthetic 100,000 cells with three repeats. Previous streaming outputs and summaries are retained in each case's `history/` directory. Completed upstream reference outputs were retained unchanged; the refresh history records their SHA256 values. All eight comparisons have identical probability and P-value arrays. The acceptance rule requires aligned names/groups, probabilities within 1e-12, and exactly equal P values.

The million-cell experiment is a synthetic feasibility test with 256 input genes, 100 LR pairs, 10 groups and 100 permutations, with a 900-second timeout and 4 GiB RSS/private-memory budgets. The final streaming engine completed inference in 29.903 seconds (60.719 seconds for the full process), with sampled peak process RSS 1,543.160 MiB and private memory 1,873.898 MiB. The upstream process exceeded the 4 GiB private-memory budget and was stopped after 58.937 seconds; it did not produce a completed reference result. This case therefore establishes neither paired numerical equivalence nor a speed ratio. It does not establish performance on a real million-cell atlas or with all LR pairs.

The lightweight [final source audit](../../validation/final_benchmark_source_audit.json) verifies that all nine reported streaming workers and the final ablation use the current four core source files, and that all 24 retained upstream reference artifacts remain unchanged. Memory samples are observed peaks rather than allocation proofs; private memory and RSS describe different quantities.
