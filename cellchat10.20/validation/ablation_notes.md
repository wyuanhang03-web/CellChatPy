# Warm-process ablation

`run_ablation.py` reuses the benchmark's independent synthetic-count generator
at 5,000 cells, 256 genes, 10 groups, 100 available LR candidates and 20 label
permutations. Object generation/copy, package imports and numerical comparison
are excluded from timing. Each variant is warmed once; three timed rounds
rotate the variant order. Raw results and source hashes are in `ablation.json`.

| Variant | Evaluated genes | Median inference seconds | Time / full engine |
| --- | ---: | ---: | ---: |
| All optimizations enabled | 59 | 0.227188 | 1.000 |
| Dependency pruning disabled | 256 | 0.203059 | 0.894 |
| Reusable sparse quantile plan disabled | 59 | 1.488311 | 6.551 |

All nine timed runs produced **exactly equal probability and P-value arrays**
relative to the fully enabled engine. The no-plan variant still uses the
batched vectorized NumPy summary fallback; it is not the upstream scalar code.

The reusable sparse plan has a clear runtime contribution in this fixture.
Dependency pruning evaluates fewer genes but did **not** improve runtime here:
its unpruned variant was about 11% faster. This small result must not be
rewritten as evidence that every optimization improves runtime, nor does it
establish a general ranking for larger data. Static planning/slicing overhead
and short-run variability can matter when total inference is below a second.

This final-source rerun was a warm single-process experiment, separate from
the main fresh-process benchmark. Other task-controlled tutorial and benchmark
workers were paused during this rerun; unrelated host activity was not controlled.
The previous prototype JSON and notes are preserved under `validation/history`.
These are diagnostic ablations, not the headline end-to-end speedup or a claim
about real-data biological analysis.
