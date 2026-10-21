# Warm-process ablation

`run_ablation.py` reuses the benchmark's independent synthetic-count generator
at 5,000 cells, 256 genes, 10 groups, 100 available LR candidates and 20 label
permutations. Object generation/copy, package imports and numerical comparison
are excluded from timing. Each variant is warmed once; three timed rounds
rotate the variant order. Raw results and source hashes are in `ablation.json`.

| Variant | Evaluated genes | Median inference seconds | Time / full engine |
| --- | ---: | ---: | ---: |
| All optimizations enabled | 59 | 0.154916 | 1.000 |
| Dependency pruning disabled | 256 | 0.132006 | 0.852 |
| Reusable sparse quantile plan disabled | 59 | 0.972433 | 6.277 |

All nine timed runs produced **exactly equal probability and P-value arrays**
relative to the fully enabled engine. The no-plan variant still uses the
batched vectorized NumPy summary fallback; it is not the upstream scalar code.

The reusable sparse plan has a clear runtime contribution in this fixture.
Dependency pruning evaluates fewer genes but did **not** improve runtime here:
its unpruned variant was about 15% faster. This small result must not be
rewritten as evidence that every optimization improves runtime, nor does it
establish a general ranking for larger data. Static planning/slicing overhead
and short-run variability can matter when total inference is below a second.

This was a warm single-process experiment, separate from the main isolated
fresh-process benchmark. A small Visium tutorial verification process was also
running on the host; these measurements were not collected under exclusive
CPU load. They are diagnostic ablations, not the headline end-to-end speedup
or a claim about real-data biological analysis.
