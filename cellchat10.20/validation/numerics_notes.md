# Numerical contract for the streaming inference engine

The reference is the immutable GitHub snapshot at commit
`986bf7ef909f1e15059aad8a9a9ee31ac675ffc3` in `../upstream`, not another
working tree in the parent directory. The implementation in `../development`
must preserve this snapshot's calculation definitions. These checks assess
compatibility; they do not independently establish biological validity.

## Requirements

1. Expression is normalized by the maximum over the selected signaling layer,
   before group summaries. Excluding genes during LR compilation must not change
   this normalizing maximum. `raw_use=False` selects the smoothed layer.
2. Group order comes from the categorical levels, including any unused levels.
   Reordering groups or LR rows must not silently relabel matrix axes.
3. A bootstrap permutation changes labels only. Shuffling expression columns and
   labels by the same order cancels the permutation. The independent upstream
   function is the oracle; a previously modified local checkout is not.
4. Every externally supplied permutation column is zero-based and contains each
   cell index exactly once. A single RNG `RandomState(seed)` sequence should match
   upstream `np.random.seed(seed)` followed by repeated `np.random.permutation`.
   Batch sizes must not change the draw sequence.
5. Tukey triMean uses linear NumPy quantiles: `(Q1 + 2 * Q2 + Q3) / 4`.
   For optimized order-statistic selection, reproduce NumPy's floating-point
   interpolation branch (the high endpoint expression for weight >= 0.5).
   Numerically close quantiles are insufficient at exact permutation ties.
6. The significance rule remains `count(Pboot > Pobs) / B`. No epsilon,
   `>=`, plus-one correction, or tolerance-based tie merging is allowed in an
   implementation claiming compatibility. Zero observed probabilities have
   their P values forced to one by the reference.
7. Complex expression is the log-space geometric mean of *available* subunits,
   in database column order. An available zero subunit collapses the geometric
   mean; a missing subunit is ignored by this upstream definition. Changing this
   rule to a stricter biological model belongs in a separately named method.
8. Coreceptor activation and inhibition multiply factors `(1 + expression)`;
   inhibition is divided after activation. Agonist and antagonist products,
   population fractions, Hill exponents, and spatial/contact factors retain
   their reference operation ordering where rounding can affect P values.
9. Sparse implicit zeros and stored explicit zeros have the same expression
   semantics. Duplicated sparse indices must be summed before quantile counting.
   Sorting/canonicalization must not mutate the user's matrix.
10. NaN handling is method-specific in the upstream scalar functions. Negative
    values may be useful for helper regression but are not valid normalized
    expression inputs. Supporting or rejecting such public inputs must be
    explicit; silently discarding negative/NaN sparse entries is forbidden.
11. A result comparison checks probabilities with a tight floating-point
    tolerance and P values by exact array equality. The test fixture includes
    identity permutations and constant-valued ties so comparisons cannot pass
    solely because there happen to be no near-ties.

## Memory expectations

One permutation and one genes-by-groups summary are sufficient at a time.
Do not allocate an internally generated cells-by-bootstrap index matrix, a
bootstrap-by-genes-by-groups tensor, or a bootstrap-by-LR-by-groups-squared
tensor. External permutation matrices already belong to the caller and are
accepted for reproducibility, but must not be duplicated wholesale.

Bounded densification of a gene batch is retained as the fallback for mean-based
methods and dense input. It avoids an all-gene sparse-to-dense conversion during
group reduction but still costs `O(batch_genes * cells)` scratch memory. Dense
inputs still require full normalization storage, and extraction/normalization
of all selected signaling genes occurs before pruning. The sparse triMean/median path
instead uses the reusable exact order-statistic plan described below.
Output storage still scales with `LR * groups**2`; a low-memory bootstrap
engine cannot remove this inherent requested output cost.

## Test organization

`test_streaming_engine.py` imports the frozen upstream modeling module under a
separate package name and compares it with the development package. It includes
same-external-permutation tests, fixed-seed tests, dense/CSR inputs, batch size
invariance, multi-subunit complexes, all four cofactor modes, population factors,
spatial/contact factors, helper edge cases, and downstream network aggregation.

Tutorial and large-data runs are separate evidence. A unit-test pass alone is
not a claim that every online tutorial, R version, or biological dataset has
been verified.

## Implemented sparse order-statistic plan

`development/CellChatPy/_sparse_quantiles.py` sorts stored expression values once
per gene and retains each value's cell index. Because label permutations do not
alter expression values, each group's sorted observations are a subsequence
of this immutable order. There is no reason to sort them again per permutation.

For a group containing `m` cells, with `s` stored entries and `a` stored NaNs,
the implicit zero count is `m-s` and the effective sample size is `m-a`.
For the `j`th non-NaN stored value in that group's ordered subsequence, its
virtual dense rank is `j` if negative and `j+(m-s)` if positive. Six selected
virtual ranks suffice to obtain the lower and upper endpoints of the 25th,
50th and 75th percentiles. Endpoints inside the virtual zero interval are zero.
NumPy's interpolation branch and arithmetic order are then applied exactly.

The optional Numba kernel uses no fast-math. It executes the same algorithm as
the Python fallback. After one-time per-gene sorting, each group assignment
requires `O(nnz + genes*groups + cells)` work, without per-group sorting or
expression densification. Persistent plan storage is `O(nnz+genes)`; output
and per-call state are not constant-memory. JIT compilation/cache loading must
be distinguished from warm execution in timing claims.

The CSR normalization path explicitly divides `matrix.data` by the maximum.
SciPy's `matrix / scalar` can multiply by a reciprocal instead, producing
different last bits from dense NumPy division. The regression suite checks the
actual normalized input to the group-summary engine with exact equality.

## Executed validation

The final combined run of release and new regression tests completed with
**303 passed, 0 skipped, 0 failed** (JUnit: `combined_regression.xml`). This
consists of 97 release tests, 93 streaming/persistence/edge tests and 113 sparse
quantile tests. Two development-copy release tests were adapted: the historical
`accelerate` keyword probe now exercises `engine='legacy'/'streaming'` without
skipping, and the version check compares the package version against project
metadata rather than a hard-coded previous release. The frozen upstream test
files remain unchanged.

The final 14 tests and two focused fixes followed the independent review in
`final_independent_review.md`: missing group labels are ignored for group
summaries without deleting cells from the permutation universe, and external
floating-point permutation indices are validated/converted one column at a
time. Unknown nonmissing group labels still raise an error.

The tests include strict near-ties (differences around `1e-13`), fixed-seed RNG
state after inference, an unused signaling gene that determines the global
normalization maximum, explicit empty LR input, malformed external permutations,
and an allocation guard that rejects sparse triMean densification and known
full-bootstrap tensor shapes. NaN warnings in adversarial tests are expected
from the reference reducers. Existing AnnData and pandas deprecation warnings
do not indicate changed inference outputs.

`sparse_quantile_microbenchmark.json` records an exploratory earlier-prototype
synthetic primitive benchmark, preceding the final missing-label fix:
20,000 cells, 128 genes, 12 groups, 35% stored values and six group assignments.
Every result is exactly equal to the scalar upstream triMean. On this run,
warm primitive median speedup was about **46.1x**, with ~0.098 s plan setup and
~0.047 s first call including cache loading. Baseline densification time was
excluded. This is a primitive-only synthetic measurement, **not** an end-to-end
speedup, a million-cell measurement, or a cross-tool performance ranking.
