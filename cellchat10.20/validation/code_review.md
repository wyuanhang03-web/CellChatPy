# Review of the streaming inference implementation

Reviewed against frozen GitHub commit
`986bf7ef909f1e15059aad8a9a9ee31ac675ffc3`, with an independent import of its
modeling module as the numerical oracle. The reviewer implemented the sparse
quantile module and tests, so this is an independent review of the main
streaming/modeling/storage integration, not an independent authorship review
of `_sparse_quantiles.py` itself.

## Findings and resolutions

1. **Resolved: sparse normalization rounding.** SciPy sparse division by a
   scalar can multiply by a reciprocal, while the frozen implementation divides
   dense entries. This changes low bits and could change a strict permutation
   exceedance at a tie. The streaming branch now explicitly divides its copied
   CSR data buffer by the global maximum. A test captures the normalized summary
   input and requires exact equality to dense NumPy division.
2. **Resolved: empty explicit LR input.** Compiled empty entity index arrays
   previously defaulted to floating dtype. They now use integer indices. Empty
   explicit LR tables and nonempty LR tables with zero measured dependencies
   both match the frozen reference.
3. **Resolved: H5AD network-axis reordering.** An actual inferred network lost
   dictionary insertion order on H5AD reload; its LR metadata retained original
   order. This pre-existing storage issue could misassociate stacked values and
   LR names. The normalizer now stores an explicit `matrix_order`, restores old
   LR order from interaction metadata where possible, and reorders matching
   dictionaries. Tests persist real inferred results, reversed LR order,
   duplicate LR names with suffixes, and deliberately reversed multi-pathway
   order. They also recompute downstream pathways after reload. Spot-network
   normalization recovers order from its interaction metadata.

The initial review found no further blocker within its tested inputs. A later
[independent final review](final_independent_review.md) identified missing-group
label handling and whole-matrix validation of floating external permutations;
both were fixed before the final source freeze. Missing-label cells stay in the
permutation universe and population denominator but do not contribute to a
group summary. Validation now works one permutation column at a time. These
conclusions are bounded by the recorded tests, not a claim about every input.

## Numerical and behavioral review

- **Randomness:** `np.random.seed(seed_use)` and one global
  `np.random.permutation(n_cells)` per replicate preserve both the permutations
  and post-inference global RNG state of the reference. Externally supplied
  permutations are checked and infer their own replicate count. Streaming does
  not shuffle expression and labels together. This intentionally retains the
  upstream global-RNG side effect; simultaneous threaded calls remain outside
  the reproducibility guarantee.
- **Model arithmetic:** LR gene/complex lookup precedence, available-subunit
  geometric means, zero-subunit behavior, coactivation/division order, agonist
  and antagonist products, Hill factors, and population fractions match the
  source. Unknown average methods retain its triMean fallback. The strict rule
  remains `(Pboot - Pobs) > 0`, with no tolerance or plus-one correction.
- **Normalization and pruning:** The global maximum is computed before static
  dependency pruning. A selected but unused signaling gene can determine that
  maximum. Signaling masks and the unmasked smoothed-layer behavior are tested
  separately. Pruning keeps dependency order consistent with the original gene
  list and does not change selected database rows.
- **Sparse summaries:** The immutable expression order is reused across label
  assignments. Implicit zeros, explicit zeros, duplicate indices, negative
  values, NaNs, infinities, small-group quantile boundaries, and the distinct
  NumPy median arithmetic have exact tests. Tests exercise both the Python
  kernel and optional Numba compilation without fast-math. Negative/NaN helper
  tests establish compatibility, not recommended biological inputs.
- **Dense fallback:** triMean/median use bounded contiguous gene batches;
  truncated and thresholded means retain scalar reduction ordering. Fixed
  external permutations and fixed-seed tests cover dense and sparse input.
- **Space and contact:** Spatial distance construction remains in the original
  path. Shared spatial weights, row-specific contact masks, forced contact,
  distance enabled/disabled, and multiplication order are checked against the
  frozen implementation. This is a group-level acceleration; spot inference
  and preprocessing were not rewritten as part of the numerical engine.
- **Empty cases:** An unused categorical group behaves like the source for all
  four supported means. A truly empty signaling selection is rejected by both
  paths. An explicit empty LR table returns an empty network; LR dependencies
  absent from an otherwise valid expression layer produce zero probabilities.
- **Mutation and persistence:** Source expression and database tables remain
  unchanged. Network values and execution metadata survive H5AD round trips.
  The old `accelerate`-keyword probe in the release test was updated to test
  the actual `engine='legacy'/'streaming'` interface instead of skipping it.

## Limits to state in a paper or release

- Memory is bounded with respect to the number of bootstrap replicates, not
  constant overall: the normalized sparse matrix and sorted plan are `O(nnz)`,
  gene/group summaries are `O(genes*groups)`, and requested network outputs are
  `O(LR*groups**2)`. Caller-supplied permutation matrices still occupy memory.
- Numba compilation/cache loading and warm kernel timings are distinct. The
  recorded approximately 46x synthetic primitive microbenchmark is not an
  end-to-end speed claim. Main performance claims should use the independent
  benchmark process and real-data runs.
- Original pathway files lacking both saved axis order and suitable metadata
  cannot have an arbitrary lost historical order reconstructed. New files store
  it. Legacy LR files can generally recover order from their interaction table.
- The frozen Python implementation is the direct automated oracle. Existing
  R-concordance tutorials provide separate evidence; this review does not
  independently rerun every R version or validate biological causality.

Evidence is recorded in `combined_regression.xml`, `storage_regression.xml`,
`engine_dispatch_regression.xml`, `test_streaming_engine.py`, and
`test_sparse_quantiles.py`. The earlier `review_edge_regression.xml` intentionally
retains the persistence failure that motivated the subsequent fix; the later
storage and combined reports supersede it.

The earlier combined suite passed **289 tests with no failures or skips** in
13.62 seconds. After the independent review's two compatibility/memory fixes
and 14 targeted tests, the final combined suite passed **303 tests with no
failures or skips** in 11.17 seconds. These times exclude process startup/import
before collection. The project version test compares against `pyproject.toml`,
matching the local development version while checking packaging consistency.
