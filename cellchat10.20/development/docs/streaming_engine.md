# Streaming inference

The experimental `1.1.0+streaming.20261020` build accelerates **group-level**
communication inference while retaining the model implemented by upstream
commit `986bf7ef909f1e15059aad8a9a9ee31ac675ffc3`. It does not change the LR
database, averaging statistic, permutation count, or significance definition.

## Installation and use

From this source directory:

```bash
python -m pip install ".[accelerate]"
```

The `accelerate` extra enables Numba compilation of the sparse order-statistic
scan. The same scan has a pure Python fallback; large inputs will be slower
without the compiler. A first call can include compilation or cache loading.

The existing tutorials select the new engine by default:

```python
cellchat = cc.compute_communication_probability(
    cellchat,
    engine="streaming",
    n_boot=100,
    seed_use=1,
    gene_batch_size=64,
    lr_batch_size=64,
)
```

`engine="legacy"` runs the published implementation retained for diagnosis.
The new execution settings are available in `cellchat.settings["execution"]`.

## How execution changes

1. Normalize with the maximum over the complete signaling layer, then calculate
   only genes needed by the selected LR complexes and all relevant cofactors.
2. For sparse triMean/median input, sort each gene's nonzero expression once.
   Label permutations select ordered subsequences of that same sequence.
   Count implicit zeros and resolve the required ranks without densifying the
   expression matrix or sorting it again for every permutation.
3. Compile interaction and cofactor lookups once, reuse identical expressions,
   and calculate LR probabilities in bounded batches.
4. Generate one permutation at a time and retain exceedance counts, rather
   than retaining all permutations and their averaged expression matrices.

Dense inputs and truncated/thresholded means use bounded gene blocks. Their
scalar reduction order is retained where floating-point ties could matter.

The probability outputs still occupy space proportional to the number of LR
pairs times the square of the number of cell groups. This is not a
constant-memory algorithm, and backed AnnData/out-of-core input is not claimed.

## Compatibility

The engine preserves the original `count(P_boot - P_observed > 0) / n_boot`
definition, including zero P values and the original rule for zero observed
probabilities. There is no adaptive stopping or sampling of cells. The global
NumPy random state follows the same seeded permutation sequence as upstream.
External permutation matrices remain supported.

Regression checks load the frozen upstream code independently and exercise
complexes, all cofactor types, population effects, spatial/contact effects,
dense and sparse storage, missing/zero subunits, and strict permutation ties.
The seven original tutorial workflows and larger data experiments have separate
execution records in the enclosing research workspace; consult their actual
reports for tested datasets, dimensions, performance and limitations.

## Persistence and tutorial correction

Network dictionaries now retain an explicit `matrix_order`, so H5AD key sorting
does not misalign LR matrices and their metadata after reload. Old LR networks
can recover order from their interaction table. This fix does not change the
inference probabilities.

The Visium tutorial's manual aggregation check now removes `p >= 0.05`, matching
the package's existing strict `p < 0.05` rule. The previous notebook used
`p > 0.05` in that assertion. Analysis parameters are unchanged.

## Scope of performance claims

Inference-kernel timings exclude installation, imports, input preparation,
differential gene selection, plotting, and downstream network analyses unless
explicitly stated otherwise. Real-data and synthetic scale experiments must be
reported separately. The spot-level spatial inference implementation is not
accelerated by this group-level engine.
