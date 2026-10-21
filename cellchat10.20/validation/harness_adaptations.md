# Tutorial execution adaptations

All seven published Read the Docs notebook sources were fetched and compared
against the fixed GitHub snapshot. Their code cells are identical; see
`tutorial_inventory.json` and `published_*.ipynb`.

The harness executes every original code cell with its original inference
parameters. The only scientific-code edit is the explicitly recorded Visium
manual-check correction described below; all other original cell sources are
unchanged. A setup cell explicitly imports the selected
upstream/development package, records inference call parameters and numerical
outputs, and verifies the import path. A completion cell checks finite network
values and saves final arrays. Original cached outputs are cleared. Only raw
input data and the original R permutation matrix are copied; no previous
inference cache is loaded.

Execution directories use short labels (`01_human_skin`, ..., `07_psoriasis`)
and a shared task-local cache because the first attempt with full notebook
names exceeded Windows' path limit while Numba imported pynndescent. This first
attempt failed before any original tutorial code ran. Its evidence remains in
the full-name directories under `results/tutorials/baseline`; those directories
are not counted as completed or numerically tested tutorials.

A second harness-only issue was corrected after successful Visium inference:
the evidence exporter must call `toarray()` for sparse probability matrices,
rather than passing the sparse object directly to `numpy.asarray`. The aborted
`baseline` evidence does not indicate a CellChat inference error. Formal
reference executions are under `results/tutorials/reference`.

The unmodified published Visium tutorial failed its own aggregation assertion:
its manual check cleared `pval > 0.05`, while the fixed upstream package's
`aggregate_network` explicitly clears `pval >= 0.05` (strict `p < 0.05`). The
manual check is corrected to `>=` in both development notebook copies. The
runner applies exactly the same one-line overlay when testing either upstream
or development. `tutorial_compatibility_patches.json` records the original and
adapted cell SHA-256 hashes, literal before/after text, and rationale. Inference
parameters, results, filters and the assertion itself are not removed or
weakened. The originally failing run is archived when `--resume` starts its
fresh rerun; `--no-tutorial-fixes` reproduces the original published behavior.

The published-source comparison in `tutorial_inventory.json` remains a
comparison of the unmodified Read the Docs sources against the unmodified
GitHub snapshot. Execution summaries separately identify applied overlays.

`--resume` reuses only already completed notebook executions produced in this
task; it never loads their computed objects into another execution. Failed or
interrupted directories are archived and rerun from raw inputs. A `PAUSE`
marker under `results/tutorials` delays starting the next tutorial to provide
exclusive memory windows for the large-data benchmarks.

The Jupyter kernel is explicitly configured with the Python interpreter used
to invoke the runner. Python bytecode writes are disabled. Matplotlib, IPython,
Jupyter, Numba and temporary directories are redirected under this task's
`results/tutorials/cache`. The first aborted attempt briefly used the normal
temporary directory for Jupyter's connection file, which Jupyter cleans up.

The requested CPU thread limit is recorded in each summary and is applied
consistently to both engines. It does not change scientific parameters.

The notebook evidence exporter records group `network` probability, P-value,
count, and weight arrays when those fields exist. An empty network snapshot
contains metadata only and is explicitly marked as such in the comparison
report; it is never evidence of numerical agreement. The three spot-based
Visium, multi-sample and multiomics notebooks leave their original
`export_results=False`, so their pathway computations are covered by complete
execution and figure comparison, not by a separate saved pathway-array audit.

The other four notebooks save fresh CellChat objects in their original final
cells. `compare_tutorial_saved_objects.py` supplements their original evidence
by inspecting these newly saved objects after notebook execution, without
recomputing an analysis or reading a historical cache. Reference and candidate
pickles are loaded in separate processes with their respective source package.
All actual fields of `network`, `pathway_network`, `spot_network`, and
`spot_pathway_network` are exported, including numerical centrality fields
where present; merged-object `network_similarity` is also included. Mixed
DataFrames are exported column by column so numeric contributions retain the
floating comparison rule while categorical labels, category order, dtype,
and ordered flags remain explicit metadata. Sparse matrices retain their native format, shape, data and
index arrays and are never converted to dense spot-by-spot matrices. Numeric
arrays are compared in bounded chunks via memory-mapped NPY files, with exact
P-values and integer/index arrays and a 1e-12 tolerance for other floating
arrays. Slot fields that do not exist (in particular spot-level P-values) are
not claimed as tested. The full seven-notebook comparison additionally requires
all four supplemental audits and nonempty psoriasis spot-probability evidence.
SHA-256 checks tie each supplement to the specific completed execution summary.
The final comparison also rechecks each saved pickle's SHA-256. The
development-only `matrix_order` storage field is validated against actual
probability keys, P-value keys and group/spot matrix dimensions before both
versions are represented by a common semantic axis. Explicit versus legacy
axis provenance is retained in the supplemental report; a real mismatch fails.
Buffered numerical comparison follows logical C-order coordinates even when
the two arrays have different physical C/F storage. A small self-check covers
mixed-order equal arrays, a wrong logical arrangement with identical physical
bytes, and strict P-value equality.

The pair driver runs applicable saved-object supplements automatically after
all requested original notebooks finish and before the final comparison.
`--only` restricts the supplemental scope to applicable selected tutorials.
On a resumed driver run, the previous driver status is archived separately;
completed notebook evidence is reused only after the original resume checks.

Final figure evidence: 147 of 148 paired PNG files have identical SHA-256
hashes. The NL/LS comparison bubble plot has a source-inherited nondeterministic
column order caused by sorting a set with a key that omits the dataset tie
break. Its points and dataset labels move together; the plotting source is
identical in both versions. See `figure_difference_notes.md` for the exact
code evidence and the existing separator-line limitation. This is not a claim
that all 148 PNG files are byte-identical.
