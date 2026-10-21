# CellChatPy 2026-10-20 inference benchmark protocol

## Scope and baseline

The baseline is the unmodified GitHub commit
`986bf7ef909f1e15059aad8a9a9ee31ac675ffc3` under `upstream/`.
It is **not** the modified `CellChatPy_GitHub/` working tree outside this task.
The candidate is the implementation under `development/`.

The primary experiment measures `compute_communication_probability` with the
same expression values, cell groups, candidate interaction rows, triMean
definition, random seed, and bootstrap count. It compares every probability and
p-value entry, plus group and interaction ordering. The reference is the frozen
Python release; the pre-existing R agreement tutorials are tested separately.

## Real data

The complete 10x Genomics fresh 68k PBMC donor A data and official Cell Ranger
analysis are downloaded from the official `cf.10xgenomics.com` host. URLs, byte
counts and SHA-256 hashes are recorded in `data/pbmc68k_download_manifest.json`.
Original archives remain under `data/`.

`prepare-pbmc` reads the MatrixMarket coordinate stream in bounded chunks.
Library sizes use **all original genes**. Counts with duplicate gene symbols are
summed before normalization. Expression is `log1p(10000 * count / library_size)`.
Only database signaling genes are persisted as a sparse matrix. All original
barcodes remain; no cells are replicated. The official analysis contains labels
for only 40,000 of the 68,579 matrix barcodes. Those partial labels are retained
separately; they are not falsely represented as full-data annotations. Shared
benchmark groups for all cells are computed once with TruncatedSVD (30 dimensions)
and MiniBatchKMeans (10 clusters), both with random seed 42, and then saved.
These are computational benchmark groups, not biological annotations. Both
inference engines consume the exact same persisted labels.

For the inference-kernel comparison, candidate interactions are fixed by complete
ligand/receptor availability, including complex subunits, in database row order.
This deliberately avoids different DE screening outcomes between runs. It is not
a claim that every tested candidate is biologically active. The original tutorial
workflows provide the separate preprocessing-through-analysis tests.

## Synthetic data

Synthetic tests use independent count draws with group-specific expression
modules and explicitly artificial gene/cell metadata. This creates nonzero
triMean communication examples at globally sparse density. Synthetic tests are
scale and numerical regression tests; they are **not** real atlas validation and
are never created by duplicating PBMC cells.

## Timing, memory, resource limits

Each run has a fresh Python child process, with the implementation imported from
its explicit source directory. CPU numerical library threads are set equally.
Implementation order alternates between repeats. The inference timer starts
after imports and object preparation, immediately before the inference call.
It includes first-call Numba initialization or loading from the on-disk JIT cache.
This is not a pure warm-kernel measurement. Full process wall time is separately
reported; it includes imports, input reading, object setup, inference and output
serialization, but not the separate raw-data conversion/download.

RSS is sampled inside inference every 20 ms and across the complete child process
tree every 100 ms. On Windows, the latter is necessary because the virtual-env
launcher spawns the actual Python process. Start RSS, inference peak RSS, peak
increment, and complete process-tree peak are distinct metrics. Sampling can miss
very short allocations. Import overhead is substantial and is retained in the
process peak.

Each child has a wall-time, RSS, and Windows private-memory limit. Private memory
is monitored separately because committed allocations may be paged out and thus
not appear in the resident set. A timeout or resource stop is recorded
with its actual status and logs, not treated as a completed measurement. No speed
ratio is computed from unequal bootstrap counts, different inputs or incomplete
reference results. This host has about 16.9 GB physical RAM and 20 logical CPUs;
low free memory and swapping can distort absolute timings. Large tests are run
sequentially and coordinated with the tutorial runner.

## Reproduction

Run from the repository workspace with the task's environment:

```powershell
& 'cellchat10.20/.venv/Scripts/python.exe' 'cellchat10.20/scripts/benchmark_scaling.py' download
& 'cellchat10.20/.venv/Scripts/python.exe' 'cellchat10.20/scripts/benchmark_scaling.py' prepare-pbmc
& 'cellchat10.20/.venv/Scripts/python.exe' 'cellchat10.20/scripts/benchmark_scaling.py' suite --dataset pbmc68k --n-boot 100 --repeats 1 --timeout 900 --max-rss-gb 4 --max-private-gb 4
& 'cellchat10.20/.venv/Scripts/python.exe' 'cellchat10.20/scripts/benchmark_scaling.py' suite --dataset synthetic --cells 100000 --genes 256 --groups 10 --density 0.05 --pairs 100 --n-boot 100 --repeats 3 --threads 1 --timeout 900 --max-rss-gb 4 --max-private-gb 4 --tag synthetic100000_b100
& 'cellchat10.20/.venv/Scripts/python.exe' 'cellchat10.20/scripts/benchmark_scaling.py' suite --dataset synthetic --cells 1000000 --genes 256 --groups 10 --density 0.05 --pairs 100 --n-boot 100 --repeats 1 --threads 1 --timeout 900 --max-rss-gb 4 --max-private-gb 4 --tag synthetic1000000_b100
```

Individual run JSON, command, logs, probability arrays and comparison summaries
are written below `results/benchmarks/`. `source_sha256_at_start` records the
actual inference source files used by each worker. `profile_*.prof` are diagnostic
profiles, whose instrumented timings must not be used for headline speed ratios.
