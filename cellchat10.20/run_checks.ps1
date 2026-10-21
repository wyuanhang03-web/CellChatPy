param([switch]$Tutorials, [switch]$LargeBenchmarks)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install provenance/requirements-tested.txt before running checks.'
}
$env:PYTHONPATH = Join-Path $taskRoot 'development'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $taskRoot 'cache/mpl'
$env:NUMBA_CACHE_DIR = Join-Path $taskRoot 'cache/numba'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Push-Location $taskRoot
try {
    & $taskPython -B bootstrap_source.py
    if ($LASTEXITCODE -ne 0) { throw 'Frozen baseline verification failed.' }
    & $taskPython -B scripts/run_regression.py
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }
    if ($Tutorials) {
        & $taskPython -B scripts/run_tutorial_pairs.py
        if ($LASTEXITCODE -ne 0) { throw 'Tutorial regression failed. Inspect the saved execution reports.' }
    }
    if ($LargeBenchmarks) {
        & $taskPython -B scripts/benchmark_scaling.py download
        if ($LASTEXITCODE -ne 0) { throw 'Dataset download failed.' }
        & $taskPython -B scripts/benchmark_scaling.py prepare-pbmc
        if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
        & $taskPython -B scripts/benchmark_scaling.py suite --dataset pbmc68k --n-boot 100 --repeats 3 --threads 1 --timeout 900 --max-rss-gb 4 --max-private-gb 4 --tag pbmc68k_b100_reproduced
        if ($LASTEXITCODE -ne 0) { throw 'Benchmark runner failed. Check per-engine completion statuses.' }
    }
}
finally {
    Pop-Location
}
