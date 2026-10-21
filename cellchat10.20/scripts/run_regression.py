"""Run the regression suite and bind its evidence to source and test hashes."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    paths = set()
    for package in ("upstream", "development"):
        paths.update(p for p in (ROOT / package / "CellChatPy").rglob("*")
                     if p.is_file() and p.suffix in {".py", ".csv", ".mtx"})
    paths.update((ROOT / "development/tests").rglob("*.py"))
    paths.update((ROOT / "validation").glob("test_*.py"))
    paths.add(Path(__file__).resolve())
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def main():
    before = snapshot()
    command = [sys.executable, "-B", "-m", "pytest", "development/tests",
               "validation/test_streaming_engine.py", "validation/test_sparse_quantiles.py",
               "-q", "-o", f"cache_dir={ROOT / 'cache/pytest'}",
               f"--basetemp={ROOT / 'cache/pytest_run'}",
               f"--junitxml={ROOT / 'validation/combined_regression.xml'}"]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    after = snapshot()
    junit = ROOT / "validation/combined_regression.xml"
    report = {"passed": result.returncode == 0 and before == after,
              "returncode": result.returncode, "started_utc": started,
              "elapsed_seconds": time.perf_counter() - clock, "command": command,
              "source_unchanged_during_execution": before == after,
              "source_sha256": before, "source_sha256_after": after,
              "junit_sha256": sha(junit) if junit.is_file() else None}
    (ROOT / "validation/regression_provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
