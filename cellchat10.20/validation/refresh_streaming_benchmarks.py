"""Rerun only streaming workers after a source fix; retain all upstream runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_scaling import compare_arrays, write_json


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rerun(old, output, limits):
    command = list(old["command"])
    command[0] = sys.executable
    command_output = Path(command[command.index("--output") + 1]).resolve()
    if command_output != output.resolve() or not output.resolve().is_relative_to(ROOT):
        raise ValueError("Worker output must stay in its original task result directory")
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(old.get("threads", 1))
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    peak_rss, peak_private, reason, next_progress = 0, 0, None, 60.0
    timeout = limits.get("timeout_seconds", 900)
    max_rss = limits.get("max_rss_gb", 4) * 2**30
    max_private = limits.get("max_private_gb", limits.get("max_rss_gb", 4)) * 2**30
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=environment)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                children = monitor.children(recursive=True)
                memories = [p.memory_info() for p in [monitor, *children] if p.is_running()]
                peak_rss = max(peak_rss, sum(m.rss for m in memories))
                peak_private = max(peak_private, sum(getattr(m, "private", m.rss) for m in memories))
            except psutil.NoSuchProcess:
                break
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                reason = "timeout"
            if peak_rss > max_rss:
                reason = "memory_budget_exceeded"
            if peak_private > max_private:
                reason = "private_memory_budget_exceeded"
            if reason:
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                break
            if elapsed >= next_progress:
                print(f"{output.parent.name}: refresh {output.stem} running {elapsed:.0f}s", flush=True)
                next_progress += 60
            time.sleep(0.1)
        process.wait()
    result = json.loads(output.read_text(encoding="utf-8")) if process.returncode == 0 else {}
    result.update(engine="streaming", repeat=old["repeat"], command=command,
                  returncode=process.returncode, process_seconds=time.monotonic() - started,
                  process_peak_rss_bytes=peak_rss, process_peak_private_bytes=peak_private,
                  threads=old.get("threads", 1),
                  status=reason or ("completed" if process.returncode == 0 else "failed"))
    write_json(output, result)
    print(f"{output.parent.name}: refresh {output.stem} {result['status']} "
          f"process={result['process_seconds']:.2f}s", flush=True)
    return result


def refresh(tag, reason):
    folder = (ROOT / "results/benchmarks" / tag).resolve()
    if folder.parent != (ROOT / "results/benchmarks").resolve():
        raise ValueError("Tag must identify one benchmark result directory")
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = folder / "history" / f"streaming_before_missing_label_fix_{stamp}"
    history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(summary_path, history / "summary.json")
    preserved = {}
    for record in summary["records"]:
        for suffix in (".json", ".log", ".npz", ".prof"):
            path = folder / f"{record['engine']}_{record['repeat']}{suffix}"
            if path.exists():
                if record["engine"] == "streaming":
                    shutil.copy2(path, history / path.name)
                elif record["engine"] == "upstream":
                    preserved[path.name] = file_hash(path)
    for index, record in enumerate(summary["records"]):
        if record["engine"] == "streaming":
            summary["records"][index] = rerun(
                record, folder / f"streaming_{record['repeat']}.json", summary["limits"])
    comparisons = []
    repeats = sorted({record["repeat"] for record in summary["records"]})
    for repeat in repeats:
        pair = {record["engine"]: record for record in summary["records"] if record["repeat"] == repeat}
        if all(pair.get(engine, {}).get("status") == "completed" for engine in ("upstream", "streaming")):
            comparisons.append({"repeat": repeat, **compare_arrays(
                folder / f"upstream_{repeat}.npz", folder / f"streaming_{repeat}.npz")})
    for name, expected in preserved.items():
        if file_hash(folder / name) != expected:
            raise RuntimeError(f"Preserved upstream output changed: {name}")
    summary["comparisons"] = comparisons
    summary["all_comparisons_pass"] = len(comparisons) == len(repeats) and all(row["pass"] for row in comparisons)
    summary.setdefault("refresh_history", []).append({
        "completed_utc": datetime.now(timezone.utc).isoformat(), "reason": reason,
        "archived_previous_streaming_outputs": str(history.relative_to(ROOT)),
        "upstream_policy": "retained original completed workers; not rerun",
        "retained_upstream_sha256": preserved,
        "acceptance_rule": "aligned axes; probability allclose at 1e-12; P values exactly equal",
    })
    write_json(summary_path, summary)
    print(f"{tag}: refreshed comparison pass={summary['all_comparisons_pass']}", flush=True)
    return summary["all_comparisons_pass"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="append")
    parser.add_argument("--reason", default="Final missing-label compatibility and bounded external-permutation validation fixes")
    args = parser.parse_args()
    tags = args.tag or ["pbmc68579_b10", "pbmc68579_b100", "synthetic5000_b20", "synthetic100000_b100"]
    outcomes = [refresh(tag, args.reason) for tag in tags]
    if not all(outcomes):
        raise SystemExit("At least one refreshed benchmark pair did not pass")


if __name__ == "__main__":
    main()
