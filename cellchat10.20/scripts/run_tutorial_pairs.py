"""Run reference/development tutorials serially, one pair at a time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from run_tutorials import NAMES, SHORT_NAMES, TASK_ROOT, write_json
from compare_tutorial_saved_objects import OBJECTS as SAVED_OBJECT_TUTORIALS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=NAMES)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Accepted for clarity; this driver always resumes verified completed executions.")
    args = parser.parse_args()
    order = args.only or [NAMES[3], NAMES[4], NAMES[5], NAMES[6], NAMES[0], NAMES[1], NAMES[2]]
    progress_path = TASK_ROOT / "results" / "tutorials" / "paired_progress.json"
    history = None
    history_reason = None
    if progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history = TASK_ROOT / "results" / "tutorials" / "attempts" / f"paired_progress_{stamp}.json"
        write_json(history, previous)
        if (previous.get("status") == "failed" and previous.get("comparison_returncode")
                and all(row.get("returncode") == 0 for row in previous.get("completed", []))
                and "supplementary_audit" not in previous):
            history_reason = "Earlier driver completed the original cells before mandatory saved-object supplemental auditing was added; its final comparison was pending that supplement."
        else:
            history_reason = "Previous driver status retained; completed notebook execution summaries are independently verified before reuse."
    results = []
    pair_comparisons = []
    for name in order:
        for label, package in (("reference", "upstream"), ("development", "development")):
            command = [sys.executable, "-B", str(TASK_ROOT / "scripts" / "run_tutorials.py"),
                       "--only", name, "--label", label, "--package-root", str(TASK_ROOT / package),
                       "--threads", str(args.threads), "--resume"]
            progress = {"status": "running", "name": name, "engine_label": label, "completed": results}
            write_json(TASK_ROOT / "results" / "tutorials" / "paired_progress.json", progress)
            print(json.dumps({"pair_start": name, "engine_label": label}), flush=True)
            started = time.perf_counter()
            outcome = subprocess.run(command, cwd=TASK_ROOT, check=False)
            results.append({"name": name, "label": label, "returncode": outcome.returncode,
                            "wall_seconds": time.perf_counter() - started})
            write_json(TASK_ROOT / "results" / "tutorials" / "paired_progress.json",
                       {"status": "between_executions", "completed": results})
        pair_output = TASK_ROOT / "validation" / ("tutorial_comparison_" + SHORT_NAMES[name] + ".json")
        pair_check = subprocess.run([sys.executable, "-B", str(TASK_ROOT / "scripts" / "compare_tutorial_results.py"),
                                     "--only", name, "--output", str(pair_output)], cwd=TASK_ROOT, check=False)
        pair_comparisons.append({"name": name, "returncode": pair_check.returncode, "report": str(pair_output)})
        write_json(TASK_ROOT / "results" / "tutorials" / "paired_progress.json",
                   {"status": "between_pairs", "completed": results, "pair_comparisons": pair_comparisons})
    supplemental_scope = [SHORT_NAMES[name] for name in order if SHORT_NAMES[name] in SAVED_OBJECT_TUTORIALS]
    supplement = {"status": "not_applicable", "returncode": 0, "requested": supplemental_scope}
    if supplemental_scope and all(row["returncode"] == 0 for row in results):
        supplement_command = [sys.executable, "-B", str(TASK_ROOT / "scripts" / "compare_tutorial_saved_objects.py")]
        if args.only:
            supplement_command.extend(["--only", *supplemental_scope])
        write_json(progress_path, {"status": "running_saved_object_audit", "completed": results,
                                   "pair_comparisons": pair_comparisons, "requested": supplemental_scope})
        started = time.perf_counter()
        supplemental_result = subprocess.run(supplement_command, cwd=TASK_ROOT, check=False)
        supplement.update(status="passed" if supplemental_result.returncode == 0 else "failed",
                          returncode=supplemental_result.returncode, wall_seconds=time.perf_counter() - started)
    elif supplemental_scope:
        supplement.update(status="blocked_by_notebook_failure", returncode=1)
    # Refresh each applicable per-pair report after the supplemental stage so
    # it cannot retain an earlier "pending" status beside a final suite pass.
    for index, name in enumerate(order):
        if SHORT_NAMES[name] not in supplemental_scope:
            continue
        pair_output = TASK_ROOT / "validation" / ("tutorial_comparison_" + SHORT_NAMES[name] + ".json")
        final_pair = subprocess.run([sys.executable, "-B", str(TASK_ROOT / "scripts" / "compare_tutorial_results.py"),
                                     "--only", name, "--require-saved-objects", "--output", str(pair_output)],
                                    cwd=TASK_ROOT, check=False)
        pair_comparisons[index] = {"name": name, "returncode": final_pair.returncode, "report": str(pair_output),
                                   "saved_object_audit_required": True}
    comparison_command = [sys.executable, "-B", str(TASK_ROOT / "scripts" / "compare_tutorial_results.py"),
                          "--require-saved-objects"]
    if args.only:
        comparison_command.extend(["--only", *args.only])
    comparison = subprocess.run(comparison_command, cwd=TASK_ROOT, check=False)
    passed = (all(row["returncode"] == 0 for row in results + pair_comparisons)
              and supplement["returncode"] == 0 and comparison.returncode == 0)
    write_json(TASK_ROOT / "results" / "tutorials" / "paired_progress.json",
               {"status": "passed" if passed else "failed", "completed": results,
                "pair_comparisons": pair_comparisons,
                "supplementary_audit": supplement,
                "previous_driver_status": str(history) if history else None,
                "previous_driver_status_reason": history_reason,
                "comparison_returncode": comparison.returncode})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
