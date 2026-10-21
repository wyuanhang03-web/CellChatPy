"""Compare core numerical evidence from full, freshly executed tutorial runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_tutorials import NAMES


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def compare_pair(reference: Path, candidate: Path, atol: float, rtol: float) -> dict:
    report = {"reference": str(reference), "candidate": str(candidate), "arrays": []}
    with np.load(reference, allow_pickle=False) as old, np.load(candidate, allow_pickle=False) as new:
        report["keys_equal"] = old.files == new.files
        for key in old.files:
            item = {"key": key, "shape": list(old[key].shape)}
            if key not in new.files:
                item.update(passed=False, error="missing array")
            elif old[key].shape != new[key].shape:
                item.update(passed=False, error="shape differs", new_shape=list(new[key].shape))
            else:
                a, b = old[key], new[key]
                item["finite"] = bool(np.isfinite(a).all() and np.isfinite(b).all())
                item["bitwise_equal"] = bool(np.array_equal(a, b))
                item["max_abs_error"] = float(np.max(np.abs(a - b))) if a.size else 0.0
                item["passed"] = bool(item["finite"] and np.allclose(a, b, rtol=rtol, atol=atol))
                if key == "pval":
                    item["significance_005_equal"] = bool(np.array_equal(a < 0.05, b < 0.05))
                    item["passed"] = item["passed"] and item["bitwise_equal"] and item["significance_005_equal"]
            report["arrays"].append(item)
    old_meta = json.loads(reference.with_suffix(".json").read_text(encoding="utf-8"))
    new_meta = json.loads(candidate.with_suffix(".json").read_text(encoding="utf-8"))
    report["labels_equal"] = old_meta == new_meta
    report["numeric_element_count"] = sum(int(np.prod(r["shape"])) for r in report["arrays"])
    report["has_numerical_content"] = report["numeric_element_count"] > 0
    report["scope"] = "numeric_arrays_and_labels" if report["has_numerical_content"] else "metadata_only_empty_network_snapshot"
    report["passed"] = report["keys_equal"] and report["labels_equal"] and all(row["passed"] for row in report["arrays"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--candidate", default="development")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--only", nargs="+", choices=NAMES, help="Validate an explicitly selected subset; the report still states whether the full seven-tutorial suite is complete.")
    parser.add_argument("--require-saved-objects", action="store_true", help="Require applicable saved-object supplements even for a selected subset (always required for the full suite).")
    parser.add_argument("--output", type=Path, help="Optional task-local JSON output path.")
    args = parser.parse_args()
    location = ROOT / "results" / "tutorials"
    report = {"reference_label": args.reference, "candidate_label": args.candidate,
              "exact_array_equality_definition": "numpy.array_equal on finite values (legacy bitwise_equal field name); PNG equality uses SHA-256 byte hashes",
              "absolute_tolerance": args.atol, "relative_tolerance": args.rtol,
              "requested_tutorials": args.only or NAMES, "tutorials": []}
    for reference in sorted((location / args.reference).glob("[0-9][0-9]_*/summary.json")):
        old = json.loads(reference.read_text(encoding="utf-8"))
        if args.only and old["name"] not in args.only:
            continue
        new_path = location / args.candidate / reference.parent.name
        item = {"name": old["name"], "directory": reference.parent.name, "comparisons": []}
        if not (new_path / "summary.json").is_file():
            item.update(passed=False, error="candidate execution missing")
            report["tutorials"].append(item)
            continue
        new = json.loads((new_path / "summary.json").read_text(encoding="utf-8"))
        item["reference_status"], item["candidate_status"] = old["status"], new["status"]
        item["reference_package_python_source_sha256"] = old.get("package_python_source_sha256")
        item["candidate_package_python_source_sha256"] = new.get("package_python_source_sha256")
        item["original_source_equal"] = old["source_sha256"] == new["source_sha256"]
        item["reference_seconds"], item["candidate_seconds"] = old["elapsed_seconds"], new["elapsed_seconds"]
        item["reference_figures"], item["candidate_figures"] = old["figures"], new["figures"]
        figure_rows = []
        reference_figures = sorted((reference.parent / "tutorial").rglob("*.png"))
        candidate_figures = sorted((new_path / "tutorial").rglob("*.png"))
        item["figure_paths_equal"] = ([p.relative_to(reference.parent / "tutorial").as_posix() for p in reference_figures]
                                       == [p.relative_to(new_path / "tutorial").as_posix() for p in candidate_figures])
        for figure in reference_figures:
            paired_figure = new_path / figure.relative_to(reference.parent)
            figure_rows.append({"figure": figure.relative_to(reference.parent).as_posix(),
                                "exists": paired_figure.is_file(),
                                "sha256_equal": paired_figure.is_file() and hashlib.sha256(figure.read_bytes()).digest() == hashlib.sha256(paired_figure.read_bytes()).digest()})
        item["figure_comparisons"] = figure_rows
        item["all_figure_files_identical"] = item["figure_paths_equal"] and all(r["sha256_equal"] for r in figure_rows)
        reference_arrays = sorted((reference.parent / "evidence").glob("*.npz"))
        candidate_arrays = sorted((new_path / "evidence").glob("*.npz"))
        item["evidence_names_equal"] = [x.name for x in reference_arrays] == [x.name for x in candidate_arrays]
        for array in reference_arrays:
            paired = new_path / "evidence" / array.name
            if not paired.is_file():
                item["comparisons"].append({"passed": False, "missing": str(paired)})
            else:
                item["comparisons"].append(compare_pair(array, paired, args.atol, args.rtol))
        item["passed"] = (old["status"] == new["status"] == "passed" and item["original_source_equal"]
                          and item["evidence_names_equal"] and bool(reference_arrays)
                          and item["figure_paths_equal"]
                          and all(row["passed"] for row in item["comparisons"]))
        item["nonempty_numerical_snapshots"] = sum(r.get("has_numerical_content", False) for r in item["comparisons"])
        item["core_comparison_scope"] = "Group network prob/pval/count/weight and labels when present; empty snapshots contain metadata only"
        item["saved_object_audit_required_for_full_suite"] = reference.parent.name in {"01_human_skin", "02_nl_ls", "03_embryonic", "07_psoriasis"}
        supplemental = ROOT / "validation" / f"tutorial_saved_objects_{reference.parent.name}.json"
        if supplemental.is_file():
            extra = json.loads(supplemental.read_text(encoding="utf-8"))
            supplement_current = (extra.get("reference_run_summary_sha256") == hashlib.sha256(reference.read_bytes()).hexdigest()
                                  and extra.get("candidate_run_summary_sha256") == hashlib.sha256((new_path / "summary.json").read_bytes()).hexdigest())
            for field, folder in (("reference", reference.parent), ("candidate", new_path)):
                pickle_path = Path(extra.get(field + "_pickle", ""))
                supplement_current = (supplement_current and pickle_path.is_file()
                                      and pickle_path.resolve().is_relative_to(folder.resolve())
                                      and file_sha256(pickle_path) == extra.get(field + "_pickle_sha256"))
            item["saved_object_audit"] = {"report": str(supplemental), "passed": extra["passed"],
                                          "current_execution_provenance_matches": supplement_current,
                                          "nonempty_spot_probability_arrays": extra.get("nonempty_spot_probability_arrays", 0),
                                          "numeric_arrays": len(extra.get("arrays", []))}
        else:
            item["saved_object_audit"] = {"status": "pending" if item["saved_object_audit_required_for_full_suite"] else "not_available_original_export_disabled"}
        if (len(args.only or NAMES) == 7 or args.require_saved_objects) and item["saved_object_audit_required_for_full_suite"]:
            item["passed"] = item["passed"] and item["saved_object_audit"].get("passed", False)
            item["passed"] = item["passed"] and item["saved_object_audit"].get("current_execution_provenance_matches", False)
            if reference.parent.name == "07_psoriasis":
                item["passed"] = item["passed"] and item["saved_object_audit"].get("nonempty_spot_probability_arrays", 0) > 0
        if reference.parent.name != "07_psoriasis":
            item["passed"] = item["passed"] and item["nonempty_numerical_snapshots"] > 0
        report["tutorials"].append(item)
    report["complete_suite"] = len(report["tutorials"]) == 7
    report["requested_scope_complete"] = len(report["tutorials"]) == len(args.only or NAMES)
    report["passed"] = report["requested_scope_complete"] and all(row["passed"] for row in report["tutorials"])
    suffix = "_selected" if args.only else ""
    path = args.output or (ROOT / "validation" / f"tutorial_comparison_{args.reference}_{args.candidate}{suffix}.json")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Comparison output must stay within this task directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "tutorials": len(report["tutorials"]), "report": str(path)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
