"""Verify current delivery evidence without rerunning inference or notebooks.

Saved arrays are compared again. Missing evidence and stale artifact/source
hashes fail the audit, including an unfinished tutorial queue.
"""
from pathlib import Path
from email.parser import BytesParser
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from benchmark_scaling import compare_arrays
from compare_tutorial_results import compare_pair
from compare_tutorial_saved_objects import OBJECTS
from run_regression import snapshot as regression_snapshot
from run_tutorials import NAMES, SHORT_NAMES

ROOT = Path(__file__).resolve().parents[1]
CORE = {"modeling.py", "_streaming.py", "_sparse_quantiles.py", "network_storage.py"}
REQUIRED_PAIRED = {"pbmc68579_b10": 1, "pbmc68579_b100": 1,
                   "synthetic5000_b20": 3, "synthetic100000_b100": 3}
MILLION = "synthetic1000000_b100"
ARCHIVE_SHA256 = "46ef4c548cd97365ba61e0335fc58424425efa5dd4e88b0211177181548ef894"
ARTIFACT_HASHES = {}


def local_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Evidence must remain inside this delivery: {path}")
    return path


def sha(path):
    path = local_path(path)
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    digest = result.hexdigest()
    ARTIFACT_HASHES[path.relative_to(ROOT).as_posix()] = digest
    return digest


def read(relative):
    path = local_path(relative)
    payload = path.read_bytes()
    ARTIFACT_HASHES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload.decode("utf-8"))


def fingerprint(package):
    base = ROOT / package
    paths = sorted((base / "CellChatPy").rglob("*.py"))
    if not paths:
        raise ValueError(f"Empty package: {package}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(base).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    checks, notes = [], []

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    def section(name, operation):
        try:
            operation()
        except Exception as exc:
            check(name, False, error=f"{type(exc).__name__}: {exc}")

    def baseline_verified():
        required = {"pinned_archive_hash", "pinned_commit", "upstream_matches_pinned_archive",
                    "upstream_package_inventory"}
        present = {c["check"]: c["passed"] for c in checks if c["check"] in required}
        return set(present) == required and all(present.values())

    def baseline():
        archive = ROOT / "provenance/upstream.zip"
        check("pinned_archive_hash", sha(archive) == ARCHIVE_SHA256)
        metadata = read("provenance/github_commit.json")
        check("pinned_commit", metadata["sha"] == "986bf7ef909f1e15059aad8a9a9ee31ac675ffc3"
              and metadata["commit"]["tree"]["sha"] == "f3c44d7cc552bbf6ac14bec25bf0d89d98c1f65d")
        with zipfile.ZipFile(archive) as source:
            items = [item for item in source.infolist() if not item.is_dir()]
            changed = [item.filename for item in items
                       if local_path(ROOT / "upstream" / item.filename).read_bytes() != source.read(item)]
            check("upstream_matches_pinned_archive", bool(items) and not changed, differences=changed)
            expected = {item.filename for item in items if item.filename.startswith("CellChatPy/")
                        and Path(item.filename).suffix in {".py", ".csv", ".mtx"}}
            current = {p.relative_to(ROOT / "upstream").as_posix()
                       for p in (ROOT / "upstream/CellChatPy").rglob("*")
                       if p.is_file() and p.suffix in {".py", ".csv", ".mtx"}}
            check("upstream_package_inventory", current == expected)

    def wheel():
        report = read("validation/wheel_integrity.json")
        path = local_path(report["wheel"])
        check("wheel_recorded_digest", report["passed"] and sha(path) == report["sha256"])
        expected = {p.relative_to(ROOT / "development").as_posix(): p
                    for p in (ROOT / "development/CellChatPy").rglob("*")
                    if p.is_file() and p.suffix in {".py", ".csv", ".mtx"}}
        with zipfile.ZipFile(path) as archive:
            present = {name for name in archive.namelist() if name.startswith("CellChatPy/")
                       and Path(name).suffix in {".py", ".csv", ".mtx"}}
            mismatches = [name for name, current in expected.items()
                          if name not in present or archive.read(name) != current.read_bytes()]
            metadata = BytesParser().parsebytes(archive.read(next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))))
            check("wheel_current_contents", bool(expected) and present == set(expected)
                  and not mismatches, files=len(expected), mismatches=mismatches)
            check("wheel_version", metadata["Version"] == "1.1.0+streaming.20261020")
        changes = read("provenance/development_changes.json")["source_changes"]
        check("exported_source_manifest_nonempty", bool(changes))
        for item in changes:
            check("exported_source_hash:" + item["path"],
                  sha(ROOT / "development" / item["path"]) == item["development_sha256"])

    def regression():
        path = ROOT / "validation/combined_regression.xml"
        tree = ET.parse(path).getroot()
        totals = {key: sum(int(x.attrib.get(key, 0)) for x in tree.iter("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
        check("regression_suite", totals["tests"] >= 303 and
              totals["tests"] == len(list(tree.iter("testcase"))) and
              all(totals[k] == 0 for k in ("failures", "errors", "skipped")), **totals)
        provenance = read("validation/regression_provenance.json")
        current = regression_snapshot()
        check("regression_current_source_and_tests", bool(current) and provenance["passed"]
              and provenance["returncode"] == 0 and provenance["source_unchanged_during_execution"]
              and provenance["source_sha256"] == provenance["source_sha256_after"] == current
              and provenance["junit_sha256"] == sha(path))

    def tutorials():
        report = read("validation/tutorial_comparison_reference_development.json")
        rows = report["tutorials"]
        check("all_seven_tutorials", report["passed"] and report.get("complete_suite")
              and report.get("reference_label") == "reference" and report.get("candidate_label") == "development"
              and len(rows) == 7 and {r["name"] for r in rows} == set(NAMES)
              and {r["directory"] for r in rows} == set(SHORT_NAMES.values())
              and all(r.get("passed") and SHORT_NAMES[r["name"]] == r["directory"] for r in rows))
        for row in rows:
            name, directory = row["name"], row["directory"]
            roots = [ROOT / "results/tutorials" / label / directory for label in ("reference", "development")]
            for label, package, folder in zip(("reference", "development"), ("upstream", "development"), roots):
                summary = read(folder / "summary.json")
                recorded = summary.get("package_python_source_sha256")
                matched = recorded == fingerprint(package)
                if recorded is None and label == "reference" and directory == "05_multi_spatial":
                    notes.append(f"{label}/{directory}: explicitly allowlisted early baseline execution lacks package/code fingerprints; frozen archive and original executed cell sources are verified directly.")
                    matched = baseline_verified() and local_path(summary["package_root"]) == ROOT / "upstream"
                source = ROOT / "upstream/docs/tutorials" / (name + ".ipynb")
                original = read(source)
                expected_indices = [i for i, c in enumerate(original["cells"]) if c["cell_type"] == "code"]
                notebook = read(folder / "executed.ipynb")
                cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
                executed = [c for c in cells if "audit_original_cell_index" in c["metadata"]]
                code = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in executed]
                recorded_code_hash = summary.get("executed_code_sha256")
                code_matches = hashlib.sha256("\n".join(code).encode()).hexdigest() == recorded_code_hash
                if recorded_code_hash is None and label == "reference" and directory == "05_multi_spatial":
                    original_code = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                                     for c in original["cells"] if c["cell_type"] == "code"]
                    code_matches = (baseline_verified() and summary.get("original_cells_unchanged") is True
                                    and not summary.get("tutorial_compatibility_patches") and code == original_code)
                prefix = "reference" if label == "reference" else "candidate"
                check(f"tutorial:{label}/{directory}", summary["status"] == "passed" and matched
                      and summary["name"] == name and summary["label"] == label
                      and local_path(summary["package_root"]) == ROOT / package
                      and summary["source_sha256"] == sha(source)
                      and summary["executed_original_code_cells"] == summary["original_code_cell_count"] == len(expected_indices)
                      and [c["metadata"]["audit_original_cell_index"] for c in executed] == expected_indices
                      and all(c.get("execution_count") is not None for c in cells)
                      and not any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", []))
                      and code_matches
                      and row[prefix + "_status"] == summary["status"]
                      and row[prefix + "_seconds"] == summary["elapsed_seconds"])
                check(f"tutorial_comparison_source:{label}/{directory}",
                      row.get(prefix + "_package_python_source_sha256") == recorded)
            old_arrays, new_arrays = [sorted((folder / "evidence").glob("*.npz")) for folder in roots]
            check(f"tutorial_array_inventory:{directory}", bool(old_arrays)
                  and [p.name for p in old_arrays] == [p.name for p in new_arrays]
                  and len(row["comparisons"]) == len(old_arrays))
            fresh = []
            for old in old_arrays:
                new = roots[1] / "evidence" / old.name
                for array in (old, new):
                    sha(array)
                    sha(array.with_suffix(".json"))
                fresh.append(compare_pair(old, new, 1e-12, 1e-12))
            check(f"tutorial_raw_comparison:{directory}", bool(fresh)
                  and all(c["passed"] for c in fresh) and fresh == row["comparisons"]
                  and (directory == "07_psoriasis" or any(c["has_numerical_content"] for c in fresh)))
            figure_names = [[p.relative_to(folder / "tutorial").as_posix()
                             for p in sorted((folder / "tutorial").rglob("*.png"))] for folder in roots]
            figures = [{"figure": "tutorial/" + relative, "exists": True,
                        "sha256_equal": sha(roots[0] / "tutorial" / relative) == sha(roots[1] / "tutorial" / relative)}
                       for relative in figure_names[0] if (roots[1] / "tutorial" / relative).is_file()]
            check(f"tutorial_figures_current:{directory}", figure_names[0] == figure_names[1]
                  and figures == row["figure_comparisons"])

        supplemental = read("validation/tutorial_saved_object_comparison.json")
        supplements = supplemental["tutorials"]
        check("four_saved_object_reports", supplemental["passed"]
              and supplemental["complete_four_saved_object_tutorials"] and len(supplements) == 4
              and {r["directory"] for r in supplements} == set(OBJECTS))
        for directory, relative_pickle in OBJECTS.items():
            extra = read(f"validation/tutorial_saved_objects_{directory}.json")
            check(f"saved_object_report:{directory}", extra["passed"] and extra["arrays"]
                  and extra["metadata_equal"] and extra["array_paths_equal"]
                  and extra["nonempty_probability_arrays"] > 0
                  and all(a["passed"] and (not a.get("exact_required") or a["bitwise_equal"]) for a in extra["arrays"])
                  and [r for r in supplements if r["directory"] == directory] == [extra]
                  and (directory != "07_psoriasis" or extra["nonempty_spot_probability_arrays"] > 0))
            for field, label in (("reference", "reference"), ("candidate", "development")):
                folder = ROOT / "results/tutorials" / label / directory
                path = folder / relative_pickle
                check(f"saved_object_current:{label}/{directory}",
                      local_path(extra[field + "_pickle"]) == path
                      and extra[field + "_pickle_sha256"] == sha(path)
                      and extra[field + "_run_summary_sha256"] == sha(folder / "summary.json"))
            matching = next(r for r in rows if r["directory"] == directory)
            check(f"tutorial_saved_object_link:{directory}",
                  matching["saved_object_audit"]["passed"]
                  and matching["saved_object_audit"]["current_execution_provenance_matches"]
                  and matching["saved_object_audit"]["numeric_arrays"] == len(extra["arrays"]))

        driver = read("results/tutorials/paired_progress.json")
        completed, pairs = driver["completed"], driver["pair_comparisons"]
        check("tutorial_driver_complete", driver["status"] == "passed"
              and driver["comparison_returncode"] == 0 and len(completed) == 14
              and {(r["name"], r["label"]) for r in completed} == {(n, l) for n in NAMES for l in ("reference", "development")}
              and all(r["returncode"] == 0 for r in completed)
              and len(pairs) == 7 and {r["name"] for r in pairs} == set(NAMES)
              and all(r["returncode"] == 0 for r in pairs)
              and driver["supplementary_audit"]["status"] == "passed"
              and driver["supplementary_audit"]["returncode"] == 0
              and set(driver["supplementary_audit"]["requested"]) == set(OBJECTS))
        for pair in pairs:
            individual = read(pair["report"])
            target = next(r for r in rows if r["name"] == pair["name"])
            check("tutorial_driver_pair:" + pair["name"], individual["passed"]
                  and individual["tutorials"] == [target])

    def benchmark(path):
        data = read(path)
        tag, records = path.parent.name, data["records"]
        keys = [(r["engine"], r["repeat"]) for r in records]
        check("benchmark_record_inventory:" + tag, bool(keys) and len(keys) == len(set(keys))
              and all(engine in {"upstream", "streaming"} for engine, _ in keys))
        completed = {}
        for record in records:
            engine, repeat = record["engine"], record["repeat"]
            worker = path.parent / f"{engine}_{repeat}.json"
            check(f"benchmark_worker_current:{tag}/{engine}/{repeat}", read(worker) == record)
            sha(worker.with_suffix(".log"))
            if record["status"] != "completed":
                check(f"benchmark_allowed_incompletion:{tag}/{engine}/{repeat}", tag == MILLION
                      and engine == "upstream" and record["status"] in
                      {"timeout", "memory_budget_exceeded", "private_memory_budget_exceeded"})
                notes.append(f"{tag}/{engine}/{repeat}: {record['status']}; no paired speed/equality claim.")
                continue
            completed[(engine, repeat)] = record
            package = "upstream" if engine == "upstream" else "development"
            recorded = record.get("source_sha256_at_start")
            expected = CORE if engine == "streaming" else {"modeling.py", "network_storage.py"}
            if recorded is None and tag == "synthetic5000_b20" and engine == "upstream" and repeat in {0, 1}:
                notes.append(f"{tag}/{engine}/{repeat}: explicitly allowlisted early immutable-baseline pilot without source fingerprint.")
                check(f"benchmark_source:{tag}/{engine}/{repeat}", baseline_verified())
            else:
                inventory_matches = (isinstance(recorded, dict) and
                                     (set(recorded) == expected if engine == "streaming" else expected <= set(recorded)))
                if (tag == "synthetic5000_b20" and engine == "upstream" and repeat == 2
                        and isinstance(recorded, dict) and set(recorded) == {"modeling.py"}):
                    notes.append(f"{tag}/{engine}/{repeat}: explicitly allowlisted early pilot recorded only modeling.py; that hash and the complete pinned baseline archive are verified, including network_storage.py.")
                    inventory_matches = baseline_verified()
                check(f"benchmark_source:{tag}/{engine}/{repeat}", inventory_matches
                      and all(local_path(ROOT / package / "CellChatPy" / name).is_relative_to(ROOT / package / "CellChatPy")
                              and sha(ROOT / package / "CellChatPy" / name) == digest
                              for name, digest in recorded.items()))
            check(f"benchmark_execution:{tag}/{engine}/{repeat}", record["returncode"] == 0
                  and local_path(record["package_file"]) == ROOT / package / "CellChatPy/__init__.py"
                  and all(math.isfinite(record[k]) and record[k] > 0 for k in
                          ("inference_seconds", "process_seconds", "process_peak_rss_bytes")))
            arrays = worker.with_suffix(".npz")
            sha(arrays)
            with np.load(arrays, allow_pickle=False) as saved:
                shape = (record["groups"], record["groups"], record["pairs"])
                check(f"benchmark_output:{tag}/{engine}/{repeat}",
                      set(saved.files) == {"prob", "pval", "groups", "names"}
                      and saved["prob"].shape == saved["pval"].shape == shape
                      and saved["names"].size == record["pairs"] > 0
                      and saved["groups"].size == record["groups"] > 0
                      and np.isfinite(saved["prob"]).all() and np.isfinite(saved["pval"]).all()
                      and (saved["pval"] >= 0).all() and (saved["pval"] <= 1).all())
        pair_repeats = {repeat for engine, repeat in completed if engine == "upstream"
                        and ("streaming", repeat) in completed}
        comparisons = data["comparisons"]
        check("benchmark_pair_inventory:" + tag, len(comparisons) == len(pair_repeats)
              and {r["repeat"] for r in comparisons} == pair_repeats)
        fresh_pass = []
        for comparison in comparisons:
            repeat = comparison["repeat"]
            old, new = completed[("upstream", repeat)], completed[("streaming", repeat)]
            fields = ("cells", "genes", "nnz", "groups", "pairs", "method", "n_boot", "seed", "threads", "candidate_selection")
            check(f"benchmark_equal_parameters:{tag}/{repeat}", all(old[k] == new[k] for k in fields)
                  and old["settings"]["parameter"] == new["settings"]["parameter"])
            fresh = compare_arrays(path.parent / f"upstream_{repeat}.npz", path.parent / f"streaming_{repeat}.npz")
            fresh = json.loads(json.dumps(fresh, default=bool))
            recorded = {key: value for key, value in comparison.items() if key != "repeat"}
            check(f"benchmark_raw_comparison:{tag}/{repeat}", fresh["pass"] and fresh == recorded)
            fresh_pass.append(fresh["pass"])
            if tag in REQUIRED_PAIRED:
                check(f"benchmark_exact_results:{tag}/{repeat}",
                      fresh["prob"]["exact_equal"] and fresh["pval"]["exact_equal"])
        check("benchmark_summary_consistency:" + tag,
              data["all_comparisons_pass"] == (bool(fresh_pass) and all(fresh_pass)
                                             and len(completed) == len(records)))
        if tag in REQUIRED_PAIRED:
            count = REQUIRED_PAIRED[tag]
            check("required_benchmark:" + tag, bool(comparisons) and data["all_comparisons_pass"]
                  and pair_repeats == set(range(count)) and len(records) == 2 * count
                  and len(completed) == 2 * count)
        if tag == MILLION:
            runs = [r for (engine, _), r in completed.items() if engine == "streaming"]
            check("million_cell_streaming_completion", bool(runs)
                  and all(r["cells"] == 1000000 and r["genes"] == 256 and r["pairs"] == 100
                          and r["groups"] == 10 and r["n_boot"] == 100 for r in runs))
        for history in data.get("refresh_history", []):
            retained = history["retained_upstream_sha256"]
            check("retained_upstream_hashes:" + tag, bool(retained)
                  and all(sha(path.parent / name) == digest for name, digest in retained.items()))

    def ablation():
        data = read("validation/ablation.json")
        recorded = data["source_sha256"]
        check("ablation_source", set(recorded) == CORE and
              all(sha(ROOT / "development/CellChatPy" / name) == digest for name, digest in recorded.items()))
        rows = data["results"]
        check("ablation_numerics", set(rows) == {"all_enabled", "no_pruning", "no_sparse_plan"}
              and all(len(row["exact_prob"]) == len(row["exact_pval"]) == len(row["evaluated_genes"]) == 3
                      and all(row["exact_prob"]) and all(row["exact_pval"]) for row in rows.values()))

    section("baseline_evidence", baseline)
    section("wheel_evidence", wheel)
    section("regression_evidence", regression)
    section("tutorial_evidence", tutorials)
    paths = sorted((ROOT / "results/benchmarks").glob("*/summary.json"))
    check("required_benchmark_directories", set(REQUIRED_PAIRED) | {MILLION} <= {p.parent.name for p in paths})
    for path in paths:
        section("benchmark_evidence:" + path.parent.name, lambda p=path: benchmark(p))
    section("ablation_evidence", ablation)
    result = {"passed": bool(checks) and all(x["passed"] for x in checks), "checks": checks, "notes": notes,
              "artifact_sha256": ARTIFACT_HASHES,
              "upstream_package_sha256": fingerprint("upstream"),
              "development_package_sha256": fingerprint("development"),
              "audit_scope": "Fresh comparison of saved arrays and direct artifact/source verification; no inference or notebook re-execution"}
    (ROOT / "validation/delivery_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(checks),
                      "failures": [c for c in checks if not c["passed"]], "notes": notes}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
