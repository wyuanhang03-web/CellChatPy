"""Execute every published tutorial in an isolated, freshly computed workspace.

Original analysis parameters are retained verbatim. Documented corrections to
independent tutorial assertions apply identically to both engines. Added
cells only choose the package import, record numerical evidence, and validate
completion. No existing CellChat result/cache is loaded by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import urljoin


TASK_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = TASK_ROOT.parent
NAMES = [
    "CellChat_humanSkin_LS",
    "CellChat_comparison_analysis_NL_LS",
    "Comparison_analysis_of_multiple_datasets_with_different_cellular_compositions",
    "CellChat_analysis_of_spatial_transcriptomics_data",
    "CellChat_analysis_of_multiple_spatial_transcriptomics_datasets",
    "CellChat_analysis_of_spatial_multiomics_data",
    "SpatialCellChat_analysis_of_spatial_transcriptomics_data",
]
INPUTS = {
    NAMES[0]: ["data_humanSkin.h5"],
    NAMES[1]: ["data_humanSkin.h5"],
    NAMES[2]: ["mouse_skin_e13.h5ad", "mouse_skin_e14.h5ad"],
    NAMES[3]: ["mouse_cortex_visium.h5"],
    NAMES[4]: ["human_intestine_A1.h5", "human_intestine_A2.h5"],
    NAMES[5]: ["mouse_spleen_spatial_multiomics.h5"],
    NAMES[6]: ["human_psoriasis.h5"],
}
SHORT_NAMES = dict(zip(NAMES, ["01_human_skin", "02_nl_ls", "03_embryonic", "04_visium", "05_multi_spatial", "06_multiomics", "07_psoriasis"]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def package_fingerprint(package_root: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted((package_root / "CellChatPy").rglob("*.py")):
        digest.update(source.relative_to(package_root).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def inventory(source_root: Path, fetch_web: bool = True) -> dict:
    import requests

    destination = TASK_ROOT / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    base = "https://cellchatpy.readthedocs.io/en/latest/"
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "readthedocs_url": base,
        "source_root": str(source_root),
        "cache_policy": "Recompute all original code cells; no persisted inference results are reused.",
        "adaptations": [
            "Original inference parameters and cells are retained; the explicit Visium manual-check threshold correction is recorded in tutorial_compatibility_patches.json and applied identically to both engines.",
            "Each notebook executes in a fresh isolated project-shaped directory under results/tutorials.",
            "Original raw inputs are copied once and linked from task-local copies into each execution directory.",
            "CellChatPy imports are explicitly resolved from the selected package root and asserted.",
            "The R-exported permutations are an original input, not an inferred-result cache.",
            "Jupyter, IPython, Numba and Matplotlib caches are confined to this task.",
            "A call-recording wrapper saves probability/P-value arrays without altering numerical arguments.",
        ],
        "tutorials": [],
    }
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (CellChatPy reproducibility audit)"
    if fetch_web:
        try:
            response = session.get(base, timeout=45)
            response.raise_for_status()
            (destination / "readthedocs_index.html").write_text(response.text, encoding="utf-8")
            links = sorted(set(re.findall(r'href="(tutorials/[^"#]+\.html)', response.text)))
            report["published_links"] = links
            report["readthedocs_status"] = response.status_code
        except Exception as exc:
            report["readthedocs_error"] = repr(exc)
    for name in NAMES:
        source = source_root / "docs" / "tutorials" / f"{name}.ipynb"
        item = {"name": name, "source_notebook": str(source), "url": base + "tutorials/" + name + ".html"}
        if source.is_file():
            notebook = json.loads(source.read_text(encoding="utf-8"))
            item["notebook_sha256"] = sha256(source)
            item["code_cells"] = sum(c["cell_type"] == "code" for c in notebook["cells"])
            item["total_cells"] = len(notebook["cells"])
        else:
            item["missing_source"] = True
        item["inputs"] = []
        for filename in INPUTS[name]:
            path = ORIGINAL_ROOT / "data" / filename
            item["inputs"].append({"path": str(path), "exists": path.is_file(),
                                   "size_bytes": path.stat().st_size if path.is_file() else None,
                                   "sha256": sha256(path) if path.is_file() else None})
        if name == NAMES[0]:
            perm = source_root / "r_permutations.csv"
            item["permutations"] = {"path": str(perm), "exists": perm.is_file(),
                                    "sha256": sha256(perm) if perm.is_file() else None}
        if fetch_web:
            try:
                response = session.get(item["url"], timeout=45)
                response.raise_for_status()
                item["web_status"] = response.status_code
                (destination / f"{name}.html").write_text(response.text, encoding="utf-8")
                downloads = re.findall(r'href="([^"#]+\.ipynb(?:\.txt)?)"', response.text)
                item["published_notebook_downloads"] = [urljoin(item["url"], d) for d in downloads]
                if downloads:
                    notebook_response = session.get(item["published_notebook_downloads"][0], timeout=45)
                    notebook_response.raise_for_status()
                    published = notebook_response.json()
                    published_code = ["".join(c["source"]) for c in published["cells"] if c["cell_type"] == "code"]
                    source_code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
                    item["published_code_matches_github"] = published_code == source_code
                    (destination / f"published_{name}.ipynb").write_bytes(notebook_response.content)
            except Exception as exc:
                item["web_error"] = repr(exc)
        report["tutorials"].append(item)
    write_json(destination / "tutorial_inventory.json", report)
    return report


def stage_input(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
            raise RuntimeError(f"Staged input changed: {destination}")
        return
    shutil.copy2(source, destination)


def prepare(name: str, source_root: Path, label: str) -> Path:
    outputs = TASK_ROOT / "results" / "tutorials"
    run_root = outputs / label / SHORT_NAMES[name]
    if (run_root / "executed.ipynb").exists():
        raise FileExistsError(f"Refusing to overwrite executed evidence: {run_root}; choose a new --label.")
    (run_root / "data").mkdir(parents=True, exist_ok=True)
    (run_root / "tutorial").mkdir(exist_ok=True)
    # A directory marker satisfies original repository-location checks. The
    # actual package is imported explicitly in the prelude, never copied here.
    (run_root / "CellChatPy").mkdir(exist_ok=True)
    for filename in INPUTS[name]:
        original = ORIGINAL_ROOT / "data" / filename
        staged = outputs / "inputs" / filename
        stage_input(original, staged)
        target = run_root / "data" / filename
        if not target.exists():
            try:
                os.link(staged, target)
            except OSError:
                shutil.copy2(staged, target)
    if name == NAMES[0]:
        stage_input(source_root / "r_permutations.csv", run_root / "r_permutations.csv")
    return run_root


def prelude(package_root: Path, run_root: Path) -> str:
    return '''# Added by the reproducibility runner: paths and evidence only.
import os as _audit_os, sys as _audit_sys, json as _audit_json
import time as _audit_time, functools as _audit_functools
from pathlib import Path as _AuditPath
import numpy as _audit_np
_audit_package = _AuditPath(PACKAGE_ROOT)
_audit_directory = _AuditPath(RUN_ROOT) / "evidence"
_audit_directory.mkdir(exist_ok=True)
_audit_sys.path.insert(0, str(_audit_package))
import CellChatPy as _audit_cc
assert _AuditPath(_audit_cc.__file__).resolve().is_relative_to(_audit_package.resolve())
print("AUDIT_PACKAGE", _audit_cc.__file__)
_audit_calls = []
_audit_original_compute = _audit_cc.compute_communication_probability
def _audit_array(value):
    return value.toarray() if hasattr(value, "toarray") else _audit_np.asarray(value)
def _audit_dump_network(obj, prefix):
    network = getattr(obj, "network", {})
    arrays, meta = {}, {"groups": [str(x) for x in obj.groups.categories]}
    for key in ("prob", "pval"):
        value = network.get(key)
        if isinstance(value, dict):
            names = list(value)
            meta[key + "_names"] = [str(x) for x in names]
            arrays[key] = _audit_np.stack([_audit_array(value[k]) for k in names]) if names else _audit_np.empty((0,))
        elif value is not None:
            arrays[key] = _audit_array(value)
    for key in ("count", "weight"):
        if network.get(key) is not None:
            arrays[key] = _audit_array(network[key])
    for key, value in arrays.items():
        if not _audit_np.isfinite(value).all():
            raise AssertionError("Nonfinite network result: " + key)
    _audit_np.savez_compressed(_audit_directory / (prefix + ".npz"), **arrays)
    (_audit_directory / (prefix + ".json")).write_text(_audit_json.dumps(meta, indent=2), encoding="utf-8")
@_audit_functools.wraps(_audit_original_compute)
def _audit_compute(*args, **kwargs):
    begin = _audit_time.perf_counter()
    result = _audit_original_compute(*args, **kwargs)
    index = len(_audit_calls)
    record = {"call_index": index, "elapsed_seconds": _audit_time.perf_counter() - begin,
              "kwargs": {k: ("array(shape=" + str(v.shape) + ")" if isinstance(v, _audit_np.ndarray) else repr(v)) for k, v in kwargs.items()},
              "shape": list(result.shape), "groups": len(result.groups.categories)}
    _audit_calls.append(record)
    _audit_dump_network(result, "inference_" + str(index))
    (_audit_directory / "inference_calls.json").write_text(_audit_json.dumps(_audit_calls, indent=2), encoding="utf-8")
    return result
_audit_cc.compute_communication_probability = _audit_compute
'''.replace("PACKAGE_ROOT", repr(str(package_root))).replace("RUN_ROOT", repr(str(run_root)))


POSTLUDE = '''# Added by the reproducibility runner: export final finite numerical evidence.
_audit_seen = set()
_audit_objects = []
for _audit_name, _audit_obj in list(globals().items()):
    if _audit_name.startswith("_audit_") or id(_audit_obj) in _audit_seen:
        continue
    if hasattr(_audit_obj, "network") and hasattr(_audit_obj, "groups") and hasattr(_audit_obj.groups, "categories"):
        _audit_seen.add(id(_audit_obj))
        _audit_dump_network(_audit_obj, "final_" + _audit_name)
        _audit_objects.append(_audit_name)
(_audit_directory / "completion.json").write_text(_audit_json.dumps({
    "all_original_cells_completed": True, "package_file": _audit_cc.__file__,
    "final_objects": _audit_objects, "inference_call_count": len(_audit_calls)
}, indent=2), encoding="utf-8")
print("AUDIT_ALL_ORIGINAL_CELLS_COMPLETED", _audit_objects)
'''


def execute(name: str, args) -> dict:
    run_root = prepare(name, args.source_root, args.label)
    cache_root = TASK_ROOT / "results" / "tutorials" / "cache"
    temporary = cache_root / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    tempfile.tempdir = str(temporary)
    for var, subdir in [("MPLCONFIGDIR", "matplotlib"), ("IPYTHONDIR", "ipython"),
                        ("JUPYTER_RUNTIME_DIR", "jupyter"), ("NUMBA_CACHE_DIR", "numba"),
                        ("JUPYTER_CONFIG_DIR", "jupyter_config")]:
        folder = cache_root / subdir
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(folder)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache")
    kernel_data = cache_root / "jupyter_data"
    kernel_spec = kernel_data / "kernels" / "cellchat-validation"
    write_json(kernel_spec / "kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "CellChat isolated validation", "language": "python",
    })
    os.environ["JUPYTER_PATH"] = str(kernel_data)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    import nbformat
    from nbclient import NotebookClient

    source = args.source_root / "docs" / "tutorials" / (name + ".ipynb")
    notebook = nbformat.read(source, as_version=4)
    original_code = [c.source for c in notebook.cells if c.cell_type == "code"]
    adaptations = []
    patch_file = TASK_ROOT / "validation" / "tutorial_compatibility_patches.json"
    if patch_file.exists() and not args.no_tutorial_fixes:
        for patch in json.loads(patch_file.read_text(encoding="utf-8")):
            if patch["name"] != name:
                continue
            cell = notebook.cells[patch["original_cell_index"]]
            if hashlib.sha256(cell.source.encode()).hexdigest() != patch["original_cell_sha256"]:
                raise AssertionError("Tutorial compatibility patch source hash differs")
            cell.source = cell.source.replace(patch["old"], patch["new"])
            assert hashlib.sha256(cell.source.encode()).hexdigest() == patch["adapted_cell_sha256"]
            adaptations.append(patch)
    adapted_code = [c.source for c in notebook.cells if c.cell_type == "code"]
    write_json(run_root / "adaptations.json", adaptations)
    for index, cell in enumerate(notebook.cells):
        cell.metadata["audit_original_cell_index"] = index
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(prelude(args.package_root, run_root), metadata={"tags": ["audit_setup"]}))
    notebook.cells.append(nbformat.v4.new_code_cell(POSTLUDE, metadata={"tags": ["audit_completion"]}))
    assert adapted_code == [c.source for c in notebook.cells if c.cell_type == "code" and "audit_original_cell_index" in c.metadata]
    nbformat.write(notebook, run_root / "prepared.ipynb")
    timings = []
    started = time.perf_counter()
    active_cell = {}

    def on_start(cell, cell_index, **kwargs):
        if cell.cell_type != "code":
            return
        active_cell.update(index=cell_index, begin=time.perf_counter())
        status = {"name": name, "status": "running", "cell_index": cell_index,
                  "original_cell_index": cell.metadata.get("audit_original_cell_index"),
                  "first_line": cell.source.splitlines()[0] if cell.source else "", "elapsed_seconds": time.perf_counter() - started}
        write_json(run_root / "status.json", status)
        print(json.dumps(status), flush=True)

    def on_complete(cell, cell_index, **kwargs):
        # nbclient 0.5's hook is sent when execution starts, not on full IOPub
        # completion; precise kernel execution timings remain in cell metadata.
        if cell.cell_type == "code":
            nbformat.write(notebook, run_root / "progress.ipynb")

    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="cellchat-validation",
                            allow_errors=False, record_timing=True,
                            resources={"metadata": {"path": str(run_root)}},
                            on_cell_start=on_start, on_cell_complete=on_complete)
    summary = {"name": name, "label": args.label, "package_root": str(args.package_root),
               "package_python_source_sha256": package_fingerprint(args.package_root),
               "source_sha256": sha256(source), "threads": args.threads,
               "original_code_cell_count": len(original_code), "original_cells_unchanged": not adaptations,
               "tutorial_compatibility_patches": adaptations,
               "executed_code_sha256": hashlib.sha256("\n".join(adapted_code).encode()).hexdigest(),
               "output_directory": str(run_root)}
    import psutil
    stop_monitor = threading.Event()
    resources = {"kernel_tree_peak_rss_bytes": 0,
                 "system_min_available_bytes": psutil.virtual_memory().available,
                 "sampling_interval_seconds": 0.25}

    def monitor_resources():
        while not stop_monitor.is_set():
            resources["system_min_available_bytes"] = min(resources["system_min_available_bytes"], psutil.virtual_memory().available)
            provisioner = getattr(client.km, "provisioner", None)
            process = getattr(provisioner, "process", None)
            pid = getattr(process, "pid", None)
            if pid is not None:
                try:
                    kernel = psutil.Process(pid)
                    rss = sum(p.memory_info().rss for p in [kernel, *kernel.children(recursive=True)] if p.is_running())
                    resources["kernel_tree_peak_rss_bytes"] = max(resources["kernel_tree_peak_rss_bytes"], rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            stop_monitor.wait(0.25)

    monitor = threading.Thread(target=monitor_resources, daemon=True)
    monitor.start()
    try:
        client.execute(cwd=str(run_root))
        errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
        if errors:
            raise AssertionError(f"Notebook contains {len(errors)} execution errors")
        executed_original = sum(c.cell_type == "code" and c.execution_count is not None and "audit_original_cell_index" in c.metadata for c in notebook.cells)
        if executed_original != len(original_code):
            raise AssertionError(f"Only {executed_original}/{len(original_code)} original code cells executed")
        if package_fingerprint(args.package_root) != summary["package_python_source_sha256"]:
            raise AssertionError("Package source changed during notebook execution; rerun against a fixed revision")
        summary["status"] = "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        summary["error_type"] = type(exc).__name__
        summary["failed_original_cell_index"] = notebook.cells[active_cell.get("index", 0)].metadata.get("audit_original_cell_index")
        (run_root / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
        summary["resource_observations"] = resources
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["executed_original_code_cells"] = sum(c.cell_type == "code" and c.execution_count is not None and "audit_original_cell_index" in c.metadata for c in notebook.cells)
        summary["figures"] = len(list(run_root.rglob("*.png")))
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                timings.append({"cell_index": index, "original_cell_index": cell.metadata.get("audit_original_cell_index"), "execution_count": cell.execution_count, "execution": dict(cell.metadata.get("execution", {}))})
        write_json(run_root / "cell_timings.json", timings)
        nbformat.write(notebook, run_root / "executed.ipynb")
        write_json(run_root / "summary.json", summary)
        write_json(run_root / "status.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=TASK_ROOT / "upstream")
    parser.add_argument("--package-root", type=Path, default=TASK_ROOT / "development")
    parser.add_argument("--label", default="development")
    parser.add_argument("--only", nargs="+", choices=NAMES)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--no-tutorial-fixes", action="store_true", help="Reproduce the original published assertion bug without the documented overlay.")
    parser.add_argument("--resume", action="store_true", help="Keep this task's completed fresh executions and archive/restart failed or interrupted executions.")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-cell timeout; original analysis counts are never reduced.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.package_root = args.package_root.resolve()
    if args.inventory_only:
        report = inventory(args.source_root, fetch_web=not args.no_web)
        print(json.dumps({"tutorials": len(report["tutorials"]), "inventory": str(TASK_ROOT / "validation" / "tutorial_inventory.json")}))
        return 0
    if not (args.package_root / "CellChatPy" / "__init__.py").is_file():
        raise FileNotFoundError(f"Package is not ready: {args.package_root}")
    summaries = []
    for name in args.only or NAMES:
        previous = TASK_ROOT / "results" / "tutorials" / args.label / SHORT_NAMES[name]
        if args.resume and previous.is_dir():
            summary_file = previous / "summary.json"
            previous_summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else {}
            current_fingerprint = package_fingerprint(args.package_root)
            previous_fingerprint = previous_summary.get("package_python_source_sha256")
            # The immutable upstream's first completed run predates fingerprint
            # instrumentation. It is safe to retain after matching its exact
            # package root; development runs always require a fingerprint.
            immutable_upstream = args.package_root == (TASK_ROOT / "upstream").resolve()
            same_package = previous_fingerprint == current_fingerprint or (
                previous_fingerprint is None and immutable_upstream
                and previous_summary.get("package_root") == str(args.package_root))
            if previous_summary.get("status") == "passed" and same_package:
                summaries.append(previous_summary)
                print(json.dumps({"status": "already_executed_in_this_task", "name": name, "evidence": str(previous)}), flush=True)
                continue
            archive = TASK_ROOT / "results" / "tutorials" / "attempts" / args.label / (SHORT_NAMES[name] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
            if not previous.resolve().is_relative_to(TASK_ROOT.resolve()) or not archive.resolve().is_relative_to(TASK_ROOT.resolve()):
                raise ValueError("Refusing to move paths outside the task workspace")
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(previous), str(archive))
        pause_file = TASK_ROOT / "results" / "tutorials" / "PAUSE"
        while pause_file.exists():
            print(json.dumps({"status": "waiting_for_resource_window", "next_tutorial": name, "pause_file": str(pause_file)}), flush=True)
            time.sleep(30)
        summaries.append(execute(name, args))
        suite_root = TASK_ROOT / "results" / "tutorials" / args.label
        all_summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(suite_root.glob("[0-9][0-9]_*/summary.json"))]
        write_json(suite_root / "suite_summary.json", all_summaries)
    return int(any(row["status"] != "passed" for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
