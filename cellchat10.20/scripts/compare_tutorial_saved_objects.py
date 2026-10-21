"""Audit only newly executed tutorial pickles, preserving sparse matrices.

Run after the notebook kernels finish. Each package version has its own worker
process, loading one pickle bundle at a time; numeric leaves are exported to memory-mappable
NPY files without densifying any sparse matrix. The separate report supplements,
and does not rewrite, the original notebook evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "tutorials"
OBJECTS = {
    "01_human_skin": "cellchat_humanSkin_LS.pkl",
    "02_nl_ls": "tutorial/cellchat_NL_LS_comparison.pkl",
    "03_embryonic": "tutorial/cellchat_mouseSkin_E13_E14_from_h5.pkl",
    "07_psoriasis": "tutorial/spatial_tutorial_output/cellchat_psoriasis_spatial.pkl",
}
SLOTS = ("network", "pathway_network", "spot_network", "spot_pathway_network", "network_similarity")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def export(label, short_name):
    package = ROOT / ("upstream" if label == "reference" else "development")
    cache = RUNS / "cache"
    for key, folder in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib"),
                        ("TEMP", "temp"), ("TMP", "temp")):
        os.environ[key] = str(cache / folder)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(package))
    import CellChatPy
    import pandas as pd
    from scipy import sparse
    assert Path(CellChatPy.__file__).resolve().is_relative_to(package.resolve())
    run = RUNS / label / short_name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "passed", "Only completed fresh tutorial outputs may be inspected"
    source = run / OBJECTS[short_name]
    target = run / "saved_object_evidence"
    target.mkdir(exist_ok=True)
    manifest = {"source_pickle": str(source), "source_pickle_sha256": digest(source),
                "run_summary_sha256": digest(run / "summary.json"),
                "package_file": CellChatPy.__file__, "objects": {}, "arrays": [], "axis_validation": []}

    def array(value, path):
        a = np.asarray(value)
        if a.dtype.kind in "OUS":
            # Names and mixed annotation tables remain explicit metadata.
            return {"type": "labels", "shape": list(a.shape), "values": flatten(a.tolist(), path + "/labels")}
        filename = f"array_{len(manifest['arrays']):06d}.npy"
        np.save(target / filename, a, allow_pickle=False)
        row = {"path": path, "file": filename, "shape": list(a.shape),
               "dtype": str(a.dtype), "size": int(a.size)}
        manifest["arrays"].append(row)
        return {"type": "array", "path": path, "shape": list(a.shape), "dtype": str(a.dtype)}

    def flatten(value, path):
        if sparse.issparse(value):
            # Preserve storage layout and all explicit/implicit-zero structure.
            if value.format not in ("csr", "csc", "coo"):
                raise TypeError(f"Unsupported sparse storage at {path}: {value.format}")
            result = {"type": "sparse", "format": value.format, "shape": list(value.shape),
                      "data": array(value.data, path + "/data")}
            for key in (("row", "col") if value.format == "coo" else ("indices", "indptr")):
                result[key] = array(getattr(value, key), path + "/" + key)
            return result
        if isinstance(value, pd.DataFrame):
            return {"type": "dataframe", "index": flatten(value.index.tolist(), path + "/index"),
                    "index_dtype": str(value.index.dtype),
                    "index_names": flatten(value.index.names, path + "/index_names"),
                    "columns": flatten(value.columns.tolist(), path + "/columns"),
                    "column_index_dtype": str(value.columns.dtype),
                    "column_names": flatten(value.columns.names, path + "/column_names"),
                    "column_values": [flatten(value.iloc[:, i], path + f"/column/{i}")
                                      for i in range(value.shape[1])]}
        if isinstance(value, pd.Series):
            result = {"type": "series", "index": flatten(value.index.tolist(), path + "/index"),
                      "index_dtype": str(value.index.dtype), "dtype": str(value.dtype),
                      "name": flatten(value.name, path + "/name")}
            if isinstance(value.dtype, pd.CategoricalDtype):
                result.update(categories=flatten(value.cat.categories.tolist(), path + "/categories"),
                              ordered=value.cat.ordered, codes=array(value.cat.codes.to_numpy(), path + "/codes"))
            elif pd.api.types.is_extension_array_dtype(value.dtype) and pd.api.types.is_numeric_dtype(value.dtype):
                kind = np.int64 if pd.api.types.is_integer_dtype(value.dtype) else np.float64
                result.update(missing=array(value.isna().to_numpy(), path + "/missing"),
                              values=array(value.to_numpy(dtype=kind, na_value=0), path + "/values"))
            else:
                result["values"] = array(value.to_numpy(), path + "/values")
            return result
        if isinstance(value, dict):
            axis = None
            if isinstance(value.get("prob"), dict) and ("groups" in value or "spots" in value):
                axis = list(value["prob"])
                saved_order = value.get("matrix_order")
                if saved_order is not None and list(saved_order) != axis:
                    raise AssertionError(f"Stored matrix_order disagrees with probability labels at {path}")
                labels = list(value.get("groups", value.get("spots")))
                expected_shape = (len(labels), len(labels))
                if any(matrix.shape != expected_shape for matrix in value["prob"].values()):
                    raise AssertionError(f"Probability matrix shape disagrees with row/column labels at {path}")
                if "pval" in value:
                    if list(value["pval"]) != axis or any(matrix.shape != expected_shape for matrix in value["pval"].values()):
                        raise AssertionError(f"P-value axes disagree with probability axes at {path}")
                manifest["axis_validation"].append({"path": path, "valid": True,
                    "matrix_order_source": "explicit_matrix_order" if saved_order is not None else "probability_dictionary_insertion_order",
                    "communication_count": len(axis), "row_column_count": len(labels),
                    "pvalue_axes_checked": "pval" in value})
            elif "matrix_order" in value:
                raise AssertionError(f"Unrecognized matrix_order outside a labeled network at {path}")
            result = {"type": "dict", "items": [[str(k), flatten(v, path + "/" + str(k))]
                                                   for k, v in value.items() if k != "matrix_order"]}
            if axis is not None:
                # The new explicit storage field and the old insertion order
                # are compared as the same semantic axis, after validating
                # the explicit field rather than silently ignoring it.
                result["semantic_matrix_order"] = axis
            return result
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "items": [flatten(v, path + f"/{i}")
                                                             for i, v in enumerate(value)]}
        if isinstance(value, np.ndarray):
            return array(value, path)
        if value is pd.NA or value is pd.NaT:
            return {"type": type(value).__name__}
        if isinstance(value, np.generic):
            return flatten(value.item(), path)
        if isinstance(value, float) and not np.isfinite(value):
            return {"type": "nonfinite_metadata_float", "value": str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported saved network field at {path}: {type(value)}")

    seen = set()
    def visit(value, path):
        if hasattr(value, "pathway_network") and id(value) not in seen:
            seen.add(id(value))
            manifest["objects"][path] = {slot: flatten(getattr(value, slot, {}), path + "/" + slot)
                                         for slot in SLOTS}
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + "/" + str(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, path + f"/{index}")

    with source.open("rb") as handle:
        obj = pickle.load(handle)
    visit(obj, "object")
    if not manifest["objects"]:
        raise AssertionError("No CellChat objects found in the fresh tutorial pickle")
    write_json(target / "manifest.json", manifest)
    print(json.dumps({"exported": short_name, "label": label, "arrays": len(manifest["arrays"])}), flush=True)


def compare_array(a_path, b_path, path):
    a = np.load(a_path, mmap_mode="r", allow_pickle=False)
    b = np.load(b_path, mmap_mode="r", allow_pickle=False)
    row = {"path": path, "shape_equal": a.shape == b.shape, "dtype_equal": a.dtype == b.dtype,
           "size": int(a.size), "bitwise_equal": True, "finite": True, "max_abs_error": 0.0,
           "reference_nonzero_elements": 0, "candidate_nonzero_elements": 0}
    if not row["shape_equal"] or not row["dtype_equal"]:
        row["passed"] = False
        return row
    exact_required = "pval" in path.split("/") or a.dtype.kind in "biu"
    row["exact_required"] = exact_required
    close = True
    # Compare logical C-order coordinates, including mixed C/F source layouts,
    # with a bounded iterator buffer rather than a potentially full ravel copy.
    iterator = np.nditer([a, b], flags=["external_loop", "buffered", "zerosize_ok"],
                         op_flags=[["readonly"], ["readonly"]], order="C", buffersize=262144)
    for x, y in iterator:
        row["reference_nonzero_elements"] += int(np.count_nonzero(x))
        row["candidate_nonzero_elements"] += int(np.count_nonzero(y))
        row["finite"] &= bool(np.isfinite(x).all() and np.isfinite(y).all())
        row["bitwise_equal"] &= bool(np.array_equal(x, y))
        close &= bool(np.allclose(x, y, rtol=1e-12, atol=1e-12))
        if x.size:
            error = np.abs(x.astype(np.float64) - y.astype(np.float64))
            row["max_abs_error"] = max(row["max_abs_error"], float(np.max(error)))
    row["passed"] = row["finite"] and (row["bitwise_equal"] if exact_required else close)
    return row


def compare(short_name):
    folders = [RUNS / label / short_name / "saved_object_evidence" for label in ("reference", "development")]
    old, new = [json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in folders]
    report = {"directory": short_name, "metadata_equal": old["objects"] == new["objects"],
              "reference_pickle": old["source_pickle"], "candidate_pickle": new["source_pickle"],
              "reference_run_summary_sha256": old["run_summary_sha256"],
              "candidate_run_summary_sha256": new["run_summary_sha256"],
              "reference_pickle_sha256": old["source_pickle_sha256"],
              "candidate_pickle_sha256": new["source_pickle_sha256"], "arrays": []}
    report["axis_validation"] = {"reference": old["axis_validation"], "candidate": new["axis_validation"],
                                  "normalization": "Validated explicit matrix_order and legacy probability insertion order are compared as the same semantic axis"}
    report["array_paths_equal"] = [r["path"] for r in old["arrays"]] == [r["path"] for r in new["arrays"]]
    if report["array_paths_equal"]:
        report["arrays"] = [compare_array(folders[0] / a["file"], folders[1] / b["file"], a["path"])
                            for a, b in zip(old["arrays"], new["arrays"])]
    report["nonempty_probability_arrays"] = sum("prob" in r["path"].split("/") and r["size"] > 0
                                                  and r["path"].split("/")[-1] not in {"indptr", "indices", "row", "col"}
                                                  for r in report["arrays"])
    report["nonempty_spot_probability_arrays"] = sum("/spot_network/prob/" in r["path"] and r["path"].endswith("/data")
                                                       and r["reference_nonzero_elements"] > 0
                                                       and r["candidate_nonzero_elements"] > 0 for r in report["arrays"])
    report["pathway_array_count"] = sum("/pathway_network/" in r["path"] for r in report["arrays"])
    report["spot_pathway_array_count"] = sum("/spot_pathway_network/" in r["path"] for r in report["arrays"])
    report["merged_similarity_array_count"] = sum("/network_similarity/" in r["path"] for r in report["arrays"])
    report["passed"] = (report["metadata_equal"] and report["array_paths_equal"]
                        and report["nonempty_probability_arrays"] > 0 and all(r["passed"] for r in report["arrays"]))
    if short_name == "07_psoriasis":
        report["passed"] &= report["nonempty_spot_probability_arrays"] > 0
    write_json(ROOT / "validation" / f"tutorial_saved_objects_{short_name}.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=OBJECTS)
    parser.add_argument("--export", choices=("reference", "development"))
    parser.add_argument("--short-name", nargs="+", choices=OBJECTS)
    args = parser.parse_args()
    if args.export:
        if not args.short_name:
            parser.error("--export requires --short-name")
        for short_name in args.short_name:
            export(args.export, short_name)
            # Process only one saved object bundle at a time. Reuse imports
            # within one package/version, never mix two versions in a worker.
            gc.collect()
        return 0
    rows = []
    requested = args.only or list(OBJECTS)
    for label in ("reference", "development"):
        subprocess.run([sys.executable, "-B", str(Path(__file__)), "--export", label,
                        "--short-name", *requested], cwd=ROOT, check=True)
    for short_name in requested:
        rows.append(compare(short_name))
        print(json.dumps({"compared": short_name, "passed": rows[-1]["passed"],
                          "arrays": len(rows[-1]["arrays"])}), flush=True)
    report = {"scope": "Newly saved tutorial objects only; no historical cache or recomputation",
              "exact_array_equality_definition": "numpy.array_equal on finite values (legacy bitwise_equal field name)",
              "sparse_policy": "Compare native sparse format, shape, data, indices and indptr; never densify",
              "requested": args.only or list(OBJECTS), "tutorials": rows,
              "complete_four_saved_object_tutorials": len(rows) == 4,
              "passed": all(r["passed"] for r in rows)}
    suffix = "_selected" if args.only and len(args.only) != 4 else ""
    write_json(ROOT / "validation" / f"tutorial_saved_object_comparison{suffix}.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
