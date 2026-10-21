"""Reproducible, isolated CellChatPy inference benchmarks.

All generated files stay below this task directory.  A benchmark is an inference
kernel benchmark: gene/interaction selection is fixed before timing and therefore
does not assert biological validity or measure an entire single-cell workflow.
Real PBMC68k data and explicitly synthetic sparse scale tests are kept separate.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "benchmarks"
PREFIX = "fresh_68k_pbmc_donor_a"
URL_BASE = f"https://cf.10xgenomics.com/samples/cell-exp/1.1.0/{PREFIX}/"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_package(source):
    source = str(Path(source).resolve())
    sys.path.insert(0, source)
    import CellChatPy as cc
    if not Path(cc.__file__).resolve().is_relative_to(Path(source)):
        raise RuntimeError(f"Incorrect package import: {cc.__file__}")
    return cc


def machine_info():
    import psutil
    import numpy as np
    import scipy
    versions = {}
    for name in ("numpy", "scipy", "pandas", "anndata", "numba", "scikit-learn", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "python": sys.version,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes_at_capture": psutil.virtual_memory().available,
        "executable": sys.executable,
        "package_versions": versions,
    }


def download(args):
    DATA.mkdir(exist_ok=True)
    manifest = []
    for suffix in ("filtered_gene_bc_matrices.tar.gz", "analysis.tar.gz"):
        name = f"{PREFIX}_{suffix}"
        path = DATA / name
        url = URL_BASE + name
        expected = {"filtered_gene_bc_matrices.tar.gz": 124442812,
                    "analysis.tar.gz": 24380463}[suffix]
        if not path.exists() or path.stat().st_size != expected:
            temporary = path.with_suffix(path.suffix + ".part")
            # Bounded HTTP ranges avoid stalled whole-file responses on this host.
            chunk_size = 1024 * 1024
            def fetch(offset):
                end = min(offset + chunk_size, expected) - 1
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                                "Range": f"bytes={offset}-{end}"})
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(request, timeout=45) as response:
                            data = response.read()
                            expected_range = f"bytes {offset}-{end}/{expected}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                                raise RuntimeError(f"Invalid HTTP range: {response.headers.get('Content-Range')}")
                            if len(data) != end - offset + 1:
                                raise RuntimeError("Incomplete HTTP range")
                        return offset, data
                    except Exception:
                        if attempt == 3:
                            raise
                raise AssertionError("unreachable")
            with temporary.open("wb") as out, concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                completed = 0
                futures = [executor.submit(fetch, offset) for offset in range(0, expected, chunk_size)]
                for future in concurrent.futures.as_completed(futures):
                    offset, data = future.result()
                    out.seek(offset)
                    out.write(data)
                    completed += len(data)
                    if completed % (16 * 1024 * 1024) == 0:
                        print(f"Downloaded {name}: {completed / 2**20:.0f} MiB", flush=True)
            if temporary.stat().st_size != expected:
                raise RuntimeError(f"Unexpected download size for {name}")
            temporary.replace(path)
        manifest.append({"filename": name, "bytes": path.stat().st_size,
                         "sha256": sha256(path), "url": url})
    write_json(DATA / "pbmc68k_download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def database_tables(source, species="human"):
    import pandas as pd
    folder = Path(source) / "CellChatPy" / "CellChatDB" / f"CellChatDB.{species}"
    return {key: pd.read_csv(folder / name, index_col=0).fillna("") for key, name in
            [("interaction", "interaction.csv"), ("complex", "complex.csv"),
             ("cofactor", "cofactor.csv"), ("gene_info", "geneInfo.csv")]}


def required_symbols(database):
    """Expand all ligand/receptor complexes and used cofactors, without importing the package."""
    result = set()
    interaction = database["interaction"]
    complex_table = database["complex"]
    cofactor = database["cofactor"]
    for col in ("ligand", "receptor"):
        for name in interaction[col].astype(str):
            if name in complex_table.index:
                result.update(str(x) for x in complex_table.loc[name] if str(x))
            elif name:
                result.add(name)
    for col in ("agonist", "antagonist", "co_A_receptor", "co_I_receptor"):
        if col in interaction:
            for name in interaction[col].astype(str):
                if name in cofactor.index:
                    result.update(str(x) for x in cofactor.loc[name] if str(x))
    return result


def prepare_pbmc(args):
    """Read MatrixMarket in chunks; store only signaling rows, normalize using all genes."""
    import numpy as np
    from scipy import sparse
    import pandas as pd
    prepare_started = time.perf_counter()
    database = database_tables(args.source)
    symbols = required_symbols(database)
    matrix_tar = DATA / f"{PREFIX}_filtered_gene_bc_matrices.tar.gz"
    analysis_tar = DATA / f"{PREFIX}_analysis.tar.gz"
    destination = DATA / "pbmc68k_signaling"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(matrix_tar, "r:gz") as archive:
        names = archive.getnames()
        gene_member = next(x for x in names if x.endswith("/genes.tsv"))
        barcode_member = next(x for x in names if x.endswith("/barcodes.tsv"))
        gene_rows = list(csv.reader(io.TextIOWrapper(archive.extractfile(gene_member)), delimiter="\t"))
        cells = [x.strip() for x in io.TextIOWrapper(archive.extractfile(barcode_member))]
        # Collapse repeated symbols by summing raw counts, before log normalization.
        genes = list(dict.fromkeys(row[1] for row in gene_rows if row[1] in symbols))
        mapping = {gene: i for i, gene in enumerate(genes)}
        gene_map = np.asarray([mapping.get(row[1], -1) for row in gene_rows], dtype=np.int32)
        matrix_member = next(x for x in names if x.endswith("/matrix.mtx"))
        stream = archive.extractfile(matrix_member)
        line = stream.readline()
        while line.startswith(b"%"):
            line = stream.readline()
        n_genes, n_cells, nnz = map(int, line.split())
        if n_genes != len(gene_rows) or n_cells != len(cells):
            raise RuntimeError("Matrix and metadata dimensions differ")
        totals = np.zeros(n_cells, dtype=np.float64)
        rows, columns, values = [], [], []
        processed = 0
        while lines := stream.readlines(8 * 1024 * 1024):
            triples = np.fromstring(b"".join(lines).decode("ascii"), sep=" ", dtype=np.int64).reshape(-1, 3)
            gi = triples[:, 0] - 1
            ci = triples[:, 1] - 1
            counts = triples[:, 2]
            totals += np.bincount(ci, weights=counts, minlength=n_cells)
            keep = gene_map[gi] >= 0
            rows.append(ci[keep].astype(np.int32))
            columns.append(gene_map[gi[keep]])
            values.append(counts[keep].astype(np.float64))
            processed += len(triples)
            if processed % 10000000 < len(triples):
                print(f"Parsed {processed:,}/{nnz:,} counts", flush=True)
        if processed != nnz:
            raise RuntimeError(f"Incomplete matrix: {processed} != {nnz}")
    expression = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                   shape=(n_cells, len(genes))).tocsr()
    del rows, columns, values
    factors = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    expression = expression.multiply(factors[:, None]).tocsr()
    np.log1p(expression.data, out=expression.data)
    with tarfile.open(analysis_tar, "r:gz") as archive:
        members = [x for x in archive.getnames() if x.endswith("clusters.csv")]
        target = next((x for x in members if "kmeans_10_clusters" in x or "/kmeans/10_clusters/" in x), None)
        if target is None:
            raise RuntimeError(f"Official k=10 cluster labels not found: {members}")
        labels = pd.read_csv(archive.extractfile(target), index_col=0).iloc[:, 0]
        aligned = labels.reindex(cells)
    official_coverage = int(aligned.notna().sum())
    aligned.rename("official_cluster_if_available").to_csv(destination / "official_partial_cluster_labels.csv")
    group_source = target
    clustering = None
    if aligned.isna().any():
        # The 2017 official analysis used a 40k subsample, while the matrix has
        # all 68,579 cells. Preserve every cell; do not invent official labels.
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        print(f"Official labels cover {official_coverage}/{n_cells}; computing fixed full-data benchmark clusters", flush=True)
        with threadpool_limits(limits=1):
            svd = TruncatedSVD(n_components=30, n_iter=5, random_state=42)
            projection = svd.fit_transform(expression)
            model = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=10, batch_size=4096)
            aligned = pd.Series(model.fit_predict(projection).astype(str), index=cells)
        np.savez_compressed(destination / "benchmark_clustering.npz", projection=projection,
                            components=svd.components_, centers=model.cluster_centers_)
        group_source = "fixed computational clusters on complete normalized signaling matrix; not biological annotations"
        clustering = {"projection": "TruncatedSVD", "n_components": 30, "n_iter": 5,
                      "clustering": "MiniBatchKMeans", "n_clusters": 10, "n_init": 10,
                      "batch_size": 4096, "random_state": 42}
    sparse.save_npz(destination / "expression.npz", expression)
    np.savez_compressed(destination / "metadata.npz", genes=np.asarray(genes), cells=np.asarray(cells),
                        groups=aligned.astype(str).to_numpy(dtype=str), library_sizes=totals)
    info = {"kind": "real", "dataset": "10x Genomics fresh 68k PBMC donor A",
            "original_shape_genes_cells": [n_genes, n_cells], "cells": n_cells,
            "signaling_genes": len(genes), "signaling_nnz": expression.nnz,
            "groups": int(aligned.nunique()), "group_source": group_source,
            "official_analysis_labeled_cells": official_coverage, "benchmark_clustering": clustering,
            "preparation_seconds": time.perf_counter() - prepare_started,
            "normalization": "log1p(10000 * raw_count / total raw UMI across all original genes)",
            "duplicate_gene_symbols": "sum raw counts before normalization",
            "scope": "all original cells; database signaling genes only; no cell duplication",
            "source_archives": [{"file": x.name, "sha256": sha256(x)} for x in (matrix_tar, analysis_tar)],
            "database_interaction_sha256": sha256(Path(args.source) / "CellChatPy/CellChatDB/CellChatDB.human/interaction.csv")}
    write_json(destination / "manifest.json", info)
    print(json.dumps(info, indent=2))


def build_object(args, cc):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    database = cc.load_database("human")
    if args.dataset == "pbmc68k":
        folder = DATA / "pbmc68k_signaling"
        expression = sparse.load_npz(folder / "expression.npz")
        meta = np.load(folder / "metadata.npz")
        genes, cells = meta["genes"], meta["cells"]
        # CellChat forbids the literal group label '0'; retain the cluster ID
        # while using explicit valid computational-group names in both engines.
        groups = np.asarray([f"benchmark_cluster_{value}" for value in meta["groups"]])
        if args.cells and args.cells < len(cells):
            # Deterministic real-data subset, never replication.
            indices = np.sort(np.random.RandomState(42).choice(len(cells), args.cells, replace=False))
            expression, cells, groups = expression[indices], cells[indices], groups[indices]
        kind = "real_full" if len(cells) == 68579 else "real_subset"
    elif args.dataset == "synthetic":
        n_cells = args.cells or 100000
        genes = np.asarray(sorted(required_symbols(database)))[:args.genes]
        rng = np.random.RandomState(12345)
        # Independent count draws; this is NOT an expanded or replicated biological atlas.
        per_cell = max(1, int(round(len(genes) * args.density)))
        rows = np.repeat(np.arange(n_cells, dtype=np.int32), per_cell)
        # State-specific enrichment makes triMean nonzero for some groups even
        # when the global matrix is sparse, so equality is not an all-zero test.
        module_count = min(args.groups, 10)
        module_width = max(1, len(genes) // module_count)
        active_module = (rows % args.groups) % module_count
        columns = (active_module * module_width + rng.randint(0, module_width, size=len(rows))).astype(np.int32)
        counts = rng.poisson(2.0, size=len(rows)).astype(np.float64) + 1.0
        expression = sparse.coo_matrix((counts, (rows, columns)), shape=(n_cells, len(genes))).tocsr()
        expression.data = np.log1p(expression.data)
        cells = np.asarray([f"synthetic_{i}" for i in range(n_cells)])
        groups = np.asarray([f"group_{i % args.groups}" for i in range(n_cells)])
        kind = "synthetic_independent_counts"
    else:
        raise ValueError(args.dataset)
    obs = pd.DataFrame({"benchmark_group": groups}, index=cells)
    obj = cc.create_cellchat(expression, metadata=obs, group_by="benchmark_group", genes=genes, cells=cells)
    obj.database = database
    obj.signaling = expression
    obj.var["is_signaling"] = True
    # All candidates are fixed by availability, with complete complexes required.
    # This isolates inference and does not claim DE-selected biological interactions.
    available = set(genes)
    complex_table = database["complex"]
    def present(name):
        if name in available:
            return True
        if name not in complex_table.index:
            return False
        subunits = [str(x) for x in complex_table.loc[name] if isinstance(x, str) and x]
        return bool(subunits) and all(x in available for x in subunits)
    interaction = database["interaction"]
    candidates = interaction[interaction["ligand"].map(present) & interaction["receptor"].map(present)].copy()
    if args.pairs:
        candidates = candidates.iloc[:args.pairs].copy()
    if not len(candidates):
        raise RuntimeError("No complete ligand-receptor candidates")
    metadata = {"kind": kind, "cells": obj.n_obs, "genes": obj.n_vars, "nnz": expression.nnz,
                "groups": len(obj.groups.categories), "pairs": len(candidates),
                "candidate_selection": "database order; complete ligand/receptor expression availability; no DE screening"}
    return obj, candidates, metadata


def worker(args):
    import psutil
    import numpy as np
    import threading
    cc = import_package(args.source)
    source_hashes = {name: sha256(Path(args.source) / "CellChatPy" / name)
                     for name in ("modeling.py", "_streaming.py", "_sparse_quantiles.py", "network_storage.py")
                     if (Path(args.source) / "CellChatPy" / name).exists()}
    obj, candidates, metadata = build_object(args, cc)
    memory_before = psutil.Process().memory_info().rss
    peak_rss = [memory_before]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.02):
            peak_rss[0] = max(peak_rss[0], psutil.Process().memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    kwargs = dict(lr_use=candidates, type_method=args.method, n_boot=args.n_boot,
                  seed_use=args.seed, population_size=args.population_size)
    if args.engine != "upstream":
        kwargs["engine"] = args.engine
        kwargs["gene_batch_size"] = args.gene_batch_size
        kwargs["lr_batch_size"] = args.lr_batch_size
    started = time.perf_counter()
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    try:
        cc.compute_communication_probability(obj, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        if args.profile:
            profiler.disable()
            profiler.dump_stats(str(Path(args.output).with_suffix(".prof")))
        peak_rss[0] = max(peak_rss[0], psutil.Process().memory_info().rss)
        stop.set()
        thread.join()
    from CellChatPy.network_storage import stack_network_field, network_names
    names = network_names(obj.network)
    prob = stack_network_field(obj.network, "prob", names=names)
    pval = stack_network_field(obj.network, "pval", names=names, fill_value=1.)
    if not np.isfinite(prob).all() or not np.isfinite(pval).all():
        raise RuntimeError("Nonfinite probability or p-value")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), prob=prob, pval=pval, names=np.asarray(names, dtype=str),
                        groups=np.asarray(obj.network["groups"], dtype=str))
    result = {**metadata, "status": "completed", "engine": args.engine,
              "package_file": cc.__file__, "inference_seconds": elapsed,
              "package_version": cc.__version__,
              "source_sha256_at_start": source_hashes,
              "inference_start_rss_bytes": memory_before, "inference_peak_rss_bytes": peak_rss[0],
              "inference_peak_increment_bytes": peak_rss[0] - memory_before,
              "method": args.method, "n_boot": args.n_boot, "seed": args.seed,
              "timing_mode": "first inference call in a fresh process; includes JIT initialization/cache loading",
              "gene_batch_size": args.gene_batch_size, "lr_batch_size": args.lr_batch_size,
              "prob_sum": float(prob.sum()), "nonzero_prob": int(np.count_nonzero(prob)),
              "pval_unique": np.unique(pval).tolist(), "settings": obj.settings,
              "machine": machine_info()}
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ("engine", "cells", "genes", "pairs", "inference_seconds", "inference_peak_rss_bytes")}))


def compare_arrays(left, right):
    import numpy as np
    a, b = np.load(left), np.load(right)
    result = {"names_equal": np.array_equal(a["names"], b["names"]),
              "groups_equal": np.array_equal(a["groups"], b["groups"])}
    for field in ("prob", "pval"):
        same_shape = a[field].shape == b[field].shape
        result[field] = {"same_shape": same_shape}
        if same_shape:
            delta = np.abs(a[field] - b[field])
            result[field].update(exact_equal=bool(np.array_equal(a[field], b[field])),
                                 allclose_1e_12=bool(np.allclose(a[field], b[field], rtol=1e-12, atol=1e-12)),
                                 max_abs_difference=float(delta.max(initial=0)),
                                 unequal_entries=int(np.count_nonzero(delta)),
                                 entries=int(delta.size))
    result["pass"] = bool(result["names_equal"] and result["groups_equal"] and
                          result["prob"].get("allclose_1e_12", False) and
                          result["pval"].get("exact_equal", False))
    result["acceptance_rule"] = "aligned axes; probability allclose at 1e-12; P values exactly equal"
    return result


def suite(args):
    import psutil
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.dataset}_{args.cells or 'full'}_{args.method}_b{args.n_boot}"
    folder = RESULTS / tag
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "machine.json", machine_info())
    records = []
    for repeat in range(args.repeats):
        # Alternating execution order reduces systematic warm-cache bias.
        engines = ["upstream", "streaming"] if repeat % 2 == 0 else ["streaming", "upstream"]
        for engine in engines:
            output = folder / f"{engine}_{repeat}.json"
            source = ROOT / ("upstream" if engine == "upstream" else "development")
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--source", str(source),
                       "--output", str(output), "--engine", engine, "--dataset", args.dataset,
                       "--cells", str(args.cells), "--genes", str(args.genes), "--groups", str(args.groups),
                       "--density", str(args.density), "--pairs", str(args.pairs), "--method", args.method,
                       "--n-boot", str(args.n_boot), "--seed", str(args.seed),
                       "--gene-batch-size", str(args.gene_batch_size), "--lr-batch-size", str(args.lr_batch_size)]
            if args.population_size:
                command.append("--population-size")
            if args.profile:
                command.append("--profile")
            environment = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(args.threads)
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            started = time.monotonic()
            peak, peak_private, stop_reason = 0, 0, None
            next_progress = 60.0
            with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=ROOT, env=environment)
                monitor = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        processes = [monitor] + monitor.children(recursive=True)
                        memories = [p.memory_info() for p in processes if p.is_running()]
                        current = sum(m.rss for m in memories)
                        peak = max(peak, current)
                        peak_private = max(peak_private, sum(getattr(m, "private", m.rss) for m in memories))
                    except psutil.NoSuchProcess:
                        break
                    if time.monotonic() - started > args.timeout:
                        stop_reason = "timeout"
                    if peak > args.max_rss_gb * 2**30:
                        stop_reason = "memory_budget_exceeded"
                    if peak_private > args.max_private_gb * 2**30:
                        stop_reason = "private_memory_budget_exceeded"
                    if stop_reason:
                        for child in monitor.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        process.kill()
                        process.wait()
                        break
                    elapsed = time.monotonic() - started
                    if elapsed >= next_progress:
                        print(f"{tag}: {engine} repeat={repeat} running {elapsed:.0f}s "
                              f"peak RSS={peak/2**20:.1f}MiB private={peak_private/2**20:.1f}MiB", flush=True)
                        next_progress += 60.0
                    time.sleep(0.1)
            process.wait()
            result = json.loads(output.read_text(encoding="utf-8")) if output.exists() and process.returncode == 0 else {}
            result.update(engine=engine, repeat=repeat, command=command, returncode=process.returncode,
                          process_seconds=time.monotonic() - started, process_peak_rss_bytes=peak,
                          process_peak_private_bytes=peak_private,
                          threads=args.threads,
                          status=stop_reason or ("completed" if process.returncode == 0 else "failed"))
            write_json(output, result)
            records.append(result)
            print(f"{tag}: {engine} repeat={repeat} status={result['status']} process={result['process_seconds']:.2f}s peak={peak/2**20:.1f}MiB", flush=True)
    comparisons = []
    for repeat in range(args.repeats):
        files = [folder / f"{engine}_{repeat}.npz" for engine in ("upstream", "streaming")]
        statuses = [r["status"] for r in records if r["repeat"] == repeat]
        if all(x.exists() for x in files) and statuses == ["completed", "completed"]:
            comparisons.append({"repeat": repeat, **compare_arrays(*files)})
    summary = {"benchmark_scope": "inference kernel; preprocessing excluded from inference_seconds, included in process_seconds",
               "dataset": args.dataset, "records": records, "comparisons": comparisons,
               "all_comparisons_pass": len(comparisons) == args.repeats and all(x["pass"] for x in comparisons),
               "limits": {"timeout_seconds": args.timeout, "max_rss_gb": args.max_rss_gb,
                          "max_private_gb": args.max_private_gb},
               "caveat": "Synthetic cases are independent simulated counts, not biological atlas validation."}
    write_json(folder / "summary.json", summary)
    print(json.dumps({"summary": str(folder / "summary.json"), "all_comparisons_pass": summary["all_comparisons_pass"]}))


def report(args):
    import statistics
    summaries = []
    for path in sorted(RESULTS.glob("*/summary.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        summaries.append((path.parent.name, result))
    lines = ["# Inference benchmark results", "",
             "Only completed equal-parameter runs receive a speed ratio. Times are medians across recorded repeats.",
             "Inference includes first-call JIT initialization/cache loading; process wall time also includes imports and setup.",
             "Real PBMC and synthetic scale experiments are separate. Synthetic cells are not a biological atlas.", "",
             "| Experiment | Cells | Genes | LR pairs | Bootstraps | Upstream inference s | Streaming inference s | Kernel speed ratio | Upstream / streaming process s | Upstream / streaming peak RSS MiB | Exact prob+pval |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    details = []
    for name, summary in summaries:
        completed = [x for x in summary["records"] if x["status"] == "completed"]
        if not completed:
            details.append(f"- {name}: no completed workers; see summary and logs.")
            continue
        sample = completed[0]
        medians = {}
        for engine in ("upstream", "streaming"):
            rows = [x for x in completed if x["engine"] == engine]
            medians[engine] = {field: statistics.median(x[field] for x in rows) if rows else None
                               for field in ("inference_seconds", "process_seconds", "process_peak_rss_bytes")}
        u, s = medians["upstream"], medians["streaming"]
        verified = summary.get("all_comparisons_pass", False)
        ratio = u["inference_seconds"] / s["inference_seconds"] if verified and u["inference_seconds"] and s["inference_seconds"] else None
        exact = verified and all(x[field].get("exact_equal", False)
                                 for x in summary["comparisons"] for field in ("prob", "pval"))
        fmt = lambda value: f"{value:.3f}" if value is not None else "—"
        memory = lambda value: fmt(value / 2**20) if value is not None else "—"
        lines.append(f"| {name} | {sample['cells']:,} | {sample['genes']:,} | {sample['pairs']:,} | {sample['n_boot']} | "
                     f"{fmt(u['inference_seconds'])} | {fmt(s['inference_seconds'])} | {fmt(ratio)} | "
                     f"{fmt(u['process_seconds'])} / {fmt(s['process_seconds'])} | "
                     f"{memory(u['process_peak_rss_bytes'])} / {memory(s['process_peak_rss_bytes'])} | "
                     f"{'yes' if exact else 'not established'} |")
        counts = {engine: sum(x["engine"] == engine for x in completed) for engine in ("upstream", "streaming")}
        failures = [f"{x['engine']} repeat {x['repeat']}: {x['status']}" for x in summary["records"] if x["status"] != "completed"]
        details.append(f"- {name}: completed repeats {counts}; " + ("; ".join(failures) if failures else "all requested workers completed.")
                       + f" [raw summary]({name}/summary.json)")
    lines += ["", "## Completion and limitations", "", *details, "",
              "Peak process-tree RSS is sampled at 100 ms. Per-inference RSS is separately available in worker JSON.",
              "This host has limited free memory; paging and background activity may affect absolute time. Paging I/O was not measured.",
              "The early synthetic5000 pilot predates full source-hash recording; later workers record four source file hashes and package version.",
              "Raw data provenance and full-cell clustering are documented in ../../data/pbmc68k_signaling/manifest.json.",
              "Candidate interactions in these kernel benchmarks use complete ligand/receptor availability, without DE screening.",
              "Original tutorial workflows are validated separately; do not treat kernel speed ratios as end-to-end pipeline ratios."]
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(RESULTS / "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("download")
    sub.add_parser("report")
    prepare = sub.add_parser("prepare-pbmc")
    prepare.add_argument("--source", type=Path, default=ROOT / "upstream")
    for action in ("worker", "suite"):
        p = sub.add_parser(action)
        p.add_argument("--dataset", choices=("pbmc68k", "synthetic"), default="pbmc68k")
        p.add_argument("--cells", type=int, default=0)
        p.add_argument("--genes", type=int, default=256)
        p.add_argument("--groups", type=int, default=10)
        p.add_argument("--density", type=float, default=.05)
        p.add_argument("--pairs", type=int, default=0)
        p.add_argument("--n-boot", type=int, default=100)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--method", default="triMean")
        p.add_argument("--population-size", action="store_true")
        p.add_argument("--profile", action="store_true")
        p.add_argument("--gene-batch-size", type=int, default=64)
        p.add_argument("--lr-batch-size", type=int, default=64)
        if action == "worker":
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--engine", choices=("upstream", "streaming", "legacy"), required=True)
        else:
            p.add_argument("--timeout", type=float, default=600)
            p.add_argument("--max-rss-gb", type=float, default=6)
            p.add_argument("--max-private-gb", type=float, default=6)
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--tag")
    args = parser.parse_args()
    try:
        {"download": download, "prepare-pbmc": prepare_pbmc, "worker": worker, "suite": suite, "report": report}[args.action](args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
