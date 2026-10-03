"""Measure the M1 performance targets on a real run directory.

Targets (SPEC, Astral single run): open under 5 s; precursor detail under 1 s warm and
under 3 s cold; memory under 2 GB.

Each scenario runs in a fresh Python process. "Cold" means the operating system's
file cache was purged for every file of the run directory first (see purge.py), so
the first reads come from disk. Two cold scenarios are measured:

* ``first_open``: an empty viewer cache, as on the first visit of a run (candidate
  indexes and the decoy-partner map are built);
* ``reopen``: the viewer cache from the first scenario, as on every later visit.

Usage::

    python benchmarks/m1_performance.py <run-dir> [--candidates 20] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHILD = r"""
import json, os, sys, time
t_start = time.perf_counter()
from mumdia_viewer.runtime import configure_environment
configure_environment()
import numpy as np
import psutil
from mumdia_viewer.runtime import configure_arrow
configure_arrow()
from mumdia_viewer.data import open_results
from mumdia_viewer.data import counts, overview
from mumdia_viewer.data.detail import precursor_detail, mirror, detail_percentiles
import pyarrow.parquet as pq

proc = psutil.Process()
out = {"import_s": time.perf_counter() - t_start, "memory_trace": {}}


def mark(name):
    m = proc.memory_info()
    out["memory_trace"][name] = round(getattr(m, "private", m.rss) / 2**20)


mark("imports")
run_dir, n = sys.argv[1], int(sys.argv[2])

t = time.perf_counter()
rs = open_results(run_dir)
out["open_results_s"] = time.perf_counter() - t
mark("open")

t = time.perf_counter()
summary = overview.run_summary(rs)
unit = counts.unit_counts(rs, 0.01)
per_run = counts.per_run_counts(rs, 0.01)
check = counts.engine_check(rs)
curves = {k: counts.id_curve(rs, k) for k in ("psm", "precursor", "peptide", "protein_group")}
hist = counts.score_histogram(rs)
timings = overview.stage_timings_table(rs)
out["overview_s"] = time.perf_counter() - t
mark("overview")
out["counts"] = {getattr(c.unit, "key", c.unit): c.n_target for c in unit}
out["engine_check_equal"] = all(bool(e.equal) for e in check if e.equal is not None)

run = rs.runs[0]
cols = ["candidate_id", "label", "q_value"]
scored = pq.read_table(run.artifact("psms_scored").path, columns=cols)
df = scored.to_pandas()
accepted = df[(df.label == "target") & (df.q_value <= 0.01)].candidate_id.to_numpy()
rng = np.random.default_rng(0)
picks = [int(c) for c in rng.choice(accepted, size=n + 1, replace=False)]

def one(cid):
    t0 = time.perf_counter()
    d = precursor_detail(rs, run, cid)
    t1 = time.perf_counter()
    m = mirror(rs, d)
    t2 = time.perf_counter()
    p = detail_percentiles(rs, d)
    t3 = time.perf_counter()
    return {"detail": t1 - t0, "mirror": t2 - t1, "percentiles": t3 - t2, "total": t3 - t0,
            "parts_ms": d.timings_ms}

first = one(picks[0])
out["detail_first"] = first
mark("first_detail")
warm = [one(c) for c in picks[1:]]
mark("warm_details")
from mumdia_viewer.data.pqio import ROW_GROUP_CACHE
from mumdia_viewer.data.spectra import SPECTRUM_CACHE
out["row_group_cache_mb"] = ROW_GROUP_CACHE.nbytes / 2**20
out["spectrum_cache_mb"] = getattr(SPECTRUM_CACHE, "nbytes", 0) / 2**20
repeat = one(picks[0])
tot = np.array([w["total"] for w in warm])
out["detail_warm_other_median_s"] = float(np.median(tot))
out["detail_warm_other_max_s"] = float(tot.max())
out["detail_warm_repeat_s"] = repeat["total"]
mem = proc.memory_info()
out["memory"] = {
    "rss_mb": mem.rss / 2**20,
    "private_mb": getattr(mem, "private", mem.rss) / 2**20,
    "peak_wset_mb": getattr(mem, "peak_wset", mem.rss) / 2**20,
}
out["open_plus_overview_s"] = out["open_results_s"] + out["overview_s"]
print("RESULT " + json.dumps(out))
"""


def run_child(run_dir: Path, n: int, cache: Path, pool: str | None = None) -> dict:
    env = dict(os.environ)
    env["MUMDIA_VIEWER_CACHE_DIR"] = str(cache)
    if pool:
        env["ARROW_DEFAULT_MEMORY_POOL"] = pool
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(run_dir), str(n)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise RuntimeError(f"benchmark child failed:\n{proc.stdout}\n{proc.stderr}")


def main() -> None:
    sys.path.insert(0, str(HERE))
    from purge import purge_tree

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--arrow-pool", default=None, help="ARROW_DEFAULT_MEMORY_POOL for the runs")
    args = ap.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="mumdia-viewer-bench-") as tmp:
        cache = Path(tmp) / "cache"
        purged = purge_tree([args.run_dir])
        results["first_open"] = run_child(args.run_dir, args.candidates, cache, args.arrow_pool)
        purge_tree([args.run_dir])
        results["reopen"] = run_child(args.run_dir, args.candidates, cache, args.arrow_pool)
        results["purged_files"] = purged
    print(json.dumps(results, indent=1))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    rows = [
        ("import (data layer)", "import_s"),
        ("open_results", "open_results_s"),
        ("open + overview data", "open_plus_overview_s"),
        ("first precursor detail + mirror + percentiles", None),
        ("warm detail, other candidates (median)", "detail_warm_other_median_s"),
        ("warm detail, other candidates (max)", "detail_warm_other_max_s"),
        ("warm detail, same candidate", "detail_warm_repeat_s"),
    ]
    print("\n| measurement | first open (empty viewer cache) | reopen (viewer cache) |")
    print("|---|---|---|")
    for label, key in rows:
        vals = []
        for scen in ("first_open", "reopen"):
            r = results[scen]
            v = r["detail_first"]["total"] if key is None else r[key]
            vals.append(f"{v * 1000:.0f} ms")
        print(f"| {label} | {vals[0]} | {vals[1]} |")
    for scen in ("first_open", "reopen"):
        m = results[scen]["memory"]
        print(
            f"| memory at end ({scen}) | private {m['private_mb']:.0f} MB, peak working set "
            f"{m['peak_wset_mb']:.0f} MB | |"
        )
        r = results[scen]
        print(
            f"  private MB by phase ({scen}): {r['memory_trace']}; row-group cache "
            f"{r['row_group_cache_mb']:.0f} MB; spectrum cache {r['spectrum_cache_mb']:.0f} MB"
        )


if __name__ == "__main__":
    main()
