"""Measure the server time of the viewer pages (M2) on a real run directory.

Each scenario runs in a fresh Python process. The app is built with ``create_app`` and
driven through Flask's test client: the page router callback is posted exactly as the
browser posts it, so the time includes building and serialising the page. Browser
rendering is not included.

Scenarios:

* ``open``: import, open the run, build the app, serve the shell and the overview;
* ``pages``: the identification browser and the precursor page of ``--candidates``
  accepted targets, each first visit (cold for that candidate) and a second visit
  (warm).

"Cold disk" means the operating system's file cache was purged for every file of the
run directory first (see purge.py).

Usage::

    python benchmarks/m2_ui_performance.py <run-dir> [--candidates 10] [--json out.json]
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
import json, sys, time
t_start = time.perf_counter()
from mumdia_viewer.runtime import configure_environment
configure_environment()
import numpy as np
import psutil
import pyarrow.parquet as pq
from mumdia_viewer.data import open_results
from mumdia_viewer.ui.app import create_app

proc = psutil.Process()
out = {"import_s": time.perf_counter() - t_start, "memory_mb": {}, "routes": []}


def mark(name):
    m = proc.memory_info()
    out["memory_mb"][name] = round(getattr(m, "private", m.rss) / 2**20)


run_dir, n, base = sys.argv[1], int(sys.argv[2]), "/bench/"
t = time.perf_counter()
rs = open_results(run_dir)
app = create_app(rs, url_base=base)
client = app.server.test_client()
out["open_and_build_s"] = time.perf_counter() - t
t = time.perf_counter()
assert client.get(base).status_code == 200
layout = client.get(base + "_dash-layout").get_json()
deps = client.get(base + "_dash-dependencies").get_json()
out["shell_s"] = time.perf_counter() - t
mark("shell")

router = next(d for d in deps if "page.children" in d["output"])


def outputs_of(spec):
    parts = spec.strip(".").split("...")
    return [dict(zip(("id", "property"), p.rsplit(".", 1))) for p in parts]


def route(path, search="", label=None):
    values = {("url", "pathname"): base + path, ("url", "search"): search,
              ("threshold", "data"): 0.01, ("scheme", "data"): "light", ("recent", "data"): []}
    body = {
        "output": router["output"],
        "outputs": outputs_of(router["output"]),
        "inputs": [{**i, "value": values.get((i["id"], i["property"]))} for i in router["inputs"]],
        "state": [{**s, "value": values.get((s["id"], s["property"]))} for s in router["state"]],
        "changedPropIds": ["url.pathname"],
    }
    t = time.perf_counter()
    r = client.post(base + "_dash-update-component", json=body)
    dt = time.perf_counter() - t
    ok = r.status_code == 200
    out["routes"].append({"page": label or path, "s": round(dt, 4), "ok": ok,
                          "bytes": len(r.data)})
    return dt, ok


route("", label="overview")
mark("overview")
if sys.argv[3] == "pages":
    route("identifications", label="identifications")
    mark("identifications")
    run = rs.runs[0]
    df = pq.read_table(run.artifact("psms_scored").path,
                       columns=["candidate_id", "label", "q_value"]).to_pandas()
    accepted = df[(df.label == "target") & (df.q_value <= 0.01)].candidate_id.to_numpy()
    picks = np.random.default_rng(0).choice(accepted, size=min(n, accepted.size), replace=False)
    run_name = run.name or ""
    for cid in picks:
        q = f"?run={run_name}&cid={int(cid)}"
        route("precursor", q, label="precursor first visit")
        route("precursor", q, label="precursor second visit")
    mark("precursors")
out["total_s"] = time.perf_counter() - t_start
print(json.dumps(out))
"""


def run_child(run_dir: Path, n: int, scenario: str, cache: Path) -> dict:
    env = {**os.environ, "MUMDIA_VIEWER_CACHE_DIR": str(cache)}
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(run_dir), str(n), scenario],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"benchmark child failed:\n{proc.stdout}\n{proc.stderr}")


def summarise(result: dict) -> dict:
    by_page: dict[str, list[float]] = {}
    for r in result["routes"]:
        by_page.setdefault(r["page"], []).append(r["s"])
    return {
        page: {"n": len(v), "median_s": sorted(v)[len(v) // 2], "max_s": max(v)}
        for page, v in by_page.items()
    }


def main() -> None:
    sys.path.insert(0, str(HERE))
    from purge import purge_tree

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--cache", type=Path, help="viewer cache directory (default: a temp dir)")
    args = ap.parse_args()
    cache = args.cache or Path(tempfile.mkdtemp(prefix="mv-bench-cache-"))
    results: dict = {"run_dir": str(args.run_dir), "cache": str(cache)}
    results["purged_files"] = purge_tree([args.run_dir])
    results["open_first_visit"] = run_child(args.run_dir, 0, "open", cache)
    purge_tree([args.run_dir])
    pages = run_child(args.run_dir, args.candidates, "pages", cache)
    results["pages_reopen"] = pages
    results["pages_summary"] = summarise(pages)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
