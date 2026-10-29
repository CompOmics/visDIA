"""Cut a real diaPASEF MuMDIA run down to the committed fixture ``tests/fixtures/im/out_tims``.

The source is a full single-run output of MuMDIA >= 0.7 on a Bruker diaPASEF ``.d``
(``features.im_features`` on, so chromatograms v4 carry per-point 1/K0). The cut keeps:

* the candidates whose scored apex falls in ``[t0, t0 + span)``: up to ``n_accepted``
  accepted targets (``q_value <= 0.01``), ``n_decoy`` decoys and ``n_other`` other targets,
  sampled with a fixed seed;
* every candidate-keyed row of those candidates (library precursors and fragments,
  run_windows, seed, extracted and its peaks, chromatograms, features, competed, scored,
  quant), with ``candidate_id`` renumbered ``0..n-1`` in the original order, so the dense
  tables (``candidate_id == row``) stay dense;
* the MS1 and MS2 scans in ``[t0 - pad, t0 + span + pad]``, each trimmed to its most
  intense peaks (``top_ms2``, ``top_ms1``; the per-peak 1/K0 stays aligned);
* the protein groups, peptides.tsv and proteins.tsv rows of the kept quant rows.

The manifest and the reports get the new row counts and a null ``content_hash`` (the
viewer then keys derived data by a footer fingerprint). ``rt_bootstrap/`` and
``sidecar_work/`` are not copied. Scores, q values and cal.json are the full run's: the
fixture tests the viewer against the parquet values, not the engine.

Usage: python subset_run.py <source run dir> <output dir> [--t0 2200 --span 15 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Tables keyed by candidate_id, by file name (the manifest names the library file).
KEYED = (
    "fragment_library_fragments.parquet",
    "fragment_library_precursors.parquet",
    "fragment_library_precursors_multihead.parquet",
    "run_windows.parquet",
    "seed_psms.parquet",
    "psms_extracted.parquet",
    "psms_extracted.parquet.peaks.parquet",
    "chromatograms.parquet",
    "features.parquet",
    "psms_competed.parquet",
    "psms_scored.parquet",
    "fragment_quant.parquet",
    "peptide_quant.parquet",
)
COPIED = (
    "cal.json",
    "features.parquet.schema.json",
    "psms_competed.parquet.schema.json",
    "seed_psms.parquet.masscal.json",
    "fragment_library_precursors_multihead.parquet.summary.json",
)


def choose(src: Path, t0: float, span: float, n: dict[str, int], seed: int) -> np.ndarray:
    """The kept original candidate ids, ascending."""
    s = pq.read_table(src / "psms_scored.parquet", columns=["candidate_id", "apex_rt", "label", "q_value"])
    s = s.to_pandas()
    s = s[(s.apex_rt >= t0) & (s.apex_rt < t0 + span)]
    target = s.label == "target"
    groups = {
        "accepted": s[target & (s.q_value <= 0.01)],
        "decoy": s[~target],
        "other": s[target & (s.q_value > 0.01)],
    }
    rng = np.random.default_rng(seed)
    kept = []
    for name, g in groups.items():
        ids = g.candidate_id.to_numpy(dtype=np.int64)
        kept.append(rng.choice(ids, size=min(n[name], ids.size), replace=False))
    return np.unique(np.concatenate(kept))


def renumber(table: pa.Table, old: np.ndarray) -> pa.Table:
    """The rows of ``old`` candidates, in file order, with ``candidate_id`` -> rank in ``old``."""
    ids = table.column("candidate_id").to_numpy().astype(np.int64)
    pos = np.searchsorted(old, ids)
    pos[pos >= old.size] = 0
    keep = old[pos] == ids
    out = table.filter(pa.array(keep))
    new = pa.array(pos[keep], type=table.schema.field("candidate_id").type)
    return out.set_column(table.schema.get_field_index("candidate_id"), "candidate_id", new)


def top_peaks(table: pa.Table, top: int) -> pa.Table:
    """Each scan's ``top`` most intense peaks, m/z ascending; ``im`` stays aligned."""
    lists = [c for c in ("mz", "intensity", "im") if c in table.column_names]
    cols = {c: table.column(c).to_pylist() for c in lists}
    for i in range(table.num_rows):
        inten = np.asarray(cols["intensity"][i], dtype=np.float32)
        if inten.size <= top:
            continue
        keep = np.sort(np.argsort(-inten, kind="stable")[:top])
        for c in lists:
            cols[c][i] = np.asarray(cols[c][i], dtype=np.float32)[keep].tolist()
    for c in lists:
        field = table.schema.field(c)
        table = table.set_column(table.schema.get_field_index(c), field, pa.array(cols[c], type=field.type))
    return table


def write(table: pa.Table, path: Path, rows: dict[str, int], row_group_size: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size, compression="zstd")
    rows[str(path)] = table.num_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--t0", type=float, default=2200.0)
    ap.add_argument("--span", type=float, default=15.0)
    ap.add_argument("--pad", type=float, default=15.0, help="seconds of spectra kept beyond the slice")
    ap.add_argument("--n-accepted", type=int, default=250)
    ap.add_argument("--n-decoy", type=int, default=250)
    ap.add_argument("--n-other", type=int, default=150)
    ap.add_argument("--top-ms2", type=int, default=200)
    ap.add_argument("--top-ms1", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src, out = a.src, a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows: dict[str, int] = {}

    old = choose(src, a.t0, a.span, {"accepted": a.n_accepted, "decoy": a.n_decoy, "other": a.n_other}, a.seed)
    for name in KEYED:
        if (src / name).exists():
            write(renumber(pq.read_table(src / name), old), out / name, rows)

    # Quant above the precursor, and the TSV reports, follow the kept peptide rows.
    pep = pq.read_table(out / "peptide_quant.parquet")
    groups = pa.array(sorted(set(pep.column("protein_group").to_pylist()) - {None}))
    pg = pq.read_table(src / "protein_group_quant.parquet")
    write(pg.filter(pc.is_in(pg.column("protein_group"), groups)), out / "protein_group_quant.parquet", rows)
    precursors = set(pq.read_table(out / "psms_scored.parquet", columns=["peptidoform"]).column(0).to_pylist())
    for tsv, key, keep in (("peptides.tsv", "precursor", precursors), ("proteins.tsv", "protein_group", set(groups.to_pylist()))):
        lines = (src / tsv).read_text().splitlines(keepends=True)
        col = lines[0].rstrip("\n").split("\t").index(key)
        (out / tsv).write_text("".join([lines[0], *(ln for ln in lines[1:] if ln.split("\t")[col] in keep)]))

    # Spectra: the scans around the slice, trimmed to their top peaks.
    lo, hi = a.t0 - a.pad, a.t0 + a.span + a.pad
    sp, so = src / "spectra", out / "spectra"
    kept_scans: dict[str, set[int]] = {}
    for name, top in (("spectra_ms2.parquet", a.top_ms2), ("spectra_ms1.parquet", a.top_ms1)):
        t = pq.read_table(sp / name)
        rt = t.column("rt_seconds").to_numpy()
        t = top_peaks(t.filter(pa.array((rt >= lo) & (rt <= hi))), top)
        kept_scans[name] = set(t.column("scan_index").to_pylist())
        write(t, so / name, rows, row_group_size=256)
    link = pq.read_table(sp / "ms2_to_ms1.parquet")
    ms2 = np.isin(link.column("ms2_scan_index").to_numpy(), list(kept_scans["spectra_ms2.parquet"]))
    ms1 = np.isin(link.column("ms1_scan_index").to_numpy(), list(kept_scans["spectra_ms1.parquet"]))
    write(link.filter(pa.array(ms2 & ms1)), so / "ms2_to_ms1.parquet", rows)
    write(pq.read_table(sp / "isolation_windows.parquet"), so / "isolation_windows.parquet", rows)

    for name in COPIED:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    # The manifest and the reports: new row counts, no content hash.
    manifest = json.loads((src / "manifest.json").read_text())
    # The directory the engine wrote to (the manifest paths are absolute).
    root = Path(os.path.commonpath([r["path"] for r in manifest["artifacts"].values()]))
    for key in list(manifest["artifacts"]):
        rec = manifest["artifacts"][key]
        rel = Path(rec["path"]).relative_to(root)
        new = out / rel
        if str(new) not in rows:
            del manifest["artifacts"][key]  # not copied (rt_bootstrap)
            continue
        rec["rows"] = rows[str(new)]
        rec["content_hash"] = None
        report = src / (str(rel) + ".report.json")
        if report.exists():
            r = json.loads(report.read_text())
            r["rows"], r["content_hash"] = rec["rows"], None
            (out / (str(rel) + ".report.json")).write_text(json.dumps(r, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for name in ("fragment_library_precursors.parquet", "psms_competed.parquet"):
        report = src / (name + ".report.json")  # artifacts the manifest names by another file
        if report.exists() and not (out / (name + ".report.json")).exists() and str(out / name) in rows:
            r = json.loads(report.read_text())
            r["rows"], r["content_hash"] = rows[str(out / name)], None
            (out / (name + ".report.json")).write_text(json.dumps(r, indent=2) + "\n")
    print(f"{old.size} candidates; {sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
