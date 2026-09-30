"""Inspect a MuMDIA run-experiment output directory for MBR facts (read-only).

Usage: python inspect_exp.py <exp_dir>
"""
import json
import math
import os
import sys

import blake3
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

exp = sys.argv[1].replace("\\", "/")


def b3(path):
    h = blake3.blake3()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def footer(path, label):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"\n### footer {label}: {os.path.relpath(path, exp)}")
    print(f"  created_by: {md.created_by!r}; format_version: {md.format_version}; rows: {md.num_rows}; row_groups: {md.num_row_groups}")
    print(f"  row group sizes: {[md.row_group(i).num_rows for i in range(md.num_row_groups)]}")
    kv = pf.schema_arrow.metadata or {}
    print(f"  key-value metadata keys: {[k.decode() for k in kv.keys()]}")
    comp = {md.row_group(0).column(j).compression for j in range(md.num_columns)} if md.num_row_groups else set()
    print(f"  compression: {sorted(comp)}")
    for i, f in enumerate(pf.schema_arrow):
        stats = []
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(i).statistics
            if st is None or not st.has_min_max:
                stats.append(None)
            else:
                stats.append((st.min, st.max, st.null_count))
        print(f"  {i+1:2d} {f.name:22s} {str(f.type):10s} nullable={f.nullable} stats(min,max,nulls)/rg={stats if len(str(stats)) < 160 else str(stats)[:160] + '...'}")
    return pf


m = json.load(open(f"{exp}/experiment_manifest.json"))
e = m["experiment"]
print("## manifest")
print("experiment.mbr:", e.get("mbr"))
print("model_identities.mbr:", m["model_identities"].get("mbr"))
print("experiment.scored_combined:", e.get("scored_combined"))
print("experiment.scored_for_quant:", e.get("scored_for_quant"))
print("experiment keys:", sorted(e.keys()))
print("artifact keys:", sorted(m["artifacts"].keys()))
sfq = m["artifacts"].get("scored_for_quant")
print("artifact scored_for_quant:", json.dumps(sfq))
for key, rec in sorted(m["artifacts"].items()):
    p = rec["path"]
    ok = os.path.exists(p)
    h = b3(p) if ok else None
    nrows = pq.ParquetFile(p).metadata.num_rows if ok and p.endswith(".parquet") else None
    print(f"  {key:26s} stage={rec['producing_stage']:16s} schema={rec['schema_name']} v{rec['schema_version']} rows={rec['rows']} file_rows={nrows} hash_ok={h == rec['content_hash']}")
print("report block:", json.dumps(e.get("report")))

sc_path = f"{exp}/scored_combined.parquet"
mbr_path = f"{exp}/scored_mbr.parquet"
tr_path = f"{exp}/mbr_transferred.parquet"
print("\nfiles present: scored_mbr", os.path.exists(mbr_path), "mbr_transferred", os.path.exists(tr_path))
for extra in ("scored_mbr.parquet.report.json", "mbr_transferred.parquet.report.json"):
    print(f"  {extra} exists:", os.path.exists(f"{exp}/{extra}"))

footer(sc_path, "scored_combined")
if os.path.exists(mbr_path):
    footer(mbr_path, "scored_mbr")
if os.path.exists(tr_path):
    footer(tr_path, "mbr_transferred")
runs = e["runs"]
for r in runs:
    footer(f"{exp}/{r}/scored.parquet", f"scored[{r}]")

con = duckdb.connect()
con.execute(f"CREATE VIEW sc AS SELECT * FROM read_parquet('{sc_path}', file_row_number = true)")
if os.path.exists(mbr_path):
    con.execute(f"CREATE VIEW mb AS SELECT * FROM read_parquet('{mbr_path}', file_row_number = true)")
    print("\n## scored_mbr content")
    print("rows sc / mb:", con.execute("SELECT (SELECT count(*) FROM sc), (SELECT count(*) FROM mb)").fetchall())
    print("is_transferred: true/false/null:", con.execute(
        "SELECT count(*) FILTER (WHERE is_transferred), count(*) FILTER (WHERE NOT is_transferred), count(*) FILTER (WHERE is_transferred IS NULL) FROM mb").fetchall())
    print("transfer_q: null / NaN / finite:", con.execute(
        "SELECT count(*) FILTER (WHERE transfer_q IS NULL), count(*) FILTER (WHERE isnan(transfer_q)), count(*) FILTER (WHERE isfinite(transfer_q)) FROM mb").fetchall())
    print("transfer_q null/NaN/finite on transferred rows:", con.execute(
        "SELECT count(*) FILTER (WHERE transfer_q IS NULL), count(*) FILTER (WHERE isnan(transfer_q)), count(*) FILTER (WHERE isfinite(transfer_q)) FROM mb WHERE is_transferred").fetchall())
    # row order and column equality against scored_combined
    cols = [f.name for f in pq.read_schema(sc_path)]
    print("column order mb[:21] == sc:", [f.name for f in pq.read_schema(mbr_path)][: len(cols)] == cols)
    lowered = {"q_value", "run_psm_q", "experiment_psm_q"}
    diffs = {}
    for c in cols:
        if c in lowered:
            q = f"SELECT count(*) FROM sc JOIN mb USING (file_row_number) WHERE NOT (sc.{c} IS NOT DISTINCT FROM mb.{c}) AND NOT mb.is_transferred"
        else:
            q = f"SELECT count(*) FROM sc JOIN mb USING (file_row_number) WHERE NOT (sc.{c} IS NOT DISTINCT FROM mb.{c})"
        diffs[c] = con.execute(q).fetchone()[0]
    print("value mismatches (row-aligned; lowered cols only on untransferred rows):", {k: v for k, v in diffs.items() if v} or "none")
    # NaN in scored_combined (float cols) that could become null
    fcols = [f.name for f in pq.read_schema(sc_path) if pa.types.is_floating(f.type)]
    nan_sc = {c: con.execute(f"SELECT count(*) FILTER (WHERE isnan({c})) FROM sc").fetchone()[0] for c in fcols}
    null_mb = {c: con.execute(f"SELECT count(*) FILTER (WHERE {c} IS NULL) FROM mb").fetchone()[0] for c in fcols}
    print("NaN counts in scored_combined float cols:", {k: v for k, v in nan_sc.items() if v} or "none")
    print("null counts in scored_mbr float cols:", {k: v for k, v in null_mb.items() if v} or "none")
    # transferred rows: all q columns
    qcols = ["q_value", "run_psm_q", "experiment_psm_q", "global_q_value", "peptide_q_value", "precursor_q", "pg_q_value"]
    rows = con.execute(
        "SELECT mb.source, mb.candidate_id, mb.peptidoform, mb.charge, mb.label, mb.transfer_q, "
        + ", ".join(f"sc.{c} AS sc_{c}, mb.{c} AS mb_{c}" for c in qcols)
        + " FROM mb JOIN sc USING (source, candidate_id) WHERE mb.is_transferred ORDER BY mb.source, mb.candidate_id").fetchall()
    print(f"\ntransferred rows: {len(rows)}")
    checks = {c: 0 for c in qcols}
    for r in rows:
        tq = r[5]
        vals = dict(zip([f"{p}_{c}" for c in qcols for p in ("sc", "mb")], r[6:]))
        for c in qcols:
            before, after = vals[f"sc_{c}"], vals[f"mb_{c}"]
            if c in lowered:
                expect = min(before, tq)
            else:
                expect = before
            if after != expect:
                checks[c] += 1
    print("q-column rule mismatches over transferred rows (lowered = min(q, transfer_q); others unchanged):", checks)
    for r in rows[:8]:
        print("  ", r[:6], "q_value %s->%s run_psm_q %s->%s exp_psm_q %s->%s global %s->%s pep %s->%s prec %s->%s pg %s->%s" % tuple(r[6:]))
    # other q stats
    print("rows where mb.q_value != mb.global_q_value:", con.execute("SELECT count(*) FROM mb WHERE q_value <> global_q_value").fetchone()[0])
    print("rows where mb.q_value != mb.experiment_psm_q:", con.execute("SELECT count(*) FROM mb WHERE q_value <> experiment_psm_q").fetchone()[0])

# per-run scored vs scored_for_quant subset
sfq_path = e["scored_for_quant"]
con.execute(f"CREATE VIEW sfq AS SELECT * FROM read_parquet('{sfq_path}', file_row_number = true)")
for i, r in enumerate(runs):
    p = f"{exp}/{r}/scored.parquet"
    con.execute(f"CREATE OR REPLACE VIEW rs AS SELECT * FROM read_parquet('{p}', file_row_number = true)")
    n_rs = con.execute("SELECT count(*) FROM rs").fetchone()[0]
    n_sub = con.execute(f"SELECT count(*) FROM sfq WHERE source = {i}").fetchone()[0]
    names = [f.name for f in pq.read_schema(p)]
    # exact equality of values, in order
    rs_rows = con.execute(f"SELECT * EXCLUDE (file_row_number) FROM rs ORDER BY file_row_number").fetchall()
    sub_rows = con.execute(f"SELECT {', '.join(names)} FROM sfq WHERE source = {i} ORDER BY file_row_number").fetchall()
    def norm(v):
        return "NaN" if isinstance(v, float) and math.isnan(v) else v
    same = [tuple(map(norm, a)) for a in rs_rows] == [tuple(map(norm, b)) for b in sub_rows]
    print(f"\nper-run {r}: rows {n_rs} == sfq source={i} rows {n_sub}: {n_rs == n_sub}; ordered values identical: {same}; columns={len(names)}")
    if "is_transferred" in names:
        print("  transferred rows in split:", con.execute("SELECT count(*) FILTER (WHERE is_transferred), count(*) FILTER (WHERE transfer_q IS NULL), count(*) FILTER (WHERE isnan(transfer_q)) FROM rs").fetchall())

# quant of transferred candidates
if os.path.exists(mbr_path):
    for i, r in enumerate(runs):
        pqp = f"{exp}/{r}/peptide_quant.parquet"
        con.execute(f"CREATE OR REPLACE VIEW pq{i} AS SELECT * FROM read_parquet('{pqp}')")
        print(f"\npeptide_quant[{r}] columns:", [f.name for f in pq.read_schema(pqp)])
        res = con.execute(f"""
            SELECT m.candidate_id, m.peptidoform, m.charge, m.q_value, m.transfer_q, q.quantity, q.quant_status, q.n_fragments_used
            FROM mb m LEFT JOIN pq{i} q USING (candidate_id)
            WHERE m.source = {i} AND m.is_transferred ORDER BY m.candidate_id""").fetchall()
        print(f"  transferred rows of run {r}: {len(res)}; with a peptide_quant row: {sum(1 for x in res if x[6] is not None)}")
        for x in res[:10]:
            print("   ", x)
        # quant gate check: rows in pq = targets with q_value<=thr or transferred
        gate = con.execute(f"""
            SELECT count(*) FROM rs_dummy
        """.replace("rs_dummy", f"(SELECT * FROM mb WHERE source = {i} AND label <> 'decoy' AND (q_value <= 0.01 OR is_transferred))")).fetchone()[0]
        npq = con.execute(f"SELECT count(*) FROM pq{i}").fetchone()[0]
        native = con.execute(f"SELECT count(*) FROM sc WHERE source = {i} AND label <> 'decoy' AND q_value <= 0.01").fetchone()[0]
        print(f"  peptide_quant rows {npq}; gate on scored_mbr (q_value<=0.01 OR is_transferred) {gate}; native gate on scored_combined {native}")

    # LFQ inclusion
    lfq = e["lfq"]
    con.execute(f"CREATE VIEW lfq_prec AS SELECT * FROM read_parquet('{lfq}.precursor.parquet')")
    con.execute(f"CREATE VIEW lfq_prot AS SELECT * FROM read_parquet('{lfq}')")
    print("\nLFQ precursor columns:", [f.name for f in pq.read_schema(lfq + '.precursor.parquet')])
    res = con.execute("""
        SELECT m.source, m.candidate_id, m.peptidoform, m.charge, m.protein_group, l.quantity AS lfq_prec
        FROM mb m LEFT JOIN lfq_prec l ON l."group" = m.peptidoform AND l.charge = m.charge AND l.run = m.source
        WHERE m.is_transferred ORDER BY 1, 2""").fetchall()
    print(f"transferred rows with precursor-LFQ value > 0: {sum(1 for x in res if x[5] and x[5] > 0)} of {len(res)}")
    for x in res[:10]:
        print("   ", x)
    res = con.execute("""
        SELECT m.source, m.protein_group, l.quantity, l.n_features
        FROM mb m LEFT JOIN lfq_prot l ON l.protein_group = m.protein_group AND l.run = m.source
        WHERE m.is_transferred ORDER BY 1, 2""").fetchall()
    print(f"transferred rows whose protein group has LFQ > 0 in that run: {sum(1 for x in res if x[2] and x[2] > 0)} of {len(res)}")

# TSVs
for tsv in ("peptides.tsv", "proteins.tsv"):
    p = f"{exp}/{tsv}"
    lines = open(p, encoding="utf-8").read().split("\n")
    header = lines[0].split("\t")
    body = [l.split("\t") for l in lines[1:] if l]
    print(f"\n{tsv}: header={header}; rows={len(body)}; trailing newline={lines[-1] == ''}")
    it = header.index("is_transferred"); tq = header.index("transfer_q")
    vals = {}
    for row in body:
        vals[(row[it], row[tq] != "")] = vals.get((row[it], row[tq] != ""), 0) + 1
    print("  (is_transferred, transfer_q non-empty) counts:", vals)
    for row in body:
        if row[it] == "true":
            print("   transferred row:", dict(zip(header, row)))
