"""Transfer-level checks on a MuMDIA MBR experiment output (read-only).

Usage: python inspect_transfers.py <exp_dir> [<reference_exp_dir_same_inputs_no_transfers>]
"""
import csv
import json
import os
import sys

import blake3
import duckdb
import pyarrow.parquet as pq

exp = sys.argv[1].replace("\\", "/")
ref = sys.argv[2].replace("\\", "/") if len(sys.argv) > 2 else None
T = float(os.environ.get("QT", "0.01"))


def b3(path):
    h = blake3.blake3()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


m = json.load(open(f"{exp}/experiment_manifest.json"))
runs = m["experiment"]["runs"]
con = duckdb.connect()
con.execute(f"CREATE VIEW sc AS SELECT * FROM read_parquet('{exp}/scored_combined.parquet', file_row_number = true)")
con.execute(f"CREATE VIEW mb AS SELECT * FROM read_parquet('{exp}/scored_mbr.parquet', file_row_number = true)")
con.execute(f"CREATE VIEW tr AS SELECT * FROM read_parquet('{exp}/mbr_transferred.parquet', file_row_number = true)")

print("## mbr_transferred.parquet")
print("arrow schema:", pq.read_schema(f"{exp}/mbr_transferred.parquet"))
print("rows:", con.execute("SELECT count(*) FROM tr").fetchone()[0])
print("row order (candidate_id, source) first 10:", con.execute("SELECT source, candidate_id FROM tr ORDER BY file_row_number LIMIT 10").fetchall())
print("sorted by (source, candidate_id)?", con.execute(
    "SELECT bool_and(ok) FROM (SELECT (source, candidate_id) >= lag((source, candidate_id)) OVER (ORDER BY file_row_number) AS ok FROM tr) WHERE ok IS NOT NULL").fetchone()[0])
print("per source:", con.execute("SELECT source, count(*), min(transfer_q), max(transfer_q), min(rt_delta), max(rt_delta) FROM tr GROUP BY 1 ORDER BY 1").fetchall())
print("labels:", con.execute("SELECT label, count(*) FROM tr GROUP BY 1").fetchall())
print("duplicate (candidate_id, source) keys:", con.execute("SELECT count(*) FROM (SELECT candidate_id, source FROM tr GROUP BY ALL HAVING count(*) > 1)").fetchone()[0])
print("null counts:", con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE {c} IS NULL)" for c in
      ["candidate_id", "source", "peptidoform", "charge", "protein_group", "label", "expected_rt", "observed_rt", "rt_delta", "transfer_q"]) + " FROM tr").fetchone())

print("\n## tr vs scored_mbr flagged rows")
print("flagged in mb:", con.execute("SELECT count(*) FROM mb WHERE is_transferred").fetchone()[0])
print("tr keys not flagged in mb:", con.execute(
    "SELECT count(*) FROM tr LEFT JOIN mb ON mb.candidate_id = tr.candidate_id AND mb.source = tr.source AND mb.is_transferred WHERE mb.candidate_id IS NULL").fetchone()[0])
print("flagged mb keys not in tr:", con.execute(
    "SELECT count(*) FROM mb LEFT JOIN tr ON mb.candidate_id = tr.candidate_id AND mb.source = tr.source WHERE mb.is_transferred AND tr.candidate_id IS NULL").fetchone()[0])
print("mismatch transfer_q / peptidoform / charge / protein_group / label:", con.execute("""
    SELECT count(*) FILTER (WHERE mb.transfer_q <> tr.transfer_q),
           count(*) FILTER (WHERE mb.peptidoform <> tr.peptidoform),
           count(*) FILTER (WHERE mb.charge <> tr.charge),
           count(*) FILTER (WHERE mb.protein_group <> tr.protein_group),
           count(*) FILTER (WHERE mb.label <> tr.label)
    FROM tr JOIN mb USING (candidate_id, source)""").fetchone())
print("observed_rt == scored apex_rt of the transferred row:", con.execute(
    "SELECT count(*) FILTER (WHERE tr.observed_rt = mb.apex_rt), count(*) FROM tr JOIN mb USING (candidate_id, source)").fetchone())
print("expected_rt == apex_rt of the same candidate in the other run (identity RT map, 2 runs):", con.execute("""
    SELECT count(*) FILTER (WHERE tr.expected_rt = o.apex_rt), count(*) FROM tr JOIN mb o ON o.candidate_id = tr.candidate_id AND o.source <> tr.source""").fetchone())
print("rt_delta == abs(observed - expected):", con.execute(
    "SELECT count(*) FILTER (WHERE rt_delta = abs(observed_rt - expected_rt)), count(*) FROM tr").fetchone())
print("pre-MBR q of transferred rows (from scored_combined): q_value range / run_psm_q range / peptide_q / pg_q:", con.execute("""
    SELECT min(sc.q_value), max(sc.q_value), min(sc.run_psm_q), max(sc.run_psm_q), min(sc.peptide_q_value), max(sc.peptide_q_value), min(sc.pg_q_value), max(sc.pg_q_value)
    FROM tr JOIN sc USING (candidate_id, source)""").fetchone())
print("the other run's row of each transferred candidate: label/q_value range:", con.execute("""
    SELECT o.label, count(*), min(o.q_value), max(o.q_value), max(o.is_transferred::INT) FROM tr JOIN mb o ON o.candidate_id = tr.candidate_id AND o.source <> tr.source GROUP BY 1""").fetchall())
print("transferred rows with run_psm_q > t (pre-MBR):", con.execute(
    f"SELECT count(*) FROM tr JOIN sc USING (candidate_id, source) WHERE sc.run_psm_q > {T}").fetchone()[0])
print("transferred rows with q_value <= quant.q_threshold (pre-MBR; quant would take them anyway):", con.execute(
    f"SELECT count(*) FROM tr JOIN sc USING (candidate_id, source) WHERE sc.q_value <= {T}").fetchone()[0])
print("distinct (peptidoform, charge) among transfers:", con.execute("SELECT count(*) FROM (SELECT DISTINCT peptidoform, charge FROM tr)").fetchone()[0])
print("distinct protein_group among transfers:", con.execute("SELECT count(DISTINCT protein_group) FROM tr").fetchone()[0])

print("\n## per-run counts: native (scored_combined) vs scored_for_quant / split")
for i, r in enumerate(runs):
    nat = con.execute(f"SELECT count(*) FROM sc WHERE source = {i} AND label = 'target' AND run_psm_q <= {T}").fetchone()[0]
    aug = con.execute(f"SELECT count(*) FROM read_parquet('{exp}/{r}/scored.parquet') WHERE label = 'target' AND run_psm_q <= {T}").fetchone()[0]
    aug_or = con.execute(f"SELECT count(*) FROM read_parquet('{exp}/{r}/scored.parquet') WHERE label = 'target' AND (run_psm_q <= {T} OR is_transferred)").fetchone()[0]
    aug_not = con.execute(f"SELECT count(*) FROM read_parquet('{exp}/{r}/scored.parquet') WHERE label = 'target' AND run_psm_q <= {T} AND NOT is_transferred").fetchone()[0]
    ntr = con.execute(f"SELECT count(*) FROM read_parquet('{exp}/{r}/scored.parquet') WHERE is_transferred").fetchone()[0]
    print(f"  {r}: native run_psm_q PSMs {nat}; split run_psm_q {aug}; split run_psm_q AND NOT transferred {aug_not}; split (run_psm_q OR transferred) {aug_or}; transfers {ntr}")

print("\n## peptides.tsv")
with open(f"{exp}/peptides.tsv", encoding="utf-8", newline="") as fh:
    rdr = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = next(rdr)
    rows = [dict(zip(header, r)) for r in rdr]
con.execute("CREATE TEMP TABLE tsv AS SELECT * FROM (VALUES " + ",".join(
    "(%s, %d, %s, %s, %d, %s, %s)" % (
        "'" + r["precursor"].replace("'", "''") + "'", int(r["charge"]), "'" + r["is_transferred"] + "'",
        "'" + r["transfer_q"] + "'", int(r["n_runs"]), "'" + r["q_value"] + "'", "'" + r["score"] + "'") for r in rows)
    + ") t(peptidoform, charge, is_tr, tq, n_runs, q_cell, score_cell)")
# keys of transfers: printed? how?
res = con.execute(f"""
    WITH k AS (SELECT DISTINCT peptidoform, charge FROM tr),
    nat AS (SELECT DISTINCT peptidoform, charge FROM mb WHERE label = 'target' AND peptide_q_value <= {T})
    SELECT (nat.peptidoform IS NOT NULL) AS has_native_winner, tsv.is_tr, count(*)
    FROM k LEFT JOIN nat USING (peptidoform, charge) LEFT JOIN tsv USING (peptidoform, charge)
    GROUP BY ALL ORDER BY ALL""").fetchall()
print("transfer keys by (has native pep_q<=t row, TSV is_transferred cell):", res)
# expected TSV transferred-row values: the first transferred row in stable (peptide_q_value, file order)
res = con.execute(f"""
    WITH first_tr AS (
      SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY peptidoform, charge ORDER BY peptide_q_value, file_row_number) AS rk
                     FROM mb WHERE label = 'target' AND (is_transferred OR peptide_q_value <= {T})) WHERE rk = 1)
    SELECT count(*) FILTER (WHERE printf('%.6f', f.transfer_q) = tsv.tq AND printf('%.4f', f.score) = tsv.score_cell AND printf('%.6f', f.peptide_q_value) = tsv.q_cell),
           count(*)
    FROM tsv JOIN first_tr f USING (peptidoform, charge) WHERE tsv.is_tr = 'true'""").fetchone()
print("TSV transferred rows matching the first accepted row (stable sort by peptide_q_value) on transfer_q, score and q cells:", res)
# n_runs rule
res = con.execute(f"""
    WITH nr AS (SELECT peptidoform, charge, count(DISTINCT source) FILTER (WHERE run_psm_q <= {T} OR is_transferred) AS n_all,
                       count(DISTINCT source) FILTER (WHERE run_psm_q <= {T} AND NOT is_transferred) AS n_native,
                       count(DISTINCT source) FILTER (WHERE is_transferred) AS n_tr
                FROM mb WHERE label = 'target' GROUP BY ALL)
    SELECT count(*) FILTER (WHERE tsv.n_runs = nr.n_all), count(*) FILTER (WHERE tsv.n_runs <> nr.n_native), count(*) FILTER (WHERE nr.n_tr > 0), count(*)
    FROM tsv JOIN nr USING (peptidoform, charge)""").fetchone()
print("n_runs == count(DISTINCT source) with (run_psm_q<=t OR is_transferred) / rows where it differs from the native-only count / rows with a transferred run / rows:", res)
print("n_runs histogram (TSV):", con.execute("SELECT n_runs, is_tr, count(*) FROM tsv GROUP BY ALL ORDER BY ALL").fetchall())
# quantity cells of transferred rows
qcols = [f"quantity_{r}" for r in runs]
empty = {c: sum(1 for r in rows if r["is_transferred"] == "true" and r[c] == "") for c in qcols}
print("empty quantity cells on transferred TSV rows:", empty)

print("\n## proteins.tsv")
with open(f"{exp}/proteins.tsv", encoding="utf-8", newline="") as fh:
    rdr = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    pheader = next(rdr)
    prows = [dict(zip(pheader, r)) for r in rdr]
print("header:", pheader, "rows:", len(prows))
for r in prows[:5]:
    print("  ", r)
pg_native = con.execute(f"SELECT count(DISTINCT protein_group) FROM mb WHERE label = 'target' AND pg_q_value <= {T} AND protein_group <> ''").fetchone()[0]
print("native protein groups at pg_q_value <= t:", pg_native, "; TSV rows flagged transferred:", sum(1 for r in prows if r["is_transferred"] == "true"))
print("q_value cells of transferred protein rows:", sorted({r["q_value"] for r in prows if r["is_transferred"] == "true"}))
print("manifest report block:", {k: m["experiment"]["report"][k] for k in ("n_precursors", "n_protein_groups", "q_threshold")})

if ref:
    print("\n## byte comparison against", os.path.basename(ref))
    files = ["scored_combined.parquet", "lfq_maxlfq.parquet", "lfq_maxlfq.parquet.peptide.parquet", "lfq_maxlfq.parquet.precursor.parquet"]
    for r in runs:
        files += [f"{r}/peptide_quant.parquet", f"{r}/protein_group_quant.parquet", f"{r}/psms_competed.parquet", f"{r}/chromatograms.parquet"]
    for f in files:
        print(f"  {f}: identical={b3(f'{exp}/{f}') == b3(f'{ref}/{f}')}")
    for f in ["scored_mbr.parquet", "peptides.tsv", "proteins.tsv"] + [f"{r}/scored.parquet" for r in runs]:
        print(f"  {f}: identical={b3(f'{exp}/{f}') == b3(f'{ref}/{f}')}")
