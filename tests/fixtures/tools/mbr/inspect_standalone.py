"""Inspect the standalone `mumdia mbr` output on EXP (read-only on EXP; DuckDB projections only)."""
import time

import duckdb
import pyarrow.parquet as pq

E = "C:/Users/robbi/OneDrive/Desktop/res_v050_mumdia"
S = "C:/Users/robbi/AppData/Local/Temp/claude/C--Users-robbi-OneDrive-Documents-GitHub-visDIA/ecfc3918-c866-4e97-9df8-df3b2807a87f/scratchpad/tmp/G3_mbr/exp_standalone"
T = 0.01

for path in (f"{S}/scored_mbr.parquet", f"{S}/mbr_transferred.parquet", f"{E}/scored_combined.parquet"):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"\n### {path.split('/')[-2]}/{path.split('/')[-1]}")
    print("created_by:", md.created_by, "| rows:", md.num_rows, "| row groups:", md.num_row_groups, [md.row_group(i).num_rows for i in range(md.num_row_groups)])
    print("kv metadata:", [k.decode() for k in (pf.schema_arrow.metadata or {}).keys()])
    names = pf.schema_arrow.names
    print("columns:", [(f.name, str(f.type), f.nullable) for f in pf.schema_arrow])
    for col in ("source", "candidate_id"):
        if col in names:
            j = names.index(col)
            st = [md.row_group(i).column(j).statistics for i in range(md.num_row_groups)]
            print(f"  {col} stats per rg:", [(s.min, s.max) if s is not None and s.has_min_max else None for s in st])

con = duckdb.connect()
con.execute("SET threads = 8")
con.execute(f"CREATE VIEW sc AS SELECT * FROM read_parquet('{E}/scored_combined.parquet', file_row_number = true)")
con.execute(f"CREATE VIEW mb AS SELECT * FROM read_parquet('{S}/scored_mbr.parquet', file_row_number = true)")
con.execute(f"CREATE VIEW tr AS SELECT * FROM read_parquet('{S}/mbr_transferred.parquet', file_row_number = true)")
t0 = time.time()
print("\nis_transferred true/false/null:", con.execute("SELECT count(*) FILTER (WHERE is_transferred), count(*) FILTER (WHERE NOT is_transferred), count(*) FILTER (WHERE is_transferred IS NULL) FROM mb").fetchone())
print("transfer_q null / NaN / finite:", con.execute("SELECT count(*) FILTER (WHERE transfer_q IS NULL), count(*) FILTER (WHERE isnan(transfer_q)), count(*) FILTER (WHERE isfinite(transfer_q)) FROM mb").fetchone())
print("row alignment: rows with a different (candidate_id, source) at the same file_row_number:", con.execute(
    "SELECT count(*) FROM sc JOIN mb USING (file_row_number) WHERE sc.candidate_id <> mb.candidate_id OR sc.source <> mb.source").fetchone()[0])
fl = [f.name for f in pq.read_schema(f"{E}/scored_combined.parquet") if str(f.type) == "double"]
print("NaN in scored_combined double columns:", {c: n for c, n in zip(fl, con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE isnan({c}))" for c in fl) + " FROM sc").fetchone()) if n})
print("null in scored_mbr double columns (excluding transfer_q):", {c: n for c, n in zip(fl, con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE {c} IS NULL)" for c in fl) + " FROM mb").fetchone()) if n})
other = ["candidate_id", "peptidoform", "charge", "label", "protein", "base_peptide_id", "apex_rt", "elution_lo", "elution_hi", "score", "peptide_q_value", "protein_group", "pg_q_value", "global_q_value", "prelim_score", "precursor_q", "selected_peak_rank"]
res = con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE sc.{c} IS DISTINCT FROM mb.{c})" for c in other) + " FROM sc JOIN mb USING (file_row_number)").fetchone()
print("value mismatches in unlowered columns:", {c: n for c, n in zip(other, res) if n} or "none")
res = con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE sc.{c} IS DISTINCT FROM mb.{c} AND NOT mb.is_transferred)" for c in ("q_value", "run_psm_q", "experiment_psm_q")) + " FROM sc JOIN mb USING (file_row_number)").fetchone()
print("lowered columns changed on untransferred rows:", res)
res = con.execute("""
    SELECT count(*),
           count(*) FILTER (WHERE mb.q_value = least(sc.q_value, mb.transfer_q)),
           count(*) FILTER (WHERE mb.run_psm_q = least(sc.run_psm_q, mb.transfer_q)),
           count(*) FILTER (WHERE mb.experiment_psm_q = least(sc.experiment_psm_q, mb.transfer_q)),
           count(*) FILTER (WHERE mb.transfer_q < sc.q_value),
           count(*) FILTER (WHERE mb.transfer_q < sc.run_psm_q),
           count(*) FILTER (WHERE mb.q_value < sc.q_value),
           count(*) FILTER (WHERE mb.global_q_value = sc.q_value)
    FROM sc JOIN mb USING (file_row_number) WHERE mb.is_transferred""").fetchone()
print("transferred rows: n, q_value==min(q,tq), run_psm_q==min, experiment_psm_q==min, tq<q_value(pre), tq<run_psm_q(pre), q_value actually lowered, global_q_value==pre q_value:", res)
print("mb rows where q_value <> global_q_value:", con.execute("SELECT count(*) FROM mb WHERE q_value <> global_q_value").fetchone()[0],
      "| where q_value <> experiment_psm_q:", con.execute("SELECT count(*) FROM mb WHERE q_value <> experiment_psm_q").fetchone()[0])
print("pre-MBR q of transferred rows: q_value min/median/max; run_psm_q min/median/max:", con.execute(
    "SELECT min(sc.q_value), median(sc.q_value), max(sc.q_value), min(sc.run_psm_q), median(sc.run_psm_q), max(sc.run_psm_q) FROM sc JOIN mb USING (file_row_number) WHERE mb.is_transferred").fetchone())
print("transfer_q min/median/max:", con.execute("SELECT min(transfer_q), median(transfer_q), max(transfer_q) FROM tr").fetchone())
print("transferred rows with pre-MBR q_value <= t (would be quantified anyway):", con.execute(
    f"SELECT count(*) FROM sc JOIN mb USING (file_row_number) WHERE mb.is_transferred AND sc.q_value <= {T}").fetchone()[0])
print("transferred rows with pre-MBR run_psm_q <= t (natively accepted per run):", con.execute(
    f"SELECT count(*) FROM sc JOIN mb USING (file_row_number) WHERE mb.is_transferred AND sc.run_psm_q <= {T}").fetchone()[0])
print("labels of transferred rows:", con.execute("SELECT label, count(*) FROM mb WHERE is_transferred GROUP BY 1").fetchall())
print("peptide_q_value of transferred rows: <= t / = 1.0 / other:", con.execute(
    f"SELECT count(*) FILTER (WHERE peptide_q_value <= {T}), count(*) FILTER (WHERE peptide_q_value = 1.0), count(*) FILTER (WHERE peptide_q_value > {T} AND peptide_q_value < 1.0) FROM mb WHERE is_transferred").fetchone())

print("\nper run: native run_psm_q PSMs (scored_combined) | augmented run_psm_q | augmented AND NOT transferred | augmented OR transferred | transfers | transfers natively accepted")
for s in range(6):
    r = con.execute(f"""
      SELECT count(*) FILTER (WHERE sc.run_psm_q <= {T}),
             count(*) FILTER (WHERE mb.run_psm_q <= {T}),
             count(*) FILTER (WHERE mb.run_psm_q <= {T} AND NOT mb.is_transferred),
             count(*) FILTER (WHERE mb.run_psm_q <= {T} OR mb.is_transferred),
             count(*) FILTER (WHERE mb.is_transferred),
             count(*) FILTER (WHERE mb.is_transferred AND sc.run_psm_q <= {T})
      FROM sc JOIN mb USING (file_row_number) WHERE sc.source = {s} AND sc.label = 'target'""").fetchone()
    print(f"  source {s}: {r}")
r = con.execute(f"""
    SELECT count(*) FILTER (WHERE sc.q_value <= {T}), count(*) FILTER (WHERE mb.q_value <= {T}), count(*) FILTER (WHERE mb.q_value <= {T} OR mb.is_transferred)
    FROM sc JOIN mb USING (file_row_number) WHERE sc.label = 'target'""").fetchone()
print("pooled target PSMs at q_value <= t: native", r[0], "| augmented", r[1], "| augmented OR transferred (= per-run quant population total)", r[2])
# precursor-level unit counts are unaffected (precursor_q untouched) but TSV rows are
r = con.execute(f"""
    SELECT count(*) FROM (SELECT DISTINCT peptidoform, charge FROM mb WHERE label = 'target' AND (peptide_q_value <= {T} OR is_transferred))""").fetchone()
r0 = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT peptidoform, charge FROM sc WHERE label = 'target' AND peptide_q_value <= {T})").fetchone()
print("peptides.tsv rows that report would write: with transfers", r[0], "| native", r0[0])
r = con.execute(f"SELECT count(DISTINCT protein_group) FROM mb WHERE label = 'target' AND protein_group <> '' AND (pg_q_value <= {T} OR is_transferred)").fetchone()
r0 = con.execute(f"SELECT count(DISTINCT protein_group) FROM sc WHERE label = 'target' AND protein_group <> '' AND pg_q_value <= {T}").fetchone()
print("proteins.tsv rows that report would write: with transfers", r[0], "| native", r0[0])
# n_runs shift
r = con.execute(f"""
    WITH a AS (SELECT peptidoform, charge,
                 count(DISTINCT source) FILTER (WHERE run_psm_q <= {T} OR is_transferred) AS n_all,
                 count(DISTINCT source) FILTER (WHERE is_transferred) AS n_tr_runs
               FROM mb WHERE label = 'target' GROUP BY ALL),
         n AS (SELECT peptidoform, charge, count(DISTINCT source) FILTER (WHERE run_psm_q <= {T}) AS n_native FROM sc WHERE label = 'target' GROUP BY ALL)
    SELECT count(*) FILTER (WHERE n_all <> n_native), count(*) FILTER (WHERE n_tr_runs > 0), max(n_all - n_native) FROM a JOIN n USING (peptidoform, charge)""").fetchone()
print("precursor keys whose n_runs (with transfers) differs from the native per-run count / keys with a transfer / max shift:", r)
print("mbr_transferred sorted by (source, candidate_id)?", con.execute(
    "SELECT bool_and(ok) FROM (SELECT (source, candidate_id) >= lag((source, candidate_id)) OVER (ORDER BY file_row_number) AS ok FROM tr) WHERE ok IS NOT NULL").fetchone()[0],
    "| source non-decreasing?", con.execute("SELECT bool_and(ok) FROM (SELECT source >= lag(source) OVER (ORDER BY file_row_number) AS ok FROM tr) WHERE ok IS NOT NULL").fetchone()[0])
print("elapsed (indicative, CPU shared):", round(time.time() - t0, 1), "s")
