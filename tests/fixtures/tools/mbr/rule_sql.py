"""Execute the data-layer SQL proposed in the facts file on real outputs (read-only)."""
import sys
import time
import duckdb

cases = [
    ("EXP + standalone mbr", "C:/Users/robbi/OneDrive/Desktop/res_v050_mumdia/scored_combined.parquet",
     "exp_standalone/mbr_transferred.parquet", "exp_standalone/scored_mbr.parquet", 0.01),
    ("fixture exp_mbr_c_tr_qt004", "exp_mbr_c_tr_qt004/scored_combined.parquet",
     "exp_mbr_c_tr_qt004/mbr_transferred.parquet", "exp_mbr_c_tr_qt004/scored_mbr.parquet", 0.004),
    ("fixture exp_mbr_c_cand0 (0-accepted, null-typed strings)", "exp_mbr_c_cand0/scored_combined.parquet",
     "exp_mbr_c_cand0/mbr_transferred.parquet", "exp_mbr_c_cand0/scored_mbr.parquet", 0.01),
]
PER_RUN = """
WITH s AS (SELECT source, candidate_id, label, run_psm_q FROM read_parquet($sc)),
     t AS (SELECT source, candidate_id FROM read_parquet($tr))
SELECT s.source,
       count(*) FILTER (WHERE s.label = 'target' AND s.run_psm_q <= $t)                 AS native_psms,
       count(t.candidate_id)                                                            AS transfers,
       count(t.candidate_id) FILTER (WHERE NOT (s.run_psm_q <= $t))                    AS added_by_mbr,
       count(*) FILTER (WHERE s.label = 'target'
                         AND (s.run_psm_q <= $t OR t.candidate_id IS NOT NULL))         AS native_or_transferred
FROM s LEFT JOIN t USING (source, candidate_id)
GROUP BY s.source ORDER BY s.source
"""
NRUNS = """
WITH s AS (SELECT source, candidate_id, peptidoform, charge, label, run_psm_q FROM read_parquet($sc)),
     t AS (SELECT source, candidate_id, true AS tr FROM read_parquet($tr)),
     x AS (SELECT s.*, coalesce(t.tr, false) AS tr, s.run_psm_q <= $t AS native
           FROM s LEFT JOIN t USING (source, candidate_id) WHERE s.label = 'target')
SELECT count(*) AS keys,
       count(*) FILTER (WHERE n_engine <> n_native) AS keys_shifted,
       sum(n_transfer_only) AS transfer_only_runs
FROM (SELECT peptidoform, charge,
             count(DISTINCT source) FILTER (WHERE native)            AS n_native,
             count(DISTINCT source) FILTER (WHERE tr AND NOT native) AS n_transfer_only,
             count(DISTINCT source) FILTER (WHERE native OR tr)      AS n_engine
      FROM x GROUP BY peptidoform, charge)
"""
CHECK_ENGINE_NRUNS = """
SELECT count(*) FROM (
  SELECT peptidoform, charge, count(DISTINCT source) FILTER (WHERE run_psm_q <= $t OR coalesce(is_transferred, false)) AS n
  FROM read_parquet($mb) WHERE label = 'target' GROUP BY ALL) a
JOIN (
  SELECT s.peptidoform, s.charge, count(DISTINCT s.source) FILTER (WHERE s.run_psm_q <= $t OR t.candidate_id IS NOT NULL) AS n
  FROM read_parquet($sc) s LEFT JOIN read_parquet($tr) t USING (source, candidate_id) WHERE s.label = 'target' GROUP BY ALL) b
USING (peptidoform, charge) WHERE a.n <> b.n
"""
con = duckdb.connect()
con.execute("SET threads = 8")
for name, sc, tr, mb, t in cases:
    cur = con.cursor()
    t0 = time.time()
    print(f"\n## {name} (t = {t})")
    for row in cur.execute(PER_RUN, {"sc": sc, "tr": tr, "t": t}).fetchall():
        print("  per run (source, native, transfers, added_by_mbr, native_or_transferred):", row)
    print("  n_runs decomposition (keys, keys_shifted, transfer_only_runs):", cur.execute(NRUNS, {"sc": sc, "tr": tr, "t": t}).fetchone())
    print("  engine n_runs rule on scored_for_quant vs scored_combined+mbr_transferred, mismatching keys:",
          cur.execute(CHECK_ENGINE_NRUNS, {"sc": sc, "tr": tr, "mb": mb, "t": t}).fetchone()[0])
    print(f"  elapsed {time.time() - t0:.2f} s (indicative; CPU shared with a background search)")
