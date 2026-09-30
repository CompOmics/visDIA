"""Compare quant/LFQ between a transfer arm and its no-transfer reference (same inputs)."""
import sys
import duckdb
tr, ref = sys.argv[1], sys.argv[2]
con = duckdb.connect()
for i, r in enumerate(("a", "c")):
    con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM '{tr}/{r}/peptide_quant.parquet'")
    con.execute(f"CREATE OR REPLACE VIEW f AS SELECT * FROM '{ref}/{r}/peptide_quant.parquet'")
    con.execute(f"CREATE OR REPLACE VIEW s AS SELECT * FROM '{tr}/{r}/scored.parquet'")
    added = con.execute("SELECT candidate_id FROM t EXCEPT SELECT candidate_id FROM f").fetchall()
    removed = con.execute("SELECT candidate_id FROM f EXCEPT SELECT candidate_id FROM t").fetchall()
    trs = con.execute("SELECT candidate_id FROM s WHERE is_transferred").fetchall()
    print(f"run {r}: peptide_quant rows tr={con.execute('SELECT count(*) FROM t').fetchone()[0]} ref={con.execute('SELECT count(*) FROM f').fetchone()[0]}; added={len(added)} removed={len(removed)}; added == transferred set: {set(added) == set(trs)}")
    print("   shared rows with different values:", con.execute(
        "SELECT count(*) FROM t JOIN f USING (candidate_id) WHERE NOT (t.quantity IS NOT DISTINCT FROM f.quantity AND t.quant_status = f.quant_status AND t.n_fragments_used = f.n_fragments_used AND t.integration_apex_rt IS NOT DISTINCT FROM f.integration_apex_rt AND t.integration_lo_rt IS NOT DISTINCT FROM f.integration_lo_rt AND t.integration_hi_rt IS NOT DISTINCT FROM f.integration_hi_rt)").fetchone()[0])
    print("   row order of tr table == scored order of gated rows:", con.execute(
        "SELECT count(*) FROM (SELECT candidate_id, row_number() OVER () AS k FROM t) a JOIN (SELECT candidate_id, row_number() OVER () AS k FROM (SELECT candidate_id FROM read_parquet('" + f"{tr}/{r}/scored.parquet" + "', file_row_number=true) WHERE label <> 'decoy' AND (q_value <= 0.004 OR is_transferred) ORDER BY file_row_number)) b USING (k) WHERE a.candidate_id <> b.candidate_id").fetchone()[0], "mismatches")
    print("   statuses of added rows:", con.execute("SELECT quant_status, count(*) FROM t WHERE candidate_id IN (SELECT candidate_id FROM s WHERE is_transferred) GROUP BY 1").fetchall())
    print("   protein_group_quant rows tr/ref:", con.execute(f"SELECT (SELECT count(*) FROM '{tr}/{r}/protein_group_quant.parquet'), (SELECT count(*) FROM '{ref}/{r}/protein_group_quant.parquet')").fetchone(),
          "; groups with changed quantity:", con.execute(f"SELECT count(*) FROM '{tr}/{r}/protein_group_quant.parquet' a JOIN '{ref}/{r}/protein_group_quant.parquet' b USING (protein_group) WHERE a.quantity IS DISTINCT FROM b.quantity OR a.n_peptides <> b.n_peptides").fetchone()[0])
# LFQ
print("LFQ protein cells tr vs ref (changed values):", con.execute(f"SELECT count(*), count(*) FILTER (WHERE a.quantity <> b.quantity) FROM '{tr}/lfq_maxlfq.parquet' a JOIN '{ref}/lfq_maxlfq.parquet' b USING (protein_group, run)").fetchone())
print("LFQ precursor cells: rows tr/ref, cells that were 0 in ref and > 0 in tr:", con.execute(f"SELECT (SELECT count(*) FROM '{tr}/lfq_maxlfq.parquet.precursor.parquet'), (SELECT count(*) FROM '{ref}/lfq_maxlfq.parquet.precursor.parquet')").fetchone(),
      con.execute(f"SELECT count(*) FROM '{tr}/lfq_maxlfq.parquet.precursor.parquet' a LEFT JOIN '{ref}/lfq_maxlfq.parquet.precursor.parquet' b USING (\"group\", charge, run) WHERE a.quantity > 0 AND coalesce(b.quantity, 0) = 0").fetchone()[0])
# derived: transferred features per LFQ protein cell
print("derived: LFQ protein cells containing >= 1 transferred precursor (run = source):", con.execute(f"""
   WITH trf AS (SELECT source AS run, protein_group, count(*) AS n_tr FROM '{tr}/scored_mbr.parquet' WHERE is_transferred GROUP BY ALL)
   SELECT count(*), sum(n_tr) FROM '{tr}/lfq_maxlfq.parquet' l JOIN trf USING (protein_group, run) WHERE l.quantity > 0""").fetchone())
