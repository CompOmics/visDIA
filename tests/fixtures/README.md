# Test fixtures

Small MuMDIA outputs used by the test suite. All were written by mumdia 0.5.0
(git 80d4874318cb, the v0.5.0 release) on 2026-09-30. They are copies, so every path
recorded in their manifests points at the directory they were written to; the viewer
re-roots those paths onto the copy (this also tests moved-directory support).

| directory | origin | purpose |
|---|---|---|
| `smoke/out` | `ci/smoke.sh` arm `out` | single run, FASTA mode, chromatograms v2 |
| `smoke/out_chrom_v1` | arm `out_chrom_v1` (`extract.chromatogram_schema = 1`) | v1 layout of the same search (trimmed: manifest, chromatograms, psms_scored) |
| `smoke/out_chrom_v2_rg1` | arm `out_chrom_v2_rg1` (`MUMDIA_CHROM_ROW_GROUP_ROWS=1`) | v2 with one row per row group (trimmed like above) |
| `smoke/out_grouped` | arm `out_grouped` | grouped run, 3 bands, default (no pooled tables) |
| `smoke/out_grouped_pool` | arm `out_grouped_pool` | same with pooled tables |
| `smoke/exp` | arm `exp` | two-run experiment |
| `smoke/test_data/fixture.fasta` | `test_data/fixture.fasta` of the MuMDIA repository | FASTA input of the smoke runs |
| `topk/out_dbl_topk3` | `tools/topk` | double-peak fixture with `extract.retain_top_peaks = promote_top_peaks = 3` |
| `mbr/exp_mbr_c_tr_qt004` | `tools/mbr` | two-run experiment with match-between-runs (48 transfers) |
| `overlap/out_ovl_bp`, `out_ovl_bp_pool` | `tools/overlap` | grouped run with overlapping bands and a non-empty `overlap_losers.parquet`, and its pooled twin |
| `overlap/out_ovl_pg_rg50`, `out_ovl_pg_pool_rg50` | `tools/overlap` | per-group calibration, 50-row row groups |
| `overlap/out_ovl128`, `out_ovl128_pool` | `tools/overlap` | 128 planned bands with 8 skipped (trimmed to the chromatogram tables) |
| `entrapment/entrap_mode` | `tools/entrapment` | `psms_scored.parquet` of a rescore in entrapment mode (`entrapment_native`) |

`tools/` holds the fixture generators (derived from `ci/make_fixture_mzml.py` of the
MuMDIA repository, Apache-2.0) and the configurations used. The smoke fixtures are
regenerated with `MUMDIA_BIN=<mumdia> PYTHON=python bash ci/smoke.sh <work>` in a MuMDIA
checkout; note that the script deletes `<work>` first.
