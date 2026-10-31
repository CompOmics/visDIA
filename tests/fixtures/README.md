# Test fixtures

Small MuMDIA outputs used by the test suite. All but `im/out_tims` were written by
mumdia 0.5.0 (git 80d4874318cb, the v0.5.0 release) on 2026-09-30. They are copies, so every path
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
| `smoke/out_qf` | arm `out_qf` (`quant.fragment_detail = true`) | single run with the engine's `quant_fragments.parquet` (trimmed: manifest, scored, quant, fragment and chromatogram tables) |
| `smoke/exp_qf` | arm `exp_qf` (`quant.fragment_detail = true`) | two-run experiment with each run's `quant_fragments.parquet` (trimmed: no spectra, features, libraries or pass directories) |
| `smoke/test_data/fixture.fasta` | `test_data/fixture.fasta` of the MuMDIA repository | FASTA input of the smoke runs |
| `topk/out_dbl_topk3` | `tools/topk` | double-peak fixture with `extract.retain_top_peaks = promote_top_peaks = 3` |
| `mbr/exp_mbr_c_tr_qt004` | `tools/mbr` | two-run experiment with match-between-runs (48 transfers) |
| `overlap/out_ovl_bp`, `out_ovl_bp_pool` | `tools/overlap` | grouped run with overlapping bands and a non-empty `overlap_losers.parquet`, and its pooled twin |
| `overlap/out_ovl_pg_rg50`, `out_ovl_pg_pool_rg50` | `tools/overlap` | per-group calibration, 50-row row groups |
| `overlap/out_ovl128`, `out_ovl128_pool` | `tools/overlap` | 128 planned bands with 8 skipped (trimmed to the chromatogram tables) |
| `entrapment/entrap_mode` | `tools/entrapment` | `psms_scored.parquet` of a rescore in entrapment mode (`entrapment_native`) |
| `im/out_tims` | `tools/im/subset_run.py` | diaPASEF single run (ion mobility), cut from a real search (see below) |

`smoke/out_qf` and `smoke/exp_qf` were written on 2026-10-31 by the MuMDIA branch that adds
`quant.fragment_detail` (`feat/quant-fragment-detail`, version 0.9.0 plus the change), with the
fixed scan window and the configuration of that branch's smoke test; every other file of
those arms was byte-identical to the same build without the key.

`tools/` holds the fixture generators (derived from `ci/make_fixture_mzml.py` of the
MuMDIA repository, Apache-2.0) and the configurations used. The smoke fixtures are
regenerated with `MUMDIA_BIN=<mumdia> PYTHON=python bash ci/smoke.sh <work>` in a MuMDIA
checkout; note that the script deletes `<work>` first.

## `im/out_tims` (diaPASEF)

A cut of a real search, not a synthetic run: mumdia 0.8.0 (git a8366300a758) on
`ttSCP_diaPASEF_Condition_A_Sample_Alpha_01_11494.d` of the ProteoBench HYE diaPASEF set
(timsTOF, 24 diaPASEF windows), FASTA `ProteoBenchFASTA_MixedSpecies_HYE.fasta`, the
`configs/examples/diapasef.json` recipe with ms2pip `timsTOF2024`, no missed cleavages,
length 7 to 30, charge 2 to 3 (`features.im_features` on, so chromatograms v4 carry
per-point 1/K0), written 2026-10-28. `tools/im/subset_run.py` (defaults) keeps 650
candidates with a scored apex in 2200 to 2215 s (250 accepted targets, 250 decoys, 150
other targets), renumbers their `candidate_id` to 0..649 in every table, keeps the scans
from 2185 to 2230 s trimmed to their 200 (MS2) or 400 (MS1) most intense peaks, and nulls
the content hashes. Scores, q values and `cal.json` are the full run's. Regenerate with

    python tests/fixtures/tools/im/subset_run.py <full run> tests/fixtures/im/out_tims
