"""Run summary, inputs, artifact table, stage timings and hash verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import blake3
import pandas as pd
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.overview import (
    artifact_table,
    inputs_table,
    run_summary,
    stage_timings_table,
    verify_hashes,
)


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _edit_json(path: Path, **fields) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- summary


def test_run_summary_single(open_fixture, fixture_dir):
    rs = open_fixture("single")
    s = run_summary(rs)
    manifest = json.loads((fixture_dir("single") / "manifest.json").read_text(encoding="utf-8"))
    assert (s.kind, s.root, s.n_runs, s.runs) == ("run", fixture_dir("single"), 1, [""])
    assert (s.mumdia_version, s.git_sha) == ("0.5.0", "80d4874318cb")
    assert s.commit_date == "2026-09-29T12:37:28+02:00"
    assert s.cli_args == manifest["cli_args"] and s.cli_args[1] == "run"
    assert s.config["features"]["set"] == "extended"
    assert s.config_hash == blake3.blake3(manifest["config_json"].encode()).hexdigest()
    assert s.model_identities["rescorer"] == "native-percolator-lite-v1"
    assert s.rescore.classifier == "native_tda" and s.rescore.mode == "target_decoy"
    assert s.grouped is None and s.mbr_strategy is None and not s.mbr_ran
    assert s.quant_q_filter["configured"] == "peptide_q"
    assert s.quant_q_filter["effective"] == "PeptideQ"
    assert s.quant_q_filter["q_threshold"] == 0.01
    assert "moved" in [n.code for n in s.notices]


def test_run_summary_experiment(open_fixture):
    s = run_summary(open_fixture("experiment"))
    assert (s.kind, s.n_runs, s.runs) == ("experiment", 2, ["a", "b"])
    assert s.mbr_strategy == "None" and not s.mbr_ran
    assert s.quant_q_filter["configured"] == "PeptideQ"
    assert s.quant_q_filter["effective"] == "PsmQ"
    assert "experiment_manifest.json" in s.quant_q_filter["source"]
    assert s.model_identities["mbr"] == "None" and s.grouped is None


def test_run_summary_mbr(open_fixture):
    s = run_summary(open_fixture("mbr"))
    assert s.runs == ["a", "c"] and s.mbr_strategy == "RtTransfer" and s.mbr_ran
    assert s.quant_q_filter["q_threshold"] == 0.004
    assert s.quant_q_filter["effective"] == "PsmQ"


def test_run_summary_grouped(open_fixture):
    g = run_summary(open_fixture("grouped")).grouped
    assert list(g) == [""]
    band = g[""]
    assert band["bands"] == ["g00", "g01", "g02"] and band["n_bands"] == 3
    assert band["skipped"] == [] and band["overlap_losers_rows"] == 0
    assert not band["pooled_chromatograms"] and not band["pooled_competed"]
    assert band["window_groups_configured"] == 3 and band["calibration"] == "PerGroup"
    pooled = run_summary(open_fixture("grouped_pool")).grouped[""]
    assert pooled["pooled_chromatograms"] and pooled["pooled_competed"]
    assert pooled["overlap_losers_rows"] is None
    wide = run_summary(open_fixture("ovl128")).grouped[""]
    assert wide["n_bands"] == 120 and wide["skipped"] == [83, 114, 115, 122, 123, 124, 125, 126]
    assert wide["window_groups_configured"] == 128 and wide["overlap_losers_rows"] == 140


# --------------------------------------------------------------------------- inputs


def test_inputs_table_statuses(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    out_dir = "C:/Users/robbi/mumdia_viewer_ref/smoke/out"  # the recorded --out-dir
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    size = (copy / "spectra" / "spectra_ms2.parquet").stat().st_size
    data["inputs"]["mzml"]["path"] = "Z:/nowhere/fixture.mzML"
    data["inputs"]["inside"] = {
        "path": f"{out_dir}/spectra/spectra_ms2.parquet",
        "bytes": size,
        "content_hash": "0" * 64,
    }
    data["inputs"]["inside_missing"] = {"path": f"{out_dir}/nope.mzML", "bytes": 5}
    data["inputs"]["wrong_size"] = {"path": "test_data/fixture.fasta", "bytes": 1}
    manifest.write_text(json.dumps(data), encoding="utf-8")

    table = inputs_table(open_results(copy)).set_index("key")
    assert table.loc["fasta", "status"] == "needs_remap"
    assert "working directory" in table.loc["fasta", "note"]
    assert table.loc["mzml", "status"] == "needs_remap"
    assert table.loc["inside", "status"] == "found"
    assert table.loc["inside", "resolution"] == "rerooted"
    assert table.loc["inside_missing", "status"] == "missing"
    assert table.loc["wrong_size", "status"] == "needs_remap"

    test_data = fixture_dir("single").parent / "test_data"
    remapped = inputs_table(open_results(copy, remaps={"test_data": test_data})).set_index("key")
    assert remapped.loc["fasta", "status"] == "found"
    assert remapped.loc["fasta", "resolution"] == "remapped"
    assert remapped.loc["fasta", "size_on_disk"] == 2723 == remapped.loc["fasta", "bytes"]
    assert remapped.loc["wrong_size", "status"] == "size_mismatch"
    assert list(remapped.columns[:5]) == [
        "recorded_path",
        "resolved_path",
        "bytes",
        "content_hash",
        "status",
    ]


def test_inputs_table_experiment_keys(open_fixture):
    keys = inputs_table(open_fixture("experiment"))["key"].tolist()
    assert keys == ["fasta", "mzml[0]", "mzml[1]"]


# --------------------------------------------------------------------------- artifacts


def test_artifact_table_shows_deleted_band_intermediates(open_fixture):
    table = artifact_table(open_fixture("grouped"))
    assert not table["key"].duplicated().any()
    deleted = table[table["status"] == "deleted_after_pooling"]
    assert sorted(deleted["key"]) == sorted(
        f"{k}[g0{i}]" for k in ("features", "psms_extracted") for i in range(3)
    )
    assert set(deleted["scope"]) == {"groups/g00", "groups/g01", "groups/g02"}
    assert all(p.startswith("C:/Users/robbi/mumdia_viewer_ref/") for p in deleted["path"])
    present = table[table["status"] == "present"].set_index("key")
    assert present.loc["psms_competed[g01]", "path"] == "groups/g01/psms_competed.parquet"
    assert present.loc["psms_competed[g01]", "version"] == "psms_competed v4"
    assert present.loc["psms_scored", "scope"] == "."
    assert present.loc["psms_scored", "content_hash_short"] == "b8acb13b3a03"
    assert str(table["rows"].dtype) == "Int64"


def test_artifact_table_lists_each_artifact_once(open_fixture):
    single = artifact_table(open_fixture("single"))
    assert not single["key"].duplicated().any() and len(single) == 18
    assert set(single["scope"]) == {"."}
    exp = artifact_table(open_fixture("experiment")).set_index("key")
    assert exp.loc["scored_combined", "scope"] == "."
    assert exp.loc["scored[b]", "scope"] == "b" and exp.loc["chromatograms[a]", "scope"] == "a"
    assert exp.loc["chromatograms[a]", "resolution"] == "fixed name"
    assert exp.loc["lfq_maxlfq.peptide", "version"].endswith("version not recorded") or (
        "inferred" in exp.loc["lfq_maxlfq.peptide", "version"]
    )


# --------------------------------------------------------------------------- timings


def test_stage_timings_take_the_largest_report_time(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    spectra = copy / "spectra"
    for name, ms in (
        ("spectra_ms2", 5000),
        ("spectra_ms1", 3000),
        ("isolation_windows", 1000),
        ("ms2_to_ms1", 2000),
    ):
        _edit_json(spectra / f"{name}.parquet.report.json", elapsed_ms=ms)
    _edit_json(copy / "chromatograms.parquet.report.json", elapsed_ms=7000)
    _edit_json(copy / "psms_extracted.parquet.report.json", elapsed_ms=4000)
    table = stage_timings_table(open_results(copy))
    timed = table[table["stage"] != "not recorded"]
    assert not timed.duplicated(["directory", "stage"]).any()
    rows = timed.set_index("stage")
    assert rows.loc["convert", "elapsed_s"] == 5.0  # the largest, not the sum 11.0
    assert rows.loc["convert", "artifacts"] == (
        "isolation_windows, ms2_to_ms1, spectra_ms1, spectra_ms2"
    )
    assert rows.loc["extract", "elapsed_s"] == 7.0
    assert set(timed["directory"]) == {"."}
    assert "report.json" in rows.loc["extract", "source"]
    last = table.iloc[-1]
    assert last["stage"] == "not recorded" and last["artifacts"] == "report"


def _summary(timings: dict, *, multihead: bool = True, bands: dict | None = None) -> dict:
    data = {"rows": 10, "repredicted": 10, "timings_s": timings}
    if multihead:
        data["multihead"] = {"heads_requested": 80, "anchors": 100, "heads": [1], "best_head": 1}
    if bands is not None:
        data["bands"] = bands
    return data


TIMINGS = {
    "read_library": 2.0,
    "model_load": 0.5,
    "reference": 0.7,
    "fit": 10.0,
    "unique": 7.0,
    "featurisation": 250.0,
    "forward": 0.0,
    "predict": 600.0,
    "rewrite": 3.0,
    "write": 5.0,
}


def test_stage_timings_include_deeplc_summaries(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    path = copy / "fragment_library_precursors_multihead.parquet.summary.json"
    path.write_text(json.dumps(_summary(TIMINGS)), encoding="utf-8")
    rows = stage_timings_table(open_results(copy)).set_index("stage")
    row = rows.loc["deeplc-multihead"]
    # Sequential phases only: featurisation and forward are part of predict.
    assert row["elapsed_s"] == pytest.approx(2.0 + 0.5 + 0.7 + 10.0 + 7.0 + 600.0 + 3.0 + 5.0)
    assert path.name in row["source"] and row["directory"] == "."
    assert "predict 600.0 s" in row["note"]

    exp = _copy(fixture_dir("experiment"), tmp_path / "exp")
    repredict = {k: v for k, v in TIMINGS.items() if k != "model_load"}
    (exp / "a" / "fragment_library_precursors_multihead.parquet.summary.json").write_text(
        json.dumps(_summary(TIMINGS)), encoding="utf-8"
    )
    (exp / "fragment_library_precursors_deeplc.parquet.summary.json").write_text(
        json.dumps(_summary(dict(repredict, model_load=9.0), multihead=False)), encoding="utf-8"
    )
    table = stage_timings_table(open_results(exp))
    multi = table[table["stage"] == "deeplc-multihead"]
    assert multi["directory"].tolist() == ["a"]
    again = table[table["stage"] == "deeplc-repredict"].iloc[0]
    # Without the multi-head fit the model load is inside predict: it is not added.
    assert again["directory"] == "." and again["elapsed_s"] == pytest.approx(627.7)
    assert table.iloc[-1]["artifacts"] == "split-by-source, quant-lfq, report"


def test_band_summaries_sharing_one_call_are_reported_once(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("grouped"), tmp_path / "run")
    for i in range(3):
        band = copy / "groups" / f"g0{i}"
        summary = _summary(TIMINGS, bands={"count": 3, "index": i, "union_unique": 50})
        (band / "lib_precursors_multihead.parquet.summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    table = stage_timings_table(open_results(copy))
    multi = table[table["stage"] == "deeplc-multihead"]
    assert len(multi) == 1
    assert multi.iloc[0]["directory"] == "groups/g00 (+2 bands)"
    assert multi.iloc[0]["elapsed_s"] == pytest.approx(628.2)
    last = table.iloc[-1]
    assert last["artifacts"] == "seed-pool, features (3 band reports deleted after pooling), report"
    bands = table[table["directory"].str.startswith("groups/g0") & (table["stage"] == "extract")]
    assert len(bands) == 3  # one row per band; parallel band times are never summed


def test_band_stages_whose_reports_were_deleted_are_listed(open_fixture, fixture_dir, tmp_path):
    """The band features report goes with features.parquet (delete_band_intermediates)."""
    table = stage_timings_table(open_fixture("grouped"))
    bands = table[table["directory"].str.startswith("groups/")]
    assert set(bands["stage"]) == {"extract", "search-seed", "compete", "rt-im-train"}
    last = table.iloc[-1]
    assert last["stage"] == "not recorded"
    assert "features (3 band reports deleted after pooling)" in last["artifacts"]
    assert "deleted after pooling" in last["source"]
    single = stage_timings_table(open_fixture("single")).iloc[-1]
    assert (
        single["artifacts"] == "report" and single["source"] == "these stages write no report.json"
    )
    # A band whose features report survives is timed, and only the others are listed.
    copy = _copy(fixture_dir("grouped"), tmp_path / "run")
    band = copy / "groups" / "g00"
    shutil.copyfile(band / "psms_competed.parquet", band / "features.parquet")
    report = json.loads((band / "psms_competed.parquet.report.json").read_text(encoding="utf-8"))
    report.update(logical_name="features", schema_name="features", schema_version=2)
    report.update(stage="features", elapsed_ms=4321)
    (band / "features.parquet.report.json").write_text(json.dumps(report), encoding="utf-8")
    table = stage_timings_table(open_results(copy))
    rows = table.set_index(["directory", "stage"])
    assert rows.loc[("groups/g00", "features"), "elapsed_s"] == pytest.approx(4.321)
    assert "features (2 band reports deleted after pooling)" in table.iloc[-1]["artifacts"]


def test_stage_timings_label_a_library_cache_hit(fixture_dir, tmp_path):
    """A cache hit copies the stored library's reports; their time is another run's."""
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("fragment_library_precursors", "fragment_library_fragments"):
        data["artifacts"][key]["producing_stage"] = "library-cache"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    stored = max(
        json.loads((copy / f"{key}.parquet.report.json").read_text(encoding="utf-8"))["elapsed_ms"]
        for key in ("fragment_library_precursors", "fragment_library_fragments")
    )
    table = stage_timings_table(open_results(copy))
    assert "predict-frag" not in set(table["stage"])
    row = table[table["stage"] == "library-cache"].iloc[0]
    assert row["directory"] == "." and pd.isna(row["elapsed_s"])
    assert row["artifacts"] == "fragment_library_fragments, fragment_library_precursors"
    assert row["source"] == "manifest producing_stage library-cache; report.json stage predict-frag"
    assert f"({stored / 1000.0:.3f} s)" in row["note"] and "copied" in row["note"]
    assert "not a time of this run" in row["note"]
    # The other stages keep their report times.
    assert set(table[table["elapsed_s"].notna()]["stage"]) >= {"convert", "extract", "rescore"}


def test_untimed_stages_of_an_mbr_experiment(open_fixture):
    table = stage_timings_table(open_fixture("mbr"))
    assert table.iloc[-1]["artifacts"] == "split-by-source, quant-lfq, mbr, report"
    assert set(table["directory"]) == {".", "a", "c"}


# --------------------------------------------------------------------------- hashes


def test_verify_hashes_on_small_artifacts(open_fixture):
    rs = open_fixture("single")
    keys = ["psms_scored", "features", "psms_competed", "peptide_quant", "run_windows"]
    table = verify_hashes(rs, keys)
    assert table["key"].tolist() == keys
    assert (table["verdict"] == "match").all()
    assert (table["recorded"] == table["computed"]).all()
    everything = verify_hashes(rs)
    assert len(everything) == 18 and (everything["verdict"] == "match").all()
    with pytest.raises(KeyError, match="no artifact or input"):
        verify_hashes(rs, ["nope"])


def test_verify_hashes_detects_a_changed_file(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    target = copy / "peptide_quant.parquet"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    rs = open_results(copy, remaps={"test_data": fixture_dir("single").parent / "test_data"})
    table = verify_hashes(rs, ["peptide_quant", "psms_scored", "fasta"]).set_index("key")
    assert table.loc["peptide_quant", "verdict"] == "mismatch"
    assert table.loc["psms_scored", "verdict"] == "match"
    assert table.loc["fasta", "verdict"] == "match" and table.loc["fasta", "scope"] == "input"


def test_verify_hashes_reports_deleted_band_intermediates(open_fixture):
    table = verify_hashes(open_fixture("grouped"), ["features[g00]", "psms_competed[g00]"])
    verdicts = dict(zip(table["key"], table["verdict"], strict=True))
    assert verdicts == {"features[g00]": "missing", "psms_competed[g00]": "match"}


def test_overview_never_writes_to_the_run_directory(fixture_dir, tmp_path):
    copy = _copy(fixture_dir("grouped"), tmp_path / "run")
    before = {p: p.stat().st_mtime_ns for p in copy.rglob("*") if p.is_file()}
    rs = open_results(copy)
    run_summary(rs)
    inputs_table(rs)
    artifact_table(rs)
    stage_timings_table(rs)
    verify_hashes(rs)
    after = {p: p.stat().st_mtime_ns for p in copy.rglob("*") if p.is_file()}
    assert before == after


# --------------------------------------------------------------------------- real data


@pytest.mark.real_data
def test_real_single_overview(real_single):
    rs = open_results(real_single)
    s = run_summary(rs)
    assert s.kind == "run" and s.model_identities["rt_predictor"].startswith("deeplc")
    table = stage_timings_table(rs).set_index("stage")
    summary = json.loads(
        (real_single / "fragment_library_precursors_multihead.parquet.summary.json").read_text(
            encoding="utf-8"
        )
    )
    phases = ["read_library", "model_load", "reference", "fit", "unique", "predict", "rewrite"]
    expected = sum(summary["timings_s"][p] for p in [*phases, "write"])
    assert table.loc["deeplc-multihead", "elapsed_s"] == pytest.approx(expected)
    assert table.loc["rescore", "elapsed_s"] > 0
    inputs = inputs_table(rs)
    assert set(inputs["key"]) == {"mzml", "lib_precursors", "lib_fragments"}
    artifacts = artifact_table(rs)
    assert not artifacts["key"].duplicated().any()
    print("\n" + table.reset_index()[["directory", "stage", "elapsed_s"]].to_string())
