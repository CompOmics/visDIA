"""Read-only guarantees, spill directories, glob-safe paths, version disagreements and the
MBR presentation of the detail page."""

import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import (
    Cache,
    SchemaVersionError,
    Status,
    counts,
    open_results,
)
from mumdia_viewer.data.candidate_index import CandidateIndex
from mumdia_viewer.data.detail import precursor_detail


def _copy(src: Path, dst: Path) -> Path:
    """Copy a fixture without its file attributes (OneDrive marks directories read-only,
    and a read-only directory cannot be removed on Windows)."""
    shutil.copytree(src, dst, copy_function=shutil.copyfile)
    for path in [dst, *dst.rglob("*")]:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | (stat.S_IEXEC if path.is_dir() else 0))
    return dst


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


# --------------------------------------------------------------------------- read-only


def test_a_cache_configured_inside_the_run_directory_writes_nothing_there(fixture_dir, tmp_path):
    run = _copy(fixture_dir("single"), tmp_path / "run")
    before = _snapshot(run)
    rs = open_results(run, cache=Cache(run / "viewer-cache"))
    assert not rs.cache.writable
    cid = int(pq.read_table(run / "psms_scored.parquet", columns=["candidate_id"])[0][0].as_py())
    precursor_detail(rs, rs.runs[0], cid)
    counts.unit_counts(rs, 0.01)
    rs.duck.close()
    assert _snapshot(run) == before
    assert not (run / "viewer-cache").exists() and not (run / ".tmp").exists()


def test_duckdb_spill_directories_are_private_and_removed(fixture_dir, tmp_path):
    cache = Cache(tmp_path / "cache")
    a = open_results(fixture_dir("single"), cache=cache)
    b = open_results(fixture_dir("experiment"), cache=cache)
    assert a.duck.temp_directory != b.duck.temp_directory
    for rs in (a, b):
        setting = rs.duck.scalar("SELECT current_setting('temp_directory')")
        assert Path(setting) == rs.duck.temp_directory
        assert not rs.duck.temp_directory.is_relative_to(rs.root)
    a.duck.close()
    assert not a.duck.temp_directory.exists()


def test_an_unwritable_cache_never_spills_into_the_working_directory(fixture_dir, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    run = _copy(fixture_dir("single"), tmp_path / "run")
    cwd = os.getcwd()
    try:
        os.chdir(run)
        rs = open_results(".", cache=Cache(blocker / "cache"))
        setting = rs.duck.scalar("SELECT current_setting('temp_directory')")
        assert setting and not Path(setting).resolve().is_relative_to(run.resolve())
    finally:
        os.chdir(cwd)


def test_hard_linked_features_stay_linked_and_unchanged(fixture_dir, tmp_path):
    run = _copy(fixture_dir("single"), tmp_path / "run")
    (run / "psms_competed.parquet").unlink()
    os.link(run / "features.parquet", run / "psms_competed.parquet")
    before = _snapshot(run)
    rs = open_results(run)
    cids = pq.read_table(run / "psms_scored.parquet", columns=["candidate_id"]).column(0)
    for cid in cids.to_pylist()[:5]:
        precursor_detail(rs, rs.runs[0], cid)
    assert os.stat(run / "features.parquet").st_nlink == 2
    assert os.path.samefile(run / "features.parquet", run / "psms_competed.parquet")
    assert _snapshot(run) == before


def test_a_moved_directory_never_reads_the_recorded_location(fixture_dir, tmp_path):
    # Make the recorded --out-dir a real directory with a different features table, so
    # the test does not depend on the machine that wrote the fixture.
    old = tmp_path / "old_location"
    old.mkdir()
    pq.write_table(pa.table({"candidate_id": pa.array([7], pa.uint32())}), old / "features.parquet")
    copy = _copy(fixture_dir("single"), tmp_path / "copy")
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["cli_args"][manifest["cli_args"].index("--out-dir") + 1]
    text = (
        (copy / "manifest.json")
        .read_text(encoding="utf-8")
        .replace(recorded.replace("\\", "\\\\"), str(old).replace("\\", "/"))
    )
    (copy / "manifest.json").write_text(text, encoding="utf-8")
    (copy / "features.parquet").unlink()
    rs = open_results(copy)
    features = rs.runs[0].artifact("features")
    assert features.status is Status.MISSING and features.path is None
    assert rs.runs[0].artifact("psms_scored").path == copy / "psms_scored.parquet"


# --------------------------------------------------------------------------- DuckDB paths


def test_a_bracketed_directory_does_not_read_its_sibling(fixture_dir, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    bracket = _copy(fixture_dir("single"), base / "run[1]")
    _copy(fixture_dir("grouped"), base / "run1")  # what the unescaped glob would match
    rs = open_results(bracket)
    got = {c.unit: c.n_target for c in counts.unit_counts(rs, 0.01)}
    table = pq.read_table(bracket / "psms_scored.parquet")
    targets = pc.equal(table["label"], "target")
    psm = pc.sum(pc.and_(targets, pc.less_equal(table["q_value"], 0.01))).as_py()
    assert got["psm"] == psm == 273


# --------------------------------------------------------------------------- versions


def test_report_version_wins_over_a_stale_manifest(fixture_dir, tmp_path):
    run = _copy(fixture_dir("single"), tmp_path / "run")
    path = run / "chromatograms.parquet.report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = 99
    path.write_text(json.dumps(report), encoding="utf-8")
    rs = open_results(run)
    chrom = rs.runs[0].artifact("chromatograms")
    assert chrom.version.source == "report" and chrom.version.version == 99
    assert "version_mismatch" in rs.notice_codes()
    with pytest.raises(SchemaVersionError, match="version 99"):
        chrom.require()


def test_report_only_versions_of_experiment_runs_are_checked(fixture_dir, tmp_path):
    exp = _copy(fixture_dir("experiment"), tmp_path / "exp")
    path = exp / "a" / "features.parquet.report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["schema_version"] = 7
    path.write_text(json.dumps(report), encoding="utf-8")
    rs = open_results(exp)
    features = rs.run("a").artifact("features")
    assert features.record is None and features.version.source == "report"
    with pytest.raises(SchemaVersionError, match="features schema version 7"):
        features.require()


def test_unrecorded_spectra_with_ion_mobility_are_refused(fixture_dir, tmp_path):
    exp = _copy(fixture_dir("experiment"), tmp_path / "exp")
    ms1 = exp / "a" / "spectra" / "spectra_ms1.parquet"
    table = pq.read_table(ms1)
    table = table.append_column(
        "im", pa.array([[] for _ in range(table.num_rows)], pa.large_list(pa.float32()))
    )
    pq.write_table(table, ms1)
    (exp / "a" / "spectra" / "spectra_ms1.parquet.report.json").unlink()
    rs = open_results(exp)
    art = rs.run("a").artifact("spectra_ms1")
    assert art.version.source == "inferred" and art.version.version == 3
    with pytest.raises(SchemaVersionError, match="ion-mobility"):
        art.require()


# --------------------------------------------------------------------------- MBR


def test_stale_mbr_files_are_not_used(fixture_dir, tmp_path):
    exp = _copy(fixture_dir("experiment"), tmp_path / "exp")
    shutil.copy(fixture_dir("mbr") / "mbr_transferred.parquet", exp / "mbr_transferred.parquet")
    rs = open_results(exp)
    assert "stale_mbr" in rs.notice_codes() and "mbr_transferred" not in rs.extra


def test_mbr_detail_shows_native_q_and_the_lowered_q_separately(fixture_dir, tmp_path):
    exp = _copy(fixture_dir("mbr"), tmp_path / "exp")
    transfers = pq.read_table(exp / "mbr_transferred.parquet").to_pandas()
    first = transfers.iloc[0]
    source, cid = int(first["source"]), int(first["candidate_id"])
    run_name = json.loads((exp / "experiment_manifest.json").read_text(encoding="utf-8"))[
        "experiment"
    ]["runs"][source]
    # Simulate the engine's lowering on the transferred row of the per-run split.
    split = exp / run_name / "scored.parquet"
    t = pq.read_table(split)
    hit = pc.equal(t["candidate_id"], cid)
    lowered = pc.if_else(hit, pa.scalar(0.0001), t["q_value"])
    t = t.set_column(t.schema.get_field_index("q_value"), "q_value", lowered)
    pq.write_table(t, split)
    rs = open_results(exp)
    d = precursor_detail(rs, rs.run(source), cid)
    native = pq.read_table(exp / "scored_combined.parquet").to_pandas()
    native = native[(native.source == source) & (native.candidate_id == cid)].iloc[0]
    shown = {q.column: q.value for q in d.q_values}
    assert shown["q_value"] == pytest.approx(float(native["q_value"]))
    assert shown["q_value"] != pytest.approx(0.0001)
    after = [e for e in d.evidence if e.key == "q_value_after_mbr"]
    assert after and after[0].value == pytest.approx(0.0001)
    assert any("match-between-runs transfer" in n for n in d.notes)


# --------------------------------------------------------------------------- detail robustness


def test_detail_survives_missing_optional_tables(fixture_dir, tmp_path):
    run = _copy(fixture_dir("single"), tmp_path / "run")
    for name in (
        "features.parquet",
        "psms_competed.parquet",
        "peptide_quant.parquet",
        "run_windows.parquet",
    ):
        (run / name).unlink()
    shutil.rmtree(run / "spectra")
    rs = open_results(run)
    cid = int(pq.read_table(run / "psms_scored.parquet", columns=["candidate_id"])[0][0].as_py())
    d = precursor_detail(rs, rs.runs[0], cid)
    assert d.chromatogram is not None
    assert d.features is None and d.window is None and d.apex_scan is None and d.quant is None
    joined = " ".join(d.notes)
    for part in ("features", "extraction window", "quant state", "spectra"):
        assert part in joined, part


# --------------------------------------------------------------------------- index cache


def test_the_cached_index_is_read_back(open_fixture, tmp_path, monkeypatch):
    artifact = open_fixture("single").runs[0].artifact("chromatograms")
    cache = Cache(tmp_path / "cache")
    first = CandidateIndex.for_artifact(artifact, cache)

    def fail(*_args, **_kwargs):
        raise AssertionError("the index was rebuilt instead of read from the cache")

    monkeypatch.setattr(CandidateIndex, "build", classmethod(fail))
    second = CandidateIndex.for_artifact(artifact, Cache(tmp_path / "cache"))
    assert np.array_equal(first.ids, second.ids) and np.array_equal(first.stops, second.stops)
