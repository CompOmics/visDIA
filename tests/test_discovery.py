"""Artifact discovery for single, experiment, grouped and moved directories."""

import json
import os
import shutil
from pathlib import Path

import pytest

from mumdia_viewer.data import (
    ArtifactNotFound,
    NotAResultDirectory,
    SchemaVersionError,
    Status,
    open_results,
)


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def test_single_run(open_fixture):
    rs = open_fixture("single")
    assert rs.kind == "run" and not rs.is_experiment
    assert [r.label for r in rs.runs] == ["run"]
    run = rs.runs[0]
    assert rs.scored is run.artifact("psms_scored")
    assert rs.scored.version.version == 4 and rs.scored.version.source == "manifest"
    assert rs.scored.rows == 284
    for kind in (
        "chromatograms",
        "features",
        "psms_competed",
        "psms_extracted",
        "run_windows",
        "seed_psms",
        "spectra_ms2",
        "spectra_ms1",
        "isolation_windows",
        "ms2_to_ms1",
        "peptide_quant",
        "protein_group_quant",
        "fragment_quant",
        "rt_library",
    ):
        assert run.has(kind), kind
    assert run.artifact("chromatograms").version.version == 2
    assert run.grouped is None
    assert {"cal", "masscal", "features_schema"} <= set(run.side_files)
    # The fixture is a copy: the recorded paths were re-rooted.
    assert "moved" in rs.notice_codes()


def test_grouped_run_default_layout(open_fixture):
    rs = open_fixture("grouped")
    run = rs.runs[0]
    g = run.grouped
    assert g is not None and [b.name for b in g.bands] == ["g00", "g01", "g02"]
    assert not g.pooled_chromatograms and not g.pooled_competed
    assert g.losers is not None and g.losers.present and g.losers.rows == 0
    band = g.band(1)
    assert band.artifact("chromatograms").present and band.artifact("run_windows").present
    # Band features and psms_extracted are listed but were deleted after pooling.
    assert band.artifact("features").status is Status.DELETED_AFTER_POOLING
    with pytest.raises(ArtifactNotFound, match="deleted after pooling"):
        band.artifact("psms_extracted").require()
    assert "deleted_after_pooling" in rs.notice_codes()
    assert g.plan["window_groups"] == 3


def test_grouped_run_pooled_layout(open_fixture):
    g = open_fixture("grouped_pool").runs[0].grouped
    assert g.pooled_chromatograms and g.pooled_competed and g.losers is None


def test_skipped_bands_and_three_digit_band_names(open_fixture):
    rs = open_fixture("ovl128")
    g = rs.runs[0].grouped
    assert len(g.bands) == 120
    assert g.skipped == [83, 114, 115, 122, 123, 124, 125, 126]
    assert g.band(127).name == "g127"
    assert g.losers.rows == 140
    # The trimmed fixture lacks most artifacts: one aggregated notice, not one per file.
    assert rs.notice_codes().count("missing_artifact") == 1


def test_experiment(open_fixture):
    rs = open_fixture("experiment")
    assert rs.kind == "experiment"
    assert [(r.name, r.index) for r in rs.runs] == [("a", 0), ("b", 1)]
    assert rs.scored.key == "scored_combined" and rs.scored.rows == 568
    assert rs.scored_for_quant is None
    a = rs.run("a")
    assert rs.run(1).name == "b"
    assert a.artifact("psms_scored").key == "scored[a]"
    assert a.artifact("psms_scored").version.source == "manifest"
    # Per-run stage tables are not in the experiment manifest; their reports give the version.
    chrom = a.artifact("chromatograms")
    assert chrom.present and chrom.record is None and chrom.version.source == "report"
    assert chrom.version.version == 2
    assert {"lfq_maxlfq", "lfq_maxlfq_peptide", "lfq_maxlfq_precursor"} <= set(rs.extra)
    assert rs.extra["lfq_maxlfq_peptide"].version.source in ("inferred", "unrecorded")
    assert rs.extra["fragment_library_precursors"].present
    assert a.artifact("rt_library") is not None
    with pytest.raises(KeyError):
        rs.run("zzz")


def test_experiment_with_match_between_runs(open_fixture):
    rs = open_fixture("mbr")
    assert rs.scored_for_quant is not None
    assert rs.scored_for_quant.path.name == "scored_mbr.parquet"
    assert rs.scored.path.name == "scored_combined.parquet"
    assert rs.extra["mbr_transferred"].present
    assert rs.manifest.experiment["mbr"] == "RtTransfer"


def test_moved_directory_uses_the_copy_only(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "moved" / "run")
    (copy / "features.parquet").unlink()
    (copy / "psms_competed.parquet").unlink()
    rs = open_results(copy)
    # The original location may still exist on this machine; it must never be used.
    features = rs.runs[0].artifact("features")
    assert features.status is Status.MISSING and features.path is None
    assert rs.scored.path == copy / "psms_scored.parquet"
    assert "missing_artifact" in rs.notice_codes()


def test_opening_never_writes_to_the_run_directory(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("experiment"), tmp_path / "exp")
    before = _snapshot(copy)
    rs = open_results(copy)
    for _, artifact in rs.all_artifacts():
        if artifact.usable:
            handle = artifact.parquet()
            handle.metadata()
            if handle.num_row_groups:
                handle.read_row_group(0, cached=True)
    rs.stage_timings()
    assert _snapshot(copy) == before


def test_not_a_result_directory(fixture_dir, tmp_path: Path):
    with pytest.raises(NotAResultDirectory, match="no manifest"):
        open_results(tmp_path)
    with pytest.raises(NotAResultDirectory, match="not a directory"):
        open_results(tmp_path / "nope")
    exp = _copy(fixture_dir("experiment"), tmp_path / "exp")
    with pytest.raises(NotAResultDirectory, match="run directory of the experiment"):
        open_results(exp / "a")
    stages = tmp_path / "stages"
    stages.mkdir()
    (stages / "x.parquet.report.json").write_text("{}")
    with pytest.raises(NotAResultDirectory, match="stage reports but no manifest"):
        open_results(stages)


def _edit_manifest(root: Path, key: str, **fields) -> None:
    path = root / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifacts"][key].update(fields)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_unknown_scored_version_refuses_to_open(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    _edit_manifest(copy, "psms_scored", schema_version=99)
    with pytest.raises(SchemaVersionError, match="psms_scored schema version 99"):
        open_results(copy)


def test_unreleased_chromatogram_version_is_refused_per_artifact(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    _edit_manifest(copy, "chromatograms", schema_version=4)
    rs = open_results(copy)
    chrom = rs.runs[0].artifact("chromatograms")
    assert not chrom.usable and "unsupported_version" in rs.notice_codes()
    with pytest.raises(SchemaVersionError, match="ion-mobility"):
        chrom.require()
    # With the opt-in the version passes the registry; the columns are still v2, so the
    # column contract refuses the mismatch when the table is read.
    rs2 = open_results(copy, allow_unreleased=True)
    assert rs2.runs[0].artifact("chromatograms").error is None


def test_hash_mismatch_between_manifest_and_report(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    _edit_manifest(copy, "run_windows", content_hash="0" * 64)
    rs = open_results(copy)
    assert "hash_mismatch" in rs.notice_codes()


def test_unfinished_writes_are_reported(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    (copy / "features.parquet.tmp-1234-5").write_bytes(b"partial")
    rs = open_results(copy)
    assert "in_progress" in rs.notice_codes()


def test_stage_timings_take_the_largest_per_stage(open_fixture):
    rs = open_fixture("single")
    timings = rs.stage_timings()
    stages = [(t.directory, t.stage) for t in timings]
    assert len(stages) == len(set(stages))
    convert = [t for t in timings if t.stage == "convert"]
    assert len(convert) == 1 and len(convert[0].artifacts) >= 2


@pytest.mark.real_data
def test_real_single_run(real_single):
    rs = open_results(real_single)
    assert rs.kind == "run" and rs.scored.version.version == 4
    run = rs.runs[0]
    assert run.has("chromatograms") and run.has("spectra_ms2") and run.has("rt_library")
    fragments = rs.extra.get("fragment_library_fragments")
    assert fragments is not None and fragments.stage == "library-input"


@pytest.mark.real_data
def test_real_experiment(real_experiment):
    rs = open_results(real_experiment)
    assert rs.kind == "experiment" and len(rs.runs) >= 2
    assert all(r.has("chromatograms") for r in rs.runs)
    assert os.path.basename(rs.scored.path) == "scored_combined.parquet"


def test_run_lookup_accepts_numpy_integers(open_fixture):
    import numpy as np

    rs = open_fixture("experiment")
    assert rs.run(np.uint32(1)).name == "b" and rs.run(np.int64(0)).name == "a"
    assert rs.run(rs.runs[0]) is rs.runs[0]
    with pytest.raises(KeyError):
        rs.run(True)
    with pytest.raises(KeyError):
        rs.run(7)


def test_memo_computes_once(open_fixture):
    rs = open_fixture("single")
    calls = []
    assert rs.memo(("test", 1), lambda: calls.append(1) or 42) == 42
    assert rs.memo(("test", 1), lambda: calls.append(1) or 43) == 42
    assert calls == [1]


def test_missing_artifact_error_names_the_expected_path(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("single"), tmp_path / "run")
    (copy / "run_windows.parquet").unlink()
    rs = open_results(copy)
    with pytest.raises(ArtifactNotFound) as err:
        rs.runs[0].artifact("run_windows").require()
    assert str(copy / "run_windows.parquet") in str(err.value)


def test_stale_mbr_files_are_reported(fixture_dir, tmp_path: Path):
    copy = _copy(fixture_dir("experiment"), tmp_path / "exp")
    shutil.copy(fixture_dir("mbr") / "mbr_transferred.parquet", copy / "mbr_transferred.parquet")
    rs = open_results(copy)
    assert "stale_mbr" in rs.notice_codes()


def test_all_artifacts_yields_each_artifact_once_with_its_directory(open_fixture):
    rs = open_fixture("grouped")
    scopes = {}
    ids = [id(a) for _, a in rs.all_artifacts()]
    assert len(ids) == len(set(ids))
    for scope, a in rs.all_artifacts():
        scopes.setdefault(scope, set()).add(a.kind)
    assert "" in scopes and "groups/g00" in scopes and "chromatograms" in scopes["groups/g01"]
    exp = open_fixture("experiment")
    assert {s for s, _ in exp.all_artifacts()} >= {"", "a", "b"}
