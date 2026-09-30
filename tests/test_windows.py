"""RT extraction windows: run_windows rows, band offsets and band-local ids.

The oracles are read with pyarrow directly: the run_windows rows, psms_extracted, the
library precursor m/z, and the MS2 scan table, from which the chromatogram axis of a
candidate is rebuilt with the engine's rule (every scan of every isolation window with
``lower <= precursor_mz <= upper`` and ``rt_lo <= rt <= rt_hi``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import ArtifactNotFound, open_results
from mumdia_viewer.data.chromatograms import ChromatogramSource
from mumdia_viewer.data.windows import AmbiguousBand, BandOffset, band_offsets, rt_window

# Band spans from the plan and the library (G4 facts F3.1; T4 F7).
OFFSETS = {
    "grouped": {"g00": (4, 1770), "g01": (1774, 1560), "g02": (3334, 470)},
    "ovl_bp": {"g00": (4, 1810), "g01": (1740, 1602), "g02": (3314, 496)},
    "ovl_rg50": {"g00": (4, 1810), "g01": (1740, 1602), "g02": (3314, 496)},
}


def f64bits(x: float) -> int:
    return int(np.array([x], dtype=np.float64).view(np.uint64)[0])


def finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def run_windows_rows(path: Path) -> dict[str, np.ndarray]:
    t = pq.read_table(path)
    return {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}


def scan_table(run_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = pq.read_table(
        run_root / "spectra" / "spectra_ms2.parquet",
        columns=["rt_seconds", "window_lower", "window_upper"],
    )
    return tuple(t.column(c).to_numpy() for c in t.column_names)  # type: ignore[return-value]


def covering_axis(scans, pmz: float, lo: float | None, hi: float | None) -> np.ndarray:
    """The engine's grid: every covering window of the run, inclusive bounds, float32."""
    rt, lower, upper = scans
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    sel = (lower <= pmz) & (pmz <= upper) & (rt >= lo) & (rt <= hi)
    return np.unique(rt[sel]).astype(np.float32)


def library_mz(run_root: Path) -> np.ndarray:
    t = pq.read_table(
        run_root / "fragment_library_precursors.parquet", columns=["candidate_id", "precursor_mz"]
    )
    ids = t.column("candidate_id").to_numpy()
    assert (ids == np.arange(ids.size)).all()
    return t.column("precursor_mz").to_numpy()


def copy_run(fixture_dir, name: str, tmp_path: Path) -> Path:
    run_dir = tmp_path / name
    shutil.copytree(fixture_dir(name), run_dir)
    return run_dir


def forget_hash(run_dir: Path, key: str, rel: str) -> None:
    """Drop the recorded content hash of a rewritten artifact, so no cache entry is reused.

    ``key`` is the manifest key and ``rel`` the file's path in the run directory.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["artifacts"][key]["content_hash"] = None
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report = run_dir / f"{rel}.report.json"
    if report.exists():
        report.unlink()


def drop_library_rows(run_dir: Path, rows: list[int]) -> None:
    """Remove library rows and renumber candidate_id, so the table still looks dense."""
    path = run_dir / "fragment_library_precursors.parquet"
    lib = pq.read_table(path)
    keep = np.setdiff1d(np.arange(lib.num_rows), np.array(rows, dtype=np.int64))
    lib = lib.take(pa.array(keep))
    lib = lib.set_column(0, "candidate_id", pa.array(np.arange(lib.num_rows), pa.uint32()))
    pq.write_table(lib, path)
    forget_hash(run_dir, "fragment_library_precursors", path.name)


def axis_failures(rs, run, true_mz: np.ndarray) -> dict[str, list[int]]:
    """Per band: candidates whose rt_window does not reproduce their chromatogram axis.

    The oracle is the engine's grid rule with the true library m/z; a candidate whose
    window raises is listed under the exception name.
    """
    src = ChromatogramSource.for_run(rs, run)
    scans = scan_table(run.root)
    out: dict[str, list[int]] = {}
    for cid in src.candidate_ids().tolist():
        axis = src.read(cid).common_axis()
        if axis is None:
            continue
        try:
            w = rt_window(rs, run, cid)
        except Exception as exc:
            out.setdefault(type(exc).__name__, []).append(cid)
            continue
        expected = covering_axis(scans, float(true_mz[cid]), w.rt_lo, w.rt_hi)
        ok = np.array_equal(expected.view(np.uint32), axis.view(np.uint32))
        out.setdefault(str(w.band), [])
        if not ok:
            out[str(w.band)].append(cid)
    return out


# --------------------------------------------------------------------------- ungrouped


@pytest.mark.parametrize("name", ["single", "experiment", "topk"])
def test_ungrouped_window_equals_the_run_windows_row(open_fixture, name):
    rs = open_fixture(name)
    for run in rs.runs:
        rows = run_windows_rows(run.artifacts["run_windows"].path)
        ex = pq.read_table(
            run.artifacts["psms_extracted"].path, columns=["candidate_id", "rt_pred_cal"]
        ).to_pylist()
        cal = json.loads(run.side_files["cal"].read_text())
        prefix = f"{run.name}/" if run.name else ""
        for row in ex:
            cid = row["candidate_id"]
            w = rt_window(rs, run, cid)
            assert w is not None and w.candidate_id == cid and w.band is None
            assert int(rows["candidate_id"][cid]) == cid
            for field in ("rt_pred_cal", "rt_lo", "rt_hi"):
                assert f64bits(getattr(w, field)) == f64bits(float(rows[field][cid])), field
            assert f64bits(w.rt_pred_cal) == f64bits(row["rt_pred_cal"])
            assert w.im_pred_cal is None and w.im_lo is None and w.im_hi is None
            assert w.bounded and w.half_width == (w.rt_hi - w.rt_lo) / 2
            assert math.isclose(w.half_width, cal["w_rt"], rel_tol=1e-9, abs_tol=1e-9)
            assert w.source == f"{prefix}run_windows.parquet row {cid}"
            assert "calibration_status loess" in w.calibration_note


def test_window_outside_the_table_is_none(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    n = pq.read_metadata(run.artifacts["run_windows"].path).num_rows
    assert rt_window(rs, run, n - 1) is not None
    assert rt_window(rs, run, n) is None
    assert rt_window(rs, run, -1) is None
    with pytest.raises(ValueError, match="not a grouped run"):
        rt_window(rs, run, 4, band="g00")


def test_missing_run_windows_is_refused(open_fixture):
    rs = open_fixture("chrom_v1")  # trimmed: the manifest records run_windows, the file is gone
    with pytest.raises(ArtifactNotFound, match="run_windows"):
        rt_window(rs, rs.runs[0], 4)


# --------------------------------------------------------------------------- band offsets


@pytest.mark.parametrize("name", ["grouped", "ovl_bp", "ovl_rg50"])
def test_band_offsets_from_the_library(open_fixture, name):
    rs = open_fixture(name)
    offsets = band_offsets(rs, rs.runs[0])
    assert {k: tuple(v) for k, v in offsets.items()} == OFFSETS[name]
    for band, bo in offsets.items():
        assert isinstance(bo, BandOffset) and bo.method == "library"
        # The library passed every check against the run's own tables.
        assert f"the library span holds n = {bo.n} precursors" in bo.checked
        assert "chromatograms.parquet and psms_competed.parquet lies inside" in bo.checked
        assert "precursor_mz agrees at" in bo.checked and "seed_psms.parquet" in bo.checked
        offset, n = bo
        assert (
            n
            == pq.read_metadata(rs.runs[0].root / "groups" / band / "run_windows.parquet").num_rows
        )
        assert bo.holds(offset) and bo.holds(offset + n - 1) and not bo.holds(offset + n)
    assert band_offsets(rs, rs.runs[0]) == offsets  # memoised, returned as a copy


@pytest.mark.parametrize("name", ["ovl_bp", "ovl_rg50"])
def test_seed_join_offsets_are_labelled_inferred(open_fixture, name):
    rs = open_fixture(name)
    inferred = band_offsets(rs, rs.runs[0], library=False)
    assert {k: tuple(v) for k, v in inferred.items()} == OFFSETS[name]
    assert all(bo.method == "inferred" and "seed" in bo.note for bo in inferred.values())


def test_ungrouped_run_has_no_band_offsets(open_fixture):
    rs = open_fixture("single")
    assert band_offsets(rs, rs.runs[0]) == {}


def test_without_the_library_the_seed_join_is_used(fixture_dir, open_fixture, tmp_path):
    run_dir = tmp_path / "ovl_bp"
    shutil.copytree(fixture_dir("ovl_bp"), run_dir)
    os.remove(run_dir / "fragment_library_precursors.parquet")
    rs = open_results(run_dir)
    run = rs.runs[0]
    offsets = band_offsets(rs, run)
    assert {k: tuple(v) for k, v in offsets.items()} == OFFSETS["ovl_bp"]
    assert {bo.method for bo in offsets.values()} == {"inferred"}
    original = open_fixture("ovl_bp")
    src = ChromatogramSource.for_run(rs, run)
    for band in ("g00", "g01", "g02"):
        cid = next(c for c in src.candidate_ids().tolist() if src.band_of(c) == band)
        w = rt_window(rs, run, cid)
        assert w.band == band and "offset inferred" in w.source
        assert w.rt_lo == rt_window(original, original.runs[0], cid).rt_lo


def test_library_span_mismatch_in_one_band_rejects_it_for_every_band(fixture_dir, tmp_path):
    run_dir = copy_run(fixture_dir, "ovl_bp", tmp_path)
    plan_path = run_dir / "groups" / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["bands"][1]["mz_hi"] = plan["bands"][1]["mz_hi"] - 50.0  # a plan the run did not use
    plan_path.write_text(json.dumps(plan))
    rs = open_results(run_dir)
    offsets = band_offsets(rs, rs.runs[0])
    assert {k: tuple(v) for k, v in offsets.items()} == OFFSETS["ovl_bp"]
    for bo in offsets.values():
        assert bo.method == "inferred"
        assert "it is used for no band" in bo.note and "band g01: the span" in bo.note


def test_wrong_library_is_used_for_no_band(fixture_dir, tmp_path):
    """Probe p2, scenario A: library rows inside band g00 removed, ids renumbered.

    The library then gives g00 a failing span check and g01, g02 offsets that are 20 rows
    too low. Every band's offset comes from the seeds, and every window reproduces the
    engine's axis rule.
    """
    true_mz = library_mz(fixture_dir("ovl_bp"))
    run_dir = copy_run(fixture_dir, "ovl_bp", tmp_path)
    drop_library_rows(run_dir, list(range(1000, 1020)))
    rs = open_results(run_dir)
    run = rs.runs[0]
    offsets = band_offsets(rs, run)
    assert {k: tuple(v) for k, v in offsets.items()} == OFFSETS["ovl_bp"]
    for bo in offsets.values():
        assert bo.method == "inferred" and "it is used for no band" in bo.note
    note = offsets["g01"].note
    assert "outside the row span [1720, 3322)" in note  # the rejected g01 offset
    failures = axis_failures(rs, run, true_mz)
    assert set(failures) == {"g00", "g01", "g02"} and not any(failures.values())
    w = rt_window(rs, run, 1750)
    assert w.band == "g01" and "band offset 1740, offset inferred" in w.source


def test_wrong_library_without_band_seeds_refuses_that_band(fixture_dir, tmp_path):
    """Probe p2, scenario B: as A, plus rows 0-1 removed and the g00 band seeds deleted."""
    true_mz = library_mz(fixture_dir("ovl_bp"))
    run_dir = copy_run(fixture_dir, "ovl_bp", tmp_path)
    drop_library_rows(run_dir, [0, 1, *range(1000, 1020)])
    os.remove(run_dir / "groups" / "g00" / "seed_psms.parquet")
    rs = open_results(run_dir)
    run = rs.runs[0]
    offsets = band_offsets(rs, run)
    assert "g00" not in offsets
    assert {k: tuple(v) for k, v in offsets.items()} == {
        k: v for k, v in OFFSETS["ovl_bp"].items() if k != "g00"
    }
    with pytest.raises(ArtifactNotFound, match="ids of band g00 cannot be placed") as err:
        rt_window(rs, run, 4)
    assert "it is used for no band" in str(err.value) and "seed_psms.parquet" in str(err.value)
    with pytest.raises(ArtifactNotFound, match="band g00 cannot be placed"):
        rt_window(rs, run, 4, band="g00")
    failures = axis_failures(rs, run, true_mz)
    assert set(failures) == {"ArtifactNotFound", "g01", "g02"}
    assert not failures["g01"] and not failures["g02"]
    held = set(ChromatogramSource.for_run(rs, run).candidate_ids().tolist())
    g00_only = {c for c in held if c < OFFSETS["ovl_bp"]["g01"][0]}
    assert g00_only <= set(failures["ArtifactNotFound"])


def test_library_with_other_precursor_mz_fails_the_precursor_check(fixture_dir, tmp_path):
    """Every span and id range agrees, but the precursor m/z values are not the run's."""
    run_dir = copy_run(fixture_dir, "ovl_bp", tmp_path)
    path = run_dir / "fragment_library_precursors.parquet"
    lib = pq.read_table(path)
    mz = lib.column("precursor_mz").to_numpy() * (1.0 + 1e-12)
    i = lib.schema.get_field_index("precursor_mz")
    pq.write_table(lib.set_column(i, "precursor_mz", pa.array(mz, pa.float64())), path)
    forget_hash(run_dir, "fragment_library_precursors", path.name)
    rs = open_results(run_dir)
    offsets = band_offsets(rs, rs.runs[0])
    assert {k: tuple(v) for k, v in offsets.items()} == OFFSETS["ovl_bp"]
    for bo in offsets.values():
        assert bo.method == "inferred"
        assert "precursor_mz differs from seed_psms.parquet" in bo.note
        assert "the span" not in bo.note and "outside the row span" not in bo.note


def test_inferred_offset_must_hold_the_band_tables(fixture_dir, tmp_path):
    """A seed join that gives a constant but wrong offset is refused, not used."""
    run_dir = copy_run(fixture_dir, "ovl_bp", tmp_path)
    os.remove(run_dir / "fragment_library_precursors.parquet")
    seed_path = run_dir / "groups" / "g01" / "seed_psms.parquet"
    seed = pq.read_table(seed_path)
    ids = seed.column("candidate_id").to_numpy().astype(np.uint32) + np.uint32(5)
    pq.write_table(seed.set_column(0, "candidate_id", pa.array(ids, pa.uint32())), seed_path)
    forget_hash(run_dir, "seed_psms[g01]", "groups/g01/seed_psms.parquet")
    rs = open_results(run_dir)
    run = rs.runs[0]
    offsets = band_offsets(rs, run)
    assert {k: tuple(v) for k, v in offsets.items()} == {
        k: v for k, v in OFFSETS["ovl_bp"].items() if k != "g01"
    }
    assert all(b.method == "inferred" for b in offsets.values())
    src = ChromatogramSource.for_run(rs, run)
    cid = next(c for c in src.candidate_ids().tolist() if src.band_of(c) == "g01")
    with pytest.raises(
        ArtifactNotFound, match=r"band g01 cannot be placed: .*seed-join offset 1735 fails"
    ) as err:
        rt_window(rs, run, cid)
    assert "outside the row span [1735, 3337)" in str(err.value)


# --------------------------------------------------------------------------- grouped windows


@pytest.mark.parametrize("name", ["ovl_rg50", "ovl_bp"])
def test_band_local_windows_reproduce_the_chromatogram_axes(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    scans = scan_table(run.root)
    mz = library_mz(run.root)
    offsets = band_offsets(rs, run)
    band_rows = {
        b: run_windows_rows(run.root / "groups" / b / "run_windows.parquet") for b in offsets
    }
    checked = compared = wrong_row = 0
    for cid in src.candidate_ids().tolist():
        chrom = src.read(cid)
        axis = chrom.common_axis()
        if axis is None:
            continue
        w = rt_window(rs, run, cid)
        assert w is not None and w.band == chrom.band == src.band_of(cid)
        assert w.bounded
        local = cid - offsets[w.band].offset
        assert f"groups/{w.band}/run_windows.parquet row {local} " in w.source
        rows = band_rows[w.band]
        assert f64bits(w.rt_lo) == f64bits(float(rows["rt_lo"][local]))
        assert f64bits(w.rt_hi) == f64bits(float(rows["rt_hi"][local]))
        expected = covering_axis(scans, float(mz[cid]), w.rt_lo, w.rt_hi)
        assert np.array_equal(expected.view(np.uint32), axis.view(np.uint32)), cid
        checked += 1
        # The library-wide id read as a band-local row gives the wrong window (G4 F10).
        if cid < len(rows["rt_lo"]):
            other = covering_axis(scans, float(mz[cid]), rows["rt_lo"][cid], rows["rt_hi"][cid])
            compared += 1
            wrong_row += not np.array_equal(other.view(np.uint32), axis.view(np.uint32))
    assert checked >= 300
    assert compared > 100 and wrong_row > 0.8 * compared


def test_unbounded_sentinel_gives_none_bounds(open_fixture):
    rs = open_fixture("grouped")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    for band in ("g00", "g01", "g02"):
        rows = run_windows_rows(run.root / "groups" / band / "run_windows.parquet")
        assert np.isnan(rows["rt_pred_cal"]).all()
        assert np.isneginf(rows["rt_lo"]).all() and np.isposinf(rows["rt_hi"]).all()
    for cid in src.candidate_ids().tolist():
        w = rt_window(rs, run, cid)
        assert w.band == src.band_of(cid)
        assert w.rt_pred_cal is None and w.rt_lo is None and w.rt_hi is None
        assert w.half_width is None and not w.bounded
        assert "unbounded" in w.calibration_note
        assert "insufficient_anchors_unbounded" in w.calibration_note


@pytest.mark.parametrize(
    ("bands", "pooled"),
    [("grouped", "grouped_pool"), ("ovl_bp", "ovl_bp_pool"), ("ovl_rg50", "ovl_rg50_pool")],
)
def test_pooled_run_windows_equal_the_unpooled_twin(open_fixture, bands, pooled):
    rs_b, rs_p = open_fixture(bands), open_fixture(pooled)
    src = ChromatogramSource.for_run(rs_b, rs_b.runs[0])
    chosen = set()
    for cid in src.candidate_ids().tolist():
        a = rt_window(rs_b, rs_b.runs[0], cid)
        b = rt_window(rs_p, rs_p.runs[0], cid)
        assert a.band == b.band, cid
        for field in ("rt_pred_cal", "rt_lo", "rt_hi"):
            x, y = getattr(a, field), getattr(b, field)
            assert (x is None and y is None) or f64bits(x) == f64bits(y)
        chosen.add(b.source.split("band chosen as ")[1])
    if bands != "grouped":
        # Overlap candidates of a pooled run are placed by matching the band tables.
        assert "the band table whose rows equal the pooled chromatogram rows" in chosen


def test_band_given_by_the_caller(open_fixture):
    rs = open_fixture("ovl_bp")
    run = rs.runs[0]
    rows = run_windows_rows(run.root / "groups" / "g01" / "run_windows.parquet")
    w = rt_window(rs, run, 1740, band="g01")
    assert w.band == "g01" and "band given by the caller" in w.source
    assert f64bits(w.rt_lo) == f64bits(float(rows["rt_lo"][0]))
    assert rt_window(rs, run, 5, band="g01") is None  # outside g01's row span
    with pytest.raises(ValueError, match="no band g99"):
        rt_window(rs, run, 1740, band="g99")
    # An integer could be a plan index or a loser-file position; it is refused.
    for value in (1, np.int64(1)):
        with pytest.raises(TypeError, match=r"ChromatogramTable\.position"):
            rt_window(rs, run, 1740, band=value)  # type: ignore[arg-type]


def test_band_argument_is_the_band_name_not_the_loser_position(open_fixture):
    """ovl128: the table at loser position 84 is g85 (bands were skipped before it)."""
    rs = open_fixture("ovl128")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    table = next(t for t in src.tables if t.position == 84)
    assert table.band == "g85" and 3229 in table.drop
    with pytest.raises(TypeError, match="differ when bands were skipped"):
        rt_window(rs, run, 3229, band=table.position)  # type: ignore[arg-type]
    # The name reaches g85; the trimmed fixture has no band run_windows.
    with pytest.raises(ArtifactNotFound, match=r"run_windows\[g85\]"):
        rt_window(rs, run, 3229, band=table.band)


def _identical_g00_windows(fixture_dir, tmp_path) -> tuple[Path, list[int]]:
    """Probe p8: a copy of ovl_bp_pool whose g00 run_windows rows of the overlap candidates
    that the run kept in g01 are overwritten with their g01 rows (identical windows, as
    under global calibration). Returns the copy and those candidates."""
    rs_b = open_results(fixture_dir("ovl_bp"))
    src_b = ChromatogramSource.for_run(rs_b, rs_b.runs[0])
    g00 = next(t for t in src_b.tables if t.band == "g00")
    kept_g01 = sorted(c for c in g00.drop if src_b.band_of(c) == "g01")
    assert kept_g01 == [1750, 1806]
    pool = copy_run(fixture_dir, "ovl_bp_pool", tmp_path)
    p00 = pool / "groups" / "g00" / "run_windows.parquet"
    t00 = pq.read_table(p00)
    t01 = pq.read_table(pool / "groups" / "g01" / "run_windows.parquet")
    cols = {c: t00.column(c).to_pylist() for c in t00.column_names}
    for c in kept_g01:
        l0, l1 = c - OFFSETS["ovl_bp"]["g00"][0], c - OFFSETS["ovl_bp"]["g01"][0]
        for f in ("rt_pred_cal", "rt_lo", "rt_hi"):
            cols[f][l0] = t01.column(f)[l1].as_py()
    pq.write_table(pa.table(cols, schema=t00.schema), p00)
    forget_hash(pool, "run_windows[g00]", "groups/g00/run_windows.parquet")
    return pool, kept_g01


def test_pooled_identical_windows_name_the_band_the_run_kept(fixture_dir, open_fixture, tmp_path):
    pool, kept = _identical_g00_windows(fixture_dir, tmp_path)
    rs_b = open_fixture("ovl_bp")
    rs_p = open_results(pool)
    for c in kept:
        wb = rt_window(rs_b, rs_b.runs[0], c)
        wp = rt_window(rs_p, rs_p.runs[0], c)
        assert wb.band == wp.band == "g01", c
        assert (wp.rt_lo, wp.rt_hi) == (wb.rt_lo, wb.rt_hi)
        assert "the band table whose rows equal the pooled chromatogram rows" in wp.source


def test_pooled_band_from_the_competed_rows(fixture_dir, open_fixture, tmp_path):
    """Without band chromatogram tables, the pooled competed row names the kept band."""
    pool, kept = _identical_g00_windows(fixture_dir, tmp_path)
    for b in ("g00", "g01", "g02"):
        os.remove(pool / "groups" / b / "chromatograms.parquet")
    rs_b = open_fixture("ovl_bp")
    rs_p = open_results(pool)
    for c in kept:
        wp = rt_window(rs_p, rs_p.runs[0], c)
        assert wp.band == rt_window(rs_b, rs_b.runs[0], c).band == "g01"
        assert "band psms_competed table whose rows equal the pooled competed rows" in wp.source


def test_pooled_identical_windows_without_evidence_leave_the_band_open(fixture_dir, tmp_path):
    """Neither band tables nor band competed tables: the band is not guessed."""
    pool, kept = _identical_g00_windows(fixture_dir, tmp_path)
    for b in ("g00", "g01", "g02"):
        os.remove(pool / "groups" / b / "chromatograms.parquet")
        os.remove(pool / "groups" / b / "psms_competed.parquet")
    rs_p = open_results(pool)
    run = rs_p.runs[0]
    rows01 = run_windows_rows(pool / "groups" / "g01" / "run_windows.parquet")
    for c in kept:
        w = rt_window(rs_p, run, c)
        assert w.band is None
        assert "band not determined; bands g00, g01 give identical windows" in w.source
        l0, l1 = c - OFFSETS["ovl_bp"]["g00"][0], c - OFFSETS["ovl_bp"]["g01"][0]
        assert f"groups/g00/run_windows.parquet row {l0} " in w.source
        assert f"groups/g01/run_windows.parquet row {l1} " in w.source
        assert f64bits(w.rt_lo) == f64bits(float(rows01["rt_lo"][l1]))
        assert "the band was not determined" in w.calibration_note
    # Where the windows differ, the band is still refused rather than guessed.
    with pytest.raises(AmbiguousBand):
        rt_window(rs_p, run, 1740)


def test_ambiguous_band_without_chromatograms(open_fixture):
    rs = open_fixture("ovl_bp")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    held = set(src.candidate_ids().tolist())
    offsets = band_offsets(rs, run)
    both = [
        c
        for c in range(offsets["g01"].offset, offsets["g00"].offset + offsets["g00"].n)
        if c not in held
    ]
    assert both
    cid = both[0]
    with pytest.raises(AmbiguousBand, match="pass band="):
        rt_window(rs, run, cid)
    a, b = rt_window(rs, run, cid, band="g00"), rt_window(rs, run, cid, band="g01")
    assert a.band == "g00" and b.band == "g01" and a.rt_lo != b.rt_lo


def test_candidate_outside_every_band(open_fixture):
    rs = open_fixture("grouped")
    run = rs.runs[0]
    assert rt_window(rs, run, 0) is None  # library rows 0..3 lie below the first band
    assert rt_window(rs, run, 3819) is None


def test_band_offsets_that_cannot_be_found(open_fixture):
    rs = open_fixture("ovl128")  # trimmed: band run_windows, library and seeds are gone
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    with pytest.raises(ArtifactNotFound, match="run_windows"):
        rt_window(rs, rs.runs[0], int(src.candidate_ids()[0]))
    rs = open_fixture("ovl128_pool")
    with pytest.raises(ArtifactNotFound, match="row spans of 120 band"):
        rt_window(rs, rs.runs[0], 30)


# --------------------------------------------------------------------------- real data


@pytest.mark.real_data
def test_real_single_windows(real_single):
    rs = open_results(real_single)
    run = rs.runs[0]
    ex = pq.read_table(
        run.artifacts["psms_extracted"].path, columns=["candidate_id", "rt_pred_cal"]
    )
    cid = ex.column("candidate_id").to_numpy()
    pred = ex.column("rt_pred_cal").to_numpy()
    sample = np.random.default_rng(20260930).choice(cid.size, 50, replace=False)
    times = []
    for k in sample.tolist():
        t0 = time.perf_counter()
        w = rt_window(rs, run, int(cid[k]))
        times.append(time.perf_counter() - t0)
        expected = finite_or_none(float(pred[k]))
        assert (w.rt_pred_cal is None and expected is None) or f64bits(w.rt_pred_cal) == f64bits(
            float(pred[k])
        )
    print(
        f"\nrt_window on {run.artifacts['run_windows'].rows} run_windows rows: median "
        f"{1e3 * np.median(times):.1f} ms, max {1e3 * np.max(times):.1f} ms"
    )
