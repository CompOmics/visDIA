"""Tests of the spectra layer: scan table, apex scan, XIC grid, MS1 relations, window scheme.

Every engine rule is checked against an independent pyarrow read of the fixture files,
and against the engine's own outputs where they encode the rule (the scored apex RTs,
the v1 chromatogram axes, the band axes of overlapping windows, the MS1 columns of
``psms_extracted``).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data import spectra as spectra_module
from mumdia_viewer.data.errors import (
    ArtifactNotFound,
    InconsistentData,
    LayoutError,
    SchemaVersionError,
)
from mumdia_viewer.data.pqio import ROW_GROUP_CACHE
from mumdia_viewer.data.spectra import (
    GRID_AXIS_LABEL,
    GRID_UNKNOWN_LABEL,
    MS1_SUM_LABEL,
    NEAREST_MS1_LABEL,
    SPARSE_GRID_LABEL,
    SPECTRUM_CACHE,
    Ms1Table,
    ScanTable,
    isolation_scheme,
    isotope_mz,
    sum_near,
)

LARGE = pa.large_list(pa.float32())
SMALL = pa.list_(pa.float32())
C13 = 1.003354835


# --------------------------------------------------------------------------- helpers


def _ms2_scalars(path: Path) -> dict[str, np.ndarray]:
    """The MS2 scalar columns read directly with pyarrow (the oracle)."""
    t = pq.read_table(
        path,
        columns=["scan_index", "rt_seconds", "window_id", "window_lower", "window_upper"],
    )
    return {
        "scan_index": t["scan_index"].to_numpy().astype(np.int64),
        "rt": t["rt_seconds"].to_numpy(),
        "window_id": t["window_id"].to_numpy().astype(np.int64),
        "lower": t["window_lower"].to_numpy(),
        "upper": t["window_upper"].to_numpy(),
    }


def _library_mz(rs) -> np.ndarray:
    lib = rs.artifact("fragment_library_precursors")
    t = pq.read_table(lib.path, columns=["candidate_id", "precursor_mz"])
    cid = t["candidate_id"].to_numpy()
    assert np.array_equal(cid, np.arange(cid.size))  # candidate_id == row
    return t["precursor_mz"].to_numpy()


def _scored_pairs(rs, run) -> tuple[np.ndarray, np.ndarray]:
    """(precursor_mz, apex_rt) of every scored row of one run."""
    scored = run.artifact("psms_scored") or rs.scored
    t = pq.read_table(scored.path, columns=["candidate_id", "apex_rt", "selected_peak_rank"])
    cid = t["candidate_id"].to_numpy().astype(np.int64)
    apex = t["apex_rt"].to_numpy()
    rank = t["selected_peak_rank"].to_numpy()
    extracted = run.artifact("psms_extracted")
    if extracted is not None and extracted.present:
        e = pq.read_table(extracted.path, columns=["candidate_id", "peak_rank", "precursor_mz"])
        by_key = {
            (int(c), int(r)): float(m)
            for c, r, m in zip(
                e["candidate_id"].to_pylist(),
                e["peak_rank"].to_pylist(),
                e["precursor_mz"].to_pylist(),
                strict=True,
            )
        }
        pmz = np.array([by_key[(int(c), int(r))] for c, r in zip(cid, rank, strict=True)])
    else:  # grouped runs delete band psms_extracted; the library holds the same value
        pmz = _library_mz(rs)[cid]
    return pmz, apex


def _nearest_oracle(ms2: dict, pmz: float, t: float) -> int:
    """Covering-window row nearest to t: ties to the lower RT, then the lower row."""
    rows = np.flatnonzero((ms2["lower"] <= pmz) & (pmz <= ms2["upper"]))
    d = np.abs(ms2["rt"][rows] - t)
    return int(rows[np.lexsort((rows, ms2["rt"][rows], d))[0]])


def _grid_oracle(ms2: dict, pmz: float, lo: float, hi: float) -> np.ndarray:
    mask = (ms2["lower"] <= pmz) & (pmz <= ms2["upper"]) & (ms2["rt"] >= lo) & (ms2["rt"] <= hi)
    rows = np.flatnonzero(mask)
    rows = rows[np.lexsort((rows, ms2["rt"][rows]))]
    r = ms2["rt"][rows]
    keep = np.ones(rows.size, dtype=bool)
    keep[1:] = r[1:] != r[:-1]
    return rows[keep]


def _bits32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _write_ms2(
    path: Path,
    windows: dict[int, tuple[float, float]],
    scans: list[tuple[int, float]],
    *,
    peaks: list[tuple[list[float] | None, list[float] | None]] | None = None,
    list_type: pa.DataType = LARGE,
    row_group_size: int | None = None,
    scan_index: list[int] | None = None,
) -> None:
    """A synthetic spectra_ms2.parquet: ``scans`` are (window_id, rt) in file order."""
    n = len(scans)
    if peaks is None:
        peaks = [([100.0 + i, 200.0 + i], [10.0, 20.0]) for i in range(n)]
    idx = scan_index if scan_index is not None else [2 * i + 1 for i in range(n)]
    lo = [windows[w][0] for w, _ in scans]
    hi = [windows[w][1] for w, _ in scans]
    table = pa.table(
        {
            "scan_index": pa.array(idx, pa.uint32()),
            "id": pa.array([f"scan={i + 1}" for i in idx], pa.string()),
            "rt_seconds": pa.array([rt for _, rt in scans], pa.float64()),
            "window_id": pa.array([w for w, _ in scans], pa.uint32()),
            "window_target": pa.array([(a + b) / 2 for a, b in zip(lo, hi, strict=True)]),
            "window_lower": pa.array(lo, pa.float64()),
            "window_upper": pa.array(hi, pa.float64()),
            "precursor_mz": pa.array([None] * n, pa.float64()),
            "precursor_charge": pa.array([None] * n, pa.int32()),
            "mz": pa.array([p[0] for p in peaks], list_type),
            "intensity": pa.array([p[1] for p in peaks], list_type),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _write_windows(path: Path, windows: dict[int, tuple[float, float]]) -> None:
    ids = sorted(windows)
    pq.write_table(
        pa.table(
            {
                "window_id": pa.array(ids, pa.uint32()),
                "target": pa.array([(windows[i][0] + windows[i][1]) / 2 for i in ids]),
                "lower": pa.array([windows[i][0] for i in ids], pa.float64()),
                "upper": pa.array([windows[i][1] for i in ids], pa.float64()),
            }
        ),
        path,
    )


def _write_ms1(path: Path, scan_index: list[int], rts: list[float]) -> None:
    n = len(rts)
    pq.write_table(
        pa.table(
            {
                "scan_index": pa.array(scan_index, pa.uint32()),
                "rt_seconds": pa.array(rts, pa.float64()),
                "mz": pa.array([[500.0 + i] for i in range(n)], LARGE),
                "intensity": pa.array([[1.0 + i] for i in range(n)], LARGE),
            }
        ),
        path,
    )


def _write_link(path: Path, ms2: list[int], ms1: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "ms2_scan_index": pa.array(ms2, pa.uint32()),
                "ms1_scan_index": pa.array(ms1, pa.int32()),
            }
        ),
        path,
    )


def _edited_copy(tmp_path: Path, source: Path, edit) -> Path:
    """A copy of a fixture whose manifest ``edit(manifest, config)`` changed."""
    root = tmp_path / source.name
    shutil.copytree(source, root)
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    cfg = json.loads(m["config_json"])
    edit(m, cfg)
    m["config_json"] = json.dumps(cfg)
    manifest.write_text(json.dumps(m), encoding="utf-8")
    return root


# --------------------------------------------------------------------------- scan table


def test_scan_table_matches_pyarrow_and_is_memoised(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    assert ScanTable.for_run(rs, run) is st
    assert ScanTable.for_run(rs, 0) is st
    path = run.artifact("spectra_ms2").path
    oracle = _ms2_scalars(path)
    assert st.n == 480
    for name in ("scan_index", "rt", "window_id", "lower", "upper"):
        assert np.array_equal(getattr(st, name), oracle[name]), name
    t = pq.read_table(path, columns=["window_target", "precursor_mz", "precursor_charge"])
    assert np.array_equal(st.target, t["window_target"].to_numpy())
    assert np.array_equal(st.precursor_mz, t["precursor_mz"].to_numpy())
    assert t["precursor_charge"].null_count == 480
    assert np.all(st.charge == 0)  # null charge
    assert not st.rt.flags.writeable
    windows = pq.read_table(run.artifact("isolation_windows").path).to_pandas()
    got = st.windows
    assert got["window_id"].tolist() == windows["window_id"].tolist()
    for col in ("target", "lower", "upper"):
        assert np.array_equal(got[col].to_numpy(), windows[col].to_numpy())
    assert st.windows_source == "isolation_windows.parquet"


def test_a_run_without_spectra(open_fixture):
    rs = open_fixture("chrom_v1")  # trimmed fixture: the spectra tables are missing
    with pytest.raises(ArtifactNotFound, match="spectra_ms2"):
        ScanTable.for_run(rs, 0)
    with pytest.raises(ArtifactNotFound, match="spectra_ms1"):
        Ms1Table.for_run(rs, 0)
    with pytest.raises(ArtifactNotFound):
        isolation_scheme(rs, 0)


def test_an_unknown_spectra_schema_version_is_refused(tmp_path, fixture_dir):
    root = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), root)
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["artifacts"]["spectra_ms2"]["schema_version"] = 99
    manifest.write_text(json.dumps(m), encoding="utf-8")
    rs = open_results(root)
    assert "unsupported_version" in rs.notice_codes()
    with pytest.raises(SchemaVersionError, match="spectra_ms2 schema version 99"):
        ScanTable.for_run(rs, 0)


def test_the_scan_table_is_reloaded_when_the_file_changes(tmp_path, fixture_dir):
    root = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), root)
    rs = open_results(root)
    first = ScanTable.for_run(rs, 0)
    assert ScanTable.for_run(rs, 0) is first
    path = rs.runs[0].artifact("spectra_ms2").path
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    second = ScanTable.for_run(rs, 0)
    assert second is not first and np.array_equal(second.rt, first.rt)


def test_scan_index_is_not_the_row(open_fixture):
    rs = open_fixture("single")
    st = ScanTable.for_run(rs, 0)
    rows = np.arange(st.n)
    assert np.any(st.scan_index != rows)
    for row in rows:
        assert st.row_of_scan_index(int(st.scan_index[row])) == row
    m1 = Ms1Table.for_run(rs, 0)
    for scan in m1.scan_index:  # MS1 scan indices are not MS2 rows
        assert st.row_of_scan_index(int(scan)) is None
    assert st.row_of_scan_index(10**9) is None


def test_run_keys_of_any_integer_type(open_fixture):
    """A ``source`` value read from a pooled scored table (uint32) addresses its run."""
    rs = open_fixture("experiment")
    source = pq.read_table(rs.scored.path, columns=["source"])["source"].to_numpy()
    assert source.dtype == np.uint32
    for value in np.unique(source):
        run = rs.run(int(value))
        assert ScanTable.for_run(rs, value) is ScanTable.for_run(rs, run)
        assert ScanTable.for_run(rs, np.int64(value)) is ScanTable.for_run(rs, run.name)
        assert Ms1Table.for_run(rs, value) is Ms1Table.for_run(rs, run)
        assert isolation_scheme(rs, value).equals(isolation_scheme(rs, run))
    for bad in (True, np.True_, 1.0, np.float64(0.0), None, [0]):
        with pytest.raises(TypeError, match="source index"):
            ScanTable.for_run(rs, bad)
    with pytest.raises(KeyError, match="no run 7"):
        ScanTable.for_run(rs, np.uint32(7))


def test_scan_index_lookup_on_unsorted_file(tmp_path):
    windows = {0: (400.0, 500.0)}
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, windows, [(0, 1.0), (0, 2.0), (0, 3.0)], scan_index=[7, 3, 5])
    st = ScanTable(path)
    assert [st.row_of_scan_index(s) for s in (7, 3, 5, 4)] == [0, 1, 2, None]


# --------------------------------------------------------------------------- windows


def test_covering_windows_inclusive_on_the_smoke_scheme(open_fixture):
    st = ScanTable.for_run(open_fixture("single"), 0)
    w = st.windows.sort_values(["lower", "upper"]).reset_index(drop=True)
    touching = overlaps = gaps = 0
    for i in range(len(w) - 1):
        up, nxt = float(w.upper[i]), float(w.lower[i + 1])
        a, b = int(w.window_id[i]), int(w.window_id[i + 1])
        if up == nxt:  # a bound shared bit for bit: both windows cover it
            touching += 1
            assert st.covering_windows(up).tolist() == [a, b]
            assert st.covering_windows(np.nextafter(up, -np.inf)).tolist() == [a]
            assert st.covering_windows(np.nextafter(up, np.inf)).tolist() == [b]
        elif up > nxt:  # overlap: both bounds and the inside are covered twice
            overlaps += 1
            for pmz in (nxt, (nxt + up) / 2, up):
                assert st.covering_windows(pmz).tolist() == [a, b]
        else:  # gap: nothing covers the inside, each bound only its own window
            gaps += 1
            assert st.covering_windows((up + nxt) / 2).tolist() == []
            assert st.covering_windows(up).tolist() == [a]
            assert st.covering_windows(nxt).tolist() == [b]
    assert (touching, overlaps, gaps) == (5, 1, 1)  # facts F6 C1.3
    assert st.covering_windows(float(w.lower[0])).tolist() == [int(w.window_id[0])]
    assert st.covering_windows(np.nextafter(float(w.lower[0]), -np.inf)).tolist() == []
    last = float(w.upper.iloc[-1])
    assert st.covering_windows(last).tolist() == [int(w.window_id.iloc[-1])]
    assert st.covering_windows(np.nextafter(last, np.inf)).tolist() == []
    assert st.covering_windows(float("nan")).tolist() == []


def test_covering_windows_synthetic(tmp_path):
    # window ids in acquisition order, not m/z order; 1 and 3 touch, 3 and 0 overlap,
    # 0 and 2 have a gap
    windows = {1: (400.0, 500.0), 3: (500.0, 600.0), 0: (599.5, 700.0), 2: (700.25, 800.0)}
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, windows, [(1, 1.0), (3, 1.1), (0, 1.2), (2, 1.3)])
    wpath = tmp_path / "windows.parquet"
    _write_windows(wpath, windows)
    for st in (ScanTable(path, wpath), ScanTable(path)):
        assert st.covering_windows(500.0).tolist() == [1, 3]
        assert st.covering_windows(np.nextafter(500.0, -np.inf)).tolist() == [1]
        assert st.covering_windows(np.nextafter(500.0, np.inf)).tolist() == [3]
        assert st.covering_windows(599.5).tolist() == [3, 0]
        assert st.covering_windows(599.75).tolist() == [3, 0]
        assert st.covering_windows(600.0).tolist() == [3, 0]
        assert st.covering_windows(700.0).tolist() == [0]
        assert st.covering_windows(700.1).tolist() == []
        assert st.covering_windows(400.0).tolist() == [1]
        assert st.covering_windows(800.0).tolist() == [2]
        assert st.covering_windows(np.nextafter(800.0, np.inf)).tolist() == []
    assert ScanTable(path).windows_source.startswith("derived from spectra_ms2")


def test_window_table_must_agree_with_the_scans(tmp_path):
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, {0: (400.0, 500.0), 1: (500.0, 600.0)}, [(0, 1.0), (1, 1.1)])
    other = tmp_path / "other.parquet"
    _write_windows(other, {0: (400.0, 500.0), 1: (500.0, 600.5)})
    with pytest.raises(InconsistentData, match="window bounds"):
        ScanTable(path, other)
    partial = tmp_path / "partial.parquet"
    _write_windows(partial, {0: (400.0, 500.0)})
    with pytest.raises(InconsistentData, match="does not list"):
        ScanTable(path, partial)


def test_isolation_scheme_smoke(open_fixture):
    rs = open_fixture("single")
    df = isolation_scheme(rs, rs.runs[0])
    assert list(df.columns) == [
        "window_id",
        "lower",
        "upper",
        "target",
        "width",
        "n_scans",
        "overlap_with_next",
        "cycle_time_s",
        "aif",
    ]
    assert len(df) == 8
    assert df["lower"].is_monotonic_increasing
    assert np.array_equal(df["width"].to_numpy(), (df["upper"] - df["lower"]).to_numpy())
    assert df["n_scans"].tolist() == [60] * 8
    ov = df["overlap_with_next"].to_numpy()
    assert np.isnan(ov[-1])
    assert int(np.sum(ov[:-1] == 0.0)) == 5
    assert ov[2] == pytest.approx(1137.1234130859375 - 1137.123291015625)  # overlap > 0
    assert ov[5] < 0  # gap between windows 5 and 6
    assert ov[5] == pytest.approx(-0.000244, abs=1e-6)
    assert np.allclose(df["cycle_time_s"].to_numpy(), 2.0, atol=1e-3)
    assert not df["aif"].any()


def test_isolation_scheme_overlapping_windows(open_fixture):
    df = isolation_scheme(open_fixture("ovl_bp"), 0)
    ov = df["overlap_with_next"].to_numpy()[:-1]
    assert np.all(ov > 20.0)  # the overlap fixture widens every window by 12.5 Th per side


# --------------------------------------------------------------------------- apex scan

APEX_FIXTURES = ["single", "grouped", "grouped_pool", "experiment", "mbr", "ovl_bp", "ovl_rg50"]


@pytest.mark.parametrize("name", APEX_FIXTURES)
def test_apex_scan_is_exact_for_every_scored_row(open_fixture, name):
    rs = open_fixture(name)
    checked = two_windows = 0
    for run in rs.runs:
        st = ScanTable.for_run(rs, run)
        oracle = _ms2_scalars(run.artifact("spectra_ms2").path)
        pmz, apex = _scored_pairs(rs, run)
        for m, t in zip(pmz, apex, strict=True):
            cover = (oracle["lower"] <= m) & (m <= oracle["upper"])
            exact = np.flatnonzero(cover & (oracle["rt"] == t))
            assert exact.size == 1, (name, run.label, m, t)
            pick = st.apex_scan(m, t)
            assert pick is not None and pick.exact
            assert pick.row == int(exact[0])
            assert pick.delta_rt == 0.0 and pick.rt == t
            assert pick.window_lower <= m <= pick.window_upper
            assert pick.scan_index == int(oracle["scan_index"][pick.row])
            two_windows += st.covering_windows(m).size == 2
            checked += 1
    assert checked > 0
    if name.startswith("ovl"):
        assert two_windows > 0  # the multi-window path was exercised


def test_apex_scan_topk_every_peak_rank(open_fixture):
    rs = open_fixture("topk")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    oracle = _ms2_scalars(run.artifact("spectra_ms2").path)
    e = pq.read_table(
        run.artifact("psms_extracted").path, columns=["peak_rank", "precursor_mz", "apex_rt"]
    )
    ranks = e["peak_rank"].to_numpy()
    assert len(ranks) == 360 and set(ranks.tolist()) == {0, 1}
    for m, t in zip(e["precursor_mz"].to_numpy(), e["apex_rt"].to_numpy(), strict=True):
        cover = (oracle["lower"] <= m) & (m <= oracle["upper"])
        exact = np.flatnonzero(cover & (oracle["rt"] == t))
        assert exact.size == 1
        pick = st.apex_scan(m, t)
        assert pick is not None and pick.exact and pick.row == int(exact[0])


@pytest.mark.parametrize("name", ["single", "topk", "ovl_bp"])
@pytest.mark.parametrize("shift", [-0.3, 0.3])
def test_shifted_apex_returns_the_nearest_scan(open_fixture, name, shift):
    rs = open_fixture(name)
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    oracle = _ms2_scalars(run.artifact("spectra_ms2").path)
    pmz, apex = _scored_pairs(rs, run)
    for m, t in zip(pmz, apex, strict=True):
        target = t + shift
        pick = st.apex_scan(m, target)
        expected = _nearest_oracle(oracle, m, target)
        assert pick is not None and not pick.exact
        assert pick.row == expected
        assert pick.delta_rt == oracle["rt"][expected] - target


def _two_window_table(tmp_path: Path) -> ScanTable:
    windows = {0: (400.0, 500.0), 1: (450.0, 550.0)}  # overlap 450-500
    scans = [(0, 10.0), (1, 11.0), (0, 12.0), (1, 12.0), (1, 13.0), (0, 14.0)]
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, windows, scans)
    return ScanTable(path)


def test_apex_scan_tie_rules(tmp_path):
    st = _two_window_table(tmp_path)
    # exact RT in both covering windows: the lower row, as the engine's first match
    pick = st.apex_scan(475.0, 12.0)
    assert (pick.row, pick.exact, pick.window_id) == (2, True, 0)
    # equal distance to 12 and 13: the lower RT, then the lower row
    pick = st.apex_scan(475.0, 12.5)
    assert (pick.row, pick.exact, pick.delta_rt) == (2, False, -0.5)
    # one covering window: 10 and 12 are 1 s away, the lower RT wins
    assert st.apex_scan(425.0, 11.0).row == 0
    assert st.apex_scan(425.0, 11.5).row == 2
    pick = st.apex_scan(525.0, 20.0)
    assert (pick.row, pick.exact, pick.delta_rt) == (4, False, -7.0)
    assert st.apex_scan(600.0, 12.0) is None  # no covering window
    assert st.apex_scan(475.0, float("nan")) is None


def test_step_order_within_a_window(tmp_path):
    st = _two_window_table(tmp_path)
    assert st.rows_in_window(0).tolist() == [0, 2, 5]
    assert st.rows_in_window(1).tolist() == [1, 3, 4]
    assert st.step(0, 1).row == 2
    assert st.step(2, 1).row == 5
    assert st.step(0, 2).row == 5
    assert st.step(5, 1) is None
    assert st.step(2, -1).row == 0
    assert st.step(0, -1) is None
    assert st.step(3, -1).row == 1
    same = st.step(3, 0)
    assert same.row == 3 and same.exact and same.delta_rt == 0.0
    ref = st.step(3, 1, reference_rt=12.0)
    assert (ref.row, ref.delta_rt, ref.exact) == (4, 1.0, False)


def test_step_order_on_the_smoke_run(open_fixture):
    st = ScanTable.for_run(open_fixture("single"), 0)
    for w in st.windows["window_id"]:
        rows = st.rows_in_window(int(w))
        assert rows.size == 60
        assert np.all(np.diff(st.rt[rows]) > 0)
        assert np.all(st.window_id[rows] == w)
        for k in range(rows.size):
            nxt = st.step(int(rows[k]), 1)
            prv = st.step(int(rows[k]), -1)
            assert (nxt.row if nxt else None) == (int(rows[k + 1]) if k + 1 < rows.size else None)
            assert (prv.row if prv else None) == (int(rows[k - 1]) if k > 0 else None)


def test_rt_order_when_the_file_is_not_rt_sorted(tmp_path):
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, {0: (400.0, 500.0)}, [(0, 14.0), (0, 10.0), (0, 12.0), (0, 12.0)])
    st = ScanTable(path)
    assert st.rows_in_window(0).tolist() == [1, 2, 3, 0]
    assert st.step(1, 1).row == 2 and st.step(2, 1).row == 3 and st.step(3, 1).row == 0
    assert st.apex_scan(450.0, 12.0).row == 2
    assert st.grid_rows(450.0, -np.inf, np.inf).tolist() == [1, 2, 0]


# --------------------------------------------------------------------------- XIC grid


def test_grid_rows_synthetic(tmp_path):
    st = _two_window_table(tmp_path)
    assert st.grid_rows(475.0, 11.0, 13.0).tolist() == [1, 2, 4]  # rt 12 kept once
    assert st.grid_rows(475.0, -np.inf, np.inf).tolist() == [0, 1, 2, 4, 5]
    assert st.grid_rows(425.0, 10.0, 12.0).tolist() == [0, 2]  # inclusive at both ends
    assert st.grid_rows(425.0, np.nextafter(10.0, np.inf), 12.0).tolist() == [2]
    assert st.grid_rows(475.0, float("nan"), 13.0).tolist() == []
    assert st.grid_rows(600.0, -np.inf, np.inf).tolist() == []


def test_grid_rows_reproduce_the_v1_axes(open_fixture, fixture_dir):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    rw = pq.read_table(run.artifact("run_windows").path, columns=["candidate_id", "rt_lo", "rt_hi"])
    assert np.array_equal(rw["candidate_id"].to_numpy(), np.arange(rw.num_rows))
    rt_lo, rt_hi = rw["rt_lo"].to_numpy(), rw["rt_hi"].to_numpy()
    e = pq.read_table(run.artifact("psms_extracted").path, columns=["candidate_id", "precursor_mz"])
    pmz = dict(zip(e["candidate_id"].to_pylist(), e["precursor_mz"].to_pylist(), strict=True))
    v1 = pq.read_table(
        fixture_dir("chrom_v1") / "chromatograms.parquet", columns=["candidate_id", "rt"]
    )
    cids = v1["candidate_id"].to_pylist()
    axes = v1["rt"].to_pylist()
    observed = set()
    checked = 0
    for cid, axis in zip(cids, axes, strict=True):
        if not axis:  # a never-observed fragment has an empty axis
            continue
        rows = st.grid_rows(pmz[cid], rt_lo[cid], rt_hi[cid])
        assert np.array_equal(_bits32(st.rt[rows]), _bits32(axis)), cid
        observed.add(cid)
        checked += 1
    assert len(observed) == 284 and checked == 2519


def _band_offsets(root: Path, lib_mz: np.ndarray) -> dict[str, int]:
    plan = json.loads((root / "groups" / "plan.json").read_text())
    return {f"g{b['index']:02d}": int(np.sum(lib_mz < b["mz_lo"])) for b in plan["bands"]}


@pytest.mark.parametrize("name", ["ovl_bp", "ovl_rg50"])
def test_grid_rows_merge_overlapping_windows(open_fixture, name):
    """Band axes of the overlap fixtures, where precursors sit in two windows."""
    rs = open_fixture(name)
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    lib_mz = _library_mz(rs)
    offsets = _band_offsets(run.root, lib_mz)
    checked = merged = 0
    for band in run.grouped.bands:
        rw = pq.read_table(band.artifact("run_windows").path, columns=["rt_lo", "rt_hi"])
        rt_lo, rt_hi = rw["rt_lo"].to_numpy(), rw["rt_hi"].to_numpy()
        pf = pq.ParquetFile(band.artifact("chromatograms").path)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["candidate_id", "rt_axis"])
            for cid, axis in zip(
                t["candidate_id"].to_pylist(), t["rt_axis"].to_pylist(), strict=True
            ):
                if not axis:  # the v2 axis is written once per candidate and row group
                    continue
                local = cid - offsets[band.name]  # band run_windows ids are band-local
                rows = st.grid_rows(lib_mz[cid], rt_lo[local], rt_hi[local])
                assert np.array_equal(_bits32(st.rt[rows]), _bits32(axis)), (band.name, cid)
                merged += st.covering_windows(lib_mz[cid]).size == 2
                checked += 1
    assert checked > 300 and merged > 40


def test_grid_label_follows_emit_window_grid(tmp_path, open_fixture, fixture_dir):
    """grid_rows is the chromatogram axis only in window-grid mode; the table says which."""
    rs = open_fixture("single")
    assert rs.config_get("extract", "emit_window_grid") is True
    st = ScanTable.for_run(rs, 0)
    assert st.emit_window_grid is True and st.grid_label == GRID_AXIS_LABEL

    def sparse(m, cfg):
        cfg["extract"]["emit_window_grid"] = False

    st = ScanTable.for_run(open_results(_edited_copy(tmp_path, fixture_dir("single"), sparse)), 0)
    assert st.emit_window_grid is False and st.grid_label == SPARSE_GRID_LABEL
    assert "not the chromatogram axis" in st.grid_label

    def unrecorded(m, cfg):
        del cfg["extract"]["emit_window_grid"]

    other = tmp_path / "unrecorded"
    other.mkdir()
    rs = open_results(_edited_copy(other, fixture_dir("single"), unrecorded))
    assert ScanTable.for_run(rs, 0).grid_label == GRID_UNKNOWN_LABEL
    direct = ScanTable(rs.runs[0].artifact("spectra_ms2").path)  # no result set: not known
    assert direct.emit_window_grid is None and direct.grid_label == GRID_UNKNOWN_LABEL


def test_grid_rows_equal_an_independent_oracle(open_fixture):
    rs = open_fixture("ovl_bp")
    st = ScanTable.for_run(rs, 0)
    oracle = _ms2_scalars(rs.runs[0].artifact("spectra_ms2").path)
    rng = np.random.default_rng(7)
    for pmz in rng.uniform(240.0, 2620.0, 300):
        lo = float(rng.uniform(-10.0, 120.0))
        hi = lo + float(rng.uniform(0.0, 40.0))
        assert np.array_equal(st.grid_rows(pmz, lo, hi), _grid_oracle(oracle, pmz, lo, hi))


# --------------------------------------------------------------------------- peaks


def test_ms2_spectrum_equals_pyarrow(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    t = pq.read_table(run.artifact("spectra_ms2").path)
    mzs, ints, ids = t["mz"].to_pylist(), t["intensity"].to_pylist(), t["id"].to_pylist()
    for row in range(st.n):
        sp = st.spectrum(row)
        assert sp.level == 2 and sp.row == row
        assert sp.mz.dtype == np.float32 and sp.intensity.dtype == np.float32
        assert np.array_equal(_bits32(sp.mz), _bits32(mzs[row]))
        assert np.array_equal(_bits32(sp.intensity), _bits32(ints[row]))
        assert sp.native_id == ids[row]
        assert sp.native_id == f"scan={sp.scan_index + 1}"
        assert sp.scan_index == int(st.scan_index[row]) and sp.rt == float(st.rt[row])
        assert sp.window_id == int(st.window_id[row])
        assert (sp.window_lower, sp.window_upper) == (float(st.lower[row]), float(st.upper[row]))
        assert sp.precursor_mz == float(st.precursor_mz[row])
        assert sp.precursor_charge is None


@pytest.mark.parametrize("list_type", [LARGE, SMALL], ids=["large_list", "list"])
def test_spectrum_in_every_row_group(tmp_path, list_type):
    rng = np.random.default_rng(3)
    n = 40
    peaks: list[tuple[list[float] | None, list[float] | None]] = []
    for _ in range(n):
        k = int(rng.integers(0, 30))
        mz = np.sort(rng.uniform(100, 2000, k)).astype(np.float32).tolist()
        inten = rng.uniform(1, 1e6, k).astype(np.float32).tolist()
        peaks.append((mz, inten))
    peaks[5] = ([], [])  # an empty scan
    peaks[11] = (None, None)  # a null list reads as empty
    windows = {0: (400.0, 500.0), 1: (500.0, 600.0)}
    scans = [(i % 2, 0.5 * i) for i in range(n)]
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, windows, scans, peaks=peaks, list_type=list_type, row_group_size=7)
    st = ScanTable(path)
    assert pq.ParquetFile(path).metadata.num_row_groups == 6
    for row in range(n):
        sp = st.spectrum(row)
        want_mz, want_int = peaks[row]
        assert np.array_equal(_bits32(sp.mz), _bits32(want_mz or []))
        assert np.array_equal(_bits32(sp.intensity), _bits32(want_int or []))
        assert sp.n_peaks == len(want_mz or [])
        assert sp.native_id == f"scan={2 * row + 2}"
    with pytest.raises(IndexError):
        st.spectrum(n)
    SPECTRUM_CACHE.clear()
    ROW_GROUP_CACHE.clear()
    uncached = st.spectrum(20, cached=False)
    assert len(SPECTRUM_CACHE) == 0 and SPECTRUM_CACHE.nbytes == 0
    assert np.array_equal(uncached.mz, st.spectrum(20).mz)
    assert len(SPECTRUM_CACHE) == 1 and SPECTRUM_CACHE.nbytes > 0
    assert ROW_GROUP_CACHE.nbytes == 0  # spectra never enter the process-wide cache


def _counting_reads(monkeypatch, st: ScanTable) -> list[int]:
    """Record the row group of every file read the scan table makes."""
    reads: list[int] = []
    original = st._handle.read_row_group

    def read(index, columns=None, *, cached=False):
        assert not cached  # never through the process-wide cache
        reads.append(index)
        return original(index, columns, cached=cached)

    monkeypatch.setattr(st._handle, "read_row_group", read)
    return reads


def test_spectrum_cache_is_bounded_and_releases_memory(tmp_path, monkeypatch):
    """The spectra cache keeps at most max_groups row groups and max_bytes (issue: an
    Astral session filled the 512 MB process-wide cache with 10-21 MB spectrum groups)."""
    n, size = 42, 7
    rng = np.random.default_rng(9)
    peaks = []
    for _ in range(n):
        mz = np.sort(rng.uniform(100, 2000, 200)).astype(np.float32).tolist()
        peaks.append((mz, rng.uniform(1, 1e6, 200).astype(np.float32).tolist()))
    path = tmp_path / "ms2.parquet"
    _write_ms2(
        path,
        {0: (400.0, 500.0)},
        [(0, float(i)) for i in range(n)],
        peaks=peaks,
        row_group_size=size,
    )
    st = ScanTable(path)
    releases: list[int] = []
    monkeypatch.setattr(spectra_module, "_release_arrow_memory", lambda: releases.append(1))
    reads = _counting_reads(monkeypatch, st)
    ROW_GROUP_CACHE.clear()
    SPECTRUM_CACHE.clear()
    monkeypatch.setattr(SPECTRUM_CACHE, "max_groups", 2)
    monkeypatch.setattr(SPECTRUM_CACHE, "release_bytes", 10**12)

    # LRU of two groups: groups 0, 1, 2 read; group 0 evicted; 1 and 2 are hits
    for rg in (0, 1, 2):
        st.spectrum(rg * size)
    assert reads == [0, 1, 2] and len(SPECTRUM_CACHE) == 2
    group = SPECTRUM_CACHE.nbytes // 2
    st.spectrum(1 * size + 3)
    st.spectrum(2 * size + 6)
    assert reads == [0, 1, 2]
    st.spectrum(0)  # evicts group 1, the least recently used
    st.spectrum(2 * size)
    assert reads == [0, 1, 2, 0] and len(SPECTRUM_CACHE) == 2
    st.spectrum(1 * size)
    assert reads == [0, 1, 2, 0, 1]
    assert ROW_GROUP_CACHE.nbytes == 0

    # byte budget: two groups fit, a third evicts the oldest
    SPECTRUM_CACHE.clear()
    monkeypatch.setattr(SPECTRUM_CACHE, "max_groups", 4)
    monkeypatch.setattr(SPECTRUM_CACHE, "max_bytes", int(group * 2.5))
    for rg in range(4):
        st.spectrum(rg * size)
    assert len(SPECTRUM_CACHE) == 2 and SPECTRUM_CACHE.nbytes <= SPECTRUM_CACHE.max_bytes

    # a group larger than half the budget is read but not kept
    SPECTRUM_CACHE.clear()
    monkeypatch.setattr(SPECTRUM_CACHE, "max_bytes", int(group * 1.5))
    st.spectrum(0)
    st.spectrum(1)
    assert len(SPECTRUM_CACHE) == 0 and reads[-2:] == [0, 0]

    # dropped peak data (an eviction, an uncached read) returns the pool's memory
    monkeypatch.setattr(SPECTRUM_CACHE, "max_bytes", 96 * 2**20)
    monkeypatch.setattr(SPECTRUM_CACHE, "max_groups", 1)
    SPECTRUM_CACHE.clear()
    st.spectrum(2 * size)  # kept; nothing dropped
    g2 = SPECTRUM_CACHE.nbytes
    monkeypatch.setattr(SPECTRUM_CACHE, "release_bytes", g2)
    releases.clear()
    st.spectrum(3 * size)  # kept; evicts group 2
    assert releases == [1] and len(SPECTRUM_CACHE) == 1
    SPECTRUM_CACHE.clear()
    releases.clear()
    monkeypatch.setattr(SPECTRUM_CACHE, "release_bytes", g2 + 1)
    st.spectrum(2 * size, cached=False)
    assert releases == [] and len(SPECTRUM_CACHE) == 0  # g2 dropped: below release_bytes
    st.spectrum(2 * size, cached=False)
    assert releases == [1]  # 2 * g2 dropped since the last release
    SPECTRUM_CACHE.clear()


def test_spectrum_cache_defaults():
    """The spectrum budget of T9 R10 and R15: 2-4 row groups, about 80-100 MB."""
    cache = spectra_module.SpectrumCache()
    assert (cache.max_groups, cache.max_bytes) == (4, 96 * 2**20)
    assert cache.keeps(47 * 2**20) and not cache.keeps(49 * 2**20)  # Astral MS1 group: 46.7 MB


def test_spectrum_refuses_mismatched_or_non_float32_lists(tmp_path):
    windows = {0: (400.0, 500.0)}
    path = tmp_path / "ms2.parquet"
    _write_ms2(path, windows, [(0, 1.0)], peaks=[([100.0, 200.0], [1.0])])
    with pytest.raises(InconsistentData, match="intensities"):
        ScanTable(path).spectrum(0)
    wide = tmp_path / "wide.parquet"
    _write_ms2(wide, windows, [(0, 1.0)], list_type=pa.large_list(pa.float64()))
    with pytest.raises(LayoutError, match="float32"):
        ScanTable(wide).spectrum(0)


def test_ms1_spectrum_equals_pyarrow(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    m1 = Ms1Table.for_run(rs, run)
    assert Ms1Table.for_run(rs, run) is m1
    t = pq.read_table(run.artifact("spectra_ms1").path)
    assert np.array_equal(m1.scan_index, t["scan_index"].to_numpy())
    assert np.array_equal(m1.rt, t["rt_seconds"].to_numpy())
    for row in range(m1.n):
        sp = m1.spectrum(row)
        assert sp.level == 1 and sp.window_id is None and sp.precursor_mz is None
        assert np.array_equal(_bits32(sp.mz), _bits32(t["mz"][row].as_py()))
        assert np.array_equal(_bits32(sp.intensity), _bits32(t["intensity"][row].as_py()))


# --------------------------------------------------------------------------- MS1 relations


def _engine_nearest_index(rts: np.ndarray, t: float) -> int:
    """A literal port of extract.rs ``nearest_index`` (partition_point on sorted RTs)."""
    p = 0
    while p < rts.size and rts[p] < t:
        p += 1
    if p == 0:
        return 0
    if p >= rts.size:
        return rts.size - 1
    return p - 1 if abs(t - rts[p - 1]) <= abs(rts[p] - t) else p


def test_ms1_nearest_tie_rule(tmp_path):
    path = tmp_path / "ms1.parquet"
    rts = [0.0, 2.0, 4.0, 6.0]
    _write_ms1(path, [0, 3, 6, 9], rts)
    m1 = Ms1Table(path)
    assert m1.nearest(1.0) == 0  # tie: the earlier scan
    assert m1.nearest(3.0) == 1
    assert m1.nearest(5.0) == 2
    assert m1.nearest(2.0) == 1
    assert m1.nearest(2.0000001) == 1
    assert m1.nearest(-5.0) == 0 and m1.nearest(99.0) == 3
    assert m1.nearest(float("nan")) is None
    rng = np.random.default_rng(1)
    probe = np.concatenate([rng.uniform(-1, 7, 500), np.arange(-1.0, 7.5, 0.5)])
    arr = np.asarray(rts)
    want = [_engine_nearest_index(arr, float(t)) for t in probe]
    assert m1.nearest_rows(probe).tolist() == want
    assert [m1.nearest(float(t)) for t in probe] == want


def _sum_near_oracle(mz: np.ndarray, inten: np.ndarray, target: float, tol: float) -> float:
    """extract.rs ``sum_near``: inclusive ppm_bounds on widened m/z, float32 accumulator."""
    d = target * tol * 1e-6
    acc = np.float32(0.0)
    for m, v in zip(mz.tolist(), inten.tolist(), strict=True):
        if target - d <= m <= target + d:
            acc = np.float32(acc + np.float32(v))
    return float(acc)


def _ms1_columns(run) -> list:
    cols = ["apex_rt", "precursor_mz", "charge", "ms1_isom1", "ms1_mono", "ms1_iso1", "ms1_iso2"]
    return list(
        pq.read_table(run.artifact("psms_extracted").path, columns=cols).to_pandas().itertuples()
    )


def test_isotope_mz_is_the_engine_expression():
    for pmz, z in ((512.2718, 2), (777.77, 3), (1500.5, 1)):
        sp = C13 / z
        assert [isotope_mz(pmz, z, k) for k in (-1, 0, 1, 2)] == [
            pmz - sp,
            pmz,
            pmz + sp,
            pmz + 2.0 * sp,
        ]


def test_sum_near_accumulates_in_float32():
    mz = np.array([499.99, 500.0, 500.005, 500.02], np.float32)
    inten = np.array([1.0, 16777216.0, 1.0, 5.0], np.float32)  # 2**24 + 1 is not a float32
    assert sum_near(mz, inten, 500.0, 20.0) == 16777216.0  # float64 would give 16777217
    assert sum_near(mz, inten, 500.0, 20.0) == _sum_near_oracle(mz, inten, 500.0, 20.0)
    assert sum_near(mz, inten, 700.0, 20.0) == 0.0
    assert sum_near(np.array([], np.float32), np.array([], np.float32), 500.0, 20.0) == 0.0


@pytest.mark.parametrize("name", ["single", "topk", "experiment"])
def test_nearest_ms1_reproduces_the_engine_ms1_columns(open_fixture, name):
    """psms_extracted ms1_isom1/mono/iso1/iso2 are sum_near in the MS1 scan nearest apex_rt."""
    rs = open_fixture(name)
    tol = rs.config_get("extract", "prec_tol_ppm")
    assert tol == 20.0
    nonzero = 0
    for run in rs.runs:
        m1 = Ms1Table.for_run(rs, run)
        for r in _ms1_columns(run):
            sp = m1.spectrum(m1.nearest(r.apex_rt))
            want = [r.ms1_isom1, r.ms1_mono, r.ms1_iso1, r.ms1_iso2]
            targets = [isotope_mz(r.precursor_mz, r.charge, k) for k in (-1, 0, 1, 2)]
            assert [sum_near(sp.mz, sp.intensity, t, tol) for t in targets] == want
            mz64 = sp.mz.astype(np.float64)
            assert [_sum_near_oracle(mz64, sp.intensity, t, tol) for t in targets] == want
            nonzero += r.ms1_mono > 0
    assert nonzero > 100


def test_preceding_ms1_equals_ms2_to_ms1(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    m1 = Ms1Table.for_run(rs, run)
    link = pq.read_table(run.artifact("ms2_to_ms1").path)
    parent = link["ms1_scan_index"].to_numpy()
    assert np.array_equal(m1.ms2_scan_index, st.scan_index)  # row-aligned with spectra_ms2
    assert np.array_equal(m1.ms1_scan_index, parent)
    differ = 0
    for row in range(st.n):
        prec = m1.preceding(row)
        assert prec is not None and int(m1.scan_index[prec]) == int(parent[row])
        assert m1.scan_index[prec] < st.scan_index[row]
        differ += prec != m1.nearest(float(st.rt[row]))
    assert 0 < differ < st.n  # the two MS1 relations are different things


def test_preceding_ms1_edge_cases(tmp_path):
    ms1 = tmp_path / "ms1.parquet"
    _write_ms1(ms1, [0, 10], [0.0, 2.0])
    link = tmp_path / "link.parquet"
    _write_link(link, [1, 2, 11, 12], [-1, 0, 10, 99])
    m1 = Ms1Table(ms1, link)
    assert m1.preceding(0) is None  # -1: no MS1 before the first MS2 scan
    assert m1.preceding(1) == 0 and m1.preceding(2) == 1
    with pytest.raises(InconsistentData, match="99"):
        m1.preceding(3)
    with pytest.raises(IndexError):
        m1.preceding(4)
    with pytest.raises(InconsistentData, match="row-aligned"):
        Ms1Table(ms1, link, n_ms2=5).preceding(0)
    with pytest.raises(ArtifactNotFound):
        Ms1Table(ms1).preceding(0)


def test_an_unreadable_ms2_to_ms1_refuses_only_the_preceding_ms1(tmp_path, fixture_dir):
    """nearest() and spectrum() do not use ms2_to_ms1, so its refusal must not block them."""

    def v99(m, cfg):
        m["artifacts"]["ms2_to_ms1"]["schema_version"] = 99

    root = _edited_copy(tmp_path, fixture_dir("single"), v99)
    rs = open_results(root)
    assert "unsupported_version" in rs.notice_codes()
    m1 = Ms1Table.for_run(rs, 0)
    row = m1.nearest(100.0)
    assert row is not None and m1.spectrum(row).n_peaks > 0
    for call in (lambda: m1.preceding(0), lambda: m1.ms1_scan_index, lambda: m1.ms2_scan_index):
        with pytest.raises(SchemaVersionError, match="ms2_to_ms1 schema version 99"):
            call()
    # a missing link table: the same, with ArtifactNotFound naming the file
    (root / "spectra" / "ms2_to_ms1.parquet").unlink()
    m1 = Ms1Table.for_run(open_results(root), 0)
    assert m1.nearest(100.0) == row
    with pytest.raises(ArtifactNotFound, match="ms2_to_ms1"):
        m1.preceding(0)


def test_ms1_labels_name_their_provenance():
    # sum_near is recomputed by the viewer; the engine's values are stored columns
    assert MS1_SUM_LABEL.startswith("viewer-recomputed (engine formula")
    assert "extract.prec_tol_ppm" in MS1_SUM_LABEL
    assert "psms_extracted ms1_*" in NEAREST_MS1_LABEL and "window-grid mode" in NEAREST_MS1_LABEL


# --------------------------------------------------------------------------- real data


@pytest.mark.real_data
def test_astral_single_run(real_single, capsys):
    rs = open_results(real_single)
    run = rs.runs[0]
    ROW_GROUP_CACHE.clear()
    t0 = time.perf_counter()
    st = ScanTable(run.artifact("spectra_ms2").parquet(), run.artifact("isolation_windows").path)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    ScanTable(run.artifact("spectra_ms2").parquet(), run.artifact("isolation_windows").path)
    t_load2 = time.perf_counter() - t0
    scheme = st.scheme()
    assert len(scheme) == 300
    assert np.all(scheme["width"].to_numpy() == 2.0)
    assert np.all(scheme["overlap_with_next"].to_numpy()[:-1] < 0)  # gaps, no overlap

    scored = pq.read_table(
        rs.scored.path, columns=["candidate_id", "apex_rt", "selected_peak_rank"]
    )
    rng = np.random.default_rng(0)
    pick_rows = rng.choice(scored.num_rows, 1000, replace=False)
    e = pq.read_table(
        run.artifact("psms_extracted").path, columns=["candidate_id", "peak_rank", "precursor_mz"]
    )
    key = {
        (c, r): m
        for c, r, m in zip(
            e["candidate_id"].to_pylist(),
            e["peak_rank"].to_pylist(),
            e["precursor_mz"].to_pylist(),
            strict=True,
        )
    }
    oracle = _ms2_scalars(run.artifact("spectra_ms2").path)
    cid = scored["candidate_id"].to_numpy()
    apex = scored["apex_rt"].to_numpy()
    rank = scored["selected_peak_rank"].to_numpy()
    rows = []
    t0 = time.perf_counter()
    for i in pick_rows:
        m = key[(int(cid[i]), int(rank[i]))]
        pick = st.apex_scan(m, float(apex[i]))
        assert pick is not None and pick.exact
        assert st.covering_windows(m).size == 1
        rows.append(pick.row)
    t_apex = (time.perf_counter() - t0) / len(pick_rows)
    for i, row in zip(pick_rows, rows, strict=True):
        m = key[(int(cid[i]), int(rank[i]))]
        cover = (oracle["lower"] <= m) & (m <= oracle["upper"])
        assert np.flatnonzero(cover & (oracle["rt"] == apex[i])).tolist() == [row]

    # the nearest MS1 scan reproduces the engine's MS1 columns (float32 sums)
    m1 = Ms1Table.for_run(rs, run)
    tol = rs.config_get("extract", "prec_tol_ppm")
    ms1 = pq.read_table(
        run.artifact("psms_extracted").path,
        columns=[
            "apex_rt",
            "precursor_mz",
            "charge",
            "ms1_isom1",
            "ms1_mono",
            "ms1_iso1",
            "ms1_iso2",
        ],
    )
    ms1 = ms1.take(pa.array(rng.choice(ms1.num_rows, 300, replace=False))).to_pandas()
    for r in ms1.itertuples():
        sp = m1.spectrum(m1.nearest(r.apex_rt))
        got = [
            sum_near(sp.mz, sp.intensity, isotope_mz(r.precursor_mz, r.charge, k), tol)
            for k in (-1, 0, 1, 2)
        ]
        assert got == [r.ms1_isom1, r.ms1_mono, r.ms1_iso1, r.ms1_iso2]

    SPECTRUM_CACHE.clear()
    sample = rows[:20]
    cold, warm = [], []
    for row in sample:
        t0 = time.perf_counter()
        st.spectrum(row)
        cold.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        st.spectrum(row)
        warm.append(time.perf_counter() - t0)
    with capsys.disabled():
        print(
            f"\n[astral] MS2 scalar load {t_load * 1e3:.0f} ms first, {t_load2 * 1e3:.0f} ms "
            f"repeat ({st.n} scans); apex_scan {t_apex * 1e6:.0f} us per call; spectrum read "
            f"first median {np.median(cold) * 1e3:.1f} ms (max {max(cold) * 1e3:.1f}), cached "
            f"median {np.median(warm) * 1e3:.2f} ms"
        )


def _process_memory_mb() -> tuple[float, float] | None:
    """(private, working set) of this process in MB on Windows; None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    c = Counters()
    c.cb = ctypes.sizeof(Counters)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return None
    return c.PrivateUsage / 2**20, c.WorkingSetSize / 2**20


@pytest.mark.real_data
def test_astral_spectrum_reads_stay_in_the_budget(real_single, capsys):
    """Spectra from 48 different row groups keep at most the spectrum budget (T9 R10, R15).

    Before the spectra had their own cache, the same reads left 457 MB of row groups in
    the process-wide cache and raised private memory by about 0.9 GB.
    """
    rs = open_results(real_single)
    st = ScanTable.for_run(rs, 0)
    offsets = st._handle.row_group_offsets()
    SPECTRUM_CACHE.clear()
    ROW_GROUP_CACHE.clear()
    before = _process_memory_mb()
    t0 = time.perf_counter()
    groups = range(0, offsets.size - 1, 3)
    for rg in groups:
        st.spectrum(int(offsets[rg]) + 5)
        assert len(SPECTRUM_CACHE) <= SPECTRUM_CACHE.max_groups
        assert SPECTRUM_CACHE.nbytes <= SPECTRUM_CACHE.max_bytes
    per_read = (time.perf_counter() - t0) / len(groups)
    assert ROW_GROUP_CACHE.nbytes == 0
    after = _process_memory_mb()
    memory = ""
    if before is not None and after is not None:
        memory = (
            f"; private {before[0]:.0f} -> {after[0]:.0f} MB, "
            f"working set {before[1]:.0f} -> {after[1]:.0f} MB"
        )
    with capsys.disabled():
        print(
            f"\n[astral] {len(groups)} spectra from {len(groups)} row groups: "
            f"{per_read * 1e3:.1f} ms per read; spectrum cache "
            f"{SPECTRUM_CACHE.nbytes / 2**20:.1f} MB in {len(SPECTRUM_CACHE)} groups{memory}"
        )
    SPECTRUM_CACHE.clear()
