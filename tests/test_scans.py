"""The spectrum browser's data (data.scans): addressing and stepping scans, the accepted
identifications near a scan, and the fragment and isotope overlays.

Every reference is computed independently here, with pyarrow, numpy and plain Python
(no DuckDB, no code of the module under test). The overlays are checked against the
precursor page's mirror (data.detail) and against the engine's own MS1 columns.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import scans
from mumdia_viewer.data.detail import mirror, precursor_detail
from mumdia_viewer.data.spectra import ScanTable

SPECTRA_FIXTURES = ["single", "experiment", "mbr", "topk", "grouped"]
SCORED_FIXTURES = ["single", "experiment", "mbr", "topk", "grouped"]


def _ms2(run):
    t = pq.read_table(
        run.artifact("spectra_ms2").path,
        columns=["scan_index", "rt_seconds", "window_id", "window_lower", "window_upper"],
    )
    return {c: t.column(c).to_numpy() for c in t.column_names}


def _ms1(run):
    t = pq.read_table(run.artifact("spectra_ms1").path, columns=["scan_index", "rt_seconds"])
    return {c: t.column(c).to_numpy() for c in t.column_names}


def _nearest(rts, rows, t):
    """The row nearest t among rows; ties to the lower RT, then the lower row."""
    best = min(rows, key=lambda r: (abs(float(rts[r]) - t), float(rts[r]), r))
    return best


# --------------------------------------------------------------------------- addressing


@pytest.mark.parametrize("name", SPECTRA_FIXTURES)
def test_locate_finds_every_scan_index(open_fixture, name):
    rs = open_fixture(name)
    for run in rs.runs:
        ms2, ms1 = _ms2(run), _ms1(run)
        for row in (0, len(ms2["scan_index"]) // 2, len(ms2["scan_index"]) - 1):
            ref = scans.locate(rs, run, int(ms2["scan_index"][row]))
            assert (ref.level, ref.row) == (2, row)
            assert ref.rt == float(ms2["rt_seconds"][row])
            assert ref.window_id == int(ms2["window_id"][row])
            assert ref.lower == float(ms2["window_lower"][row])
        for row in (0, len(ms1["scan_index"]) - 1):
            ref = scans.locate(rs, run, int(ms1["scan_index"][row]))
            assert (ref.level, ref.row) == (1, row) and ref.window_id is None
        used = set(ms2["scan_index"].tolist()) | set(ms1["scan_index"].tolist())
        missing = max(used) + 1
        assert scans.locate(rs, run, missing) is None


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_nearest_scan_equals_brute_force(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[-1]
    ms2, ms1 = _ms2(run), _ms1(run)
    windows = sorted(set(ms2["window_id"].tolist()))
    rng = np.random.default_rng(7)
    lo, hi = float(ms2["rt_seconds"].min()), float(ms2["rt_seconds"].max())
    targets = [*rng.uniform(lo - 5, hi + 5, 25).tolist(), float(ms2["rt_seconds"][11])]
    # A target halfway between two scans of one window: the tie goes to the earlier one.
    w0 = [r for r in range(len(ms2["window_id"])) if ms2["window_id"][r] == windows[0]]
    w0.sort(key=lambda r: (ms2["rt_seconds"][r], r))
    targets.append((float(ms2["rt_seconds"][w0[3]]) + float(ms2["rt_seconds"][w0[4]])) / 2)
    for t in targets:
        for w in (windows[0], windows[-1]):
            rows = [r for r in range(len(ms2["window_id"])) if ms2["window_id"][r] == w]
            ref = scans.nearest_scan(rs, run, t, level=2, window_id=w)
            assert ref.row == _nearest(ms2["rt_seconds"], rows, t)
        ref = scans.nearest_scan(rs, run, t, level=2)
        assert ref.row == _nearest(ms2["rt_seconds"], range(len(ms2["rt_seconds"])), t)
        ref = scans.nearest_scan(rs, run, t, level=1)
        assert ref.row == _nearest(ms1["rt_seconds"], range(len(ms1["rt_seconds"])), t)
    assert scans.nearest_scan(rs, run, float("nan")) is None
    with pytest.raises(KeyError):
        scans.nearest_scan(rs, run, 10.0, level=2, window_id=10**6)


@pytest.mark.parametrize("name", ["single", "mbr"])
def test_steps_in_window_and_in_run(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    ms2, ms1 = _ms2(run), _ms1(run)
    # Acquisition order: every scan_index of both tables, ascending.
    order = sorted(
        [(int(s), 2, r) for r, s in enumerate(ms2["scan_index"])]
        + [(int(s), 1, r) for r, s in enumerate(ms1["scan_index"])]
    )
    ref = scans.scan_ref(rs, run, order[0][1], order[0][2])
    seen = [ref.scan_index]
    while True:
        nxt = scans.step(rs, run, ref, 1, scope="run")
        if nxt is None:
            break
        seen.append(nxt.scan_index)
        ref = nxt
    assert seen == [o[0] for o in order]
    assert scans.step(rs, run, ref, -2, scope="run").scan_index == order[-3][0]
    # The same window, in (rt, row) order.
    w = int(ms2["window_id"][5])
    rows = sorted(
        (r for r in range(len(ms2["window_id"])) if ms2["window_id"][r] == w),
        key=lambda r: (ms2["rt_seconds"][r], r),
    )
    ref = scans.scan_ref(rs, run, 2, rows[0])
    walked = [ref.row]
    while (ref := scans.step(rs, run, ref, 1)) is not None:
        walked.append(ref.row)
    assert walked == rows
    assert scans.step(rs, run, scans.scan_ref(rs, run, 2, rows[2]), -2).row == rows[0]
    # MS1 scans in RT order.
    m_rows = sorted(range(len(ms1["rt_seconds"])), key=lambda r: (ms1["rt_seconds"][r], r))
    ref = scans.scan_ref(rs, run, 1, m_rows[1])
    assert scans.step(rs, run, ref, 1).row == m_rows[2]
    assert scans.step(rs, run, ref, -1).row == m_rows[0]
    assert scans.step(rs, run, ref, -2) is None
    with pytest.raises(ValueError):
        scans.step(rs, run, ref, 1, scope="level")  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        scans.scan_ref(rs, run, 2, len(ms2["rt_seconds"]))


def test_ms1_links_follow_the_link_table_and_the_nearest_rule(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    ms2, ms1 = _ms2(run), _ms1(run)
    link = pq.read_table(run.artifact("ms2_to_ms1").path).to_pydict()
    for row in (0, 17, len(ms2["rt_seconds"]) - 1):
        ref = scans.scan_ref(rs, run, 2, row)
        got = scans.ms1_links(rs, run, ref)
        parent = link["ms1_scan_index"][row]
        if parent < 0:
            assert got["preceding"] is None
        else:
            assert got["preceding"].scan_index == parent
        want = _nearest(ms1["rt_seconds"], range(len(ms1["rt_seconds"])), ref.rt)
        assert got["nearest"].row == want
    ms1_ref = scans.scan_ref(rs, run, 1, 0)
    assert scans.ms1_links(rs, run, ms1_ref)["nearest"] is None


def test_top_peaks_picks_the_most_intense_apart():
    mz = np.array([100.0, 100.5, 200.0, 300.0, 300.2, 400.0])
    it = np.array([5.0, 9.0, 1.0, 7.0, 8.0, 7.0])
    got = scans.top_peaks(mz, it, 3, min_gap=1.0).tolist()
    assert got == [1, 4, 5]  # 300.0 is 0.2 from 300.2, so 400.0 is the third
    assert scans.top_peaks(mz, it, 10, lo=150, hi=350, min_gap=0.0).tolist() == [4, 3, 2]
    # Equal intensities: the lower index first.
    assert scans.top_peaks(mz, it, 2, lo=250, min_gap=0.0).tolist() == [4, 3]
    assert scans.top_peaks(mz[:0], it[:0], 3).size == 0


# --------------------------------------------------------------------------- candidates


def _reference_frame(rs, run, t):
    """Accepted rows of a run with their precursor m/z, joined in plain Python."""
    s = pq.read_table(
        rs.scored.path,
        columns=[
            "candidate_id",
            "label",
            "apex_rt",
            "elution_lo",
            "elution_hi",
            "run_psm_q",
            "selected_peak_rank",
            "source",
            "score",
        ],
    ).to_pylist()
    ex = run.artifact("psms_extracted")
    if ex is not None and ex.usable:
        e = pq.read_table(ex.path, columns=["candidate_id", "peak_rank", "precursor_mz"])
        pmz = {
            (c, k): m
            for c, k, m in zip(*(e.column(n).to_pylist() for n in e.column_names), strict=True)
        }
        get = lambda r: pmz.get((r["candidate_id"], r["selected_peak_rank"] or 0))  # noqa: E731
    else:
        lib = run.artifacts.get("fragment_library_precursors") or rs.extra.get(
            "fragment_library_precursors"
        )
        e = pq.read_table(lib.path, columns=["candidate_id", "precursor_mz"]).to_pydict()
        pmz = dict(zip(e["candidate_id"], e["precursor_mz"], strict=True))
        get = lambda r: pmz.get(r["candidate_id"])  # noqa: E731
    out = []
    for r in s:
        if rs.is_experiment and r["source"] != run.index:
            continue
        if r["run_psm_q"] is None or r["run_psm_q"] > t:
            continue
        out.append({**r, "precursor_mz": get(r)})
    return out


@pytest.mark.parametrize("name", SCORED_FIXTURES)
@pytest.mark.parametrize("t", [0.01, 0.1])
def test_candidates_near_equal_brute_force(open_fixture, name, t):
    rs = open_fixture(name)
    for run in rs.runs:
        ref_rows = _reference_frame(rs, run, t)
        st = ScanTable.for_run(rs, run)
        # Scans at the apexes of accepted targets (so lists are not empty) and one at random.
        apexes = [r for r in ref_rows if r["label"] == "target" and r["precursor_mz"] is not None]
        picks = []
        for r in apexes[:: max(1, len(apexes) // 6)][:6]:
            w = st.covering_windows(r["precursor_mz"])
            if w.size:
                picks.append(scans.nearest_scan(rs, run, r["apex_rt"], window_id=int(w[0])))
        picks.append(scans.scan_ref(rs, run, 2, st.n // 3))
        picks.append(scans.scan_ref(rs, run, 1, 3))
        for ref in picks:
            for mode, delta, decoys in (
                ("apex", 5.0, False),
                ("apex", 1.0, True),
                ("elution", 0, False),
            ):
                got = scans.candidates_near(
                    rs, run, ref, t, delta=delta, mode=mode, include_decoys=decoys
                )
                want = []
                for r in ref_rows:
                    if r["label"] != "target" and not (decoys and r["label"] == "decoy"):
                        continue
                    if mode == "apex":
                        if r["apex_rt"] is None or abs(r["apex_rt"] - ref.rt) > delta:
                            continue
                    elif not (
                        r["elution_lo"] is not None and r["elution_lo"] <= ref.rt <= r["elution_hi"]
                    ):
                        continue
                    if ref.level == 2:
                        m = r["precursor_mz"]
                        if m is None or not ref.lower <= m <= ref.upper:
                            continue
                    want.append(r)
                want.sort(key=lambda r: (abs(r["apex_rt"] - ref.rt), r["candidate_id"]))
                assert got["candidate_id"].tolist() == [r["candidate_id"] for r in want]
                if len(got):
                    assert np.allclose(got["delta_rt"], [r["apex_rt"] - ref.rt for r in want])
                    assert (got["run_psm_q"] <= t).all()
                assert "run_psm_q" in got.attrs["label"]
                assert "isolation window" in got.attrs["rule"] or ref.level == 1


@pytest.mark.parametrize("name", ["single", "experiment", "grouped"])
def test_apex_histogram_and_default_scan(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    rows = [r for r in _reference_frame(rs, run, 0.01) if r["label"] == "target"]
    edges = np.linspace(0, 130, 14)
    apex = np.array([r["apex_rt"] for r in rows], dtype=float)
    assert (
        scans.apex_histogram(rs, run, 0.01, edges).tolist() == np.histogram(apex, edges)[0].tolist()
    )
    lo, hi = 500.0, 650.0
    inside = np.array(
        [
            r["apex_rt"]
            for r in rows
            if r["precursor_mz"] is not None and lo <= r["precursor_mz"] <= hi
        ]
    )
    got = scans.apex_histogram(rs, run, 0.01, edges, window=(lo, hi))
    assert got.tolist() == np.histogram(inside, edges)[0].tolist()
    ref, cid = scans.default_scan(rs, run, 0.01)
    best = max(rows, key=lambda r: (r["score"], -r["candidate_id"])) if rows else None
    if best is not None:
        assert cid == best["candidate_id"]
        assert ref.level == 2 and ref.rt == best["apex_rt"]
        assert ref.lower <= best["precursor_mz"] <= ref.upper


def test_candidates_use_the_library_when_psms_extracted_is_gone(open_fixture):
    rs = open_fixture("grouped")
    acc = scans.accepted_set(rs, rs.runs[0], 0.01)
    assert "library" in acc.mz_source
    assert acc.frame["precursor_mz"].notna().all()
    single = scans.accepted_set(open_fixture("single"), None, 0.01)
    assert "psms_extracted" in single.mz_source


def test_candidate_arguments_are_checked(open_fixture):
    rs = open_fixture("single")
    ref = scans.scan_ref(rs, None, 2, 0)
    with pytest.raises(ValueError):
        scans.candidates_near(rs, None, ref, 0.01, mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        scans.candidates_near(rs, None, ref, 0.01, delta=-1)
    with pytest.raises(ValueError):
        scans.accepted_set(rs, None, 0)


def test_population_label_names_the_column_and_transfers(open_fixture):
    assert "run_psm_q" in scans.population_label(open_fixture("single"), None, 0.01)
    exp = open_fixture("experiment")
    assert f"run {exp.runs[1].label}" in scans.population_label(exp, exp.runs[1].name, 0.01)
    mbr = open_fixture("mbr")
    assert "match-between-runs" in scans.population_label(mbr, mbr.runs[0].name, 0.01)


# --------------------------------------------------------------------------- overlays


@pytest.mark.parametrize("name", ["single", "experiment", "topk"])
def test_fragment_overlay_equals_the_precursor_mirror(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    rows = [r for r in _reference_frame(rs, run, 0.01) if r["label"] == "target"][:5]
    assert rows
    st = ScanTable.for_run(rs, run)
    compared = 0
    for r in rows:
        d = precursor_detail(rs, run, r["candidate_id"])
        m = mirror(rs, d)
        if m is None:
            continue
        spec = st.spectrum(m.pick.row)
        ov = scans.fragment_overlay(rs, run, r["candidate_id"], spec)
        assert ov.fragments["theo_mz"].tolist() == m.fragments["theo_mz"].tolist()
        assert [(x.fragment, x.peak, x.ppm_raw) for x in ov.matches] == [
            (x.fragment, x.peak, x.ppm_raw) for x in m.matches
        ]
        assert ov.label == m.tolerance.match_label
        counts = scans.matched_counts(rs, run, [r["candidate_id"]], spec)
        assert counts[r["candidate_id"]] == (len(m.matches), len(m.fragments))
        compared += 1
    assert compared >= 3


def test_isotope_overlay_reproduces_the_engine_ms1_columns(open_fixture):
    """In the MS1 scan nearest apex_rt the sum equals psms_extracted's ms1_* (bit for bit)."""
    rs = open_fixture("single")
    run = rs.runs[0]
    ex = pq.read_table(
        run.artifact("psms_extracted").path,
        columns=["candidate_id", "apex_rt", "precursor_mz", "charge", "ms1_mono", "ms1_iso1"],
    ).to_pylist()
    ms1 = _ms1(run)
    checked = 0
    for r in ex[:40]:
        if r["apex_rt"] is None or not np.isfinite(r["apex_rt"]):
            continue
        row = _nearest(ms1["rt_seconds"], range(len(ms1["rt_seconds"])), r["apex_rt"])
        ref = scans.scan_ref(rs, run, 1, row)
        from mumdia_viewer.data.spectra import Ms1Table

        spec = Ms1Table.for_run(rs, run).spectrum(ref.row)
        ov = scans.isotope_overlay(rs, r["candidate_id"], r["precursor_mz"], r["charge"], spec)
        by_k = {int(x["k"]): x for x in ov.rows}
        assert np.float32(by_k[0]["sum"]) == np.float32(r["ms1_mono"])
        assert np.float32(by_k[1]["sum"]) == np.float32(r["ms1_iso1"])
        checked += 1
    assert checked >= 10
    assert scans.isotope_overlay(rs, 1, 500.0, 0, spec) is None
