"""Mobilograms of diaPASEF candidates (``tims`` fixture).

The centroid source is checked stick by stick against the spectra tables; the raw source
through a recorded ``mumdia mobilogram`` answer for the fixture's candidates 11 and 12
(``fixtures/im/mobilogram_engine.json``, MuMDIA 0.9.0 on the fixture's tables and the full
``.d`` it was cut from) and a stand-in for the engine process. The test marked ``engine``
runs the real binary: set ``MUMDIA_BIN`` and ``MUMDIA_VIEWER_TIMS_RAW`` (the
``ttSCP_diaPASEF_Condition_A_Sample_Alpha_01_11494.d`` of the ProteoBench HYE diaPASEF set).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import ViewerError
from mumdia_viewer.data import mobilogram as mob
from mumdia_viewer.data.detail import precursor_detail
from mumdia_viewer.data.fragments import within_ppm
from mumdia_viewer.data.spectra import Ms1Table, ScanTable, Spectrum
from mumdia_viewer.ui import detail as detail_page
from mumdia_viewer.ui import detail_mobilogram as mobc
from mumdia_viewer.ui.detail_view import fragments_of
from mumdia_viewer.ui.state import PageContext
from test_ui_spectra import _walk

RECORDED = Path(__file__).parent / "fixtures" / "im" / "mobilogram_engine.json"
RECORDED_IDS = (11, 12)
# Read before the autouse fixture below clears them.
ENGINE = os.environ.get("MUMDIA_BIN")
TIMS_RAW = os.environ.get("MUMDIA_VIEWER_TIMS_RAW")


@pytest.fixture(autouse=True)
def _no_engine(monkeypatch):
    """No engine unless a test provides one."""
    monkeypatch.delenv(mob.ENGINE_ENV, raising=False)
    mob.set_engine(None)
    with mobc._LOCK:
        mobc._CACHE.clear()
    yield
    mob.set_engine(None)


@pytest.fixture(scope="module")
def tims(open_fixture):
    return open_fixture("tims")


@pytest.fixture(scope="module")
def recorded():
    return json.loads(RECORDED.read_text())


def _accepted(rs, n):
    s = pq.read_table(
        rs.runs[0].artifact("psms_scored").path, columns=["candidate_id", "label", "q_value"]
    ).to_pandas()
    s = s[(s.label == "target") & (s.q_value <= 0.01)].sort_values("candidate_id")
    return [int(c) for c in s.candidate_id.to_numpy()[:: max(1, len(s) // n)][:n]]


def _detail(rs, cid):
    return precursor_detail(rs, rs.runs[0].name, cid)


def _frame_spectra(scans, d, rt):
    windows = scans.covering_windows(d.precursor_mz)
    rows = np.concatenate([scans.rows_in_window(int(w)) for w in windows])
    return [scans.spectrum(int(r)) for r in rows[scans.rt[rows] == rt]]


def test_every_centroid_stick_is_a_spectra_peak_in_its_slot(tims):
    run = tims.runs[0]
    scans = ScanTable.for_run(tims, run)
    ms1 = Ms1Table.for_run(tims, run)
    tol_ms1 = mob.prec_tol_ppm(tims)
    checked = 0
    for cid in _accepted(tims, 8):
        d = _detail(tims, cid)
        m = mob.centroid_mobilogram(tims, d)
        assert m.source == "centroids" and m.label == mob.CENTROID_LABEL
        assert [f.offset for f in m.ms2_frames] == list(range(-2, 3))
        # The centre frame is the frame of the apex scan.
        centre = next(f for f in m.ms2_frames if f.offset == 0)
        assert centre.rt == d.apex_scan.rt
        names = [t.frag_name for t in d.chromatogram.fragments() if t.predicted_intensity > 0]
        assert [i.name for i in m.ions if i.kind == "fragment"] == names
        assert [i.name for i in m.ions if i.kind == "ms1"] == [
            t.frag_name for t in d.chromatogram.ms1()
        ]
        theo = {t.frag_name: t.frag_mz for t in d.chromatogram.traces}
        for k, frame in enumerate(m.ms2_frames):
            spectra = _frame_spectra(scans, d, frame.rt)
            assert spectra
            for ion in (i for i in m.ions if i.kind == "fragment"):
                im, inten = ion.frames[k]
                t32 = float(np.float32(theo[ion.name]))
                # Every matching peak of the frame is a stick, and every stick is one.
                want = []
                for s in spectra:
                    mz64 = s.mz.astype(np.float64)
                    q = mz64 / (1.0 + d.tolerance.ppm_at(mz64) * 1e-6)
                    hit = np.flatnonzero(within_ppm(np.full(q.shape, t32), q, d.tolerance.tol_ppm))
                    for j in hit:
                        assert s.window_im_lower <= s.im[j] <= s.window_im_upper
                        want.append((float(s.im[j]), float(s.intensity[j])))
                assert sorted(zip(im.tolist(), inten.tolist(), strict=True)) == sorted(want)
                checked += len(want)
        for k, frame in enumerate(m.ms1_frames):
            row = ms1.row_of_scan_index(int(frame.label.rsplit(" ", 1)[1]))
            s = ms1.spectrum(row)
            for ion in (i for i in m.ions if i.kind == "ms1"):
                im, inten = ion.frames[k]
                for x, y in zip(im, inten, strict=True):
                    j = np.flatnonzero((s.im == np.float32(x)) & (s.intensity == np.float32(y)))
                    assert j.size, (cid, ion.name, x, y)
                    mz = float(s.mz[j[0]])
                    assert abs(mz - ion.mz) <= ion.mz * tol_ms1 * 1e-6 * (1 + 1e-9)
                    assert m.slot_im[0] <= x <= m.slot_im[1]
    assert checked > 50


def test_overlapping_slots_of_one_frame_count_a_peak_once():
    def spec(row, lo, hi, im):
        im = np.asarray(im, dtype=np.float32)
        return Spectrum(
            level=2,
            row=row,
            scan_index=row,
            rt=1.0,
            mz=np.full(im.size, 500.0, dtype=np.float32),
            intensity=np.arange(1, im.size + 1, dtype=np.float32),
            im=im,
            window_im_lower=lo,
            window_im_upper=hi,
        )

    # Slot 1 covers 0.80-0.90, slot 0 0.85-0.95: their peaks at 0.86 and 0.88 are the
    # same events, kept from the slot with the lower bound.
    a = spec(0, 0.85, 0.95, [0.86, 0.88, 0.93])
    b = spec(1, 0.80, 0.90, [0.81, 0.86, 0.88])
    _, inten, im = mob.merge_slots([a, b])
    assert sorted(im.tolist()) == pytest.approx([0.81, 0.86, 0.88, 0.93])
    assert inten.tolist() == [1.0, 2.0, 3.0, 3.0]  # b's three peaks, then a's 0.93
    assert mob.merge_slots([])[0].size == 0


def test_three_d_runs_have_no_mobilogram(open_fixture):
    rs = open_fixture("single")
    s = pq.read_table(rs.runs[0].artifact("psms_scored").path, columns=["candidate_id", "label"])
    cid = int(s.to_pandas().query("label == 'target'").candidate_id.iloc[0])
    d = precursor_detail(rs, rs.runs[0].name, cid)
    with pytest.raises(ViewerError, match="no ion mobility"):
        mob.centroid_mobilogram(rs, d)
    assert (
        mobc.mobilogram_card(PageContext(rs=rs, base="/"), d, fragments_of(d.chromatogram)) is None
    )
    assert mob.engine_status(rs, rs.runs[0]) == (False, "not a diaPASEF run")


def test_the_recorded_engine_answer_parses(tims, recorded):
    ext = pq.read_table(
        tims.runs[0].artifact("psms_extracted").path, columns=["candidate_id", "apex_im"]
    ).to_pandas()
    for cid in RECORDED_IDS:
        m = mob.parse_engine_json(recorded, cid)
        raw = next(c for c in recorded["candidates"] if c["candidate_id"] == cid)
        assert m.source == "raw" and m.bin == pytest.approx(0.002)
        assert [f.offset for f in m.ms2_frames] == [-2, -1, 0, 1, 2]
        assert [f.label for f in m.ms2_frames] == [
            f"frame={f['frame_id']}" for f in raw["ms2_frames"]
        ]
        d = _detail(tims, cid)
        names = [t.frag_name for t in d.chromatogram.fragments() if t.predicted_intensity > 0]
        assert [i.name for i in m.ions if i.kind == "fragment"] == names
        # The raw source reads the frames the centroid source reads.
        c = mob.centroid_mobilogram(tims, d)
        assert [f.label for f in c.ms2_frames] == [f.label for f in m.ms2_frames]
        frag = [i for i in m.ions if i.kind == "fragment"]
        for ion, src in zip(frag, raw["ions"], strict=False):
            im, total = m.profile(ion, 2)
            assert np.allclose(total, src["intensity"])
            _, centre = m.profile(ion, 0)
            assert np.allclose(centre, src["per_frame"][2])
            assert im.size == len(raw["im"])
        # The summed fragment profile peaks at the extracted apex 1/K0.
        im = np.asarray(raw["im"])
        total = np.sum([m.profile(i, 2)[1] for i in frag], axis=0)
        apex = float(ext[ext.candidate_id == cid].apex_im.iloc[0])
        assert abs(im[int(np.argmax(total))] - apex) < 0.01
    with pytest.raises(ViewerError, match="could not profile"):
        mob.parse_engine_json(recorded, 99999)
    with pytest.raises(ViewerError, match="does not list"):
        mob.parse_engine_json(recorded, 5)
    with pytest.raises(ViewerError, match="format version"):
        mob.parse_engine_json({**recorded, "format_version": 2}, 11)
    with pytest.raises(ViewerError, match="not a mumdia-mobilogram"):
        mob.parse_engine_json({"format": "x"}, 11)


def test_the_engine_command_names_every_input(tims):
    d = _detail(tims, 11)
    cmd = mob.engine_command(
        tims,
        d,
        engine=Path("mumdia"),
        raw=Path("run.d"),
        centre_rt=2201.5,
        frames=2,
        bin_width=0.002,
    )
    run = tims.runs[0]
    args = {a: b for a, b in itertools.pairwise(cmd) if a.startswith("--")}
    assert cmd[:2] == ["mumdia", "mobilogram"] and "--json" in cmd
    assert args["--candidate"] == "11" and args["--peak-rank"] == "0"
    assert args["--raw"] == "run.d" and float(args["--rt"]) == 2201.5
    assert Path(args["--psms-extracted"]) == run.artifact("psms_extracted").path
    assert Path(args["--chromatograms"]) == d.chromatogram.table
    assert Path(args["--run-windows"]) == run.artifact("run_windows").path
    assert Path(args["--mass-cal"]).name == "seed_psms.parquet.masscal.json"
    assert args["--run-dir"] == str(run.root)


class _FakeEngine:
    """Stands in for ``subprocess.run``: answers ``--version`` and ``mobilogram`` with the
    recorded document and counts the calls."""

    def __init__(self, recorded, returncode=0, stderr=""):
        self.recorded, self.returncode, self.stderr = recorded, returncode, stderr
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "mumdia 0.9.0\n", "")
        if self.returncode:
            return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)
        return subprocess.CompletedProcess(cmd, 0, json.dumps(self.recorded), "")


@pytest.fixture
def fake_engine(tmp_path, monkeypatch, recorded):
    exe = tmp_path / "mumdia"
    exe.write_text("")
    raw = tmp_path / "run.d"
    raw.mkdir()
    (raw / "analysis.tdf").write_text("")
    mob.set_engine(exe)
    monkeypatch.setattr(mob, "raw_input", lambda rs, run: raw)
    fake = _FakeEngine(recorded)
    monkeypatch.setattr(mob.subprocess, "run", fake)
    return fake


def test_engine_answers_are_cached(tims, fake_engine):
    d = _detail(tims, 12)
    assert mob.engine_status(tims, d.run) == (True, "")
    m = mob.engine_mobilogram(tims, d, centre_rt=2205.0786)
    assert m.source == "raw" and m.candidate_id == 12
    runs = [c for c in fake_engine.calls if c[1] == "mobilogram"]
    assert len(runs) == 1
    again = mob.engine_mobilogram(tims, d, centre_rt=2205.0786)
    assert len([c for c in fake_engine.calls if c[1] == "mobilogram"]) == 1
    assert again.ion("y4").frames[2][1].tolist() == m.ion("y4").frames[2][1].tolist()
    # Another centre RT is another request.
    mob.engine_mobilogram(tims, d, centre_rt=2206.0)
    assert len([c for c in fake_engine.calls if c[1] == "mobilogram"]) == 2


def test_an_engine_failure_falls_back_to_the_centroids(tims, fake_engine):
    fake_engine.returncode, fake_engine.stderr = 1, "log line\nError: the .d is damaged\n"
    d = _detail(tims, 11)
    with pytest.raises(ViewerError, match=r"the \.d is damaged"):
        mob.engine_mobilogram(tims, d, centre_rt=1.0)
    m, why = mobc.get_mobilogram(tims, d, "raw", 1.0)
    assert m.source == "centroids" and "damaged" in why
    _, note = mobc.render(tims, d, fragments_of(d.chromatogram), source="raw", k=2, norm="abs")
    assert "Raw profile not available" in json.dumps(note, default=str)


def test_engine_timeout_is_reported(tims, fake_engine, monkeypatch):
    def slow(cmd, **kwargs):
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "mumdia 0.9.0", "")
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(mob.subprocess, "run", slow)
    with pytest.raises(ViewerError, match="did not answer within 3 s"):
        mob.engine_mobilogram(tims, _detail(tims, 11), centre_rt=7.0, timeout=3)


def test_without_an_engine_the_raw_source_is_disabled(tims):
    d = _detail(tims, 11)
    ok, why = mob.engine_status(tims, d.run)
    assert not ok and "MUMDIA_BIN" in why
    card = mobc.mobilogram_card(PageContext(rs=tims, base="/"), d, fragments_of(d.chromatogram))
    assert card.id == "pd-mob-card"
    parts = {c.id: c for c in _walk(card) if isinstance(getattr(c, "id", None), str)}
    source = parts["pd-mob-source"]
    assert source.value == "centroids"
    assert [x.get("disabled", False) for x in source.data] == [True, False]
    assert mob.CENTROID_LABEL in json.dumps(parts["pd-mob-note"].children, default=str)


def test_the_figure_draws_sticks_markers_and_the_windows(tims, recorded):
    d = _detail(tims, 11)
    frags = fragments_of(d.chromatogram)
    m = mob.centroid_mobilogram(tims, d)
    fig = mobc.mobilogram_figure(m, frags, d, retrace_hw=mobc.retrace_half_width(tims))
    names = {s.name for s in fig.layout.shapes}
    assert {"im-window", "im-pred", "apex-im", "retrace-band"} <= names
    w = d.window
    band = next(s for s in fig.layout.shapes if s.name == "im-window")
    assert (band.x0, band.x1) == (w.im_lo, w.im_hi)
    hw = tims.config_get("retrace", "im_half_width", default=0.015)
    rb = next(s for s in fig.layout.shapes if s.name == "retrace-band")
    assert rb.x1 - rb.x0 == pytest.approx(2 * hw)
    # One marker trace per predicted fragment, the XIC's colour, its sticks in its group.
    tops = [t for t in fig.data if (t.meta or {}).get("kind") == "tops"]
    by_index = {f.index: f for f in frags}
    for t in tops:
        f = by_index[t.meta["frag"]]
        assert t.marker.color == f.colour and t.legendgroup == f"f{f.index}"
        im, y = m.profile(m.ion(f.name), 2)
        assert np.allclose(np.asarray(t.x, dtype=float), np.round(im, 5), atol=1e-6)
        assert np.allclose(np.asarray(t.y, dtype=float), y)
    # Per-ion scaling and the centre frame alone.
    scaled = mobc.mobilogram_figure(m, frags, d, norm="ion", k=0, hidden={frags[0].index})
    for t in scaled.data:
        ys = [v for v in (() if t.y is None else t.y) if v is not None]
        if ys:
            assert max(ys) <= 1.0 + 1e-12
    assert all(
        t.visible == "legendonly"
        for t in scaled.data
        if (t.meta or {}).get("frag") == frags[0].index
    )
    raw = mob.parse_engine_json(recorded, 11)
    rfig = mobc.mobilogram_figure(raw, frags, d)
    lines = [t for t in rfig.data if (t.meta or {}).get("frag") is not None]
    assert lines and all(t.mode == "lines" for t in lines)
    assert rfig.layout.meta["source"] == "raw"


def test_the_callback_follows_the_shown_scan_and_the_controls(tims):
    d = _detail(tims, 12)
    key = {"run": tims.runs[0].name, "cid": 12}
    apex = d.apex_scan
    _, note = mobc.update(
        tims,
        key,
        {"row": apex.row, "rt": apex.rt},
        "centroids",
        "2",
        "abs",
        [],
        "light",
        detail_page.get_detail,
    )
    text = json.dumps(note, default=str)
    assert "the apex" in text and "5 MS2 frames" in text
    # A scan two frames later moves the centre.
    scans = ScanTable.for_run(tims, d.run)
    later = scans.step(apex.row, 2)
    fig2, note2 = mobc.update(
        tims,
        key,
        {"row": later.row, "rt": later.rt},
        "centroids",
        "0",
        "ion",
        [],
        "dark",
        detail_page.get_detail,
    )
    text2 = json.dumps(note2, default=str)
    assert "the shown scan" in text2 and "1 MS2 frame " in text2
    assert f"{later.rt:.2f} s" in text2
    assert fig2.layout.meta["frames"] == 0 and fig2.layout.meta["norm"] == "ion"


def test_cli_refuses_an_engine_that_is_not_a_file(fixture_dir, tmp_path):
    from mumdia_viewer.cli import main

    assert main([str(fixture_dir("tims")), "--engine", str(tmp_path / "nope")]) == 2


@pytest.mark.engine
def test_the_engine_reads_the_raw_mobilogram_of_the_fixture(fixture_dir, recorded):
    if not ENGINE or not Path(ENGINE).is_file():
        pytest.skip("MUMDIA_BIN is not set")
    if not TIMS_RAW or not Path(TIMS_RAW).is_dir():
        pytest.skip("MUMDIA_VIEWER_TIMS_RAW is not set")
    from mumdia_viewer.data import open_results

    report = json.loads(
        (fixture_dir("tims") / "spectra" / "spectra_ms2.parquet.report.json").read_text()
    )
    rs = open_results(fixture_dir("tims"), remaps={report["params"]["mzml"]: TIMS_RAW})
    mob.set_engine(ENGINE)
    d = precursor_detail(rs, rs.runs[0].name, 11)
    assert mob.engine_status(rs, d.run) == (True, "")
    m = mob.engine_mobilogram(rs, d)
    want = mob.parse_engine_json(recorded, 11)
    assert [f.label for f in m.ms2_frames] == [f.label for f in want.ms2_frames]
    for ion in want.ions:
        got = m.ion(ion.name)
        assert got is not None
        for (gi, gy), (wi, wy) in zip(got.frames, ion.frames, strict=True):
            assert np.allclose(gi, wi) and np.allclose(gy, wy), ion.name
