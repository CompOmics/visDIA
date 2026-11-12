"""SCIEX ZT Scan DIA runs: the Q1 sweep, the cycle traces, the patch and the page.

The run is synthetic (:mod:`ztscan_fixture`): spectra_ms2 v3 with the Q1 sweep, cal.json
with the Q1 calibration and psms_extracted v6 with the per-fragment decisions, the signal
planted with known amplitudes through the calibrated transmission. The engine's
cycle-level traces (chromatograms v5, psms_extracted v9) are the same run written with
``cycle_traces=True``. Tests marked ``real_data`` read ``MUMDIA_VIEWER_REAL_ZTSCAN`` (a MuMDIA
0.12.0 ZT Scan run or experiment directory).
"""

from __future__ import annotations

import json
import math
import os
import socket
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from plotly.io import to_json as to_json_plotly

from mumdia_viewer.data import ViewerError, open_results
from mumdia_viewer.data import ztscan as zt
from mumdia_viewer.data.detail import precursor_detail
from mumdia_viewer.data.errors import SchemaVersionError
from mumdia_viewer.data.schemas import (
    check_version,
    chromatogram_layout,
    is_ztscan_version,
    supported_versions,
)
from mumdia_viewer.ui import detail as detail_page
from mumdia_viewer.ui import detail_figures as dfig
from mumdia_viewer.ui import detail_ztscan as ztc
from mumdia_viewer.ui.detail_view import fragments_of, scan_grid
from mumdia_viewer.ui.state import PageContext
from test_ui_spectra import _walk
from ztscan_fixture import STEP, build_ztscan_run


@pytest.fixture(scope="module")
def zrun(tmp_path_factory):
    truth = build_ztscan_run(tmp_path_factory.mktemp("zt_steps"))
    return truth, open_results(truth.root)


@pytest.fixture(scope="module")
def zcyc(tmp_path_factory):
    truth = build_ztscan_run(tmp_path_factory.mktemp("zt_cycles"), cycle_traces=True)
    return truth, open_results(truth.root)


def _ids(tree) -> set[str]:
    return {json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)}


def _has(tree, id_: str) -> bool:
    return json.dumps(id_) in _ids(tree)


# --------------------------------------------------------------------------- physics


def test_averagine_is_the_normalised_poisson():
    for mass in (500.0, 1500.0, 3200.0):
        lam = 0.000594 * mass
        p = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(6)])
        np.testing.assert_allclose(zt.averagine_isotopes(mass, 6), p / p.sum(), rtol=1e-12)


def test_transmission_is_the_trapezoid():
    h, e = 2.5, 4.0
    x = np.array([0.0, 0.5, 2.5 - 2.0, 2.5, 4.5, 5.0, -2.5])
    np.testing.assert_allclose(zt.transmission(x, h, e), [1, 1, 1, 0.5, 0, 0, 0.5])
    np.testing.assert_allclose(zt.transmission(np.array([2.4, 2.6]), 2.5, 0.0), [1, 0])


def test_isotope_weights_follow_the_fragment_share():
    w_small = zt.isotope_weights(600.0, 2, 200.0, 1)
    w_big = zt.isotope_weights(600.0, 2, 1000.0, 1)
    assert w_small.sum() == pytest.approx(1.0) and w_big.sum() == pytest.approx(1.0)
    # A heavier fragment keeps the heavy isotopes itself: its window sees more of M0.
    assert w_big[0] > w_small[0]


def test_model_from_the_report_and_from_cal_json_agree():
    from ztscan_fixture import CAL_BLOCK, MODEL_BLOCK

    a = zt.model_from_block(MODEL_BLOCK, "report")
    b = zt.model_from_block(CAL_BLOCK, "cal")
    assert (a.step_mz, a.half_width_mz, a.edge_width_mz) == (
        b.step_mz,
        b.half_width_mz,
        b.edge_width_mz,
    )
    assert a.tolerances == b.tolerances
    for pm in (450.0, 520.3, 880.0):
        assert a.offset_mz(pm, 500.0, 100.0) == pytest.approx(b.offset_mz(pm, 500.0, 100.0))
    assert a.offset_mz(400.0, None, 0.0) == pytest.approx(0.12)
    assert a.offset_mz(650.0, None, 0.0) == pytest.approx(0.07)
    assert a.offset_mz(2000.0, None, 0.0) == pytest.approx(0.02)  # held at the end
    with pytest.raises(ViewerError, match="status"):
        zt.model_from_block({**CAL_BLOCK, "status": "too_few_anchors"}, "cal")
    with pytest.raises(ViewerError, match="not a window"):
        zt.model_from_block({**MODEL_BLOCK, "half_width_mz": 0.0}, "report")


def test_candidate_centre_is_mass_plus_mean_isotope_minus_offset():
    from ztscan_fixture import MODEL_BLOCK

    m = zt.model_from_block(MODEL_BLOCK, "report")
    c = zt.CandQ1(m, 520.3, 2, 100.0, ["y5"], [589.33])
    pre = zt.averagine_isotopes((520.3 - zt.PROTON) * 2, 6)
    ibar = float(np.dot(np.arange(6), pre))
    assert c.centre == pytest.approx(
        520.3 + zt.ISOTOPE_SPACING / 2 * ibar - m.offset_mz(520.3, None, 100.0)
    )
    # The fragment's expected profile peaks at its own centre and is 0 beyond the reach.
    f = c.frags[0]
    assert float(c.expected(0, f.centre)) == pytest.approx(1.0, abs=0.05)
    assert float(c.expected(0, f.centre - m.reach_mz - 0.5)) == 0.0
    assert float(c.expected(0, f.centre + m.reach_mz + 2.0)) < 1e-5  # the far isotopes
    # Shifting by mu steps moves the profile by mu steps.
    t = f.centre + 1.3
    assert float(c.expected(0, t, mu=2.0)) == pytest.approx(float(c.expected(0, t - 2 * m.step_mz)))


def test_collapse_is_the_least_squares_amplitude():
    cyc = np.array([3, 3, 3, 4, 4])
    rt = np.array([10.0, 10.001, 10.002, 11.0, 11.001], dtype=np.float32)
    w = np.array([0.5, 1.0, 0.5, 1.0, 0.25])
    x = np.array([50.0, 100.0, 50.0, 80.0, 20.0])
    mrt, val, c, n = zt._collapse(cyc, rt, x, w)
    np.testing.assert_allclose(val, [100.0, 80.0])
    assert c.tolist() == [3, 4] and n.tolist() == [3, 2]
    assert mrt[0] == pytest.approx(float(np.float32(rt[:3].astype(np.float64).mean())))
    # Every weight 0: the value is 0, as the engine's.
    assert zt._collapse(cyc[:1], rt[:1], x[:1], np.zeros(1))[1].tolist() == [0.0]


# --------------------------------------------------------------------------- schemas


def test_the_zt_scan_versions_are_read_and_the_im_branch_still_refused():
    names = ["scan_index", "rt_seconds", "mz", "intensity", "q1_cycle", "q1_step"]
    assert 3 in supported_versions("spectra_ms2", columns=names)
    check_version("spectra_ms2", 3, where="x", columns=names)
    # The ion-mobility branch's v3 (a per-peak `im`, no sweep) is refused as before.
    with pytest.raises(SchemaVersionError, match="ion-mobility branch"):
        check_version("spectra_ms2", 3, where="x", columns=["scan_index", "mz", "im"])
    with pytest.raises(SchemaVersionError, match="ion-mobility branch"):
        check_version("spectra_ms2", 3, where="x")
    for v in (6, 7, 8, 9):
        check_version("psms_extracted", v, where="x", columns=["candidate_id"])
    check_version("psms_extracted", 5, where="x", columns=["q1_corr"])
    with pytest.raises(SchemaVersionError):
        check_version("psms_extracted", 5, where="x", columns=["apex_im"])
    for v in (3, 4):
        check_version("peptide_quant", v, where="x")
    assert is_ztscan_version("chromatograms", 6, None)
    for v in (5, 6):
        check_version("chromatograms", v, where="x")


def test_cycle_trace_chromatogram_layouts():
    v1 = pa.schema(
        [("rt", pa.large_list(pa.float32())), ("intensity", pa.large_list(pa.float32()))]
    )
    v2 = pa.schema(
        [(c, pa.large_list(pa.float32())) for c in ("rt_axis", "intensity_trimmed")]
        + [("trace_offset", pa.uint32()), ("trace_len", pa.uint32())]
    )
    assert chromatogram_layout(v1).schema_version == 1
    assert chromatogram_layout(v2).schema_version == 2
    assert chromatogram_layout(v1.append(pa.field("q1_class", pa.float32()))).schema_version == 5
    assert chromatogram_layout(v2.append(pa.field("q1_class", pa.float32()))).schema_version == 6


def test_only_zt_scan_runs_are_zt_scan_runs(zrun, open_fixture):
    _, rs = zrun
    assert rs.runs[0].is_ztscan
    assert not rs.notices
    for name in ("single", "experiment", "tims"):
        other = open_fixture(name)
        for run in other.runs:
            assert not run.is_ztscan
            assert zt.ztscan_run(other, run) is None


def test_the_sweep_of_the_run(zrun):
    truth, rs = zrun
    zr = zt.ztscan_run(rs, rs.runs[0])
    assert zr is zt.ztscan_run(rs, rs.runs[0])  # memoised
    s = zr.sweep
    assert (s.n_cycles, s.n_steps) == (40, 60)
    assert s.step_mz == pytest.approx(STEP)
    assert s.cycle_s == pytest.approx(1.0)
    np.testing.assert_allclose(s.step_target, truth.targets)
    rows = s.cell(np.array([0, 5, 39, 40]), np.array([0, 59, 60]))
    assert rows[0, 0] == 0 and rows[1, 1] == 5 * 60 + 59 and rows[2, 1] == 39 * 60 + 59
    assert (rows[3] == -1).all() and (rows[:, 2] == -1).all()
    assert zr.model is not None and zr.model.source.startswith("psms_extracted report")
    assert "60 Q1 steps" in zr.method_label and "+-1.9 m/z" in zr.method_label


# --------------------------------------------------------------------------- cycle traces


def test_cycle_traces_recover_the_planted_amplitudes(zrun):
    truth, rs = zrun
    c0 = truth.cands[0]
    d = precursor_detail(rs, rs.runs[0], 0)
    ct = zt.cycle_traces(rs, d)
    assert ct.collapsed and ct.source == "viewer" and ct.has_steps
    assert ct.label == zt.CYCLE_LABEL
    assert ct.classes == dict(enumerate(c0.classes))
    for t in ct.fragments:
        f = t.index
        if c0.amp[f] == 0:
            assert t.value.size == 0  # never observed
            continue
        want = np.array([c0.amp[f] * truth.elution(c0, int(k)) for k in t.cycles])
        if c0.mu[f] == 0:
            np.testing.assert_allclose(t.value, want, rtol=1e-4, atol=1e-3)
        else:
            # Centred four steps away: under this precursor's profile it collapses lower.
            assert t.value.max() < 0.75 * want.max()
        # Every cycle of the trace has its covering steps (5 inside the sweep).
        assert set(t.n[1:-1].tolist()) == {5}
    # One point per cycle at the float32 mean RT of the cycle's samples.
    t = ct.fragment(2)
    chrom = d.chromatogram.fragments()[2]
    for rt in t.rt:
        pts = chrom.rt[np.round(chrom.rt.astype(np.float64)) == np.round(rt)]
        assert rt == pytest.approx(float(np.float32(pts.astype(np.float64).mean())))
    assert ct.apex_cycle == c0.apex_cycle
    # The MS1 rows: one value per cycle.
    assert {m.name for m in ct.ms1} == {"ms1_mono", "ms1_iso1", "ms1_iso2"}
    assert all(np.all(np.diff(m.cycles) == 1) for m in ct.ms1)


def test_engine_cycle_traces_are_taken_as_written(zcyc):
    truth, rs = zcyc
    d = precursor_detail(rs, rs.runs[0], 0)
    assert d.chromatogram.layout.schema_version == 5
    ct = zt.cycle_traces(rs, d)
    assert ct.source == "engine" and not ct.has_steps and ct.label == zt.ENGINE_CYCLE_LABEL
    c0 = truth.cands[0]
    frags = d.chromatogram.fragments()
    assert [t.q1_class for t in frags] == c0.classes
    for t, tr in zip(ct.fragments, frags, strict=True):
        np.testing.assert_array_equal(t.value, tr.intensity.astype(np.float64))
        if t.value.size:
            want = np.array([c0.amp[t.index] * truth.elution(c0, int(k)) for k in t.cycles])
            if c0.mu[t.index] == 0:
                np.testing.assert_allclose(t.value, want, rtol=1e-4, atol=1e-3)


def test_a_non_zt_run_has_no_cycle_traces(open_fixture):
    rs = open_fixture("single")
    cid = int(
        rs.duck.df(
            "SELECT candidate_id FROM read_parquet($p) LIMIT 1", {"p": str(rs.scored.require())}
        ).iloc[0, 0]
    )
    d = precursor_detail(rs, rs.runs[0], cid)
    with pytest.raises(ViewerError, match="not a ZT Scan run"):
        zt.cycle_traces(rs, d)
    assert ztc.zt_traces(rs, d) == (None, None)


# --------------------------------------------------------------------------- patch and profile


def test_the_patch_holds_the_planted_cells(zrun):
    truth, rs = zrun
    zt.CELL_CACHE.clear()
    d = precursor_detail(rs, rs.runs[0], 0)
    p = zt.q1_patch(rs, d)
    c0 = truth.cands[0]
    assert p.apex_cycle == c0.apex_cycle and p.apex2d_cycle == c0.apex_cycle
    assert p.cycles.tolist() == list(range(c0.apex_cycle - 3, c0.apex_cycle + 4))
    reach = truth.model.reach_mz + 2 * STEP
    want_steps = np.flatnonzero(np.abs(truth.targets - p.centre_mz) <= reach)
    assert p.steps.tolist() == want_steps.tolist()
    assert p.read.row_groups >= 2 and p.read.cells == p.cycles.size * p.steps.size
    for f, mz in enumerate(c0.mz):
        for ci, c in enumerate(p.cycles):
            for si, s in enumerate(p.steps):
                planted = truth.planted[(int(c), int(s))].get(mz, 0.0)
                assert p.values[f, ci, si] == pytest.approx(planted, rel=1e-6, abs=1e-3)
    # The unfragmented precursor's isotopes, planted through their own windows.
    assert np.nanmax(p.precursor[0]) > 0
    assert p.fitted_centre_steps == pytest.approx(0.1) and p.width == 1.0
    assert p.classes.tolist() == c0.classes
    # Kept: the same object; a wider read reuses the cached cells.
    assert zt.q1_patch(rs, d) is p
    wide = zt.q1_patch(rs, d, half_cycles=6)
    assert wide.cycles.size == 13 and wide.read.cached == p.read.cells
    assert any("13 cycles read" in n for n in wide.notes)


def test_the_profile_scales_the_fragments_onto_the_model(zrun):
    truth, rs = zrun
    c0 = truth.cands[0]
    d = precursor_detail(rs, rs.runs[0], 0)
    p = zt.q1_patch(rs, d)
    prof = zt.q1_profile(p)
    assert prof.cycles.tolist() == [c0.apex_cycle]
    for f in range(len(c0.names)):
        if c0.amp[f] == 0:
            assert prof.amplitude[f] == 0
        elif c0.mu[f] == 0:
            assert prof.amplitude[f] == pytest.approx(c0.amp[f], rel=1e-4)
            np.testing.assert_allclose(
                prof.observed[f], prof.amplitude[f] * prof.expected[f], rtol=1e-4, atol=0.6
            )
        else:
            # Its peak sits mu steps from its expected centre.
            k = int(np.nanargmax(prof.observed[f]))
            off = (p.targets[k] - p.expected_centre[f]) / STEP
            assert off == pytest.approx(c0.mu[f], abs=0.6)
    summed = zt.q1_profile(p, half=2)
    assert summed.cycles.size == 5
    assert summed.amplitude[0] == pytest.approx(
        c0.amp[0] * sum(truth.elution(c0, c) for c in range(18, 23)), rel=1e-4
    )


def test_the_patch_needs_a_model(zrun, tmp_path):
    truth = build_ztscan_run(tmp_path / "nomodel")
    # Without the extract report's model and without cal.json there is no model.
    rep = truth.root / "psms_extracted.parquet.report.json"
    r = json.loads(rep.read_text())
    r["params"].pop("q1_fragments")
    rep.write_text(json.dumps(r))
    (truth.root / "cal.json").unlink()
    rs = open_results(truth.root)
    d = precursor_detail(rs, rs.runs[0], 0)
    with pytest.raises(ViewerError, match="no Q1 model"):
        zt.q1_patch(rs, d)
    ct = zt.cycle_traces(rs, d)
    assert ct.label == zt.UNIFORM_LABEL and any("averaged" in n for n in ct.notes)


# --------------------------------------------------------------------------- figures and page


def _zt_traces(fig):
    return [t for t in fig.data if isinstance(t.meta, dict) and "zt" in t.meta]


def test_the_xic_of_a_zt_run(zrun):
    truth, rs = zrun
    d = precursor_detail(rs, rs.runs[0], 0)
    frags = fragments_of(d.chromatogram)
    ct, why = ztc.zt_traces(rs, d)
    assert why is None
    grid = scan_grid(rs, d)
    for mode in dfig.ZT_MODES:
        fig = dfig.xic_figure(d, frags, grid, zt=ct, zt_mode=mode)
        show_cycle, show_step, _, _ = dfig.ZT_STYLE[mode]
        cyc = [t for t in _zt_traces(fig) if t.meta["zt"] == "cycle" and t.meta.get("qf") != "fill"]
        steps = [t for t in _zt_traces(fig) if t.meta["zt"] == "step"]
        assert cyc and steps
        assert all(t.meta["zhide"] is (not show_cycle) for t in cyc)
        assert all(t.meta["zhide"] is (not show_step) for t in steps)
        assert fig.layout.meta["zt"] == mode
        assert set(fig.layout.meta["yz"]) == set(dfig.ZT_MODES)
    fig = dfig.xic_figure(d, frags, grid, zt=ct)
    by_frag = {
        t.meta["frag"]: t for t in _zt_traces(fig) if t.meta["zt"] == "cycle" and "frag" in t.meta
    }
    c0 = truth.cands[0]
    for f in frags:
        if f.index in by_frag:
            dash = dfig.Q1_CLASS_STYLE[c0.classes[f.index]][0]
            assert by_frag[f.index].line.dash == dash
    # One click target per Q1 cycle.
    bars = [t for t in fig.data if t.type == "bar"]
    assert bars and len(bars[0].x) == max(t.cycles.size for t in ct.fragments)


def test_the_xic_of_any_other_run_is_unchanged(open_fixture):
    rs = open_fixture("single")
    cid = int(
        rs.duck.df(
            "SELECT candidate_id FROM read_parquet($p) WHERE label = 'target' "
            "ORDER BY q_value LIMIT 1",
            {"p": str(rs.scored.require())},
        ).iloc[0, 0]
    )
    d = precursor_detail(rs, rs.runs[0], cid)
    frags = fragments_of(d.chromatogram)
    grid = scan_grid(rs, d)
    plain = dfig.xic_figure(d, frags, grid)
    assert to_json_plotly(dfig.xic_figure(d, frags, grid, zt=None)) == to_json_plotly(plain)
    assert not _zt_traces(plain) and "yz" not in plain.layout.meta
    tree = detail_page.layout(PageContext(rs=rs, base="/t/", query={"run": "", "cid": str(cid)}))
    assert not _has(tree, "pd-zt-card") and not _has(tree, "pd-xic-zt")


def test_the_page_of_a_zt_run(zrun):
    _, rs = zrun
    tree = detail_page.layout(PageContext(rs=rs, base="/t/", query={"run": "", "cid": "0"}))
    for id_ in (
        "pd-zt-card",
        "pd-xic-zt",
        "pd-zt-frag",
        "pd-zt-cycles",
        "pd-zt-span",
        "pd-zt-table",
    ):
        assert _has(tree, id_), id_
    text = to_json_plotly(tree)
    assert "one value per cycle, viewer-derived" in text
    assert "Q1 class" in text and "q1f_n_belong" in text
    # The identification page's compact preview draws the cycle traces too.
    card = detail_page.preview(PageContext(rs=rs, base="/"), "", 0)
    assert '"zt":"cycle"' in to_json_plotly(card).replace(" ", "")


def test_the_page_of_a_run_with_engine_cycle_traces(zcyc):
    _, rs = zcyc
    tree = detail_page.layout(PageContext(rs=rs, base="/t/", query={"run": "", "cid": "0"}))
    assert _has(tree, "pd-zt-card") and not _has(tree, "pd-xic-zt")
    assert "the engine's cycle-level trace" in to_json_plotly(tree)
    d = precursor_detail(rs, rs.runs[0], 0)
    ct, _ = ztc.zt_traces(rs, d)
    fig = dfig.xic_figure(d, fragments_of(d.chromatogram), scan_grid(rs, d), zt=ct, zt_mode="both")
    assert fig.layout.meta["zt"] == "cycles"
    assert not [t for t in _zt_traces(fig) if t.meta["zt"] == "step"]
    # The engine's separable model (psms_extracted v9): A_f * E_c at the model's cycle RTs.
    model = {t.meta["frag"]: t for t in _zt_traces(fig) if t.meta.get("model")}
    truth, _ = zcyc
    c0 = truth.cands[0]
    assert set(model) == {f for f, a in enumerate(c0.amp) if a > 0}
    y2 = model[2]
    want = [c0.amp[2] * truth.elution(c0, c0.apex_cycle + k) for k in range(-3, 4)]
    np.testing.assert_allclose(np.asarray(y2.y, dtype=float), want, rtol=1e-3, atol=0.1)
    _, _, table, _ = ztc.update(
        rs,
        {"run": "", "cid": 0},
        None,
        "all",
        "3",
        "frag",
        "0",
        "norm",
        [],
        "light",
        detail_page.get_detail,
    )
    assert "model R" in json.dumps(table.to_plotly_json(), default=str)


def test_the_q1_card_callback(zrun):
    truth, rs = zrun
    d = detail_page.get_detail(rs, "", 0)
    key = {"run": "", "cid": 0}
    patch_fig, prof_fig, _, note = ztc.update(
        rs, key, None, "all", "3", "frag", "0", "norm", [], "light", detail_page.get_detail
    )
    heat = [t for t in patch_fig.data if t.type == "heatmap"]
    n_obs = len(fragments_of(d.chromatogram))
    assert len(heat) == n_obs
    names = {s.name for s in patch_fig.layout.shapes}
    assert {"expected", "fitted", "own", "apex"} <= names
    assert patch_fig.layout.coloraxis.cmax == 1
    assert "Viewer read" in json.dumps(note, default=str)
    # One fragment beside the precursor M0; shared scale; hidden fragments left out.
    one, *_ = ztc.update(
        rs, key, None, "2", "6", "abs", "2", "abs", [], "dark", detail_page.get_detail
    )
    assert len([t for t in one.data if t.type == "heatmap"]) == 2
    hid, *_ = ztc.update(
        rs, key, None, "all", "3", "frag", "0", "norm", [0, 1], "light", detail_page.get_detail
    )
    assert len([t for t in hid.data if t.type == "heatmap"]) == n_obs - 2
    # The profile: the model, the engine's fit and the precursor in MS2.
    names = [t.name for t in prof_fig.data]
    assert "calibrated transmission" in names and "precursor in MS2" in names
    assert any("engine's fit" in (n or "") for n in names)
    # Following the shown scan: a scan 8 cycles after the apex moves the patch there.
    zr = zt.ztscan_run(rs, rs.runs[0])
    row = int(zr.sweep.cell(np.array([truth.cands[0].apex_cycle + 8]), np.array([27]))[0, 0])
    _, r2, *_ = ztc.update(
        rs, key, {"row": row}, "all", "3", "frag", "0", "norm", [], "light", detail_page.get_detail
    )
    assert "Q1 cycle 28" in json.dumps(r2.to_plotly_json(), default=str)


def test_the_q1_card_says_why_without_a_patch(zrun, monkeypatch):
    _, rs = zrun
    monkeypatch.setattr(zt, "q1_patch", lambda *a, **k: (_ for _ in ()).throw(ViewerError("boom")))
    _, _, table, note = ztc.update(
        rs,
        {"run": "", "cid": 0},
        None,
        "all",
        "3",
        "frag",
        "0",
        "norm",
        [],
        "light",
        detail_page.get_detail,
    )
    assert "No Q1 patch: boom" in note and table is None


def _chromium():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            p.chromium.launch().close()
    except Exception:
        return None
    return sync_playwright


SYNC_PLAYWRIGHT = _chromium()


@pytest.mark.skipif(SYNC_PLAYWRIGHT is None, reason="Chromium for playwright is not installed")
def test_the_zt_page_in_chromium(zrun):
    """The card fills after the page is shown; the XIC switches between cycles and steps."""
    from werkzeug.serving import make_server

    from mumdia_viewer.ui.app import create_app

    _, rs = zrun
    app = create_app(rs, url_base="/")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    vis = """(z) => { const gd = document.querySelector('#pd-xic-wrap .js-plotly-plot');
      return gd.data.filter(t => t.meta && t.meta.zt === z && t.meta.frag !== undefined
        && t.meta.qf !== 'fill').map(t => t.visible === true); }"""
    try:
        with SYNC_PLAYWRIGHT() as p:
            chromium = p.chromium.launch()
            page = chromium.new_page(viewport={"width": 1500, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/precursor?cid=0")
            page.wait_for_selector("#pd-zt-card", timeout=60000)
            page.wait_for_function(
                "() => { const g = document.querySelector('#pd-zt-card .js-plotly-plot');"
                " return g && g.data && g.data.some(t => t.type === 'heatmap'); }",
                timeout=60000,
            )
            assert all(page.evaluate(vis, "cycle")) and not any(page.evaluate(vis, "step"))
            page.locator("#pd-xic-card label:has-text('Cycles + steps')").click()
            page.wait_for_function(f"() => ({vis})('step').every(v => v)", timeout=10000)
            assert all(page.evaluate(vis, "cycle"))
            page.locator("#pd-xic-card label:has-text('Steps')").last.click()
            page.wait_for_function(f"() => !({vis})('cycle').some(v => v)", timeout=10000)
            assert all(page.evaluate(vis, "step"))
            assert not errors, errors
            chromium.close()
    finally:
        server.shutdown()


def _serve(rs):
    from werkzeug.serving import make_server

    from mumdia_viewer.ui.app import create_app

    app = create_app(rs, url_base="/")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


@pytest.mark.skipif(SYNC_PLAYWRIGHT is None, reason="Chromium for playwright is not installed")
def test_a_non_zt_page_in_chromium_has_no_zt_parts(open_fixture):
    """Another run's precursor page has no ZT control or card, and no script error."""
    rs = open_fixture("single")
    cid = int(
        rs.duck.df(
            "SELECT candidate_id FROM read_parquet($p) WHERE label = 'target' "
            "ORDER BY q_value LIMIT 1",
            {"p": str(rs.scored.require())},
        ).iloc[0, 0]
    )
    server, base = _serve(rs)
    try:
        with SYNC_PLAYWRIGHT() as p:
            chromium = p.chromium.launch()
            page = chromium.new_page(viewport={"width": 1500, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/precursor?cid={cid}")
            page.wait_for_selector("#pd-xic-card .js-plotly-plot", timeout=60000)
            page.locator("#pd-xic-card label").filter(has_text="Log").click()
            page.wait_for_function(
                "() => document.querySelector('#pd-xic-wrap .js-plotly-plot')"
                ".layout.yaxis.type === 'log'",
                timeout=10000,
            )
            assert page.locator("#pd-zt-card").count() == 0
            assert page.locator("#pd-xic-zt").count() == 0
            assert not errors, errors
            chromium.close()
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- real data


def _real_zt() -> Path:
    value = os.environ.get("MUMDIA_VIEWER_REAL_ZTSCAN")
    if not value:
        pytest.skip("MUMDIA_VIEWER_REAL_ZTSCAN is not set")
    return Path(value)


@pytest.mark.real_data
def test_real_zt_run_cycle_traces_and_patch():
    """On a MuMDIA ZT Scan run: every accepted target sampled has cycle traces whose cycles
    ascend, and a patch whose expected centre lies inside the cells read."""
    rs = open_results(_real_zt())
    run = rs.runs[0]
    assert run.is_ztscan
    df = rs.duck.df(
        "SELECT candidate_id FROM read_parquet($p) WHERE label = 'target' AND run_psm_q <= 0.01 "
        "AND source = $s ORDER BY candidate_id LIMIT 4000",
        {"p": str(rs.scored.require()), "s": run.index},
    )
    for cid in df["candidate_id"].to_numpy()[::1000]:
        d = precursor_detail(rs, run, int(cid))
        ct = zt.cycle_traces(rs, d)
        assert ct.collapsed
        for t in ct.fragments:
            assert np.all(np.diff(t.cycles) >= 1)
        p = zt.q1_patch(rs, d)
        assert p.targets[0] < p.centre_mz < p.targets[-1]
