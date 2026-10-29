"""Ion mobility (diaPASEF) on the committed cut of a real timsTOF run (``tims`` fixture).

Every value the viewer shows is checked against the parquet files it came from; the 3D
fixtures must show no ion-mobility element.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import calibration as cal_data
from mumdia_viewer.data import qc as qcd
from mumdia_viewer.data.detail import detail_percentiles, precursor_detail
from mumdia_viewer.data.features import EVIDENCE_FEATURES
from mumdia_viewer.data.spectra import Ms1Table, ScanTable
from mumdia_viewer.ui import calibration_figures as cfig
from mumdia_viewer.ui import spectra as sp
from mumdia_viewer.ui import spectra_cards
from mumdia_viewer.ui.app import PAGES
from mumdia_viewer.ui.detail_view import evidence_groups
from mumdia_viewer.ui.state import PageContext
from test_ui_spectra import _walk

THREE_D = ("single", "experiment", "grouped", "topk", "mbr")
# Ids of the elements only an ion-mobility run shows.
IM_IDS = ("pd-ev-im", "pd-im-card", "cal-im-card", "cal-im-err-card")


@pytest.fixture(scope="module")
def tims(open_fixture):
    return open_fixture("tims")


def _table(rs, kind, columns=None):
    return pq.read_table(rs.runs[0].artifact(kind).path, columns=columns).to_pandas()


def _accepted(rs, n):
    s = _table(rs, "psms_scored", ["candidate_id", "label", "q_value"])
    s = s[(s.label == "target") & (s.q_value <= 0.01)].sort_values("candidate_id")
    return s.candidate_id.to_numpy()[:: max(1, len(s) // n)][:n]


def _json(tree):
    return json.dumps(tree.to_plotly_json(), default=str)


def test_opens_by_default_with_ion_mobility(tims, open_fixture):
    # open_fixture opens without allow_unreleased: the diaPASEF versions are accepted.
    run = tims.runs[0]
    assert run.has_im
    assert run.artifact("chromatograms").version.version == 4
    assert run.artifact("psms_extracted").version.version == 4
    for name in THREE_D:
        assert not any(r.has_im for r in open_fixture(name).runs), name


def test_precursor_evidence_equals_the_tables(tims):
    run = tims.runs[0]
    ext = _table(tims, "psms_extracted")
    win = _table(tims, "run_windows", ["candidate_id", "im_pred_cal", "im_lo", "im_hi"])
    sel = _table(tims, "psms_scored", ["candidate_id", "selected_peak_rank"])
    feats = _table(tims, "features")
    for cid in _accepted(tims, 5):
        d = precursor_detail(tims, run.name, int(cid))
        rank = int(sel.set_index("candidate_id").selected_peak_rank.fillna(0)[cid])
        row = ext[(ext.candidate_id == cid) & (ext.peak_rank == rank)].iloc[0]
        w = win[win.candidate_id == cid].iloc[0]  # dense: row == candidate_id
        assert win.index[win.candidate_id == cid][0] == cid
        g = {e.key: e for items in evidence_groups(d.evidence).values() for e in items}
        assert g["apex_im"].value == pytest.approx(row.apex_im)
        assert g["apex_im_mad"].group == "fragments"
        assert g["apex_im_mad"].value == pytest.approx(row.apex_im_mad)
        assert g["im_pred_cal"].value == pytest.approx(w.im_pred_cal)
        assert g["im_window"].value == pytest.approx((w.im_lo, w.im_hi))
        assert g["im_error"].value == pytest.approx(row.apex_im - w.im_pred_cal)
        assert g["im_error"].derived
        f = feats[(feats.candidate_id == cid) & (feats.peak_rank == rank)].iloc[0]
        for key in ("im_elution_frag_sd", "im_elution_sd", "has_ms1_im"):
            assert g[key].value == pytest.approx(f[key]), key
        # The engine's im_error feature is the same difference.
        assert f.im_error == pytest.approx(row.apex_im - w.im_pred_cal, abs=1e-6)


def test_ion_mobility_features_are_ranked(tims):
    run = tims.runs[0]
    d = precursor_detail(tims, run.name, int(_accepted(tims, 1)[0]))
    rows = {r.feature: r for r in detail_percentiles(tims, d)}
    for name in ("im_error_abs", "im_frag_mad", "im_elution_frag_sd", "ms1_im_error_abs"):
        assert name in rows, name
        assert rows[name].pct_target is not None and rows[name].pct_decoy is not None
    im = [n for n in EVIDENCE_FEATURES if "im" in n.split("_")]
    assert set(im) <= set(d.features)


def test_chromatogram_im_is_aligned(tims):
    run = tims.runs[0]
    d = precursor_detail(tims, run.name, int(_accepted(tims, 1)[0]))
    traces = [t for t in d.chromatogram.traces if t.observed]
    assert traces and all(t.im is not None and t.im.size == t.intensity.size for t in traces)
    vals = np.concatenate([t.im[t.im > 0] for t in traces])
    assert vals.size and vals.min() > 0.5 and vals.max() < 1.6


def test_calibration_im_error_card(tims, fixture_dir):
    run = tims.runs[0]
    cal = cal_data.cal_record(tims, run.name)
    raw = json.loads((fixture_dir("tims") / "cal.json").read_text())
    assert cal.has_im and cal.w_im == pytest.approx(raw["w_im"])
    assert set(cal.im_per_charge) == set(raw["im_per_charge"])
    e = cal_data.accepted_errors(tims, run.name, 0.01)
    # The frame's im_error is the feature of the accepted rows' selected peaks.
    feats = _table(tims, "features", ["candidate_id", "peak_rank", "im_error"])
    sel = _table(tims, "psms_scored", ["candidate_id", "selected_peak_rank"]).fillna(0)
    want = sel.merge(
        feats,
        left_on=["candidate_id", "selected_peak_rank"],
        right_on=["candidate_id", "peak_rank"],
    )
    want = want.set_index("candidate_id").im_error
    got = e.frame.set_index("candidate_id").im_error.dropna()
    assert len(got) > 50
    assert np.allclose(got.to_numpy(), want[got.index].to_numpy())
    fig = cfig.im_error_figure(e, cal.w_im)
    assert len(fig.data[0].x) == len(got)
    edges = sorted(s.y0 for s in fig.layout.shapes if s.name in ("edge-lo", "edge-hi"))
    assert edges == pytest.approx([-cal.w_im, cal.w_im])


def test_window_scheme_and_identifications(tims):
    run = tims.runs[0]
    iw = pq.read_table(run.artifact("isolation_windows").path).to_pandas()
    ws = qcd.window_scheme(tims, run.name).frame.sort_values("window_id")
    iw = iw.sort_values("window_id")
    assert np.allclose(ws.im_lower.to_numpy(), iw.im_lower.to_numpy())
    assert np.allclose(ws.im_upper.to_numpy(), iw.im_upper.to_numpy())
    ids = qcd.ids_im(tims, run.name, 0.01)
    assert len(ids) == len(_accepted(tims, 10**6)) and (ids.apex_im > 0).all()


def test_spectra_carry_per_peak_im(tims):
    run = tims.runs[0]
    raw = pq.read_table(run.artifact("spectra_ms2").path).slice(3, 1).to_pylist()[0]
    table = ScanTable.for_run(tims, run)
    spec = table.spectrum(3)
    assert np.allclose(spec.im, raw["im"]) and spec.im.size == spec.mz.size
    assert spec.window_im_lower == pytest.approx(raw["window_im_lower"])
    assert spec.window_im_upper == pytest.approx(raw["window_im_upper"])
    lo, hi = float(np.quantile(spec.im, 0.25)), float(np.quantile(spec.im, 0.75))
    cut = spec.im_slice(lo, hi)
    assert cut.n_peaks == int(((spec.im >= lo) & (spec.im <= hi)).sum())
    assert ((cut.im >= lo) & (cut.im <= hi)).all() and cut.mz.size == cut.intensity.size
    assert Ms1Table.for_run(tims, run).spectrum(0).im is not None


def test_spectrum_browser_filters_to_the_candidate_window(tims):
    run = tims.runs[0]
    cid = int(_accepted(tims, 1)[0])
    ref, cid, _ = sp.start_of(tims, run, {"run": run.name, "cid": str(cid)}, 0.01)
    args = dict(t=0.01, near=spectra_cards.DEFAULT_NEAR, decoys=False, cid=cid, n_labels=10)
    every = sp.view(tims, run, ref, scale="window", scheme="light", im_mode="all", **args)
    cand = sp.view(tims, run, ref, scale="window", scheme="light", im_mode="cand", **args)
    w = precursor_detail(tims, run.name, cand["cid"]).window
    kept = cand["im_fig"].data
    n_kept = sum(len(t.x) for t in kept if (t.meta or {}).get("kind") == "kept")
    full = every["spec"]
    assert n_kept == int(((full.im >= w.im_lo) & (full.im <= w.im_hi)).sum())
    assert "candidate's window" in cand["im_note"]
    # The spectrum's peak tops carry 1/K0 colours on the run's fixed scale.
    tops = next(t for t in every["fig"].data if (t.meta or {}).get("kind") == "peaks")
    assert tops.marker.colorscale is not None
    assert (tops.marker.cmin, tops.marker.cmax) == sp.im_bounds(tims, run)
    slider = sp.view(
        tims,
        run,
        ref,
        scale="window",
        scheme="light",
        im_mode="slider",
        im_custom=[0.0, 0.1],
        **args,
    )
    assert "0 of" in slider["im_note"]


def _page(rs, page, run, cid):
    q = {
        "precursor": {"run": run.name, "cid": str(cid)},
        "spectra": {"run": run.name, "cid": str(cid)},
    }
    q |= {k: {"run": run.name} for k in ("calibration", "qc")}
    return PAGES[page].layout(PageContext(rs=rs, base="/", query=q.get(page, {})))


def _im_block_hidden(tree) -> bool:
    block = next(c for c in _walk(tree) if getattr(c, "id", None) == "sp-im-block")
    return (block.style or {}).get("display") == "none"


def test_every_page_builds_with_its_ion_mobility_panels(tims):
    run = tims.runs[0]
    cid = int(_accepted(tims, 1)[0])
    trees = {p: _page(tims, p, run, cid) for p in PAGES}
    pages = {p: _json(t) for p, t in trees.items()}
    for i in IM_IDS:
        assert any(i in s for s in pages.values()), i
    assert "ion mobility" in pages["overview"]
    assert "1/K0" in pages["qc"] and "1/K0" in pages["runs"]
    assert pages["calibration"].index("cal-err-card") < pages["calibration"].index(
        "cal-im-err-card"
    )
    assert not _im_block_hidden(trees["spectra"])


@pytest.mark.parametrize("name", THREE_D)
def test_three_d_pages_show_no_ion_mobility(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    s = (
        _table(rs, "psms_scored", ["candidate_id", "label"])
        if run.artifact("psms_scored")
        else None
    )
    cid = int(s[s.label == "target"].candidate_id.iloc[0]) if s is not None else 0
    for page in PAGES:
        tree = _page(rs, page, run, cid)
        text = _json(tree)
        assert not any(i in text for i in IM_IDS), page
        if page == "spectra":  # the 1/K0 controls exist for the callback, hidden
            assert _im_block_hidden(tree)
        else:
            assert "1/K0" not in text, page


@pytest.mark.real_data
def test_real_tims_run(real_tims):
    from mumdia_viewer.data import open_results

    rs = open_results(real_tims)
    run = rs.runs[0]
    assert run.has_im
    cal = cal_data.cal_record(rs, run.name)
    assert cal.has_im and cal.im_per_charge
    lo, hi = sp.im_bounds(rs, run)
    assert 0.4 < lo < hi < 1.8
    e = cal_data.accepted_errors(rs, run.name, 0.01)
    assert np.isfinite(e.frame["im_error"]).sum() > 1000
