"""The spectrum browser, built on the fixtures without a browser, its pure helpers, and
its callbacks through the server (posted to /_dash-update-component as the browser posts
them)."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import scans as sc
from mumdia_viewer.data.spectra import ScanTable
from mumdia_viewer.ui import spectra, spectra_cards, spectra_figures
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "grouped", "experiment", "topk", "mbr", "chrom_v1"]
SPEC = json.dumps(spectra.SPEC_FIG, separators=(",", ":"), sort_keys=True)
NAV = json.dumps(spectra.NAV_FIG, separators=(",", ":"), sort_keys=True)


def _walk(node):
    if isinstance(node, Component):
        yield node
        yield from _walk(getattr(node, "children", None))
        for prop in ("label", "leftSection", "rightSection", "right", "title", "custom_spinner"):
            value = getattr(node, prop, None)
            if isinstance(value, Component | list):
                yield from _walk(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)


def _ids(tree) -> list[str]:
    return [json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)]


def _text(tree) -> str:
    return json.dumps(tree.to_plotly_json(), default=str)


def _stores(tree) -> dict:
    return {c.id: c.data for c in _walk(tree) if type(c).__name__ == "Store"}


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_page_builds_with_unique_ids(open_fixture, name, scheme):
    rs = open_fixture(name)
    tree = spectra.layout(PageContext(rs=rs, base="/", scheme=scheme))
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    colour = "#c1c2c5" if scheme == "dark" else "#343a40"
    for g in graphs:
        assert isinstance(g.id, dict) and g.id["type"] == "fig"
        assert g.figure.layout.template.layout.font.color == colour
    if name == "chrom_v1":
        assert "cannot be read" in _text(tree)
        return
    for needed in (
        "sp-key",
        "sp-scan",
        "sp-cid",
        "sp-step",
        "sp-goto",
        "scan-prev",
        "scan-next",
        "sp-grid",
        "sp-slider",
        "sp-window",
        "sp-level",
    ):
        assert json.dumps(needed) in ids, needed
    assert len(graphs) == 2
    assert ("sp-run" in " ".join(ids)) == rs.is_experiment


def test_page_states_its_rules(open_fixture):
    text = _text(spectra.layout(PageContext(rs=open_fixture("single"), base="/")))
    assert "run_psm_q" in text
    assert "isolation window" in text and "lower <= precursor_mz <= upper" in text
    assert "viewer match" in text and "derived" in text
    assert "psms_extracted" in text  # where the precursor m/z come from


def test_default_view_selects_the_best_target(open_fixture):
    rs = open_fixture("single")
    tree = spectra.layout(PageContext(rs=rs, base="/"))
    stores = _stores(tree)
    ref, cid = sc.default_scan(rs, None, 0.01)
    assert stores["sp-scan"]["row"] == ref.row and stores["sp-scan"]["level"] == 2
    assert stores["sp-cid"] == cid
    grid = next(c for c in _walk(tree) if getattr(c, "id", None) == "sp-grid")
    keys = [r["cid"] for r in grid.rowData]
    assert cid in keys
    selected = [r["cid"] for r in grid.rowData if r["_sel"]]
    assert selected == [cid]
    # Fragment matches of every listed candidate (MS2).
    assert all("matched" in r and r["n_lib"] >= r["matched"] for r in grid.rowData)


def test_address_entries(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    si = int(st.scan_index[40])
    ref, cid, notice = spectra.start_of(rs, run, {"scan": str(si), "cid": "7"}, 0.01)
    assert ref.scan_index == si and cid == 7 and notice is None
    ref, _, notice = spectra.start_of(rs, run, {"scan": "999999999"}, 0.01)
    assert "999999999" in notice and ref.level == 2
    w = int(st.windows["window_id"].iloc[2])
    ref, _, notice = spectra.start_of(rs, run, {"rt": "50", "window": str(w)}, 0.01)
    assert ref.window_id == w and notice is None
    assert ref.row == sc.nearest_scan(rs, run, 50.0, level=2, window_id=w).row
    ref, _, _ = spectra.start_of(rs, run, {"rt": "50", "level": "1"}, 0.01)
    assert ref.level == 1
    ref, _, notice = spectra.start_of(rs, run, {"rt": "50", "window": "12345"}, 0.01)
    assert "12345" in notice and ref.level == 2
    text = _text(spectra.layout(PageContext(rs=rs, base="/", query={"scan": "999999999"})))
    assert "No scan with scan_index" in text


def test_experiment_run_choice(open_fixture):
    rs = open_fixture("experiment")
    last = rs.runs[-1]
    run, notice = spectra.run_of(rs, {"run": last.name})
    assert run is last and notice is None
    run, notice = spectra.run_of(rs, {"run": "nope"})
    assert run is rs.runs[0] and "nope" in notice
    tree = spectra.layout(PageContext(rs=rs, base="/", query={"run": last.name}))
    assert _stores(tree)["sp-key"]["run"] == last.name


# --------------------------------------------------------------------------- helpers


def test_parse_near_and_edges():
    assert spectra_cards.parse_near("apex:10") == ("apex", 10.0)
    assert spectra_cards.parse_near("elution") == ("elution", 0.0)
    assert spectra_cards.parse_near("junk") == ("apex", 5.0)
    assert spectra_cards.parse_near("apex:-1") == ("apex", 5.0)
    edges = spectra.nav_edges(1620.0)
    assert edges[0] == 0 and edges[-1] >= 1620 and np.allclose(np.diff(edges), 20.0)
    assert spectra.nav_edges(120.0)[1] == 1.0


def test_navigate_to_every_input(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    st = ScanTable.for_run(rs, run)
    cur = sc.scan_ref(rs, run, 2, st.n // 2)
    nxt = spectra.navigate_to(rs, run, cur, "sp-step", step={"n": 1}, scope="window")
    assert nxt == sc.step(rs, run, cur, 1)
    nxt = spectra.navigate_to(rs, run, cur, "sp-step", step={"n": -1}, scope="run")
    assert nxt == sc.step(rs, run, cur, -1, scope="run")
    assert spectra.navigate_to(rs, run, cur, "sp-step", step=None) is None
    got = spectra.navigate_to(rs, run, cur, "sp-scan-input", scan_value=int(st.scan_index[3]))
    assert got.row == 3
    assert spectra.navigate_to(rs, run, cur, "sp-scan-input", scan_value=10**9) is None
    got = spectra.navigate_to(rs, run, cur, "sp-rt-input", rt_value=12.5)
    assert got == sc.nearest_scan(rs, run, 12.5, level=2, window_id=cur.window_id)
    got = spectra.navigate_to(rs, run, cur, "sp-slider", slider_value=80)
    assert got == sc.nearest_scan(rs, run, 80.0, level=2, window_id=cur.window_id)
    got = spectra.navigate_to(rs, run, cur, "sp-level", level_value="1")
    assert got.level == 1 and got == sc.nearest_scan(rs, run, cur.rt, level=1)
    other = int(st.windows["window_id"].iloc[0])
    got = spectra.navigate_to(rs, run, cur, "sp-window", window_value=str(other))
    assert got.window_id == other
    click = {"points": [{"x": 30.0}]}
    got = spectra.navigate_to(rs, run, cur, spectra.NAV_FIG, click=click)
    assert got == sc.nearest_scan(rs, run, 30.0, level=2, window_id=cur.window_id)
    got = spectra.navigate_to(rs, run, cur, "sp-goto", goto={"level": 1, "row": 2, "ts": 1})
    assert (got.level, got.row) == (1, 2)
    assert spectra.navigate_to(rs, run, cur, "sp-goto", goto={"level": 1, "row": 10**7}) is None
    assert spectra.navigate_to(rs, run, cur, "unknown") is None


def test_view_of_an_ms1_scan_shows_isotopes(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    ref2, cid = sc.default_scan(rs, run, 0.01)
    ref1 = sc.nearest_scan(rs, run, ref2.rt, level=1)
    v = spectra.view(
        rs,
        run,
        ref1,
        t=0.01,
        near="apex:5",
        decoys=False,
        cid=cid,
        n_labels=10,
        scale="window",
        scheme="light",
        window_hint=ref2.window_id,
    )
    assert v["cid"] == cid
    assert v["frag_title"] == "Precursor isotopes"
    assert "sum_near" in json.dumps(v["fig"].to_plotly_json()) or "isotope" in _text(
        v["frag_body"][1]
    )
    kinds = [t.meta.get("kind") for t in v["fig"].data if t.meta]
    assert "isotope-mark" in kinds
    # The MS2 scan of the remembered window, as a button of the scan card.
    assert f"MS2 in window {ref2.window_id}" in json.dumps(
        [c.to_plotly_json() if isinstance(c, Component) else c for c in v["details"]], default=str
    )


# --------------------------------------------------------------------------- figures


def test_spectrum_figure_traces_and_labels(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    ref, cid = sc.default_scan(rs, run, 0.01)
    spec = ScanTable.for_run(rs, run).spectrum(ref.row)
    fig = spectra_figures.spectrum_figure(spec, "light", n_labels=5)
    kinds = [t.meta["kind"] for t in fig.data]
    assert kinds[:3] == ["stems", "peaks", "labels"]
    labels = fig.data[2]
    assert len(labels.x) == min(5, spec.n_peaks)
    # The labels are the most intense peaks (none of them closer than 1.5 % of the range).
    want = sc.top_peaks(
        spec.mz, spec.intensity, 5, lo=fig.layout.xaxis.range[0], hi=fig.layout.xaxis.range[1]
    )
    assert sorted(round(float(spec.mz[i]), 4) for i in want) == sorted(
        round(float(x), 4) for x in labels.x
    )
    # Plain lists on the peak trace (spectra.js reads them).
    assert isinstance(fig.to_plotly_json()["data"][1]["x"], list)
    ov = sc.fragment_overlay(rs, run, cid, spec)
    from mumdia_viewer.ui.detail_view import fragments_of

    frags = fragments_of(ov.chromatogram)
    fig = spectra_figures.spectrum_figure(spec, "dark", overlay=ov, frags=frags, scale="matched")
    kinds = [t.meta["kind"] for t in fig.data]
    assert kinds.count("library") == ov.n_library
    assert kinds.count("matched") == ov.n_matched == kinds.count("matched-label")
    assert fig.layout.yaxis.range[0] < 0  # the library is mirrored below
    empty = spectra_figures.spectrum_figure(None, "light")
    assert isinstance(empty, go.Figure) and not empty.data


def test_window_scale_leaves_out_the_isolation_window(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    ref, _ = sc.default_scan(rs, run, 0.01)
    spec = ScanTable.for_run(rs, run).spectrum(ref.row)
    fig = spectra_figures.spectrum_figure(spec, "light", scale="window")
    mz = np.asarray(spec.mz, dtype=float)
    it = np.asarray(spec.intensity, dtype=float)
    outside = (mz < spec.window_lower) | (mz > spec.window_upper)
    rel = np.asarray([c[1] for c in fig.data[1].customdata])
    k = int(np.argmax(np.where(outside, it, -1)))
    assert rel[k] == pytest.approx(100.0, abs=0.01)


def test_nav_figure(open_fixture):
    edges = np.arange(0, 101, 10.0)
    counts = np.arange(10)
    fig = spectra_figures.nav_figure(edges, counts, rt=42.0, band=(37.0, 47.0), rt_max=100.0)
    assert list(fig.layout.xaxis.range) == [0.0, 100.0]
    assert [t.meta["kind"] for t in fig.data] == ["apexes", "target"]
    assert fig.layout.shapes[-1].x0 == 42.0


# --------------------------------------------------------------------------- callbacks


def _key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _prop_id(i) -> str:
    cid = i["id"]
    if isinstance(cid, dict):
        cid = json.dumps(cid, separators=(",", ":"), sort_keys=True)
    return f"{cid}.{i['property']}"


def _post(client, key, inputs, state, triggered=None):
    outputs = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        outputs.append({"id": json.loads(cid) if cid.startswith("{") else cid, "property": prop})
    body = {
        "output": key,
        "outputs": outputs if "..." in key else outputs[0],
        "inputs": inputs,
        "changedPropIds": [_prop_id(i) for i in (triggered or [inputs[0]])],
        "state": state,
    }
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code in (200, 204), resp.data[:800]
    return resp.get_json()["response"] if resp.status_code == 200 else {}


@pytest.fixture
def single_app(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    tree = spectra.layout(PageContext(rs=rs, base="/"))
    return rs, app, app.server.test_client(), _stores(tree)


def test_navigate_callback_steps(single_app):
    rs, app, client, stores = single_app
    key = _key(app, "sp-scan.data...sp-scan-input.value")
    cur = stores["sp-scan"]
    inputs = [
        {"id": "sp-step", "property": "data", "value": {"n": 1, "ts": 1}},
        {"id": "sp-scan-input", "property": "value", "value": cur["scan_index"]},
        {"id": "sp-rt-input", "property": "value", "value": cur["rt"]},
        {"id": "sp-level", "property": "value", "value": "2"},
        {"id": "sp-window", "property": "value", "value": str(cur["window"])},
        {"id": "sp-slider", "property": "value", "value": cur["rt"]},
        {"id": spectra.NAV_FIG, "property": "clickData", "value": None},
        {"id": "sp-goto", "property": "data", "value": None},
    ]
    state = [
        {"id": "sp-key", "property": "data", "value": stores["sp-key"]},
        {"id": "sp-scan", "property": "data", "value": cur},
        {"id": "sp-scope", "property": "value", "value": "window"},
    ]
    out = _post(client, key, inputs, state)
    want = sc.step(rs, None, sc.scan_ref(rs, None, 2, cur["row"]), 1)
    assert out["sp-scan"]["data"]["row"] == want.row
    assert out["sp-scan-input"]["value"] == want.scan_index
    # An unknown scan_index puts the box back to the shown scan.
    inputs[1]["value"] = 10**9
    out = _post(client, key, inputs, state, triggered=[inputs[1]])
    assert "sp-scan" not in out and out["sp-scan-input"]["value"] == cur["scan_index"]
    # MS1: the window control keeps its window, and the store remembers it.
    inputs[3]["value"] = "1"
    out = _post(client, key, inputs, state, triggered=[inputs[3]])
    assert out["sp-scan"]["data"]["level"] == 1
    assert out["sp-scan"]["data"]["window"] == cur["window"]


def test_show_callback_follows_threshold_and_selection(single_app):
    rs, app, client, stores = single_app
    key = _key(app, "sp-title.children")
    cur = stores["sp-scan"]
    inputs = [
        {"id": "sp-scan", "property": "data", "value": cur},
        {"id": "threshold", "property": "data", "value": 0.05},
        {"id": "sp-near", "property": "value", "value": "apex:20"},
        {"id": "sp-decoys", "property": "checked", "value": True},
        {"id": "sp-cid", "property": "data", "value": stores["sp-cid"]},
        {"id": "sp-labels", "property": "value", "value": "25"},
        {"id": "sp-scale", "property": "value", "value": "base"},
    ]
    state = [
        {"id": "sp-key", "property": "data", "value": stores["sp-key"]},
        {"id": "scheme", "property": "data", "value": "dark"},
    ]
    out = _post(client, key, inputs, state, triggered=[inputs[1]])
    ref = sc.scan_ref(rs, None, cur["level"], cur["row"])
    want = sc.candidates_near(rs, None, ref, 0.05, delta=20, include_decoys=True)
    rows = out["sp-grid"]["rowData"]
    assert [r["cid"] for r in rows] == want["candidate_id"].tolist()
    assert out["sp-cand-count"]["children"] == f"{len(want):,}"
    assert out[SPEC]["figure"]["layout"]["template"]["layout"]["font"]["color"] == "#c1c2c5"
    assert out[NAV]["figure"]["data"]
    assert "0.05" in json.dumps(out["sp-grid"]["columnDefs"])
    # Another candidate: its fragments.
    if len(rows) > 1:
        other = rows[1]["cid"]
        inputs[4]["value"] = other
        out = _post(client, key, inputs, state, triggered=[inputs[4]])
        sel = [r["cid"] for r in out["sp-grid"]["rowData"] if r["_sel"]]
        assert sel == [other]
        assert "sp-cid" not in out  # the selection stands
        assert out["sp-grid"]["scrollTo"]["rowId"] == str(other)


def test_two_apps_keep_their_spectra_callbacks(open_fixture):
    a = create_app(open_fixture("single"), url_base="/a/")
    b = create_app(open_fixture("experiment"), url_base="/b/")
    for app in (a, b):
        keys = " ".join(app.callback_map)
        assert "sp-title.children" in keys and "sp-scan.data" in keys
