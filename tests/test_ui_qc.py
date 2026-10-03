"""The run QC page, built on the fixtures without a browser, and its callbacks through
the server (posted to /_dash-update-component as the browser posts them)."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import qc as qcd
from mumdia_viewer.ui import qc, qc_cards, qc_figures
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "grouped", "experiment", "topk", "mbr", "chrom_v1"]
RT = json.dumps(qc.RT_FIG, separators=(",", ":"), sort_keys=True)


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


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_page_builds_with_unique_ids(open_fixture, name, scheme):
    rs = open_fixture(name)
    tree = qc.layout(PageContext(rs=rs, base="/", scheme=scheme))
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for needed in ("qc-key", "qc-scan", "qc-bin", "qc-level", "qc-bin-grid", "qc-dist"):
        assert json.dumps(needed) in ids
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    assert graphs and all(isinstance(g.id, dict) and g.id["type"] == "fig" for g in graphs)
    colour = "#c1c2c5" if scheme == "dark" else "#343a40"
    for g in graphs:
        assert g.figure.layout.template.layout.font.color == colour


def test_page_states_its_rules(open_fixture):
    text = _text(qc.layout(PageContext(rs=open_fixture("single"), base="/")))
    assert "run_psm_q" in text and "viewer" in text
    assert "not followed by P" in text  # the missed-cleavage rule
    assert "top_peaks_ms2" in text or "Reading the peak lists" in text


def test_page_without_spectra_says_so(open_fixture):
    text = _text(qc.layout(PageContext(rs=open_fixture("chrom_v1"), base="/")))
    assert "spectra_ms2" in text and "cannot be read" in text


def test_experiment_runs(open_fixture):
    rs = open_fixture("experiment")
    last = rs.runs[-1]
    run, notice = qc.run_of(rs, {"run": last.name})
    assert run is last and notice is None
    run, notice = qc.run_of(rs, {"run": "nope"})
    assert run is rs.runs[0] and "nope" in notice
    tree = qc.layout(PageContext(rs=rs, base="/", query={"run": last.name}))
    key = next(c for c in _walk(tree) if getattr(c, "id", None) == "qc-key")
    assert key.data["run"] == last.name
    assert f"run {last.label}".lower() in _text(tree).lower()
    assert json.dumps("qc-run") in _ids(tree)
    single = qc.run_of(open_fixture("single"), {"run": "x"})
    assert single == (open_fixture("single").runs[0], None)


def test_default_selections(open_fixture):
    rs = open_fixture("single")
    scan = qc.default_scan(rs, rs.runs[0], 1)
    sig = qcd.scan_signals(rs, None, 1)
    assert scan == {"level": 1, "row": int(np.argmax(sig.tic))}
    ids = qc.ids_frame(rs, rs.runs[0], 0.01)
    sel = qc.default_bin(ids)
    assert ids["targets"].iloc[sel["index"]] == ids["targets"].max()


def test_rt_figure_traces_and_plain_customdata(open_fixture):
    rs = open_fixture("single")
    fig, note = qc.rt_figure_for(
        rs,
        rs.runs[0],
        level=1,
        window=None,
        x_range=None,
        t=0.01,
        scan=None,
        sel=None,
        scheme="dark",
    )
    assert isinstance(fig, go.Figure) and "every scan" in note
    names = [t.name for t in fig.data]
    assert names[qc_figures.TRACE["tic"]] == "MS1 TIC"
    assert names[qc_figures.TRACE["targets"]] == "accepted target PSMs"
    # Plain lists: dcc.Graph drops typed arrays from its click data.
    data = fig.to_plotly_json()["data"]
    for i in (qc_figures.TRACE["tic"], qc_figures.TRACE["bp"], qc_figures.TRACE["targets"]):
        assert isinstance(data[i]["customdata"], list | tuple)
        assert isinstance(data[i]["customdata"][0], list | tuple)
    shapes = qc_figures.selection_shapes(10.0, (2.0, 4.0))
    assert [s["name"] for s in shapes] == ["qc-bin", "qc-scan"]


def test_ms2_view_draws_an_envelope_above_the_limit(open_fixture, monkeypatch):
    rs = open_fixture("single")
    monkeypatch.setattr(qc, "ENVELOPE_ABOVE", 50)
    monkeypatch.setattr(qc, "ENVELOPE_BINS", 20)
    tic, _bp, n_view, envelope = qc.series(rs, rs.runs[0], 2, None, None)
    assert envelope and n_view == qcd.scan_signals(rs, None, 2).n
    assert tic.rows.size < n_view and np.all(np.diff(tic.rt) >= 0)
    one = int(qcd.window_scheme(rs, None).frame["window_id"].iloc[0])
    tic, _, n_view, envelope = qc.series(rs, rs.runs[0], 2, one, None)
    assert not envelope and tic.rows.size == n_view


def test_bin_records_and_links(open_fixture):
    rs = open_fixture("experiment")
    run = rs.runs[1]
    df = qcd.ids_in_rt_range(rs, run, 0.05, 0.0, 1e9, closed=True, limit=5)
    rows = qc_cards.bin_records(df)
    assert [r["cid"] for r in rows] == df["candidate_id"].tolist()
    assert all(isinstance(r["_key"], str) and "_href" not in r for r in rows)
    link = qc_cards.precursor_link("/b/", run.name, rows[0]["cid"])
    assert link == f"/b/precursor?run={run.name}&cid={rows[0]['cid']}"
    assert qc_cards.precursor_link("/b/", "", 7) == "/b/precursor?cid=7"
    card = qc_cards.bin_card(
        base="/b/",
        run_name=run.name,
        lo=0.0,
        hi=1.0,
        total=len(rows),
        rows=rows,
        threshold=0.01,
        population="p",
        limit=10,
    )
    grid = next(c for c in _walk(card) if getattr(c, "id", None) == qc_cards.GRID_ID)
    assert grid.dashGridOptions["context"] == {"base": "/b/", "run": run.name}


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
        # A single-output callback takes the output itself, not a list.
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
    tree = qc.layout(PageContext(rs=rs, base="/"))
    stores = {c.id: c.data for c in _walk(tree) if type(c).__name__ == "Store"}
    return rs, app, app.server.test_client(), stores


def test_select_a_bar_and_a_scan(single_app):
    rs, app, client, stores = single_app
    key = _key(app, "qc-scan-title.children")
    ids = qc.ids_frame(rs, rs.runs[0], 0.01)
    k = int(np.argmin(np.where(ids["targets"] > 0, ids["targets"], 10**9)))
    x = float((ids["bin_lo"].iloc[k] + ids["bin_hi"].iloc[k]) / 2)
    state = [
        {"id": "qc-key", "property": "data", "value": stores["qc-key"]},
        {"id": "qc-view", "property": "data", "value": stores["qc-view"]},
        {"id": "qc-scan", "property": "data", "value": stores["qc-scan"]},
        {"id": "qc-bin", "property": "data", "value": stores["qc-bin"]},
        {"id": "threshold", "property": "data", "value": 0.01},
        {"id": "scheme", "property": "data", "value": "light"},
    ]
    click = {"points": [{"curveNumber": qc_figures.TRACE["targets"], "x": x}]}
    inputs = [
        {"id": qc.RT_FIG, "property": "clickData", "value": click},
        {"id": "scan-prev", "property": "n_clicks", "value": 0},
        {"id": "scan-next", "property": "n_clicks", "value": 0},
    ]
    out = _post(client, key, inputs, state)
    assert out["qc-bin"]["data"]["index"] == k
    assert len(out["qc-bin-grid"]["rowData"]) == int(ids["targets"].iloc[k])
    assert out["qc-bin-count"]["children"] == f"{int(ids['targets'].iloc[k]):,}"
    # A TIC point: its customdata names the row.
    row = 7
    click = {"points": [{"curveNumber": qc_figures.TRACE["tic"], "customdata": [row, 0, 0]}]}
    inputs[0]["value"] = click
    out = _post(client, key, inputs, state)
    assert out["qc-scan"]["data"] == {"level": 1, "row": row}
    # The next scan.
    state[2]["value"] = {"level": 1, "row": row}
    inputs[2]["value"] = 1
    out = _post(client, key, inputs, state, triggered=[inputs[2]])
    assert out["qc-scan"]["data"] == {"level": 1, "row": row + 1}
    # MS2: the next scan of the same isolation window.
    st = qcd.ScanTable.for_run(rs, rs.runs[0])
    state[2]["value"] = {"level": 2, "row": 0}
    out = _post(client, key, inputs, state, triggered=[inputs[2]])
    nxt = out["qc-scan"]["data"]["row"]
    assert st.window_id[nxt] == st.window_id[0] and st.rt[nxt] > st.rt[0]


def test_threshold_and_views_answer(single_app):
    rs, app, client, stores = single_app
    key = _key(app, "qc-fact-accepted.children")
    out = _post(
        client,
        key,
        [{"id": "threshold", "property": "data", "value": 0.05}],
        [
            {"id": "qc-key", "property": "data", "value": stores["qc-key"]},
            {"id": "qc-bin", "property": "data", "value": stores["qc-bin"]},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
    )
    total = qcd.ids_across_rt(rs, None, 0.05, qc.edges_of(rs, rs.runs[0])).attrs["total"]
    assert out["qc-fact-accepted"]["children"] == f"{total:,}"
    assert "0.05" in out["qc-fact-accepted-q"]["children"]
    view = _key(app, "qc-window-box.style")
    out = _post(
        client,
        view,
        [
            {"id": "qc-level", "property": "value", "value": "2"},
            {"id": "qc-window", "property": "value", "value": "all"},
        ],
        [
            {"id": "qc-key", "property": "data", "value": stores["qc-key"]},
            {"id": "qc-xrange", "property": "data", "value": None},
            {"id": "qc-scan", "property": "data", "value": stores["qc-scan"]},
            {"id": "qc-bin", "property": "data", "value": stores["qc-bin"]},
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "scheme", "property": "data", "value": "light"},
        ],
    )
    assert out["qc-view"]["data"] == {"level": 2, "window": None}
    assert out[RT]["figure"]["data"][qc_figures.TRACE["tic"]]["name"] == "MS2 TIC"
    assert out["qc-window-box"]["style"] == {}
    peaks = _key(app, "qc-peaks-slot.children")
    out = _post(
        client,
        peaks,
        [{"id": "qc-need", "property": "data", "value": {"run": ""}}],
        [{"id": "scheme", "property": "data", "value": "light"}],
    )
    assert "Peaks per MS2 spectrum" in json.dumps(out)


def test_two_apps_keep_their_qc_callbacks(open_fixture):
    a = create_app(open_fixture("single"), url_base="/a/")
    b = create_app(open_fixture("experiment"), url_base="/b/")
    for app in (a, b):
        keys = " ".join(app.callback_map)
        assert "qc-scan-title.children" in keys and "qc-peaks-slot.children" in keys
