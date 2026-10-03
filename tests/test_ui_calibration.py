"""The calibration page, built on the fixtures without a browser, and its callbacks
answered through the server."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import calibration as data
from mumdia_viewer.ui import calibration, calibration_cards, calibration_figures
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "grouped", "experiment", "topk", "mbr", "ovl_bp"]


def _walk(node):
    if isinstance(node, Component):
        yield node
        yield from _walk(getattr(node, "children", None))
        for prop in ("label", "leftSection", "rightSection", "right", "title"):
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
def test_page_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    for query in ({}, {"run": rs.runs[-1].name}):
        tree = calibration.layout(PageContext(rs=rs, base="/", query=query))
        ids = _ids(tree)
        assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
        for needed in ("cal-key", "cal-err-count", "cal-mass-count", "cal-pt-fit", "cal-pt-err"):
            assert json.dumps(needed) in ids
        graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
        names = {g.id["name"] for g in graphs}
        assert {"cal-fit", "cal-resid", "cal-err", "cal-offset", "cal-mass"} <= names


def test_experiment_page_has_a_run_selector_and_follows_the_address(open_fixture):
    rs = open_fixture("experiment")
    ctx = PageContext(rs=rs, base="/", query={"run": "b"})
    assert calibration.target(ctx) == ("b", None)
    tree = calibration.layout(ctx)
    sel = [c for c in _walk(tree) if getattr(c, "id", None) == "cal-run"]
    assert sel and sel[0].value == "b"
    key = next(c for c in _walk(tree) if getattr(c, "id", None) == "cal-key")
    assert key.data == {"run": "b", "band": None}
    assert "run b" in _text(tree)
    unknown = PageContext(rs=rs, base="/", query={"run": "zz"})
    assert calibration.target(unknown) == ("a", None)
    assert calibration.calibration_href("/t/", "b") == "/t/calibration?run=b"
    single = calibration.layout(PageContext(rs=open_fixture("single"), base="/"))
    assert json.dumps("cal-run") not in _ids(single)


def test_grouped_page_offers_its_bands(open_fixture):
    rs = open_fixture("ovl_bp")
    tree = calibration.layout(PageContext(rs=rs, base="/", query={"band": "g01"}))
    sel = [c for c in _walk(tree) if getattr(c, "id", None) == "cal-band"]
    assert sel and sel[0].value == "g01"
    assert [d["value"] for d in sel[0].data] == ["g00", "g01", "g02"]
    assert "band g01" in _text(tree)


def test_the_page_says_the_residuals_are_in_sample_and_the_rebuild_agrees(open_fixture):
    text = _text(calibration.layout(PageContext(rs=open_fixture("single"), base="/")))
    assert "not an error estimate" in text
    assert "rebuild equals cal.json" in text
    assert "viewer" in text  # the derived numbers are marked


def test_figures_take_the_scheme_and_name_their_lines(open_fixture):
    rs = open_fixture("single")
    a = data.rt_anchors(rs, rs.runs[0])
    e = data.accepted_errors(rs, rs.runs[0], 0.01)
    m = data.masscal_record(rs, rs.runs[0])
    for scheme in ("light", "dark"):
        figs = [
            calibration_figures.fit_figure(a, scheme),
            calibration_figures.residual_figure(a, data.cal_record(rs, rs.runs[0]), scheme),
            calibration_figures.error_figure(e, scheme),
            calibration_figures.error_figure(e, scheme, view="fraction"),
            calibration_figures.offset_figure(m, None, scheme),
            calibration_figures.mass_figure(e, m, scheme),
            calibration_figures.mass_figure(e, m, scheme, view="gradient"),
        ]
        for fig in figs:
            assert isinstance(fig, go.Figure)
            assert fig.layout.template.layout.font.color == (
                "#c1c2c5" if scheme == "dark" else "#343a40"
            )
    fit = calibration_figures.fit_figure(a)
    assert {"rt_axis", "home", "curve"} <= set(fit.layout.meta)
    assert {s.name for s in calibration_figures.error_figure(e).layout.shapes} >= {
        "zero",
        "edge-hi",
        "edge-lo",
    }


def test_every_anchor_is_drawn_once_per_panel(open_fixture):
    rs = open_fixture("topk")
    a = data.rt_anchors(rs, rs.runs[0])
    traces = calibration_figures.anchor_traces(a)
    rows = np.concatenate(list(traces.values()))
    # each anchor appears in the top panel and in the residual panel
    assert sorted(rows.tolist()) == sorted(list(range(a.n)) * 2)


def test_identification_points_map_back_to_their_rows(open_fixture):
    rs = open_fixture("single")
    e = data.accepted_errors(rs, rs.runs[0], 0.01)
    fig = calibration_figures.error_figure(e)
    rows = calibration_figures.id_rows(e, "rt_error")
    assert len(fig.data[0].x) == rows.size
    k = rows.size // 2
    event = {"points": [{"curveNumber": 0, "pointIndex": k}]}
    assert calibration.id_row(event, e, "rt_error") == rows[k]
    assert float(fig.data[0].x[k]) == pytest.approx(e.frame["apex_rt"].iloc[rows[k]], abs=1e-3)
    assert (
        calibration.id_row({"points": [{"curveNumber": 1, "pointIndex": 0}]}, e, "rt_error") is None
    )


def test_empty_states_explain_themselves(open_fixture):
    rs = open_fixture("grouped")
    s = data.scopes(rs, rs.runs[0])[0]
    a = data.rt_anchors(rs, s.run, s.key)
    fig = calibration_figures.fit_figure(a)
    assert "No anchors" in fig.layout.annotations[0].text
    e = data.accepted_errors(rs, s.run, 0.01, s.key)
    assert "no RT calibration" in calibration_figures.error_figure(e).layout.annotations[0].text


def test_point_bars_name_the_point(open_fixture):
    rs = open_fixture("single")
    a = data.rt_anchors(rs, rs.runs[0])
    row = int(np.flatnonzero(a.frame["scored"].to_numpy())[0])
    bar = _text(calibration_cards.anchor_bar(rs, "/", a, row, 0.01))
    assert "precursor?cid=" in bar and "residual" in bar
    e = data.accepted_errors(rs, rs.runs[0], 0.01)
    bar = _text(calibration_cards.id_bar(rs, "/", e, 0, "frag_mass_err_median"))
    assert f"cid={int(e.frame['candidate_id'].iloc[0])}" in bar
    assert "Hover" in _text(calibration_cards.anchor_bar(rs, "/", a, None, 0.01))


# --------------------------------------------------------------------------- callbacks


def _key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _prop_id(value) -> str:
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _key_by_input(app, needle: str, prop: str) -> str:
    keys = [
        k
        for k, v in app.callback_map.items()
        if any(
            needle in json.dumps(i.get("id"), sort_keys=True) and i.get("property") == prop
            for i in v.get("inputs", [])
        )
    ]
    assert len(keys) == 1, keys
    return keys[0]


def _post(client, key, inputs, state, changed=None):
    outputs = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        prop = prop.split("@")[0]
        outputs.append({"id": json.loads(cid) if cid.startswith("{") else cid, "property": prop})
    body = {
        "output": key,
        "outputs": outputs if len(outputs) > 1 else outputs[0],
        "inputs": inputs,
        "changedPropIds": [
            f"{_prop_id(i['id'])}.{i['property']}"
            for i in inputs
            if changed is None or i["id"] == changed
        ],
        "state": state,
    }
    return client.post("/_dash-update-component", json=body)


FIT = {"name": "cal-fit", "type": "fig"}
ERR = {"name": "cal-err", "type": "fig"}
MASS = {"name": "cal-mass", "type": "fig"}


def test_hover_of_a_webgl_anchor_fills_the_point_bar(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    a = data.rt_anchors(rs, rs.runs[0])
    curve, rows = next(iter(calibration_figures.anchor_traces(a).items()))
    event = {"points": [{"curveNumber": curve, "pointIndex": 2}]}  # no customdata
    resp = _post(
        client,
        _key(app, "cal-pt-fit.children"),
        [{"id": FIT, "property": "hoverData", "value": event}],
        [
            {"id": "cal-key", "property": "data", "value": {"run": "", "band": None}},
            {"id": "threshold", "property": "data", "value": 0.01},
        ],
    )
    assert resp.status_code == 200, resp.data[:400]
    text = json.dumps(resp.get_json())
    assert str(a.frame["peptidoform"].iloc[int(rows[2])]).split("[")[0][:6] in text


def test_threshold_rebuilds_the_error_card(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    resp = _post(
        client,
        _key(app, "cal-err-count.children"),
        [
            {"id": "threshold", "property": "data", "value": 0.05},
            {"id": "cal-err-view", "property": "value", "value": "fraction"},
        ],
        [
            {"id": "cal-key", "property": "data", "value": {"run": "b", "band": None}},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
    )
    assert resp.status_code == 200, resp.data[:400]
    out = resp.get_json()["response"]
    e = data.accepted_errors(rs, "b", 0.05)
    assert out["cal-err-count"]["children"] == f"{e.n:,}"
    assert out["cal-err-q"]["children"] == "run_psm_q ≤ 0.05"
    assert out["cal-err-meta"]["data"] == {"t": 0.05, "view": "fraction"}


def test_click_opens_the_precursor_page(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    e = data.accepted_errors(rs, "a", 0.01)
    rows = calibration_figures.id_rows(e, "rt_error")
    event = {"points": [{"curveNumber": 0, "pointIndex": 0}]}
    resp = _post(
        client,
        _key_by_input(app, "cal-mass", "clickData"),
        [
            {"id": FIT, "property": "clickData", "value": None},
            {"id": ERR, "property": "clickData", "value": event},
            {"id": MASS, "property": "clickData", "value": None},
        ],
        [
            {"id": "cal-key", "property": "data", "value": {"run": "a", "band": None}},
            {"id": "cal-err-meta", "property": "data", "value": {"t": 0.01, "view": "seconds"}},
            {"id": "cal-mass-meta", "property": "data", "value": {}},
        ],
        changed=ERR,
    )
    assert resp.status_code == 200, resp.data[:400]
    cid = int(e.frame["candidate_id"].iloc[rows[0]])
    assert resp.get_json()["response"]["url"]["href"] == f"/precursor?run=a&cid={cid}"
