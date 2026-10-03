"""The app shell and the overview page, built on the fixtures without a browser."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data.counts import unit_counts
from mumdia_viewer.ui import figures, overview
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import (
    THRESHOLD_STOPS,
    PageContext,
    href,
    page_of,
    parse_threshold,
    query_of,
    threshold_options,
)

FIXTURES = ["single", "grouped", "experiment", "topk", "mbr", "ovl_bp"]


def _walk(node):
    if isinstance(node, Component):
        yield node
        children = getattr(node, "children", None)
        yield from _walk(children)
        for prop in ("label", "leftSection", "rightSection", "right", "title"):
            value = getattr(node, prop, None)
            if isinstance(value, Component | list):
                yield from _walk(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)


def _ids(tree) -> list:
    return [json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)]


@pytest.mark.parametrize("name", FIXTURES)
def test_overview_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    tree = overview.layout(PageContext(rs=rs, base="/"))
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for unit in overview.CARD_UNITS:
        assert json.dumps(f"kpi-n-{unit}") in ids
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    assert graphs and all(isinstance(g.id, dict) and g.id["type"] == "fig" for g in graphs)


@pytest.mark.parametrize("name", ["single", "experiment", "mbr"])
def test_slider_counts_are_the_unit_counts_at_each_stop(open_fixture, name):
    rs = open_fixture(name)
    data = overview._slider_data(rs, overview._curves(rs))
    assert data["stops"] == list(THRESHOLD_STOPS)
    for i, t in enumerate(THRESHOLD_STOPS):
        counts = {c.unit: c for c in unit_counts(rs, t)}
        for unit in overview.CARD_UNITS:
            assert data["targets"][unit][i] == counts[unit].n_target, (unit, t)
            assert data["decoys"][unit][i] == counts[unit].n_decoy, (unit, t)


def test_curve_grid_holds_every_stop():
    for t in THRESHOLD_STOPS:
        assert t in overview.CURVE_GRID
    assert list(overview.CURVE_GRID) == sorted(set(overview.CURVE_GRID))


def test_threshold_sections_follow_the_threshold(open_fixture):
    rs = open_fixture("experiment")
    for t in (0.001, 0.05):
        parts = overview._threshold_sections(PageContext(rs=rs, base="/", threshold=t))
        text = json.dumps([p.to_plotly_json() for p in parts], default=str)
        assert f"run_psm_q \\u2264 {t:g}" in text or f"run_psm_q ≤ {t:g}" in text


def test_figures_take_the_scheme(open_fixture):
    rs = open_fixture("single")
    curves = overview._curves(rs)
    for scheme in ("light", "dark"):
        fig = figures.id_curves_figure(curves, 0.01, {}, scheme)
        assert isinstance(fig, go.Figure)
        font = fig.layout.template.layout.font.color
        assert font == ("#c1c2c5" if scheme == "dark" else "#343a40")
        names = [s.name for s in fig.layout.shapes]
        assert names == ["threshold"]
    empty = figures.score_histogram_figure(None, "dark")
    assert empty.layout.annotations[0].text == "no scored rows"


def test_state_helpers():
    assert [parse_threshold(o["value"]) for o in threshold_options()] == list(THRESHOLD_STOPS)
    assert parse_threshold("abc") == 0.01 and parse_threshold(1.0) == 0.01
    assert page_of("/tok/identifications", "/tok/") == "identifications"
    assert page_of("/tok/", "/tok/") == "overview" and page_of("/tok/nope", "/tok/") == "overview"
    assert query_of("?run=r1&cid=5") == {"run": "r1", "cid": "5"}
    assert href("/t/", "precursor", {"run": "", "cid": 3}) == "/t/precursor?cid=3"
    assert href("/t/", "overview") == "/t/"


def test_two_apps_in_one_process_keep_their_own_callbacks(open_fixture):
    """Callbacks registered on the app, not in Dash's global list (which the first app to
    serve a request would take whole)."""
    first = create_app(open_fixture("single"), url_base="/a/")
    second = create_app(open_fixture("experiment"), url_base="/b/")
    a = first.server.test_client().get("/a/_dash-dependencies").get_json()
    b = second.server.test_client().get("/b/_dash-dependencies").get_json()

    def clientside(deps):
        return sorted(d["output"] for d in deps if d.get("clientside_function"))

    assert clientside(a) and clientside(a) == clientside(b)
    assert len(a) == len(b)


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_app_builds_and_serves_the_shell(open_fixture, name):
    rs = open_fixture(name)
    app = create_app(rs, url_base="/abc/")
    client = app.server.test_client()
    assert client.get("/abc/").status_code == 200
    layout = client.get("/abc/_dash-layout").get_json()
    assert "provider" in json.dumps(layout)
    deps = client.get("/abc/_dash-dependencies").get_json()
    outputs = " ".join(d["output"] for d in deps)
    assert "page.children" in outputs and "threshold.data" in outputs


def test_compare_serves_both_viewers_on_one_server(open_fixture):
    from mumdia_viewer.ui.app import create_compare_apps

    a, b = create_compare_apps(open_fixture("single"), open_fixture("experiment"), url_base="/c/")
    assert a.server is b.server
    client = a.server.test_client()
    assert client.get("/c/").status_code == 200 and client.get("/c/b/").status_code == 200
    assert a.mv_compare() is open_fixture("experiment")
    assert b.mv_compare() is open_fixture("single")
    assert a.mv_compare_base() == "/c/b/" and b.mv_compare_base() == "/c/"
    deps_a = client.get("/c/_dash-dependencies").get_json()
    deps_b = client.get("/c/b/_dash-dependencies").get_json()
    assert len(deps_a) == len(deps_b) > 0
