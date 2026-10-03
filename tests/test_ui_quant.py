"""The quant QC page, built on the fixtures without a browser."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import quantqc as Q
from mumdia_viewer.ui import quant, quant_cards, quant_figures
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "experiment", "mbr", "grouped", "topk"]
SHELL = {"threshold", "scheme", "url", "q-select", "templates", "recent", "page"}


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


def _ids(tree) -> list:
    return [json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)]


def _text(tree) -> str:
    return json.dumps(tree.to_plotly_json() if isinstance(tree, Component) else tree, default=str)


def _layout(rs, **kw):
    return quant.layout(PageContext(rs=rs, base="/", **kw))


@pytest.mark.parametrize("name", FIXTURES)
def test_layout_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    tree = _layout(rs)
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    stores = [c for c in _walk(tree) if getattr(c, "id", None) == "mv-conditions"]
    assert len(stores) == 1 and stores[0].storage_type == "local"
    assert getattr(stores[0], "data", None) is None  # the browser's value wins
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    assert graphs and all(isinstance(g.id, dict) and g.id["type"] == "fig" for g in graphs)
    text = _text(tree)
    assert "Quant QC" in text and "computed by the viewer" in text
    if rs.is_experiment:
        assert "LFQ matrix" in text and json.dumps("qq-heat-cluster") in ids
    if len(rs.runs) > 1:
        for run in rs.runs:
            assert json.dumps({"run": run.name, "type": "qq-cond"}, sort_keys=True) in ids
    else:
        assert "A CV needs at least two runs" in text


def _string_ids(deps: list[dict]) -> set[str]:
    return {d["id"] for d in deps if not d["id"].startswith("{")}


@pytest.mark.parametrize("name", FIXTURES)
def test_every_page_callback_finds_its_ids(open_fixture, name):
    """A callback with some inputs in the layout and others missing fails in the browser."""
    rs = open_fixture(name)
    app = create_app(rs, url_base="/")
    present = {json.loads(i) if i.startswith('"') else None for i in _ids(_layout(rs))} - {None}
    deps = app.server.test_client().get("/_dash-dependencies").get_json()
    for dep in deps:
        inputs = _string_ids(dep["inputs"])
        ours = {i for i in inputs if i.startswith("qq-") or i == "mv-conditions"}
        if not ours:
            continue
        ids = inputs | _string_ids(dep["state"])
        parts = [part.strip(".") for part in dep["output"].split("...")]
        outputs = {part.rsplit(".", 1)[0] for part in parts if part and not part.startswith("{")}
        missing = {i for i in ids | outputs if i not in present and i not in SHELL}
        assert not missing, (dep["output"], missing)


def test_answers_on_the_experiment(open_fixture):
    rs = open_fixture("mbr")
    stored = {"a": "X", "c": "X"}
    fig, sub, body, _ = quant.dist_answer(rs, stored, "lfq", "all", 0.01, "light")
    assert isinstance(fig, go.Figure) and "MaxLFQ" in sub and "viewer" in sub
    assert "qq-missing" in _text(body)
    fig, table, sub = quant.cv_answer(rs, stored, "lfq", "all", "all", 0.01, "dark")
    assert fig.layout.template.layout.font.color == "#c1c2c5"
    assert "n - 1" in sub and "every run" in sub
    m = Q.quant_matrix(rs, "precursor", "lfq")
    cv = Q.condition_cvs(m, stored)
    assert f"{cv.conditions[0].median:.1f}%" in _text(table)
    fig, count, sub = quant.heat_answer(rs, stored, "all", 0.01, "relative", True, None, "light")
    assert count == f"{Q.quant_matrix(rs, 'protein', 'lfq').n_keys:,} protein groups"
    assert "clustered" in sub
    group = Q.find_groups(rs, "")[0]
    fig, _, _, psub, link = quant.profile_answer(
        rs, group, stored, "lfq", "all", 0.01, "/b/", "light"
    )
    assert link == f"/b/protein?group={group.replace('|', '%7C')}"
    assert "ringed" in psub  # MBR ran: transfers are marked
    names = [t.name for t in fig.data]
    assert "MaxLFQ" in names and "protein_group_quant" in names
    body, states_sub = quant.states_answer(rs, 0.01)
    assert "MBR transfers" in _text(body) and "run_psm_q" in states_sub


def test_answers_on_the_single_run(open_fixture):
    rs = open_fixture("single")
    fig, sub, body, _ = quant.dist_answer(rs, None, "lfq", "accepted", 0.01, "light")
    assert "per-run quant" in sub  # a single run has no LFQ, whatever the toolbar says
    assert "have no quantity" in _text(body)
    fig, table, sub = quant.cv_answer(rs, None, "quant", "accepted", "all", 0.01, "light")
    assert table is None and "one run" in fig.layout.annotations[0].text
    fig, count, _ = quant.heat_answer(rs, None, "all", 0.01, "relative", False, None, "light")
    assert count == "" and "experiment" in fig.layout.annotations[0].text
    fig, head, _, _, _ = quant.profile_answer(rs, None, None, "quant", "all", 0.01, "/", "light")
    assert head is None


def test_group_options_keep_the_value(open_fixture):
    rs = open_fixture("experiment")
    opts = quant.group_options(rs, "fix01", "OTHER", "all", 0.01)
    assert opts[0]["value"] == "OTHER"
    assert any("FIX01" in o["value"] for o in opts[1:])
    assert quant.default_group(rs, "WANTED", 0.01) == "WANTED"
    # No group is accepted at 0.01 in the fixture: the most abundant LFQ group is taken.
    assert quant.default_group(rs, None, 0.01) == Q.find_groups(rs, "")[0]


def _key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k or k == needle]
    keys = [k for k in keys if k == needle] or keys
    assert len(keys) == 1, keys
    return keys[0]


def _post(client, key, inputs, state=()):
    outputs = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        outputs.append(
            {"id": json.loads(cid) if cid.startswith("{") else cid, "property": prop.split("@")[0]}
        )
    body = {
        "output": key,
        "outputs": outputs if key.startswith("..") else outputs[0],
        "inputs": list(inputs),
        "changedPropIds": [
            f"{json.dumps(i['id']) if isinstance(i['id'], dict) else i['id']}.{i['property']}"
            for i in inputs
        ],
        "state": list(state),
    }
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code == 200, resp.data[:800]
    return resp.get_json()["response"]


def test_cv_callback_follows_the_stored_conditions(open_fixture):
    rs = open_fixture("mbr")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    key = _key(app, "qq-cv-table.children")
    inputs = [
        {"id": "mv-conditions", "property": "data", "value": {"a": "X", "c": "X"}},
        {"id": "qq-source", "property": "value", "value": "lfq"},
        {"id": "qq-keys", "property": "value", "value": "all"},
        {"id": "qq-need", "property": "value", "value": "all"},
        {"id": "threshold", "property": "data", "value": 0.01},
    ]
    out = _post(client, key, inputs, [{"id": "scheme", "property": "data", "value": "light"}])
    assert "condition X" in json.dumps(out)
    inputs[0]["value"] = {}  # the suggestion: one run per condition, no CV
    out = _post(client, key, inputs, [{"id": "scheme", "property": "data", "value": "light"}])
    assert "one run" in json.dumps(out)


def test_search_callback(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    key = _key(app, "qq-group.data")
    out = _post(
        app.server.test_client(),
        key,
        [{"id": "qq-group", "property": "searchValue", "value": "FIX02"}],
        [
            {"id": "qq-group", "property": "value", "value": None},
            {"id": "qq-keys", "property": "value", "value": "all"},
            {"id": "threshold", "property": "data", "value": 0.01},
        ],
    )
    values = [o["value"] for o in out["qq-group"]["data"]]
    assert values and all("FIX02" in v for v in values)


def test_figures():
    d = np.array([[1.0, 2.0, np.nan], [4.0, 8.0, 16.0], [5.0, np.nan, np.nan]])
    m = Q.QuantMatrix(
        "protein",
        "quant",
        ("a", "b", "c"),
        __import__("pandas").DataFrame({"protein_group": ["G0", "G1", "G2"]}),
        d,
    )
    cond = {"a": "X", "b": "X", "c": "Y"}
    dist = Q.quantity_distribution(m, bins=8)
    fig = quant_figures.distribution_figure({"protein": dist}, cond, "light")
    assert any("condition X" in (t.name or "") for t in fig.data)
    h = Q.heatmap(m, cond)
    light = quant_figures.heatmap_figure(h, {"X": "#000000", "Y": "#111111"}, "light")
    dark = quant_figures.heatmap_figure(h, {}, "dark")
    assert light.data[0].colorscale[2][1] == quant_figures.CENTRE["light"]
    assert dark.data[0].colorscale[2][1] == quant_figures.CENTRE["dark"]
    assert light.data[0].meta == "qq-relative" and len(light.data) == 1
    assert light.layout.plot_bgcolor == quant_figures.MISSING_CELL
    marked = quant_figures.heatmap_figure(h, {}, "light", selected="G1")
    assert any(s.name == "selected" for s in marked.layout.shapes)
    assert quant_figures.compact(1.8e6) == "1.80 M" and quant_figures.compact(None) == "missing"
    assert quant_figures.condition_colours({"a": "Y", "b": "X", "c": "Y"}, ["a", "b", "c"]) == {
        "Y": quant_figures.theme.SERIES[0],
        "X": quant_figures.theme.SERIES[1],
    }


def test_short_file_names():
    files = {
        "r0": "LFQ_Astral_DIA_15min_50ng_Condition_A_REP1.mzML",
        "r1": "LFQ_Astral_DIA_15min_50ng_Condition_B_REP1.mzML",
        "r2": None,
    }
    short = quant_cards.short_names(files)
    assert short == {"r0": "…A_REP1.mzML", "r1": "…B_REP1.mzML", "r2": None}
    assert quant_cards.short_names({"": "one.mzML"}) == {"": "one.mzML"}


def test_protein_href():
    assert quant.protein_href("/t/", "ATLA3_HUMAN") == "/t/protein?group=ATLA3_HUMAN"
    assert quant.protein_href("/t/", None) == "/t/protein"
