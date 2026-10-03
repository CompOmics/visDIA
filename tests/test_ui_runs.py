"""The across-runs page and the condition-ratio page, built on the fixtures without a
browser."""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import across as A
from mumdia_viewer.ui import ratios, runs, runs_cards, runs_figures
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "experiment", "mbr", "grouped", "topk"]
SHELL = {"threshold", "scheme", "url", "q-select", "templates", "recent", "page", "mv-conditions"}


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


def _precursor(rs) -> tuple[str, int]:
    df = pq.read_table(rs.scored.path).to_pandas()
    df = df[df["label"] == "target"].sort_values("score", ascending=False)
    return str(df["peptidoform"].iloc[0]), int(df["charge"].iloc[0])


def _ctx(rs, **kw):
    return PageContext(rs=rs, base="/", **kw)


@pytest.mark.parametrize("name", FIXTURES)
def test_runs_layout_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    pep, z = _precursor(rs)
    for query in ({"peptidoform": pep, "charge": str(z)}, {}):
        tree = runs.layout(_ctx(rs, query=query))
        ids = _ids(tree)
        assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
        graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
        assert {g.id["name"] for g in graphs} == {"xr-xic", "xr-quant", "xr-q"}
        text = _text(tree)
        assert "XICs across runs" in text and "Grouped q" in text
        for run in rs.runs:
            assert json.dumps({"run": run.name, "type": "xr-cond"}, sort_keys=True) in ids
        if not rs.is_experiment:
            assert "This is a single run" in text
        if not query:
            assert "No precursor in the address" in text


def test_runs_layout_for_an_unknown_precursor(open_fixture):
    rs = open_fixture("experiment")
    tree = runs.layout(_ctx(rs, query={"peptidoform": "NOTAPEPTIDE", "charge": "2"}))
    text = _text(tree)
    assert "Precursor not found" in text and "no scored row" in text


@pytest.mark.parametrize("name", FIXTURES)
def test_ratios_layout_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    tree = ratios.layout(_ctx(rs))
    ids = _ids(tree)
    assert len(ids) == len(set(ids))
    text = _text(tree)
    assert "Condition ratios" in text
    if len(rs.runs) < 2:
        assert "single run" in text and "cr-a" not in ids
    else:
        for i in ("cr-a", "cr-b", "cr-suffixes", "cr-hye", "cr-expected"):
            assert json.dumps(i) in ids
        assert "entered by you" in text and "not measured" in text


def _string_ids(deps: list[dict]) -> set[str]:
    return {d["id"] for d in deps if not d["id"].startswith("{")}


@pytest.mark.parametrize("name", ["experiment", "mbr"])
def test_every_page_callback_finds_its_ids(open_fixture, name):
    rs = open_fixture(name)
    app = create_app(rs, url_base="/")
    pep, z = _precursor(rs)
    present = set()
    for tree in (
        runs.layout(_ctx(rs, query={"peptidoform": pep, "charge": str(z)})),
        ratios.layout(_ctx(rs)),
    ):
        present |= {json.loads(i) for i in _ids(tree) if i.startswith('"')}
    deps = app.server.test_client().get("/_dash-dependencies").get_json()
    seen = 0
    for dep in deps:
        inputs = _string_ids(dep["inputs"])
        if not any(i.startswith(("xr-", "cr-")) for i in inputs | _string_ids(dep["state"])):
            continue
        seen += 1
        ids = inputs | _string_ids(dep["state"])
        parts = [part.strip(".") for part in dep["output"].split("...")]
        outputs = {part.rsplit(".", 1)[0] for part in parts if part and not part.startswith("{")}
        missing = {i for i in ids | outputs if i not in present and i not in SHELL}
        assert not missing, (dep["output"], missing)
        # The shared conditions store is read as a state: a callback with an input in
        # the app shell and the others on this page would fire on every page.
        assert "mv-conditions" not in inputs, dep["output"]
    assert seen >= 8


# --------------------------------------------------------------------------- helpers


def test_pick_values_and_links():
    assert runs_cards.pick_value("PEPM[Oxidation]K", 2) == "PEPM[Oxidation]K|2"
    assert runs_cards.parse_pick("PEP|K|3") == ("PEP|K", 3)
    assert runs_cards.parse_pick("PEP") is None
    assert runs_cards.parse_pick("PEP|x") is None
    assert runs_cards.runs_href("/t/", "PEPM[Oxidation]K", 2) == (
        "/t/runs?peptidoform=PEPM%5BOxidation%5DK&charge=2"
    )
    assert runs_cards.run_href("/", "", 7) == "/precursor?cid=7"
    assert runs_cards.run_href("/", "r1", float("nan")) is None


def test_click_targets(open_fixture):
    rs = open_fixture("experiment")
    pep, z = _precursor(rs)
    a = A.precursor_across(rs, pep, z)
    row = a.rows[a.rows["scored"]].iloc[0]
    want = f"/precursor?run={row['run']}&cid={int(row['candidate_id'])}"
    assert runs.click_target("/", a, {"points": [{"customdata": row["run"]}]}) == want
    assert runs.click_target("/", a, {"points": [{"customdata": [row["run"], 1.0]}]}) == want
    assert runs.click_target("/", a, {"points": [{"customdata": "nope"}]}) is None
    assert runs.click_target("/", a, None) is None
    assert ratios.point_href("/", {"points": [{"text": "PEPK 2+"}]}) == (
        "/runs?peptidoform=PEPK&charge=2"
    )
    assert ratios.point_href("/", {"points": [{"text": "ALBU_HUMAN"}]}) == (
        "/protein?group=ALBU_HUMAN"
    )
    assert ratios.point_href("/", {}) is None


def test_expected_ratios_follow_the_conditions():
    assert ratios.flip_ratio("2:1") == "1:2"
    assert ratios.flip_ratio("1/4") == "4:1"
    assert ratios.flip_ratio("0.5") == "1:0.5"
    assert ratios.flip_ratio("") == ""
    store = {"pair": ["A", "B"], "values": {"_YEAST": "2:1", "_ECOLI": "1:4"}}
    sfx = ("_HUMAN", "_YEAST", "_ECOLI")
    assert ratios.expected_log2(store, sfx, "A", "B") == {
        "_HUMAN": None,
        "_YEAST": 1.0,
        "_ECOLI": -2.0,
    }
    # The ratios describe the samples: B over A turns them round.
    assert ratios.expected_log2(store, sfx, "B", "A")["_YEAST"] == -1.0
    assert ratios.expected_log2(store, sfx, "B", "A")["_ECOLI"] == 2.0
    assert ratios.expected_log2({"_YEAST": "2:1"}, sfx, "B", "A")["_YEAST"] == 1.0
    assert ratios.expected_log2(None, sfx) == dict.fromkeys(sfx)
    assert ratios.clean_suffixes(["_HUMAN", " _HUMAN", "", "_X"]) == ("_HUMAN", "_X")
    assert ratios.clean_suffixes(None) == A.DEFAULT_SUFFIXES


def test_ratio_answers(open_fixture):
    rs = open_fixture("mbr")
    stored = {"a": "X", "c": "Y"}
    exp = {"pair": ["X", "Y"], "values": {"_TEST": "2:1"}}
    fig, table, count, sub, fact = ratios.level_answer(
        rs, "precursor", stored, "X", "Y", "quant", "median", "one", 0.5, ("_TEST",), exp, "light"
    )
    assert isinstance(fig, go.Figure) and " in both" in count
    assert "precursors in both" in _text(fact)
    assert "TEST" in _text(table) and "+1.00" in _text(table)
    assert "median of per-run quant" in _text(sub)
    # The expected ratio is a dashed line at log2(2) = 1.
    assert any(s.line.dash == "dash" and s.x0 == 1.0 for s in fig.layout.shapes)
    opts, a, _, b, lines = ratios.conditions_answer(rs, stored, None, None)
    assert [o["value"] for o in opts] == ["X", "Y"] and (a, b) == ("X", "Y")
    _, a, _, b, lines = ratios.conditions_answer(rs, {"a": "X", "c": "X"}, "X", "Y")
    assert a == "X" and b is None and "one condition" in _text(lines)
    fig = ratios.scatter_answer(
        rs, "precursor", stored, "X", "Y", "quant", "median", "one", 0.5, ("_TEST",), exp, "dark"
    )
    assert fig.layout.template.layout.font.color == "#c1c2c5"
    assert all(" " in t and t.endswith("+") for tr in fig.data for t in tr.text)
    fig, *_ = ratios.level_answer(
        open_fixture("single"),
        "protein",
        None,
        None,
        None,
        "lfq",
        "median",
        "one",
        0.01,
        ("_TEST",),
        None,
        "light",
    )
    assert "single run" in fig.layout.annotations[0].text


# --------------------------------------------------------------------------- figures


@pytest.mark.parametrize("name", ["single", "mbr"])
def test_figures(open_fixture, name):
    rs = open_fixture(name)
    pep, z = _precursor(rs)
    a = A.precursor_across(rs, pep, z)
    xics = A.run_xics(rs, a)
    grid = runs_figures.xic_grid_figure(a, xics, 0.01, "light")
    assert len(grid.layout.annotations) >= len(rs.runs)  # one title per panel
    assert all(isinstance(t.customdata[0], str) for t in grid.data)
    shared = runs_figures.xic_grid_figure(a, xics, 0.01, "dark", shared=True)
    assert shared.layout.template.layout.font.color == "#c1c2c5"
    over = runs_figures.xic_overlay_figure(a, xics, 0.01, "light")
    ys = [np.asarray(t.y, dtype=float) for t in over.data if t.y is not None and t.y[0] is not None]
    assert all(np.nanmax(y) <= 1.0 + 1e-9 for y in ys)  # each run on its own maximum
    q = runs_figures.quantity_figure(a, "light")
    bars = q.data[0]
    for state, y in zip(a.rows["state"], bars.y, strict=True):
        assert (y is None) == (state != "quantified")  # no bar, never a 0
    qf = runs_figures.q_figure(a, 0.01, "light")
    assert qf.layout.yaxis.type == "log"


def test_quantity_figure_marks_transfers(open_fixture):
    rs = open_fixture("mbr")
    tr = pq.read_table(rs.artifact("mbr_transferred").path).to_pandas()
    scored = pq.read_table(rs.scored.path).to_pandas()
    pr = scored[scored["candidate_id"] == tr["candidate_id"].iloc[0]].iloc[0]
    a = A.precursor_across(rs, str(pr["peptidoform"]), int(pr["charge"]))
    fig = runs_figures.quantity_figure(a, "light")
    shapes = list(fig.data[0].marker.pattern.shape)
    assert shapes == ["/" if t else "" for t in a.rows["transferred"]]
    assert "/" in shapes


# --------------------------------------------------------------------------- callbacks


def _key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
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


def test_mode_and_threshold_callbacks(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    pep, z = _precursor(rs)
    key_state = {"id": "xr-key", "property": "data", "value": {"peptidoform": pep, "charge": z}}
    k = [x for x in app.callback_map if "xr-xic" in x and not x.startswith("..")]
    assert len(k) == 1
    out = _post(
        client,
        k[0],
        [
            {"id": "xr-mode", "property": "value", "value": "overlay"},
            {"id": "xr-scale", "property": "value", "value": "shared"},
        ],
        [
            key_state,
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "scheme", "property": "data", "value": "light"},
        ],
    )
    assert "xr-overlay" in json.dumps(out)
    out = _post(
        client,
        _key(app, "xr-strip-box.children"),
        [{"id": "threshold", "property": "data", "value": 0.0001}],
        [
            key_state,
            {"id": "xr-mode", "property": "value", "value": "grid"},
            {"id": "xr-scale", "property": "value", "value": "own"},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
    )
    assert "q ≤ 1e-4" in json.dumps(out, ensure_ascii=False)
    out = _post(
        client,
        _key(app, "xr-pick.data"),
        [{"id": "xr-pick", "property": "searchValue", "value": pep[:4]}],
        [
            {"id": "xr-pick", "property": "value", "value": f"{pep}|{z}"},
            {"id": "xr-pick", "property": "data", "value": []},
        ],
    )
    values = [o["value"] for o in out["xr-pick"]["data"]]
    assert values[0] == f"{pep}|{z}" and len(values) > 1


def test_ratio_callback(open_fixture):
    rs = open_fixture("mbr")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    k = [x for x in app.callback_map if "cr-tab-protein" in x]
    assert len(k) == 1
    inputs = [
        {"id": "cr-conds", "property": "data", "value": {"a": "X", "c": "Y"}},
        {"id": "cr-a", "property": "value", "value": "X"},
        {"id": "cr-b", "property": "value", "value": "Y"},
        {"id": "cr-source", "property": "value", "value": "lfq"},
        {"id": "cr-summary", "property": "value", "value": "mean"},
        {"id": "cr-need", "property": "value", "value": "one"},
        {"id": "threshold", "property": "data", "value": 0.01},
        {"id": "cr-suffixes", "property": "value", "value": ["_TEST"]},
        {"id": "cr-expected", "property": "data", "value": None},
    ]
    out = json.dumps(_post(client, k[0], inputs, [{"id": "scheme", "property": "data"}]))
    assert "mean of MaxLFQ" in out and "Expected X : Y" in out
    out = _post(
        client,
        _key(app, "cr-conds.data"),
        [{"id": "cr-init", "property": "data", "value": 0}],
        [
            {"id": "mv-conditions", "property": "data", "value": {"a": "X", "c": "Y"}},
            {"id": "cr-a", "property": "value", "value": None},
            {"id": "cr-b", "property": "value", "value": None},
        ],
    )
    assert out["cr-conds"]["data"] == {"a": "X", "c": "Y"}
    assert out["cr-a"]["value"] == "X" and out["cr-b"]["value"] == "Y"
