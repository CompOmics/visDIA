"""The compare page, built on the fixtures without a browser, and its callbacks through
the server (posted to /_dash-update-component as the browser posts them), with A at /
and B at /b/ on one server (create_compare_apps)."""

from __future__ import annotations

import json

import pytest
from dash.development.base_component import Component
from plotly.utils import PlotlyJSONEncoder

from mumdia_viewer.data import compare as cd
from mumdia_viewer.ui import compare, compare_cards, compare_figures
from mumdia_viewer.ui.app import create_app, create_compare_apps
from mumdia_viewer.ui.state import PageContext

PAIRS = [("single", "grouped"), ("single", "experiment"), ("experiment", "mbr")]


def _walk(node):
    if isinstance(node, Component):
        yield node
        yield from _walk(getattr(node, "children", None))
        for prop in ("label", "leftSection", "rightSection", "right", "title", "subtitle"):
            value = getattr(node, prop, None)
            if isinstance(value, Component | list):
                yield from _walk(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)


def _ids(tree) -> list[str]:
    return [json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)]


def _text(tree) -> str:
    return json.dumps(tree, cls=PlotlyJSONEncoder)


# --------------------------------------------------------------------------- layout


def test_without_compare_shows_the_command(open_fixture):
    rs = open_fixture("single")
    tree = compare.layout(PageContext(rs=rs, base="/"))
    text = _text(tree)
    assert "--compare" in text and str(rs.root).replace("\\", "\\\\") in text
    assert "mumdia-viewer" in text and "b/" in text


@pytest.mark.parametrize(("x", "y"), PAIRS)
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_page_builds_with_unique_ids(open_fixture, x, y, scheme):
    a, b = open_fixture(x), open_fixture(y)
    create_compare_apps(a, b, url_base="/")  # registers the base of B for "/"
    tree = compare.layout(PageContext(rs=a, base="/", compare=b, scheme=scheme))
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for needed in ("cmp-unit", "cmp-overlap", "cmp-counts", "cmp-detail", "cmp-prov-card"):
        assert json.dumps(needed) in ids
    text = _text(tree)
    for word in ("precursor_q", "peptide_q_value", "pg_q_value", "Jaccard", "config_json"):
        assert word in text
    assert a.manifest.git_sha in text
    # The links of the hero go to B's viewer.
    assert '"/b/compare"' in text and '"/b/"' in text


def test_header_counts_are_each_sides(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    tree = compare.layout(PageContext(rs=a, base="/", compare=b, threshold=0.05))
    counts = next(c for c in _walk(tree) if getattr(c, "id", None) == "cmp-counts")
    text = _text(counts)
    for rs in (a, b):
        for n in cd.side_counts(rs, 0.05).values():
            assert f"{n:,}" in text


def test_overlap_rows_show_the_counts(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    overlaps = cd.overlap(a, b, 0.01)
    rows = compare_cards.overlap_rows(overlaps, "peptide")
    assert [r.id["unit"] for r in rows] == list(cd.COMPARE_UNITS)
    assert [("cmp-ov-row-active" in r.className) for r in rows] == [False, True, False]
    text = _text(rows)
    for o in overlaps:
        for n in (o.n_a, o.n_b, o.n_only_a, o.n_both, o.n_only_b):
            assert f"{n:,}" in text


def test_detail_has_scatters_and_tables(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    tree = compare.detail(a, b, unit="precursor", t=0.01, scheme="dark", base="/", b_base="/b/")
    ids = _ids(tree)
    for needed in ("cmp-grid-a", "cmp-grid-b", "cmp-pair", "cmp-quant-slot", "cmp-pick"):
        assert json.dumps(needed) in ids
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    assert {g.id["name"] for g in graphs} == {"cmp-score", "cmp-quant"}
    for g in graphs:
        assert g.figure["layout"]["template"]["layout"]["font"]["color"] == "#c1c2c5"
    grid_a = next(c for c in _walk(tree) if getattr(c, "id", None) == "cmp-grid-a")
    grid_b = next(c for c in _walk(tree) if getattr(c, "id", None) == "cmp-grid-b")
    ua = cd.unique_keys(a, b, "precursor", 0.01, "a")
    ub = cd.unique_keys(a, b, "precursor", 0.01, "b")
    assert len(grid_a.rowData) == len(ua) and len(grid_b.rowData) == len(ub)
    assert all(r["_href"].startswith("/precursor?") for r in grid_a.rowData)
    assert all(r["_href"].startswith("/b/precursor?") for r in grid_b.rowData)
    assert grid_b.dashGridOptions["context"]["external"] is True


def test_unique_records_link_to_pages(open_fixture):
    a, b = open_fixture("experiment"), open_fixture("mbr")
    df = cd.unique_keys(a, b, "precursor", 0.5, "b")
    rows = compare_cards.unique_records(df, "precursor", "/b/")
    for r, k in zip(rows, df.itertuples(), strict=False):
        assert r["_href"] == f"/b/precursor?run={k.run}&cid={int(k.cid)}"
    pg = cd.unique_keys(a, open_fixture("ovl_bp"), "protein_group", 0.5, "a")
    rows = compare_cards.unique_records(pg, "protein_group", "/")
    assert all(r["_href"].startswith("/protein?group=") for r in rows)
    assert (
        compare_cards.unique_records(df, "precursor", None)[0]["_href"] is None if len(df) else True
    )


def test_grid_limit(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    df = cd.unique_keys(a, b, "precursor", 0.01, "a")
    assert len(compare_cards.unique_records(df, "precursor", "/", limit=3)) == min(3, len(df))


# --------------------------------------------------------------------------- figures


def test_sample_rows_is_fixed():
    assert compare_figures.sample_rows(10).tolist() == list(range(10))
    one = compare_figures.sample_rows(100_000, 500)
    two = compare_figures.sample_rows(100_000, 500)
    assert one.tolist() == two.tolist() and len(set(one.tolist())) == 500
    assert one.tolist() == sorted(one.tolist())


def test_score_figure_points_are_rows(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    shared = cd.shared_keys(a, b, "precursor", 0.01)
    fig = compare_figures.score_figure(shared, unit_noun="precursors", a_label="x", b_label="y")
    pts = fig["data"][1]
    assert pts["customdata"] == list(range(len(shared)))
    assert pts["x"][0] == pytest.approx(shared["a_score"].iloc[0], rel=1e-5)
    assert compare_figures.figure_points(fig) == len(shared)
    empty = compare_figures.score_figure(shared.head(0), unit_noun="x", a_label="a", b_label="b")
    assert empty.layout.annotations


def test_quantity_figure_and_ratio(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    df = cd.quantity_pairs(a, b, 0.01, cd.run_pairs(a, b))
    fig = compare_figures.quantity_figure(df)
    assert fig["layout"]["xaxis"]["type"] == "log"
    ok = (df["a_quantity"] > 0) & (df["b_quantity"] > 0)
    assert compare_figures.figure_points(fig) == int(ok.sum())
    ratio = compare_figures.log2_median_ratio(df)
    import numpy as np

    want = float(np.median(np.log2(df.loc[ok, "b_quantity"] / df.loc[ok, "a_quantity"])))
    assert ratio == pytest.approx(want)


# --------------------------------------------------------------------------- callbacks


def _key(app, needle: str) -> str:
    if needle in app.callback_map:
        return needle
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _prop_id(i) -> str:
    cid = i["id"]
    if isinstance(cid, dict):
        cid = json.dumps(cid, separators=(",", ":"), sort_keys=True)
    return f"{cid}.{i['property']}"


def _post(client, key, inputs, state, triggered=None, base="/"):
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
    resp = client.post(f"{base}_dash-update-component", json=body)
    assert resp.status_code in (200, 204), resp.data[:800]
    return resp.get_json()["response"] if resp.status_code == 200 else {}


@pytest.fixture
def apps(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    app_a, app_b = create_compare_apps(a, b, url_base="/")
    return a, b, app_a, app_b, app_a.server.test_client()


def test_detail_follows_the_threshold(apps):
    a, b, app, _, client = apps
    key = _key(app, "cmp-detail.children")
    out = _post(
        client,
        key,
        [{"id": "threshold", "property": "data", "value": 0.5}],
        [
            {"id": "cmp-unit", "property": "value", "value": "protein_group"},
            {"id": "scheme", "property": "data", "value": "light"},
            {"id": "cmp-top-ms", "property": "data", "value": 12},
        ],
    )
    text = json.dumps(out)
    assert "pg_q_value" in text and "protein groups" in text.lower()
    assert "cmp-score-slot" in text and "cmp-tables" in text and "cmp-quant-slot" in text
    o = next(o for o in cd.overlap(a, b, 0.5) if o.unit == "protein_group")
    assert f"{o.n_both:,} shared" in json.dumps(out["cmp-unit-summary"])


def test_unit_replaces_scores_and_tables_only(apps):
    a, b, app, _, client = apps
    key = _key(app, "cmp-tables.children")
    assert "cmp-quant-slot" not in key
    out = _post(
        client,
        key,
        [{"id": "cmp-unit", "property": "value", "value": "peptide"}],
        [
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
    )
    assert "peptide_q_value" in json.dumps(out["cmp-tables"])
    assert "Scores of shared peptides" in json.dumps(out["cmp-score-slot"])
    o = next(o for o in cd.overlap(a, b, 0.01) if o.unit == "peptide")
    assert f"{o.n_only_b:,} only in B" in json.dumps(out["cmp-unit-summary"])
    assert out["cmp-pick"]["children"] is None


def test_top_follows_threshold(apps):
    a, _b, app, _, client = apps
    key = _key(app, "cmp-counts.children")
    out = _post(
        client,
        key,
        [{"id": "threshold", "property": "data", "value": 0.05}],
        [{"id": "cmp-unit", "property": "value", "value": "peptide"}],
    )
    assert out["cmp-overlap-t"]["children"] == "q ≤ 0.05"
    text = json.dumps(out["cmp-counts"])
    for n in cd.side_counts(a, 0.05).values():
        assert f"{n:,}" in text
    assert "cmp-ov-row-active" in json.dumps(out["cmp-overlap"])


def test_pick_links_into_both_sides(apps):
    a, b, app, _, client = apps
    key = _key(app, "cmp-pick.children")
    shared = cd.shared_keys(a, b, "precursor", 0.01)
    state = [
        {"id": "threshold", "property": "data", "value": 0.01},
        {"id": "cmp-unit", "property": "value", "value": "precursor"},
        {"id": "cmp-pair", "property": "value", "value": "0"},
    ]
    inputs = [
        {
            "id": compare.SCORE_FIG,
            "property": "clickData",
            "value": {"points": [{"customdata": 2}]},
        },
        {"id": compare.QUANT_FIG, "property": "clickData", "value": None},
    ]
    out = _post(client, key, inputs, state)
    text = json.dumps(out)
    r = shared.iloc[2]
    assert f"/precursor?cid={int(r['a_cid'])}" in text
    assert f"/b/precursor?cid={int(r['b_cid'])}" in text
    # A quantity point.
    df = cd.quantity_pairs(a, b, 0.01, cd.run_pairs(a, b)[:1])
    inputs[1]["value"] = {"points": [{"customdata": 0}]}
    out = _post(client, key, inputs, state, triggered=[inputs[1]])
    assert f"cid={int(df.iloc[0]['a_cid'])}" in json.dumps(out)


def test_pair_and_export(apps):
    a, b, app, _, client = apps
    key = _key(app, "cmp-quant-slot.children")
    out = _post(
        client,
        key,
        [{"id": "cmp-pair", "property": "value", "value": "0"}],
        [
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "scheme", "property": "data", "value": "light"},
        ],
    )
    assert "Per run" in json.dumps(out)
    key = _key(app, "cmp-download-b.data")
    out = _post(
        client,
        key,
        [{"id": "cmp-export-b", "property": "n_clicks", "value": 1}],
        [
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "cmp-unit", "property": "value", "value": "precursor"},
        ],
    )
    data = out["cmp-download-b"]["data"]
    lines = data["content"].strip().splitlines()
    assert lines[0].split("\t")[0] == "key" and "precursor_q" in lines[0]
    assert len(lines) - 1 == len(cd.unique_keys(a, b, "precursor", 0.01, "b"))


def test_b_viewer_compares_the_other_way(apps):
    a, b, app_a, app_b, client = apps
    assert app_a.mv_compare_base() == "/b/" and app_b.mv_compare_base() == "/"
    assert compare.b_base_of("/b/") == "/"
    key = _key(app_b, "cmp-counts.children")
    out = _post(
        client,
        key,
        [{"id": "threshold", "property": "data", "value": 0.01}],
        [{"id": "cmp-unit", "property": "value", "value": "precursor"}],
        base="/b/",
    )
    o = cd.overlap(b, a, 0.01)[0]
    assert f"{o.n_only_a:,}" in json.dumps(out["cmp-overlap"])


def test_callbacks_without_b_do_nothing(open_fixture):
    app = create_app(open_fixture("single"), url_base="/")
    client = app.server.test_client()
    key = _key(app, "cmp-counts.children")
    out = _post(
        client,
        key,
        [{"id": "threshold", "property": "data", "value": 0.01}],
        [{"id": "cmp-unit", "property": "value", "value": "precursor"}],
    )
    assert out == {}
