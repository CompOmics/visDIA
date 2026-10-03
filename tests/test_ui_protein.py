"""The protein page, on the fixtures.

The page's rows must be the data layer's (data.protein, identification_table); its
callbacks answer through the Flask test client; the page in Chromium keeps the panels
linked (a peptide click fills the precursors and moves the coverage outline, a coverage
click selects the peptide, Enter opens the precursor page). The Chromium part is skipped
when Chromium is not installed.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
from dash.development.base_component import Component
from plotly.io.json import to_json_plotly

from mumdia_viewer.data.fasta import Fasta, protein_coverage
from mumdia_viewer.data.protein import (
    group_from_peptides,
    group_peptides,
    peptide_precursors,
    peptide_run_matrix,
    protein_group,
    quant_by_run,
)
from mumdia_viewer.ui import protein, protein_figures, protein_view
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURE_FASTA = Path(__file__).parent / "fixtures" / "smoke" / "test_data" / "fixture.fasta"
GROUP = "sp|FIXT01|FIX01_TEST"
DECOY = "DECOY_sp|FIXT06|FIX06_TEST"
FIXTURES = ["single", "experiment", "mbr", "grouped", "topk"]


def _walk(node):
    if isinstance(node, Component):
        yield node
        yield from _walk(getattr(node, "children", None))
        for prop in ("label", "leftSection", "rightSection", "title", "right"):
            value = getattr(node, prop, None)
            if isinstance(value, Component | list):
                yield from _walk(value)
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _walk(item)


def _ids(tree) -> list[str]:
    return [json.dumps(c.id, sort_keys=True) for c in _walk(tree) if getattr(c, "id", None)]


def _find(tree, id_):
    for c in _walk(tree):
        if getattr(c, "id", None) == id_:
            return c
    raise AssertionError(f"no component {id_!r}")


def _has(tree, id_) -> bool:
    return any(getattr(c, "id", None) == id_ for c in _walk(tree))


def _text(tree) -> str:
    return json.dumps(json.loads(to_json_plotly(tree)), ensure_ascii=False)


@pytest.fixture(scope="module")
def fasta():
    return Fasta.read([FIXTURE_FASTA])


def _page(rs, query, *, fasta=None, t=0.1, scheme="light"):
    return protein.layout(
        PageContext(rs=rs, base="/t/", threshold=t, query=query, fasta=fasta, scheme=scheme)
    )


# --------------------------------------------------------------------------- address


def test_protein_href_and_address():
    assert protein.protein_href("/t/", "A_HUMAN;B_HUMAN") == "/t/protein?group=A_HUMAN%3BB_HUMAN"
    assert protein.protein_href("/t/", "A", peptide=5) == "/t/protein?group=A&peptide=5"
    assert protein.protein_href("/t/", None) == "/t/protein"
    want = protein.parse_address({"group": " A;B ", "peptide": "12", "member": "B"})
    assert want == {"group": "A;B", "peptide": 12, "member": "B", "search": ""}
    bad = protein.parse_address({"peptide": "x1", "search": "  macf1 "})
    assert bad["group"] is None and bad["peptide"] is None and bad["search"] == "macf1"


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("with_fasta", [False, True])
def test_group_page_builds_with_unique_ids(open_fixture, fasta, name, with_fasta):
    rs = open_fixture(name)
    for group in (GROUP, DECOY):
        tree = _page(rs, {"group": group}, fasta=fasta if with_fasta else None)
        ids = _ids(tree)
        assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
        for required in (
            "pp-root",
            "pp-key",
            "pp-sel",
            "pp-pre-req",
            "pp-pre-data",
            "pp-strip",
            "pp-members-slot",
            "pp-cov-body",
            "pp-pep-grid",
            "pp-pre-grid",
        ):
            assert json.dumps(required) in ids, (group, required)
        assert _has(tree, protein.MATRIX) == rs.is_experiment
        key = _find(tree, "pp-key").data
        assert key["group"] == group and key["decoy"] == group.startswith("DECOY_")


def test_peptide_rows_are_the_data_layer_rows(open_fixture, fasta):
    rs = open_fixture("single")
    tree = _page(rs, {"group": GROUP}, fasta=fasta)
    grid = _find(tree, "pp-pep-grid")
    want = group_peptides(rs, GROUP).rows
    assert [r["base_peptide_id"] for r in grid.rowData] == want["base_peptide_id"].tolist()
    assert all(r["_key"] == str(r["base_peptide_id"]) for r in grid.rowData)
    assert all(r["_href"].startswith("/t/precursor?") for r in grid.rowData)
    # The first peptide is selected; its precursors come on the browser's request.
    first = str(want["base_peptide_id"].iloc[0])
    assert grid.selectedRows == {"ids": [first]}
    assert _find(tree, "pp-pre-grid").rowData == []
    # Positions: the coverage spans (1-based), the first occurrence.
    cov = protein_coverage(rs, fasta, GROUP, threshold=0.1)
    for r in grid.rowData:
        spans = cov.spans_of(r["base_peptide_id"])
        if spans:
            assert (r["_start"], r["_end"], r["_n_pos"]) == (
                spans[0].start + 1,
                spans[0].end,
                len(spans),
            )
        else:
            assert r["_start"] is None and r["_n_pos"] == 0
    # Rollup: the peptides of the protein's top-N sum, with their ranks.
    m = peptide_run_matrix(rs, GROUP, 0.1)
    ranks = dict(zip(m["base_peptide_id"], m["rollup_rank"], strict=True))
    inroll = set(m.loc[m["in_rollup"], "base_peptide_id"])
    for r in grid.rowData:
        if r["base_peptide_id"] in inroll:
            assert r["_rollup_rank"] == ranks[r["base_peptide_id"]]
        else:
            assert r["_rollup_rank"] is None
    cols = [d["colId"] for d in grid.columnDefs]
    assert cols[0] == "_valid" and "_pos" in cols and "_rollup" in cols and cols[-1] == "_open"


def test_selection_from_the_address(open_fixture):
    rs = open_fixture("experiment")
    ids = group_peptides(rs, GROUP).rows["base_peptide_id"].tolist()
    tree = _page(rs, {"group": GROUP, "peptide": str(ids[3])})
    assert _find(tree, "pp-pep-grid").selectedRows == {"ids": [str(ids[3])]}
    assert _find(tree, "pp-sel").data["peptide"] == ids[3]
    # A peptide of another group falls back to the first row.
    tree = _page(rs, {"group": GROUP, "peptide": "999999999"})
    assert _find(tree, "pp-sel").data["peptide"] == ids[0]


def test_group_row_from_peptides_equals_the_protein_group_table(open_fixture):
    for name in ("single", "experiment", "mbr", "grouped"):
        rs = open_fixture(name)
        for group in (GROUP, DECOY, "sp|FIXT14|FIX14_TEST"):
            a = protein_group(rs, group)
            b = group_from_peptides(rs, group, group_peptides(rs, group))
            for col in ("candidate_id", "source", "pg_q_value", "score", "n_peptides", "label"):
                assert a.get(col) == b.get(col), (name, group, col)


def test_strip_follows_the_threshold(open_fixture):
    rs = open_fixture("single")
    pg = protein_group(rs, GROUP)
    peps = group_peptides(rs, GROUP).rows
    q = pg.get("pg_q_value")
    for t, verdict in ((0.1, "passes"), (0.01, "does not pass")):
        assert (q <= t) == (verdict == "passes")
        ctx = PageContext(rs=rs, base="/", threshold=t)
        text = _text(
            protein_view.strip(
                ctx,
                pg,
                peptides=peps,
                quant=quant_by_run(rs, GROUP),
                matrix=None,
                coverage=None,
                scales=protein.scales_of(rs),
                rescorer=None,
            )
        )
        n_pass = int((peps["peptide_q_value"] <= t).sum())
        assert verdict in text and f"{n_pass:,} of the group's {len(peps):,} peptides" in text


def test_missing_group_and_search_pages(open_fixture):
    rs = open_fixture("single")
    tree = _page(rs, {"group": "NOPE_X;sp|FIXT01|FIX01_TEST"})
    assert _has(tree, "pp-missing") and not _has(tree, "pp-pep-grid")
    tree = _page(rs, {"group": "FIXT0"})  # not a group: suggestions that contain it
    text = _text(tree)
    assert "No protein group" in text and "sp|FIXT01|FIX01_TEST" in text
    tree = _page(rs, {"search": "fix1"})
    grid = _find(tree, "pp-find-grid")
    assert {r["protein_group"] for r in grid.rowData} == {
        f"sp|FIXT{i}|FIX{i}_TEST" for i in ("10", "11", "12", "13", "14", "15", "16")
    }
    assert all(r["_protein"].startswith("/t/protein?group=") for r in grid.rowData)
    assert "_open" not in [d["colId"] for d in grid.columnDefs]
    assert _find(tree, "pp-find-input").value == "fix1"


def test_decoy_group_has_no_coverage(open_fixture, fasta):
    rs = open_fixture("single")
    tree = _page(rs, {"group": DECOY}, fasta=fasta)
    assert "decoy protein group has no sequence" in _text(_find(tree, "pp-cov-body"))
    rows = _find(tree, "pp-pep-grid").rowData
    assert rows and all(r["label"] == "decoy" for r in rows)


def test_without_fasta_the_page_says_how_to_add_one(open_fixture):
    tree = _page(open_fixture("single"), {"group": GROUP})
    assert "--fasta" in _text(_find(tree, "pp-cov-body"))
    assert "_pos" not in [d["colId"] for d in _find(tree, "pp-pep-grid").columnDefs]


# --------------------------------------------------------------------------- figures


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_matrix_figure(open_fixture, scheme):
    rs = open_fixture("mbr")
    m = peptide_run_matrix(rs, GROUP, 0.01)
    rows, n_empty, n_cut = protein_figures.matrix_rows(m)
    assert n_cut == 0 and len(rows) + n_empty == m["base_peptide_id"].nunique()
    fig = protein_figures.matrix_figure(
        m, scheme, threshold=0.01, base_href="/t/", experiment=True, rows=rows, selected=rows[0]
    )
    assert fig.layout.meta["rows"] == rows
    traces = {t.name: t for t in fig.data}
    sub = m[m["base_peptide_id"].isin(rows)]
    assert len(traces["identified"].x) == int(sub["identified"].sum())
    assert len(traces["transfer"].x) == int((sub["n_transferred"] > 0).sum())
    ramp = traces["quantity"].colorscale
    assert (ramp[0][1] < ramp[-1][1]) == (scheme == "dark")  # light to dark, reversed in dark
    hrefs = [c[1] for row in traces["quantity"].customdata for c in row if c[1]]
    assert hrefs and all(h.startswith("/t/precursor?run=") for h in hrefs)
    assert [s.name for s in fig.layout.shapes] == ["selected"]


def test_quantity_per_run_figure_and_ticks(open_fixture):
    rs = open_fixture("experiment")
    fig = protein_figures.quantity_per_run_figure(quant_by_run(rs, GROUP), "light")
    assert [t.name for t in fig.data] == ["protein_group_quant", "MaxLFQ"]
    assert fig.layout.yaxis.type != "log" and fig.layout.yaxis.rangemode == "tozero"
    for lo, hi in ((3.5, 4.7), (3.9, 4.1), (2.1, 6.3)):
        ticks = protein_figures.nice_ticks(lo, hi)
        assert len(ticks) >= 2 and all(lo - 1e-9 <= v <= hi + 1e-9 for v in ticks)
    assert protein_figures.compact(99349.6) == "99.3 k" and protein_figures.compact(None) == ""


# --------------------------------------------------------------------------- callbacks


def _key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _outputs(key: str) -> list[dict]:
    out = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        cid = json.loads(cid) if cid.startswith("{") else cid
        out.append({"id": cid, "property": prop.split("@")[0]})
    return out


def _post(client, key, inputs, state=()):
    outputs = _outputs(key)
    body = {
        "output": key,
        "outputs": outputs if key.startswith("..") else outputs[0],
        "inputs": list(inputs),
        "state": list(state),
        "changedPropIds": [
            f"{i['id']}.{i['property']}" for i in inputs if isinstance(i["id"], str)
        ],
    }
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code in (200, 204), resp.data[:800]
    return resp.get_json()["response"] if resp.status_code == 200 else {}


def test_precursors_callback(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    bp = int(group_peptides(rs, GROUP).rows["base_peptide_id"].iloc[1])
    key = _key(app, "pp-pre-data.data")
    out = _post(
        client,
        key,
        [{"id": "pp-pre-req", "property": "data", "value": {"peptide": bp, "n": 3, "t": 0.1}}],
        [{"id": "pp-key", "property": "data", "value": {"group": GROUP}}],
    )
    data = out["pp-pre-data"]["data"]
    want = peptide_precursors(rs, GROUP, bp).rows
    assert data["peptide"] == bp and data["n"] == 3 and data["total"] == len(want)
    assert [(r["run"], r["candidate_id"]) for r in data["rows"]] == list(
        zip(want["run"], want["candidate_id"], strict=True)
    )
    assert data["mark"] == "run_psm_q" and data["count"].endswith(f"of {len(want)}")
    # Not a request of the page: nothing.
    out = _post(
        client,
        key,
        [{"id": "pp-pre-req", "property": "data", "value": None}],
        [{"id": "pp-key", "property": "data", "value": {"group": GROUP}}],
    )
    assert out == {}


def test_threshold_callbacks(open_fixture, fasta):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/", fasta=fasta)
    client = app.server.test_client()
    key = _key(app, "pp-strip.children")
    out = _post(
        client,
        key,
        [
            {"id": "threshold", "property": "data", "value": 0.001},
            {"id": "pp-cov-req", "property": "data", "value": None},
        ],
        [
            {"id": "pp-key", "property": "data", "value": {"group": GROUP, "experiment": True}},
            {"id": "pp-sel", "property": "data", "value": {"peptide": None}},
        ],
    )
    assert "q ≤ 0.001" in json.dumps(out["pp-strip"]["children"], ensure_ascii=False)
    assert 'data-threshold": "0.001"' in json.dumps(out["pp-cov-body"]["children"])
    mkey = _key(app, "pp-matrix")
    out = _post(
        client,
        mkey,
        [
            {"id": "threshold", "property": "data", "value": 0.05},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
        [
            {"id": "pp-key", "property": "data", "value": {"group": GROUP, "experiment": True}},
            {"id": "pp-sel", "property": "data", "value": {"peptide": None}},
        ],
    )
    fig_out = next(v for k, v in out.items() if "pp-matrix" in k and k.startswith("{"))
    assert fig_out["figure"]["layout"]["meta"]["scheme"] == "dark"
    assert out["pp-matrix-t"]["children"] == "run_psm_q ≤ 0.05"


def test_search_callback(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    out = _post(
        client,
        _key(app, "pp-find-grid.rowData"),
        [
            {"id": "pp-find-input", "property": "value", "value": "FIX0"},
            {"id": "threshold", "property": "data", "value": 0.1},
        ],
    )
    rows = out["pp-find-grid"]["rowData"]
    assert len(rows) == 9 and out["pp-find-count"]["children"] == "9"


def test_app_registers_the_page_callbacks(open_fixture):
    app = create_app(open_fixture("single"), url_base="/abc/")
    deps = json.dumps(app.server.test_client().get("/abc/_dash-dependencies").get_json())
    for needle in ("pp-pre-data.data", "pp-strip.children", "pp-find-grid.rowData", "pp-matrix"):
        assert needle in deps, needle


# --------------------------------------------------------------------------- Chromium


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
def test_the_page_in_chromium(open_fixture, fasta):
    """A peptide click fills the precursors and moves the outline; a coverage click
    selects the peptide; Enter on a precursor opens its page."""
    from werkzeug.serving import make_server

    rs = open_fixture("single")
    app = create_app(rs, url_base="/", fasta=fasta)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    peps = group_peptides(rs, GROUP).rows
    ids = [str(i) for i in peps["base_peptide_id"]]
    seqs = dict(zip(ids, peps["sequence"], strict=True))
    ready = """(want) => {
      const s = document.getElementById('pp-pre-subject');
      const p = document.getElementById('pp-panel-pre');
      return s && s.innerText === want && p && !p.classList.contains('pp-loading'); }"""
    sel = (
        "() => window.dash_ag_grid.getApi('pp-pep-grid').getSelectedNodes()"
        ".map(n => n.data._key)[0]"
    )
    try:
        with SYNC_PLAYWRIGHT() as p:
            chromium = p.chromium.launch()
            page = chromium.new_page(viewport={"width": 1440, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/protein?group=sp%7CFIXT01%7CFIX01_TEST")
            page.wait_for_function(ready, arg=seqs[ids[0]], timeout=60000)
            # A click on a peptide: precursors, outline and address follow.
            page.locator('#pp-pep-grid .ag-row[row-index="2"] [col-id="peptidoform"]').click()
            page.wait_for_function(ready, arg=seqs[ids[2]], timeout=15000)
            assert page.evaluate(sel) == ids[2]
            page.wait_for_function(
                "(k) => document.querySelector('#pp-cov-body .mvc-bar').getAttribute("
                "'data-selected') === k",
                arg=ids[2],
                timeout=5000,
            )
            assert f"peptide={ids[2]}" in page.evaluate("() => location.search")
            want = peptide_precursors(rs, GROUP, int(ids[2])).rows
            n = page.evaluate(
                "() => window.dash_ag_grid.getApi('pp-pre-grid').getDisplayedRowCount()"
            )
            assert n == len(want)
            # A click on the coverage lanes selects that peptide in the grid.
            page.locator(f'#pp-cov-body .mvc-lane-pep[data-pep="{ids[4]}"]').first.click()
            page.wait_for_function(ready, arg=seqs[ids[4]], timeout=15000)
            assert page.evaluate(sel) == ids[4]
            # Enter on the precursors grid opens the selected precursor's page.
            page.locator('#pp-pre-grid .ag-row[row-index="0"] [col-id="peptidoform"]').click()
            page.keyboard.press("Enter")
            page.wait_for_function("() => location.pathname.endsWith('/precursor')", timeout=10000)
            assert errors == []
            chromium.close()
    finally:
        server.shutdown()
