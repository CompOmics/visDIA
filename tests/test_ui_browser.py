"""The identification page (linked panels), on the fixtures.

Every row of every panel must be the data layer's (identification_table): the first
table with the page's filters, a child table with the parent's exact key and no q
filter. The address must reproduce the level, the filters and the selection, and a
refused query must answer the grid with no rows and the data layer's message.

The browser code (assets/browser.js) is tested twice: its pure functions in Node (the
address, the static "islands" of the preview, the selection rule the browser shares
with the server), and the page in Chromium (clicks, keys, Enter, Back). Each part is
skipped when Node or Chromium is not available.
"""

from __future__ import annotations

import json
import math
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dash.development.base_component import Component
from plotly.io.json import to_json_plotly

from mumdia_viewer.data.tables import Q_COLUMNS, TableQuery, identification_table
from mumdia_viewer.ui import browser, browser_grid, browser_panels
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURES = ["single", "experiment", "grouped", "topk", "mbr"]
UNITS = ["protein_group", "peptide", "precursor"]
# The fixtures' protein groups pass at q <= 0.1 only.
T = {"t": "0.1"}
BROWSER_JS = Path(browser.__file__).parent / "assets" / "browser.js"


def _walk(node):
    if isinstance(node, Component):
        yield node
        yield from _walk(getattr(node, "children", None))
        for prop in ("label", "leftSection", "rightSection", "title", "data", "right"):
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
    """The whole component tree as JSON text (nested components included, unescaped)."""
    return json.dumps(json.loads(to_json_plotly(tree)), ensure_ascii=False)


def _page(rs, query, base="/t/"):
    return browser.layout(PageContext(rs=rs, base=base, threshold=0.01, query=query))


def _first(rs, f):
    return browser.first_block(rs, "/", f)[1][0]


def _need(row, kind="group", **kw):
    need = {
        "kind": kind,
        "group": row.get("protein_group"),
        "peptide": row.get("base_peptide_id"),
        "run": row.get("run") or "",
        "cid": row.get("candidate_id"),
        "decoys": False,
        "n": 1,
    }
    return {**need, **kw}


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("unit", UNITS)
def test_layout_builds_the_panels_of_each_level(open_fixture, name, unit):
    rs = open_fixture(name)
    tree = _page(rs, {"unit": unit, **T})
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    common = ("ib-shell", "ib-grid", "ib-view", "ib-sel", "ib-prec", "ib-preview", "ib-unit")
    for required in common:
        assert json.dumps(required) in ids, required
    # The panels of each level: protein groups | peptides | precursors, then the preview.
    assert _has(tree, "ib-pep-grid") == (unit == "protein_group")
    assert _has(tree, "ib-pre-grid") == (unit != "precursor")
    # The child panels are asked for by the browser: stores at the levels that have them.
    assert _has(tree, "ib-need") == (unit != "precursor")
    assert _has(tree, "ib-prefetch") == (unit != "precursor")
    grid = _find(tree, "ib-grid")
    assert grid.rowModelType == "infinite"
    assert grid.dashGridOptions["maxConcurrentDatasourceRequests"] == 1
    assert grid.dashGridOptions["context"]["threshold"] == 0.1
    assert grid.dashGridOptions["rowHeight"] == browser_panels.ROW_HEIGHT == 26
    assert _find(tree, "ib-view").data["unit"] == unit
    root = _find(tree, "ib-root")
    assert root.__getattribute__("data-unit") == unit
    assert root.__getattribute__("data-base") == "/t/"
    # The drawer is gone: the preview panel (with its collapse handle) replaces it.
    assert not _has(tree, "ib-drawer")
    assert _has(tree, "ib-pv-toggle") and _has(tree, "ib-notice")


def test_layout_runs_only_the_first_block(open_fixture, monkeypatch):
    """The child panels are not queried while the page is built (they load after)."""
    rs = open_fixture("single")
    calls = []
    real = browser.identification_table

    def spy(rs_, query):
        calls.append(query)
        return real(rs_, query)

    monkeypatch.setattr(browser, "identification_table", spy)
    monkeypatch.setattr(browser.ahead, "ENABLED", False)
    tree = _page(rs, T)
    assert [q.unit for q in calls] == ["protein_group"]
    assert _find(tree, "ib-pep-grid").rowData == []
    assert "ib-loading" in _find(tree, "ib-panel-pep").className


@pytest.mark.parametrize("name", ["single", "experiment"])
@pytest.mark.parametrize("unit", UNITS)
def test_top_panel_shows_the_data_layer_total_and_q_column(open_fixture, name, unit):
    rs = open_fixture(name)
    page = identification_table(rs, TableQuery(unit=unit, threshold=0.1, limit=1))
    tree = _page(rs, {"unit": unit, **T})
    assert _find(tree, "ib-top-count").children == f"{page.total:,}"
    assert _find(tree, "ib-top-q").children == f"{page.q_column} ≤ 0.1"
    help_text = _text(_find(tree, "ib-top-help").label)
    # One plain line first (the count, its unit, its q column), the description below.
    noun = browser_panels.NOUNS[unit]
    assert f"{page.total:,} {noun}s with {page.q_column} ≤ 0.1" in help_text
    assert page.description[:60] in help_text


def test_refused_query_is_shown_with_the_data_layer_message(open_fixture):
    rs = open_fixture("experiment")
    tree = _page(rs, {"unit": "peptide", "in_run": "a"})
    with pytest.raises(Exception) as exc:
        identification_table(rs, TableQuery(unit="peptide", run="a"))
    empty = _find(tree, "ib-empty")
    assert "ib-empty-on" in empty.className
    assert str(exc.value)[:50] in _text(empty)
    assert _find(tree, "ib-top-count").children == "-"
    # No first row, so nothing is selected and the child panel is empty.
    assert _find(tree, "ib-sel").data == {"group": None, "peptide": None, "run": "", "cid": None}
    assert _find(tree, "ib-pre-grid").rowData == []
    assert _find(tree, "ib-prec").data is None


def test_grouped_levels_of_an_experiment_disable_run_and_quant(open_fixture):
    rs = open_fixture("experiment")
    tree = _page(rs, {"unit": "peptide"})
    assert _find(tree, "ib-run").disabled is True
    assert _find(tree, "ib-quant").disabled is True
    tree = _page(rs, {"unit": "precursor"})
    assert not _find(tree, "ib-run").disabled
    single = _page(open_fixture("single"), {})
    assert _find(single, "ib-run").style == {"display": "none"}


# --------------------------------------------------------------------------- selection


@pytest.mark.parametrize("name", ["single", "experiment", "mbr"])
def test_default_selection_follows_the_first_row(open_fixture, name):
    """No selection in the address: the first row of each panel is selected."""
    rs = open_fixture(name)
    f = browser.Filters.from_query(T, threshold=0.01)
    first = identification_table(rs, f.table_query(limit=1)).rows.iloc[0]
    sel = browser.initial_selection(f, browser.Selection(), _first(rs, f))
    group = str(first["protein_group"])
    assert sel.group == group
    # The group's row shown names the peptide and the precursor.
    assert sel.peptide == int(first["base_peptide_id"])
    assert sel.cid == int(first["candidate_id"])
    assert sel.run == (str(first["run"]) if rs.is_experiment else "")
    tree = _page(rs, T)
    assert browser.Selection.from_store(_find(tree, "ib-sel").data) == sel
    assert _find(tree, "ib-prec").data["cid"] == sel.cid
    # The children are the data layer's: every row of the parent, no q filter.
    p = browser.children_payload(rs, "/", _need(dict(first)))
    peps = identification_table(
        rs, TableQuery(unit="peptide", protein_group=group, threshold=None, limit=2000)
    )
    assert [r["base_peptide_id"] for r in p["pep"]["rows"]] == peps.rows["base_peptide_id"].tolist()
    assert p["pep"]["total"] == peps.total
    pres = identification_table(
        rs,
        TableQuery(unit="precursor", base_peptide_id=sel.peptide, threshold=None, limit=2000),
    )
    assert [r["candidate_id"] for r in p["pre"]["rows"]] == pres.rows["candidate_id"].tolist()
    assert "no q filter" in _text(p["pre"]["help"])
    assert p["sel"] == sel.store() and p["found"] is True
    assert p["prec"]["cid"] == sel.cid


def test_selection_from_the_address_is_kept_and_checked(open_fixture):
    rs = open_fixture("experiment")
    f = browser.Filters.from_query(T, threshold=0.01)
    groups = identification_table(rs, f.table_query(limit=50)).rows
    group = str(groups["protein_group"].iloc[-1])
    peps = browser.child_table(rs, "/", "peptide", group=group)
    peptide = int(peps.rows[-1]["base_peptide_id"])
    pres = browser.child_table(rs, "/", "precursor", peptide=peptide)
    target = pres.rows[-1]
    want = browser.Selection(group, peptide, target["run"], int(target["candidate_id"]))
    assert browser.initial_selection(f, want, _first(rs, f)) == want
    tree = _page(rs, {**T, **want.query("protein_group")})
    assert browser.Selection.from_store(_find(tree, "ib-sel").data) == want
    assert _find(tree, "ib-prec").data["cid"] == want.cid
    # The children callback checks the wanted rows against their parents.
    need = {**want.store(), "kind": "group", "decoys": False, "n": 3}
    p = browser.children_payload(rs, "/", need)
    assert p["sel"] == want.store() and p["n"] == 3
    # A peptide that is not in the group: the group's first peptide and its first row.
    p = browser.children_payload(rs, "/", {**need, "peptide": -5})
    assert p["sel"]["peptide"] == peps.rows[0]["base_peptide_id"]
    assert p["sel"]["cid"] == p["pre"]["rows"][0]["candidate_id"]
    # A precursor that is not a row of the peptide: the peptide's first (best) row.
    p = browser.children_payload(rs, "/", {**need, "cid": -7})
    assert p["sel"]["peptide"] == peptide
    assert p["sel"]["cid"] == pres.rows[0]["candidate_id"]


def test_an_unknown_group_in_the_address_is_not_found(open_fixture):
    rs = open_fixture("single")
    f = browser.Filters.from_query(T, threshold=0.01)
    first = _first(rs, f)
    # The page starts from the address; the children answer says the group has no rows
    # and the browser falls back to the first row with a notice.
    sel = browser.initial_selection(f, browser.Selection(group="NOT_A_GROUP"), first)
    assert sel.group == "NOT_A_GROUP" and sel.peptide is None and sel.cid is None
    p = browser.children_payload(rs, "/", {"kind": "group", "group": "NOT_A_GROUP", "n": 1})
    assert p["found"] is False and p["pep"]["rows"] == [] and p["pre"] is None
    p = browser.children_payload(rs, "/", {"kind": "peptide", "peptide": 123456789, "n": 1})
    assert p["found"] is False


@pytest.mark.parametrize("unit", ["peptide", "precursor"])
def test_other_levels_select_the_first_row(open_fixture, unit):
    rs = open_fixture("single")
    f = browser.Filters(unit=unit)
    first = identification_table(rs, f.table_query(limit=1)).rows.iloc[0]
    sel = browser.initial_selection(f, browser.Selection(), _first(rs, f))
    assert sel.group is None
    assert sel.cid == int(first["candidate_id"])
    assert (sel.peptide is None) == (unit == "precursor")
    # The address's precursor is kept as given (the preview checks it).
    want = browser.Selection(cid=99, run="")
    assert browser.initial_selection(browser.Filters(unit="precursor"), want, None).cid == 99


def test_child_tables_follow_the_decoy_switch(open_fixture):
    rs = open_fixture("single")
    page = identification_table(
        rs, TableQuery(unit="precursor", include_decoys=True, threshold=None, limit=5000)
    )
    decoys = page.rows[page.rows["label"] == "decoy"]
    peptide = int(decoys["base_peptide_id"].iloc[0])
    hidden = browser.child_table(rs, "/", "precursor", peptide=peptide)
    shown = browser.child_table(rs, "/", "precursor", peptide=peptide, decoys=True)
    assert all(r["label"] == "target" for r in hidden.rows)
    assert {r["label"] for r in shown.rows} >= {"decoy"}


def test_child_rows_carry_only_their_columns(open_fixture):
    """A child grid carries the columns it can show and the keys, not the protein text."""
    rs = open_fixture("experiment")
    f = browser.Filters(t=0.1)
    group = _first(rs, f)["protein_group"]
    child = browser.child_table(rs, "/", "peptide", group=group)
    keys = set(child.rows[0])
    assert {"protein", "protein_group", "source"}.isdisjoint(keys)
    assert {"base_peptide_id", "candidate_id", "run", "peptidoform", "label", "_key"} <= keys
    assert set(Q_COLUMNS) & keys == set(Q_COLUMNS) & set(child.columns)
    # The child grid defines exactly the columns its rows carry.
    defs = browser._child_defs(rs, "peptide", browser_grid.scales_of(rs), None, 0.1)
    cols = {d["colId"] for d in defs} - {"_valid", "_open"}
    assert cols == set(child.columns)


def test_long_child_tables_load_in_blocks(open_fixture, monkeypatch):
    rs = open_fixture("single")
    f = browser.Filters(t=0.1)
    group = _first(rs, f)["protein_group"]
    whole = browser.child_table(rs, "/", "peptide", group=group)
    monkeypatch.setattr(browser, "CHILD_MAX", 3)
    p = browser.children_payload(rs, "/", {"kind": "group", "group": group, "n": 1})
    assert len(p["pep"]["rows"]) == 3 and p["pep"]["total"] == whole.total > 3
    assert p["pep"]["note"] == f"3 of {whole.total:,} loaded · any q"
    keys = [r["_key"] for r in p["pep"]["rows"]]
    offset = 3
    while offset < whole.total:
        more = browser.children_payload(
            rs, "/", {"kind": "more", "unit": "peptide", "group": group, "offset": offset, "n": 1}
        )["more"]
        assert more["offset"] == offset and more["key"] == group
        keys += [r["_key"] for r in more["rows"]]
        offset += len(more["rows"])
    assert keys == [r["_key"] for r in whole.rows]


def test_locate_finds_a_row_of_the_first_table(open_fixture, monkeypatch):
    rs = open_fixture("single")
    f = browser.Filters(unit="precursor", t=0.1, sort="apex_rt", desc=False)
    rows = identification_table(rs, f.table_query(limit=40)).rows
    key = browser_grid.row_key("precursor", {"run": "", "candidate_id": rows["candidate_id"][17]})
    assert browser.locate_index(rs, f.store(), key, ["apex_rt", False]) == 17
    monkeypatch.setattr(browser, "LOCATE_BLOCK", 5)
    assert browser.locate_index(rs, f.store(), key, ["apex_rt", False]) == 17
    assert browser.locate_index(rs, f.store(), ":-1", None) is None
    # The vectorised row ids are the grid's.
    for unit in UNITS:
        page = identification_table(rs, browser.Filters(unit=unit, t=0.1).table_query(limit=20))
        recs = browser_grid.records(page.rows, "/", unit)
        assert browser.row_keys(unit, page.rows).tolist() == [r["_key"] for r in recs]


# --------------------------------------------------------------------------- address


def test_filters_round_trip_through_the_address():
    views = [
        browser.Filters(),
        browser.Filters(unit="peptide", q="q_value", search="LGE", charge="3", decoys=True),
        browser.Filters(protein="_YEAST", mod="Oxidation", quant="not_selected", t=0.05),
        browser.Filters(unit="precursor", run="r1", q="q_value", sort="apex_rt", desc=False),
    ]
    for f in views:
        assert browser.Filters.from_query(f.query(), threshold=0.01) == f
        assert browser.Filters.from_store(f.store()) == f
    # The run filter is in_run; run is the selected precursor's run.
    assert browser.Filters(unit="precursor", run="r1").query()["in_run"] == "r1"
    assert "unit" not in browser.Filters().query()


def test_filters_from_a_bad_address_fall_back():
    f = browser.Filters.from_query(
        {"unit": "psm", "q": "nope", "charge": "x", "t": "0.03", "decoys": "maybe"}, threshold=0.02
    )
    assert f.unit == "protein_group" and f.q == "" and f.charge == "" and not f.decoys
    # Only the header's stops are thresholds; otherwise the header's value holds.
    assert f.t == 0.02
    assert browser.Filters.from_query({"t": "0.05"}, threshold=0.01).t == 0.05
    # The default column, chosen explicitly, is the default.
    assert browser.Filters.from_query({"q": "pg_q_value"}, threshold=0.01).q == ""
    query = {"unit": "precursor", "q": "run_psm_q", "in_run": "a"}
    assert browser.Filters.from_query(query, threshold=0.01).q == ""
    query = {"unit": "precursor", "q": "precursor_q", "in_run": "a"}
    assert browser.Filters.from_query(query, threshold=0.01).q == "precursor_q"


def test_selection_round_trip_and_level_keys():
    sel = browser.Selection(group="A_HUMAN;B_HUMAN", peptide=12, run="r2", cid=345)
    assert browser.Selection.from_query(sel.query("protein_group")) == sel
    assert sel.query("peptide") == {"peptide": "12", "run": "r2", "cid": "345"}
    assert sel.query("precursor") == {"run": "r2", "cid": "345"}
    assert browser.Selection.from_store(sel.store()) == sel
    # Without a candidate id a run means nothing; bad numbers are dropped.
    assert browser.Selection.from_query({"run": "r1", "peptide": "x"}) == browser.Selection()
    assert browser.Selection.from_query({"cid": "12.0"}).cid == 12


def test_table_query_and_sort():
    f = browser.Filters(
        unit="peptide", q="q_value", charge="2", protein="X", mod="Ox", quant="quantified"
    )
    q = f.table_query(offset=200, limit=100, sort=("apex_rt", False))
    assert q == TableQuery(
        unit="peptide",
        q_column="q_value",
        threshold=0.01,
        charge=2,
        protein="X",
        modification="Ox",
        quant_status="quantified",
        sort_by="apex_rt",
        descending=False,
        offset=200,
        limit=100,
    )
    assert browser.sort_of({"sortModel": [{"colId": "score", "sort": "asc"}]}, f) == (
        "score",
        False,
    )
    assert browser.sort_of({"sortModel": []}, f) == ("score", True)
    precursor = browser.Filters(unit="precursor", run="a")
    assert browser.q_active(precursor, experiment=True) == "run_psm_q"
    assert browser.q_active(browser.Filters(), experiment=True) == "pg_q_value"


def test_chips_name_every_active_filter():
    assert browser_panels.chip_row(browser.Filters()) == []
    f = browser.Filters(unit="precursor", search="K", charge="3", run="a", decoys=True)
    f = browser.Filters(**{**f.store(), "quant": "quantified"})
    keys = [c.id["key"] for c in browser_panels.chip_row(f)]
    # The popover's filters first, then the search and the decoy switch.
    assert keys == ["charge", "quant", "run", "search", "decoys"]
    assert browser_panels.n_popover_filters(f) == 3


def test_row_filters_of_a_grouped_level_say_they_test_the_winning_row():
    f = browser.Filters(charge="3", search="PLEC")
    assert browser.fast_mode(f)
    texts = [_text(c) for c in browser_panels.chip_row(f, fast=True)]
    assert all("(winning row)" in t for t in texts)
    # Another q column: the filters select rows, a group is listed when one passes.
    g = browser.Filters(charge="3", q="q_value")
    assert not browser.fast_mode(g)
    assert not browser.fast_mode(browser.Filters(unit="precursor", charge="3"))
    assert "(winning row)" not in _text(browser_panels.chip_row(g, fast=False))
    tree = browser_panels.toolbar(
        _FakeRs(),
        f,
        default_q="pg_q_value",
        charges=[2, 3],
        statuses=[],
        chips=[],
        n_filters=1,
        fast=True,
    )
    note = _find(tree, "ib-fast-note")
    assert note.children == browser_panels.FAST_NOTE and "ib-hidden" not in note.className


class _FakeRs:
    is_experiment = False
    runs = ()


def test_q_options_hold_the_seven_columns_and_follow_the_run():
    options = browser_panels.q_options(True, "a", "precursor", "run_psm_q")
    items = [i for g in options for i in g["items"]]
    assert sorted(i["value"] for i in items) == sorted(Q_COLUMNS)
    by = {i["value"]: i for i in items}
    assert by["run_psm_q"]["label"].endswith("(default)")
    assert all(by[c]["disabled"] for c in ("precursor_q", "peptide_q_value", "pg_q_value"))
    plain = browser_panels.q_options(False, "", "peptide", "peptide_q_value")
    plain_items = {i["value"]: i for g in plain for i in g["items"]}
    assert plain_items["peptide_q_value"]["label"].endswith("(default)")
    assert not any(i["disabled"] for i in plain_items.values())


# --------------------------------------------------------------------------- rows


@pytest.mark.parametrize("name", ["single", "experiment", "mbr"])
@pytest.mark.parametrize("unit", UNITS)
def test_rows_are_the_data_layer_rows_in_its_order(open_fixture, name, unit):
    rs = open_fixture(name)
    f = browser.Filters(unit=unit, decoys=True, t=0.1)
    request = {"startRow": 0, "endRow": 40, "sortModel": [{"colId": "apex_rt", "sort": "asc"}]}
    response, _, total, error, q = browser.rows_response(rs, request, f.store(), "/t/")
    assert error is None
    want = identification_table(rs, f.table_query(offset=0, limit=40, sort=("apex_rt", False)))
    assert response["rowCount"] == want.total == total
    rows = response["rowData"]
    assert [r["candidate_id"] for r in rows] == want.rows["candidate_id"].tolist()
    assert q == want.q_column
    for r in rows:
        assert r["_key"] == browser_grid.row_key(unit, r)
        assert r["_href"].startswith("/t/precursor?")
        assert f"cid={r['candidate_id']}" in r["_href"]
        if rs.is_experiment:
            assert f"run={r['run']}" in r["_href"]
        json.dumps(r)  # plain JSON values only
    assert len({r["_key"] for r in rows}) == len(rows)


def test_rows_page_through_the_whole_table(open_fixture):
    rs = open_fixture("single")
    f = browser.Filters(unit="precursor", decoys=True, t=0.1)
    seen = []
    start = 0
    while True:
        response, *_ = browser.rows_response(
            rs, {"startRow": start, "endRow": start + 50}, f.store(), "/"
        )
        seen += [r["_key"] for r in response["rowData"]]
        if start + 50 >= response["rowCount"]:
            break
        start += 50
    assert len(seen) == len(set(seen)) == response["rowCount"]


def test_a_refused_query_answers_with_no_rows(open_fixture):
    rs = open_fixture("single")
    f = browser.Filters(sort="nope")
    response, page, total, error, _ = browser.rows_response(rs, {"startRow": 0}, f.store(), "/")
    assert response == {"rowData": [], "rowCount": 0}
    assert page is None and total is None
    assert error.startswith("cannot sort the protein_group table by 'nope'")
    # A sort from the grid replaces the address's.
    request = {"startRow": 0, "endRow": 10, "sortModel": [{"colId": "score", "sort": "desc"}]}
    f = browser.Filters(sort="nope", t=0.1)
    response, *_ = browser.rows_response(rs, request, f.store(), "/")
    assert response["rowCount"] > 0


def test_records_hold_plain_json_values_and_row_keys():
    df = pd.DataFrame(
        {
            "candidate_id": np.array([5, 7], dtype=np.int64),
            "source": np.array([0, 1], dtype=np.uint32),
            "run": [None, "r1"],
            "protein_group": ["A_HUMAN", "B_YEAST;C_YEAST"],
            "base_peptide_id": np.array([11, 12], dtype=np.int64),
            "quantity": [np.nan, 2.5],
            "is_winner": np.array([True, False]),
            "quant_status": [None, "quantified"],
        }
    )
    rows = browser_grid.records(df, "/b/", "precursor")
    assert rows[0]["quantity"] is None and rows[1]["quantity"] == 2.5
    assert rows[0]["is_winner"] is True and isinstance(rows[0]["candidate_id"], int)
    assert rows[0]["run"] == "" and rows[0]["_key"] == ":5" and rows[1]["_key"] == "r1:7"
    assert rows[0]["_href"] == "/b/precursor?cid=5"
    assert rows[1]["_href"] == "/b/precursor?run=r1&cid=7"
    assert [r["_key"] for r in browser_grid.records(df, "/", "peptide")] == ["11", "12"]
    groups = browser_grid.records(df, "/", "protein_group")
    assert [r["_key"] for r in groups] == ["A_HUMAN", "B_YEAST;C_YEAST"]
    json.dumps(rows)
    # Only some columns (a child grid's): the keys and the address still come.
    some = browser_grid.records(df, "/", "precursor", ["candidate_id", "run", "quantity"])
    assert set(some[1]) == {"candidate_id", "run", "quantity", "_key", "_href"}
    assert some[1]["_key"] == "r1:7"


# --------------------------------------------------------------------------- columns


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("role", ["top", "child"])
def test_column_defs(open_fixture, unit, role):
    rs = open_fixture("experiment")
    query = TableQuery(unit=unit, limit=1, threshold=0.1 if role == "top" else None)
    page = identification_table(rs, query)
    active = page.q_column if role == "top" else None
    scales = browser_grid.scales_of(rs)
    defs = browser_grid.column_defs(
        unit,
        list(page.rows.columns),
        page.column_labels,
        role=role,
        experiment=True,
        q_active=active,
        threshold=0.1 if role == "top" else None,
        winner_matters=role == "child",
        scales=scales,
        marks_at=None if role == "top" else 0.05,
    )
    by = {d["colId"]: d for d in defs}
    # The validation mark first, the open icon last, both pinned.
    assert defs[0]["colId"] == "_valid" and defs[0]["pinned"] == "left"
    assert defs[-1]["colId"] == "_open" and defs[-1]["pinned"] == "right"
    mark = browser_grid.validation_column(unit, role, True, active)
    assert defs[0]["cellRendererParams"]["qField"] == mark
    assert mark in defs[0]["headerTooltip"]
    # The mark's tooltip names the threshold it tests; its template has "{t}".
    t = "0.1" if role == "top" else "0.05"
    assert f"{mark} ≤ {t} (the header threshold)" in defs[0]["headerTooltip"]
    assert "{t}" in defs[0]["cellRendererParams"]["tipTemplate"]
    extra = {"_valid", "_open"} | ({"_species"} if unit == "protein_group" else set())
    assert set(by) - extra == set(page.rows.columns)
    key = "protein_group" if unit == "protein_group" else "peptidoform"
    assert by[key]["lockVisible"] is True
    # Every q column has a bar on the fixed -log10 scale and the data layer's words.
    for c in Q_COLUMNS:
        assert page.column_labels[c] in by[c]["headerTooltip"]
        params = by[c]["cellRendererParams"]
        assert by[c]["cellRenderer"] == "IbBar"
        assert (params["scale"], params["min"], params["max"]) == ("neglog10", 1.0, 1e-4)
        assert "-log10(q)" in by[c]["headerTooltip"]
        assert params["width"] == (30 if role == "top" else 24)
    # The score bar is scaled on the scored table's range, the same for every row.
    lo, hi = scales.score
    assert (by["score"]["cellRendererParams"]["min"], by["score"]["cellRendererParams"]["max"]) == (
        lo,
        hi,
    )
    if role == "top":
        assert by[active]["hide"] is False and "initialHide" not in by[active]
        assert by[active]["headerTooltip"].startswith(f"Filter column of this table: {active} <=")
    # Counts at any q say so and have a grey bar; counts at the threshold do not.
    for c in ("n_precursors", "n_runs", "n_peptides"):
        if c not in by:
            continue
        anyq = c == "n_precursors" or role == "child"
        assert ("ib-head-anyq" in str(by[c].get("headerClass"))) == anyq, c
        colour = by[c]["cellRendererParams"]["colour"]
        assert (colour == "var(--ib-bar-anyq)") == anyq, c
    # Columns that may go when the grid does not fit say when (and only shown ones).
    for d in defs:
        order = (d.get("context") or {}).get("hideOrder")
        if order is not None:
            assert order == browser_grid.hide_order(unit, role, d["colId"])
            assert not d.get("initialHide")
    # The Columns menu lists every column but the fixed ones and the key.
    menu = [o["value"] for o in browser_grid.column_options(defs)]
    assert key not in menu and "_valid" not in menu and "_open" not in menu
    shown = browser_grid.visible_columns(defs)
    assert set(shown) <= set(menu)


def test_columns_menu_of_a_single_run_leaves_out_run_and_source(open_fixture):
    rs = open_fixture("single")
    page = identification_table(rs, TableQuery(unit="precursor", limit=1))
    defs = browser_grid.column_defs(
        "precursor",
        list(page.rows.columns),
        page.column_labels,
        experiment=False,
        q_active=page.q_column,
        threshold=0.01,
        winner_matters=False,
    )
    single = [o["value"] for o in browser_grid.column_options(defs, experiment=False)]
    both = [o["value"] for o in browser_grid.column_options(defs, experiment=True)]
    assert "run" not in single and "source" not in single
    assert {"run", "source"} <= set(both)
    peptide = identification_table(rs, TableQuery(unit="peptide", limit=1))
    defs = browser_grid.column_defs(
        "peptide",
        list(peptide.rows.columns),
        peptide.column_labels,
        experiment=False,
        q_active=peptide.q_column,
        threshold=0.01,
        winner_matters=False,
    )
    labels = {o["value"]: o["label"] for o in browser_grid.column_options(defs)}
    assert labels["n_precursors"] == "precursors (any q)"


def test_validation_columns():
    vc = browser_grid.validation_column
    assert vc("precursor", "child", False, None) == "q_value"
    assert vc("precursor", "child", True, None) == "run_psm_q"
    assert vc("peptide", "child", True, None) == "peptide_q_value"
    assert vc("protein_group", "top", True, "q_value") == "q_value"


def test_default_columns_per_table():
    cols = [
        "protein_group",
        "peptidoform",
        "charge",
        "run",
        "protein",
        "score",
        "q_value",
        "run_psm_q",
        "peptide_q_value",
        "pg_q_value",
        "n_peptides",
        "n_precursors",
        "n_runs",
        "apex_rt",
        "quantity",
        "quant_state",
        "is_winner",
        "candidate_id",
    ]
    dc = browser_grid.default_columns
    pg = dc(
        "protein_group", "top", cols, experiment=False, q_column="pg_q_value", winner_matters=False
    )
    # The q column the marks test comes right after the key column.
    assert pg[:5] == ["protein_group", "_species", "pg_q_value", "n_peptides", "n_runs"]
    assert "is_winner" not in pg and "candidate_id" not in pg
    pep = dc(
        "peptide", "child", cols, experiment=True, q_column="peptide_q_value", winner_matters=True
    )
    assert pep[:4] == ["peptidoform", "peptide_q_value", "score", "n_precursors"]
    assert "is_winner" in pep and "run" in pep and "protein" not in pep
    pre = dc("precursor", "child", cols, experiment=True, q_column="run_psm_q", winner_matters=True)
    assert pre[:5] == ["peptidoform", "charge", "run", "run_psm_q", "score"]
    top = dc(
        "protein_group", "top", cols, experiment=False, q_column="q_value", winner_matters=True
    )
    assert top[2:4] == ["pg_q_value", "q_value"]


def test_bar_scales_are_stated():
    scales = browser_grid.Scales(score=(0.0, 1.0), quantity=(10.0, 1e6), pg_quantity=None, n_runs=6)
    params, tip = browser_grid.bar_params("n_peptides", "protein_group", scales)
    assert (params["scale"], params["min"], params["max"]) == ("log10", 1, 1000)
    assert "log10" in tip
    params, tip = browser_grid.bar_params("n_runs", "peptide", scales)
    assert params["max"] == 6 and "6 runs" in tip
    params, tip = browser_grid.bar_params("n_runs", "peptide", scales, anyq=True, width=24)
    assert params["colour"] == "var(--ib-bar-anyq)" and params["width"] == 24
    assert "at any q" in tip
    params, tip = browser_grid.bar_params("quantity", "precursor", scales)
    assert (params["min"], params["max"]) == (10.0, 1e6) and "peptide_quant" in tip
    assert "never 0" in tip and "quant state" in tip
    # No protein_group_quant range (an experiment): no bar.
    assert browser_grid.bar_params("quantity", "protein_group", scales) is None
    assert browser_grid.bar_params("apex_rt", "precursor", scales) is None
    assert browser_grid.any_q("n_precursors", 0.01) and browser_grid.any_q("n_runs", None)
    assert not browser_grid.any_q("n_runs", 0.01)


# --------------------------------------------------------------------------- facets


def test_facets(open_fixture):
    rs = open_fixture("single")
    fac = browser_grid.facets(rs)
    page = identification_table(rs, TableQuery(threshold=None, include_decoys=True, limit=10000))
    assert set(fac.charges) == set(page.rows["charge"].astype(int))
    assert math.isclose(fac.score_hi, float(page.rows["score"].max()))
    q = page.rows["quantity"].dropna()
    q = q[q > 0]
    assert fac.quantity is not None
    assert fac.quantity[0] <= float(q.min()) and fac.quantity[1] >= float(q.max())
    assert browser_grid.facets(rs) is fac
    mods = browser_grid.modifications(rs)
    assert all(isinstance(m, str) and m for m in mods)
    tokens = page.rows["peptidoform"].str.findall(r"\[([^\]]*)\]").explode().dropna()
    assert set(mods) >= set(tokens)
    columns, labels = browser_grid.child_meta(rs, "peptide")
    assert browser_grid.child_meta(rs, "peptide") == (columns, labels)
    assert "peptidoform" in columns and "any q" in labels.get("n_precursors", "any q")


# --------------------------------------------------------------------------- preview


def test_preview_embeds_the_detail_preview(open_fixture):
    rs = open_fixture("single")
    f = browser.Filters(unit="precursor")
    first = _first(rs, f)
    ctx = PageContext(rs=rs, base="/")
    card, ok = browser_panels.preview(ctx, browser.prec_data(first))
    assert getattr(card, "id", None) == "pv-card" and ok is True
    empty, ok = browser_panels.preview(ctx, None)
    assert "No precursor is selected" in _text(empty) and ok is True
    # A candidate this result set lacks: the card says so, and the page falls back.
    _, ok = browser_panels.preview(ctx, {"run": "", "cid": 999999999})
    assert ok is False
    assert browser_panels.warm_detail(rs, "", first["candidate_id"]) is True
    assert browser_panels.warm_detail(rs, "", 999999999) is False


def test_the_first_precursor_is_read_ahead(open_fixture, monkeypatch):
    rs = open_fixture("topk")
    from mumdia_viewer.ui import browser_ahead, detail

    f = browser.Filters(unit="precursor", t=0.1)
    first = _first(rs, f)
    seen = []
    real = browser_panels.warm_detail

    def warm(rs_, run, cid):
        seen.append((run, cid))
        return real(rs_, run, cid)

    monkeypatch.setattr(browser_panels, "warm_detail", warm)
    monkeypatch.setattr(browser_ahead, "_JOBS", type(browser_ahead._JOBS)())
    _page(rs, {"unit": "precursor", **T})
    browser_ahead.wait_detail(rs, "", first["candidate_id"])
    assert seen == [("", first["candidate_id"])]
    assert detail._is_cached(rs, "", first["candidate_id"])
    # Off: nothing is read ahead.
    monkeypatch.setattr(browser_ahead, "ENABLED", False)
    monkeypatch.setattr(browser_ahead, "_JOBS", type(browser_ahead._JOBS)())
    seen.clear()
    _page(rs, {"unit": "precursor", "sort": "apex_rt", **T})
    assert seen == []


def test_preview_of_an_experiment_needs_its_run(open_fixture):
    rs = open_fixture("experiment")
    first = _first(rs, browser.Filters(unit="precursor"))
    ctx = PageContext(rs=rs, base="/")
    _, ok = browser_panels.preview(ctx, browser.prec_data(first))
    assert ok is True
    _, ok = browser_panels.preview(ctx, {"run": "", "cid": first["candidate_id"]})
    assert ok is False


def test_a_failing_preview_does_not_stop_the_page(open_fixture, monkeypatch):
    rs = open_fixture("single")
    from mumdia_viewer.ui import detail

    def broken(ctx, run, cid):
        raise RuntimeError("boom")

    monkeypatch.setattr(detail, "preview", broken)
    card, _ = browser_panels.preview(PageContext(rs=rs, base="/"), {"run": "", "cid": 5, "row": {}})
    text = _text(card)
    assert "RuntimeError: boom" in text and "Open precursor page" in text


def test_preview_falls_back_to_a_summary(open_fixture, monkeypatch):
    rs = open_fixture("single")
    from mumdia_viewer.ui import detail

    monkeypatch.delattr(detail, "preview")
    prec = {"run": "", "cid": 5, "row": {"peptidoform": "PEM[Oxidation]K", "charge": 2}}
    card, _ = browser_panels.preview(PageContext(rs=rs, base="/x/"), prec)
    text = _text(card)
    assert "Open precursor page" in text and "/x/precursor?cid=5" in text
    assert "not available" in text


# --------------------------------------------------------------------------- callbacks


def _callback_key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _outputs(key: str) -> list[dict[str, str]]:
    out = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        prop = prop.split("@")[0]
        out.append({"id": cid, "property": prop})
    return out


def _post(client, key, inputs, state=()):
    outputs = _outputs(key)
    body = {
        "output": key,
        # A callback with one output takes it as one object, several as a list.
        "outputs": outputs if key.startswith("..") else outputs[0],
        "inputs": list(inputs),
        "changedPropIds": [f"{i['id']}.{i['property']}" for i in inputs],
        "state": list(state),
    }
    resp = client.post("/_dash-update-component", json=body)
    assert resp.status_code in (200, 204), resp.data[:800]
    return resp.get_json()["response"] if resp.status_code == 200 else {}


def test_rows_callback_answers_the_grid(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    key = _callback_key(app, "ib-grid.getRowsResponse")
    f = browser.Filters(t=0.1)
    request = {"startRow": 0, "endRow": 30, "sortModel": [{"colId": "score", "sort": "desc"}]}
    state = [
        {"id": "ib-view", "property": "data", "value": f.store()},
        {"id": "ib-sig", "property": "data", "value": "old view"},
        {"id": "ib-cols", "property": "value", "value": ["score"]},
        {"id": "ib-defs", "property": "data", "value": None},
    ]
    inputs = [{"id": "ib-grid", "property": "getRowsRequest", "value": request}]
    out = _post(client, key, inputs, state)
    want = identification_table(rs, f.table_query(offset=0, limit=30))
    grid = out["ib-grid"]
    assert grid["getRowsResponse"]["rowCount"] == want.total
    assert len(grid["getRowsResponse"]["rowData"]) == min(30, want.total)
    # A new view brings the count, the q column, the column definitions and the first
    # block, which the browser uses to keep or move the selection.
    assert out["ib-top-count"]["children"] == f"{want.total:,}"
    assert out["ib-top-q"]["children"] == "pg_q_value ≤ 0.1"
    assert want.description[:40] in json.dumps(out["ib-top-help"], ensure_ascii=False)
    assert "columnDefs" in grid
    first = out["ib-first"]["data"]
    assert first["row"]["protein_group"] == want.rows["protein_group"].iloc[0]
    assert first["keys"] == want.rows["protein_group"].tolist()
    # The same view again (a scroll): only the rows.
    state[1]["value"] = browser.shown_key_of(f, None)
    state[3]["value"] = out["ib-defs"]["data"]
    inputs[0]["value"] = {"startRow": 0, "endRow": 30, "sortModel": []}
    out = _post(client, key, inputs, state)
    assert set(out) == {"ib-grid"} and set(out["ib-grid"]) == {"getRowsResponse"}


def test_rows_callback_answers_a_refused_query(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    key = _callback_key(app, "ib-grid.getRowsResponse")
    f = browser.Filters(unit="peptide", run="a")
    state = [
        {"id": "ib-view", "property": "data", "value": f.store()},
        {"id": "ib-sig", "property": "data", "value": None},
        {"id": "ib-cols", "property": "value", "value": []},
        {"id": "ib-defs", "property": "data", "value": None},
    ]
    inputs = [{"id": "ib-grid", "property": "getRowsRequest", "value": {"startRow": 0}}]
    out = _post(client, key, inputs, state)
    assert out["ib-grid"]["getRowsResponse"] == {"rowData": [], "rowCount": 0}
    assert "experiment-wide" in json.dumps(out["ib-empty"])
    assert "ib-empty-on" in out["ib-empty"]["className"]
    assert out["ib-first"]["data"]["row"] is None


def test_children_callback_fills_both_panels_in_one_answer(open_fixture):
    rs = open_fixture("experiment")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    f = browser.Filters(t=0.1)
    first = _first(rs, f)
    key = _callback_key(app, "ib-children.data")
    need = _need(first, n=7)
    out = _post(client, key, [{"id": "ib-need", "property": "data", "value": need}])
    p = out["ib-children"]["data"]
    want = identification_table(
        rs,
        TableQuery(
            unit="peptide", protein_group=first["protein_group"], threshold=None, limit=2000
        ),
    )
    assert [r["base_peptide_id"] for r in p["pep"]["rows"]] == want.rows["base_peptide_id"].tolist()
    assert p["n"] == 7 and p["kind"] == "group" and p["found"] is True
    assert p["sel"]["cid"] == first["candidate_id"] and p["sel"]["run"] == first["run"]
    assert p["pre"]["mark"] == "run_psm_q" and p["pep"]["mark"] == "peptide_q_value"
    # An empty request (a store that mounts without data) answers nothing.
    out = _post(client, key, [{"id": "ib-need", "property": "data", "value": None}])
    assert out == {}


def test_prefetch_callback_answers_ahead(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    f = browser.Filters(t=0.1)
    rows = browser.first_block(rs, "/", f)[1]
    key = _callback_key(app, "ib-prefetched.data")
    needs = [_need(rows[1]), _need(rows[2]), _need(rows[3])]
    out = _post(
        client, key, [{"id": "ib-prefetch", "property": "data", "value": {"needs": needs, "n": 2}}]
    )
    data = out["ib-prefetched"]["data"]
    assert data["n"] == 2 and len(data["payloads"]) == browser.PREFETCH_MAX
    assert data["payloads"][0]["pep"]["key"] == rows[1]["protein_group"]


def test_preview_callback_answers_the_browser_only(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    first = _first(rs, browser.Filters(unit="precursor"))
    key = _callback_key(app, "ib-preview-data.data")
    # The browser's request: the threshold of the marks comes with it.
    prec = {**browser.prec_data(first), "t": 0.05, "n": 4}
    out = _post(
        client,
        key,
        [{"id": "ib-prec", "property": "data", "value": prec}],
        [{"id": "scheme", "property": "data", "value": "dark"}],
    )
    data = out["ib-preview-data"]["data"]
    assert data["card"]["props"]["id"] == "pv-card"
    assert (data["ok"], data["t"], data["n"], data["cid"]) == (True, 0.05, 4, first["candidate_id"])
    # The threshold is no input of the preview (a threshold change rebuilds no card).
    spec = next(d for d in client.get("/_dash-dependencies").get_json() if d["output"] == key)
    assert [i["id"] for i in spec["inputs"]] == ["ib-prec"]
    # Not a request of the browser's (no request number): no card.
    out = _post(
        client,
        key,
        [{"id": "ib-prec", "property": "data", "value": browser.prec_data(first)}],
        [{"id": "scheme", "property": "data", "value": None}],
    )
    assert out == {}


def test_locate_callback(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    f = browser.Filters(t=0.1)
    rows = browser.first_block(rs, "/", f)[1]
    key = _callback_key(app, "ib-located.data")
    req = {"key": rows[4]["_key"], "sort": ["score", True], "n": 9}
    out = _post(
        client,
        key,
        [{"id": "ib-locate-req", "property": "data", "value": req}],
        [{"id": "ib-view", "property": "data", "value": f.store()}],
    )
    assert out["ib-located"]["data"] == {"n": 9, "key": rows[4]["_key"], "index": 4}


def test_rebuild_callback(open_fixture):
    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    # An address the router did not rebuild: the page builds it in place.
    key = _callback_key(app, "ib-shell.children")
    out = _post(
        client,
        key,
        [{"id": "ib-address", "property": "data", "value": "?unit=peptide&t=0.1"}],
        [
            {"id": "threshold", "property": "data", "value": 0.01},
            {"id": "scheme", "property": "data", "value": None},
        ],
    )
    text = json.dumps(out["ib-shell"]["children"])
    assert '"data-unit": "peptide"' in text and "ib-pre-grid" in text


def test_app_registers_the_page_callbacks(open_fixture):
    app = create_app(open_fixture("single"), url_base="/abc/")
    client = app.server.test_client()
    deps = json.dumps(client.get("/abc/_dash-dependencies").get_json())
    for needle in (
        "ib-grid.getRowsResponse",
        "ib-view.data",
        "ib-children.data",
        "ib-prefetched.data",
        "ib-preview-data.data",
        "ib-located.data",
        "ib-shell.children",
        "ib-mod.data",
    ):
        assert needle in deps, needle


def test_help_bodies_lead_with_one_plain_line(open_fixture):
    rs = open_fixture("single")
    f = browser.Filters(t=0.1)
    child = browser.child_table(rs, "/", "peptide", group=_first(rs, f)["protein_group"])
    body = browser.child_help(child, "peptide", "peptide_q_value")
    line = body.children[0]
    assert line.className == "ib-help-line"
    assert line.children.startswith("Every peptide of the selected protein group, passing or not")
    # The data layer's description follows in smaller print; the key hints are gone.
    assert body.children[-1].className == "ib-help-small"
    assert body.children[-1].children == child.description
    assert "Enter" not in _text(body)


# --------------------------------------------------------------------------- browser.js in Node

NODE = shutil.which("node")

NODE_TEST = r"""
const fs = require('fs');
const noop = () => {};
global.window = {
  addEventListener: noop,
  localStorage: { getItem: () => null, setItem: noop },
  dash_clientside: {},
  history: {},
  location: { search: '', pathname: '/identifications' },
};
global.document = {
  addEventListener: noop,
  documentElement: { setAttribute: noop },
  getElementById: () => null,
};
eval(fs.readFileSync(process.argv[2], 'utf8'));
const M = window.mvbInternals;
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = {};
out.addresses = cases.addresses.map((c) => M.addressOf(c.view, c.sel, c.sort));
out.norm = M.norm('?b=2&a=1&a=3&c=');
// Islands: a static table (with tooltips) becomes one host; a graph and a link stay.
const H = (type, props) => ({ namespace: 'dash_html_components', type, props });
const tip = (label, child) => ({
  namespace: 'dash_mantine_components',
  type: 'Tooltip',
  props: { label, children: child },
});
const cells = [];
for (let i = 0; i < 14; i++) {
  const mark = H('Div', { 'data-frag': String(i), children: String(i) });
  cells.push(H('Td', { className: 'c', children: tip('fragment ' + i, mark) }));
}
const tbody = H('Tbody', { children: H('Tr', { children: cells }) });
const ladder = H('Table', { children: tbody });
const table = H('Div', { className: 'pv-ladder-scroll', children: ladder });
const help = tip('help text', H('Span', { className: 'mv-help', children: 'i' }));
const graph = {
  namespace: 'dash_core_components',
  type: 'Graph',
  props: { id: { type: 'fig', name: 'pv-xic' }, figure: { layout: { height: 240 } } },
};
const link = H('A', { href: '/x', children: 'open' });
const pep = H('Span', { children: [H('Span', { children: 'a' }), H('Span', { children: 'b' })] });
const head = H('Div', { className: 'pv-head', children: [link, pep] });
const xic = H('Div', { children: [help, graph] });
const body = H('Div', { className: 'pv-body', children: [xic, table] });
const card = {
  namespace: 'dash_mantine_components',
  type: 'Card',
  props: { id: 'pv-card', children: [head, body] },
};
const isl = {};
const tree = M.islandize(card, isl);
const graphs = [];
const deferred = M.deferGraphs(tree, graphs);
out.islands = Object.keys(isl).length;
out.hostClasses = JSON.stringify(deferred).match(/"className":"[^"]*","data-ib-island"/g);
out.keepsLink = JSON.stringify(deferred).includes('"type":"A"');
out.keepsHelpTooltip = JSON.stringify(deferred).includes('help text');
out.graphs = graphs.map((g) => g.node.props.id.name);
out.graphHost = JSON.stringify(deferred).includes('"minHeight":"240px"');
out.collapsed = M.collapseTree(card).props.children.length;
// The selection rule the browser shares with the server.
const rows = [
  { base_peptide_id: 5, candidate_id: 1, run: 'a' },
  { base_peptide_id: 7, candidate_id: 2, run: 'b' },
];
out.pick = [7, 9].map((p) => M.pickBy(rows, 'base_peptide_id', p).candidate_id);
out.pickPrecursor = ['b', 'a'].map((r) => M.pickPrecursor(rows, r, 2).candidate_id);
const cache = new M.LRU(3);
const pre = { key: 7, rows: rows, total: 2, mark: 'q_value' };
const peps = { key: 'G', rows: [{ base_peptide_id: 7, peptidoform: 'PEP' }], total: 1 };
cache.set(M.partKey('pep', 'G', false), peps);
cache.set(M.partKey('pre', 7, false), pre);
const want = { kind: 'group', group: 'G', peptide: 5, run: 'a', cid: 1, decoys: false, n: 3 };
const hit = M.resolveFromCache(want, cache);
out.resolved = hit && hit.sel;
out.miss = M.resolveFromCache({ ...want, group: 'H', n: 4 }, cache);
const marks = [
  { q: 0.001, label: 'target' },
  { q: 0.2, label: 'target' },
  { q: 0.0001, label: 'decoy' },
  { q: null },
];
out.pass = M.passCount(marks, 'q', 0.01);
const lru = new M.LRU(2); lru.set('a', 1); lru.set('b', 2); lru.get('a'); lru.set('c', 3);
out.lru = [lru.has('a'), lru.has('b'), lru.has('c')];
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_browser_js_pure_functions(tmp_path):
    views = [
        (browser.Filters(), browser.Selection(group="A;B", peptide=4, cid=9)),
        (
            browser.Filters(unit="peptide", q="q_value", search="LGE", charge="3", decoys=True),
            browser.Selection(peptide=4, run="r1", cid=9),
        ),
        (browser.Filters(unit="precursor", run="r1", sort="apex_rt", desc=False, t=0.05), None),
    ]
    cases = {
        "addresses": [
            {
                "view": f.store(),
                "sel": sel.store() if sel else None,
                "sort": {"col": f.sort, "desc": f.desc},
            }
            for f, sel in views
        ]
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    script = tmp_path / "test.js"
    script.write_text(NODE_TEST, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(script), str(BROWSER_JS), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout)
    # The browser writes the address the server reads (Filters.query + Selection.query).
    for (f, sel), got in zip(views, out["addresses"], strict=True):
        want = {**f.query(), **(sel.query(f.unit) if sel else {})}
        assert dict(pair.split("=", 1) for pair in got.lstrip("?").split("&")) == {
            k: str(v).replace(";", "%3B") for k, v in want.items()
        }
    assert out["norm"] == "a=1&b=2"
    # One island (the ion table, its tooltips as titles); the link, the info icon's
    # tooltip and the graph stay Dash components; the graph waits in a host of its height.
    assert out["islands"] == 2 and out["keepsLink"] and out["keepsHelpTooltip"]
    assert any("pv-ladder-scroll" in h for h in out["hostClasses"])
    assert out["graphs"] == ["pv-xic"] and out["graphHost"]
    assert out["collapsed"] == 1
    assert out["pick"] == [2, 1] and out["pickPrecursor"] == [2, 1]
    # The wanted peptide is not in the kept group: its first peptide and that one's
    # first row, as the server answers.
    assert out["resolved"] == {"group": "G", "peptide": 7, "run": "a", "cid": 1}
    assert out["miss"] is None
    assert out["pass"] == 1
    assert out["lru"] == [True, False, True]


# --------------------------------------------------------------------------- the page in Chromium


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
def test_the_page_in_chromium(open_fixture):
    """Clicks and keys move the selection, Enter opens the selected row, Back restores."""
    from werkzeug.serving import make_server

    rs = open_fixture("single")
    app = create_app(rs, url_base="/")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = make_server("127.0.0.1", port, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    sel_js = """() => ['ib-grid', 'ib-pep-grid', 'ib-pre-grid'].map(id => {
      const api = window.dash_ag_grid && window.dash_ag_grid.getApi(id);
      if (!api) return null;
      return api.getSelectedNodes().map(n => n.data ? n.data._key : '?').join(','); })"""
    loaded = """() => ['pep', 'pre'].every(k => {
      const e = document.getElementById('ib-panel-' + k);
      return e && !e.classList.contains('ib-loading'); })"""
    try:
        with SYNC_PLAYWRIGHT() as p:
            chromium = p.chromium.launch()
            page = chromium.new_page(viewport={"width": 1440, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            def selected(want, i=0, timeout=15000):
                """Wait until grid i selects the row with this id."""
                page.wait_for_function(
                    f"(want) => ({sel_js})()[{i}] === want", arg=want, timeout=timeout
                )

            def cell(index):
                return page.locator(
                    f'#ib-grid .ag-row[row-index="{index}"] [col-id="protein_group"]'
                )

            page.goto(base + "/identifications?t=0.1")
            page.wait_for_selector("#pv-card", timeout=60000)
            page.wait_for_function(loaded, timeout=30000)
            rows = browser.first_block(rs, "/", browser.Filters(t=0.1))[1]
            selected(rows[0]["_key"])
            assert " of " in page.locator("#ib-pep-count").inner_text()
            # A click selects the row; the child panels follow and the address holds it.
            cell(1).click()
            selected(rows[1]["_key"])
            page.wait_for_function(
                "(g) => decodeURIComponent(location.search).indexOf('group=' + g) >= 0",
                arg=rows[1]["_key"],
                timeout=10000,
            )
            page.wait_for_function(loaded, timeout=20000)
            page.wait_for_function(
                "(g) => g.indexOf(document.getElementById('ib-pep-subject').innerText"
                ".split(' ')[0]) >= 0",
                arg=rows[1]["_key"],
                timeout=10000,
            )
            # ArrowUp on row 0 keeps the selection on row 0 (no jump to another block).
            cell(0).click()
            selected(rows[0]["_key"])
            page.wait_for_function(loaded, timeout=20000)
            page.keyboard.press("ArrowUp")
            page.wait_for_timeout(600)
            assert page.evaluate(sel_js)[0] == rows[0]["_key"]
            # PageDown moves the focus and the selection together; Enter opens that row.
            page.keyboard.press("PageDown")
            page.wait_for_function(
                f"(first) => {{ const k = ({sel_js})()[0]; return k && k !== first; }}",
                arg=rows[0]["_key"],
                timeout=10000,
            )
            picked = page.evaluate(sel_js)[0]
            row = next(r for r in rows if r["_key"] == picked)
            page.wait_for_function(
                "(g) => decodeURIComponent(location.search).indexOf('group=' + g) >= 0",
                arg=picked,
                timeout=10000,
            )
            before = page.evaluate("() => location.search")
            page.keyboard.press("Enter")
            page.wait_for_function("() => location.pathname.endsWith('/precursor')", timeout=10000)
            assert f"cid={row['candidate_id']}" in page.evaluate("() => location.search")
            # Back: the identification page with the selection of the address.
            page.go_back()
            selected(picked, timeout=30000)
            page.wait_for_function(loaded, timeout=30000)
            assert page.evaluate("() => location.search") == before
            assert errors == []
            chromium.close()
    finally:
        server.shutdown()
