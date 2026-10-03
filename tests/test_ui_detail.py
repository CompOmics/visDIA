"""The precursor page and its preview, built on the fixtures without a browser.

The page must show the data layer's values: the ion table and the sequence diagram place
exactly the library fragments of the candidate and mark exactly the matches of the shown
scan; the validation marks test the engine's q columns against the threshold, and a
grouped column that the row does not win is tested on its group's winning row on every
surface. q text never reads as the threshold when the value differs from it. The ids
must be unique, the preview's must not collide with the page's, and the callbacks must
answer through the server.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from dash.development.base_component import Component
from plotly.io.json import to_json_plotly

from mumdia_viewer.data import open_results
from mumdia_viewer.data.detail import mirror
from mumdia_viewer.data.duck import sql_path
from mumdia_viewer.data.fragments import PeakMatch
from mumdia_viewer.data.mbr import transfers_for_run
from mumdia_viewer.ui import (
    detail,
    detail_cards,
    detail_figures,
    detail_ions,
    detail_preview,
    detail_view,
)
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.icons import icon
from mumdia_viewer.ui.state import PageContext, href

FIXTURES = ["single", "experiment", "grouped", "topk", "mbr", "ovl_bp"]
CHECK = icon("check").style["maskImage"]
CROSS = icon("x").style["maskImage"]
TOOLS = Path(__file__).parent / "fixtures" / "tools" / "entrapment"


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


def _text(tree) -> str:
    return json.dumps(json.loads(to_json_plotly(tree)), ensure_ascii=False)


def _classes(tree, cls: str) -> list[Component]:
    return [c for c in _walk(tree) if cls in str(getattr(c, "className", "") or "").split()]


def _frag_of(c) -> str | None:
    return getattr(c, "data-frag", None)


def _mark_kind(badge) -> str | None:
    """'check', 'cross', 'D', 'E' or None for a ThemeIcon of a validation mark."""
    if "pd-quant-mark" in str(getattr(badge, "className", "") or ""):
        return None  # the quantity tile's mark is not a validation mark
    child = badge.children
    if isinstance(child, str):
        return child if child in ("D", "E") else None
    if isinstance(child, Component):
        cls = str(getattr(child, "className", "") or "")
        style = getattr(child, "style", None) or {}
        if "pd-ico-check" in cls or style.get("maskImage") == CHECK:
            return "check"
        if "pd-ico-x" in cls or style.get("maskImage") == CROSS:
            return "cross"
    return None


def _badges(tree) -> list[str]:
    return [
        k
        for c in _walk(tree)
        if type(c).__name__ == "ThemeIcon" and (k := _mark_kind(c)) is not None
    ]


def _marks(tree) -> tuple[int, int]:
    """(check marks, cross marks) of the validation marks in a tree."""
    kinds = _badges(tree)
    return kinds.count("check"), kinds.count("cross")


def _candidates(rs, n: int = 1, label: str = "target") -> list[tuple[str, int]]:
    """The best-q rows of a label, as (run name, candidate id)."""
    df = rs.duck.df(
        "SELECT source, candidate_id FROM read_parquet($p) WHERE label = $l "
        "ORDER BY q_value, candidate_id LIMIT $n",
        {"p": sql_path(rs.scored.require()), "l": label, "n": n},
    )
    out = []
    for source, cid in zip(df["source"], df["candidate_id"], strict=True):
        run = rs.run(int(source)).name if rs.is_experiment else ""
        out.append((run, int(cid)))
    return out


def _ctx(rs, run: str, cid: int, **kw) -> PageContext:
    return PageContext(rs=rs, base="/t/", query={"run": run, "cid": str(cid)}, **kw)


def _non_winner(rs) -> tuple[str, int]:
    """A target row of an experiment that wins none of its grouped columns."""
    df = rs.duck.df(
        "SELECT source, candidate_id FROM read_parquet($p) WHERE label = 'target' AND "
        "precursor_q = 1.0 AND peptide_q_value = 1.0 AND pg_q_value = 1.0 AND q_value < 0.01 "
        "ORDER BY q_value, candidate_id LIMIT 1",
        {"p": sql_path(rs.scored.require())},
    )
    assert len(df), "no non-winner row in the fixture"
    return rs.run(int(df.iloc[0]["source"])).name, int(df.iloc[0]["candidate_id"])


# --------------------------------------------------------------------------- numbers


def test_q_text_never_reads_as_the_threshold():
    fmt = detail_view.fmt_q_at
    # Just above the threshold: more digits until the text differs and stays above.
    text = fmt(0.0100033, 0.01)
    assert text != "0.0100" and float(text) > 0.01 and text == "0.010003"
    assert fmt(0.0099996, 0.01) == "0.0099996" and float(fmt(0.0099996, 0.01)) <= 0.01
    assert fmt(1.0004e-4, 1e-4) == "1.0004e-4"
    # Values far from the threshold keep the grid's short form.
    assert fmt(3.79794910748196e-05, 0.01) == "3.80e-5"
    assert fmt(0.4025, 0.01) == "0.4025" and fmt(0.01, 0.01) == "0.0100"
    assert fmt(0, 0.01) == "0" and fmt(None, 0.01) == "" and fmt(float("nan")) == ""
    # The figure hover does not follow the threshold: unambiguous at every stop.
    assert detail_figures.q_text_any(0.010003) == "0.010003"
    assert detail_figures.q_text_any(0.2) == "0.2000"


def test_validation_tooltip_uses_the_threshold_aware_text():
    mark = detail_cards.mark(0.0100033, 0.01, label="target", column="q_value")
    assert mark.label == "does not pass: q_value 0.010003 > 0.01"
    assert _mark_kind(mark.children) == "cross"
    ok = detail_cards.mark(0.0099996, 0.01, label="target", column="q_value")
    assert ok.label == "passes: q_value 0.0099996 ≤ 0.01" and _mark_kind(ok.children) == "check"
    assert _mark_kind(detail_cards.mark(0.5, 0.01, label="decoy", column="q").children) == "D"
    spike = detail_cards.mark(0.001, 0.01, label="target", column="q", spike=True)
    assert _mark_kind(spike.children) == "E" and spike.label.startswith("entrapment spike-in")


def test_score_digits_tell_the_rows_apart():
    assert detail_view.score_digits([0.999999, 0.999996, 0.999582]) == 6
    assert detail_view.score_digits([1.0, 0.585991, 0.580282, 0.221028]) == 4
    assert detail_view.score_digits([0.5, 0.5]) == 4  # a tie stays a tie
    assert detail_view.score_digits([0.1234567891, 0.1234567892]) == 8
    assert detail_view.fmt_score(0.999996, 6) == "0.999996"
    assert detail_view.fmt_score(None) == ""


def test_score_bounds_ignore_an_outlier(open_fixture):
    rs = open_fixture("ovl_bp")
    lo, hi, _ = detail_view.score_range(rs)
    b = detail_view.score_bounds(rs)
    assert lo < b.lo < b.hi <= hi
    # The fixture's lowest score is an outlier: the bar's range starts far above it.
    assert b.lo - lo > 10 * (hi - b.hi + 1e-9) and "percentile" in b.text
    assert b.clamp_note(lo).startswith(" (below") and b.clamp_note((b.lo + b.hi) / 2) == ""
    assert detail_view.score_bounds(rs) is b


# --------------------------------------------------------------------------- layout


@pytest.mark.parametrize("name", FIXTURES)
def test_layout_builds_with_unique_ids(open_fixture, name):
    rs = open_fixture(name)
    run, cid = _candidates(rs)[0]
    tree = detail.layout(_ctx(rs, run, cid))
    ids = _ids(tree)
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for required in (
        "pd-key",
        "pd-verdict",
        "pd-seq",
        "pd-ladder",
        "pd-frag-table",
        "pd-frag-view",
        "pd-q-table",
        "pd-comp-table",
        "pd-partner-body",
        "pd-xic-view",
        "pd-xview",
        "pd-mirror-next",
        "scan-prev",
        "scan-next",
        "scan-apex",
    ):
        assert json.dumps(required) in ids, required
    graphs = [c for c in _walk(tree) if type(c).__name__ == "Graph"]
    assert graphs and all(g.id["type"] == "fig" and g.id["name"].startswith("pd-") for g in graphs)
    assert "pd-linked" in tree.className.split()
    # The linked panels: XIC and retention time left, the spectrum and its fragments right
    # (detail.css orders them XIC, spectrum, fragments, RT on narrow screens).
    grid = _classes(tree, "pd-linked-grid")[0]
    cols = [[c.id for c in col.children] for col in grid.children]
    assert cols == [["pd-xic-card", "pd-rt-card"], ["pd-mirror-card", "pd-frag-card"]]


def test_xic_names_its_page_and_leaves_the_shown_scan_to_the_overlay(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    tree = detail.layout(_ctx(rs, run, cid))
    xic = _find(tree, {"type": "fig", "name": "pd-xic"}).figure
    assert xic.layout.meta["key"] == f"{run}|{cid}" and xic.layout.meta["step"] > 0
    assert not [s for s in xic.layout.shapes if s.name == "scan"]
    mark = _find(tree, "pd-scanmark")
    assert [c.className for c in mark.children] == ["pd-scanband", "pd-scanline"]
    # The page's script starts the overlay on the scan of the pd-scan store (the apex).
    d = detail.get_detail(rs, run, cid)
    assert _find(tree, "pd-scan").data["row"] == d.apex_scan.row
    # The page's plots take their height from the page's CSS.
    assert xic.layout.height is None
    assert _find(tree, {"type": "fig", "name": "pd-mirror"}).figure.layout.height is None


def test_xic_opens_on_the_peak_and_offers_the_window(open_fixture):
    # The grouped fixture's RT windows are wider than its peaks (the smoke run's are not).
    rs = open_fixture("grouped")
    for run, cid in _candidates(rs, n=60):
        d = detail.get_detail(rs, run, cid)
        grid = detail_view.scan_grid(rs, d)
        views = detail_cards.xic_views(d, grid)
        if "peak" in views and views["peak"]["slider"] is not None:
            break
    else:
        pytest.skip("every candidate's peak covers most of its window")
    xr = detail_figures.x_range(d, grid)
    peak, window = views["peak"]["range"], views["window"]["range"]
    m = d.markers
    assert peak[0] <= m["elution_lo"] and m["elution_hi"] <= peak[1] and peak[0] <= m["apex_rt"]
    assert window == xr["range"] and peak[1] - peak[0] < 0.8 * (window[1] - window[0])
    s_peak, s_window = views["peak"]["slider"], views["window"]["slider"]
    assert s_window["min"] == 0 and s_window["max"] == grid.size - 1
    assert 0 <= s_peak["min"] < s_peak["max"] <= grid.size - 1
    inside = [i for i, rt in enumerate(grid.rt) if peak[0] <= rt <= peak[1]]
    assert (s_peak["min"], s_peak["max"]) == (inside[0], inside[-1])
    fig = detail_figures.xic_figure(d, detail_view.fragments_of(d.chromatogram), grid, view="peak")
    assert list(fig.layout.xaxis.range) == peak and fig.layout.meta["view"] == "peak"
    assert set(fig.layout.meta["views"]) == {"peak", "window"}
    # An edge note per window bound; inside the plotted range it shows only out of view.
    notes = {a.name: a for a in fig.layout.annotations if (a.name or "").startswith("edge-")}
    for e in fig.layout.meta["edges"]:
        visible = e["side"] == "always" or detail_figures.edge_visible(e["x"], e["side"], peak)
        assert notes[e["name"]].visible == visible


def test_missing_candidate_gives_a_message(open_fixture):
    rs = open_fixture("single")
    assert "No candidate id" in _text(detail.layout(PageContext(rs=rs, base="/", query={})))
    text = _text(detail.layout(PageContext(rs=rs, base="/", query={"cid": "999999999"})))
    assert "Candidate 999999999 is not a scored row of this run" in text
    assert "is not a whole number of 0 or more" in _text(
        detail.layout(PageContext(rs=rs, base="/", query={"cid": "abc"}))
    )


def test_address_parsing():
    assert detail.parse_cid("10298669") == (10298669, None)
    assert detail.parse_cid(" 12 ") == (12, None)
    assert detail.parse_cid("10298669.0") == (10298669, None)
    assert detail.parse_cid("abc")[0] is None and "'abc'" in detail.parse_cid("abc")[1]
    assert detail.parse_cid("-5")[0] is None and detail.parse_cid(None)[0] is None


def test_a_single_run_ignores_the_run_of_the_address(open_fixture):
    rs = open_fixture("single")
    _, cid = _candidates(rs)[0]
    tree = detail.layout(PageContext(rs=rs, base="/", query={"cid": str(cid), "run": "r2"}))
    assert json.dumps("pd-key") in _ids(tree)


def test_an_experiment_address_without_a_run_offers_the_runs(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _candidates(rs)[0]
    tree = detail.layout(PageContext(rs=rs, base="/", query={"cid": str(cid)}))
    text = _text(tree)
    assert "The address names no run" in text
    links = [c.href for c in _walk(tree) if type(c).__name__ == "Link" and "precursor" in c.href]
    assert href("/", "precursor", {"run": run, "cid": cid}) in links
    tree = detail.layout(PageContext(rs=rs, base="/", query={"cid": str(cid), "run": "zz"}))
    text = _text(tree)
    # The KeyError's message, not its repr (which quoted it).
    assert "No run 'zz'" in text and '\\"no run' not in text


# --------------------------------------------------------------------------- preview


@pytest.mark.parametrize("name", FIXTURES)
def test_preview_is_one_static_card_with_pv_ids(open_fixture, name):
    rs = open_fixture(name)
    run, cid = _candidates(rs)[0]
    ctx = _ctx(rs, run, cid)
    card = detail.preview(ctx, run, cid)
    assert type(card).__name__ == "Card" and card.id == "pv-card"
    ids = _ids(card)
    assert len(ids) == len(set(ids))
    for i in ids:
        value = json.loads(i)
        name_ = value["name"] if isinstance(value, dict) else value
        assert name_.startswith("pv-"), i
    graphs = [c for c in _walk(card) if type(c).__name__ == "Graph"]
    assert sorted(g.id["name"] for g in graphs) == ["pv-spec", "pv-xic"]
    assert all(g.config.get("displayModeBar") is False for g in graphs)
    link = _find(card, "pv-open")
    assert link.href == href("/t/", "precursor", {"run": run, "cid": cid})
    # The four key q columns, with the run's own PSM q in an experiment.
    text = _text(card)
    first = "run_psm_q" if rs.is_experiment else "q_value"
    for column in (first, "precursor_q", "peptide_q_value", "pg_q_value"):
        assert f'"{column}"' in text, column
    # The preview leaves its detail in the page's cache: the page then opens warm.
    assert detail.get_detail(rs, run, cid) is detail.get_detail(rs, run, cid)


def test_preview_and_page_ids_do_not_collide(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    ctx = _ctx(rs, run, cid)
    page = set(_ids(detail.layout(ctx)))
    pv = set(_ids(detail.preview(ctx, run, cid)))
    assert not page & pv


def test_preview_of_an_unreadable_candidate_says_why(open_fixture):
    rs = open_fixture("single")
    card = detail.preview(PageContext(rs=rs, base="/"), "", 999999999)
    assert card.id == "pv-card" and "999999999 is not a scored row" in _text(card)
    card = detail.preview(PageContext(rs=rs, base="/"), "", "abc")
    assert "is not a whole number of 0 or more" in _text(card) and "invalid literal" not in _text(
        card
    )


def test_preview_marks_a_decoy_with_d(open_fixture):
    rs = open_fixture("single")
    found = _candidates(rs, label="decoy")
    if not found:
        pytest.skip("no decoy row in the fixture")
    run, cid = found[0]
    card = detail.preview(_ctx(rs, run, cid), run, cid)
    assert _marks(card) == (0, 0)
    assert "D" in _badges(card)


def test_preview_marks_follow_the_threshold(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    q = d.scored["q_value"]
    loose = detail.preview(_ctx(rs, run, cid, threshold=0.999), run, cid)
    strict = detail.preview(_ctx(rs, run, cid, threshold=q / 10 if q > 0 else 1e-300), run, cid)
    assert _marks(loose)[0] > _marks(strict)[0]
    assert _marks(strict)[1] >= 1


def test_preview_shows_the_quant_state(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _non_winner(rs)
    d = detail.get_detail(rs, run, cid)
    card = detail.preview(_ctx(rs, run, cid), run, cid)
    if d.quant is None:
        pytest.skip("no quant state in the fixture")
    assert d.quant.state.replace("_", " ") in _text(card)
    # A grouped column this row does not win shows its group's value with the mark of it.
    tiles = {t.column: t for t in detail_view.q_tiles(rs, d, detail_preview.EXPERIMENT_COLUMNS)}
    assert tiles["precursor_q"].value is None and tiles["precursor_q"].group_winner is not None
    assert "group" in _text(card)


# --------------------------------------------------------------------------- q values


def _surface_marks(tree) -> list[str]:
    return _badges(tree)


def test_tiles_and_q_table_agree_on_a_non_winner(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _non_winner(rs)
    d = detail.get_detail(rs, run, cid)
    for t in (0.001, 0.01, 0.05, 0.5):
        ctx = PageContext(rs=rs, base="/", threshold=t)
        tiles = {x.column: x for x in detail_view.q_tiles(rs, d, None)}
        # Every grouped column of this row is on another row: the winner's value is tested.
        for column in ("precursor_q", "peptide_q_value", "pg_q_value"):
            tile = tiles[column]
            assert tile.value is None and tile.group_winner is not None, column
            assert tile.group_winner.value < 1.0 and tile.tested == tile.group_winner.value
        expected = {
            c: "check" if x.tested is not None and x.tested <= t else "cross"
            for c, x in tiles.items()
        }
        strip = detail_cards.verdict_body(ctx, d, detail_view.verdict_tiles(rs, d))
        strip_tiles = [x for x in detail_view.verdict_tiles(rs, d)]
        assert _surface_marks(strip) == [expected[x.column] for x in strip_tiles]
        table = detail_cards.q_table(ctx, d)
        rows = [e.key for e in d.evidence if e.group == "q"]
        assert _surface_marks(table) == [expected[c] for c in rows], t
        exact_checks = sum(1 for c in rows if expected[c] == "check")
        assert _marks(table) == (exact_checks, len(rows) - exact_checks)


def test_the_group_winners_are_the_rows_that_hold_the_q(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _non_winner(rs)
    d = detail.get_detail(rs, run, cid)
    p = sql_path(rs.scored.require())
    s = d.scored
    keys = {
        "precursor_q": (
            "peptidoform = $a AND charge = $b",
            {"a": s["peptidoform"], "b": int(s["charge"])},
        ),
        "peptide_q_value": ("base_peptide_id = $a", {"a": int(s["base_peptide_id"])}),
        "pg_q_value": ("protein_group = $a", {"a": s["protein_group"]}),
    }
    for column, (where, params) in keys.items():
        w = detail_view.group_winner(rs, d, column)
        held = rs.duck.df(
            f"SELECT source, candidate_id, {column} AS q FROM read_parquet($p) "
            f"WHERE {where} AND {column} < 1.0",
            {"p": p, **params},
        )
        assert len(held) == 1, column
        row = held.iloc[0]
        winner = (rs.run(int(row["source"])).label, int(row["candidate_id"]))
        assert (w.run, w.candidate_id) == winner, column
        assert w.value == pytest.approx(float(row["q"]))


def test_q_table_units_join_unit_and_scope(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _candidates(rs)[0]
    text = _text(
        detail_cards.q_table(PageContext(rs=rs, base="/"), detail.get_detail(rs, run, cid))
    )
    assert "; experiment-wide" in text and ") (" not in text


def test_verdict_of_a_single_run_leaves_out_the_equal_run_psm_q(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    columns = [t.column for t in detail_view.verdict_tiles(rs, d)]
    assert "run_psm_q" not in columns and columns[0] == "q_value"
    exp = open_fixture("experiment")
    r2, c2 = _candidates(exp)[0]
    assert "run_psm_q" in [
        t.column for t in detail_view.verdict_tiles(exp, detail.get_detail(exp, r2, c2))
    ]


def test_decoy_summary_says_counts_against_the_fdr_only_when_one_passes():
    tiles = [
        detail_view.QTile("q_value", 0.3, "u", "PSM", "psm", "this run", False, None, None, "")
    ]
    assert "counts against the FDR" not in detail_cards.summary(tiles, 0.01, decoy=True)
    assert "counts against the FDR" in detail_cards.summary(tiles, 0.5, decoy=True)


def test_threshold_rebuilds_the_marks(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    key = {"run": run, "cid": cid}
    d = detail.get_detail(rs, run, cid)
    loose = detail.follow_threshold(rs, "/", key, 0.999)
    strict = detail.follow_threshold(rs, "/", key, 1e-300)
    assert len(loose) == 4
    n_strip = len(detail_view.verdict_tiles(rs, d))
    n_table = len([e for e in d.evidence if e.group == "q"])
    # The best row of the fixture passes every column at 0.999 and none at 1e-300.
    assert _marks(loose[0]) == (n_strip, 0) and _marks(strict[0]) == (0, n_strip)
    assert _marks(loose[1]) == (n_table, 0) and _marks(strict[1]) == (0, n_table)
    assert "At q ≤ 0.999" in _text(loose[0])


# --------------------------------------------------------------------------- ion ladder


def _frag(index: int, name: str, mz: float) -> detail_view.Fragment:
    ion, rest = name[0], name[1:]
    ordinal, _, charge = rest.partition("^")
    return detail_view.Fragment(
        index=index,
        name=name,
        ion=ion if ion in "by" else None,
        ordinal=int(ordinal) if ordinal.isdigit() else None,
        charge=int(charge or 1) if ordinal.isdigit() else None,
        mz=mz,
        obs_mz=mz,
        predicted=0.5,
        observed=True,
        colour="#1c7ed6" if ion == "b" else "#e03131",
    )


def _match(index: int, mz: float, ppm: float = 1.5) -> PeakMatch:
    return PeakMatch(
        fragment=index,
        peak=index,
        obs_mz=mz * (1 + ppm * 1e-6),
        obs_intensity=1000.0 + index,
        ppm_raw=ppm,
        ppm_corrected=ppm,
        theo_mz=mz,
        query_mz=mz,
    )


@pytest.fixture
def synthetic():
    # PEPTM[Oxidation]IDEK: 9 residues. b2, b5, b5^2, y3, y7, y7^2; b9 is the whole
    # peptide (row 9, no cleavage); z4 is not a b or y ion; b12 does not fit.
    frags = [
        _frag(0, "b2", 227.10),
        _frag(1, "b5", 556.25),
        _frag(2, "b5^2", 278.63),
        _frag(3, "y3", 391.20),
        _frag(4, "y7", 853.37),
        _frag(5, "y7^2", 427.19),
        _frag(6, "b9", 1001.4),
        _frag(7, "z4", 500.0),
        _frag(8, "b12", 1300.0),
    ]
    m = SimpleNamespace(matches=[_match(1, 556.25), _match(3, 391.20, -2.0), _match(5, 427.19)])
    return detail_view.ion_ladder("PEPTM[Oxidation]IDEK", frags, m), frags


def test_ion_ladder_places_only_the_library_fragments(synthetic):
    ladder, frags = synthetic
    assert ladder.n == 9 and ladder.residues[4].mods == ("Oxidation",)
    assert ladder.b_charges == (1, 2) and ladder.y_charges == (1, 2)
    assert set(ladder.cells) == {
        ("b", 2, 1),
        ("b", 5, 1),
        ("b", 5, 2),
        ("y", 3, 1),
        ("y", 7, 1),
        ("y", 7, 2),
        ("b", 9, 1),
    }
    assert [f.name for f in ladder.unplaced] == ["z4", "b12"]
    assert ladder.n_library == 7 and ladder.n_matched == 3
    assert ladder.cell("b", 5, 1).matched and not ladder.cell("b", 5, 2).matched
    assert ladder.cell("y", 7, 2).match.fragment == 5
    assert ladder.cell("b", 3, 1) is None
    empty = detail_view.ion_ladder("PEPTM[Oxidation]IDEK", frags, None)
    assert not empty.scan and empty.n_matched == 0 and empty.n_library == 7
    assert detail_ions.ladder_caption(ladder) == "3 of 7 library fragments matched"
    assert "not on the sequence" in detail_ions.ladder_help(ladder)


def test_sequence_diagram_marks_each_cleavage(synthetic):
    ladder, _ = synthetic
    tree = detail_ions.sequence_diagram(ladder, hidden={4})
    marks = {_frag_of(m): m for m in _walk(tree) if "pd-mark" in str(getattr(m, "className", ""))}
    marks = {k: v for k, v in marks.items() if k is not None}
    # b9 has no cleavage (it is the whole peptide): no mark; the others have one each.
    assert sorted(marks) == ["0", "1", "2", "3", "4", "5"]
    assert "pd-mark-on" in marks["1"].className and "pd-mark-lib" in marks["0"].className
    assert "pd-off" in marks["4"].className
    # The page's one tooltip shows each mark's library m/z and match (no Mantine tooltip).
    assert "Matched in the shown scan" in getattr(marks["1"], "data-tip")
    assert "Not matched" in getattr(marks["0"], "data-tip")
    slots = {
        _frag_of(m): s.style
        for s in _classes(tree, "pd-seq-slot")
        for m in _walk(s.children)
        if _frag_of(m)
    }
    n, letters = 9, 4  # two b charge rows: the letters are on row 4
    # b_i sits between residue i and i + 1 (grid column 2i), y_j between n - j and n - j + 1.
    assert slots["0"]["gridColumn"] == 4 and slots["1"]["gridColumn"] == 10
    assert slots["3"]["gridColumn"] == 2 * (n - 3) and slots["4"]["gridColumn"] == 2 * (n - 7)
    # Charge 1 next to the letters, charge 2 one row further out.
    assert slots["1"]["gridRow"] == letters - 1 and slots["2"]["gridRow"] == letters - 2
    assert slots["4"]["gridRow"] == letters + 1 and slots["5"]["gridRow"] == letters + 2
    letters_ = [c for c in _classes(tree, "pd-seq-aa")]
    assert [c.children[0] for c in letters_] == list("PEPTMIDEK")
    assert all(c.style["gridRow"] == letters for c in letters_)
    assert "Oxidation" in getattr(letters_[4], "data-tip")


def test_ladder_table_rows_hold_b_i_and_y_n_minus_i_plus_1(synthetic):
    ladder, _ = synthetic
    table = detail_ions.ladder_table(ladder)
    body = table.children[1].children
    assert len(body) == 9
    # Row i: #, b(1+), b(2+), residue, y(1+), y(2+), #.
    frag_of = _frag_of
    row2 = body[1].children
    assert row2[0].children == "2" and frag_of(row2[1]) == "0" and row2[-1].children == "8"
    row5 = body[4].children
    assert frag_of(row5[1]) == "1" and frag_of(row5[2]) == "2"
    assert row5[3].children[0] == "M" and "Oxidation" in getattr(row5[3], "data-tip")
    row7 = body[6].children  # y3 is in row n - 3 + 1 = 7
    assert frag_of(row7[4]) == "3"
    row3 = body[2].children  # y7 and y7^2 in row 3
    assert frag_of(row3[4]) == "4" and frag_of(row3[5]) == "5"
    assert "pd-lc-on" in row5[1].className and "pd-lc-lib" in row5[2].className
    assert "-2.0 ppm" in _text(row7[4])
    assert frag_of(body[8].children[1]) == "6"  # b9 in the last row
    assert not [c for c in _walk(table) if type(c).__name__ == "Tooltip"]
    compact = detail_ions.ladder_table(ladder, compact=True)
    cells = [c for c in _walk(compact) if _frag_of(c)]
    # Narrow cells: no dot and no ppm (it is in the tooltip).
    assert cells and not _classes(compact, "pd-lc-dot") and not _classes(compact, "pd-lc-ppm")
    assert all("ppm raw" in getattr(c, "data-tip") for c in cells if "pd-lc-on" in c.className)


@pytest.mark.parametrize("name", FIXTURES)
def test_ion_ladder_on_fixtures_holds_every_fragment_once(open_fixture, name):
    rs = open_fixture(name)
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    frags = detail_view.fragments_of(d.chromatogram)
    m = mirror(rs, d)
    ladder = detail_view.ion_ladder(d.scored["peptidoform"], frags, m)
    assert ladder.n_library + len(ladder.unplaced) == len(frags)
    for (ion, ordinal, charge), cell in ladder.cells.items():
        f = cell.fragment
        assert (f.ion, f.ordinal, f.charge or 1) == (ion, ordinal, charge)
        assert 1 <= ordinal <= ladder.n
    matched = {mt.fragment for mt in m.matches} if m is not None else set()
    assert {c.fragment.index for c in ladder.cells.values() if c.matched} == matched - {
        f.index for f in ladder.unplaced
    }


# --------------------------------------------------------------------------- mirror


def test_mirror_keeps_one_mz_range_for_every_scan(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    grid = detail_view.scan_grid(rs, d)
    frags = detail_view.fragments_of(d.chromatogram)
    apex = mirror(rs, d)
    mz = detail_view.mz_range(apex)
    assert mz[0] < float(apex.spectrum.mz.min()) and mz[1] > float(apex.spectrum.mz.max())
    ranges = set()
    for row in (int(grid.rows[0]), int(grid.rows[-1])):
        m = mirror(rs, d, row=row)
        fig = detail_figures.mirror_figure(m, frags, x_range=mz)
        ranges.add(tuple(fig.layout.xaxis.range))
        out = detail.show_scan(rs, {"run": run, "cid": cid}, {"row": row}, "matched", [], "light")
        ranges.add(tuple(out[0].layout.xaxis.range))
    assert ranges == {tuple(mz)}


def test_mirror_annotations_do_not_collide(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    frags = detail_view.fragments_of(d.chromatogram)
    fig = detail_figures.mirror_figure(mirror(rs, d), frags, outside=True)
    top = {a.name: a.text for a in fig.layout.annotations}
    # Short texts: the scale on the left, the outside note on the right of one line.
    assert len(top["observed"]) + len(top["outside"]) < 90
    assert fig.layout.modebar.orientation == "v" and fig.layout.margin.r >= 30


def test_competition_hover_prints_a_grouped_q_on_its_winner_only(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _non_winner(rs)
    d = detail.get_detail(rs, run, cid)
    fig = detail_figures.competition_figure(d.competition, experiment=True)
    texts = [t for tr in fig.data if tr.hovertext for t in tr.hovertext]
    frame = d.competition.sort_values("score", ascending=False, kind="mergesort")
    assert len(texts) == len(frame)
    losers = [t for t in texts if "precursor_q: not the winner" in t]
    assert losers and all("precursor_q 1.0000" not in t for t in texts)
    assert all("peptide_q_value 1.0000" not in t for t in texts)
    n_prec = int((~frame["wins_precursor"].astype(bool)).sum())
    assert len(losers) == n_prec


def test_competition_table_of_a_non_winner(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _non_winner(rs)
    d = detail.get_detail(rs, run, cid)
    table = detail_cards.comp_table(PageContext(rs=rs, base="/"), d)
    dashes = [c for c in _classes(table, "pd-dash")]
    df = d.competition
    losers = int((~df["wins_peptide"].astype(bool)).sum()) + int(
        (~df["wins_precursor"].astype(bool)).sum()
    )
    assert len(dashes) == losers
    # This row is shown without a link to itself; the other rows link to their pages.
    links = [c.href for c in _walk(table) if type(c).__name__ == "Link"]
    own = href("/", "precursor", {"run": run, "cid": cid})
    assert own not in links and len(links) == len(df) - 1


def test_competition_scores_tell_the_rows_apart(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    digits = detail_view.page_score_digits(d)
    texts = [detail_view.fmt_score(v, digits) for v in d.competition["score"].drop_duplicates()]
    assert len(set(texts)) == len(texts)


def test_competition_tick_labels_draw_the_modifications():
    html = detail_figures.pep_html("DECOY_PEPTM[Oxidation]IDEK")
    assert html.startswith("<span style='color:") and "DECOY_" in html
    assert "M<sup>ox</sup>" in html
    long = detail_figures.pep_html("A" * 40, max_chars=10)
    assert long.endswith("…") and long.count("A") == 10


def test_entrapment_spike_ins_get_an_e(fixture_dir, tmp_path):
    dst = tmp_path / "run"
    dst.mkdir()
    src = fixture_dir("entrapment")
    for f in ("psms_scored.parquet", "psms_scored.parquet.report.json"):
        shutil.copy2(src / f, dst / f)
    report = json.loads((dst / "psms_scored.parquet.report.json").read_text(encoding="utf-8"))
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    (dst / "manifest.json").write_text(
        json.dumps(
            {
                "mumdia_version": "0.5.0",
                "cli_args": ["mumdia", "rescore", "--out-dir", str(dst)],
                "config_json": json.dumps(config),
                "artifacts": {
                    "psms_scored": {
                        "path": str(dst / "psms_scored.parquet"),
                        "schema_name": "psms_scored",
                        "schema_version": 4,
                        "rows": report["rows"],
                        "content_hash": report["content_hash"],
                        "producing_stage": "rescore",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    rs = open_results(dst)
    df = rs.duck.df(
        "SELECT candidate_id FROM read_parquet($p) WHERE label = 'target' AND "
        "protein LIKE '%ENTRAP_%' AND protein NOT LIKE '%REAL_%' ORDER BY q_value LIMIT 1",
        {"p": sql_path(rs.scored.require())},
    )
    if df.empty:
        pytest.skip("no spike-in row in the fixture")
    cid = int(df.iloc[0]["candidate_id"])
    d = detail.precursor_detail(rs, "", cid)
    assert detail_cards.is_spike(d)
    table = detail_cards.comp_table(PageContext(rs=rs, base="/"), d)
    assert "E" in _badges(table)
    assert "entrapment spike-in" in _text(detail_cards.hero(PageContext(rs=rs, base="/"), d, []))


# --------------------------------------------------------------------------- cards


def test_notes_are_one_compact_card(open_fixture):
    rs = open_fixture("chrom_v1")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    card = detail_cards.notes(d)
    rows = _classes(card, "pd-note-row")
    shown = [n for n in d.notes if not n.startswith(detail_cards.HERO_NOTES)]
    assert card.id == "pd-notes" and len(rows) == len(shown)
    visible = " ".join(c.children for c in _classes(card, "pd-note-text"))
    assert "C:\\" not in visible and "/Users/" not in visible
    if len(rows) > 2:
        assert [c for c in _walk(card) if type(c).__name__ == "Spoiler"]


def test_short_paths_keeps_the_file_names():
    text = (
        "run_windows is missing: C:\\Users\\a\\run_windows.parquet (recorded as C:/x/b/w.parquet)"
    )
    assert detail_cards.short_paths(text) == (
        "run_windows is missing: run_windows.parquet (recorded as w.parquet)"
    )
    assert detail_cards.short_paths("see /home/u/x/y.parquet now") == "see y.parquet now"
    assert detail_cards.short_paths("no path here") == "no path here"
    # Relative paths stay as they are.
    assert detail_cards.short_paths("band g00 (groups/g00/x.parquet)") == (
        "band g00 (groups/g00/x.parquet)"
    )


def test_decoy_and_transfer_notes_are_hero_chips(open_fixture):
    rs = open_fixture("single")
    found = _candidates(rs, label="decoy")
    if not found:
        pytest.skip("no decoy row")
    run, cid = found[0]
    d = detail.get_detail(rs, run, cid)
    assert "this row is a decoy" in d.notes
    card = detail_cards.notes(d)
    assert card is None or "this row is a decoy" not in _text(card).lower()


def test_back_link_selects_this_row(open_fixture):
    rs = open_fixture("experiment")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    link = detail_cards.back_link(PageContext(rs=rs, base="/b/"), d)
    s = d.scored
    assert link.href == href(
        "/b/",
        "identifications",
        {
            "group": s["protein_group"],
            "peptide": int(s["base_peptide_id"]),
            "run": run,
            "cid": cid,
        },
    )


def test_rt_window_note_only_when_it_extends_past_the_plot(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    grid = detail_view.scan_grid(rs, d)
    xr = detail_figures.x_range(d, grid)
    text = _text(detail_cards.marker_key(d, xr))
    past = not (xr["lo_inside"] and xr["hi_inside"])
    assert ("past the plot" in text) == past


def test_footer_says_when_the_detail_was_cached(open_fixture):
    rs = open_fixture("topk")
    run, cid = _candidates(rs)[0]
    with detail._LOCK:
        detail._CACHE.pop((id(rs), run, cid), None)
    cold = _text(_find(detail.layout(_ctx(rs, run, cid)), "pd-footer"))
    warm = _text(_find(detail.layout(_ctx(rs, run, cid)), "pd-footer"))
    assert "Page built in" in cold and "(cached)" not in cold and "(cached)" in warm


def test_rt_error_and_matches_are_marked_viewer_derived(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    tree = detail.layout(_ctx(rs, run, cid))
    hero = _classes(tree, "pd-hero")[0]
    assert "derived" in _text(hero) and "apex_rt - rt_pred_cal" in _text(hero)
    for part in ("pd-seq", "pd-frag-table", "pd-ladder"):
        assert "viewer match" in _text(_find(tree, part)), part


# --------------------------------------------------------------------------- callbacks


def test_show_scan_rebuilds_the_diagram_and_the_ion_table(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    grid = detail_view.scan_grid(rs, d)
    assert grid is not None and grid.size > 1
    key = {"run": run, "cid": cid}
    for row in (int(grid.rows[0]), int(grid.rows[-1])):
        out = detail.show_scan(rs, key, {"row": row}, "matched", [], "light")
        assert len(out) == 7
        m = mirror(rs, d, row=row)
        ladder = detail_view.ion_ladder(
            d.scored["peptidoform"], detail_view.fragments_of(d.chromatogram), m
        )
        on = [c for c in _classes(out[5], "pd-mark-on") if _frag_of(c)]
        expected = [c for c in ladder.cells.values() if c.matched and c.fragment.ordinal < ladder.n]
        assert len(on) == len(expected)
        assert len(_classes(out[6], "pd-lc-on")) == ladder.n_matched
        assert out[4]["row"] == row


def test_competition_shows_grouped_q_on_winners_only(open_fixture):
    rs = open_fixture("single")
    for run, cid in _candidates(rs, n=40):
        d = detail.get_detail(rs, run, cid)
        if len(d.competition) > 1 and not d.competition["wins_peptide"].all():
            break
    else:
        pytest.skip("no base peptide with several rows in the fixture")
    table = detail_cards.comp_table(PageContext(rs=rs, base="/"), d)
    dashes = len(_classes(table, "pd-dash"))
    losers = int((~d.competition["wins_peptide"].astype(bool)).sum()) + int(
        (~d.competition["wins_precursor"].astype(bool)).sum()
    )
    assert dashes == losers
    # Winner chips only for groups of more than one row.
    n_prec = d.competition.groupby(["peptidoform", "charge"]).size()
    chips = _text(table).count("precursor winner")
    assert chips == int((n_prec > 1).sum())


def test_score_range_is_the_scored_table_range(open_fixture):
    rs = open_fixture("experiment")
    lo, hi, source = detail_view.score_range(rs)
    got = rs.duck.rows(
        "SELECT min(score), max(score) FROM read_parquet($p)", {"p": sql_path(rs.scored.require())}
    )[0]
    assert (lo, hi) == pytest.approx(got)
    assert source and detail_view.score_range(rs) == (lo, hi, source)


def test_mbr_transfer_is_labelled(open_fixture):
    rs = open_fixture("mbr")
    for r in rs.runs:
        df = transfers_for_run(rs, r.index)
        if len(df):
            run, cid = r.name, int(df.iloc[0]["candidate_id"])
            break
    else:
        pytest.skip("no transfer in the fixture")
    ctx = _ctx(rs, run, cid)
    text = _text(detail.layout(ctx))
    assert "MBR transfer" in text and "transfer_q" in text
    assert json.dumps("pd-transfer-card") in _ids(detail.layout(ctx))
    assert "MBR transfer" in _text(detail.preview(ctx, run, cid))


def _callback_key(app, needle: str) -> str:
    keys = [k for k in app.callback_map if needle in k]
    assert len(keys) == 1, keys
    return keys[0]


def _post(client, key, inputs, state):
    outputs = []
    for part in key.strip(".").split("..."):
        cid, prop = part.rsplit(".", 1)
        outputs.append({"id": json.loads(cid) if cid.startswith("{") else cid, "property": prop})
    body = {
        "output": key,
        "outputs": outputs,
        "inputs": inputs,
        "changedPropIds": [f"{i['id']}.{i['property']}" for i in inputs],
        "state": state,
    }
    return client.post("/_dash-update-component", json=body)


def test_callbacks_answer_through_the_server(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    grid = detail_view.scan_grid(rs, d)
    app = create_app(rs, url_base="/")
    client = app.server.test_client()
    mz = detail_view.mz_range(mirror(rs, d))
    key = {"run": run, "cid": cid, "mz": mz}
    scan = _callback_key(app, "pd-seq.children")
    resp = _post(
        client,
        scan,
        [
            {"id": "pd-scan", "property": "data", "value": {"row": int(grid.rows[0])}},
            {"id": "pd-scale", "property": "value", "value": "base"},
        ],
        [
            {"id": "pd-hidden", "property": "data", "value": []},
            {"id": "pd-key", "property": "data", "value": key},
            {"id": "scheme", "property": "data", "value": "dark"},
        ],
    )
    assert resp.status_code == 200, resp.data[:500]
    out = resp.get_json()["response"]
    assert {"pd-seq", "pd-ladder", "pd-frag-table", "pd-nav", "pd-mirror-next"} <= set(out)
    assert out["pd-nav"]["data"]["row"] == int(grid.rows[0])
    # The mirror goes to the page's script through a store (it keeps a user's zoom).
    fig = out["pd-mirror-next"]["data"]
    assert fig["layout"]["template"]["layout"]["font"]["color"] == "#c1c2c5"
    assert fig["layout"]["xaxis"]["range"] == mz
    thr = _callback_key(app, "pd-partner-body.children")
    resp = _post(
        client,
        thr,
        [{"id": "threshold", "property": "data", "value": 0.05}],
        [{"id": "pd-key", "property": "data", "value": key}],
    )
    assert resp.status_code == 200, resp.data[:500]
    out = resp.get_json()["response"]
    assert "At q ≤ 0.05" in json.dumps(out["pd-verdict"], ensure_ascii=False)


def test_figures_send_float32_arrays(open_fixture):
    rs = open_fixture("single")
    run, cid = _candidates(rs)[0]
    d = detail.get_detail(rs, run, cid)
    frags = detail_view.fragments_of(d.chromatogram)
    fig = detail_figures.mirror_figure(mirror(rs, d), frags)
    # Dash sends a figure as Plotly's JSON: numpy arrays as base64 typed arrays.
    observed = json.loads(to_json_plotly(fig))["data"][0]
    assert observed["x"]["dtype"] == "f4" and observed["y"]["dtype"] == "f4"
