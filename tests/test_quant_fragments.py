"""The fragments behind each quantity: the engine's table and the viewer-derived sources.

``smoke/out_qf`` and ``smoke/exp_qf`` were written with ``quant.fragment_detail`` on, so they
carry the engine's ``quant_fragments.parquet``; every other fixture predates it. Each source
must reproduce ``peptide_quant.quantity`` for every quantified candidate, and the
viewer-derived sources must name the engine's fragments and areas where both exist.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from dash.development.base_component import Component

from mumdia_viewer.data import across as A
from mumdia_viewer.data import quant_fragments as QF
from mumdia_viewer.data.errors import SchemaVersionError
from mumdia_viewer.data.schemas import check_version, supported_versions
from mumdia_viewer.ui import detail, detail_figures, runs, runs_figures
from mumdia_viewer.ui.state import PageContext


@pytest.fixture(scope="module")
def qf_fixtures(open_fixture):
    return {"single": open_fixture("single_qf"), "experiment": open_fixture("experiment_qf")}


def _quantified(rs, run) -> list[tuple[int, float, int]]:
    """(candidate_id, quantity, n_fragments_used) of every quantified peptide_quant row."""
    t = pq.read_table(
        run.artifact("peptide_quant").require(),
        columns=["candidate_id", "quantity", "n_fragments_used"],
    ).to_pylist()
    return [
        (r["candidate_id"], r["quantity"], r["n_fragments_used"])
        for r in t
        if r["quantity"] is not None
    ]


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


def _text(tree) -> str:
    return json.dumps(tree.to_plotly_json() if isinstance(tree, Component) else tree, default=str)


# --------------------------------------------------------------------------- schemas


def test_the_new_schema_versions_are_read():
    assert 6 in supported_versions("fragment_quant")
    assert {1, 5} <= supported_versions("fragment_quant")
    assert supported_versions("quant_fragments") == frozenset({1})
    check_version("fragment_quant", 6, where="x")
    check_version("quant_fragments", 1, where="x")
    with pytest.raises(SchemaVersionError, match="quant_fragments schema version 2"):
        check_version("quant_fragments", 2, where="x")


# --------------------------------------------------------------------------- the engine's table


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_the_engine_table_reproduces_every_quantity(qf_fixtures, name):
    rs = qf_fixtures[name]
    n = 0
    for run in rs.runs:
        assert run.artifact("quant_fragments") is not None, run.name
        for cid, quantity, used in _quantified(rs, run):
            acc = QF.quant_fragments(rs, run, cid)
            assert acc.source == QF.SOURCE_ENGINE and not acc.derived
            assert acc.rule == "observed_area"
            assert acc.verified is True, acc.message
            assert acc.reproduced == quantity  # bit for bit, in rank order
            assert acc.n_selected == used
            assert list(acc.rows.columns) == list(QF.COLUMNS)
            assert (acc.rows["weight"] == 1.0).all()
            n += 1
    assert n == {"single": 153, "experiment": 636}[name]


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_the_derived_sources_name_the_engines_fragments(qf_fixtures, name):
    rs = qf_fixtures[name]
    sources = [QF.SOURCE_CHROMATOGRAMS] + ([QF.SOURCE_FRAGMENT_QUANT] if name == "single" else [])
    for run in rs.runs:
        for cid, quantity, _ in _quantified(rs, run):
            engine = QF.quant_fragments(rs, run, cid, sources=[QF.SOURCE_ENGINE])
            for source in sources:
                acc = QF.quant_fragments(rs, run, cid, sources=[source])
                assert acc.source == source and acc.derived and acc.verified, acc.message
                assert acc.reproduced == quantity
                assert acc.selected_names() == engine.selected_names()
                assert acc.rows["area"].tolist() == engine.rows["area"].tolist()
                assert acc.rows["rank"].tolist() == engine.rows["rank"].tolist()


def test_an_experiments_run_fragment_quant_is_never_used(qf_fixtures):
    rs = qf_fixtures["experiment"]
    run = rs.runs[0]
    cid = _quantified(rs, run)[0][0]
    acc = QF.quant_fragments(rs, run, cid, sources=[QF.SOURCE_FRAGMENT_QUANT])
    assert acc.source == QF.SOURCE_NONE
    assert acc.message == QF.NOT_REPRODUCIBLE
    assert any("first-pass export" in t for t in acc.tried)


# --------------------------------------------------------------------------- older outputs


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("single", QF.SOURCE_FRAGMENT_QUANT, 150),
        ("single", QF.SOURCE_CHROMATOGRAMS, 150),
        ("tims", QF.SOURCE_FRAGMENT_QUANT, 238),
        ("experiment", QF.SOURCE_CHROMATOGRAMS, 564),
    ],
)
def test_older_outputs_are_derived_and_verified(open_fixture, name, source, expected):
    """Outputs without the engine's table: the single runs from fragment_quant and the
    configured rule (the diaPASEF run's cross_run rule falls back to predicted in a single
    run), the walked-window runs from their chromatograms, each quantity reproduced."""
    rs = open_fixture(name)
    n = 0
    for run in rs.runs:
        assert run.artifact("quant_fragments") is None
        for cid, quantity, used in _quantified(rs, run):
            acc = QF.quant_fragments(rs, run, cid, sources=[source])
            assert acc.source == source and acc.verified, (cid, acc.message, acc.tried)
            assert acc.reproduced == quantity
            assert acc.n_selected == used
            n += 1
    assert n == expected


def test_the_default_order_and_what_cannot_be_derived(open_fixture):
    tims = open_fixture("tims")
    run = tims.runs[0]
    cid = _quantified(tims, run)[0][0]
    # Baseline and envelope: the chromatograms cannot reproduce the areas.
    acc = QF.quant_fragments(tims, run, cid, sources=[QF.SOURCE_CHROMATOGRAMS])
    assert acc.source == QF.SOURCE_NONE and "quant.fragment_detail" in acc.message
    assert "baseline" in acc.tried[-1] and "envelope" in acc.tried[-1]
    # The default order takes fragment_quant first.
    acc = QF.quant_fragments(tims, run, cid)
    assert acc.source == QF.SOURCE_FRAGMENT_QUANT and acc.rule == "predicted"
    assert "derived" in acc.label and "verified" in acc.label


def test_a_quantity_that_does_not_match_is_not_reproducible(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    cid, quantity, used = _quantified(rs, run)[0]
    state = dict(
        state="quantified",
        quantity=quantity * 2,
        n_fragments_used=used,
        integration_apex_rt=None,
        integration_lo_rt=None,
        integration_hi_rt=None,
    )
    acc = QF.quant_fragments(rs, run, cid, quant=state)
    assert acc.source == QF.SOURCE_NONE and not acc.available
    assert acc.message == QF.NOT_REPRODUCIBLE
    assert len(acc.tried) == 3  # engine absent, fragment_quant and chromatograms refused


def test_the_cross_run_step_of_an_experiment_is_not_derived(open_fixture, monkeypatch):
    rs = open_fixture("experiment")
    run = rs.runs[0]
    cid = _quantified(rs, run)[0][0]
    monkeypatch.setattr(
        QF, "_quant_config", lambda rs: QF._QuantConfig(fragment_selection="cross_run")
    )
    acc = QF.quant_fragments(rs, run, cid)
    assert acc.source == QF.SOURCE_NONE and acc.message == QF.NOT_REPRODUCIBLE
    assert "cross-run step" in acc.tried[-1]


def test_the_check_of_each_rule():
    rows = QF._top_n_rows(
        ["y3", "y4", "b2"], np.array([10.0, 0.0, 5.0]), np.array([0.1, 1.0, 0.5]), "predicted", 2
    )
    # y4 has no area, so the predicted rule ranks b2 then y3.
    assert rows["rank"].tolist() == [2, 0, 1]
    assert rows["selected"].tolist() == [True, False, True]
    inputs = QF.QuantInputs("quantified", 15.0, 2, None, None, None)
    assert QF._check(rows, "predicted", inputs)[:2] == (15.0, True)
    weighted = rows.assign(weight=[2.0, np.nan, 0.5])
    assert QF.reproduce(weighted, "cross_run") == 2.0 * 10.0 + 0.5 * 5.0
    pattern = QF._check(rows.assign(weight=np.nan), "pattern_fit", inputs)
    assert pattern == (None, True, pattern[2]) and "fitted" in pattern[2]


# --------------------------------------------------------------------------- the pages


def test_the_precursor_page_marks_the_fragments_used(qf_fixtures):
    rs = qf_fixtures["single"]
    run = rs.runs[0]
    cid, _, used = _quantified(rs, run)[0]
    tree = detail.layout(PageContext(rs=rs, base="/", query={"cid": str(cid)}))
    text = _text(tree)
    assert "pd-qf-list" in text and "quant area" in text
    switch = [c for c in _walk(tree) if getattr(c, "id", None) == "pd-xic-quant"]
    assert len(switch) == 1 and switch[0].value == "highlight" and switch[0].style is None
    d = detail.get_detail(rs, "", cid)
    assert d.quant_fragments is not None and d.quant_fragments.source == QF.SOURCE_ENGINE
    from mumdia_viewer.ui import detail_view as view

    frags = view.fragments_of(d.chromatogram)
    for mode in detail_figures.QF_MODES:
        fig = detail_figures.xic_figure(d, frags, None, quant_mode=mode)
        roles = [t.meta.get("qf") for t in fig.data if isinstance(t.meta, dict)]
        assert roles.count("used") == used
        assert roles.count("fill") >= 1
        assert fig.layout.meta["qf"] == mode
        others = [t for t in fig.data if isinstance(t.meta, dict) and t.meta.get("qf") == "other"]
        if mode == "only":
            assert all(t.visible == "legendonly" for t in others)
        if mode == "off":
            fills = [t for t in fig.data if isinstance(t.meta, dict) and t.meta.get("qf") == "fill"]
            assert all(t.visible == "legendonly" for t in fills)


def test_a_derived_selection_is_marked_derived(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    cid = _quantified(rs, run)[0][0]
    tree = detail.layout(PageContext(rs=rs, base="/", query={"cid": str(cid)}))
    lst = [c for c in _walk(tree) if getattr(c, "id", None) == "pd-qf-list"]
    assert len(lst) == 1 and "derived" in _text(lst[0])
    assert "pd-th-derived" in _text(tree)


def test_the_across_runs_page_shows_the_fragment_areas(qf_fixtures):
    rs = qf_fixtures["experiment"]
    df = pq.read_table(rs.scored.path).to_pandas()
    df = df[df["label"] == "target"].sort_values("score", ascending=False)
    pep, z = str(df["peptidoform"].iloc[0]), int(df["charge"].iloc[0])
    a = A.precursor_across(rs, pep, z)
    xics = A.run_xics(rs, a)
    af = runs.fragment_areas(rs, a, xics)
    assert af.area.shape == (len(rs.runs), len(af.fragments))
    for i, row in enumerate(a.rows.to_dict("records")):
        if row["state"] == "quantified":
            assert af.selected[i].sum() == row["n_fragments_used"]
    assert af.any_available and not af.any_derived
    fig = runs_figures.fragment_area_figure(af)
    assert len([s for s in fig.layout.shapes if s.name == "used"]) == int(af.selected.sum())
    tree = runs.layout(PageContext(rs=rs, base="/", query={"peptidoform": pep, "charge": str(z)}))
    graphs = {c.id["name"] for c in _walk(tree) if type(c).__name__ == "Graph"}
    assert "xr-qf" in graphs
