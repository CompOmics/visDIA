"""The coverage views: what the server sends, the pages' slots and callbacks, the CLI flag."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plotly.utils import PlotlyJSONEncoder

from mumdia_viewer.cli import _parser
from mumdia_viewer.data.fasta import Fasta, protein_coverage
from mumdia_viewer.ui import browser, coverage, detail
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext

FIXTURE_FASTA = Path(__file__).parent / "fixtures" / "smoke" / "test_data" / "fixture.fasta"
GROUP = "sp|FIXT14|FIX14_TEST"


@pytest.fixture(scope="module")
def fasta() -> Fasta:
    return Fasta.read([FIXTURE_FASTA])


def _json(tree) -> str:
    return json.dumps(tree, cls=PlotlyJSONEncoder)


def _props(component) -> dict:
    return component.to_plotly_json()["props"]


def _find(tree, cls: str):
    """Every component in ``tree`` whose className contains ``cls``."""
    out = []

    def walk(node):
        if isinstance(node, list | tuple):
            for n in node:
                walk(n)
            return
        if not hasattr(node, "to_plotly_json"):
            return
        if cls in (getattr(node, "className", "") or "").split():
            out.append(node)
        for attr in ("children", "label"):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)

    walk(tree)
    return out


def test_the_bar_carries_the_states_and_spans(open_fixture, fasta):
    rs = open_fixture("single")
    cov = protein_coverage(rs, fasta, GROUP, threshold=0.01)
    selected = cov.spans[0].base_peptide_id
    bar = coverage.bar(cov, selected=selected)
    props = _props(bar)
    assert props["data-length"] == str(cov.length)
    states = props["data-states"]
    assert len(states) == cov.length and set(states) <= {"0", "1", "2"}
    assert states == "".join(str(int(s)) for s in cov.states())
    spans = json.loads(props["data-spans"])
    assert [(s[0], s[1], s[2], bool(s[3]), s[4]) for s in spans] == [
        (s.start, s.end, s.base_peptide_id, s.passes, s.sequence) for s in cov.spans
    ]
    assert props["data-selected"] == str(selected)
    assert props["data-sequence"] == cov.entry.sequence
    assert props["data-group"] == GROUP and props["data-threshold"] == "0.01"


def test_the_strip_names_what_it_shows(open_fixture, fasta):
    rs = open_fixture("single")
    cov = protein_coverage(rs, fasta, GROUP, threshold=0.05)
    strip = coverage.strip(cov)
    summary = _find(strip, "mvc-summary")[0]
    assert summary.children == coverage.summary_text(cov)
    assert "passing" in summary.children and "all" in summary.children
    text = coverage.help_text(cov)
    assert "Computed by the viewer" in text and "0.05" in text and "Green" in text
    assert _find(strip, "mvc-bar")


def test_messages_without_a_fasta_or_for_a_decoy(open_fixture, fasta):
    rs = open_fixture("single")
    none = browser.coverage_answer(rs, None, {"group": GROUP})
    assert "mvc-message" in none.className and "--fasta" in _json(none)
    decoy = browser.coverage_answer(rs, fasta, {"group": f"DECOY_{GROUP}", "t": 0.01})
    assert "mvc-warn" in decoy.className
    empty = browser.coverage_answer(rs, fasta, {"group": ""})
    assert "Select a protein group" in _json(empty)
    ok = browser.coverage_answer(rs, fasta, {"group": GROUP, "t": 0.01, "peptide": None})
    assert "mvc-strip" in ok.className and _find(ok, "mvc-bar")


def test_the_precursor_card(open_fixture, fasta):
    rs = open_fixture("single")
    cov = protein_coverage(rs, fasta, GROUP)
    body = detail.coverage_body(
        rs, fasta, {"group": GROUP, "peptide": cov.spans[0].base_peptide_id}, 0.01
    )
    for cls in ("mvc-bar", "mvc-lanes", "mvc-seq"):
        found = _find(body, cls)
        assert found and _props(found[0])["data-selected"] == str(cov.spans[0].base_peptide_id)
    assert "--fasta" in _json(detail.coverage_body(rs, None, {"group": GROUP}, None))


def test_the_pages_have_their_slots(open_fixture, fasta):
    rs = open_fixture("single")
    page = browser.layout(PageContext(rs=rs, base="/", fasta=fasta))
    text = _json(page)
    assert '"ib-cov"' in text and '"ib-cov-req"' in text
    no_fasta = _json(browser.layout(PageContext(rs=rs, base="/")))
    assert "--fasta" in no_fasta
    peptide_level = browser.layout(
        PageContext(rs=rs, base="/", fasta=fasta, query={"unit": "peptide"})
    )
    assert '"ib-cov"' not in _json(peptide_level)


def test_the_app_takes_a_fasta(open_fixture, fasta):
    app = create_app(open_fixture("single"), url_base="/x/", fasta=fasta)
    assert app.mv_fasta() is fasta
    deps = app.server.test_client().get("/x/_dash-dependencies").get_json()
    outputs = " ".join(d["output"] for d in deps)
    assert "ib-cov.children" in outputs and "pd-cov.children" in outputs


def test_the_cli_takes_fasta_files():
    args = _parser().parse_args(["run_dir", "--fasta", "a.fasta", "--fasta", "b.fasta"])
    assert args.fasta == ["a.fasta", "b.fasta"]
    assert _parser().parse_args(["run_dir"]).fasta == []
