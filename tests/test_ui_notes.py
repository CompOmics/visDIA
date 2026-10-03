"""The notes UI: the precursor card, saving through it, the notes page, the shell."""

from __future__ import annotations

import json

from plotly.utils import PlotlyJSONEncoder

from mumdia_viewer.data.notes import NoteBook
from mumdia_viewer.ui import notes
from mumdia_viewer.ui.app import create_app
from mumdia_viewer.ui.state import PageContext


def _json(tree) -> str:
    return json.dumps(tree, cls=PlotlyJSONEncoder)


def test_card_save_and_page(open_fixture, tmp_path, monkeypatch):
    monkeypatch.setenv("MUMDIA_VIEWER_NOTES_DIR", str(tmp_path / "notes"))
    rs = open_fixture("single")
    ctx = PageContext(rs=rs, base="/x/")
    scored = {
        "peptidoform": "PEPK",
        "charge": 2,
        "protein_group": "P",
        "label": "target",
        "q_value": 0.002,
        "score": 0.8,
    }
    card = _json(notes.note_card(ctx, "", 11, scored))
    assert "pd-note-verdict" in card and "No verdict yet." in card
    key = {"run": "", "cid": 11, "snapshot": scored}
    text, failed = notes.save_note(rs, key, "rejected", "noisy apex")
    assert not failed and text.startswith("Rejected, saved ")
    assert NoteBook(rs).get("", 11).comment == "noisy apex"
    again = _json(notes.note_card(ctx, "", 11, scored))
    assert '"value": "rejected"' in again and "noisy apex" in again
    page = _json(notes.layout(ctx))
    assert "nt-grid" in page and "PEPK" in page and "1 rejected" in page
    text, failed = notes.save_note(rs, key, "", "")
    assert not failed and "removed" in text and NoteBook(rs).get("", 11) is None
    text, failed = notes.save_note(rs, {"run": "r9", "cid": 1}, "accepted", "")
    assert failed and "no run name" in text


def test_the_shell_counts_notes(open_fixture, tmp_path, monkeypatch):
    monkeypatch.setenv("MUMDIA_VIEWER_NOTES_DIR", str(tmp_path / "n2"))
    app = create_app(open_fixture("single"), url_base="/y/")
    client = app.server.test_client()
    layout = json.dumps(client.get("/y/_dash-layout").get_json())
    assert "nav-notes-count" in layout and "notes-version" in layout
    outputs = " ".join(d["output"] for d in client.get("/y/_dash-dependencies").get_json())
    assert "nav-notes-count.children" in outputs and "nt-download.data" in outputs
    assert "pd-note-status.children" in outputs


def test_grid_rows_carry_verdicts_on_precursors_only(open_fixture, tmp_path, monkeypatch):
    from mumdia_viewer.data.tables import TableQuery, identification_table
    from mumdia_viewer.ui.browser_grid import records

    monkeypatch.setenv("MUMDIA_VIEWER_NOTES_DIR", str(tmp_path / "n3"))
    rs = open_fixture("single")
    page = identification_table(rs, TableQuery(unit="precursor", limit=5))
    cid = int(page.rows["candidate_id"].iloc[1])
    NoteBook(rs).put("", cid, "unsure", "check MS1")
    rows = records(page.rows, "/x/", "precursor", notes=notes.note_map(rs))
    marked = [r for r in rows if r.get("_note")]
    assert [(r["candidate_id"], r["_note"], r["_note_text"]) for r in marked] == [
        (cid, "unsure", "check MS1")
    ]
    peptides = identification_table(rs, TableQuery(unit="peptide", limit=5))
    rows = records(peptides.rows, "/x/", "peptide", notes=notes.note_map(rs))
    assert not any("_note" in r for r in rows)
