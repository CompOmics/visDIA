"""Validation notes: storage outside the run, round trips, validation and export."""

from __future__ import annotations

import io
import json
import threading

import pandas as pd
import pytest

from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.notes import VERDICTS, NoteBook, default_notes_root

SNAP = {
    "peptidoform": "PEPTIDEK",
    "charge": 2,
    "protein_group": "P1",
    "label": "target",
    "q_value": 0.001,
    "score": 0.9,
    "ignored": "x",
}


def test_round_trip_and_snapshot(open_fixture, tmp_path):
    rs = open_fixture("single")
    book = NoteBook(rs, root=tmp_path / "notes")
    assert book.notes() == [] and book.get("", 5) is None
    first = book.put("", 5, "unsure", "  check the y ions  ", snapshot=SNAP)
    assert first.verdict == "unsure" and first.comment == "check the y ions"
    assert first.peptidoform == "PEPTIDEK" and first.charge == 2 and first.q_value == 0.001
    assert book.path.parent == tmp_path / "notes" and book.path.exists()
    second = book.put("", 5, "accepted", "fine after all", snapshot={"peptidoform": "OTHER"})
    assert second.created == first.created and second.verdict == "accepted"
    assert second.peptidoform == "PEPTIDEK"  # the snapshot of the first note is kept
    assert book.get("", 5) == second
    book.put("run", 7, "rejected", "", snapshot=None)  # "run" is the single run's name
    assert {n.candidate_id for n in book.notes()} == {5, 7}
    assert book.counts() == {"accepted": 1, "rejected": 1, "unsure": 0}
    assert book.delete("", 7) and not book.delete("", 7)
    assert [n.candidate_id for n in book.notes()] == [5]
    assert not list((tmp_path / "notes").glob(".notes-*"))  # no temporary file left


def test_export(open_fixture, tmp_path):
    rs = open_fixture("single")
    book = NoteBook(rs, root=tmp_path)
    book.put("", 1, "accepted", "tab\tand\nnewline", snapshot=SNAP)
    frame = pd.read_csv(io.StringIO(book.to_tsv()), sep="\t")
    assert list(frame["candidate_id"]) == [1] and frame["verdict"][0] == "accepted"
    data = json.loads(book.to_json())
    assert data["scored_identity"] == rs.scored.identity() and len(data["notes"]) == 1
    assert data["notes"][0]["comment"] == "tab\tand\nnewline"


def test_validation(open_fixture, tmp_path):
    rs = open_fixture("single")
    book = NoteBook(rs, root=tmp_path)
    with pytest.raises(ViewerError, match="verdict"):
        book.put("", 1, "maybe")
    with pytest.raises(ViewerError, match="no run name"):
        book.put("r0", 1, "accepted")
    with pytest.raises(ViewerError, match="integer"):
        book.put("", "x", "accepted")
    with pytest.raises(ViewerError, match="not negative"):
        book.put("", -1, "accepted")
    with pytest.raises(ViewerError, match="longer"):
        book.put("", 1, "accepted", "x" * 5000)
    assert set(VERDICTS) == {"accepted", "rejected", "unsure"}


def test_experiment_runs(open_fixture, tmp_path):
    rs = open_fixture("experiment")
    book = NoteBook(rs, root=tmp_path)
    run = rs.runs[1].name
    book.put(run, 3, "accepted")
    assert book.get(run, 3).run == run
    with pytest.raises(ViewerError, match="not a run"):
        book.put("", 3, "accepted")


def test_never_inside_the_run(open_fixture):
    rs = open_fixture("single")
    with pytest.raises(ViewerError, match="inside"):
        NoteBook(rs, root=rs.root / "notes")


def test_default_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MUMDIA_VIEWER_NOTES_DIR", str(tmp_path / "n"))
    assert default_notes_root() == tmp_path / "n"
    monkeypatch.delenv("MUMDIA_VIEWER_NOTES_DIR")
    assert default_notes_root().name == "notes"


def test_a_broken_file_is_reported(open_fixture, tmp_path):
    rs = open_fixture("single")
    book = NoteBook(rs, root=tmp_path)
    tmp_path.mkdir(exist_ok=True)
    book.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ViewerError, match="not valid JSON"):
        book.notes()


def test_concurrent_writes_keep_every_note(open_fixture, tmp_path):
    rs = open_fixture("single")
    book = NoteBook(rs, root=tmp_path)

    def write(i: int) -> None:
        book.put("", i, VERDICTS[i % 3], f"note {i}")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(book.notes()) == 40
