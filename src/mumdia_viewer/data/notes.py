"""Validation notes: a person's verdict on an identification, kept by the viewer.

A note marks one scored row (a run and a candidate) as accepted, rejected or unsure, with
a comment. Notes are the user's own data, so they are kept in the viewer's notes
directory, never in a run directory, and not in the viewer's cache (which may be
cleared):

* ``MUMDIA_VIEWER_NOTES_DIR`` when it is set;
* ``%APPDATA%\\mumdia-viewer\\notes`` on Windows;
* ``~/Library/Application Support/mumdia-viewer/notes`` on macOS;
* ``$XDG_DATA_HOME/mumdia-viewer/notes`` (when absolute) or
  ``~/.local/share/mumdia-viewer/notes`` elsewhere.

There is one JSON file per result set, named by the identity of its scored table (the
content hash the engine recorded), so the notes follow the result set when its directory
moves. A note keeps a snapshot of the row it judges (peptidoform, charge, protein group,
label, ``q_value``, score), so an export reads without the run. Writes are atomic (a
temporary file, then ``os.replace``) and serialized within the process.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import ResultSet
from .errors import ViewerError

__all__ = ["VERDICTS", "Note", "NoteBook", "default_notes_root"]

VERDICTS = ("accepted", "rejected", "unsure")
ENV_VAR = "MUMDIA_VIEWER_NOTES_DIR"
COMMENT_MAX = 4000
FORMAT = 1

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def default_notes_root() -> Path:
    """The notes directory (see the module docstring); not created here."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return (
            (Path(base) if base else Path.home() / "AppData" / "Roaming")
            / "mumdia-viewer"
            / "notes"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mumdia-viewer" / "notes"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local" / "share"
    return base / "mumdia-viewer" / "notes"


def _lock(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # getuser raises several types when no name is known
        return ""


def _clean(value: Any) -> Any:
    """A JSON-ready scalar (numpy and pandas scalars to Python, NaN to None)."""
    if hasattr(value, "item"):
        with contextlib.suppress(ValueError, AttributeError):
            value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


@dataclass(frozen=True)
class Note:
    """One verdict. ``run`` is the run name (empty in a single run)."""

    run: str
    candidate_id: int
    verdict: str
    comment: str
    author: str
    created: str
    updated: str
    peptidoform: str | None = None
    charge: int | None = None
    protein_group: str | None = None
    label: str | None = None
    q_value: float | None = None
    score: float | None = None

    @property
    def key(self) -> str:
        return f"{self.run}:{self.candidate_id}"


SNAPSHOT_FIELDS = ("peptidoform", "charge", "protein_group", "label", "q_value", "score")
NOTE_FIELDS = tuple(f.name for f in fields(Note))


class NoteBook:
    """The notes of one result set."""

    def __init__(self, rs: ResultSet, root: Path | None = None) -> None:
        self.rs = rs
        self.root = Path(root) if root is not None else default_notes_root()
        identity = re.sub(r"[^A-Za-z0-9_.-]", "_", rs.scored.identity())
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", rs.root.name)[:60] or "results"
        self.path = self.root / f"{name}__{identity[:80]}.json"
        resolved = self.path.resolve()
        for run in rs.runs:
            if resolved.is_relative_to(Path(run.root).resolve()):
                raise ViewerError("The notes directory is inside a run directory; set another.")
        if resolved.is_relative_to(rs.root.resolve()):
            raise ViewerError("The notes directory is inside the result directory; set another.")

    # ------------------------------------------------------------------ storage

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ViewerError(f"cannot read the notes file {self.path}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ViewerError(f"the notes file {self.path} is not valid JSON: {exc}") from exc
        notes = data.get("notes") if isinstance(data, dict) else None
        return notes if isinstance(notes, dict) else {}

    def _write(self, notes: dict[str, dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "result_set": str(self.rs.root),
            "scored_identity": self.rs.scored.identity(),
            "notes": notes,
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".notes-", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------ queries

    def _check(self, run: str, cid: Any) -> tuple[str, int]:
        run = str(run or "")
        if self.rs.is_experiment:
            if run not in {r.name for r in self.rs.runs}:
                raise ViewerError(f"{run!r} is not a run of this experiment.")
        elif run not in ("", "run"):
            raise ViewerError("A single run has no run name; leave it empty.")
        else:
            run = ""
        try:
            number = int(cid)
        except (TypeError, ValueError):
            raise ViewerError(f"candidate id {cid!r} is not an integer.") from None
        if number < 0:
            raise ViewerError("A candidate id is not negative.")
        return run, number

    def get(self, run: str, cid: Any) -> Note | None:
        run, number = self._check(run, cid)
        with _lock(self.path):
            raw = self._read().get(f"{run}:{number}")
        return _note(raw)

    def notes(self) -> list[Note]:
        """Every note, the most recently updated first."""
        with _lock(self.path):
            raw = self._read()
        out = [n for n in (_note(v) for v in raw.values()) if n is not None]
        return sorted(out, key=lambda n: n.updated, reverse=True)

    def counts(self) -> dict[str, int]:
        found = {v: 0 for v in VERDICTS}
        for n in self.notes():
            found[n.verdict] = found.get(n.verdict, 0) + 1
        return found

    # ------------------------------------------------------------------ changes

    def put(
        self,
        run: str,
        cid: Any,
        verdict: str,
        comment: str = "",
        *,
        snapshot: Mapping[str, Any] | None = None,
    ) -> Note:
        """Add or change the note of a row; the snapshot is kept from the first note."""
        run, number = self._check(run, cid)
        if verdict not in VERDICTS:
            raise ViewerError(f"verdict {verdict!r} is not one of {', '.join(VERDICTS)}.")
        text = str(comment or "").strip()
        if len(text) > COMMENT_MAX:
            raise ViewerError(f"the comment is longer than {COMMENT_MAX} characters.")
        key = f"{run}:{number}"
        with _lock(self.path):
            notes = self._read()
            old = notes.get(key) or {}
            now = _now()
            record = {
                "run": run,
                "candidate_id": number,
                "verdict": verdict,
                "comment": text,
                "author": old.get("author") or _user(),
                "created": old.get("created") or now,
                "updated": now,
            }
            for f in SNAPSHOT_FIELDS:
                record[f] = old.get(f) if f in old else _clean((snapshot or {}).get(f))
            notes[key] = record
            self._write(notes)
        note = _note(record)
        assert note is not None
        return note

    def delete(self, run: str, cid: Any) -> bool:
        run, number = self._check(run, cid)
        with _lock(self.path):
            notes = self._read()
            if notes.pop(f"{run}:{number}", None) is None:
                return False
            self._write(notes)
        return True

    # ------------------------------------------------------------------ export

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(n) for n in self.notes()], columns=list(NOTE_FIELDS))

    def to_tsv(self) -> str:
        return self.frame().to_csv(sep="\t", index=False, lineterminator="\n")

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": FORMAT,
                "result_set": str(self.rs.root),
                "scored_identity": self.rs.scored.identity(),
                "notes": [asdict(n) for n in self.notes()],
            },
            indent=1,
        )


def _note(raw: Mapping[str, Any] | None) -> Note | None:
    if not raw or raw.get("verdict") not in VERDICTS:
        return None
    try:
        return Note(**{f: raw.get(f) for f in NOTE_FIELDS if f in raw or f in SNAPSHOT_FIELDS})
    except TypeError:
        return None
