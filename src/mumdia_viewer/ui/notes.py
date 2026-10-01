"""Validation notes: the verdict card of the precursor page and the notes page.

A note is the user's verdict on one scored row (accepted, rejected or unsure, with a
comment), kept by :class:`data.notes.NoteBook` in the viewer's notes directory, never in
the run directory. On the precursor page the keys A, R and U set and save a verdict
(``assets/notes.js``); the notes page lists every note of the result set and exports
them as TSV or JSON.
"""

from __future__ import annotations

from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, dcc, html, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.notes import VERDICTS, Note, NoteBook

from .icons import icon
from .state import PageContext, href
from .widgets import fmt_q, section

VERDICT_COLOURS = {"accepted": "green", "rejected": "red", "unsure": "yellow"}
VERDICT_WORDS = {"accepted": "Accepted", "rejected": "Rejected", "unsure": "Unsure"}
SNAPSHOT = ("peptidoform", "charge", "protein_group", "label", "q_value", "score")


def _book(rs: ResultSet) -> tuple[NoteBook | None, str | None]:
    try:
        return NoteBook(rs), None
    except ViewerError as exc:
        return None, str(exc)


def _when(stamp: str) -> str:
    """An ISO time in UTC as ``2026-10-01 16:04:05 UTC``."""
    return stamp.replace("T", " ").replace("+00:00", "") + " UTC"


def status_text(note: Note | None) -> str:
    if note is None:
        return "No verdict yet."
    who = f" by {note.author}" if note.author else ""
    return f"{VERDICT_WORDS[note.verdict]}, saved {_when(note.updated)}{who}."


def verdict_badge(verdict: str | None) -> Any:
    if not verdict:
        return dmc.Badge("no verdict", color="gray", variant="light", size="sm")
    return dmc.Badge(
        VERDICT_WORDS[verdict],
        color=VERDICT_COLOURS[verdict],
        variant="light",
        size="sm",
        style={"textTransform": "none"},
    )


# --------------------------------------------------------------------------- precursor card


def note_card(ctx: PageContext, run: str, cid: int, scored: dict[str, Any]) -> Any:
    """The verdict card of a precursor page."""
    book, error = _book(ctx.rs)
    note = None
    if book is not None:
        try:
            note = book.get(run, cid)
        except ViewerError as exc:
            error = str(exc)
    snapshot = {k: scored.get(k) for k in SNAPSHOT}
    controls = dmc.Group(
        [
            dmc.SegmentedControl(
                id="pd-note-verdict",
                data=[
                    {"value": "accepted", "label": "Accepted"},
                    {"value": "rejected", "label": "Rejected"},
                    {"value": "unsure", "label": "Unsure"},
                    {"value": "", "label": "No verdict"},
                ],
                value=note.verdict if note else "",
                size="sm",
                radius="xl",
            ),
            dmc.Textarea(
                id="pd-note-comment",
                value=note.comment if note else "",
                placeholder="Comment (optional): what you checked, what looks wrong",
                autosize=True,
                minRows=1,
                maxRows=4,
                style={"flex": "1 1 260px", "minWidth": "200px"},
            ),
            dmc.Button(
                "Save",
                id="pd-note-save",
                n_clicks=0,
                variant="light",
                leftSection=icon("check", 14),
                disabled=book is None,
            ),
        ],
        align="flex-start",
        gap="sm",
    )
    return section(
        "Your verdict",
        controls,
        dmc.Text(
            error or status_text(note),
            id="pd-note-status",
            size="xs",
            c="red" if error else "dimmed",
            mt=6,
        ),
        dcc.Store(id="pd-note-key", data={"run": run, "cid": int(cid), "snapshot": snapshot}),
        subtitle="Kept by the viewer in its notes directory, not in the run. Keys: A accepted, "
        "R rejected, U unsure (each saves at once).",
        right=dcc.Link(
            dmc.Group([icon("table", 14), dmc.Text("All notes", size="sm")], gap=4),
            href=href(ctx.base, "notes"),
        ),
    )


def save_note(
    rs: ResultSet, key: dict[str, Any] | None, verdict: str, comment: str
) -> tuple[str, bool]:
    """Save (or, without a verdict, delete) a note; (status text, failed)."""
    book, error = _book(rs)
    if book is None or not key:
        return error or "Nothing to save.", True
    try:
        if not verdict:
            gone = book.delete(key.get("run") or "", key.get("cid"))
            return ("The verdict was removed." if gone else "No verdict yet."), False
        note = book.put(
            key.get("run") or "",
            key.get("cid"),
            verdict,
            comment or "",
            snapshot=key.get("snapshot") or {},
        )
    except ViewerError as exc:
        return str(exc), True
    return status_text(note), False


# --------------------------------------------------------------------------- notes page


def _rows(base: str, notes: list[Note]) -> list[dict[str, Any]]:
    out = []
    for n in notes:
        out.append(
            {
                "verdict": n.verdict,
                "peptidoform": n.peptidoform,
                "charge": n.charge,
                "run": n.run,
                "candidate_id": n.candidate_id,
                "protein_group": n.protein_group,
                "label": n.label,
                "q_value": n.q_value,
                "q_text": fmt_q(n.q_value),
                "score": n.score,
                "comment": n.comment,
                "author": n.author,
                "updated": _when(n.updated),
                "_href": href(base, "precursor", {"run": n.run, "cid": n.candidate_id}),
            }
        )
    return out


COLUMNS = [
    {
        "field": "verdict",
        "headerName": "verdict",
        "width": 110,
        "cellClassRules": {
            "nt-accepted": "params.value == 'accepted'",
            "nt-rejected": "params.value == 'rejected'",
            "nt-unsure": "params.value == 'unsure'",
        },
    },
    {
        "field": "peptidoform",
        "headerName": "peptidoform",
        "cellRenderer": "MvPeptidoform",
        "minWidth": 220,
        "flex": 1,
    },
    {"field": "charge", "headerName": "z", "width": 70},
    {"field": "run", "headerName": "run", "width": 90},
    {"field": "candidate_id", "headerName": "candidate", "width": 110},
    {"field": "protein_group", "headerName": "protein group", "minWidth": 160},
    {
        "field": "q_text",
        "headerName": "q_value",
        "width": 110,
        "headerTooltip": "q_value of the row when the note was first saved (the engine's)",
    },
    {
        "field": "score",
        "headerName": "score",
        "width": 100,
        "valueFormatter": {"function": "params.value == null ? '' : params.value.toFixed(4)"},
    },
    {
        "field": "comment",
        "headerName": "comment",
        "minWidth": 260,
        "flex": 2,
        "wrapText": True,
        "autoHeight": True,
    },
    {"field": "author", "headerName": "author", "width": 90},
    {"field": "updated", "headerName": "updated", "width": 180},
]


def layout(ctx: PageContext) -> Any:
    book, error = _book(ctx.rs)
    if book is None:
        return section("Validation notes", dmc.Alert(error, color="red", variant="light"))
    try:
        notes = book.notes()
    except ViewerError as exc:
        return section("Validation notes", dmc.Alert(str(exc), color="red", variant="light"))
    counts = {v: sum(n.verdict == v for n in notes) for v in VERDICTS}
    header = html.Div(
        [
            dmc.Text("Validation", className="mv-eyebrow"),
            html.Div("Notes", className="mv-title"),
            dmc.Text(str(book.path), className="mv-path", c="dimmed", mt=2),
        ]
    )
    summary = dmc.Group(
        [
            dmc.Badge(
                f"{counts[v]:,} {VERDICT_WORDS[v].lower()}",
                color=VERDICT_COLOURS[v],
                variant="light",
                size="lg",
                style={"textTransform": "none"},
            )
            for v in VERDICTS
        ],
        gap="xs",
    )
    actions = dmc.Group(
        [
            dmc.SegmentedControl(
                id="nt-filter",
                data=[{"value": "all", "label": "All"}]
                + [{"value": v, "label": VERDICT_WORDS[v]} for v in VERDICTS],
                value="all",
                size="xs",
                radius="xl",
            ),
            dmc.Button(
                "TSV",
                id="nt-tsv",
                n_clicks=0,
                size="xs",
                variant="light",
                leftSection=icon("external", 13),
                disabled=not notes,
            ),
            dmc.Button(
                "JSON",
                id="nt-json",
                n_clicks=0,
                size="xs",
                variant="light",
                leftSection=icon("external", 13),
                disabled=not notes,
            ),
            dcc.Download(id="nt-download"),
        ],
        gap="xs",
    )
    grid = dag.AgGrid(
        id="nt-grid",
        rowData=_rows(ctx.base, notes),
        columnDefs=COLUMNS,
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        dashGridOptions={
            "rowHeight": 30,
            "headerHeight": 34,
            "animateRows": False,
            "overlayNoRowsTemplate": "No notes yet. Open a precursor page and give a verdict "
            "(keys A, R, U).",
        },
        className="ag-theme-quartz nt-grid",
        style={"height": "min(70vh, 640px)", "width": "100%"},
    )
    body = section(
        "Notes of this result set",
        grid,
        count=len(notes),
        help="Each note is a verdict on one scored row, with a snapshot of the row taken "
        "when the note was first saved. Click a row to open its precursor page.",
        right=actions,
        p="sm",
    )
    return dmc.Stack([header, summary, body], gap="lg")


def register(app, get_rs, base: str) -> None:
    @app.callback(
        Output("pd-note-status", "children"),
        Output("pd-note-status", "c"),
        Output("notes-version", "data", allow_duplicate=True),
        Input("pd-note-save", "n_clicks"),
        State("pd-note-verdict", "value"),
        State("pd-note-comment", "value"),
        State("pd-note-key", "data"),
        State("notes-version", "data"),
        prevent_initial_call=True,
    )
    def save(n, verdict, comment, key, version):
        if not n:
            return no_update, no_update, no_update
        text, failed = save_note(get_rs(), key, verdict or "", comment or "")
        return text, "red" if failed else "dimmed", (version or 0) + (0 if failed else 1)

    @app.callback(
        Output("nt-grid", "filterModel"),
        Input("nt-filter", "value"),
        prevent_initial_call=True,
    )
    def filter_rows(value):
        if value in (None, "all"):
            return {}
        return {"verdict": {"filterType": "text", "type": "equals", "filter": value}}

    @app.callback(
        Output("nt-download", "data"),
        Input("nt-tsv", "n_clicks"),
        Input("nt-json", "n_clicks"),
        prevent_initial_call=True,
    )
    def download(n_tsv, n_json):
        from dash import ctx

        book, _error = _book(get_rs())
        if book is None:
            return no_update
        stem = book.path.stem
        if ctx.triggered_id == "nt-json":
            return {"content": book.to_json(), "filename": f"{stem}.notes.json"}
        return {"content": book.to_tsv(), "filename": f"{stem}.notes.tsv"}

    app.clientside_callback(
        """function (cell, rows) {
            if (!cell || !rows || cell.colId === "comment") {
                return window.dash_clientside.no_update;
            }
            const row = rows[cell.rowIndex];
            return row && row._href ? row._href : window.dash_clientside.no_update;
        }""",
        Output("url", "href", allow_duplicate=True),
        Input("nt-grid", "cellClicked"),
        State("nt-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
