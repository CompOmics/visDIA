"""Precursor detail page (P0 view 3): why one identification was accepted."""

from __future__ import annotations

import threading
from collections import OrderedDict

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.detail import (
    PrecursorDetail,
    detail_percentiles,
    mirror,
    precursor_detail,
)

from . import figures
from .components import CARD, MUTED, SECTION, fmt, graph, notices, precursor_href, table

_CACHE: OrderedDict[tuple, PrecursorDetail] = OrderedDict()
_LOCK = threading.Lock()
_MAX = 32


def get_detail(rs: ResultSet, run: str, cid: int) -> PrecursorDetail:
    """The assembled detail, kept for the mirror-plot callbacks (a small LRU)."""
    key = (id(rs), run, int(cid))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    d = precursor_detail(rs, run, cid)
    with _LOCK:
        _CACHE[key] = d
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return d


def _header(d: PrecursorDetail, base: str) -> html.Div:
    s = d.scored
    label = s.get("label")
    badge = {
        "background": "#d62728" if label == "decoy" else "#2ca02c",
        "color": "#fff",
        "borderRadius": "4px",
        "padding": "1px 6px",
        "marginLeft": "8px",
        "fontSize": "0.8em",
    }
    run = d.run.label if d.run.name else "single run"
    return html.Div(
        [
            html.H2([f"{s.get('peptidoform')}  {s.get('charge')}+", html.Span(label, style=badge)]),
            html.Div(
                f"protein {s.get('protein')} | candidate {d.candidate_id} | run {run} | "
                f"apex {fmt(s.get('apex_rt'))} s | score {fmt(s.get('score'))} | "
                f"selected peak rank {d.selected_peak_rank} of {len(d.peaks)}",
                style=MUTED,
            ),
            dcc.Link("back to the identifications", href=f"{base}identifications"),
        ]
    )


def _q_table(d: PrecursorDetail) -> html.Div:
    rows = [
        {
            "q column": q.column,
            "value": fmt(q.display_value)
            if q.display_value is not None
            else "not the group winner",
            "unit": q.unit,
            "scope": q.scope,
        }
        for q in d.q_values
    ]
    return html.Div([html.H3("q values", style=SECTION), table(pd.DataFrame(rows))])


def _evidence(d: PrecursorDetail) -> html.Div:
    rows = [
        {
            "group": e.group,
            "evidence": e.label,
            "value": fmt(e.value),
            "unit": e.unit,
            "source": e.source,
            "note": e.note or "",
            "derived": "yes" if e.derived else "",
        }
        for e in d.evidence
        if e.group != "q"
    ]
    return html.Div(
        [html.H3("Evidence summary", style=SECTION), table(pd.DataFrame(rows), page_size=60)]
    )


def _competition(d: PrecursorDetail, base: str) -> html.Div:
    df = d.competition.copy()
    if df.empty:
        return html.Div()
    keep = [
        "run",
        "label",
        "peptidoform",
        "charge",
        "score",
        "q_value",
        "run_psm_q",
        "precursor_q",
        "peptide_q_value",
        "wins_peptide",
        "wins_precursor",
        "is_this_row",
    ]
    df = df[[c for c in keep if c in df]]
    return html.Div(
        [
            html.H3("Base-peptide competition (picked target-decoy, all runs)", style=SECTION),
            html.Div(
                "wins_peptide / wins_precursor: the engine's winner rule (score, decoy first "
                "on a tie, file row order) applied by the viewer; it reproduces the engine's "
                "sparse q columns.",
                style=MUTED,
            ),
            table(df, page_size=12),
        ]
    )


def _partner(d: PrecursorDetail, base: str) -> html.Div:
    p = d.partner
    children = [html.H3("Exact decoy partner", style=SECTION), html.Div(p.reason, style=MUTED)]
    if p.candidate_id is not None:
        if p.rows.empty:
            children.append(html.Div(f"candidate {p.candidate_id}: not scored in any run"))
        else:
            children.append(
                table(
                    p.rows[
                        [
                            c
                            for c in (
                                "run",
                                "label",
                                "peptidoform",
                                "charge",
                                "score",
                                "q_value",
                                "run_psm_q",
                            )
                            if c in p.rows
                        ]
                    ]
                )
            )
            first = p.rows.iloc[0]
            children.append(
                dcc.Link(
                    f"open candidate {p.candidate_id}",
                    href=precursor_href(
                        base, str(first["run"]) if d.run.name else "", p.candidate_id
                    ),
                )
            )
    return html.Div(children)


def layout(rs: ResultSet, base: str, run: str, cid: int) -> html.Div:
    try:
        d = get_detail(rs, run, cid)
    except ViewerError as exc:
        return html.Div([html.H2("Precursor detail"), html.Div(str(exc))])
    m = None
    try:
        m = mirror(rs, d)
    except ViewerError as exc:
        d.notes.append(f"spectrum unavailable: {exc}")
    try:
        pct = detail_percentiles(rs, d)
    except ViewerError as exc:
        pct = []
        d.notes.append(f"feature percentiles unavailable: {exc}")
    state = {
        "run": run,
        "cid": int(cid),
        "row": m.pick.row if m else None,
        "apex": m.pick.row if m else None,
    }
    note_box = notices([("", n) for n in d.notes], kind="info")
    return html.Div(
        [
            _header(d, base),
            note_box or html.Div(),
            _q_table(d),
            html.Div([graph(figures.xic_figure(d))], style=SECTION),
            graph(figures.ms1_figure(d)),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button("previous scan", id="mirror-prev", n_clicks=0),
                            html.Button(
                                "apex scan",
                                id="mirror-apex",
                                n_clicks=0,
                                style={"marginLeft": "6px"},
                            ),
                            html.Button(
                                "next scan",
                                id="mirror-next",
                                n_clicks=0,
                                style={"marginLeft": "6px"},
                            ),
                            html.Span(" same isolation window, ordered by RT", style=MUTED),
                        ]
                    ),
                    graph(figures.mirror_figure(m), id="mirror-graph"),
                    dcc.Store(id="mirror-state", data=state),
                ],
                style={**CARD, **SECTION},
            ),
            _evidence(d),
            graph(figures.percentile_figure(pct)),
            _competition(d, base),
            _partner(d, base),
            html.Div(f"assembled in {sum(d.timings_ms.values()):.0f} ms", style=MUTED),
        ]
    )


def register(app, get_rs) -> None:
    """Callbacks of the detail page."""

    @app.callback(
        Output("mirror-graph", "figure"),
        Output("mirror-state", "data"),
        Input("mirror-prev", "n_clicks"),
        Input("mirror-apex", "n_clicks"),
        Input("mirror-next", "n_clicks"),
        State("mirror-state", "data"),
        prevent_initial_call=True,
    )
    def step(_prev, _apex, _next, state):
        if not state or state.get("row") is None:
            return no_update, no_update
        rs = get_rs()
        d = get_detail(rs, state["run"], state["cid"])
        current = mirror(rs, d, row=state["row"])
        target = state["row"]
        if ctx.triggered_id == "mirror-prev" and current and current.previous_row is not None:
            target = current.previous_row
        elif ctx.triggered_id == "mirror-next" and current and current.next_row is not None:
            target = current.next_row
        elif ctx.triggered_id == "mirror-apex":
            target = state["apex"]
        m = mirror(rs, d, row=target)
        title = "Spectrum mirror" if target == state["apex"] else "Spectrum mirror (neighbour scan)"
        return figures.mirror_figure(m, title=title), {**state, "row": target}
