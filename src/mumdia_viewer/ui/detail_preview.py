"""The compact precursor panel of the identification page (``ui.detail.preview``).

One card of about 320 px: a header line (peptidoform, charge, label, the key q values
with validation marks, the score, the quant state and the link to the precursor page)
over three panels: the fragment and MS1 XICs, the sequence fragmentation diagram over
the annotated spectrum of the apex scan, and the ion table in ladder form. It is static
(no callback); hovering a fragment anywhere in the card highlights it in the others
(``assets/detail.js``, the ``pd-linked`` group). Ids start with ``pv-``. The marks and
the number formats are the precursor page's (:mod:`.detail_cards`, :mod:`.detail_view`).
"""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc
from dash import dcc, html

from mumdia_viewer.data.detail import MirrorData, PrecursorDetail

from . import detail_cards as cards
from . import detail_figures as dfig
from . import detail_ions as ions
from . import detail_view as view
from .icons import icon
from .state import PageContext, href
from .widgets import chip, graph, peptidoform, spark_bar

# The key q columns of the header: PSM, precursor, peptide and protein group.
SINGLE_COLUMNS = ("q_value", "precursor_q", "peptide_q_value", "pg_q_value")
EXPERIMENT_COLUMNS = ("run_psm_q", "precursor_q", "peptide_q_value", "pg_q_value")
STATIC = {"displayModeBar": False}
XIC_HEIGHT = 240


def spectrum_height(ladder: view.IonLadder) -> int:
    """The spectrum's height: the XIC's minus the compact diagram's (so the columns align).

    The compact diagram is about 26 px plus 11 px per row of marks (one row per fragment
    charge on each side, at least one each).
    """
    rows = max(len(ladder.b_charges), 1) + max(len(ladder.y_charges), 1)
    return XIC_HEIGHT - (26 + 11 * rows)


MARKERS = (
    "Green band: the identification's elution bounds; green line: apex_rt; violet dashed "
    "box: quant's integration bounds; grey dotted line: rt_pred_cal; grey dash-dot lines: "
    "the RT window. The plot shows the identification's peak with a margin; the precursor "
    "page shows the whole window. Bottom: the MS1 isotope traces (mono, +1, +2)."
)


def _q_item(ctx: PageContext, d: PrecursorDetail, t: view.QTile) -> Any:
    """One key q value of the header: the mark, the column and the value.

    Narrow cards show the mark and the value only (detail.css); the tooltip names the
    column, its unit and, for a grouped column this row does not win, the group's row.
    """
    thr = ctx.threshold
    parts = [f"{t.column}: {t.unit}; {t.scope}."]
    if t.tested is not None:
        label = str(d.scored.get("label") or "")
        parts.insert(
            0, cards.sentence(cards.valid_text(t.tested, thr, label, cards.tested_column(t)))
        )
    if t.grouped:
        parts.append(cards.winner_text(ctx, t).replace("; the value links to it", ""))
    if t.value is not None:
        value: Any = html.Span(view.fmt_q_at(t.value, thr), className="pv-qv")
    elif t.group_winner is not None:
        value = html.Span(
            [html.Span("group ", className="pv-qg"), view.fmt_q_at(t.group_winner.value, thr)],
            className="pv-qv",
        )
    else:
        value = html.Span("not on this row", className="pv-qv pv-qg")
    text = html.Span([html.Span(t.column, className="pv-qc"), value], className="pv-qtext")
    badge = cards.q_mark(ctx, d, t, 18, bare=True)
    return dmc.Tooltip(
        html.Span([badge, text], className="pv-q"), label=" ".join(p for p in parts if p), w=340
    )


def _quant_chip(d: PrecursorDetail) -> Any:
    q = d.quant
    if q is None:
        return None
    colour = cards.STATE_COLOURS.get(q.state, "gray")
    amount = f": {q.quantity:,.0f} {view.QUANT_UNIT}" if q.quantity is not None else ""
    return chip(
        q.state.replace("_", " "),
        colour,
        size="sm",
        tip=f"peptide_quant{amount}. {cards.sentence(q.reason)}",
    )


def _header(ctx: PageContext, d: PrecursorDetail, run: str) -> Any:
    rs = ctx.rs
    s = d.scored
    label = str(s.get("label") or "")
    ident: list[Any] = [
        html.Span(peptidoform(s.get("peptidoform"), size="1.02rem"), className="pv-pep"),
        dmc.Badge(
            f"{s.get('charge')}+",
            size="sm",
            variant="outline",
            color="gray",
            radius="sm",
            style={"textTransform": "none"},
        ),
        chip(
            label or "no label",
            "orange" if label == "decoy" else "indigo",
            size="sm",
            tip="label column of the scored table",
        ),
    ]
    if cards.is_spike(d):
        ident.append(chip("spike-in", "grape", size="sm", tip="an entrapment spike-in target"))
    if rs.is_experiment:
        ident.append(chip(f"run {d.run.label}", "gray", size="sm", tip="the run of this row"))
    if d.transfer:
        ident.append(
            chip(
                "MBR transfer",
                "lime",
                size="sm",
                tip="match-between-runs transferred this identification into this run; the "
                "q values shown are the rescorer's native values",
            )
        )
    quant = _quant_chip(d)
    if quant is not None:
        ident.append(quant)
    columns = EXPERIMENT_COLUMNS if rs.is_experiment else SINGLE_COLUMNS
    tiles = view.q_tiles(rs, d, columns)
    b = view.score_bounds(rs)
    score = s.get("score")
    r = d.rescore
    score_item = html.Span(
        [
            html.Span("score", className="pv-qc"),
            spark_bar(
                score,
                lo=b.lo,
                hi=b.hi,
                width=42,
                text=view.fmt_score(score, view.page_score_digits(d)),
                colour=cards.SCORE_BAR,
                tip=f"score column; rescorer {r.classifier or 'not recorded'}; higher is "
                f"better. Bar: linear {b.text}{b.clamp_note(score)}.",
            ),
        ],
        className="pv-score",
    )
    target = href(
        ctx.base, "precursor", {"run": run if rs.is_experiment else "", "cid": d.candidate_id}
    )
    open_link = dcc.Link(
        [html.Span("Open precursor page"), icon("right", 14)],
        href=target,
        className="pv-open",
        id="pv-open",
    )
    return html.Div(
        [
            html.Div(ident, className="pv-ident"),
            html.Div([*[_q_item(ctx, d, t) for t in tiles], score_item], className="pv-qs"),
            open_link,
        ],
        className="pv-head",
    )


def _caption(title: str, text: str | None = None, help_: str | None = None) -> Any:
    kids: list[Any] = [html.Span(title, className="pv-cap-title")]
    if text:
        kids.append(html.Span(text, className="pv-cap-text"))
    if help_:
        kids.append(
            dmc.Tooltip(html.Span(icon("info", 12), className="mv-help"), label=help_, w=340)
        )
    return html.Div(kids, className="pv-cap")


def preview_card(
    ctx: PageContext,
    d: PrecursorDetail,
    run: str,
    frags: list[view.Fragment],
    grid: view.ScanGrid | None,
    m: MirrorData | None,
    why: str,
) -> Any:
    scheme = ctx.scheme
    n_obs = sum(1 for f in frags if f.observed)
    ladder = view.ion_ladder(d.scored.get("peptidoform"), frags, m)
    xic = dfig.xic_figure(d, frags, grid, scheme, compact=True, height=XIC_HEIGHT)
    spec = dfig.spectrum_figure(m, frags, scheme, height=spectrum_height(ladder), why_empty=why)
    if m is not None:
        p = m.pick
        at = "apex scan" if d.apex_scan is not None and p.row == d.apex_scan.row else "scan"
        spec_text = (
            f"{at} {p.scan_index} · RT {p.rt:.2f} s · {ladder.n_matched} of "
            f"{ladder.n_library} matched"
        )
        spec_help = (
            f"The MS2 scan of the precursor's isolation window at the apex (scan_index "
            f"{p.scan_index}, window {p.window_lower:.2f} to {p.window_upper:.2f} m/z). "
            f"{dfig.spectrum_scale(m)} {cards.match_text(m)} {m.tolerance.label}. Above: the "
            "sequence with a mark for each library fragment, b ions above and y ions "
            "below; filled when matched in this scan, outlined when not."
        )
    else:
        spec_text, spec_help = view.short_paths(why), why
    panels = [
        html.Div(
            [
                _caption(
                    "XIC",
                    f"{n_obs} of {len(frags)} fragments observed" if frags else None,
                    MARKERS,
                ),
                html.Div(graph("pv-xic", xic, config=STATIC), className="pv-graph"),
            ],
            className="pv-panel pv-xic",
        ),
        html.Div(
            [
                _caption("Spectrum", spec_text, spec_help),
                ions.sequence_diagram(ladder, compact=True),
                html.Div(graph("pv-spec", spec, config=STATIC), className="pv-graph"),
            ],
            className="pv-panel pv-spec",
        ),
        html.Div(
            [
                _caption(
                    "Ion table",
                    "library m/z",
                    ions.ladder_help(ladder, compact=True),
                ),
                html.Div(ions.ladder_table(ladder, compact=True), className="pv-ladder-scroll"),
            ],
            className="pv-panel pv-ladder",
        ),
    ]
    return dmc.Card(
        [_header(ctx, d, run), html.Div(panels, className="pv-body")],
        p="sm",
        className="pv-root pd-linked",
        id="pv-card",
    )


def preview_missing(ctx: PageContext, run: str, cid: Any, text: str) -> Any:
    return dmc.Card(
        dmc.Group(
            [
                dmc.ThemeIcon(icon("alert", 16), color="gray", variant="light", radius="xl"),
                dmc.Text(text, size="sm", c="dimmed"),
            ],
            gap="sm",
        ),
        p="md",
        className="pv-root pv-missing",
        id="pv-card",
    )
