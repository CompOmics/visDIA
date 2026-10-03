"""The cards of the precursor page (components only; the page and its callbacks are in
:mod:`.detail`). Every value comes from the data layer; viewer-derived values carry a
"derived" or "viewer match" badge and say how they were computed.

The validation marks of the page and of its preview are made here (:func:`mark`,
:func:`q_mark`, :func:`unknown_mark`): the mark of ``widgets.validation_icon`` with a
tooltip whose q text never reads as the threshold (:func:`detail_view.fmt_q_at`), and E
for an entrapment spike-in.
"""

from __future__ import annotations

import math
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import dcc, html

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.competition import WINNER_SOURCE
from mumdia_viewer.data.detail import (
    DEFAULT_PERCENTILE_FEATURES,
    EvidenceItem,
    MirrorData,
    PrecursorDetail,
)
from mumdia_viewer.data.features import FeaturePercentile
from mumdia_viewer.data.fragments import MATCH_LABEL
from mumdia_viewer.data.spectra import ScanTable

from . import detail_figures as dfig
from . import detail_ions as ions
from . import detail_view as view
from .icons import icon
from .state import PageContext, href, stop_label
from .widgets import (
    chip,
    data_table,
    empty,
    graph,
    pbar,
    peptidoform,
    section,
    spark_bar,
    validation_icon,
)

STATE_COLOURS = {"quantified": "green", "not_quantifiable": "yellow", "not_selected": "gray"}
UNIT_TILE = {"psm": "indigo", "precursor": "violet", "peptide": "teal", "protein_group": "pink"}
# Score bars: one colour on the page, the identification grid's (grape; detail.css).
SCORE_BAR = "var(--pd-bar-score)"
# Short units for the evidence rows; the full unit is in the row's tooltip.
SHORT_UNIT = {
    "Pearson r": "r",
    "cosine": "cos",
    "normalized angle [0, 1]": "",
    "ppm (raw)": "ppm raw",
    "fragments": "",
    "count": "",
    "counts": "",
    "fraction": "",
    "flag": "",
    "rank": "",
    "ln(1 + intensity)": "ln",
    "fraction of the half-window": "of half-window",
    "intensity x s": view.QUANT_UNIT,
    "higher is better": "",
    "q value": "",
}
# Evidence rows whose note is shown under the value (other notes are in the tooltip).
INLINE_NOTES = frozenset({"selected_peak_rank"})
# Data-layer notes that the hero shows as chips (the label chip, the MBR chip).
HERO_NOTES = ("this row is a decoy", "match-between-runs transfer into this run")


# --------------------------------------------------------------------------- helpers


def num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rt_text(v: Any, digits: int = 2) -> str:
    f = num(v)
    return "" if f is None else f"{f:.{digits}f} s"


def interval(lo: Any, hi: Any, unit: str = "s", digits: int = 2) -> str:
    a, b = num(lo), num(hi)
    if a is None and b is None:
        return "unbounded"
    left = "-inf" if a is None else f"{a:.{digits}f}"
    right = "+inf" if b is None else f"{b:.{digits}f}"
    return f"[{left}, {right}] {unit}".strip()


def tip(child: Any, label: Any, *, block: bool = False, **kwargs: Any) -> Any:
    """A tooltip; ``block`` keeps the full width of a block child (the wrapper shrinks)."""
    extra = {"boxWrapperProps": {"w": "100%"}} if block else {}
    return dmc.Tooltip(child, label=label, **extra, **kwargs)


def info(text: str, size: int = 13) -> Any:
    return tip(html.Span(icon("info", size), className="pd-info"), text, position="top")


def derived_badge(text: str) -> Any:
    return tip(
        dmc.Badge(
            "derived", size="xs", color="grape", variant="light", style={"textTransform": "none"}
        ),
        f"Viewer-derived: {text}",
    )


def match_text(m: MirrorData | None) -> str:
    """What a "matched in the shown scan" means: the viewer's match, with the engine's rule."""
    label = m.label if m is not None and m.label else MATCH_LABEL
    return (
        f"Viewer-derived: {label}. The viewer matches the shown scan's peaks to the library "
        "fragments with the engine's rule and the extraction's tolerance and mass offset; "
        "the engine stores no per-scan matches."
    )


def match_badge(m: MirrorData | None) -> Any:
    return tip(
        dmc.Badge(
            "viewer match",
            size="xs",
            color="grape",
            variant="light",
            style={"textTransform": "none"},
            className="pd-match-badge",
        ),
        match_text(m),
        w=340,
    )


def precursor_href(ctx: PageContext, run_name: str, cid: int) -> str:
    run = run_name if ctx.rs.is_experiment else ""
    return href(ctx.base, "precursor", {"run": run, "cid": int(cid)})


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def tidy(note: str | None) -> str:
    """A data-layer note for display: a repeated leading phrase is said once."""
    if not note:
        return ""
    for lead in ("not computed: not computed", "not quantifiable: not quantifiable"):
        if note.startswith(lead):
            note = note[len(lead.split(":")[0]) + 2 :]
    return note


def card(title: str, *children: Any, **kwargs: Any) -> Any:
    """A panel of the page: ``widgets.section`` without the empty children."""
    return section(title, *[c for c in children if c is not None], **kwargs)


def back_link(ctx: PageContext, d: PrecursorDetail | None = None) -> Any:
    """Back to the identification page, with this row's protein group, peptide and row
    selected (the page's script goes back in the history instead when the previous page
    was the identification page, which keeps its search and filters)."""
    query: dict[str, Any] = {}
    if d is not None:
        s = d.scored
        bpid = s.get("base_peptide_id")
        query = {
            "group": s.get("protein_group") or None,
            "peptide": int(bpid) if bpid is not None else None,
            "run": d.run.name if ctx.rs.is_experiment else None,
            "cid": d.candidate_id,
            "decoys": "1" if d.is_decoy else None,
        }
    return dcc.Link(
        dmc.Group([icon("left", 14), dmc.Text("Identifications", size="sm")], gap=4),
        href=href(ctx.base, "identifications", query),
        className="pd-back",
        id="pd-back",
    )


# --------------------------------------------------------------------------- marks


def ico(name: str, size: int = 14) -> Any:
    """The check or the cross of ``icons.icon`` drawn by a CSS class (detail.css).

    A page has a mark on every row of its tables; the class keeps the icon's data URI
    out of each of them.
    """
    return html.Span(
        className=f"pd-ico pd-ico-{name}", style={"width": f"{size}px", "height": f"{size}px"}
    )


def valid_text(value: Any, threshold: float | None, label: str, column: str) -> str:
    """The tooltip of a validation mark, with q text that never reads as the threshold."""
    shown = view.fmt_q_at(value, threshold) or "not set"
    v = num(value)
    if label == "decoy":
        return f"decoy; {column} {shown}"
    if threshold is not None and v is not None and v <= threshold:
        return f"passes: {column} {shown} ≤ {threshold:g}"
    cut = f" > {threshold:g}" if threshold is not None and v is not None else ""
    return f"does not pass: {column} {shown}{cut}"


def mark(
    value: Any,
    threshold: float | None,
    *,
    label: str,
    column: str,
    size: int = 18,
    spike: bool = False,
    bare: bool = False,
) -> Any:
    """The validation mark of ``widgets.validation_icon`` (check, cross, D for a decoy).

    The tooltip's q text never reads as the threshold (0.010003 is not 0.0100 at
    q ≤ 0.01); ``spike`` marks an entrapment spike-in with E, as the grid does;
    ``bare`` gives the badge alone, for a tile whose own tooltip says what it tests.
    """
    out = validation_icon(value, threshold, label=label, column=column, size=size)
    badge = out.children
    text = valid_text(value, threshold, label, column)
    if spike:
        badge.children = "E"
        badge.color = "grape"
        text = f"entrapment spike-in; {text}"
    elif not isinstance(badge.children, str):
        # The same drawing as the widget's icon, from the page's stylesheet.
        badge.children = ico("check" if badge.color == "green" else "x", size - 6)
    out.label = text
    return badge if bare else out


def unknown_mark(text: str | None, size: int = 18) -> Any:
    """A neutral mark for a q that cannot be tested on this row (it is on another row).

    ``text`` None gives the badge alone.
    """
    badge = dmc.ThemeIcon(
        html.Span(className="pd-neutral"),
        size=size,
        radius="xl",
        color="gray",
        variant="light",
        style={"fontSize": f"{size - 7}px", "fontWeight": 700},
    )
    return badge if text is None else dmc.Tooltip(badge, label=text, w=320)


def is_spike(d: PrecursorDetail) -> bool:
    """Whether this row is an entrapment spike-in (entrapment mode: the competition flags it)."""
    df = d.competition
    if df is None or df.empty or "is_entrapment" not in df:
        return False
    this = df[df["is_this_row"].astype(bool)]
    return bool(this["is_entrapment"].iloc[0]) if not this.empty else False


def tested_column(t: view.QTile) -> str:
    """The column a mark tests, with the winning row when the value is on another row."""
    if t.value is None and t.group_winner is not None:
        return f"{t.column} of its group's winning row (candidate {t.group_winner.candidate_id})"
    return t.column


def q_mark(
    ctx: PageContext,
    d: PrecursorDetail,
    t: view.QTile,
    size: int = 18,
    *,
    bare: bool = False,
) -> Any:
    """The validation mark of one q column of this row at the threshold.

    The row's own value when it has one; for a grouped column that this row does not
    win, the value of its group's winning row; otherwise a neutral mark.
    """
    label = str(d.scored.get("label") or "")
    if t.tested is not None:
        return mark(
            t.tested,
            ctx.threshold,
            label=label,
            column=tested_column(t),
            size=size,
            spike=is_spike(d),
            bare=bare,
        )
    return unknown_mark(
        None
        if bare
        else f"{t.column} is stored on the winning row of its group; this row holds 1.0, which "
        "is not a q value, and the winning row was not found.",
        size,
    )


def group_noun(column: str) -> str:
    return view.GROUP_NOUNS.get(column, "group")


def winner_text(ctx: PageContext, t: view.QTile) -> str:
    """Where the q of a grouped column is, for the tooltips of the tile and the q table."""
    noun = group_noun(t.column)
    if not t.grouped:
        return ""
    if t.winner:
        if t.n_group == 1:
            return f"This row is the only scored row of its {noun}, so it carries the q."
        rows = f", one of {t.n_group} scored rows," if t.n_group else ""
        return (
            f"This row is the winning row of its {noun}{rows} and carries the q. Winner "
            f"flag: {t.winner_source}."
        )
    w = t.group_winner
    if w is None:
        return (
            f"This row is not the winning row of its {noun}: it holds 1.0, which is not a q "
            "value. The winning row was not found."
        )
    where = f" in run {w.run}" if ctx.rs.is_experiment else ""
    return (
        f"This row is not the winning row of its {noun}: it holds 1.0, which is not a q "
        f"value. The group's q, {view.fmt_q_at(w.value, ctx.threshold)}, is on candidate "
        f"{w.candidate_id}{where} ({w.peptidoform} {w.charge}+, {w.label}); the link opens "
        f"it. Found by: {w.how}."
    )


def several_rows(t: view.QTile) -> bool:
    """Whether a grouped column's group has more than this row (unknown counts as yes)."""
    return t.n_group is None or t.n_group > 1


def group_link(ctx: PageContext, w: view.GroupWinner, text: str, cls: str = "pd-link") -> Any:
    return dcc.Link(text, href=precursor_href(ctx, w.run_name, w.candidate_id), className=cls)


# --------------------------------------------------------------------------- header


def _fact(label: str, value: str, text: str, *, derived: bool = False) -> Any:
    head: Any = dmc.Text(label, className="pd-fact-label")
    if derived:
        head = html.Div(
            [head, html.Span("derived", className="pd-th-derived")], className="pd-fact-head"
        )
    return tip(
        dmc.Paper(
            [head, dmc.Text(value, className="pd-fact-value")],
            withBorder=True,
            px="md",
            py=8,
            radius="md",
            className="pd-fact",
        ),
        text,
    )


def hero(ctx: PageContext, d: PrecursorDetail, frags: list[view.Fragment]) -> Any:
    s = d.scored
    rs = ctx.rs
    label = str(s.get("label") or "")
    decoy = label == "decoy"
    where = f"run {d.run.label}" if rs.is_experiment else "single run"
    pills = [
        dmc.Badge(
            f"{s.get('charge')}+",
            size="lg",
            variant="outline",
            color="gray",
            radius="sm",
            style={"textTransform": "none"},
        ),
        chip(
            label or "label not recorded",
            "orange" if decoy else "indigo",
            variant="filled",
            size="lg",
            tip="label column of the scored table"
            + ("; a decoy row: a decoy that passes counts against the FDR" if decoy else ""),
        ),
    ]
    if is_spike(d):
        pills.append(
            chip(
                "entrapment spike-in",
                "grape",
                size="lg",
                tip="an entrapment target (the spike-in test of the entrapment settings): it "
                "counts in the entrapment FDP",
            )
        )
    if d.band:
        pills.append(
            chip(
                f"band {d.band}",
                "cyan",
                size="lg",
                tip="grouped extraction: this candidate's chromatogram, RT window and tolerance "
                f"come from window group {d.band} (groups/{d.band})",
                left=icon("layers", 13),
            )
        )
    if d.transfer:
        note = next((n for n in d.notes if n.startswith(HERO_NOTES[1])), "")
        pills.append(
            chip(
                "MBR transfer",
                "lime",
                variant="filled",
                size="lg",
                tip=sentence(note)
                if note
                else "match-between-runs transferred this identification into this run",
            )
        )
    title = dmc.Group(
        [html.Div(peptidoform(s.get("peptidoform"), size="1.85rem"), className="pd-title"), *pills],
        gap=10,
        align="center",
    )
    chips: list[Any] = []
    for p in [x for x in str(s.get("protein") or "").split(";") if x]:
        chips.append(
            dcc.Link(
                chip(
                    p,
                    "orange" if p.startswith("DECOY_") else "gray",
                    tip="protein column; opens the identifications of this protein",
                ),
                href=href(ctx.base, "identifications", {"search": p}),
                className="pd-chip-link",
            )
        )
    pg = str(s.get("protein_group") or "")
    if pg:
        chips.append(
            dcc.Link(
                chip(
                    f"group {pg}",
                    "pink",
                    tip="protein_group column; opens the protein group in the identifications",
                    left=icon("layers", 12),
                ),
                href=href(ctx.base, "identifications", {"unit": "protein_group", "search": pg}),
                className="pd-chip-link",
            )
        )
    cid = str(d.candidate_id)
    chips.append(
        dmc.Group(
            [
                dmc.Text(f"candidate {cid}", className="pd-mono", size="xs"),
                dcc.Clipboard(content=cid, title="Copy the candidate id", className="pd-copy"),
            ],
            gap=4,
            className="pd-cid",
        )
    )
    chips.append(dmc.Text(f"{where} · {rs.root.name}", size="xs", c="dimmed"))
    left = dmc.Stack(
        [
            html.Div(
                [back_link(ctx, d), dmc.Text(f"Precursor · {where}", className="mv-eyebrow", mt=6)]
            ),
            title,
            dmc.Group(chips, gap=8),
        ],
        gap=8,
    )
    peak = d.selected_peak or {}
    n_match = num(peak.get("n_matched_fragments"))
    n_pred = num(peak.get("n_predicted_fragments"))
    n_obs = sum(1 for f in frags if f.observed)
    w = d.window
    apex = num(s.get("apex_rt"))
    pred = w.rt_pred_cal if w is not None else None
    err = apex - pred if apex is not None and pred is not None else None
    counted = n_match is not None and n_pred is not None
    facts = [
        _fact(
            "precursor m/z",
            f"{d.precursor_mz:.4f}" if d.precursor_mz is not None else "n/a",
            f"precursor_mz of the selected peak ({d.peaks_source})",
        ),
        _fact("apex RT", rt_text(apex) or "n/a", "apex_rt of the scored row, seconds"),
        _fact(
            "RT error",
            f"{err:+.2f} s" if err is not None else "n/a",
            "Viewer-derived: apex_rt - rt_pred_cal (run_windows.parquet), seconds",
            derived=err is not None,
        ),
        _fact(
            "fragments",
            f"{n_match:.0f} / {n_pred:.0f}" if counted else f"{n_obs} / {len(frags)}",
            "n_matched_fragments / n_predicted_fragments of the selected peak (matched in "
            "any scan of the RT window)"
            if counted
            else "fragment rows observed in the XIC / predicted fragment rows",
        ),
    ]
    return dmc.Group(
        [left, dmc.Group(facts, gap="sm", wrap="nowrap", className="pd-facts")],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="pd-hero",
    )


# --------------------------------------------------------------------------- verdict


def summary(tiles: list[view.QTile], t: float, *, decoy: bool = False) -> str:
    """The verdict in one sentence (the tooltip of the strip's title)."""
    own = [x for x in tiles if x.value is not None]
    grp = [x for x in tiles if x.value is None and x.tested is not None]
    other = sum(1 for x in tiles if x.value is None and x.tested is None)
    head = f"At q ≤ {stop_label(t)}: "
    if decoy:
        n = sum(1 for x in own if x.value <= t)
        return (
            head + f"a decoy row; {n} of its {len(own)} q values on this row are at or below "
            "the threshold" + (" (a passing decoy counts against the FDR)." if n else ".")
        )
    counts = (
        (sum(1 for x in own if x.value <= t), "pass on this row"),
        (sum(1 for x in own if x.value > t), "fail on this row"),
        (sum(1 for x in grp if x.tested <= t), "pass on the winning row of their group"),
        (sum(1 for x in grp if x.tested > t), "fail on the winning row of their group"),
        (other, "stored on another row of the group"),
    )
    parts = [f"{n} {what}" for n, what in counts if n]
    return head + ("; ".join(parts) if parts else "no q column") + "."


def short_summary(tiles: list[view.QTile], t: float, *, decoy: bool = False) -> str:
    """The verdict in a few words (the strip's title): passing q columns of all tested."""
    if decoy:
        own = [x.value for x in tiles if x.value is not None]
        return f"decoy · {sum(1 for v in own if v <= t)} of {len(own)} pass"
    tested = [x.tested for x in tiles if x.tested is not None]
    return f"{sum(1 for v in tested if v <= t)} of {len(tested)} pass"


def _vtile(
    head: str, value: Any, text: str, colour: str, *, badge: Any = None, side: Any = None
) -> Any:
    """A tile of the verdict strip: the column name (and ``side``) over the mark and value."""
    top = html.Div([html.Span(head, className="pd-vt-col"), side], className="pd-vt-top")
    line = html.Div([badge, html.Div(value, className="pd-vt-val")], className="pd-vt-line")
    return tip(
        html.Div([top, line], className=f"pd-vt pd-vt-{colour}"),
        text,
        block=True,
        position="bottom",
        w=360,
        openDelay=150,
    )


def _q_tile(ctx: PageContext, d: PrecursorDetail, t: view.QTile) -> Any:
    label = str(d.scored.get("label") or "")
    thr = ctx.threshold
    parts = []
    if t.tested is not None:
        parts.append(sentence(valid_text(t.tested, thr, label, tested_column(t))))
    parts.append(f"{t.column}: {t.unit}; {t.scope}.")
    if t.grouped:
        parts.append(winner_text(ctx, t))
    text = " ".join(p for p in parts if p)
    if t.value is not None:
        value: Any = view.fmt_q_at(t.value, thr)
    elif t.group_winner is not None:
        w = t.group_winner
        value = [
            html.Span("group", className="pd-vt-group"),
            group_link(ctx, w, view.fmt_q_at(w.value, thr), "pd-link pd-vt-link"),
        ]
    else:
        value = html.Span("not on this row", className="pd-vt-none")
    colour = UNIT_TILE.get(t.unit_key, "gray")
    return _vtile(t.column, value, text, colour, badge=q_mark(ctx, d, t, 16, bare=True))


def _score_tile(ctx: PageContext, d: PrecursorDetail, digits: int) -> Any:
    s, r = d.scored, d.rescore
    b = view.score_bounds(ctx.rs)
    bar = spark_bar(s.get("score"), lo=b.lo, hi=b.hi, width=40, text="", colour=SCORE_BAR)
    text = (
        f"score column; rescorer {r.label}; model {r.model_identity or 'not recorded'}; higher "
        f"is better. prelim_score {view.fmt_score(s.get('prelim_score'), 2)}: competition uses "
        f"it to pick the winner of a key; it is not a classifier input. Bar: linear {b.text}"
        f"{b.clamp_note(s.get('score'))}."
    )
    value = html.Div(
        [html.Span(view.fmt_score(s.get("score"), digits)), bar], className="pd-vt-score"
    )
    side = html.Span(r.classifier or "", className="pd-vt-side")
    return _vtile("score", value, text, "grape", side=side)


def _quant_tile(d: PrecursorDetail) -> Any:
    q = d.quant
    if q is None:
        return _vtile(
            "quantity",
            html.Span("n/a", className="pd-vt-none"),
            "The quant tables are not readable.",
            "gray",
        )
    colour = STATE_COLOURS.get(q.state, "gray")
    state_text = q.state.replace("_", " ")
    badge = None
    if q.quantity is not None:
        # Quantified: a green mark, as a passing q column has.
        value: Any = f"{q.quantity:,.0f}"
        badge = dmc.ThemeIcon(
            ico("check", 10),
            size=16,
            radius="xl",
            color=colour,
            variant="light",
            className="pd-quant-mark",
        )
    else:
        # No quantity: the state is the value (never 0).
        value = html.Span(state_text, className="pd-vt-none")
    unit = f" Unit: {view.QUANT_UNIT}." if q.quantity is not None else ""
    used = f" {q.n_fragments_used} fragments used." if q.n_fragments_used is not None else ""
    text = f"peptide_quant: {state_text}. {sentence(q.reason)}{used}{unit}"
    return _vtile("quantity", value, text, "green" if colour == "green" else "gray", badge=badge)


def verdict_body(ctx: PageContext, d: PrecursorDetail, tiles: list[view.QTile]) -> list[Any]:
    """The verdict strip at ``ctx.threshold`` (rebuilt when the threshold changes)."""
    t = ctx.threshold
    digits = view.page_score_digits(d)
    head = tip(
        html.Div(
            [
                html.Div("Verdict", className="mv-section-title"),
                html.Div(f"q ≤ {stop_label(t)}", className="pd-vh-t"),
                html.Div(short_summary(tiles, t, decoy=d.is_decoy), className="pd-vh-n"),
            ],
            id="pd-verdict-sum",
            className="pd-vhead",
        ),
        summary(tiles, t, decoy=d.is_decoy)
        + " A grouped column that this row does not win is tested on its group's winning "
        "row. Hover a tile for its unit and provenance.",
        w=340,
        position="bottom-start",
    )
    cells = [_q_tile(ctx, d, x) for x in tiles] + [_score_tile(ctx, d, digits), _quant_tile(d)]
    return [html.Div([head, *cells], className="pd-vstrip")]


def verdict(ctx: PageContext, d: PrecursorDetail, tiles: list[view.QTile]) -> Any:
    return dmc.Card(verdict_body(ctx, d, tiles), p="sm", id="pd-verdict")


short_paths = view.short_paths


def _note_kind(note: str) -> tuple[str, str]:
    low = note.lower()
    if "decoy" in low and "partner" not in low:
        return "orange", "alert"
    if "transfer" in low:
        return "lime", "info"
    if any(
        w in low
        for w in ("unavailable", "refused", "missing", "unbounded", "fallback", "approximate")
    ):
        return "yellow", "alert"
    return "blue", "info"


def notes(d: PrecursorDetail) -> Any:
    """The data layer's notes as one compact card: one line each, paths in the tooltips.

    The decoy and MBR notes are chips of the hero. More than two notes are collapsed.
    """
    items = [n for n in d.notes if not n.startswith(HERO_NOTES)]
    if not items:
        return None
    rows = []
    for n in items:
        colour, name = _note_kind(n)
        rows.append(
            tip(
                html.Div(
                    [
                        html.Span(icon(name, 14), className=f"pd-note-icon pd-note-{colour}"),
                        html.Span(sentence(short_paths(n)), className="pd-note-text"),
                    ],
                    className="pd-note-row",
                ),
                sentence(n),
                block=True,
                position="bottom-start",
                w=560,
                openDelay=250,
            )
        )
    body: Any = html.Div(rows, className="pd-notes-list")
    if len(rows) > 2:
        body = dmc.Spoiler(
            body,
            maxHeight=42,
            showLabel=f"Show all {len(rows)} notes",
            hideLabel="Show fewer",
            className="pd-notes-spoiler",
        )
    return card(
        "Notes",
        body,
        count=len(rows),
        help="What the data layer could not read, assumed or approximated for this page. "
        "Hover a note for its full text.",
        id="pd-notes",
        p="sm",
    )


# --------------------------------------------------------------------------- XIC


def _key(label: str, value: str, text: str, kind: str, colour: str) -> Any:
    return tip(
        html.Span(
            [
                html.Span(className=f"pd-sw pd-sw-{kind}", style={"--pd-c": colour}),
                html.Span(label, className="pd-key-label"),
                html.Span(value, className="pd-key-value") if value else None,
            ],
            className="pd-key",
        ),
        text,
    )


def marker_key(d: PrecursorDetail, xr: dict[str, Any]) -> Any:
    m = d.markers
    items = []
    if m.get("elution_lo") is not None:
        items.append(
            _key(
                "elution",
                interval(m.get("elution_lo"), m.get("elution_hi"), digits=1),
                "elution_lo and elution_hi of the scored row: the identification's bounds",
                "band",
                "#2f9e44",
            )
        )
    if m.get("apex_rt") is not None:
        items.append(
            _key("apex", rt_text(m.get("apex_rt")), "apex_rt of the scored row", "solid", "#2f9e44")
        )
    if m.get("integration_lo_rt") is not None:
        items.append(
            _key(
                "integration",
                interval(m.get("integration_lo_rt"), m.get("integration_hi_rt"), digits=1),
                "integration_lo_rt and integration_hi_rt of peptide_quant: what quant "
                f"integrated (integration apex {rt_text(m.get('integration_apex_rt'))})",
                "box",
                "#7048e8",
            )
        )
    if m.get("rt_pred_cal") is not None:
        source = f" ({d.window.source})" if d.window else ""
        items.append(
            _key(
                "prediction",
                rt_text(m.get("rt_pred_cal")),
                f"rt_pred_cal: the calibrated RT prediction{source}",
                "dot",
                "#868e96",
            )
        )
    w = d.window
    if w is not None:
        # The grid holds only the scans inside the window; the window extends past the
        # plotted range when a bound is drawn as an edge note only.
        past = bool(w.bounded and not (xr.get("lo_inside") and xr.get("hi_inside")))
        note = (
            " The window extends past the plotted RT range: the grid holds only the scans "
            "inside it."
            if past
            else ""
        )
        text = (
            f"rt_lo and rt_hi ({w.source}): the extraction window.{note} {w.calibration_note}"
            if w.bounded
            else f"Unbounded: no RT calibration for this candidate ({w.source})."
        )
        value = interval(w.rt_lo, w.rt_hi, digits=1) + (" (past the plot)" if past else "")
        items.append(_key("RT window", value, text, "dashdot", "#adb5bd"))
    peaks = [p for p in m.get("peaks", []) if p.get("apex_rt") is not None]
    if len(peaks) > 1:
        items.append(
            _key(
                "other peaks",
                str(len(peaks) - 1),
                "extracted peaks of this candidate that the rescorer did not select "
                "(peak_rank in psms_extracted); arrows at the top of the plot",
                "arrow",
                "#f08c00",
            )
        )
    items.append(
        _key(
            "shown scan",
            "",
            "the MS2 scan in the mirror: click the plot, drag the slider, use the buttons or "
            "the arrow keys",
            "scan",
            "#4263eb",
        )
    )
    return html.Div(items, className="pd-keys")


def scrub_params(grid: view.ScanGrid | None, rng: Any) -> dict[str, Any] | None:
    """The scan slider for one RT range of the XIC: the grid scans inside it, lined up.

    The slider spans the grid scans inside ``rng`` and its margins put its ends under
    those scans on the plot (the figure margins are ``detail_figures.XIC_MARGIN``; the
    graph's wrapper is 6 px wider than the card on each side).
    """
    if grid is None or grid.size < 2 or not rng:
        return None
    rt = grid.rt
    eps = 1e-6
    inside = np.flatnonzero((rt >= rng[0] - eps) & (rt <= rng[1] + eps))
    if inside.size < 2:
        return None
    i0, i1 = int(inside[0]), int(inside[-1])
    span = float(rng[1]) - float(rng[0])
    f_lo = max(0.0, (float(rt[i0]) - float(rng[0])) / span)
    f_hi = max(0.0, (float(rng[1]) - float(rt[i1])) / span)
    left = dfig.XIC_MARGIN["l"] - 6
    right = dfig.XIC_MARGIN["r"] - 6
    style = {
        "marginLeft": f"calc({left}px + (100% - {left + right}px) * {f_lo:.4f})",
        "marginRight": f"calc({right}px + (100% - {left + right}px) * {f_hi:.4f})",
    }
    apex = grid.apex_index
    marks = [{"value": apex, "label": "apex"}] if apex is not None and i0 <= apex <= i1 else []
    return {"min": i0, "max": i1, "style": style, "marks": marks}


def xic_views(d: PrecursorDetail, grid: view.ScanGrid | None) -> dict[str, Any]:
    """The two views of the XIC (peak and whole window) with their slider parameters."""
    xr = dfig.x_range(d, grid)
    peak = dfig.peak_range(d, xr)
    window = xr.get("range")
    out: dict[str, Any] = {
        "window": {"range": window, "slider": scrub_params(grid, window)},
        "view": "window",
    }
    if peak is not None:
        slider = scrub_params(grid, peak)
        if slider is not None or grid is None or grid.size < 2:
            out["peak"] = {"range": peak, "slider": slider}
            out["view"] = "peak"
    return out


def _scrubber(grid: view.ScanGrid | None, views: dict[str, Any]) -> Any:
    n = grid.size if grid is not None else 0
    params = (views.get(views["view"]) or {}).get("slider") or {}
    apex = grid.apex_index if grid is not None else None
    slider = dmc.Slider(
        id="pd-scrub",
        min=params.get("min", 0),
        max=params.get("max", max(n - 1, 1)),
        step=1,
        value=apex if apex is not None else params.get("min", 0),
        updatemode="drag",
        disabled=n < 2,
        marks=params.get("marks", []),
        size="sm",
        color="indigo",
        label={
            "function": "pdScanLabel",
            "options": {
                "rt": [round(float(v), 3) for v in grid.rt] if grid is not None else [],
                "scan": [int(s) for s in grid.scan_index] if grid is not None else [],
            },
        },
        thumbLabel="Scan of the XIC grid",
    )
    if grid is not None and n:
        hint = tip(
            dmc.Text(
                f"Scan slider over the grid scans in view ({n} in the RT window); ← and → "
                "step every scan",
                size="xs",
                c="dimmed",
            ),
            sentence(grid.label),
        )
    else:
        hint = dmc.Text(
            "No scan grid: no readable spectra or no precursor m/z.", size="xs", c="dimmed"
        )
    return html.Div(
        [
            html.Div(
                slider, style=params.get("style", {}), className="pd-scrub", id="pd-scrub-box"
            ),
            html.Div(hint, className="pd-hint"),
        ],
        id="pd-scrub-wrap",
    )


def xic_card(
    ctx: PageContext,
    d: PrecursorDetail,
    frags: list[view.Fragment],
    grid: view.ScanGrid | None,
    key: str | None = None,
    views: dict[str, Any] | None = None,
) -> Any:
    views = views or xic_views(d, grid)
    fig = dfig.xic_figure(d, frags, grid, ctx.scheme, key=key, view=views["view"])
    xr = dfig.x_range(d, grid)
    n_obs = sum(1 for f in frags if f.observed)
    sub = f"{n_obs} of {len(frags)} predicted fragments observed"
    if grid is not None and grid.size:
        sub += f" in {grid.size} grid scans"
    help_ = (
        "Every fragment trace of the candidate over retention time (top) and the MS1 isotope "
        "traces (bottom). The plot opens on the identification's peak; Whole window shows "
        "the RT window. Click a scan to show it in the mirror. Click a legend entry to hide "
        "a fragment in both plots; double click to show it alone. Hover a trace to find the "
        "fragment in the spectrum, on the sequence and in the tables."
    )
    axis = dmc.SegmentedControl(
        id="pd-xic-axis",
        data=[{"value": "linear", "label": "Linear"}, {"value": "log", "label": "Log"}],
        value="linear",
        size="xs",
        radius="xl",
    )
    span = views["window"]["range"]
    peak = views.get("peak")
    view_switch = dmc.SegmentedControl(
        id="pd-xic-view",
        data=[
            {"value": "peak", "label": "Peak"},
            {"value": "window", "label": "Whole window"},
        ],
        value=views["view"],
        size="xs",
        radius="xl",
        style=None if peak is not None else {"display": "none"},
    )
    if peak is not None and span:
        view_switch = tip(
            view_switch,
            f"Peak: {peak['range'][0]:.1f} to {peak['range'][1]:.1f} s, the elution and "
            "integration bounds, the apex, the prediction and the other peaks with a margin "
            f"of 2.5 peak widths. Whole window: {span[0]:.1f} to {span[1]:.1f} s.",
            w=300,
        )
    return card(
        "Fragment and MS1 XICs",
        marker_key(d, xr),
        html.Div(
            [
                graph("pd-xic", fig),
                # The shown scan, drawn over the plot by assets/detail.js.
                html.Div(
                    [html.Div(className="pd-scanband"), html.Div(className="pd-scanline")],
                    id="pd-scanmark",
                    className="pd-scanmark",
                ),
            ],
            id="pd-xic-wrap",
            className="pd-graph-wrap",
        ),
        _scrubber(grid, views),
        dcc.Store(id="pd-xview", data=views),
        right=dmc.Group([view_switch, axis], gap=6, wrap="nowrap"),
        subtitle=sub,
        help=help_,
        count=len(frags),
        id="pd-xic-card",
    )


# --------------------------------------------------------------------------- retention time


def gauge(d: PrecursorDetail, xr: dict[str, Any]) -> Any:
    """The apex in the RT window, with the prediction and the elution and integration bounds."""
    m = d.markers
    w = d.window
    if w is not None and w.bounded:
        lo, hi, what = float(w.rt_lo), float(w.rt_hi), "RT window"
    elif xr.get("axis") is not None:
        lo, hi = xr["axis"]
        what = "XIC axis (the RT window is unbounded)"
    else:
        return None
    span = hi - lo if hi > lo else 1.0

    def pos(x: Any) -> float | None:
        v = num(x)
        return None if v is None else max(0.0, min(100.0, 100.0 * (v - lo) / span))

    def band(a: float | None, b: float | None, cls: str) -> Any:
        if a is None or b is None:
            return None
        return html.Div(
            className=cls, style={"left": f"{a:.2f}%", "width": f"{max(b - a, 0.8):.2f}%"}
        )

    parts: list[Any] = [
        html.Div(className="pd-gauge-track"),
        band(pos(m.get("elution_lo")), pos(m.get("elution_hi")), "pd-gauge-elution"),
        band(
            pos(m.get("integration_lo_rt")), pos(m.get("integration_hi_rt")), "pd-gauge-integration"
        ),
    ]
    p = pos(m.get("rt_pred_cal"))
    x = pos(m.get("apex_rt"))
    if p is not None:
        side = "left" if x is not None and x > p else "right"
        parts += [
            html.Div(className="pd-gauge-pred", style={"left": f"{p:.2f}%"}),
            html.Span(
                "pred.",
                className=f"pd-gauge-tag pd-gauge-tag-{side} pd-tag-pred",
                style={"left": f"{p:.2f}%"},
            ),
        ]
    if x is not None:
        side = "right" if p is not None and x > p else "left"
        parts += [
            html.Div(className="pd-gauge-apex", style={"left": f"{x:.2f}%"}),
            html.Span(
                "apex",
                className=f"pd-gauge-tag pd-gauge-tag-{side} pd-tag-apex",
                style={"left": f"{x:.2f}%"},
            ),
        ]
    labels = html.Div(
        [
            html.Span(f"{lo:.1f} s", className="pd-gauge-end"),
            html.Span(what, className="pd-gauge-what"),
            html.Span(f"{hi:.1f} s", className="pd-gauge-end"),
        ],
        className="pd-gauge-labels",
    )
    return html.Div(
        [html.Div([q for q in parts if q is not None], className="pd-gauge"), labels],
        className="pd-gauge-box",
    )


def rt_card(d: PrecursorDetail, xr: dict[str, Any], pct: dict[str, FeaturePercentile]) -> Any:
    items = view.evidence_groups(d.evidence).get("rt", [])
    w = d.window
    return card(
        "Retention time",
        gauge(d, xr),
        *[evidence_row(e, pct.get(e.key)) for e in items],
        dmc.Text(sentence(w.calibration_note), size="xs", c="dimmed", mt="xs")
        if w is not None and w.rt_pred_cal is not None
        else None,
        subtitle="The apex against the calibrated prediction and the extraction window",
        id="pd-rt-card",
    )


# --------------------------------------------------------------------------- mirror


def scan_nav(enabled: bool) -> Any:
    def arrow(id_: str, name: str, label: str, key: str) -> Any:
        return tip(
            dmc.ActionIcon(
                icon(name, 16),
                id=id_,
                variant="default",
                size="md",
                radius="md",
                n_clicks=0,
                disabled=not enabled,
                **{"aria-label": label},
            ),
            f"{label} ({key} arrow key)",
        )

    return dmc.Group(
        [
            arrow("scan-prev", "left", "Previous scan", "left"),
            tip(
                dmc.Button(
                    "Apex",
                    id="scan-apex",
                    variant="light",
                    size="compact-sm",
                    radius="md",
                    n_clicks=0,
                    disabled=not enabled,
                    leftSection=icon("target", 14),
                ),
                "Back to the apex scan",
            ),
            arrow("scan-next", "right", "Next scan", "right"),
        ],
        gap=6,
        wrap="nowrap",
    )


def scan_info(rs: ResultSet, d: PrecursorDetail, m: MirrorData | None) -> Any:
    """The badges that describe the shown scan."""
    if m is None:
        return dmc.Text("No scan shown.", size="xs", c="dimmed")
    p = m.pick
    apex = d.apex_scan.row if d.apex_scan is not None else None
    at_apex = p.row == apex
    items = [
        chip(
            f"scan_index {p.scan_index}",
            "gray",
            size="sm",
            tip=f"MS2 row {p.row} of spectra_ms2.parquet; scan_index is the run-global spectrum "
            "counter, not the row",
        ),
        chip(f"RT {p.rt:.2f} s", "gray", size="sm", tip="rt_seconds of the scan"),
        chip(
            "apex scan" if at_apex else f"{p.delta_rt:+.2f} s from the apex",
            "green" if at_apex else "gray",
            size="sm",
            tip="the scan whose rt_seconds equals apex_rt (the scan the engine used)"
            if at_apex
            else "rt_seconds - apex_rt",
        ),
        chip(
            f"window {p.window_lower:.2f} to {p.window_upper:.2f}",
            "gray",
            size="sm",
            tip=f"isolation window {p.window_id} of the scan (m/z)",
        ),
        chip(f"{m.spectrum.n_peaks:,} peaks", "gray", size="sm", tip="peaks stored for the scan"),
    ]
    if view.in_window(p.rt, d) is False:
        items.append(
            chip(
                "outside the RT window",
                "orange",
                size="sm",
                tip="the scan's RT lies outside [rt_lo, rt_hi]",
            )
        )
    return dmc.Group(items, gap=6)


def tolerance_row(d: PrecursorDetail, m: MirrorData | None) -> Any:
    tol = d.tolerance
    if tol is None:
        return dmc.Text(
            "The extraction tolerance is unknown, so no peak is matched.", size="xs", c="dimmed"
        )
    items = [
        chip(
            f"tolerance {tol.tol_ppm:.3g} ppm",
            "orange" if tol.assumed else "indigo",
            size="sm",
            tip=f"{sentence(tol.label)} Source: {tol.source}.",
        ),
        chip(
            "m/z-dependent offset" if tol.uses_grid else f"offset {tol.offset_ppm:+.3g} ppm",
            "gray",
            size="sm",
            tip=f"mass calibration grid ({tol.grid_source})"
            if tol.uses_grid
            else "the run's fragment mass offset, applied to the observed m/z before matching",
        ),
        chip(f"matcher {tol.matcher}", "gray", size="sm", tip="config_json extract.matcher"),
    ]
    if tol.assumed:
        items.append(
            chip(
                "assumed",
                "orange",
                size="sm",
                tip="no extraction record was found: these are the configured values",
            )
        )
    if m is not None:
        items.append(
            chip(
                f"{len(m.matches)} of {len(m.fragments)} matched",
                "teal",
                size="sm",
                tip=f"{match_text(m)} {sentence(m.tolerance.label)}",
            )
        )
    return dmc.Group(items, gap=6)


def mirror_card(
    ctx: PageContext,
    d: PrecursorDetail,
    frags: list[view.Fragment],
    m: MirrorData | None,
    why: str,
    ladder: view.IonLadder,
    x_range: list[float] | None = None,
) -> Any:
    outside = view.in_window(m.pick.rt, d) is False if m is not None else None
    fig = dfig.mirror_figure(m, frags, ctx.scheme, outside=outside, why_empty=why, x_range=x_range)
    scale = dmc.SegmentedControl(
        id="pd-scale",
        data=[{"value": "matched", "label": "Matched"}, {"value": "base", "label": "Base peak"}],
        value="matched",
        size="xs",
        radius="xl",
    )
    return card(
        "Spectrum mirror",
        dmc.Group(
            [scan_nav(m is not None), html.Div(scan_info(ctx.rs, d, m), id="pd-scan-info")],
            gap="sm",
            align="center",
            wrap="nowrap",
            mb=4,
        ),
        html.Div(ions.sequence_diagram(ladder, match=match_badge(m)), id="pd-seq"),
        html.Div(graph("pd-mirror", fig), id="pd-mirror-wrap", className="pd-graph-wrap"),
        html.Div(tolerance_row(d, m), className="pd-tol", id="pd-tol"),
        right=tip(
            scale,
            "Scale of the observed peaks: % of the highest matched peak, or of the base peak "
            "of the scan",
        ),
        subtitle="The isolation-window scan against the library's predicted fragments",
        help="The observed MS2 spectrum of the shown scan (up) against the candidate's library "
        "fragments (down). Labels give the fragment and its raw ppm error. Above the "
        "spectrum, the sequence marks each library fragment: filled when it is matched in "
        "this scan, outlined when not. Step the scans with the buttons, the arrow keys, the "
        "slider under the XIC or a click on the XIC. A zoom stays while you step.",
        id="pd-mirror-card",
    )


def _frag_name(f: view.Fragment) -> Any:
    if f.ion is None or f.ordinal is None:
        return html.Span(f.name)
    base = f"{f.ion}{f.ordinal}"
    return html.Span(base if (f.charge or 1) == 1 else [base, html.Sup(f"{f.charge}+")])


def _ppm(v: float | None) -> Any:
    return "" if v is None else html.Span(f"{v:+.2f}", className="pd-ppm")


def _yes(v: bool, no: str = "no") -> Any:
    if v:
        return html.Span(ico("check", 13), className="pd-yes")
    return html.Span(no, className="pd-no")


def fragment_table(
    m: MirrorData | None, frags: list[view.Fragment], hidden: set[int] | None = None
) -> Any:
    """The fragment table of the shown scan: one row per predicted fragment."""
    rows = view.fragment_rows(m, frags)
    if not rows:
        return dmc.Text("No predicted fragments.", size="sm", c="dimmed")
    hidden = hidden or set()

    def th(text: Any, label: str | None = None, numeric: bool = False) -> Any:
        # The page's one tooltip (data-tip): the table is rebuilt on every scan step.
        cell = html.Span(text, className="pd-th-tip", **{"data-tip": label}) if label else text
        return dmc.TableTh(cell, className="mv-num" if numeric else None)

    head = dmc.TableThead(
        dmc.TableTr(
            [
                th("fragment"),
                th("theor. m/z", "the library fragment m/z (float32, as extract holds it)", True),
                th("obs. m/z", "m/z of the matched peak in the shown scan (viewer match)", True),
                th("ppm raw", view.PPM_LABELS["raw"], True),
                th(
                    ["ppm corr.", html.Br(), html.Span("derived", className="pd-th-derived")],
                    "Viewer-derived: " + view.PPM_LABELS["corrected"],
                    True,
                ),
                th("pred.", "predicted_intensity of the library fragment", True),
                th(
                    "matched", "a peak of the shown scan within the tolerance (" + MATCH_LABEL + ")"
                ),
                th(
                    "in XIC",
                    "observed in the XIC: a matched peak in at least one scan of the RT window",
                ),
            ]
        )
    )
    body = []
    for r in rows:
        f = r.fragment
        cls = (
            "pd-frow"
            + (" pd-off" if f.index in hidden else "")
            + ("" if r.matched else " pd-unmatched")
        )
        name = [html.Span(className="pd-swatch", style={"background": f.colour}), _frag_name(f)]
        body.append(
            dmc.TableTr(
                [
                    dmc.TableTd(html.Span(name, className="pd-fname")),
                    dmc.TableTd(f"{f.mz:.4f}", className="mv-num"),
                    dmc.TableTd(
                        f"{r.obs_mz:.4f}" if r.obs_mz is not None else "", className="mv-num"
                    ),
                    dmc.TableTd(_ppm(r.ppm_raw), className="mv-num"),
                    dmc.TableTd(_ppm(r.ppm_corrected), className="mv-num"),
                    dmc.TableTd(f"{f.predicted:.3f}", className="mv-num"),
                    dmc.TableTd(_yes(r.matched)),
                    dmc.TableTd(_yes(f.observed, "never")),
                ],
                className=cls,
                **{"data-frag": str(f.index)},
            )
        )
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        verticalSpacing=5,
        horizontalSpacing=6,
        className="mv-table pd-ftable",
    )
    caption: Any = "No scan shown."
    if m is not None:
        caption = [
            html.Span(
                f"Shown scan: scan_index {m.pick.scan_index}, RT {m.pick.rt:.2f} s; "
                f"{len(m.matches)} of {len(rows)} fragments matched",
                className="pd-lad-cap-text",
            ),
            match_badge(m),
        ]
    return html.Div(
        [
            html.Div(caption, className="pd-lad-cap pd-ftable-caption"),
            dmc.TableScrollContainer(table, minWidth=500),
        ]
    )


def ladder_panel(
    ladder: view.IonLadder, hidden: set[int] | None = None, m: MirrorData | None = None
) -> Any:
    """The ion table in ladder form with its one-line caption (the fragment card's second view)."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(ions.ladder_caption(ladder), className="pd-lad-cap-text"),
                    match_badge(m) if ladder.scan else None,
                    dmc.Tooltip(
                        html.Span(icon("info", 12), className="mv-help"),
                        label=ions.ladder_help(ladder),
                        w=360,
                    ),
                ],
                className="pd-lad-cap",
            ),
            html.Div(ions.ladder_table(ladder, hidden=hidden or set()), className="pd-ladder-wrap"),
        ]
    )


FRAG_VIEWS = [
    {"value": "list", "label": "Fragment list"},
    {"value": "ladder", "label": "Ion table"},
]


def fragment_card(m: MirrorData | None, frags: list[view.Fragment], ladder: view.IonLadder) -> Any:
    switch = dmc.SegmentedControl(
        id="pd-frag-view",
        data=FRAG_VIEWS,
        value="list",
        size="xs",
        radius="xl",
        persistence=True,
        persistence_type="local",
    )
    return card(
        "Fragments in the shown scan",
        html.Div(fragment_table(m, frags), id="pd-frag-table"),
        html.Div(ladder_panel(ladder, m=m), id="pd-ladder", style={"display": "none"}),
        right=switch,
        subtitle="Hover a fragment to find it in the plots; click it to hide it",
        help="The fragment list gives every library fragment of the candidate with its match "
        "in the shown scan. The ion table puts the same fragments on the sequence (PeptideShaker's "
        "ladder form): one row per residue, b ions on the left and y ions on the right. "
        "Hover a fragment to find it in the XIC, the spectrum and on the sequence; click it "
        "to hide it in both plots, and again to show it. The matches are the viewer's.",
        count=len(frags),
        id="pd-frag-card",
    )


def nav_data(rs: ResultSet, d: PrecursorDetail, m: MirrorData | None) -> dict[str, Any]:
    """Where the scan buttons lead from the shown scan (same isolation window)."""
    apex = d.apex_scan
    out: dict[str, Any] = {
        "row": None,
        "prev_row": None,
        "prev_rt": None,
        "next_row": None,
        "next_rt": None,
        "apex_row": apex.row if apex is not None else None,
        "apex_rt": float(np.float32(apex.rt)) if apex is not None else None,
    }
    if m is None:
        return out
    out["row"] = m.pick.row
    try:
        scans = ScanTable.for_run(rs, d.run)
    except ViewerError:
        return out
    for key, row in (("prev", m.previous_row), ("next", m.next_row)):
        if row is not None:
            out[f"{key}_row"] = int(row)
            out[f"{key}_rt"] = float(np.float32(scans.rt[int(row)]))
    return out


# --------------------------------------------------------------------------- evidence


def _short_unit(unit: str) -> str:
    return SHORT_UNIT.get(unit, unit)


def evidence_row(e: EvidenceItem, pct: FeaturePercentile | None = None) -> Any:
    note = tidy(e.note)
    text = f"Source: {e.source}." + (f" {sentence(note)}" if note else "")
    if e.unit and e.unit not in ("higher is better", "q value"):
        text += f" Unit: {e.unit}."
    value = view.evidence_value(e)
    label = tip(html.Span(e.label, className="pd-ev-label"), text, position="top-start")
    left = html.Div([label, derived_badge(e.source) if e.derived else None], className="pd-ev-left")
    extra = None
    if value:
        unit = _short_unit(e.unit or "")
        right = html.Div(
            [
                html.Span(value, className="pd-ev-value"),
                html.Span(unit, className="pd-ev-unit") if unit else None,
            ],
            className="pd-ev-right",
        )
        if e.key in INLINE_NOTES and note:
            extra = html.Div(sentence(note), className="pd-ev-note")
    else:
        right = html.Div(html.Span("n/a", className="pd-ev-na"), className="pd-ev-right")
        extra = html.Div(sentence(note or "not available"), className="pd-ev-note")
    parts = [html.Div([left, right], className="pd-ev-head")]
    if extra is not None:
        parts.append(extra)
    if pct is not None and (pct.pct_target is not None or pct.pct_decoy is not None):
        parts.append(pct_line(pct))
    return html.Div(parts, className="mv-evidence-row pd-ev")


def pct_line(p: FeaturePercentile) -> Any:
    bits = []
    if p.pct_target is not None:
        bits.append(f"targets {p.pct_target:.0f}%")
    if p.pct_decoy is not None:
        bits.append(f"decoys {p.pct_decoy:.0f}%")
    text = ""
    if p.pct_target is not None:
        text += f"{p.pct_target:.1f}% of {p.n_target:,} target rows"
    if p.pct_decoy is not None:
        text += (" and " if text else "") + f"{p.pct_decoy:.1f}% of {p.n_decoy:,} decoy rows"
    text += f" have a value ≤ this one. Population: {p.population}."
    return html.Div(
        [
            html.Div(pbar(p.pct_target, p.pct_decoy), className="pd-pbar"),
            tip(html.Span(" · ".join(bits), className="pd-pct-text"), text, position="bottom-end"),
        ],
        className="pd-pct",
    )


def _ev_card(
    title: str,
    items: list[EvidenceItem],
    pct: dict[str, FeaturePercentile],
    *extra: Any,
    subtitle: str | None = None,
    id_: str | None = None,
) -> Any:
    rows = [evidence_row(e, pct.get(e.key)) for e in items]
    return card(title, *rows, *extra, subtitle=subtitle, id=id_)


def isotopes(d: PrecursorDetail) -> Any:
    peak = d.selected_peak or {}
    vals = [(k, num(peak.get(k))) for k in ("ms1_mono", "ms1_iso1", "ms1_iso2")]
    if all(v is None for _, v in vals):
        return None
    top = max((v for _, v in vals if v is not None), default=0.0) or 1.0
    names = {"ms1_mono": "mono", "ms1_iso1": "+1", "ms1_iso2": "+2"}
    rows = []
    for k, v in vals:
        width = 0.0 if v is None else 100.0 * v / top
        rows.append(
            html.Div(
                [
                    html.Span(names[k], className="pd-iso-name"),
                    html.Div(
                        html.Div(className="pd-iso-fill", style={"width": f"{width:.1f}%"}),
                        className="pd-iso-track",
                    ),
                    html.Span(f"{v:,.0f}" if v is not None else "n/a", className="pd-iso-value"),
                ],
                className="pd-iso-row",
            )
        )
    head = dmc.Group(
        [
            dmc.Text("Isotope pattern at the apex", size="xs", fw=600),
            info(
                "ms1_mono, ms1_iso1 and ms1_iso2 of the selected peak (psms_extracted): summed "
                "MS1 peaks near each isotope m/z in the MS1 scan nearest the apex. Bars are "
                "relative to the largest of the three.",
                12,
            ),
        ],
        gap=4,
    )
    return html.Div([head, *rows], className="pd-iso")


def peaks_table(d: PrecursorDetail) -> Any:
    if d.peaks.empty or len(d.peaks) < 2 or "peak_rank" not in d.peaks:
        return None
    rows = []
    for _, r in d.peaks.sort_values("peak_rank").iterrows():
        rank = int(r["peak_rank"])
        selected = rank == d.selected_peak_rank
        rows.append(
            [
                html.Span(str(rank), className="pd-num"),
                rt_text(r.get("apex_rt")),
                f"{num(r.get('apex_intensity')) or 0:,.0f}" if "apex_intensity" in r else "",
                f"{num(r.get('n_matched_fragments')) or 0:.0f}"
                if "n_matched_fragments" in r
                else "",
                chip("selected", "green", size="xs")
                if selected
                else chip("not selected", "orange", size="xs"),
            ]
        )
    return html.Div(
        [
            dmc.Text("Extracted peaks of this candidate", size="xs", fw=600, mt="sm", mb=4),
            data_table(rows, ["rank", "apex", "apex int.", "matched", ""], numeric=[0, 2, 3]),
        ]
    )


def evidence_legend() -> Any:
    """How to read the percentile bars of the evidence cards."""
    return dmc.Group(
        [
            html.Div("Evidence", className="mv-section-title"),
            dmc.Group(
                [
                    html.Span(
                        [html.Span(className="pd-lg-fill"), "rank among the run's targets"],
                        className="pd-lg",
                    ),
                    html.Span(
                        [html.Span(className="pd-lg-mark"), "rank among the run's decoys"],
                        className="pd-lg",
                    ),
                    dmc.Text(
                        "Ranks over the scored rows of the run, not FDR-filtered; hover a value "
                        "for its source and a rank for its population.",
                        size="xs",
                        c="dimmed",
                    ),
                ],
                gap="md",
            ),
        ],
        justify="space-between",
        className="pd-ev-legend",
    )


def evidence_grid(d: PrecursorDetail, pct: dict[str, FeaturePercentile]) -> Any:
    g = view.evidence_groups(d.evidence)
    cards = [
        _ev_card(
            "Fragments",
            g.get("fragments", []),
            pct,
            id_="pd-ev-fragments",
            subtitle="Matches in the RT window and at the apex scan",
        ),
        _ev_card(
            "Co-elution",
            g.get("coelution", []),
            pct,
            id_="pd-ev-coelution",
            subtitle="Fragment traces inside the elution window",
        ),
        _ev_card(
            "Mass error",
            g.get("mass", []),
            pct,
            id_="pd-ev-mass",
            subtitle="Fragment m/z against the library; raw ppm centre on the run's offset",
        ),
        _ev_card(
            "MS1 support",
            g.get("ms1", []),
            pct,
            isotopes(d),
            id_="pd-ev-ms1",
            subtitle="The precursor isotopes in the MS1 scan nearest the apex",
        ),
        _ev_card(
            "Interference and peaks",
            g.get("interference", []) + g.get("peaks", []),
            pct,
            peaks_table(d),
            id_="pd-ev-interference",
            subtitle="Shared signal, and the extracted peak the rescorer chose",
        ),
        quant_card(d),
    ]
    return dmc.Stack(
        [
            evidence_legend(),
            dmc.SimpleGrid(cards, cols={"base": 1, "md": 2, "xl": 3}, spacing="lg"),
        ],
        gap="sm",
    )


# --------------------------------------------------------------------------- q table


def q_value_cell(ctx: PageContext, t: view.QTile) -> Any:
    """The value of one q column in the q table: this row's, or its group winner's."""
    thr = ctx.threshold
    if t.value is not None:
        return dmc.Text(view.fmt_q_at(t.value, thr), className="pd-num", size="sm", fw=600)
    w = t.group_winner
    if w is None:
        return tip(dmc.Text("not on this row", size="xs", c="dimmed"), t.text)
    where = f", run {w.run}" if ctx.rs.is_experiment else ""
    return html.Div(
        [
            html.Div(
                [
                    html.Span("group ", className="pd-qg"),
                    html.Span(view.fmt_q_at(w.value, thr), className="pd-num pd-qgv"),
                ],
                className="pd-qgroup",
            ),
            html.Div(
                group_link(ctx, w, f"candidate {w.candidate_id}{where}"), className="pd-small"
            ),
        ]
    )


def q_table(ctx: PageContext, d: PrecursorDetail) -> Any:
    """Every q column of the scored row with its validation mark at ``ctx.threshold``.

    Built from the same tiles as the verdict (:func:`detail_view.q_tiles`), so a column
    has the same mark and value in both: a grouped column that this row does not win
    shows its group's winning row and is tested on it.
    """
    tiles = {t.column: t for t in view.q_tiles(ctx.rs, d, None)}
    label = str(d.scored.get("label") or "")
    spike = is_spike(d)
    rows = []
    for e in [x for x in d.evidence if x.group == "q"]:
        t = tiles.get(e.key)
        prov: Any = None
        if t is not None:
            mark_ = q_mark(ctx, d, t, 16)
            value = q_value_cell(ctx, t)
            if t.grouped and t.winner and several_rows(t):
                prov = chip("group winner", "green", size="xs", tip=winner_text(ctx, t))
            elif t.grouped and not t.winner:
                prov = chip("not the winner", "gray", size="xs", tip=winner_text(ctx, t))
            unit = f"{t.unit}; {t.scope}"
        else:
            mark_ = mark(e.value, ctx.threshold, label=label, column=e.label, size=16, spike=spike)
            value = dmc.Text(
                view.fmt_q_at(e.value, ctx.threshold), className="pd-num", size="sm", fw=600
            )
            prov = chip("after MBR", "lime", size="xs", tip=e.note or "")
            unit = f"{e.note or ''}; {d.run.label}"
        rows.append(
            [
                mark_,
                tip(dmc.Text(e.label, className="pd-mono", size="xs"), f"Source: {e.source}"),
                value,
                html.Div(
                    [dmc.Text(unit, size="xs", c="dimmed", className="pd-unit"), prov],
                    className="pd-q-unit",
                ),
            ]
        )
    return data_table(rows, ["", "column", "value", "unit and scope"])


def q_card(ctx: PageContext, d: PrecursorDetail) -> Any:
    return card(
        "q values",
        html.Div(q_table(ctx, d), id="pd-q-table"),
        subtitle="Every q column of the scored row, with its unit",
        help="The mark tests each column against the threshold of the header. A grouped "
        "column (precursor_q, peptide_q_value, pg_q_value) is set on its group's winning row "
        "only; the other rows hold 1.0, which is not a q value, so the group's value and its "
        "row are shown and tested instead. In an experiment the grouped columns are "
        "experiment-wide and run_psm_q is the q within this run.",
        count=len(d.q_values),
        id="pd-q-card",
    )


# --------------------------------------------------------------------------- quant, MBR


def _bars(
    rows: list[tuple[str, float | None, float | None, str]],
    marks: list[tuple[float | None, str]],
) -> Any:
    """Intervals on one RT scale, one bar per row, with marks (the apex) on every bar."""
    values = [v for _, a, b, _ in rows for v in (a, b) if v is not None]
    values += [v for v, _ in marks if v is not None]
    if len(values) < 2:
        return None
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.06, 0.5)
    lo, hi = lo - pad, hi + pad

    def pos(v: float) -> float:
        return 100.0 * (v - lo) / (hi - lo)

    out = []
    for label, a, b, colour in rows:
        if a is None or b is None:
            continue
        fill = html.Div(
            className="pd-wbar-fill",
            style={
                "left": f"{pos(a):.2f}%",
                "width": f"{max(pos(b) - pos(a), 0.8):.2f}%",
                "background": colour,
            },
        )
        ticks = [
            html.Div(className="pd-wbar-mark", style={"left": f"{pos(v):.2f}%", "background": c})
            for v, c in marks
            if v is not None
        ]
        out.append(
            html.Div(
                [
                    html.Span(label, className="pd-wbar-label"),
                    html.Div([fill, *ticks], className="pd-wbar-track"),
                    html.Span(interval(a, b), className="pd-wbar-value"),
                ],
                className="pd-wbar",
            )
        )
    return html.Div(out, className="pd-wbars")


def _defs(rows: list[tuple[str, str, str]]) -> Any:
    """A compact list of (label, value, tooltip) pairs."""
    items = []
    for k, v, why in rows:
        key = html.Span(k, className="pd-def-key")
        items.append(
            html.Div(
                [tip(key, why) if why else key, html.Span(v, className="pd-def-value")],
                className="pd-def",
            )
        )
    return html.Div(items, className="pd-defs")


def quant_card(d: PrecursorDetail) -> Any:
    q = d.quant
    if q is None:
        return card(
            "Quantification",
            dmc.Text("No quant state is available.", size="sm", c="dimmed"),
            id="pd-quant-card",
        )
    colour = STATE_COLOURS.get(q.state, "gray")
    if q.quantity is not None:
        big = html.Div(f"{q.quantity:,.0f}", className="pd-big")
        unit = dmc.Text(f"{view.QUANT_UNIT}, peptide_quant quantity", size="xs", c="dimmed")
    else:
        none = "not quantifiable" if q.state == "not_quantifiable" else "no quantity"
        big = html.Div(none, className="pd-big pd-big-none")
        unit = dmc.Text(
            "quantity is null: not quantifiable, never 0"
            if q.state == "not_quantifiable"
            else "no peptide_quant row for this candidate",
            size="xs",
            c="dimmed",
        )
    m = d.markers
    bars = _bars(
        [
            ("identification", num(m.get("elution_lo")), num(m.get("elution_hi")), "#2f9e44"),
            ("integration", q.integration_lo_rt, q.integration_hi_rt, "#7048e8"),
        ],
        [(num(m.get("apex_rt")), "var(--mantine-color-text)")],
    )
    facts: list[tuple[str, str, str]] = []
    if q.status:
        facts.append(("quant_status", q.status, "peptide_quant quant_status"))
    if q.n_fragments_used is not None:
        facts.append(("fragments used", str(q.n_fragments_used), "peptide_quant n_fragments_used"))
    if q.integration_apex_rt is not None:
        facts.append(
            (
                "integration apex",
                rt_text(q.integration_apex_rt),
                "peptide_quant integration_apex_rt",
            )
        )
    if q.gate_q is not None:
        facts.append(
            (
                "gate q",
                dfig.q_text_any(q.gate_q),
                "the gate column of the scored table that quant read",
            )
        )
    if q.native_q is not None:
        facts.append(
            (
                "native q (before MBR)",
                dfig.q_text_any(q.native_q),
                "the same column in scored_combined.parquet",
            )
        )
    if q.window_also_covers:
        facts.append(
            (
                "window also covers peak",
                ", ".join(str(r) for r in q.window_also_covers),
                "other peak ranks whose apex lies inside the integration window",
            )
        )
    head = [
        big,
        dmc.Badge(
            q.state.replace("_", " "),
            color=colour,
            variant="light",
            style={"textTransform": "none"},
        ),
    ]
    if q.from_transfer:
        head.append(chip("MBR transfer", "lime", size="sm"))
    return card(
        "Quantification",
        dmc.Group(head, gap="sm", align="center"),
        unit,
        bars,
        _defs(facts) if facts else None,
        dmc.Text(sentence(q.reason), size="xs", c="dimmed", mt="xs"),
        subtitle="The integration window against the identification's bounds",
        id="pd-quant-card",
    )


def transfer_card(d: PrecursorDetail) -> Any:
    t = d.transfer
    if not t:
        return None
    rows = []
    for key, value, why in view.transfer_rows(t):
        if key.endswith("_rt") or key == "rt_delta":
            shown = rt_text(value)
        else:
            shown = dfig.q_text_any(value) if value is not None else ""
        rows.append((key, shown or "not recorded", why))
    return card(
        "Match-between-runs transfer",
        dmc.Alert(
            "This identification was transferred into this run. The q values of the verdict "
            "are the rescorer's native values (scored_combined.parquet); the lowered values that "
            "quant and the report used are listed here.",
            color="lime",
            variant="light",
            p="xs",
            icon=icon("info", 16),
        ),
        _defs(rows),
        subtitle=f"mbr_transferred.parquet, run {t.get('run')}",
        id="pd-transfer-card",
    )


# --------------------------------------------------------------------------- competition


def _th(text: str, label: str | None = None, numeric: bool = False) -> Any:
    cell = tip(html.Span(text, className="pd-th-tip"), label) if label else text
    return dmc.TableTh(cell, className="mv-num" if numeric else None)


def _sparse_q(value: Any, wins: bool, column: str, noun: str, threshold: float) -> Any:
    """A grouped q of one competition row: the value on the group's winner, else a dash.

    The column's header says what the dash means; the dash's own title says it briefly.
    """
    if wins:
        return view.fmt_q_at(value, threshold)
    return html.Span(
        "-", className="pd-dash", title=f"not the winner of its {noun}: {column} holds 1.0"
    )


def comp_table(ctx: PageContext, d: PrecursorDetail) -> Any:
    """The competing rows, best first, with validation marks at ``ctx.threshold`` and score bars.

    This row is highlighted (and not a link); the winners of the base peptide and of each
    precursor carry the grouped q values (dashes elsewhere). Winner chips are shown only
    for groups of more than one row.
    """
    df = d.competition
    if df is None or df.empty:
        return empty("No competition rows.")
    exp = ctx.rs.is_experiment
    thr = ctx.threshold
    vcol = "run_psm_q" if exp else "q_value"
    hrefs = [
        precursor_href(ctx, str(r.run), int(r.candidate_id)) for r in df.itertuples(index=False)
    ]
    n_prec = df.groupby(["peptidoform", "charge"])["candidate_id"].transform("size").astype(int)
    frame = df.assign(pd_href=hrefs, pd_nprec=n_prec).sort_values(
        "score", ascending=False, kind="mergesort"
    )
    b = view.score_bounds(ctx.rs)
    digits = view.page_score_digits(d)
    spikes = "is_entrapment" in frame
    head_cells = [
        _th(
            "",
            f"validation: {vcol} of each row at the threshold; D marks a decoy"
            + ("; E an entrapment spike-in" if spikes else ""),
        ),
        _th("precursor"),
        _th("label"),
    ]
    if exp:
        head_cells.append(_th("run"))
    head_cells += [
        _th("score", f"rescorer score, {digits} decimals; bar: linear {b.text}"),
        _th("q_value", "PSM rows pooled over the whole rescore", True),
    ]
    if exp:
        head_cells.append(_th("run_psm_q", "PSM rows of each row's own run", True))
    head_cells += [
        _th(
            "precursor_q",
            "set on the winning row of each (peptidoform, charge) only; a dash on the others",
            True,
        ),
        _th(
            "peptide_q_value",
            "set on the winning row of the base peptide only; a dash on the others",
            True,
        ),
    ]
    body = []
    for r in frame.itertuples(index=False):
        won = []
        if r.wins_peptide and len(frame) > 1:
            won.append(
                chip(
                    "peptide winner",
                    "green",
                    size="xs",
                    tip="winner of its base peptide: the row that carries peptide_q_value",
                )
            )
        if r.wins_precursor and int(r.pd_nprec) > 1:
            won.append(
                chip(
                    "precursor winner",
                    "teal",
                    size="xs",
                    tip="winner of its (peptidoform, charge): the row that carries precursor_q",
                )
            )
        pep = html.Span(
            [peptidoform(str(r.peptidoform), size="0.82em"), f" {int(r.charge)}+"],
            className="pd-cpep",
        )
        full = f"{r.peptidoform} {int(r.charge)}+, candidate {int(r.candidate_id)}"
        if r.is_this_row:
            name_el: Any = html.Span(pep, title=f"{full}: this row")
        else:
            name_el = dcc.Link(
                pep, href=r.pd_href, className="pd-link", title=f"{full}; opens its page"
            )
        name = html.Div(
            [name_el, dmc.Group(won, gap=4, mt=2) if won else None], className="pd-cname"
        )
        decoy = r.label == "decoy"
        spike = bool(getattr(r, "is_entrapment", False)) if spikes else False
        cells: list[tuple[Any, bool]] = [
            (
                mark(getattr(r, vcol), thr, label=str(r.label), column=vcol, size=16, spike=spike),
                False,
            ),
            (name, False),
            (chip(str(r.label), "orange" if decoy else "indigo", size="xs"), False),
        ]
        if exp:
            cells.append((str(r.run), False))
        cells.append(
            (
                spark_bar(
                    r.score,
                    lo=b.lo,
                    hi=b.hi,
                    width=56,
                    text=view.fmt_score(r.score, digits),
                    colour=SCORE_BAR,
                ),
                False,
            )
        )
        cells.append((view.fmt_q_at(r.q_value, thr), True))
        if exp:
            cells.append((view.fmt_q_at(r.run_psm_q, thr), True))
        cells.append(
            (
                _sparse_q(r.precursor_q, bool(r.wins_precursor), "precursor_q", "precursor", thr),
                True,
            )
        )
        cells.append(
            (
                _sparse_q(
                    r.peptide_q_value,
                    bool(r.wins_peptide),
                    "peptide_q_value",
                    "base peptide",
                    thr,
                ),
                True,
            )
        )
        body.append(
            dmc.TableTr(
                [dmc.TableTd(c, className="mv-num" if num_ else None) for c, num_ in cells],
                className="pd-this-row" if r.is_this_row else None,
            )
        )
    table = dmc.Table(
        [dmc.TableThead(dmc.TableTr(head_cells)), dmc.TableTbody(body)],
        verticalSpacing=5,
        horizontalSpacing=8,
        highlightOnHover=True,
        className="mv-table pd-ctable",
    )
    if len(body) > 8:
        return dmc.ScrollArea(table, h=340, type="auto", offsetScrollbars=True)
    return dmc.TableScrollContainer(table, minWidth=560)


def competition_card(ctx: PageContext, d: PrecursorDetail) -> Any:
    df = d.competition
    exp = ctx.rs.is_experiment
    if df is None or df.empty:
        return card(
            "Base-peptide competition",
            html.Div(empty("No competition rows."), id="pd-comp-table"),
            id="pd-comp-card",
        )
    hrefs = [
        precursor_href(ctx, str(r.run), int(r.candidate_id)) for r in df.itertuples(index=False)
    ]
    fig = dfig.competition_figure(
        df, ctx.scheme, hrefs=hrefs, experiment=exp, digits=view.page_score_digits(d)
    )
    scope = "every run of the experiment" if exp else "this run"
    return card(
        "Base-peptide competition",
        html.Div(graph("pd-comp", fig), className="pd-graph-wrap", id="pd-comp-wrap"),
        html.Div(comp_table(ctx, d), id="pd-comp-table"),
        subtitle=f"Every scored row of base_peptide_id {d.scored.get('base_peptide_id')} in "
        f"{scope}",
        help="peptide_q_value is a picked target-decoy competition on this key: the best row "
        "of the base peptide carries it. precursor_q is set on the best row of each "
        "(peptidoform, charge). Click a dot or a peptidoform to open that row; this row is "
        f"highlighted. Winner flags: {WINNER_SOURCE}.",
        count=len(df),
        id="pd-comp-card",
    )


def partner_body(ctx: PageContext, d: PrecursorDetail) -> list[Any]:
    """The partner card's content at ``ctx.threshold`` (rebuilt when the threshold changes)."""
    p = d.partner
    if p.candidate_id is None:
        return [dmc.Text(sentence(p.reason), size="sm")]
    if p.rows is None or p.rows.empty:
        return [
            dmc.Group(
                [
                    dmc.Text(f"candidate {p.candidate_id}", className="pd-mono", size="sm"),
                    chip(
                        "not scored",
                        "gray",
                        size="sm",
                        tip="the partner has no row in the scored table",
                    ),
                ],
                gap=6,
            ),
            dmc.Text(
                f"The exact library partner was not scored in any run ({p.reason}).",
                size="xs",
                c="dimmed",
            ),
        ]
    exp = ctx.rs.is_experiment
    thr = ctx.threshold
    vcol = "run_psm_q" if exp else "q_value"
    b = view.score_bounds(ctx.rs)
    digits = view.page_score_digits(d)
    s = d.scored
    entries: list[dict[str, Any]] = [
        {
            "who": "this row",
            "peptidoform": s.get("peptidoform"),
            "charge": s.get("charge"),
            "label": s.get("label"),
            "run": d.run.label,
            "score": s.get("score"),
            "q_value": s.get("q_value"),
            "run_psm_q": s.get("run_psm_q"),
            "href": None,
        }
    ]
    for r in p.rows.itertuples(index=False):
        entries.append(
            {
                "who": "partner",
                "peptidoform": r.peptidoform,
                "charge": r.charge,
                "label": r.label,
                "run": r.run,
                "score": r.score,
                "q_value": r.q_value,
                "run_psm_q": r.run_psm_q,
                "href": precursor_href(ctx, str(r.run), int(r.candidate_id)),
                "cid": int(r.candidate_id),
            }
        )
    blocks = []
    for e in entries:
        decoy = e["label"] == "decoy"
        pep = peptidoform(str(e["peptidoform"]), size="0.86em")
        name = dcc.Link(pep, href=e["href"], className="pd-link") if e["href"] else pep
        head = [
            mark(e[vcol], thr, label=str(e["label"]), column=vcol, size=16),
            html.Span([name, f" {int(e['charge'])}+"], className="pd-prow-name"),
            chip(str(e["label"]), "orange" if decoy else "indigo", size="xs"),
            dmc.Text(e["who"] + (f", run {e['run']}" if exp else ""), size="xs", c="dimmed"),
        ]
        values = [
            _kv(
                "score",
                spark_bar(
                    e["score"],
                    lo=b.lo,
                    hi=b.hi,
                    width=64,
                    text=view.fmt_score(e["score"], digits),
                    colour=SCORE_BAR,
                ),
            ),
            _kv("q_value", view.fmt_q_at(e["q_value"], thr)),
        ]
        if exp:
            values.append(_kv("run_psm_q", view.fmt_q_at(e["run_psm_q"], thr)))
        blocks.append(
            html.Div(
                [
                    html.Div(head, className="pd-prow-head"),
                    html.Div(values, className="pd-prow-vals"),
                ],
                className="pd-prow" + (" pd-prow-this" if e["href"] is None else ""),
            )
        )
    note = f"Marks: {vcol} at the threshold. Score bars: linear {b.text}. {sentence(p.reason)}" + (
        "" if exp else " In a single run run_psm_q equals q_value."
    )
    return [*blocks, dmc.Text(note, size="xs", c="dimmed", mt=4)]


def _kv(label: str, value: Any) -> Any:
    return html.Div(
        [html.Span(label, className="pd-kv-label"), html.Span(value, className="pd-kv-value")],
        className="pd-kv",
    )


def partner_card(ctx: PageContext, d: PrecursorDetail) -> Any:
    other = "target" if d.is_decoy else "decoy"
    return card(
        f"Exact {other} partner",
        html.Div(partner_body(ctx, d), id="pd-partner-body"),
        subtitle="The other row of the candidate's library peptidoform_id",
        help="In an imported library each peptidoform_id holds one target and one decoy "
        "precursor. The partner's scored rows are shown with this row; a FASTA-built library "
        "has no such key (see the base-peptide competition).",
        id="pd-partner-card",
    )


# --------------------------------------------------------------------------- features


def pct_notes(rows: list[FeaturePercentile]) -> list[Any]:
    """The population of the ranks and the features that were not ranked."""
    pops = list(dict.fromkeys(r.population.split(";")[0] for r in rows if r.population))
    unranked = [r for r in rows if r.pct_target is None and r.pct_decoy is None]
    parts: list[Any] = []
    if pops:
        parts.append(
            dmc.Text(
                f"Population: {pops[0]}. Rows that fail a feature's validity rule (its sentinel "
                "values) are not ranked; hover a dot for the rule.",
                size="xs",
                c="dimmed",
            )
        )
    for r in unranked:
        parts.append(
            dmc.Text(f"{r.feature}: not ranked; {r.note or r.population}", size="xs", c="dimmed")
        )
    return parts


def features_card(ctx: PageContext, d: PrecursorDetail, rows: list[FeaturePercentile]) -> Any:
    choices = view.percentile_choices(d)
    chosen = [c for c in DEFAULT_PERCENTILE_FEATURES if c in choices]
    boxes = dmc.CheckboxGroup(
        dmc.Stack(
            [
                dmc.Checkbox(
                    label=html.Span(
                        [
                            html.Span(c, className="pd-mono"),
                            " ",
                            html.Span(view.feature_label(c), className="pd-dim"),
                        ]
                    ),
                    value=c,
                    size="xs",
                )
                for c in choices
            ],
            gap=6,
        ),
        id="pd-feat-select",
        value=chosen,
    )
    chooser = dmc.Popover(
        [
            dmc.PopoverTarget(
                dmc.Button(
                    f"{len(chosen)} features",
                    id="pd-feat-btn",
                    variant="default",
                    size="compact-sm",
                    radius="xl",
                    rightSection=icon("layers", 13),
                    disabled=not choices,
                )
            ),
            dmc.PopoverDropdown(dmc.ScrollArea(boxes, h=320, type="auto", offsetScrollbars=True)),
        ],
        width=430,
        position="bottom-end",
        shadow="md",
        withArrow=True,
    )
    return card(
        "Features against the run's targets and decoys",
        html.Div(
            graph("pd-pct", dfig.percentile_figure(rows, ctx.scheme)),
            className="pd-graph-wrap",
            id="pd-pct-wrap",
        ),
        html.Div(pct_notes(rows), id="pd-pct-notes"),
        right=chooser,
        subtitle="Percentiles among the run's scored targets and decoys (not FDR-filtered)",
        help="This candidate's value (selected peak) as a percentile of the run's scored "
        "target rows and of its decoy rows. The rows are not FDR-filtered. Hover a dot for "
        "the feature's definition and its validity rule; choose the features on the right.",
        id="pd-feat-card",
    )


def footer(d: PrecursorDetail, extra: dict[str, float], *, cached: bool = False) -> Any:
    """How long the page took: this page's build first, then the parts.

    ``cached`` says that the precursor detail came from the page's cache (assembled
    earlier, for example by the identification page's preview).
    """
    total = sum(d.timings_ms.values())
    parts = sorted(d.timings_ms.items(), key=lambda kv: -kv[1])
    detail = ", ".join(f"{k} {v:.0f}" for k, v in parts if v >= 1)
    page = extra.get("page")
    more = ", ".join(f"{k} {v:.0f} ms" for k, v in extra.items() if k != "page")
    assembled = (
        f"precursor detail assembled earlier in {total:.0f} ms (cached)"
        if cached
        else f"precursor detail assembled in {total:.0f} ms"
    )
    text = (f"Page built in {page:.0f} ms; " if page is not None else "") + assembled
    return dmc.Group(
        [
            icon("clock", 14),
            tip(
                dmc.Text(text + (f"; {more}" if more else ""), size="xs", c="dimmed"),
                f"precursor_detail parts (ms): {detail}",
            ),
        ],
        gap=6,
        className="pd-footer",
        id="pd-footer",
    )


def missing(
    ctx: PageContext, text: str, links: list[tuple[str, str]] | None = None, d: Any = None
) -> Any:
    """The page for an address it cannot show: the reason, and links that can (``links``)."""
    body: list[Any] = [empty(text, "alert")]
    if links:
        body.append(
            dmc.Group(
                [
                    dcc.Link(
                        dmc.Badge(
                            label, variant="light", size="lg", style={"textTransform": "none"}
                        ),
                        href=target,
                        className="pd-chip-link",
                    )
                    for label, target in links
                ],
                gap=8,
                justify="center",
                mt=-36,
                mb="sm",
            )
        )
    return dmc.Stack([back_link(ctx, d), dmc.Card(body, p="lg", id="pd-missing")], gap="md")
