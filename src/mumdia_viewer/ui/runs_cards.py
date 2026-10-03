"""Building blocks of the across-runs page: hero, q strip, run table, siblings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import dcc, html

from mumdia_viewer.data.across import ROW_LABELS, GroupQ, PrecursorAcross, RunXic

from .icons import icon
from .protein_view import protein_href
from .runs_figures import TRANSFER, run_name
from .state import PageContext, href, stop_label
from .widgets import chip, fmt_q, peptidoform, spark_bar, validation_icon

UNIT_TILE = {"precursor_q": "violet", "peptide_q_value": "teal", "pg_q_value": "pink"}
STATE_BADGE = {
    "quantified": "green",
    "not_quantifiable": "yellow",
    "not_selected": "gray",
    "not_scored": "gray",
}
Q_LO, Q_HI = 1.0, 1e-4  # q bars: -log10 from 1 to 1e-4 (longer is smaller)


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compact(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f} G"
    if a >= 1e6:
        return f"{value / 1e6:.2f} M"
    if a >= 1e3:
        return f"{value / 1e3:.1f} k"
    return f"{value:.0f}"


def runs_href(base: str, peptidoform_text: str, charge: int) -> str:
    return href(base, "runs", {"peptidoform": peptidoform_text, "charge": int(charge)})


def run_href(base: str, run: str, cid: Any) -> str | None:
    if cid is None or (isinstance(cid, float) and math.isnan(cid)):
        return None
    return href(base, "precursor", {"run": run, "cid": int(cid)})


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def derived(text: str) -> Any:
    return tip(html.Span("derived", className="xr-derived"), f"Viewer-derived: {text}")


def fact(label: str, value: str, text: str, *, is_derived: bool = False) -> Any:
    head: Any = dmc.Text(label, className="xr-fact-label")
    if is_derived:
        head = html.Div([head, html.Span("derived", className="xr-derived")], className="xr-fhead")
    return tip(
        dmc.Paper(
            [head, dmc.Text(value, className="xr-fact-value")],
            withBorder=True,
            px="md",
            py=8,
            radius="md",
            className="xr-fact",
        ),
        text,
    )


# --------------------------------------------------------------------------- hero


def picker(value: str | None, options: list[dict[str, str]]) -> Any:
    return dmc.Select(
        id="xr-pick",
        data=options,
        value=value,
        searchable=True,
        clearable=False,
        allowDeselect=False,
        nothingFoundMessage="No target precursor contains this text",
        placeholder="Find a precursor",
        leftSection=icon("search", 14),
        size="sm",
        radius="xl",
        w=330,
        limit=30,
        maxDropdownHeight=320,
        comboboxProps={"shadow": "md"},
        className="xr-pick",
        **{"aria-label": "Another precursor"},
    )


def pick_value(peptidoform_text: str, charge: int) -> str:
    return f"{peptidoform_text}|{int(charge)}"


def parse_pick(value: Any) -> tuple[str, int] | None:
    text = str(value or "")
    pep, sep, z = text.rpartition("|")
    if not sep or not pep:
        return None
    try:
        return pep, int(z)
    except ValueError:
        return None


def pick_label(peptidoform_text: str, charge: int, n_runs: Any, q: Any) -> str:
    runs = f"{int(n_runs)} run{'s' if int(n_runs) != 1 else ''}" if n_runs is not None else ""
    qt = f"precursor_q {fmt_q(q)}" if _finite(q) is not None else ""
    return " · ".join(x for x in (f"{peptidoform_text} {int(charge)}+", runs, qt) if x)


def hero(ctx: PageContext, a: PrecursorAcross, xics: Sequence[RunXic], options: list) -> Any:
    rs = ctx.rs
    decoy = a.label == "decoy"
    kind = f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"
    pills = [
        dmc.Badge(
            f"{a.charge}+",
            size="lg",
            variant="outline",
            color="gray",
            radius="sm",
            style={"textTransform": "none"},
        ),
        chip(
            a.label,
            "orange" if decoy else "indigo",
            variant="filled",
            size="lg",
            tip="label column of the scored table",
        ),
    ]
    if a.mbr and a.rows["transferred"].any():
        n = int(a.rows["transferred"].sum())
        pills.append(
            chip(
                f"MBR transfer in {n} run{'s' if n != 1 else ''}",
                "lime",
                variant="filled",
                size="lg",
                tip="match-between-runs transferred this precursor into these runs (transfer "
                "table); their quantities are hatched and their run_psm_q after MBR is shown "
                "apart",
            )
        )
    title = dmc.Group(
        [html.Div(peptidoform(a.peptidoform, size="1.85rem"), className="xr-title"), *pills],
        gap=10,
        align="center",
    )
    chips: list[Any] = []
    if a.protein_group:
        target = (
            protein_href(ctx.base, a.protein_group)
            if not a.protein_group.startswith("DECOY_")
            else href(ctx.base, "identifications", {"search": a.protein_group})
        )
        chips.append(
            dcc.Link(
                chip(
                    f"group {a.protein_group}",
                    "pink",
                    tip="protein_group column; opens the protein page",
                    left=icon("layers", 12),
                ),
                href=target,
                className="xr-chip-link",
            )
        )
    if a.base_peptide_id is not None:
        chips.append(
            dcc.Link(
                chip(
                    f"base peptide {a.base_peptide_id}",
                    "teal",
                    tip="base_peptide_id; opens the peptide's precursors in the identifications",
                ),
                href=href(
                    ctx.base,
                    "identifications",
                    {"group": a.protein_group, "peptide": a.base_peptide_id},
                ),
                className="xr-chip-link",
            )
        )
    cids = {int(c) for c in a.rows["candidate_id"].dropna()}
    if len(cids) == 1:
        chips.append(
            tip(
                dmc.Text(f"candidate {next(iter(cids))}", className="xr-mono", size="xs"),
                "candidate_id: the same library candidate in every run that scored it",
            )
        )
    chips.append(dmc.Text(f"{kind} · {rs.root.name}", size="xs", c="dimmed"))
    best = _best_run(a)
    back = (
        dcc.Link(
            dmc.Group([icon("left", 14), dmc.Text("Precursor detail", size="sm")], gap=4),
            href=run_href(ctx.base, best[0], best[1]) or href(ctx.base, "identifications"),
            className="xr-back",
        )
        if best is not None
        else dcc.Link(
            dmc.Group([icon("left", 14), dmc.Text("Identifications", size="sm")], gap=4),
            href=href(ctx.base, "identifications"),
            className="xr-back",
        )
    )
    left = dmc.Stack(
        [
            html.Div([back, dmc.Text(f"Across runs · {kind}", className="mv-eyebrow", mt=6)]),
            title,
            dmc.Group(chips, gap=8),
        ],
        gap=8,
        className="xr-hero-left",
    )
    t = ctx.threshold
    n = a.n_runs
    rq = a.rows["run_psm_q"].to_numpy(dtype="float64")
    passing = int(np.sum(np.isfinite(rq) & (rq <= t)))
    apex = a.rows["apex_rt"].to_numpy(dtype="float64", na_value=np.nan)
    apex = apex[np.isfinite(apex)]
    spread = float(apex.max() - apex.min()) if apex.size > 1 else None
    facts = [
        fact(
            "scored",
            f"{a.n_scored} / {n}",
            "Runs with a scored row of this (peptidoform, charge) in the pooled scored table",
        ),
        fact(
            f"run_psm_q ≤ {stop_label(t)}",
            f"{passing} / {n}",
            "Runs whose row passes the header threshold on run_psm_q (this run's own "
            "target-decoy re-run; the per-run q column)",
        ),
        fact(
            "quantified",
            f"{a.n_quantified} / {n}",
            "Runs with a peptide_quant quantity for the row (a missing quantity is never 0)",
        ),
    ]
    if spread is not None:
        facts.append(
            fact(
                "apex RT range",
                f"{spread:.1f} s",
                "Viewer-derived: the largest minus the smallest apex_rt over the runs that "
                "scored the precursor; each run has its own RT axis (no alignment)",
                is_derived=True,
            )
        )
    right = dmc.Stack(
        [
            picker(pick_value(a.peptidoform, a.charge), options),
            dmc.Group(facts, gap="sm", wrap="nowrap", className="xr-facts"),
        ],
        gap="sm",
        align="flex-end",
        className="xr-hero-right",
    )
    return dmc.Group(
        [left, right], justify="space-between", align="flex-end", gap="lg", className="xr-hero"
    )


def _best_run(a: PrecursorAcross) -> tuple[str, int] | None:
    rows = a.rows[a.rows["scored"]]
    if rows.empty:
        return None
    r = rows.sort_values("run_psm_q", kind="mergesort").iloc[0]
    return str(r["run"]), int(r["candidate_id"])


# --------------------------------------------------------------------------- grouped q strip


def _winner_text(g: GroupQ, experiment: bool) -> str:
    if g.value is None:
        return f"{g.column}: no row of the group {g.group}."
    where = f"run {run_name(g.winner_run or '')}" if experiment else "this run"
    who = (
        "a row of this precursor"
        if g.this_precursor
        else f"{g.winner_peptidoform} {g.winner_charge}+ ({g.winner_label})"
    )
    scope = "experiment-wide" if experiment else "this run"
    return (
        f"{g.column}: {g.unit}; {scope}. The group ({g.group}) has its q on one winning row "
        f"only, the other rows hold 1.0. Winning row: {who}, {where}, candidate "
        f"{g.winner_cid}. Found by {g.source}."
    )


def q_strip(ctx: PageContext, a: PrecursorAcross) -> Any:
    t = ctx.threshold
    scope = "experiment-wide" if a.experiment else "this run"
    head = tip(
        html.Div(
            [
                html.Div("Grouped q", className="mv-section-title"),
                html.Div(f"q ≤ {stop_label(t)}", className="xr-vh-t"),
                html.Div(scope, className="xr-vh-n"),
            ],
            className="xr-vhead",
        ),
        "The grouped q columns are set on each group's winning row only. "
        + (
            "In an experiment they are experiment-wide: one value for the group over every "
            "run, not a per-run value. Per-run values are run_psm_q (table below)."
            if a.experiment
            else "In a single run they are this run's values."
        ),
        w=340,
        position="bottom-start",
    )
    tiles = []
    for g in a.grouped:
        mark = validation_icon(g.value, t, label=g.winner_label, column=g.column, size=16)
        if g.value is None:
            value: Any = html.Span("no row", className="xr-vt-none")
        else:
            value = fmt_q(g.value, t)
        side = (
            html.Span(
                "this precursor"
                if g.this_precursor
                else f"on {g.winner_peptidoform} {g.winner_charge}+"[:28],
                className="xr-vt-side",
            )
            if g.value is not None
            else None
        )
        where = (
            html.Span(f"winner in {run_name(g.winner_run or '')}", className="xr-vt-where")
            if g.value is not None and a.experiment
            else None
        )
        tiles.append(
            tip(
                html.Div(
                    [
                        html.Div(
                            [html.Span(g.column, className="xr-vt-col"), side],
                            className="xr-vt-top",
                        ),
                        html.Div(
                            [mark, html.Div(value, className="xr-vt-val"), where],
                            className="xr-vt-line",
                        ),
                    ],
                    className=f"xr-vt xr-vt-{UNIT_TILE.get(g.column, 'gray')}",
                ),
                _winner_text(g, a.experiment),
                w=360,
                boxWrapperProps={"w": "100%"},
                position="bottom",
                openDelay=150,
            )
        )
    return dmc.Card(html.Div([head, *tiles], className="xr-vstrip"), p="sm", id="xr-strip")


# --------------------------------------------------------------------------- run table


def _th(text: str, tip_text: str | None = None, *, num: bool = False) -> Any:
    label: Any = text
    if tip_text:
        label = tip(html.Span([text, icon("info", 10)], className="xr-th"), tip_text)
    return dmc.TableTh(label, className="mv-num" if num else None)


def run_table(
    ctx: PageContext, a: PrecursorAcross, xics: Sequence[RunXic], suggestion: dict[str, str]
) -> Any:
    t = ctx.threshold
    rows = a.rows
    scores = rows["score"].to_numpy(dtype="float64")
    s_hi = max(1.0, float(np.nanmax(scores))) if np.isfinite(scores).any() else 1.0
    s_lo = min(0.0, float(np.nanmin(scores))) if np.isfinite(scores).any() else 0.0
    quants = np.concatenate(
        [rows["quantity"].to_numpy(dtype="float64"), rows["lfq_quantity"].to_numpy(dtype="float64")]
    )
    good = quants[np.isfinite(quants) & (quants > 0)]
    q_hi = float(good.max()) if good.size else 1.0
    q_lo = float(good.min()) / 10.0 if good.size else 0.1
    q_scale = f"log10 from {compact(q_lo)} to {compact(q_hi)} (the largest value of the table)"
    head = dmc.TableThead(
        dmc.TableTr(
            [
                _th("", "Validation on run_psm_q (this run's own q) at the header threshold"),
                _th("run", "The run; its condition (from the Quant QC conditions) under it"),
                _th("score", f"{ROW_LABELS['score']}. Bar: linear {s_lo:g} to {s_hi:g}."),
                _th("run_psm_q", ROW_LABELS["run_psm_q"] + ". Bar: -log10 from 1 to 1e-4."),
                _th("q_value", ROW_LABELS["q_value"] + ". Bar: -log10 from 1 to 1e-4."),
                _th("apex RT", ROW_LABELS["apex_rt"], num=True),
                _th(
                    "RT error",
                    "Viewer-derived: apex_rt - rt_pred_cal of the run (run_windows), seconds",
                    num=True,
                ),
                _th("elution", ROW_LABELS["elution"], num=True),
                _th("quant state", "The quant state of the row (peptide_quant and the quant gate)"),
                _th("quantity", f"{ROW_LABELS['quantity']}. Bar: {q_scale}."),
                _th("MaxLFQ", f"{ROW_LABELS['lfq_quantity']}. Bar: {q_scale}."),
                _th("", None),
            ]
        )
    )
    body = []
    for i, row in enumerate(rows.to_dict("records")):
        run = str(row["run"])
        scored = bool(row["scored"])
        x = xics[i] if i < len(xics) else None
        link = run_href(ctx.base, run, row["candidate_id"]) if scored else None
        q = _finite(row["run_psm_q"])
        name: Any = html.Span(run_name(run), className="xr-run")
        if link:
            name = dcc.Link(name, href=link, className="xr-link")
        cond = html.Span(
            suggestion.get(run, ""), id={"type": "xr-cond", "run": run}, className="xr-cond"
        )
        state = str(row["state"])
        badge = dmc.Badge(
            state.replace("_", " "),
            color=STATE_BADGE.get(state, "gray"),
            variant="light",
            size="sm",
            style={"textTransform": "none"},
        )
        badge = tip(badge, str(row.get("reason") or state), w=360)
        if not scored:
            cells = [
                html.Span(className="xr-none-mark"),
                html.Div([name, cond], className="xr-run-cell"),
                html.Span("not scored in this run", className="xr-dim"),
                "",
                "",
                "",
                "",
                "",
                badge,
                "",
                spark_bar(
                    row["lfq_quantity"],
                    lo=q_lo,
                    hi=q_hi,
                    scale="log10",
                    width=46,
                    text=compact(_finite(row["lfq_quantity"])),
                    colour="var(--mantine-color-pink-5)",
                )
                if _finite(row["lfq_quantity"]) is not None
                else "",
                "",
            ]
        else:
            err = (
                (x.apex_rt - x.rt_pred_cal)
                if x is not None and x.apex_rt is not None and x.rt_pred_cal is not None
                else None
            )
            tr = bool(row["transferred"])
            qty = _finite(row["quantity"])
            qcell: Any = (
                spark_bar(
                    qty,
                    lo=q_lo,
                    hi=q_hi,
                    scale="log10",
                    width=46,
                    text=compact(qty),
                    colour=TRANSFER if tr else "var(--mantine-color-violet-5)",
                )
                if qty is not None
                else html.Span("missing", className="xr-dim")
            )
            if tr:
                qcell = html.Div(
                    [
                        qcell,
                        tip(
                            html.Span("MBR", className="xr-mbr"),
                            "match-between-runs transfer"
                            + (
                                f"; transfer_q {fmt_q(row['transfer_q'])}"
                                if _finite(row["transfer_q"]) is not None
                                else ""
                            )
                            + (
                                f"; run_psm_q after MBR {fmt_q(row['run_psm_q_after_mbr'])}"
                                if _finite(row["run_psm_q_after_mbr"]) is not None
                                else ""
                            ),
                        ),
                    ],
                    className="xr-qcell",
                )
            lfq = _finite(row["lfq_quantity"])
            cells = [
                validation_icon(q, t, label=row["label"], column="run_psm_q", size=18),
                html.Div([name, cond], className="xr-run-cell"),
                spark_bar(row["score"], lo=s_lo, hi=s_hi, width=46, text=f"{row['score']:.4f}"),
                spark_bar(
                    q,
                    lo=Q_LO,
                    hi=Q_HI,
                    scale="neglog10",
                    width=46,
                    text=fmt_q(q, t),
                    colour="var(--mantine-color-green-6)"
                    if q is not None and q <= t
                    else "var(--mantine-color-gray-5)",
                ),
                spark_bar(
                    row["q_value"],
                    lo=Q_LO,
                    hi=Q_HI,
                    scale="neglog10",
                    width=46,
                    text=fmt_q(row["q_value"], t),
                    colour="var(--mantine-color-indigo-4)",
                ),
                f"{row['apex_rt']:.2f}" if _finite(row["apex_rt"]) is not None else "",
                html.Span(f"{err:+.2f}", className="xr-num") if err is not None else "",
                (
                    f"{row['elution_lo']:.1f} to {row['elution_hi']:.1f}"
                    if _finite(row["elution_lo"]) is not None
                    and _finite(row["elution_hi"]) is not None
                    else ""
                ),
                badge,
                qcell,
                spark_bar(
                    lfq,
                    lo=q_lo,
                    hi=q_hi,
                    scale="log10",
                    width=46,
                    text=compact(lfq),
                    colour="var(--mantine-color-pink-5)",
                )
                if lfq is not None
                else html.Span("missing", className="xr-dim")
                if a.experiment
                else html.Span("n/a", className="xr-dim"),
                tip(
                    dcc.Link(icon("external", 14), href=link, className="xr-open"),
                    f"Open the precursor page of {run_name(run)}",
                )
                if link
                else "",
            ]
        numeric = {5, 6, 7}
        body.append(
            dmc.TableTr(
                [
                    dmc.TableTd(c, className="mv-num" if j in numeric else None)
                    for j, c in enumerate(cells)
                ],
                className="xr-row" + ("" if scored else " xr-row-off"),
            )
        )
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        highlightOnHover=True,
        verticalSpacing=6,
        horizontalSpacing="sm",
        className="mv-table xr-table",
    )
    return dmc.TableScrollContainer(table, minWidth=960)


# --------------------------------------------------------------------------- siblings


def siblings(ctx: PageContext, a: PrecursorAcross, limit: int = 12) -> Any:
    s = a.siblings
    if s.empty:
        return dmc.Text(
            "No other precursor of this base peptide was scored.", size="xs", c="dimmed"
        )
    items = []
    for row in s.head(limit).to_dict("records"):
        decoy = row["label"] == "decoy"
        mark = validation_icon(
            row["precursor_q"], ctx.threshold, label=row["label"], column="precursor_q", size=16
        )
        body = html.Div(
            [
                mark,
                peptidoform(str(row["peptidoform"]), size="0.82rem"),
                html.Span(f"{int(row['charge'])}+", className="xr-sib-z"),
                html.Span(
                    f"{int(row['n_runs'])} run{'s' if int(row['n_runs']) != 1 else ''}",
                    className="xr-sib-n",
                ),
            ],
            className="xr-sib" + (" xr-sib-decoy" if decoy else ""),
        )
        items.append(
            dcc.Link(
                body,
                href=runs_href(ctx.base, str(row["peptidoform"]), int(row["charge"])),
                className="xr-sib-link",
            )
        )
    more = [dmc.Text(f"and {len(s) - limit} more", size="xs", c="dimmed")] if len(s) > limit else []
    return html.Div([*items, *more], className="xr-sibs")
