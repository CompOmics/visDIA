"""Run overview page (P0 view 1)."""

from __future__ import annotations

import json

import pandas as pd
from dash import dcc, html

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import counts as counts_mod
from mumdia_viewer.data import overview as overview_mod
from mumdia_viewer.data.entrapment import entrapment_fdp
from mumdia_viewer.data.mbr import mbr_info, transfer_counts
from mumdia_viewer.data.units import UNITS

from . import figures
from .components import MUTED, SECTION, card, details, fmt, graph, notices, table

CURVE_UNITS = ("psm", "precursor", "peptide", "protein_group")


def layout(rs: ResultSet, t: float) -> html.Div:
    summary = overview_mod.run_summary(rs)
    parts: list = [
        html.H2("Run overview"),
        html.Div(
            [
                html.Span(
                    f"{'Experiment' if rs.is_experiment else 'Single run'}  ",
                    style={"fontWeight": "bold"},
                ),
                html.Span(str(rs.root), style={"fontFamily": "monospace"}),
            ]
        ),
        html.Div(
            f"MuMDIA {summary.mumdia_version or '?'}, git {summary.git_sha or '?'} "
            f"({summary.commit_date or 'date not recorded'}); rescorer {summary.rescore.label}; "
            f"competition key {summary.rescore.group_by_display or 'not recorded'}"
            + (f"; {summary.n_runs} runs: {', '.join(summary.runs)}" if rs.is_experiment else ""),
            style=MUTED,
        ),
    ]
    note_box = notices([(n.code, n.message) for n in rs.notices])
    if note_box is not None:
        parts.append(note_box)

    # Identification counts, each labelled with its row unit and q column.
    unit_counts = counts_mod.unit_counts(rs, t)
    checks = {c.unit: c for c in counts_mod.engine_check(rs)}
    cards = []
    for c in unit_counts:
        check = checks.get(c.unit)
        sub = c.population
        if check is not None and check.engine is not None and abs(t - 0.01) < 1e-12:
            sub += " | engine report: " + (
                f"{check.engine:,} (equal)" if check.equal else f"{check.engine:,} (DIFFERENT)"
            )
        if c.n_decoy:
            sub += f" | decoys at the same cut: {c.n_decoy:,}"
        cards.append(
            card(UNITS[c.unit].plural if c.unit in UNITS else c.unit, f"{c.n_target:,}", sub)
        )
        cards[-1].children.insert(2, html.Div(c.label, style={"fontSize": "0.85em"}))
    parts += [
        html.H3(f"Identifications at q <= {t:g}", style=SECTION),
        html.Div(cards, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
    ]

    if rs.is_experiment:
        per_run = counts_mod.per_run_counts(rs, t)
        labels = per_run.attrs.get("labels", {}) if hasattr(per_run, "attrs") else {}
        parts += [
            html.H3("Per run (run_psm_q, PSM-level FDR within each run)", style=SECTION),
            table(per_run, labels=labels),
            html.Div(per_run.attrs.get("note", ""), style=MUTED),
        ]

    # Curves and the score distribution.
    q_max = max(0.05, 2 * t)
    curves, labels = {}, {}
    for unit in CURVE_UNITS:
        try:
            curves[unit] = counts_mod.id_curve(rs, unit, q_max=q_max, points=200)
            labels[unit] = f"{UNITS[unit].plural} ({UNITS[unit].q_column})"
        except (ViewerError, ValueError):
            continue
    parts.append(
        html.Div(
            [
                html.Div(graph(figures.id_curve_figure(curves, t, labels)), style={"flex": "1"}),
                html.Div(
                    graph(figures.score_histogram_figure(counts_mod.score_histogram(rs))),
                    style={"flex": "1"},
                ),
            ],
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap", **SECTION},
        )
    )

    # Entrapment and match-between-runs, only when they apply.
    try:
        fdp = entrapment_fdp(rs, t)
    except ViewerError as exc:
        fdp, parts = [], [*parts, html.Div(f"entrapment FDP unavailable: {exc}", style=MUTED)]
    if fdp:
        df = pd.DataFrame(
            [
                {
                    "unit": r.unit,
                    "run": r.run,
                    "spike-ins (E)": r.spike_ins,
                    "real targets (R)": r.real,
                    "ratio r": r.ratio,
                    "FDP": r.fdp,
                    "largest accepted q": r.largest_accepted_q,
                }
                for r in fdp
            ]
        )
        parts += [
            html.H3("Entrapment FDP = (r x E + 1) / R", style=SECTION),
            table(df),
            html.Div(fdp[0].label, style=MUTED),
        ]
    info = mbr_info(rs) if rs.is_experiment else None
    if info is not None and info.ran:
        parts += [
            html.H3("Match-between-runs", style=SECTION),
            html.Div(f"strategy {info.strategy}; {fmt(info.n_transfers)} transfers", style=MUTED),
            table(transfer_counts(rs, t)),
        ]

    report = overview_mod.engine_report_numbers(rs)
    if report:
        parts.append(
            html.Div(
                [
                    html.H3("Engine report numbers", style=SECTION),
                    html.Ul([html.Li(f"{n.key} = {fmt(n.value)}: {n.label}") for n in report]),
                ]
            )
        )

    timings = overview_mod.stage_timings_table(rs)
    parts.append(details("Stage timings (from the reports)", table(timings, page_size=30)))
    parts.append(details("Inputs", table(overview_mod.inputs_table(rs), page_size=20)))
    parts.append(details("Artifacts", table(overview_mod.artifact_table(rs), page_size=25)))
    parts.append(
        details(
            "Configuration (config_json)",
            dcc.Markdown(f"```json\n{json.dumps(summary.config, indent=1)}\n```"),
        )
    )
    return html.Div(parts)
