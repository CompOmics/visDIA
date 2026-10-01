"""A static HTML report of the overview: one file that opens offline.

The report holds what the overview page shows at one threshold:
- the identity of the run;
- the counts per unit with their q columns and the engine agreement;
- the identification curves, the score distribution and the per-run counts;
- the stage timings, the inputs, the artifacts and the configuration.

plotly.js is written into the file, so it needs no network. Every number is MuMDIA's
own column, or says how the viewer derived it, as on the page.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

from mumdia_viewer import __version__
from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import counts as counts_mod
from mumdia_viewer.data import overview as overview_mod
from mumdia_viewer.data.units import UNITS

from . import figures
from .overview import CARD_UNITS, _curves
from .state import stop_label
from .theme import GRAPH_CONFIG

STYLE = """
body { font-family: Inter, 'Segoe UI', system-ui, sans-serif; margin: 0; color: #212529;
       background: #f8f9fa; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 4px 0 2px; letter-spacing: -0.02em; }
h2 { font-size: 13px; letter-spacing: 0.06em; text-transform: uppercase; color: #868e96;
     margin: 30px 0 10px; }
.eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
           color: #4263eb; }
.path, code { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px; color: #495057; }
.card { background: #fff; border: 1px solid #dee2e6; border-radius: 12px; padding: 14px 16px;
        margin-bottom: 14px; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.kpi .n { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.kpi .unit { font-weight: 650; }
.kpi .q { font-family: Consolas, monospace; font-size: 12px; color: #4263eb; }
.small { font-size: 12px; color: #868e96; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e9ecef;
         vertical-align: top; }
th { color: #495057; font-weight: 650; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
pre { white-space: pre-wrap; font-size: 11.5px; background: #f1f3f5; padding: 10px;
      border-radius: 8px; max-height: 420px; overflow: auto; }
.chips span { display: inline-block; margin: 2px 4px 2px 0; padding: 2px 8px;
              border-radius: 999px; background: #edf2ff; color: #364fc7; font-size: 12px; }
footer { margin-top: 40px; font-size: 12px; color: #868e96; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _table(df: pd.DataFrame | None, *, labels: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return '<p class="small">None.</p>'
    cols = list(df.columns)
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(df[c])}
    head = "".join(f"<th>{_e((labels or {}).get(c, c))}</th>" for c in cols)
    rows = []
    for record in df.itertuples(index=False, name=None):
        cells = []
        for c, v in zip(cols, record, strict=True):
            text = "" if v is None or (isinstance(v, float) and v != v) else v
            if isinstance(text, float):
                text = f"{text:.4g}"
            elif isinstance(text, int) and not isinstance(text, bool):
                text = f"{text:,}"
            cls = ' class="num"' if c in numeric else ""
            cells.append(f"<td{cls}>{_e(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _figure(fig: Any) -> str:
    config = {**GRAPH_CONFIG, "responsive": True}
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=config)


def overview_report(rs: ResultSet, threshold: float = 0.01) -> str:
    """The report as one HTML document (plotly.js inline)."""
    t = float(threshold)
    summary = overview_mod.run_summary(rs)
    r = summary.rescore
    kind = f"Experiment, {summary.n_runs} runs" if rs.is_experiment else "Single run"
    chips = [
        f"MuMDIA {summary.mumdia_version or 'version not recorded'}",
        f"git {summary.git_sha or 'not recorded'} ({summary.commit_date or 'no date'})",
        f"rescorer {r.classifier or 'not recorded'} ({r.model_identity or 'no model id'})",
        f"competition {r.group_by_display or 'not recorded'}",
    ]
    for key, value in (summary.model_identities or {}).items():
        if key not in ("rescorer", "feature_schema_id"):
            chips.append(f"{key} {value}")
    parts: list[str] = [
        f'<div class="eyebrow">{_e(kind)}</div><h1>{_e(rs.root.name)}</h1>',
        f'<div class="path">{_e(rs.root)}</div>',
        '<div class="chips" style="margin-top:10px">'
        + "".join(f"<span>{_e(c)}</span>" for c in chips)
        + "</div>",
    ]
    if rs.notices:
        parts.append("<h2>Notices</h2><div class='card'>")
        parts += [f"<p><b>{_e(n.code)}</b>: {_e(n.message)}</p>" for n in rs.notices]
        parts.append("</div>")

    parts.append(f"<h2>Identifications at q &le; {_e(stop_label(t))}</h2><div class='kpis'>")
    try:
        found = {c.unit: c for c in counts_mod.unit_counts(rs, t)}
    except (ViewerError, ValueError) as exc:
        found = {}
        parts.append(f'<p class="small">{_e(exc)}</p>')
    for unit in CARD_UNITS:
        c = found.get(unit)
        u = UNITS[unit]
        parts.append(
            "<div class='card kpi'>"
            f"<div class='unit'>{_e(u.plural)}</div>"
            f"<div class='n'>{_e(f'{c.n_target:,}' if c else '-')}</div>"
            f"<div class='q'>{_e(u.q_column)} &le; {_e(stop_label(t))}</div>"
            f"<div class='small'>{_e(u.distinct_label)}"
            + (f"; {c.n_decoy:,} decoys pass the same cut" if c else "")
            + "</div></div>"
        )
    parts.append("</div>")
    notes = [f"{UNITS[u].plural}: {found[u].note}" for u in CARD_UNITS if u in found]
    if notes:
        parts.append("<div class='small'>" + "<br>".join(_e(n) for n in notes) + "</div>")

    try:
        checks = counts_mod.engine_check(rs)
        rows = [
            {
                "unit": UNITS[c.unit].plural if c.unit in UNITS else c.unit,
                "viewer": c.viewer,
                "engine report": c.engine,
                "equal": "yes" if c.equal else ("not in the report" if c.engine is None else "NO"),
            }
            for c in checks
        ]
        parts.append("<h2>Engine agreement at q &le; 0.01</h2><div class='card'>")
        parts.append(_table(pd.DataFrame(rows)) + "</div>")
    except ViewerError as exc:
        parts.append(f"<p class='small'>{_e(exc)}</p>")

    curves = _curves(rs)
    labels = {u: f"{UNITS[u].plural} ({UNITS[u].q_column})" for u in curves}
    parts.append("<h2>Identifications against the threshold</h2><div class='card'>")
    parts.append(_figure(figures.id_curves_figure(curves, t, labels, "light")))
    parts.append(
        "<div class='small'>Exact counts of the engine's q columns at each threshold; no q "
        "value is recomputed.</div></div>"
    )
    try:
        hist = counts_mod.score_histogram(rs)
        parts.append("<h2>Score distribution</h2><div class='card'>")
        parts.append(_figure(figures.score_histogram_figure(hist, "light")) + "</div>")
    except ViewerError:
        pass
    if rs.is_experiment:
        try:
            per_run = counts_mod.per_run_counts(rs, t)
            run_labels = per_run.attrs.get("labels", {})
            parts.append("<h2>Per run (run_psm_q)</h2><div class='card'>")
            parts.append(_figure(figures.per_run_figure(per_run, run_labels, "light")))
            parts.append(_table(per_run.drop(columns=["source"], errors="ignore")))
            parts.append(f"<div class='small'>{_e(per_run.attrs.get('note', ''))}</div></div>")
        except (ViewerError, ValueError) as exc:
            parts.append(f"<p class='small'>{_e(exc)}</p>")
    try:
        timings = overview_mod.stage_timings_table(rs)
        parts.append("<h2>Stage timings</h2><div class='card'>")
        parts.append(_figure(figures.stage_timings_figure(timings, "light")))
        shown = timings[
            [c for c in ("directory", "stage", "elapsed_s", "artifacts") if c in timings]
        ]
        parts.append(_table(shown) + "</div>")
    except ViewerError:
        pass
    for title, build, keep in (
        ("Inputs", overview_mod.inputs_table, ["key", "resolved_path", "bytes", "status"]),
        (
            "Artifacts",
            overview_mod.artifact_table,
            ["scope", "key", "version", "rows", "status", "stage", "content_hash_short"],
        ),
    ):
        try:
            frame = build(rs)
            frame = frame[[c for c in keep if c in frame.columns]]
            parts.append(f"<h2>{title}</h2><div class='card'>{_table(frame)}</div>")
        except ViewerError as exc:
            parts.append(f"<h2>{title}</h2><p class='small'>{_e(exc)}</p>")
    config = json.dumps(summary.config or {}, indent=2, sort_keys=True)
    parts.append(f"<h2>Configuration (config_json)</h2><pre>{_e(config)}</pre>")
    if summary.cli_args:
        parts.append(f"<h2>Command line</h2><pre>{_e(' '.join(summary.cli_args))}</pre>")
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<footer>Generated by mumdia-viewer {_e(__version__)} on {_e(now)} from "
        f"{_e(rs.root)}. Every number is MuMDIA's own column unless it says how the viewer "
        "derived it.</footer>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(rs.root.name)}: MuMDIA overview</title><style>{STYLE}</style>"
        f"<script>{get_plotlyjs()}</script></head><body><main>"
        + "".join(parts)
        + "</main></body></html>"
    )
