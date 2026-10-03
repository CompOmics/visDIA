"""Figures of the quant QC page: pure functions of :mod:`mumdia_viewer.data.quantqc` results.

A violin here is drawn from the viewer's histogram of a run or condition (one shape per
group, scaled to the same width), with the 25th to 75th percentile as a bar and the
median as a dot; the card that holds the figure says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mumdia_viewer.data.quantqc import (
    CvResult,
    Distribution,
    Heatmap,
    ProteinProfile,
    condition_groups,
)

from . import theme
from .figures import empty_figure, themed

LEVEL_TITLES = {"precursor": "Precursors", "protein": "Protein groups"}
LEVEL_NOUNS = {"precursor": "precursors", "protein": "protein groups"}
LEVEL_COLOURS = {
    "precursor": theme.UNIT_COLOURS["precursor"],
    "protein": theme.UNIT_COLOURS["protein_group"],
}
# Diverging scale of the relative heatmap; its centre follows the theme (quant.js swaps
# it when the theme changes), so a cell at the row mean is never the grey of a missing
# cell.
CENTRE = {"light": "#f1f3f5", "dark": "#3a3b3d"}


def diverging(scheme: str) -> list[list[Any]]:
    return [
        [0.0, "#1864ab"],
        [0.25, "#4dabf7"],
        [0.5, CENTRE["dark" if scheme == "dark" else "light"]],
        [0.75, "#ff8787"],
        [1.0, "#c92a2a"],
    ]


MISSING_CELL = "rgba(134, 142, 150, 0.45)"
RELATIVE_RANGE = 2.0  # log2 units either side of the row mean


def run_label(run: str) -> str:
    return run or "single run"


def condition_colours(conditions: Mapping[str, str], runs: Sequence[str]) -> dict[str, str]:
    """A colour per condition (the categorical palette, in the order of the first run)."""
    out: dict[str, str] = {}
    for run in runs:
        c = conditions.get(run)
        if c is not None and c not in out:
            out[c] = theme.SERIES[len(out) % len(theme.SERIES)]
    return out


def _rgba(colour: str, alpha: float) -> str:
    c = colour.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def compact(value: float | None) -> str:
    """A quantity in a short form: 1.8 M, 27.5 k, 412."""
    if value is None or not math.isfinite(value):
        return "missing"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f} G"
    if a >= 1e6:
        return f"{value / 1e6:.2f} M"
    if a >= 1e4:
        return f"{value / 1e3:.1f} k"
    return f"{value:.4g}"


def _power_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    start, stop = math.floor(lo), math.ceil(hi)
    vals = list(range(start, stop + 1))
    return [float(v) for v in vals], [f"10<sup>{v}</sup>" for v in vals]


def _violin(
    fig: go.Figure,
    x0: float,
    counts: np.ndarray,
    edges: np.ndarray,
    colour: str,
    text: str,
    *,
    col: int,
    name: str,
    legend: bool,
) -> None:
    """One violin from a histogram: trimmed to the bins with data, its fill hoverable."""
    if counts.sum() == 0:
        return
    nz = np.flatnonzero(counts)
    lo, hi = int(nz[0]), int(nz[-1])
    centres = (edges[:-1] + edges[1:]) / 2.0
    y = np.concatenate([[edges[lo]], centres[lo : hi + 1], [edges[hi + 1]]])
    w = np.concatenate([[0.0], 0.42 * counts[lo : hi + 1] / counts.max(), [0.0]])
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x0 + w, (x0 - w)[::-1]]),
            y=np.concatenate([y, y[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(color=colour, width=1.2, shape="spline", smoothing=0.6),
            fillcolor=_rgba(colour, 0.30),
            hoveron="fills",
            hoverinfo="text",
            text=text,
            name=name,
            legendgroup=name,
            showlegend=legend,
        ),
        row=1,
        col=col,
    )


def _box(fig: go.Figure, x0: float, p25: float, p50: float, p75: float, colour: str, col: int):
    fig.add_trace(
        go.Scatter(
            x=[x0, x0],
            y=[p25, p75],
            mode="lines",
            line=dict(color=colour, width=6),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=[x0],
            y=[p50],
            mode="markers",
            marker=dict(color="#ffffff", size=7, line=dict(color=colour, width=2)),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=col,
    )


def distribution_figure(
    dists: Mapping[str, Distribution | None],
    conditions: Mapping[str, str],
    scheme: str = "light",
    *,
    height: int = 340,
) -> go.Figure:
    """log10 quantity per run as violins, one panel per level (precursors, protein groups)."""
    levels = [lvl for lvl in ("precursor", "protein") if dists.get(lvl) is not None]
    if not levels:
        return empty_figure("no quantities", scheme, height=height)
    fig = make_subplots(
        rows=1,
        cols=len(levels),
        horizontal_spacing=0.11,
        subplot_titles=[LEVEL_TITLES[lvl] for lvl in levels],
    )
    seen: set[str] = set()
    for c, level in enumerate(levels, start=1):
        d = dists[level]
        assert d is not None
        colours = condition_colours(conditions, d.runs)
        noun = LEVEL_NOUNS[level]
        lows, highs = [], []
        for j, run in enumerate(d.runs):
            row = d.table.iloc[j]
            cond = conditions.get(run, "")
            colour = colours.get(cond, theme.SERIES[0])
            if row["n"] == 0:
                continue
            lows.append(d.edges[np.flatnonzero(d.counts[j])[0]])
            highs.append(d.edges[np.flatnonzero(d.counts[j])[-1] + 1])
            text = (
                f"<b>{run_label(run)}</b>"
                + (f" · condition {cond}" if cond else "")
                + f"<br>{int(row['n']):,} {noun} with a value"
                + (f", {int(row['missing']):,} without" if row["missing"] else "")
                + f"<br>median {compact(10 ** row['p50'])}"
                + f"<br>25th to 75th percentile {compact(10 ** row['p25'])} to "
                f"{compact(10 ** row['p75'])}"
                + f"<br>5th to 95th percentile {compact(10 ** row['p5'])} to "
                f"{compact(10 ** row['p95'])}"
            )
            legend = bool(cond) and cond not in seen
            seen.add(cond)
            _violin(
                fig,
                j,
                d.counts[j],
                d.edges,
                colour,
                text,
                col=c,
                name=f"condition {cond}" if cond else run_label(run),
                legend=legend,
            )
            _box(fig, j, row["p25"], row["p50"], row["p75"], colour, c)
        fig.update_xaxes(
            tickvals=list(range(len(d.runs))),
            ticktext=[run_label(r) for r in d.runs],
            range=[-0.6, len(d.runs) - 0.4],
            showgrid=False,
            zeroline=False,
            row=1,
            col=c,
        )
        if lows:
            tv, tt = _power_ticks(min(lows), max(highs))
            fig.update_yaxes(
                tickvals=tv,
                ticktext=tt,
                range=[min(lows) - 0.15, max(highs) + 0.15],
                title_text="quantity (log scale)" if c == 1 else None,
                row=1,
                col=c,
            )
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)
        ann.x = ann.x - 0.5 / len(levels) + 0.005
        ann.xanchor = "left"
        ann.yshift = 6
    return themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=60, r=12, t=58, b=36),
        legend=dict(y=1.13),
        hovermode="closest",
    )


def missing_figure(
    missing: Mapping[str, pd.DataFrame],
    per_runs: Mapping[str, pd.DataFrame] | None,
    scheme: str = "light",
    *,
    accepted: bool,
    height: int = 340,
) -> go.Figure:
    """Missing values per run (bars per level) and, for experiments, keys by runs with a value.

    ``missing`` and ``per_runs`` are :func:`.quantqc.missing_by_run` and
    :func:`.quantqc.runs_with_value` per level.
    """
    levels = [lvl for lvl in ("precursor", "protein") if lvl in missing]
    if not levels:
        return empty_figure("no quantities", scheme, height=height)
    two = per_runs is not None and len(next(iter(missing.values()))) > 1
    fig = make_subplots(
        rows=2 if two else 1,
        cols=1,
        vertical_spacing=0.24,
        subplot_titles=(
            ["Without a value, per run", "With a value in n runs"] if two else ["Without a value"]
        ),
    )
    whose = "accepted " if accepted else ""
    for level in levels:
        df = missing[level]
        noun = LEVEL_NOUNS[level]
        fig.add_trace(
            go.Bar(
                x=[run_label(r) for r in df["run"]],
                y=df["missing_pct"],
                name=LEVEL_TITLES[level],
                legendgroup=level,
                marker=dict(color=LEVEL_COLOURS[level], line=dict(width=0)),
                customdata=np.stack([df["missing"], df["keys"]], axis=1),
                hovertemplate="%{x}: %{customdata[0]:,} of %{customdata[1]:,} "
                + f"{whose}{noun} without a value (%{{y:.1f}}%)<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if two and per_runs is not None:
            p = per_runs[level]
            p = p[p["n_runs"] > 0] if (p["keys"].iloc[0] == 0) else p
            fig.add_trace(
                go.Bar(
                    x=p["n_runs"].astype(str),
                    y=p["pct"],
                    name=LEVEL_TITLES[level],
                    legendgroup=level,
                    showlegend=False,
                    marker=dict(color=LEVEL_COLOURS[level], line=dict(width=0)),
                    customdata=p["keys"],
                    hovertemplate="%{customdata:,} "
                    + f"{whose}{noun}"
                    + " with a value in %{x} runs (%{y:.1f}%)<extra></extra>",
                ),
                row=2,
                col=1,
            )
    fig.update_yaxes(title_text="% of keys", rangemode="tozero", row=1, col=1)
    if two:
        fig.update_yaxes(title_text="% of keys", rangemode="tozero", row=2, col=1)
        fig.update_xaxes(title_text="runs with a value", type="category", row=2, col=1)
    fig.update_xaxes(type="category", row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=12)
        ann.x = 0.0
        ann.xanchor = "left"
    return themed(
        fig,
        scheme,
        height=height,
        barmode="group",
        bargap=0.3,
        margin=dict(l=56, r=12, t=58, b=40 if two else 30),
        legend=dict(y=1.12),
    )


def cv_figure(
    results: Mapping[str, CvResult | None],
    colours: Mapping[str, str],
    scheme: str = "light",
    *,
    height: int = 330,
) -> tuple[go.Figure, dict[str, int]]:
    """CV per condition as violins, one panel per level, and the CVs above the axis.

    Returns the figure and, per level, the number of CVs above the plotted range.
    """
    levels = [lvl for lvl in ("precursor", "protein") if results.get(lvl) is not None]
    above: dict[str, int] = {}
    groups = [c for lvl in levels for c in results[lvl].conditions if c.n_cv]  # type: ignore[union-attr]
    if not groups:
        return empty_figure("no CV: no condition has enough values", scheme, height=height), above
    every = np.concatenate([c.cv for c in groups])
    top = float(max(40.0, min(150.0, math.ceil(np.percentile(every, 98) / 10.0) * 10.0)))
    edges = np.linspace(0.0, top, 61)
    fig = make_subplots(
        rows=1,
        cols=len(levels),
        horizontal_spacing=0.1,
        subplot_titles=[LEVEL_TITLES[lvl] for lvl in levels],
    )
    seen: set[str] = set()
    for col, level in enumerate(levels, start=1):
        res = results[level]
        assert res is not None
        names = [c.condition for c in res.conditions]
        above[level] = 0
        for i, c in enumerate(res.conditions):
            colour = colours.get(c.condition, theme.SERIES[i % len(theme.SERIES)])
            if not c.n_cv:
                continue
            counts = np.histogram(np.clip(c.cv, 0, top), bins=edges)[0]
            above[level] += int((c.cv > top).sum())
            q25, q50, q75 = np.percentile(c.cv, [25, 50, 75])
            noun = LEVEL_NOUNS[level]
            text = (
                f"<b>condition {c.condition}</b> ({', '.join(run_label(r) for r in c.runs)})"
                f"<br>{c.n_cv:,} {noun} with a CV (each needs {c.need} values)"
                f"<br>median CV {q50:.1f}%<br>25th to 75th percentile {q25:.1f}% to {q75:.1f}%"
                f"<br>{c.share_below(20):.0f}% of the CVs are at most 20%"
            )
            _violin(
                fig,
                i,
                counts,
                edges,
                colour,
                text,
                col=col,
                name=f"condition {c.condition}",
                legend=c.condition not in seen,
            )
            seen.add(c.condition)
            _box(fig, i, q25, q50, q75, colour, col)
            fig.add_annotation(
                x=i,
                y=q50,
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                text=f"<b>{q50:.1f}%</b>",
                showarrow=False,
                xanchor="left",
                xshift=9,
                font=dict(size=11),
            )
        fig.update_xaxes(
            tickvals=list(range(len(names))),
            ticktext=names,
            range=[-0.6, len(names) - 0.4],
            showgrid=False,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            range=[0, top],
            ticksuffix="%",
            title_text="CV" if col == 1 else None,
            row=1,
            col=col,
        )
    for ann in fig.layout.annotations:
        if ann.text in LEVEL_TITLES.values():
            ann.font = dict(size=12)
            ann.x = ann.x - 0.5 / len(levels) + 0.005
            ann.xanchor = "left"
            ann.yshift = 6
    fig = themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=56, r=12, t=58, b=32),
        legend=dict(y=1.13),
    )
    return fig, above


def heatmap_figure(
    h: Heatmap,
    colours: Mapping[str, str],
    scheme: str = "light",
    *,
    relative: bool = True,
    selected: str | None = None,
    height: int = 520,
) -> go.Figure:
    """The LFQ matrix: one row per key, one column per run, missing cells in grey."""
    if not len(h.groups):
        return empty_figure("no protein group with an LFQ value", scheme, height=height)
    z = h.relative if relative else h.log10
    runs = [run_label(r) for r in h.runs]
    if relative:
        zmin, zmax = -RELATIVE_RANGE, RELATIVE_RANGE
        scale, title = diverging(scheme), "log2 to<br>row mean"
        hover = "log2 to the row mean %{z:.2f}"
    else:
        finite = z[np.isfinite(z)]
        zmin = float(np.floor(finite.min())) if finite.size else 0.0
        zmax = float(np.ceil(finite.max())) if finite.size else 1.0
        scale, title = "Viridis", "log10<br>MaxLFQ"
        hover = "log10 MaxLFQ %{z:.2f}"
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z.astype("float32"),
            x=runs,
            y=h.groups,
            zmin=zmin,
            zmax=zmax,
            colorscale=scale,
            colorbar=dict(title=dict(text=title, side="top"), thickness=10, len=0.6, y=0.42),
            hovertemplate="<b>%{y}</b><br>%{x}<br>" + hover + "<extra></extra>",
            hoverongaps=False,
            meta="qq-relative" if relative else "qq-log10",
        )
    )
    # Condition bars above the columns.
    spans: list[tuple[str, int, int]] = []
    for j, cond in enumerate(h.conditions):
        if spans and spans[-1][0] == cond and spans[-1][2] == j - 1:
            spans[-1] = (cond, spans[-1][1], j)
        else:
            spans.append((cond, j, j))
    for cond, a, b in spans:
        colour = colours.get(cond, theme.PREDICTION)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=a - 0.48,
            x1=b + 0.48,
            y0=1.012,
            y1=1.032,
            fillcolor=colour,
            line=dict(width=0),
        )
        fig.add_annotation(
            x=(a + b) / 2.0,
            y=1.035,
            xref="x",
            yref="paper",
            yanchor="bottom",
            text=cond or "no condition",
            showarrow=False,
            font=dict(size=11, color=colour),
        )
    if selected is not None:
        hits = np.flatnonzero(h.groups == selected)
        if hits.size:
            i = int(hits[0])
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y",
                x0=-0.012,
                x1=1.0,
                y0=i - 0.5 - max(0.5, len(h.groups) / 300.0),
                y1=i + 0.5 + max(0.5, len(h.groups) / 300.0),
                line=dict(color=theme.TARGET, width=1.5),
                fillcolor="rgba(0,0,0,0)",
                name="selected",
            )
    fig.update_yaxes(
        autorange="reversed", showticklabels=False, showgrid=False, ticks="", title_text=None
    )
    fig.update_xaxes(side="bottom", showgrid=False, type="category")
    # A missing cell is not drawn: the plot area behind it is grey (one trace, so the
    # names of the rows are sent once).
    return themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=12, r=8, t=40, b=30),
        hovermode="closest",
        dragmode="zoom",
        plot_bgcolor=MISSING_CELL,
    )


def profile_figure(
    p: ProteinProfile,
    conditions: Mapping[str, str],
    colours: Mapping[str, str],
    scheme: str = "light",
    *,
    max_precursors: int = 60,
    height: int = 300,
) -> tuple[go.Figure, int]:
    """One protein group across the runs (log axis), grouped by condition.

    The group's MaxLFQ (line), its protein_group_quant quantity (dashed) and the MaxLFQ
    of its precursors (thin lines, the ``max_precursors`` with the highest mean). A cell
    with match-between-runs transfers among its features is ringed. Returns the figure
    and the number of precursors drawn.
    """
    groups = condition_groups(conditions, p.runs)
    order = [r for runs in groups.values() for r in runs]
    order += [r for r in p.runs if r not in order]
    col = {r: j for j, r in enumerate(p.runs)}
    idx = [col[r] for r in order]
    x = [run_label(r) for r in order]
    fig = go.Figure()
    # Condition bands behind the points.
    start = 0
    for cond, runs in groups.items():
        colour = colours.get(cond, theme.PREDICTION)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=start - 0.45,
            x1=start + len(runs) - 0.55,
            y0=0,
            y1=1,
            fillcolor=_rgba(colour, 0.07),
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            x=start + (len(runs) - 1) / 2.0,
            y=1.0,
            xref="x",
            yref="paper",
            yanchor="bottom",
            text=f"condition {cond}",
            showarrow=False,
            font=dict(size=11, color=colour),
        )
        start += len(runs)
    shown = 0
    prec = p.precursor_lfq if p.precursor_lfq is not None else p.precursor_quant
    prec_name = "precursor MaxLFQ" if p.precursor_lfq is not None else "precursor quantity"
    if prec is not None and len(prec):
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.nanmean(np.where(np.isfinite(prec), np.log10(prec), np.nan), axis=1)
        means = np.nan_to_num(means, nan=-np.inf)
        keep = np.argsort(-means, kind="stable")[:max_precursors]
        for n, i in enumerate(keep):
            y = prec[i, idx]
            if not np.isfinite(y).any():
                continue
            shown += 1
            label = f"{p.precursors['peptidoform'].iloc[i]} {int(p.precursors['charge'].iloc[i])}+"
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(color=theme.PREDICTION, width=1),
                    marker=dict(size=4),
                    opacity=0.35,
                    name=prec_name,
                    legendgroup="prec",
                    showlegend=n == 0,
                    hovertemplate=label + "<br>%{x}: %{y:.4s}<extra>" + prec_name + "</extra>",
                )
            )
    quant = p.quant.set_index("run")["quantity"].reindex(list(p.runs)).to_numpy(dtype=float)
    if np.isfinite(quant).any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=quant[idx],
                mode="lines+markers",
                line=dict(color=LEVEL_COLOURS["protein"], width=2, dash="dash"),
                marker=dict(size=8, symbol="circle-open", line=dict(width=2)),
                name="protein_group_quant",
                hovertemplate="%{x}: %{y:.4s}<extra>protein_group_quant (not normalized)</extra>",
            )
        )
    if p.lfq is not None:
        y = p.lfq[idx]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color=theme.TARGET, width=3),
                marker=dict(size=10, color=theme.TARGET),
                name="MaxLFQ",
                hovertemplate="%{x}: %{y:.4s}<extra>MaxLFQ (median-ratio normalized)</extra>",
            )
        )
        if p.lfq_transferred is not None:
            moved = p.lfq_transferred[idx]
            sel = np.flatnonzero((moved > 0) & np.isfinite(y))
            if sel.size:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i] for i in sel],
                        y=y[sel],
                        mode="markers",
                        marker=dict(
                            size=20,
                            symbol="circle-open",
                            color="#82c91e",
                            line=dict(width=2.5, color="#82c91e"),
                        ),
                        name="with MBR transfers",
                        customdata=moved[sel],
                        hovertemplate="%{x}: %{customdata} transferred precursor(s) among the "
                        "features (match-between-runs)<extra></extra>",
                    )
                )
    values = [v for v in (p.lfq, quant, prec) if v is not None]
    finite = np.concatenate([np.ravel(v)[np.isfinite(np.ravel(v))] for v in values])
    if not finite.size:
        return empty_figure("no quantity in any run", scheme, height=height), shown
    lo, hi = math.log10(float(finite.min())), math.log10(float(finite.max()))
    fig.update_yaxes(
        type="log",
        range=[lo - 0.25, hi + 0.25],
        title_text="quantity (log scale)",
        exponentformat="power",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=x, showgrid=False)
    fig = themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=60, r=12, t=56, b=30),
        legend=dict(y=1.16),
        hovermode="closest",
    )
    return fig, shown
