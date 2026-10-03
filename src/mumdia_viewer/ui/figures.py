"""Plotly figures built from data-layer results.

Every function here is a pure function of data-layer objects (no Dash, no I/O), so the
figures can be tested and exported without a browser. Retention times are seconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

from . import theme

# --------------------------------------------------------------------------- overview


def themed(fig: go.Figure, scheme: str = "light", **layout) -> go.Figure:
    """Apply the viewer's Plotly template for ``scheme`` and the given layout."""
    fig.update_layout(template=theme.template(scheme), **layout)
    return fig


def empty_figure(message: str, scheme: str = "light", *, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=13)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return themed(fig, scheme, height=height, margin=dict(l=10, r=10, t=10, b=10))


def id_curves_figure(
    curves: Mapping[str, pd.DataFrame],
    threshold: float,
    labels: Mapping[str, str],
    scheme: str = "light",
    *,
    marker_label: str | None = None,
) -> go.Figure:
    """Identifications against the q threshold (log axis), one line per unit.

    Each curve holds exact counts at its thresholds (data.counts.counts_at). The
    threshold marker is a shape and an annotation named "threshold", which the
    client moves while the threshold slider is dragged.
    """
    fig = go.Figure()
    for unit, df in curves.items():
        if df is None or df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["q"],
                y=df["count"],
                mode="lines",
                name=labels.get(unit, unit),
                # PSMs dashed: in a run competed on (peptidoform, charge) the PSM and
                # precursor curves coincide, and the dash keeps both visible.
                line=dict(
                    color=theme.UNIT_COLOURS.get(unit),
                    width=2.4,
                    dash="dash" if unit == "psm" else "solid",
                ),
                hovertemplate="%{y:,}<extra>" + labels.get(unit, unit) + "</extra>",
            )
        )
    fig.add_shape(
        type="line",
        name="threshold",
        x0=threshold,
        x1=threshold,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=theme.PREDICTION, width=1.5, dash="dot"),
    )
    fig.add_annotation(
        name="threshold",
        x=math.log10(threshold),
        xref="x",
        y=1,
        yref="paper",
        yanchor="top",
        xanchor="left",
        xshift=6,
        showarrow=False,
        text=marker_label or f"q ≤ {threshold:g}",
        font=dict(size=11),
    )
    themed(
        fig,
        scheme,
        height=330,
        hovermode="x unified",
        xaxis=dict(
            type="log",
            title="q threshold (each unit on its own q column)",
            exponentformat="power",
            showspikes=False,
        ),
        yaxis=dict(title="target identifications", rangemode="tozero"),
    )
    return fig


def score_histogram_figure(df: pd.DataFrame, scheme: str = "light") -> go.Figure:
    """Rescorer score distribution by label (overlaid bars)."""
    if df is None or df.empty:
        return empty_figure("no scored rows", scheme)
    centres = (df["bin_lo"] + df["bin_hi"]) / 2.0
    width = float((df["bin_hi"] - df["bin_lo"]).median())
    fig = go.Figure()
    for column, colour, name in (
        ("target", theme.TARGET, "targets"),
        ("decoy", theme.DECOY, "decoys"),
        ("spike_in", theme.SPIKE, "entrapment spike-ins"),
    ):
        if column in df:
            fig.add_trace(
                go.Bar(
                    x=centres,
                    y=df[column],
                    width=width,
                    name=name,
                    marker=dict(color=colour, line=dict(width=0)),
                    opacity=0.62,
                    hovertemplate="score %{x:.3g}<br>%{y:,} rows<extra>" + name + "</extra>",
                )
            )
    return themed(
        fig,
        scheme,
        height=330,
        barmode="overlay",
        bargap=0,
        xaxis_title="rescorer score (higher is better)",
        yaxis_title="scored rows",
    )


def stage_timings_figure(df: pd.DataFrame, scheme: str = "light") -> go.Figure:
    """Wall time of each recorded stage; one stacked colour per directory."""
    timed = df[df["elapsed_s"].notna()] if df is not None and not df.empty else df
    if timed is None or timed.empty:
        return empty_figure("no stage report records a time", scheme)
    order = timed.groupby("stage")["elapsed_s"].sum().sort_values().index.tolist()
    directories = list(dict.fromkeys(timed["directory"]))
    fig = go.Figure()
    many = len(directories) > 1
    for i, directory in enumerate(directories):
        part = timed[timed["directory"] == directory].set_index("stage")
        values = [
            float(part["elapsed_s"].get(stage, 0.0)) if stage in part.index else None
            for stage in order
        ]
        artifacts = [
            str(part["artifacts"].get(stage, "")) if stage in part.index else "" for stage in order
        ]
        name = "experiment directory" if directory == "." else f"run {directory}"
        fig.add_trace(
            go.Bar(
                x=values,
                y=order,
                orientation="h",
                name=name,
                marker=dict(
                    color=theme.TARGET if not many else theme.SERIES[i % len(theme.SERIES)],
                    line=dict(width=0),
                ),
                customdata=artifacts,
                hovertemplate="%{y}: %{x:.2f} s<br>writes %{customdata}<extra>" + name + "</extra>",
                showlegend=many,
            )
        )
    return themed(
        fig,
        scheme,
        height=max(220, 36 + 30 * len(order)),
        barmode="stack",
        bargap=0.35,
        xaxis_title="wall time (s), from report.json elapsed_ms",
        yaxis=dict(automargin=True),
        margin=dict(l=10, r=16, t=34 if many else 12, b=44),
    )


PER_RUN_NAMES = {
    "target_psms": "target PSMs",
    "precursors": "precursors (derived)",
    "peptides": "peptides (derived)",
    "protein_groups": "protein groups (derived)",
}


def per_run_figure(df: pd.DataFrame, labels: Mapping[str, str], scheme: str = "light") -> go.Figure:
    """Per-run counts (run_psm_q) as grouped bars; the hover gives each full label."""
    if df is None or df.empty:
        return empty_figure("no runs", scheme)
    fig = go.Figure()
    for column, unit in (
        ("target_psms", "psm"),
        ("precursors", "precursor"),
        ("peptides", "peptide"),
        ("protein_groups", "protein_group"),
    ):
        if column not in df:
            continue
        fig.add_trace(
            go.Bar(
                x=df["run"].astype(str),
                y=df[column],
                name=PER_RUN_NAMES[column],
                marker=dict(color=theme.UNIT_COLOURS[unit], line=dict(width=0)),
                hovertemplate="%{x}: %{y:,}<br>" + labels.get(column, column) + "<extra></extra>",
            )
        )
    return themed(
        fig,
        scheme,
        height=320,
        barmode="group",
        bargap=0.25,
        xaxis_title="run",
        yaxis_title="count",
    )
