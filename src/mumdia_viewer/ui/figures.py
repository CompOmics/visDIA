"""Plotly figures built from data-layer results.

Every function here is a pure function of data-layer objects (no Dash, no I/O), so the
figures can be tested and exported without a browser. Retention times are seconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mumdia_viewer.data.detail import MirrorData, PrecursorDetail

from . import theme

ION_COLOURS = {"b": "#1f77b4", "y": "#d62728"}
OTHER_COLOUR = "#7f7f7f"
TARGET_COLOUR = "#1f77b4"
DECOY_COLOUR = "#d62728"
SPIKE_COLOUR = "#ff7f0e"

# The legend sits below the x axis, so it never covers the title or the traces.
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=20, t=60, b=110),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    hoverlabel=dict(namelength=-1),
)


def _empty(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, **_LAYOUT)
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


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


# --------------------------------------------------------------------------- XIC panels


def _vline(fig: go.Figure, xs, *, colour: str, dash: str, label: str) -> None:
    """Vertical marker line(s) at ``xs`` (a value or a tuple) with one legend entry."""
    values = xs if isinstance(xs, tuple) else (xs,)
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values:
        return
    for x in values:
        fig.add_vline(x=x, line_color=colour, line_dash=dash, line_width=1.5)
    # An invisible trace gives the lines one legend entry.
    fig.add_trace(
        go.Scatter(
            x=[values[0]],
            y=[None],
            mode="lines",
            name=label,
            line=dict(color=colour, dash=dash),
            hoverinfo="skip",
        )
    )


def _bounds(d: PrecursorDetail) -> tuple[float | None, float | None]:
    lo = d.markers.get("rt_lo")
    hi = d.markers.get("rt_hi")
    return lo, hi


def xic_figure(d: PrecursorDetail, *, include_ms1: bool = False) -> go.Figure:
    """Fragment XICs of one candidate with the identification and quant markers."""
    chrom = d.chromatogram
    title = "Fragment XICs"
    if chrom is None:
        return _empty(title, "no chromatogram rows for this candidate")
    fig = go.Figure()
    traces = chrom.traces if include_ms1 else chrom.fragments()
    shown = 0
    for t in traces:
        if not t.observed:
            continue
        shown += 1
        fig.add_trace(
            go.Scatter(
                x=t.rt.astype(float),
                y=t.intensity.astype(float),
                mode="lines+markers",
                name=t.frag_name,
                line=dict(width=1.5),
                marker=dict(size=4),
                legendgroup=t.frag_name,
                hovertemplate=(
                    f"{t.frag_name}<br>m/z {t.frag_mz:.4f}<br>predicted "
                    f"{t.predicted_intensity:.3g}<br>RT %{{x:.2f}} s<br>intensity "
                    "%{y:.4g}<extra></extra>"
                ),
            )
        )
    if shown == 0:
        return _empty(title, "no fragment of this candidate was observed")
    m = d.markers
    lo, hi = m.get("elution_lo"), m.get("elution_hi")
    if lo is not None and hi is not None:
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor="#2ca02c",
            opacity=0.12,
            line_width=0,
            annotation_text="elution bounds",
            annotation_position="top left",
        )
    _vline(fig, m.get("apex_rt"), colour="#2ca02c", dash="solid", label="identification apex")
    _vline(
        fig,
        (m.get("integration_lo_rt"), m.get("integration_hi_rt")),
        colour="#9467bd",
        dash="dash",
        label="quant integration bounds",
    )
    _vline(
        fig, m.get("rt_pred_cal"), colour="#8c564b", dash="dot", label="calibrated RT prediction"
    )
    _vline(fig, _bounds(d), colour="#bcbd22", dash="longdash", label="extraction window")
    alternatives = [p for p in m.get("peaks", []) if not p["selected"] and p["apex_rt"]]
    if alternatives:
        ymax = max(float(t.intensity.max()) for t in traces if t.observed)
        fig.add_trace(
            go.Scatter(
                x=[p["apex_rt"] for p in alternatives],
                y=[ymax * 1.05] * len(alternatives),
                mode="markers+text",
                text=[f"peak {p['peak_rank']}" for p in alternatives],
                textposition="top center",
                marker=dict(symbol="triangle-down", size=11, color="#ff7f0e"),
                name="alternative peaks (not selected)",
                hovertemplate="alternative peak %{text}<br>apex %{x:.2f} s<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, xaxis_title="retention time (s)", yaxis_title="intensity", **_LAYOUT
    )
    # Keep the view on the extraction window when it is finite and not absurdly wide.
    axis = chrom.common_axis()
    if axis is not None and axis.size:
        fig.update_xaxes(range=[float(axis.min()) - 1.0, float(axis.max()) + 1.0])
    return fig


def ms1_figure(d: PrecursorDetail) -> go.Figure:
    """The MS1 isotope XICs (summed MS1 peaks at mono, +1, +2 in the nearest MS1 scan)."""
    chrom = d.chromatogram
    title = "MS1 isotope XICs"
    if chrom is None or not chrom.ms1():
        return _empty(title, "no MS1 isotope rows")
    fig = go.Figure()
    colours = {"ms1_mono": "#17becf", "ms1_iso1": "#1f77b4", "ms1_iso2": "#9edae5"}
    for t in chrom.ms1():
        if not t.observed:
            continue
        fig.add_trace(
            go.Scatter(
                x=t.rt.astype(float),
                y=t.intensity.astype(float),
                mode="lines+markers",
                name=t.frag_name,
                line=dict(color=colours.get(t.frag_name), width=1.5),
                marker=dict(size=4),
                hovertemplate=(
                    f"{t.frag_name}<br>m/z {t.frag_mz:.4f}<br>RT %{{x:.2f}} s<br>"
                    "intensity %{y:.4g}<extra></extra>"
                ),
            )
        )
    m = d.markers
    _vline(fig, m.get("apex_rt"), colour="#2ca02c", dash="solid", label="identification apex")
    fig.update_layout(
        title=title,
        xaxis_title="retention time (s)",
        yaxis_title="intensity",
        height=280,
        **_LAYOUT,
    )
    axis = chrom.common_axis()
    if axis is not None and axis.size:
        fig.update_xaxes(range=[float(axis.min()) - 1.0, float(axis.max()) + 1.0])
    return fig


# --------------------------------------------------------------------------- mirror


def mirror_figure(m: MirrorData | None, *, title: str = "Spectrum mirror") -> go.Figure:
    """Observed MS2 peaks (up) against the predicted fragments (down, scaled to -100).

    Matched peaks are coloured by ion type and annotated with the fragment name and the
    ppm error; the match uses the extraction's tolerance and mass offset.
    """
    if m is None:
        return _empty(title, "no spectrum")
    spec = m.spectrum
    mz = spec.mz.astype(float)
    inten = spec.intensity.astype(float)
    top = float(inten.max()) if inten.size else 1.0
    rel = 100.0 * inten / top if top > 0 else inten
    fig = go.Figure()
    xs, ys = _stems(mz, rel)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#bbbbbb", width=1),
            name="observed peaks",
            hoverinfo="skip",
        )
    )
    frags = m.fragments
    pred = frags["predicted_intensity"].to_numpy(dtype=float)
    pmax = float(pred.max()) if pred.size else 1.0
    prel = -100.0 * pred / pmax if pmax > 0 else pred
    for ion in ("b", "y", None):
        sel = (frags["ion"] == ion) if ion is not None else ~frags["ion"].isin(["b", "y"])
        if not sel.any():
            continue
        px, py = _stems(frags.loc[sel, "theo_mz"].to_numpy(float), prel[sel.to_numpy()])
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines",
                line=dict(color=ION_COLOURS.get(ion or "", OTHER_COLOUR), width=2),
                name=f"predicted {ion or 'other'} fragments",
                hoverinfo="skip",
            )
        )
    if m.matches:
        names = frags["name"].tolist()
        ions = frags["ion"].tolist()
        mx = [mt.obs_mz for mt in m.matches]
        my = [100.0 * mt.obs_intensity / top if top > 0 else 0.0 for mt in m.matches]
        sx, sy = _stems(np.array(mx), np.array(my))
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                line=dict(color="#2ca02c", width=2),
                name="matched peaks",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mx,
                y=my,
                mode="markers+text",
                text=[names[mt.fragment] for mt in m.matches],
                textposition="top center",
                marker=dict(
                    size=6,
                    color=[
                        ION_COLOURS.get(ions[mt.fragment] or "", OTHER_COLOUR) for mt in m.matches
                    ],
                ),
                customdata=[[mt.ppm_raw, mt.ppm_corrected, mt.theo_mz] for mt in m.matches],
                hovertemplate=(
                    "%{text}<br>observed m/z %{x:.4f}<br>predicted m/z %{customdata[2]:.4f}"
                    "<br>ppm (raw) %{customdata[0]:.2f}<br>ppm (offset-corrected, "
                    "viewer-derived) %{customdata[1]:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line_color="#444", line_width=1)
    pick = m.pick
    subtitle = (
        f"scan {pick.scan_index} at {pick.rt:.2f} s, window "
        f"{pick.window_lower:.2f}-{pick.window_upper:.2f}; "
        f"{len(m.matches)} of {len(frags)} predicted fragments matched within "
        f"{m.tolerance.tol_ppm:.2f} ppm (offset {m.tolerance.offset_ppm:+.2f} ppm)"
    )
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>"),
        xaxis_title="m/z",
        yaxis_title="relative intensity (%)",
        yaxis_range=[-110, 125],
        height=420,
        **_LAYOUT,
    )
    return fig


def _stems(x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    """Line segments from 0 to each y, as one trace (None separates the stems)."""
    xs: list = []
    ys: list = []
    for xi, yi in zip(np.asarray(x, float), np.asarray(y, float), strict=False):
        xs += [xi, xi, None]
        ys += [0.0, yi, None]
    return xs, ys


def percentile_figure(rows: Sequence) -> go.Figure:
    """Feature percentiles of the candidate among valid targets and decoys of its run."""
    rows = [r for r in rows if r.pct_target is not None or r.pct_decoy is not None]
    if not rows:
        return _empty("Feature percentiles", "no ranked features")
    names = [r.feature for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=[r.pct_target for r in rows],
            orientation="h",
            name="percentile among targets",
            marker_color=TARGET_COLOUR,
        )
    )
    fig.add_trace(
        go.Bar(
            y=names,
            x=[r.pct_decoy for r in rows],
            orientation="h",
            name="percentile among decoys",
            marker_color=DECOY_COLOUR,
        )
    )
    fig.update_layout(
        title="Feature percentiles (valid rows of this run, not FDR-filtered)",
        barmode="group",
        xaxis_title="share of rows with a value <= this candidate's (%)",
        xaxis_range=[0, 100],
        height=40 + 26 * len(rows),
        **_LAYOUT,
    )
    fig.update_yaxes(autorange="reversed")
    return fig
