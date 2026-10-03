"""Plotly figures of the run QC page (pure functions of data-layer results).

The RT tracks share one RT axis (one figure with four rows), so zooming one track zooms
all of them. The tracks: the TIC and the base peak of the chosen MS level (the viewer's
sum and maximum of each scan's intensities), the accepted identifications per RT bin
and the MS2 scan rate. Trace order is fixed (:data:`TRACE`); the page's callbacks patch
traces by these indexes. Retention times are seconds.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mumdia_viewer.data.qc import IdDistributions, PeakCounts, WindowScheme

from . import theme
from .figures import empty_figure, themed

# Trace indexes of the RT figure.
TRACE = {"tic": 0, "bp": 1, "targets": 2, "decoys": 3, "rate": 4}
LEVEL_COLOURS = {1: theme.MS1[0], 2: theme.TARGET}
FILLS = {1: "rgba(16, 152, 173, 0.16)", 2: "rgba(66, 99, 235, 0.13)"}
ACCEPTED = theme.ACCEPT
RATE = theme.PREDICTION
SELECT_BAND = "rgba(66, 99, 235, 0.10)"
SCAN_LINE = "#f08c00"
CAP = "#e03131"
RT_HEIGHT = 560
UIREVISION = "qc-rt"


@dataclass(frozen=True)
class SignalTrace:
    """The scans drawn in the TIC and base-peak tracks.

    ``rows`` are rows of the spectra table, in RT order; ``rt``, ``tic``, ``base_peak``,
    ``base_peak_mz`` and ``scan_index`` are their values. ``n_total`` is the number of
    scans in the view (one window or all), ``envelope`` True when ``rows`` is the M4
    selection of a larger view.
    """

    level: int
    rows: np.ndarray
    rt: np.ndarray
    tic: np.ndarray
    base_peak: np.ndarray
    base_peak_mz: np.ndarray
    scan_index: np.ndarray
    n_total: int
    envelope: bool
    window: str | None = None


def plain(values: Any) -> list[Any]:
    """Customdata as nested lists with None for a non-finite value.

    A NumPy array goes to Plotly as a binary typed array, and dcc.Graph leaves typed
    arrays out of its click and hover data, so a click could not say which scan it hit.
    """
    a = np.asarray(values, dtype=np.float64)
    out = a.astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def _level_name(level: int) -> str:
    return "MS1" if level == 1 else "MS2"


def signal_customdata(s: SignalTrace, *, with_mz: bool = False) -> list[list[Any]]:
    """``[row, scan_index]`` (and the base peak m/z) per drawn point, as plain lists.

    The click data of a point names its row this way: dcc.Graph leaves typed arrays out
    of its event data. Integers and a rounded m/z keep the payload small.
    """
    rows = s.rows.astype(np.int64).tolist()
    scans = s.scan_index.astype(np.int64).tolist()
    if not with_mz:
        return [[r, k] for r, k in zip(rows, scans, strict=True)]
    mz = [round(v, 4) if math.isfinite(v) else None for v in s.base_peak_mz.astype(float).tolist()]
    return [[r, k, m] for r, k, m in zip(rows, scans, mz, strict=True)]


def typed(values: Any) -> dict[str, str]:
    """A float32 array in Plotly's typed-array form (for a Patch; figures do it themselves)."""
    data = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return {"dtype": "f4", "bdata": base64.b64encode(data.tobytes()).decode("ascii")}


def _signal_traces(tic_s: SignalTrace, bp_s: SignalTrace) -> tuple[go.Scattergl, go.Scattergl]:
    """The TIC and base-peak traces. Each has its own rows: an envelope keeps the extremes
    of its own series."""
    name = _level_name(tic_s.level)
    where = f" ({tic_s.window})" if tic_s.window else ""
    colour = LEVEL_COLOURS.get(tic_s.level, theme.TARGET)
    tic = go.Scattergl(
        x=tic_s.rt.astype(np.float32),
        y=tic_s.tic.astype(np.float32),
        mode="lines",
        line=dict(color=colour, width=1.2),
        fill="tozeroy",
        fillcolor=FILLS.get(tic_s.level, FILLS[2]),
        customdata=signal_customdata(tic_s),
        name=f"{name} TIC",
        hovertemplate=(
            "RT %{x:.2f} s<br>TIC %{y:.4s}<br>scan_index %{customdata[1]:.0f}"
            f"<extra>{name} TIC{where}</extra>"
        ),
        showlegend=False,
    )
    bp = go.Scattergl(
        x=bp_s.rt.astype(np.float32),
        y=bp_s.base_peak.astype(np.float32),
        mode="lines",
        line=dict(color=colour, width=1.0),
        customdata=signal_customdata(bp_s, with_mz=True),
        name=f"{name} base peak",
        hovertemplate=(
            "RT %{x:.2f} s<br>base peak %{y:.4s} at m/z %{customdata[2]:.4f}"
            f"<br>scan_index %{{customdata[1]:.0f}}<extra>{name} base peak{where}</extra>"
        ),
        showlegend=False,
    )
    return tic, bp


def bin_customdata(ids: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [ids["bin_lo"].to_numpy(), ids["bin_hi"].to_numpy(), ids["decoys"].to_numpy()]
    ).astype(np.float64)


def bin_opacity(n: int, selected: int | None) -> list[float]:
    return [1.0 if selected is None or i == selected else 0.55 for i in range(n)]


def _id_traces(ids: pd.DataFrame, t_label: str, selected: int | None) -> tuple[go.Bar, go.Bar]:
    centres = ((ids["bin_lo"] + ids["bin_hi"]) / 2.0).to_numpy()
    widths = (ids["bin_hi"] - ids["bin_lo"]).to_numpy()
    targets = go.Bar(
        x=centres,
        y=ids["targets"].to_numpy(),
        width=widths,
        marker=dict(color=ACCEPTED, opacity=bin_opacity(len(ids), selected), line=dict(width=0)),
        customdata=plain(bin_customdata(ids)),
        name="accepted target PSMs",
        hovertemplate=(
            "apex RT %{customdata[0]:.0f} to %{customdata[1]:.0f} s<br>"
            f"%{{y:,}} accepted target PSMs ({t_label})<br>"
            "decoy PSMs passing the same cut: %{customdata[2]:,}<extra>click: list them</extra>"
        ),
        showlegend=False,
    )
    decoys = go.Bar(
        x=centres,
        y=ids["decoys"].to_numpy(),
        width=widths,
        marker=dict(color=theme.DECOY, line=dict(width=0)),
        name="decoys passing",
        hovertemplate=(
            f"decoy PSMs passing {t_label}: %{{y:,}}<extra>decoys (a diagnostic)</extra>"
        ),
        showlegend=False,
    )
    return targets, decoys


def _rate_trace(rate: pd.DataFrame) -> go.Scatter:
    centres = ((rate["bin_lo"] + rate["bin_hi"]) / 2.0).to_numpy()
    custom = np.column_stack(
        [
            rate["bin_lo"].to_numpy(),
            rate["bin_hi"].to_numpy(),
            rate["cycle_s"].to_numpy(),
            rate["ms2_scans"].to_numpy(),
            rate["ms1_per_s"].to_numpy(),
        ]
    ).astype(np.float64)
    return go.Scatter(
        x=centres,
        y=rate["ms2_per_s"].to_numpy(),
        mode="lines",
        line=dict(color=RATE, width=1.6, shape="hvh"),
        fill="tozeroy",
        fillcolor="rgba(134, 142, 150, 0.12)",
        customdata=plain(custom),
        name="MS2 scans per second",
        hovertemplate=(
            "RT %{customdata[0]:.0f} to %{customdata[1]:.0f} s<br>"
            "%{y:.1f} MS2 scans/s (%{customdata[3]:,} scans)<br>"
            "cycle time %{customdata[2]:.3f} s (median step of one window)<br>"
            "%{customdata[4]:.2f} MS1 scans/s<extra>scan rate</extra>"
        ),
        showlegend=False,
    )


def selection_shapes(
    scan_rt: float | None, band: tuple[float, float] | None
) -> list[dict[str, Any]]:
    """The selected scan (a line across the tracks) and the selected RT bin (a band)."""
    shapes: list[dict[str, Any]] = []
    if band is not None:
        shapes.append(
            dict(
                type="rect",
                name="qc-bin",
                xref="x",
                yref="paper",
                x0=float(band[0]),
                x1=float(band[1]),
                y0=0,
                y1=1,
                fillcolor=SELECT_BAND,
                line=dict(width=0),
                layer="below",
            )
        )
    if scan_rt is not None and np.isfinite(scan_rt):
        shapes.append(
            dict(
                type="line",
                name="qc-scan",
                xref="x",
                yref="paper",
                x0=float(scan_rt),
                x1=float(scan_rt),
                y0=0,
                y1=1,
                line=dict(color=SCAN_LINE, width=1.4, dash="dot"),
            )
        )
    return shapes


def rt_figure(
    tic_s: SignalTrace | None,
    bp_s: SignalTrace | None,
    ids: pd.DataFrame | None,
    rate: pd.DataFrame | None,
    *,
    t_label: str,
    bin_label: str,
    scheme: str = "light",
    scan_rt: float | None = None,
    band: tuple[float, float] | None = None,
    selected_bin: int | None = None,
    x_range: Sequence[float] | None = None,
    message: str | None = None,
) -> go.Figure:
    """The four RT tracks on one shared RT axis (see the module docstring)."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.31, 0.23, 0.27, 0.19],
    )
    level = tic_s.level if tic_s is not None else 1
    if tic_s is not None and bp_s is not None:
        tic, bp = _signal_traces(tic_s, bp_s)
    else:
        tic = go.Scattergl(x=[], y=[], showlegend=False, name="TIC")
        bp = go.Scattergl(x=[], y=[], showlegend=False, name="base peak")
    fig.add_trace(tic, row=1, col=1)
    fig.add_trace(bp, row=2, col=1)
    if ids is not None and len(ids):
        targets, decoys = _id_traces(ids, t_label, selected_bin)
    else:
        targets = go.Bar(x=[], y=[], showlegend=False, name="accepted target PSMs")
        decoys = go.Bar(x=[], y=[], showlegend=False, name="decoys passing")
    fig.add_trace(targets, row=3, col=1)
    fig.add_trace(decoys, row=3, col=1)
    fig.add_trace(
        _rate_trace(rate)
        if rate is not None and len(rate)
        else go.Scatter(x=[], y=[], showlegend=False, name="MS2 scans per second"),
        row=4,
        col=1,
    )
    name = _level_name(level)
    axis = dict(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    for row in range(1, 5):
        fig.update_xaxes(row=row, col=1, **axis)
    fig.update_xaxes(row=4, col=1, title_text="retention time (s)")
    fig.update_yaxes(
        row=1, col=1, title_text=f"{name} TIC", exponentformat="SI", rangemode="tozero"
    )
    fig.update_yaxes(
        row=2, col=1, title_text=f"{name} base peak", exponentformat="SI", rangemode="tozero"
    )
    fig.update_yaxes(row=3, col=1, title_text=bin_label, rangemode="tozero")
    fig.update_yaxes(row=4, col=1, title_text="MS2 scans/s", rangemode="tozero")
    layout: dict[str, Any] = dict(
        height=RT_HEIGHT,
        barmode="overlay",
        bargap=0,
        hovermode="x",
        margin=dict(l=64, r=14, t=10, b=44),
        uirevision=UIREVISION,
        shapes=selection_shapes(scan_rt, band),
    )
    if x_range is not None:
        layout["xaxis"] = dict(range=[float(x_range[0]), float(x_range[1])], autorange=False)
    if message:
        layout["annotations"] = [
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.86,
                showarrow=False,
                font=dict(size=12),
            )
        ]
    themed(fig, scheme, **layout)
    for ax in ("yaxis", "yaxis2", "yaxis3", "yaxis4"):
        fig.layout[ax].title.font.size = 11
        fig.layout[ax].title.standoff = 6
    return fig


# --------------------------------------------------------------------------- spectrum


def spectrum_figure(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    level: int,
    scheme: str = "light",
    window: tuple[float, float] | None = None,
    base_peak: tuple[float, float] | None = None,
) -> go.Figure:
    """One scan as a stick plot (the peaks as stored by convert), the base peak marked.

    For an MS2 scan the isolation window is drawn as a band.
    """
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return empty_figure("the scan has no peaks", scheme, height=250)
    xs = np.empty(mz.size * 3, dtype=np.float32)
    ys = np.empty(mz.size * 3, dtype=np.float32)
    xs[0::3], xs[1::3], xs[2::3] = mz, mz, np.nan
    ys[0::3], ys[1::3], ys[2::3] = 0.0, intensity, np.nan
    colour = LEVEL_COLOURS.get(level, theme.TARGET)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=colour, width=1),
            hoverinfo="skip",
            showlegend=False,
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=mz.astype(np.float32),
            y=intensity.astype(np.float32),
            mode="markers",
            marker=dict(size=4, color=colour, opacity=0),
            hovertemplate="m/z %{x:.4f}<br>intensity %{y:.4s}<extra></extra>",
            showlegend=False,
        )
    )
    shapes = []
    if window is not None:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=float(window[0]),
                x1=float(window[1]),
                y0=0,
                y1=1,
                fillcolor="rgba(240, 140, 0, 0.18)",
                line=dict(width=0),
                layer="below",
            )
        )
    annotations = []
    if base_peak is not None and np.isfinite(base_peak[0]):
        annotations.append(
            dict(
                x=float(base_peak[0]),
                y=float(base_peak[1]),
                text=f"{base_peak[0]:.4f}",
                showarrow=False,
                yshift=9,
                font=dict(size=10),
            )
        )
    lo, hi = float(mz.min()), float(mz.max())
    pad = max(1.0, 0.02 * (hi - lo))
    return themed(
        fig,
        scheme,
        height=250,
        margin=dict(l=52, r=12, t=14, b=40),
        hovermode="closest",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="m/z", range=[lo - pad, hi + pad]),
        yaxis=dict(title="intensity", exponentformat="SI", rangemode="tozero"),
    )


# --------------------------------------------------------------------------- windows


def windows_figure(
    ws: WindowScheme, scheme: str = "light", *, selected: int | None = None
) -> go.Figure:
    """The isolation windows as m/z ranges, one row per window in acquisition order.

    Each window is a thick segment from its lower to its upper bound; an invisible marker
    at its centre carries the hover and the click.
    """
    df = ws.frame
    if df is None or not len(df):
        return empty_figure("no isolation windows", scheme)
    n = len(df)
    width = 7 if n <= 40 else 4 if n <= 120 else 2.5

    def segments(part: pd.DataFrame) -> tuple[list[float | None], list[float | None]]:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for lo, hi, y in zip(part["lower"], part["upper"], part["order"], strict=True):
            xs += [float(lo), float(hi), None]
            ys += [float(y), float(y), None]
        return xs, ys

    fig = go.Figure()
    xs, ys = segments(df)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=theme.TARGET, width=width),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if selected is not None:
        part = df[df["window_id"] == selected]
        if len(part):
            xs, ys = segments(part)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=theme.DECOY, width=width + 2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            row = part.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[(row["lower"] + row["upper"]) / 2.0],
                    y=[row["order"]],
                    mode="markers+text",
                    marker=dict(size=11, color=theme.DECOY, line=dict(width=0)),
                    text=[f"window {int(row['window_id'])}"],
                    textposition="middle right",
                    textfont=dict(size=11, color=theme.DECOY),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    custom = np.column_stack(
        [
            df["window_id"].to_numpy(),
            df["lower"].to_numpy(),
            df["upper"].to_numpy(),
            df["n_scans"].to_numpy(),
            df["cycle_time_s"].to_numpy(),
            df["overlap_with_next"].to_numpy(),
        ]
    ).astype(np.float64)
    fig.add_trace(
        go.Scatter(
            x=((df["lower"] + df["upper"]) / 2.0).to_numpy(),
            y=df["order"].to_numpy(),
            mode="markers",
            marker=dict(size=max(8, width * 2), color=theme.TARGET, opacity=0),
            customdata=plain(custom),
            hovertemplate=(
                "window %{customdata[0]:.0f}: %{customdata[1]:.3f} to %{customdata[2]:.3f} "
                "m/z<br>position %{y} in the acquisition order<br>%{customdata[3]:,} scans, "
                "cycle %{customdata[4]:.3f} s<br>gap (-) or overlap (+) with the next window "
                "%{customdata[5]:.4f} Th<extra>click: its MS2 TIC</extra>"
            ),
            showlegend=False,
        )
    )
    return themed(
        fig,
        scheme,
        height=330,
        margin=dict(l=56, r=12, t=10, b=44),
        hovermode="y",
        xaxis=dict(title="isolation window (m/z)"),
        yaxis=dict(title="acquisition order", autorange="reversed", tickformat="d"),
    )


# --------------------------------------------------------------------------- peaks


def peaks_figure(pc: PeakCounts, scheme: str = "light") -> go.Figure:
    """Peaks per MS2 spectrum: a histogram with the percentiles and the conversion cap."""
    h = pc.histogram
    if h is None or not len(h):
        return empty_figure("no spectra", scheme, height=230)
    centres = ((h["bin_lo"] + h["bin_hi"]) / 2.0).to_numpy()
    fig = go.Figure(
        go.Bar(
            x=centres,
            y=h["n"].to_numpy(),
            width=(h["bin_hi"] - h["bin_lo"]).to_numpy(),
            marker=dict(color=theme.TARGET, opacity=0.75, line=dict(width=0)),
            customdata=plain(np.column_stack([h["bin_lo"], h["bin_hi"]])),
            hovertemplate=(
                "%{customdata[0]:,.0f} to %{customdata[1]:,.0f} peaks: %{y:,} spectra"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    shapes, annotations = [], []
    for name in ("p25", "p50", "p95"):
        v = pc.percentiles.get(name)
        if v is None:
            continue
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=v,
                x1=v,
                y0=0,
                y1=1,
                line=dict(color=theme.PREDICTION, width=1, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=v,
                y=1,
                xref="x",
                yref="paper",
                text="median" if name == "p50" else name,
                showarrow=False,
                yanchor="bottom",
                font=dict(size=10),
            )
        )
    if pc.cap:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=pc.cap,
                x1=pc.cap,
                y0=0,
                y1=1,
                line=dict(color=CAP, width=1.6, dash="dash"),
            )
        )
        annotations.append(
            dict(
                x=pc.cap,
                y=0.82,
                xref="x",
                yref="paper",
                text=f"cap {pc.cap}",
                showarrow=False,
                xanchor="left",
                xshift=4,
                font=dict(size=10, color=CAP),
            )
        )
    return themed(
        fig,
        scheme,
        height=230,
        margin=dict(l=56, r=12, t=22, b=42),
        bargap=0,
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="peaks per MS2 spectrum", rangemode="tozero"),
        yaxis=dict(title="spectra", rangemode="tozero"),
    )


# --------------------------------------------------------------------------- distributions


def count_bars(
    df: pd.DataFrame,
    column: str,
    *,
    scheme: str = "light",
    xtitle: str,
    noun: str,
    total: int,
    tick_suffix: str = "",
    height: int = 210,
) -> go.Figure:
    """Counts of accepted PSMs per value of ``column`` (charge, length, missed cleavages)."""
    if df is None or not len(df) or total == 0:
        return empty_figure("no accepted identification", scheme, height=height)
    x = df[column].astype(float).to_numpy()
    y = df["n"].to_numpy()
    share = 100.0 * y / max(1, total)
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker=dict(color=ACCEPTED, opacity=0.85, line=dict(width=0)),
            customdata=plain(share),
            hovertemplate=(
                f"{noun} %{{x}}{tick_suffix}: %{{y:,}} PSMs (%{{customdata:.1f}}%)<extra></extra>"
            ),
            showlegend=False,
        )
    )
    few = len(df) <= 8
    return themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=52, r=10, t=10, b=42),
        bargap=0.25 if few else 0.08,
        xaxis=dict(
            title=xtitle,
            tickmode="array" if few else "auto",
            tickvals=list(x) if few else None,
            ticktext=[f"{int(v)}{tick_suffix}" for v in x] if few else None,
        ),
        yaxis=dict(title="PSMs", rangemode="tozero"),
    )


def mods_rows(d: IdDistributions) -> list[Mapping[str, Any]]:
    """The modification table rows: name, short tag, colour, site and the counts."""
    out = []
    for r in d.mods.to_dict("records"):
        short, colour = theme.mod_style(str(r["tag"]))
        out.append(
            {
                "name": theme.mod_name(str(r["tag"])),
                "tag": str(r["tag"]),
                "short": short,
                "colour": colour,
                "site": str(r["site"]),
                "psms": int(r["psms"]),
                "sites": int(r["sites"]),
                "with_site": int(r["with_site"]),
            }
        )
    return out
