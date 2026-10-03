"""Figures of the calibration page: the RT fit and its residuals, the RT error of the
accepted identifications, the fragment mass offset and the fragment mass errors.

Every function is a pure function of :mod:`mumdia_viewer.data.calibration` results (no
Dash, no I/O) and finishes with :func:`figures.themed`. Retention times are seconds.

Scatter points carry their row in the data-layer frame as ``customdata``; the page's
callbacks look the row up for the point bar (hover) and the precursor link (click).
Lines and shapes carry a ``name`` so the page's script and the tests can find them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mumdia_viewer.data.calibration import (
    AcceptedErrors,
    Anchors,
    CalRecord,
    MassCalRecord,
    binned_quantiles,
)

from . import figures, theme

# A curve drawn with at most this many points (an even sample of the engine's rows).
CURVE_POINTS = 1200
# Points of one scatter are drawn with WebGL above this count.
GL_AT = 2000
# The running quantile lines of the accepted rows: bins across the gradient.
RT_BINS = 48
WINDOW_FILL = "rgba(134, 142, 150, 0.16)"
WINDOW_LINE = "rgba(134, 142, 150, 0.55)"
CURVE = "#5f3dc4"  # violet 9: the engine's fitted map
INSIDE = theme.TARGET
OUTSIDE = theme.WARN
QUANTILE = "#1c7ed6"  # blue 7: the viewer's running quantiles
QUANTILE_LINE = "#f59f00"  # yellow 7: drawn over the indigo points, in both themes
MASS = "#0c8599"  # cyan 8: the fragment mass calibration
TOL_FILL = "rgba(12, 133, 153, 0.12)"
HEIGHT = 450
FIT_HEIGHT = 480
SMALL = 300
MASS_HEIGHT = 360

ERROR_VIEWS = ("seconds", "fraction")
MASS_VIEWS = ("distribution", "gradient")


def _f32(values: Any) -> np.ndarray:
    """Values as float32 (Plotly sends numpy arrays as typed arrays; RTs and iRTs need no
    more than float32 for a display)."""
    return np.asarray(values, dtype=np.float64).astype(np.float32)


def _scatter(n: int, **kwargs: Any) -> go.Scatter | go.Scattergl:
    return go.Scattergl(**kwargs) if n > GL_AT else go.Scatter(**kwargs)


def _pad(lo: float, hi: float, frac: float = 0.03) -> list[float]:
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return [lo - frac * span, hi + frac * span]


def _even(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    """At most ``n`` rows of ``df`` (sorted by ``column``), evenly spaced in ``column``:
    for each of ``n`` equally spaced values, the row nearest to it. Every row kept is a
    row of the input, so a line through them passes through the engine's values."""
    if len(df) <= n:
        return df
    x = df[column].to_numpy(dtype=np.float64)
    targets = np.linspace(x[0], x[-1], n)
    idx = np.unique(np.clip(np.searchsorted(x, targets), 0, len(x) - 1))
    return df.iloc[idx]


def _vline(
    name: str, x: float, colour: str, dash: str, label: str | None = None, **kw: Any
) -> dict:
    shape: dict[str, Any] = dict(
        type="line",
        name=name,
        x0=x,
        x1=x,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=colour, width=kw.pop("width", 1.4), dash=dash),
        layer="above",
    )
    if label:
        shape["label"] = dict(
            text=label,
            textposition=kw.pop("position", "end"),
            textangle=0,
            font=dict(size=10, color=colour),
            xanchor=kw.pop("anchor", "left"),
            yanchor="top",
            padding=3,
        )
    return shape


def _hline(
    name: str, y: float, colour: str, dash: str, label: str | None = None, **kw: Any
) -> dict:
    shape: dict[str, Any] = dict(
        type="line",
        name=name,
        y0=y,
        y1=y,
        yref="y",
        x0=0,
        x1=1,
        xref="paper",
        line=dict(color=colour, width=kw.pop("width", 1.3), dash=dash),
        layer="above",
    )
    if label:
        shape["label"] = dict(
            text=label,
            textposition=kw.pop("position", "end"),
            font=dict(size=10, color=colour),
            yanchor=kw.pop("yanchor", "bottom"),
            padding=2,
        )
    return shape


# --------------------------------------------------------------------------- RT fit


def anchor_range(a: Anchors) -> tuple[float, float] | None:
    """The anchors' library iRT range (None without anchors)."""
    x = a.frame["irt"].to_numpy(dtype=np.float64) if a.n else np.zeros(0)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(x.min()), float(x.max())


def central(a: Anchors, q: float = 0.005) -> pd.DataFrame:
    """The curve rows inside the central range of the sampled library iRT (the 0.5th to
    the 99.5th percentile) or the anchors' range, whichever is wider. A library can hold
    a few rows far outside the gradient (an imported iRT of -23,000 on the Astral HYE
    library); the drawn curve leaves them out."""
    c = a.curve
    if len(c) == 0:
        return c
    x = c["irt"].to_numpy(dtype=np.float64)
    lo, hi = float(np.quantile(x, q)), float(np.quantile(x, 1 - q))
    rng = anchor_range(a)
    if rng is not None:
        lo, hi = min(lo, rng[0]), max(hi, rng[1])
    return c[(x >= lo) & (x <= hi)]


def residual_limit(a: Anchors) -> float:
    """The half-range of the residual panel: 1.6 half-windows, or 1.1 times the 99th
    percentile of |residual| when the windows are unbounded. Anchors beyond it are drawn
    at the panel's edge."""
    hw = ((a.frame["rt_hi"] - a.frame["rt_lo"]) / 2.0).to_numpy(dtype=np.float64)
    hw = hw[np.isfinite(hw)]
    if hw.size:
        return float(1.6 * np.max(hw))
    res = np.abs(a.frame["residual"].to_numpy(dtype=np.float64))
    res = res[np.isfinite(res)]
    return float(np.quantile(res, 0.99)) * 1.1 if res.size else 1.0


def fit_figure(a: Anchors, scheme: str = "light") -> go.Figure:
    """The anchors with the engine's fitted map and RT window, and their residuals.

    Top: x the library ``predicted_irt`` the fit read, y the seed ``observed_rt``; the line
    is ``rt_pred_cal`` of ``run_windows`` at the anchors and at evenly spaced library rows
    (solid over the anchors' iRT range, dashed beyond it, where the engine continues the
    curve linearly, ``calibrate.rs``), the band ``[rt_lo, rt_hi]``. Bottom, on the same x:
    the in-sample residual ``observed_rt - rt_pred_cal`` with the band of the half-window;
    residuals beyond the panel (:func:`residual_limit`) are drawn at its edge as
    triangles. Anchors outside their window are in the warning colour. Every anchor
    point's ``customdata`` is its row in ``a.frame``.
    """
    if a.error or a.n == 0:
        text = a.error or "No anchors: the calibration had no confident target seed PSM."
        return figures.empty_figure(text, scheme, height=FIT_HEIGHT)
    df = a.frame
    rows = np.arange(len(df), dtype=np.int32)
    inside = df["in_window"].to_numpy(dtype=bool)
    x_all = df["irt"].to_numpy(dtype=np.float64)
    y_all = df["observed_rt"].to_numpy(dtype=np.float64)
    res = df["residual"].to_numpy(dtype=np.float64)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.66, 0.34], vertical_spacing=0.05
    )
    curve = _even(central(a), "irt", CURVE_POINTS)
    lo_a, hi_a = anchor_range(a) or (float("nan"), float("nan"))
    band = dict(color=WINDOW_LINE, width=1, dash="dashdot")
    if len(curve):
        cx = curve["irt"].to_numpy(dtype=np.float64)
        pred = curve["rt_pred_cal"].to_numpy(dtype=np.float64)
        lo_c = curve["rt_lo"].to_numpy(dtype=np.float64)
        hi_c = curve["rt_hi"].to_numpy(dtype=np.float64)
        for row, lo, hi, name in (
            (1, lo_c, hi_c, "RT window [rt_lo, rt_hi]"),
            (2, lo_c - pred, hi_c - pred, None),
        ):
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.sum() < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_f32(cx[ok]),
                    y=_f32(lo[ok]),
                    mode="lines",
                    line=band,
                    hoverinfo="skip",
                    showlegend=False,
                    name="window low",
                    legendgroup="window",
                    meta={"kind": "window"},
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=_f32(cx[ok]),
                    y=_f32(hi[ok]),
                    mode="lines",
                    line=band,
                    fill="tonexty",
                    fillcolor=WINDOW_FILL,
                    hoverinfo="skip",
                    name=name or "window",
                    showlegend=name is not None,
                    legendgroup="window",
                    meta={"kind": "window"},
                ),
                row=row,
                col=1,
            )
        inner = (cx >= lo_a) & (cx <= hi_a)
        for part, dash, name, legend in (
            (inner, "solid", "fitted map (rt_pred_cal)", True),
            (cx <= lo_a, "dash", "beyond the anchors", True),
            (cx >= hi_a, "dash", "beyond the anchors", False),
        ):
            if part.sum() < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_f32(cx[part]),
                    y=_f32(pred[part]),
                    mode="lines",
                    line=dict(color=CURVE, width=2.2, dash=dash),
                    name=name,
                    showlegend=legend,
                    legendgroup="extrapolated" if dash == "dash" else "curve",
                    meta={"kind": "curve"},
                    # The anchors' point bar gives rt_pred_cal; the line must not take
                    # their hover.
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
    limit = residual_limit(a)
    clipped = np.isfinite(res) & (np.abs(res) > limit)
    for mask, colour, name, size, opacity in (
        (inside, INSIDE, "anchors in their window", 4, 0.5),
        (~inside, OUTSIDE, "anchors outside their window", 5, 0.85),
    ):
        if not mask.any():
            continue
        marker = dict(color=colour, size=size, opacity=opacity, line=dict(width=0))
        fig.add_trace(
            _scatter(
                int(mask.sum()),
                x=_f32(x_all[mask]),
                y=_f32(y_all[mask]),
                mode="markers",
                marker=marker,
                customdata=rows[mask],
                name=name,
                legendgroup=name,
                meta={"kind": "anchors"},
                hovertemplate="iRT %{x:.2f}<br>observed %{y:.1f} s<extra>anchor</extra>",
            ),
            row=1,
            col=1,
        )
        keep = mask & ~clipped
        if keep.any():
            fig.add_trace(
                _scatter(
                    int(keep.sum()),
                    x=_f32(x_all[keep]),
                    y=_f32(res[keep]),
                    mode="markers",
                    marker=marker,
                    customdata=rows[keep],
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    meta={"kind": "anchors"},
                    hovertemplate="iRT %{x:.2f}<br>residual %{y:+.2f} s<extra>anchor</extra>",
                ),
                row=2,
                col=1,
            )
        cut = mask & clipped
        if cut.any():
            fig.add_trace(
                go.Scatter(
                    x=_f32(x_all[cut]),
                    y=_f32(np.sign(res[cut]) * limit),
                    mode="markers",
                    marker=dict(
                        color=colour,
                        size=7,
                        opacity=0.9,
                        line=dict(width=0),
                        symbol=np.where(res[cut] > 0, "triangle-up", "triangle-down").tolist(),
                    ),
                    customdata=rows[cut],
                    text=[f"{v:+.1f}" for v in res[cut]],
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    meta={"kind": "anchors", "clipped": True},
                    hovertemplate="iRT %{x:.2f}<br>residual %{text} s (beyond the panel, "
                    "drawn at its edge)<extra>anchor</extra>",
                ),
                row=2,
                col=1,
            )
    fig.add_shape(
        type="line",
        name="zero",
        xref="x2 domain",
        x0=0,
        x1=1,
        yref="y2",
        y0=0,
        y1=0,
        line=dict(color=theme.PREDICTION, width=1, dash="dot"),
    )
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x_rng = _pad(float(x_all[finite].min()), float(x_all[finite].max())) if finite.any() else None
    y_rng = _pad(float(y_all[finite].min()), float(y_all[finite].max())) if finite.any() else None
    y2_rng = [-1.06 * limit, 1.06 * limit]
    look = _even(central(a), "irt", 300)
    meta: dict[str, Any] = {
        "rt_axis": "y",
        "home": {"xaxis": x_rng, "xaxis2": x_rng, "yaxis": y_rng, "yaxis2": y2_rng},
        "curve": {
            "x": [round(float(v), 3) for v in look["irt"]],
            "y": [round(float(v), 2) for v in look["rt_pred_cal"]],
        },
        "clipped": int(clipped.sum()),
        "limit": limit,
    }
    fig = figures.themed(
        fig,
        scheme,
        height=FIT_HEIGHT,
        hovermode="closest",
        uirevision="cal-fit",
        showlegend=False,
        margin=dict(l=62, r=16, t=22, b=46),
        meta=meta,
    )
    fig.update_xaxes(range=x_rng, showspikes=False)
    fig.update_xaxes(title_text="library predicted_irt (the iRT the fit read)", row=2, col=1)
    fig.update_yaxes(showspikes=False, zeroline=False)
    fig.update_yaxes(title_text="observed RT (s)", range=y_rng, row=1, col=1)
    fig.update_yaxes(title_text="residual (s)", range=y2_rng, row=2, col=1)
    return fig


def anchor_traces(a: Anchors) -> dict[int, np.ndarray]:
    """For every anchor trace of :func:`fit_figure`: its number and the anchor rows of its
    points, in order. Dash's ``hoverData`` of a WebGL trace has no ``customdata``, so the
    page maps ``(curveNumber, pointIndex)`` back to a row with this."""
    fig = fit_figure(a)
    out = {}
    for i, trace in enumerate(fig.data):
        if (trace.meta or {}).get("kind") == "anchors" and trace.customdata is not None:
            out[i] = np.asarray(trace.customdata, dtype=np.int64)
    return out


def residual_figure(a: Anchors, cal: CalRecord | None, scheme: str = "light") -> go.Figure:
    """The histogram of the anchors' in-sample residuals ``observed_rt - rt_pred_cal``.

    Lines: 0, the median, the half-window ``±w_rt`` and the residual percentile behind it
    (``±w_rt / multiplier``). The range is about ``±1.3 w_rt``; anchors beyond it are
    counted in an annotation.
    """
    if a.error or a.n < 2:
        text = a.error or "Fewer than two anchors: no residuals."
        return figures.empty_figure(text, scheme, height=SMALL - 40)
    res = a.frame["residual"].to_numpy(dtype=np.float64)
    res = res[np.isfinite(res)]
    w = cal.w_rt if cal is not None and cal.w_rt is not None else None
    half = 1.3 * w if w else float(np.quantile(np.abs(res), 0.995)) * 1.1 or 1.0
    edges = np.linspace(-half, half, 81)
    counts, _ = np.histogram(np.clip(res, -half, half), bins=edges)
    beyond = int(np.sum(np.abs(res) > half))
    mids = (edges[:-1] + edges[1:]) / 2.0
    fig = go.Figure(
        go.Bar(
            x=_f32(mids),
            y=counts,
            width=float(edges[1] - edges[0]),
            marker=dict(color=INSIDE, opacity=0.72, line=dict(width=0)),
            name="anchors",
            hovertemplate="%{x:+.2f} s: %{y:,} anchors<extra></extra>",
        )
    )
    med = float(np.quantile(res, 0.5)) if res.size else 0.0
    shapes = [_vline("zero", 0.0, theme.PREDICTION, "dot", width=1)]
    if cal is not None and cal.residual_median_s is not None:
        med = cal.residual_median_s
    shapes.append(_vline("median", med, CURVE, "dash"))
    if w:
        shapes.append(_vline("window-lo", -w, WINDOW_LINE, "dashdot", width=1.6))
        shapes.append(_vline("window-hi", w, WINDOW_LINE, "dashdot", width=1.6))
        p = cal.p_rt_width if cal is not None else None
        if p and abs(p - w) > 1e-9:
            shapes.append(_vline("p-lo", -p, theme.PREDICTION, "dot", width=1))
            shapes.append(_vline("p-hi", p, theme.PREDICTION, "dot", width=1))
    for s in shapes:
        fig.add_shape(**s)
    if beyond:
        fig.add_annotation(
            name="beyond",
            text=f"{beyond:,} beyond ±{half:.0f} s, in the end bins",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=10, color=theme.PREDICTION),
        )
    return figures.themed(
        fig,
        scheme,
        height=SMALL - 40,
        bargap=0.04,
        showlegend=False,
        margin=dict(l=52, r=12, t=26, b=42),
        xaxis=dict(title="observed - rt_pred_cal (s)", range=[-half, half], zeroline=False),
        yaxis=dict(title="anchors", rangemode="tozero"),
    )


# --------------------------------------------------------------------------- RT error


def id_rows(e: AcceptedErrors, y_column: str) -> np.ndarray:
    """The rows of ``e.frame`` drawn in a scatter of ``y_column`` against ``apex_rt``.

    The scatter is the figure's first trace and carries no ``customdata``: point ``k`` of
    it is row ``id_rows(e, column)[k]`` (the page's hover and click callbacks map a point
    back with this function).
    """
    if not len(e.frame) or y_column not in e.frame:
        return np.zeros(0, dtype=np.int64)
    x = e.frame["apex_rt"].to_numpy(dtype=np.float64)
    y = e.frame[y_column].to_numpy(dtype=np.float64)
    return np.flatnonzero(np.isfinite(x) & np.isfinite(y))


def error_column(view: str) -> str:
    """The frame column the RT error figure draws in a view."""
    return "rt_error_rel" if view == "fraction" else "rt_error"


def error_figure(e: AcceptedErrors, scheme: str = "light", *, view: str = "seconds") -> go.Figure:
    """The RT error of the accepted identifications across the gradient.

    x ``apex_rt``; y the feature ``rt_error_signed`` (``apex_rt - rt_pred_cal``, seconds)
    or, with ``view="fraction"``, that error over the candidate's half-window
    (viewer-derived). Lines: 0, the window edges, and the 5th, 50th and 95th percentiles
    of the error in bins of apex RT (viewer-derived, numpy). The points are the first
    trace, in the rows of :func:`id_rows`.
    """
    if e.error:
        return figures.empty_figure(f"Not available: {e.error}", scheme, height=HEIGHT)
    df = e.frame
    column = error_column(view)
    y = df[column].to_numpy(dtype=np.float64) if len(df) else np.zeros(0)
    x = df["apex_rt"].to_numpy(dtype=np.float64) if len(df) else np.zeros(0)
    ok = np.zeros(len(df), dtype=bool)
    ok[id_rows(e, column)] = True
    if not ok.any():
        why = (
            "No accepted identification at this threshold."
            if not len(df)
            else "No accepted row has a valid RT error (no RT calibration: the engine writes 0)."
        )
        return figures.empty_figure(why, scheme, height=HEIGHT)
    fig = go.Figure()
    unit = "of the half-window" if view == "fraction" else "s"
    fmt = ".3f" if view == "fraction" else "+.2f"
    fig.add_trace(
        _scatter(
            int(ok.sum()),
            x=_f32(x[ok]),
            y=_f32(y[ok]),
            mode="markers",
            marker=dict(
                color=INSIDE, size=3 if ok.sum() > 20_000 else 4, opacity=0.28, line=dict(width=0)
            ),
            name="accepted identifications",
            meta={"kind": "ids"},
            hovertemplate=f"apex %{{x:.1f}} s<br>RT error %{{y:{fmt}}} {unit}"
            "<extra>accepted</extra>",
        )
    )
    bins = binned_quantiles(x[ok], y[ok], bins=RT_BINS, min_count=20)
    if len(bins):
        for col, name, dash, width in (
            ("q05", "5th and 95th percentile", "dot", 1.4),
            ("q50", "median in bins of apex RT", "solid", 2.4),
            ("q95", "95th percentile", "dot", 1.4),
        ):
            fig.add_trace(
                # WebGL, after the points: an SVG line would lie under the WebGL points.
                go.Scattergl(
                    x=_f32(bins["x_mid"]),
                    y=_f32(bins[col]),
                    mode="lines",
                    line=dict(color=QUANTILE_LINE, width=width, dash=dash),
                    name=name,
                    showlegend=col != "q95",
                    legendgroup="q" if col != "q50" else "median",
                    meta={"kind": "quantile", "q": col},
                    hovertemplate=f"apex %{{x:.0f}} s<br>{col} %{{y:{fmt}}} {unit}"
                    "<extra>viewer-derived</extra>",
                )
            )
    hw = df["half_width"].to_numpy(dtype=np.float64)
    hw = hw[np.isfinite(hw)]
    shapes = [_hline("zero", 0.0, theme.PREDICTION, "dot", width=1)]
    edge = None
    if view == "fraction":
        edge = 1.0
    elif hw.size and float(np.min(hw)) == float(np.max(hw)):
        edge = float(hw[0])
    if edge is not None:
        shapes.append(_hline("edge-hi", edge, WINDOW_LINE, "dashdot"))
        shapes.append(_hline("edge-lo", -edge, WINDOW_LINE, "dashdot"))
    for s in shapes:
        fig.add_shape(**s)
    top = float(np.max(np.abs(y[ok])))
    if edge is not None:
        top = max(top, edge)
    x_rng = _pad(float(x[ok].min()), float(x[ok].max()))
    y_rng = [-1.08 * top, 1.08 * top]
    y_title = (
        "rt_error_signed / half-window (viewer-derived)"
        if view == "fraction"
        else "rt_error_signed: apex - rt_pred_cal (s)"
    )
    return figures.themed(
        fig,
        scheme,
        height=HEIGHT,
        hovermode="closest",
        uirevision="cal-error",
        showlegend=False,
        margin=dict(l=60, r=16, t=22, b=46),
        xaxis=dict(title="apex_rt of the identification (s)", range=x_rng, showspikes=False),
        yaxis=dict(title=y_title, range=y_rng, zeroline=False, showspikes=False),
        meta={"view": view, "home": {"xaxis": x_rng, "yaxis": y_rng}},
    )


# --------------------------------------------------------------------------- mass


def offset_figure(
    m: MassCalRecord | None,
    mz_range: tuple[float, float, str] | None,
    scheme: str = "light",
) -> go.Figure:
    """The fragment mass offset extract applied, against m/z, with the tolerance band.

    With an m/z grid (two or more nodes): the grid nodes as recorded, joined by straight
    lines (extract interpolates linearly between nodes and keeps the end values outside
    them), the band ``offset(m/z) ± frag_tol_ppm``. Without one: the constant
    ``frag_ppm_offset`` over the fragment m/z range, the band ``± frag_tol_ppm``.
    """
    if m is None or m.frag_ppm_offset is None:
        return figures.empty_figure("No mass calibration record.", scheme, height=SMALL - 40)
    lo, hi = (mz_range[0], mz_range[1]) if mz_range else (150.0, 2000.0)
    tol = m.frag_tol_ppm or 0.0
    if m.uses_grid:
        gx, gy = m.grid_mz, m.grid_ppm
        x = np.concatenate([[min(lo, gx[0])], gx, [max(hi, gx[-1])]])
        y = np.concatenate([[gy[0]], gy, [gy[-1]]])
        name = "m/z-dependent offset (mz_cal_grid)"
    else:
        x = np.array([lo, hi])
        y = np.array([m.frag_ppm_offset, m.frag_ppm_offset])
        name = "constant offset (frag_ppm_offset)"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y - tol,
            mode="lines",
            line=dict(color=MASS, width=1, dash="dashdot"),
            hoverinfo="skip",
            showlegend=False,
            name="tolerance low",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y + tol,
            mode="lines",
            line=dict(color=MASS, width=1, dash="dashdot"),
            fill="tonexty",
            fillcolor=TOL_FILL,
            name=f"± frag_tol_ppm {tol:.2f}",
            hovertemplate="m/z %{x:.0f}<br>upper edge %{y:+.2f} ppm<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers" if m.uses_grid else "lines",
            line=dict(color=MASS, width=2.6),
            marker=dict(size=5, color=MASS),
            name=name,
            hovertemplate="m/z %{x:.1f}<br>offset %{y:+.3f} ppm<extra></extra>",
        )
    )
    fig.add_shape(**_hline("zero", 0.0, theme.PREDICTION, "dot", width=1))
    span = max(tol * 1.6, abs(m.frag_ppm_offset) + tol * 1.3, 2.0)
    centre = float(np.mean(y))
    return figures.themed(
        fig,
        scheme,
        height=SMALL - 40,
        hovermode="closest",
        margin=dict(l=56, r=14, t=30, b=44),
        xaxis=dict(title="fragment m/z", range=_pad(float(x.min()), float(x.max()), 0.01)),
        yaxis=dict(
            title="ppm (observed vs library)", range=[centre - span, centre + span], zeroline=False
        ),
    )


def mass_figure(
    e: AcceptedErrors,
    m: MassCalRecord | None,
    scheme: str = "light",
    *,
    column: str = "frag_mass_err_median",
    view: str = "distribution",
) -> go.Figure:
    """The fragment mass error of the accepted identifications (raw ppm, one value per row).

    ``view="distribution"``: a histogram, with the run's ``frag_ppm_offset`` and the
    tolerance edges ``offset ± frag_tol_ppm`` (the values are uncorrected, so they centre
    on the offset). ``view="gradient"``: the values against ``apex_rt`` with the offset,
    the tolerance band and the running percentiles (viewer-derived).
    """
    if e.error:
        return figures.empty_figure(f"Not available: {e.error}", scheme, height=MASS_HEIGHT)
    v = e.frame[column].to_numpy(dtype=np.float64) if len(e.frame) else np.zeros(0)
    ok = np.isfinite(v)
    if not ok.any():
        return figures.empty_figure(
            "No accepted row has a valid value.", scheme, height=MASS_HEIGHT
        )
    off = m.frag_ppm_offset if m is not None else None
    tol = m.frag_tol_ppm if m is not None else None
    fig = go.Figure()
    if view == "gradient":
        x = e.frame["apex_rt"].to_numpy(dtype=np.float64)
        good = np.zeros(len(e.frame), dtype=bool)
        good[id_rows(e, column)] = True
        # The points first (trace 0, the rows of id_rows), then the tolerance band.
        fig.add_trace(
            _scatter(
                int(good.sum()),
                x=_f32(x[good]),
                y=_f32(v[good]),
                mode="markers",
                marker=dict(
                    color=INSIDE,
                    size=3 if good.sum() > 20_000 else 4,
                    opacity=0.25,
                    line=dict(width=0),
                ),
                name="accepted identifications",
                meta={"kind": "ids"},
                hovertemplate="apex %{x:.1f} s<br>%{y:+.2f} ppm<extra>accepted</extra>",
            )
        )
        if off is not None and tol is not None:
            xs = [float(x[good].min()), float(x[good].max())]
            line = dict(color=MASS, width=1, dash="dashdot")
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[off - tol] * 2,
                    mode="lines",
                    line=line,
                    hoverinfo="skip",
                    showlegend=False,
                    name="tol low",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[off + tol] * 2,
                    mode="lines",
                    line=line,
                    hoverinfo="skip",
                    name="offset ± frag_tol_ppm",
                )
            )
        bins = binned_quantiles(x[good], v[good], bins=RT_BINS, min_count=20)
        for col, name, dash, width in (
            ("q05", "5th and 95th percentile", "dot", 1.4),
            ("q50", "median in bins of apex RT", "solid", 2.4),
            ("q95", "95th percentile", "dot", 1.4),
        ):
            if len(bins):
                fig.add_trace(
                    go.Scattergl(
                        x=_f32(bins["x_mid"]),
                        y=_f32(bins[col]),
                        mode="lines",
                        line=dict(color=QUANTILE_LINE, width=width, dash=dash),
                        name=name,
                        showlegend=col != "q95",
                        legendgroup="q" if col != "q50" else "median",
                        meta={"kind": "quantile", "q": col},
                        hovertemplate=f"apex %{{x:.0f}} s<br>{col} %{{y:+.2f}} ppm"
                        "<extra>viewer-derived</extra>",
                    )
                )
        if off is not None:
            fig.add_shape(**_hline("offset", off, MASS, "dash", width=1.6))
        top = float(np.quantile(np.abs(v[good] - (off or 0.0)), 0.999)) if good.any() else 1.0
        if tol:
            top = max(top, tol * 1.15)
        centre = off or 0.0
        x_rng = _pad(float(x[good].min()), float(x[good].max()))
        y_rng = [centre - top, centre + top]
        return figures.themed(
            fig,
            scheme,
            height=MASS_HEIGHT,
            hovermode="closest",
            uirevision="cal-mass-gradient",
            showlegend=False,
            margin=dict(l=56, r=14, t=22, b=44),
            xaxis=dict(title="apex_rt of the identification (s)", range=x_rng, showspikes=False),
            yaxis=dict(title=f"{column} (raw ppm)", range=y_rng, zeroline=False, showspikes=False),
            meta={"view": view, "home": {"xaxis": x_rng, "yaxis": y_rng}},
        )
    vals = v[ok]
    centre = off if off is not None else float(np.median(vals))
    half = max(float(np.quantile(np.abs(vals - centre), 0.995)), (tol or 0.0) * 1.15, 1.0)
    edges = np.linspace(centre - half, centre + half, 91)
    counts, _ = np.histogram(np.clip(vals, edges[0], edges[-1]), bins=edges)
    fig.add_trace(
        go.Bar(
            x=_f32((edges[:-1] + edges[1:]) / 2.0),
            y=counts,
            width=float(edges[1] - edges[0]),
            marker=dict(color=INSIDE, opacity=0.72, line=dict(width=0)),
            name="accepted identifications",
            hovertemplate="%{x:+.2f} ppm: %{y:,} rows<extra></extra>",
        )
    )
    shapes = [_vline("zero", 0.0, theme.PREDICTION, "dot", width=1)]
    if off is not None:
        shapes.append(_vline("offset", off, MASS, "dash", width=1.6))
        if tol:
            shapes.append(_vline("tol-lo", off - tol, MASS, "dashdot"))
            shapes.append(_vline("tol-hi", off + tol, MASS, "dashdot"))
    for s in shapes:
        fig.add_shape(**s)
    beyond = int(np.sum(np.abs(vals - centre) > half))
    if beyond:
        fig.add_annotation(
            name="beyond",
            text=f"{beyond:,} beyond ±{half:.1f} ppm of the offset (in the end bins)",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=10, color=theme.PREDICTION),
        )
    return figures.themed(
        fig,
        scheme,
        height=MASS_HEIGHT,
        bargap=0.04,
        showlegend=False,
        margin=dict(l=56, r=14, t=22, b=44),
        xaxis=dict(title=f"{column} (raw ppm)", range=[edges[0], edges[-1]], zeroline=False),
        yaxis=dict(title="accepted identifications", rangemode="tozero"),
        meta={"view": view},
    )


def funnel_rows(funnel: Sequence[tuple[str, int]]) -> list[tuple[str, int, int]]:
    """(step, rows left, rows removed by the step) of the anchor funnel."""
    out = []
    prev = None
    for step, n in funnel:
        out.append((step, n, 0 if prev is None else prev - n))
        prev = n
    return out
