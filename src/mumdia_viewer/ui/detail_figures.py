"""Figures of the precursor page: the XIC panels, the spectrum mirror, the competition and
the feature percentiles.

Every function is a pure function of data-layer objects (no Dash, no I/O) and finishes
with :func:`figures.themed`. Retention times are seconds.

Fragment traces carry ``meta = {"frag": k, "op": base opacity}``, where ``k`` is the
fragment's index among the candidate's fragment rows (see :mod:`.detail_view`). The page's
script uses it to link a fragment across the XIC, the mirror and the fragment table.
Shapes and annotations carry a ``name`` so the script (and the tests) can find them.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mumdia_viewer.data.detail import MirrorData, PrecursorDetail
from mumdia_viewer.data.features import FeaturePercentile

from . import figures, theme
from .detail_view import (
    Fragment,
    ScanGrid,
    feature_label,
    feature_unit,
    fmt_q_at,
    fmt_score,
    format_value,
    score_digits,
    short_paths,
    wrap_text,
)
from .state import THRESHOLD_STOPS
from .widgets import parse_peptidoform

# The page's XIC and mirror take their height from the page's CSS (detail.css: 520 and
# 400 px, shorter below Mantine's lg breakpoint), so their figures leave layout.height
# unset; the compact figures of the identification page have a fixed height.
COMPACT_XIC_HEIGHT = 250
COMPACT_SPECTRUM_HEIGHT = 200
SPECTRUM_TOP = 134.0
OBSERVED = "rgba(134, 142, 150, 0.42)"
CLIP = 110.0  # observed peaks above this (in % of the reference) are drawn clipped
LABEL_FONT = 10
# The XIC's margins: the scan slider under the plot is lined up with its plot area
# (detail_cards.scrubber). The right margin holds the vertical modebar, which would
# otherwise cover the legend.
XIC_MARGIN = {"l": 64, "r": 36, "t": 48, "b": 46}
MIRROR_MARGIN = {"l": 56, "r": 36, "t": 26, "b": 44}
MODEBAR = {"orientation": "v"}


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _r(values: Any, digits: int) -> np.ndarray:
    """``values`` rounded to ``digits``, as float32.

    Plotly sends numpy arrays as base64 typed arrays, so float32 halves the page's
    figures; the spectra and the chromatograms are stored as float32, so nothing is lost.
    """
    return np.round(np.asarray(values, dtype=np.float64), digits).astype(np.float32)


def _empty(message: str, scheme: str, height: int | None) -> go.Figure:
    """An empty figure with ``message`` on short lines (absolute paths as file names)."""
    text = "<br>".join(wrap_text(short_paths(message), 64))
    fig = figures.empty_figure(text, scheme, height=height or 260)
    if height is None:
        fig.layout.height = None
    return fig


# --------------------------------------------------------------------------- XIC


def x_range(d: PrecursorDetail, grid: ScanGrid | None) -> dict[str, Any]:
    """The plotted RT range and how the RT window relates to it.

    The range is the XIC axis (the grid scans) padded by half a scan step. The window
    bounds are drawn when they lie inside the padded range or close to it (within a
    quarter of the axis span, the range then grows to show them); otherwise the plot
    marks that the window continues past its edge.
    """
    axis = None
    if grid is not None and grid.size:
        axis = grid.rt
    elif d.chromatogram is not None:
        ax = d.chromatogram.common_axis()
        if ax is None:
            parts = [t.rt for t in d.chromatogram.traces if t.observed]
            ax = np.concatenate(parts) if parts else None
        axis = None if ax is None or ax.size == 0 else np.asarray(ax, dtype=np.float64)
    if axis is None or axis.size == 0:
        return {"range": None, "lo_inside": False, "hi_inside": False, "axis": None}
    a0, a1 = float(np.min(axis)), float(np.max(axis))
    step = float(np.median(np.diff(np.sort(axis)))) if axis.size > 1 else 1.0
    pad = max(step / 2.0, 1e-3)
    x0, x1 = a0 - pad, a1 + pad
    span = max(a1 - a0, step)
    w = d.window
    lo = w.rt_lo if w is not None else None
    hi = w.rt_hi if w is not None else None
    lo_inside = hi_inside = False
    if lo is not None and lo >= x0 - 0.25 * span:
        x0 = min(x0, lo - pad / 2)
        lo_inside = True
    if hi is not None and hi <= x1 + 0.25 * span:
        x1 = max(x1, hi + pad / 2)
        hi_inside = True
    return {
        "range": [x0, x1],
        "lo_inside": lo_inside,
        "hi_inside": hi_inside,
        "axis": (a0, a1),
        "step": step,
    }


def peak_range(d: PrecursorDetail, xr: dict[str, Any]) -> list[float] | None:
    """The XIC's opening view: the identification's peak and a few peak widths around it.

    The view covers the elution and integration bounds, the apex, the calibrated
    prediction and the other extracted peaks that lie in the plotted range, padded by
    2.5 peak widths (at least three scan steps) on each side and at least ten scan steps
    wide. None when it would show most of the range anyway (the whole window is then
    the only view).
    """
    rng = xr.get("range")
    if not rng:
        return None
    m = d.markers
    keys = (
        "elution_lo",
        "elution_hi",
        "integration_lo_rt",
        "integration_hi_rt",
        "apex_rt",
        "rt_pred_cal",
    )
    pts = [_finite(m.get(k)) for k in keys]
    pts += [_finite(p.get("apex_rt")) for p in m.get("peaks", [])]
    pts = [p for p in pts if p is not None and rng[0] <= p <= rng[1]]
    if not pts:
        return None
    lo, hi = min(pts), max(pts)
    step = float(xr.get("step") or 1.0)
    e_lo, e_hi = _finite(m.get("elution_lo")), _finite(m.get("elution_hi"))
    width = e_hi - e_lo if e_lo is not None and e_hi is not None and e_hi > e_lo else 0.0
    pad = max(2.5 * width, 3.0 * step)
    x0, x1 = max(rng[0], lo - pad), min(rng[1], hi + pad)
    short = 10.0 * step - (x1 - x0)
    if short > 0:
        x0, x1 = max(rng[0], x0 - short / 2), min(rng[1], x1 + short / 2)
    if x1 - x0 >= 0.8 * (rng[1] - rng[0]):
        return None
    return [round(x0, 3), round(x1, 3)]


def _vline(
    name: str,
    x: float,
    colour: str,
    dash: str,
    width: float,
    label: str | None,
    position: str = "end",
    anchor: str = "left",
    size: int = 10,
) -> dict[str, Any]:
    shape: dict[str, Any] = dict(
        type="line",
        name=name,
        x0=x,
        x1=x,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=colour, width=width, dash=dash),
        layer="above",
    )
    if label:
        shape["label"] = dict(
            text=label,
            textposition=position,
            textangle=0,
            font=dict(size=size, color=colour),
            xanchor=anchor,
            yanchor="top",
            padding=3,
        )
    return shape


def xic_figure(
    d: PrecursorDetail,
    frags: Sequence[Fragment],
    grid: ScanGrid | None,
    scheme: str = "light",
    *,
    hidden: Collection[int] = (),
    compact: bool = False,
    height: int | None = None,
    key: str | None = None,
    view: str = "peak",
) -> go.Figure:
    """Fragment XICs (top) and MS1 isotope XICs (bottom) on one shared RT axis.

    Markers: the identification's elution bounds (green band) and apex (green line),
    quant's integration bounds (violet dashed box), the calibrated RT prediction (grey
    dotted line), the RT window (grey dash-dot lines, or an edge note when it lies
    outside the shown range) and the other extracted peaks (arrows at the top). Invisible
    bars over the grid scans make every RT column clickable. The shown scan is not in
    the figure: the page draws it as an overlay (``pd-scanmark``), so that stepping
    through the scans does not redraw the plot; ``layout.meta`` holds the scan step it
    needs.

    ``view="peak"`` opens on :func:`peak_range` (the elution peak with a few peak widths
    around it), ``"window"`` on the whole plotted RT window; ``layout.meta["views"]``
    holds both ranges and ``meta["edges"]`` the window bounds whose edge notes the
    page's script shows when a bound lies outside the shown range. ``compact`` gives the
    static thumbnail of the identification page: no legend and no click targets.
    ``key`` (``"<run>|<candidate id>"``) names the page in ``layout.meta``, so the page's
    script moves the band of its own figure only. ``height`` None leaves the height to
    the page's CSS.
    """
    tall = height if height is not None else (COMPACT_XIC_HEIGHT if compact else None)
    chrom = d.chromatogram
    if chrom is None:
        return _empty("No chromatogram rows for this candidate.", scheme, tall)
    traces = chrom.fragments()
    hidden = set(hidden)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06 if compact else 0.075,
    )
    ms1_top = float(fig.layout.yaxis2.domain[1])
    xr = x_range(d, grid)
    peak = peak_range(d, xr)
    views = {"window": xr.get("range"), "peak": peak or xr.get("range")}
    view = view if peak is not None else "window"
    shown = views[view]
    observed = [traces[f.index] for f in frags if f.observed]
    ymax = max((float(np.max(t.intensity)) for t in observed if t.intensity.size), default=0.0)
    ms1 = [t for t in chrom.ms1() if t.observed]
    m1max = max((float(np.max(t.intensity)) for t in ms1 if t.intensity.size), default=0.0)
    # The intensity axes of each view are scaled to the traces inside it.
    head = 1.2 if compact else 1.16
    y_views = {
        name: [0.0, _peak_height(observed, rng) * head] if ymax > 0 else None
        for name, rng in views.items()
    }
    y2_views = {
        name: [0.0, _peak_height(ms1, rng) * 1.1] if m1max > 0 else None
        for name, rng in views.items()
    }

    # Click targets: one invisible bar per grid scan in each row.
    if grid is not None and grid.size and not compact:
        width = grid.spacing()
        custom = np.stack([grid.scan_index, grid.rows], axis=1)
        for row, top in ((1, ymax), (2, m1max)):
            fig.add_trace(
                go.Bar(
                    x=grid.rt32,
                    y=np.full(grid.size, top if top > 0 else 1.0, dtype=np.float32),
                    width=width,
                    marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
                    customdata=custom,
                    hovertemplate=(
                        "scan_index %{customdata[0]}<br>RT %{x:.2f} s<br>"
                        "<i>click to show this scan</i><extra></extra>"
                    ),
                    showlegend=False,
                    name="scan" if row == 1 else "scan (MS1 row)",
                    meta={"kind": "scan"},
                ),
                row=row,
                col=1,
            )

    for f in frags:
        t = traces[f.index]
        visible = "legendonly" if f.index in hidden else True
        if f.observed:
            fig.add_trace(
                go.Scatter(
                    x=_r(t.rt, 3),
                    y=_r(t.intensity, 1),
                    mode="lines" if compact else "lines+markers",
                    name=f.html,
                    legendgroup=f"f{f.index}",
                    line=dict(color=f.colour, width=1.6 if compact else 1.8, shape="linear"),
                    marker=dict(size=4, color=f.colour),
                    meta={"frag": f.index, "op": 1.0},
                    visible=visible,
                    hovertemplate=(
                        f"<b>{f.html}</b> · m/z {f.mz:.4f}<br>RT %{{x:.2f}} s<br>"
                        "intensity %{y:,.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        elif not compact:
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name=f"{f.html} (never observed)",
                    legendgroup=f"f{f.index}",
                    line=dict(color=f.colour, width=1.5, dash="dot"),
                    meta={"frag": f.index, "op": 0.5},
                    opacity=0.5,
                    visible=visible,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    names = {"ms1_mono": "mono", "ms1_iso1": "+1", "ms1_iso2": "+2"}
    for i, t in enumerate(ms1):
        colour = theme.MS1[i % len(theme.MS1)]
        fig.add_trace(
            go.Scatter(
                x=_r(t.rt, 3),
                y=_r(t.intensity, 1),
                mode="lines" if compact else "lines+markers",
                name=names.get(t.frag_name, t.frag_name),
                line=dict(color=colour, width=1.6 if compact else 1.8),
                marker=dict(size=3.5, color=colour),
                legend="legend2",
                meta={"kind": "ms1", "name": t.frag_name},
                hovertemplate=(
                    f"<b>MS1 {names.get(t.frag_name, t.frag_name)}</b> ({t.frag_name}) · m/z "
                    f"{t.frag_mz:.4f}<br>RT %{{x:.2f}} s<br>intensity %{{y:,.0f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    if ms1 and m1max <= 0:
        fig.add_annotation(
            name="ms1-zero",
            text="the MS1 isotope traces are 0 in every grid scan",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.55,
            showarrow=False,
            font=dict(size=11, color=theme.PREDICTION),
        )
    if not ms1:
        fig.add_annotation(
            name="no-ms1",
            text="no MS1 isotope rows (sparse mode or no MS1 data)",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=11),
        )

    shapes, notes, edges = _xic_markers(d, xr, shown, compact=compact)
    step = xr.get("step") or (grid.spacing() if grid is not None else 1.0)
    annotations = _peak_annotations(d, compact=compact) + notes
    if not observed:
        annotations.append(
            dict(
                name="no-fragments",
                text="no fragment of this candidate was observed in its RT window",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11 if compact else 12),
            )
        )
    for a in annotations:
        fig.add_annotation(**a)
    for s in shapes:
        fig.add_shape(**s)
    legends: dict[str, Any] = {
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="toggleitem",
            itemwidth=30,
            tracegroupgap=0,
        ),
        "legend2": dict(
            orientation="h",
            yanchor="bottom",
            y=ms1_top + 0.004,
            xanchor="right",
            x=1,
            font=dict(size=10),
            itemwidth=30,
            bgcolor="rgba(0,0,0,0)",
        ),
    }
    figures.themed(
        fig,
        scheme,
        height=tall,
        hovermode="closest",
        hoverdistance=12,
        uirevision="pv-xic" if compact else "pd-xic",
        bargap=0,
        margin=dict(l=46, r=8, t=8, b=30) if compact else dict(XIC_MARGIN),
        showlegend=not compact,
        modebar=MODEBAR,
        meta={
            "xr": xr.get("range"),
            "views": views,
            "view": view,
            "y": y_views,
            "y2": y2_views,
            "edges": edges,
            "step": step,
            "key": key,
        },
        **({} if compact else legends),
    )
    yr = y_views[view]
    small = dict(tickfont=dict(size=10), title_font=dict(size=10)) if compact else {}
    fig.update_yaxes(
        title_text="fragments" if compact else "fragment intensity",
        tickformat="~s",
        rangemode="tozero",
        row=1,
        col=1,
        nticks=4 if compact else None,
        **small,
        **({"range": yr} if yr else {}),
    )
    fig.update_yaxes(
        title_text="MS1" if compact else "MS1 isotopes",
        tickformat="~s",
        rangemode="tozero",
        row=2,
        col=1,
        nticks=3 if compact else None,
        **small,
        **(
            {"range": [0, 1], "showticklabels": False}
            if m1max <= 0
            else {"range": y2_views[view]}
            if y2_views[view]
            else {}
        ),
    )
    fig.update_xaxes(showspikes=False)
    fig.update_xaxes(
        title_text="RT (s)" if compact else "retention time (s)",
        row=2,
        col=1,
        **(dict(title_standoff=2, **small) if compact else {}),
    )
    if compact:
        fig.update_xaxes(tickfont=dict(size=10), row=1, col=1)
    if shown:
        fig.update_xaxes(range=list(shown))
    return fig


def _peak_height(traces: Sequence[Any], rng: Sequence[float] | None) -> float:
    """The highest intensity of ``traces`` inside the RT range ``rng`` (all of it for None).

    Falls back to the highest anywhere when nothing inside the range is above 0.
    """
    best = top = 0.0
    for t in traces:
        inten = np.asarray(t.intensity, dtype=np.float64)
        if inten.size == 0:
            continue
        top = max(top, float(np.nanmax(inten)))
        if rng:
            rt = np.asarray(t.rt, dtype=np.float64)
            inside = inten[(rt >= rng[0]) & (rt <= rng[1])]
            if inside.size:
                best = max(best, float(np.nanmax(inside)))
        else:
            best = top
    return best if best > 0 else top


def edge_visible(x: float, side: str, shown: Sequence[float] | None) -> bool:
    """Whether the edge note of a window bound shows: the bound lies outside ``shown``."""
    if not shown:
        return True
    return x < shown[0] if side == "lo" else x > shown[1]


def _xic_markers(
    d: PrecursorDetail,
    xr: dict[str, Any],
    shown: Sequence[float] | None = None,
    *,
    compact: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """The marker shapes, the edge notes and the edge records (``meta["edges"]``) of the XIC.

    A window bound inside the plotted range is a line; its edge note shows only while the
    bound lies outside the shown range (``shown``, the opening view). A bound outside the
    plotted range has its note only.
    """
    m = d.markers
    shapes: list[dict] = []
    notes: list[dict] = []
    edges: list[dict] = []
    size = 9 if compact else 10
    lo, hi = _finite(m.get("elution_lo")), _finite(m.get("elution_hi"))
    if lo is not None and hi is not None:
        shapes.append(
            dict(
                type="rect",
                name="elution",
                x0=lo,
                x1=hi,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                fillcolor=theme.ELUTION,
                line=dict(width=0),
                layer="below",
            )
        )
    ilo, ihi = _finite(m.get("integration_lo_rt")), _finite(m.get("integration_hi_rt"))
    if ilo is not None and ihi is not None:
        shapes.append(
            dict(
                type="rect",
                name="integration",
                x0=ilo,
                x1=ihi,
                xref="x",
                y0=0.004,
                y1=0.996,
                yref="paper",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=theme.INTEGRATION, width=1.3, dash="dash"),
                layer="above",
            )
        )
    pred = _finite(m.get("rt_pred_cal"))
    apex = _finite(m.get("apex_rt"))
    # The two labels sit at the top, each on the side away from the other line.
    apex_first = apex is not None and pred is not None and apex <= pred
    if pred is not None:
        shapes.append(
            _vline(
                "prediction",
                pred,
                theme.PREDICTION,
                "dot",
                1.6,
                "pred.",
                "end",
                "left" if apex_first or apex is None else "right",
                size,
            )
        )
    if apex is not None:
        shapes.append(
            _vline(
                "apex",
                apex,
                theme.APEX,
                "solid",
                2.0,
                "apex",
                "end",
                "right" if apex_first else "left",
                size,
            )
        )
    w_lo, w_hi = _finite(m.get("rt_lo")), _finite(m.get("rt_hi"))
    shown = shown or xr.get("range")
    for name, x, side, inside in (
        ("window_lo", w_lo, "lo", xr.get("lo_inside")),
        ("window_hi", w_hi, "hi", xr.get("hi_inside")),
    ):
        if x is None:
            continue
        if inside:
            shapes.append(_vline(name, x, theme.WINDOW, "dashdot", 1.4, None))
        note = f"edge-{name}"
        # Inside the plotted range the note shows only while the bound is out of view.
        # The notes sit in the bottom corners, beside the axis title, away from the
        # labels of the lines at the top.
        edges.append({"name": note, "x": x, "side": side if inside else "always"})
        notes.append(
            dict(
                name=note,
                text=f"◂ RT window from {x:.1f} s" if side == "lo" else f"RT window to {x:.1f} s ▸",
                xref="x2 domain",
                yref="y2 domain",
                x=0.0 if side == "lo" else 1.0,
                y=0.0,
                xanchor="left" if side == "lo" else "right",
                yanchor="top",
                yshift=-17 if compact else -24,
                showarrow=False,
                visible=True if not inside else edge_visible(x, side, shown),
                font=dict(size=size, color=theme.PREDICTION),
            )
        )
    return shapes, notes, edges


def _peak_annotations(d: PrecursorDetail, *, compact: bool = False) -> list[dict]:
    """Arrows at the top of the fragment panel for every extracted peak, when there are several.

    The selected peak's arrow has no text (the apex line and its label mark it); the
    arrows sit below the row of line labels at the top of the panel, so the labels do
    not overlap.
    """
    peaks = [p for p in d.markers.get("peaks", []) if _finite(p.get("apex_rt")) is not None]
    if len(peaks) < 2:
        return []
    rows = {}
    if not d.peaks.empty and "peak_rank" in d.peaks:
        for _, r in d.peaks.iterrows():
            rows[int(r["peak_rank"])] = r
    out = []
    for p in peaks:
        rank = int(p["peak_rank"])
        r = rows.get(rank)
        bits = [f"peak_rank {rank}", f"apex {float(p['apex_rt']):.2f} s"]
        if r is not None:
            for col, label in (
                ("apex_intensity", "apex intensity"),
                ("n_matched_fragments", "matched fragments"),
                ("coelution_run", "co-elution run"),
            ):
                v = _finite(r.get(col)) if col in r else None
                if v is not None:
                    bits.append(f"{label} {v:,.0f}")
        bits.append(
            "selected by the rescorer"
            if p["selected"]
            else "not selected (its score is not stored)"
        )
        colour = theme.APEX if p["selected"] else theme.WARN
        out.append(
            dict(
                name=f"peak-{rank}",
                x=float(p["apex_rt"]),
                xref="x",
                y=0.8,
                yref="y domain",
                text="" if p["selected"] else f"peak {rank}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.6,
                arrowcolor=colour,
                ax=0,
                ay=-10 if compact else -13,
                yanchor="bottom",
                font=dict(size=9 if compact else 10, color=colour),
                hovertext="<br>".join(bits),
            )
        )
    return out


# --------------------------------------------------------------------------- mirror


def _stems(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segments from 0 to each y as one trace (NaN separates the stems)."""
    n = int(np.asarray(x).size)
    xs = np.empty(3 * n, dtype=np.float32)
    ys = np.empty(3 * n, dtype=np.float32)
    xs[0::3] = x
    xs[1::3] = x
    xs[2::3] = np.nan
    ys[0::3] = 0.0
    ys[1::3] = y
    ys[2::3] = np.nan
    return xs, ys


def _label_heights(
    xs: Sequence[float], ys: Sequence[float], span: float, *, top: float = 150.0
) -> list[float]:
    """Label positions that do not overlap: a label moves up a tier when a placed one is near."""
    width = span * 0.085
    height = 11.0
    placed: list[tuple[float, float]] = []
    out = []
    for x, y in zip(xs, ys, strict=True):
        h = min(y, CLIP) + 4.0
        while any(abs(x - px) < width and abs(h - py) < height for px, py in placed) and h < top:
            h += height
        placed.append((x, h))
        out.append(h)
    return out


def _unmatched(n: int, matches: Sequence[Any]) -> np.ndarray:
    """A mask of the peaks that no shown fragment matched (the hover of plain peaks).

    A matched peak is left out: its fragment trace carries its hover, so hovering it
    names (and highlights) the fragment.
    """
    free = np.ones(n, dtype=bool)
    for mt in matches:
        if 0 <= mt.peak < n:
            free[mt.peak] = False
    return free


def _peak_tops(
    fig: go.Figure, mz: np.ndarray, shown: np.ndarray, rel: np.ndarray, free: np.ndarray, what: str
) -> None:
    """Invisible markers on the tops of the peaks no fragment matched: their hover.

    The hover gives the m/z and the height in % of ``what``; a clipped peak (drawn at
    110%) has a trace of its own that carries its true height.
    """
    cut = free & (rel > CLIP)
    plain = free & ~cut
    common = dict(
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        showlegend=False,
        name="observed peak tops",
        meta={"kind": "observed"},
    )
    fig.add_trace(
        go.Scatter(
            x=_r(mz[plain], 4),
            y=_r(shown[plain], 1),
            hovertemplate=f"m/z %{{x:.4f}}<br>%{{y:.1f}}% of {what}<extra>observed peak</extra>",
            **common,
        )
    )
    if cut.any():
        fig.add_trace(
            go.Scatter(
                x=_r(mz[cut], 4),
                y=_r(shown[cut], 1),
                customdata=_r(rel[cut], 1),
                hovertemplate=(
                    f"m/z %{{x:.4f}}<br>%{{customdata:.1f}}% of {what} (drawn clipped at "
                    f"{CLIP:.0f}%)<extra>observed peak</extra>"
                ),
                **common,
            )
        )


def mirror_figure(
    m: MirrorData | None,
    frags: Sequence[Fragment],
    scheme: str = "light",
    *,
    hidden: Collection[int] = (),
    scale: str = "matched",
    outside: bool | None = None,
    why_empty: str = "No spectrum for this candidate.",
    x_range: Sequence[float] | None = None,
    height: int | None = None,
) -> go.Figure:
    """The observed MS2 spectrum (up) against the predicted fragments (down).

    ``scale="matched"`` draws the observed peaks in % of the highest matched peak (peaks
    above 110% are clipped and marked); ``"base"`` in % of the base peak. The predicted
    fragments are in % of the highest predicted intensity. Matched peaks carry the
    fragment colour of the XIC and a label with the fragment name and the raw ppm error.
    ``x_range`` is the candidate's m/z range (:func:`detail_view.mz_range`): the same for
    every scan, so a user zoom survives a scan step. ``height`` None leaves the height to
    the page's CSS.
    """
    if m is None:
        return _empty(why_empty, scheme, height)
    hidden = set(hidden)
    spec = m.spectrum
    mz = spec.mz.astype(np.float64)
    inten = spec.intensity.astype(np.float64)
    by_index = {f.index: f for f in frags}
    matches = [mt for mt in m.matches if mt.fragment in by_index]
    base = float(inten.max()) if inten.size else 1.0
    top_match = max((mt.obs_intensity for mt in matches), default=0.0)
    by_match = scale == "matched" and top_match > 0
    ref = top_match if by_match else base
    ref = ref if ref > 0 else 1.0
    rel = 100.0 * inten / ref
    clipped = rel > CLIP
    shown = np.minimum(rel, CLIP)
    theo = m.fragments["theo_mz"].to_numpy(dtype=np.float64)
    if x_range is None:
        lo = float(mz.min()) if mz.size else 0.0
        hi = float(mz.max()) if mz.size else 1.0
        if theo.size:
            lo, hi = min(lo, float(theo.min())), max(hi, float(theo.max()))
        pad = 0.03 * max(hi - lo, 1.0)
        x_range = [lo - pad, hi + pad]
    span = max(float(x_range[1]) - float(x_range[0]), 1.0) / 1.06
    what = "the highest matched peak" if by_match else "the base peak"

    fig = go.Figure()
    xs, ys = _stems(_r(mz, 4), _r(shown, 1))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=OBSERVED, width=1),
            hoverinfo="skip",
            showlegend=False,
            name="observed peaks",
            meta={"kind": "observed"},
        )
    )
    _peak_tops(fig, mz, shown, rel, _unmatched(mz.size, matches), what)

    frag_rows = m.fragments
    pred = frag_rows["predicted_intensity"].to_numpy(dtype=np.float64)
    pmax = float(pred.max()) if pred.size and pred.max() > 0 else 1.0
    matched = {mt.fragment: mt for mt in matches}
    for f in frags:
        if f.index >= len(frag_rows):
            continue
        mt = matched.get(f.index)
        p = round(-100.0 * float(pred[f.index]) / pmax, 1)
        theo_mz = round(float(theo[f.index]), 4)
        op = 1.0 if mt is not None else 0.4
        visible = f.index not in hidden
        status = "matched in this scan" if mt is not None else "not matched in this scan"
        xic = "observed in the XIC" if f.observed else "never observed in the XIC"
        fig.add_trace(
            go.Scatter(
                x=[theo_mz, theo_mz],
                y=[0.0, p],
                mode="lines+markers",
                line=dict(color=f.colour, width=2.2, dash="solid" if f.observed else "dot"),
                marker=dict(size=[0, 6], color=f.colour),
                opacity=op,
                visible=visible,
                showlegend=False,
                name=f"{f.text} predicted",
                meta={"frag": f.index, "op": op},
                hovertemplate=(
                    f"<b>{f.html}</b> predicted<br>m/z {theo_mz:.4f} · intensity "
                    f"{pred[f.index]:.3f}<br>{status}<br>{xic}<extra></extra>"
                ),
            )
        )
    if matches:
        order = sorted(matches, key=lambda mt: mt.obs_mz)
        heights = _label_heights(
            [mt.obs_mz for mt in order],
            [min(100.0 * mt.obs_intensity / ref, CLIP) for mt in order],
            span,
        )
        for mt, h in zip(order, heights, strict=True):
            f = by_index[mt.fragment]
            y = round(min(100.0 * mt.obs_intensity / ref, CLIP), 1)
            obs = round(float(mt.obs_mz), 4)
            visible = f.index not in hidden
            hover = (
                f"<b>{f.html}</b> matched<br>observed m/z {mt.obs_mz:.4f}<br>"
                f"theoretical m/z {mt.theo_mz:.4f}<br>ppm raw {mt.ppm_raw:+.2f} · "
                f"offset-corrected {mt.ppm_corrected:+.2f} (viewer-derived)<br>"
                f"intensity {mt.obs_intensity:,.0f} ({100.0 * mt.obs_intensity / ref:.1f}%)"
                "<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=[obs, obs],
                    y=[0.0, y],
                    mode="lines+markers",
                    line=dict(color=f.colour, width=2.4),
                    marker=dict(size=[0, 7], color=f.colour),
                    visible=visible,
                    showlegend=False,
                    name=f"{f.text} observed",
                    meta={"frag": f.index, "op": 1.0},
                    hovertemplate=hover,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[obs],
                    y=[round(h, 1)],
                    mode="text",
                    text=[f"<b>{f.html}</b> {mt.ppm_raw:+.1f}"],
                    textposition="top center",
                    textfont=dict(size=LABEL_FONT, color=f.colour),
                    visible=visible,
                    showlegend=False,
                    name=f"{f.text} label",
                    meta={"frag": f.index, "op": 1.0},
                    hovertemplate=hover,
                )
            )
    title_top = f"observed · % of {what}" + (
        " · taller peaks clipped" if bool(clipped.any()) else ""
    )
    annotations = [
        dict(
            name="observed",
            text=title_top,
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            yshift=4,
            showarrow=False,
            font=dict(size=10),
        ),
        dict(
            name="predicted",
            text="predicted · % of the highest library intensity",
            xref="paper",
            yref="paper",
            x=0.0,
            y=0.0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=10),
        ),
    ]
    if outside:
        annotations.append(
            dict(
                name="outside",
                text="<b>outside the RT window</b>",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0,
                xanchor="right",
                yanchor="bottom",
                yshift=4,
                showarrow=False,
                font=dict(size=10, color=theme.WARN),
            )
        )
    for a in annotations:
        fig.add_annotation(**a)
    return figures.themed(
        fig,
        scheme,
        height=height,
        hovermode="closest",
        hoverdistance=10,
        uirevision="pd-mirror",
        showlegend=False,
        margin=dict(MIRROR_MARGIN),
        modebar=MODEBAR,
        xaxis=dict(title="m/z", showspikes=False, range=[float(v) for v in x_range]),
        yaxis=dict(
            title="relative intensity (%)",
            range=[-116, 158],
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["100", "50", "0", "50", "100"],
            zeroline=True,
            zerolinewidth=1.4,
            fixedrange=False,
        ),
    )


def spectrum_figure(
    m: MirrorData | None,
    frags: Sequence[Fragment],
    scheme: str = "light",
    *,
    height: int = COMPACT_SPECTRUM_HEIGHT,
    why_empty: str = "No spectrum for this candidate.",
) -> go.Figure:
    """The annotated MS2 spectrum of the scan of ``m``, without the mirror half.

    Observed peaks are grey, in % of the highest matched peak (of the base peak when
    nothing matched; taller peaks are clipped at 110%, see :func:`spectrum_scale`);
    matched peaks are drawn in their fragment's XIC colour and labelled with the
    fragment name. The hover gives the observed and library m/z, the raw ppm error and
    the intensity. This is the static spectrum of the identification page.
    """
    if m is None:
        return _empty(why_empty, scheme, height)
    spec = m.spectrum
    mz = spec.mz.astype(np.float64)
    inten = spec.intensity.astype(np.float64)
    by_index = {f.index: f for f in frags}
    matches = [mt for mt in m.matches if mt.fragment in by_index]
    base = float(inten.max()) if inten.size else 1.0
    top_match = max((mt.obs_intensity for mt in matches), default=0.0)
    ref = top_match if top_match > 0 else base
    ref = ref if ref > 0 else 1.0
    rel = 100.0 * inten / ref
    shown = np.minimum(rel, CLIP)
    lo = float(mz.min()) if mz.size else 0.0
    hi = float(mz.max()) if mz.size else 1.0
    theo = m.fragments["theo_mz"].to_numpy(dtype=np.float64)
    if theo.size:
        lo, hi = min(lo, float(theo.min())), max(hi, float(theo.max()))
    span = max(hi - lo, 1.0)
    fig = go.Figure()
    xs, ys = _stems(_r(mz, 4), _r(shown, 1))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=OBSERVED, width=1),
            hoverinfo="skip",
            showlegend=False,
            name="observed peaks",
            meta={"kind": "observed"},
        )
    )
    what = "the highest matched peak" if top_match > 0 else "the base peak"
    _peak_tops(fig, mz, shown, rel, _unmatched(mz.size, matches), what)
    order = sorted(matches, key=lambda mt: mt.obs_mz)
    heights = _label_heights(
        [mt.obs_mz for mt in order],
        [min(100.0 * mt.obs_intensity / ref, CLIP) for mt in order],
        span,
        top=SPECTRUM_TOP - 12.0,
    )
    for mt, h in zip(order, heights, strict=True):
        f = by_index[mt.fragment]
        y = round(min(100.0 * mt.obs_intensity / ref, CLIP), 1)
        obs = round(float(mt.obs_mz), 4)
        hover = (
            f"<b>{f.html}</b> matched<br>observed m/z {mt.obs_mz:.4f} · library {mt.theo_mz:.4f}"
            f"<br>ppm raw {mt.ppm_raw:+.2f}<br>intensity {mt.obs_intensity:,.0f} "
            f"({100.0 * mt.obs_intensity / ref:.1f}%)<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[obs, obs],
                y=[0.0, y],
                mode="lines",
                line=dict(color=f.colour, width=2.2),
                showlegend=False,
                name=f"{f.text} observed",
                meta={"frag": f.index, "op": 1.0},
                hovertemplate=hover,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[obs],
                y=[round(h, 1)],
                mode="text",
                text=[f"<b>{f.html}</b>"],
                textposition="top center",
                textfont=dict(size=9, color=f.colour),
                showlegend=False,
                name=f"{f.text} label",
                meta={"frag": f.index, "op": 1.0},
                hovertemplate=hover,
            )
        )
    return figures.themed(
        fig,
        scheme,
        height=height,
        hovermode="closest",
        hoverdistance=10,
        uirevision="pv-spec",
        showlegend=False,
        margin=dict(l=40, r=8, t=6, b=30),
        xaxis=dict(
            title=dict(text="m/z", standoff=2, font=dict(size=10)),
            tickfont=dict(size=10),
            showspikes=False,
            range=[lo - 0.03 * span, hi + 0.03 * span],
        ),
        yaxis=dict(
            title=dict(text="rel. intensity (%)", font=dict(size=10)),
            tickfont=dict(size=10),
            range=[0, SPECTRUM_TOP],
            tickvals=[0, 50, 100],
            zeroline=True,
        ),
    )


def spectrum_scale(m: MirrorData | None) -> str:
    """What 100% means in :func:`spectrum_figure` (for its caption)."""
    if m is None:
        return ""
    top = max((mt.obs_intensity for mt in m.matches), default=0.0)
    inten = m.spectrum.intensity
    base = float(np.max(inten)) if inten.size else 0.0
    if top <= 0:
        return "Intensity in % of the base peak (no fragment matched)."
    clipped = base > top * CLIP / 100.0
    return "Intensity in % of the highest matched peak" + (
        "; taller peaks are clipped at 110%." if clipped else "."
    )


# --------------------------------------------------------------------------- competition


def q_text_any(value: Any) -> str:
    """A q value that reads unambiguously at every threshold stop of the header.

    For text that does not follow the threshold (a figure's hover): the longest of
    :func:`detail_view.fmt_q_at` over the stops, so 0.010003 never reads as 0.0100.
    """
    return max((fmt_q_at(value, t) for t in THRESHOLD_STOPS), key=len, default="")


def pep_html(text: str, max_chars: int = 26) -> str:
    """A peptidoform for Plotly text, drawn like ``widgets.peptidoform``.

    A modified residue in its modification's colour with the short tag as a superscript,
    ``DECOY_`` in the decoy colour; longer than ``max_chars`` residues, it ends with an
    ellipsis.
    """
    pf = parse_peptidoform(text)
    parts: list[str] = []
    if pf.decoy:
        parts.append(f"<span style='color:{theme.DECOY}'>DECOY_</span>")

    def mod(label: str, mods: tuple[str, ...]) -> str:
        styles = [theme.mod_style(t) for t in mods]
        tag = "+".join(s for s, _ in styles)
        return f"<span style='color:{styles[0][1]}'>{label}<sup>{tag}</sup></span>"

    if pf.nterm:
        parts.append(mod("n", pf.nterm) + "-")
    parts += [mod(r.aa, r.mods) if r.mods else r.aa for r in pf.residues[:max_chars]]
    if len(pf.residues) > max_chars:
        parts.append("…")
    elif pf.cterm:
        parts.append("-" + mod("c", pf.cterm))
    return "".join(parts)


def competition_figure(
    df: pd.DataFrame,
    scheme: str = "light",
    *,
    hrefs: Sequence[str] = (),
    experiment: bool = False,
    max_rows: int = 24,
    digits: int | None = None,
) -> go.Figure:
    """Score of every row of the base-peptide competition, best first.

    Colour gives the label; a star marks this row and a green ring the winner of the
    base-peptide key. A click on a dot opens that row's page (``customdata`` holds the
    address). The hover prints a grouped q only on the row that wins its group (the
    others store 1.0, which is not a q value); scores have ``digits`` decimals, enough
    to tell the rows apart (:func:`detail_view.score_digits`).
    """
    if df is None or df.empty:
        return figures.empty_figure("No competing rows.", scheme, height=160)
    frame = df.reset_index(drop=True).copy()
    frame["href"] = list(hrefs) if len(hrefs) == len(frame) else ""
    frame = frame.sort_values("score", ascending=False, kind="mergesort")
    keep = frame.head(max_rows)
    extra = frame[(frame["is_this_row"] | frame["wins_peptide"]) & ~frame.index.isin(keep.index)]
    frame = pd.concat([keep, extra])
    n = len(frame)
    ys = list(range(n))[::-1]
    digits = digits if digits is not None else score_digits(frame["score"])
    labels = []
    for r in frame.itertuples(index=False):
        labels.append(
            f"{pep_html(str(r.peptidoform))} {int(r.charge)}+"
            + (f" · {r.run}" if experiment else "")
        )
    fig = go.Figure()
    sparse = {
        "precursor_q": ("wins_precursor", "its (peptidoform, charge)"),
        "peptide_q_value": ("wins_peptide", "the base peptide"),
    }

    def hover(r) -> str:
        flags = []
        if r.wins_peptide:
            flags.append("winner of the base peptide")
        if r.wins_precursor:
            flags.append("winner of its (peptidoform, charge)")
        if r.is_this_row:
            flags.append("this row")
        lines = []
        for c in ("q_value", "run_psm_q", "precursor_q", "peptide_q_value"):
            if c not in frame.columns or (c == "run_psm_q" and not experiment):
                continue
            value = getattr(r, c)
            if c in sparse and not bool(getattr(r, sparse[c][0])):
                lines.append(f"{c}: not the winner of {sparse[c][1]} (stored 1.0, not a q value)")
            elif value is not None and not pd.isna(value):
                lines.append(f"{c} {q_text_any(value)}")
        return (
            f"<b>{pep_html(str(r.peptidoform), 60)}</b> {int(r.charge)}+ · {r.label}"
            + (f" · run {r.run}" if experiment else "")
            + f"<br>candidate {int(r.candidate_id)} · score {fmt_score(r.score, digits)}<br>"
            + "<br>".join(lines)
            + ("<br>" + "; ".join(flags) if flags else "")
        )

    rows = list(frame.itertuples(index=False))
    for label, colour, name in (
        ("target", theme.TARGET, "target"),
        ("decoy", theme.DECOY, "decoy"),
    ):
        idx = [i for i, r in enumerate(rows) if r.label == label and not r.is_this_row]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter(
                x=[float(rows[i].score) for i in idx],
                y=[ys[i] for i in idx],
                mode="markers",
                marker=dict(size=11, color=colour, line=dict(width=0)),
                customdata=[rows[i].href for i in idx],
                hovertext=[hover(rows[i]) for i in idx],
                hovertemplate="%{hovertext}<extra></extra>",
                name=name,
            )
        )
    win = [i for i, r in enumerate(rows) if r.wins_peptide]
    if win:
        fig.add_trace(
            go.Scatter(
                x=[float(rows[i].score) for i in win],
                y=[ys[i] for i in win],
                mode="markers",
                marker=dict(
                    size=24, symbol="circle-open", color=theme.ACCEPT, line=dict(width=2.2)
                ),
                hoverinfo="skip",
                name="base-peptide winner",
            )
        )
    this = [i for i, r in enumerate(rows) if r.is_this_row]
    if this:
        fig.add_trace(
            go.Scatter(
                x=[float(rows[i].score) for i in this],
                y=[ys[i] for i in this],
                mode="markers",
                marker=dict(
                    size=17,
                    symbol="star",
                    color=[theme.DECOY if rows[i].label == "decoy" else theme.TARGET for i in this],
                    line=dict(width=0),
                ),
                customdata=[rows[i].href for i in this],
                hovertext=[hover(rows[i]) for i in this],
                hovertemplate="%{hovertext}<extra></extra>",
                name="this row",
            )
        )
    scores = [float(r.score) for r in rows]
    s0, s1 = min(scores), max(scores)
    pad = (s1 - s0) * 0.08 if s1 > s0 else max(abs(s1) * 0.1, 0.1)
    return figures.themed(
        fig,
        scheme,
        height=max(170, 70 + 30 * n),
        hovermode="closest",
        margin=dict(l=10, r=16, t=30, b=40),
        xaxis=dict(title="rescorer score (higher is better)", range=[s0 - pad, s1 + pad]),
        yaxis=dict(
            tickvals=ys,
            ticktext=labels,
            range=[-0.7, n - 0.3],
            automargin=True,
            showgrid=True,
            tickfont=dict(size=11, family=theme.MONO),
        ),
    )


# --------------------------------------------------------------------------- percentiles


def percentile_figure(rows: Sequence[FeaturePercentile], scheme: str = "light") -> go.Figure:
    """Percentile of the candidate's value among the run's target and decoy rows.

    One line per feature: a dot for the targets (indigo) and one for the decoys (orange),
    joined by a thin line. Features that are not ranked are left out; the page lists
    them with the reason.
    """
    ranked = [r for r in rows if r.pct_target is not None or r.pct_decoy is not None]
    if not ranked:
        return figures.empty_figure("No ranked feature.", scheme, height=160)
    n = len(ranked)
    ys = list(range(n))[::-1]
    fig = go.Figure()
    for r, y in zip(ranked, ys, strict=True):
        a = r.pct_target if r.pct_target is not None else r.pct_decoy
        b = r.pct_decoy if r.pct_decoy is not None else r.pct_target
        fig.add_trace(
            go.Scatter(
                x=[a, b],
                y=[y + 0.13, y - 0.13],
                mode="lines",
                line=dict(color=theme.WINDOW, width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def text(r: FeaturePercentile) -> str:
        unit = feature_unit(r.feature)
        value = format_value(r.value, unit, r.feature)
        parts = [f"<b>{feature_label(r.feature)}</b> ({r.feature})", f"value {value} {unit}"]
        if r.pct_target is not None:
            parts.append(f"{r.pct_target:.1f}% of {r.n_target:,} target rows ≤ this value")
        if r.pct_decoy is not None:
            parts.append(f"{r.pct_decoy:.1f}% of {r.n_decoy:,} decoy rows ≤ this value")
        if r.rule:
            parts.append(f"valid rows: {r.rule}")
        if r.valid is False:
            parts.append("this candidate's value fails the validity rule")
        return "<br>".join(parts)

    hover = [text(r) for r in ranked]
    # Targets sit a little above the line and decoys a little below, so equal ranks stay
    # visible.
    for attr, colour, name, symbol, dy in (
        ("pct_target", theme.TARGET, "percentile among targets", "circle", 0.13),
        ("pct_decoy", theme.DECOY, "percentile among decoys", "diamond", -0.13),
    ):
        pts = [(getattr(r, attr), y + dy, h) for r, y, h in zip(ranked, ys, hover, strict=True)]
        pts = [p for p in pts if p[0] is not None]
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers",
                marker=dict(size=10, color=colour, symbol=symbol, line=dict(width=0)),
                name=name,
                hovertext=[p[2] for p in pts],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    labels = [r.feature for r in ranked]
    return figures.themed(
        fig,
        scheme,
        height=max(200, 70 + 26 * n),
        hovermode="closest",
        margin=dict(l=10, r=20, t=34, b=44),
        xaxis=dict(
            title="share of the run's rows with a value ≤ this candidate's (%)",
            range=[-2, 102],
            tickvals=[0, 25, 50, 75, 100],
            zeroline=False,
        ),
        yaxis=dict(
            tickvals=ys,
            ticktext=labels,
            range=[-0.7, n - 0.3],
            automargin=True,
            showgrid=True,
            tickfont=dict(size=11, family=theme.MONO),
        ),
        shapes=[
            dict(
                type="line",
                x0=50,
                x1=50,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=theme.WINDOW, width=1, dash="dot"),
                layer="below",
            )
        ],
    )
