"""The mobilogram card of the precursor page (diaPASEF runs only).

Intensity against 1/K0 for every predicted fragment (the XIC's colours and legend) and,
below, the MS1 isotopes, in the frame of the scan the XIC shows (the apex scan when the
page opens) and optionally one or two frames either side. Two sources
(:mod:`mumdia_viewer.data.mobilogram`): the raw timsTOF events binned by ``mumdia
mobilogram`` when an engine binary and the ``.d`` are available, otherwise the convert
centroids, which carry one 1/K0 per peak and are drawn as sticks. Markers: the
candidate's 1/K0 window (grey band), its calibrated 1/K0 prediction (dotted), the apex
1/K0 (green), the MS1 apex 1/K0 (dashed, MS1 row) and, on a run searched with retrace,
the band retrace summed in (dashed box).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Collection, Sequence
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update
from plotly.subplots import make_subplots

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import mobilogram as mob
from mumdia_viewer.data.detail import PrecursorDetail

from . import figures, theme
from .detail_figures import MODEBAR, XIC_MARGIN, _empty, _r
from .detail_view import Fragment, fragments_of
from .state import PageContext
from .widgets import graph, section

MOB = {"type": "fig", "name": "pd-mob"}
HEIGHT = 420
MS1_NAMES = {"ms1_mono": "mono", "ms1_iso1": "+1", "ms1_iso2": "+2"}
FRAME_CHOICES = (0, 1, 2)

_CACHE: OrderedDict[tuple, mob.Mobilogram] = OrderedDict()
_LOCK = threading.Lock()
_MAX = 24


def retrace_half_width(rs: ResultSet) -> float | None:
    """``retrace.im_half_width`` when the run was searched with retrace, else None."""
    if not rs.config_get("retrace", "enabled", default=False):
        return None
    try:
        hw = float(rs.config_get("retrace", "im_half_width", default=0.015))
    except (TypeError, ValueError):
        return None
    return hw if np.isfinite(hw) and hw > 0 else None


def _sticks(im: np.ndarray, y: np.ndarray) -> tuple[list, list]:
    """Vertical sticks from 0 to ``y`` at ``im``, as one line trace (None separates)."""
    xs: list = []
    ys: list = []
    for a, b in zip(np.round(im, 5).tolist(), np.asarray(y, dtype=float).tolist(), strict=True):
        xs += [a, a, None]
        ys += [0.0, b, None]
    return xs, ys


def _scaled(y: np.ndarray, norm: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if norm == "ion" and y.size and float(np.max(y)) > 0:
        return y / float(np.max(y))
    return y


def mobilogram_figure(
    m: mob.Mobilogram,
    frags: Sequence[Fragment],
    d: PrecursorDetail,
    scheme: str = "light",
    *,
    k: int = 2,
    norm: str = "abs",
    hidden: Collection[int] = (),
    retrace_hw: float | None = None,
) -> go.Figure:
    """The mobilogram figure: fragments on top, MS1 isotopes below, over 1/K0.

    ``k`` is the number of frames either side of the centre frame that are included,
    ``norm`` ``"abs"`` (raw intensity) or ``"ion"`` (each ion scaled to its maximum).
    Fragment traces carry ``meta = {"frag": index}`` as on the XIC; the stick (centroid)
    and marker traces of one fragment share its legend group. Shapes carry a ``name``.
    """
    hidden = set(hidden)
    raw = m.source == "raw"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.08
    )
    ms1_top = float(fig.layout.yaxis2.domain[1])
    unit = "relative intensity" if norm == "ion" else "intensity"
    any_signal = False
    for f in frags:
        ion = m.ion(f.name)
        if ion is None:
            continue
        im, y = m.profile(ion, k)
        y = _scaled(y, norm)
        any_signal |= bool(y.size and np.max(y) > 0)
        visible = "legendonly" if f.index in hidden else True
        hover = (
            f"<b>{f.html}</b> · m/z {f.mz:.4f}<br>1/K0 %{{x:.4f}}<br>{unit} "
            "%{y:,.3~g}<extra></extra>"
        )
        group = f"f{f.index}"
        if raw:
            fig.add_trace(
                go.Scatter(
                    x=_r(im, 5),
                    y=y,
                    mode="lines",
                    name=f.html,
                    legendgroup=group,
                    line=dict(color=f.colour, width=1.7),
                    meta={"frag": f.index, "op": 1.0},
                    visible=visible,
                    hovertemplate=hover,
                ),
                row=1,
                col=1,
            )
            continue
        sx, sy = _sticks(im, y)
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                legendgroup=group,
                showlegend=False,
                line=dict(color=f.colour, width=1.4),
                meta={"frag": f.index, "op": 1.0, "kind": "sticks"},
                visible=visible,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_r(im, 5),
                y=y,
                mode="markers",
                name=f.html,
                legendgroup=group,
                marker=dict(color=f.colour, size=5),
                meta={"frag": f.index, "op": 1.0, "kind": "tops"},
                visible=visible,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
    for i, name in enumerate(MS1_NAMES):
        ion = m.ion(name)
        if ion is None:
            continue
        im, y = m.profile(ion, k)
        y = _scaled(y, norm)
        colour = theme.MS1[i % len(theme.MS1)]
        hover = (
            f"<b>MS1 {MS1_NAMES[name]}</b> ({name}) · m/z {ion.mz:.4f}<br>1/K0 %{{x:.4f}}<br>"
            f"{unit} %{{y:,.3~g}}<extra></extra>"
        )
        common = dict(legend="legend2", legendgroup=name, meta={"kind": "ms1", "name": name})
        if raw:
            fig.add_trace(
                go.Scatter(
                    x=_r(im, 5),
                    y=y,
                    mode="lines",
                    name=MS1_NAMES[name],
                    line=dict(color=colour, width=1.6),
                    hovertemplate=hover,
                    **common,
                ),
                row=2,
                col=1,
            )
            continue
        sx, sy = _sticks(im, y)
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines",
                showlegend=False,
                line=dict(color=colour, width=1.3),
                hoverinfo="skip",
                **common,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=_r(im, 5),
                y=y,
                mode="markers",
                name=MS1_NAMES[name],
                marker=dict(color=colour, size=4.5),
                hovertemplate=hover,
                **common,
            ),
            row=2,
            col=1,
        )
    if not m.ms1_frames:
        fig.add_annotation(
            name="no-ms1",
            text="no MS1 isotope rows or MS1 spectra",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=11),
        )
    if not any_signal:
        fig.add_annotation(
            name="no-signal",
            text="no fragment signal in these frames",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
        )
    _markers(fig, d, retrace_hw)
    lo_hi = m.slot_im
    w = d.window
    if lo_hi is None and w is not None and w.im_lo is not None and w.im_hi is not None:
        lo_hi = (w.im_lo, w.im_hi)
    figures.themed(
        fig,
        scheme,
        height=HEIGHT,
        hovermode="closest",
        uirevision="pd-mob",
        margin=dict(XIC_MARGIN),
        modebar=MODEBAR,
        meta={"kind": "mobilogram", "source": m.source, "frames": int(k), "norm": norm},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            itemwidth=30,
            tracegroupgap=0,
        ),
        legend2=dict(
            orientation="h",
            yanchor="bottom",
            y=ms1_top + 0.004,
            xanchor="right",
            x=1,
            font=dict(size=10),
            itemwidth=30,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_yaxes(title_text=f"fragment {unit}", tickformat="~s", rangemode="tozero", row=1)
    fig.update_yaxes(title_text="MS1 isotopes", tickformat="~s", rangemode="tozero", row=2)
    fig.update_xaxes(title_text="1/K0 (V s cm⁻²)", tickformat=".3f", row=2, col=1)
    if lo_hi is not None:
        pad = 0.004
        fig.update_xaxes(range=[lo_hi[0] - pad, lo_hi[1] + pad])
    return fig


def _markers(fig: go.Figure, d: PrecursorDetail, retrace_hw: float | None) -> None:
    """The 1/K0 window, prediction, apex, MS1 apex and retrace band, as named shapes."""
    w = d.window
    peak = d.selected_peak or {}

    def finite(v: Any) -> float | None:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) else None

    whole = dict(xref="x", yref="paper", y0=0, y1=1)
    if w is not None and w.im_lo is not None and w.im_hi is not None:
        fig.add_shape(
            type="rect",
            x0=w.im_lo,
            x1=w.im_hi,
            fillcolor=theme.WINDOW,
            opacity=0.16,
            line=dict(width=0),
            layer="below",
            name="im-window",
            **whole,
        )
    if w is not None and w.im_pred_cal is not None:
        fig.add_shape(
            type="line",
            x0=w.im_pred_cal,
            x1=w.im_pred_cal,
            line=dict(color=theme.PREDICTION, dash="dot", width=1.5),
            name="im-pred",
            **whole,
        )
    apex = finite(peak.get("apex_im"))
    if apex is not None:
        fig.add_shape(
            type="line",
            x0=apex,
            x1=apex,
            line=dict(color=theme.APEX, width=1.6),
            name="apex-im",
            **whole,
        )
        if retrace_hw is not None:
            fig.add_shape(
                type="rect",
                x0=apex - retrace_hw,
                x1=apex + retrace_hw,
                line=dict(color=theme.INTEGRATION, dash="dash", width=1.2),
                fillcolor="rgba(0,0,0,0)",
                name="retrace-band",
                **whole,
            )
    ms1 = finite(peak.get("ms1_apex_im"))
    if ms1 is not None:
        fig.add_shape(
            type="line",
            x0=ms1,
            x1=ms1,
            xref="x2",
            yref="y2 domain",
            y0=0,
            y1=1,
            line=dict(color=theme.MS1[0], dash="dash", width=1.4),
            name="ms1-apex-im",
        )


# --------------------------------------------------------------------------- data


def get_mobilogram(
    rs: ResultSet, d: PrecursorDetail, source: str, centre_rt: float | None
) -> tuple[mob.Mobilogram, str | None]:
    """The mobilogram of ``d`` from ``source``, falling back from ``raw`` to the centroids
    (the second value then says why). Kept in a small LRU per page view."""
    rt = None if centre_rt is None else round(float(centre_rt), 4)
    key = (id(rs), d.run.name, int(d.candidate_id), int(d.selected_peak_rank), source, rt)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit, None
    why = None
    m = None
    if source == "raw":
        try:
            m = mob.engine_mobilogram(rs, d, centre_rt=centre_rt)
        except ViewerError as exc:
            why = str(exc)
    if m is None:
        m = mob.centroid_mobilogram(rs, d, centre_rt=centre_rt)
    if why is None:
        with _LOCK:
            _CACHE[key] = m
            while len(_CACHE) > _MAX:
                _CACHE.popitem(last=False)
    return m, why


def _note(m: mob.Mobilogram, k: int, centred_on_scan: bool, why: str | None) -> Any:
    n2, n1 = m.within(k)
    frames = [f.label for f in m.ms2_frames if abs(f.offset) <= k]
    span = "the centre frame" if k == 0 else f"the centre frame and {k} either side"
    parts: list[Any] = [
        html.B("Raw events" if m.source == "raw" else "Centroids"),
        f": {m.label}. ",
        f"Centred on {'the shown scan' if centred_on_scan else 'the apex'} at "
        f"{m.centre_rt:.2f} s; {span}: {n2} MS2 frame{'s' if n2 != 1 else ''} "
        f"({', '.join(frames)}), {n1} MS1 scan{'s' if n1 != 1 else ''}.",
    ]
    if m.bin is not None:
        parts.append(f" Bin width {m.bin:g}.")
    for n in m.notes:
        parts.append(f" {n[0].upper()}{n[1:]}.")
    if why:
        parts.append(html.Div(f"Raw profile not available ({why}); showing the centroids."))
    return parts


def render(
    rs: ResultSet,
    d: PrecursorDetail,
    frags: Sequence[Fragment],
    *,
    source: str,
    k: int,
    norm: str,
    hidden: Collection[int] = (),
    scheme: str = "light",
    centre_rt: float | None = None,
    centred_on_scan: bool = False,
) -> tuple[go.Figure, Any]:
    """The figure and the note under it, for the card and its callback."""
    try:
        m, why = get_mobilogram(rs, d, source, centre_rt)
    except ViewerError as exc:
        return _empty(f"No mobilogram: {exc}", scheme, HEIGHT), str(exc)
    fig = mobilogram_figure(
        m, frags, d, scheme, k=k, norm=norm, hidden=hidden, retrace_hw=retrace_half_width(rs)
    )
    return fig, _note(m, k, centred_on_scan, why)


# --------------------------------------------------------------------------- card


def mobilogram_card(
    ctx: PageContext,
    d: PrecursorDetail,
    frags: list[Fragment],
    *,
    centre_rt: float | None = None,
) -> Any:
    """The mobilogram card; None on a run without ion mobility. ``centre_rt`` is the RT
    of the scan the page opens on (the apex scan)."""
    if not d.run.has_im:
        return None
    ok, why = mob.engine_status(ctx.rs, d.run)
    source = "raw" if ok else "centroids"
    k = mob.FRAMES
    fig, note = render(
        ctx.rs, d, frags, source=source, k=k, norm="abs", scheme=ctx.scheme, centre_rt=centre_rt
    )
    seg = dict(size="xs", radius="xl")
    controls = dmc.Group(
        [
            dmc.Tooltip(
                dmc.SegmentedControl(
                    id="pd-mob-source",
                    data=[
                        {"value": "raw", "label": "Raw", "disabled": not ok},
                        {"value": "centroids", "label": "Centroids"},
                    ],
                    value=source,
                    **seg,
                ),
                label=(
                    "Raw: the .d's events binned in 1/K0 by mumdia mobilogram. Centroids: "
                    "the converted peaks, one 1/K0 each."
                    + ("" if ok else f" Raw is not available: {why}.")
                ),
                w=320,
                multiline=True,
            ),
            dmc.SegmentedControl(
                id="pd-mob-frames",
                data=[
                    {"value": "0", "label": "Frame"},
                    {"value": "1", "label": "±1"},
                    {"value": "2", "label": "±2"},
                ],
                value=str(k),
                **seg,
            ),
            dmc.SegmentedControl(
                id="pd-mob-norm",
                data=[
                    {"value": "abs", "label": "Absolute"},
                    {"value": "ion", "label": "Per ion"},
                ],
                value="abs",
                **seg,
            ),
        ],
        gap=6,
        wrap="wrap",
        justify="flex-end",
    )
    help_ = (
        "Intensity against 1/K0 for every predicted fragment (top, the XIC's colours) and the "
        "MS1 isotopes (bottom), in the frame of the scan the XIC shows and the frames either "
        "side. Raw sums the timsTOF events of the slots that isolate the precursor; "
        "Centroids are the converted peaks, one intensity-weighted 1/K0 per peak, so a "
        "profile is not drawn. Grey band: the 1/K0 window; dotted: the calibrated 1/K0 "
        "prediction; green: the apex 1/K0; dashed line (MS1): the MS1 apex 1/K0; dashed box: "
        "the band retrace summed in."
    )
    return section(
        "Mobilogram",
        graph("pd-mob", fig),
        html.Div(note, id="pd-mob-note", className="pd-mob-note"),
        right=controls,
        subtitle=(
            "Raw timsTOF events in 1/K0"
            if ok
            else "Convert centroids, one 1/K0 per peak, not a profile"
        ),
        help=help_,
        id="pd-mob-card",
    )


# --------------------------------------------------------------------------- callback


def update(
    rs: ResultSet,
    key: dict,
    scan: dict | None,
    source: str | None,
    frames: str | None,
    norm: str | None,
    hidden: list[int] | None,
    scheme: str,
    get_detail: Any,
) -> tuple[go.Figure, Any]:
    """The card's figure and note for the shown scan and the controls."""
    d = get_detail(rs, key.get("run", ""), int(key["cid"]))
    rt = (scan or {}).get("rt")
    centre = float(rt) if rt is not None else None
    peak = d.selected_peak or {}
    apex = peak.get("apex_rt", d.scored.get("apex_rt"))
    on_scan = centre is not None and apex is not None and abs(centre - float(apex)) > 1e-3
    try:
        k = int(frames) if frames is not None else mob.FRAMES
    except ValueError:
        k = mob.FRAMES
    k = k if k in FRAME_CHOICES else mob.FRAMES
    return render(
        rs,
        d,
        fragments_of(d.chromatogram),
        source=source or "centroids",
        k=k,
        norm=norm or "abs",
        hidden={int(h) for h in (hidden or [])},
        scheme=scheme,
        centre_rt=centre,
        centred_on_scan=on_scan,
    )


def register(app, get_rs, get_detail) -> None:
    """The card's callback: the shown scan (``pd-scan``), the source, the frames, the
    normalisation and the fragments hidden on the XIC redraw the figure."""

    @app.callback(
        Output(MOB, "figure", allow_duplicate=True),
        Output("pd-mob-note", "children"),
        Input("pd-scan", "data"),
        Input("pd-mob-source", "value"),
        Input("pd-mob-frames", "value"),
        Input("pd-mob-norm", "value"),
        Input("pd-hidden", "data"),
        State("pd-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _mobilogram(scan, source, frames, norm, hidden, key, scheme):
        if not key:
            return no_update, no_update
        return update(
            get_rs(),
            key,
            scan,
            source,
            frames,
            norm,
            hidden,
            "dark" if scheme == "dark" else "light",
            get_detail,
        )
