"""The ZT Scan parts of the precursor page (SCIEX ZT Scan DIA runs only).

A ZT Scan run sweeps the quadrupole in steps far narrower than its window, so each
fragment is seen in five to seven consecutive steps of every cycle, each at its own
transmission (:mod:`mumdia_viewer.data.ztscan`). The page then shows:

* on the XIC, one value per Q1 cycle (the transmission-weighted collapse of the cycle's
  steps, the estimator of the engine's ``quant.q1_fragments``), with the per-step samples
  faint behind it or alone, and each fragment marked with the engine's Q1 class
  (``q1f_frag_class``: belongs, interfered, absent) by its line dash and marker
  (:func:`zt_traces`, ``detail_figures.xic_figure``);
* the **Q1 sweep** card: each fragment's RT x Q1-step patch around the apex (small
  multiples, or one fragment beside the unfragmented precursor), with its expected Q1
  centre, the engine's fitted shared centre and its own centre as lines; and the Q1
  profile at the shown cycle (the apex cycle when the page opens): each fragment's
  intensity across the steps against the calibrated transmission, with the precursor's own
  isotopes in MS2.

The patch is read from the spectra after the page is shown (its callback), and the read is
started in the background while the page is built, so the card fills as soon as it can.
Every number the card computes says that the viewer derived it; the engine's own are the
``q1f_*`` columns of ``psms_extracted``, shown with their names.
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections import OrderedDict
from collections.abc import Collection, Sequence
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import ztscan as zt
from mumdia_viewer.data.detail import PrecursorDetail

from . import figures, theme
from .detail_figures import MODEBAR, Q1_CLASS_STYLE, Q1_CLASS_WORD, _empty, _r
from .detail_view import Fragment, fragments_of
from .state import PageContext
from .widgets import graph, section

PATCH = {"type": "fig", "name": "pd-zt-patch"}
PROFILE = {"type": "fig", "name": "pd-zt-prof"}
PATCH_HEIGHT = 470
PROFILE_HEIGHT = 380
CYCLE_CHOICES = (3, 6, 10)
MAX_PANELS = 16
SCALE = "Viridis"
# Marker lines of the patch, readable on the Viridis ramp: the expected centre, the
# engine's fitted shared centre and the fragment's own centre, and the apex cycles.
EXPECTED_LINE = "#ffffff"
FITTED_LINE = "#ff922b"
OWN_LINE = "#f783ac"
APEX_LINE = "rgba(255,255,255,0.85)"
CLASS_BADGE = {1: "teal", 2: "orange", 0: "gray"}

_TRACES: OrderedDict[tuple, tuple[Any, str | None]] = OrderedDict()
_LOCK = threading.Lock()
_MAX = 32


# --------------------------------------------------------------------------- data


def zt_traces(rs: ResultSet, d: PrecursorDetail) -> tuple[zt.CycleTraces | None, str | None]:
    """The candidate's cycle traces and None, or None and why not; (None, None) on a run
    that is not a ZT Scan run. Kept in a small LRU."""
    if not d.run.is_ztscan:
        return None, None
    key = (id(rs), d.run.name, int(d.candidate_id), int(d.selected_peak_rank))
    with _LOCK:
        hit = _TRACES.get(key)
        if hit is not None:
            _TRACES.move_to_end(key)
            return hit
    try:
        value: tuple[Any, str | None] = (zt.cycle_traces(rs, d), None)
    except ViewerError as exc:
        value = (None, str(exc))
    with _LOCK:
        _TRACES[key] = value
        while len(_TRACES) > _MAX:
            _TRACES.popitem(last=False)
    return value


def prefetch(rs: ResultSet, d: PrecursorDetail) -> None:
    """Start reading the candidate's patch in the background (the card's callback then
    finds it cached, or waits for the same read)."""

    def run() -> None:
        with contextlib.suppress(Exception):  # the card's callback reports a failure
            zt.q1_patch(rs, d)

    threading.Thread(target=run, name="mv-zt-prefetch", daemon=True).start()


def _scan_cycle(rs: ResultSet, d: PrecursorDetail, scan: dict | None) -> int | None:
    """The Q1 cycle of the shown scan (None when no scan is given or it is not in the
    sweep)."""
    row = (scan or {}).get("row")
    if row is None:
        return None
    zr = zt.ztscan_run(rs, d.run)
    if zr is None or not 0 <= int(row) < zr.sweep.cycle.size:
        return None
    return int(zr.sweep.cycle[int(row)])


# --------------------------------------------------------------------------- figures


def _shown(frags: Sequence[Fragment], hidden: Collection[int]) -> list[Fragment]:
    out = [f for f in frags if f.index not in hidden]
    if len(out) > MAX_PANELS:
        out = sorted(out, key=lambda f: -f.predicted)[:MAX_PANELS]
        order = {f.index: i for i, f in enumerate(frags)}
        out.sort(key=lambda f: order[f.index])
    return out


def _class_of(patch: zt.Q1Patch, f: int) -> int | None:
    if patch.classes is None or f >= patch.classes.size:
        return None
    return int(patch.classes[f])


def _title(f: Fragment, klass: int | None) -> str:
    tag = Q1_CLASS_STYLE.get(klass, ("", "", ""))[2] if klass is not None else ""
    word = Q1_CLASS_WORD.get(klass, "") if klass is not None else ""
    colour = f.colour
    return f"<span style='color:{colour}'><b>{f.html}</b></span>{tag} {word}".strip()


def _lines(
    patch: zt.Q1Patch, f: int | None, xref: str, yref: str, compact: bool
) -> list[dict[str, Any]]:
    """Vertical lines at the expected, fitted and own Q1 centres of ion ``f`` (None: the
    precursor M0), horizontal lines at the apex cycle and the 2D apex."""
    out: list[dict[str, Any]] = []
    step = patch.step_mz
    if f is None:
        centre = float(patch.precursor_centre[0])
        own = None
    else:
        centre = float(patch.expected_centre[f])
        own = None
        if patch.frag_offset_steps is not None and math.isfinite(patch.frag_offset_steps[f]):
            own = centre + float(patch.frag_offset_steps[f]) * step
    w = 1.2 if compact else 1.6

    def v(name: str, x: float, colour: str, dash: str) -> dict[str, Any]:
        return dict(
            type="line",
            name=name,
            x0=x,
            x1=x,
            xref=xref,
            y0=0,
            y1=1,
            yref=f"{yref} domain",
            line=dict(color=colour, width=w, dash=dash),
            layer="above",
        )

    out.append(v("expected", centre, EXPECTED_LINE, "solid"))
    if patch.fitted_centre_steps is not None:
        out.append(v("fitted", centre + patch.fitted_centre_steps * step, FITTED_LINE, "dash"))
    if own is not None:
        out.append(v("own", own, OWN_LINE, "dot"))
    for name, c, dash in (
        ("apex", patch.apex_cycle, "dashdot"),
        ("apex2d", patch.apex2d_cycle, "dot"),
    ):
        if c is None or (name == "apex2d" and c == patch.apex_cycle):
            continue
        ci = patch.cycle_index(c)
        if ci is None or not math.isfinite(patch.cycle_rt[ci]):
            continue
        y = float(patch.cycle_rt[ci])
        out.append(
            dict(
                type="line",
                name=name,
                x0=0,
                x1=1,
                xref=f"{xref} domain",
                y0=y,
                y1=y,
                yref=yref,
                line=dict(color=APEX_LINE, width=1.0, dash=dash),
                layer="above",
            )
        )
    return out


def _heat(
    patch: zt.Q1Patch,
    z: np.ndarray,
    name: str,
    scale_max: float,
    *,
    colour_axis: str = "coloraxis",
    meta: dict | None = None,
) -> go.Heatmap:
    rt = patch.cycle_rt
    y = np.where(np.isfinite(rt), rt, np.nan)
    cyc = np.repeat(patch.cycles[:, None], patch.targets.size, axis=1)
    raw = np.round(np.nan_to_num(z, nan=-1.0), 0)
    custom = np.stack([cyc, raw], axis=-1)
    zz = z / scale_max if scale_max > 0 else z
    return go.Heatmap(
        z=np.round(zz, 4),
        x=np.round(patch.targets, 4),
        y=np.round(y, 3),
        customdata=custom,
        coloraxis=colour_axis,
        hovertemplate=(
            f"<b>{name}</b><br>Q1 step target %{{x:.3f}} m/z<br>cycle %{{customdata[0]}}, RT "
            "%{y:.2f} s<br>intensity %{customdata[1]:,.0f}<extra></extra>"
        ),
        xgap=1,
        ygap=1,
        meta=meta or {},
        name=name,
    )


def patch_figure(
    patch: zt.Q1Patch,
    frags: Sequence[Fragment],
    scheme: str = "light",
    *,
    fragment: int | None = None,
    scale: str = "frag",
    hidden: Collection[int] = (),
) -> go.Figure:
    """The RT x Q1 patch: one heatmap per fragment (``fragment`` None), or fragment
    ``fragment`` beside the unfragmented precursor's M0. ``scale`` ``frag`` scales each
    heatmap to its own maximum, ``abs`` all to the largest.

    Lines: the expected Q1 centre (white), the engine's fitted shared centre (orange,
    dashed), the fragment's own centre (pink, dotted), the apex cycle (dash-dot) and the
    decision's 2D apex (dotted) when it differs. Cells without a spectrum are blank.
    """
    by_index = {f.index: f for f in frags}
    if fragment is not None and fragment in by_index:
        panels: list[tuple[str, Any, np.ndarray, int | None]] = [
            (
                _title(by_index[fragment], _class_of(patch, fragment)),
                by_index[fragment],
                patch.values[fragment],
                fragment,
            ),
            (
                "<b>precursor M0</b> (unfragmented, in MS2)",
                None,
                patch.precursor[0],
                None,
            ),
        ]
        cols, rows = 2, 1
    else:
        shown = _shown(frags, hidden)
        panels = [
            (_title(f, _class_of(patch, f.index)), f, patch.values[f.index], f.index)
            for f in shown
            if f.index < patch.values.shape[0]
        ]
        if not panels:
            return _empty("Every fragment is hidden on the XIC.", scheme, PATCH_HEIGHT)
        cols = 4 if len(panels) > 6 else 3 if len(panels) > 4 else len(panels)
        rows = math.ceil(len(panels) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[p[0] for p in panels],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.025 if cols > 2 else 0.06,
        vertical_spacing=min(0.12, 0.34 / max(rows, 1)) if rows > 1 else 0.1,
    )
    finite = [np.nanmax(p[2]) for p in panels if np.isfinite(p[2]).any()]
    gmax = float(max(finite)) if finite else 0.0
    shapes: list[dict[str, Any]] = []
    for i, (_, f, z, idx) in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        local = float(np.nanmax(z)) if np.isfinite(z).any() else 0.0
        smax = local if scale == "frag" else gmax
        name = f.text if f is not None else "precursor M0"
        meta = {"frag": idx, "op": 1.0} if idx is not None else {}
        fig.add_trace(_heat(patch, z.astype(np.float64), name, smax, meta=meta), row=r, col=c)
        k = i + 1
        xref = "x" if k == 1 else f"x{k}"
        yref = "y" if k == 1 else f"y{k}"
        shapes += _lines(patch, idx, xref, yref, compact=len(panels) > 2)
    for s in shapes:
        fig.add_shape(**s)
    title = "fraction of each fragment's maximum" if scale == "frag" else "intensity"
    figures.themed(
        fig,
        scheme,
        height=PATCH_HEIGHT if rows > 1 else 380,
        margin=dict(l=64, r=20, t=40, b=48),
        coloraxis=dict(
            colorscale=SCALE,
            cmin=0,
            cmax=1 if scale == "frag" else max(gmax, 1.0),
            colorbar=dict(
                title=dict(text=title, side="right", font=dict(size=10)),
                thickness=10,
                len=0.8,
                tickformat="~s" if scale != "frag" else ".1f",
            ),
        ),
        modebar=MODEBAR,
        uirevision="pd-zt-patch",
        meta={"kind": "q1patch"},
    )
    for a in fig.layout.annotations:
        a.font = dict(size=11)
    fig.update_xaxes(tickformat=".1f", nticks=4, showgrid=False)
    fig.update_yaxes(tickformat=".0f", nticks=5, showgrid=False)
    for c in range(1, cols + 1):
        fig.update_xaxes(
            title_text="Q1 step target (m/z)", row=rows, col=c, title_font=dict(size=10)
        )
    fig.update_yaxes(title_text="RT (s)", row=1, col=1, title_font=dict(size=10))
    if rows > 1:
        for r in range(2, rows + 1):
            fig.update_yaxes(title_text="RT (s)", row=r, col=1, title_font=dict(size=10))
    return fig


def profile_figure(
    patch: zt.Q1Patch,
    prof: zt.Q1Profile,
    frags: Sequence[Fragment],
    scheme: str = "light",
    *,
    norm: str = "norm",
    hidden: Collection[int] = (),
    title: str = "",
) -> go.Figure:
    """Each fragment's intensity across the Q1 steps of ``prof`` against the calibrated
    transmission.

    ``norm`` ``norm``: x is the step target against the ion's own expected centre, in
    steps, and y the intensity over the ion's maximum, so an ion where the model puts it
    follows the grey curve (the precursor's isotope envelope through the calibrated window,
    peak 1; dashed: at the engine's fitted shared centre and width) and an ion centred
    elsewhere peaks away from 0. ``abs``: the raw intensities against the step target
    (m/z), each fragment's expected profile scaled by its least-squares amplitude as a
    dotted line. The unfragmented precursor's isotopes in MS2 are black diamonds.
    """
    fig = go.Figure()
    step = patch.step_mz
    shown = [f for f in frags if f.index not in hidden and f.index < prof.observed.shape[0]]
    ink = "#212529" if scheme != "dark" else "#e9ecef"
    reach = (patch.cand.model.reach_mz / step) + 1.5
    if norm == "norm":
        u = np.linspace(-reach, reach, 241)
        env = patch.cand.envelope(patch.centre_mz + u * step)
        env = env / env.max() if env.max() > 0 else env
        fig.add_trace(
            go.Scatter(
                x=_r(u, 3),
                y=_r(env, 4),
                mode="lines",
                name="calibrated transmission",
                line=dict(color=theme.PREDICTION, width=4),
                opacity=0.55,
                hovertemplate=(
                    "calibrated transmission at the expected centre<br>%{x:.2f} steps: "
                    "%{y:.2f}<extra></extra>"
                ),
            )
        )
        if patch.fitted_centre_steps is not None:
            om = patch.width if patch.width else 1.0
            envf = patch.cand.envelope(patch.centre_mz + u * step, patch.fitted_centre_steps, om)
            envf = envf / envf.max() if envf.max() > 0 else envf
            fig.add_trace(
                go.Scatter(
                    x=_r(u, 3),
                    y=_r(envf, 4),
                    mode="lines",
                    name="engine's fit (q1f_centre, q1f_width)",
                    line=dict(color=FITTED_LINE, width=2, dash="dash"),
                    hovertemplate=(
                        f"at the fitted shared centre {patch.fitted_centre_steps:+.2f} steps, "
                        f"width x{om:g}<br>%{{x:.2f}} steps: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )
    for f in shown:
        k = f.index
        amp = float(prof.amplitude[k])
        obs = prof.observed[k]
        if not np.isfinite(obs).any() or np.nanmax(obs) <= 0:
            continue
        klass = _class_of(patch, k)
        dash, symbol, tag = Q1_CLASS_STYLE.get(klass, ("solid", "circle", ""))
        word = Q1_CLASS_WORD.get(klass, "not recorded") if klass is not None else "not recorded"
        top = float(np.nanmax(obs))
        if norm == "norm":
            x = (prof.targets - patch.expected_centre[k]) / step
            y = obs / top
        else:
            x = prof.targets
            y = obs
        custom = np.stack([prof.targets, np.nan_to_num(obs, nan=0.0)], axis=1)
        fig.add_trace(
            go.Scatter(
                x=_r(x, 3),
                y=_r(y, 4 if norm == "norm" else 1),
                customdata=np.round(custom, 3),
                mode="lines+markers",
                name=f"{f.html}{tag}",
                legendgroup=f"f{k}",
                line=dict(color=f.colour, width=1.6, dash=dash),
                marker=dict(size=7, color=f.colour, symbol=symbol, line=dict(width=1.2)),
                meta={"frag": k, "op": 1.0},
                hovertemplate=(
                    f"<b>{f.html}</b> ({word})<br>step target %{{customdata[0]:.3f}} m/z<br>"
                    "intensity %{customdata[1]:,.0f}"
                    + (
                        f"<br>/ its maximum {top:,.0f} = %{{y:.2f}}<br>%{{x:+.2f}} steps from "
                        "its expected centre"
                        if norm == "norm"
                        else ""
                    )
                    + "<extra></extra>"
                ),
            )
        )
        if norm != "norm" and amp > 0:
            fig.add_trace(
                go.Scatter(
                    x=_r(prof.targets, 3),
                    y=_r(prof.expected[k] * amp, 1),
                    mode="lines",
                    legendgroup=f"f{k}",
                    showlegend=False,
                    line=dict(color=f.colour, width=1, dash="dot"),
                    meta={"frag": k, "op": 0.7},
                    opacity=0.7,
                    hoverinfo="skip",
                )
            )
    # The unfragmented precursor's isotopes in MS2.
    pshown = False
    for j, name in enumerate(patch.precursor_names):
        obs = prof.precursor_observed[j]
        if not np.isfinite(obs).any() or np.nanmax(obs) <= 0:
            continue
        if norm == "norm":
            x = (prof.targets - patch.precursor_centre[j]) / step
            y = obs / float(np.nanmax(obs))
        else:
            x, y = prof.targets, obs
        custom = np.stack([prof.targets, np.nan_to_num(obs, nan=0.0)], axis=1)
        fig.add_trace(
            go.Scatter(
                x=_r(x, 3),
                y=_r(y, 4 if norm == "norm" else 1),
                customdata=np.round(custom, 3),
                mode="lines+markers",
                name="precursor in MS2" if not pshown else name,
                legendgroup="precursor",
                showlegend=not pshown,
                line=dict(color=ink, width=1.2, dash="dot"),
                marker=dict(size=7, color=ink, symbol="diamond-open" if j else "diamond"),
                hovertemplate=(
                    f"<b>precursor {name}</b> (m/z {patch.precursor_mz[j]:.4f}, unfragmented, "
                    "in MS2)<br>step target %{customdata[0]:.3f} m/z<br>intensity "
                    "%{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )
        pshown = True
    shapes = []
    if norm == "norm":
        shapes.append(
            dict(
                type="line",
                name="expected",
                x0=0,
                x1=0,
                xref="x",
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=theme.PREDICTION, width=1.2),
            )
        )
        if patch.fitted_centre_steps is not None:
            x = patch.fitted_centre_steps
            shapes.append(
                dict(
                    type="line",
                    name="fitted",
                    x0=x,
                    x1=x,
                    xref="x",
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=FITTED_LINE, width=1.2, dash="dash"),
                )
            )
    figures.themed(
        fig,
        scheme,
        height=PROFILE_HEIGHT,
        margin=dict(l=60, r=20, t=34, b=46),
        modebar=MODEBAR,
        hovermode="closest",
        uirevision="pd-zt-prof",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=10)),
        meta={"kind": "q1profile", "norm": norm},
    )
    for s in shapes:
        fig.add_shape(**s)
    if title:
        fig.add_annotation(
            text=title,
            xref="paper",
            yref="paper",
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=10, color=theme.PREDICTION),
        )
    if norm == "norm":
        fig.update_xaxes(title_text="Q1 step target against the ion's expected centre (steps)")
        fig.update_yaxes(title_text="intensity / the ion's maximum", range=[0, 1.12])
    else:
        fig.update_xaxes(title_text="Q1 step target (m/z)")
        fig.update_yaxes(title_text="intensity", tickformat="~s", rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- text and tables


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    if abs(f) >= 1000 or (f == int(f) and abs(f) < 1e6):
        return f"{f:,.0f}" if f == int(f) else f"{f:,.1f}"
    return f"{f:.{digits}g}"


def decision_table(
    patch: zt.Q1Patch, prof: zt.Q1Profile, frags: Sequence[Fragment], row: dict | None
) -> Any:
    """Each fragment's Q1 decision (the engine's lists) and its amplitude at the shown
    cycles (viewer-derived)."""
    nf = len(patch.names)
    off = patch.frag_offset_steps
    resid = zt._list(row, "q1f_frag_resid", nf)
    corr = zt._list(row, "q1f_frag_corr2d", nf)
    s_r2 = zt._list(row, "q1s_frag_r2", nf)
    s_if = zt._list(row, "q1s_frag_interf", nf)
    sep = s_r2 is not None or s_if is not None

    def num(v: np.ndarray | None, k: int, fmt: str) -> Any:
        return dmc.TableTd(
            format(v[k], fmt) if v is not None and math.isfinite(v[k]) else "",
            className="mv-num",
        )

    def th(text: Any, tip: str, numeric: bool = False) -> Any:
        return dmc.TableTh(
            html.Span(text, className="pd-th-tip", **{"data-tip": tip}),
            className="mv-num" if numeric else None,
        )

    head = dmc.TableThead(
        dmc.TableTr(
            [
                th("fragment", "the library fragment (chromatogram row)"),
                th("Q1 class", "q1f_frag_class of psms_extracted: the engine's decision"),
                th(
                    "own centre",
                    "q1f_frag_offset: the fragment's own Q1 centre against its expected one, "
                    "in steps (the engine's)",
                    True,
                ),
                th(
                    "1 - R²",
                    "q1f_frag_resid: its residual under the shared fit (the engine's)",
                    True,
                ),
                th("2D corr.", "q1f_frag_corr2d: its 2D correlation (the engine's)", True),
                *(
                    [
                        th(
                            "model R²",
                            "q1s_frag_r2: R² of its RT x Q1 cells under its own component of "
                            "the engine's separable model",
                            True,
                        ),
                        th(
                            "interf.",
                            "q1s_frag_interf: the share of its patch intensity above its own "
                            "component (the engine's)",
                            True,
                        ),
                    ]
                    if sep
                    else []
                ),
                th(
                    ["amplitude", html.Br(), html.Span("derived", className="pd-th-derived")],
                    "Viewer-derived: sum(w x)/sum(w^2) over the shown cycles' steps under the "
                    "fragment's calibrated Q1 profile, its intensity at full transmission",
                    True,
                ),
            ]
        )
    )
    body = []
    for f in frags:
        k = f.index
        if k >= nf:
            continue
        klass = _class_of(patch, k)
        badge = (
            dmc.Badge(Q1_CLASS_WORD[klass], color=CLASS_BADGE[klass], variant="light", size="sm")
            if klass in Q1_CLASS_WORD
            else ""
        )
        name = [html.Span(className="pd-swatch", style={"background": f.colour}), f.text]
        body.append(
            dmc.TableTr(
                [
                    dmc.TableTd(html.Span(name, className="pd-fname")),
                    dmc.TableTd(badge),
                    dmc.TableTd(
                        f"{off[k]:+.2f}" if off is not None and math.isfinite(off[k]) else "",
                        className="mv-num",
                    ),
                    dmc.TableTd(
                        f"{resid[k]:.2f}" if resid is not None and math.isfinite(resid[k]) else "",
                        className="mv-num",
                    ),
                    dmc.TableTd(
                        f"{corr[k]:.2f}" if corr is not None and math.isfinite(corr[k]) else "",
                        className="mv-num",
                    ),
                    *([num(s_r2, k, ".2f"), num(s_if, k, ".2f")] if sep else []),
                    dmc.TableTd(f"{prof.amplitude[k]:,.0f}", className="mv-num"),
                ],
                className="pd-frow",
                **{"data-frag": str(k)},
            )
        )
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        verticalSpacing=4,
        horizontalSpacing=6,
        className="mv-table pd-ftable",
    )
    return dmc.TableScrollContainer(table, minWidth=420)


def summary(row: dict | None, model: zt.Q1Model | None) -> Any:
    """The engine's per-candidate Q1 columns, by name."""
    items = zt.summary_items(row)
    cells = [
        dmc.Tooltip(
            html.Span(
                [
                    html.Span(col, className="pd-key-label"),
                    html.Span(_fmt(v), className="pd-key-value"),
                ],
                className="pd-key",
            ),
            label=f"{col} (psms_extracted): {text}",
            w=280,
            multiline=True,
        )
        for col, text, v in items
    ]
    if model is not None and model.tolerances:
        t = model.tolerances
        cells.append(
            dmc.Tooltip(
                html.Span(
                    [
                        html.Span("tolerances", className="pd-key-label"),
                        html.Span(
                            f"offset {t['offset_steps']:.2f} · centre "
                            f"{t['centre_steps']:.2f} steps · corr {t['corr2d_min']:.2f}",
                            className="pd-key-value",
                        ),
                    ],
                    className="pd-key",
                ),
                label=(
                    "The decision's tolerances learned from the seed (" + model.source + "): a "
                    "fragment belongs when its own centre is within the offset tolerance of the "
                    "shared centre (clamped to the centre tolerance of the expected one) and its "
                    "2D correlation reaches the floor."
                ),
                w=320,
                multiline=True,
            )
        )
    if not cells:
        return None
    return html.Div(cells, className="pd-keys")


def _profile_title(prof: zt.Q1Profile, patch: zt.Q1Patch, shown_cycle: int | None) -> str:
    c = prof.cycles
    what = f"Q1 cycle {int(c[0])}" if c.size == 1 else f"Q1 cycles {int(c[0])}-{int(c[-1])} summed"
    tags = []
    if shown_cycle is not None and int(c[c.size // 2]) == shown_cycle:
        tags.append("the shown scan's cycle")
    if patch.apex_cycle in c.tolist():
        tags.append("apex cycle" if c.size == 1 else "around the apex")
    if (
        patch.apex2d_cycle is not None
        and patch.apex2d_cycle != patch.apex_cycle
        and patch.apex2d_cycle in c.tolist()
    ):
        tags.append("2D apex")
    return what + (f" ({', '.join(tags)})" if tags else "")


def _note(patch: zt.Q1Patch, zr: zt.ZtRun | None) -> Any:
    r = patch.read
    parts: list[Any] = [
        html.B("Viewer read"),
        f": {patch.n_cycles} cycles x {patch.targets.size} Q1 steps "
        f"({patch.targets[0]:.2f}-{patch.targets[-1]:.2f} m/z), {r.cells} spectra"
        + (f" ({r.cached} cached)" if r.cached else "")
        + f", {r.row_groups} row group{'s' if r.row_groups != 1 else ''} of spectra_ms2 in "
        f"{r.ms / 1000:.2f} s. Each cell is the nearest peak within {patch.tol_ppm:.3g} ppm of "
        "the offset-corrected fragment m/z (the engine's patch rule). ",
    ]
    if zr is not None and zr.model is not None:
        parts.append(
            f"Expected centres from the {zr.model.label}; the fitted centre, the own centres "
            "and the classes are the engine's q1f_* columns. "
        )
    for n in patch.notes:
        parts.append(f"{n[0].upper()}{n[1:]}. ")
    return parts


# --------------------------------------------------------------------------- card


def zt_card(ctx: PageContext, d: PrecursorDetail, frags: list[Fragment]) -> Any:
    """The Q1 sweep card; None on any run that is not a ZT Scan run. The figures are
    filled by the card's callback (:func:`update`)."""
    if not d.run.is_ztscan:
        return None
    zr = zt.ztscan_run(ctx.rs, d.run)
    if zr is None:
        return None
    if zr.model is not None and d.tolerance is not None and d.apex_scan is not None:
        prefetch(ctx.rs, d)
    seg = dict(size="xs", radius="xl")
    options = [{"value": "all", "label": "All fragments"}] + [
        {"value": str(f.index), "label": f.text} for f in frags
    ]
    controls = dmc.Group(
        [
            dmc.Select(
                id="pd-zt-frag",
                data=options,
                value="all",
                size="xs",
                w=150,
                allowDeselect=False,
                comboboxProps={"withinPortal": True},
            ),
            dmc.Tooltip(
                dmc.SegmentedControl(
                    id="pd-zt-cycles",
                    data=[{"value": str(k), "label": f"±{k}"} for k in CYCLE_CHOICES],
                    value=str(CYCLE_CHOICES[0]),
                    **seg,
                ),
                label=(
                    "Q1 cycles read either side of the apex (or of the shown scan's cycle); "
                    "the engine's decision reads ±3."
                ),
                w=260,
                multiline=True,
            ),
            dmc.SegmentedControl(
                id="pd-zt-scale",
                data=[
                    {"value": "frag", "label": "Per fragment"},
                    {"value": "abs", "label": "Shared"},
                ],
                value="frag",
                **seg,
            ),
        ],
        gap=6,
        wrap="wrap",
        justify="flex-end",
    )
    profile_controls = dmc.Group(
        [
            dmc.Tooltip(
                dmc.SegmentedControl(
                    id="pd-zt-span",
                    data=[
                        {"value": "0", "label": "Cycle"},
                        {"value": "2", "label": "±2 cycles"},
                    ],
                    value="0",
                    **seg,
                ),
                label=(
                    "Cycle: the shown scan's cycle (the apex cycle when the page opens). ±2: "
                    "summed over two cycles either side, as the engine's decision sums its "
                    "profiles around the 2D apex."
                ),
                w=280,
                multiline=True,
            ),
            dmc.SegmentedControl(
                id="pd-zt-norm",
                data=[
                    {"value": "norm", "label": "Scaled"},
                    {"value": "abs", "label": "Intensity"},
                ],
                value="norm",
                **seg,
            ),
        ],
        gap=6,
        justify="flex-end",
    )
    wait = _empty("Reading the Q1 sweep ...", ctx.scheme, PATCH_HEIGHT)
    wait_p = _empty("Reading the Q1 sweep ...", ctx.scheme, PROFILE_HEIGHT)
    model_text = (
        zr.model.label if zr.model is not None else f"no Q1 model recorded ({zr.model_note})"
    )
    help_ = (
        "Top: each fragment's intensity over the Q1 steps (x, the step's target m/z) and the "
        "Q1 cycles (y, RT) around the apex, as the engine's per-fragment decision reads it. A "
        "fragment that belongs lights up at its expected Q1 centre (white line), where the "
        "calibrated window transmits the precursor; an interfered one is centred elsewhere, "
        "where the precursor it really comes from is. Orange dashed: the engine's fitted "
        "shared centre; pink dotted: the fragment's own centre; horizontal lines: the apex "
        "cycle (dash-dot) and the decision's 2D apex (dotted). Bottom: the Q1 profile at the "
        "shown cycle, each fragment scaled to its maximum and placed against its own expected "
        "centre, over the calibrated transmission (grey), with the unfragmented precursor's "
        "isotopes in MS2 (diamonds)."
    )
    return section(
        "Q1 sweep (ZT Scan)",
        summary(zt.q1f_row(ctx.rs, d), zr.model),
        dcc.Loading(graph("pd-zt-patch", wait), type="dot", delay_show=300),
        html.Div(
            [
                html.Div(
                    dmc.Text("Q1 profile", size="xs", fw=600, tt="uppercase", c="dimmed"),
                    className="pd-zt-prof-title",
                ),
                profile_controls,
            ],
            className="pd-zt-prof-head",
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
        ),
        dcc.Loading(graph("pd-zt-prof", wait_p), type="dot", delay_show=300),
        html.Div(id="pd-zt-table"),
        html.Div(id="pd-zt-note", className="pd-mob-note"),
        right=controls,
        subtitle=f"{zr.method_label}; {model_text}",
        help=help_,
        id="pd-zt-card",
    )


# --------------------------------------------------------------------------- callback


def render(
    rs: ResultSet,
    d: PrecursorDetail,
    *,
    scan: dict | None = None,
    fragment: str | None = "all",
    cycles: str | None = None,
    scale: str | None = "frag",
    span: str | None = "0",
    norm: str | None = "norm",
    hidden: Collection[int] = (),
    scheme: str = "light",
) -> tuple[go.Figure, go.Figure, Any, Any]:
    """The card's patch figure, profile figure, decision table and note."""
    frags = fragments_of(d.chromatogram)
    try:
        k = int(cycles) if cycles is not None else CYCLE_CHOICES[0]
    except ValueError:
        k = CYCLE_CHOICES[0]
    k = k if k in CYCLE_CHOICES else CYCLE_CHOICES[0]
    shown_cycle = _scan_cycle(rs, d, scan)
    try:
        patch = zt.q1_patch(rs, d, half_cycles=k)
        if shown_cycle is not None and patch.cycle_index(shown_cycle) is None:
            patch = zt.q1_patch(rs, d, half_cycles=k, centre_cycle=shown_cycle)
    except ViewerError as exc:
        msg = f"No Q1 patch: {exc}"
        return _empty(msg, scheme, PATCH_HEIGHT), _empty(msg, scheme, PROFILE_HEIGHT), None, msg
    try:
        half = int(span) if span is not None else 0
    except ValueError:
        half = 0
    prof = zt.q1_profile(patch, cycle=shown_cycle, half=max(half, 0))
    sel = None
    if fragment not in (None, "all"):
        try:
            sel = int(fragment)
        except ValueError:
            sel = None
    off = {int(h) for h in hidden}
    pfig = patch_figure(patch, frags, scheme, fragment=sel, scale=scale or "frag", hidden=off)
    rfig = profile_figure(
        patch,
        prof,
        frags,
        scheme,
        norm=norm or "norm",
        hidden=off,
        title=_profile_title(prof, patch, shown_cycle),
    )
    table = decision_table(patch, prof, frags, zt.q1f_row(rs, d))
    return pfig, rfig, table, _note(patch, zt.ztscan_run(rs, d.run))


def update(
    rs: ResultSet,
    key: dict,
    scan: dict | None,
    fragment: str | None,
    cycles: str | None,
    scale: str | None,
    span: str | None,
    norm: str | None,
    hidden: list[int] | None,
    scheme: str,
    get_detail: Any,
) -> tuple[go.Figure, go.Figure, Any, Any]:
    """The card's outputs for the page's candidate, the shown scan and the controls."""
    d = get_detail(rs, key.get("run", ""), int(key["cid"]))
    return render(
        rs,
        d,
        scan=scan,
        fragment=fragment,
        cycles=cycles,
        scale=scale,
        span=span,
        norm=norm,
        hidden=hidden or (),
        scheme=scheme,
    )


def register(app, get_rs, get_detail) -> None:
    """The card's callback: it fills the card when the page is shown and follows the
    shown scan (``pd-scan``), the controls and the fragments hidden on the XIC."""

    @app.callback(
        Output(PATCH, "figure"),
        Output(PROFILE, "figure"),
        Output("pd-zt-table", "children"),
        Output("pd-zt-note", "children"),
        Input("pd-key", "data"),
        Input("pd-scan", "data"),
        Input("pd-zt-frag", "value"),
        Input("pd-zt-cycles", "value"),
        Input("pd-zt-scale", "value"),
        Input("pd-zt-span", "value"),
        Input("pd-zt-norm", "value"),
        Input("pd-hidden", "data"),
        State("scheme", "data"),
    )
    def _zt(key, scan, fragment, cycles, scale, span, norm, hidden, scheme):
        if not key:
            return no_update, no_update, no_update, no_update
        try:
            return update(
                get_rs(),
                key,
                scan,
                fragment,
                cycles,
                scale,
                span,
                norm,
                hidden,
                "dark" if scheme == "dark" else "light",
                get_detail,
            )
        except (ViewerError, KeyError, ValueError) as exc:
            msg = f"No Q1 patch: {exc}"
            return no_update, no_update, None, msg
