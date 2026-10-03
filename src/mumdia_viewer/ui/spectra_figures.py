"""Plotly figures of the spectrum browser (pure functions of data-layer results).

The spectrum is a stick plot of the peaks as convert stored them, in % of the base peak
(or of the highest matched peak), with the most intense peaks labelled with their m/z.
A selected candidate adds its library fragments: matched peaks in the fragment's colour
with the fragment name and the raw ppm error, and the library's fragments mirrored
below (the precursor page's convention). In an MS1 scan a selected candidate adds its
precursor isotopes instead.

Trace order of the spectrum is fixed: 0 the stems, 1 the peak tops (hover, and the data
the browser relabels from), 2 the m/z labels. ``meta.kind`` names them for spectra.js.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from mumdia_viewer.data.scans import FragmentOverlay, IsotopeOverlay, top_peaks
from mumdia_viewer.data.spectra import Spectrum

from . import theme
from .figures import empty_figure, themed

SPECTRUM_HEIGHT = 400
NAV_HEIGHT = 112
NAV_MARGIN = {"l": 52, "r": 14, "t": 6, "b": 24}
CLIP = 110.0
OBSERVED = "rgba(134, 142, 150, 0.55)"
LEVEL_COLOURS = {1: theme.MS1[0], 2: theme.TARGET}
WINDOW_FILL = "rgba(240, 140, 0, 0.16)"
SCAN_LINE = "#f08c00"
BAND_FILL = "rgba(66, 99, 235, 0.10)"
NAV_BARS = "rgba(47, 158, 68, 0.75)"
TIC_LINE = "rgba(134, 142, 150, 0.85)"
TIC_FILL = "rgba(134, 142, 150, 0.14)"
ISOTOPE = theme.MS1[0]
LABEL_FONT = 10
ISOTOPE_NAMES = {-1: "M-1", 0: "M", 1: "M+1", 2: "M+2"}


def _r(values: Any, digits: int) -> np.ndarray:
    return np.round(np.asarray(values, dtype=np.float64), digits)


def _stems(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segments from 0 to each y as one trace (NaN separates the stems)."""
    n = int(np.asarray(x).size)
    xs = np.empty(3 * n, dtype=np.float64)
    ys = np.empty(3 * n, dtype=np.float64)
    xs[0::3], xs[1::3], xs[2::3] = x, x, np.nan
    ys[0::3], ys[1::3], ys[2::3] = 0.0, y, np.nan
    return xs, ys


def label_heights(xs: Sequence[float], ys: Sequence[float], span: float, top: float) -> list[float]:
    """Positions of labels so that near labels do not overlap: a label moves up a tier."""
    width = span * 0.06
    tier = 7.0
    placed: list[tuple[float, float]] = []
    out = []
    for x, y in zip(xs, ys, strict=True):
        h = min(float(y), CLIP) + 3.0
        while any(abs(x - px) < width and abs(h - py) < tier for px, py in placed) and h < top:
            h += tier
        placed.append((x, h))
        out.append(h)
    return out


def _mz_labels(
    mz: np.ndarray, rel: np.ndarray, n: int, lo: float, hi: float, skip: set[int]
) -> tuple[list[float], list[float], list[str]]:
    """The top-``n`` peaks in [lo, hi] (not in ``skip``) as label positions and texts."""
    idx = [int(i) for i in top_peaks(mz, rel, n + len(skip), lo=lo, hi=hi) if int(i) not in skip]
    idx = sorted(idx[:n], key=lambda i: mz[i])
    xs = [float(mz[i]) for i in idx]
    heights = label_heights(xs, [float(min(rel[i], CLIP)) for i in idx], hi - lo, 150.0)
    return xs, [round(h, 1) for h in heights], [f"{mz[i]:.2f}" for i in idx]


def spectrum_figure(
    spec: Spectrum | None,
    scheme: str = "light",
    *,
    n_labels: int = 10,
    overlay: FragmentOverlay | None = None,
    frags: Sequence[Any] = (),
    isotopes: IsotopeOverlay | None = None,
    scale: str = "base",
    precursor_mz: float | None = None,
    candidate: int | None = None,
    why_empty: str = "No scan to show.",
) -> go.Figure:
    """One scan as a stick plot; see the module docstring.

    ``frags`` are the precursor page's fragments (:func:`.detail_view.fragments_of`):
    their colours and names. ``scale="matched"`` draws the observed peaks in % of the
    highest matched peak when the overlay has matches; ``scale="window"`` in % of the
    highest peak outside the isolation window of an MS2 scan (the unfragmented precursor
    is often the base peak); with MS1 isotopes, in % of the highest peak of the shown
    precursor region. Taller peaks are clipped at 110 %.
    """
    if spec is None:
        return empty_figure(why_empty, scheme, height=SPECTRUM_HEIGHT)
    mz = np.asarray(spec.mz, dtype=np.float64)
    inten = np.asarray(spec.intensity, dtype=np.float64)
    if mz.size == 0:
        return empty_figure("This scan has no peaks.", scheme, height=SPECTRUM_HEIGHT)
    by_index = {f.index: f for f in frags}
    matches = [
        m for m in (overlay.matches if overlay is not None else []) if m.fragment in by_index
    ]
    base = float(inten.max()) if inten.size else 1.0
    top_match = max((m.obs_intensity for m in matches), default=0.0)
    by_match = scale == "matched" and top_match > 0
    ref = top_match if by_match else base
    what = "the highest matched peak" if by_match else "the base peak"
    if scale == "window" and spec.level == 2 and spec.window_lower is not None:
        outside = (mz < float(spec.window_lower)) | (mz > float(spec.window_upper))
        if outside.any() and float(inten[outside].max()) > 0:
            ref = float(inten[outside].max())
            what = "the highest peak outside the isolation window"
    if scale == "window" and isotopes is not None and spec.level == 1:
        # The precursor's region of an MS1 scan: % of the highest peak shown around it.
        a, b = isotopes.rows[0]["mz"] - 2.5, isotopes.rows[-1]["mz"] + 2.5
        near = (mz >= a) & (mz <= b)
        if near.any() and float(inten[near].max()) > 0:
            ref = float(inten[near].max())
            what = f"the highest peak from {a:.1f} to {b:.1f} m/z"
    ref = ref if ref > 0 else 1.0
    rel = 100.0 * inten / ref
    shown = np.minimum(rel, CLIP)
    mirror = overlay is not None
    colour = (
        OBSERVED
        if (mirror or isotopes is not None)
        else LEVEL_COLOURS.get(spec.level, theme.TARGET)
    )

    lo, hi = float(mz.min()), float(mz.max())
    if overlay is not None and overlay.n_library:
        theo = overlay.fragments["theo_mz"].to_numpy(dtype=np.float64)
        lo, hi = min(lo, float(theo.min())), max(hi, float(theo.max()))
    pad = max(1.0, 0.02 * (hi - lo))
    x_range = [lo - pad, hi + pad]
    view = (lo - pad, hi + pad)
    if isotopes is not None:
        first, last = isotopes.rows[0]["mz"], isotopes.rows[-1]["mz"]
        view = (first - 2.5, last + 2.5)

    fig = go.Figure()
    xs, ys = _stems(_r(mz, 4), _r(shown, 2))
    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=colour, width=1),
            hoverinfo="skip",
            showlegend=False,
            name="peaks",
            meta={"kind": "stems"},
        )
    )
    fig.add_trace(
        # Plain lists (not typed arrays): spectra.js reads them to relabel after a zoom.
        go.Scattergl(
            x=_r(mz, 4).tolist(),
            y=_r(shown, 2).tolist(),
            customdata=np.column_stack([_r(inten, 0), _r(rel, 2)]).tolist(),
            mode="markers",
            marker=dict(size=5, color=colour, opacity=0),
            hovertemplate=(
                "m/z %{x:.4f}<br>intensity %{customdata[0]:,.0f}"
                f"<br>%{{customdata[1]:.1f}}% of {what}<extra></extra>"
            ),
            showlegend=False,
            name="peak tops",
            meta={"kind": "peaks"},
        )
    )
    matched_peaks = {m.peak for m in matches}
    lx, ly, lt = _mz_labels(mz, rel, int(n_labels), view[0], view[1], matched_peaks)
    fig.add_trace(
        go.Scatter(
            x=lx,
            y=ly,
            text=lt,
            mode="text",
            textposition="top center",
            textfont=dict(size=LABEL_FONT),
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="m/z labels",
            meta={"kind": "labels", "n": int(n_labels)},
        )
    )
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    top = 150.0 if mirror else 125.0
    if spec.level == 2 and spec.window_lower is not None and spec.window_upper is not None:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=float(spec.window_lower),
                x1=float(spec.window_upper),
                y0=0,
                y1=top,
                fillcolor=WINDOW_FILL,
                line=dict(width=0),
                layer="below",
            )
        )
        annotations.append(
            dict(
                x=(float(spec.window_lower) + float(spec.window_upper)) / 2,
                y=top,
                xref="x",
                yref="y",
                text="isolation window",
                showarrow=False,
                yanchor="top",
                font=dict(size=9, color=SCAN_LINE),
            )
        )
    if precursor_mz is not None and np.isfinite(precursor_mz) and spec.level == 2:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=float(precursor_mz),
                x1=float(precursor_mz),
                y0=0,
                y1=top - 14,
                line=dict(color=theme.INTEGRATION, width=1.2, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=float(precursor_mz),
                y=top - 14,
                xref="x",
                yref="y",
                text="precursor m/z",
                showarrow=False,
                xanchor="left",
                xshift=3,
                yanchor="middle",
                font=dict(size=9, color=theme.INTEGRATION),
            )
        )
    if overlay is not None:
        _add_fragments(fig, overlay, frags, matches, ref, x_range)
    if isotopes is not None:
        _add_isotopes(fig, isotopes, mz, rel, shown, ref)
    if mirror:
        annotations += [
            dict(
                text=f"observed · % of {what}"
                + (" · taller peaks clipped" if (rel > CLIP).any() else ""),
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.0,
                xanchor="left",
                yanchor="bottom",
                yshift=2,
                showarrow=False,
                font=dict(size=10),
            ),
            dict(
                text="library fragments · % of the highest predicted intensity",
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
        yaxis = dict(
            title="relative intensity (%)",
            range=[-116, 156],
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["100", "50", "0", "50", "100"],
            zeroline=True,
            zerolinewidth=1.4,
        )
    else:
        annotations.append(
            dict(
                text=f"% of {what} · {ref:,.0f} = 100 %"
                + (" · taller peaks clipped" if (rel > CLIP).any() else ""),
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.0,
                xanchor="left",
                yanchor="bottom",
                yshift=2,
                showarrow=False,
                font=dict(size=10),
            )
        )
        yaxis = dict(
            title="relative intensity (%)",
            range=[0, 128],
            tickvals=[0, 25, 50, 75, 100],
            zeroline=True,
        )
    xaxis: dict[str, Any] = dict(title="m/z", showspikes=False, range=list(view))
    revision = f"sp-{spec.level}" + (f"-{candidate}" if isotopes is not None else "")
    return themed(
        fig,
        scheme,
        height=SPECTRUM_HEIGHT,
        hovermode="closest",
        hoverdistance=8,
        showlegend=False,
        uirevision=revision,
        margin=dict(l=56, r=36, t=22, b=42),
        modebar={"orientation": "v"},
        shapes=shapes,
        annotations=annotations,
        xaxis=xaxis,
        yaxis=yaxis,
        meta={"full": x_range, "n_labels": int(n_labels)},
    )


def _add_fragments(
    fig: go.Figure,
    ov: FragmentOverlay,
    frags: Sequence[Any],
    matches: Sequence[Any],
    ref: float,
    x_range: Sequence[float],
) -> None:
    table = ov.fragments
    pred = table["predicted_intensity"].to_numpy(dtype=np.float64)
    theo = table["theo_mz"].to_numpy(dtype=np.float64)
    pmax = float(pred.max()) if pred.size and pred.max() > 0 else 1.0
    matched = {m.fragment: m for m in matches}
    for f in frags:
        if f.index >= len(table):
            continue
        mt = matched.get(f.index)
        p = round(-100.0 * float(pred[f.index]) / pmax, 1)
        t = round(float(theo[f.index]), 4)
        status = "matched in this scan" if mt is not None else "not matched in this scan"
        xic = "observed in the XIC" if f.observed else "never observed in the XIC"
        fig.add_trace(
            go.Scatter(
                x=[t, t],
                y=[0.0, p],
                mode="lines+markers",
                line=dict(color=f.colour, width=2.0, dash="solid" if f.observed else "dot"),
                marker=dict(size=[0, 5], color=f.colour),
                opacity=1.0 if mt is not None else 0.38,
                showlegend=False,
                name=f"{f.text} library",
                meta={"kind": "library", "frag": f.index},
                hovertemplate=(
                    f"<b>{f.html}</b> library<br>m/z {t:.4f} · predicted "
                    f"{pred[f.index]:.3f}<br>{status}<br>{xic}<extra></extra>"
                ),
            )
        )
    if not matches:
        return
    order = sorted(matches, key=lambda m: m.obs_mz)
    span = max(float(x_range[1]) - float(x_range[0]), 1.0)
    heights = label_heights(
        [m.obs_mz for m in order],
        [min(100.0 * m.obs_intensity / ref, CLIP) for m in order],
        span,
        150.0,
    )
    by_index = {f.index: f for f in frags}
    for mt, h in zip(order, heights, strict=True):
        f = by_index[mt.fragment]
        y = round(min(100.0 * mt.obs_intensity / ref, CLIP), 1)
        obs = round(float(mt.obs_mz), 4)
        hover = (
            f"<b>{f.html}</b> matched<br>observed m/z {mt.obs_mz:.4f} · library "
            f"{mt.theo_mz:.4f}<br>ppm raw {mt.ppm_raw:+.2f} · offset-corrected "
            f"{mt.ppm_corrected:+.2f} (viewer-derived)<br>intensity {mt.obs_intensity:,.0f} "
            f"({100.0 * mt.obs_intensity / ref:.1f}%)<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[obs, obs],
                y=[0.0, y],
                mode="lines",
                line=dict(color=f.colour, width=2.4),
                showlegend=False,
                name=f"{f.text} observed",
                meta={"kind": "matched", "frag": f.index},
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
                showlegend=False,
                name=f"{f.text} label",
                meta={"kind": "matched-label", "frag": f.index},
                hovertemplate=hover,
            )
        )


def _add_isotopes(
    fig: go.Figure,
    iso: IsotopeOverlay,
    mz: np.ndarray,
    rel: np.ndarray,
    shown: np.ndarray,
    ref: float,
) -> None:
    for row in iso.rows:
        k = int(row["k"])
        inside = np.flatnonzero((mz >= row["lo"]) & (mz <= row["hi"]))
        name = ISOTOPE_NAMES.get(k, f"M{k:+d}")
        if inside.size:
            xs, ys = _stems(_r(mz[inside], 4), _r(shown[inside], 2))
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=ISOTOPE, width=2.6),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{name} peaks",
                    meta={"kind": "isotope"},
                )
            )
        top = float(shown[inside].max()) if inside.size else 0.0
        hover = (
            f"<b>{name}</b> m/z {row['mz']:.4f}<br>window {row['lo']:.4f} to {row['hi']:.4f} "
            f"(± {iso.tol_ppm:g} ppm)<br>{int(row['n_peaks'])} peaks, sum {row['sum']:,.0f} "
            "(viewer-recomputed with the engine's sum_near rule)<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[round(row["mz"], 4)],
                y=[round(top + 4.0, 1)],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=9, color=ISOTOPE),
                text=[f"<b>{name}</b>"],
                textposition="top center",
                textfont=dict(size=LABEL_FONT, color=ISOTOPE),
                showlegend=False,
                name=name,
                meta={"kind": "isotope-mark"},
                hovertemplate=hover,
            )
        )


# --------------------------------------------------------------------------- navigator


def nav_figure(
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    rt: float,
    band: tuple[float, float] | None,
    rt_max: float,
    tic: tuple[np.ndarray, np.ndarray] | None = None,
    what: str = "",
    scheme: str = "light",
) -> go.Figure:
    """The run's RT axis: accepted apexes per RT bin, the TIC (when known), the scan.

    Every bin has a transparent full-height bar, so a click anywhere on the strip picks
    an RT. The x axis is fixed to ``[0, rt_max]`` with fixed margins (:data:`NAV_MARGIN`)
    so the RT slider under it lines up.
    """
    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    width = float(np.median(np.diff(edges))) if edges.size > 1 else 1.0
    top = float(counts.max()) if counts.size and counts.max() > 0 else 1.0
    fig = go.Figure()
    if tic is not None and tic[0].size:
        t_rt, t_val = tic
        t_top = float(np.nanmax(t_val)) if t_val.size else 1.0
        fig.add_trace(
            go.Scattergl(
                x=_r(t_rt, 2),
                y=_r(np.asarray(t_val) / (t_top or 1.0) * top, 3),
                mode="lines",
                line=dict(color=TIC_LINE, width=1),
                fill="tozeroy",
                fillcolor=TIC_FILL,
                hoverinfo="skip",
                showlegend=False,
                name="TIC",
                meta={"kind": "tic"},
            )
        )
    fig.add_trace(
        go.Bar(
            x=_r(centres, 2),
            y=counts,
            width=width,
            marker=dict(color=NAV_BARS, line=dict(width=0)),
            customdata=np.column_stack([_r(edges[:-1], 1), _r(edges[1:], 1)]),
            hovertemplate=(
                f"%{{y:,}} {what}<br>apex_rt %{{customdata[0]}} to %{{customdata[1]}} s"
                "<br>click to go to this RT<extra></extra>"
            ),
            showlegend=False,
            name="accepted apexes",
            meta={"kind": "apexes"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=_r(centres, 2),
            y=np.full(centres.size, top * 1.08),
            width=width,
            marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
            hovertemplate="RT %{x:.0f} s · click to go here<extra></extra>",
            showlegend=False,
            name="click target",
            meta={"kind": "target"},
        )
    )
    shapes = []
    if band is not None:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=float(band[0]),
                x1=float(band[1]),
                y0=0,
                y1=1,
                fillcolor=BAND_FILL,
                line=dict(width=0),
                layer="below",
            )
        )
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=float(rt),
            x1=float(rt),
            y0=0,
            y1=1,
            line=dict(color=SCAN_LINE, width=2),
        )
    )
    return themed(
        fig,
        scheme,
        height=NAV_HEIGHT,
        barmode="overlay",
        bargap=0,
        hovermode="closest",
        dragmode=False,
        showlegend=False,
        margin=dict(NAV_MARGIN),
        shapes=shapes,
        xaxis=dict(range=[0.0, float(rt_max)], fixedrange=True, showgrid=False, ticksuffix=" s"),
        yaxis=dict(
            range=[0.0, top * 1.08],
            fixedrange=True,
            showgrid=False,
            title=dict(text="apexes", font=dict(size=10)),
            tickfont=dict(size=9),
            nticks=3,
        ),
    )
