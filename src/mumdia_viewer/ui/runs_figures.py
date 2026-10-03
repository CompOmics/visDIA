"""Figures of the across-runs page and the condition-ratio page (pure functions).

Every figure takes data-layer objects (:mod:`mumdia_viewer.data.across`) and a scheme,
and ends with :func:`figures.themed`. Retention times are seconds. The figures carry no
titles: the cards have them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mumdia_viewer.data.across import MIXED, OTHER, PrecursorAcross, RatioResult, RunXic

from . import figures, theme
from .detail_view import fragments_of
from .widgets import fmt_q

STATE_COLOURS = {
    "quantified": theme.ACCEPT,
    "not_quantifiable": theme.WARN,
    "not_selected": theme.PREDICTION,
    "not_scored": theme.WINDOW,
}
TRANSFER = "#74b816"  # lime 7: match-between-runs transfers
SPECIES_PALETTE = ["#4c6ef5", "#f59f00", "#12b886", "#e64980", "#7950f2", "#15aabf", "#fd7e14"]
SPECIES_EXTRA = {MIXED: "#868e96", OTHER: "#adb5bd"}
COLS = 3
PANEL_HEIGHT = 210


def run_name(run: str) -> str:
    return run or "single run"


def species_name(suffix: str) -> str:
    return suffix.lstrip("_") if suffix not in SPECIES_EXTRA else suffix


def species_colours(suffixes: Sequence[str]) -> dict[str, str]:
    out = {s: SPECIES_PALETTE[i % len(SPECIES_PALETTE)] for i, s in enumerate(suffixes)}
    out.update(SPECIES_EXTRA)
    return out


def _rgba(colour: str, alpha: float) -> str:
    c = colour.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def run_colour(i: int) -> str:
    return theme.SERIES[i % len(theme.SERIES)]


# --------------------------------------------------------------------------- XICs


def _xic_range(x: RunXic, axis: np.ndarray | None) -> list[float] | None:
    """The shown RT range of a run's panel: the elution and integration bounds with a
    margin of one elution width (at least 6 s), within the plotted trace."""
    marks = [
        v
        for v in (x.elution_lo, x.elution_hi, x.integration_lo, x.integration_hi, x.apex_rt)
        if _finite(v) is not None
    ]
    if not marks:
        if axis is not None and axis.size:
            return [float(axis[0]), float(axis[-1])]
        return None
    lo, hi = min(marks), max(marks)
    width = (x.elution_hi or 0) - (x.elution_lo or 0) if x.elution_lo and x.elution_hi else 0
    pad = max(width, 6.0)
    lo, hi = lo - pad, hi + pad
    if axis is not None and axis.size:
        lo, hi = max(lo, float(axis[0])), min(hi, float(axis[-1]))
        if hi <= lo:
            lo, hi = float(axis[0]), float(axis[-1])
    return [round(lo, 3), round(hi, 3)]


def _panel_title(across: PrecursorAcross, i: int, threshold: float) -> str:
    row = across.rows.iloc[i]
    name = run_name(str(row["run"]))
    if not bool(row["scored"]):
        return f"<b>{name}</b> · not scored"
    q = _finite(row["run_psm_q"])
    mark = "✓" if q is not None and q <= threshold else "✗"
    tr = " · <span style='color:#74b816'>MBR</span>" if bool(row["transferred"]) else ""
    return f"<b>{name}</b> · run_psm_q {fmt_q(q, threshold)} {mark}{tr}"


def xic_grid_figure(
    across: PrecursorAcross,
    xics: Sequence[RunXic],
    threshold: float,
    scheme: str = "light",
    *,
    shared: bool = False,
) -> go.Figure:
    """Small multiples: one panel per run, each on its own RT axis.

    Each panel has the run's fragment traces (b blue, y red, the precursor page's
    colours), the identification's elution bounds (green band) and apex (green line),
    quant's integration bounds (violet dashed box) and the calibrated RT prediction
    (grey dotted). ``shared`` puts every panel on the highest intensity of all runs.
    A click on a trace opens the precursor page of that run (``customdata`` is the run).
    """
    n = len(xics)
    cols = min(COLS, max(1, n))
    rows = math.ceil(n / cols)
    titles = [_panel_title(across, i, threshold) for i in range(n)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.06 if cols > 1 else 0.0,
        vertical_spacing=min(0.16, 0.42 / rows) if rows > 1 else 0.0,
    )
    seen: set[str] = set()
    ymax_all = 0.0
    per_panel: list[float] = []
    colours: dict[str, str] = {}
    labels: dict[str, str] = {}
    for x in xics:
        for f in fragments_of(x.chromatogram):
            colours.setdefault(f.name, f.colour)
            labels.setdefault(f.name, f.html)
    for i, x in enumerate(xics):
        r, c = i // cols + 1, i % cols + 1
        chrom = x.chromatogram
        axis = None
        top = 0.0
        if chrom is not None:
            for f in fragments_of(chrom):
                t = chrom.fragments()[f.index]
                if not t.observed:
                    continue
                axis = t.rt if axis is None else axis
                y = np.asarray(t.intensity, dtype="float64")
                top = max(top, float(y.max()) if y.size else 0.0)
                fig.add_trace(
                    go.Scatter(
                        x=np.round(np.asarray(t.rt, dtype="float64"), 3),
                        y=np.round(y, 1),
                        mode="lines",
                        name=labels.get(f.name, f.html),
                        legendgroup=f.name,
                        showlegend=f.name not in seen,
                        line=dict(color=colours.get(f.name, f.colour), width=1.5),
                        customdata=[x.run] * len(y),
                        hovertemplate=(
                            f"<b>{run_name(x.run)}</b> · {f.html}<br>RT %{{x:.2f}} s<br>"
                            "intensity %{y:,.0f}<br><i>click to open this run</i><extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                seen.add(f.name)
        per_panel.append(top)
        ymax_all = max(ymax_all, top)
        rng = _xic_range(x, None if axis is None else np.asarray(axis, dtype="float64"))
        xref = "x" if i == 0 else f"x{i + 1}"
        yref = "y" if i == 0 else f"y{i + 1}"
        if x.elution_lo is not None and x.elution_hi is not None:
            fig.add_shape(
                type="rect",
                x0=x.elution_lo,
                x1=x.elution_hi,
                xref=xref,
                y0=0,
                y1=1,
                yref=f"{yref} domain",
                fillcolor=theme.ELUTION,
                line=dict(width=0),
                layer="below",
            )
        if x.integration_lo is not None and x.integration_hi is not None:
            fig.add_shape(
                type="rect",
                x0=x.integration_lo,
                x1=x.integration_hi,
                xref=xref,
                y0=0.005,
                y1=0.995,
                yref=f"{yref} domain",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=theme.INTEGRATION, width=1.2, dash="dash"),
            )
        if x.rt_pred_cal is not None:
            fig.add_shape(
                type="line",
                x0=x.rt_pred_cal,
                x1=x.rt_pred_cal,
                xref=xref,
                y0=0,
                y1=1,
                yref=f"{yref} domain",
                line=dict(color=theme.PREDICTION, width=1.4, dash="dot"),
            )
        if x.apex_rt is not None:
            fig.add_shape(
                type="line",
                x0=x.apex_rt,
                x1=x.apex_rt,
                xref=xref,
                y0=0,
                y1=1,
                yref=f"{yref} domain",
                line=dict(color=theme.APEX, width=1.8),
            )
        if chrom is None:
            fig.add_annotation(
                text=x.note or "no chromatogram",
                xref=f"{xref} domain",
                yref=f"{yref} domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11, color=theme.PREDICTION),
            )
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
            continue
        if rng is not None:
            fig.update_xaxes(range=rng, row=r, col=c)
        fig.update_yaxes(tickformat="~s", rangemode="tozero", nticks=4, row=r, col=c)
        fig.update_xaxes(nticks=5, row=r, col=c)
    if shared and ymax_all > 0:
        for i in range(n):
            if per_panel[i] > 0:
                fig.update_yaxes(range=[0, ymax_all * 1.08], row=i // cols + 1, col=i % cols + 1)
    for a in fig.layout.annotations[:n]:
        a.update(font=dict(size=11.5))
    fig.update_xaxes(title_text="retention time (s)", title_font=dict(size=10), row=rows)
    fig.update_yaxes(title_text="intensity", title_font=dict(size=10), col=1)
    fig.update_xaxes(tickfont=dict(size=10), showspikes=False)
    fig.update_yaxes(tickfont=dict(size=10))
    return figures.themed(
        fig,
        scheme,
        height=max(260, rows * PANEL_HEIGHT + 90),
        margin=dict(l=56, r=16, t=64, b=44),
        hovermode="closest",
        uirevision="xr-xic",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0 + 46 / max(260, rows * PANEL_HEIGHT + 90),
            x=0,
            font=dict(size=11),
            itemwidth=30,
            groupclick="togglegroup",
        ),
        meta={"kind": "xr-grid"},
    )


def xic_overlay_figure(
    across: PrecursorAcross,
    xics: Sequence[RunXic],
    threshold: float,
    scheme: str = "light",
    *,
    shared: bool = False,
) -> go.Figure:
    """Every run's summed fragment trace on one axis, aligned on its apex.

    x is ``RT - apex_rt`` of the run (seconds); y is the viewer's sum of the run's
    fragment traces, or, unless ``shared``, that sum divided by its maximum (so the
    shapes compare). The view opens on four times the widest elution half-width (at
    least 15 s) around the apex; the whole traces are plotted.
    """
    fig = go.Figure()
    any_trace = False
    reach = 0.0  # the farthest elution bound from its apex, over the runs
    for i, x in enumerate(xics):
        colour = run_colour(i)
        name = run_name(x.run)
        s = x.summed()
        if s is None or x.apex_rt is None:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"{name} ({x.note or 'no trace'})",
                    line=dict(color=colour, width=1.5, dash="dot"),
                    visible="legendonly",
                )
            )
            continue
        axis, y = s
        top = float(y.max()) if y.size else 0.0
        yy = y if shared or top <= 0 else y / top
        dx = axis - float(x.apex_rt)
        row = across.rows.iloc[i]
        q = _finite(row["run_psm_q"])
        tr = bool(row["transferred"])
        fig.add_trace(
            go.Scatter(
                x=np.round(dx, 3),
                y=np.round(yy, 4 if not shared else 1),
                mode="lines+markers",
                name=f"{name}" + (" (MBR)" if tr else ""),
                line=dict(color=colour, width=2.0, dash="dash" if tr else "solid"),
                marker=dict(size=4, color=colour),
                customdata=np.stack([np.full(axis.size, x.run), np.round(axis, 2)], axis=1),
                hovertemplate=(
                    f"<b>{name}</b> · run_psm_q {fmt_q(q, threshold)}<br>"
                    "RT %{customdata[1]} s (apex %{x:+.2f} s)<br>"
                    + ("summed intensity %{y:,.0f}" if shared else "share of the maximum %{y:.2f}")
                    + "<br><i>click to open this run</i><extra></extra>"
                ),
            )
        )
        any_trace = True
        if x.elution_lo is not None and x.elution_hi is not None:
            reach = max(reach, abs(x.elution_lo - x.apex_rt), abs(x.elution_hi - x.apex_rt))
    if not any_trace:
        return figures.empty_figure("No run has a chromatogram for this precursor.", scheme)
    fig.add_vline(x=0, line=dict(color=theme.APEX, width=1.6))
    fig.add_annotation(
        x=0,
        y=1,
        yref="y domain",
        text="apex",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=3,
        font=dict(size=10, color=theme.APEX),
    )
    half = max(4.0 * reach, 15.0)
    fig.update_xaxes(title_text="RT - the run's apex_rt (s)", zeroline=False, range=[-half, half])
    fig.update_yaxes(
        title_text="summed fragment intensity" if shared else "share of the run's maximum",
        tickformat="~s" if shared else ".1f",
        rangemode="tozero",
    )
    return figures.themed(
        fig,
        scheme,
        height=380,
        margin=dict(l=60, r=16, t=40, b=50),
        hovermode="closest",
        uirevision="xr-xic",
        meta={"kind": "xr-overlay"},
    )


# --------------------------------------------------------------------------- quantity and q


def quantity_figure(across: PrecursorAcross, scheme: str = "light") -> go.Figure:
    """peptide_quant quantity per run (bars) and the MaxLFQ precursor quantity (diamonds).

    A run without a quantity has no bar: its state is written at the base (never 0).
    Transfers (match-between-runs) are hatched.
    """
    rows = across.rows
    names = [run_name(r) for r in rows["run"]]
    q = rows["quantity"].to_numpy(dtype="float64")
    tr = rows["transferred"].astype(bool).to_numpy()
    colours = [TRANSFER if t else theme.UNIT_COLOURS["precursor"] for t in tr]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names,
            y=np.where(np.isfinite(q), q, None),
            name="peptide_quant",
            marker=dict(
                color=colours,
                pattern=dict(shape=["/" if t else "" for t in tr], fgcolor="rgba(255,255,255,0.6)"),
                line=dict(width=0),
            ),
            customdata=np.stack(
                [
                    rows["run"].astype(str).to_numpy(),
                    rows["n_fragments_used"].astype(str).to_numpy(),
                    np.where(tr, "MBR transfer", "native"),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>peptide_quant.quantity %{y:,.0f}<br>"
                "fragments used %{customdata[1]} · %{customdata[2]}"
                "<br><i>click to open this run</i><extra></extra>"
            ),
            width=0.6 if len(names) > 2 else 0.3,
        )
    )
    lfq = rows["lfq_quantity"].to_numpy(dtype="float64")
    if np.isfinite(lfq).any():
        fig.add_trace(
            go.Scatter(
                x=names,
                y=np.where(np.isfinite(lfq), lfq, None),
                mode="markers",
                name="MaxLFQ (precursor)",
                marker=dict(
                    symbol="diamond",
                    size=11,
                    color=theme.UNIT_COLOURS["protein_group"],
                    line=dict(width=1.2, color="white"),
                ),
                customdata=rows["run"].astype(str).to_numpy(),
                hovertemplate="<b>%{x}</b><br>MaxLFQ precursor %{y:,.0f}<extra></extra>",
            )
        )
    top = np.nanmax(np.concatenate([q, lfq])) if np.isfinite(np.concatenate([q, lfq])).any() else 1
    for name, state in zip(names, rows["state"], strict=True):
        if state == "quantified":
            continue
        fig.add_annotation(
            x=name,
            y=0,
            yref="y",
            text=str(state).replace("_", " ").replace("not ", "not<br>"),
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color=STATE_COLOURS.get(str(state), theme.PREDICTION)),
        )
    fig.update_yaxes(title_text="quantity", tickformat="~s", range=[0, float(top) * 1.15])
    fig.update_xaxes(type="category")
    return figures.themed(
        fig,
        scheme,
        height=300,
        margin=dict(l=60, r=16, t=36, b=40),
        bargap=0.35,
        hovermode="closest",
        meta={"kind": "xr-quant"},
    )


def q_figure(across: PrecursorAcross, threshold: float, scheme: str = "light") -> go.Figure:
    """run_psm_q and q_value per run on a log axis, with the threshold.

    run_psm_q (this run's own target-decoy re-run) is filled: green at or below the
    threshold, grey above; the pooled q_value is an open diamond. After match-between-
    runs, a transferred row's lowered run_psm_q is a lime cross.
    """
    rows = across.rows
    names = [run_name(r) for r in rows["run"]]
    rq = rows["run_psm_q"].to_numpy(dtype="float64")
    pq = rows["q_value"].to_numpy(dtype="float64")
    floor = 1e-6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=names,
            y=np.where(np.isfinite(rq), np.maximum(rq, floor), None),
            mode="markers",
            name="run_psm_q (this run)",
            marker=dict(
                size=12,
                color=[
                    theme.ACCEPT if v <= threshold else theme.PREDICTION
                    for v in np.nan_to_num(rq, nan=1)
                ],
                line=dict(width=1.2, color="white"),
            ),
            customdata=rq,
            hovertemplate="<b>%{x}</b><br>run_psm_q %{customdata:.3g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=names,
            y=np.where(np.isfinite(pq), np.maximum(pq, floor), None),
            mode="markers",
            name="q_value (pooled)",
            marker=dict(size=10, symbol="diamond-open", color=theme.TARGET, line=dict(width=1.6)),
            customdata=pq,
            hovertemplate="<b>%{x}</b><br>q_value %{customdata:.3g} (pooled)<extra></extra>",
        )
    )
    after = rows["run_psm_q_after_mbr"]
    if across.mbr and after.notna().any():
        a = after.to_numpy(dtype="float64", na_value=np.nan)
        fig.add_trace(
            go.Scatter(
                x=names,
                y=np.where(np.isfinite(a), np.maximum(a, floor), None),
                mode="markers",
                name="run_psm_q after MBR",
                marker=dict(size=11, symbol="x-thin-open", color=TRANSFER, line=dict(width=2.4)),
                customdata=a,
                hovertemplate="<b>%{x}</b><br>run_psm_q after MBR %{customdata:.3g}<extra></extra>",
            )
        )
    for name, ok in zip(names, rows["scored"], strict=True):
        if not ok:
            fig.add_annotation(
                x=name,
                y=0.04,
                yref="y domain",
                yanchor="bottom",
                text="not<br>scored",
                showarrow=False,
                font=dict(size=10, color=theme.PREDICTION),
            )
    fig.add_hline(
        y=threshold,
        line=dict(color=theme.ACCEPT, width=1.2, dash="dash"),
        annotation_text=f"q ≤ {threshold:g}",
        annotation_position="top left",
        annotation_font=dict(size=10, color=theme.ACCEPT),
    )
    finite = np.concatenate([rq[np.isfinite(rq)], pq[np.isfinite(pq)]])
    lo = math.floor(math.log10(max(floor, float(finite.min()) if finite.size else threshold))) - 0.3
    lo = min(lo, math.log10(threshold) - 0.5)
    fig.update_yaxes(type="log", title_text="q", range=[lo, 0.05], exponentformat="power", dtick=1)
    fig.update_xaxes(type="category")
    return figures.themed(
        fig,
        scheme,
        height=300,
        margin=dict(l=60, r=16, t=36, b=40),
        hovermode="closest",
        meta={"kind": "xr-q"},
    )


# --------------------------------------------------------------------------- ratios


def ratio_density_figure(
    result: RatioResult,
    edges: np.ndarray,
    counts: Mapping[str, np.ndarray],
    expected: Mapping[str, float | None],
    scheme: str = "light",
    *,
    height: int = 330,
) -> go.Figure:
    """Histograms of log2(A/B) per species as filled step lines (the viewer's bins).

    y is the share of the species' keys with both values in each bin, so species of
    different sizes compare. Solid line: the species' median; dashed: the expected ratio
    the user entered.
    """
    if not counts:
        return figures.empty_figure("No key has a value in both conditions.", scheme, height=height)
    colours = species_colours(result.suffixes)
    centres = (edges[:-1] + edges[1:]) / 2.0
    width = float(edges[1] - edges[0]) if edges.size > 1 else 0.1
    fig = go.Figure()
    med = result.species.set_index("species")
    shapes: list[dict[str, Any]] = []
    for sp, c in counts.items():
        total = int(c.sum())
        if total == 0:
            continue
        share = c / total
        colour = colours.get(sp, theme.PREDICTION)
        faint = sp in SPECIES_EXTRA
        fig.add_trace(
            go.Scatter(
                x=np.round(centres, 4),
                y=np.round(share, 5),
                mode="lines",
                line=dict(color=colour, width=1.2 if faint else 2.0, shape="hvh"),
                fill="tozeroy",
                fillcolor=_rgba(colour, 0.06 if faint else 0.16),
                name=f"{species_name(sp)} ({total:,})",
                legendgroup=sp,
                visible="legendonly" if faint else True,
                customdata=c,
                hovertemplate=(
                    f"<b>{species_name(sp)}</b><br>log2 A/B %{{x:.2f}} ± {width / 2:.2f}<br>"
                    "%{customdata:,} keys (%{y:.1%})<extra></extra>"
                ),
            )
        )
        m = med.loc[sp, "median"] if sp in med.index else np.nan
        if np.isfinite(m) and not faint:
            shapes.append(
                dict(
                    type="line",
                    x0=m,
                    x1=m,
                    y0=0,
                    y1=1,
                    yref="y domain",
                    line=dict(color=colour, width=2),
                    label=dict(
                        text=f"{m:+.2f}",
                        textposition="end",
                        textangle=0,
                        font=dict(size=10, color=colour),
                        xanchor="left",
                        yanchor="top",
                        padding=2,
                    ),
                )
            )
        e = expected.get(sp)
        if e is not None and not faint:
            shapes.append(
                dict(
                    type="line",
                    x0=e,
                    x1=e,
                    y0=0,
                    y1=0.92,
                    yref="y domain",
                    line=dict(color=colour, width=1.6, dash="dash"),
                )
            )
    for s in shapes:
        fig.add_shape(**s)
    r = result.table["log2_ratio"].to_numpy(dtype="float64")
    r = r[np.isfinite(r)]
    view: dict[str, Any] = {}
    if r.size > 20:
        lo, hi = np.percentile(r, [0.5, 99.5])
        view = {"range": [float(lo) - 0.5, float(hi) + 0.5]}
    fig.update_xaxes(title_text=f"log2 ({result.a} / {result.b})", zeroline=True, **view)
    fig.update_yaxes(title_text="share of the species' keys", tickformat=".0%", rangemode="tozero")
    return figures.themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=60, r=16, t=36, b=46),
        hovermode="closest",
        meta={"kind": "cr-density", "level": result.level},
    )


def ratio_scatter_figure(
    result: RatioResult,
    expected: Mapping[str, float | None],
    scheme: str = "light",
    *,
    height: int = 420,
) -> go.Figure:
    """log2(A/B) against log10 of the mean of the two condition values, per species.

    WebGL points; the solid lines are the species' medians, the dashed lines the expected
    ratios entered. Each point's ``text`` names what a click opens: the protein group, or
    ``"<peptidoform> <charge>+"``.
    """
    t = result.table
    ok = np.isfinite(t["log2_ratio"].to_numpy(dtype="float64"))
    if not ok.any():
        return figures.empty_figure("No key has a value in both conditions.", scheme, height=height)
    colours = species_colours(result.suffixes)
    fig = go.Figure()
    d = t[ok]
    for sp in (*result.suffixes, MIXED, OTHER):
        part = d[d["species"] == sp]
        if part.empty:
            continue
        colour = colours.get(sp, theme.PREDICTION)
        if result.level == "protein":
            text = part["protein_group"].astype(str).to_numpy()
        else:
            text = (
                part["peptidoform"].astype(str) + " " + part["charge"].astype(str) + "+"
            ).to_numpy()
        fig.add_trace(
            go.Scattergl(
                x=part["log10_mean"].to_numpy(dtype="float32"),
                y=part["log2_ratio"].to_numpy(dtype="float32"),
                mode="markers",
                name=f"{species_name(sp)} ({len(part):,})",
                marker=dict(size=4 if len(d) > 20000 else 5, color=colour, opacity=0.5),
                visible="legendonly" if sp in SPECIES_EXTRA else True,
                text=text,
                hovertemplate=(
                    "<b>%{text}</b><br>log2 A/B %{y:.2f}<br>log10 mean %{x:.2f}"
                    "<br><i>click to open</i><extra></extra>"
                ),
            )
        )
    med = result.species.set_index("species")
    for sp in result.suffixes:
        colour = colours.get(sp, theme.PREDICTION)
        m = med.loc[sp, "median"] if sp in med.index else np.nan
        if np.isfinite(m):
            fig.add_hline(y=float(m), line=dict(color=colour, width=1.6))
        e = expected.get(sp)
        if e is not None:
            fig.add_hline(y=float(e), line=dict(color=colour, width=1.4, dash="dash"))
    fig.update_xaxes(title_text="log10 of the mean of the two condition values")
    fig.update_yaxes(title_text=f"log2 ({result.a} / {result.b})")
    return figures.themed(
        fig,
        scheme,
        height=height,
        margin=dict(l=60, r=16, t=36, b=46),
        hovermode="closest",
        uirevision=f"cr-ma-{result.level}",
        meta={"kind": "cr-ma", "level": result.level},
    )
