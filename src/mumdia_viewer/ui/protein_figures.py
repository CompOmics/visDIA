"""Figures of the protein page: the quantity per run and the peptides-by-runs heatmap.

Colours follow the data-visualisation rules of the app: two series (protein_group_quant
and MaxLFQ) take the first two slots of the categorical order (indigo, teal; validated
for CVD separation and contrast on the light and the dark card), and the heatmap's
magnitude is one blue ramp (its two lightest steps left out, so that they never read as
the grey of "no quantity"), light to dark on the light card and the reverse on the dark
card (so that a small quantity recedes towards the surface in both). A cell without a
quantity is a neutral grey, never the low end of the ramp. Figures have no titles (the
card has the title) and are built for one colour scheme; the page rebuilds the heatmap
when the scheme changes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import template

# Categorical slots 1 and 2 (theme.SERIES order: indigo 6, then teal 7 for contrast).
PG_COLOUR = "#4c6ef5"
LFQ_COLOUR = "#0ca678"
# One sequential hue (blue 100 to 700), light to dark; reversed on the dark card.
BLUE_RAMP = [
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
MISSING = {"light": "#e9ecef", "dark": "#373a40"}
INK = {"light": "#1a1b1e", "dark": "#f1f3f5"}
TRANSFER = "#82c91e"  # lime 6: the app's match-between-runs mark
# Rows of the heatmap (peptides): the cell height, and the most rows drawn.
ROW_PX = 17
MAX_ROWS = 150


def _ramp(scheme: str) -> list[list[Any]]:
    steps = BLUE_RAMP if scheme != "dark" else BLUE_RAMP[::-1]
    n = len(steps) - 1
    return [[i / n, c] for i, c in enumerate(steps)]


def _ramp_colour(frac: float, scheme: str) -> str:
    steps = BLUE_RAMP if scheme != "dark" else BLUE_RAMP[::-1]
    i = round(max(0.0, min(1.0, frac)) * (len(steps) - 1))
    return steps[i]


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def compact(value: Any) -> str:
    """A quantity in a few characters: 7,340; 99.3 k; 1.40 M."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f} G"
    if a >= 1e6:
        return f"{v / 1e6:.2f} M"
    if a >= 1e4:
        return f"{v / 1e3:.1f} k"
    return f"{v:,.0f}"


def _empty(scheme: str, text: str, height: int = 180) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(
        template=template(scheme),
        height=height,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    return fig


# --------------------------------------------------------------------------- per run


def quantity_per_run_figure(df: pd.DataFrame, scheme: str, *, height: int = 230) -> go.Figure:
    """Grouped bars per run on one linear axis from zero: protein_group_quant and MaxLFQ.

    Both are intensities of the same group in the same run (one axis, no second scale;
    a bar's length is its value, so the axis starts at zero).
    A run without a value has no bar; its hover and the table below say why.
    """
    if df is None or df.empty:
        return _empty(scheme, "no runs")
    runs = df["run"].astype(str).tolist()
    fig = go.Figure()
    has_any = False

    def add(column: str, name: str, colour: str, how: str) -> None:
        nonlocal has_any
        if column not in df:
            return
        values = df[column].astype("float64")
        if values.notna().any():
            has_any = True
        state = df["state"].astype(object).where(df["state"].notna(), "") if "state" in df else ""
        texts = []
        for i, v in enumerate(values):
            if math.isfinite(v):
                texts.append(f"{v:,.0f}")
            elif column == "quantity":
                st = str(state.iloc[i]).replace("_", " ") if len(df) else ""
                texts.append(f"no quantity ({st or 'not recorded'})")
            else:
                texts.append("no value (no feature in this run)")
        fig.add_trace(
            go.Bar(
                x=runs,
                y=values,
                name=name,
                marker={"color": colour, "line": {"width": 0}},
                customdata=np.array(texts, dtype=object),
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{customdata}}<br>{how}<extra></extra>",
            )
        )

    add(
        "quantity",
        "protein_group_quant",
        PG_COLOUR,
        "top-N sum of the per-peptide maxima (quant.rs)",
    )
    add("lfq", "MaxLFQ", LFQ_COLOUR, "lfq_maxlfq.parquet (MaxLFQ across the runs)")
    if not has_any:
        return _empty(scheme, "no quantity in any run", height)
    fig.update_layout(
        template=template(scheme),
        height=height,
        barmode="group",
        bargap=0.34,
        bargroupgap=0.12,
        barcornerradius=4,
        margin={"l": 58, "r": 12, "t": 30, "b": 34},
        legend={"orientation": "h", "y": 1.04, "yanchor": "bottom", "x": 0, "xanchor": "left"},
        hovermode="closest",
        yaxis={"title": {"text": "quantity"}, "exponentformat": "SI", "rangemode": "tozero"},
        xaxis={"title": {"text": None}, "type": "category"},
    )
    return fig


# --------------------------------------------------------------------------- matrix


def nice_ticks(lo: float, hi: float, most: int = 6) -> list[float]:
    """log10 positions of 1-2-5 values between 10**lo and 10**hi (at most ``most``)."""
    out: list[float] = []
    for steps in ((1.0,), (1.0, 5.0), (1.0, 2.0, 5.0), tuple(float(i) for i in range(1, 10))):
        cand = []
        for k in range(math.floor(lo) - 1, math.ceil(hi) + 1):
            for m in steps:
                v = math.log10(m) + k
                if lo - 1e-9 <= v <= hi + 1e-9:
                    cand.append(v)
        if len(cand) >= 2:
            out = cand
            if len(cand) >= 3:
                break
    if len(out) < 2:
        out = [lo, hi]
    while len(out) > most:
        out = out[::2]
    return out


def matrix_rows(m: pd.DataFrame, *, max_rows: int = MAX_ROWS) -> tuple[list[int], int, int]:
    """The peptides the heatmap draws, in order, and how many it leaves out.

    A peptide with neither a quantity nor an identification at the threshold in any run
    is left out (its row would be empty); the others keep the peptide table's order,
    sorted by the number of runs with a quantity (most first). At most ``max_rows`` are
    drawn. Returns (base_peptide_ids, n_empty, n_cut).
    """
    if m.empty:
        return [], 0, 0
    g = m.groupby("base_peptide_id", sort=False)
    n_q = g["quantity"].count()
    n_id = g["identified"].sum()
    order = pd.Series(np.arange(len(n_q)), index=n_q.index)
    keep = (n_q > 0) | (n_id > 0)
    n_empty = int((~keep).sum())
    shown = pd.DataFrame({"n_q": n_q[keep], "n_id": n_id[keep], "order": order[keep]})
    shown = shown.sort_values(["n_q", "n_id", "order"], ascending=[False, False, True])
    ids = [int(i) for i in shown.index]
    n_cut = max(0, len(ids) - max_rows)
    return ids[:max_rows], n_empty, n_cut


def matrix_figure(
    m: pd.DataFrame,
    scheme: str,
    *,
    threshold: float,
    base_href: str,
    experiment: bool,
    rows: list[int] | None = None,
    selected: int | None = None,
) -> go.Figure:
    """The peptides (rows) by the runs (columns): log10 quantity, q marks, transfers.

    Cell colour: log10 of the peptide's quantity in the run (the maximum over its
    quantified precursors, the value of the engine's rollup); grey: no quantity. A check
    mark: the peptide has a row with run_psm_q <= threshold in the run. A lime ring: a
    match-between-runs transfer among the peptide's rows in the run. ``customdata``
    carries the precursor page of each cell (the quantity's precursor, else the
    best-scoring row) for the click.
    """
    if rows is None:
        rows, _, _ = matrix_rows(m)
    if m.empty or not rows:
        return _empty(scheme, "no peptide has a quantity or an identification in any run")
    runs = list(dict.fromkeys(m["run"].astype(str)))
    sub = m[m["base_peptide_id"].isin(rows)]
    pos = {pid: i for i, pid in enumerate(rows)}
    col = {r: j for j, r in enumerate(runs)}
    n_r, n_c = len(rows), len(runs)
    z = np.full((n_r, n_c), np.nan)
    missing = np.full((n_r, n_c), np.nan)
    text = np.full((n_r, n_c), "", dtype=object)
    href = np.full((n_r, n_c), "", dtype=object)
    labels = [""] * n_r
    marks_x: list[str] = []
    marks_y: list[str] = []
    marks_colour: list[str] = []
    ring_x: list[str] = []
    ring_y: list[str] = []
    q = sub["quantity"].to_numpy(dtype="float64")
    finite = q[np.isfinite(q) & (q > 0)]
    lo = float(np.log10(finite.min())) if finite.size else 0.0
    hi = float(np.log10(finite.max())) if finite.size else 1.0
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    t = f"{threshold:g}"
    tick = {r.base_peptide_id: r.sequence for r in sub.itertuples(index=False)}
    for pid in rows:
        seq = str(tick.get(pid, pid))
        labels[pos[pid]] = seq if len(seq) <= 20 else seq[:18] + "…"
    # Row labels must be unique for a categorical axis: repeat sequences get a suffix.
    seen: dict[str, int] = {}
    for i, lab in enumerate(labels):
        seen[lab] = seen.get(lab, 0) + 1
        if seen[lab] > 1:
            labels[i] = f"{lab} ({seen[lab]})"
    for r in sub.itertuples(index=False):
        i, j = pos[int(r.base_peptide_id)], col[str(r.run)]
        qty = float(r.quantity) if r.quantity == r.quantity else math.nan
        has_q = math.isfinite(qty) and qty > 0
        if has_q:
            z[i, j] = math.log10(qty)
        else:
            missing[i, j] = 1.0
        best = float(r.best_run_psm_q) if r.best_run_psm_q == r.best_run_psm_q else math.nan
        parts = [f"<b>{r.sequence}</b> · run {r.run}" if experiment else f"<b>{r.sequence}</b>"]
        if has_q:
            parts.append(
                f"quantity {qty:,.0f}: the largest of {int(r.n_quantified)} quantified "
                f"precursor{'s' if int(r.n_quantified) != 1 else ''} (peptide_quant)"
            )
            if bool(r.in_rollup):
                parts.append(f"in the protein's top-N sum of this run (rank {int(r.rollup_rank)})")
        elif int(r.n_quant_rows):
            parts.append("no quantity: not quantifiable (peptide_quant rows with a null quantity)")
        else:
            parts.append("no quantity: no precursor passed the run's quant gate")
        if math.isfinite(best):
            verdict = "passes" if best <= threshold else "does not pass"
            parts.append(
                f"best run_psm_q {best:.3g} ({verdict} ≤ {t}); {int(r.n_rows)} scored "
                f"row{'s' if int(r.n_rows) != 1 else ''}"
            )
        else:
            parts.append("no scored row in this run")
        if int(r.n_transferred):
            parts.append(
                f"match-between-runs transfer: {int(r.n_transferred)} row"
                f"{'s' if int(r.n_transferred) != 1 else ''}"
                + (" (the quantity's precursor)" if bool(r.quantity_from_transfer) else "")
            )
        cid = r.quantity_cid if has_q and r.quantity_cid is not pd.NA else r.best_cid
        if cid is not pd.NA and cid is not None and cid == cid:
            run_name = str(r.run) if experiment else ""
            query = f"run={run_name}&cid={int(cid)}" if run_name else f"cid={int(cid)}"
            href[i, j] = f"{base_href}precursor?{query}"
            parts.append("click: open its precursor page")
        text[i, j] = "<br>".join(parts)
        if bool(r.identified):
            marks_x.append(runs[j])
            marks_y.append(labels[i])
            frac = (z[i, j] - lo) / (hi - lo) if has_q else None
            if frac is None:
                marks_colour.append(INK[scheme if scheme == "dark" else "light"])
            else:
                cell = _ramp_colour(frac, scheme)
                marks_colour.append("#ffffff" if _luminance(cell) < 0.3 else "#1a1b1e")
        if int(r.n_transferred):
            ring_x.append(runs[j])
            ring_y.append(labels[i])
    custom = np.dstack([text, href])
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=missing,
            x=runs,
            y=labels,
            colorscale=[[0, MISSING[scheme]], [1, MISSING[scheme]]],
            showscale=False,
            xgap=2,
            ygap=2,
            customdata=custom,
            hovertemplate="%{customdata[0]}<extra></extra>",
            name="no quantity",
        )
    )
    tickvals = nice_ticks(lo, hi)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=runs,
            y=labels,
            zmin=lo,
            zmax=hi,
            colorscale=_ramp(scheme),
            xgap=2,
            ygap=2,
            customdata=custom,
            hovertemplate="%{customdata[0]}<extra></extra>",
            colorbar={
                "title": {"text": "quantity", "side": "top", "font": {"size": 11}},
                "tickvals": tickvals,
                "ticktext": [compact(round(10**v)) for v in tickvals],
                "thickness": 10,
                "len": min(1.0, max(0.35, 220 / max(1, n_r * ROW_PX))),
                "y": 1,
                "yanchor": "top",
                "outlinewidth": 0,
                "tickfont": {"size": 10},
            },
            name="quantity",
        )
    )
    if marks_x:
        fig.add_trace(
            go.Scatter(
                x=marks_x,
                y=marks_y,
                mode="text",
                text=["✓"] * len(marks_x),
                textfont={"color": marks_colour, "size": 11},
                hoverinfo="skip",
                showlegend=False,
                name="identified",
            )
        )
    if ring_x:
        fig.add_trace(
            go.Scatter(
                x=ring_x,
                y=ring_y,
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "size": 15,
                    "color": TRANSFER,
                    "line": {"width": 2, "color": TRANSFER},
                },
                hoverinfo="skip",
                showlegend=False,
                name="transfer",
            )
        )
    height = max(150, n_r * ROW_PX + 64)
    shapes = []
    if selected is not None and selected in pos:
        shapes.append(_row_shape(pos[selected], n_r, scheme))
    fig.update_layout(
        template=template(scheme),
        height=height,
        margin={"l": 8, "r": 8, "t": 26, "b": 8},
        showlegend=False,
        hovermode="closest",
        xaxis={
            "side": "top",
            "type": "category",
            "showgrid": False,
            "fixedrange": True,
            "tickfont": {"size": 11},
        },
        yaxis={
            "type": "category",
            "autorange": "reversed",
            "showgrid": False,
            "fixedrange": True,
            "automargin": True,
            "tickfont": {
                "family": "'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
                "size": 10.5,
            },
        },
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=shapes,
        meta={"rows": [int(p) for p in rows], "runs": runs, "scheme": scheme},
        dragmode=False,
    )
    return fig


def _row_shape(i: int, n_rows: int, scheme: str) -> dict[str, Any]:
    """An outline around row ``i`` of the heatmap (the selected peptide)."""
    colour = "#4263eb" if scheme != "dark" else "#91a7ff"
    return {
        "type": "rect",
        "xref": "paper",
        "yref": "y",
        "x0": 0,
        "x1": 1,
        "y0": i - 0.5,
        "y1": i + 0.5,
        "line": {"color": colour, "width": 2},
        "fillcolor": "rgba(0,0,0,0)",
        "layer": "above",
        "name": "selected",
    }
