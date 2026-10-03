"""Plotly figures of the compare page (pure functions of :mod:`mumdia_viewer.data.compare`
results).

Both scatters draw the shared identifications of the two result sets: A on x, B on y,
with the line y = x. A large set is drawn as a fixed random sample (the seed is fixed,
so the same points come back after every redraw and a click finds its row again); the
card says how many are drawn. Each point's customdata is its row in the shared table,
as a plain list (dcc.Graph leaves binary typed arrays out of click data).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figures import empty_figure, themed

# Side colours: blue for A, ochre for B (checked for colour-vision deficiency with the
# dataviz validator, light and dark); shared is the neutral grey between them.
A_COLOUR = "#1c7ed6"
B_COLOUR = "#c96a00"
BOTH_COLOUR = "#adb5bd"
POINT = "#5c7cfa"
DIAGONAL = "rgba(134, 142, 150, 0.85)"
MAX_POINTS = 10_000
SEED = 20261003
HEIGHT = 380


def sample_rows(n: int, limit: int = MAX_POINTS) -> np.ndarray:
    """The rows drawn of ``n``: all of them, or a fixed random sample of ``limit`` in
    row order."""
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _diagonal(lo: float, hi: float, *, log: bool = False) -> dict[str, Any]:
    return dict(
        type="scatter",
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        line=dict(color=DIAGONAL, width=1.2, dash="dash"),
        name="y = x",
        hoverinfo="skip",
        showlegend=False,
    )


def _floats(values: np.ndarray) -> list[float]:
    """Plain floats rounded to 6 significant digits (a short JSON payload)."""
    return [float(f"{v:.6g}") for v in values.tolist()]


def _figure(traces: list[dict[str, Any]], scheme: str, **layout: Any) -> dict[str, Any]:
    """A figure as a plain dict: Plotly's validation of 15,000 points costs more than
    the data. The layout (template included) is built and validated as usual."""
    shell = themed(go.Figure(), scheme, **layout)
    fig = shell.to_plotly_json()
    fig["data"] = traces
    return fig


def _range(values: np.ndarray, *, log: bool = False) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if log:
        v = v[v > 0]
    if not v.size:
        return (0.0, 1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def score_figure(
    shared: pd.DataFrame,
    *,
    unit_noun: str,
    a_label: str,
    b_label: str,
    scheme: str = "light",
    colour: str = POINT,
) -> Any:
    """A score against B score of the shared keys (``data.compare.shared_keys``)."""
    if shared.empty:
        return empty_figure(f"No {unit_noun} passes on both sides.", scheme, height=HEIGHT)
    rows = sample_rows(len(shared))
    sub = shared.iloc[rows]
    x = sub["a_score"].to_numpy(float)
    y = sub["b_score"].to_numpy(float)
    text = sub["key"].astype(str).tolist()
    lo = min(_range(x)[0], _range(y)[0])
    hi = max(_range(x)[1], _range(y)[1])
    traces = [
        _diagonal(lo, hi),
        dict(
            type="scattergl",
            x=_floats(x),
            y=_floats(y),
            mode="markers",
            marker=dict(size=4, color=colour, opacity=0.38 if len(rows) > 2000 else 0.7),
            text=text,
            customdata=[int(r) for r in rows],
            hovertemplate="%{text}<br>A score %{x:.4f} · B score %{y:.4f}<extra></extra>",
            name=unit_noun,
            showlegend=False,
        ),
    ]
    return _figure(
        traces,
        scheme,
        height=HEIGHT,
        margin=dict(l=58, r=14, t=10, b=46),
        uirevision="cmp-score",
        xaxis=dict(title=dict(text=f"A score ({a_label})")),
        yaxis=dict(title=dict(text=f"B score ({b_label})")),
    )


def quantity_figure(
    df: pd.DataFrame,
    *,
    scheme: str = "light",
    a_label: str = "A",
    b_label: str = "B",
) -> Any:
    """log10 quantity of A against B (``data.compare.quantity_pairs``); only points with
    a positive quantity on both sides are drawn."""
    ok = (df["a_quantity"] > 0) & (df["b_quantity"] > 0) if len(df) else pd.Series([], dtype=bool)
    idx = np.flatnonzero(ok.to_numpy()) if len(df) else np.array([], dtype=int)
    if not idx.size:
        return empty_figure(
            "No shared precursor has a quantity on both sides.", scheme, height=HEIGHT
        )
    rows = idx[sample_rows(idx.size)]
    sub = df.iloc[rows]
    x = sub["a_quantity"].to_numpy(float)
    y = sub["b_quantity"].to_numpy(float)
    many_pairs = df["a_run"].nunique() > 1 and df.attrs.get("mode") == "per run"
    text = [
        f"{k}" + (f"<br>A {ra or 'run'} · B {rb or 'run'}" if many_pairs else "")
        for k, ra, rb in zip(sub["key"], sub["a_run"], sub["b_run"], strict=True)
    ]
    lo = min(_range(x, log=True)[0], _range(y, log=True)[0])
    hi = max(_range(x, log=True)[1], _range(y, log=True)[1])
    traces = [
        _diagonal(lo, hi, log=True),
        dict(
            type="scattergl",
            x=_floats(x),
            y=_floats(y),
            mode="markers",
            marker=dict(size=4, color=POINT, opacity=0.35 if len(rows) > 2000 else 0.7),
            text=text,
            customdata=[int(r) for r in rows],
            hovertemplate="%{text}<br>A %{x:.4g}<br>B %{y:.4g}<extra></extra>",
            showlegend=False,
        ),
    ]
    return _figure(
        traces,
        scheme,
        height=HEIGHT,
        margin=dict(l=62, r=14, t=10, b=46),
        uirevision="cmp-quant",
        xaxis=dict(
            type="log", title=dict(text=f"A quantity ({a_label})"), exponentformat="power", dtick=1
        ),
        yaxis=dict(
            type="log", title=dict(text=f"B quantity ({b_label})"), exponentformat="power", dtick=1
        ),
    )


def log2_median_ratio(df: pd.DataFrame) -> float | None:
    """The viewer's median of log2(B / A) over the points with both quantities."""
    if df.empty:
        return None
    ok = (df["a_quantity"] > 0) & (df["b_quantity"] > 0)
    if not ok.any():
        return None
    r = np.log2(df.loc[ok, "b_quantity"].to_numpy(float) / df.loc[ok, "a_quantity"].to_numpy(float))
    value = float(np.median(r))
    return value if math.isfinite(value) else None


def drawn_note(n_drawn: int, n_total: int, noun: str) -> str:
    if n_drawn >= n_total:
        return f"All {n_total:,} {noun} are drawn."
    return f"{n_drawn:,} of {n_total:,} {noun} are drawn (a fixed random sample)."


def figure_points(fig: Any) -> int:
    """The number of points of the scatter trace of a figure built here."""
    data = fig.get("data", []) if isinstance(fig, dict) else list(fig.data)
    return sum(len(t.get("customdata") or []) for t in data if isinstance(t, dict))
