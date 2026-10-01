"""Small layout helpers shared by the pages."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from dash import dash_table, dcc, html

CARD = {
    "border": "1px solid #ddd",
    "borderRadius": "6px",
    "padding": "10px 14px",
    "background": "#fff",
}
MUTED = {"color": "#666", "fontSize": "0.85em"}
SECTION = {"marginTop": "18px"}


def fmt(value: Any, *, digits: int = 4) -> str:
    """A compact text form of a number (empty for None and NaN)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    if isinstance(value, tuple | list):
        return " - ".join(fmt(v, digits=digits) for v in value)
    return str(value)


def card(title: str, value: str, subtitle: str | None = None) -> html.Div:
    children = [html.Div(title, style=MUTED), html.Div(value, style={"fontSize": "1.6em"})]
    if subtitle:
        children.append(html.Div(subtitle, style=MUTED))
    return html.Div(children, style={**CARD, "minWidth": "220px", "flex": "1"})


def notices(items: Iterable[tuple[str, str]], *, kind: str = "warning") -> html.Div | None:
    colours = {"warning": ("#fff8e1", "#f0c36d"), "info": ("#eef6ff", "#9cc3f5")}
    bg, border = colours.get(kind, colours["info"])
    rows = [html.Li([html.B(code + ": ") if code else None, text]) for code, text in items]
    if not rows:
        return None
    return html.Div(
        html.Ul(rows, style={"margin": "0", "paddingLeft": "18px"}),
        style={**CARD, "background": bg, "borderColor": border, **SECTION},
    )


def table(
    df: pd.DataFrame,
    *,
    id: str | None = None,
    page_size: int = 15,
    columns: Sequence[str] | None = None,
    labels: dict[str, str] | None = None,
    **kwargs: Any,
) -> dash_table.DataTable:
    """A read-only DataTable of a small DataFrame (formatted as text)."""
    cols = list(columns) if columns is not None else list(df.columns)
    data = [{c: fmt(row.get(c)) for c in cols} for row in df[cols].to_dict("records")]
    params: dict[str, Any] = dict(
        data=data,
        columns=[{"name": (labels or {}).get(c, c), "id": c} for c in cols],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "monospace",
            "fontSize": "0.85em",
            "textAlign": "left",
            "maxWidth": "420px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={"fontWeight": "bold"},
        **kwargs,
    )
    if id is not None:
        params["id"] = id
    return dash_table.DataTable(**params)


def details(summary: str, *children: Any) -> html.Details:
    return html.Details(
        [html.Summary(summary, style={"cursor": "pointer"}), *children], style=SECTION
    )


def precursor_href(base: str, run: str, cid: int) -> str:
    return f"{base}precursor?{urlencode({'run': run, 'cid': int(cid)})}"


def graph(figure, **kwargs: Any) -> dcc.Graph:
    return dcc.Graph(
        figure=figure,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}},
        **kwargs,
    )
