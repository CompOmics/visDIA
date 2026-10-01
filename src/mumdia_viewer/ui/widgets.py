"""Shared building blocks of the pages (Mantine components)."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlencode

import dash_mantine_components as dmc
from dash import dcc, html

from .icons import icon
from .theme import GRAPH_CONFIG, UNIT_MANTINE

_MOD = re.compile(r"(\[[^\]]*\]|\([^)]*\))")


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
        if math.isinf(value):
            return "unbounded"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    if isinstance(value, tuple | list):
        return " to ".join(fmt(v, digits=digits) for v in value)
    return str(value)


def fmt_q(value: Any) -> str:
    if value is None:
        return ""
    v = float(value)
    if math.isnan(v):
        return ""
    if v == 0:
        return "0"
    return f"{v:.2e}" if v < 0.001 else f"{v:.4f}"


def peptidoform(text: str | None, *, size: str = "1em") -> html.Span:
    """A ProForma-lite peptidoform with modifications as superscripts."""
    if not text:
        return html.Span("")
    parts: list = []
    rest = text
    if rest.startswith("DECOY_"):
        parts.append(html.Span("DECOY_", className="mv-pep-decoy"))
        rest = rest[len("DECOY_") :]
    for token in _MOD.split(rest):
        if not token:
            continue
        if _MOD.fullmatch(token):
            parts.append(html.Span(token[1:-1], className="mv-pep-mod"))
        else:
            parts.append(token)
    return html.Span(parts, className="mv-pep", style={"fontSize": size})


def chip(
    text: str,
    colour: str = "gray",
    *,
    tip: str | None = None,
    variant: str = "light",
    left: Any = None,
    size: str = "md",
) -> Any:
    badge = dmc.Badge(
        text,
        color=colour,
        variant=variant,
        leftSection=left,
        size=size,
        style={"textTransform": "none"},
    )
    if tip:
        return dmc.Tooltip(badge, label=tip)
    return badge


def section(
    title: str,
    *children: Any,
    right: Any = None,
    subtitle: str | None = None,
    id: str | None = None,
    p: str = "md",
) -> dmc.Card:
    head = [html.Div(title, className="mv-section-title")]
    if subtitle:
        head.append(dmc.Text(subtitle, size="xs", c="dimmed", mt=2))
    header = (
        dmc.Group([html.Div(head), right], justify="space-between", align="flex-start", mb="sm")
        if right is not None
        else html.Div(head, style={"marginBottom": "10px"})
    )
    kwargs = {"id": id} if id else {}
    return dmc.Card([header, *children], p=p, **kwargs)


def graph(
    name: str,
    figure: Any,
    *,
    height: int | None = None,
    id_: Any = None,
    config: dict | None = None,
) -> dcc.Graph:
    """A Plotly graph registered for instant re-theming (pattern-matching id)."""
    style = {"height": f"{height}px"} if height else {}
    return dcc.Graph(
        id=id_ or {"type": "fig", "name": name},
        figure=figure,
        config={**GRAPH_CONFIG, **(config or {})},
        style=style,
        className="mv-graph",
    )


def stat(
    label: str, value: Any, *, hint: str | None = None, colour: str | None = None, sub: Any = None
) -> html.Div:
    """A labelled value in a metric strip."""
    title = dmc.Text(
        label, size="xs", c="dimmed", fw=600, tt="uppercase", style={"letterSpacing": "0.04em"}
    )
    if hint:
        title = dmc.Tooltip(dmc.Group([title, icon("info", 12)], gap=4), label=hint)
    body = [
        title,
        dmc.Text(
            value if isinstance(value, str) else fmt(value),
            size="lg",
            fw=650,
            c=colour,
            style={"fontVariantNumeric": "tabular-nums"},
        ),
    ]
    if sub is not None:
        body.append(sub if not isinstance(sub, str) else dmc.Text(sub, size="xs", c="dimmed"))
    return html.Div(body)


def pbar(target_pct: float | None, decoy_pct: float | None) -> html.Div:
    """Percentile bar: filled to the target percentile, with a mark at the decoy one."""
    children = []
    if target_pct is not None:
        children.append(html.Div(className="mv-pbar-fill", style={"width": f"{target_pct:.1f}%"}))
    if decoy_pct is not None:
        children.append(
            html.Div(className="mv-pbar-mark", style={"left": f"calc({decoy_pct:.1f}% - 1px)"})
        )
    return html.Div(children, className="mv-pbar")


def empty(text: str, icon_name: str = "info") -> dmc.Center:
    return dmc.Center(
        dmc.Stack(
            [icon(icon_name, 28), dmc.Text(text, c="dimmed", size="sm")], align="center", gap=6
        ),
        h=160,
    )


def notice_list(items: Sequence[tuple[str, str]]) -> dmc.Stack:
    return dmc.Stack(
        [
            dmc.Alert(
                text,
                title=code.replace("_", " ") if code else None,
                color="yellow",
                variant="light",
                icon=icon("alert", 18),
            )
            for code, text in items
        ],
        gap="xs",
    )


def fmt_bytes(n: Any) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0
    return ""


def status_badge(status: Any) -> Any:
    text = str(status or "")
    colour = (
        "green"
        if text in ("present", "found", "ok", "equal")
        else "red"
        if text in ("missing", "unreadable", "error")
        else "yellow"
    )
    return dmc.Badge(
        text or "unknown",
        color=colour,
        size="sm",
        variant="light",
        style={"textTransform": "none", "overflow": "visible"},
        miw="fit-content",
    )


def data_table(
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
    *,
    numeric: Sequence[int] = (),
    max_height: int | None = None,
    striped: bool = False,
    min_width: int | None = None,
) -> Any:
    """A compact Mantine table; cells may hold components. ``numeric`` are column indexes.

    ``max_height`` makes the body scroll under a sticky header; ``min_width`` makes a wide
    table scroll sideways instead of squeezing its columns.
    """
    right = set(numeric)
    head = dmc.TableThead(
        dmc.TableTr(
            [
                dmc.TableTh(c, className="mv-num" if i in right else None)
                for i, c in enumerate(columns)
            ]
        )
    )
    body = dmc.TableTbody(
        [
            dmc.TableTr(
                [
                    dmc.TableTd(
                        fmt(v) if isinstance(v, bool | int | float) else v,
                        className="mv-num" if i in right else None,
                    )
                    for i, v in enumerate(row)
                ]
            )
            for row in rows
        ]
    )
    table = dmc.Table(
        [head, body],
        highlightOnHover=True,
        striped=striped,
        verticalSpacing=6,
        horizontalSpacing="sm",
        className="mv-table",
        stickyHeader=max_height is not None,
    )
    if max_height is not None:
        return dmc.ScrollArea(table, h=max_height, type="auto", offsetScrollbars=True)
    if min_width:
        return dmc.TableScrollContainer(table, minWidth=min_width)
    return table


def frame_table(
    df: Any,
    *,
    labels: dict[str, str] | None = None,
    max_height: int | None = None,
    min_width: int | None = None,
) -> Any:
    """A data_table of a small DataFrame, with numbers right-aligned."""
    import pandas as pd

    cols = list(df.columns)
    numeric = [i for i, c in enumerate(cols) if pd.api.types.is_numeric_dtype(df[c])]
    rows = [[_cell(v) for v in record] for record in df.itertuples(index=False, name=None)]
    return data_table(
        rows,
        [(labels or {}).get(c, c) for c in cols],
        numeric=numeric,
        max_height=max_height,
        min_width=min_width,
    )


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if hasattr(value, "item"):
        with contextlib.suppress(ValueError, AttributeError):
            value = value.item()
    if isinstance(value, float):
        return fmt(value)
    return value if isinstance(value, int | str) else str(value)


def precursor_href(base: str, run: str, cid: int) -> str:
    return f"{base}precursor?{urlencode({'run': run, 'cid': int(cid)})}"


def unit_colour(unit: str) -> str:
    return UNIT_MANTINE.get(unit, "gray")
