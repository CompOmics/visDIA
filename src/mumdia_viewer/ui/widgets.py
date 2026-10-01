"""Shared building blocks of the pages (Mantine components)."""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import dash_mantine_components as dmc
from dash import dcc, html

from .icons import icon
from .theme import GRAPH_CONFIG, UNIT_MANTINE, mod_name, mod_style


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


def fmt_q(value: Any, threshold: float | None = None) -> str:
    """A q value in the grid's form: four decimals from 0.001, else ``3.80e-5``.

    A value that differs from ``threshold`` but would read as it (0.010003 as 0.0100 at
    q <= 0.01) gets more digits until it does not, so a printed value never contradicts
    its pass or fail mark.
    """
    if value is None:
        return ""
    v = float(value)
    if math.isnan(v):
        return ""
    if v == 0:
        return "0"

    def form(extra: int) -> str:
        if v < 0.001:
            mantissa, _, exponent = f"{v:.{2 + extra}e}".partition("e")
            return f"{mantissa}e{int(exponent)}"
        return f"{v:.{4 + extra}f}"

    text = form(0)
    if threshold is None or v == threshold:
        return text
    for extra in range(1, 14):
        shown = float(text)
        if shown != threshold and (shown <= threshold) == (v <= threshold):
            return text
        text = form(extra)
    return repr(v)


@dataclass(frozen=True)
class Residue:
    aa: str
    mods: tuple[str, ...] = ()


@dataclass(frozen=True)
class Peptidoform:
    """A ProForma-lite peptidoform split into residues and their modifications."""

    decoy: bool
    residues: tuple[Residue, ...]
    nterm: tuple[str, ...] = ()
    cterm: tuple[str, ...] = ()

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)


_TAG = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)")
_CTERM = re.compile(r"-((?:\[[^\]]*\]|\([^)]*\))+)$")


def parse_peptidoform(text: str | None) -> Peptidoform:
    """Split ``[Acetyl]-PEM[Oxidation]TIDE-[Amidated]`` (or ``M(Oxidation)``) into residues.

    A tag after a residue modifies that residue; tags before ``-`` at the start are
    N-terminal, tags after ``-`` at the end C-terminal. Any other character is kept as a
    residue, so nothing of the text is lost.
    """
    rest = text or ""
    decoy = rest.startswith("DECOY_")
    if decoy:
        rest = rest[len("DECOY_") :]
    nterm: list[str] = []
    while True:
        m = _TAG.match(rest)
        if m is None or not rest[m.end() :].startswith("-"):
            break
        nterm.append(m.group(1) if m.group(1) is not None else m.group(2))
        rest = rest[m.end() + 1 :]
    cterm: list[str] = []
    cm = _CTERM.search(rest)
    if cm is not None:
        cterm = [a or b for a, b in _TAG.findall(cm.group(1))]
        rest = rest[: cm.start()]
    residues: list[Residue] = []
    pos = 0
    while pos < len(rest):
        m = _TAG.match(rest, pos)
        if m is not None:
            tag = m.group(1) if m.group(1) is not None else m.group(2)
            if residues:
                last = residues[-1]
                residues[-1] = Residue(last.aa, (*last.mods, tag))
            else:
                nterm.append(tag)
            pos = m.end()
            continue
        residues.append(Residue(rest[pos]))
        pos += 1
    return Peptidoform(decoy, tuple(residues), tuple(nterm), tuple(cterm))


def _mod_span(label: str, mods: tuple[str, ...]) -> html.Span:
    styles = [mod_style(m) for m in mods]
    names = ", ".join(mod_name(m) for m in mods)
    return html.Span(
        [label, html.Sup("+".join(t for t, _ in styles), className="mv-pep-tag")],
        className="mv-pep-modres",
        style={"color": styles[0][1]},
        title=names if label in ("n", "c") else f"{label}: {names}",
    )


def peptidoform(text: str | None, *, size: str = "1em") -> html.Span:
    """A peptidoform as PeptideShaker draws it: modified residues coloured and tagged.

    The tag is a short form of the modification (``ox``, ``cam``, ``ph``); the tooltip of
    the residue names it in full, the tooltip of the whole gives the text as written.
    ``DECOY_`` is shown in the decoy colour.
    """
    if not text:
        return html.Span("")
    pf = parse_peptidoform(text)
    parts: list = []
    if pf.decoy:
        parts.append(html.Span("DECOY_", className="mv-pep-decoy"))
    if pf.nterm:
        parts.append(_mod_span("n", pf.nterm))
        parts.append(html.Span("-", className="mv-pep-term"))
    plain = ""
    for r in pf.residues:
        if r.mods:
            if plain:
                parts.append(plain)
                plain = ""
            parts.append(_mod_span(r.aa, r.mods))
        else:
            plain += r.aa
    if plain:
        parts.append(plain)
    if pf.cterm:
        parts.append(html.Span("-", className="mv-pep-term"))
        parts.append(_mod_span("c", pf.cterm))
    return html.Span(parts, className="mv-pep", style={"fontSize": size}, title=text)


def validation_icon(
    q: Any,
    threshold: float | None,
    *,
    label: str | None = "target",
    column: str = "q",
    size: int = 18,
    spike: bool = False,
) -> Any:
    """PeptideShaker-style validation mark: does the row pass the threshold?

    Green check: ``q <= threshold`` on the engine's column; grey cross: above it, or no
    value; an orange D marks a decoy, a grape E an entrapment spike-in. The tooltip
    states the column, the value and the cut.
    """
    try:
        value = float(q)
    except (TypeError, ValueError):
        value = float("nan")
    shown = fmt_q(value, threshold) or "not set"
    if label == "decoy":
        mark, colour, tip = "D", "orange", f"decoy; {column} {shown}"
    elif threshold is not None and not math.isnan(value) and value <= threshold:
        mark, colour = icon("check", size - 6), "green"
        tip = f"passes: {column} {shown} ≤ {threshold:g}"
    else:
        cut = f" > {threshold:g}" if threshold is not None and not math.isnan(value) else ""
        mark, colour = icon("x", size - 6), "gray"
        tip = f"does not pass: {column} {shown}{cut}"
    if spike:
        mark, colour, tip = "E", "grape", f"entrapment spike-in; {tip}"
    badge = dmc.ThemeIcon(
        mark,
        size=size,
        radius="xl",
        color=colour,
        variant="light",
        style={"fontSize": f"{size - 8}px", "fontWeight": 700},
    )
    return dmc.Tooltip(badge, label=tip, multiline=False, w="auto")


def spark_bar(
    value: Any,
    *,
    lo: float,
    hi: float,
    scale: str = "linear",
    colour: str = "var(--mantine-color-indigo-6)",
    text: str | None = None,
    width: int = 64,
    tip: str | None = None,
) -> Any:
    """An in-cell bar with its value, like PeptideShaker's JSparklines.

    ``scale`` is ``linear``, ``log10`` or ``neglog10`` (for q values, so that a longer
    bar is a smaller q). The bar is a display only; ``tip`` should say what it encodes.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")

    def tr(x: float) -> float:
        if scale == "log10":
            return math.log10(max(x, 1e-300))
        if scale == "neglog10":
            return -math.log10(max(x, 1e-300))
        return x

    frac = 0.0
    if not math.isnan(v):
        a, b = tr(lo), tr(hi)
        frac = 0.0 if b == a else min(1.0, max(0.0, (tr(v) - a) / (b - a)))
    bar = html.Div(
        html.Div(
            className="mv-spark-fill",
            style={"width": f"{100 * frac:.1f}%", "background": colour},
        ),
        className="mv-spark",
        style={"width": f"{width}px"},
    )
    label = text if text is not None else (fmt(v) if not math.isnan(v) else "")
    body = html.Div([bar, html.Span(label, className="mv-spark-text")], className="mv-spark-cell")
    return dmc.Tooltip(body, label=tip, multiline=False, w="auto") if tip else body


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
    count: Any = None,
    help: str | None = None,
) -> dmc.Card:
    """A card with a small uppercase title, like a PeptideShaker panel.

    ``count`` is shown as a badge after the title (for example the rows of a table),
    ``help`` as an info icon with a tooltip.
    """
    title_row: list[Any] = [html.Div(title, className="mv-section-title")]
    if count is not None:
        title_row.append(
            dmc.Badge(
                count if isinstance(count, str) else fmt(count),
                size="sm",
                color="gray",
                variant="light",
                style={"textTransform": "none"},
            )
        )
    if help:
        title_row.append(dmc.Tooltip(html.Span(icon("info", 13), className="mv-help"), label=help))
    head = [dmc.Group(title_row, gap=6) if len(title_row) > 1 else title_row[0]]
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
