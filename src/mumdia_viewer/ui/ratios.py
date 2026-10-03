"""Condition ratios per species (P2 view 9).

log2(A / B) of the protein groups and the precursors between two conditions, per
species, for mixed-species benchmarks. The runs' conditions come from the shared store
``mv-conditions`` (edited on the Quant QC page; empty means the suggestion from the
mzML names). The species of a key is the viewer's rule on its protein group's member
names: the entry-name suffix, from a list the user edits. Expected ratios are what the
user enters (the HYE benchmark design is offered as a button and named as such); the
viewer never infers them. Every ratio, summary and histogram is the viewer's
(:func:`mumdia_viewer.data.across.condition_ratios`).

Ids start with ``cr-``; the browser code is ``window.dash_clientside.mvr``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import dash_mantine_components as dmc
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import across as A
from mumdia_viewer.data.quantqc import condition_groups, resolve_conditions, suggest_conditions

from . import runs_figures as rf
from .figures import empty_figure
from .icons import icon
from .protein_view import protein_href
from .quant_cards import loading, qq_section
from .runs_cards import fact, runs_href
from .state import PageContext, href, parse_threshold, stop_label
from .widgets import empty, graph, spark_bar

log = logging.getLogger(__name__)

CONDITIONS_STORE = "mv-conditions"
LEVELS = (("protein", "Protein groups"), ("precursor", "Precursors"))
LEVEL_NOUN = {"protein": "protein groups", "precursor": "precursors"}
SOURCE_TIP = (
    "MaxLFQ: the experiment's lfq_maxlfq tables (after one median-ratio size factor per "
    "run). Per-run quant: each run's own peptide_quant (the maximum over the run's rows of a "
    "precursor) and protein_group_quant tables, not normalized across runs."
)
SUMMARY_TIP = (
    "How a condition's value is formed from its runs: the median or the mean of the linear "
    "quantities of the condition's runs that have a value. Computed by the viewer."
)
NEED_TIP = (
    "The values a condition needs before it has a value: at least one, at least two (one "
    "when the condition has a single run), or a value in every run of the condition."
)
DESIGN_TEXT = (
    "The HYE benchmark design (ProteoBench mixed-species LFQ): human 1:1, yeast 2:1, E. coli "
    "1:4, condition A over B. Use it only when your samples follow this design: the viewer "
    "cannot know a sample's composition, so these values are entered by you."
)


def _scheme(value: Any) -> str:
    return "dark" if value == "dark" else "light"


def _source(rs: ResultSet, value: Any) -> str:
    return "lfq" if rs.is_experiment and value != "quant" else "quant"


def _segment(id_: str, data: list[tuple[str, str]], value: str, colour: str = "indigo") -> Any:
    return dmc.SegmentedControl(
        id=id_,
        data=[{"value": v, "label": label} for v, label in data],
        value=value,
        size="xs",
        radius="md",
        color=colour,
    )


def _labelled(label: str, tip: str, control: Any) -> Any:
    return html.Div(
        [
            dmc.Tooltip(
                html.Span([label, icon("info", 11)], className="cr-tool-label"),
                label=tip,
                position="bottom",
            ),
            control,
        ],
        className="cr-tool",
    )


def flip_ratio(text: Any) -> str:
    """``"2:1"`` written the other way round (``"1:2"``); a bare ``r`` becomes ``"1:r"``."""
    t = str(text or "").strip()
    if not t:
        return t
    for sep in (":", "/"):
        if sep in t:
            a, _, b = t.partition(sep)
            return f"{b.strip()}:{a.strip()}"
    return f"1:{t}" if A.parse_ratio(t) is not None else t


def stored_values(stored: Any, a: Any, b: Any) -> dict[str, str]:
    """The entered ratios as ``a : b``.

    The store is ``{"pair": [A, B], "values": {suffix: "x:y"}}``: the ratios are kept
    with the conditions they were entered for, so they describe the samples. Entered for
    ``(b, a)``, they are turned round; entered for any other pair, they are used as
    written. A plain ``{suffix: "x:y"}`` mapping is read as written.
    """
    if not isinstance(stored, Mapping):
        return {}
    values = stored.get("values") if isinstance(stored.get("values"), Mapping) else stored
    values = {str(k): str(v) for k, v in values.items() if isinstance(v, str | int | float)}
    pair = stored.get("pair")
    if isinstance(pair, list | tuple) and len(pair) == 2 and a != b and list(pair) == [b, a]:
        return {k: flip_ratio(v) for k, v in values.items()}
    return values


def expected_log2(
    stored: Any, suffixes: Any, a: Any = None, b: Any = None
) -> dict[str, float | None]:
    """log2 of the expected A:B of each suffix (None where nothing valid was entered)."""
    values = stored_values(stored, a, b)
    return {str(s): A.parse_ratio(values.get(str(s))) for s in suffixes or ()}


def clean_suffixes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return A.DEFAULT_SUFFIXES
    out: list[str] = []
    for v in value:
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return tuple(out)


# --------------------------------------------------------------------------- parts


def hero(ctx: PageContext, *, with_facts: bool = True) -> Any:
    rs = ctx.rs
    kind = f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"
    title = html.Div(
        [
            dmc.Text(f"Condition ratios · {kind}", className="mv-eyebrow"),
            html.Div("Condition ratios", className="mv-title"),
            dmc.Text(
                "log2 A/B per species from the engine's quantities. The ratios, the condition "
                "summaries, the species and the histograms are the viewer's.",
                size="xs",
                c="dimmed",
                mt=2,
            ),
        ]
    )
    facts = [
        html.Div(
            fact(
                "protein groups in both",
                "…",
                "Accepted protein groups (pg_q_value at the header threshold) with a value in "
                "both conditions",
            ),
            id="cr-fact-protein",
        ),
        html.Div(
            fact(
                "precursors in both",
                "…",
                "Accepted precursors (precursor_q at the header threshold) with a value in both "
                "conditions",
            ),
            id="cr-fact-precursor",
        ),
    ]
    return dmc.Group(
        [title, dmc.Group(facts if with_facts else [], gap="sm", wrap="nowrap")],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="xr-hero",
    )


def toolbar(ctx: PageContext, names: list[str], a: str, b: str) -> Any:
    opts = [{"value": n, "label": n} for n in names]
    pair = html.Div(
        [
            html.Span("A", className="cr-ab cr-ab-a"),
            dmc.Select(
                id="cr-a",
                data=opts,
                value=a,
                allowDeselect=False,
                size="xs",
                radius="md",
                w=120,
                comboboxProps={"shadow": "md"},
                **{"aria-label": "Condition A"},
            ),
            dmc.Tooltip(
                dmc.ActionIcon(
                    icon("compare", 14),
                    id="cr-swap",
                    variant="subtle",
                    color="gray",
                    n_clicks=0,
                    **{"aria-label": "Swap A and B"},
                ),
                label="Swap A and B",
            ),
            html.Span("B", className="cr-ab cr-ab-b"),
            dmc.Select(
                id="cr-b",
                data=opts,
                value=b,
                allowDeselect=False,
                size="xs",
                radius="md",
                w=120,
                comboboxProps={"shadow": "md"},
                **{"aria-label": "Condition B"},
            ),
        ],
        className="cr-tool",
    )
    pill = dmc.Tooltip(
        html.Div(
            [
                icon("target", 14),
                html.Span(f"precursor_q · pg_q_value ≤ {stop_label(ctx.threshold)}", id="cr-t"),
            ],
            className="cr-t",
        ),
        label="The keys are the accepted ones at the header threshold: precursors at "
        "precursor_q, protein groups at pg_q_value (target rows, experiment-wide).",
        position="bottom",
    )
    parts = [
        _labelled("Conditions", "A over B: the ratio is log2(A / B).", pair),
        _labelled(
            "Quantities",
            SOURCE_TIP,
            _segment("cr-source", [("lfq", "MaxLFQ"), ("quant", "Per-run quant")], "lfq"),
        ),
        _labelled(
            "Summary",
            SUMMARY_TIP,
            _segment("cr-summary", [("median", "Median"), ("mean", "Mean")], "median", "teal"),
        ),
        _labelled(
            "Needs",
            NEED_TIP,
            _segment(
                "cr-need",
                [("one", "≥ 1 value"), ("two", "≥ 2 values"), ("all", "Every run")],
                "two",
                "teal",
            ),
        ),
        pill,
    ]
    return dmc.Card(html.Div(parts, className="cr-tools"), className="cr-toolbar", p=6)


def species_inputs(suffixes: tuple[str, ...], stored: Any, a: Any = None, b: Any = None) -> Any:
    values = stored_values(stored, a, b)
    colours = rf.species_colours(suffixes)
    rows = []
    for s in suffixes:
        rows.append(
            html.Div(
                [
                    html.Span(className="cr-dot", style={"background": colours[s]}),
                    html.Span(rf.species_name(s), className="cr-sp-name"),
                    dmc.TextInput(
                        id={"type": "cr-exp", "sp": s},
                        value=str(values.get(s) or ""),
                        placeholder="A:B, e.g. 2:1",
                        size="xs",
                        radius="md",
                        debounce=400,
                        w=110,
                        className="cr-exp-input",
                        **{"aria-label": f"Expected A:B of {rf.species_name(s)}"},
                    ),
                ],
                className="cr-sp-row",
            )
        )
    if not rows:
        return dmc.Text("Add an entry-name suffix such as _HUMAN.", size="xs", c="dimmed")
    return html.Div(rows, className="cr-sp-rows")


def species_card(suggestion: Mapping[str, str]) -> Any:
    suffixes = A.DEFAULT_SUFFIXES
    tags = dmc.TagsInput(
        id="cr-suffixes",
        value=list(suffixes),
        data=[],
        size="xs",
        radius="md",
        placeholder="Add a suffix and press Enter",
        persistence=True,
        persistence_type="local",
        clearable=False,
        splitChars=[",", " "],
        **{"aria-label": "Species suffixes"},
    )
    buttons = dmc.Group(
        [
            dmc.Tooltip(
                dmc.Button(
                    "HYE benchmark design",
                    id="cr-hye",
                    n_clicks=0,
                    size="compact-xs",
                    variant="light",
                    color="indigo",
                    leftSection=icon("flask", 12),
                ),
                label=DESIGN_TEXT,
            ),
            dmc.Button(
                "Clear",
                id="cr-clear",
                n_clicks=0,
                size="compact-xs",
                variant="subtle",
                color="gray",
                leftSection=icon("x", 11),
            ),
        ],
        gap=6,
    )
    return qq_section(
        "Species and expected ratios",
        dmc.Text("Entry-name suffixes", className="cr-label"),
        tags,
        dmc.Text(
            "Expected A : B (entered by you)", id="cr-exp-label", className="cr-label", mt="sm"
        ),
        html.Div(species_inputs(suffixes, None), id="cr-species"),
        buttons,
        html.Div(
            "The HYE button enters human 1:1, yeast 2:1, E. coli 1:4: the benchmark design of "
            "the ProteoBench mixed-species samples, entered by you, not measured. An empty "
            "field draws no expected line.",
            className="cr-how",
        ),
        dmc.Text("Conditions", className="cr-label", mt="sm"),
        html.Div(id="cr-cond-text", className="cr-cond-text"),
        dcc.Link(
            dmc.Group(
                [
                    dmc.Text("Edit the conditions on the Quant QC page", size="xs"),
                    icon("right", 11),
                ],
                gap=2,
            ),
            href="quant",
            id="cr-quant-link",
            className="xr-link",
        ),
        help="A key's species: the entry-name suffix of its protein group's members (ALBU_HUMAN "
        "is _HUMAN), the viewer's rule. A group whose members have different suffixes is "
        "'mixed', one without a listed suffix 'other'; both are counted but not drawn by "
        "default. The suffixes and the expected ratios are kept in this browser.",
        subtitle="The viewer's species rule and your expected ratios",
        className="cr-species-card",
    )


def _th(text: str, tip: str | None = None, *, num: bool = True) -> Any:
    label: Any = text
    if tip:
        label = dmc.Tooltip(html.Span([text, icon("info", 10)], className="xr-th"), label=tip)
    return dmc.TableTh(label, className="mv-num" if num else None)


def _fmt_log2(v: Any, signed: bool = True) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    return f"{f:+.2f}" if signed else f"{f:.2f}"


def species_table(result: A.RatioResult, expected: Mapping[str, float | None]) -> Any:
    """Per species: keys, keys in both conditions (bar: share), only A, only B, and the
    median log2(A/B) against the expected ratio. The rest is in the species' tooltip."""
    colours = rf.species_colours(result.suffixes)
    sp = result.species
    total = int(sp["keys"].sum())
    head = dmc.TableThead(
        dmc.TableTr(
            [
                _th(
                    "species",
                    "The viewer's suffix rule on the protein group's members; hover a species "
                    "for its spread and the keys without a value",
                    num=False,
                ),
                _th("keys", f"Accepted {LEVEL_NOUN[result.level]} of the species"),
                _th(
                    "in both",
                    "Keys with a value in both conditions (a ratio). Bar: the share of the "
                    "species' keys, 0 to 100 %.",
                ),
                _th(
                    "only A · B",
                    f"Keys with a value in {result.a} only, and in {result.b} only",
                ),
                _th("median", "Median of log2(A/B) over the keys in both (the viewer's)"),
                _th("expected", "log2 of the expected A:B you entered"),
                _th("Δ", "median - expected, log2; orange beyond ±0.5"),
            ]
        )
    )
    body = []
    for row in sp.to_dict("records"):
        if row["keys"] == 0:
            continue
        name = row["species"]
        e = expected.get(name)
        delta = row["median"] - e if e is not None and math.isfinite(row["median"]) else None
        share = row["both"] / row["keys"] if row["keys"] else 0.0
        spread = (
            f"IQR {_fmt_log2(row['q25'])} to {_fmt_log2(row['q75'])}, mean "
            f"{_fmt_log2(row['mean'])}, sd {_fmt_log2(row['sd'], signed=False) or 'n/a'} "
            "(numpy; percentiles with linear interpolation, sd with n - 1). "
            if math.isfinite(row["q25"])
            else ""
        )
        what = {
            A.MIXED: "Protein groups whose members carry different listed suffixes. ",
            A.OTHER: "Protein groups without a listed suffix. ",
        }.get(name, f"Protein groups whose members end with {name}. ")
        cells = [
            dmc.Tooltip(
                html.Span(
                    [
                        html.Span(className="cr-dot", style={"background": colours.get(name)}),
                        rf.species_name(name),
                    ],
                    className="cr-sp-cell",
                ),
                label=f"{what}{spread}{row['neither']:,} keys have no value in either condition.",
                w=320,
            ),
            f"{row['keys']:,}",
            spark_bar(
                share * 100,
                lo=0,
                hi=100,
                width=32,
                text=f"{row['both']:,}",
                colour=colours.get(name, "var(--mantine-color-gray-5)"),
                tip=f"{share:.1%} of the species' keys",
            ),
            f"{row['only_a']:,} · {row['only_b']:,}",
            html.B(_fmt_log2(row["median"])),
            _fmt_log2(e) if e is not None else html.Span("none", className="xr-dim"),
            html.Span(
                _fmt_log2(delta),
                className="cr-delta"
                + (" cr-delta-far" if delta is not None and abs(delta) > 0.5 else ""),
            )
            if delta is not None
            else "",
        ]
        body.append(
            dmc.TableTr(
                [
                    dmc.TableTd(c, className=None if j == 0 else "mv-num")
                    for j, c in enumerate(cells)
                ],
                className="cr-faint" if name in (A.MIXED, A.OTHER) else None,
            )
        )
    body.append(
        dmc.TableTr(
            [
                dmc.TableTd(
                    dmc.Tooltip(
                        html.Span("all"),
                        label=f"{int(sp['neither'].sum()):,} keys have no value in either "
                        "condition (missing, or too few values for the rule).",
                    ),
                    className="cr-total",
                ),
                dmc.TableTd(f"{total:,}", className="mv-num cr-total"),
                dmc.TableTd(f"{int(sp['both'].sum()):,}", className="mv-num cr-total"),
                dmc.TableTd(
                    f"{int(sp['only_a'].sum()):,} · {int(sp['only_b'].sum()):,}",
                    className="mv-num cr-total",
                ),
                *[dmc.TableTd("", className="cr-total") for _ in range(3)],
            ]
        )
    )
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        verticalSpacing=5,
        horizontalSpacing=8,
        highlightOnHover=True,
        className="mv-table cr-table",
    )
    return dmc.TableScrollContainer(table, minWidth=480)


def level_card(level: str, scheme: str) -> Any:
    return qq_section(
        f"log2 A/B · {dict(LEVELS)[level]}",
        loading(
            html.Div(
                [
                    graph(f"cr-dens-{level}", empty_figure("", scheme, height=330)),
                    html.Div(id=f"cr-tab-{level}", className="cr-tab"),
                ]
            )
        ),
        count="",
        count_id=f"cr-count-{level}",
        help=(
            "Histogram of log2(A/B) per species on 0.1-wide bins (the viewer's), as the share "
            "of the species' keys in each bin, so species of different sizes compare. Solid "
            "line and number: the species' median; dashed: the expected ratio you entered. "
            "Ratios beyond ±6 are put in the outer bins. Click a legend entry to hide a species."
        ),
        subtitle="",
        subtitle_id=f"cr-sub-{level}",
    )


# --------------------------------------------------------------------------- answers


def _ratio(
    rs: ResultSet,
    level: str,
    stored: Any,
    a: Any,
    b: Any,
    source: Any,
    summary: Any,
    need: Any,
    t: float,
    suffixes: tuple[str, ...],
) -> A.RatioResult:
    key = (
        "ratios.result",
        level,
        _source(rs, source),
        tuple(sorted(resolve_conditions(rs, stored).items())),
        a,
        b,
        summary,
        need,
        t,
        suffixes,
    )
    return A.cached(
        rs,
        key,
        lambda: A.condition_ratios(
            rs,
            level,
            _source(rs, source),
            stored,
            a,
            b,
            summary=summary if summary in A.SUMMARIES else "median",
            need=need if need in A.NEEDS else "two",
            accepted_at=t,
            suffixes=suffixes,
        ),
    )


def level_answer(
    rs: ResultSet,
    level: str,
    stored: Any,
    a: Any,
    b: Any,
    source: Any,
    summary: Any,
    need: Any,
    t: float,
    suffixes: tuple[str, ...],
    expected_store: Any,
    scheme: str,
) -> tuple[Any, Any, str, Any, Any]:
    """(figure, table, count, subtitle, fact) of one level."""
    expected = expected_log2(expected_store, suffixes, a, b)
    noun = LEVEL_NOUN[level]
    try:
        r = _ratio(rs, level, stored, a, b, source, summary, need, t, suffixes)
    except ViewerError as exc:
        return (
            empty_figure(str(exc), scheme, height=330),
            None,
            "",
            str(exc),
            fact(f"{noun} in both", "n/a", str(exc)),
        )
    edges, counts, clipped = A.ratio_histogram(r)
    fig = rf.ratio_density_figure(r, edges, counts, expected, scheme)
    n_clip = sum(clipped.values())
    needs = {"one": "≥ 1 value", "two": "≥ 2 values", "all": "a value in every run"}[r.need]
    src = "MaxLFQ" if r.source == "lfq" else "per-run quant"
    sub = dmc.Tooltip(
        html.Span(
            f"{r.a} ({', '.join(rf.run_name(x) for x in r.runs_a)}) over {r.b} "
            f"({', '.join(rf.run_name(x) for x in r.runs_b)}) · {r.summary} of {src} · "
            f"{needs} per condition" + (f" · {n_clip:,} beyond the axis" if n_clip else ""),
            className="cr-sub",
        ),
        label=r.label
        + (f" {n_clip:,} ratios beyond ±6 are counted in the outer bins." if n_clip else ""),
        w=380,
    )
    keys = len(r.table)
    fact_box = fact(
        f"{noun} in both",
        f"{r.n_both:,}",
        f"Accepted {noun} ({'precursor_q' if level == 'precursor' else 'pg_q_value'} ≤ "
        f"{stop_label(t)}) with a value in both {r.a} and {r.b}, of {keys:,} accepted. "
        f"{r.label}",
    )
    return fig, species_table(r, expected), f"{r.n_both:,} of {keys:,} in both", sub, fact_box


def scatter_answer(
    rs: ResultSet,
    level: str,
    stored: Any,
    a: Any,
    b: Any,
    source: Any,
    summary: Any,
    need: Any,
    t: float,
    suffixes: tuple[str, ...],
    expected_store: Any,
    scheme: str,
) -> Any:
    try:
        r = _ratio(rs, level, stored, a, b, source, summary, need, t, suffixes)
    except ViewerError as exc:
        return empty_figure(str(exc), scheme, height=420)
    return rf.ratio_scatter_figure(r, expected_log2(expected_store, suffixes, a, b), scheme)


def conditions_answer(rs: ResultSet, stored: Any, a: Any, b: Any) -> tuple[Any, ...]:
    """Options and values of the A and B selects, and the conditions in words."""
    resolved = resolve_conditions(rs, stored)
    groups = condition_groups(resolved, [r.name for r in rs.runs])
    names = list(groups)
    opts = [{"value": n, "label": n} for n in names]
    a = a if a in groups else (names[0] if names else None)
    b = b if b in groups and b != a else next((n for n in names if n != a), None)
    lines = [
        html.Div(
            [
                html.Span(name, className="cr-cond-name"),
                html.Span(", ".join(rf.run_name(r) for r in runs), className="cr-cond-runs"),
            ],
            className="cr-cond-line",
        )
        for name, runs in groups.items()
    ]
    if len(names) < 2:
        lines.append(
            dmc.Text("Every run is in one condition: a ratio needs two.", size="xs", c="orange")
        )
    return opts, a, opts, b, lines


def _single(ctx: PageContext) -> Any:
    return dmc.Stack(
        [
            hero(ctx, with_facts=False),
            dmc.Card(
                empty(
                    "Condition ratios need an experiment with at least two conditions. This is "
                    "a single run: it has one.",
                    "scale",
                )
            ),
        ],
        gap="lg",
    )


def layout(ctx: PageContext) -> Any:
    rs = ctx.rs
    if len(rs.runs) < 2:
        return html.Div(_single(ctx), className="cr-root")
    suggestion = suggest_conditions(rs)
    groups = condition_groups(suggestion, [r.name for r in rs.runs])
    names = list(groups)
    a = names[0]
    b = names[1] if len(names) > 1 else names[0]
    ma_card = qq_section(
        "Ratio against abundance",
        loading(graph("cr-ma", empty_figure("", ctx.scheme, height=420))),
        help="Each key with a value in both conditions: log2(A/B) against log10 of the mean of "
        "the two condition values (the viewer's). Solid lines: the species' medians; dashed: "
        "the expected ratios you entered. Click a point to open its protein page, or a "
        "precursor across the runs.",
        subtitle="Ratio compression and intensity dependence; click a point to open it",
        right=_segment("cr-ma-level", list(LEVELS), "protein"),
    )
    return html.Div(
        dmc.Stack(
            [
                dcc.Store(id="cr-expected", storage_type="local"),
                # cr-init starts follow_conditions on load; cr-conds is the resolved
                # {run: condition} that the figures follow.
                dcc.Store(id="cr-init", data=0),
                dcc.Store(id="cr-conds"),
                dcc.Store(id="cr-design", data=A.HYE_DESIGN),
                hero(ctx),
                toolbar(ctx, names, a, b),
                dmc.Grid(
                    [
                        dmc.GridCol(species_card(suggestion), span={"base": 12, "md": 5, "lg": 4}),
                        dmc.GridCol(ma_card, span={"base": 12, "md": 7, "lg": 8}),
                    ],
                    gutter="lg",
                ),
                dmc.Grid(
                    [
                        dmc.GridCol(level_card("protein", ctx.scheme), span={"base": 12, "lg": 6}),
                        dmc.GridCol(
                            level_card("precursor", ctx.scheme), span={"base": 12, "lg": 6}
                        ),
                    ],
                    gutter="lg",
                ),
            ],
            gap="lg",
        ),
        className="cr-root",
    )


# --------------------------------------------------------------------------- callbacks


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""
    app.clientside_callback(
        ClientsideFunction("mvr", "swap"),
        Output("cr-a", "value", allow_duplicate=True),
        Output("cr-b", "value", allow_duplicate=True),
        Output({"type": "cr-exp", "sp": ALL}, "value", allow_duplicate=True),
        Input("cr-swap", "n_clicks"),
        State("cr-a", "value"),
        State("cr-b", "value"),
        State({"type": "cr-exp", "sp": ALL}, "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvr", "design"),
        Output({"type": "cr-exp", "sp": ALL}, "value"),
        Input("cr-hye", "n_clicks"),
        State({"type": "cr-exp", "sp": ALL}, "id"),
        State("cr-design", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvr", "clear"),
        Output({"type": "cr-exp", "sp": ALL}, "value", allow_duplicate=True),
        Input("cr-clear", "n_clicks"),
        State({"type": "cr-exp", "sp": ALL}, "id"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvr", "expected"),
        Output("cr-expected", "data"),
        Input({"type": "cr-exp", "sp": ALL}, "value"),
        State({"type": "cr-exp", "sp": ALL}, "id"),
        State("cr-expected", "data"),
        State("cr-a", "value"),
        State("cr-b", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("cr-a", "data"),
        Output("cr-a", "value"),
        Output("cr-b", "data"),
        Output("cr-b", "value"),
        Output("cr-cond-text", "children"),
        Output("cr-quant-link", "href"),
        Output("cr-conds", "data"),
        Input("cr-init", "data"),
        State(CONDITIONS_STORE, "data"),
        State("cr-a", "value"),
        State("cr-b", "value"),
    )
    def follow_conditions(_init, stored, a, b):
        rs = get_rs()
        resolved = resolve_conditions(rs, stored)
        return (*conditions_answer(rs, resolved, a, b), href(base, "quant"), resolved)

    @app.callback(
        Output("cr-species", "children"),
        Input("cr-suffixes", "value"),
        State("cr-expected", "data"),
        State("cr-a", "value"),
        State("cr-b", "value"),
    )
    def follow_suffixes(suffixes, stored, a, b):
        return species_inputs(clean_suffixes(suffixes), stored, a, b)

    # The page's copy of the conditions (cr-conds), not the shared store, drives the
    # figures: a callback with an input in the app shell would fire on every page.
    inputs = [
        Input("cr-conds", "data"),
        Input("cr-a", "value"),
        Input("cr-b", "value"),
        Input("cr-source", "value"),
        Input("cr-summary", "value"),
        Input("cr-need", "value"),
        Input("threshold", "data"),
        Input("cr-suffixes", "value"),
        Input("cr-expected", "data"),
    ]

    @app.callback(
        Output({"type": "fig", "name": "cr-dens-protein"}, "figure", allow_duplicate=True),
        Output("cr-tab-protein", "children"),
        Output("cr-count-protein", "children"),
        Output("cr-sub-protein", "children"),
        Output("cr-fact-protein", "children"),
        Output({"type": "fig", "name": "cr-dens-precursor"}, "figure", allow_duplicate=True),
        Output("cr-tab-precursor", "children"),
        Output("cr-count-precursor", "children"),
        Output("cr-sub-precursor", "children"),
        Output("cr-fact-precursor", "children"),
        Output("cr-t", "children"),
        Output("cr-exp-label", "children"),
        *inputs,
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow(stored, a, b, source, summary, need, t, suffixes, expected, scheme):
        if stored is None:  # before follow_conditions has resolved the conditions
            return (no_update,) * 12
        rs = get_rs()
        thr = parse_threshold(t)
        sfx = clean_suffixes(suffixes)
        out: list[Any] = []
        for level, _ in LEVELS:
            out.extend(
                level_answer(
                    rs,
                    level,
                    stored,
                    a,
                    b,
                    source,
                    summary,
                    need,
                    thr,
                    sfx,
                    expected,
                    _scheme(scheme),
                )
            )
        out.append(f"precursor_q · pg_q_value ≤ {stop_label(thr)}")
        out.append(f"Expected {a} : {b} (entered by you)" if a and b else "Expected A : B")
        return tuple(out)

    @app.callback(
        Output({"type": "fig", "name": "cr-ma"}, "figure", allow_duplicate=True),
        Input("cr-ma-level", "value"),
        *inputs,
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow_ma(level, stored, a, b, source, summary, need, t, suffixes, expected, scheme):
        if stored is None:
            return no_update
        return scatter_answer(
            get_rs(),
            level if level in dict(LEVELS) else "protein",
            stored,
            a,
            b,
            source,
            summary,
            need,
            parse_threshold(t),
            clean_suffixes(suffixes),
            expected,
            _scheme(scheme),
        )

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input({"type": "fig", "name": "cr-ma"}, "clickData"),
        prevent_initial_call=True,
    )
    def open_point(click):
        return point_href(base, click) or no_update


def point_href(base: str, click: Any) -> str | None:
    """The page a click on the ratio scatter opens: a protein group's protein page, or a
    precursor across the runs (the point's text is ``"<peptidoform> <charge>+"``)."""
    try:
        text = click["points"][0]["text"]
    except (TypeError, KeyError, IndexError):
        return None
    if not isinstance(text, str) or not text:
        return None
    pep, _, z = text.rpartition(" ")
    if pep and z.endswith("+") and z[:-1].isdigit():
        return runs_href(base, pep, int(z[:-1]))
    return protein_href(base, text)
