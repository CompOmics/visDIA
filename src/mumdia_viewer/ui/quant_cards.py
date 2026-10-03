"""Building blocks of the quant QC page: header, toolbar, editor and tables."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import dash_mantine_components as dmc
import pandas as pd
from dash import dcc, html

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data.mbr import mbr_info
from mumdia_viewer.data.quant import quant_gate
from mumdia_viewer.data.quantqc import (
    CvResult,
    ProteinProfile,
    size_factors,
)

from .icons import icon
from .quant_figures import LEVEL_NOUNS, compact, run_label
from .state import PageContext, href, stop_label
from .widgets import chip, fmt, spark_bar, validation_icon

SOURCE_TIPS = {
    "lfq": "MaxLFQ: the experiment's lfq_maxlfq tables (protein and precursor level), after "
    "one median-ratio size factor per run (run-experiment always normalizes so).",
    "quant": "Per-run quant: each run's own peptide_quant and protein_group_quant tables "
    "(top-N sums, not normalized across runs).",
}
KEYS_TIP = (
    "Accepted: the keys accepted at the header threshold on their own unit's q column, "
    "precursors at precursor_q and protein groups at pg_q_value (target rows; experiment-wide "
    "in an experiment), each with a row, so an accepted key without a quantity counts as "
    "missing. All quantified: every key with a value in any run (the LFQ key list is not "
    "FDR-controlled at the key's unit)."
)
NEED_TIP = (
    "The values a CV needs: a value in every run of its condition, or at least two values. A "
    "condition with one run gives no CV."
)


def qq_section(
    title: str,
    *children: Any,
    count: Any = None,
    count_id: str | None = None,
    help: str | Any = None,
    help_id: str | None = None,
    subtitle: Any = None,
    subtitle_id: str | None = None,
    right: Any = None,
    id: str | None = None,
    className: str | None = None,
) -> dmc.Card:
    """``widgets.section`` with ids on the count, the help and the subtitle (live parts)."""
    row: list[Any] = [html.Div(title, className="mv-section-title")]
    if count is not None or count_id is not None:
        kwargs = {"id": count_id} if count_id else {}
        row.append(
            dmc.Badge(
                count if isinstance(count, str) else fmt(count) if count is not None else "",
                size="sm",
                color="gray",
                variant="light",
                style={"textTransform": "none"},
                **kwargs,
            )
        )
    if help is not None or help_id is not None:
        kwargs = {"id": help_id} if help_id else {}
        row.append(
            dmc.Tooltip(
                html.Span(icon("info", 13), className="mv-help"),
                label=help if help is not None else "",
                **kwargs,
            )
        )
    head: list[Any] = [dmc.Group(row, gap=6)]
    if subtitle is not None or subtitle_id is not None:
        kwargs = {"id": subtitle_id} if subtitle_id else {}
        head.append(dmc.Text(subtitle, size="xs", c="dimmed", mt=2, **kwargs))
    header = (
        dmc.Group([html.Div(head), right], justify="space-between", align="flex-start", mb="sm")
        if right is not None
        else html.Div(head, style={"marginBottom": "10px"})
    )
    extra = {"id": id} if id else {}
    return dmc.Card([header, *children], p="md", h="100%", className=className, **extra)


def loading(child: Any) -> Any:
    return dcc.Loading(
        child,
        delay_show=250,
        overlay_style={"visibility": "visible", "opacity": 0.5},
        custom_spinner=dmc.Loader(type="dots", size="md"),
    )


def note(text: str, colour: str = "dimmed") -> Any:
    return dmc.Text(text, size="xs", c=colour)


# --------------------------------------------------------------------------- header


def _fact(label: str, value: str, tip: str, *, derived: bool = False) -> Any:
    head: Any = dmc.Text(label, className="qq-fact-label")
    if derived:
        head = html.Div([head, html.Span("derived", className="qq-derived")], className="qq-fhead")
    return dmc.Tooltip(
        dmc.Paper(
            [head, dmc.Text(value, className="qq-fact-value")],
            withBorder=True,
            px="md",
            py=8,
            radius="md",
            className="qq-fact",
        ),
        label=tip,
    )


def _report_params(rs: ResultSet) -> dict[str, Any]:
    for run in rs.runs:
        art = run.artifact("peptide_quant")
        if art is not None and art.report is not None:
            return dict(art.report.params)
    return {}


def header(ctx: PageContext, quant_rows: Mapping[str, int], lfq_keys: Mapping[str, int]) -> Any:
    """Eyebrow, title, the quant settings as chips, and a few key numbers."""
    rs = ctx.rs
    gate = quant_gate(rs)
    params = _report_params(rs)
    kind = f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"
    chips: list[Any] = []
    differs = (
        rs.is_experiment
        and gate.configured is not None
        and gate.effective is not None
        and gate.configured != gate.effective
    )
    gate_text = (
        f"quant gate {gate.q_column} ≤ {gate.threshold:g}"
        if gate.q_column and gate.threshold is not None
        else "quant gate not recorded"
    )
    chips.append(
        chip(
            gate_text,
            "yellow" if differs else "gray",
            tip=f"Quant selected {gate.describe(rs.is_experiment)}. {gate.note} Source: "
            f"{gate.source}.",
            left=icon("target", 12),
        )
    )
    if params.get("top_n_fragments") is not None:
        chips.append(
            chip(
                f"top {params['top_n_fragments']} fragments",
                "violet",
                tip="quant.top_n_fragments: a precursor quantity is the sum of its top-N "
                f"positive fragment areas (fragment_selection {params.get('fragment_selection')}, "
                f"bound_peak {params.get('bound_peak')}); from the quant report.",
            )
        )
    if params.get("rollup") is not None:
        chips.append(
            chip(
                f"protein {params['rollup']} of {params.get('top_n_peptides')} peptides",
                "pink",
                tip="quant.rollup and top_n_peptides: a protein group quantity sums the top-N "
                "maximum quantities of its base peptides; from the quant report.",
            )
        )
    if rs.is_experiment:
        sf = size_factors(rs)
        text = "MaxLFQ, median-ratio normalized"
        tip = (
            "lfq_maxlfq.parquet and its .precursor and .peptide tables: MaxLFQ over the runs' "
            "peptide_quant features, after one median-ratio size factor per run "
            "(run_experiment.rs passes NormalizeMethod::MedianRatio; the engine logs the "
            "factors only)."
        )
        if sf is not None and sf.attrs.get("constant"):
            parts = ", ".join(f"{r} {f:.3f}" for r, f in zip(sf["run"], sf["factor"], strict=True))
            tip += f" Size factors derived by the viewer (quantity / precursor MaxLFQ): {parts}."
        chips.append(chip(text, "indigo", tip=tip))
    info = mbr_info(rs) if rs.is_experiment else None
    if info is not None and info.ran:
        chips.append(
            chip(
                f"MBR {fmt(info.n_transfers) if info.n_transfers is not None else ''} transfers",
                "lime",
                tip="Match-between-runs ran: quant admits every transfer whatever its q, and "
                "no quant table flags it; the viewer marks transfers from the transfer table.",
            )
        )
    title = html.Div(
        [
            dmc.Text(f"Quant QC · {kind}", className="mv-eyebrow"),
            html.Div("Quantification", className="mv-title"),
            dmc.Text(
                "Quantities, missing values and CVs of the engine's quant tables. Every "
                "summary on this page is computed by the viewer.",
                size="xs",
                c="dimmed",
                mt=2,
            ),
        ]
    )
    facts: list[Any] = []
    n = len(rs.runs)
    for level, kind_name in (("precursor", "peptide_quant"), ("protein", "protein_group_quant")):
        rows = quant_rows.get(level)
        if rows is None:
            continue
        label = "precursors quantified" if level == "precursor" else "protein groups quantified"
        if n > 1:
            value = fmt(round(rows / n))
            tip = (
                f"Rows of {kind_name} with a quantity, mean per run over {n} runs "
                f"({fmt(rows)} in total)."
            )
            label += " / run"
        else:
            value, tip = fmt(rows), f"Rows of {kind_name} with a quantity."
        facts.append(_fact(label, value, tip))
    if rs.is_experiment and lfq_keys.get("protein") is not None:
        facts.append(
            _fact(
                "LFQ protein groups",
                fmt(lfq_keys["protein"]),
                "Protein groups of lfq_maxlfq.parquet: every group with a quant-gated feature "
                "in any run; not FDR-controlled at the protein unit.",
            )
        )
    return dmc.Group(
        [
            dmc.Stack([title, dmc.Group(chips, gap=6)], gap="sm", className="qq-hero-left"),
            dmc.Group(facts, gap="sm", wrap="nowrap", className="qq-facts"),
        ],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="qq-hero",
    )


# --------------------------------------------------------------------------- toolbar


def _segment(id_: str, data: list[tuple[str, str]], value: str, colour: str = "indigo") -> Any:
    return dmc.SegmentedControl(
        id=id_,
        data=[{"value": v, "label": label} for v, label in data],
        value=value,
        size="xs",
        radius="md",
        color=colour,
    )


def _labelled(label: str, tip: str, control: Any, *, hidden: bool = False) -> Any:
    return html.Div(
        [
            dmc.Tooltip(
                html.Span([label, icon("info", 11)], className="qq-tool-label"),
                label=tip,
                position="bottom",
            ),
            control,
        ],
        className="qq-tool",
        style={"display": "none"} if hidden else None,
    )


def toolbar(ctx: PageContext) -> Any:
    rs = ctx.rs
    experiment = rs.is_experiment
    source = _segment(
        "qq-source",
        [("lfq", "MaxLFQ"), ("quant", "Per-run quant")],
        "lfq" if experiment else "quant",
    )
    keys = _segment("qq-keys", [("accepted", "Accepted"), ("all", "All quantified")], "accepted")
    need = _segment("qq-need", [("all", "Every run"), ("two", "≥ 2 values")], "all", "teal")
    pill = dmc.Tooltip(
        html.Div(
            [
                icon("target", 14),
                html.Span(f"precursor_q · pg_q_value ≤ {stop_label(ctx.threshold)}", id="qq-t"),
            ],
            className="qq-t",
            role="button",
            tabIndex="0",
        ),
        label="The header threshold: with Accepted keys, the precursors with precursor_q and "
        "the protein groups with pg_q_value at or below it; the accepted identifications of "
        "each run use run_psm_q. Click to change it.",
        position="bottom",
    )
    single = dmc.Tooltip(
        dmc.Badge(
            "per-run quant",
            color="gray",
            variant="light",
            size="lg",
            radius="md",
            style={"textTransform": "none"},
        ),
        label="A single run has no LFQ: the quantities are its peptide_quant and "
        "protein_group_quant tables.",
    )
    parts = [
        _labelled(
            "Quantities",
            SOURCE_TIPS["lfq"] + " " + SOURCE_TIPS["quant"],
            source,
            hidden=not experiment,
        ),
        html.Div(
            single, className="qq-tool", style=None if not experiment else {"display": "none"}
        ),
        _labelled("Keys", KEYS_TIP, keys),
        pill,
        _labelled("CV needs", NEED_TIP, need, hidden=not experiment),
    ]
    return dmc.Card(html.Div(parts, className="qq-tools"), className="qq-toolbar", p=6)


# --------------------------------------------------------------------------- conditions


SUGGESTION_TIP = (
    "Suggested from the mzML file names of the manifest: the token after Condition (or "
    "Cond), else the name without its replicate token (REP1, R2, run3, 01) and without the "
    "tokens every run shares. Change a condition to group the runs your way: your choice is "
    "kept in this browser (localStorage) and used by every CV on this page."
)


def short_names(files: Mapping[str, str | None]) -> dict[str, str | None]:
    """File names without the leading tokens that every file shares (with an ellipsis)."""
    names = {k: v for k, v in files.items() if v}
    if len(names) < 2:
        return dict(files)
    parts = {k: re.split(r"(?<=[_\-. ])", v) for k, v in names.items()}
    lists = list(parts.values())
    n = 0
    while all(len(x) > n + 1 for x in lists) and all(x[n] == lists[0][n] for x in lists):
        n += 1
    out: dict[str, str | None] = {}
    for k, v in files.items():
        out[k] = ("…" + "".join(parts[k][n:])) if v and n else v
    return out


def conditions_card(rs: ResultSet, suggestion: Mapping[str, str], files: Mapping[str, Any]) -> Any:
    rows = []
    shown = short_names(files)
    for run in rs.runs:
        name = files.get(run.name)
        rows.append(
            html.Div(
                [
                    html.Span(
                        [
                            html.Span(
                                className="qq-dot",
                                id={"type": "qq-dot", "run": run.name},
                            ),
                            html.Span(run_label(run.name), className="qq-run"),
                        ],
                        className="qq-run-cell",
                    ),
                    html.Div(
                        dmc.Tooltip(
                            html.Span(
                                shown.get(run.name) or "no mzML recorded", className="qq-file"
                            ),
                            label=name or "The manifest records no mzML input for this run.",
                            w="auto",
                            multiline=False,
                        ),
                        className="qq-file-cell",
                    ),
                    dmc.TextInput(
                        id={"type": "qq-cond", "run": run.name},
                        value=suggestion.get(run.name, ""),
                        size="xs",
                        radius="md",
                        debounce=350,
                        className="qq-cond-input",
                        **{"aria-label": f"Condition of {run_label(run.name)}"},
                    ),
                ],
                className="qq-cond-row",
            )
        )
    footer = dmc.Group(
        [
            dmc.Button(
                "Use the suggestion",
                id="qq-cond-reset",
                n_clicks=0,
                size="compact-xs",
                variant="subtle",
                color="gray",
                leftSection=icon("x", 11),
            ),
            dmc.Text("", id="qq-cond-state", size="xs", c="dimmed"),
        ],
        justify="space-between",
        mt=8,
    )
    head = html.Div(
        [html.Span("run"), html.Span("mzML file"), html.Span("condition")],
        className="qq-cond-row qq-cond-head",
    )
    how = html.Div(
        "Each run's condition is suggested from its mzML file name: the token after "
        "Condition, else the name without its replicate number. Type another name to "
        "regroup the runs; the CVs, the matrix and the profile follow, and this browser "
        "keeps your choice.",
        className="qq-cond-how",
    )
    return qq_section(
        "Conditions",
        html.Div([head, *rows], className="qq-cond-table"),
        footer,
        how,
        count=f"{len(rs.runs)} run{'s' if len(rs.runs) != 1 else ''}",
        help=SUGGESTION_TIP,
        subtitle="Runs grouped into conditions for the CVs; kept in this browser.",
        className="qq-cond-card",
    )


# --------------------------------------------------------------------------- tables


def _bar_cell(
    value: Any, total: Any, colour: str, *, tip: str | None = None, width: int = 46
) -> Any:
    if value is None or pd.isna(value):
        return html.Span("n/a", className="qq-na")
    v = int(value)
    hi = float(total) if total is not None and not pd.isna(total) and total else 1.0
    return spark_bar(v, lo=0.0, hi=max(hi, 1.0), colour=colour, text=fmt(v), width=width, tip=tip)


STATE_COLUMNS = (
    ("accepted", "accepted", "var(--mantine-color-indigo-5)"),
    ("quantified", "quantified", "var(--mantine-color-green-6)"),
    ("not_quantifiable", "not quantifiable", "var(--mantine-color-orange-6)"),
    ("not_selected", "not selected", "var(--mantine-color-gray-6)"),
)
EXTRA_COLUMNS = (
    ("quantified_not_accepted", "not accepted", "var(--mantine-color-violet-5)"),
    ("transfers", "MBR transfers", "var(--mantine-color-lime-6)"),
)
TABLE_NAMES = {
    "peptide_quant": "peptide_quant rows",
    "protein_group_quant": "protein_group_quant rows",
}


def _th(
    text: str,
    tip: str | None = None,
    *,
    numeric: bool = True,
    className: str | None = None,
    **kwargs: Any,
) -> Any:
    label: Any = text
    if tip:
        label = dmc.Tooltip(html.Span(text, className="qq-th"), label=tip)
    classes = " ".join(c for c in ("qq-th-cell", "mv-num" if numeric else None, className) if c)
    # html.Th: Mantine's TableTh takes no colSpan or rowSpan (styled in quant.css).
    return html.Th(label, className=classes, **kwargs)


def states_table(states: pd.DataFrame | None, status: pd.DataFrame | None, t: float) -> Any:
    """One row per run: its accepted identifications by quant state, and the engine's rows.

    ``states`` is :func:`.quantqc.accepted_quant_states`, ``status`` is
    :func:`.quantqc.status_matrix`; either may be None (not readable).
    """
    labels = dict(states.attrs.get("labels", {})) if states is not None else {}
    extra = [c for c in EXTRA_COLUMNS if states is not None and c[0] in states]
    groups: list[tuple[str, int, str]] = []
    if states is not None:
        q = states.attrs.get("q_column", "run_psm_q")
        groups.append(
            (
                f"Accepted identifications ({q} ≤ {stop_label(t)})",
                len(STATE_COLUMNS),
                labels.get("accepted", ""),
            )
        )
        groups.append(
            (
                "Quantified rows",
                len(extra),
                "Rows of the run's peptide_quant with a quantity, beyond the accepted ones",
            )
        )
    status_cols: list[tuple[str, str, str]] = []
    if status is not None:
        for table in ("peptide_quant", "protein_group_quant"):
            sub = status[status["table"] == table]
            for row in sub.to_dict("records"):
                status_cols.append((table, row["status"], row["description"]))
            if len(sub):
                groups.append(
                    (
                        TABLE_NAMES[table],
                        len(sub),
                        f"Rows of each run's {table} by quant_status (the engine's counts)",
                    )
                )
    top = [_th("", numeric=False, rowSpan=2)]
    for text, span, tip in groups:
        if span:
            top.append(_th(text, tip, numeric=False, colSpan=span, className="qq-group-th"))
    second = []
    if states is not None:
        second += [_th(label, labels.get(key, label)) for key, label, _ in STATE_COLUMNS]
        second += [_th(label, labels.get(key, label)) for key, label, _ in extra]
    second += [
        _th(st, f"{table}: {desc}", className="mv-num qq-status-th")
        for table, st, desc in status_cols
    ]
    head = dmc.TableThead([dmc.TableTr(top), dmc.TableTr(second)])
    names = (
        list(states["run"])
        if states is not None
        else list(status.columns[3:])
        if (status is not None)
        else []
    )
    by_run = states.set_index("run") if states is not None else None
    body = []
    for run in names:
        if status is not None and run not in status.columns:
            continue
        cells = [dmc.TableTd(html.Span(run_label(run), className="qq-run"))]
        if by_run is not None:
            row = by_run.loc[run]
            total = row.get("accepted")
            for key, label, colour in STATE_COLUMNS:
                cells.append(
                    dmc.TableTd(
                        _bar_cell(
                            row.get(key), total, colour, tip=labels.get(key, label), width=28
                        ),
                        className="mv-num",
                    )
                )
            for key, label, colour in extra:
                cells.append(
                    dmc.TableTd(
                        _bar_cell(
                            row.get(key),
                            row.get("quant_rows"),
                            colour,
                            tip=labels.get(key, label),
                            width=28,
                        ),
                        className="mv-num",
                    )
                )
        if status is not None:
            for table, st, _ in status_cols:
                hit = status[(status["table"] == table) & (status["status"] == st)]
                n = int(hit[run].iloc[0]) if len(hit) else 0
                total = int(status.loc[status["table"] == table, run].sum())
                good = st == "quantified"
                tip = f"{table} of {run_label(run)}: {n:,} rows with quant_status {st}"
                tr_col = f"n_transferred_{run}"
                if len(hit) and tr_col in hit and not pd.isna(hit[tr_col].iloc[0]):
                    tip += (
                        f"; {int(hit[tr_col].iloc[0]):,} of them are match-between-runs transfers"
                    )
                cells.append(
                    dmc.TableTd(
                        _bar_cell(
                            n,
                            total,
                            "var(--mantine-color-green-6)"
                            if good
                            else "var(--mantine-color-orange-6)",
                            tip=tip,
                            width=28,
                        ),
                        className="mv-num",
                    )
                )
        body.append(dmc.TableTr(cells))
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        highlightOnHover=True,
        verticalSpacing=5,
        horizontalSpacing="xs",
        className="mv-table qq-table qq-states",
    )
    legend = [
        html.Span(
            [
                html.Span(
                    st,
                    className="qq-status "
                    + ("qq-status-ok" if st == "quantified" else "qq-status-no"),
                ),
                html.Span(
                    f" {table}: " + (desc.split(": ", 1)[1] if desc.startswith(st + ":") else desc),
                    className="qq-legend-text",
                ),
            ],
            className="qq-legend-item",
        )
        for table, st, desc in status_cols
    ]
    return html.Div(
        [
            dmc.TableScrollContainer(table, minWidth=680, type="native"),
            html.Div(legend, className="qq-legend") if legend else None,
        ]
    )


def cv_table(results: Mapping[str, CvResult | None], colours: Mapping[str, str]) -> Any:
    """Per condition: its runs and, per level, the CVs, their median and the share at most 20%."""
    levels = [lvl for lvl in ("precursor", "protein") if results.get(lvl) is not None]
    if not levels:
        return None
    first = results[levels[0]]
    assert first is not None
    top = [_th("condition", numeric=False, rowSpan=2), _th("runs", numeric=False, rowSpan=2)]
    second = []
    for lvl in levels:
        top.append(
            _th(
                LEVEL_NOUNS[lvl].capitalize(),
                f"CVs of the {LEVEL_NOUNS[lvl]} (the viewer's)",
                numeric=False,
                colSpan=3,
                className="qq-group-th",
            )
        )
        second += [
            _th(
                "with a CV",
                f"{LEVEL_NOUNS[lvl]} with the values a CV needs, of those with a "
                "value in the condition",
            ),
            _th("median", "median CV"),
            _th("≤ 20%", "share of the CVs at most 20%"),
        ]
    head = dmc.TableThead([dmc.TableTr(top), dmc.TableTr(second)])
    body = []
    for i, c in enumerate(first.conditions):
        colour = colours.get(c.condition, "gray")
        runs = ", ".join(run_label(r) for r in c.runs)
        cells = [
            dmc.TableTd(
                html.Span(
                    [html.Span(className="qq-dot", style={"background": colour}), c.condition],
                    className="qq-run-cell qq-strong",
                )
            ),
            dmc.TableTd(
                html.Div(
                    [
                        html.Span(runs, className="qq-runs-list"),
                        html.Span(
                            c.reason if c.reason else f"each CV needs {c.need} values",
                            className="qq-need" + (" qq-na" if c.reason else ""),
                        ),
                    ]
                )
            ),
        ]
        for lvl in levels:
            res = results[lvl]
            assert res is not None
            cc = res.conditions[i]
            med = cc.median
            share = cc.share_below(20)
            cells += [
                dmc.TableTd(
                    _bar_cell(
                        cc.n_cv,
                        cc.n_with_value or 1,
                        colour,
                        tip=f"{cc.n_cv:,} of the {cc.n_with_value:,} {LEVEL_NOUNS[lvl]} with a "
                        f"value in condition {c.condition} have the {cc.need} values a CV needs",
                    ),
                    className="mv-num",
                ),
                dmc.TableTd(
                    f"{med:.1f}%" if med is not None else html.Span("none", className="qq-na"),
                    className="mv-num qq-strong",
                ),
                dmc.TableTd(f"{share:.0f}%" if share is not None else "", className="mv-num"),
            ]
        body.append(dmc.TableTr(cells))
    table = dmc.Table(
        [head, dmc.TableTbody(body)],
        highlightOnHover=True,
        verticalSpacing=5,
        horizontalSpacing="xs",
        className="mv-table qq-table qq-cvt",
    )
    return dmc.TableScrollContainer(table, minWidth=560, type="native")


def missing_single(missing: Mapping[str, pd.DataFrame], accepted: bool, t: float) -> Any:
    """The missing values of a single run, as two lines with bars."""
    lines = []
    for level in ("precursor", "protein"):
        df = missing.get(level)
        if df is None or not len(df):
            continue
        row = df.iloc[0]
        keys, miss = int(row["keys"]), int(row["missing"])
        column = "precursor_q" if level == "precursor" else "pg_q_value"
        pct = float(row["missing_pct"]) if keys else math.nan
        what = (
            f"accepted {LEVEL_NOUNS[level]} ({column} ≤ {stop_label(t)})"
            if accepted
            else f"quantified {LEVEL_NOUNS[level]}"
        )
        lines.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(fmt(miss), className="qq-big"),
                            html.Span(f" of {fmt(keys)} {what} have no quantity"),
                        ]
                    ),
                    spark_bar(
                        miss,
                        lo=0,
                        hi=max(keys, 1),
                        colour="var(--mantine-color-orange-6)",
                        text=f"{pct:.1f}%" if keys else "no keys",
                        width=160,
                    ),
                ],
                className="qq-miss-line",
            )
        )
    return html.Div(lines, className="qq-miss-single")


def _q_mark(p: ProteinProfile, t: float) -> Any:
    if p.pg_q_value is None:
        return dmc.Badge("no target row", color="gray", variant="light", size="sm")
    return dmc.Group(
        [
            validation_icon(p.pg_q_value, t, column="pg_q_value"),
            dmc.Text(f"pg_q_value {fmt(p.pg_q_value)}", size="xs", c="dimmed", className="qq-mono"),
        ],
        gap=6,
    )


SPECIES_COLOURS = {"HUMAN": "indigo", "YEAST": "yellow", "ECOLI": "teal", "MOUSE": "pink"}


def profile_head(p: ProteinProfile, t: float, base: str, cvs: Mapping[str, float | None]) -> Any:
    members = [m for m in p.group.split(";") if m]
    name = members[0] if members else p.group
    more = f" +{len(members) - 1}" if len(members) > 1 else ""
    chips: list[Any] = [_q_mark(p, t)]
    for sp in p.species[:3]:
        chips.append(
            chip(
                sp,
                SPECIES_COLOURS.get(sp, "gray"),
                size="sm",
                tip="Species from the entry-name suffix of the group's members (the viewer's "
                "rule).",
            )
        )
    if p.n_features is not None:
        chips.append(
            chip(
                f"{p.n_features} features",
                "gray",
                size="sm",
                tip="n_features of lfq_maxlfq.parquet: distinct precursors with a positive "
                "quantity in any run of the group.",
            )
        )
    for cond, cv in cvs.items():
        chips.append(
            chip(
                f"CV {cond} {cv:.1f}%" if cv is not None else f"CV {cond} n/a",
                "teal" if cv is not None else "gray",
                size="sm",
                tip="The viewer's CV of the group's MaxLFQ over the runs of the condition "
                "with a value (sample SD / mean); n/a when a run of the condition has no value"
                " or the condition has one run.",
            )
        )
    return html.Div(
        [
            dmc.Group(
                [
                    dmc.Tooltip(
                        html.Span([name, html.Span(more, className="qq-more")], className="qq-pg"),
                        label=p.group,
                    ),
                ],
                gap=6,
            ),
            dmc.Group(chips, gap=6, mt=4),
        ]
    )


def profile_table(p: ProteinProfile, conditions: Mapping[str, str], order: Sequence[str]) -> Any:
    quant = p.quant.set_index("run")
    col = {r: j for j, r in enumerate(p.runs)}
    mbr = p.lfq_transferred is not None
    head = ["run", "condition"]
    if p.lfq is not None:
        head.append("MaxLFQ")
    head += ["protein quant", "status", "peptides"]
    if mbr:
        head.append("MBR")
    numeric = {"MaxLFQ", "protein quant", "peptides", "MBR"}
    rows = []
    for r in order:
        j = col[r]
        q = quant.loc[r] if r in quant.index else None
        status = q["quant_status"] if q is not None else None
        cells = [
            dmc.TableTd(html.Span(run_label(r), className="qq-run")),
            dmc.TableTd(conditions.get(r, "")),
        ]
        if p.lfq is not None:
            v = float(p.lfq[j])
            cells.append(
                dmc.TableTd(
                    compact(v) if math.isfinite(v) else html.Span("missing", className="qq-na"),
                    className="mv-num",
                )
            )
        qv = float(q["quantity"]) if q is not None and not pd.isna(q["quantity"]) else math.nan
        cells.append(
            dmc.TableTd(
                compact(qv)
                if math.isfinite(qv)
                else html.Span(
                    "no row" if status is None else "not quantifiable", className="qq-na"
                ),
                className="mv-num",
            )
        )
        cells.append(
            dmc.TableTd(
                dmc.Tooltip(
                    html.Span(str(status), className="qq-mono"), label=str(q["description"])
                )
                if status is not None and q is not None
                else html.Span("no protein_group_quant row", className="qq-na")
            )
        )
        n_pep = q["n_peptides"] if q is not None else None
        cells.append(
            dmc.TableTd(
                "" if n_pep is None or pd.isna(n_pep) else str(int(n_pep)), className="mv-num"
            )
        )
        if mbr:
            assert p.lfq_transferred is not None
            n = int(p.lfq_transferred[j])
            cells.append(dmc.TableTd(str(n) if n else "", className="mv-num"))
        rows.append(dmc.TableTr(cells))
    return dmc.Table(
        [
            dmc.TableThead(
                dmc.TableTr(
                    [dmc.TableTh(h, className="mv-num" if h in numeric else None) for h in head]
                )
            ),
            dmc.TableTbody(rows),
        ],
        verticalSpacing=4,
        horizontalSpacing="xs",
        className="mv-table qq-table qq-ptable",
    )


def error_text(exc: Exception) -> Any:
    return dmc.Alert(
        str(exc), color="yellow", variant="light", icon=icon("alert", 16), p="xs", title=None
    )


def open_button(base: str, group: str | None) -> Any:
    return dcc.Link(
        dmc.Button(
            "Open protein page",
            rightSection=icon("right", 14),
            size="xs",
            radius="md",
            variant="light",
        ),
        href=href(base, "protein", {"group": group}) if group else href(base, "protein"),
        id="qq-open",
        className="qq-open",
    )
