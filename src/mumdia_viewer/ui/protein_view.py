"""Components of the protein page: the header, the verdict strip, the members, the panels.

The peptides and the precursors of the group are dense grids in the identification
page's vocabulary (the column definitions and cell renderers of
:mod:`.browser_grid`), so a peptide reads the same on both pages. Everything here only
lays out what :mod:`mumdia_viewer.data.protein` returns; numbers the viewer derives say
so in their tooltips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash import dcc, html

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data.fasta import Coverage, ProteinEntry
from mumdia_viewer.data.protein import SPECIES_RULE, ProteinGroup, species_of

from . import browser_panels as bp
from .browser_grid import Scales, column_defs
from .icons import icon
from .protein_figures import compact
from .state import href, stop_label
from .widgets import fmt, fmt_q, peptidoform, section, spark_bar, validation_icon

# Species colours: the identification grid's (browser.js SPECIES).
SPECIES_COLOURS = {
    "HUMAN": "indigo",
    "YEAST": "yellow",
    "ECOLI": "teal",
    "MOUSE": "pink",
    "RAT": "grape",
    "BOVIN": "orange",
    "ARATH": "lime",
    "DROME": "cyan",
    "CAEEL": "violet",
}
# Members shown in the title; the others are in the members card.
TITLE_MEMBERS = 3
PEP_GRID = "pp-pep-grid"
PRE_GRID = "pp-pre-grid"
FIND_GRID = "pp-find-grid"
SCORE_BAR = "var(--ib-bar-score)"


def protein_href(base: str, group: str | None, **query: Any) -> str:
    """The address of a protein group's page: ``<base>protein?group=<group>``.

    ``query`` adds ``peptide`` (a base_peptide_id to select) or ``member`` (the member
    whose coverage is shown). Without a group it is the page's search.
    """
    return href(base, "protein", {"group": group or None, **query})


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def help_icon(text: Any, *, w: int = 380, id_: str | None = None) -> Any:
    kwargs = {"id": id_} if id_ else {}
    return dmc.Tooltip(
        html.Span(icon("info", 13), className="mv-help"),
        label=text,
        w=w,
        multiline=True,
        position="bottom-start",
        **kwargs,
    )


def help_body(line: str, *more: str | None, small: str | None = None) -> Any:
    """A tooltip's content: one plain line, then details in smaller text."""
    return bp.help_body(line, *more, small=small)


def where(rs: ResultSet) -> str:
    return f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"


def _member(name: str, *, first: bool) -> Any:
    kids: list[Any] = []
    text = name
    if text.startswith("DECOY_"):
        kids.append(html.Span("DECOY_", className="mv-pep-decoy"))
        text = text[len("DECOY_") :]
    kids.append(text)
    return html.Span(kids, className="pp-member-first" if first else "pp-member")


def species_chips(group: str, size: str = "md") -> list[Any]:
    out = []
    for code in species_of(group):
        out.append(
            dmc.Tooltip(
                dmc.Badge(
                    code,
                    color=SPECIES_COLOURS.get(code, "gray"),
                    variant="light",
                    size=size,
                    radius="sm",
                    rightSection=html.Span("rule", className="pp-rule"),
                    style={"textTransform": "none"},
                ),
                label=f"Species {code}: {SPECIES_RULE}.",
                w=320,
                multiline=True,
            )
        )
    return out


# --------------------------------------------------------------------------- header


def _fact(label: str, value: str, text: str, *, derived: bool = False) -> Any:
    head: Any = html.Div(label, className="pp-fact-label")
    if derived:
        head = html.Div(
            [head, html.Span("derived", className="pp-derived")], className="pp-fact-head"
        )
    return tip(
        html.Div([head, html.Div(value, className="pp-fact-value")], className="pp-fact"),
        text,
        w=300,
        multiline=True,
    )


def hero(
    ctx: Any,
    pg: ProteinGroup,
    *,
    entry: ProteinEntry | None,
    member: str | None,
    n_rows: int | None,
    n_precursors: int | None,
    fasta_label: str | None,
) -> Any:
    """The page header: the members, the label, the species, the links, the key facts."""
    rs = ctx.rs
    group = pg.protein_group
    members = list(pg.members) or [group]
    title: list[Any] = [_member(members[0], first=True)]
    for m in members[1:TITLE_MEMBERS]:
        title.append(html.Span(";", className="pp-sep"))
        title.append(_member(m, first=False))
    if len(members) > TITLE_MEMBERS:
        title.append(
            html.Span(
                f"+{len(members) - TITLE_MEMBERS} more",
                className="pp-more",
                title="; ".join(members[TITLE_MEMBERS:]),
            )
        )
    chips: list[Any] = [
        dmc.Tooltip(
            dmc.Badge(
                pg.label,
                color="orange" if pg.decoy else "indigo",
                variant="filled",
                size="lg",
                radius="sm",
                style={"textTransform": "none"},
            ),
            label="A decoy protein group (its string starts with DECOY_; its rows are the "
            "decoy rows)"
            if pg.decoy
            else "A target protein group (label of its rows)",
        ),
        *species_chips(group, "lg"),
    ]
    if len(members) > 1:
        chips.append(
            dmc.Badge(
                f"{len(members)} members",
                color="gray",
                variant="light",
                size="lg",
                radius="sm",
                style={"textTransform": "none"},
            )
        )
    links: list[Any] = []
    cid = pg.get("candidate_id")
    if cid is not None:
        run = str(pg.get("run") or "") if rs.is_experiment else ""
        links.append(
            dcc.Link(
                dmc.Badge(
                    [
                        html.Span("winning precursor ", className="pp-link-lead"),
                        peptidoform(pg.get("peptidoform"), size="1em"),
                        html.Span(f" {pg.get('charge')}+"),
                        html.Span(f" · run {run}" if run else ""),
                    ],
                    color="pink",
                    variant="light",
                    size="md",
                    radius="sm",
                    leftSection=icon("peak", 12),
                    style={"textTransform": "none", "cursor": "pointer"},
                    className="pp-chip-link",
                ),
                href=href(ctx.base, "precursor", {"run": run, "cid": int(cid)}),
                title="The group's winning row (pg_q_value is set on it): open its precursor page",
            )
        )
    first = members[0].removeprefix("DECOY_")
    links.append(
        dcc.Link(
            dmc.Badge(
                "identifications",
                color="gray",
                variant="light",
                size="md",
                radius="sm",
                leftSection=icon("table", 12),
                style={"textTransform": "none", "cursor": "pointer"},
                className="pp-chip-link",
            ),
            href=href(
                ctx.base,
                "identifications",
                {"search": first, "group": group, "decoys": "1" if pg.decoy else None},
            ),
            title="Open this group in the identification page (its linked panels)",
        )
    )
    line: list[Any] = []
    if entry is not None:
        acc = entry.accession or entry.key
        line.append(html.Span(acc, className="pp-acc"))
        line.append(
            html.Span(
                entry.description or "no description in the FASTA header",
                className="pp-desc" if entry.description else "pp-desc pp-dim",
            )
        )
        if len(members) > 1 and member:
            line.append(html.Span(f"(member {member})", className="pp-dim"))
    elif fasta_label is None:
        line.append(
            html.Span(
                [
                    "No FASTA: start the viewer with ",
                    html.Code("--fasta <file>"),
                    " for descriptions, lengths and coverage.",
                ],
                className="pp-dim",
            )
        )
    elif member:
        line.append(html.Span(f"{member} is not in {fasta_label}.", className="pp-dim"))
    eyebrow = html.Div(
        [
            dcc.Link(
                [icon("left", 13), html.Span("Protein groups")],
                href=protein_href(ctx.base, None),
                className="pp-back",
                title="Search the protein groups",
            ),
            html.Span(f"Protein group · {where(rs)}", className="mv-eyebrow"),
        ],
        className="pp-eyebrow",
    )
    left = html.Div(
        [
            eyebrow,
            html.Div(
                [
                    html.Div(title, className="pp-title", title=group),
                    dcc.Clipboard(
                        content=group, title="Copy the protein group", className="pp-copy"
                    ),
                ],
                className="pp-title-row",
            ),
            html.Div([*chips, *links], className="pp-chips"),
            html.Div(line, className="pp-line") if line else None,
        ],
        className="pp-hero-left",
    )
    facts = []
    if entry is not None:
        facts.append(
            _fact(
                "length",
                f"{entry.length:,} aa",
                f"Residues of {member or entry.key} in {fasta_label} (the FASTA, not the engine)",
            )
        )
    facts.append(
        _fact(
            "peptides",
            fmt(pg.get("n_peptides")) if pg.found else "-",
            pg.labels.get("n_peptides", "peptides of this group (distinct base_peptide_id, any q)"),
        )
    )
    if n_precursors is not None:
        facts.append(
            _fact(
                "precursors",
                f"{n_precursors:,}",
                "Viewer-derived: the sum of n_precursors over the group's peptides, which is "
                "the number of distinct (peptidoform, charge) of the group (same label, any q, "
                "all runs)",
                derived=True,
            )
        )
    if n_rows is not None:
        facts.append(
            _fact(
                "scored rows",
                f"{n_rows:,}",
                "Rows of the scored table with this protein group and its label (any q"
                + (", all runs)" if rs.is_experiment else ")"),
            )
        )
    return html.Div(
        [left, html.Div(facts, className="pp-facts")], className="pp-hero", id="pp-hero"
    )


# --------------------------------------------------------------------------- verdict strip


def _vtile(head: str, value: Any, text: Any, colour: str, *, badge: Any = None, side: Any = None):
    top = html.Div([html.Span(head, className="pp-vt-col"), side], className="pp-vt-top")
    line = html.Div([badge, html.Div(value, className="pp-vt-val")], className="pp-vt-line")
    return dmc.Tooltip(
        html.Div([top, line], className=f"pp-vt pp-vt-{colour}"),
        label=text,
        w=360,
        multiline=True,
        position="bottom",
        openDelay=150,
        boxWrapperProps={"w": "100%"},
    )


def strip(
    ctx: Any,
    pg: ProteinGroup,
    *,
    peptides: pd.DataFrame,
    quant: pd.DataFrame | None,
    matrix: pd.DataFrame | None,
    coverage: Coverage | None,
    scales: Scales,
    rescorer: str | None,
) -> list[Any]:
    """The verdict strip at ``ctx.threshold`` (rebuilt when the threshold changes)."""
    rs, t = ctx.rs, ctx.threshold
    q = pg.get("pg_q_value")
    passes = q is not None and q <= t and not pg.decoy
    scope = ", experiment-wide" if rs.is_experiment else ""
    if pg.decoy:
        summary = "decoy group"
    elif q is None:
        summary = "no scored row"
    else:
        summary = "passes" if passes else "does not pass"
    head = dmc.Tooltip(
        html.Div(
            [
                html.Div("Verdict", className="mv-section-title"),
                html.Div(f"q ≤ {stop_label(t)}", className="pp-vh-t"),
                html.Div(summary, className="pp-vh-n" + (" pp-vh-pass" if passes else "")),
            ],
            className="pp-vhead",
        ),
        label=(
            f"At q ≤ {stop_label(t)} (the header threshold): the group {summary} on pg_q_value"
            f"{scope}. Hover a tile for its unit and provenance."
        ),
        w=320,
        multiline=True,
        position="bottom-start",
    )
    tiles: list[Any] = [head]
    # pg_q_value
    label = "decoy" if pg.decoy else "target"
    mark = validation_icon(q, t, label=label, column="pg_q_value", size=18)
    shown = fmt_q(q, t) if q is not None else "-"
    tiles.append(
        _vtile(
            "pg_q_value",
            shown,
            help_body(
                f"pg_q_value {shown} of the group's winning row"
                + (
                    f": {'passes' if passes else 'does not pass'} ≤ {stop_label(t)}."
                    if q is not None
                    else "."
                ),
                "pg_q_value is the q per protein group (the best PSM of each group, picked "
                "target-decoy competition), set on the group's winning row only; the other "
                f"rows hold 1.0{scope}.",
            ),
            "pink",
            badge=mark.children,
        )
    )
    # peptides
    n_all = len(peptides)
    n_pass = int((peptides["peptide_q_value"].astype("float64") <= t).sum()) if n_all else 0
    tiles.append(
        _vtile(
            "peptides",
            html.Span(
                [html.Span(f"{n_pass:,}"), html.Span(f" of {n_all:,}", className="pp-vt-of")]
            ),
            help_body(
                f"{n_pass:,} of the group's {n_all:,} peptides pass peptide_q_value ≤ "
                f"{stop_label(t)}.",
                "Peptides: unique base_peptide_id of the group's rows (any q). "
                "peptide_q_value is the picked target-decoy q per base_peptide_id, set on the "
                f"winning row of each peptide{scope}; a peptide whose decoy won holds 1.0.",
            ),
            "teal",
            side=html.Span("pass", className="pp-vt-side"),
        )
    )
    # best score
    score = pg.get("score")
    if score is not None and scales.score is not None:
        lo, hi = scales.score
        bar = spark_bar(score, lo=lo, hi=hi, width=40, text="", colour=SCORE_BAR)
        value: Any = html.Div([html.Span(f"{float(score):.4f}"), bar], className="pp-vt-score")
        bar_text = (
            f" Bar: linear from the lowest ({lo:.3g}) to the highest ({hi:.3g}) score of "
            "the scored table."
        )
    else:
        value, bar_text = (f"{float(score):.4f}" if score is not None else "-"), ""
    tiles.append(
        _vtile(
            "best score",
            value,
            help_body(
                "score of the group's winning row (its highest-scoring row), from "
                f"psms_scored.score; higher is better.{bar_text}",
                f"Rescorer: {rescorer}." if rescorer else None,
            ),
            "grape",
        )
    )
    # quantity
    tiles.append(_quant_tile(rs, pg, quant))
    # coverage
    if coverage is not None and coverage.entry is not None:
        frac = coverage.fraction_passing
        tiles.append(
            _vtile(
                "coverage",
                html.Span(
                    [
                        html.Span(f"{100 * frac:.1f}%" if frac is not None else "-"),
                        html.Span(
                            f" {100 * (coverage.fraction_any or 0):.1f}% all", className="pp-vt-of"
                        ),
                    ]
                ),
                help_body(
                    "Viewer-derived. "
                    + (
                        f"{100 * frac:.1f}% of the {coverage.length:,} residues of "
                        f"{coverage.member} are covered by the {coverage.n_passing:,} passing "
                        f"peptides, {100 * (coverage.fraction_any or 0):.1f}% by all "
                        f"{coverage.n_peptides:,} peptides of the group."
                        if frac is not None
                        else ""
                    ),
                    small=coverage.note,
                ),
                "green",
                side=html.Span("derived", className="pp-derived"),
            )
        )
    # runs (experiments)
    if rs.is_experiment and matrix is not None and len(matrix):
        runs_id = matrix.loc[matrix["identified"], "run"].nunique()
        tiles.append(
            _vtile(
                "runs with a PSM",
                html.Span(
                    [
                        html.Span(f"{runs_id:,}"),
                        html.Span(f" of {len(rs.runs):,}", className="pp-vt-of"),
                    ]
                ),
                help_body(
                    f"Runs in which a row of this group has run_psm_q ≤ {stop_label(t)} "
                    f"({runs_id} of {len(rs.runs)}).",
                    "run_psm_q is the PSM-level q within each run (per-run identification; "
                    "the grouped q columns are experiment-wide). It is not a protein-level "
                    "result: the group passes or fails on pg_q_value only"
                    + (
                        "; native values (scored_combined.parquet)."
                        if matrix.attrs.get("mbr")
                        else "."
                    ),
                ),
                "indigo",
                side=html.Span("identified", className="pp-vt-side"),
            )
        )
    return [html.Div(tiles, className="pp-vstrip")]


def _quant_tile(rs: ResultSet, pg: ProteinGroup, quant: pd.DataFrame | None) -> Any:
    if quant is None or quant.empty:
        return _vtile(
            "quantity",
            html.Span("n/a", className="pp-vt-none"),
            "The quant tables cannot be read.",
            "gray",
        )
    labels = quant.attrs.get("labels", {})
    rollup = quant.attrs.get("rollup")
    top_n = quant.attrs.get("top_n")
    rule = (
        f"the top-{top_n} sum of the per-peptide maxima"
        if rollup == "TopNSum" and top_n
        else f"the {rollup or 'recorded'} rollup of the per-peptide maxima"
    )
    if not rs.is_experiment:
        row = quant.iloc[0]
        qty = row.get("quantity")
        state = str(row.get("state") or "not_selected")
        if qty is not None and math.isfinite(float(qty)):
            badge = dmc.ThemeIcon(
                icon("check", 10), size=16, radius="xl", color="green", variant="light"
            )
            n_pep = row.get("n_peptides")
            text = help_body(
                f"protein_group_quant.quantity {float(qty):,.0f}: {rule} of the "
                f"{int(n_pep) if not pd.isna(n_pep) else '?'} quantified peptides (quant.rs).",
                "The peptides of the sum are marked in the peptides table (rollup column). "
                "Unit: intensity \u00d7 s.",
                small=labels.get("quantity"),
            )
            return _vtile("quantity", compact(qty), text, "green", badge=badge)
        words = state.replace("_", " ")
        return _vtile(
            "quantity",
            html.Span(words, className="pp-vt-none"),
            help_body(
                f"No protein quantity: {words}.",
                "No peptide of the group passed the quant gate."
                if state == "not_selected"
                else "The group's peptides have no positive quantity.",
                small=labels.get("state"),
            ),
            "gray",
        )
    n = len(quant)
    n_q = int(quant["quantity"].notna().sum())
    n_l = int(quant["lfq"].notna().sum()) if "lfq" in quant else 0
    return _vtile(
        "quantified",
        html.Span([html.Span(f"{n_q:,}"), html.Span(f" of {n:,} runs", className="pp-vt-of")]),
        help_body(
            f"protein_group_quant has a quantity in {n_q} of {n} runs ({rule}); MaxLFQ has a "
            f"value in {n_l} of {n} runs.",
            "The quantity per run card shows both; the peptides \u00d7 runs card the per-peptide "
            "values.",
        ),
        "green" if n_q else "gray",
    )


# --------------------------------------------------------------------------- members


def members_card(
    ctx: Any,
    pg: ProteinGroup,
    entries: Mapping[str, ProteinEntry | None],
    covs: Mapping[str, Coverage | None],
    shown: str | None,
    fasta_label: str | None,
) -> Any:
    """The members of a group of several: names, FASTA facts and coverage per member."""
    rows = []
    for m in pg.members:
        e = entries.get(m)
        cov = covs.get(m)
        on = m == shown
        name = html.Button(
            _member(m, first=True),
            className="pp-mbtn" + (" is-on" if on else ""),
            title=f"Show the coverage of {m}" if e is not None else m,
            disabled=e is None or pg.decoy,
            **{"data-member": m, "data-group": pg.protein_group},
        )
        found = "-"
        cover = "-"
        if cov is not None and cov.entry is not None:
            n_found = len({s.base_peptide_id for s in cov.spans})
            found = f"{n_found:,} of {cov.n_peptides:,}"
            fp, fa = cov.fraction_passing or 0.0, cov.fraction_any or 0.0
            cover = html.Span(
                [
                    spark_bar(fp, lo=0, hi=1, width=34, text="", colour="var(--mvc-pass)"),
                    html.Span(f"{100 * fp:.1f}%", className="pp-num"),
                    html.Span(f" {100 * fa:.1f}% all", className="pp-dim pp-small"),
                ],
                className="pp-cover",
            )
        rows.append(
            html.Tr(
                [
                    html.Td(name),
                    html.Td(html.Span(species_chips(m, "sm"), className="pp-chips-inline")),
                    html.Td(e.accession or e.key if e is not None else "-", className="pp-mono"),
                    html.Td(f"{e.length:,}" if e is not None else "-", className="pp-num"),
                    html.Td(
                        (e.description or html.Span("no description", className="pp-dim"))
                        if e is not None
                        else html.Span(
                            "not in the FASTA" if fasta_label else "no FASTA", className="pp-dim"
                        ),
                        className="pp-desc-cell",
                    ),
                    html.Td(found, className="pp-num"),
                    html.Td(cover),
                ],
                className="is-on" if on else None,
            )
        )
    head = html.Tr(
        [
            html.Th("member"),
            html.Th("species"),
            html.Th("accession"),
            html.Th("length", className="pp-num"),
            html.Th("description"),
            html.Th(
                tip(
                    html.Span("peptides in sequence", className="pp-th-tip"),
                    "Viewer-derived: the group's peptides found by exact search in this "
                    "member's sequence (any q)",
                ),
                className="pp-num",
            ),
            html.Th(
                tip(
                    html.Span("coverage", className="pp-th-tip"),
                    "Viewer-derived: residues covered by passing peptides (bar), and by all "
                    "of the group's peptides",
                )
            ),
        ]
    )
    table = html.Table([html.Thead(head), html.Tbody(rows)], className="pp-members")
    note = (
        "Descriptions, lengths and coverage come from "
        + (fasta_label or "a FASTA (start the viewer with --fasta)")
        + ". Click a member to show its coverage."
    )
    return section(
        "Members",
        html.Div(table, className="pp-members-box"),
        count=len(pg.members),
        help="The proteins of the group string, in the engine's order (protein_group is the "
        "protein string; members are separated by ';'). " + note,
        id="pp-members",
    )


# --------------------------------------------------------------------------- panels


def panel(
    key: str,
    title: str,
    body: Any,
    *,
    count: Any,
    count_tip: str | None,
    help_text: Any,
    subject: Any = None,
    right: Any = None,
    loading: bool = False,
    above: Any = None,
) -> Any:
    """A linked panel: the title bar (title, subject, count badge, help) and a grid."""
    row: list[Any] = [html.Span(title, className="pp-ptitle-text")]
    if subject is not None:
        row.append(html.Span(subject, id=f"pp-{key}-subject", className="pp-subject"))
    row.append(
        html.Span(
            dmc.Badge(
                count,
                id=f"pp-{key}-count",
                size="sm",
                color="gray",
                variant="light",
                className="pp-count",
                style={"textTransform": "none"},
            ),
            id=f"pp-{key}-countbox",
            className="pp-countbox",
            title=count_tip or "",
        )
    )
    row.append(help_icon(help_text, w=440, id_=f"pp-{key}-help"))
    header = html.Div(
        [html.Div(row, className="pp-ptitle"), html.Div(right, className="pp-pright")],
        className="pp-phead",
    )
    return html.Div(
        dmc.Card(
            [
                header,
                *([above] if above is not None else []),
                html.Div(body, className="pp-gridbox"),
            ],
            p=0,
            className="pp-pcard",
        ),
        id=f"pp-panel-{key}",
        className=f"pp-panel pp-panel-{key}" + (" pp-loading" if loading else ""),
    )


def _events(panel_key: str) -> dict[str, list[str]]:
    return {
        "rowClicked": [f"ppRowClicked(params, '{panel_key}')"],
        "rowDoubleClicked": [f"ppRowDoubleClicked(params, '{panel_key}')"],
        "cellKeyDown": [f"ppKeyDown(params, '{panel_key}')"],
        "cellFocused": [f"ppFocused(params, '{panel_key}')"],
        "firstDataRendered": [f"ppFirstData(params, '{panel_key}')"],
        "rowDataUpdated": [f"ppRowDataUpdated(params, '{panel_key}')"],
    }


def grid(
    grid_id: str,
    panel_key: str,
    defs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    base: str,
    threshold: float,
    *,
    selected: str | None,
    empty_text: str,
) -> Any:
    """A dense grid with the identification page's look; rows sorted in the browser."""
    return dag.AgGrid(
        id=grid_id,
        rowData=rows,
        columnDefs=defs,
        defaultColDef=bp.DEFAULT_COL,
        getRowId="params.data._key",
        selectedRows=bp.selected_rows(selected),
        dashGridOptions={
            **bp.GRID_OPTIONS,
            "rowBuffer": 12,
            "context": {"base": base, "threshold": threshold, "panel": panel_key},
            "localeText": {"noRowsToShow": empty_text},
            "suppressScrollOnNewData": True,
        },
        eventListeners=_events(panel_key),
        style={"height": "100%", "width": "100%"},
        className="ib-grid ib-grid-child pp-grid",
    )


def pos_def(member: str | None) -> dict[str, Any]:
    """The viewer-derived position column of the peptides grid (from the coverage)."""
    return {
        "colId": "_pos",
        "field": "_start",
        "headerName": "position",
        "headerTooltip": (
            "Viewer-derived: the first and the last residue (1-based) of the peptide in the "
            f"sequence of {member or 'the shown member'}, from an exact search in the FASTA; "
            "+n when it occurs more than once. Sorts by the first residue."
        ),
        "cellRenderer": "PpPos",
        "type": "rightAligned",
        "initialWidth": 92,
        "minWidth": 60,
        "headerClass": "ib-head-derived",
        "sortingOrder": ["asc", "desc"],
        "context": {"hideOrder": 1},
    }


def rollup_def(top_n: int | None, rollup: str | None) -> dict[str, Any]:
    rule = f"top {top_n}" if rollup == "TopNSum" and top_n else str(rollup or "rollup")
    return {
        "colId": "_rollup",
        "field": "_rollup_rank",
        "headerName": "rollup",
        "headerTooltip": (
            f"Viewer-derived: the peptide is in the {rule} sum that gives the protein's "
            "protein_group_quant.quantity (quant.rs rollup_protein_bases over the per-peptide "
            "maxima of peptide_quant); the number is the rank of the peptide's quantity. The "
            "tests check that the sum equals the engine's protein quantity."
        ),
        "cellRenderer": "PpRollup",
        "type": "rightAligned",
        "initialWidth": 88,
        "minWidth": 56,
        "headerClass": "ib-head-derived",
        "sortingOrder": ["asc", "desc"],
        "context": {"hideOrder": 2},
    }


def peptide_defs(
    rs: ResultSet,
    columns: Sequence[str],
    labels: Mapping[str, str],
    scales: Scales,
    rescorer: str | None,
    t: float,
    *,
    member: str | None,
    with_pos: bool,
    rollup: tuple[int | None, str | None] | None,
) -> list[dict[str, Any]]:
    defs = column_defs(
        "peptide",
        columns,
        labels,
        role="child",
        experiment=rs.is_experiment,
        q_active=None,
        threshold=None,
        winner_matters=True,
        scales=scales,
        rescorer=rescorer,
        marks_at=t,
    )
    extra = []
    if with_pos:
        extra.append(("peptidoform", pos_def(member)))
    if rollup is not None:
        extra.append(("quantity", rollup_def(*rollup)))
    for after, d in extra:
        idx = next((i for i, x in enumerate(defs) if x.get("colId") == after), len(defs) - 2)
        defs.insert(idx + 1, d)
    return _hide_orders(defs, PEP_HIDE)


# The order in which a narrow peptides grid hides columns (the highest first): on the
# protein page the quantity and the position stay longest.
PEP_HIDE = {
    "is_winner": 10,
    "run": 9,
    "apex_rt": 8,
    "quant_state": 7,
    "n_runs_transfer_only": 7,
    "n_precursors": 6,
    "n_runs": 5,
    "score": 4,
    "_rollup": 3,
    "_pos": 2,
    "quantity": 1,
}


def _hide_orders(defs: list[dict[str, Any]], orders: Mapping[str, int]) -> list[dict[str, Any]]:
    for d in defs:
        col = d.get("colId")
        if col == "peptidoform" and d.get("flex"):
            d["context"] = {**(d.get("context") or {}), "fitWidth": 150}
        if col in orders and not d.get("initialHide") and not d.get("lockVisible"):
            d["context"] = {**(d.get("context") or {}), "hideOrder": orders[col]}
    return defs


def precursor_defs(
    rs: ResultSet,
    columns: Sequence[str],
    labels: Mapping[str, str],
    scales: Scales,
    rescorer: str | None,
    t: float,
) -> list[dict[str, Any]]:
    return column_defs(
        "precursor",
        columns,
        labels,
        role="child",
        experiment=rs.is_experiment,
        q_active=None,
        threshold=None,
        winner_matters=True,
        scales=scales,
        rescorer=rescorer,
        marks_at=t,
    )


def pass_count(rows: Sequence[Mapping[str, Any]], column: str, t: float) -> int:
    """Rows that pass ``column <= t`` (targets; a decoy row never counts)."""
    n = 0
    for r in rows:
        v = r.get(column)
        if r.get("label") != "decoy" and v is not None and v == v and float(v) <= t:
            n += 1
    return n


def count_text(rows: Sequence[Mapping[str, Any]], total: int, column: str, t: float) -> str:
    return f"{pass_count(rows, column, t):,} of {total:,}"


def count_tip(
    noun: str, rows: Sequence[Mapping[str, Any]], total: int, column: str, t: float
) -> str:
    n = pass_count(rows, column, t)
    return (
        f"{n:,} of {total:,} {noun} pass {column} ≤ {stop_label(t)} (the validation marks' "
        "test). The table lists every row of its parent, at any q."
    )


def subject_of(row: Mapping[str, Any] | None) -> Any:
    if not row:
        return ""
    seq = str(row.get("sequence") or "")
    if not seq:
        from .widgets import parse_peptidoform

        seq = parse_peptidoform(row.get("peptidoform")).sequence
    return html.Span(
        seq,
        className="pp-subject-seq",
        title=f"base_peptide_id {row.get('base_peptide_id')}; its precursors are every scored "
        "row of this base peptide in this group",
    )


# --------------------------------------------------------------------------- notes


def note_line(text: str, *, warn: bool = False) -> Any:
    return html.Div(
        [icon("alert" if warn else "info", 14), html.Span(text)],
        className="pp-note" + (" pp-note-warn" if warn else ""),
    )


def missing_group(ctx: Any, group: str, suggestions: pd.DataFrame | None) -> Any:
    """The page for a group string that no scored row carries."""
    items = []
    if suggestions is not None and len(suggestions):
        for g in suggestions["protein_group"].head(12):
            items.append(
                dcc.Link(
                    dmc.Badge(
                        str(g),
                        variant="light",
                        color="pink",
                        size="lg",
                        radius="sm",
                        style={"textTransform": "none", "cursor": "pointer"},
                    ),
                    href=protein_href(ctx.base, str(g)),
                )
            )
    return dmc.Card(
        dmc.Stack(
            [
                dmc.ThemeIcon(
                    icon("search", 20), color="gray", variant="light", size=40, radius="xl"
                ),
                dmc.Text(f"No protein group {group!r} in this result set", fw=650),
                dmc.Text(
                    "A protein group is named by its exact string (members separated by ';'). "
                    + (
                        "Groups that contain its first member:"
                        if items
                        else "Search the groups by a part of the name."
                    ),
                    size="sm",
                    c="dimmed",
                    ta="center",
                    maw=640,
                ),
                dmc.Group(items, gap=6, justify="center") if items else None,
                dcc.Link(
                    dmc.Button("Search the protein groups", variant="light", size="compact-sm"),
                    href=protein_href(ctx.base, None, search=group.split(";")[0]),
                ),
            ],
            align="center",
            gap=8,
            py="lg",
        ),
        id="pp-missing",
    )
