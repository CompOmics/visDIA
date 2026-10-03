"""Protein page (P1 view 8): one protein group, its peptides, precursors and quantities.

Address: ``protein?group=<protein group>[&peptide=<base_peptide_id>][&member=<member>]``
(build it with :func:`protein_href`). ``group`` is the exact protein string of the
scored table (members separated by ``;``), ``peptide`` the selected peptide and
``member`` the member whose coverage is shown. Without ``group`` the page is a search
of the protein groups (``protein?search=<text>``).

The page, in the manner of PeptideShaker:

* the header: the members, the label, the species (the viewer's rule), the winning
  precursor, and with a FASTA the accession, the description and the length;
* the verdict strip at the header threshold: ``pg_q_value`` with its validation mark,
  the passing peptides, the best score, the quantity, the coverage (viewer-derived) and
  in an experiment the runs in which the group is identified (``run_psm_q``);
* the members of a group of several, each with its FASTA facts and coverage;
* the sequence coverage (:func:`.coverage.card_body`), linked to the peptides;
* the peptides of the group (every one, at any q) beside the precursors of the
  selected peptide; a click on a peptide (in the grid or on the coverage) selects it
  everywhere, and a precursor opens its precursor page;
* in an experiment, the quantity per run (``protein_group_quant`` and MaxLFQ) and the
  peptides-by-runs matrix (quantities on a log scale, ``run_psm_q`` marks, transfers).

Every number comes from :mod:`mumdia_viewer.data.protein` and the data layer;
``assets/protein.js`` keeps the panels linked and rewrites the address.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dash_mantine_components as dmc
import pandas as pd
from dash import ClientsideFunction, Input, Output, State, dcc, html
from dash import no_update as NO

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.fasta import Coverage, Fasta, protein_coverage
from mumdia_viewer.data.protein import (
    ProteinGroup,
    find_groups,
    group_from_peptides,
    group_peptides,
    matrix_sides,
    peptide_precursors,
    peptide_run_matrix,
)
from mumdia_viewer.data.rescore import rescore_info

from . import coverage
from . import protein_figures as pf
from . import protein_view as pv
from .browser_grid import child_columns, child_meta, column_defs, records, safe_columns, scales_of
from .protein_view import protein_href
from .state import PageContext, parse_threshold, stop_label
from .widgets import graph, section

log = logging.getLogger(__name__)

__all__ = ["layout", "protein_href", "register"]

# Groups are named by long strings (60 members on the Astral run's largest).
GROUP_MAX = 4000
SEARCH_MAX = 200
MATRIX = {"type": "fig", "name": "pp-matrix"}
QRUN = {"type": "fig", "name": "pp-qrun"}


# --------------------------------------------------------------------------- address


def _text(value: Any, n: int = 200) -> str:
    return str(value if value is not None else "").strip()[:n]


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return int(f) if math.isfinite(f) and f == int(f) else None


def parse_address(query: Mapping[str, Any]) -> dict[str, Any]:
    """The group, the peptide, the member and the search text of an address query."""
    return {
        "group": _text(query.get("group"), GROUP_MAX) or None,
        "peptide": _int(query.get("peptide")),
        "member": _text(query.get("member"), 200) or None,
        "search": _text(query.get("search"), 200),
    }


def pre_mark(rs: ResultSet) -> str:
    """The q column of the precursors' marks: q_value; run_psm_q in an experiment."""
    return "run_psm_q" if rs.is_experiment else "q_value"


# --------------------------------------------------------------------------- data


def _coverage(
    rs: ResultSet, fasta: Fasta | None, pg: ProteinGroup, member: str | None, t: float
) -> tuple[Coverage | None, str | None]:
    """The coverage of a member (None without a FASTA), or the reason there is none."""
    if fasta is None:
        return None, None
    if pg.decoy:
        return None, "A decoy protein group has no sequence in the FASTA, so no coverage."
    try:
        return protein_coverage(rs, fasta, pg.protein_group, member=member, threshold=t), None
    except ViewerError as exc:
        return None, str(exc)


def _positions(cov: Coverage | None) -> dict[int, tuple[int, int, int]]:
    """base_peptide_id -> (first start, its end, occurrences), 1-based, from the spans."""
    out: dict[int, tuple[int, int, int]] = {}
    if cov is None:
        return out
    for s in cov.spans:
        hit = out.get(s.base_peptide_id)
        if hit is None:
            out[s.base_peptide_id] = (s.start + 1, s.end, 1)
        else:
            a, b, n = hit
            out[s.base_peptide_id] = (
                (a, b, n + 1) if a <= s.start + 1 else (s.start + 1, s.end, n + 1)
            )
    return out


def peptide_records(
    rows: pd.DataFrame,
    base: str,
    columns: list[str],
    *,
    positions: Mapping[int, tuple[int, int, int]] | None,
    rollup: Mapping[int, int] | None,
) -> list[dict[str, Any]]:
    """The peptide grid's rows: the data layer's columns plus the viewer-derived ones."""
    out = records(rows, base, "peptide", columns)
    for r in out:
        bp = r.get("base_peptide_id")
        if positions is not None:
            hit = positions.get(int(bp)) if bp is not None else None
            r["_start"], r["_end"], r["_n_pos"] = hit if hit else (None, None, 0)
        if rollup is not None:
            r["_rollup_rank"] = rollup.get(int(bp)) if bp is not None else None
    return out


def _rollup_ranks(matrix: pd.DataFrame | None) -> dict[int, int]:
    """A single run's peptides in the protein's rollup and their ranks."""
    if matrix is None or matrix.empty:
        return {}
    part = matrix[matrix["in_rollup"]]
    return {
        int(b): int(r) for b, r in zip(part["base_peptide_id"], part["rollup_rank"], strict=True)
    }


def _matrix(rs: ResultSet, group: str, t: float, peptides: pd.DataFrame) -> pd.DataFrame | None:
    try:
        return peptide_run_matrix(rs, group, t, peptides=peptides)
    except ViewerError as exc:
        log.info("no peptide matrix for %s: %s", group, exc)
        return None


def _sides(rs: ResultSet, group: str, t: float) -> None:
    """Read the matrix's per-run parts ahead (a failure is reported where they are used)."""
    try:
        matrix_sides(rs, group, t)
    except Exception:  # read ahead only; _matrix reports the failure
        log.debug("matrix sides of %s failed ahead", group, exc_info=True)


def _quant(rs: ResultSet, group: str, t: float) -> pd.DataFrame | None:
    try:
        return matrix_sides(rs, group, t)[0]
    except ViewerError as exc:
        log.info("no quantity per run for %s: %s", group, exc)
        return None


def precursor_part(
    rs: ResultSet, base: str, group: str, peptide: int | None, t: float
) -> dict[str, Any]:
    """The precursors panel's content for one peptide (the browser applies it)."""
    mark = pre_mark(rs)
    if peptide is None:
        return {"peptide": None, "rows": [], "total": 0, "count": "0", "tip": "", "mark": mark}
    try:
        page = peptide_precursors(rs, group, int(peptide))
    except (ViewerError, ValueError) as exc:
        return {
            "peptide": peptide,
            "rows": [],
            "total": 0,
            "count": "-",
            "tip": "",
            "error": str(exc),
            "mark": mark,
            "help": pv.help_body("The data layer refused this table.", small=str(exc)),
        }
    rows = records(page.rows, base, "precursor", child_columns(list(page.rows.columns)))
    return {
        "peptide": int(peptide),
        "rows": rows,
        "total": int(page.total),
        "count": pv.count_text(rows, page.total, mark, t),
        "tip": pv.count_tip("precursor rows", rows, page.total, mark, t),
        "mark": mark,
        "help": precursor_help(rs, page.description, mark),
    }


def precursor_help(rs: ResultSet, description: str, mark: str) -> Any:
    return pv.help_body(
        "Every scored row of the selected peptide in this group, passing or not (no q "
        f"filter). The marks test {mark} at the header threshold"
        + (" (the PSM-level q within each run)." if rs.is_experiment else "."),
        "Double-click a row, press Enter or use its open icon to open its precursor page.",
        small=description,
    )


def peptide_help(rs: ResultSet, description: str, *, positions: bool, rollup: bool) -> Any:
    more = [
        "Click a peptide (or a peptide on the coverage) to see its precursors; double-click "
        "or Enter opens the precursor page of the row shown.",
    ]
    if positions:
        more.append("position: viewer-derived, from an exact search in the FASTA sequence.")
    if rollup:
        more.append(
            "rollup: viewer-derived, the peptides whose per-peptide maxima sum to the "
            "protein's quantity (the engine's top-N rule, checked to give its value)."
        )
    if rs.is_experiment:
        more.append(
            "The peptide table of an experiment is experiment-wide and has no quant columns; "
            "the peptides \u00d7 runs card gives the quantity of each peptide in each run."
        )
    return pv.help_body(
        "Every peptide of this protein group, passing or not (no q filter). The marks test "
        "peptide_q_value at the header threshold.",
        *more,
        small=description,
    )


# --------------------------------------------------------------------------- cards


def coverage_body(cov: Coverage | None, why: str | None, fasta: Fasta | None, pep: int | None):
    if fasta is None:
        return coverage.no_fasta()
    if cov is None:
        return coverage.message(why or "No coverage.", warn=True)
    return coverage.card_body(cov, selected=pep)


def coverage_card(ctx: PageContext, body: Any) -> Any:
    """The coverage view; its own head line (coverage.head) is the card's title."""
    return dmc.Card(
        html.Div(
            body,
            id="pp-cov-body",
            className="pp-cov",
            **{"data-cov-req": "pp-cov-req"},
        ),
        id="pp-cov-card",
        p="sm",
        className="pp-cov-card",
    )


def quant_table(df: pd.DataFrame) -> Any:
    labels = df.attrs.get("labels", {})
    head = ["run", "protein_group_quant", "state", "peptides"]
    tips = [None, labels.get("quantity"), labels.get("state"), labels.get("n_peptides")]
    if "lfq" in df:
        head += ["MaxLFQ", "features"]
        tips += [labels.get("lfq"), labels.get("lfq_n_features")]
    if "n_transferred" in df:
        head.append("transfers")
        tips.append(labels.get("n_transferred"))
    body = []
    for r in df.to_dict("records"):
        qty = r.get("quantity")
        state = str(r.get("state") or "")
        cells = [
            html.Td(str(r.get("run") or "run"), className="pp-mono"),
            html.Td(
                f"{qty:,.0f}"
                if qty is not None and qty == qty
                else html.Span("-", className="pp-dim"),
                className="pp-num",
            ),
            html.Td(
                html.Span(
                    [html.Span(className=f"ib-dot ib-dot-{state}"), state.replace("_", " ")],
                    className="ib-state",
                )
                if state
                else ""
            ),
            html.Td(
                "" if pd.isna(r.get("n_peptides")) else f"{int(r['n_peptides'])}",
                className="pp-num",
            ),
        ]
        if "lfq" in df:
            lfq = r.get("lfq")
            cells.append(
                html.Td(
                    f"{lfq:,.0f}"
                    if lfq is not None and lfq == lfq
                    else html.Span("-", className="pp-dim"),
                    className="pp-num",
                )
            )
            n_f = r.get("lfq_n_features")
            cells.append(html.Td("" if pd.isna(n_f) else f"{int(n_f)}", className="pp-num"))
        if "n_transferred" in df:
            n_t = r.get("n_transferred")
            cells.append(html.Td("" if pd.isna(n_t) else f"{int(n_t)}", className="pp-num"))
        body.append(html.Tr(cells))
    heads = []
    for h, tp in zip(head, tips, strict=True):
        cls = "pp-num" if h not in ("run", "state") else None
        content = (
            pv.tip(html.Span(h, className="pp-th-tip"), tp, w=320, multiline=True) if tp else h
        )
        heads.append(html.Th(content, className=cls))
    return html.Table([html.Thead(html.Tr(heads)), html.Tbody(body)], className="pp-qtable")


def quant_card(ctx: PageContext, df: pd.DataFrame | None) -> Any:
    if df is None:
        return section(
            "Quantity per run", pv.note_line("The quant tables cannot be read.", warn=True)
        )
    notes = df.attrs.get("notes") or []
    rollup, top_n = df.attrs.get("rollup"), df.attrs.get("top_n")
    rule = f"the top-{top_n} sum" if rollup == "TopNSum" and top_n else f"the {rollup} rollup"
    return section(
        "Quantity per run",
        graph("pp-qrun", pf.quantity_per_run_figure(df, ctx.scheme)),
        quant_table(df),
        *[pv.note_line(n, warn=True) for n in notes],
        help=(
            f"protein_group_quant.quantity of each run: {rule} of the per-peptide maxima "
            "(quant.rs), from the run's peptide_quant rows (its quant gate). MaxLFQ: the "
            "experiment's lfq_maxlfq.parquet, which combines the runs; 0.0 there is shown as "
            "missing. Both are intensities on one linear axis from zero; a run without a value "
            "has no bar."
        ),
        subtitle="protein_group_quant and MaxLFQ in each run",
        id="pp-qrun-card",
    )


def matrix_card(ctx: PageContext, m: pd.DataFrame | None, selected: int | None) -> Any:
    if m is None:
        return section(
            "Peptides \u00d7 runs", pv.note_line("The quant tables cannot be read.", warn=True)
        )
    rows, n_empty, n_cut = pf.matrix_rows(m)
    fig = pf.matrix_figure(
        m,
        ctx.scheme,
        threshold=ctx.threshold,
        base_href=ctx.base,
        experiment=ctx.rs.is_experiment,
        rows=rows,
        selected=selected,
    )
    n_pep = m["base_peptide_id"].nunique()
    notes = []
    if n_empty:
        notes.append(
            f"{n_empty:,} of the {n_pep:,} peptides have neither a quantity nor a row at "
            f"run_psm_q ≤ {stop_label(ctx.threshold)} in any run; they are not drawn."
        )
    if n_cut:
        notes.append(
            f"The {pf.MAX_ROWS} peptides quantified in the most runs are drawn; {n_cut:,} more "
            "are in the peptides table."
        )
    key = html.Div(
        [
            html.Span(className="pp-key pp-key-q"),
            html.Span("log10 quantity", className="pp-key-text"),
            html.Span(className="pp-key pp-key-none"),
            html.Span("no quantity", className="pp-key-text"),
            html.Span("✓", className="pp-key-check"),
            html.Span(
                f"run_psm_q ≤ {stop_label(ctx.threshold)}",
                className="pp-key-text",
                id="pp-matrix-t",
            ),
            *(
                [
                    html.Span(className="pp-key pp-key-ring"),
                    html.Span("MBR transfer", className="pp-key-text"),
                ]
                if m.attrs.get("mbr")
                else []
            ),
        ],
        className="pp-keyline",
    )
    labels = m.attrs.get("labels", {})
    return section(
        "Peptides \u00d7 runs",
        key,
        html.Div(
            graph("pp-matrix", fig, config={"displayModeBar": False}),
            className="pp-matrix-box",
            id="pp-matrix-box",
        ),
        *[pv.note_line(n) for n in notes],
        count=f"{len(rows):,} of {n_pep:,}",
        help=pv.help_body(
            "Each cell is one peptide in one run. Colour: the peptide's quantity in the run "
            "(log scale); grey: no quantity. ✓: a row of the peptide has run_psm_q at or below "
            "the header threshold in that run.",
            labels.get("quantity"),
            labels.get("identified"),
            small="Rows: the peptides with a quantity or an identification in some run, those "
            "quantified in the most runs first. Click a cell to open the precursor page of its "
            "quantity's precursor (or of the run's best row).",
        ),
        subtitle="quantity per peptide and run, log scale; marks at the header threshold",
        id="pp-matrix-card",
    )


# --------------------------------------------------------------------------- pages


def _stores(key: dict[str, Any], sel: dict[str, Any]) -> list[Any]:
    return [
        dcc.Store(id="pp-key", data=key),
        dcc.Store(id="pp-sel", data=sel),
        dcc.Store(id="pp-pre-req"),
        dcc.Store(id="pp-pre-data"),
        dcc.Store(id="pp-cov-req"),
        dcc.Store(id="pp-sync"),
    ]


def group_page(ctx: PageContext, group: str, want: Mapping[str, Any]) -> Any:
    t0 = time.perf_counter()
    rs, t, base = ctx.rs, ctx.threshold, ctx.base
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = (now - since) * 1000.0
        return now

    # The per-run sides of the matrix run beside the table queries (which share the data
    # layer's table cache and run one at a time).
    pool = ThreadPoolExecutor(max_workers=1)
    sides = pool.submit(_sides, rs, group, t)
    pool.shutdown(wait=False)
    peps = group_peptides(rs, group)
    pg = group_from_peptides(rs, group, peps)
    tt = lap("group", t0)
    if not pg.found:
        first = group.split(";")[0].removeprefix("DECOY_")
        try:
            suggestions = find_groups(rs, first, threshold=t, limit=12).rows
        except ViewerError:
            suggestions = None
        return html.Div(
            [
                *_stores({"group": group, "missing": True}, {}),
                pv.missing_group(ctx, group, suggestions),
            ],
            id="pp-root",
            className="pp-root",
        )
    fasta = ctx.fasta
    members = list(pg.members)
    member = want.get("member") if want.get("member") in members else None
    cov, why = _coverage(rs, fasta, pg, member, t)
    tt = lap("coverage", tt)
    shown = cov.member if cov is not None else member
    entries = {m: fasta.get(m) for m in members} if fasta is not None else {}
    sides.result()
    matrix = _matrix(rs, group, t, peps.rows)
    quant = _quant(rs, group, t)
    tt = lap("matrix", tt)
    rows_df = peps.rows
    ids = [int(b) for b in rows_df["base_peptide_id"]] if len(rows_df) else []
    sel = want.get("peptide") if want.get("peptide") in ids else (ids[0] if ids else None)
    positions = _positions(cov) if cov is not None else None
    single_quant = not rs.is_experiment and "quantity" in rows_df.columns
    rollup = _rollup_ranks(matrix) if single_quant else None
    columns = child_columns(list(rows_df.columns))
    pep_rows = peptide_records(rows_df, base, columns, positions=positions, rollup=rollup)
    # The precursors of the selected peptide: the browser asks for them once the page is
    # drawn (pp-pre-req), so the page does not wait for a third table query.
    pre = {"rows": [], "count": "-", "tip": "", "help": None}
    scales = scales_of(rs)
    rescorer = rescore_info(rs).label
    pep_defs = pv.peptide_defs(
        rs,
        columns,
        peps.column_labels,
        scales,
        rescorer,
        t,
        member=shown,
        with_pos=positions is not None,
        rollup=(quant.attrs.get("top_n"), quant.attrs.get("rollup"))
        if single_quant and quant is not None
        else None,
    )
    pre_columns, pre_labels = child_meta(rs, "precursor")
    pre_defs = pv.precursor_defs(rs, child_columns(pre_columns), pre_labels, scales, rescorer, t)
    sel_row = next((r for r in pep_rows if r.get("base_peptide_id") == sel), None)
    pre_sel = pre["rows"][0]["_key"] if pre["rows"] else None
    # In a single run each precursor is one scored row, so the rows add nothing there.
    n_rows = int(matrix["n_rows"].sum()) if matrix is not None and rs.is_experiment else None
    n_prec = int(rows_df["n_precursors"].sum()) if "n_precursors" in rows_df else None
    hero = pv.hero(
        ctx,
        pg,
        entry=cov.entry if cov is not None else (entries.get(members[0]) if entries else None),
        member=shown,
        n_rows=n_rows,
        n_precursors=n_prec,
        fasta_label=fasta.label if fasta is not None else None,
    )
    strip_children = pv.strip(
        ctx,
        pg,
        peptides=rows_df,
        quant=quant,
        matrix=matrix,
        coverage=cov,
        scales=scales,
        rescorer=rescorer,
    )
    members_slot = html.Div(id="pp-members-slot")
    if len(members) > 1:
        covs = _member_coverages(rs, fasta, pg, t) if fasta is not None else {}
        members_slot.children = pv.members_card(
            ctx, pg, entries, covs, shown, fasta.label if fasta is not None else None
        )
    pep_panel = pv.panel(
        "pep",
        "Peptides",
        pv.grid(
            pv.PEP_GRID,
            "pep",
            pep_defs,
            pep_rows,
            base,
            t,
            selected=str(sel) if sel is not None else None,
            empty_text="No peptides",
        ),
        count=pv.count_text(pep_rows, peps.total, "peptide_q_value", t),
        count_tip=pv.count_tip("peptides", pep_rows, peps.total, "peptide_q_value", t),
        help_text=peptide_help(
            rs, peps.description, positions=positions is not None, rollup=rollup is not None
        ),
        right=html.Span(
            "any q",
            className="pp-note-any",
            title="No q filter: every peptide of the group, passing or not.",
        ),
    )
    pre_panel = pv.panel(
        "pre",
        "Precursors of",
        pv.grid(
            pv.PRE_GRID,
            "pre",
            pre_defs,
            pre["rows"],
            base,
            t,
            selected=pre_sel,
            empty_text="No precursors",
        ),
        loading=sel is not None,
        count=pre["count"],
        count_tip=pre["tip"],
        help_text=pre.get("help") or precursor_help(rs, "", pre_mark(rs)),
        subject=pv.subject_of(sel_row),
        right=html.Span(
            "any q",
            className="pp-note-any",
            title="No q filter: every scored row of the peptide, passing or not.",
        ),
    )
    with_cov = cov is not None and cov.entry is not None
    main = html.Div(
        [pep_panel, pre_panel, coverage_card(ctx, coverage_body(cov, why, fasta, sel))],
        className="pp-main " + ("pp-main-cov" if with_cov else "pp-main-nocov"),
    )
    blocks: list[Any] = [
        hero,
        dmc.Card(strip_children, id="pp-strip", p="sm", className="pp-strip-card"),
        members_slot,
        main,
    ]
    if rs.is_experiment:
        blocks.append(
            html.Div(
                [quant_card(ctx, quant), matrix_card(ctx, matrix, sel)],
                className="pp-exp-row",
            )
        )
    key = {
        "group": group,
        "decoy": pg.decoy,
        "experiment": rs.is_experiment,
        "pre_mark": pre_mark(rs),
        "member": shown,
        "page": uuid.uuid4().hex,
        "fasta": fasta is not None,
    }
    timings["page"] = (time.perf_counter() - t0) * 1000.0
    log.debug("protein page %s: %s", group, {k: round(v) for k, v in timings.items()})
    return html.Div(
        [*_stores(key, {"peptide": sel, "member": shown}), *blocks],
        id="pp-root",
        className="pp-root",
        **{
            "data-group": group,
            "data-base": base,
            "data-page": key["page"],
            "data-peptide": "" if sel is None else str(sel),
            "data-member": shown or "",
            "data-experiment": "1" if rs.is_experiment else "0",
            "data-pre-mark": pre_mark(rs),
            "data-ms": f"{timings['page']:.0f}",
        },
    )


def _member_coverages(
    rs: ResultSet, fasta: Fasta, pg: ProteinGroup, t: float
) -> dict[str, Coverage | None]:
    out: dict[str, Coverage | None] = {}
    if pg.decoy:
        return out
    for m in pg.members:
        if fasta.get(m) is None:
            out[m] = None
            continue
        try:
            out[m] = protein_coverage(rs, fasta, pg.protein_group, member=m, threshold=t)
        except ViewerError:
            out[m] = None
    return out


def find_rows(
    rs: ResultSet, base: str, text: str, t: float
) -> tuple[list[dict[str, Any]], int, str, list[str], dict[str, str]]:
    """The search's grid rows, the number of matches, the data layer's description, and
    the table's columns and labels."""
    page = find_groups(rs, text, threshold=t, limit=SEARCH_MAX)
    rows = records(page.rows, base, "protein_group")
    for r in rows:
        r["_protein"] = protein_href(base, str(r.get("protein_group") or ""))
    columns, labels = list(page.rows.columns), dict(page.column_labels)
    return rows, int(page.total), page.description, columns, labels


def find_defs(
    rs: ResultSet, columns: list[str], labels: Mapping[str, str], t: float
) -> list[dict[str, Any]]:
    defs = column_defs(
        "protein_group",
        columns,
        labels,
        role="child",
        experiment=rs.is_experiment,
        q_active=None,
        threshold=None,
        winner_matters=False,
        scales=scales_of(rs),
        rescorer=rescore_info(rs).label,
        marks_at=t,
    )
    # A row opens the protein page; the precursor page is one more click away.
    return [d for d in defs if d.get("colId") != "_open"]


def find_summary(total: int, shown: int, text: str, t: float) -> tuple[str, str]:
    if text:
        line = f"{total:,} protein groups contain “{text}” (any q; targets)."
    else:
        line = f"{total:,} protein groups pass pg_q_value ≤ {stop_label(t)} (targets)."
    if total > shown:
        line += f" The first {shown:,} by best score are listed; refine the search for others."
    return (f"{shown:,} of {total:,}" if total > shown else f"{total:,}"), line


def search_page(ctx: PageContext, text: str) -> Any:
    rs, t = ctx.rs, ctx.threshold
    try:
        rows, total, description, columns, labels = find_rows(rs, ctx.base, text, t)
        error = None
    except ViewerError as exc:
        rows, total, description, error = [], 0, "", str(exc)
        columns, labels = safe_columns(rs, "protein_group", threshold=None)
    count, line = find_summary(total, len(rows), text, t)
    grid = pv.grid(
        pv.FIND_GRID,
        "find",
        find_defs(rs, columns, labels, t),
        rows,
        ctx.base,
        t,
        selected=None,
        empty_text="No protein group contains this text",
    )
    head = html.Div(
        [
            html.Div(f"Protein groups · {pv.where(rs)}", className="mv-eyebrow"),
            html.Div("Find a protein group", className="mv-title"),
            dmc.Text(
                "Type part of a protein name (the protein group string, any case). A row opens "
                "the protein page: its peptides, coverage and quantities.",
                size="sm",
                c="dimmed",
            ),
            dmc.TextInput(
                id="pp-find-input",
                value=text,
                placeholder="for example ATLA3, MACF1 or _YEAST",
                leftSection=pv.icon("search", 16),
                debounce=300,
                radius="md",
                size="md",
                w=420,
                mt="sm",
                **{"aria-label": "Search the protein groups (substring, any case)"},
            ),
        ],
        className="pp-find-head",
    )
    panel = pv.panel(
        "find",
        "Protein groups",
        grid,
        count=count,
        count_tip=line,
        help_text=pv.help_body(line, small=error or description),
        right=html.Span("click a row to open it", className="pp-note-any"),
    )
    return html.Div(
        [
            *_stores({"search": text, "page": uuid.uuid4().hex}, {}),
            head,
            html.Div(panel, className="pp-find-panel"),
        ],
        id="pp-root",
        className="pp-root pp-root-find",
        **{"data-base": ctx.base, "data-search": "1"},
    )


def layout(ctx: PageContext) -> Any:
    want = parse_address(ctx.query)
    if not want["group"]:
        return search_page(ctx, want["search"])
    return group_page(ctx, want["group"], want)


# --------------------------------------------------------------------------- callbacks


def _ctx(get_rs, base: str, t: Any, scheme: Any, fasta: Fasta | None = None) -> PageContext:
    return PageContext(
        rs=get_rs(),
        base=base,
        threshold=parse_threshold(t),
        scheme="dark" if scheme == "dark" else "light",
        fasta=fasta,
    )


def follow(
    rs: ResultSet,
    base: str,
    fasta: Fasta | None,
    key: Mapping[str, Any],
    sel: Mapping[str, Any] | None,
    cov_req: Mapping[str, Any] | None,
    t: float,
) -> tuple[Any, Any, Any, str | None]:
    """The strip, the members card and the coverage at ``t`` for the shown member."""
    group = str(key.get("group") or "")
    ctx = PageContext(rs=rs, base=base, threshold=t, fasta=fasta)
    peps = group_peptides(rs, group)
    pg = group_from_peptides(rs, group, peps)
    member = (cov_req or {}).get("member") or (sel or {}).get("member") or key.get("member")
    member = member if member in pg.members else None
    cov, why = _coverage(rs, fasta, pg, member, t)
    matrix = _matrix(rs, group, t, peps.rows) if rs.is_experiment else None
    quant = _quant(rs, group, t)
    strip_children = pv.strip(
        ctx,
        pg,
        peptides=peps.rows,
        quant=quant,
        matrix=matrix,
        coverage=cov,
        scales=scales_of(rs),
        rescorer=rescore_info(rs).label,
    )
    members = NO
    shown = cov.member if cov is not None else member
    if len(pg.members) > 1:
        entries = {m: fasta.get(m) for m in pg.members} if fasta is not None else {}
        covs = _member_coverages(rs, fasta, pg, t) if fasta is not None else {}
        members = pv.members_card(ctx, pg, entries, covs, shown, fasta.label if fasta else None)
    pep = (sel or {}).get("peptide")
    pep = (cov_req or {}).get("peptide") if pep is None else pep
    body = coverage_body(cov, why, fasta, _int(pep))
    return strip_children, members, body, shown


def register(app, get_rs, base: str) -> None:
    """Callbacks of the protein page (app.callback and app.clientside_callback only)."""
    get_fasta = getattr(app, "mv_fasta", lambda: None)

    # The precursors of the selected peptide (asked for by the browser).
    @app.callback(
        Output("pp-pre-data", "data"),
        Input("pp-pre-req", "data"),
        State("pp-key", "data"),
        prevent_initial_call=True,
    )
    def precursors(req, key):
        if not req or not key or not key.get("group") or req.get("peptide") is None:
            return NO
        t0 = time.perf_counter()
        out = precursor_part(
            get_rs(),
            base,
            str(key["group"]),
            _int(req.get("peptide")),
            parse_threshold(req.get("t")),
        )
        out["n"] = req.get("n")
        log.debug("protein precursors in %.0f ms", (time.perf_counter() - t0) * 1000.0)
        return out

    app.clientside_callback(
        ClientsideFunction("mvp", "applyPre"),
        Output("pp-sync", "data"),
        Input("pp-pre-data", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvp", "threshold"),
        Output("pp-sync", "data", allow_duplicate=True),
        Input("threshold", "data"),
        State("pp-key", "data"),
        prevent_initial_call=True,
    )

    # The parts that test the threshold or show the member: strip, members, coverage.
    @app.callback(
        Output("pp-strip", "children"),
        Output("pp-members-slot", "children"),
        Output("pp-cov-body", "children"),
        Input("threshold", "data"),
        Input("pp-cov-req", "data"),
        State("pp-key", "data"),
        State("pp-sel", "data"),
        prevent_initial_call=True,
    )
    def follow_threshold(t, cov_req, key, sel):
        if not key or not key.get("group") or key.get("missing"):
            return NO, NO, NO
        t0 = time.perf_counter()
        strip_children, members, body, _shown = follow(
            get_rs(), base, get_fasta(), key, sel, cov_req, parse_threshold(t)
        )
        log.debug("protein follow in %.0f ms", (time.perf_counter() - t0) * 1000.0)
        return strip_children, members, body

    # The heatmap: its marks test the threshold, its ramp follows the colour scheme.
    @app.callback(
        Output(MATRIX, "figure", allow_duplicate=True),
        Output("pp-matrix-t", "children"),
        Input("threshold", "data"),
        Input("scheme", "data"),
        State("pp-key", "data"),
        State("pp-sel", "data"),
        prevent_initial_call=True,
    )
    def matrix_follow(t, scheme, key, sel):
        if not key or not key.get("group") or not key.get("experiment"):
            return NO, NO
        rs = get_rs()
        ctx = _ctx(get_rs, base, t, scheme)
        group = str(key["group"])
        m = _matrix(rs, group, ctx.threshold, group_peptides(rs, group).rows)
        if m is None:
            return NO, NO
        rows, _, _ = pf.matrix_rows(m)
        fig = pf.matrix_figure(
            m,
            ctx.scheme,
            threshold=ctx.threshold,
            base_href=base,
            experiment=rs.is_experiment,
            rows=rows,
            selected=_int((sel or {}).get("peptide")),
        )
        return fig, f"run_psm_q ≤ {stop_label(ctx.threshold)}"

    # The search.
    @app.callback(
        Output(pv.FIND_GRID, "rowData"),
        Output("pp-find-count", "children"),
        Output("pp-find-countbox", "title"),
        Output("pp-find-help", "label"),
        Input("pp-find-input", "value"),
        Input("threshold", "data"),
        prevent_initial_call=True,
    )
    def search(text, t):
        rs = get_rs()
        words = _text(text)
        threshold = parse_threshold(t)
        try:
            rows, total, description, _, _ = find_rows(rs, base, words, threshold)
            error = None
        except ViewerError as exc:
            rows, total, description, error = [], 0, "", str(exc)
        count, line = find_summary(total, len(rows), words, threshold)
        return rows, count, line, pv.help_body(line, small=error or description)

    app.clientside_callback(
        ClientsideFunction("mvp", "searchAddress"),
        Output("pp-sync", "data", allow_duplicate=True),
        Input("pp-find-input", "value"),
        prevent_initial_call=True,
    )
