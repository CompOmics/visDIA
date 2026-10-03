"""The grids of the identification page: column definitions, bar scales, facets, records.

Three tables share one vocabulary: the protein-group, peptide and precursor tables of
:mod:`mumdia_viewer.data.tables`. Each can be the first table of the page (role
``"top"``: AG Grid's infinite row model, sorted and paged by the data layer) or a child
table (role ``"child"``: every row of one parent, loaded whole and sorted in the
browser). The cell renderers named here live in ``assets/browser.js``; they wrap the
shared ones of ``assets/clientside.js`` (``MvValidation``, ``MvBar``, ``MvPeptidoform``).

In-cell bars (PeptideShaker's JSparklines) have one scale per column, the same on every
row, stated in the column's header tooltip: scores from the scored table's range, q
values on -log10 from 1 to 1e-4, counts on fixed limits and quantities on a log scale
from the quant table's range (see :func:`bar_params`). A bar is a display only. A count
taken at any q (``n_precursors`` always; ``n_runs`` in a child table, which has no q
filter) has a grey bar and an "any q" mark on its header, so it does not read as a
count at the header threshold.

Column definitions use AG Grid's ``initial*`` attributes (``initialHide``,
``initialWidth``, ``initialSort``) so that new definitions (new header tooltips after a
threshold change) keep what the user did: hidden columns, widths and the sort.

A child grid carries only the columns it can show (:func:`child_columns`): its default
columns, the few its Columns menu offers, and the keys and flags the page needs. The
long protein strings are left out (about half of the payload).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.duck import sql_path
from mumdia_viewer.data.tables import (
    GROUPED_Q,
    Q_COLUMNS,
    QUANT_STATES,
    UNIT_Q_COLUMN,
    TableQuery,
    identification_table,
)

from .state import href, stop_label

GRID_ID = "ib-grid"
PEP_GRID_ID = "ib-pep-grid"
PRE_GRID_ID = "ib-pre-grid"
# Rows per block of the infinite row model (one server request per block).
BLOCK_SIZE = 100

# Bars: the fixed limits of the q and count columns.
Q_BAR_FULL = 1e-4
N_PEPTIDES_MAX = 1000
N_PRECURSORS_MAX = 10
BAR_WIDTH = 30
# Child panels are half as wide as the first table: narrower bars.
CHILD_BAR_WIDTH = 24
# Counts taken at any q: n_precursors always, the run and peptide counts when the table
# has no q filter (a child table).
ANY_Q_ALWAYS = frozenset({"n_precursors"})
ANY_Q_WITHOUT_FILTER = frozenset({"n_runs", "n_runs_transfer_only", "n_peptides"})

# What each scored or joined column holds, for the header tooltips. The data layer's
# ``column_labels`` (the q columns and every derived column) take precedence.
MEANINGS: dict[str, str] = {
    "candidate_id": "psms_scored.candidate_id: the library candidate of this row",
    "run": "The run of this row (the run directory; source in the pooled scored table)",
    "source": "psms_scored.source: the run index of this row in the pooled scored table",
    "peptidoform": (
        "psms_scored.peptidoform: a modified residue is drawn in its modification's colour "
        "with a short tag (ox, cam, ph); hover it for the full name. DECOY_ in orange"
    ),
    "charge": "psms_scored.charge: precursor charge",
    "label": "psms_scored.label: target or decoy",
    "protein": "psms_scored.protein: every protein the peptide maps to",
    "protein_group": (
        "psms_scored.protein_group: the members of the group, separated by ';' (the first "
        "in bold, the others as chips; hover the cell for all of them)"
    ),
    "base_peptide_id": (
        "psms_scored.base_peptide_id: the peptide key that pairs a target with its decoy"
    ),
    "score": "psms_scored.score: the rescorer's score (higher is better)",
    "prelim_score": (
        "psms_scored.prelim_score: competition uses it to pick the winner of a key; it is "
        "not a classifier input"
    ),
    "selected_peak_rank": (
        "psms_scored.selected_peak_rank: the extracted peak the scored row chose (0 is "
        "the best peak of extraction)"
    ),
    "apex_rt": "psms_scored.apex_rt: apex of the identification, seconds",
    "elution_lo": "psms_scored.elution_lo: start of the elution bounds, seconds",
    "elution_hi": "psms_scored.elution_hi: end of the elution bounds, seconds",
}

# Header text of each column (the q columns keep their engine names).
HEADERS: dict[str, str] = {
    "candidate_id": "candidate",
    "run": "run",
    "source": "source",
    "peptidoform": "peptidoform",
    "sequence": "sequence",
    "charge": "z",
    "label": "label",
    "protein": "protein",
    "protein_group": "protein group",
    "base_peptide_id": "base peptide",
    "score": "score",
    "prelim_score": "prelim score",
    "selected_peak_rank": "peak rank",
    "apex_rt": "apex RT (s)",
    "elution_lo": "elution start (s)",
    "elution_hi": "elution end (s)",
    "n_precursors": "precursors",
    "n_peptides": "peptides",
    "n_runs": "runs",
    "n_runs_transfer_only": "transfer-only runs",
    "is_winner": "winner",
    "quantity": "quantity",
    "quant_status": "quant_status",
    "quant_state": "quant state",
    "quant_n_peptides": "quant peptides",
    "is_transferred": "MBR",
    "transfer_q": "transfer_q",
    "is_entrapment": "spike-in",
    "_species": "species",
}

# Renderer kind of each column; anything else is plain text.
KINDS: dict[str, str] = {
    "candidate_id": "id",
    "base_peptide_id": "id",
    "source": "int",
    "run": "run",
    "peptidoform": "pep",
    "sequence": "mono",
    "charge": "charge",
    "label": "label",
    "protein": "text",
    "protein_group": "group",
    "score": "score",
    "prelim_score": "num",
    "selected_peak_rank": "int",
    "apex_rt": "rt",
    "elution_lo": "rt",
    "elution_hi": "rt",
    "n_precursors": "count",
    "n_peptides": "count",
    "n_runs": "count",
    "n_runs_transfer_only": "int",
    "quant_n_peptides": "int",
    "is_winner": "winner",
    "quantity": "quantity",
    "quant_status": "text",
    "quant_state": "quant",
    "is_transferred": "transfer",
    "transfer_q": "plainq",
    "is_entrapment": "spike",
    **{c: "q" for c in Q_COLUMNS},
}

RENDERERS = {
    "pep": "IbPep",
    "group": "IbGroup",
    "label": "IbLabel",
    "charge": "IbCharge",
    "q": "IbBar",
    "score": "IbBar",
    "count": "IbBar",
    "quantity": "IbBar",
    "plainq": "IbNumber",
    "quant": "IbQuantState",
    "winner": "IbWinner",
    "transfer": "IbFlag",
    "spike": "IbFlag",
    "run": "IbRun",
    "rt": "IbNumber",
    "num": "IbNumber",
    "int": "IbNumber",
    "id": "IbNumber",
    "mono": "IbText",
    "text": "IbText",
}
NUMERIC = {"q", "score", "count", "quantity", "plainq", "rt", "num", "int", "id"}
WIDTHS = {
    "pep": 230,
    "group": 240,
    "text": 150,
    "label": 74,
    "charge": 42,
    "q": 104,
    "score": 100,
    "count": 82,
    "quantity": 102,
    "plainq": 96,
    "quant": 124,
    "rt": 84,
    "num": 96,
    "int": 84,
    "id": 100,
    "run": 52,
    "winner": 84,
    "transfer": 76,
    "spike": 84,
    "mono": 170,
}
# Columns whose header needs more than their kind's width.
COLUMN_WIDTHS = {
    "peptide_q_value": 118,
    "experiment_psm_q": 130,
    "global_q_value": 122,
    "precursor_q": 108,
    "n_runs_transfer_only": 128,
    "n_precursors": 88,
    "selected_peak_rank": 92,
    "elution_lo": 120,
    "elution_hi": 116,
    "quant_status": 116,
    "quant_n_peptides": 112,
}
# The bar columns' value widths (browser.css): the bars line up down a column.
BAR_CLASS = {"q": "ib-bar-q", "score": "ib-bar-score", "count": "ib-bar-count"}
# The extra width of a count column whose header says "any q" (none: the mark goes
# under the header text, browser.css).
ANY_Q_EXTRA = 0


# --------------------------------------------------------------------------- facets


@dataclass(frozen=True)
class Facets:
    """Values the filters and the bars need, read once per result set.

    ``score_lo``/``score_hi`` scale the score bars; ``quantity`` and ``pg_quantity`` are
    the smallest and largest positive quantity of the peptide_quant tables (all runs)
    and of protein_group_quant (single runs), which scale the quantity bars;
    ``statuses`` are the raw ``quant_status`` strings that are not quant states.
    """

    charges: tuple[int, ...]
    score_lo: float | None
    score_hi: float | None
    statuses: tuple[str, ...]
    quantity: tuple[float, float] | None = None
    pg_quantity: tuple[float, float] | None = None


_FACETS: dict[tuple[Any, ...], Facets] = {}
_MODS: dict[tuple[Any, ...], tuple[str, ...]] = {}
_FACET_LOCK = threading.Lock()


def _facet_key(rs: ResultSet) -> tuple[Any, ...]:
    return (id(rs), rs.scored.parquet().stamp)


def _range(rows: Sequence[Any]) -> tuple[float, float] | None:
    lo = [_finite(r[0]) for r in rows]
    hi = [_finite(r[1]) for r in rows]
    los = [v for v in lo if v is not None]
    his = [v for v in hi if v is not None]
    if not los or not his:
        return None
    a, b = min(los), max(his)
    return (a, b) if b > a else None


QUANTITY_RANGE_SQL = (
    "SELECT min(quantity), max(quantity) FROM read_parquet(?) "
    "WHERE quantity > 0 AND isfinite(quantity)"
)


def _artifact_rows(rs: ResultSet, sql: str, art: Any) -> list[Any]:
    """The rows of a facet query on one artifact; none when it cannot be read."""
    if art is None or not art.usable:
        return []
    try:
        return list(rs.duck.execute(sql, [sql_path(art.require())]).fetchall())
    except Exception:
        return []


def facets(rs: ResultSet) -> Facets:
    """Charges, score and quantity ranges and raw quant statuses of ``rs`` (cached)."""
    key = _facet_key(rs)
    with _FACET_LOCK:
        hit = _FACETS.get(key)
    if hit is not None:
        return hit
    path = sql_path(rs.scored.require())
    try:
        charges = tuple(
            int(r[0])
            for r in rs.duck.execute(
                "SELECT DISTINCT charge FROM read_parquet(?) WHERE charge IS NOT NULL ORDER BY 1",
                [path],
            ).fetchall()
        )
        lo, hi = rs.duck.execute(
            "SELECT min(score), max(score) FROM read_parquet(?) WHERE isfinite(score)", [path]
        ).fetchone() or (None, None)
    except Exception:
        charges, lo, hi = (), None, None
    statuses: set[str] = set()
    q_ranges: list[Any] = []
    pg_ranges: list[Any] = []
    for run in rs.runs:
        art = run.artifact("peptide_quant")
        status_sql = (
            "SELECT DISTINCT quant_status FROM read_parquet(?) WHERE quant_status IS NOT NULL"
        )
        statuses.update(str(r[0]) for r in _artifact_rows(rs, status_sql, art))
        q_ranges += _artifact_rows(rs, QUANTITY_RANGE_SQL, art)
        if not rs.is_experiment:
            pg = run.artifact("protein_group_quant")
            pg_ranges += _artifact_rows(rs, QUANTITY_RANGE_SQL, pg)
    out = Facets(
        charges=charges,
        score_lo=_finite(lo),
        score_hi=_finite(hi),
        statuses=tuple(sorted(s for s in statuses if s not in QUANT_STATES)),
        quantity=_range([r for r in q_ranges if r]),
        pg_quantity=_range([r for r in pg_ranges if r]),
    )
    with _FACET_LOCK:
        _FACETS[key] = out
    return out


def modifications(rs: ResultSet) -> tuple[str, ...]:
    """The modification names in the peptidoforms, most frequent first (cached).

    About 0.1 s on a single Astral run and 0.2 s on six runs, so the page asks for them
    after it has loaded.
    """
    key = _facet_key(rs)
    with _FACET_LOCK:
        hit = _MODS.get(key)
    if hit is not None:
        return hit
    sql = (
        "SELECT m, count(*) AS n FROM (SELECT unnest(regexp_extract_all(peptidoform, "
        r"'\[([^\]]*)\]|\(([^)]*)\)', 1)) AS m FROM (SELECT DISTINCT peptidoform FROM "
        "read_parquet(?))) WHERE m IS NOT NULL AND m <> '' GROUP BY 1 ORDER BY 2 DESC, 1"
    )
    try:
        rows = rs.duck.execute(sql, [sql_path(rs.scored.require())]).fetchall()
        mods = tuple(str(r[0]) for r in rows)
    except Exception:
        mods = ()
    with _FACET_LOCK:
        _MODS[key] = mods
    return mods


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- bars


@dataclass(frozen=True)
class Scales:
    """The fixed scales of the in-cell bars of one result set."""

    score: tuple[float, float] | None = None
    quantity: tuple[float, float] | None = None
    pg_quantity: tuple[float, float] | None = None
    n_runs: int = 1


def scales_of(rs: ResultSet) -> Scales:
    fac = facets(rs)
    score = (
        (fac.score_lo, fac.score_hi)
        if fac.score_lo is not None and fac.score_hi is not None and fac.score_hi > fac.score_lo
        else None
    )
    return Scales(
        score=score,
        quantity=fac.quantity,
        pg_quantity=fac.pg_quantity,
        n_runs=max(1, len(rs.runs)),
    )


def _g(v: float) -> str:
    return f"{v:.3g}"


def any_q(column: str, threshold: float | None) -> bool:
    """Whether a count column counts rows at any q in a table with this threshold."""
    return column in ANY_Q_ALWAYS or (threshold is None and column in ANY_Q_WITHOUT_FILTER)


def bar_params(
    column: str,
    table: str,
    scales: Scales,
    *,
    anyq: bool = False,
    width: int = BAR_WIDTH,
) -> tuple[dict[str, Any], str] | None:
    """The MvBar parameters of a column and the sentence that states its scale.

    None when the column has no bar (or its scale is unknown). ``anyq`` marks a count at
    any q: its bar is grey.
    """
    kind = KINDS.get(column)
    out = _bar_params(column, table, scales, kind, width)
    if out is None or not anyq:
        return out
    params, tip = out
    return {**params, "colour": "var(--ib-bar-anyq)"}, tip + (
        "; grey: a count at any q, not at the header threshold"
    )


def _bar_params(
    column: str, table: str, scales: Scales, kind: str | None, width: int
) -> tuple[dict[str, Any], str] | None:
    if kind == "q":
        params = {
            "scale": "neglog10",
            "min": 1.0,
            "max": Q_BAR_FULL,
            "passColour": "var(--ib-bar-pass)",
            "failColour": "var(--ib-bar-fail)",
            "format": "q",
            "width": width,
        }
        tip = (
            "bar: -log10(q), empty at q = 1 and full at q ≤ 1e-4; green when the row "
            "passes the header threshold on this column, grey when not"
        )
        if column in GROUPED_Q:
            tip += "; rows that are not their group's winning row hold 1.0 (no bar)"
        return params, tip
    if column == "score" and scales.score is not None:
        lo, hi = scales.score
        params = {
            "scale": "linear",
            "min": lo,
            "max": hi,
            "colour": "var(--ib-bar-score)",
            "format": "score",
            "width": width,
        }
        return params, (
            f"bar: linear from the lowest ({_g(lo)}) to the highest ({_g(hi)}) score of "
            "the scored table"
        )
    if column == "n_peptides":
        params = {
            "scale": "log10",
            "min": 1,
            "max": N_PEPTIDES_MAX,
            "colour": "var(--ib-bar-peptide)",
            "format": "int",
            "width": width,
        }
        return params, f"bar: log10 scale, empty at 1 and full at {N_PEPTIDES_MAX:,} peptides"
    if column == "n_precursors":
        params = {
            "scale": "linear",
            "min": 0,
            "max": N_PRECURSORS_MAX,
            "colour": "var(--ib-bar-precursor)",
            "format": "int",
            "width": width,
        }
        return params, (
            f"bar: linear from 0 to {N_PRECURSORS_MAX} precursors ({N_PRECURSORS_MAX} or "
            "more fill it)"
        )
    if column == "n_runs":
        n = scales.n_runs
        params = {
            "scale": "linear",
            "min": 0,
            "max": n,
            "colour": "var(--ib-bar-runs)",
            "format": "int",
            "width": width,
        }
        return params, f"bar: linear from 0 to the {n} runs of the experiment"
    if column == "quantity":
        rng = scales.pg_quantity if table == "protein_group" else scales.quantity
        if rng is None:
            return None
        source = "protein_group_quant" if table == "protein_group" else "peptide_quant"
        params = {
            "scale": "log10",
            "min": rng[0],
            "max": rng[1],
            "colour": "var(--ib-bar-quantity)",
            "format": "compact",
            "width": width,
        }
        return params, (
            f"bar: log10 scale from the smallest ({_g(rng[0])}) to the largest "
            f"({_g(rng[1])}) positive quantity of {source}; a row without a quantity "
            "shows its quant state instead (not quantifiable or not selected, never 0)"
        )
    return None


# --------------------------------------------------------------------------- columns


def validation_column(table: str, role: str, experiment: bool, q_active: str | None) -> str:
    """The q column of a table's validation marks.

    The first table uses its own filter column. A child table shows every row (no q
    filter) and marks it on its unit's column: ``peptide_q_value`` for peptides; for
    precursors ``q_value`` in a single run and ``run_psm_q`` (the PSM-level q within
    each run) in an experiment, whose grouped columns are experiment-wide.
    """
    if role == "top" and q_active:
        return q_active
    if table == "precursor":
        return "run_psm_q" if experiment else "q_value"
    return UNIT_Q_COLUMN[table]


def default_columns(
    table: str,
    role: str,
    columns: Sequence[str],
    *,
    experiment: bool,
    q_column: str,
    winner_matters: bool,
) -> list[str]:
    """The columns shown by default, in display order (only those the table has).

    The q column of the validation marks comes right after the key column (and, for
    precursors, after the charge and the run that tell the rows apart), so the value
    the mark tests stays in view on a narrow screen.
    """
    top = role == "top"
    if table == "protein_group":
        wanted: list[str | None] = [
            "protein_group",
            "_species",
            "pg_q_value",
            q_column,
            "n_peptides",
            "n_runs",
            "n_runs_transfer_only",
            "score",
            "quantity",
            "quant_state",
            "is_winner" if winner_matters else None,
        ]
    elif table == "peptide":
        wanted = [
            "peptidoform",
            "peptide_q_value",
            q_column,
            "score",
            "n_precursors",
            "n_runs",
            "n_runs_transfer_only",
            "protein" if top else None,
            "apex_rt",
            "run" if experiment and not top else None,
            "quantity",
            "quant_state",
            "is_winner" if (winner_matters or not top) else None,
        ]
    else:
        wanted = [
            "peptidoform",
            "charge",
            "run" if experiment else None,
            # Transfers are shown next to their run (and ringed on the validation mark).
            "is_transferred",
            q_column,
            "run_psm_q" if experiment and top else None,
            "score",
            "protein" if top else None,
            "apex_rt",
            "quantity",
            "quant_state",
        ]
    present = set(columns) | ({"_species"} if "protein_group" in columns else set())
    out: list[str] = []
    for c in wanted:
        if c and c in present and c not in out:
            out.append(c)
    return out


# The columns a grid may hide when it is created and its columns do not fit its width
# (the Columns menus show them again), in this order: the highest number first. A
# missing quantity names its quant state in the cell, so the quant state column goes
# early; the peptides panel gives up its quantity before the precursors panel beside it.
HIDE_TOP = {
    "is_winner": 8,
    "apex_rt": 7,
    "quant_state": 6,
    "_species": 5,
    "quantity": 4,
    "run_psm_q": 3,
    "protein": 2,
}
HIDE_CHILD = {"is_winner": 7, "apex_rt": 6, "quant_state": 5}
HIDE_PEPTIDE_CHILD = {**HIDE_CHILD, "run": 4, "quantity": 3}
HIDE_PRECURSOR_CHILD = {**HIDE_CHILD, "quantity": 2}


def hide_order(table: str, role: str, column: str) -> int | None:
    """Where a column comes in the order a grid that does not fit hides columns, or None."""
    if role == "top":
        return HIDE_TOP.get(column)
    rules = HIDE_PEPTIDE_CHILD if table == "peptide" else HIDE_PRECURSOR_CHILD
    return rules.get(column)


def header_tip(
    column: str,
    labels: Mapping[str, str],
    *,
    table: str,
    q_active: str | None,
    threshold: float | None,
    rescorer: str | None,
    scales: Scales,
) -> str:
    """The header tooltip of a column: what it holds (the data layer's words) and its bar."""
    if column == "_species":
        return (
            "Viewer-derived (the viewer's rule, not a column of the data): the species from "
            "the entry-name suffix of the group's members (_HUMAN, _YEAST, _ECOLI and other "
            "UniProt mnemonics)."
        )
    if column == "score" and table == "protein_group":
        text = (
            "best score: psms_scored.score of the row shown, which is the group's winning "
            "row (its highest-scoring row) unless that row does not pass the filters "
            "(is_winner False)"
        )
    else:
        text = labels.get(column) or MEANINGS.get(column) or column
    if column == "score" and rescorer:
        text += f". Rescorer: {rescorer}"
    if column == q_active and threshold is not None:
        text = f"Filter column of this table: {column} <= {stop_label(threshold)}. " + text
    bar = bar_params(column, table, scales, anyq=any_q(column, threshold))
    if bar is not None:
        text += ". " + bar[1][0].upper() + bar[1][1:]
    return text


def mark_tip(q_mark: str, threshold: float | None, label: str, spike: bool) -> str:
    """The validation column's header tooltip (``{t}`` stands for a threshold of None)."""
    t = stop_label(threshold) if threshold is not None else "{t}"
    return (
        f"Validation: ✓ when {q_mark} ≤ {t} (the header threshold), ✕ when not, D for a "
        "decoy" + (", E for an entrapment spike-in" if spike else "") + f". {label}."
    )


def column_defs(
    table: str,
    columns: Sequence[str],
    labels: Mapping[str, str],
    *,
    role: str = "top",
    experiment: bool,
    q_active: str | None,
    threshold: float | None,
    winner_matters: bool,
    scales: Scales | None = None,
    rescorer: str | None = None,
    sort: tuple[str, bool] | None = None,
    marks_at: float | None = None,
) -> list[dict[str, Any]]:
    """AG Grid column definitions of one grid.

    The validation mark first (pinned), then the key column, the default columns, every
    other column hidden (the Columns menu shows them), and the open icon pinned on the
    right. ``q_active`` is the table's filter column (the first table) or None (a child
    table, no q filter); ``threshold`` is the table's q filter (None for a child table)
    and ``marks_at`` the header threshold the marks test (for the tooltip). A column's
    ``context.hideOrder`` says when it goes if the grid's columns do not fit (browser.js).
    """
    scales = scales or Scales()
    top = role == "top"
    q_mark = validation_column(table, role, experiment, q_active)
    present = set(columns)
    shown = default_columns(
        table,
        role,
        columns,
        experiment=experiment,
        q_column=q_mark,
        winner_matters=winner_matters,
    )
    key = "protein_group" if table == "protein_group" else "peptidoform"
    spike = "is_entrapment" in present
    at = marks_at if marks_at is not None else threshold
    q_words = labels.get(q_mark) or q_mark
    params: dict[str, Any] = {
        "qField": q_mark,
        "labelField": "label",
        # The header tooltip with "{t}" for the threshold: the browser fills it in again
        # when the header threshold changes.
        "tipTemplate": mark_tip(q_mark, None, q_words, spike),
    }
    if spike:
        params["spikeField"] = "is_entrapment"
    if "is_transferred" in present:
        params["transferField"] = "is_transferred"
    defs: list[dict[str, Any]] = [
        {
            "colId": "_valid",
            "headerName": "",
            "field": q_mark,
            "cellRenderer": "IbValid",
            "cellRendererParams": params,
            "pinned": "left",
            "lockPosition": "left",
            "width": 40,
            "minWidth": 40,
            "maxWidth": 40,
            "resizable": False,
            "suppressMovable": True,
            "sortable": not top,
            "sortingOrder": ["asc", "desc"],
            "headerClass": "ib-head-valid",
            "cellClass": "ib-cell-valid",
            "headerTooltip": mark_tip(q_mark, at, q_words, spike),
        }
    ]
    if table == "precursor":
        defs.append(
            {
                "colId": "_note",
                "headerName": "",
                "field": "_note",
                "cellRenderer": "MvNote",
                "pinned": "left",
                "lockPosition": "left",
                "width": 34,
                "minWidth": 34,
                "maxWidth": 34,
                "resizable": False,
                "suppressMovable": True,
                "sortable": not top,
                "headerClass": "ib-head-note",
                "headerTooltip": "Your verdict (validation notes): A accepted, R rejected, "
                "U unsure. Give one on the precursor page (keys A, R, U).",
            }
        )
    bar_width = BAR_WIDTH if top else CHILD_BAR_WIDTH
    rest = [c for c in columns if c not in shown]
    for column in [*shown, *rest]:
        kind = "species" if column == "_species" else KINDS.get(column, "text")
        anyq = kind == "count" and any_q(column, threshold)
        width = COLUMN_WIDTHS.get(column, WIDTHS.get(kind, 84 if kind == "species" else 130))
        if anyq:
            width += ANY_Q_EXTRA
        if not top and (kind in BAR_CLASS or kind == "quantity"):
            width -= BAR_WIDTH - CHILD_BAR_WIDTH
        d: dict[str, Any] = {
            "field": "protein_group" if column == "_species" else column,
            "colId": column,
            "headerName": "best score"
            if column == "score" and table == "protein_group"
            else HEADERS.get(column, column),
            "headerTooltip": header_tip(
                column,
                labels,
                table=table,
                q_active=q_active,
                threshold=threshold,
                rescorer=rescorer,
                scales=scales,
            ),
            "cellRenderer": "IbSpecies" if kind == "species" else RENDERERS.get(kind, "IbText"),
            "initialWidth": width,
            "minWidth": 48,
        }
        if kind == "charge":
            d["minWidth"] = 40
        if kind in NUMERIC:
            d["type"] = "rightAligned"
        classes: list[str] = []
        head: list[str] = []
        bar = bar_params(column, table, scales, anyq=anyq, width=bar_width)
        if bar is not None:
            d["cellRendererParams"] = {**bar[0], "tip": bar[1]}
            classes += ["ib-cell-bar", BAR_CLASS.get(kind, "ib-bar-quantity")]
        elif kind in ("q", "score", "count", "quantity"):
            # No known scale: the value without a bar.
            d["cellRenderer"] = "IbNumber"
            d["cellRendererParams"] = {"kind": "q" if kind == "q" else "num"}
        elif kind in ("rt", "num", "int", "id", "plainq"):
            d["cellRendererParams"] = {"kind": "q" if kind == "plainq" else kind}
        elif kind in ("transfer", "spike"):
            d["cellRendererParams"] = (
                {"text": "transfer", "colour": "lime"}
                if kind == "transfer"
                else {"text": "spike-in", "colour": "grape"}
            )
        if anyq:
            head.append("ib-head-anyq")
        if kind == "q":
            d["sortingOrder"] = ["asc", "desc"]
        if kind == "species":
            d["sortable"] = False
            head.append("ib-head-derived")
        if kind in ("text", "mono"):
            d["tooltipField"] = column
        elif kind == "group":
            # The members of a group of several, one per line.
            d["tooltipValueGetter"] = {"function": "mvbGroupTip(params)"}
        if kind == "quantity":
            d["tooltipValueGetter"] = {"function": "mvbQuantityTip(params)"}
        order = hide_order(table, role, column)
        if order is not None and column in shown:
            d["context"] = {"hideOrder": order}
        if column == key and table == "protein_group":
            # The group takes the width the other columns leave (the table is wide).
            d["hide"] = False
            d["lockVisible"] = True
            d["lockPosition"] = "left"
            d["flex"] = 1
            d["minWidth"] = 220
            classes.append("ib-cell-key")
        elif column == key and top:
            d["pinned"] = "left"
            d["lockPosition"] = "left"
            d["lockVisible"] = True
            d["hide"] = False
            classes.append("ib-cell-key")
            d["initialWidth"] = 210
        elif column == key:
            # A child panel is half as wide as the first table: the peptidoform takes
            # the width the other columns leave. In the peptides panel it is the row's
            # identity, so the fit (browser.js) keeps it readable (fitWidth); the
            # precursors of one peptide share its sequence (the panel's title).
            d["lockPosition"] = "left"
            d["lockVisible"] = True
            d["hide"] = False
            d["flex"] = 1
            d["minWidth"] = 110
            if table == "peptide":
                d["context"] = {"fitWidth": 160}
            classes.append("ib-cell-key")
        elif top and column == q_active:
            d["hide"] = False
            head.append("ib-head-active")
            classes.append("ib-cell-active")
        else:
            d["initialHide"] = column not in shown
        if classes:
            d["cellClass"] = " ".join(classes)
        if head:
            d["headerClass"] = " ".join(head)
        if column == "protein" and top:
            # It takes the width that the other columns leave.
            d["flex"] = 1
            d["minWidth"] = 120
        if sort is not None and column == sort[0]:
            d["initialSort"] = "desc" if sort[1] else "asc"
        defs.append(d)
    defs.append(
        {
            "colId": "_open",
            "headerName": "",
            "field": "_href",
            "cellRenderer": "IbOpen",
            "pinned": "right",
            "width": 40,
            "minWidth": 40,
            "maxWidth": 40,
            "sortable": False,
            "resizable": False,
            "suppressMovable": True,
            "lockPosition": "right",
            "cellClass": "ib-cell-open",
            "headerTooltip": "Open the precursor page of the row's precursor (or double-click "
            "the row, or press Enter)",
        }
    )
    return defs


FIXED = ("_valid", "_note", "_open")


def column_options(
    defs: Sequence[Mapping[str, Any]], *, experiment: bool = True
) -> list[dict[str, str]]:
    """The Columns menu: every column but the mark, the open icon and the key column.

    A single run leaves out ``run`` and ``source`` (empty and 0 there).
    """
    out = []
    for d in defs:
        col = d.get("colId")
        if col in FIXED or d.get("lockVisible"):
            continue
        if not experiment and col in ("run", "source"):
            continue
        label = str(d["headerName"])
        if col == "_species":
            label = "species (viewer rule)"
        elif "ib-head-anyq" in str(d.get("headerClass") or ""):
            label += " (any q)"
        out.append({"value": str(col), "label": label})
    return out


def visible_columns(defs: Sequence[Mapping[str, Any]]) -> list[str]:
    """The columns of ``defs`` that are shown when the grid is created (menu columns)."""
    out = []
    for d in defs:
        if d.get("colId") in FIXED or d.get("lockVisible"):
            continue
        hidden = d.get("hide") if "hide" in d else d.get("initialHide", False)
        if not hidden:
            out.append(str(d["colId"]))
    return out


# --------------------------------------------------------------------------- rows


def _plain(value: Any) -> Any:
    """A JSON-ready value: numpy scalars as Python values, NaN and NA as None."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def row_key(unit: str, row: Mapping[str, Any]) -> str:
    """The grid row id of a table row.

    Protein groups: the group; peptides: the ``base_peptide_id``; precursors:
    ``run:candidate_id`` (the run name, empty in a single run), which is also what the
    selection holds, so the browser finds a selected row by its id.
    """
    if unit == "protein_group":
        return str(row.get("protein_group"))
    if unit == "peptide":
        return str(row.get("base_peptide_id"))
    return f"{row.get('run') or ''}:{row.get('candidate_id')}"


def records(
    df: pd.DataFrame,
    base: str,
    unit: str = "precursor",
    columns: Sequence[str] | None = None,
    notes: Mapping[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Rows of a table page for a grid, with the row id and the precursor page address.

    ``columns`` keeps only those columns (a child grid's, see :func:`child_columns`).
    ``notes`` (``"run:candidate_id"`` to the user's verdict and comment) puts the
    verdict on precursor rows (``_note``); a protein or peptide row would show the
    verdict of its representative precursor, so it gets none.
    """
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    out: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        row = {k: _plain(v) for k, v in record.items()}
        if row.get("run") is None and "run" in row:
            row["run"] = ""
        cid = row.get("candidate_id")
        row["_key"] = row_key(unit, row)
        row["_href"] = (
            href(base, "precursor", {"run": row.get("run") or "", "cid": int(cid)})
            if cid is not None
            else None
        )
        if notes and unit == "precursor" and cid is not None:
            note = notes.get(f"{row.get('run') or ''}:{int(cid)}")
            if note:
                row["_note"], row["_note_text"] = note
        out.append(row)
    return out


# The columns a child grid carries: what it shows, what its Columns menu offers, and
# the keys and flags of the page. The protein strings (long, and the same on most rows
# of one parent) and the run index are left out.
CHILD_KEEP = (
    "peptidoform",
    "charge",
    "label",
    "run",
    "candidate_id",
    "base_peptide_id",
    "score",
    *Q_COLUMNS,
    "n_precursors",
    "n_runs",
    "n_runs_transfer_only",
    "apex_rt",
    "selected_peak_rank",
    "quantity",
    "quant_state",
    "is_winner",
    "is_transferred",
    "transfer_q",
    "is_entrapment",
)


def child_columns(columns: Sequence[str]) -> list[str]:
    """The columns of a child table that its grid carries, in the table's order."""
    keep = set(CHILD_KEEP)
    return [c for c in columns if c in keep]


_CHILD_META: dict[tuple[Any, ...], tuple[list[str], dict[str, str]]] = {}


def child_meta(rs: ResultSet, unit: str) -> tuple[list[str], dict[str, str]]:
    """The columns and labels of a child table (no q filter), the same for every parent.

    Read once per result set from a query that matches no parent (cached), so a page
    can build its child grids before it knows their rows.
    """
    key = (*_facet_key(rs), unit)
    with _FACET_LOCK:
        hit = _CHILD_META.get(key)
    if hit is not None:
        return hit
    none: dict[str, Any] = (
        {"protein_group": "\x00"} if unit == "peptide" else {"base_peptide_id": -1}
    )
    out = safe_columns(rs, unit, threshold=None, **none)
    with _FACET_LOCK:
        _CHILD_META[key] = out
    return out


def winner_matters(unit: str, q_column: str | None) -> bool:
    """Whether a grouped first table can show rows that are not their group's winner.

    With the unit's own q column (and a threshold below 1) every row shown is a winner;
    with another q column the row shown can be a loser (``is_winner`` False).
    """
    return unit in UNIT_Q_COLUMN and unit != "precursor" and q_column != UNIT_Q_COLUMN[unit]


def safe_columns(rs: ResultSet, unit: str, **kwargs: Any) -> tuple[list[str], dict[str, str]]:
    """The columns and labels of a unit's table, for a page whose query failed.

    The column set depends on the unit and the result set only, so this table has the
    same columns as the refused one. ``kwargs`` are TableQuery fields (for example
    ``threshold=None`` for the labels of a child table).
    """
    try:
        page = identification_table(rs, TableQuery(unit=unit, limit=0, **kwargs))  # type: ignore[arg-type]
    except (ViewerError, ValueError):
        base = ["peptidoform", "charge", "label", "protein", "score", *Q_COLUMNS]
        if unit == "protein_group":
            base = ["protein_group", *base]
        return base, {}
    return list(page.rows.columns), dict(page.column_labels)
