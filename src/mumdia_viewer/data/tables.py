"""Identification tables: precursors, peptides and protein groups, filtered, sorted and paged.

Row units:

* **precursor**: one row per pooled scored row, that is one candidate in one run
  (``(source, candidate_id)``). An experiment shows the rows of every run with the run
  name and ``source``. Quant columns come from each run's ``peptide_quant``, joined on
  ``(source, candidate_id)``.
* **peptide** (one row per ``base_peptide_id``) and **protein_group** (one row per
  ``protein_group``). The filters (label, q column and threshold, charge, protein,
  modification, search, quant status) select scored rows, and a key is in the table
  when at least one of its rows passes them. The total is therefore
  ``COUNT(DISTINCT key)`` over the passing rows, the same count as the overview's. The
  row shown for a key is its winning row when that row passes the filters; otherwise it
  is the best passing row, and ``is_winner`` is False. The winning row is the engine's
  (``grouped_q`` in rescore.rs): the only row of the group whose grouped q is below 1.0,
  or, for a winner at the cap of 1.0, the row that the rule picks (highest ``score``, a
  decoy first on an exact score tie, then file order). Passing rows are ranked by the
  same order. In entrapment mode decoys do not compete and are not shown, and an
  entrapment row wins a tie. With the unit's own q column and a threshold below 1 only
  winning rows pass, so every row shown is a winner.

The grouped q columns (``precursor_q``, ``peptide_q_value``, ``pg_q_value``) are set on
the winning row of each group only (1.0 elsewhere) and are experiment-wide in an
experiment, so a table is never filtered by run on them.

All filtering, sorting and paging run in DuckDB over the parquet files with a column
projection. For large tables the result of a query (without its order and page) is
materialised once in a table of the in-memory DuckDB database: the whole result when it
has at most :data:`MAX_MATERIALISED_ROWS` rows, otherwise its order only (position and
row number, one table per sort), and the rows of a page are then read back by row
number. Every cached table holds all that its pages need, so evicting one table never
breaks another. These are ordinary tables, not TEMP tables: a TEMP table is private to
the cursor that made it, and every thread has its own cursor.
"""

from __future__ import annotations

import contextlib
import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from .discovery import ResultSet, Run
from .duck import sql_ident, sql_path
from .errors import ViewerError
from .quant import (
    STRIP_SQL,
    QuantGate,
    entrapment_condition,
    mbr_ran,
    quant_gate,
    quant_relation,
    quant_table_problems,
    resolve_run,
    run_entrapment_condition,
    table_problem,
    transfer_relation,
)
from .rescore import rescore_info

Unit = Literal["precursor", "peptide", "protein_group"]

# The seven q columns of psms_scored v4, in file order.
Q_COLUMNS: tuple[str, ...] = (
    "q_value",
    "peptide_q_value",
    "pg_q_value",
    "global_q_value",
    "run_psm_q",
    "experiment_psm_q",
    "precursor_q",
)
UNIT_Q_COLUMN: dict[str, str] = {
    "precursor": "precursor_q",
    "peptide": "peptide_q_value",
    "protein_group": "pg_q_value",
}
GROUPED_Q: dict[str, str] = {
    "precursor_q": "(peptidoform, charge)",
    "peptide_q_value": "base_peptide_id",
    "pg_q_value": "protein_group",
}
# The key and the name of the groups of the peptide and protein-group tables.
GROUP_KEY: dict[str, str] = {"peptide": "base_peptide_id", "protein_group": "protein_group"}
GROUP_NOUN: dict[str, str] = {"peptide": "base peptide", "protein_group": "protein group"}
QUANT_STATES = ("quantified", "not_quantifiable", "not_selected")

# Tables whose scored table has at least this many rows are materialised per query.
MATERIALISE_MIN_ROWS = 200_000
# A result with at most this many rows is materialised whole; a larger one only as its
# order (position and row number per sort), and each page's rows are read by row number.
MAX_MATERIALISED_ROWS = 250_000
# Budget of the per-result-set table cache (tables, and rows x columns). A result whose
# order table alone would exceed the cell budget is not cached: every page then runs the
# whole query again.
MAX_CACHED_TABLES = 12
MAX_CACHED_CELLS = 40_000_000
MAX_LIMIT = 10_000


@dataclass(frozen=True)
class TableQuery:
    """One request for an identification table.

    ``q_column=None`` selects the unit's own column (``precursor_q``, ``peptide_q_value``
    or ``pg_q_value``); in an experiment a precursor table filtered by ``run`` uses
    ``run_psm_q``. ``threshold=None`` applies no q filter. ``protein`` is a
    case-insensitive substring of the protein string; ``modification`` a case-sensitive
    substring of the peptidoform (for example ``Oxidation``); ``search`` a
    case-insensitive substring of the peptidoform or the protein. ``quant_status`` is a
    quant state (``quantified``, ``not_quantifiable``, ``not_selected``) or a raw
    ``quant_status`` string. ``run`` (experiments) is a run name or ``source`` index.
    In the peptide and protein-group tables every filter selects scored rows, and a
    key is shown when one of its rows passes (see the module docstring).
    """

    unit: Unit = "precursor"
    q_column: str | None = None
    threshold: float | None = 0.01
    include_decoys: bool = False
    charge: int | None = None
    protein: str | None = None
    modification: str | None = None
    quant_status: str | None = None
    search: str | None = None
    run: str | int | None = None
    sort_by: str = "score"
    descending: bool = True
    offset: int = 0
    limit: int = 50

    def signature(self) -> tuple[Any, ...]:
        """The fields that define the filtered row set (not its order or page)."""
        skip = {"sort_by", "descending", "offset", "limit"}
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in skip)


@dataclass
class TablePage:
    """One page of an identification table.

    ``total`` is the number of rows that pass the filters. ``description`` states the
    row unit, the q column and the filters in words. ``column_labels`` says how each
    derived column was computed.
    """

    rows: pd.DataFrame
    total: int
    query: TableQuery
    description: str
    column_labels: dict[str, str] = field(default_factory=dict)
    q_column: str | None = None


# --------------------------------------------------------------------------- SQL helpers


class SqlFragment:
    """SQL text with its bound positional parameters, kept in textual order.

    ``text`` is the SQL; ``params`` are the values of its ``?`` placeholders in order.
    """

    def __init__(self, text: str = "", *params: Any) -> None:
        self.parts: list[str] = [text] if text else []
        self.params: list[Any] = list(params)

    def add(self, text: str, *params: Any) -> SqlFragment:
        self.parts.append(text)
        self.params.extend(params)
        return self

    def extend(self, other: SqlFragment) -> SqlFragment:
        self.parts.append(other.text)
        self.params.extend(other.params)
        return self

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _q_text(column: str, experiment: bool, entrapment: bool) -> str:
    """What a q column estimates, in words."""
    if column == "q_value":
        text = (
            "q_value is the PSM-level q pooled over all runs (not a per-run FDR)"
            if experiment
            else "q_value is the PSM-level q"
        )
    elif column in ("global_q_value", "experiment_psm_q"):
        text = f"{column} is a copy of the pooled PSM-level q_value"
    elif column == "run_psm_q":
        text = "run_psm_q is the PSM-level q within each run"
    elif column == "precursor_q":
        text = "precursor_q is the precursor-level q per (peptidoform, charge)"
        if experiment:
            text += (
                "; it is experiment-wide and set on the winning row of each precursor over "
                "all runs (1.0 on the other rows), so each accepted precursor appears once, "
                "in the run of its winning row"
            )
        else:
            text += "; in a single run it equals q_value on every target row"
    elif column == "peptide_q_value":
        text = (
            "peptide_q_value is the picked target-decoy q per base_peptide_id, set on the "
            "winning row of each peptide (1.0 on the other rows)"
        )
        if experiment:
            text += "; it is experiment-wide"
    elif column == "pg_q_value":
        text = "pg_q_value is the q per protein group, set on the winning row (1.0 elsewhere)"
        if experiment:
            text += "; it is experiment-wide"
    else:
        text = f"{column} is not a known q column"
    if entrapment:
        text += " (entrapment-mode estimate: the rescorer ran in entrapment mode)"
    return text


def _fmt_t(value: float) -> str:
    return f"{value:g}"


# --------------------------------------------------------------------------- table cache


@dataclass
class _Entry:
    name: str
    rows: int
    cells: int


class _TableCache:
    """Tables materialised in the result set's DuckDB database, least recently used first.

    Every entry is self-contained: a page reads one entry and nothing else, so an
    eviction can never remove a table that another entry needs.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entries: OrderedDict[Any, _Entry] = OrderedDict()

    def get(self, key: Any) -> _Entry | None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
            return entry

    def create(self, rs: ResultSet, key: Any, query: SqlFragment, n_columns: int) -> _Entry:
        """Materialise ``query`` under ``key``, then evict older entries over the budget.

        The new entry is never evicted by its own creation, so the page that asked for
        it can read it; it is the first to go when a later query needs the room.
        """
        with self.lock:
            hit = self.get(key)
            if hit is not None:
                return hit
            name = "mumdia_viewer_" + uuid.uuid4().hex
            rs.duck.execute(f"CREATE TABLE {name} AS {query.text}", query.params)
            rows = int(rs.duck.scalar(f"SELECT count(*) FROM {name}") or 0)
            entry = _Entry(name, rows, rows * max(1, n_columns))
            self.entries[key] = entry
            self._evict(rs, keep=key)
            return entry

    def _evict(self, rs: ResultSet, keep: Any) -> None:
        cells = sum(e.cells for e in self.entries.values())
        while len(self.entries) > 1 and (
            len(self.entries) > MAX_CACHED_TABLES or cells > MAX_CACHED_CELLS
        ):
            key, entry = next(iter(self.entries.items()))
            if key == keep:
                break
            self.entries.pop(key)
            cells -= entry.cells
            rs.duck.execute(f"DROP TABLE IF EXISTS {entry.name}")

    def clear(self, rs: ResultSet) -> None:
        with self.lock:
            for entry in self.entries.values():
                rs.duck.execute(f"DROP TABLE IF EXISTS {entry.name}")
            self.entries.clear()


def _cache(rs: ResultSet) -> _TableCache:
    cache = rs._memo.get(("tables", "cache"))
    if cache is None:
        cache = rs._memo.setdefault(("tables", "cache"), _TableCache())
    return cache


def drop_table_cache(rs: ResultSet) -> None:
    """Drop every table this module materialised for ``rs`` (frees DuckDB memory)."""
    _cache(rs).clear(rs)
    for key in [
        k
        for k in rs._memo
        if isinstance(k, tuple) and k[:2] in (("tables", "total"), ("tables", "flags"))
    ]:
        rs._memo.pop(key, None)


# --------------------------------------------------------------------------- context


@dataclass
class _Context:
    rs: ResultSet
    query: TableQuery
    q_column: str | None
    scored_path: str
    scored_names: set[str]
    scored_rows: int
    experiment: bool
    entrapment: bool
    mbr: bool
    run: Run | None
    large: bool
    stamps: tuple[Any, ...]
    notes: list[str] = field(default_factory=list)

    @property
    def unit(self) -> str:
        return self.query.unit

    @property
    def grouped(self) -> bool:
        return self.query.unit in GROUP_KEY

    @property
    def fast(self) -> bool:
        """True when only winning rows can pass: the unit's own column below 1.

        The engine writes a grouped q below 1.0 to the winning row of a group only, so
        each key then has at most one passing row and no ranking is needed.
        """
        t = self.query.threshold
        return (
            self.grouped
            and self.q_column == UNIT_Q_COLUMN[self.unit]
            and t is not None
            and float(t) < 1.0
        )

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)


def _is_int(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _stamps(rs: ResultSet) -> tuple[Any, ...]:
    """(size, mtime) of every file a table reads, so a rewritten file is never served stale."""
    out: list[Any] = [rs.scored.parquet().stamp]
    for run in rs.runs:
        for kind in ("peptide_quant", "protein_group_quant"):
            art = run.artifact(kind)
            if art is not None and art.usable:
                out.append((run.index, kind, art.parquet().stamp))
    tr = rs.artifact("mbr_transferred")
    if tr is not None and tr.usable:
        out.append(("mbr", tr.parquet().stamp))
    return tuple(out)


def _validate(rs: ResultSet, query: TableQuery) -> _Context:
    if query.unit not in UNIT_Q_COLUMN:
        raise ViewerError(
            f"unknown table unit {query.unit!r}; use precursor, peptide or protein_group."
        )
    handle = rs.scored.parquet()
    names = set(handle.schema.names)
    present_q = [c for c in Q_COLUMNS if c in names]
    run: Run | None = None
    if query.run is not None and query.run != "":
        if not rs.is_experiment and query.run not in (0, "run"):
            raise ViewerError("a single run has no run filter; leave run unset.")
        if rs.is_experiment:
            if query.unit != "precursor":
                raise ViewerError(
                    f"the {query.unit} table of an experiment is experiment-wide: each row is "
                    "a row of its group over all runs and its q column is experiment-wide. "
                    "For per-run rows use the precursor table with a run (run_psm_q); for "
                    "per-run presence use the n_runs column."
                )
            run = resolve_run(rs, query.run)
    q_column = query.q_column
    if q_column is None:
        q_column = "run_psm_q" if run is not None else UNIT_Q_COLUMN[query.unit]
    if q_column not in present_q:
        raise ViewerError(
            f"q column {q_column!r} is not in the scored table; available: {', '.join(present_q)}."
        )
    if run is not None and q_column in GROUPED_Q:
        raise ViewerError(
            f"{q_column} is experiment-wide and set on one winning row per group, so a count "
            f"of one run on it counts the experiment's winners that happen to sit in that "
            "run, not the run's identifications. Use run_psm_q for a per-run table."
        )
    if query.threshold is not None:
        t = float(query.threshold)
        if math.isnan(t):
            raise ViewerError("the q threshold is NaN; use a number or None.")
    if not _is_int(query.offset) or query.offset < 0:
        raise ViewerError("offset must be a non-negative integer.")
    if not _is_int(query.limit) or not 0 <= query.limit <= MAX_LIMIT:
        raise ViewerError(f"limit must be an integer from 0 to {MAX_LIMIT}.")
    if query.charge is not None and not _is_int(query.charge):
        raise ViewerError("charge must be an integer.")
    info = rescore_info(rs)
    return _Context(
        rs=rs,
        query=query,
        q_column=q_column,
        scored_path=sql_path(rs.scored.require()),
        scored_names=names,
        scored_rows=handle.num_rows,
        experiment=rs.is_experiment,
        entrapment=info.mode == "entrapment",
        mbr=mbr_ran(rs),
        run=run,
        large=handle.num_rows >= MATERIALISE_MIN_ROWS,
        stamps=_stamps(rs),
    )


# --------------------------------------------------------------------------- entrapment


def entrapment_sql(
    marker: str | None,
    exclude: str | None = None,
    contaminants: list[str] | tuple[str, ...] = (),
    alias: str = "w",
) -> SqlFragment:
    """True on entrapment (spike-in) rows, by the engine's rule (rescore.rs).

    A non-decoy row is entrapment when its protein string contains ``marker``, does not
    contain ``exclude`` and contains none of the ``contaminants`` (case-sensitive
    substrings). Without a marker no row is entrapment.
    """
    text, params = entrapment_condition(marker, exclude, tuple(contaminants or ()), alias)
    return SqlFragment(text, *params)


def _entrapment_expr(rs: ResultSet, alias: str) -> SqlFragment:
    text, params = run_entrapment_condition(rs, alias)
    return SqlFragment(text, *params)


def winner_sql(
    key: str, path: str, *, entrapment: bool, is_entrapment: SqlFragment | None = None
) -> SqlFragment:
    """SQL selecting the ``file_row_number`` (as ``rn``) of the winning row of each group.

    ``key`` is a column of the alias ``w`` (for example ``w.base_peptide_id``) and
    ``path`` the scored parquet file. The engine rule (``grouped_q`` in rescore.rs):
    highest ``score``; on an exact score tie a decoy replaces a target; otherwise the
    earlier row in file order stays. In entrapment mode decoys are skipped and an
    entrapment row (``is_entrapment``) replaces another row on a tie.
    """
    s = SqlFragment(
        "SELECT file_row_number AS rn FROM (SELECT file_row_number, row_number() OVER ("
    )
    s.add(f"PARTITION BY {key} ORDER BY score DESC, ")
    if entrapment:
        tie = is_entrapment if is_entrapment is not None else SqlFragment("false")
        s.add("(").extend(tie).add(") DESC, ")
    else:
        s.add("(label = 'decoy') DESC, ")
    s.add("file_row_number) AS rk FROM read_parquet(?, file_row_number = true) w", path)
    if entrapment:
        s.add(" WHERE label <> 'decoy'")
    s.add(") WHERE rk = 1")
    return s


def _entrapment_label(ctx: _Context) -> str | None:
    """The is_entrapment column label in entrapment mode, or None when rows cannot be flagged."""
    if not ctx.entrapment:
        return None
    marker = ctx.rs.config_get("rescore", "entrapment_marker")
    if not marker:
        ctx.note(
            "Entrapment mode, but no rescore.entrapment_marker is recorded; spike-in rows "
            "cannot be flagged."
        )
        return None
    exclude = ctx.rs.config_get("rescore", "entrapment_exclude")
    tokens = ctx.rs.config_get("rescore", "entrapment_contaminant_markers", default=[]) or []
    label = f"entrapment spike-in by the engine rule: a target whose protein contains {marker!r}"
    if exclude:
        label += f", not {exclude!r}"
    if tokens:
        label += ", and none of " + ", ".join(repr(t) for t in tokens)
    return label


# --------------------------------------------------------------------------- quant gate


def _gates(ctx: _Context) -> list[tuple[Run, QuantGate]]:
    """The quant gate of each run whose quant columns the table shows."""
    if ctx.run is not None:
        runs = [ctx.run]
    elif ctx.experiment:
        runs = list(ctx.rs.runs)
    else:
        runs = ctx.rs.runs[:1]
    kinds = ["peptide_quant"]
    if ctx.unit == "protein_group":
        kinds.append("protein_group_quant")
    out: list[tuple[Run, QuantGate]] = []
    for run in runs:
        if all(table_problem(ctx.rs, run, kind) is None for kind in kinds):
            out.append((run, quant_gate(ctx.rs, run)))
    return out


def _gate_sentence(ctx: _Context) -> str | None:
    """How the quant columns were gated, in one or two sentences (T2 R10)."""
    gates = _gates(ctx)
    if not gates:
        return None
    distinct = {g.key: g for _, g in gates}
    if len(distinct) == 1:
        gate = next(iter(distinct.values()))
        if ctx.unit == "protein_group":
            where = "protein_group_quant rolls up the peptide_quant rows, which hold"
        elif ctx.experiment:
            where = "each run's peptide_quant holds"
        else:
            where = "peptide_quant holds"
        text = f"Quant columns: {where} {gate.describe(ctx.experiment)}."
        columns = {gate.q_column}
    else:
        parts = [f"run {run.label}: {gate.describe(ctx.experiment)}" for run, gate in gates]
        text = "Quant columns: peptide_quant holds, per run, " + "; ".join(parts) + "."
        columns = {g.q_column for g in distinct.values()}
    if ctx.q_column is not None and None not in columns and ctx.q_column not in columns:
        gate_cols = ", ".join(sorted(str(c) for c in columns))
        text += (
            f" The q filter of this table ({ctx.q_column}) is not the quant gate column "
            f"({gate_cols}), so a row that passes the filter can be not_selected."
        )
    return text


def _gate_label(ctx: _Context) -> str:
    """The quant gate in a few words, for the quant_state column label."""
    gates = _gates(ctx)
    distinct = {g.key: g for _, g in gates}
    if len(distinct) == 1:
        return f"outside the quant gate, {next(iter(distinct.values())).label}"
    return "outside its run's quant gate (see the description)"


# --------------------------------------------------------------------------- relations


def _runs_join(ctx: _Context, alias: str) -> SqlFragment:
    """LEFT JOIN of the run names on ``source``."""
    values = ", ".join("(?::UINTEGER, ?::VARCHAR)" for _ in ctx.rs.runs)
    params: list[Any] = []
    for r in ctx.rs.runs:
        params += [int(r.index), r.name]
    return SqlFragment(
        f" LEFT JOIN (SELECT * FROM (VALUES {values}) v(source, run)) r "
        f"ON r.source = {alias}.source",
        *params,
    )


def _peptide_quant_join(ctx: _Context) -> tuple[SqlFragment, list[tuple[str, str]]]:
    """LEFT JOIN of every run's peptide_quant on (source, candidate_id), and its items."""
    union = quant_relation(ctx.rs, "peptide_quant", ("candidate_id", "quantity", "quant_status"))
    for run, problem in quant_table_problems(ctx.rs, "peptide_quant"):
        scope = f"of run {run.label} are" if ctx.experiment else "are"
        ctx.note(f"The quant columns {scope} empty: {problem}")
    if union is None:
        items = [
            ("NULL::DOUBLE", "quantity"),
            ("NULL::VARCHAR", "quant_status"),
            ("NULL::VARCHAR", "quant_state"),
        ]
        return SqlFragment(), items
    u_sql, u_params, sources = union
    join = SqlFragment(
        " LEFT JOIN (SELECT source, candidate_id, any_value(quantity) AS quantity, "
        f"any_value(quant_status) AS quant_status FROM ({u_sql}) GROUP BY 1, 2) q "
        "ON q.source = s.source AND q.candidate_id = s.candidate_id",
        *u_params,
    )
    listed = ", ".join(str(s) for s in sources)
    state = (
        f"CASE WHEN s.source NOT IN ({listed}) THEN NULL "
        "WHEN q.candidate_id IS NULL THEN 'not_selected' "
        "WHEN q.quantity IS NULL OR isnan(q.quantity) THEN 'not_quantifiable' "
        "ELSE 'quantified' END"
    )
    return join, [
        ("q.quantity", "quantity"),
        ("q.quant_status", "quant_status"),
        (state, "quant_state"),
    ]


def _protein_quant_join(ctx: _Context) -> tuple[SqlFragment, list[tuple[str, str]]]:
    """LEFT JOIN of the single run's protein_group_quant on protein_group, and its items."""
    run = ctx.rs.runs[0]
    art = run.artifact("protein_group_quant")
    problem = table_problem(ctx.rs, run, "protein_group_quant")
    if problem is not None or art is None:
        ctx.note(f"The protein table has no quant columns: {problem}")
        return SqlFragment(), []
    art.parquet()
    join = SqlFragment(
        " LEFT JOIN (SELECT protein_group, any_value(quantity) AS quantity, "
        "any_value(quant_status) AS quant_status, any_value(n_peptides) AS "
        "quant_n_peptides FROM read_parquet(?) GROUP BY 1) g "
        "ON g.protein_group = s.protein_group",
        sql_path(art.require()),
    )
    return join, [
        ("g.quantity", "quantity"),
        ("g.quant_status", "quant_status"),
        (
            "CASE WHEN g.protein_group IS NULL THEN 'not_selected' "
            "WHEN g.quantity IS NULL OR isnan(g.quantity) THEN 'not_quantifiable' "
            "ELSE 'quantified' END",
            "quant_state",
        ),
        ("g.quant_n_peptides", "quant_n_peptides"),
    ]


def _transfer_join(ctx: _Context) -> tuple[SqlFragment, list[tuple[str, str]]]:
    rel = transfer_relation(ctx.rs)
    if rel is None:
        ctx.note(
            "Match-between-runs ran, but no transfer table was found (mbr_transferred.parquet "
            "or is_transferred in scored_for_quant); transfers cannot be flagged."
        )
        return SqlFragment(), [
            ("NULL::BOOLEAN", "is_transferred"),
            ("NULL::DOUBLE", "transfer_q"),
        ]
    t_sql, t_params = rel
    join = SqlFragment(
        f" LEFT JOIN ({t_sql}) t ON t.source = s.source AND t.candidate_id = s.candidate_id",
        *t_params,
    )
    return join, [
        ("t.candidate_id IS NOT NULL", "is_transferred"),
        ("t.transfer_q", "transfer_q"),
    ]


def _has_quant(ctx: _Context) -> bool:
    """Whether the unit shows quant columns: precursors always; grouped units in a single run."""
    return ctx.unit == "precursor" or not ctx.experiment


def _row_relation(
    ctx: _Context, restrict: SqlFragment | None = None
) -> tuple[SqlFragment, list[str]]:
    """``x``: scored rows with their row-level joined columns, and the column names.

    ``restrict`` is a condition on the scored alias ``s`` (a page's row numbers).
    """
    names = ctx.scored_names
    items: list[tuple[str, str]] = [
        (f"s.{c}", c)
        for c in (
            "candidate_id",
            "source",
            "peptidoform",
            "charge",
            "label",
            "protein",
            "protein_group",
            "base_peptide_id",
            "score",
            "prelim_score",
        )
    ]
    items += [(f"s.{c}", c) for c in Q_COLUMNS if c in names]
    if "selected_peak_rank" in names:
        items.append(("s.selected_peak_rank", "selected_peak_rank"))
    else:
        items.append(("0::INTEGER", "selected_peak_rank"))
    items += [(f"s.{c}", c) for c in ("apex_rt", "elution_lo", "elution_hi")]
    items.append(("s.file_row_number", "rn"))
    joins = SqlFragment()
    if _has_quant(ctx):
        if ctx.unit == "protein_group":
            q_join, q_items = _protein_quant_join(ctx)
        else:
            q_join, q_items = _peptide_quant_join(ctx)
        joins.extend(q_join)
        items += q_items
    if ctx.mbr and ctx.unit == "precursor":
        t_join, t_items = _transfer_join(ctx)
        joins.extend(t_join)
        items += t_items
    head = SqlFragment("SELECT ")
    if _entrapment_label(ctx) is not None:
        ent = _entrapment_expr(ctx.rs, "s")
        head.add(", ".join(f"{e} AS {sql_ident(n)}" for e, n in items) + ", ")
        head.extend(ent).add(" AS is_entrapment")
        items.append(("", "is_entrapment"))
    else:
        head.add(", ".join(f"{e} AS {sql_ident(n)}" for e, n in items))
    head.add(" FROM read_parquet(?, file_row_number = true) s", ctx.scored_path)
    head.extend(joins)
    if restrict is not None:
        head.add(" WHERE ").extend(restrict)
    return head, [n for _, n in items]


def _filters(ctx: _Context, columns: list[str]) -> tuple[SqlFragment, list[str]]:
    """WHERE conditions on the row relation ``x`` and their descriptions."""
    q = ctx.query
    conds: list[SqlFragment] = []
    words: list[str] = []
    if q.threshold is not None and ctx.q_column is not None:
        conds.append(SqlFragment(f"x.{sql_ident(ctx.q_column)} <= ?", float(q.threshold)))
    if not q.include_decoys:
        conds.append(SqlFragment("x.label = 'target'"))
        words.append("targets only")
    elif ctx.grouped and ctx.entrapment:
        words.append("targets (decoys do not compete in entrapment mode)")
    else:
        words.append("targets and decoys")
    if ctx.grouped and ctx.entrapment:
        conds.append(SqlFragment("x.label <> 'decoy'"))
    if ctx.run is not None:
        conds.append(SqlFragment("x.source = ?", int(ctx.run.index)))
        words.append(f"run {ctx.run.label}")
    elif ctx.experiment and q.unit == "precursor":
        words.append("all runs")
    if q.charge is not None:
        conds.append(SqlFragment("x.charge = ?", int(q.charge)))
        words.append(f"charge {int(q.charge)}")
    protein = _clean_text(q.protein)
    if protein is not None:
        conds.append(SqlFragment("contains(lower(x.protein), lower(?))", protein))
        words.append(f"protein contains {protein!r} (case-insensitive)")
    mod = _clean_text(q.modification)
    if mod is not None:
        conds.append(SqlFragment("contains(x.peptidoform, ?)", mod))
        words.append(f"peptidoform contains {mod!r}")
    search = _clean_text(q.search)
    if search is not None:
        conds.append(
            SqlFragment(
                "(contains(lower(x.peptidoform), lower(?)) "
                "OR contains(lower(x.protein), lower(?)))",
                search,
                search,
            )
        )
        words.append(f"peptidoform or protein contains {search!r} (case-insensitive)")
    status = _clean_text(q.quant_status)
    if status is not None:
        if "quant_state" not in columns:
            raise ViewerError(
                f"the {q.unit} table of "
                + ("an experiment" if ctx.experiment else "this run")
                + " has no quant columns; filter the precursor table by quant status instead."
            )
        if status in QUANT_STATES:
            conds.append(SqlFragment("x.quant_state = ?", status))
            words.append(f"quant state {status}")
        else:
            conds.append(SqlFragment("x.quant_status = ?", status))
            words.append(f"quant_status {status!r}")
    where = SqlFragment()
    for i, cond in enumerate(conds):
        where.add(" WHERE " if i == 0 else " AND ").extend(cond)
    return where, words


def _pick(ctx: _Context) -> SqlFragment:
    """``d``: one passing row per key (grouped units).

    Under :attr:`_Context.fast` every passing row is a winner, one per key. Otherwise the
    passing rows of each key are ranked: the engine's winning row first (its grouped q
    is below 1.0), then highest score, a decoy (entrapment mode: an entrapment row)
    first on an exact tie, then file order.
    """
    if ctx.fast:
        return SqlFragment("SELECT * FROM f")
    key = GROUP_KEY[ctx.unit]
    uq = UNIT_Q_COLUMN[ctx.unit]
    tie = _entrapment_expr(ctx.rs, "f") if ctx.entrapment else SqlFragment("(f.label = 'decoy')")
    return (
        SqlFragment(
            "SELECT * EXCLUDE (_rk) FROM (SELECT f.*, row_number() OVER ("
            f"PARTITION BY f.{key} ORDER BY (f.{uq} < 1) DESC, f.score DESC, "
        )
        .extend(tie)
        .add(" DESC, f.rn) AS _rk FROM f) WHERE _rk = 1")
    )


def _winner_ctes(ctx: _Context) -> tuple[SqlFragment, tuple[str, str]]:
    """CTEs that decide whether each row of ``d`` is its group's winning row, and the item.

    Under :attr:`_Context.fast` every row of ``d`` is a winner. Otherwise a row whose
    grouped q is below 1.0 is the winner (the engine writes the group q to the winning
    row only). A group of ``d`` without such a row has its winner at the cap of 1.0;
    only for those groups (usually none) the engine rule ranks the group's rows.
    """
    if ctx.fast:
        return SqlFragment(), ("true", "is_winner")
    key = GROUP_KEY[ctx.unit]
    uq = UNIT_Q_COLUMN[ctx.unit]
    tie = _entrapment_expr(ctx.rs, "w") if ctx.entrapment else SqlFragment("(w.label = 'decoy')")
    skip = " AND w.label <> 'decoy'" if ctx.entrapment else ""
    sql = SqlFragment(
        f", sub1 AS (SELECT DISTINCT {key} AS gkey FROM read_parquet(?) "
        f"WHERE {uq} < 1 AND {key} IN (SELECT {key} FROM d)), "
        f"cap AS (SELECT DISTINCT {key} AS gkey FROM d WHERE {key} NOT IN "
        "(SELECT gkey FROM sub1 WHERE gkey IS NOT NULL)), "
        "cw AS (SELECT rn FROM (SELECT w.file_row_number AS rn, row_number() OVER ("
        f"PARTITION BY w.{key} ORDER BY w.score DESC, ",
        ctx.scored_path,
    )
    sql.extend(tie).add(
        " DESC, w.file_row_number) AS rk FROM read_parquet(?, file_row_number = true) w"
    )
    sql.params.append(ctx.scored_path)
    sql.add(f" WHERE w.{key} IN (SELECT gkey FROM cap){skip}) WHERE rk = 1)")
    return sql, (f"(d.{uq} < 1 OR d.rn IN (SELECT rn FROM cw))", "is_winner")


def _count_ctes(ctx: _Context) -> tuple[SqlFragment, list[tuple[str, str]]]:
    """CTEs with the per-group counts of the keys of ``d`` (``c``), and their items.

    ``k`` holds every row (both labels, all runs) of the keys in ``d``, so the counts
    cover whole groups, but only the groups shown are read.
    """
    key = GROUP_KEY[ctx.unit]
    t = ctx.query.threshold
    peptide = ctx.unit == "peptide"
    k = SqlFragment(f", k AS (SELECT s.{key} AS gkey, s.label, s.source, ")
    k.add("s.peptidoform, s.charge" if peptide else "s.base_peptide_id, s.peptide_q_value")
    tr_join = SqlFragment()
    if ctx.experiment:
        if t is not None:
            k.add(", (s.run_psm_q <= ?) AS native", float(t))
        else:
            k.add(", true AS native")
        rel = transfer_relation(ctx.rs) if ctx.mbr else None
        if rel is not None:
            k.add(", t.candidate_id IS NOT NULL AS tr")
            tr_join = SqlFragment(
                f" LEFT JOIN ({rel[0]}) t ON t.source = s.source AND t.candidate_id = "
                "s.candidate_id",
                *rel[1],
            )
        else:
            k.add(", false AS tr")
    k.add(" FROM read_parquet(?) s", ctx.scored_path)
    k.extend(tr_join)
    k.add(f" WHERE s.{key} IN (SELECT {key} FROM d))")
    items: list[tuple[str, str]] = []
    if peptide:
        k.add(
            ", p AS (SELECT gkey, label, count(DISTINCT (peptidoform, charge)) AS n_precursors "
            "FROM k GROUP BY 1, 2)"
        )
        items.append(("c.n_precursors", "n_precursors"))
    elif t is not None:
        k.add(
            ", p AS (SELECT gkey, label, count(DISTINCT base_peptide_id) FILTER "
            "(WHERE peptide_q_value <= ?) AS n_peptides FROM k GROUP BY 1, 2)",
            float(t),
        )
        items.append(("c.n_peptides", "n_peptides"))
    else:
        k.add(
            ", p AS (SELECT gkey, label, count(DISTINCT base_peptide_id) AS n_peptides "
            "FROM k GROUP BY 1, 2)"
        )
        items.append(("c.n_peptides", "n_peptides"))
    if ctx.experiment:
        k.add(
            ", rr AS (SELECT gkey, label, source, bool_or(native) AS native, bool_or(tr) AS tr "
            "FROM k GROUP BY 1, 2, 3), r2 AS (SELECT gkey, label, count(*) FILTER (WHERE native) "
            "AS n_runs, count(*) FILTER (WHERE tr AND NOT native) AS n_runs_transfer_only "
            "FROM rr GROUP BY 1, 2), c AS (SELECT p.*, coalesce(r2.n_runs, 0) AS n_runs, "
            "coalesce(r2.n_runs_transfer_only, 0) AS n_runs_transfer_only FROM p "
            "LEFT JOIN r2 ON r2.gkey = p.gkey AND r2.label = p.label)"
        )
        items.append(("c.n_runs", "n_runs"))
        if ctx.mbr:
            items.append(("c.n_runs_transfer_only", "n_runs_transfer_only"))
    else:
        k.add(", c AS (SELECT * FROM p)")
    return k, items


def _group_items(ctx: _Context) -> list[tuple[str, str]]:
    """The per-group items of a grouped table, in display order (counts, then is_winner)."""
    return [*_count_ctes(ctx)[1], _winner_ctes(ctx)[1]]


def _final_items(
    ctx: _Context, alias: str, row_columns: list[str], group_items: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The SELECT items of the table, in display order (``rn`` last)."""
    a = alias
    scored = [
        "candidate_id",
        "run",
        "source",
        "peptidoform",
        "charge",
        "label",
        "protein",
        "protein_group",
        "base_peptide_id",
        "score",
        "prelim_score",
        *[c for c in Q_COLUMNS if c in row_columns],
        "selected_peak_rank",
        "apex_rt",
        "elution_lo",
        "elution_hi",
    ]
    if ctx.unit == "peptide":
        scored.remove("base_peptide_id")
        scored = ["base_peptide_id", "sequence", *scored]
    elif ctx.unit == "protein_group":
        scored.remove("protein_group")
        scored = ["protein_group", *scored]
    items: list[tuple[str, str]] = []
    for name in scored:
        if name == "run":
            items.append(("r.run", "run"))
        elif name == "sequence":
            items.append((STRIP_SQL.format(col=f"{a}.peptidoform"), "sequence"))
        else:
            items.append((f"{a}.{sql_ident(name)}", name))
    items += group_items
    extra = [
        "quantity",
        "quant_status",
        "quant_state",
        "quant_n_peptides",
        "is_transferred",
        "transfer_q",
        "is_entrapment",
    ]
    items += [(f"{a}.{sql_ident(n)}", n) for n in extra if n in row_columns]
    items.append((f"{a}.rn", "rn"))
    return items


def _select(items: list[tuple[str, str]]) -> str:
    return ", ".join(f"{e} AS {sql_ident(n)}" for e, n in items)


@dataclass
class _Plan:
    """The SQL of one table query: select, count, order and page-fetch forms."""

    ctx: _Context
    row_columns: list[str]
    columns: list[str]
    words: list[str]
    flags: list[tuple[str, str]]

    @classmethod
    def build(cls, ctx: _Context) -> _Plan:
        _, row_columns = _row_relation(ctx)
        _, words = _filters(ctx, row_columns)
        group_items = _group_items(ctx) if ctx.grouped else []
        items = _final_items(ctx, "a", row_columns, group_items)
        columns = [n for _, n in items if n != "rn"]
        flags: list[tuple[str, str]] = []
        if "is_transferred" in columns:
            flags.append(("is_transferred", "is_transferred"))
        if "is_entrapment" in columns:
            flags.append(("is_entrapment", "is_entrapment"))
        if ctx.grouped and not ctx.fast:
            flags.append(("not_winner", "NOT is_winner"))
        return cls(ctx, row_columns, columns, words, flags)

    @property
    def flag_columns(self) -> list[str]:
        """The columns that the flag conditions read."""
        return ["is_winner" if name == "not_winner" else name for name, _ in self.flags]

    def _prefix(self) -> SqlFragment:
        x, _ = _row_relation(self.ctx)
        where, _ = _filters(self.ctx, self.row_columns)
        return (
            SqlFragment("WITH x AS (")
            .extend(x)
            .add("), f AS (SELECT * FROM x")
            .extend(where)
            .add(")")
        )

    def _grouped(self, sql: SqlFragment, keep: set[str] | None, page: bool) -> SqlFragment:
        """Append the per-group CTEs and the final SELECT over ``d`` to ``sql``.

        ``keep`` limits the output to these columns (plus ``rn``); the count CTEs are
        then left out unless a kept column needs them. ``page`` joins the page table.
        """
        ctx = self.ctx
        w_sql, w_item = _winner_ctes(ctx)
        sql.extend(w_sql)
        c_sql, c_items = _count_ctes(ctx)
        counts = keep is None or any(n in keep for _, n in c_items)
        if counts:
            sql.extend(c_sql)
        items = _final_items(ctx, "d", self.row_columns, [*(c_items if counts else []), w_item])
        if keep is not None:
            items = [(e, n) for e, n in items if n in keep or n == "rn"]
        sql.add(f" SELECT {_select(items)}")
        sql.add(" FROM d JOIN page ON page.rn = d.rn" if page else " FROM d")
        if keep is None or "run" in keep:
            sql.extend(_runs_join(ctx, "d"))
        if counts:
            sql.add(f" LEFT JOIN c ON c.gkey = d.{GROUP_KEY[ctx.unit]} AND c.label = d.label")
        if page:
            sql.add(" ORDER BY page.pos")
        return sql

    def select(self, keep: set[str] | None = None) -> SqlFragment:
        """The whole table: every passing row (precursors) or key (grouped units).

        All columns and ``rn`` last, or only the ``keep`` columns and ``rn``.
        """
        ctx = self.ctx
        sql = self._prefix()
        if not ctx.grouped:
            items = _final_items(ctx, "f", self.row_columns, [])
            if keep is not None:
                items = [(e, n) for e, n in items if n in keep or n == "rn"]
            sql.add(f" SELECT {_select(items)} FROM f")
            if keep is None or "run" in keep:
                sql.extend(_runs_join(ctx, "f"))
            return sql
        sql.add(", d AS (").extend(_pick(ctx)).add(")")
        return self._grouped(sql, keep, page=False)

    def order(self, query: TableQuery) -> SqlFragment:
        """Position and row number of every row under the query's sort, plus the flags."""
        keep = {query.sort_by, *_TIEBREAK[query.unit], *self.flag_columns}
        select = self.select(keep)
        flags = "".join(f", {sql_ident(c)}" for c in self.flag_columns)
        return (
            SqlFragment(
                f"SELECT row_number() OVER (ORDER BY {_order_by(query)}) AS pos, rn{flags} FROM ("
            )
            .extend(select)
            .add(") z")
        )

    def count(self) -> SqlFragment:
        """The number of rows of the table, without the per-group columns."""
        sql = self._prefix()
        if self.ctx.grouped and not self.ctx.fast:
            return sql.add(f" SELECT count(DISTINCT {GROUP_KEY[self.ctx.unit]}) FROM f")
        return sql.add(" SELECT count(*) FROM f")

    def fetch(self, order_table: str, start: int, stop: int) -> SqlFragment:
        """The rows at positions ``start + 1 .. stop`` of an order table, in order.

        The row set is known, so no filter or ranking runs; the per-group columns are
        computed for the keys of the page only.
        """
        ctx = self.ctx
        sql = SqlFragment(
            f"WITH page AS (SELECT pos, rn FROM {order_table} WHERE pos > ? AND pos <= ?)",
            int(start),
            int(stop),
        )
        x, _ = _row_relation(ctx, SqlFragment("s.file_row_number IN (SELECT rn FROM page)"))
        sql.add(", x AS (").extend(x).add(")")
        if not ctx.grouped:
            items = _final_items(ctx, "x", self.row_columns, [])
            sql.add(f" SELECT {_select(items)} FROM x JOIN page ON page.rn = x.rn")
            return sql.extend(_runs_join(ctx, "x")).add(" ORDER BY page.pos")
        sql.add(", d AS (SELECT * FROM x)")
        return self._grouped(sql, None, page=True)


# --------------------------------------------------------------------------- labels


_ROW_UNIT = {
    "precursor": ("precursors", "candidate_id"),
    "peptide": ("peptides", "unique base_peptide_id"),
    "protein_group": ("protein groups", "unique protein_group"),
}


def _labels(ctx: _Context, columns: list[str]) -> dict[str, str]:
    """How each derived column was computed."""
    labels: dict[str, str] = {}
    t = ctx.query.threshold
    unit = ctx.unit
    noun = GROUP_NOUN.get(unit, "")
    if "sequence" in columns:
        labels["sequence"] = "stripped sequence of the row shown (display only)"
    if "n_precursors" in columns:
        labels["n_precursors"] = (
            "scored precursors of this peptide (distinct peptidoform, charge; same label; "
            "any q, all runs)"
        )
    if "n_peptides" in columns:
        labels["n_peptides"] = (
            f"peptides of this group (distinct base_peptide_id with peptide_q_value <= {_fmt_t(t)})"
            if t is not None
            else "peptides of this group (distinct base_peptide_id, any q)"
        )
    if "n_runs" in columns:
        where = f"run_psm_q <= {_fmt_t(t)}" if t is not None else "any q"
        what = "peptide" if unit == "peptide" else "group"
        labels["n_runs"] = (
            f"runs with a row of this {what} (same label) with {where} (PSM-level q within "
            "each run)"
        )
        if ctx.mbr:
            labels["n_runs"] += "; native values from scored_combined.parquet"
    if "n_runs_transfer_only" in columns:
        where = f"run_psm_q <= {_fmt_t(t)}" if t is not None else "any q"
        what = "peptide" if unit == "peptide" else "group"
        labels["n_runs_transfer_only"] = (
            f"runs with a match-between-runs transfer of this {what} but no native row "
            f"with {where} (derived from mbr_transferred.parquet); not counted in n_runs"
        )
    if "is_winner" in columns:
        uq = UNIT_Q_COLUMN[unit]
        labels["is_winner"] = (
            f"True when the row shown is the winning row of its {noun}: the only row of the "
            f"group whose {uq} is below 1.0 (a winner at the cap of 1.0 is picked by the "
            "engine rule). False when that row does not pass the filters (for example a "
            f"hidden decoy won); the row shown then holds {uq} 1.0"
        )
    gate_label = _gate_label(ctx) if "quant_state" in columns else ""
    if "quantity" in columns and unit == "precursor":
        labels["quantity"] = (
            "peptide_quant.quantity of this run (top-N fragment area sum, intensity x s); "
            "missing when not quantifiable or not selected, never 0"
        )
        labels["quant_status"] = "peptide_quant.quant_status (engine string)"
        labels["quant_state"] = (
            "quantified; not_quantifiable (a peptide_quant row with a null quantity); "
            f"not_selected (no peptide_quant row: the row is {gate_label})"
        )
    elif "quantity" in columns and unit == "peptide":
        gates = _gates(ctx)
        how = gates[0][1].describe(False) if gates else "the rows of an unrecorded gate"
        labels["quantity"] = (
            f"peptide_quant.quantity of the row shown; peptide_quant holds {how}; missing is "
            "never 0"
        )
        labels["quant_status"] = "peptide_quant.quant_status of the row shown (engine string)"
        labels["quant_state"] = (
            "quant state of the row shown: quantified; not_quantifiable (a peptide_quant row "
            f"with a null quantity); not_selected (no peptide_quant row: {gate_label})"
        )
    elif "quantity" in columns:
        labels["quantity"] = (
            "protein_group_quant.quantity (top-N sum of the per-base-peptide maxima of the "
            "quantified precursors); missing is never 0"
        )
        labels["quant_status"] = "protein_group_quant.quant_status (engine string)"
        labels["quant_state"] = (
            "quantified; not_quantifiable (null quantity); not_selected (no "
            "protein_group_quant row: no peptide of the group passed the quant gate)"
        )
        labels["quant_n_peptides"] = (
            "protein_group_quant.n_peptides: base peptides with a positive quantity"
        )
    if "is_transferred" in columns:
        labels["is_transferred"] = (
            "match-between-runs transfer of this (source, candidate_id), from "
            "mbr_transferred.parquet"
        )
        labels["transfer_q"] = "transfer q of the MBR worker (permuted-RT null); not an FDR at t"
    ent_label = _entrapment_label(ctx) if "is_entrapment" in columns else None
    if ent_label is not None:
        labels["is_entrapment"] = ent_label
    for c in Q_COLUMNS:
        if c in ctx.scored_names:
            labels.setdefault(c, _q_text(c, ctx.experiment, ctx.entrapment))
    return labels


def _describe(ctx: _Context, total: int, flags: dict[str, int], words: list[str]) -> str:
    q = ctx.query
    noun, key = _ROW_UNIT[q.unit]
    if q.unit == "precursor" and ctx.experiment:
        noun, key = "precursor rows", "(source, candidate_id)"
    if q.threshold is not None and ctx.q_column is not None:
        scope = "experiment-wide " if ctx.experiment and ctx.q_column in GROUPED_Q else ""
        cut = f"{scope}{ctx.q_column} <= {_fmt_t(float(q.threshold))}"
    else:
        cut = "no q filter"
    text = f"{total:,} {noun} ({key}, {cut})"
    if words:
        text += ", " + ", ".join(words)
    text += "."
    if q.threshold is not None and ctx.q_column is not None:
        text += " " + _q_text(ctx.q_column, ctx.experiment, ctx.entrapment) + "."
    if ctx.grouped:
        group = GROUP_NOUN[q.unit]
        uq = UNIT_Q_COLUMN[q.unit]
        scope = " over all runs" if ctx.experiment else ""
        tie = (
            "decoys do not compete (entrapment mode); an entrapment row wins an exact tie"
            if ctx.entrapment
            else "a decoy wins an exact score tie"
        )
        if ctx.fast:
            text += (
                f" Each row is the winning row of its {group}{scope}, the only row of the group "
                f"whose {uq} is below 1.0 (the engine picks it by highest score; {tie}); the "
                "other filters apply to that row."
            )
        else:
            text += (
                f" The filters select rows; a {group} is listed when one of its rows{scope} "
                "passes them, and the row shown is the first passing row: the group's winning "
                f"row, then highest score ({tie}), then file order."
            )
            n_other = flags.get("not_winner", 0)
            text += (
                f" On {n_other:,} of the {total:,} rows the group's winning row does not pass "
                f"the filters (is_winner is False), so the row shown holds {uq} 1.0."
                if n_other
                else " Every row shown is its group's winning row."
            )
            if q.threshold is not None and float(q.threshold) >= 1.0 and ctx.q_column in GROUPED_Q:
                text += (
                    f" {ctx.q_column} <= {_fmt_t(float(q.threshold))} keeps every row, also "
                    "the rows that lost their group (they hold 1.0), so this counts every "
                    f"{group} with a passing row, not the accepted ones."
                )
    if ctx.mbr and q.unit == "precursor":
        text += (
            " Match-between-runs ran: the q values are the native (pre-MBR) values of "
            "scored_combined.parquet, so a transfer passes the q filter only on its native q; "
            "is_transferred flags the transferred (source, candidate_id) rows"
        )
        if "is_transferred" in flags:
            text += f" ({flags['is_transferred']:,} of the {total:,} rows)"
        text += "."
    elif ctx.mbr:
        text += (
            " Match-between-runs ran: the q values and n_runs are native (pre-MBR, "
            "scored_combined.parquet); n_runs_transfer_only counts the runs reached only "
            "through transfers."
        )
    if ctx.entrapment:
        text += (
            " Entrapment mode: decoys do not compete in the grouped q columns, and the "
            "entrapment spike-ins are targets by label"
        )
        if "is_entrapment" in flags:
            text += (
                f"; is_entrapment flags them ({flags['is_entrapment']:,} of the {total:,} rows), "
                "and the engine's own target counts exclude them"
            )
        text += "."
    if _has_quant(ctx):
        gate_text = _gate_sentence(ctx)
        if gate_text is not None:
            text += " " + gate_text
    if "selected_peak_rank" not in ctx.scored_names:
        text += " selected_peak_rank is not in this psms_scored version and is shown as 0."
    for note in ctx.notes:
        text += " " + note
    return text


# --------------------------------------------------------------------------- entry point


def identification_table(rs: ResultSet, query: TableQuery | None = None) -> TablePage:
    """One page of the precursor, peptide or protein-group table (see the module docstring).

    Decoys are hidden unless ``include_decoys``. The sort column must be one of the
    table's columns; rows are ordered by it with nulls last, then by the row key.
    """
    query = query or TableQuery()
    ctx = _validate(rs, query)
    cache = _cache(rs)
    # Large tables page from cached DuckDB tables; the lock keeps one thread from
    # dropping a table (cache eviction) while another reads it.
    with cache.lock if ctx.large else contextlib.nullcontext():
        return _page(rs, query, ctx, cache)


_TIEBREAK = {
    "precursor": ["source", "candidate_id"],
    "peptide": ["base_peptide_id", "label"],
    "protein_group": ["protein_group", "label"],
}


def _order_by(query: TableQuery) -> str:
    direction = "DESC" if query.descending else "ASC"
    return f"{sql_ident(query.sort_by)} {direction} NULLS LAST, " + ", ".join(
        sql_ident(c) for c in _TIEBREAK[query.unit] if c != query.sort_by
    )


def _flag_counts(
    rs: ResultSet, source: SqlFragment, flags: list[tuple[str, str]]
) -> tuple[int, dict[str, int]]:
    """Rows of a FROM item and of each flag condition."""
    extra = "".join(f", count(*) FILTER (WHERE {expr})" for _, expr in flags)
    row = rs.duck.execute(f"SELECT count(*){extra} FROM {source.text}", source.params).fetchone()
    values = row or (0,) * (1 + len(flags))
    counts = {name: int(v or 0) for (name, _), v in zip(flags, values[1:], strict=True)}
    return int(values[0]), counts


def _memo_flags(
    rs: ResultSet, key: tuple[Any, ...], source: SqlFragment, flags: list[tuple[str, str]]
) -> tuple[int, dict[str, int]]:
    """Total and flag counts of a query signature, counted once on ``source``."""
    memo_key = ("tables", "flags", *key)
    hit = rs._memo.get(memo_key)
    if hit is None:
        hit = _flag_counts(rs, source, flags)
        rs._memo[memo_key] = hit
        rs._memo[("tables", "total", *key)] = hit[0]
    return hit


def _memo_total(rs: ResultSet, key: tuple[Any, ...], plan: _Plan) -> int:
    """The total of a query signature from the cheap count query, once."""
    memo_key = ("tables", "total", *key)
    hit = rs._memo.get(memo_key)
    if hit is None:
        sql = plan.count()
        hit = int(rs.duck.scalar(sql.text, sql.params) or 0)
        rs._memo[memo_key] = hit
    return int(hit)


_Result = tuple[pd.DataFrame, int, dict[str, int]]


def _direct_page(rs: ResultSet, plan: _Plan, key: tuple[Any, ...], query: TableQuery) -> _Result:
    """One page from the whole query (small tables, or a result too large to cache)."""
    select = plan.select()
    total, flags = _memo_flags(rs, key, SqlFragment("(").extend(select).add(")"), plan.flags)
    rows = rs.duck.df(
        f"SELECT * FROM ({select.text}) p ORDER BY {_order_by(query)} LIMIT ? OFFSET ?",
        [*select.params, query.limit, query.offset],
    )
    return rows, total, flags


def _cached_page(
    rs: ResultSet, plan: _Plan, key: tuple[Any, ...], query: TableQuery, cache: _TableCache
) -> _Result:
    """One page of a large table from its cached result or its cached order.

    A result of at most :data:`MAX_MATERIALISED_ROWS` rows is materialised whole and
    serves every sort. A larger one is materialised as its order under this sort (one
    table per sort), and the page's rows are then read by row number. A hit reads its
    one table and builds nothing else.
    """
    wide_key = ("filtered", *key)
    entry = cache.get(wide_key)
    if entry is None and _memo_total(rs, key, plan) <= MAX_MATERIALISED_ROWS:
        entry = cache.create(rs, wide_key, plan.select(), len(plan.columns) + 1)
    if entry is not None:
        total, flags = _memo_flags(rs, key, SqlFragment(entry.name), plan.flags)
        rows = rs.duck.df(
            f"SELECT * FROM {entry.name} ORDER BY {_order_by(query)} LIMIT ? OFFSET ?",
            [query.limit, query.offset],
        )
        return rows, total, flags
    total = _memo_total(rs, key, plan)
    per_row = 2 + len(plan.flags)
    order_key = ("order", *key, query.sort_by, query.descending)
    entry = cache.get(order_key)
    if entry is None and total * per_row <= MAX_CACHED_CELLS:
        entry = cache.create(rs, order_key, plan.order(query), per_row)
    if entry is None:
        plan.ctx.note(
            f"The result has {total:,} rows, more than the table cache holds, so every page "
            "runs the whole query again."
        )
        return _direct_page(rs, plan, key, query)
    total, flags = _memo_flags(rs, key, SqlFragment(entry.name), plan.flags)
    # An empty range still runs, so an empty page keeps the column types.
    start = min(query.offset, total)
    stop = max(start, min(query.offset + query.limit, total))
    fetch = plan.fetch(entry.name, start, stop)
    return rs.duck.df(fetch.text, fetch.params), total, flags


def _page(rs: ResultSet, query: TableQuery, ctx: _Context, cache: _TableCache) -> TablePage:
    plan = _Plan.build(ctx)
    columns = plan.columns
    if query.sort_by not in columns:
        raise ViewerError(
            f"cannot sort the {query.unit} table by {query.sort_by!r}; sortable columns: "
            + ", ".join(columns)
            + "."
        )
    key = (query.signature(), ctx.q_column, ctx.stamps)
    if ctx.large:
        rows, total, flags = _cached_page(rs, plan, key, query, cache)
    else:
        rows, total, flags = _direct_page(rs, plan, key, query)
    return TablePage(
        rows=rows[columns].reset_index(drop=True),
        total=int(total),
        query=query,
        description=_describe(ctx, int(total), flags, plan.words),
        column_labels=_labels(ctx, columns),
        q_column=ctx.q_column,
    )


def page_count(page: TablePage) -> int:
    """The number of pages of ``page.query.limit`` rows (at least 1)."""
    limit = page.query.limit or 1
    return max(1, -(-page.total // limit))


def with_page(query: TableQuery, offset: int) -> TableQuery:
    """The same query at another offset."""
    return replace(query, offset=max(0, int(offset)))
