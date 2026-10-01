"""Identification page (P0 view 2): linked panels in the manner of PeptideShaker.

The starting level picks the first table: **protein groups** (the default) shows the
protein-group table, then the peptides of the selected group beside the precursors of
the selected peptide, then the preview of the selected precursor
(:func:`mumdia_viewer.ui.detail.preview`). **Peptides** starts from the peptide table,
**precursors** from the precursor table. A selection updates every panel below it, and
the first row of each table is selected when the page opens, so no panel is empty.

Every row and count comes from :func:`mumdia_viewer.data.tables.identification_table`:

* the first table uses AG Grid's infinite row model; the ``rows`` callback answers each
  block request with the rows and the total, sorted and paged by the data layer;
* a child table holds every row of its parent (``protein_group=`` or
  ``base_peptide_id=``, ``threshold=None``), passing or not, so its validation marks are
  informative. It is loaded whole in blocks of :data:`CHILD_MAX` rows and sorted in the
  browser. The decoy switch applies to every table, the other filters to the first.

The layout holds the first block only. The child panels are filled by the ``children``
callback, which answers one request with both panels of a protein group (the peptides,
and the precursors of the peptide the parent row shows), so a click costs one round
trip; the browser keeps the answers (``assets/browser.js``) and asks ahead for the next
row of the table once the selection rests (``prefetch``). A parent selects the peptide
and the precursor of the row it shows (its winning row, or the best row that passes the
filters): the child panels open on the row whose values the parent's row displays.

The preview of the selected precursor comes from the ``preview`` callback. The browser
asks for it when the selection rests, inserts it after the child panels have their rows,
and draws its static tables as plain HTML; it also tells whether the precursor of a
link exists (``ok``), so a link to a candidate that is not in the result set falls back
to the first row with a notice. While the browser draws a new page, the server reads the
detail of the page's first precursor (:mod:`.browser_ahead`).

The address holds the level (``unit``), the filters (``q``, ``search``, ``charge``,
``protein``, ``mod``, ``quant``, ``in_run``, ``decoys``, ``sort``, ``order``, ``t``) and
the selection (``group``, ``peptide``, ``run``, ``cid``), so a link restores the view.
The browser rewrites it with ``history.replaceState``, which the router does not see;
a link to the address the router built last (which it would not rebuild) is rebuilt by
:func:`register`'s ``rebuild`` callback.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html
from dash import no_update as NO

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.export import table_export
from mumdia_viewer.data.fasta import Fasta, protein_coverage
from mumdia_viewer.data.rescore import rescore_info
from mumdia_viewer.data.tables import (
    Q_COLUMNS,
    UNIT_Q_COLUMN,
    TablePage,
    TableQuery,
    identification_table,
)

from . import browser_ahead as ahead
from . import browser_panels as panels
from . import coverage
from .browser_grid import (
    BLOCK_SIZE,
    GRID_ID,
    PEP_GRID_ID,
    PRE_GRID_ID,
    Scales,
    child_columns,
    child_meta,
    column_defs,
    column_options,
    facets,
    modifications,
    records,
    safe_columns,
    scales_of,
    visible_columns,
    winner_matters,
)
from .icons import icon
from .state import (
    DEFAULT_THRESHOLD,
    THRESHOLD_STOPS,
    PageContext,
    parse_threshold,
    query_of,
    stop_label,
)
from .widgets import fmt

log = logging.getLogger(__name__)

UNITS = panels.LEVELS
DEFAULT_UNIT = "protein_group"
TEXT_MAX = 200
# A protein group can list many members (60 on the Astral run's largest).
GROUP_MAX = 4000
# Child tables are loaded in blocks of this many rows (one block for nearly every
# parent; TITIN_HUMAN of the six-run Astral experiment has 2,572 peptides).
CHILD_MAX = 2000
# "Show the selected row": the server reads the first table in blocks of this many rows
# (the data layer's largest page) until it finds the row, up to LOCATE_MAX rows.
LOCATE_BLOCK = 10_000
LOCATE_MAX = 2_000_000
# Requests the browser may ask ahead in one call.
PREFETCH_MAX = 2
# The fields of a row that the preview's fallback summary shows.
PREC_FIELDS = (
    "peptidoform",
    "charge",
    "label",
    "run",
    "score",
    "q_value",
    "run_psm_q",
    "precursor_q",
    "peptide_q_value",
    "pg_q_value",
)


# --------------------------------------------------------------------------- view


def _stop(value: object) -> float | None:
    """A threshold from the address when it is one of the header's stops, else None."""
    try:
        t = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    for s in THRESHOLD_STOPS:
        if math.isclose(t, s, rel_tol=1e-9):
            return s
    return None


def _text(value: object, n: int = TEXT_MAX) -> str:
    return str(value if value is not None else "").strip()[:n]


def _int(value: object) -> int | None:
    """An integer from an address or a store (``"12"``, ``12``, ``12.0``), else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        f = float(str(value).strip())
    except ValueError:
        return None
    return int(f) if math.isfinite(f) and f == int(f) else None


def default_q(unit: str, run: str) -> str:
    """The q column the data layer uses when none is chosen (``q_column=None``)."""
    return "run_psm_q" if run and unit == "precursor" else UNIT_Q_COLUMN[unit]


@dataclass(frozen=True)
class Filters:
    """The first table of the page: its unit (the starting level) and its filters.

    ``run`` is the run filter of an experiment's precursor table (address key
    ``in_run``); ``run`` in the address is the selected precursor's run.
    """

    unit: str = DEFAULT_UNIT
    q: str = ""
    search: str = ""
    charge: str = ""
    protein: str = ""
    mod: str = ""
    quant: str = ""
    run: str = ""
    decoys: bool = False
    t: float = DEFAULT_THRESHOLD
    sort: str = "score"
    desc: bool = True

    @classmethod
    def from_query(cls, query: Mapping[str, str], *, threshold: float) -> Filters:
        """Filters from an address query; invalid values fall back to the defaults.

        The run and the sort column are kept as given: the data layer checks them and
        its message is shown when they do not fit the table.
        """
        unit = query.get("unit", "")
        unit = unit if unit in UNITS else DEFAULT_UNIT
        charge = _text(query.get("charge"))
        order = _text(query.get("order")).lower()
        run = _text(query.get("in_run"), 64)
        q = query.get("q", "")
        # The default column, chosen explicitly, is the default.
        q = q if q in Q_COLUMNS and q != default_q(unit, run) else ""
        return cls(
            unit=unit,
            q=q,
            search=_text(query.get("search")),
            charge=charge if charge.isdigit() and 0 < int(charge) < 100 else "",
            protein=_text(query.get("protein")),
            mod=_text(query.get("mod")),
            quant=_text(query.get("quant"), 64),
            run=run,
            decoys=_text(query.get("decoys")).lower() in ("1", "true", "yes", "on"),
            t=_stop(query.get("t")) or threshold,
            sort=_text(query.get("sort"), 64) or "score",
            desc=order != "asc",
        )

    @classmethod
    def from_store(cls, data: Mapping[str, Any] | None) -> Filters:
        data = dict(data or {})
        try:
            t = float(data.get("t", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            t = DEFAULT_THRESHOLD
        unit = data.get("unit")
        return cls(
            unit=unit if unit in UNITS else DEFAULT_UNIT,
            q=str(data.get("q") or "") if data.get("q") in Q_COLUMNS else "",
            search=_text(data.get("search")),
            charge=_text(data.get("charge")),
            protein=_text(data.get("protein")),
            mod=_text(data.get("mod")),
            quant=_text(data.get("quant")),
            run=_text(data.get("run")),
            decoys=bool(data.get("decoys")),
            t=t if 0 < t < 1 else DEFAULT_THRESHOLD,
            sort=_text(data.get("sort")) or "score",
            desc=bool(data.get("desc", True)),
        )

    def store(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """What the first table's rows depend on (not the sort)."""
        d = self.store()
        d.pop("sort")
        d.pop("desc")
        return json.dumps(d, sort_keys=True)

    def query(self) -> dict[str, str]:
        """The address query of this view (defaults left out; the threshold always in)."""
        out: dict[str, str] = {}
        if self.unit != DEFAULT_UNIT:
            out["unit"] = self.unit
        for key in ("q", "search", "charge", "protein", "mod", "quant"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.run:
            out["in_run"] = self.run
        if self.decoys:
            out["decoys"] = "1"
        if self.sort != "score" or not self.desc:
            out["sort"] = self.sort
            out["order"] = "desc" if self.desc else "asc"
        out["t"] = repr(self.t)
        return out

    def table_query(
        self, *, offset: int = 0, limit: int = BLOCK_SIZE, sort: tuple[str, bool] | None = None
    ) -> TableQuery:
        sort_by, desc = sort if sort is not None else (self.sort, self.desc)
        return TableQuery(
            unit=self.unit,  # type: ignore[arg-type]
            q_column=self.q or None,
            threshold=self.t,
            include_decoys=self.decoys,
            charge=int(self.charge) if self.charge.isdigit() else None,
            protein=self.protein or None,
            modification=self.mod or None,
            quant_status=self.quant or None,
            search=self.search or None,
            run=self.run or None,
            sort_by=sort_by,
            descending=desc,
            offset=max(0, int(offset)),
            limit=max(0, min(int(limit), 1000)),
        )


@dataclass(frozen=True)
class Selection:
    """The selected rows: a protein group, a peptide and a precursor ``(run, cid)``."""

    group: str | None = None
    peptide: int | None = None
    run: str = ""
    cid: int | None = None

    @classmethod
    def from_query(cls, query: Mapping[str, Any]) -> Selection:
        cid = _int(query.get("cid"))
        return cls(
            group=_text(query.get("group"), GROUP_MAX) or None,
            peptide=_int(query.get("peptide")),
            run=_text(query.get("run"), 64) if cid is not None else "",
            cid=cid,
        )

    from_store = from_query

    def store(self) -> dict[str, Any]:
        return {"group": self.group, "peptide": self.peptide, "run": self.run, "cid": self.cid}

    def query(self, unit: str) -> dict[str, str]:
        """The address keys of the selection that the level shows."""
        out: dict[str, str] = {}
        if unit == "protein_group" and self.group:
            out["group"] = self.group
        if unit != "precursor" and self.peptide is not None:
            out["peptide"] = str(self.peptide)
        if self.cid is not None:
            if self.run:
                out["run"] = self.run
            out["cid"] = str(self.cid)
        return out

    def is_empty(self) -> bool:
        return self.group is None and self.peptide is None and self.cid is None


def sort_of(request: Mapping[str, Any] | None, f: Filters) -> tuple[str, bool]:
    """The sort of a grid request (its sort model), else the view's sort."""
    model = (request or {}).get("sortModel") or []
    if model and model[0].get("colId"):
        return str(model[0]["colId"]), model[0].get("sort") != "asc"
    return f.sort, f.desc


def q_active(f: Filters, experiment: bool) -> str:
    """The q column the first table filters on (the data layer's choice for the default)."""
    return f.q or default_q(f.unit, f.run if experiment else "")


def fast_mode(f: Filters) -> bool:
    """Whether the first table lists winning rows only, so its row filters test that row.

    The data layer's rule: a grouped table on its unit's own q column below 1 (the
    engine writes a grouped q below 1.0 to the winning row of a group only).
    """
    return f.unit != "precursor" and not f.q and f.t < 1.0


# --------------------------------------------------------------------------- tables


@dataclass
class Child:
    """A child table: the rows of one parent (no q filter), one block of them."""

    unit: str
    rows: list[dict[str, Any]]
    total: int
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    offset: int = 0


def child_table(
    rs: ResultSet,
    base: str,
    unit: str,
    *,
    group: str | None = None,
    peptide: int | None = None,
    decoys: bool = False,
    offset: int = 0,
) -> Child:
    """The peptides of a protein group or the precursors of a peptide, best score first.

    One block of :data:`CHILD_MAX` rows from ``offset``, with the columns the child grid
    carries (:func:`browser_grid.child_columns`).
    """
    query = TableQuery(
        unit=unit,  # type: ignore[arg-type]
        threshold=None,
        include_decoys=decoys,
        protein_group=group,
        base_peptide_id=peptide,
        sort_by="score",
        descending=True,
        offset=max(0, int(offset)),
        limit=CHILD_MAX,
    )
    try:
        page = identification_table(rs, query)
    except (ViewerError, ValueError) as exc:
        return Child(unit, [], 0, error=str(exc), offset=offset)
    columns = child_columns(list(page.rows.columns))
    return Child(
        unit,
        records(page.rows, base, unit, columns),
        int(page.total),
        page.description,
        dict(page.column_labels),
        columns,
        offset=offset,
    )


def _pick(rows: Sequence[Mapping[str, Any]], key: str, *wanted: Any) -> dict[str, Any] | None:
    """The first row whose ``key`` is one of ``wanted`` (in order), else the first row."""
    for w in wanted:
        if w is None:
            continue
        for r in rows:
            if r.get(key) == w:
                return dict(r)
    return dict(rows[0]) if rows else None


def _pick_precursor(
    rows: Sequence[Mapping[str, Any]], *wanted: tuple[Any, Any]
) -> dict[str, Any] | None:
    for run, cid in wanted:
        if cid is None:
            continue
        for r in rows:
            if r.get("candidate_id") == cid and (r.get("run") or "") == (run or ""):
                return dict(r)
    return dict(rows[0]) if rows else None


def first_block(
    rs: ResultSet, base: str, f: Filters
) -> tuple[TablePage | None, list[dict[str, Any]], str | None]:
    """The first block of the first table (it also warms the data layer's cache)."""
    try:
        page = identification_table(rs, f.table_query(offset=0, limit=BLOCK_SIZE))
    except (ViewerError, ValueError) as exc:
        return None, [], str(exc)
    return page, records(page.rows, base, f.unit), None


def initial_selection(f: Filters, want: Selection, first: Mapping[str, Any] | None) -> Selection:
    """The selection a new page starts from: the address's, else the first row's.

    Only the keys the level shows count. A key the address does not give comes from the
    first row when the row is the selected parent (a group's row names its peptide and
    precursor). The rows of a child table are not known yet: the ``children`` callback
    checks the wanted peptide and precursor against them and falls back to the first.
    """
    fg = str(first.get("protein_group")) if first and first.get("protein_group") else None
    fp = _int(first.get("base_peptide_id")) if first else None
    frun = str(first.get("run") or "") if first else ""
    fcid = _int(first.get("candidate_id")) if first else None
    if f.unit == "protein_group":
        group = want.group or fg
        peptide = want.peptide if want.group else None
        if peptide is None and group is not None and group == fg:
            peptide = fp
        run, cid = want.run, want.cid
        if cid is None and group == fg and peptide == fp:
            run, cid = frun, fcid
        return Selection(group=group, peptide=peptide, run=run if cid is not None else "", cid=cid)
    if f.unit == "peptide":
        peptide = want.peptide if want.peptide is not None else fp
        run, cid = want.run, want.cid
        if cid is None and peptide == fp:
            run, cid = frun, fcid
        return Selection(peptide=peptide, run=run if cid is not None else "", cid=cid)
    if want.cid is not None:
        return Selection(run=want.run, cid=want.cid)
    return Selection(run=frun, cid=fcid)


def prec_data(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The store of the selected precursor: its run and candidate id, and a few values."""
    if not row or row.get("candidate_id") is None:
        return None
    return {
        "run": str(row.get("run") or ""),
        "cid": _int(row.get("candidate_id")),
        "row": {k: row.get(k) for k in PREC_FIELDS if k in row},
    }


# --------------------------------------------------------------------------- panel words


def mark_column(unit: str, experiment: bool) -> str:
    if unit == "peptide":
        return "peptide_q_value"
    return "run_psm_q" if experiment else "q_value"


def top_line(f: Filters, total: int | None, active: str) -> str:
    """The first line of the first table's help: what the rows are and how they pass."""
    noun = panels.NOUNS[f.unit] + "s"
    if total is None:
        return f"The data layer refused this {panels.NOUNS[f.unit]} table."
    who = "targets only" if not f.decoys else "targets and decoys"
    return f"{total:,} {noun} with {active} ≤ {stop_label(f.t)} ({who})."


def top_help(
    f: Filters, total: int | None, active: str, description: str, error: str | None
) -> Any:
    if error:
        return panels.help_body(top_line(f, None, active), small=error)
    return panels.help_body(
        top_line(f, total, active),
        panels.FAST_NOTE if fast_mode(f) else None,
        small=description,
    )


def child_help(child: Child | None, unit: str, mark: str) -> Any:
    what = (
        "Every peptide of the selected protein group"
        if unit == "peptide"
        else "Every scored row of the selected peptide"
    )
    line = (
        f"{what}, passing or not (no q filter). The marks test {mark} at the header "
        "threshold; the badge counts the rows that pass."
    )
    rule = "The decoy switch applies; the other filters apply to the first table only."
    if child is not None and child.error:
        return panels.help_body(
            line, rule, small=f"The data layer refused this table: {child.error}"
        )
    return panels.help_body(line, rule, small=child.description if child is not None else None)


def child_note(child: Child | None, mark: str) -> str:
    """The short note of a child panel's title bar (its title says the rest)."""
    if child is None:
        return ""
    if child.error:
        return "refused by the data layer"
    if child.total > len(child.rows):
        return f"{len(child.rows):,} of {child.total:,} loaded · any q"
    return "any q"


def note_title(mark: str) -> str:
    return (
        f"No q filter: every row of the parent, passing or not. The marks test {mark} at the "
        "header threshold."
    )


# --------------------------------------------------------------------------- children


def child_part(child: Child, key: Any, subject: Any, mark: str) -> dict[str, Any]:
    """One child panel's content, as the browser applies it (and keeps it)."""
    return {
        "key": key,
        "rows": child.rows,
        "total": child.total,
        "offset": child.offset,
        "mark": mark,
        "subject": subject,
        "help": child_help(child, child.unit, mark),
        "note": child_note(child, mark),
        "error": child.error,
    }


def children_payload(rs: ResultSet, base: str, need: Mapping[str, Any]) -> dict[str, Any]:
    """The child panels of a selection, in one answer.

    ``need``: ``kind`` (``"group"``: the peptides of ``group`` and the precursors of the
    peptide its row shows; ``"peptide"``: the precursors of ``peptide``; ``"more"``: the
    next block of a long child table), the wanted ``peptide``, ``run`` and ``cid``,
    ``decoys`` and the browser's request number ``n``. The answer's ``sel`` is the
    selection after the check: the wanted peptide and precursor when they are rows of
    their parent, else the parent's first (best-scoring) row. ``found`` is False when the
    parent has no rows (a link to a group or peptide this result set lacks).
    """
    experiment = rs.is_experiment
    kind = str(need.get("kind") or "group")
    decoys = bool(need.get("decoys"))
    out: dict[str, Any] = {"n": need.get("n"), "kind": kind, "decoys": decoys}
    if kind == "more":
        unit = "peptide" if need.get("unit") == "peptide" else "precursor"
        child = child_table(
            rs,
            base,
            unit,
            group=str(need.get("group") or "") if unit == "peptide" else None,
            peptide=_int(need.get("peptide")) if unit == "precursor" else None,
            decoys=decoys,
            offset=_int(need.get("offset")) or 0,
        )
        out["more"] = {
            "unit": unit,
            "key": need.get("group") if unit == "peptide" else _int(need.get("peptide")),
            "rows": child.rows,
            "total": child.total,
            "offset": child.offset,
            "error": child.error,
        }
        return out
    group = str(need.get("group") or "") or None
    peptide = _int(need.get("peptide"))
    run, cid = str(need.get("run") or ""), _int(need.get("cid"))
    peptidoform = need.get("peptidoform")
    out["found"] = True
    if kind == "group":
        pep = child_table(rs, base, "peptide", group=group, decoys=decoys)
        mark = mark_column("peptide", experiment)
        out["pep"] = child_part(pep, group, panels.group_subject(group), mark)
        row = _pick(pep.rows, "base_peptide_id", peptide)
        if row is None:
            out["found"] = pep.error is not None
            out["pre"] = None
            out["sel"] = {"group": group, "peptide": None, "run": "", "cid": None}
            out["prec"] = None
            return out
        if _int(row.get("base_peptide_id")) != peptide:
            # The parent's precursor belongs to the peptide it named.
            run, cid = "", None
        peptide = _int(row.get("base_peptide_id"))
        peptidoform = row.get("peptidoform")
    if peptide is None:
        out.update(found=False, pre=None, prec=None)
        out["sel"] = {"group": group, "peptide": None, "run": "", "cid": None}
        return out
    pre = child_table(rs, base, "precursor", peptide=peptide, decoys=decoys)
    if not pre.rows and pre.error is None:
        out["found"] = False
    text = peptidoform or (pre.rows[0].get("peptidoform") if pre.rows else None)
    mark = mark_column("precursor", experiment)
    out["pre"] = child_part(pre, peptide, panels.peptide_subject(text, peptide), mark)
    prec = _pick_precursor(pre.rows, (run, cid))
    out["sel"] = {
        "group": group,
        "peptide": peptide,
        "run": str(prec.get("run") or "") if prec else "",
        "cid": _int(prec.get("candidate_id")) if prec else None,
    }
    out["prec"] = prec_data(prec)
    return out


def row_keys(unit: str, rows: pd.DataFrame) -> pd.Series:
    """The grid row ids of a table page (:func:`browser_grid.row_key`), as strings."""
    if unit == "protein_group":
        return rows["protein_group"].astype(str)
    if unit == "peptide":
        return rows["base_peptide_id"].astype(str)
    run = rows["run"].fillna("").astype(str) if "run" in rows else ""
    return run + ":" + rows["candidate_id"].astype(str)


def locate_index(
    rs: ResultSet, store: Mapping[str, Any] | None, key: str, sort: Sequence[Any] | None
) -> int | None:
    """The position of a row (its grid id) in the first table under a sort, or None.

    It reads the table from the data layer (which keeps the filtered set) in blocks of
    :data:`LOCATE_BLOCK` rows until it finds the row; the Astral precursor table (88,140
    rows) takes nine blocks.
    """
    f = Filters.from_store(store)
    order = (str(sort[0]), bool(sort[1])) if sort and len(sort) == 2 else (f.sort, f.desc)
    base = f.table_query(sort=order)
    offset = 0
    while offset < LOCATE_MAX:
        query = replace(base, offset=offset, limit=LOCATE_BLOCK)
        try:
            page = identification_table(rs, query)
        except (ViewerError, ValueError):
            return None
        if len(page.rows):
            hits = np.flatnonzero(row_keys(f.unit, page.rows).to_numpy() == key)
            if hits.size:
                return offset + int(hits[0])
        offset += LOCATE_BLOCK
        if offset >= page.total:
            return None
    return None


# --------------------------------------------------------------------------- layout


def _empty(
    rs: ResultSet, f: Filters, total: int | None, error: str | None, description: str = ""
) -> Any:
    """The message over the first table's body: the data layer's refusal, or no rows.

    With no rows, ``description`` (the data layer's words for the table) says how the
    filters select rows, for example that a group is represented by its winning row.
    """
    if error:
        hint = (
            "Click a column header to sort by that column, or press Reset."
            if "cannot sort" in error
            else "Remove the filter the message names (its chip), or press Reset."
        )
        return panels.empty_state(
            "The data layer refused this table",
            f"{error} {hint}",
            icon_name="alert",
            red=True,
        )
    if total == 0:
        noun = panels.NOUNS[f.unit] + "s"
        q = q_active(f, rs.is_experiment)
        return panels.empty_state(
            f"No {noun} pass these filters",
            f"The table keeps rows with {q} ≤ {stop_label(f.t)}"
            + (", targets only" if not f.decoys else "")
            + ". Remove a filter, or raise the threshold in the header.",
            icon_name="search",
            detail=description or None,
        )
    return None


def _child_defs(
    rs: ResultSet, unit: str, scales: Scales, rescorer: str | None, t: float
) -> list[dict[str, Any]]:
    """The column definitions of a child grid (the same for every parent)."""
    columns, labels = child_meta(rs, unit)
    return column_defs(
        unit,
        child_columns(columns),
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


def _child_panel(
    rs: ResultSet,
    key: str,
    title: str,
    grid_id: str,
    defs: list[dict[str, Any]],
    base: str,
    t: float,
    *,
    unit: str,
    subject: Any,
    empty_text: str,
    loading: bool,
    above: Any = None,
) -> Any:
    mark = mark_column(unit, rs.is_experiment)
    shown = visible_columns(defs)
    return panels.panel(
        key,
        title,
        panels.child_grid(grid_id, key, defs, base, t, empty_text),
        count="-" if loading else "0",
        help_text=child_help(None, unit, mark),
        subject=subject,
        right=html.Div(
            [
                html.Span(
                    "any q", id=f"ib-{key}-note", className="ib-note", title=note_title(mark)
                ),
                panels.columns_menu(
                    column_options(defs, experiment=rs.is_experiment),
                    shown,
                    prefix=f"ib-{key}-cols",
                ),
            ],
            className="ib-pright",
        ),
        loading=loading,
        above=above,
    )


def coverage_slot(ctx: PageContext) -> Any:
    """The coverage strip of the peptides panel; the browser asks for it per group."""
    if ctx.fasta is None:
        return html.Div(coverage.no_fasta(), id="ib-cov", className="ib-cov")
    return html.Div(
        coverage.message("Coverage of the selected protein group follows."),
        id="ib-cov",
        className="ib-cov",
    )


def coverage_answer(rs: ResultSet, fasta: Fasta | None, request: Mapping[str, Any] | None) -> Any:
    """The strip for a request ``{group, member, peptide, t}`` of the browser."""
    if fasta is None:
        return coverage.no_fasta()
    group = str((request or {}).get("group") or "")
    if not group:
        return coverage.message("Select a protein group to see its coverage.")
    try:
        cov = protein_coverage(
            rs,
            fasta,
            group,
            member=(request or {}).get("member") or None,
            threshold=parse_threshold((request or {}).get("t")),
        )
    except ViewerError as exc:
        return coverage.message(str(exc), warn=True)
    return coverage.strip(cov, selected=_int((request or {}).get("peptide")))


def content(ctx: PageContext) -> list[Any]:
    """The page for one address (also used by the ``rebuild`` callback).

    It runs one query, the first block of the first table; the child panels and the
    preview are asked for by the browser once that table shows its rows.
    """
    t0 = time.perf_counter()
    rs = ctx.rs
    experiment = rs.is_experiment
    f = Filters.from_query(ctx.query, threshold=ctx.threshold)
    want = Selection.from_query(ctx.query)
    top_page, rows, error = first_block(rs, ctx.base, f)
    first = rows[0] if rows else None
    sel = initial_selection(f, want, first)
    info = rescore_info(rs)
    fac = facets(rs)
    scales = scales_of(rs)
    if top_page is not None:
        columns = list(top_page.rows.columns)
        labels = dict(top_page.column_labels)
        total: int | None = int(top_page.total)
        description = top_page.description
        active = top_page.q_column or q_active(f, experiment)
    else:
        columns, labels = safe_columns(rs, f.unit)
        total, description, active = None, "", q_active(f, experiment)
    defs = column_defs(
        f.unit,
        columns,
        labels,
        role="top",
        experiment=experiment,
        q_active=active,
        threshold=f.t,
        winner_matters=winner_matters(f.unit, active),
        scales=scales,
        rescorer=info.label,
        sort=(f.sort, f.desc),
    )
    shown = visible_columns(defs)
    empty = _empty(rs, f, total, error, description)
    top = panels.panel(
        "top",
        panels.TITLES[f.unit],
        [
            panels.top_grid(defs, ctx.base, total, f.t),
            html.Div(
                empty, id="ib-empty", className="ib-empty" + (" ib-empty-on" if empty else "")
            ),
        ],
        count=fmt(total) if total is not None else "-",
        help_text=top_help(f, total, active, description, error),
        extra=panels.q_chip(f"{active} ≤ {stop_label(f.t)}", f.unit),
        right=html.Div(
            [
                panels.locate_action(),
                html.Span("", id="ib-range", className="ib-range"),
                html.Button(
                    [icon("external", 12), "TSV"],
                    id="ib-tsv",
                    n_clicks=0,
                    className="ib-tsv",
                    title="Download every row of this table under its filters as TSV "
                    "(the engine's columns as they are)",
                ),
                dcc.Download(id="ib-download"),
                panels.columns_menu(column_options(defs, experiment=experiment), shown),
            ],
            className="ib-pright",
        ),
    )
    blocks = [top]
    stores: list[Any] = []
    selected_first = first is not None and (
        _int(first.get("candidate_id")) == sel.cid and str(first.get("run") or "") == sel.run
    )
    if f.unit in ("protein_group", "peptide"):
        if f.unit == "protein_group":
            blocks.append(
                _child_panel(
                    rs,
                    "pep",
                    "Peptides of",
                    PEP_GRID_ID,
                    _child_defs(rs, "peptide", scales, info.label, f.t),
                    ctx.base,
                    f.t,
                    unit="peptide",
                    subject=panels.group_subject(sel.group),
                    empty_text="No peptides",
                    loading=sel.group is not None,
                    above=coverage_slot(ctx),
                )
            )
            stores.append(dcc.Store(id="ib-cov-req"))
        text = first.get("peptidoform") if first and selected_first else None
        blocks.append(
            _child_panel(
                rs,
                "pre",
                "Precursors of",
                PRE_GRID_ID,
                _child_defs(rs, "precursor", scales, info.label, f.t),
                ctx.base,
                f.t,
                unit="precursor",
                subject=panels.peptide_subject(text, sel.peptide),
                empty_text="No precursors",
                loading=sel.group is not None or sel.peptide is not None,
            )
        )
        stores += [
            dcc.Store(id="ib-need"),
            dcc.Store(id="ib-children"),
            dcc.Store(id="ib-prefetch"),
            dcc.Store(id="ib-prefetched"),
        ]
    blocks.append(panels.preview_slot())
    prec = (
        prec_data(first)
        if selected_first
        else ({"run": sel.run, "cid": sel.cid, "row": {}} if sel.cid is not None else None)
    )
    stores += [
        dcc.Store(id="ib-view", data=f.store()),
        dcc.Store(id="ib-sig", data=shown_key_of(f, error)),
        dcc.Store(id="ib-defs", data=defs_key_of(labels, active, f.t)),
        dcc.Store(id="ib-sel", data=sel.store()),
        dcc.Store(id="ib-prec", data=prec),
        dcc.Store(id="ib-first"),
        dcc.Store(id="ib-address"),
        dcc.Store(id="ib-boot", data=f.unit),
        dcc.Store(id="ib-cols-defaults", data=shown),
        dcc.Store(id="ib-sync"),
        dcc.Store(id="ib-built", data=dict(ctx.query)),
        dcc.Store(id="ib-locate-req"),
        dcc.Store(id="ib-located"),
        # A token per built page: the client tells a new page from a new view.
        dcc.Store(id="ib-page", data=uuid.uuid4().hex),
    ]
    fast = fast_mode(f)
    if sel.cid is not None:
        # While the browser draws the page, the server reads the preview's detail.
        ahead.start_detail(rs, sel.run, sel.cid, panels.warm_detail)
    page = html.Div(
        [
            *stores,
            panels.toolbar(
                rs,
                f,
                default_q=default_q(f.unit, f.run),
                charges=fac.charges,
                statuses=fac.statuses,
                chips=panels.chip_row(f, fast=fast),
                n_filters=panels.n_popover_filters(f),
                fast=fast,
            ),
            panels.notice_line(),
            html.Div(blocks, className="ib-panels", **{"data-level": f.unit}),
        ],
        id="ib-root",
        className="ib-root",
        **{
            "data-unit": f.unit,
            "data-experiment": "1" if experiment else "0",
            "data-base": ctx.base,
        },
    )
    log.debug("identifications layout in %.0f ms", (time.perf_counter() - t0) * 1000.0)
    return [page]


def layout(ctx: PageContext) -> Any:
    return html.Div(content(ctx), id="ib-shell")


# --------------------------------------------------------------------------- callbacks


CONTROLS = ("ib-q", "ib-search", "ib-charge", "ib-protein", "ib-mod", "ib-quant", "ib-run")


def rows_response(
    rs: ResultSet,
    request: Mapping[str, Any] | None,
    store: Mapping[str, Any] | None,
    base: str,
) -> tuple[dict[str, Any], TablePage | None, int | None, str | None, str]:
    """The answer to one grid request: (response, page or None, total, error, q column).

    It never raises: a refused query answers with no rows and the data layer's message,
    so the grid never waits for ever.
    """
    f = Filters.from_store(store)
    start = int((request or {}).get("startRow") or 0)
    end = int((request or {}).get("endRow") or start + BLOCK_SIZE)
    sort = sort_of(request, f)
    try:
        page = identification_table(rs, f.table_query(offset=start, limit=end - start, sort=sort))
    except (ViewerError, ValueError) as exc:
        return {"rowData": [], "rowCount": 0}, None, None, str(exc), q_active(f, rs.is_experiment)
    except Exception as exc:
        log.exception("identification table request failed")
        return (
            {"rowData": [], "rowCount": 0},
            None,
            None,
            f"Unexpected error: {type(exc).__name__}: {exc}",
            q_active(f, rs.is_experiment),
        )
    data = records(page.rows, base, f.unit)
    return (
        {"rowData": data, "rowCount": int(page.total)},
        page,
        int(page.total),
        None,
        page.q_column or q_active(f, rs.is_experiment),
    )


def first_of(
    rs: ResultSet,
    request: Mapping[str, Any] | None,
    response: Mapping[str, Any],
    store: Mapping[str, Any] | None,
    base: str,
    error: str | None,
) -> dict[str, Any]:
    """The first block of a new view, for the client's selection (keep or first row)."""
    if error:
        return {"row": None, "keys": [], "at": time.time()}
    rows = response.get("rowData") or []
    if int((request or {}).get("startRow") or 0) != 0:
        block = dict(request or {}, startRow=0, endRow=BLOCK_SIZE)
        rows = rows_response(rs, block, store, base)[0].get("rowData") or []
    first = rows[0] if rows else None
    return {
        "row": {k: v for k, v in first.items()} if first else None,
        "keys": [r.get("_key") for r in rows],
        "at": time.time(),
    }


def preview_answer(ctx: PageContext, prec: Mapping[str, Any] | None) -> dict[str, Any]:
    """The preview card of a request, with what it answers (the browser keeps it)."""
    card, ok = panels.preview(ctx, prec)
    prec = prec or {}
    return {
        "card": card,
        "ok": ok,
        "run": str(prec.get("run") or ""),
        "cid": _int(prec.get("cid")),
        "t": ctx.threshold,
        "n": prec.get("n"),
    }


def register(app, get_rs, base: str) -> None:
    """Callbacks of the identification page."""
    get_fasta = getattr(app, "mv_fasta", lambda: None)

    # Every row of the first table under its filters, as TSV.
    @app.callback(
        Output("ib-download", "data"),
        Output("ib-notice-text", "children", allow_duplicate=True),
        Output("ib-notice", "className", allow_duplicate=True),
        Input("ib-tsv", "n_clicks"),
        State("ib-view", "data"),
        prevent_initial_call=True,
    )
    def download_table(n, store):
        if not n:
            return NO, NO, NO
        try:
            out = table_export(get_rs(), Filters.from_store(store).table_query())
        except (ViewerError, ValueError) as exc:
            return NO, f"The table was not exported: {exc}", "ib-notice"
        return {"content": out.text, "filename": out.filename}, NO, NO

    # The coverage strip of the selected protein group (asked for by the browser).
    @app.callback(
        Output("ib-cov", "children"),
        Input("ib-cov-req", "data"),
        prevent_initial_call=True,
    )
    def coverage_of(request):
        if not request or not request.get("group"):
            return NO
        t0 = time.perf_counter()
        out = coverage_answer(get_rs(), get_fasta(), request)
        log.debug("coverage in %.0f ms", (time.perf_counter() - t0) * 1000.0)
        return out

    # Controls and the header threshold make the view (client side, no round trip).
    app.clientside_callback(
        ClientsideFunction("mvb", "filters"),
        Output("ib-view", "data"),
        *[Input(c, "value") for c in CONTROLS],
        Input("ib-decoys", "checked"),
        Input("threshold", "data"),
        State("ib-view", "data"),
        prevent_initial_call=True,
    )
    # A new view empties the first table's block cache and rewrites the address; on the
    # first call of a page it writes the address and brings the header threshold to the
    # view's (a link with t= sets it).
    app.clientside_callback(
        ClientsideFunction("mvb", "sync"),
        Output("ib-sync", "data"),
        Input("ib-view", "data"),
        State("q-select", "value"),
        State("ib-page", "data"),
        State("ib-built", "data"),
    )
    # The page's first selection (the address's or the first row's); after that the
    # browser keeps the selection.
    app.clientside_callback(
        ClientsideFunction("mvb", "sel"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-sel", "data"),
        State("ib-view", "data"),
        State("ib-page", "data"),
        State("ib-built", "data"),
        State("ib-prec", "data"),
        prevent_initial_call="initial_duplicate",
    )
    # The first block of a new view: keep the selection when it is there, else select
    # the first row.
    app.clientside_callback(
        ClientsideFunction("mvb", "first"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-first", "data"),
        prevent_initial_call=True,
    )
    # Answers from the server, applied (and kept) by the browser.
    app.clientside_callback(
        ClientsideFunction("mvb", "children"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-children", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvb", "prefetched"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-prefetched", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvb", "preview"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-preview-data", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvb", "located"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-located", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvb", "level"),
        Output("ib-sync", "data", allow_duplicate=True),
        Input("ib-unit", "value"),
        State("ib-view", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvb", "clear"),
        *[Output(c, "value", allow_duplicate=True) for c in CONTROLS],
        Output("ib-decoys", "checked", allow_duplicate=True),
        Input({"type": "ib-chip", "key": ALL}, "n_clicks"),
        Input("ib-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    for panel, menu in (("top", "ib-cols"), ("pep", "ib-pep-cols"), ("pre", "ib-pre-cols")):
        app.clientside_callback(
            f"function (value) {{ return window.dash_clientside.mvb.columns('{panel}', value); }}",
            Output("ib-sync", "data", allow_duplicate=True),
            Input(menu, "value"),
            prevent_initial_call=True,
        )
    app.clientside_callback(
        ClientsideFunction("mvb", "defaultColumns"),
        Output("ib-cols", "value", allow_duplicate=True),
        Input("ib-cols-default", "n_clicks"),
        State("ib-cols-defaults", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output(GRID_ID, "getRowsResponse"),
        Output("ib-top-count", "children"),
        Output("ib-top-q", "children"),
        Output("ib-top-help", "label"),
        Output("ib-sig", "data"),
        Output("ib-empty", "children"),
        Output("ib-empty", "className"),
        Output(GRID_ID, "columnDefs"),
        Output("ib-cols", "value", allow_duplicate=True),
        Output("ib-defs", "data"),
        Output("ib-first", "data"),
        Input(GRID_ID, "getRowsRequest"),
        State("ib-view", "data"),
        State("ib-sig", "data"),
        State("ib-cols", "value"),
        State("ib-defs", "data"),
        prevent_initial_call=True,
    )
    def rows(request, store, shown_key, shown_cols, shown_defs):
        t0 = time.perf_counter()
        rs = get_rs()
        f = Filters.from_store(store)
        response, page, total, error, q_column = rows_response(rs, request, store, base)
        key = shown_key_of(f, error)
        log.debug(
            "rows %s-%s in %.0f ms",
            (request or {}).get("startRow"),
            (request or {}).get("endRow"),
            (time.perf_counter() - t0) * 1000.0,
        )
        if key == shown_key:
            # The panel already shows this view (a scroll or a sort): only the rows.
            return response, NO, NO, NO, NO, NO, NO, NO, NO, NO, NO
        if page is not None:
            columns, labels = list(page.rows.columns), dict(page.column_labels)
            description = page.description
        else:
            columns, labels = safe_columns(rs, f.unit)
            description = ""
        active = (page.q_column if page is not None else None) or q_column
        defs_key = defs_key_of(labels, active, f.t)
        defs: Any = NO
        cols = list(shown_cols or [])
        if defs_key != shown_defs:
            # New column definitions only when their tooltips or the q column change.
            defs = column_defs(
                f.unit,
                columns,
                labels,
                role="top",
                experiment=rs.is_experiment,
                q_active=active,
                threshold=f.t,
                winner_matters=winner_matters(f.unit, active),
                scales=scales_of(rs),
                rescorer=rescore_info(rs).label,
                sort=sort_of(request, f),
            )
            if active in {d.get("colId") for d in defs} and active not in cols:
                cols.append(active)
        empty = _empty(rs, f, total, error, description)
        return (
            response,
            fmt(total) if total is not None else "-",
            f"{active} ≤ {stop_label(f.t)}",
            top_help(f, total, active, description, error),
            key,
            empty,
            "ib-empty" + (" ib-empty-on" if empty else ""),
            defs,
            cols if cols != list(shown_cols or []) else NO,
            defs_key,
            first_of(rs, request, response, store, base, error),
        )

    # The child panels of a selection: one request fills both panels of a protein group.
    @app.callback(
        Output("ib-children", "data"),
        Input("ib-need", "data"),
        prevent_initial_call=True,
    )
    def children(need):
        if not need:
            return NO
        t0 = time.perf_counter()
        out = children_payload(get_rs(), base, need)
        log.debug("children %s in %.0f ms", need.get("kind"), (time.perf_counter() - t0) * 1000.0)
        return out

    # Asked ahead: the children of the next row, kept by the browser. The precursor that
    # row would show is read into the detail cache too, so its preview is quick (the
    # card itself is not sent ahead).
    @app.callback(
        Output("ib-prefetched", "data"),
        Input("ib-prefetch", "data"),
        prevent_initial_call=True,
    )
    def prefetch(request):
        needs = list((request or {}).get("needs") or [])[:PREFETCH_MAX]
        if not needs:
            return NO
        rs = get_rs()
        payloads = [children_payload(rs, base, need) for need in needs]
        sel = payloads[0].get("sel") or {}
        if sel.get("cid") is not None:
            panels.warm_detail(rs, str(sel.get("run") or ""), sel["cid"])
        return {"n": (request or {}).get("n"), "payloads": payloads}

    # The preview of the selected precursor, asked for by the browser (with the header
    # threshold of its marks), so neither a threshold change nor a pending selection
    # rebuilds a card nobody waits for.
    @app.callback(
        Output("ib-preview-data", "data"),
        Input("ib-prec", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def preview(prec, scheme):
        if not prec or prec.get("n") is None:
            # Not a request of the browser's (a store that mounts without data sets it
            # to undefined, which Dash reports as a change).
            return NO
        t0 = time.perf_counter()
        rs = get_rs()
        ctx = PageContext(
            rs=rs,
            base=base,
            threshold=parse_threshold(prec.get("t")),
            scheme="dark" if scheme == "dark" else "light",
        )
        ahead.wait_detail(
            rs, str(prec.get("run") or "") if rs.is_experiment else "", prec.get("cid")
        )
        out = preview_answer(ctx, prec)
        log.debug("preview in %.0f ms", (time.perf_counter() - t0) * 1000.0)
        return out

    # "Show the selected row": its position in the first table under the shown sort.
    @app.callback(
        Output("ib-located", "data"),
        Input("ib-locate-req", "data"),
        State("ib-view", "data"),
        prevent_initial_call=True,
    )
    def locate(request, store):
        if not request or not request.get("key"):
            return NO
        index = locate_index(get_rs(), store, str(request["key"]), request.get("sort"))
        return {"n": request.get("n"), "key": request["key"], "index": index}

    @app.callback(
        Output("ib-chips", "children"),
        Output("ib-t", "children"),
        Output("ib-q", "data"),
        Output("ib-q", "value", allow_duplicate=True),
        Output("ib-run", "disabled"),
        Output("ib-quant", "disabled"),
        Output("ib-nfilters", "children"),
        Output("ib-nfilters", "className"),
        Output("ib-fast-note", "children"),
        Output("ib-fast-note", "className"),
        Input("ib-view", "data"),
        State("ib-q", "value"),
        prevent_initial_call=True,
    )
    def chips(store, q_value):
        rs = get_rs()
        f = Filters.from_store(store)
        grouped_exp = rs.is_experiment and f.unit != "precursor"
        default = default_q(f.unit, f.run)
        q_shown = f.q or default
        n = panels.n_popover_filters(f)
        fast = fast_mode(f)
        return (
            panels.chip_row(f, fast=fast),
            f"q ≤ {stop_label(f.t)}",
            panels.q_options(rs.is_experiment, f.run, f.unit, default),
            q_shown if q_shown != q_value else NO,
            grouped_exp and not f.run,
            grouped_exp and not f.quant,
            str(n) if n else "",
            "ib-nfilters" + ("" if n else " ib-hidden"),
            panels.FAST_NOTE if fast else "",
            "ib-fast-note" + ("" if fast else " ib-hidden"),
        )

    @app.callback(
        Output("ib-shell", "children"),
        Input("ib-address", "data"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def rebuild(search, t, scheme):
        """An address the router did not rebuild (the one it built last): build it here."""
        if search is None:
            return NO
        return content(
            PageContext(
                rs=get_rs(),
                base=base,
                threshold=parse_threshold(t),
                scheme="dark" if scheme == "dark" else "light",
                query=query_of(search),
                fasta=get_fasta(),
            )
        )

    @app.callback(
        Output("ib-mod", "data"),
        Input("ib-boot", "data"),
        State("ib-mod", "value"),
    )
    def load_modifications(_unit, value):
        mods = list(modifications(get_rs()))
        if value and value not in mods:
            mods.append(value)
        return mods


def precursor_key(run: str | None, cid: int | None) -> str | None:
    """The grid row id of a precursor (``browser_grid.row_key``), or None."""
    return f"{run or ''}:{cid}" if cid is not None else None


def defs_key_of(labels: Mapping[str, str], q_column: str | None, threshold: float) -> str:
    """What the column definitions depend on: the labels, the q column, the threshold."""
    return json.dumps([dict(labels), q_column, threshold], sort_keys=True)


def shown_key_of(f: Filters, error: str | None) -> str:
    """What the panel title, the alert and the column labels show: the view and its error."""
    return f.signature() + "|" + (error or "")
