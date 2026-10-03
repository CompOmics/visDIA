"""One protein group: its peptides and precursors, its quantities per run, and a search.

The protein page (P1 view 8) reads everything about one protein group here. Most of it
is the data layer's identification tables (:func:`.tables.identification_table`) with
the exact ``protein_group`` and ``base_peptide_id`` filters and no q filter, so the
rows are the same as those of the identification page. In ``psms_scored`` the
``protein_group`` column is the ``protein`` string itself (``write_scored_table`` in
rescore.rs hands the same array to both), so a group is named by that string, and its
members are the proteins separated by ``;``. A decoy group carries ``DECOY_`` and holds
the decoy rows only.

Two parts are derived by the viewer, and say so in their labels:

* the quantity of a peptide in a run (:func:`peptide_run_matrix`): the maximum positive
  ``peptide_quant.quantity`` over that peptide's quantified precursors in the run. It is
  the value the engine's protein rollup takes per base peptide
  (``add_protein_base_quantity`` in quant.rs), and the top-N sum of these values is the
  run's ``protein_group_quant.quantity`` (``rollup_protein_bases``), which the tests
  check on every group. ``in_rollup`` marks the peptides of that sum.
* the identification of a peptide in a run: whether one of its target rows has
  ``run_psm_q <= t`` in that run (the PSM-level q within the run; native values from
  ``scored_combined.parquet`` when match-between-runs ran).

The species of a group comes from the entry-name suffix of its members (``_HUMAN``,
``_YEAST``, ``_ECOLI``): the viewer's rule (:func:`species_of`), not a column.
"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .discovery import ResultSet
from .duck import sql_path
from .errors import ViewerError
from .mbr import mbr_ran
from .quant import lfq_matrix, protein_quant, quant_relation, transfer_relation
from .tables import MAX_LIMIT, TablePage, TableQuery, identification_table

__all__ = [
    "MATRIX_COLUMNS",
    "SPECIES_RULE",
    "ProteinGroup",
    "find_groups",
    "group_from_peptides",
    "group_members",
    "group_peptides",
    "is_decoy_group",
    "matrix_sides",
    "peptide_precursors",
    "peptide_run_matrix",
    "protein_group",
    "quant_by_run",
    "species_of",
]

# A UniProt mnemonic species code: an upper-case letter and 1 to 5 letters or digits.
_SPECIES = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")
SPECIES_RULE = (
    "the viewer's rule, not a column of the data: the species is the entry-name suffix of "
    "each member of the group after its last '_' (HUMAN in ATLA3_HUMAN; DECOY_ removed), "
    "when it looks like a UniProt mnemonic"
)

# The columns of peptide_run_matrix, in order.
MATRIX_COLUMNS = [
    "base_peptide_id",
    "sequence",
    "run",
    "source",
    "n_rows",
    "best_run_psm_q",
    "n_identified",
    "identified",
    "best_cid",
    "n_quant_rows",
    "n_quantified",
    "quantity",
    "quantity_cid",
    "rollup_rank",
    "in_rollup",
    "n_transferred",
    "quantity_from_transfer",
]


def group_members(protein_group: str) -> tuple[str, ...]:
    """The proteins of a group string (``A;B``), as written (a decoy keeps its ``DECOY_``)."""
    return tuple(p.strip() for p in str(protein_group).split(";") if p.strip())


def is_decoy_group(protein_group: str) -> bool:
    """A decoy group: its string starts with ``DECOY_`` (its rows are the decoy rows)."""
    return str(protein_group).startswith("DECOY_")


def species_of(protein_group: str) -> tuple[str, ...]:
    """The species codes of a group's members, in order of first appearance.

    The viewer's rule (:data:`SPECIES_RULE`): the text after the last ``_`` of each
    member, ``DECOY_`` removed, when it is an upper-case UniProt mnemonic of 2 to 6
    characters. A member without one adds nothing.
    """
    out: list[str] = []
    for member in group_members(protein_group):
        name = member[len("DECOY_") :] if member.startswith("DECOY_") else member
        i = name.rfind("_")
        if i <= 0:
            continue
        code = name[i + 1 :]
        if _SPECIES.match(code) and code not in out:
            out.append(code)
    return tuple(out)


def _plain(value: Any) -> Any:
    """A Python value: numpy scalars as Python, NaN and NA as None."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


# --------------------------------------------------------------------------- the group


@dataclass(frozen=True)
class ProteinGroup:
    """A protein group and the data layer's row for it.

    ``row`` is the group's row of the protein-group table without a q filter (the
    winning row of the group, with ``pg_q_value``, ``n_peptides`` at any q, and in a
    single run the ``protein_group_quant`` columns); None when no scored row carries this
    exact string. ``label`` is ``decoy`` for a ``DECOY_`` group, else ``target``.
    """

    protein_group: str
    members: tuple[str, ...]
    label: str
    row: dict[str, Any] | None
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.row is not None

    @property
    def decoy(self) -> bool:
        return self.label == "decoy"

    def get(self, column: str) -> Any:
        return None if self.row is None else self.row.get(column)


def protein_group(rs: ResultSet, group: str) -> ProteinGroup:
    """The data layer's row of one protein group (exact string), or ``row=None``.

    The row comes from the protein-group table with ``protein_group=group``, no q
    filter and both labels, so a decoy group is found too.
    """
    text = str(group)
    page = identification_table(
        rs,
        TableQuery(
            unit="protein_group",
            protein_group=text,
            threshold=None,
            include_decoys=True,
            sort_by="score",
            limit=1,
        ),
    )
    row = None
    if len(page.rows):
        row = {k: _plain(v) for k, v in page.rows.iloc[0].to_dict().items()}
    return ProteinGroup(
        protein_group=text,
        members=group_members(text),
        label="decoy" if is_decoy_group(text) else "target",
        row=row,
        description=page.description,
        labels=dict(page.column_labels),
    )


_ROW_FIELDS = (
    "candidate_id",
    "run",
    "source",
    "peptidoform",
    "charge",
    "label",
    "base_peptide_id",
    "score",
    "pg_q_value",
    "apex_rt",
)


def group_from_peptides(rs: ResultSet, group: str, peptides: TablePage) -> ProteinGroup:
    """The group's row taken from its peptide rows, else :func:`protein_group`.

    The winning row of a group is its highest-scoring row (the engine writes
    ``pg_q_value`` below 1.0 to it), so it is also the best row of its own peptide and is
    one of the rows of :func:`group_peptides`. When one peptide row holds a
    ``pg_q_value`` below 1.0 it is the group's winning row, and ``n_peptides`` is the
    number of peptide rows; otherwise (a winner at the cap of 1.0) the protein-group
    table is asked. The tests check both ways against each other on every group.
    """
    rows = peptides.rows
    if len(rows) and "pg_q_value" in rows:
        below = rows[rows["pg_q_value"].astype("float64") < 1.0]
        if len(below) == 1:
            r = below.iloc[0]
            row = {k: _plain(r.get(k)) for k in _ROW_FIELDS if k in rows.columns}
            row["n_peptides"] = int(peptides.total)
            row["is_winner"] = True
            row["protein_group"] = str(group)
            return ProteinGroup(
                protein_group=str(group),
                members=group_members(group),
                label="decoy" if is_decoy_group(group) else "target",
                row=row,
                description=(
                    "The group's winning row: the one row of the group whose pg_q_value is "
                    "below 1.0, found among the group's peptide rows. " + peptides.description
                ),
                labels={"n_peptides": "peptides of this group (distinct base_peptide_id, any q)"},
            )
    return protein_group(rs, group)


def _all_rows(rs: ResultSet, query: TableQuery) -> TablePage:
    """Every row of a table query, read in blocks of the data layer's largest page."""
    first = identification_table(rs, query)
    if first.total <= len(first.rows):
        return first
    parts = [first.rows]
    offset = len(first.rows)
    while offset < first.total:
        page = identification_table(rs, replace(query, offset=offset))
        if not len(page.rows):
            break
        parts.append(page.rows)
        offset += len(page.rows)
    rows = pd.concat(parts, ignore_index=True)
    return TablePage(
        rows=rows,
        total=first.total,
        query=query,
        description=first.description,
        column_labels=first.column_labels,
        q_column=first.q_column,
    )


def group_peptides(rs: ResultSet, group: str) -> TablePage:
    """Every peptide of a protein group at any q, best score first.

    The peptide table of the data layer with ``protein_group=group`` and no q filter;
    the label follows the group (the decoy rows for a ``DECOY_`` group, else the
    targets). Each row is a ``base_peptide_id``; ``peptide_q_value`` is the engine's
    value of the row shown (its winning row; 1.0 when that row did not win).
    """
    return _all_rows(
        rs,
        TableQuery(
            unit="peptide",
            protein_group=str(group),
            threshold=None,
            include_decoys=is_decoy_group(group),
            sort_by="score",
            descending=True,
            limit=MAX_LIMIT,
        ),
    )


def peptide_precursors(rs: ResultSet, group: str, base_peptide_id: int) -> TablePage:
    """Every scored row of one peptide of a group, at any q, best score first.

    In an experiment the rows of every run, each with its run. The label follows the
    group, so the decoy partner rows (same ``base_peptide_id``, the ``DECOY_`` group) are
    not among a target group's precursors.
    """
    return _all_rows(
        rs,
        TableQuery(
            unit="precursor",
            protein_group=str(group),
            base_peptide_id=int(base_peptide_id),
            threshold=None,
            include_decoys=is_decoy_group(group),
            sort_by="score",
            descending=True,
            limit=MAX_LIMIT,
        ),
    )


# --------------------------------------------------------------------------- search


def find_groups(
    rs: ResultSet, text: str | None, *, threshold: float = 0.01, limit: int = 200
) -> TablePage:
    """Protein groups whose string contains ``text`` (case-insensitive), best score first.

    Target groups at any q, so the validation marks tell which pass. The data layer's
    ``protein`` filter is used: it is a substring of the protein string, which is the
    protein group. Without text the page holds the groups that pass
    ``pg_q_value <= threshold``. ``total`` counts every match; ``rows`` holds the
    first ``limit``.
    """
    words = (text or "").strip()
    if not 0 < int(limit) <= MAX_LIMIT:
        raise ViewerError(f"limit must be from 1 to {MAX_LIMIT}.")
    return identification_table(
        rs,
        TableQuery(
            unit="protein_group",
            protein=words or None,
            threshold=None if words else float(threshold),
            include_decoys=False,
            sort_by="score",
            descending=True,
            limit=int(limit),
        ),
    )


# --------------------------------------------------------------------------- quantities


def _lfq_long(rs: ResultSet) -> pd.DataFrame | None:
    """The protein MaxLFQ table in long form (one row per group and run), read once."""
    if not rs.is_experiment:
        return None

    def read() -> pd.DataFrame | None:
        try:
            return lfq_matrix(rs, "protein", wide=False)
        except ViewerError:
            return None

    return rs.memo(("protein_page", "lfq_long"), read)


def quant_by_run(rs: ResultSet, group: str) -> pd.DataFrame:
    """The quantity of a protein group in each run: protein_group_quant and MaxLFQ.

    One row per run, in run order. ``quantity``, ``quant_status``, ``state`` and
    ``n_peptides`` are the run's ``protein_group_quant`` row (state ``not_selected``
    with no quantity when the run has no row for the group: no peptide passed its quant
    gate). ``lfq`` and ``lfq_n_features`` are the experiment's MaxLFQ protein table
    (``lfq_maxlfq.parquet``; 0.0 there means no feature and is NaN here). With
    match-between-runs, ``n_transferred`` counts the run's quantified precursors of the
    group that are transfers. ``attrs['labels']`` say what each column holds and
    ``attrs['notes']`` why a table could not be read.
    """
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    rollup: Any = None
    top_n: Any = None
    lfq = _lfq_long(rs)
    lfq_rows = lfq[lfq["protein_group"] == group] if lfq is not None else None
    for run in rs.runs:
        rec: dict[str, Any] = {"run": run.name, "source": int(run.index)}
        try:
            pq = protein_quant(rs, run, group)
        except ViewerError as exc:
            notes.append(f"{run.label}: {exc}")
            pq = None
            rec["state"] = None
        if pq is not None:
            rollup = pq.get("rollup", rollup)
            top_n = pq.get("top_n_peptides", top_n)
            rec.update(
                quantity=pq.get("quantity"),
                quant_status=pq.get("quant_status"),
                state=pq.get("state"),
                n_peptides=pq.get("n_peptides"),
            )
            if "n_transferred_precursors" in pq:
                rec["n_transferred"] = pq["n_transferred_precursors"]
        elif "state" not in rec:
            # No row: no peptide of the group passed the run's quant gate, so no
            # quantified precursor of it is a transfer either.
            rec["state"] = "not_selected"
            if mbr_ran(rs):
                rec["n_transferred"] = 0
        if lfq_rows is not None:
            hit = lfq_rows[lfq_rows["source"] == run.index]
            if len(hit):
                rec["lfq"] = _plain(hit["quantity"].iloc[0])
                rec["lfq_n_features"] = _plain(hit["n_features"].iloc[0])
        records.append(rec)
    columns = ["run", "source", "quantity", "quant_status", "state", "n_peptides"]
    if rs.is_experiment:
        columns += ["lfq", "lfq_n_features"]
    if mbr_ran(rs):
        columns.append("n_transferred")
    df = pd.DataFrame.from_records(records, columns=columns)
    for c in ("quantity", "lfq"):
        if c in df:
            df[c] = df[c].astype("float64")
    for c in ("n_peptides", "lfq_n_features", "n_transferred"):
        if c in df:
            df[c] = df[c].astype("Int64")
    if lfq is None and rs.is_experiment:
        notes.append("The MaxLFQ protein table (lfq_maxlfq.parquet) cannot be read.")
    rule = (
        f"{rollup} of the per-peptide maxima"
        + (f" (top {top_n})" if rollup == "TopNSum" and top_n is not None else "")
        if rollup
        else "the rollup of the per-peptide maxima"
    )
    labels = {
        "quantity": f"protein_group_quant.quantity of the run: {rule} (quant.rs); missing "
        "when no peptide of the group passed the run's quant gate, never 0",
        "quant_status": "protein_group_quant.quant_status (engine string)",
        "state": "quantified; not_quantifiable (a row with a null quantity); not_selected "
        "(no protein_group_quant row in the run)",
        "n_peptides": "protein_group_quant.n_peptides: base peptides with a positive quantity "
        "in the run (before the top-N cut)",
        "lfq": "MaxLFQ protein quantity of the run (lfq_maxlfq.parquet); 0.0 in the file is "
        "missing (no feature in the run)",
        "lfq_n_features": "lfq_maxlfq.n_features: precursors of the group with a positive "
        "quantity in any run",
        "n_transferred": "quantified precursors of the group in the run that are "
        "match-between-runs transfers (derived from mbr_transferred.parquet)",
    }
    df.attrs.update(
        labels={c: labels[c] for c in df.columns if c in labels},
        notes=notes,
        rollup=rollup,
        top_n=int(top_n) if isinstance(top_n, int | float) else None,
    )
    return df


def _quant_side(rs: ResultSet, group: str, top_n: int | None, rollup: Any) -> pd.DataFrame:
    """Per (source, base_peptide_id): the run's peptide_quant rows of the group.

    ``quantity`` is the maximum positive quantity, ``quantity_cid`` its candidate, and
    ``rollup_rank`` the rank of that maximum among the group's peptides of the run
    (largest first; an exact tie keeps the lower base_peptide_id first, as the engine's
    stable sort of the per-peptide maxima, collected in base_peptide_id order, does).
    """
    union = quant_relation(
        rs, "peptide_quant", ("candidate_id", "base_peptide_id", "protein_group", "quantity")
    )
    empty = pd.DataFrame(
        columns=[
            "source",
            "base_peptide_id",
            "n_quant_rows",
            "n_quantified",
            "quantity",
            "quantity_cid",
            "quantity_from_transfer",
            "rollup_rank",
        ]
    )
    if union is None:
        return empty
    u_sql, u_params, _ = union
    positive = "q.quantity > 0 AND isfinite(q.quantity)"
    tr = transfer_relation(rs) if mbr_ran(rs) else None
    if tr is not None:
        join = f" LEFT JOIN ({tr[0]}) t ON t.source = q.source AND t.candidate_id = q.candidate_id"
        tr_params = list(tr[1])
        from_tr = (
            f"arg_max(t.candidate_id IS NOT NULL, q.quantity) FILTER (WHERE {positive}) "
            "AS quantity_from_transfer"
        )
    else:
        join, tr_params = "", []
        from_tr = "NULL::BOOLEAN AS quantity_from_transfer"
    sql = (
        "WITH a AS (SELECT q.source, q.base_peptide_id, count(*) AS n_quant_rows, "
        f"count(*) FILTER (WHERE {positive}) AS n_quantified, "
        f"max(q.quantity) FILTER (WHERE {positive}) AS quantity, "
        f"arg_max(q.candidate_id, q.quantity) FILTER (WHERE {positive}) AS quantity_cid, "
        f"{from_tr} "
        f"FROM ({u_sql}) q{join} WHERE q.protein_group = ? GROUP BY 1, 2) "
        "SELECT a.*, CASE WHEN a.quantity IS NULL THEN NULL ELSE row_number() OVER ("
        "PARTITION BY a.source, a.quantity IS NULL ORDER BY a.quantity DESC, "
        "a.base_peptide_id) END AS rollup_rank FROM a ORDER BY 1, 2"
    )
    df = rs.duck.df(sql, [*u_params, *tr_params, str(group)])
    return df if len(df) else empty


def _id_side(rs: ResultSet, group: str, threshold: float) -> pd.DataFrame:
    """Per (source, base_peptide_id): the group's scored rows in each run.

    ``best_run_psm_q`` is the smallest ``run_psm_q`` of the peptide's rows in the run,
    ``n_identified`` the rows with ``run_psm_q <= threshold``, ``best_cid`` the
    candidate of the highest-scoring row (a tie takes the first in file order), and
    ``n_transferred`` the rows that are match-between-runs transfers.
    """
    path = sql_path(rs.scored.require())
    label = "decoy" if is_decoy_group(group) else "target"
    tr = transfer_relation(rs) if mbr_ran(rs) else None
    if tr is not None:
        join = f" LEFT JOIN ({tr[0]}) t ON t.source = s.source AND t.candidate_id = s.candidate_id"
        tr_params = list(tr[1])
        n_tr = "count(t.candidate_id) AS n_transferred"
    else:
        join, tr_params = "", []
        n_tr = "0::BIGINT AS n_transferred"
    sql = (
        "SELECT s.source, s.base_peptide_id, count(*) AS n_rows, "
        "min(s.run_psm_q) AS best_run_psm_q, "
        "count(*) FILTER (WHERE s.run_psm_q <= ?) AS n_identified, "
        "arg_max(s.candidate_id, struct_pack(score := s.score, "
        "rn := -CAST(s.file_row_number AS BIGINT))) "
        f"AS best_cid, {n_tr} "
        "FROM (SELECT source, candidate_id, base_peptide_id, run_psm_q, score, file_row_number "
        "FROM read_parquet(?, file_row_number = true) WHERE protein_group = ? AND label = ?) s"
        f"{join} GROUP BY 1, 2 ORDER BY 1, 2"
    )
    return rs.duck.df(sql, [float(threshold), path, str(group), label, *tr_params])


_SIDES: OrderedDict[tuple[Any, ...], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = (
    OrderedDict()
)
_SIDES_LOCK = threading.Lock()
_SIDES_MAX = 32


def matrix_sides(
    rs: ResultSet, group: str, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The per-run parts of :func:`peptide_run_matrix`: quant_by_run, identification, quant.

    Kept for the last groups and thresholds asked for (a small LRU), so a page can read
    them in a thread while it runs its table queries, and a callback reuses them.
    """
    key = (id(rs), rs.scored.identity(), str(group), float(threshold))
    with _SIDES_LOCK:
        hit = _SIDES.get(key)
        if hit is not None:
            _SIDES.move_to_end(key)
            return hit
    pq_runs = quant_by_run(rs, group)
    ids = _id_side(rs, group, float(threshold))
    quant = _quant_side(rs, group, pq_runs.attrs.get("top_n"), pq_runs.attrs.get("rollup"))
    value = (pq_runs, ids, quant)
    with _SIDES_LOCK:
        _SIDES[key] = value
        while len(_SIDES) > _SIDES_MAX:
            _SIDES.popitem(last=False)
    return value


def peptide_run_matrix(
    rs: ResultSet,
    group: str,
    threshold: float,
    *,
    peptides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The peptides of a protein group by the runs: quantity and identification per run.

    One row per peptide (``base_peptide_id``, every peptide of :func:`group_peptides`,
    in its order) and run (in run order). Identification, from the scored table
    (``scored_combined.parquet`` in an experiment, native when match-between-runs ran):
    ``n_rows`` (scored rows of the peptide in the run, any q), ``best_run_psm_q``,
    ``n_identified`` (rows with ``run_psm_q <= threshold``), ``identified`` and
    ``best_cid`` (the highest-scoring row). Quantity, from the run's ``peptide_quant``:
    ``n_quant_rows``, ``n_quantified``, ``quantity`` (viewer-derived: the maximum
    positive quantity over the peptide's precursors, the value of the engine's protein
    rollup), ``quantity_cid``, ``rollup_rank`` and ``in_rollup`` (in the top-N sum that
    gives ``protein_group_quant.quantity``). With match-between-runs, ``n_transferred``
    and ``quantity_from_transfer`` (the quantity's precursor is a transfer).

    ``attrs['labels']`` describe the columns; ``attrs['rollup']`` and
    ``attrs['top_n']`` give the engine's rollup.
    """
    t = float(threshold)
    if not 0 < t <= 1:
        raise ViewerError(f"the threshold {threshold!r} is not a q value in (0, 1].")
    if peptides is None:
        peptides = group_peptides(rs, group).rows
    pq_runs, ids, quant = matrix_sides(rs, group, t)
    rollup = pq_runs.attrs.get("rollup")
    top_n = pq_runs.attrs.get("top_n")
    runs = pd.DataFrame(
        {"run": [r.name for r in rs.runs], "source": [int(r.index) for r in rs.runs]}
    )
    pep = pd.DataFrame(
        {
            "base_peptide_id": peptides["base_peptide_id"].astype("int64").to_numpy(),
            "sequence": peptides["sequence"].astype(str).to_numpy()
            if "sequence" in peptides
            else "",
            "_order": np.arange(len(peptides)),
        }
    )
    grid = pep.merge(runs, how="cross")
    for side in (ids, quant):
        if len(side):
            side = side.copy()
            side["source"] = side["source"].astype("int64")
            side["base_peptide_id"] = side["base_peptide_id"].astype("int64")
            grid = grid.merge(side, on=["source", "base_peptide_id"], how="left")
    out = pd.DataFrame(index=grid.index)
    out["base_peptide_id"] = grid["base_peptide_id"].astype("int64")
    out["sequence"] = grid["sequence"].astype(str)
    out["run"] = grid["run"].astype(str)
    out["source"] = grid["source"].astype("int64")

    def col(name: str, kind: str) -> pd.Series:
        if name not in grid:
            return pd.Series(pd.NA if kind != "float64" else np.nan, index=grid.index).astype(kind)
        return grid[name].astype(kind)

    out["n_rows"] = col("n_rows", "Int64").fillna(0).astype("int64")
    out["best_run_psm_q"] = col("best_run_psm_q", "float64")
    out["n_identified"] = col("n_identified", "Int64").fillna(0).astype("int64")
    out["identified"] = out["n_identified"] > 0
    out["best_cid"] = col("best_cid", "Int64")
    out["n_quant_rows"] = col("n_quant_rows", "Int64").fillna(0).astype("int64")
    out["n_quantified"] = col("n_quantified", "Int64").fillna(0).astype("int64")
    out["quantity"] = col("quantity", "float64")
    out["quantity_cid"] = col("quantity_cid", "Int64")
    out["rollup_rank"] = col("rollup_rank", "Int64")
    if rollup == "TopNSum" and top_n is not None:
        in_rollup = out["rollup_rank"].le(int(top_n)).fillna(False).astype(bool)
    elif rollup == "Sum":
        in_rollup = out["quantity"].notna()
    else:
        in_rollup = pd.Series(False, index=out.index)
    out["in_rollup"] = in_rollup.to_numpy(dtype=bool)
    out["n_transferred"] = col("n_transferred", "Int64").fillna(0).astype("int64")
    out["quantity_from_transfer"] = col("quantity_from_transfer", "boolean").fillna(False)
    out["quantity_from_transfer"] = out["quantity_from_transfer"].astype(bool)
    out = out.iloc[np.lexsort((grid["source"].to_numpy(), grid["_order"].to_numpy()))]
    out = out.reset_index(drop=True)[MATRIX_COLUMNS]
    tt = f"{t:g}"
    native = " (native, scored_combined.parquet)" if mbr_ran(rs) else ""
    rule = f"top {top_n} sum" if rollup == "TopNSum" and top_n is not None else str(rollup or "")
    labels = {
        "n_rows": "scored rows of the peptide in the run (any q)",
        "best_run_psm_q": f"smallest run_psm_q of the peptide's rows in the run{native}",
        "n_identified": f"rows of the peptide with run_psm_q <= {tt} in the run{native}",
        "identified": f"the peptide has a row with run_psm_q <= {tt} in the run (PSM-level q "
        "within the run; not a peptide-level FDR)",
        "best_cid": "candidate of the peptide's highest-scoring row in the run",
        "n_quant_rows": "peptide_quant rows of the peptide in the run (rows that passed the "
        "run's quant gate)",
        "n_quantified": "those rows with a positive quantity",
        "quantity": "viewer-derived: the largest positive peptide_quant.quantity of the "
        "peptide's precursors in the run, the per-peptide value of the engine's protein "
        "rollup (quant.rs add_protein_base_quantity); missing is never 0",
        "quantity_cid": "candidate of that precursor",
        "rollup_rank": "rank of the quantity among the group's peptides in the run (largest first)",
        "in_rollup": f"viewer-derived: the peptide is in the {rule} of the run that gives "
        "protein_group_quant.quantity (quant.rs rollup_protein_bases)",
        "n_transferred": "rows of the peptide in the run that are match-between-runs "
        "transfers (mbr_transferred.parquet)",
        "quantity_from_transfer": "the precursor of the quantity is a match-between-runs transfer",
    }
    out.attrs.update(
        labels=labels,
        threshold=t,
        rollup=rollup,
        top_n=top_n,
        mbr=mbr_ran(rs),
    )
    return out
