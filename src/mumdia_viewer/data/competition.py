"""Target-decoy competition around one identification.

Two views of "the decoy of this identification":

* **The base-peptide competition.** ``peptide_q_value`` is a picked target-decoy
  competition over ``base_peptide_id``: a target peptide and its decoy share the key,
  and the best-scoring row of the key wins. In an experiment the competition is
  experiment-wide, so the rows come from every run of the pooled scored table.
* **The exact decoy partner.** Each library ``peptidoform_id`` holds exactly one target
  and one decoy precursor in imported (DIA-NN) libraries; the partner of a candidate is
  the other row of its ``peptidoform_id``. FASTA-built libraries have no such key, which
  :func:`partner_map` detects and reports.

The winner of a key follows the engine (``grouped_q``, rescore.rs): highest score, a
decoy wins an exact score tie, then the earlier row of the pooled table. In entrapment
mode decoys do not compete and an entrapment row wins a tie; the caller passes the
entrapment classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .artifacts import Artifact
from .discovery import ResultSet
from .duck import sql_path

# Unit of each q column: (what the estimate is computed over, grouped?).
Q_COLUMNS: dict[str, tuple[str, bool]] = {
    "q_value": ("PSM rows pooled over the whole rescore", False),
    "experiment_psm_q": ("PSM rows pooled over the whole rescore (equals q_value)", False),
    "global_q_value": ("alias of the pooled PSM q (q_value before any MBR update)", False),
    "run_psm_q": ("PSM rows of this run only, target-decoy re-run per run", False),
    "precursor_q": ("the best row of each (peptidoform, charge)", True),
    "peptide_q_value": (
        "the best row of each base_peptide_id (picked target-decoy competition)",
        True,
    ),
    "pg_q_value": ("the best row of each protein_group", True),
}

GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "precursor_q": ("peptidoform", "charge"),
    "peptide_q_value": ("base_peptide_id",),
    "pg_q_value": ("protein_group",),
}


@dataclass(frozen=True)
class QValue:
    """One q column of one scored row, with its unit."""

    column: str
    value: float
    unit: str
    grouped: bool
    winner: bool | None
    scope: str

    @property
    def text(self) -> str:
        if self.grouped and self.winner is False:
            return f"{self.column}: not the winner of its group (stored 1.0); {self.unit}"
        return f"{self.column} = {self.value:.6g}; {self.unit} ({self.scope})"


def q_values(
    rs: ResultSet, row: dict, winners: dict[str, bool | None] | None = None
) -> list[QValue]:
    """The q columns of a scored row with their units.

    ``winners`` gives the winner flag of each grouped column (from :func:`competition`).
    """
    scope = "experiment-wide" if rs.is_experiment else "this run"
    out = []
    for column, (unit, grouped) in Q_COLUMNS.items():
        if column not in row or row[column] is None:
            continue
        value = float(row[column])
        col_scope = "this run" if column == "run_psm_q" else scope
        winner = None
        if grouped:
            winner = (winners or {}).get(column)
            if winner is None:
                winner = value < 1.0
        out.append(QValue(column, value, unit, grouped, winner, col_scope))
    return out


def base_peptide_rows(
    rs: ResultSet,
    base_peptide_id: int,
    *,
    entrapment: tuple[str, dict] | None = None,
) -> pd.DataFrame:
    """Every pooled scored row of one base peptide (targets and decoys, all runs).

    Columns include ``file_row_number`` (the engine's tie-break order) and ``run``.
    ``entrapment`` is the spike-in test as (SQL expression, named parameters), from
    :func:`mumdia_viewer.data.entrapment.entrapment_expr`; it adds ``is_entrapment``.
    """
    params: dict = {"path": sql_path(rs.scored.require()), "bpid": int(base_peptide_id)}
    extra = ""
    if entrapment is not None:
        expr, ent_params = entrapment
        extra = f", {expr} AS is_entrapment"
        params.update(ent_params)
    df = rs.duck.df(
        "SELECT file_row_number, source, candidate_id, peptidoform, charge, label, protein, "
        "protein_group, score, prelim_score, q_value, run_psm_q, precursor_q, "
        f"peptide_q_value, pg_q_value, apex_rt{extra} "
        "FROM read_parquet($path, file_row_number = true) WHERE base_peptide_id = $bpid "
        "ORDER BY score DESC",
        params,
    )
    names = {r.index: r.label for r in rs.runs}
    df["run"] = df["source"].map(lambda s: names.get(int(s), str(s)))
    return df


def winner_index(df: pd.DataFrame, *, entrapment: pd.Series | None = None) -> int | None:
    """Index label of the engine's winner among ``df`` rows (one competition key).

    Decoy mode: highest score, decoy first on an exact tie, then lower
    ``file_row_number``. Entrapment mode (``entrapment`` given, True for spike-ins):
    decoys are excluded and an entrapment row wins a tie.
    """
    if df.empty:
        return None
    frame = df.copy()
    if entrapment is not None:
        frame = frame[frame["label"] != "decoy"]
        if frame.empty:
            return None
        tie = entrapment.reindex(frame.index).fillna(False).astype(bool)
    else:
        tie = frame["label"] == "decoy"
    frame = frame.assign(_tie=tie.astype(int))
    frame = frame.sort_values(
        ["score", "_tie", "file_row_number"], ascending=[False, False, True], kind="mergesort"
    )
    return frame.index[0]


def competition(
    rs: ResultSet,
    source: int,
    candidate_id: int,
    base_peptide_id: int,
    *,
    entrapment: tuple[str, dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, bool | None]]:
    """The base-peptide competition of one scored row and its winner flags.

    Returns the rows (with ``is_this_row``, ``wins_peptide`` and ``wins_precursor``) and
    the winner flag of this row for ``peptide_q_value`` and ``precursor_q``. The
    precursor group is a subset of the base-peptide rows, so one query serves both.
    """
    df = base_peptide_rows(rs, base_peptide_id, entrapment=entrapment)
    flags_col = df["is_entrapment"].astype(bool) if entrapment is not None else None
    df["is_this_row"] = (df["source"] == source) & (df["candidate_id"] == candidate_id)
    win = winner_index(df, entrapment=flags_col)
    df["wins_peptide"] = df.index == win
    df["wins_precursor"] = False
    flags: dict[str, bool | None] = {"peptide_q_value": None, "precursor_q": None}
    this = df[df["is_this_row"]]
    if not this.empty:
        flags["peptide_q_value"] = bool(this["wins_peptide"].iloc[0])
        key = (this["peptidoform"].iloc[0], int(this["charge"].iloc[0]))
        group = df[(df["peptidoform"] == key[0]) & (df["charge"] == key[1])]
        pwin = winner_index(group, entrapment=flags_col)
        df.loc[df.index == pwin, "wins_precursor"] = True
        flags["precursor_q"] = bool(df.loc[this.index[0], "wins_precursor"])
    return df, flags


_VALID = "exact partner: the other row of the library peptidoform_id"
_INVALID = (
    "the library does not pair each peptidoform_id with exactly one target and one decoy "
    "(a FASTA-built library has no exact partner key)"
)


@dataclass(frozen=True)
class PartnerMap:
    """``partner[cid]`` is the candidate sharing the library ``peptidoform_id`` (or -1)."""

    partner: np.ndarray | None
    valid: bool
    reason: str
    library: str | None


def _library(rs: ResultSet) -> Artifact | None:
    lib = rs.extra.get("fragment_library_precursors")
    if lib is None and rs.runs:
        lib = rs.runs[0].artifact("fragment_library_precursors")
    return lib if lib is not None and lib.usable else None


def partner_map(rs: ResultSet) -> PartnerMap:
    """Build (or load) the exact target-decoy partner map of the searched library.

    Valid only when every ``peptidoform_id`` has exactly one target and one decoy row
    and ``candidate_id`` equals the row index; otherwise ``valid`` is False with the
    reason, and no partner is shown.
    """
    memo = rs._memo.get("partner_map")
    if memo is not None:
        return memo
    lib = _library(rs)
    if lib is None:
        result = PartnerMap(
            None, False, "the searched library precursor table is not available", None
        )
        rs._memo["partner_map"] = result
        return result
    handle = lib.parquet()
    if "peptidoform_id" not in handle.schema.names:
        result = PartnerMap(None, False, "the library has no peptidoform_id column", str(lib.path))
        rs._memo["partner_map"] = result
        return result
    identity = lib.identity()
    cached = rs.cache.load_arrays(identity, "partner_map_v1")
    if cached is not None:
        valid = bool(cached["valid"][0])
        reason = _VALID if valid else _INVALID
        result = PartnerMap(cached["partner"] if valid else None, valid, reason, str(lib.path))
        rs._memo["partner_map"] = result
        return result
    table = handle.read(["candidate_id", "peptidoform_id", "label"])
    cid = table.column("candidate_id").to_numpy().astype(np.int64)
    pid = table.column("peptidoform_id").to_numpy().astype(np.int64)
    is_decoy = np.asarray(table.column("label").to_pylist(), dtype=object) == "decoy"
    n = cid.size
    valid = bool(np.array_equal(cid, np.arange(n)))
    partner = np.full(n, -1, dtype=np.int64)
    if valid:
        order = np.argsort(pid, kind="stable")
        sp = pid[order]
        valid = n % 2 == 0 and bool(np.all(sp[0::2] == sp[1::2]))
        if valid and n > 2:
            valid = bool(np.all(sp[2::2] != sp[1:-1:2]))
        if valid:
            a, b = order[0::2], order[1::2]
            valid = bool(np.all(is_decoy[a] != is_decoy[b]))
            partner[a], partner[b] = b, a
    rs.cache.save_arrays(
        identity, "partner_map_v1", partner=partner.astype(np.int64), valid=np.array([valid])
    )
    reason = _VALID if valid else _INVALID
    result = PartnerMap(partner if valid else None, valid, reason, str(lib.path))
    rs._memo["partner_map"] = result
    return result


def exact_partner(rs: ResultSet, candidate_id: int) -> int | None:
    """The candidate_id of the exact library partner, or None when there is none."""
    pm = partner_map(rs)
    if not pm.valid or pm.partner is None or not 0 <= candidate_id < pm.partner.size:
        return None
    other = int(pm.partner[candidate_id])
    return other if other >= 0 else None


def scored_rows_of(rs: ResultSet, candidate_id: int) -> pd.DataFrame:
    """Every pooled scored row of one candidate (one per run where it was scored)."""
    df = rs.duck.df(
        "SELECT source, candidate_id, peptidoform, charge, label, protein, score, q_value, "
        "run_psm_q, precursor_q, peptide_q_value, pg_q_value, apex_rt "
        "FROM read_parquet(?) WHERE candidate_id = ? ORDER BY source",
        [sql_path(rs.scored.require()), int(candidate_id)],
    )
    names = {r.index: r.label for r in rs.runs}
    df["run"] = df["source"].map(lambda s: names.get(int(s), str(s)))
    return df
