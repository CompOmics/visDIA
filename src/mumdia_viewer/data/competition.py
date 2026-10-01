"""Target-decoy competition around one identification.

Two views of "the decoy of this identification":

* **The base-peptide competition.** ``peptide_q_value`` is a picked target-decoy
  competition over ``base_peptide_id``: a target peptide and its decoy share the key,
  and the best-scoring row of the key wins. In an experiment the competition is
  experiment-wide, so the rows come from every run of the pooled scored table.
* **The exact decoy partner.** In imported (DIA-NN) libraries each ``peptidoform_id``
  holds exactly one target and one decoy precursor; the partner of a candidate is the
  other row of its ``peptidoform_id``. FASTA-built libraries have no such key
  (``peptidoform_id`` is unique per row), which :func:`exact_partner` reports.

The winner of a key follows the engine (``grouped_q``, rescore.rs): highest score, a
decoy wins an exact score tie, then the earlier row of the pooled table. In entrapment
mode decoys do not compete and an entrapment row wins a tie; the caller passes the
spike-in test. The winner flags are the viewer's application of that rule, labelled
as such; on every fixture and on the Astral runs they equal the engine's sparse q.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .artifacts import Artifact
from .discovery import ResultSet
from .duck import sql_path
from .rescore import RescoreInfo, precursor_q_is_precursor_unit

# Unit of each q column in target-decoy mode: (what the estimate is computed over, grouped?).
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

WINNER_SOURCE = (
    "viewer-derived: the engine's winner rule (score, decoy first on a tie, file row order) "
    "applied to the pooled scored table"
)


@dataclass(frozen=True)
class QValue:
    """One q column of one scored row, with its unit.

    For a grouped column, ``winner`` says whether this row is its group's winner (the
    only row that carries the group's q); a non-winner stores 1.0, which is not a q
    value, so :attr:`display_value` is None for it.
    """

    column: str
    value: float
    unit: str
    grouped: bool
    winner: bool | None
    scope: str
    winner_source: str | None = None

    @property
    def display_value(self) -> float | None:
        if self.grouped and self.winner is False:
            return None
        return self.value

    @property
    def text(self) -> str:
        if self.grouped and self.winner is False:
            return (
                f"{self.column}: not the winner of its group, so no q is stored for this row "
                f"(the column holds 1.0); the group's q is on its winning row. Unit: {self.unit}"
            )
        return f"{self.column} = {self.value:.6g}; {self.unit} ({self.scope})"


def q_values(
    rs: ResultSet,
    row: dict,
    winners: dict[str, bool | None] | None = None,
    *,
    info: RescoreInfo | None = None,
) -> list[QValue]:
    """The q columns of a scored row with their units.

    ``winners`` gives the winner flag of each grouped column (from :func:`competition`);
    without it a grouped value below 1 marks the winner. ``info`` (the rescorer) adapts
    the units: in entrapment mode every column is an entrapment estimate, and under
    ``compete.group_by = base_peptide`` ``precursor_q`` counts base peptides.
    """
    scope = "experiment-wide" if rs.is_experiment else "this run"
    entrapment = info is not None and info.mode == "entrapment"
    base_peptide = info is not None and not precursor_q_is_precursor_unit(info)
    out = []
    for column, (unit, grouped) in Q_COLUMNS.items():
        if column not in row or row[column] is None:
            continue
        value = float(row[column])
        col_scope = "this run" if column == "run_psm_q" else scope
        if entrapment:
            unit = (
                "entrapment estimate (ratio x spike-ins + 1) / real targets over "
                + unit.replace(", target-decoy re-run per run", ", re-run per run").replace(
                    " (picked target-decoy competition)", ""
                )
                + ("; decoys do not compete" if grouped else "")
            )
        if column == "precursor_q" and base_peptide:
            unit += (
                "; compete.group_by = base_peptide left about one form per peptide, so this "
                "is a base-peptide unit"
            )
        winner, source = None, None
        if grouped:
            flag = (winners or {}).get(column)
            if flag is None:
                winner, source = value < 1.0, "the stored value (below 1 marks the winner)"
            else:
                winner, source = flag, WINNER_SOURCE
        out.append(QValue(column, value, unit, grouped, winner, col_scope, source))
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

    Returns the rows, with ``is_this_row``, ``wins_peptide`` (the winner of the
    base-peptide key) and ``wins_precursor`` (the winner of each row's own
    ``(peptidoform, charge)`` group, set for every precursor in the frame), and the
    winner flags of this row for ``peptide_q_value`` and ``precursor_q``. Every
    precursor group is a subset of the base-peptide rows, so one query serves both.
    Pass ``entrapment`` (the spike-in test) in entrapment mode, where decoys do not
    compete and a spike-in wins a tie.
    """
    df = base_peptide_rows(rs, base_peptide_id, entrapment=entrapment)
    flags_col = df["is_entrapment"].astype(bool) if entrapment is not None else None
    df["is_this_row"] = (df["source"] == source) & (df["candidate_id"] == candidate_id)
    win = winner_index(df, entrapment=flags_col)
    df["wins_peptide"] = df.index == win
    df["wins_precursor"] = False
    for _, group in df.groupby(["peptidoform", "charge"], sort=False):
        pwin = winner_index(group, entrapment=flags_col)
        if pwin is not None:
            df.loc[pwin, "wins_precursor"] = True
    flags: dict[str, bool | None] = {"peptide_q_value": None, "precursor_q": None}
    this = df[df["is_this_row"]]
    if not this.empty:
        flags["peptide_q_value"] = bool(this["wins_peptide"].iloc[0])
        flags["precursor_q"] = bool(this["wins_precursor"].iloc[0])
    return df, flags


# --------------------------------------------------------------------------- partner


@dataclass(frozen=True)
class PartnerLookup:
    """The exact library partner of one candidate, or why there is none."""

    candidate_id: int | None
    reason: str
    library: str | None


def library_precursors(rs: ResultSet) -> Artifact | None:
    """The searched library precursor table (candidate_id is its row index)."""
    lib = rs.extra.get("fragment_library_precursors")
    if lib is None and rs.runs:
        lib = rs.runs[0].artifact("fragment_library_precursors")
    return lib if lib is not None and lib.usable else None


def exact_partner(
    rs: ResultSet,
    candidate_id: int,
    *,
    peptidoform: str | None = None,
    charge: int | None = None,
) -> PartnerLookup:
    """The other row of the candidate's library ``peptidoform_id``, when it is exact.

    One query returns every library row of the candidate's ``peptidoform_id``. The
    partner is exact when that key holds exactly the candidate and one row of the other
    label. When ``peptidoform`` and ``charge`` are given (the scored row's), the
    candidate's library row must carry them; otherwise the table is not the library the
    run searched (for example a wrongly remapped input) and no partner is shown.
    """
    lib = library_precursors(rs)
    if lib is None:
        return PartnerLookup(None, "the searched library precursor table is not available", None)
    handle = lib.parquet()
    if "peptidoform_id" not in handle.schema.names:
        return PartnerLookup(None, "the library has no peptidoform_id column", str(lib.path))
    rows = rs.duck.rows(
        "SELECT candidate_id, peptidoform_id, label, peptidoform, charge "
        "FROM read_parquet($p) WHERE peptidoform_id = "
        "(SELECT peptidoform_id FROM read_parquet($p) WHERE candidate_id = $c)",
        {"p": sql_path(lib.path), "c": int(candidate_id)},
    )
    own = [r for r in rows if int(r[0]) == int(candidate_id)]
    if len(own) != 1:
        return PartnerLookup(
            None, f"candidate {candidate_id} is not a row of the library table", str(lib.path)
        )
    _, _, label, lib_peptidoform, lib_charge = own[0]
    if (peptidoform is not None and lib_peptidoform != peptidoform) or (
        charge is not None and int(lib_charge) != int(charge)
    ):
        return PartnerLookup(
            None,
            f"the library row of candidate {candidate_id} is {lib_peptidoform}/{lib_charge}, "
            f"not the scored {peptidoform}/{charge}: this table is not the library the run "
            "searched",
            str(lib.path),
        )
    others = [r for r in rows if int(r[0]) != int(candidate_id)]
    if not others:
        return PartnerLookup(
            None,
            "no other library row shares this peptidoform_id (a FASTA-built library has no "
            "exact partner key; see the base-peptide competition)",
            str(lib.path),
        )
    if len(others) > 1 or others[0][2] == label:
        return PartnerLookup(
            None,
            f"this peptidoform_id holds {len(rows)} library rows, not one target and one decoy",
            str(lib.path),
        )
    return PartnerLookup(
        int(others[0][0]),
        "exact partner: the other row of the library peptidoform_id",
        str(lib.path),
    )


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
