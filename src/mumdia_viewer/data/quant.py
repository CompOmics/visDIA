"""Quantification: the quant gate, per-candidate quant states, protein quant and LFQ.

What the engine writes (MuMDIA 0.5.0):

* ``peptide_quant.parquet`` has one row per scored row that passed the quant gate
  ``label != 'decoy' AND (is_transferred OR (isfinite(q) AND q <= q_threshold))``, where
  ``q`` is the column that ``quant.q_filter`` selects. No row means "not selected for
  quant". A row with a null ``quantity`` means "not quantifiable", and ``quant_status``
  says why. Neither is ever shown as zero.
* A single run honours the configured ``quant.q_filter``. Under the default
  ``PeptideQ`` only the winning precursor of each accepted base peptide is quantified,
  because ``peptide_q_value`` is set on the winning row only. ``run-experiment`` forces
  ``PsmQ`` (the pooled ``q_value``) for every run and records the substitution in
  ``experiment.quant_q_filter = {configured, effective}``.
* No quant or LFQ table flags match-between-runs (MBR) transfers. They are flagged here
  by joining ``mbr_transferred.parquet`` on ``(source, candidate_id)``, and only when the
  experiment manifest records that MBR ran (:func:`.mbr.mbr_ran`). A transfer table left
  over from an earlier run into the same directory is never used.
* The LFQ tables are dense: one row per key and run. An LFQ ``quantity`` of 0.0 means
  that the run has no feature for the key (missing), not a zero abundance; it is
  returned as NaN.
* A quant or LFQ table with a schema version this viewer does not read is refused with
  the version error that discovery recorded (:class:`SchemaVersionError`); a table that
  is absent is refused with the location where it was expected.

Every function reads the parquet files through DuckDB with a column projection. Nothing
is written anywhere.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from .artifacts import Artifact
from .discovery import RUN_FILES, ResultSet, Run
from .duck import sql_ident, sql_path
from .entrapment import spike_in_condition
from .errors import ArtifactNotFound, ViewerError
from .mbr import mbr_info, mbr_ran
from .reports import normalise_enum
from .rescore import rescore_info

QuantStateName = Literal["quantified", "not_quantifiable", "not_selected"]
LfqLevel = Literal["protein", "peptide", "precursor"]

# Plain descriptions of the engine's quant_status strings (quant.rs). Strings that are
# not listed here are shown verbatim.
PEPTIDE_STATUS: dict[str, str] = {
    "quantified": "quantified: the sum of the top-N positive fragment areas over the "
    "integration window",
    "no_fragment_traces": "not quantifiable: the candidate has no fragment chromatogram "
    "rows in the tables that quant read",
    "no_positive_fragment_area": "not quantifiable: no fragment has a finite area above zero",
    "no_fragments_selected": "not quantifiable: no fragment was selected "
    "(quant.top_n_fragments is 0)",
    "nonfinite_quantity": "not quantifiable: the sum of the selected fragment areas is not "
    "finite or not above zero",
}

PROTEIN_STATUS: dict[str, str] = {
    "quantified": "quantified: the top-N sum of the maximum quantity of each base peptide "
    "of the group",
    "no_quantifiable_peptide": "not quantifiable: no peptide of the group has a positive quantity",
}

# quant.q_filter (normalised spelling) -> the scored column it gates on.
Q_FILTER_COLUMN: dict[str, str] = {
    "peptideq": "peptide_q_value",
    "precursorq": "precursor_q",
    "psmq": "q_value",
    "runpsmq": "run_psm_q",
}
_ENGINE_SPELLING: dict[str, str] = {
    "peptideq": "PeptideQ",
    "precursorq": "PrecursorQ",
    "psmq": "PsmQ",
    "runpsmq": "RunPsmQ",
}
# The grouped q columns: set on the one winning row of each group, 1.0 elsewhere.
GROUPED_Q_COLUMNS: dict[str, str] = {
    "peptide_q_value": "base_peptide_id",
    "precursor_q": "(peptidoform, charge)",
    "pg_q_value": "protein_group",
}

# Label of the protein-level LFQ n_features column (quant.rs: distinct precursors with a
# positive quantity in any run of the group; constant across the group's rows).
N_FEATURES_LABEL = "precursors in any run"


def describe_status(status: str | None, table: str = "peptide_quant") -> str:
    """The plain description of a ``quant_status`` string; unknown strings verbatim."""
    if status is None:
        return "no quant_status"
    known = PROTEIN_STATUS if table == "protein_group_quant" else PEPTIDE_STATUS
    return known.get(str(status), str(status))


def engine_q_filter(value: Any) -> str | None:
    """A ``quant.q_filter`` value in the engine's Debug spelling (``run_psm_q`` -> ``RunPsmQ``).

    Unknown values are returned verbatim.
    """
    if value is None:
        return None
    return _ENGINE_SPELLING.get(normalise_enum(value), str(value))


def q_filter_column(value: Any) -> str | None:
    """The scored column a ``quant.q_filter`` value gates on, or None when unknown."""
    if value is None:
        return None
    return Q_FILTER_COLUMN.get(normalise_enum(value))


# --------------------------------------------------------------------------- runs


def resolve_run(rs: ResultSet, run: Run | str | int | None) -> Run:
    """A run of ``rs`` by object, name or ``source`` index.

    ``None`` names the only run of a single-run result; an experiment needs a run.
    """
    if isinstance(run, Run):
        return run
    if run is None or (not rs.is_experiment and run in ("", 0, "run")):
        if len(rs.runs) == 1:
            return rs.runs[0]
        names = ", ".join(r.label for r in rs.runs)
        raise ViewerError(f"this experiment has {len(rs.runs)} runs ({names}); name one.")
    if isinstance(run, bool) or not isinstance(run, str | int | np.integer):
        raise ViewerError(f"a run is named by its name or its source index, not {run!r}.")
    try:
        return rs.run(int(run) if isinstance(run, np.integer) else run)
    except KeyError as exc:
        raise ViewerError(str(exc.args[0]) if exc.args else str(exc)) from None


def _usable(artifact: Artifact | None) -> bool:
    return artifact is not None and artifact.usable


def _expected_path(rs: ResultSet, artifact: Artifact) -> str:
    """Where a recorded artifact should be in the opened directory (re-rooted), else as recorded."""
    recorded = artifact.recorded_path
    if recorded:
        parts = rs.resolver.relative_parts(recorded)
        if parts is not None:
            return str(rs.root.joinpath(*parts))
        return recorded
    return artifact.key


def artifact_problem(
    rs: ResultSet, artifact: Artifact | None, name: str, expected: Path | None = None
) -> str | None:
    """Why an artifact cannot be read, in words, or None when it can.

    ``name`` is the file name used in the message; ``expected`` is where a file that
    discovery did not find was looked for. An artifact with an unsupported schema
    version returns the error that discovery recorded, which names the file, the
    version found and the versions this viewer reads.
    """
    if artifact is None:
        where = f" (looked for {expected})" if expected is not None else ""
        return f"{name} was not found{where}."
    if artifact.error is not None:
        return artifact.error
    if not artifact.present or artifact.path is None:
        text = f"{name} is missing: {_expected_path(rs, artifact)} does not exist"
        if artifact.recorded_path:
            text += f" (recorded as {artifact.recorded_path})"
        return text + "."
    return None


def table_problem(rs: ResultSet, run: Run, kind: str) -> str | None:
    """Why the ``kind`` table of ``run`` (for example ``peptide_quant``) cannot be read."""
    name = RUN_FILES.get(kind, f"{kind}.parquet")
    return artifact_problem(rs, run.artifact(kind), name, run.root / name)


def require_artifact(
    rs: ResultSet, artifact: Artifact | None, what: str, name: str, expected: Path | None
) -> Artifact:
    """The artifact when it can be read; otherwise raise a clear error.

    An unsupported schema version raises :class:`SchemaVersionError` with the message
    that discovery recorded (``Artifact.require``). An absent file raises
    :class:`ArtifactNotFound` with the location where it was expected.
    """
    if artifact is not None and artifact.present and artifact.error is not None:
        artifact.require()  # raises SchemaVersionError(artifact.error)
    problem = artifact_problem(rs, artifact, name, expected)
    if problem is not None or artifact is None:
        raise ArtifactNotFound(f"{what} cannot be read: {problem}")
    artifact.parquet()  # checks the column contract once
    return artifact


def _run_table(rs: ResultSet, run: Run, kind: str) -> Artifact:
    name = RUN_FILES.get(kind, f"{kind}.parquet")
    return require_artifact(
        rs, run.artifact(kind), f"the {kind} table of {run.label}", name, run.root / name
    )


def run_scored(rs: ResultSet, run: Run) -> tuple[Artifact, bool]:
    """The scored table quant read for ``run``, and whether it holds other runs' rows.

    A run directory of an experiment has its own ``scored.parquet`` (the split of the
    table quant read: ``scored_for_quant``). When it is missing, the pooled table is
    used with a ``source`` filter.
    """
    own = run.artifact("psms_scored")
    if _usable(own):
        assert own is not None
        return own, own is rs.scored and rs.is_experiment
    return rs.scored, rs.is_experiment


# --------------------------------------------------------------------------- MBR

# ``mbr_ran`` is :func:`.mbr.mbr_ran`, imported above: only the manifest says whether
# match-between-runs ran (``experiment.mbr``, else ``model_identities.mbr``). A
# ``mbr_transferred.parquet`` or ``scored_mbr.parquet`` left over from an earlier run into
# the same directory is not a signal, and no function of this module reads it.


def transfer_relation(rs: ResultSet) -> tuple[str, list[Any]] | None:
    """SQL selecting ``(source, candidate_id, transfer_q)``, one row per accepted transfer.

    None unless match-between-runs ran (:func:`.mbr.mbr_ran`): a transfer table left
    over from an earlier run into the same directory is never read. When MBR ran, the
    source is the transfer table of :func:`.mbr.mbr_info`, ``mbr_transferred.parquet``
    (a pair accepted twice keeps its last row, as the worker does). Without that file
    the flagged rows of ``scored_for_quant`` (``scored_mbr.parquet``) are used; None
    when neither can be read. A NaN ``transfer_q`` is returned as null.
    """
    info = mbr_info(rs)
    if not info.ran:
        return None
    art = info.transfers
    if _usable(art):
        assert art is not None
        art.parquet()
        sql = (
            "SELECT source::UINTEGER AS source, candidate_id::UINTEGER AS candidate_id, "
            "CASE WHEN isnan(arg_max(transfer_q, file_row_number)) THEN NULL "
            "ELSE arg_max(transfer_q, file_row_number) END AS transfer_q "
            "FROM read_parquet(?, file_row_number = true) "
            "WHERE source IS NOT NULL AND candidate_id IS NOT NULL GROUP BY 1, 2"
        )
        return sql, [sql_path(art.require())]
    sfq = info.scored_for_quant
    if sfq is not None and sfq.usable and sfq.parquet().has_column("is_transferred"):
        tq = "transfer_q" if sfq.parquet().has_column("transfer_q") else "NULL::DOUBLE"
        sql = (
            "SELECT source::UINTEGER AS source, candidate_id::UINTEGER AS candidate_id, "
            f"CASE WHEN isnan(any_value({tq})) THEN NULL ELSE any_value({tq}) END "
            "AS transfer_q FROM read_parquet(?) WHERE coalesce(is_transferred, false) "
            "GROUP BY 1, 2"
        )
        return sql, [sql_path(sfq.require())]
    return None


# --------------------------------------------------------------------------- quant tables


def quant_relation(
    rs: ResultSet, kind: str, columns: Sequence[str]
) -> tuple[str, list[Any], list[int]] | None:
    """A UNION ALL of every run's ``kind`` table (``peptide_quant`` or
    ``protein_group_quant``) with a ``source`` column added from the run order.

    Returns (sql, params, sources with a table), or None when no run has one.
    """
    parts: list[str] = []
    params: list[Any] = []
    sources: list[int] = []
    cols = ", ".join(sql_ident(c) for c in columns)
    for run in rs.runs:
        art = run.artifact(kind)
        if not _usable(art):
            continue
        assert art is not None
        art.parquet()  # checks the column contract once
        parts.append(f"SELECT {int(run.index)}::UINTEGER AS source, {cols} FROM read_parquet(?)")
        params.append(sql_path(art.require()))
        sources.append(int(run.index))
    if not parts:
        return None
    return " UNION ALL ".join(parts), params, sources


def quant_table_problems(rs: ResultSet, kind: str) -> list[tuple[Run, str]]:
    """The runs whose ``kind`` table :func:`quant_relation` leaves out, with the reason."""
    out: list[tuple[Run, str]] = []
    for run in rs.runs:
        problem = table_problem(rs, run, kind)
        if problem is not None:
            out.append((run, problem))
    return out


@dataclass(frozen=True)
class QuantGate:
    """The rule that selected the scored rows of one run for quant.

    ``q_filter``, ``configured`` and ``effective`` use the engine's Debug spelling
    (``PsmQ``). ``q_column`` is the scored column the gate compares with ``threshold``.
    ``source`` names where the effective values were read.
    """

    q_filter: str | None
    q_column: str | None
    threshold: float | None
    source: str
    configured: str | None
    effective: str | None
    note: str
    transfers_admitted: bool = False

    @property
    def label(self) -> str:
        if self.q_column is None or self.threshold is None:
            return "quant gate not recorded"
        text = f"target rows with {self.q_column} <= {self.threshold:g}"
        if self.transfers_admitted:
            text += ", plus match-between-runs transfers (admitted whatever their q)"
        return text

    @property
    def key(self) -> tuple[Any, ...]:
        """The values that define the gate (runs with equal keys share one description)."""
        return (
            self.q_column,
            self.threshold,
            self.effective,
            self.configured,
            self.transfers_admitted,
        )

    def describe(self, experiment: bool) -> str:
        """The scored rows that quant selected, and why that column, as a noun phrase.

        Example: ``the target rows with q_value <= 0.01 (PsmQ: the pooled PSM-level q,
        experiment-wide, not run_psm_q; run-experiment forces PsmQ, the configured
        RunPsmQ is not applied)``.
        """
        if self.q_column is None or self.threshold is None:
            return "the rows of a quant gate that is not recorded"
        text = f"the {self.label}"
        spelled = self.effective or "the recorded q_filter"
        if self.q_column == "peptide_q_value" and not experiment:
            why = f"{spelled}: the winning precursor of each accepted base peptide"
        elif self.q_column == "q_value" and experiment:
            why = f"{spelled}: the pooled PSM-level q, experiment-wide, not run_psm_q"
        elif self.q_column == "q_value":
            why = f"{spelled}: the PSM-level q"
        elif self.q_column == "run_psm_q":
            why = f"{spelled}: the PSM-level q within the run"
        elif self.q_column == "precursor_q":
            why = f"{spelled}: the precursor-level q"
        else:
            why = spelled
        if experiment and self.configured and self.effective and self.configured != self.effective:
            why += (
                f"; run-experiment forces {self.effective}, the configured {self.configured} "
                "is not applied"
            )
        return f"{text} ({why})"


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def quant_gate(rs: ResultSet, run: Run | str | int | None = None) -> QuantGate:
    """The quant gate of a run, from its ``peptide_quant.parquet.report.json``.

    ``params.q_filter`` and ``params.q_threshold`` of the report are the values quant
    applied. For an experiment ``experiment.quant_q_filter`` gives the configured and the
    effective value (``run-experiment`` always applies ``PsmQ``). ``run=None`` takes the
    only run of a single run, or the first run of an experiment (the gate is the same
    for every run of an experiment).
    """
    if run is None and rs.is_experiment and rs.runs:
        run = rs.runs[0]
    target = resolve_run(rs, run)
    art = target.artifact("peptide_quant")
    params = art.report.params if art is not None and art.report is not None else {}
    exp = rs.manifest.experiment or {}
    recorded = exp.get("quant_q_filter") if rs.is_experiment else None
    recorded = recorded if isinstance(recorded, dict) else {}

    effective_raw = params.get("q_filter")
    threshold = _float_or_none(params.get("q_threshold"))
    sources: list[str] = []
    if effective_raw is not None and art is not None and art.report is not None:
        sources.append(f"{art.report.path.name} params.q_filter")
    if effective_raw is None and recorded.get("effective") is not None:
        effective_raw = recorded.get("effective")
        sources.append("experiment_manifest.json experiment.quant_q_filter.effective")
    if effective_raw is None and not rs.is_experiment:
        effective_raw = rs.config_get("quant", "q_filter")
        if effective_raw is not None:
            sources.append("config_json quant.q_filter (no quant report)")
    if threshold is not None and art is not None and art.report is not None:
        sources.append("params.q_threshold")
    else:
        threshold = _float_or_none(rs.config_get("quant", "q_threshold"))
        if threshold is not None:
            sources.append("config_json quant.q_threshold")

    if rs.is_experiment:
        configured_raw = recorded.get("configured", rs.config_get("quant", "q_filter"))
    else:
        configured_raw = rs.config_get("quant", "q_filter", default=effective_raw)
    effective = engine_q_filter(effective_raw)
    configured = engine_q_filter(configured_raw)
    q_column = q_filter_column(effective_raw)
    transfers = mbr_ran(rs)

    notes: list[str] = []
    if rs.is_experiment:
        pair = f"configured {configured or 'not recorded'}, effective {effective or 'not recorded'}"
        if q_column == "q_value":
            notes.append(
                "run-experiment gates every run's quant on the pooled q_value (PSM level, "
                f"experiment-wide) and ignores the configured quant.q_filter ({pair})."
            )
        else:
            notes.append(f"Per-run quant of this experiment gates on {q_column} ({pair}).")
        if configured is not None and effective is not None and configured != effective:
            notes.append(
                "The per-run identification counts use run_psm_q, so the quantified "
                "precursors of a run can differ from its identified precursors."
            )
        rec_eff = engine_q_filter(recorded.get("effective"))
        if rec_eff is not None and effective is not None and rec_eff != effective:
            notes.append(
                f"The experiment manifest records effective {rec_eff}, but this run's quant "
                f"report records {effective}; the report is used."
            )
    elif q_column == "peptide_q_value":
        notes.append(
            "peptide_q_value is set on the winning row of each base peptide only (1.0 on "
            "the other rows), so quant selects one precursor per accepted base peptide; "
            "the other precursors of that peptide get no quantity."
        )
    elif q_column is not None:
        notes.append(f"A single run honours the configured quant.q_filter ({effective}).")
    if q_column is None and effective_raw is not None:
        notes.append(f"The q_filter value {effective_raw!r} is not known to this viewer.")
    if effective_raw is None:
        notes.append("No quant report and no configuration record the quant gate.")
    if transfers:
        notes.append(
            "Match-between-runs transfers are admitted whatever their q value; no quant "
            "table flags them, so the viewer flags them from mbr_transferred.parquet."
        )
    return QuantGate(
        q_filter=effective,
        q_column=q_column,
        threshold=threshold,
        source="; ".join(sources) if sources else "not recorded",
        configured=configured,
        effective=effective,
        note=" ".join(notes),
        transfers_admitted=transfers,
    )


# --------------------------------------------------------------------------- quant state


@dataclass(frozen=True)
class QuantState:
    """The quant outcome of one scored row (candidate) in one run.

    ``state`` is ``quantified`` (a quantity exists), ``not_quantifiable`` (a
    ``peptide_quant`` row with a null quantity; ``status`` says why) or ``not_selected``
    (no ``peptide_quant`` row; ``reason`` explains the gate). A missing quantity is None,
    never zero. ``from_transfer`` marks a match-between-runs transfer in this run.

    ``gate_q`` is the gate column of the scored table that quant read (after
    match-between-runs it is lowered on transferred rows). ``native_q`` is the same
    column in ``scored_combined.parquet`` (pre-MBR); it is set only when
    match-between-runs ran. ``window_also_covers`` lists the other peak ranks of the
    candidate whose apex lies inside the integration window (top-K peak promotion).
    """

    state: QuantStateName
    status: str | None
    quantity: float | None
    n_fragments_used: int | None
    integration_apex_rt: float | None
    integration_lo_rt: float | None
    integration_hi_rt: float | None
    reason: str
    from_transfer: bool = False
    gate_q: float | None = None
    native_q: float | None = None
    window_also_covers: tuple[int, ...] = ()


_STATE_COLUMNS = [
    "candidate_id",
    "state",
    "status",
    "quantity",
    "n_fragments_used",
    "integration_apex_rt",
    "integration_lo_rt",
    "integration_hi_rt",
    "reason",
    "from_transfer",
    "transfer_q",
    "gate_q",
    "native_q",
    "window_also_covers",
]

# The group of each grouped q column: in words, and as columns of the scored table.
_GROUP_NOUN: dict[str, str] = {
    "peptide_q_value": "base peptide",
    "precursor_q": "precursor (peptidoform, charge)",
    "pg_q_value": "protein group",
}
_GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "peptide_q_value": ("base_peptide_id",),
    "precursor_q": ("peptidoform", "charge"),
    "pg_q_value": ("protein_group",),
}


def _clean(value: Any) -> Any:
    """None for SQL NULL and NaN; plain Python numbers otherwise."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _fmt_many(values: pd.Series) -> pd.Series:
    """Format q values with ``%.6g``, element-wise; missing values become ''."""
    arr = values.to_numpy(dtype="float64", na_value=np.nan)
    out = np.char.mod("%.6g", np.nan_to_num(arr, nan=0.0)).astype(object)
    out[np.isnan(arr)] = ""
    return pd.Series(out, index=values.index, dtype=object)


def _flag(df: pd.DataFrame, column: str) -> np.ndarray:
    """A boolean column as a numpy array, with missing values as False."""
    return df[column].astype("boolean").fillna(False).to_numpy(dtype=bool)


def _ints(values: pd.Series) -> pd.Series:
    """Integers as text, element-wise; missing values become ''."""
    arr = values.astype("Int64")
    out = arr.map(lambda v: "" if pd.isna(v) else str(int(v))).astype(object)
    return pd.Series(out, index=values.index, dtype=object)


def _fail_reasons(
    df: pd.DataFrame,
    fail: np.ndarray,
    col: str,
    thr: float,
    shown: pd.Series,
    *,
    experiment: bool,
    run_names: dict[int, str],
) -> pd.Series:
    """Reasons of the target rows whose gate q fails the threshold.

    For a grouped (sparse) gate column the winner of the row's group decides the text:
    the winner is the row whose grouped q is below 1.0 (the engine writes the group q to
    the winning row only), or, when a winner sits at the cap of 1.0, the row that the
    engine's rule picks (highest score, a decoy first on an exact tie, then file order).
    """
    base = f"not selected: {col} = " + shown + f" fails the quant gate {col} <= {thr:g}"
    out = pd.Series("", index=df.index, dtype=object)
    out[fail] = base[fail]
    if col not in _GROUP_NOUN:
        if col == "q_value" and experiment:
            out[fail] = out[fail] + " (the pooled PSM q that run-experiment gates quant on)"
        return out
    noun = _GROUP_NOUN[col]
    known = _flag(df, "w_known")
    own = _flag(df, "w_self")
    capped = (df["gate_q"].astype("float64") >= 1.0).to_numpy(dtype=bool)
    win = fail & known & own
    at_cap = win & capped
    out[at_cap] = out[at_cap] + (
        f" (this row is the winning row of its {noun}; the q of a winning row can be at the "
        "cap of 1.0)"
    )
    lose = fail & known & ~own
    if not lose.any():
        return out
    in_run = _flag(df, "w_in_run")
    has_row = _flag(df, "w_has_row")
    null_q = _flag(df, "w_quantity_null")
    decoy = (df["w_label"].astype(object) == "decoy").to_numpy(dtype=bool)
    cid = _ints(df["w_cid"])
    runs = df["w_source"].map(
        lambda s: "" if pd.isna(s) else run_names.get(int(s), f"source {int(s)}")
    )
    where = ("candidate " + cid).where(in_run, "candidate " + cid + " in run " + runs)
    sibling = lose & ~decoy & in_run & has_row
    status = df["w_status"].astype(object).where(df["w_status"].notna(), "no quant_status")
    state = pd.Series("quantified", index=df.index, dtype=object).where(
        ~null_q, "not quantifiable: " + status
    )
    out[sibling] = (
        f"not selected: another precursor of this {noun} (" + where[sibling] + ") is its "
        "winning row and was selected for quant (" + state[sibling] + f"); {col} is set on "
        f"the winning row only ({col} = " + shown[sibling] + " on this row), so quant takes "
        f"one precursor per {noun}"
    )
    lost_to_decoy = lose & decoy
    out[lost_to_decoy] = (
        base[lost_to_decoy]
        + f": the winning row of this {noun} is a decoy row ("
        + where[lost_to_decoy]
        + "), and every other row holds 1.0"
    )
    other = lose & ~decoy & ~sibling
    out[other] = (
        base[other] + f": the winning row of this {noun} is " + where[other] + ", and every "
        "other row holds 1.0"
    )
    unselected_winner = other & in_run & ~has_row
    out[unselected_winner] = out[unselected_winner] + " (the winning row is not selected either)"
    return out


def _reasons(
    df: pd.DataFrame,
    gate: QuantGate,
    *,
    top_n: int | None,
    experiment: bool,
    run_label: str,
    run_names: dict[int, str],
) -> pd.Series:
    """The plain-text reason of every row of a quant-state frame, column by column.

    The frame holds ``has_row``, ``in_scored``, ``label``, ``gate_q``, the winner
    columns (``w_known``, ``w_self``, ``w_source``, ``w_cid``, ``w_label``, ``w_in_run``,
    ``w_has_row``, ``w_quantity_null``, ``w_status``), ``quantity``, ``quant_status``,
    ``n_fragments_used``, the integration columns, ``from_transfer``, ``transfer_q``,
    ``native_q`` and ``covers_text``.
    """
    reason = pd.Series("", index=df.index, dtype=object)
    has = df["has_row"].to_numpy(dtype=bool)
    in_scored = df["in_scored"].to_numpy(dtype=bool)
    decoy = (df["label"] == "decoy").to_numpy(dtype=bool)
    col, thr = gate.q_column, gate.threshold

    # Not selected: no peptide_quant row.
    missing = ~has & ~in_scored
    reason[missing] = (
        "not selected: candidate "
        + df.loc[missing, "candidate_id"].astype("int64").map(str).astype(object)
        + f" has no scored row in {run_label}"
    )
    reason[~has & in_scored & decoy] = "not selected: quant never selects decoy rows"
    rest = ~has & in_scored & ~decoy
    if col is None or thr is None:
        reason[rest] = "not selected: no peptide_quant row (the quant gate is not recorded)"
    else:
        gq = df["gate_q"].astype("float64")
        passes = (gq <= thr).to_numpy(dtype=bool)
        finite = np.isfinite(gq.to_numpy())
        reason[rest & ~finite] = f"not selected: {col} is not finite on this row"
        rest = rest & finite
        shown = pd.Series("", index=df.index, dtype=object)
        shown[rest] = _fmt_many(gq[rest])
        ok = rest & passes
        reason[ok] = (
            f"no peptide_quant row, although {col} = "
            + shown[ok]
            + f" passes the recorded gate {col} <= {thr:g}; the quant table may not belong "
            "to this scored table"
        )
        fail = rest & ~passes
        texts = _fail_reasons(df, fail, col, thr, shown, experiment=experiment, run_names=run_names)
        reason[fail] = texts[fail]

    # Selected: a peptide_quant row, with or without a quantity.
    status = df["quant_status"].astype(object)
    known = status.map(PEPTIDE_STATUS)
    base = known.where(known.notna(), status).where(status.notna(), "no quant_status")
    null_q = df["quantity"].isna().to_numpy(dtype=bool)
    nq = has & null_q
    reason[nq] = base[nq] + "; the quantity is missing, not zero"
    said = nq & (status == "quantified").to_numpy(dtype=bool)
    reason[said] = reason[said] + " (the status says quantified, but the quantity is null)"
    qd = has & ~null_q
    reason[qd] = base[qd]
    odd = qd & (status.notna() & (status != "quantified")).to_numpy(dtype=bool)
    reason[odd] = reason[odd] + " (a quantity is present although the status is not 'quantified')"
    if top_n is not None:
        used = df["n_fragments_used"].astype("float64")
        few = qd & ((used > 0) & (used < top_n)).to_numpy(dtype=bool)
        reason[few] = (
            reason[few]
            + "; "
            + used[few].astype("int64").map(str).astype(object)
            + f" fragment(s) summed, fewer than quant.top_n_fragments = {top_n}"
        )
    bounds = df[["integration_apex_rt", "integration_lo_rt", "integration_hi_rt"]]
    whole = qd & bounds.isna().all(axis=1).to_numpy(dtype=bool)
    reason[whole] = reason[whole] + "; whole trace integrated (no integration bounds recorded)"
    covers = has & df["covers_text"].notna().to_numpy(dtype=bool)
    reason[covers] = (
        reason[covers]
        + "; the integration window also covers the apex of another peak of this candidate ("
        + df.loc[covers, "covers_text"].astype(object)
        + "), which the rescorer did not select"
    )
    tr = has & df["from_transfer"].to_numpy(dtype=bool)
    if tr.any():
        tq = df["transfer_q"].astype("float64")
        with_q = tr & tq.notna().to_numpy(dtype=bool)
        reason[tr] = reason[tr] + "; admitted as a match-between-runs transfer"
        reason[with_q] = reason[with_q] + " (transfer_q " + _fmt_many(tq)[with_q] + ")"
        if col is not None and thr is not None:
            nat = df["native_q"].astype("float64")
            shown_nat = _fmt_many(nat)
            has_nat = tr & nat.notna().to_numpy(dtype=bool)
            nat_ok = has_nat & (nat <= thr).to_numpy(dtype=bool)
            nat_fail = has_nat & ~nat_ok
            reason[nat_ok] = (
                reason[nat_ok]
                + f"; its native {col} "
                + shown_nat[nat_ok]
                + " (scored_combined.parquet) passes the gate as well"
            )
            reason[nat_fail] = (
                reason[nat_fail]
                + f"; its native {col} "
                + shown_nat[nat_fail]
                + f" (scored_combined.parquet) fails the gate {col} <= {thr:g}, so this row "
                "is quantified only through the transfer"
            )
    return reason


def _peak_tables(run: Run) -> tuple[list[Artifact], str]:
    """The tables with one row per ``(candidate_id, peak_rank)`` of a run, and their name.

    The run's ``psms_extracted``, ``features`` or ``psms_competed`` table, in this order;
    for a grouped run without them, the band tables when the run has no overlap losers
    (a loser band's rows belong to another band's candidate). An empty list says why
    no table is used.
    """
    need = ("candidate_id", "peak_rank", "apex_rt")

    def fits(art: Artifact | None) -> bool:
        if not _usable(art):
            return False
        assert art is not None
        try:
            handle = art.parquet()
        except (ViewerError, OSError):
            return False
        return all(handle.has_column(c) for c in need)

    for kind in ("psms_extracted", "features", "psms_competed"):
        art = run.artifact(kind)
        if fits(art):
            assert art is not None
            return [art], RUN_FILES.get(kind, kind)
    layout = run.grouped
    if layout is None or not layout.bands:
        return [], "no table with peak_rank rows was found"
    losers = layout.losers
    if losers is not None and losers.usable:
        n_losers = losers.rows if losers.rows is not None else losers.parquet().num_rows
        if n_losers:
            return [], (
                "the band tables are not used: the run has overlap losers, so a band table can "
                "hold rows of another band's candidates"
            )
    tables: list[Artifact] = []
    for band in layout.bands:
        for kind in ("psms_extracted", "psms_competed"):
            art = band.artifact(kind)
            if fits(art):
                assert art is not None
                tables.append(art)
                break
    if not tables:
        return [], "no table with peak_rank rows was found"
    return tables, "the band psms_extracted or psms_competed tables"


def _has_alternative_peaks(tables: list[Artifact]) -> bool:
    """False only when the footer statistics show ``peak_rank`` 0 on every row."""
    for art in tables:
        stats = art.parquet().column_statistics("peak_rank")
        if any(s is None or s[1] is None or int(s[1]) > 0 for s in stats):
            return True
    return False


def quant_states(
    rs: ResultSet, run: Run | str | int | None, cids: Iterable[int] | np.ndarray
) -> pd.DataFrame:
    """The quant state of many candidates of one run, in the order given.

    Columns: ``candidate_id``, ``state``, ``status``, ``quantity``, ``n_fragments_used``,
    ``integration_apex_rt``, ``integration_lo_rt``, ``integration_hi_rt``, ``reason``,
    ``from_transfer``, ``transfer_q``, ``gate_q`` (the gate column of the scored table
    that quant read), ``native_q`` (the same column in ``scored_combined.parquet``; only
    when match-between-runs ran) and ``window_also_covers`` (the other peak ranks whose
    apex lies inside the integration window, as text such as ``"0"``, or None).
    ``quantity`` is NaN when missing, never 0.

    A run whose ``peptide_quant`` table cannot be read raises: an unsupported schema
    version raises :class:`SchemaVersionError`, an absent table
    :class:`ArtifactNotFound`.
    """
    target = resolve_run(rs, run)
    ids = np.asarray(list(cids) if not isinstance(cids, np.ndarray) else cids)
    if ids.size == 0:
        return pd.DataFrame(columns=_STATE_COLUMNS)
    if ids.dtype.kind not in "iu" or int(ids.min()) < 0 or int(ids.max()) >= 2**32:
        raise ViewerError("candidate ids must be integers from 0 to 2**32 - 1.")
    pq_art = _run_table(rs, target, "peptide_quant")
    gate = quant_gate(rs, target)
    scored, pooled = run_scored(rs, target)
    scored_names = set(scored.parquet().schema.names)
    gate_col = gate.q_column if gate.q_column in scored_names else None
    gate_expr = sql_ident(gate_col) if gate_col else "NULL::DOUBLE"
    rank_expr = "selected_peak_rank" if "selected_peak_rank" in scored_names else "0"
    src_filter = " AND source = ?" if pooled else ""
    index = int(target.index)

    # The ids are registered as an Arrow table on this thread's cursor (a long Python
    # list parameter is slow to bind). The tables are read with a hash semi-join on them
    # plus a range that lets the footer statistics prune row groups.
    view = "mumdia_viewer_ids_" + uuid.uuid4().hex
    params: list[Any] = []
    ctes = [f"c AS (SELECT candidate_id, pos FROM {view})"]
    lo_id, hi_id = int(ids.min()), int(ids.max())
    in_ids = "candidate_id BETWEEN ? AND ? AND candidate_id IN (SELECT candidate_id FROM c)"
    ctes.append(
        "s AS (SELECT candidate_id, any_value(label) AS label, "
        "any_value(base_peptide_id) AS base_peptide_id, any_value(peptidoform) AS peptidoform, "
        "any_value(charge) AS charge, any_value(protein_group) AS protein_group, "
        f"any_value({gate_expr}) AS gate_q, any_value({rank_expr}) AS selected_peak_rank "
        f"FROM read_parquet(?) WHERE {in_ids}{src_filter} GROUP BY candidate_id)"
    )
    params += [sql_path(scored.require()), lo_id, hi_id]
    if pooled:
        params.append(index)
    pq_path = sql_path(pq_art.require())
    ctes.append(
        "q AS (SELECT candidate_id, any_value(quantity) AS quantity, "
        "any_value(quant_status) AS quant_status, any_value(n_fragments_used) AS n_fragments_used, "
        "any_value(integration_apex_rt) AS integration_apex_rt, "
        "any_value(integration_lo_rt) AS integration_lo_rt, "
        "any_value(integration_hi_rt) AS integration_hi_rt "
        f"FROM read_parquet(?) WHERE {in_ids} GROUP BY 1)"
    )
    params += [pq_path, lo_id, hi_id]
    joins = [
        "LEFT JOIN s ON s.candidate_id = c.candidate_id",
        "LEFT JOIN q ON q.candidate_id = c.candidate_id",
    ]
    items = [
        "c.pos",
        "c.candidate_id",
        "s.candidate_id IS NOT NULL AS in_scored",
        "s.label",
        "s.gate_q",
        "q.candidate_id IS NOT NULL AS has_row",
        "q.quantity",
        "q.quant_status",
        "q.n_fragments_used",
        "q.integration_apex_rt",
        "q.integration_lo_rt",
        "q.integration_hi_rt",
    ]

    # The winner of the group of each failing target row, for a grouped gate column.
    pooled_names = set(rs.scored.parquet().schema.names)
    if (
        gate_col in _GROUP_COLUMNS
        and gate.threshold is not None
        and gate_col in pooled_names
        and "score" in pooled_names
    ):
        keys = _GROUP_COLUMNS[gate_col]
        entrapment = _entrapment_mode(rs)
        ctes.append(
            f"wk AS (SELECT DISTINCT {', '.join('s.' + k for k in keys)} FROM s "
            "LEFT JOIN q ON q.candidate_id = s.candidate_id WHERE q.candidate_id IS NULL "
            "AND s.label <> 'decoy' AND isfinite(s.gate_q) AND s.gate_q > ?)"
        )
        params.append(float(gate.threshold))
        # The spike-in test of the identification counts (entrapment.count_classes).
        tie_sql, tie_params = spike_in_condition(rs, "p") if entrapment else ("false", [])
        on_keys = " AND ".join(f"wk.{k} = p.{k}" for k in keys)
        ctes.append(
            "g AS (SELECT p.file_row_number AS rn, p.source, p.candidate_id, p.label, p.score, "
            f"p.{sql_ident(gate_col)} AS gq, {', '.join('p.' + k for k in keys)}, "
            f"{tie_sql} AS is_ent FROM read_parquet(?, file_row_number = true) p "
            f"JOIN wk ON {on_keys})"
        )
        params += [*tie_params, sql_path(rs.scored.require())]
        key_list = ", ".join(keys)
        ctes.append(f"cap AS (SELECT {key_list} FROM g GROUP BY ALL HAVING NOT bool_or(gq < 1))")
        tie = "is_ent DESC" if entrapment else "(label = 'decoy') DESC"
        skip = " WHERE g.label <> 'decoy'" if entrapment else ""
        on_cap = " AND ".join(f"cap.{k} = g.{k}" for k in keys)
        # One row per group: the row below 1.0, else the rule's winner of a group whose
        # winner sits at the cap. QUALIFY keeps one row even if a file had two below 1.0.
        ctes.append(
            f"w AS (SELECT * FROM (SELECT {key_list}, source AS w_source, candidate_id AS w_cid, "
            "label AS w_label, gq AS w_q FROM g WHERE gq < 1 "
            f"UNION ALL SELECT {key_list}, source, candidate_id, label, gq FROM ("
            f"SELECT g.*, row_number() OVER (PARTITION BY {', '.join('g.' + k for k in keys)} "
            f"ORDER BY g.score DESC, {tie}, g.rn) AS rk FROM g JOIN cap ON {on_cap}{skip}) "
            f"WHERE rk = 1) u QUALIFY row_number() OVER (PARTITION BY {key_list} "
            "ORDER BY w_q, w_source, w_cid) = 1)"
        )
        ctes.append(
            "wq AS (SELECT candidate_id AS w_cid, any_value(quantity) AS w_quantity, "
            "any_value(quant_status) AS w_status FROM read_parquet(?) "
            f"WHERE candidate_id IN (SELECT w_cid FROM w WHERE w_source = {index}) GROUP BY 1)"
        )
        params.append(pq_path)
        joins.append("LEFT JOIN w ON " + " AND ".join(f"w.{k} = s.{k}" for k in keys))
        joins.append(f"LEFT JOIN wq ON w.w_source = {index} AND wq.w_cid = w.w_cid")
        items += [
            "w.w_cid IS NOT NULL AS w_known",
            f"(w.w_source = {index} AND w.w_cid = c.candidate_id) AS w_self",
            "w.w_source",
            "w.w_cid",
            "w.w_label",
            f"(w.w_source = {index}) AS w_in_run",
            "wq.w_cid IS NOT NULL AS w_has_row",
            "wq.w_quantity IS NULL AS w_quantity_null",
            "wq.w_status",
        ]
    else:
        items += [
            "false AS w_known",
            "false AS w_self",
            "NULL::UINTEGER AS w_source",
            "NULL::UINTEGER AS w_cid",
            "NULL::VARCHAR AS w_label",
            "false AS w_in_run",
            "false AS w_has_row",
            "true AS w_quantity_null",
            "NULL::VARCHAR AS w_status",
        ]

    # Match-between-runs: the transfer flag and the native (pre-MBR) gate q.
    transfers = transfer_relation(rs) if gate.transfers_admitted else None
    if transfers is not None:
        t_sql, t_params = transfers
        ctes.append(f"t AS (SELECT candidate_id, transfer_q FROM ({t_sql}) WHERE source = ?)")
        params += [*t_params, index]
        joins.append("LEFT JOIN t ON t.candidate_id = c.candidate_id")
        items += ["t.candidate_id IS NOT NULL AS from_transfer", "t.transfer_q"]
        if gate_col is not None and gate_col in pooled_names:
            ctes.append(
                f"n AS (SELECT candidate_id, any_value({sql_ident(gate_col)}) AS native_q "
                f"FROM read_parquet(?) WHERE source = ? AND {in_ids} GROUP BY 1)"
            )
            params += [sql_path(rs.scored.require()), index, lo_id, hi_id]
            joins.append("LEFT JOIN n ON n.candidate_id = c.candidate_id")
            items.append("n.native_q")
        else:
            items.append("NULL::DOUBLE AS native_q")
    else:
        items += [
            "false AS from_transfer",
            "NULL::DOUBLE AS transfer_q",
            "NULL::DOUBLE AS native_q",
        ]

    # Top-K peak promotion: other peak ranks whose apex lies inside the integration window.
    peak_tables, peak_source = _peak_tables(target)
    peaks_checked = bool(peak_tables) and _has_alternative_peaks(peak_tables)
    if peaks_checked:
        paths = [sql_path(a.require()) for a in peak_tables]
        ctes.append(
            "e AS (SELECT candidate_id, peak_rank, apex_rt "
            f"FROM read_parquet(?, union_by_name = true) WHERE {in_ids})"
        )
        params += [paths if len(paths) > 1 else paths[0], lo_id, hi_id]
        ctes.append(
            "ew AS (SELECT e.candidate_id, "
            "string_agg(CAST(e.peak_rank AS VARCHAR), ', ' ORDER BY e.peak_rank) AS covers, "
            "string_agg(printf('peak_rank %d at %.2f s', e.peak_rank, e.apex_rt), ', ' "
            "ORDER BY e.peak_rank) AS covers_text "
            "FROM e JOIN s ON s.candidate_id = e.candidate_id "
            "JOIN q ON q.candidate_id = e.candidate_id "
            "WHERE e.peak_rank <> s.selected_peak_rank "
            "AND CAST(e.apex_rt AS FLOAT) BETWEEN q.integration_lo_rt AND q.integration_hi_rt "
            "GROUP BY 1)"
        )
        joins.append("LEFT JOIN ew ON ew.candidate_id = c.candidate_id")
        items += ["ew.covers", "ew.covers_text"]
        peaks_note = (
            "Each integration window was compared with the apex of every other peak rank of "
            f"the candidate in {peak_source}."
        )
    else:
        items += ["NULL::VARCHAR AS covers", "NULL::VARCHAR AS covers_text"]
        peaks_note = (
            f"No alternative peak to compare with the integration windows: {peak_source}."
            if not peak_tables
            else f"No alternative peak to compare with the integration windows: {peak_source} "
            "holds peak_rank 0 only."
        )

    sql = (
        "WITH "
        + ", ".join(ctes)
        + " SELECT "
        + ", ".join(items)
        + " FROM c "
        + " ".join(joins)
        + " ORDER BY c.pos"
    )
    cur = rs.duck.cursor()
    cur.register(
        view,
        pa.table(
            {
                "candidate_id": pa.array(ids.astype(np.uint32)),
                "pos": pa.array(np.arange(ids.size, dtype=np.int64)),
            }
        ),
    )
    try:
        df = cur.execute(sql, params).df()
    finally:
        cur.unregister(view)
    top_n_raw = pq_art.report.params.get("top_n_fragments") if pq_art.report else None
    top_n = int(top_n_raw) if isinstance(top_n_raw, int | float) else None
    has = df["has_row"].to_numpy(dtype=bool)
    null_q = df["quantity"].isna().to_numpy(dtype=bool)
    state = np.where(~has, "not_selected", np.where(null_q, "not_quantifiable", "quantified"))
    status = df["quant_status"].astype(object)
    covers = df["covers"].astype(object)
    out = pd.DataFrame(
        {
            "candidate_id": df["candidate_id"].astype("int64"),
            "state": pd.Series(state, index=df.index, dtype=object),
            "status": status.where(status.notna() & has, None),
            "quantity": df["quantity"].astype("float64"),
            "n_fragments_used": df["n_fragments_used"].astype("Int64"),
            "integration_apex_rt": df["integration_apex_rt"].astype("float64"),
            "integration_lo_rt": df["integration_lo_rt"].astype("float64"),
            "integration_hi_rt": df["integration_hi_rt"].astype("float64"),
            "reason": _reasons(
                df,
                gate,
                top_n=top_n,
                experiment=rs.is_experiment,
                run_label=target.label,
                run_names={int(r.index): r.label for r in rs.runs},
            ),
            "from_transfer": df["from_transfer"].astype(bool),
            "transfer_q": df["transfer_q"].astype("float64"),
            "gate_q": df["gate_q"].astype("float64"),
            "native_q": df["native_q"].astype("float64"),
            "window_also_covers": covers.where(covers.notna() & has, None),
        },
        columns=_STATE_COLUMNS,
    )
    out.attrs["gate"] = gate
    out.attrs["run"] = target.name
    out.attrs["peaks"] = peaks_note
    return out


def _entrapment_mode(rs: ResultSet) -> bool:
    """True when the rescorer ran in entrapment mode (its q columns are entrapment estimates)."""
    return rescore_info(rs).mode == "entrapment"


def quant_state(rs: ResultSet, run: Run | str | int | None, cid: int) -> QuantState:
    """The quant state of one candidate in one run (see :func:`quant_states`)."""
    row = quant_states(rs, run, [int(cid)]).iloc[0]
    n_used = row["n_fragments_used"]
    covers = row["window_also_covers"]
    return QuantState(
        state=row["state"],
        status=row["status"],
        quantity=_clean(row["quantity"]),
        n_fragments_used=None if pd.isna(n_used) else int(n_used),
        integration_apex_rt=_clean(row["integration_apex_rt"]),
        integration_lo_rt=_clean(row["integration_lo_rt"]),
        integration_hi_rt=_clean(row["integration_hi_rt"]),
        reason=str(row["reason"]),
        from_transfer=bool(row["from_transfer"]),
        gate_q=_clean(row["gate_q"]),
        native_q=_clean(row["native_q"]),
        window_also_covers=(
            tuple(int(v) for v in str(covers).split(", ")) if isinstance(covers, str) else ()
        ),
    )


# --------------------------------------------------------------------------- proteins


def protein_quant(
    rs: ResultSet, run: Run | str | int | None, protein_group: str
) -> dict[str, Any] | None:
    """The ``protein_group_quant`` row of one protein group in one run, or None.

    The lookup is by the exact ``protein_group`` string. None means that no peptide of
    the group passed that run's quant gate. The table is not a list of identified
    proteins: filter on ``pg_q_value`` for that. With match-between-runs, the derived
    ``n_transferred_precursors`` counts the group's quantified precursors of this run
    that are transfers.

    A table that cannot be read raises :class:`SchemaVersionError` (unsupported version)
    or :class:`ArtifactNotFound` (absent).
    """
    target = resolve_run(rs, run)
    art = _run_table(rs, target, "protein_group_quant")
    rows = rs.duck.rows(
        "SELECT protein_group, quantity, quant_status, n_peptides FROM read_parquet(?) "
        "WHERE protein_group = ?",
        [sql_path(art.require()), str(protein_group)],
    )
    if not rows:
        return None
    group, quantity, status, n_peptides = rows[0]
    quantity = _clean(quantity)
    params = art.report.params if art.report is not None else {}
    out: dict[str, Any] = {
        "run": target.name,
        "source": int(target.index),
        "protein_group": group,
        "quantity": quantity,
        "quant_status": status,
        "state": "quantified" if quantity is not None else "not_quantifiable",
        "description": describe_status(status, "protein_group_quant"),
        "n_peptides": None if n_peptides is None else int(n_peptides),
        "n_peptides_label": (
            "base peptides with a positive quantity in this run (before the top-N cut)"
        ),
        "rollup": params.get("rollup"),
        "top_n_peptides": params.get("top_n_peptides"),
    }
    if len(rows) > 1:
        out["note"] = f"{len(rows)} rows carry this protein_group; the first is shown."
    if mbr_ran(rs):
        transfers = transfer_relation(rs)
        pq_art = target.artifact("peptide_quant")
        n_tr = 0
        if transfers is not None and _usable(pq_art):
            assert pq_art is not None
            t_sql, t_params = transfers
            n_tr = int(
                rs.duck.scalar(
                    f"SELECT count(*) FROM read_parquet(?) q JOIN ({t_sql}) t "
                    "ON t.candidate_id = q.candidate_id AND t.source = ? "
                    "WHERE q.protein_group = ? AND q.quantity > 0",
                    [sql_path(pq_art.require()), *t_params, int(target.index), str(group)],
                )
                or 0
            )
        out["n_transferred_precursors"] = n_tr
        out["n_transferred_label"] = (
            "quantified precursors of this group in this run that are match-between-runs "
            "transfers (derived from mbr_transferred.parquet)"
        )
    return out


def _transfer_source(rs: ResultSet) -> str:
    """The file that :func:`transfer_relation` reads, by name."""
    info = mbr_info(rs)
    if _usable(info.transfers):
        return "mbr_transferred.parquet"
    sfq = info.scored_for_quant
    name = sfq.path.name if sfq is not None and sfq.path is not None else "scored_for_quant"
    return f"is_transferred of {name}"


def _status_counts(
    rs: ResultSet,
    run: Run,
    kind: str,
    art: Artifact,
    transfers: tuple[str, list[Any]] | None,
    peptide_table: Artifact | None,
) -> list[tuple[Any, int, int | None]]:
    """(quant_status, rows, transferred rows or None) of one quant table of one run.

    ``peptide_quant`` rows are transfers when their ``(source, candidate_id)`` is in the
    transfer relation. A ``protein_group_quant`` row counts when the group holds such a
    row in the run's ``peptide_quant`` (any quantity). None when there is no transfer
    relation, or when the run's peptide_quant cannot be read for a protein table.
    """
    path = sql_path(art.require())
    if transfers is None or (kind == "protein_group_quant" and peptide_table is None):
        rows = rs.duck.rows(
            "SELECT quant_status, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1", [path]
        )
        return [(status, int(n), None) for status, n in rows]
    t_sql, t_params = transfers
    moved = f"(SELECT candidate_id FROM ({t_sql}) WHERE source = ?)"
    if kind == "peptide_quant":
        rows = rs.duck.rows(
            "SELECT q.quant_status, count(*), count(t.candidate_id) FROM read_parquet(?) q "
            f"LEFT JOIN {moved} t ON t.candidate_id = q.candidate_id GROUP BY 1 ORDER BY 1",
            [path, *t_params, int(run.index)],
        )
    else:
        assert peptide_table is not None
        rows = rs.duck.rows(
            "WITH tg AS (SELECT DISTINCT q.protein_group FROM read_parquet(?) q "
            f"JOIN {moved} t ON t.candidate_id = q.candidate_id) "
            "SELECT g.quant_status, count(*), count(tg.protein_group) FROM read_parquet(?) g "
            "LEFT JOIN tg ON tg.protein_group = g.protein_group GROUP BY 1 ORDER BY 1",
            [sql_path(peptide_table.require()), *t_params, int(run.index), path],
        )
    return [(status, int(n), int(n_tr)) for status, n, n_tr in rows]


def quant_status_breakdown(rs: ResultSet) -> pd.DataFrame:
    """Rows per ``quant_status`` of every run's ``peptide_quant`` and ``protein_group_quant``.

    Columns: ``run``, ``table``, ``status``, ``n`` (rows) and ``description``. These are
    counts of the engine's own rows; scored rows without a quant row (not selected) are
    not part of either table. A table that cannot be read (absent, or an unsupported
    schema version) has no rows here; ``attrs['notes']`` names it and says why.

    When match-between-runs ran (:func:`.mbr.mbr_ran`), quant admits every transfer
    whatever its q and no quant table flags it, so the rows of each status include the
    transfers. ``n_transferred`` (after ``n``, derived) then counts them: the
    ``peptide_quant`` rows whose ``(source, candidate_id)`` is an accepted transfer in
    ``mbr_transferred.parquet``, and the ``protein_group_quant`` groups that contain
    such a row. It is missing (NA) when no transfer table can be read. The column labels
    are in ``attrs['labels']``.
    """
    records: list[dict[str, Any]] = []
    notes: list[str] = []
    ran = mbr_ran(rs)
    transfers = transfer_relation(rs) if ran else None
    for run in rs.runs:
        peptide_table: Artifact | None = None
        for kind in ("peptide_quant", "protein_group_quant"):
            art = run.artifact(kind)
            problem = table_problem(rs, run, kind)
            if problem is not None:
                notes.append(f"{run.label} {kind} is not counted: {problem}")
                continue
            assert art is not None
            art.parquet()
            if kind == "peptide_quant":
                peptide_table = art
            elif ran and transfers is not None and peptide_table is None:
                notes.append(
                    f"{run.label} protein_group_quant: n_transferred is missing, because the "
                    "run's peptide_quant cannot be read."
                )
            for status, n, n_tr in _status_counts(rs, run, kind, art, transfers, peptide_table):
                record = {"run": run.name, "table": kind, "status": status, "n": n}
                if ran:
                    record["n_transferred"] = n_tr
                record["description"] = describe_status(status, kind)
                records.append(record)
    columns = ["run", "table", "status", "n", "description"]
    labels = {"n": "rows of the table with this quant_status (engine rows)"}
    if ran:
        columns.insert(4, "n_transferred")
        source = _transfer_source(rs) if transfers is not None else "no readable transfer table"
        labels["n_transferred"] = (
            "rows of n that are match-between-runs transfers (derived from "
            f"{source}): peptide_quant rows whose (source, candidate_id) is an accepted "
            "transfer; protein groups that contain such a peptide_quant row of the run"
        )
        if transfers is None:
            notes.append(
                "Match-between-runs ran, but no transfer table can be read "
                "(mbr_transferred.parquet, or is_transferred in scored_for_quant), so the "
                "transfers in these counts cannot be counted; n_transferred is missing."
            )
        else:
            notes.append(
                "Match-between-runs ran: quant admits every transfer whatever its q, and no "
                "quant table flags it, so the counts of each status include transfers. "
                f"n_transferred counts them (derived from {source})."
            )
    out = pd.DataFrame.from_records(records, columns=columns)
    if ran:
        out["n_transferred"] = out["n_transferred"].astype("Int64")
    out.attrs["notes"] = notes
    out.attrs["labels"] = labels
    return out


# --------------------------------------------------------------------------- LFQ

_LFQ_ARTIFACT = {
    "protein": "lfq_maxlfq",
    "peptide": "lfq_maxlfq_peptide",
    "precursor": "lfq_maxlfq_precursor",
}
_LFQ_KEYS: dict[str, tuple[str, ...]] = {
    "protein": ("protein_group",),
    "peptide": ("group", "charge"),
    "precursor": ("group", "charge"),
}
_LFQ_KEY_LABELS: dict[str, dict[str, str]] = {
    "protein": {"protein_group": "protein group (peptide_quant.protein_group)"},
    "peptide": {
        "group": "stripped sequence (modifications and DECOY_ removed)",
        "charge": "charge (-1: all charges of the sequence)",
    },
    "precursor": {"group": "peptidoform", "charge": "precursor charge"},
}
_LFQ_N_FEATURES_LABELS = {
    "protein": N_FEATURES_LABEL,
    "peptide": "precursors of this sequence in any run",
    "precursor": "features of this precursor (1 under MaxLFQ)",
}

# report.rs::strip: drop a DECOY_ prefix and every [...] and (...) block, keep letters.
STRIP_SQL = (
    "regexp_replace(regexp_replace(regexp_replace({col}, '^DECOY_', ''), "
    "'\\[[^\\]]*\\]|\\([^)]*\\)', '', 'g'), '[^A-Za-z]', '', 'g')"
)


def _accepted_filter(rs: ResultSet, level: str, threshold: float) -> tuple[str, list[Any]]:
    """A WHERE clause keeping LFQ keys accepted experiment-wide at ``threshold``."""
    scored = sql_path(rs.scored.require())
    if level == "protein":
        return (
            "protein_group IN (SELECT DISTINCT protein_group FROM read_parquet(?) "
            "WHERE label = 'target' AND pg_q_value <= ?)",
            [scored, threshold],
        )
    if level == "peptide":
        strip = STRIP_SQL.format(col="peptidoform")
        return (
            f'"group" IN (SELECT DISTINCT {strip} FROM read_parquet(?) '
            "WHERE label = 'target' AND peptide_q_value <= ?)",
            [scored, threshold],
        )
    return (
        '("group", charge) IN (SELECT DISTINCT peptidoform, charge FROM read_parquet(?) '
        "WHERE label = 'target' AND precursor_q <= ?)",
        [scored, threshold],
    )


def _transferred_cells(rs: ResultSet, level: str) -> tuple[str, list[Any]] | None:
    """SQL of (keys..., run, n_transferred): transferred precursors among each cell's features."""
    transfers = transfer_relation(rs)
    union = quant_relation(
        rs, "peptide_quant", ("candidate_id", "peptidoform", "charge", "protein_group", "quantity")
    )
    if transfers is None or union is None:
        return None
    t_sql, t_params = transfers
    u_sql, u_params, _ = union
    if level == "protein":
        keys = "p.protein_group AS protein_group"
    elif level == "peptide":
        keys = f'{STRIP_SQL.format(col="p.peptidoform")} AS "group", -1::INTEGER AS charge'
    else:
        keys = 'p.peptidoform AS "group", p.charge AS charge'
    sql = (
        f"SELECT {keys}, p.source::INTEGER AS run, count(*) AS n_transferred "
        f"FROM ({u_sql}) p JOIN ({t_sql}) t "
        "ON t.source = p.source AND t.candidate_id = p.candidate_id "
        "WHERE p.quantity > 0 AND isfinite(p.quantity) "
        "GROUP BY ALL"
    )
    return sql, [*u_params, *t_params]


def lfq_matrix(
    rs: ResultSet,
    level: LfqLevel = "protein",
    wide: bool = True,
    *,
    accepted_at: float | None = None,
) -> pd.DataFrame:
    """The experiment's MaxLFQ table at one level, with 0.0 (missing) as NaN.

    ``level`` is ``protein`` (``lfq_maxlfq.parquet``, key ``protein_group``), ``peptide``
    (``.peptide.parquet``, keys ``group`` = stripped sequence and ``charge`` = -1) or
    ``precursor`` (``.precursor.parquet``, keys ``group`` = peptidoform and ``charge``).

    ``wide=True`` gives one row per key with ``n_features`` and one column per run name
    (in ``experiment.runs`` order). ``wide=False`` gives one row per key and run with the
    columns ``run`` (name), ``source``, ``quantity`` and ``n_features``. With
    match-between-runs, ``n_transferred`` (long) or ``n_transferred_<run>`` (wide) count
    the transferred precursors among the cell's features (derived).

    The LFQ key list is every key with a quant-gated feature in any run; it is not
    FDR-controlled at the key's unit. ``accepted_at=t`` keeps only the keys accepted
    experiment-wide at ``t`` (protein: ``pg_q_value``; peptide: the stripped sequence of
    a target row with ``peptide_q_value <= t``; precursor: ``precursor_q``).
    """
    if not rs.is_experiment:
        raise ViewerError(
            "LFQ is computed only across the runs of an experiment; a single run has "
            "per-run quantities in peptide_quant and protein_group_quant."
        )
    if level not in _LFQ_ARTIFACT:
        raise ViewerError(f"unknown LFQ level {level!r}; use protein, peptide or precursor.")
    name = {
        "protein": "lfq_maxlfq.parquet",
        "peptide": "lfq_maxlfq.parquet.peptide.parquet",
        "precursor": "lfq_maxlfq.parquet.precursor.parquet",
    }[level]
    expected = rs.root / name
    protein_table = rs.artifact("lfq_maxlfq")
    if level != "protein" and protein_table is not None and protein_table.path is not None:
        # The sibling tables sit next to the protein table (experiment.lfq + ".<level>.parquet").
        expected = protein_table.path.with_name(protein_table.path.name + f".{level}.parquet")
    art = require_artifact(
        rs,
        rs.artifact(_LFQ_ARTIFACT[level]),
        f"the {level} LFQ table of the experiment",
        name,
        expected,
    )
    keys = _LFQ_KEYS[level]
    key_sql = ", ".join(sql_ident(k) for k in keys)
    where, where_params = "", []
    if accepted_at is not None:
        clause, where_params = _accepted_filter(rs, level, float(accepted_at))
        where = f" WHERE {clause}"
    base = (
        f"SELECT {key_sql}, run::INTEGER AS run, "
        "CASE WHEN quantity = 0 THEN NULL ELSE quantity END AS quantity, n_features "
        f"FROM read_parquet(?){where}"
    )
    params: list[Any] = [sql_path(art.require()), *where_params]
    names = {int(r.index): r.name for r in rs.runs}
    mbr = mbr_ran(rs)
    transferred = _transferred_cells(rs, level) if mbr else None
    labels = dict(_LFQ_KEY_LABELS[level])
    labels["n_features"] = _LFQ_N_FEATURES_LABELS[level]
    description = (
        f"MaxLFQ {level} table: one row per key"
        + (" and run" if not wide else "")
        + "; quantity 0.0 in the file means no feature in that run and is shown as missing. "
        + (
            f"Keys accepted experiment-wide at {accepted_at:g} only."
            if accepted_at is not None
            else "The key list is every key with a quant-gated feature in any run; it is not "
            "FDR-controlled at this unit."
        )
    )
    if mbr:
        description += (
            " Match-between-runs ran: the values include transferred features, which the "
            "LFQ table does not flag; n_transferred counts them (derived)."
        )
    if not wide:
        sql = f"SELECT l.* FROM ({base}) l"
        if transferred is not None:
            t_sql, t_params = transferred
            on = " AND ".join(f"x.{sql_ident(k)} = l.{sql_ident(k)}" for k in keys)
            sql = (
                f"SELECT l.*, coalesce(x.n_transferred, 0) AS n_transferred FROM ({base}) l "
                f"LEFT JOIN ({t_sql}) x ON {on} AND x.run = l.run"
            )
            params += t_params
        sql += f" ORDER BY {', '.join('l.' + sql_ident(k) for k in keys)}, l.run"
        df = rs.duck.df(sql, params)
        df.insert(len(keys), "source", df["run"].astype("int64"))
        df["run"] = [names.get(int(s), str(int(s))) for s in df["source"]]
        cols = [*keys, "run", "source", "quantity", "n_features"]
        if "n_transferred" in df.columns:
            cols.append("n_transferred")
            labels["n_transferred"] = (
                "transferred precursors among this cell's features (derived from "
                "mbr_transferred.parquet)"
            )
        df = df[cols]
        df.attrs.update(level=level, description=description, column_labels=labels)
        return df

    run_cols: list[str] = []
    select = [*[f"l.{sql_ident(k)}" for k in keys], "max(l.n_features) AS n_features"]
    select.append("min(l.n_features) <> max(l.n_features) AS _varies")
    taken = set(keys) | {"n_features"}
    for idx in sorted(names):
        name = names[idx]
        col = name if name not in taken else f"{name} (run)"
        taken.add(col)
        run_cols.append(col)
        select.append(f"max(CASE WHEN l.run = {idx} THEN l.quantity END) AS {sql_ident(col)}")
    joins = ""
    tr_cols: list[str] = []
    if transferred is not None:
        t_sql, t_params = transferred
        on = " AND ".join(f"x.{sql_ident(k)} = l.{sql_ident(k)}" for k in keys)
        joins = f" LEFT JOIN ({t_sql}) x ON {on} AND x.run = l.run"
        params += t_params
        for idx in sorted(names):
            col = f"n_transferred_{names[idx]}"
            tr_cols.append(col)
            select.append(
                f"coalesce(max(CASE WHEN l.run = {idx} THEN x.n_transferred END), 0) "
                f"AS {sql_ident(col)}"
            )
    group = ", ".join(f"l.{sql_ident(k)}" for k in keys)
    sql = f"SELECT {', '.join(select)} FROM ({base}) l{joins} GROUP BY {group} ORDER BY {group}"
    df = rs.duck.df(sql, params)
    varies = bool(df["_varies"].any()) if len(df) else False
    df = df.drop(columns="_varies")
    for col in run_cols:
        df[col] = df[col].astype("float64")
    if varies:
        labels["n_features"] += " (maximum over runs; it differs between runs for some keys)"
    for col in tr_cols:
        labels[col] = "transferred precursors among this cell's features (derived)"
    present = rs.duck.rows("SELECT DISTINCT run FROM read_parquet(?)", [sql_path(art.require())])
    unknown = sorted({int(r[0]) for r in present} - set(names))
    if unknown:
        description += f" Run indexes {unknown} are not in experiment.runs and are left out."
    df.attrs.update(
        level=level, description=description, column_labels=labels, run_columns=run_cols
    )
    return df
