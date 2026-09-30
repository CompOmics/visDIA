"""Match-between-runs (MBR): whether it ran, what it transferred, and native counts.

MBR runs inside ``run-experiment`` when ``mbr.strategy`` is not ``none``. The only
reliable signal is the experiment manifest: ``experiment.mbr`` (also
``model_identities.mbr``) is the Rust name of the strategy, ``"None"`` when MBR did not
run. The files are not a signal: a later run without MBR into the same directory
leaves an old ``scored_mbr.parquet`` and ``mbr_transferred.parquet`` behind.

When MBR ran:

* ``scored_mbr.parquet`` (``experiment.scored_for_quant``) is the pooled table plus
  ``is_transferred`` and ``transfer_q``. On transferred rows it lowers ``q_value``,
  ``run_psm_q`` and ``experiment_psm_q`` to ``min(q, transfer_q)``. Quant, LFQ, the
  per-run ``scored.parquet`` split and the TSV report read it. It is written even when
  no transfer was accepted.
* ``mbr_transferred.parquet`` lists the accepted transfers, one row per
  ``(source, candidate_id)``, with the RT evidence. It has no manifest record and no
  report. With candidates but no accepted transfer its string columns have the Arrow
  type null; this module never reads strings from it.
* ``scored_combined.parquet`` is not changed. Native identification counts come from
  it only. Filtering the MBR tables with ``NOT is_transferred`` is wrong: a transferred
  row can also pass ``run_psm_q`` natively.

A transfer is accepted at ``transfer_q <= mbr.q_transfer`` under a permuted-RT null.
It is not FDR-controlled at the identification threshold, so transfers are always
shown as a separate, labelled number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import Artifact
from .discovery import ResultSet
from .duck import sql_path
from .entrapment import count_classes
from .errors import ViewerError
from .reports import normalise_enum
from .units import check_threshold, execute_bound, format_threshold, run_key

__all__ = [
    "MbrInfo",
    "mbr_info",
    "mbr_ran",
    "n_runs_decomposition",
    "transfer_counts",
    "transfer_of",
    "transfers_for_run",
]

_EMPTY_TRANSFERS = (
    "SELECT CAST(NULL AS UINTEGER) AS source, CAST(NULL AS UINTEGER) AS candidate_id WHERE false"
)

_TRANSFER_COLUMNS = [
    "candidate_id",
    "peptidoform",
    "charge",
    "label",
    "protein_group",
    "apex_rt",
    "score",
    "native_q_value",
    "native_run_psm_q",
    "native_precursor_q",
    "native_peptide_q_value",
    "native_pg_q_value",
    "transfer_q",
    "expected_rt",
    "observed_rt",
    "rt_delta",
    "q_value_after_mbr",
    "run_psm_q_after_mbr",
]


@dataclass(frozen=True)
class MbrInfo:
    """What the manifest and the files say about match-between-runs (frozen).

    ``scored_for_quant`` and ``transfers`` are set only when MBR ran; a file left over
    from an earlier run is reported in ``notes`` instead. ``n_transfers`` is the number
    of accepted transfers (0 when MBR did not run; None when it ran but no transfer
    table is readable). ``notes`` is a tuple, so a caller cannot change the memoised
    object.
    """

    ran: bool
    strategy: str | None
    scored_for_quant: Artifact | None
    transfers: Artifact | None
    n_transfers: int | None
    notes: tuple[str, ...] = ()


def _strategy(rs: ResultSet) -> str | None:
    exp = rs.manifest.experiment or {}
    value = exp.get("mbr")
    if value is None:
        value = rs.manifest.model_identities.get("mbr")
    return None if value is None else str(value)


def mbr_ran(rs: ResultSet) -> bool:
    """True when the manifest records an MBR strategy other than ``None``."""
    strategy = _strategy(rs)
    return strategy is not None and normalise_enum(strategy) != "none"


def _columns(artifact: Artifact | None) -> list[str]:
    if artifact is None or not artifact.usable:
        return []
    try:
        return list(artifact.parquet().schema.names)
    except (ViewerError, OSError):
        return []


def _count_flagged(rs: ResultSet, artifact: Artifact) -> int:
    """Rows flagged ``is_transferred`` in an MBR-augmented scored table."""
    row = execute_bound(
        rs.duck,
        "SELECT count(*) FROM read_parquet($p) WHERE coalesce(is_transferred, false)",
        {"p": sql_path(artifact.require())},
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _transfer_rows(artifact: Artifact | None) -> int | None:
    """Rows of a readable transfer table, else None."""
    if artifact is None or not artifact.usable:
        return None
    try:
        return int(artifact.parquet().num_rows)
    except (ViewerError, OSError):
        return None


def mbr_info(rs: ResultSet) -> MbrInfo:
    """Whether MBR ran, its tables and the transfer count, with notes for the user."""
    key = ("mbr_info", rs.scored.identity())
    if key in rs._memo:
        return rs._memo[key]
    strategy = _strategy(rs)
    ran = mbr_ran(rs)
    notes: list[str] = []
    table = rs.artifact("mbr_transferred")
    if "is_transferred" in _columns(rs.scored):
        notes.append(
            f"The pooled scored table {rs.scored.key} itself carries is_transferred: it is an "
            "MBR-augmented table, so its q_value and run_psm_q are lowered on transferred rows."
        )
    if not ran:
        for name in ("scored_mbr.parquet", "mbr_transferred.parquet"):
            if (rs.root / name).is_file():
                notes.append(
                    f"{name} is present, but the manifest records mbr = "
                    f"{strategy or 'nothing'}: the file is left over from an earlier run "
                    "into this directory and was not used by this one."
                )
        info = MbrInfo(False, strategy, None, None, 0, tuple(notes))
        rs._memo[key] = info
        return info

    sfq = rs.scored_for_quant
    if sfq is None:
        notes.append(
            "MBR ran, but scored_for_quant is scored_combined.parquet: the worker wrote no "
            "augmented table (MuMDIA 0.1.0 with no transfer candidates)."
        )
    else:
        notes.append(
            f"Match-between-runs ran (strategy {strategy}). Quant, LFQ and the TSV report read "
            f"{sfq.path.name if sfq.path else sfq.key} (scored_for_quant), in which q_value, "
            "run_psm_q and experiment_psm_q are lowered on transferred rows. The "
            "identification counts of this viewer come from scored_combined.parquet, which "
            "MBR does not change."
        )
    n_transfers = _transfer_rows(table)
    if table is None or n_transfers is None:
        notes.append("mbr_transferred.parquet is missing or unreadable.")
        table = None
        if sfq is not None and "is_transferred" in _columns(sfq):
            n_transfers = _count_flagged(rs, sfq)
            notes.append(f"The transfers are read from is_transferred of {sfq.path.name}.")
    else:
        notes.append("mbr_transferred.parquet has no manifest record and no recorded content hash.")
        if sfq is not None and "is_transferred" in _columns(sfq):
            flagged = _count_flagged(rs, sfq)
            if flagged != n_transfers:
                notes.append(
                    f"mbr_transferred.parquet has {n_transfers:,} rows, but {flagged:,} rows of "
                    f"{sfq.path.name} are flagged is_transferred."
                )
    if n_transfers is not None:
        q_transfer = rs.config_get("mbr", "q_transfer")
        accept = f" at transfer_q <= {format_threshold(q_transfer)}" if q_transfer else ""
        verb = "transfer was" if n_transfers == 1 else "transfers were"
        notes.append(
            f"{n_transfers:,} {verb} accepted{accept} (permuted-RT null). A transfer is not "
            "FDR-controlled at the identification threshold; transfers are shown separately "
            "and never added to the identification counts."
        )
    exp = rs.manifest.experiment or {}
    n_runs = exp.get("n_runs") or len(rs.runs)
    min_anchor = rs.config_get("mbr", "min_anchor_runs", default=2)
    try:
        n_runs_i, min_anchor_i = int(n_runs), int(min_anchor)
    except (TypeError, ValueError):
        n_runs_i, min_anchor_i = len(rs.runs), 2
    if n_runs_i - 1 < min_anchor_i:
        notes.append(
            f"No transfer is possible in this experiment (derived from the configuration): "
            f"a precursor needs mbr.min_anchor_runs = {min_anchor_i} other runs in which it "
            f"is confident, and each run here has only {n_runs_i - 1} other run(s)."
        )
    notes.append(
        "The worker's candidate count, its null draws and the RT window were printed to the "
        "console only; the output does not record them."
    )
    info = MbrInfo(True, strategy, sfq, table, n_transfers, tuple(notes))
    rs._memo[key] = info
    return info


def _transfer_relation(rs: ResultSet, info: MbrInfo) -> tuple[str, dict[str, Any]] | None:
    """SQL for the accepted transfers as ``(source, candidate_id)``, or None without MBR."""
    if not info.ran:
        return None
    if info.transfers is not None:
        if (_transfer_rows(info.transfers) or 0) == 0:
            return _EMPTY_TRANSFERS, {}
        return (
            "SELECT DISTINCT CAST(source AS UINTEGER) AS source, "
            "CAST(candidate_id AS UINTEGER) AS candidate_id FROM read_parquet($transfers) "
            "WHERE source IS NOT NULL AND candidate_id IS NOT NULL",
            {"transfers": sql_path(info.transfers.require())},
        )
    sfq = info.scored_for_quant
    if sfq is not None and "is_transferred" in _columns(sfq):
        return (
            "SELECT DISTINCT CAST(source AS UINTEGER) AS source, "
            "CAST(candidate_id AS UINTEGER) AS candidate_id FROM read_parquet($transfers) "
            "WHERE coalesce(is_transferred, false)",
            {"transfers": sql_path(sfq.require())},
        )
    return _EMPTY_TRANSFERS, {}


def _run_label(rs: ResultSet, source: int) -> str:
    try:
        return rs.run(source).label
    except KeyError:
        return f"source {source}"


def _transfers_identity(info: MbrInfo) -> str | None:
    art = info.transfers if info.transfers is not None else info.scored_for_quant
    if art is None or not art.usable:
        return None
    try:
        return art.identity()
    except (ViewerError, OSError):
        return None


def transfer_counts(rs: ResultSet, t: float = 0.01) -> pd.DataFrame:
    """Native PSMs and MBR transfers per run.

    Columns: ``run``, ``source``, ``native_psms`` (target rows with ``run_psm_q <= t`` in
    ``scored_combined.parquet``), ``spike_in_psms`` when the library has entrapment
    markers, ``transfers`` (accepted transfers into the run), ``added_by_mbr``
    (transferred rows that do not pass ``run_psm_q <= t`` natively) and
    ``native_or_transferred`` (``label = 'target'`` rows that pass natively or were
    transferred; the acceptance rule of ``n_runs`` in ``peptides.tsv``). The column
    labels are in ``attrs['labels']``.

    ``native_psms`` and ``spike_in_psms`` use the classes of the identification counts
    (:func:`.entrapment.count_classes`), so they equal ``target_psms`` and
    ``spike_in_psms`` of :func:`.counts.per_run_counts`. In entrapment mode
    ``native_psms`` counts real targets and the spike-ins are in ``spike_in_psms``.
    ``native_or_transferred`` follows the engine's report and quant rule, which keeps
    every ``label = 'target'`` row, spike-ins included.
    """
    t = check_threshold("run_psm", t)
    info = mbr_info(rs)
    cls = count_classes(rs)
    key = ("mbr_transfer_counts", rs.scored.identity(), _transfers_identity(info), t, cls.key)
    if key in rs._memo:
        return rs._memo[key].copy()
    rel = _transfer_relation(rs, info)
    params: dict[str, Any] = {**cls.params, "scored": sql_path(rs.scored.require()), "t": t}
    # The class predicates read label and protein; the transfer relation has neither
    # column, so the unqualified names resolve to the scored table.
    native = f"count(*) FILTER (WHERE {cls.target} AND run_psm_q <= $t) AS native_psms"
    spike = f"count(*) FILTER (WHERE {cls.spike} AND run_psm_q <= $t) AS spike_in_psms"
    columns = "source, candidate_id, label, run_psm_q" + (", protein" if cls.params else "")
    if rel is None:
        sql = (
            f"SELECT source, {native}, {spike}, 0 AS transfers, 0 AS added_by_mbr, "
            "count(*) FILTER (WHERE label = 'target' AND run_psm_q <= $t) "
            "AS native_or_transferred FROM read_parquet($scored) GROUP BY source ORDER BY source"
        )
        unmatched = 0
    else:
        rel_sql, rel_params = rel
        params.update(rel_params)
        sql = (
            f"WITH s AS (SELECT {columns} FROM read_parquet($scored)), tr AS ({rel_sql}) "
            f"SELECT s.source AS source, {native}, {spike}, "
            "count(tr.candidate_id) AS transfers, "
            "count(tr.candidate_id) FILTER (WHERE NOT (s.run_psm_q <= $t)) AS added_by_mbr, "
            "count(*) FILTER (WHERE s.label = 'target' AND (s.run_psm_q <= $t "
            "OR tr.candidate_id IS NOT NULL)) AS native_or_transferred "
            "FROM s LEFT JOIN tr USING (source, candidate_id) GROUP BY s.source ORDER BY s.source"
        )
        row = execute_bound(
            rs.duck,
            f"WITH s AS (SELECT source, candidate_id FROM read_parquet($scored)), "
            f"tr AS ({rel_sql}) SELECT count(*) FROM tr ANTI JOIN s USING (source, "
            "candidate_id)",
            params,
        ).fetchone()
        unmatched = int(row[0]) if row is not None else 0
    found = {int(r[0]): r[1:] for r in execute_bound(rs.duck, sql, params).fetchall()}
    names = ["native_psms", "spike_in_psms", "transfers", "added_by_mbr", "native_or_transferred"]
    records = []
    sources = [r.index for r in rs.runs] + sorted(
        s for s in found if s not in {r.index for r in rs.runs}
    )
    for source in sources:
        values = found.get(source, (0,) * len(names))
        records.append(
            {
                "run": _run_label(rs, source),
                "source": source,
                **{n: int(v) for n, v in zip(names, values, strict=True)},
            }
        )
    df = pd.DataFrame(records, columns=["run", "source", *names])
    if not cls.spike_present:
        df = df.drop(columns=["spike_in_psms"])
    tt = format_threshold(t)
    if cls.excludes_spike_ins:
        native_label = (
            f"real target PSMs (rows, run_psm_q <= {tt}) in scored_combined.parquet: native "
            "identifications, before MBR; spike-ins are in spike_in_psms"
        )
        if not cls.markers_recorded:
            native_label += (
                "; real targets by the viewer's spike-in rule, which the run does not record"
            )
    elif cls.spike_present:
        native_label = (
            f"target PSMs (rows, run_psm_q <= {tt}) in scored_combined.parquet: native "
            "identifications, before MBR; spike-ins included, as in the engine's counts"
        )
    else:
        native_label = (
            f"target PSMs (rows, run_psm_q <= {tt}) in scored_combined.parquet: native "
            "identifications, before MBR"
        )
    labels = {
        "native_psms": native_label,
        "spike_in_psms": f"spike-in PSMs (rows, run_psm_q <= {tt}) in scored_combined.parquet, "
        "before MBR; "
        + ("not in native_psms" if cls.excludes_spike_ins else "also counted in native_psms"),
        "transfers": "accepted MBR transfers into the run (mbr_transferred.parquet); "
        "accepted on transfer_q under a permuted-RT null, not FDR-controlled at "
        f"run_psm_q <= {tt}",
        "added_by_mbr": f"transferred rows that do not pass run_psm_q <= {tt} natively",
        "native_or_transferred": f"label = 'target' rows with run_psm_q <= {tt} or transferred "
        "(the acceptance rule of n_runs in peptides.tsv)"
        + ("; spike-ins included, as in the TSV report and quant" if cls.spike_present else ""),
    }
    note = _transfer_note(info, unmatched)
    if cls.spike_present and cls.note:
        note += " " + cls.note
    df.attrs.update(
        {
            "threshold": t,
            "q_column": "run_psm_q",
            "mbr_ran": info.ran,
            "labels": {c: labels[c] for c in df.columns if c in labels},
            "note": note,
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


def _transfer_note(info: MbrInfo, unmatched: int) -> str:
    if not info.ran:
        return "MBR did not run: every transfer count is 0."
    if not info.n_transfers:
        return "MBR ran and accepted no transfer: every transfer count is 0."
    if unmatched:
        return f"{unmatched:,} transfer(s) match no row of scored_combined.parquet."
    return "Every transfer matches a row of scored_combined.parquet."


def _empty_transfers() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in _TRANSFER_COLUMNS})


def transfers_for_run(rs: ResultSet, source: int | str) -> pd.DataFrame:
    """The accepted transfers into one run, with their native values.

    One row per transfer, sorted by ``candidate_id``. ``peptidoform`` to ``score`` and
    the ``native_*`` q columns come from ``scored_combined.parquet``; ``transfer_q`` and
    the RT evidence (``expected_rt``, ``observed_rt``, ``rt_delta``, seconds) from
    ``mbr_transferred.parquet``; ``q_value_after_mbr`` and ``run_psm_q_after_mbr`` are
    the lowered values in ``scored_for_quant`` that quant and the report used. A null
    or NaN ``transfer_q`` is returned as NaN (no transfer q recorded).
    """
    run = rs.run(run_key(source))
    info = mbr_info(rs)
    key = ("mbr_transfers_for_run", rs.scored.identity(), _transfers_identity(info), run.index)
    if key in rs._memo:
        return rs._memo[key].copy()
    if not info.ran:
        df = _empty_transfers()
        df.attrs["note"] = "MBR did not run."
        rs._memo[key] = df.copy()
        return df
    params: dict[str, Any] = {"scored": sql_path(rs.scored.require()), "i": int(run.index)}
    table = info.transfers
    sfq = info.scored_for_quant
    if table is not None and (_transfer_rows(table) or 0) == 0:
        df = _empty_transfers()
        df.attrs["note"] = "MBR accepted no transfer."
        rs._memo[key] = df.copy()
        return df
    if table is not None:
        names = set(_columns(table))
        params["transfers"] = sql_path(table.require())
        extra = []
        for c in ("expected_rt", "observed_rt", "rt_delta"):
            extra.append(
                f"CAST({c} AS DOUBLE) AS {c}" if c in names else f"CAST(NULL AS DOUBLE) AS {c}"
            )
        tq = (
            "CASE WHEN transfer_q IS NULL OR isnan(CAST(transfer_q AS DOUBLE)) THEN NULL "
            "ELSE CAST(transfer_q AS DOUBLE) END AS transfer_q"
            if "transfer_q" in names
            else "CAST(NULL AS DOUBLE) AS transfer_q"
        )
        tr_sql = (
            f"SELECT CAST(candidate_id AS UINTEGER) AS candidate_id, {', '.join(extra)}, {tq} "
            "FROM read_parquet($transfers) WHERE source = $i AND candidate_id IS NOT NULL"
        )
    elif sfq is not None and "is_transferred" in _columns(sfq):
        names = set(_columns(sfq))
        params["transfers"] = sql_path(sfq.require())
        tq = (
            "CASE WHEN transfer_q IS NULL OR isnan(transfer_q) THEN NULL ELSE transfer_q END"
            if "transfer_q" in names
            else "CAST(NULL AS DOUBLE)"
        )
        tr_sql = (
            "SELECT CAST(candidate_id AS UINTEGER) AS candidate_id, "
            "CAST(NULL AS DOUBLE) AS expected_rt, CAST(NULL AS DOUBLE) AS observed_rt, "
            f"CAST(NULL AS DOUBLE) AS rt_delta, {tq} AS transfer_q FROM read_parquet($transfers) "
            "WHERE source = $i AND coalesce(is_transferred, false)"
        )
    else:
        df = _empty_transfers()
        df.attrs["note"] = "MBR ran, but no transfer table is readable."
        rs._memo[key] = df.copy()
        return df
    if sfq is not None and sfq.usable:
        params["sfq"] = sql_path(sfq.require())
        after = (
            "LEFT JOIN (SELECT CAST(candidate_id AS UINTEGER) AS candidate_id, "
            "q_value AS q_value_after_mbr, run_psm_q AS run_psm_q_after_mbr "
            "FROM read_parquet($sfq) WHERE source = $i) m USING (candidate_id)"
        )
        after_cols = "m.q_value_after_mbr, m.run_psm_q_after_mbr"
    else:
        after = ""
        after_cols = (
            "CAST(NULL AS DOUBLE) AS q_value_after_mbr, CAST(NULL AS DOUBLE) AS run_psm_q_after_mbr"
        )
    sql = (
        f"WITH tr AS ({tr_sql}), "
        "s AS (SELECT candidate_id, peptidoform, charge, label, protein_group, apex_rt, score, "
        "q_value, run_psm_q, precursor_q, peptide_q_value, pg_q_value "
        "FROM read_parquet($scored) WHERE source = $i) "
        "SELECT tr.candidate_id, s.peptidoform, s.charge, s.label, s.protein_group, s.apex_rt, "
        "s.score, s.q_value AS native_q_value, s.run_psm_q AS native_run_psm_q, "
        "s.precursor_q AS native_precursor_q, s.peptide_q_value AS native_peptide_q_value, "
        "s.pg_q_value AS native_pg_q_value, tr.transfer_q, tr.expected_rt, tr.observed_rt, "
        f"tr.rt_delta, {after_cols} "
        f"FROM tr LEFT JOIN s USING (candidate_id) {after} ORDER BY tr.candidate_id"
    )
    df = execute_bound(rs.duck, sql, params).df()
    df = df[_TRANSFER_COLUMNS]
    df.attrs.update(
        {
            "run": run.label,
            "source": run.index,
            "labels": {
                "native_q_value": "pooled q_value in scored_combined.parquet (before MBR)",
                "native_run_psm_q": "run_psm_q in scored_combined.parquet (before MBR)",
                "transfer_q": "q at which the transfer was accepted (permuted-RT null); "
                "missing when not recorded",
                "q_value_after_mbr": "q_value used by quant and the report after MBR "
                "(min of the native q and transfer_q)",
                "run_psm_q_after_mbr": "run_psm_q used by the report after MBR",
                "rt_delta": "abs(observed_rt - expected_rt), seconds",
            },
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def transfer_of(rs: ResultSet, source: int | str, cid: int) -> dict[str, Any] | None:
    """The transfer of candidate ``cid`` into run ``source``, or None when it was not transferred.

    The dict has the columns of :func:`transfers_for_run` plus ``run`` and ``source``;
    missing values are None.
    """
    run = rs.run(run_key(source))
    df = transfers_for_run(rs, run.index)
    if df.empty:
        return None
    hit = df[df["candidate_id"] == int(cid)]
    if hit.empty:
        return None
    record = {k: _plain(v) for k, v in hit.iloc[0].to_dict().items()}
    record["run"] = run.label
    record["source"] = run.index
    return record


def _report_threshold(rs: ResultSet) -> tuple[float, str]:
    """The threshold of the TSV report, and where it was read.

    ``experiment.report.q_threshold``, else ``quant.q_threshold`` (the value
    ``run-experiment`` gives the report stage), else 0.01. A later
    ``mumdia report --experiment-dir`` can rewrite the TSVs at another threshold.
    """
    exp = rs.manifest.experiment or {}
    report = exp.get("report") if isinstance(exp.get("report"), dict) else {}
    candidates = (
        (report.get("q_threshold"), "experiment.report.q_threshold"),
        (
            rs.config_get("quant", "q_threshold"),
            "config_json quant.q_threshold, the value run-experiment gives the report",
        ),
    )
    for value, source in candidates:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number, source
    return 0.01, "the viewer default; the manifest records no report threshold"


def n_runs_decomposition(
    rs: ResultSet,
    t: float | None = None,
    *,
    level: str = "precursor",
    include_zero: bool = False,
) -> pd.DataFrame:
    """The TSV's ``n_runs`` rule at a threshold, split into native and transfer runs.

    Per key (``(peptidoform, charge)`` for ``level='precursor'``, non-empty
    ``protein_group`` for ``level='protein_group'``) over ``label = 'target'`` rows:

    * ``n_runs_native``: runs with a row at ``run_psm_q <= t`` in
      ``scored_combined.parquet``;
    * ``n_runs_transfer_only``: further runs in which the key was only transferred;
    * ``n_runs_engine``: their sum, the engine's rule (``run_psm_q <= t`` or
      transferred).

    ``t`` defaults to the report threshold (``experiment.report.q_threshold``, else
    ``quant.q_threshold``, else 0.01). ``n_runs_engine`` equals the ``n_runs`` column of
    ``peptides.tsv`` (``proteins.tsv`` for protein groups) only at that threshold, and
    the label says so. Keys with ``n_runs_engine = 0`` are left out unless
    ``include_zero`` is True. These are PSM-level acceptances per run, not precursor- or
    protein-level FDR. Like the TSV, the rule keeps spike-ins in entrapment runs.
    """
    if level not in ("precursor", "protein_group"):
        raise ValueError(f"level must be 'precursor' or 'protein_group', not {level!r}.")
    report_t, report_source = _report_threshold(rs)
    t = check_threshold("run_psm", report_t if t is None else t)
    info = mbr_info(rs)
    key = (
        "mbr_n_runs",
        rs.scored.identity(),
        _transfers_identity(info),
        t,
        level,
        include_zero,
    )
    if key in rs._memo:
        return rs._memo[key].copy()
    keys = ["peptidoform", "charge"] if level == "precursor" else ["protein_group"]
    keycols = ", ".join(keys)
    nonempty = " AND protein_group <> ''" if level == "protein_group" else ""
    rel = _transfer_relation(rs, info)
    rel_sql, rel_params = rel if rel is not None else (_EMPTY_TRANSFERS, {})
    params: dict[str, Any] = {"scored": sql_path(rs.scored.require()), "t": t, **rel_params}
    having = "" if include_zero else " HAVING n_runs_engine > 0"
    sql = (
        f"WITH s AS (SELECT source, candidate_id, {keycols}, run_psm_q FROM "
        f"read_parquet($scored) WHERE label = 'target'{nonempty}), tr AS ({rel_sql}), "
        "x AS (SELECT s.*, tr.candidate_id IS NOT NULL AS is_tr, s.run_psm_q <= $t AS native "
        "FROM s LEFT JOIN tr USING (source, candidate_id)) "
        f"SELECT {keycols}, count(DISTINCT source) FILTER (WHERE native) AS n_runs_native, "
        "count(DISTINCT source) FILTER (WHERE native OR is_tr) AS n_runs_engine "
        f"FROM x GROUP BY {keycols}{having} ORDER BY {keycols}"
    )
    df = execute_bound(rs.duck, sql, params).df()
    df["n_runs_native"] = df["n_runs_native"].astype("int64")
    df["n_runs_engine"] = df["n_runs_engine"].astype("int64")
    df["n_runs_transfer_only"] = df["n_runs_engine"] - df["n_runs_native"]
    df = df[[*keys, "n_runs_native", "n_runs_transfer_only", "n_runs_engine"]]
    tt = format_threshold(t)
    report_tt = format_threshold(report_t)
    tsv = "peptides.tsv" if level == "precursor" else "proteins.tsv"
    engine_label = f"runs with run_psm_q <= {tt} or a transfer (the TSV's n_runs rule)"
    if t == report_t:
        engine_label += f"; {tt} is the report threshold ({report_source})"
    else:
        engine_label += (
            f"; it equals {tsv} n_runs only at the report threshold ({report_tt}, {report_source})"
        )
    notes = []
    if count_classes(rs).spike_present:
        notes.append(
            "The rule counts label = 'target' rows, so spike-ins are included, as in the TSV."
        )
    df.attrs.update(
        {
            "threshold": t,
            "report_threshold": report_t,
            "level": level,
            "labels": {
                "n_runs_native": f"runs with a target row at run_psm_q <= {tt} in "
                "scored_combined.parquet (PSM-level FDR within each run)",
                "n_runs_transfer_only": "further runs in which the key was only transferred by MBR",
                "n_runs_engine": engine_label,
            },
            "note": " ".join(notes),
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df
