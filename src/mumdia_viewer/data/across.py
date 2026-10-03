"""One precursor across the runs of an experiment, and condition ratios (P2 view 9).

**One precursor across runs** (:func:`precursor_across`, :func:`run_xics`). A precursor
is a ``(peptidoform, charge)``. In an experiment the pooled scored table
(``scored_combined.parquet``, ``source`` = run index) holds one row per run where the
precursor was scored; a run without a row did not score it. For every run the viewer
reads, without recomputing anything:

* the scored row: ``score``, ``q_value``, ``run_psm_q`` (this run only), ``apex_rt``,
  ``elution_lo`` and ``elution_hi``;
* the quant state of that row (:func:`.quant.quant_states`): a quantity from
  ``peptide_quant``, or why there is none (``not_quantifiable`` with a
  ``quant_status``, ``not_selected`` by the quant gate). A missing quantity is NaN,
  never 0;
* with match-between-runs, whether the row is a transfer (:func:`.mbr.transfers_for_run`);
* the MaxLFQ precursor quantity of the run (``lfq_maxlfq.parquet.precursor.parquet``);
* the chromatogram, the extraction window (``run_windows``) and quant's integration
  bounds, for the XICs.

The grouped q columns are experiment-wide and set on one winning row of each group
(other rows hold 1.0). :class:`GroupQ` gives each group's value from its winning row and
says which run holds it. The winner of the precursor and the base-peptide group follows
the engine's rule (:func:`.competition.winner_index`, labelled viewer-derived); the
protein group's value is the minimum ``pg_q_value`` over the group's rows, which is the
winning row's value because every other row holds 1.0.

**Condition ratios** (:func:`condition_ratios`). The viewer groups the runs into
conditions (``mv-conditions``, :func:`.quantqc.resolve_conditions`), summarises each
key's quantities over the runs of a condition that have a value (the median or the mean
of the linear quantities; a condition needs a set number of values), and takes
``log2(A / B)``. The species of a key is the viewer's rule on its protein group's member
names: the entry-name suffix (``_HUMAN``, ``_YEAST``, ``_ECOLI``, editable). Expected
ratios are what the user enters (for example a benchmark design); the viewer never
infers them. Nothing here is the engine's result, and every label says so.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .chromatograms import CandidateChromatogram, ChromatogramSource
from .competition import Q_COLUMNS, WINNER_SOURCE, base_peptide_rows, winner_index
from .discovery import ResultSet, Run
from .duck import sql_path
from .errors import ViewerError
from .mbr import mbr_ran, transfers_for_run
from .quant import quant_states
from .quantqc import KEY_COLUMNS, condition_groups, quant_matrix, resolve_conditions
from .windows import rt_window

__all__ = [
    "DEFAULT_SUFFIXES",
    "HYE_DESIGN",
    "NEEDS",
    "SUMMARIES",
    "GroupQ",
    "PrecursorAcross",
    "RatioResult",
    "RunXic",
    "cached",
    "condition_ratios",
    "default_precursor",
    "find_precursors",
    "parse_ratio",
    "precursor_across",
    "precursor_groups",
    "ratio_histogram",
    "run_xics",
    "species_by_suffix",
]

# --------------------------------------------------------------------------- across runs

ROW_COLUMNS = (
    "run",
    "source",
    "scored",
    "n_rows",
    "candidate_id",
    "label",
    "protein_group",
    "score",
    "q_value",
    "run_psm_q",
    "apex_rt",
    "elution_lo",
    "elution_hi",
    "selected_peak_rank",
    "state",
    "status",
    "quantity",
    "n_fragments_used",
    "integration_apex_rt",
    "integration_lo_rt",
    "integration_hi_rt",
    "reason",
    "transferred",
    "transfer_q",
    "q_value_after_mbr",
    "run_psm_q_after_mbr",
    "lfq_quantity",
)

ROW_LABELS: dict[str, str] = {
    "score": "score: the rescorer's score of the run's row (scored table)",
    "q_value": "q_value: PSM rows pooled over the whole rescore (experiment-wide)",
    "run_psm_q": "run_psm_q: PSM rows of this run only, target-decoy re-run per run",
    "apex_rt": "apex_rt: the identification's apex (s)",
    "elution": "elution_lo to elution_hi: the identification's elution bounds (s)",
    "quantity": (
        "peptide_quant.quantity of the run's row: the sum of the top-N positive fragment "
        "areas over the integration window; not normalized across runs"
    ),
    "lfq_quantity": (
        "MaxLFQ precursor quantity of the run (lfq_maxlfq.parquet.precursor.parquet), "
        "after one median-ratio size factor per run; 0.0 (no feature) is shown as missing"
    ),
    "integration": "integration_lo_rt to integration_hi_rt of peptide_quant: what quant integrated",
}

STATE_TEXT = {
    "quantified": "quantified",
    "not_quantifiable": "not quantifiable",
    "not_selected": "not selected",
    "not_scored": "not scored",
}


@dataclass(frozen=True)
class GroupQ:
    """The experiment-wide value of one grouped q column for this precursor's group.

    ``value`` is the q on the group's winning row (None when the group has no row).
    ``winner_run`` / ``winner_cid`` / ``winner_peptidoform`` / ``winner_charge`` /
    ``winner_label`` locate that row; ``this_precursor`` says whether it is a row of this
    precursor. ``source`` says how the winning row was found.
    """

    column: str
    unit: str
    group: str
    value: float | None
    winner_run: str | None
    winner_cid: int | None
    winner_peptidoform: str | None
    winner_charge: int | None
    winner_label: str | None
    this_precursor: bool
    source: str


@dataclass
class PrecursorAcross:
    """One precursor in every run of the result set (see the module docstring).

    ``rows`` has one row per run of ``rs.runs`` (in order) with :data:`ROW_COLUMNS`;
    ``scored`` is False for a run without a scored row (``state`` ``not_scored``).
    ``siblings`` lists the other precursors of the base peptide (peptidoform, charge,
    label, n_runs, best precursor_q).
    """

    peptidoform: str
    charge: int
    label: str
    protein_group: str | None
    base_peptide_id: int | None
    experiment: bool
    mbr: bool
    rows: pd.DataFrame
    grouped: tuple[GroupQ, ...]
    siblings: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.rows)

    @property
    def n_scored(self) -> int:
        return int(self.rows["scored"].sum())

    @property
    def n_quantified(self) -> int:
        return int((self.rows["state"] == "quantified").sum())

    def group(self, column: str) -> GroupQ | None:
        for g in self.grouped:
            if g.column == column:
                return g
        return None


def _plain(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _lfq_precursor(rs: ResultSet, peptidoform: str, charge: int) -> dict[int, float]:
    """MaxLFQ precursor quantity per run index (0.0 and non-finite left out)."""
    if not rs.is_experiment:
        return {}
    art = rs.artifact("lfq_maxlfq_precursor")
    if art is None or not art.usable:
        return {}
    df = rs.duck.df(
        "SELECT run::INTEGER AS run, quantity FROM read_parquet(?) "
        'WHERE "group" = ? AND charge = ? AND isfinite(quantity) AND quantity > 0',
        [sql_path(art.require()), peptidoform, int(charge)],
    )
    return {int(r): float(q) for r, q in zip(df["run"], df["quantity"], strict=True)}


def _group_winner(
    rs: ResultSet, column: str, group: str, rows: pd.DataFrame, this: tuple[str, int]
) -> GroupQ:
    unit = Q_COLUMNS[column][0]
    if rows.empty:
        return GroupQ(column, unit, group, None, None, None, None, None, None, False, "no row")
    if column == "pg_q_value":
        i = rows["pg_q_value"].astype("float64").idxmin()
        source = (
            "the minimum pg_q_value over the group's rows (the winning row's value; the other "
            "rows hold 1.0)"
        )
    else:
        i = winner_index(rows)
        source = WINNER_SOURCE
    r = rows.loc[i]
    names = {run.index: run.name for run in rs.runs}
    return GroupQ(
        column=column,
        unit=unit,
        group=group,
        value=_plain(float(r[column])),
        winner_run=names.get(int(r["source"]), str(r["source"])),
        winner_cid=int(r["candidate_id"]),
        winner_peptidoform=str(r["peptidoform"]),
        winner_charge=int(r["charge"]),
        winner_label=str(r["label"]),
        this_precursor=(str(r["peptidoform"]), int(r["charge"])) == this,
        source=source,
    )


def _pg_rows(rs: ResultSet, group: str) -> pd.DataFrame:
    return rs.duck.df(
        "SELECT source, candidate_id, peptidoform, charge, label, score, pg_q_value "
        "FROM read_parquet(?) WHERE protein_group = ? ORDER BY pg_q_value, source",
        [sql_path(rs.scored.require()), group],
    )


def _row_of(run: Run, rows: pd.DataFrame) -> tuple[dict[str, Any], int]:
    """The run's best-scoring row of the precursor and the number of its rows."""
    mine = rows[rows["source"] == run.index]
    if mine.empty:
        return {}, 0
    best = mine.sort_values("score", ascending=False, kind="mergesort").iloc[0]
    return {k: _plain(v) for k, v in best.to_dict().items()}, len(mine)


def _build_across(rs: ResultSet, peptidoform: str, charge: int) -> PrecursorAcross:
    scored = sql_path(rs.scored.require())
    head = rs.duck.df(
        "SELECT base_peptide_id, label, protein_group, count(*) AS n FROM read_parquet(?) "
        "WHERE peptidoform = ? AND charge = ? GROUP BY ALL ORDER BY n DESC",
        [scored, peptidoform, int(charge)],
    )
    if head.empty:
        raise ViewerError(
            f"{peptidoform} {charge}+ has no scored row in this result set "
            "(the scored table holds only scored candidates)."
        )
    notes: list[str] = []
    bpid = _plain(head["base_peptide_id"].iloc[0])
    label = str(head["label"].iloc[0])
    group = _plain(head["protein_group"].iloc[0])
    if len(head) > 1:
        notes.append(
            f"The precursor's rows carry {len(head)} combinations of base_peptide_id, label "
            "and protein_group; the most frequent is shown."
        )
    bp = base_peptide_rows(rs, int(bpid)) if bpid is not None else pd.DataFrame()
    mine = (
        bp[(bp["peptidoform"] == peptidoform) & (bp["charge"] == int(charge))]
        if not bp.empty
        else bp
    )
    this = (peptidoform, int(charge))
    grouped = [
        _group_winner(rs, "precursor_q", f"{peptidoform} {charge}+", mine, this),
        _group_winner(
            rs, "peptide_q_value", f"base_peptide_id {bpid}", bp if bpid is not None else mine, this
        ),
    ]
    if group:
        grouped.append(_group_winner(rs, "pg_q_value", str(group), _pg_rows(rs, str(group)), this))
    full = rs.duck.df(
        "SELECT source, candidate_id, label, protein_group, score, q_value, run_psm_q, apex_rt, "
        "elution_lo, elution_hi, "
        + (
            "selected_peak_rank"
            if rs.scored.parquet().has_column("selected_peak_rank")
            else "0 AS selected_peak_rank"
        )
        + " FROM read_parquet(?) WHERE peptidoform = ? AND charge = ? ORDER BY source",
        [scored, peptidoform, int(charge)],
    )
    lfq = _lfq_precursor(rs, peptidoform, int(charge))
    with_mbr = rs.is_experiment and mbr_ran(rs)
    records: list[dict[str, Any]] = []
    for run in rs.runs:
        row, n = _row_of(run, full)
        rec: dict[str, Any] = {c: None for c in ROW_COLUMNS}
        rec.update(run=run.name, source=int(run.index), scored=n > 0, n_rows=n)
        rec["lfq_quantity"] = lfq.get(int(run.index))
        rec["transferred"] = False
        if n == 0:
            rec.update(
                state="not_scored",
                reason=f"{peptidoform} {charge}+ has no scored row in {run.label}.",
            )
            records.append(rec)
            continue
        if n > 1:
            notes.append(
                f"{run.label}: {n} scored rows of this precursor; the best-scoring one "
                f"(candidate {row['candidate_id']}) is shown."
            )
        for k in (
            "candidate_id",
            "label",
            "protein_group",
            "score",
            "q_value",
            "run_psm_q",
            "apex_rt",
            "elution_lo",
            "elution_hi",
            "selected_peak_rank",
        ):
            rec[k] = row.get(k)
        cid = int(row["candidate_id"])
        try:
            qs = quant_states(rs, run, [cid]).iloc[0]
            rec.update(
                state=str(qs["state"]),
                status=_plain(qs["status"]),
                quantity=_plain(qs["quantity"]),
                n_fragments_used=_plain(qs["n_fragments_used"]),
                integration_apex_rt=_plain(qs["integration_apex_rt"]),
                integration_lo_rt=_plain(qs["integration_lo_rt"]),
                integration_hi_rt=_plain(qs["integration_hi_rt"]),
                reason=str(qs["reason"]),
                transferred=bool(qs["from_transfer"]),
                transfer_q=_plain(qs["transfer_q"]),
            )
        except ViewerError as exc:
            rec.update(state="not_selected", reason=f"no quant state: {exc}")
        if with_mbr:
            tr = transfers_for_run(rs, run.index)
            hit = tr[tr["candidate_id"] == cid] if not tr.empty else tr
            if not hit.empty:
                h = hit.iloc[0]
                rec["transferred"] = True
                rec["transfer_q"] = _plain(h["transfer_q"])
                rec["q_value_after_mbr"] = _plain(h["q_value_after_mbr"])
                rec["run_psm_q_after_mbr"] = _plain(h["run_psm_q_after_mbr"])
        records.append(rec)
    rows = pd.DataFrame.from_records(records, columns=list(ROW_COLUMNS))
    for c in ("quantity", "lfq_quantity", "score", "q_value", "run_psm_q", "transfer_q"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce").astype("float64")
    siblings = _siblings(bp, this)
    return PrecursorAcross(
        peptidoform=peptidoform,
        charge=int(charge),
        label=label,
        protein_group=None if group is None else str(group),
        base_peptide_id=None if bpid is None else int(bpid),
        experiment=rs.is_experiment,
        mbr=with_mbr,
        rows=rows,
        grouped=tuple(grouped),
        siblings=siblings,
        notes=notes,
    )


def _siblings(bp: pd.DataFrame, this: tuple[str, int]) -> pd.DataFrame:
    cols = ["peptidoform", "charge", "label", "n_runs", "precursor_q"]
    if bp.empty:
        return pd.DataFrame(columns=cols)
    g = (
        bp.groupby(["peptidoform", "charge", "label"], sort=False)
        .agg(n_runs=("source", "nunique"), precursor_q=("precursor_q", "min"))
        .reset_index()
    )
    keep = ~((g["peptidoform"] == this[0]) & (g["charge"] == this[1]))
    g = g[keep].sort_values(["label", "precursor_q", "charge"], ascending=[False, True, True])
    return g[cols].reset_index(drop=True)


def precursor_across(rs: ResultSet, peptidoform: str, charge: int) -> PrecursorAcross:
    """One precursor in every run (memoised on the result set; see the module docstring).

    Raises :class:`ViewerError` when the precursor has no scored row.
    """
    key = ("across.precursor", rs.scored.identity(), str(peptidoform), int(charge))
    return rs.memo(key, lambda: _build_across(rs, str(peptidoform), int(charge)))


@dataclass(frozen=True)
class RunXic:
    """What the XIC of one run shows: the trace and its markers (seconds)."""

    run: str
    candidate_id: int | None
    chromatogram: CandidateChromatogram | None
    apex_rt: float | None
    elution_lo: float | None
    elution_hi: float | None
    integration_lo: float | None
    integration_hi: float | None
    rt_pred_cal: float | None
    rt_lo: float | None
    rt_hi: float | None
    note: str | None = None

    def summed(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(axis, the viewer's sum of the fragment traces) on the common axis, or None."""
        if self.chromatogram is None:
            return None
        m = self.chromatogram.matrix(include_ms1=False)
        if m is None:
            return None
        axis, _, values = m
        return np.asarray(axis, dtype="float64"), values.astype("float64").sum(axis=0)


def _one_xic(rs: ResultSet, run: Run, rec: Mapping[str, Any]) -> RunXic:
    cid = rec.get("candidate_id")
    base = dict(
        run=run.name,
        apex_rt=_plain(rec.get("apex_rt")),
        elution_lo=_plain(rec.get("elution_lo")),
        elution_hi=_plain(rec.get("elution_hi")),
        integration_lo=_plain(rec.get("integration_lo_rt")),
        integration_hi=_plain(rec.get("integration_hi_rt")),
    )
    if not rec.get("scored") or cid is None:
        return RunXic(
            candidate_id=None,
            chromatogram=None,
            rt_pred_cal=None,
            rt_lo=None,
            rt_hi=None,
            note="not scored in this run",
            **base,
        )
    cid = int(cid)
    note = None
    chrom = None
    try:
        chrom = ChromatogramSource.for_run(rs, run).read(cid)
        if chrom is None:
            note = "no chromatogram rows for this candidate"
    except ViewerError as exc:
        note = f"chromatogram not readable: {exc}"
    pred = lo = hi = None
    try:
        w = rt_window(rs, run, cid)
        if w is not None:
            pred, lo, hi = w.rt_pred_cal, w.rt_lo, w.rt_hi
    except ViewerError:
        pass
    return RunXic(
        candidate_id=cid,
        chromatogram=chrom,
        rt_pred_cal=pred,
        rt_lo=lo,
        rt_hi=hi,
        note=note,
        **base,
    )


def run_xics(rs: ResultSet, across: PrecursorAcross, *, workers: int = 6) -> list[RunXic]:
    """The XIC of the precursor in every run (in run order), read in parallel.

    Memoised on the result set. A run without a scored row gives a RunXic without a
    chromatogram and the note ``not scored in this run``.
    """
    key = ("across.xics", rs.scored.identity(), across.peptidoform, across.charge)

    def build() -> list[RunXic]:
        records = across.rows.to_dict("records")
        jobs = list(zip(rs.runs, records, strict=True))
        if len(jobs) <= 1:
            return [_one_xic(rs, run, rec) for run, rec in jobs]
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            return list(pool.map(lambda j: _one_xic(rs, j[0], j[1]), jobs))

    return rs.memo(key, build)


def find_precursors(
    rs: ResultSet, text: str | None, *, limit: int = 30, t: float | None = None
) -> pd.DataFrame:
    """Target precursors whose peptidoform contains ``text`` (any case).

    Columns ``peptidoform``, ``charge``, ``n_runs`` (runs with a scored row) and
    ``precursor_q`` (the minimum over its rows: the experiment-wide value on its winning
    row), best first. ``t`` keeps the precursors with ``precursor_q <= t``.
    """
    needle = (text or "").strip()
    where = "label = 'target'"
    params: list[Any] = [sql_path(rs.scored.require())]
    if needle:
        where += " AND contains(lower(peptidoform), ?)"
        params.append(needle.lower())
    having = ""
    if t is not None:
        having = " HAVING min(precursor_q) <= ?"
        params.append(float(t))
    params.append(int(limit))
    return rs.duck.df(
        "SELECT peptidoform, charge, count(DISTINCT source) AS n_runs, "
        "min(precursor_q) AS precursor_q, max(score) AS score FROM read_parquet(?) "
        f"WHERE {where} GROUP BY 1, 2{having} "
        "ORDER BY precursor_q, n_runs DESC, score DESC, peptidoform, charge LIMIT ?",
        params,
    )


def default_precursor(rs: ResultSet, t: float, *, pool: int = 50) -> tuple[str, int] | None:
    """A precursor to open without one in the address (memoised).

    Among the ``pool`` best-scoring target rows accepted at ``t`` on ``precursor_q``
    (the precursors' winning rows), the precursor scored in the most runs, then the
    highest score. Without an accepted row, the best-scoring target row.
    """

    def build() -> tuple[str, int] | None:
        path = sql_path(rs.scored.require())
        df = rs.duck.df(
            "WITH best AS (SELECT peptidoform, charge, score FROM read_parquet(?) "
            "WHERE label = 'target' AND precursor_q <= ? ORDER BY score DESC, peptidoform, "
            "charge LIMIT ?) "
            "SELECT b.peptidoform, b.charge FROM best b JOIN (SELECT peptidoform, charge, "
            "count(DISTINCT source) AS n FROM read_parquet(?) WHERE label = 'target' AND "
            "(peptidoform, charge) IN (SELECT (peptidoform, charge) FROM best) GROUP BY 1, 2) "
            "c USING (peptidoform, charge) ORDER BY c.n DESC, b.score DESC, b.peptidoform, "
            "b.charge LIMIT 1",
            [path, float(t), int(pool), path],
        )
        if df.empty:
            df = rs.duck.df(
                "SELECT peptidoform, charge FROM read_parquet(?) WHERE label = 'target' "
                "ORDER BY score DESC, peptidoform, charge LIMIT 1",
                [path],
            )
        if df.empty:
            return None
        return str(df["peptidoform"].iloc[0]), int(df["charge"].iloc[0])

    return rs.memo(("across.default", rs.scored.identity(), float(t), int(pool)), build)


# --------------------------------------------------------------------------- ratios

DEFAULT_SUFFIXES: tuple[str, ...] = ("_HUMAN", "_YEAST", "_ECOLI")
# The HYE benchmark design (ProteoBench mixed-species LFQ): condition A over B. A
# suggestion the user enters; the viewer cannot know a sample's composition.
HYE_DESIGN: dict[str, str] = {"_HUMAN": "1:1", "_YEAST": "2:1", "_ECOLI": "1:4"}
SUMMARIES = ("median", "mean")
NEEDS = ("one", "two", "all")
MIXED = "mixed"
OTHER = "other"


def species_by_suffix(group: str | None, suffixes: Sequence[str]) -> str:
    """The species of a protein group by the viewer's suffix rule.

    The group is split into members at ``;`` and ``DECOY_`` is removed. A member belongs
    to the first suffix it ends with (case-sensitive, so ``_HUMAN`` matches
    ``ALBU_HUMAN``). The group's species is that suffix when every member with a match
    has the same one; :data:`MIXED` when members match different suffixes; :data:`OTHER`
    when no member matches.
    """
    found: set[str] = set()
    for member in str(group or "").split(";"):
        name = member.strip().removeprefix("DECOY_")
        for s in suffixes:
            if s and name.endswith(s):
                found.add(s)
                break
    if not found:
        return OTHER
    if len(found) > 1:
        return MIXED
    return next(iter(found))


_RATIO = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(?:[:/]\s*([0-9]*\.?[0-9]+))?\s*$")


def parse_ratio(text: Any) -> float | None:
    """log2 of a ratio written ``2:1``, ``1/4`` or ``0.25``; None when not a positive ratio."""
    if text is None:
        return None
    m = _RATIO.match(str(text))
    if m is None:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) is not None else 1.0
    if a <= 0 or b <= 0:
        return None
    return math.log2(a / b)


def precursor_groups(rs: ResultSet, t: float | None) -> pd.DataFrame:
    """The protein group of each target precursor (memoised).

    With ``t``, from the rows with ``precursor_q <= t`` (the winning rows); without, the
    group of the row with the lowest ``precursor_q``. Columns ``peptidoform``, ``charge``,
    ``protein_group``.
    """

    def build() -> pd.DataFrame:
        where = "label = 'target'"
        params: list[Any] = [sql_path(rs.scored.require())]
        if t is not None:
            where += " AND precursor_q <= ?"
            params.append(float(t))
        df = rs.duck.df(
            "SELECT peptidoform, charge, arg_min(protein_group, precursor_q) AS protein_group "
            f"FROM read_parquet(?) WHERE {where} GROUP BY 1, 2",
            params,
        )
        df["charge"] = df["charge"].astype("int64")
        return df

    key = ("across.pg", rs.scored.identity(), None if t is None else float(t))
    return rs.memo(key, build)


@dataclass(frozen=True)
class RatioResult:
    """log2(A / B) of one level between two conditions (the viewer's computation).

    ``table`` has the key columns, ``protein_group``, ``species``, ``n_a`` / ``n_b`` (runs
    of the condition with a value), ``value_a`` / ``value_b`` (the condition summary,
    NaN when the condition has fewer values than it needs), ``log2_ratio`` (NaN unless
    both values exist) and ``log10_mean`` (log10 of the mean of the two values).
    ``species`` has one row per species (the suffixes in order, then mixed and other):
    ``keys``, ``both``, ``only_a``, ``only_b``, ``neither``, and over the keys with both
    values ``median``, ``mean``, ``q25``, ``q75`` and ``sd`` (n - 1) of log2(A/B).
    """

    level: str
    source: str
    a: str
    b: str
    runs_a: tuple[str, ...]
    runs_b: tuple[str, ...]
    summary: str
    need: str
    need_a: int
    need_b: int
    accepted_at: float | None
    suffixes: tuple[str, ...]
    table: pd.DataFrame = field(repr=False)
    species: pd.DataFrame = field(repr=False)
    label: str = ""
    note: str = ""

    @property
    def n_both(self) -> int:
        return int(np.isfinite(self.table["log2_ratio"].to_numpy(dtype="float64")).sum())


def _need(need: str, n: int) -> int:
    if need == "all":
        return n
    if need == "two":
        return min(2, n)
    return 1


def _summarise(values: np.ndarray, summary: str, need: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.isfinite(values).sum(axis=1)
    out = np.full(values.shape[0], np.nan)
    ok = (n >= max(1, need)) & (n > 0)
    if ok.any():
        sub = values[ok]
        with np.errstate(all="ignore"):
            out[ok] = np.nanmedian(sub, axis=1) if summary == "median" else np.nanmean(sub, axis=1)
    return out, n


def ratio_rule(summary: str, need: str, source_name: str) -> str:
    needs = {
        "one": "at least one value",
        "two": "at least two values (one when the condition has one run)",
        "all": "a value in every run of the condition",
    }[need]
    return (
        f"log2(A / B), computed by the viewer: each condition's value is the {summary} of the "
        f"linear {source_name} quantities over the condition's runs with a value, and needs "
        f"{needs}. A missing quantity is never 0."
    )


def condition_ratios(
    rs: ResultSet,
    level: str = "protein",
    source: str = "lfq",
    conditions: Mapping[str, Any] | None = None,
    a: str | None = None,
    b: str | None = None,
    *,
    summary: str = "median",
    need: str = "one",
    accepted_at: float | None = 0.01,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
) -> RatioResult:
    """log2(A / B) for every key of one level (see :class:`RatioResult`).

    ``conditions`` is the stored mapping ``{run: condition}`` (None: the suggestion;
    :func:`.quantqc.resolve_conditions`). ``a`` and ``b`` default to the first two
    conditions. ``source`` is ``lfq`` (MaxLFQ, experiments only) or ``quant`` (each run's
    own table). ``accepted_at`` keeps the keys accepted at that threshold on their own
    q column (precursor_q, pg_q_value); None keeps every key with a quantity.
    """
    if summary not in SUMMARIES:
        raise ViewerError(f"unknown summary {summary!r}; use median or mean.")
    if need not in NEEDS:
        raise ViewerError(f"unknown need {need!r}; use one, two or all.")
    if not rs.is_experiment and source == "lfq":
        source = "quant"
    resolved = resolve_conditions(rs, conditions)
    groups = condition_groups(resolved, [r.name for r in rs.runs])
    names = list(groups)
    if len(names) < 2:
        raise ViewerError(
            "condition ratios need two conditions; "
            + (
                "a single run has one."
                if len(rs.runs) < 2
                else f"every run is in condition {names[0]!r}. Group the runs on the Quant QC page."
            )
        )
    a = a if a in groups else names[0]
    b = b if b in groups and b != a else next(n for n in names if n != a)
    m = quant_matrix(rs, level, source, accepted_at=accepted_at)
    col = {r: j for j, r in enumerate(m.runs)}
    va = m.values[:, [col[r] for r in groups[a]]]
    vb = m.values[:, [col[r] for r in groups[b]]]
    need_a, need_b = _need(need, len(groups[a])), _need(need, len(groups[b]))
    sa, na = _summarise(va, summary, need_a)
    sb, nb = _summarise(vb, summary, need_b)
    with np.errstate(all="ignore"):
        ratio = np.log2(sa / sb)
        mean = np.log10((sa + sb) / 2.0)
    ratio[~(np.isfinite(sa) & np.isfinite(sb))] = np.nan
    mean[~np.isfinite(ratio)] = np.nan
    sfx = tuple(s for s in (str(x).strip() for x in suffixes) if s)
    keys = cached(
        rs,
        ("across.keys", level, source, accepted_at, sfx, m.n_keys, rs.scored.identity()),
        lambda: _keys_with_species(rs, m, level, accepted_at, sfx),
    ).copy()
    table = keys.assign(
        n_a=na.astype("int64"),
        n_b=nb.astype("int64"),
        value_a=sa,
        value_b=sb,
        log2_ratio=ratio,
        log10_mean=mean,
    )
    species = _species_table(table, sfx)
    src_name = "MaxLFQ" if source == "lfq" else "per-run quant"
    note = m.note if source == "quant" else ""
    return RatioResult(
        level=level,
        source=source,
        a=a,
        b=b,
        runs_a=tuple(groups[a]),
        runs_b=tuple(groups[b]),
        summary=summary,
        need=need,
        need_a=need_a,
        need_b=need_b,
        accepted_at=accepted_at,
        suffixes=sfx,
        table=table,
        species=species,
        label=ratio_rule(summary, need, src_name),
        note=note,
    )


def _keys_with_species(
    rs: ResultSet, m: Any, level: str, accepted_at: float | None, sfx: tuple[str, ...]
) -> pd.DataFrame:
    """The matrix keys with their protein group and species (shared by every summary)."""
    keys = m.keys[list(KEY_COLUMNS[level])].reset_index(drop=True).copy()
    if level == "protein":
        keys["protein_group"] = keys["protein_group"].astype(str)
    else:
        pg = precursor_groups(rs, accepted_at)
        keys = keys.merge(pg, on=["peptidoform", "charge"], how="left")
        if len(keys) != m.n_keys:  # pragma: no cover - the merge keys are unique
            raise ViewerError("a precursor maps to more than one protein group.")
    keys["species"] = _species_column(keys["protein_group"], sfx)
    return keys


def _species_column(groups: pd.Series, suffixes: tuple[str, ...]) -> np.ndarray:
    """:func:`species_by_suffix` of every group, computed once per distinct group."""
    text = groups.astype(object).where(groups.notna(), "").astype(str).to_numpy()
    uniq, inverse = np.unique(text, return_inverse=True)
    species = np.array([species_by_suffix(g, suffixes) for g in uniq], dtype=object)
    return species[inverse] if len(text) else np.array([], dtype=object)


def cached(rs: ResultSet, key: Any, factory: Any) -> Any:
    """``rs.memo`` with one lock per key: concurrent callbacks build a value once."""
    lock = rs.memo(("across.lock", key), threading.Lock)
    with lock:
        return rs.memo(key, factory)


def _species_table(table: pd.DataFrame, suffixes: Sequence[str]) -> pd.DataFrame:
    rows = []
    has_a = np.isfinite(table["value_a"].to_numpy(dtype="float64"))
    has_b = np.isfinite(table["value_b"].to_numpy(dtype="float64"))
    sp = table["species"].to_numpy()
    r = table["log2_ratio"].to_numpy(dtype="float64")
    for name in (*suffixes, MIXED, OTHER):
        sel = sp == name
        both = sel & has_a & has_b
        x = r[both]
        rows.append(
            {
                "species": name,
                "keys": int(sel.sum()),
                "both": int(both.sum()),
                "only_a": int((sel & has_a & ~has_b).sum()),
                "only_b": int((sel & ~has_a & has_b).sum()),
                "neither": int((sel & ~has_a & ~has_b).sum()),
                "median": float(np.median(x)) if x.size else np.nan,
                "mean": float(np.mean(x)) if x.size else np.nan,
                "q25": float(np.percentile(x, 25)) if x.size else np.nan,
                "q75": float(np.percentile(x, 75)) if x.size else np.nan,
                "sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ratio_histogram(
    result: RatioResult, *, width: float = 0.1, clip: float = 6.0
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Histogram of log2(A/B) per species on shared bins of ``width`` (the viewer's).

    The bins span the finite ratios, limited to ``[-clip, clip]``; ratios outside fall in
    the first or last bin and are counted in the third result (``{species: n clipped}``).
    Returns ``(edges, {species: counts}, clipped)`` for the species with values.
    """
    t = result.table
    r = t["log2_ratio"].to_numpy(dtype="float64")
    ok = np.isfinite(r)
    if not ok.any():
        return np.array([-1.0, 1.0]), {}, {}
    lo = max(-clip, math.floor(float(np.min(r[ok])) / width) * width)
    hi = min(clip, math.ceil(float(np.max(r[ok])) / width) * width)
    if hi <= lo:
        lo, hi = lo - width, hi + width
    n = max(1, round((hi - lo) / width))
    edges = lo + width * np.arange(n + 1)
    counts: dict[str, np.ndarray] = {}
    clipped: dict[str, int] = {}
    sp = t["species"].to_numpy()
    for name in (*result.suffixes, MIXED, OTHER):
        x = r[ok & (sp == name)]
        if not x.size:
            continue
        clipped[name] = int(((x < lo) | (x > hi)).sum())
        counts[name] = np.histogram(np.clip(x, lo, hi), bins=edges)[0]
    return edges, counts, clipped
