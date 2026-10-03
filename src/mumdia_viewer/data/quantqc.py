"""Quant QC (P1 view 7): the viewer's summaries of the engine's quantities.

What the engine writes (MuMDIA 0.5.0; docs/12 and ``stages/quant.rs``):

* ``peptide_quant.parquet``, per run: one row per scored row that passed the quant gate.
  ``quantity`` is the sum of the top-N positive fragment areas; it is null when the row
  is not quantifiable, and ``quant_status`` says why.
* ``protein_group_quant.parquet``, per run: the top-N sum of the maximum quantity of each
  base peptide of the group (``rollup`` and ``top_n_peptides`` of the quant report).
* ``lfq_maxlfq.parquet`` with ``.precursor.parquet`` and ``.peptide.parquet``, in an
  experiment: MaxLFQ over the runs' ``peptide_quant`` features (the maximum quantity of
  each ``(peptidoform, charge)`` in each run), after one median-ratio size factor per run.
  ``run_experiment.rs`` passes ``NormalizeMethod::MedianRatio`` to ``run_lfq_combine``;
  the engine logs the factors and writes them nowhere. A quantity of 0.0 means that the
  run has no feature for the key.

Everything this module computes is the viewer's, and every label says so: the quantity
matrix of a level (keys by runs, NaN where a run has no value), distributions of log10
quantity, missing values, coefficients of variation (CV) within user-defined conditions,
the quant state of each run's accepted identifications, the size factors, the order of
the LFQ heatmap and the profile of one protein group. No quantity, q value or score is
recomputed, and nothing is written anywhere.
"""

from __future__ import annotations

import math
import re
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .discovery import ResultSet, Run
from .duck import sql_ident, sql_path
from .errors import ArtifactNotFound, ViewerError
from .mbr import mbr_ran
from .quant import (
    describe_status,
    lfq_matrix,
    protein_quant,
    quant_relation,
    quant_status_breakdown,
    quant_table_problems,
    transfer_relation,
)

__all__ = [
    "LEVELS",
    "SOURCES",
    "ConditionCv",
    "CvResult",
    "Distribution",
    "Heatmap",
    "ProteinProfile",
    "QuantMatrix",
    "accepted_keys",
    "accepted_quant_states",
    "condition_cvs",
    "condition_groups",
    "find_groups",
    "heatmap",
    "missing_by_run",
    "protein_profile",
    "quant_matrix",
    "quantity_distribution",
    "resolve_conditions",
    "run_files",
    "runs_with_value",
    "size_factors",
    "species_of",
    "status_matrix",
    "suggest_conditions",
]

LEVELS: tuple[str, ...] = ("precursor", "protein")
SOURCES: tuple[str, ...] = ("quant", "lfq")
KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "precursor": ("peptidoform", "charge"),
    "protein": ("protein_group",),
}
UNIT_LABELS: dict[str, str] = {
    "precursor": "precursors (peptidoform, charge)",
    "protein": "protein groups (protein_group)",
}
VALUE_LABELS: dict[tuple[str, str], str] = {
    ("quant", "precursor"): (
        "the run's precursor quantity: peptide_quant.quantity (the sum of the top-N "
        "fragment areas), the maximum over the run's rows of one (peptidoform, charge) as "
        "quant-lfq builds its features; not normalized"
    ),
    ("quant", "protein"): (
        "the run's protein group quantity: protein_group_quant.quantity (the top-N sum of "
        "the maximum quantity of each base peptide of the group); not normalized"
    ),
    ("lfq", "precursor"): (
        "MaxLFQ precursor quantity (lfq_maxlfq.parquet.precursor.parquet): one feature per "
        "precursor, so the run's quantity divided by the run's median-ratio size factor"
    ),
    ("lfq", "protein"): (
        "MaxLFQ protein group quantity (lfq_maxlfq.parquet), after one median-ratio size "
        "factor per run"
    ),
}
SOURCE_NAMES = {"quant": "per-run quant", "lfq": "MaxLFQ"}
# A CV is the viewer's; the rule in words (``need`` and the source are added per result).
CV_RULE = (
    "CV = sample standard deviation (n - 1) / mean of the condition's quantities, in "
    "percent, over the runs of the condition with a value; computed by the viewer on the "
    "linear quantities"
)


# --------------------------------------------------------------------------- runs


_MZML_KEY = re.compile(r"^mzml(?:\[(\d+)\])?$", re.IGNORECASE)
_COMPRESSED = re.compile(r"\.(?:gz|bz2|xz|zst|zip)$", re.IGNORECASE)
_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")
_SPLIT = re.compile(r"[_\-.\s]+")
_CONDITION_WORDS = frozenset({"condition", "cond"})
_CONDITION_JOINED = re.compile(r"^(?:condition|cond)([A-Za-z0-9]+)$", re.IGNORECASE)
_REPLICATE_WORDS = frozenset(
    {"rep", "replicate", "r", "run", "inj", "injection", "tech", "techrep", "bio", "biorep"}
)
_REPLICATE = re.compile(
    r"^(?:rep|replicate|r|run|inj|injection|tech|techrep|bio|biorep|br|tr)?\d+$", re.IGNORECASE
)


def _file_name(path: str) -> str:
    return re.split(r"[\\/]", str(path))[-1]


def _stem(name: str) -> str:
    return _EXTENSION.sub("", _COMPRESSED.sub("", name))


def run_files(rs: ResultSet) -> dict[str, str | None]:
    """The mzML file name of each run, from the manifest inputs (None when not recorded).

    A single run records its file as ``mzml`` (or ``mzml[0]``), an experiment as
    ``mzml[<i>]``, where ``i`` is the run's source index (the order of
    ``experiment.runs``). Only the file name is returned, not the directory.
    """
    by_index: dict[int, str] = {}
    for key, record in rs.manifest.inputs.items():
        m = _MZML_KEY.match(str(key))
        if m is None or not record.path:
            continue
        by_index.setdefault(int(m.group(1)) if m.group(1) is not None else 0, record.path)
    out: dict[str, str | None] = {}
    for run in rs.runs:
        path = by_index.get(int(run.index))
        if path is None and not rs.is_experiment and len(by_index) == 1:
            path = next(iter(by_index.values()))
        out[run.name] = _file_name(path) if path else None
    return out


def _tokens(name: str) -> list[str]:
    return [t for t in _SPLIT.split(_stem(name)) if t]


def _condition_token(tokens: Sequence[str]) -> str | None:
    for i, token in enumerate(tokens):
        if token.lower() in _CONDITION_WORDS and i + 1 < len(tokens):
            return tokens[i + 1]
        m = _CONDITION_JOINED.match(token)
        if m is not None:
            return m.group(1)
    return None


def _drop_replicate(tokens: list[str]) -> list[str]:
    if len(tokens) >= 3 and tokens[-2].lower() in _REPLICATE_WORDS and tokens[-1].isdigit():
        return tokens[:-2]
    if len(tokens) >= 2 and _REPLICATE.match(tokens[-1]):
        return tokens[:-1]
    return tokens


def _common_prefix(lists: Sequence[Sequence[str]]) -> int:
    n = min(len(x) for x in lists)
    i = 0
    while i < n and all(x[i] == lists[0][i] for x in lists):
        i += 1
    return i


def _common_suffix(lists: Sequence[Sequence[str]], prefix: int) -> int:
    n = min(len(x) for x in lists) - prefix
    i = 0
    while i < n and all(x[len(x) - 1 - i] == lists[0][len(lists[0]) - 1 - i] for x in lists):
        i += 1
    return i


def suggest_conditions(rs: ResultSet) -> dict[str, str]:
    """A condition for every run, suggested from the mzML file names of the manifest.

    The rule, applied to the file name of each run (:func:`run_files`):

    1. The name is cut at its extension (and a ``.gz``-like compression suffix) and split
       into tokens at ``_``, ``-``, ``.`` and spaces.
    2. A token ``Condition`` or ``Cond`` (any case) names the condition with the token
       after it, and ``ConditionX`` / ``CondX`` names it ``X``:
       ``LFQ_Astral_DIA_15min_50ng_Condition_A_REP1.mzML`` gives ``A``.
    3. Otherwise a final replicate token is removed (``REP1``, ``rep_2``, ``R3``,
       ``run4``, ``inj1``, a bare number such as ``01``), then the tokens that every
       such run shares at the start and at the end. What is left, joined with ``_``,
       is the condition: ``HeLa_ctrl_01`` and ``HeLa_treat_02`` give ``ctrl`` and
       ``treat``.
    4. A run whose name leaves nothing in step 3 (every run shares it, for example a
       single run) takes its name after the replicate token is removed.
    5. A run without a recorded mzML file takes its run name.

    The suggestion is a starting point for the user, who can change it; the viewer keeps
    the user's choice in the browser (``resolve_conditions``).
    """
    files = run_files(rs)
    out: dict[str, str] = {}
    cores: dict[str, list[str]] = {}
    for run in rs.runs:
        name = files.get(run.name)
        tokens = _tokens(name) if name else []
        if not tokens:
            out[run.name] = run.label
            continue
        named = _condition_token(tokens)
        if named is not None:
            out[run.name] = named
        else:
            cores[run.name] = _drop_replicate(tokens)
    if cores:
        lists = list(cores.values())
        prefix = _common_prefix(lists)
        suffix = _common_suffix(lists, prefix)
        for run_name, tokens in cores.items():
            rest = tokens[prefix : len(tokens) - suffix]
            out[run_name] = "_".join(rest) if rest else "_".join(tokens)
    return {run.name: out[run.name] for run in rs.runs}


def resolve_conditions(rs: ResultSet, stored: Mapping[str, Any] | None) -> dict[str, str]:
    """The condition of every run: the user's stored choice, else the suggestion.

    ``stored`` is the mapping the browser keeps (``{run name: condition}``). A run that
    is missing from it, or whose value is empty or not text, takes the suggested
    condition (:func:`suggest_conditions`); surrounding spaces are removed. Names that
    are not runs of ``rs`` are ignored, so None or ``{}`` means the suggestion.
    """
    out = suggest_conditions(rs)
    if not isinstance(stored, Mapping):
        return out
    for run in rs.runs:
        value = stored.get(run.name)
        if isinstance(value, str) and value.strip():
            out[run.name] = value.strip()
    return out


def condition_groups(conditions: Mapping[str, str], runs: Sequence[str]) -> dict[str, list[str]]:
    """The runs of each condition, conditions in the order of their first run."""
    groups: dict[str, list[str]] = {}
    for run in runs:
        if run in conditions:
            groups.setdefault(conditions[run], []).append(run)
    return groups


# --------------------------------------------------------------------------- matrices


def _memo(rs: ResultSet, key: Any, factory: Any) -> Any:
    """``rs.memo`` with one lock per key, so concurrent callbacks build a value once."""
    lock = rs.memo(("quantqc.lock", key), threading.Lock)
    with lock:
        return rs.memo(key, factory)


@dataclass(frozen=True)
class QuantMatrix:
    """Quantities of one level and source: keys by runs (the viewer's arrangement).

    ``values[i, j]`` is the quantity of key ``i`` in run ``runs[j]``; NaN means that the
    run has no value (no row, a null quantity, or 0.0 in an LFQ table), never zero.
    ``transferred`` counts the match-between-runs transfers among a cell's features
    (None when MBR did not run). ``accepted_at`` is set when the keys are the keys
    accepted at that threshold (:func:`accepted_keys`); the matrix then has a row for
    every accepted key, with NaN in every run for an accepted key without a quantity.
    """

    level: str
    source: str
    runs: tuple[str, ...]
    keys: pd.DataFrame = field(repr=False)
    values: np.ndarray = field(repr=False)
    transferred: np.ndarray | None = field(default=None, repr=False)
    label: str = ""
    accepted_at: float | None = None
    note: str = ""

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def unit(self) -> str:
        return UNIT_LABELS[self.level]

    @property
    def source_name(self) -> str:
        return SOURCE_NAMES[self.source]

    def key_index(self) -> pd.Index:
        """The keys as an index (a MultiIndex for precursors)."""
        return _index(self.keys, self.level)

    def frame(self) -> pd.DataFrame:
        """Keys and one column per run (NaN where missing)."""
        out = self.keys.reset_index(drop=True).copy()
        for j, run in enumerate(self.runs):
            out[run] = self.values[:, j]
        return out


def _index(keys: pd.DataFrame, level: str) -> pd.Index:
    if level == "precursor":
        return pd.MultiIndex.from_arrays(
            [keys["peptidoform"].astype(str).to_numpy(), keys["charge"].astype("int64").to_numpy()]
        )
    return pd.Index(keys["protein_group"].astype(str).to_numpy())


def _check(level: str, source: str) -> None:
    if level not in LEVELS:
        raise ViewerError(f"unknown level {level!r}; use precursor or protein.")
    if source not in SOURCES:
        raise ViewerError(f"unknown quantity source {source!r}; use quant or lfq.")


def _identity(rs: ResultSet, level: str, source: str) -> tuple[Any, ...]:
    if source == "lfq":
        kind = "lfq_maxlfq" if level == "protein" else "lfq_maxlfq_precursor"
        art = rs.artifact(kind)
        return (kind, art.identity() if art is not None and art.usable else None)
    kind = "peptide_quant" if level == "precursor" else "protein_group_quant"
    out: list[Any] = [kind]
    for run in rs.runs:
        art = run.artifact(kind)
        out.append(art.identity() if art is not None and art.usable else None)
    if mbr_ran(rs):
        out.append(rs.scored.identity())
    return tuple(out)


def _scatter(
    rs: ResultSet,
    cells: str,
    params: list[Any],
    keys: tuple[str, ...],
    order: Sequence[int],
    with_transfers: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Keys by runs from a relation of (keys..., source, q[, n_tr]), keys sorted."""
    key_sql = ", ".join(sql_ident(k) for k in keys)
    extra = ", c.n_tr" if with_transfers else ""
    sql = (
        f"WITH c AS ({cells}) SELECT {key_sql}, c.source, c.q{extra}, "
        f"(dense_rank() OVER (ORDER BY {key_sql}) - 1)::BIGINT AS k FROM c ORDER BY k"
    )
    df = rs.duck.df(sql, params)
    n = int(df["k"].iloc[-1]) + 1 if len(df) else 0
    col = {int(s): j for j, s in enumerate(order)}
    values = np.full((n, len(order)), np.nan)
    transferred = np.zeros((n, len(order)), dtype=np.int64) if with_transfers else None
    if len(df):
        k = df["k"].to_numpy(dtype=np.int64)
        j = df["source"].map(col)
        ok = j.notna().to_numpy()
        jj = j[ok].astype("int64").to_numpy()
        values[k[ok], jj] = df["q"].to_numpy(dtype="float64", na_value=np.nan)[ok]
        if transferred is not None:
            transferred[k[ok], jj] = df["n_tr"].fillna(0).to_numpy(dtype=np.int64)[ok]
    first = ~df.duplicated("k").to_numpy() if len(df) else np.zeros(0, dtype=bool)
    key_df = df.loc[first, list(keys)].reset_index(drop=True)
    if "charge" in key_df:
        key_df["charge"] = key_df["charge"].astype("int64")
    return key_df, values, transferred


def _quant_build(rs: ResultSet, level: str) -> QuantMatrix:
    kind = "peptide_quant" if level == "precursor" else "protein_group_quant"
    keys = KEY_COLUMNS[level]
    columns = ("candidate_id", "peptidoform", "charge", "quantity")
    union = quant_relation(
        rs, kind, columns if level == "precursor" else ("protein_group", "quantity")
    )
    if union is None:
        problems = "; ".join(f"{run.label}: {p}" for run, p in quant_table_problems(rs, kind))
        raise ArtifactNotFound(f"no run has a readable {kind} table ({problems}).")
    sql, params, _ = union
    key_sql = ", ".join(f"u.{sql_ident(k)}" for k in keys)
    good = "CASE WHEN isfinite(u.quantity) AND u.quantity > 0 THEN u.quantity END"
    transfers = transfer_relation(rs) if mbr_ran(rs) else None
    if transfers is None:
        cells = f"SELECT {key_sql}, u.source, max({good}) AS q FROM ({sql}) u GROUP BY ALL"
        cell_params = list(params)
    elif level == "precursor":
        t_sql, t_params = transfers
        cells = (
            f"SELECT {key_sql}, u.source, max({good}) AS q, "
            f"count(t.candidate_id) FILTER (WHERE {good} IS NOT NULL) AS n_tr "
            f"FROM ({sql}) u LEFT JOIN ({t_sql}) t "
            "ON t.source = u.source AND t.candidate_id = u.candidate_id GROUP BY ALL"
        )
        cell_params = [*params, *t_params]
    else:
        t_sql, t_params = transfers
        pep = quant_relation(rs, "peptide_quant", ("candidate_id", "protein_group", "quantity"))
        if pep is None:
            cells = f"SELECT {key_sql}, u.source, max({good}) AS q FROM ({sql}) u GROUP BY ALL"
            cell_params = list(params)
            transfers = None
        else:
            p_sql, p_params, _ = pep
            cells = (
                f"WITH g AS (SELECT {key_sql}, u.source, max({good}) AS q FROM ({sql}) u "
                "GROUP BY ALL), x AS (SELECT p.protein_group, p.source, count(*) AS n_tr "
                f"FROM ({p_sql}) p JOIN ({t_sql}) t ON t.source = p.source AND "
                "t.candidate_id = p.candidate_id WHERE isfinite(p.quantity) AND p.quantity > 0 "
                "GROUP BY ALL) SELECT g.protein_group, g.source, g.q, coalesce(x.n_tr, 0) "
                "AS n_tr FROM g LEFT JOIN x ON x.protein_group = g.protein_group "
                "AND x.source = g.source"
            )
            cell_params = [*params, *p_params, *t_params]
    order = [int(r.index) for r in rs.runs]
    key_df, values, transferred = _scatter(
        rs, cells, cell_params, keys, order, with_transfers=transfers is not None
    )
    missing = [run.label for run, _ in quant_table_problems(rs, kind)]
    note = (
        f"Runs without a readable {kind} table have no values: {', '.join(missing)}."
        if missing
        else ""
    )
    return QuantMatrix(
        level=level,
        source="quant",
        runs=tuple(r.name for r in rs.runs),
        keys=key_df,
        values=values,
        transferred=transferred,
        label=VALUE_LABELS[("quant", level)],
        note=note,
    )


def _lfq_build(rs: ResultSet, level: str) -> QuantMatrix:
    wide = lfq_matrix(rs, level, wide=True)  # type: ignore[arg-type]
    run_cols = list(wide.attrs.get("run_columns", []))
    names = [r.name for r in rs.runs]
    values = (
        wide[run_cols].to_numpy(dtype="float64", na_value=np.nan)
        if run_cols
        else np.full((len(wide), len(names)), np.nan)
    )
    with np.errstate(invalid="ignore"):
        values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    tr_cols = [f"n_transferred_{n}" for n in names]
    transferred = (
        wide[tr_cols].fillna(0).to_numpy(dtype=np.int64)
        if all(c in wide.columns for c in tr_cols)
        else None
    )
    if level == "precursor":
        keys = pd.DataFrame(
            {
                "peptidoform": wide["group"].astype(str).to_numpy(),
                "charge": wide["charge"].astype("int64").to_numpy(),
            }
        )
    else:
        keys = pd.DataFrame({"protein_group": wide["protein_group"].astype(str).to_numpy()})
    keys["n_features"] = wide["n_features"].astype("int64").to_numpy()
    # lfq_matrix orders its rows by the keys.
    return QuantMatrix(
        level=level,
        source="lfq",
        runs=tuple(names),
        keys=keys,
        values=values,
        transferred=transferred,
        label=VALUE_LABELS[("lfq", level)],
        note=str(wide.attrs.get("description", "")),
    )


def accepted_keys(rs: ResultSet, level: str, t: float) -> pd.DataFrame:
    """The keys accepted at ``t`` on their own unit's q column, target rows only.

    Precursors: ``(peptidoform, charge)`` of the rows with ``precursor_q <= t``; protein
    groups: ``protein_group`` of the rows with ``pg_q_value <= t``. Both columns are set
    on each group's winning row (experiment-wide in an experiment), so this is the
    identification unit of the key. The keys are sorted.
    """
    _check(level, "quant")
    column = "precursor_q" if level == "precursor" else "pg_q_value"
    handle = rs.scored.parquet()
    if not handle.has_column(column):
        raise ViewerError(f"the scored table has no {column} column, so no key is accepted.")
    keys = KEY_COLUMNS[level]
    key_sql = ", ".join(sql_ident(k) for k in keys)

    def build() -> pd.DataFrame:
        df = rs.duck.df(
            f"SELECT DISTINCT {key_sql} FROM read_parquet(?) WHERE label = 'target' "
            f"AND {sql_ident(column)} <= ? ORDER BY {key_sql}",
            [sql_path(rs.scored.require()), float(t)],
        )
        if "charge" in df:
            df["charge"] = df["charge"].astype("int64")
        else:
            df["protein_group"] = df["protein_group"].astype(str)
        return df

    return _memo(rs, ("quantqc.accepted", rs.scored.identity(), level, float(t)), build)


def _restrict(m: QuantMatrix, keys: pd.DataFrame, t: float) -> QuantMatrix:
    where = m.key_index().get_indexer(_index(keys, m.level))
    values = np.full((len(keys), len(m.runs)), np.nan)
    found = where >= 0
    values[found] = m.values[where[found]]
    transferred = None
    if m.transferred is not None:
        transferred = np.zeros((len(keys), len(m.runs)), dtype=np.int64)
        transferred[found] = m.transferred[where[found]]
    out_keys = keys[list(KEY_COLUMNS[m.level])].reset_index(drop=True).copy()
    if "n_features" in m.keys:
        nf = np.zeros(len(keys), dtype=np.int64)
        nf[found] = m.keys["n_features"].to_numpy(dtype=np.int64)[where[found]]
        out_keys["n_features"] = nf
    return QuantMatrix(
        level=m.level,
        source=m.source,
        runs=m.runs,
        keys=out_keys,
        values=values,
        transferred=transferred,
        label=m.label,
        accepted_at=float(t),
        note=m.note,
    )


def quant_matrix(
    rs: ResultSet,
    level: str = "precursor",
    source: str = "quant",
    *,
    accepted_at: float | None = None,
) -> QuantMatrix:
    """The quantities of one level as keys by runs (memoised on the result set).

    ``level`` is ``precursor`` (key ``(peptidoform, charge)``) or ``protein`` (key
    ``protein_group``). ``source`` is ``quant``, the runs' own tables
    (``peptide_quant.quantity``, the maximum over a run's rows of one key as quant-lfq
    builds its features; ``protein_group_quant.quantity``), or ``lfq``, the
    experiment's MaxLFQ tables (``lfq_maxlfq.parquet`` and ``.precursor.parquet``).
    A null, non-finite or non-positive quantity, and an LFQ 0.0, is NaN (missing).

    Without ``accepted_at`` the keys are every key with a row in any run's table (for
    LFQ: the LFQ key list, which is not FDR-controlled at the key's unit). With
    ``accepted_at=t`` the keys are the keys accepted at ``t`` (:func:`accepted_keys`),
    each with a row, NaN where a run has no quantity.

    A single run has no LFQ: ``source='lfq'`` raises :class:`ViewerError`.
    """
    _check(level, source)
    ident = _identity(rs, level, source)

    def build() -> QuantMatrix:
        return _lfq_build(rs, level) if source == "lfq" else _quant_build(rs, level)

    base: QuantMatrix = _memo(rs, ("quantqc.matrix", level, source, ident), build)
    if accepted_at is None:
        return base
    t = float(accepted_at)
    return _memo(
        rs,
        ("quantqc.matrix", level, source, ident, t),
        lambda: _restrict(base, accepted_keys(rs, level, t), t),
    )


# --------------------------------------------------------------------------- summaries


@dataclass(frozen=True)
class Distribution:
    """Per-run distribution of log10 quantity (the viewer's percentiles and histograms).

    ``table`` has one row per run: ``run``, ``n`` (keys with a value), ``missing`` and
    the percentiles ``p5``, ``p25``, ``p50``, ``p75``, ``p95`` of log10 quantity
    (``numpy.percentile``, linear interpolation; NaN without values). ``counts[j]`` is
    the histogram of run ``j`` on the shared ``edges`` (log10 quantity).
    """

    runs: tuple[str, ...]
    table: pd.DataFrame
    edges: np.ndarray
    counts: np.ndarray
    label: str


PERCENTILES = (5, 25, 50, 75, 95)


def quantity_distribution(m: QuantMatrix, bins: int = 64) -> Distribution:
    """The distribution of log10 quantity of every run of ``m``."""
    logs = [np.log10(m.values[np.isfinite(m.values[:, j]), j]) for j in range(len(m.runs))]
    present = [x for x in logs if x.size]
    if present:
        lo = float(min(x.min() for x in present))
        hi = float(max(x.max() for x in present))
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.zeros((len(m.runs), bins), dtype=np.int64)
    rows = []
    for j, (run, x) in enumerate(zip(m.runs, logs, strict=True)):
        if x.size:
            counts[j] = np.histogram(x, bins=edges)[0]
            pct = np.percentile(x, PERCENTILES)
        else:
            pct = np.full(len(PERCENTILES), np.nan)
        rows.append(
            {
                "run": run,
                "n": int(x.size),
                "missing": int(m.n_keys - x.size),
                **{f"p{p}": float(v) for p, v in zip(PERCENTILES, pct, strict=True)},
            }
        )
    table = pd.DataFrame(rows, columns=["run", "n", "missing", *[f"p{p}" for p in PERCENTILES]])
    label = (
        f"log10 of {m.label}; percentiles and histograms computed by the viewer over the "
        f"{m.unit} with a value in each run"
    )
    return Distribution(runs=m.runs, table=table, edges=edges, counts=counts, label=label)


def missing_by_run(m: QuantMatrix) -> pd.DataFrame:
    """Keys with and without a value in each run (counted by the viewer).

    Columns ``run``, ``keys`` (rows of the matrix), ``present``, ``missing`` and
    ``missing_pct`` (percent of ``keys``; NaN for an empty matrix).
    """
    present = np.isfinite(m.values).sum(axis=0) if m.n_keys else np.zeros(len(m.runs), int)
    keys = m.n_keys
    out = pd.DataFrame(
        {
            "run": list(m.runs),
            "keys": keys,
            "present": present.astype(np.int64),
            "missing": (keys - present).astype(np.int64),
        }
    )
    out["missing_pct"] = 100.0 * out["missing"] / keys if keys else np.nan
    return out


def runs_with_value(m: QuantMatrix) -> pd.DataFrame:
    """The number of keys with a value in exactly ``n_runs`` runs, for 0 to all runs."""
    n = np.isfinite(m.values).sum(axis=1) if m.n_keys else np.zeros(0, dtype=int)
    counts = np.bincount(n, minlength=len(m.runs) + 1)
    out = pd.DataFrame({"n_runs": np.arange(len(counts)), "keys": counts.astype(np.int64)})
    out["pct"] = 100.0 * out["keys"] / m.n_keys if m.n_keys else np.nan
    return out


@dataclass(frozen=True)
class ConditionCv:
    """The CVs of one condition (the viewer's; see :data:`CV_RULE`).

    ``cv`` holds the CV in percent of every key with at least ``need`` values in the
    condition, and ``rows`` the row of each such key in the matrix. ``reason`` is set
    when the condition gives no CV at all (one run).
    """

    condition: str
    runs: tuple[str, ...]
    need: int
    n_keys: int
    n_with_value: int
    cv: np.ndarray = field(repr=False)
    rows: np.ndarray = field(repr=False)
    reason: str = ""

    @property
    def n_cv(self) -> int:
        return int(self.cv.size)

    @property
    def median(self) -> float | None:
        return float(np.median(self.cv)) if self.cv.size else None

    def share_below(self, pct: float) -> float | None:
        """Percent of the CVs at or below ``pct`` percent."""
        return float(100.0 * np.mean(self.cv <= pct)) if self.cv.size else None


@dataclass(frozen=True)
class CvResult:
    """The CVs of every condition at one level and source."""

    level: str
    source: str
    conditions: tuple[ConditionCv, ...]
    all_runs: bool
    rule: str
    label: str

    def of(self, condition: str) -> ConditionCv | None:
        return next((c for c in self.conditions if c.condition == condition), None)


def _cvs(x: np.ndarray, need: int) -> tuple[np.ndarray, np.ndarray]:
    """(CV in percent, row index) of the rows of ``x`` with at least ``need`` values."""
    ok = np.isfinite(x)
    n = ok.sum(axis=1)
    rows = np.flatnonzero(n >= max(need, 2))
    if rows.size == 0:
        return np.zeros(0), rows
    sub = np.where(ok[rows], x[rows], 0.0)
    k = n[rows].astype("float64")
    mean = sub.sum(axis=1) / k
    dev = np.where(ok[rows], x[rows] - mean[:, None], 0.0)
    sd = np.sqrt((dev**2).sum(axis=1) / (k - 1.0))
    return 100.0 * sd / mean, rows


def condition_cvs(
    m: QuantMatrix, conditions: Mapping[str, str], *, all_runs: bool = True
) -> CvResult:
    """The CV of every key within every condition (the viewer's computation).

    The CV of a key in a condition is the sample standard deviation (n - 1) of its
    quantities over the condition's runs with a value, divided by their mean, in
    percent, on the linear quantities of ``m``. With ``all_runs`` (the default) a CV
    needs a value in every run of the condition; otherwise it needs at least two
    values. A condition with one run gives no CV. Runs of ``m`` without a condition are
    left out, and so are conditions without a run of ``m``.
    """
    groups = condition_groups(conditions, m.runs)
    col = {run: j for j, run in enumerate(m.runs)}
    out: list[ConditionCv] = []
    for name, runs in groups.items():
        x = m.values[:, [col[r] for r in runs]]
        n_with = int(np.isfinite(x).any(axis=1).sum()) if m.n_keys else 0
        need = len(runs) if all_runs else 2
        if len(runs) < 2:
            out.append(
                ConditionCv(
                    condition=name,
                    runs=tuple(runs),
                    need=2,
                    n_keys=m.n_keys,
                    n_with_value=n_with,
                    cv=np.zeros(0),
                    rows=np.zeros(0, dtype=np.int64),
                    reason="one run: a CV needs at least two values",
                )
            )
            continue
        cv, rows = _cvs(x, need)
        out.append(
            ConditionCv(
                condition=name,
                runs=tuple(runs),
                need=need,
                n_keys=m.n_keys,
                n_with_value=n_with,
                cv=cv,
                rows=rows,
            )
        )
    need_text = (
        "each CV needs a value in every run of its condition"
        if all_runs
        else "each CV needs at least two values"
    )
    return CvResult(
        level=m.level,
        source=m.source,
        conditions=tuple(out),
        all_runs=all_runs,
        rule=f"{CV_RULE}; {need_text}",
        label=m.label,
    )


# --------------------------------------------------------------------------- per run


def _run_psm_column(rs: ResultSet) -> str:
    handle = rs.scored.parquet()
    if handle.has_column("run_psm_q"):
        return "run_psm_q"
    if handle.has_column("q_value") and not rs.is_experiment:
        return "q_value"
    raise ViewerError("the scored table has no run_psm_q column, so no run has accepted rows.")


def accepted_quant_states(rs: ResultSet, t: float) -> pd.DataFrame:
    """The quant state of the accepted identifications of each run (counted by the viewer).

    The accepted rows of a run are its distinct ``candidate_id`` of target rows
    (``label = 'target'``) with ``run_psm_q <= t`` in the scored table that the
    identification counts read (``scored_combined.parquet`` with ``source`` = the run in
    an experiment: native, before match-between-runs). Columns:

    * ``accepted``; of these ``quantified`` (a ``peptide_quant`` row with a positive
      finite quantity), ``not_quantifiable`` (a row with a null or unusable quantity)
      and ``not_selected`` (no row: the quant gate did not select it);
    * ``quantified_not_accepted``: quantified rows whose candidate is not accepted at
      ``run_psm_q <= t`` (the quant gate admitted them on its own q column, or as
      transfers);
    * ``quant_rows``: rows of the run's ``peptide_quant``;
    * ``transfers`` (only when MBR ran): quantified rows that are accepted
      match-between-runs transfers.

    A run without a readable ``peptide_quant`` has NA counts; ``attrs['notes']`` says
    why. ``attrs['labels']`` names every column. Memoised per threshold (a copy is
    returned).
    """
    key = ("quantqc.states", rs.scored.identity(), _identity(rs, "precursor", "quant"), float(t))
    return _memo(rs, key, lambda: _accepted_quant_states(rs, t)).copy()


def _accepted_quant_states(rs: ResultSet, t: float) -> pd.DataFrame:
    column = _run_psm_column(rs)
    scored = sql_path(rs.scored.require())
    transfers = transfer_relation(rs) if mbr_ran(rs) else None
    problems = {run.name: p for run, p in quant_table_problems(rs, "peptide_quant")}
    has_source = rs.scored.parquet().has_column("source")
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    good = "q.quantity IS NOT NULL AND isfinite(q.quantity) AND q.quantity > 0"
    for run in rs.runs:
        art = run.artifact("peptide_quant")
        if run.name in problems or art is None or not art.usable:
            notes.append(f"{run.label}: {problems.get(run.name, 'no peptide_quant table')}")
            rows.append({"run": run.name})
            continue
        art.parquet()
        where = " AND source = ?" if (rs.is_experiment and has_source) else ""
        params: list[Any] = [scored, float(t)]
        if where:
            params.append(int(run.index))
        params.append(sql_path(art.require()))
        tr_cte, tr_col = "", "NULL::BIGINT AS transfers"
        if transfers is not None:
            t_sql, t_params = transfers
            tr_cte = f", t AS (SELECT candidate_id FROM ({t_sql}) WHERE source = ?)"
            params += [*t_params, int(run.index)]
            tr_col = (
                "count(*) FILTER (WHERE q.candidate_id IS NOT NULL AND q.ok AND "
                "q.candidate_id IN (SELECT candidate_id FROM t)) AS transfers"
            )
        sql = (
            "WITH a AS (SELECT DISTINCT candidate_id FROM read_parquet(?) "
            f"WHERE label = 'target' AND {sql_ident(column)} <= ?{where}), "
            f"q AS (SELECT candidate_id, bool_or({good}) AS ok FROM read_parquet(?) q "
            f"GROUP BY 1){tr_cte} "
            "SELECT count(a.candidate_id) AS accepted, "
            "count(*) FILTER (WHERE a.candidate_id IS NOT NULL AND q.ok) AS quantified, "
            "count(*) FILTER (WHERE a.candidate_id IS NOT NULL AND q.candidate_id IS NOT NULL "
            "AND NOT q.ok) AS not_quantifiable, "
            "count(*) FILTER (WHERE a.candidate_id IS NOT NULL AND q.candidate_id IS NULL) "
            "AS not_selected, "
            "count(*) FILTER (WHERE a.candidate_id IS NULL AND q.ok) AS quantified_not_accepted, "
            f"count(q.candidate_id) AS quant_rows, {tr_col} "
            "FROM a FULL OUTER JOIN q ON q.candidate_id = a.candidate_id"
        )
        values = rs.duck.rows(sql, params)[0]
        names = [
            "accepted",
            "quantified",
            "not_quantifiable",
            "not_selected",
            "quantified_not_accepted",
            "quant_rows",
            "transfers",
        ]
        rows.append({"run": run.name, **dict(zip(names, values, strict=True))})
    columns = [
        "run",
        "accepted",
        "quantified",
        "not_quantifiable",
        "not_selected",
        "quantified_not_accepted",
        "quant_rows",
    ]
    if transfers is not None:
        columns.append("transfers")
    out = pd.DataFrame.from_records(rows, columns=columns)
    for c in columns[1:]:
        out[c] = out[c].astype("Int64")
    where_text = (
        "scored_combined.parquet, the run's rows (native, before match-between-runs)"
        if rs.is_experiment
        else rs.scored.path.name
        if rs.scored.path is not None
        else "the scored table"
    )
    labels = {
        "accepted": f"accepted identifications of the run: distinct candidate_id of target rows "
        f"with {column} <= {t:g} ({where_text})",
        "quantified": "accepted rows with a positive quantity in the run's peptide_quant",
        "not_quantifiable": "accepted rows that quant selected, with a null quantity "
        "(quant_status says why)",
        "not_selected": "accepted rows without a peptide_quant row: the quant gate did not "
        "select them",
        "quantified_not_accepted": f"quantified rows whose candidate is not accepted at "
        f"{column} <= {t:g}: the quant gate admitted them on its own q column"
        + (" or as transfers" if transfers is not None else ""),
        "quant_rows": "rows of the run's peptide_quant",
    }
    if transfers is not None:
        labels["transfers"] = (
            "quantified rows that are accepted match-between-runs transfers (derived from the "
            "transfer table)"
        )
    out.attrs.update(labels=labels, notes=notes, threshold=float(t), q_column=column)
    return out


def status_matrix(rs: ResultSet) -> pd.DataFrame:
    """Rows per ``quant_status`` and run, one column per run (from the engine's tables).

    One row per (``table``, ``status``) of :func:`.quant.quant_status_breakdown`, with
    ``description`` and one column per run name (0 when that run's table has no row of
    the status). With match-between-runs, ``n_transferred_<run>`` columns follow.
    ``attrs`` carries the breakdown's ``notes`` and ``labels``. Memoised (a copy is
    returned).
    """
    key = ("quantqc.status", _identity(rs, "precursor", "quant"), _identity(rs, "protein", "quant"))
    return _memo(rs, key, lambda: _status_matrix(rs)).copy()


def _status_matrix(rs: ResultSet) -> pd.DataFrame:
    b = quant_status_breakdown(rs)
    names = [r.name for r in rs.runs]
    keys = b[["table", "status"]].drop_duplicates()
    order = {"peptide_quant": 0, "protein_group_quant": 1}
    keys = keys.assign(
        _t=keys["table"].map(order).fillna(9), _q=keys["status"].ne("quantified")
    ).sort_values(["_t", "_q", "status"])
    out = keys[["table", "status"]].reset_index(drop=True)
    out["description"] = [
        describe_status(s, t) for t, s in zip(out["table"], out["status"], strict=True)
    ]
    for name in names:
        sub = b[b["run"] == name].set_index(["table", "status"])["n"]
        out[name] = [
            int(sub.get((t, s), 0)) for t, s in zip(out["table"], out["status"], strict=True)
        ]
    if "n_transferred" in b.columns:
        for name in names:
            sub = b[b["run"] == name].set_index(["table", "status"])["n_transferred"]
            out[f"n_transferred_{name}"] = pd.array(
                [sub.get((t, s), pd.NA) for t, s in zip(out["table"], out["status"], strict=True)],
                dtype="Int64",
            )
    out.attrs.update(notes=list(b.attrs.get("notes", [])), labels=dict(b.attrs.get("labels", {})))
    return out


def size_factors(rs: ResultSet) -> pd.DataFrame | None:
    """The per-run median-ratio size factors of the LFQ, derived by the viewer.

    ``run_lfq_combine`` divides every feature of a run by the run's size factor and logs
    the factors only. With one feature per precursor (MaxLFQ), the precursor-level LFQ
    value is the run's quantity divided by that factor, so the factor is the ratio
    ``quantity / LFQ`` of any precursor of the run. Columns: ``run``, ``factor`` (the
    median ratio), ``n`` (precursors with both values) and ``spread`` (the largest
    relative deviation of a ratio from the factor). ``attrs['constant']`` is True when
    every spread is below 1e-6, that is, when the derivation holds; when it is False the
    factors are not shown as the engine's. None in a single run or without both tables.
    """
    if not rs.is_experiment:
        return None
    try:
        quant = quant_matrix(rs, "precursor", "quant")
        lfq = quant_matrix(rs, "precursor", "lfq")
    except ViewerError:
        return None
    where = lfq.key_index().get_indexer(quant.key_index())
    found = where >= 0
    rows = []
    for j, run in enumerate(quant.runs):
        q = quant.values[found, j]
        v = lfq.values[where[found], j]
        ok = np.isfinite(q) & np.isfinite(v) & (v > 0)
        ratio = q[ok] / v[ok]
        if ratio.size:
            factor = float(np.median(ratio))
            spread = float(np.max(np.abs(ratio / factor - 1.0)))
        else:
            factor, spread = math.nan, math.nan
        rows.append({"run": run, "factor": factor, "n": int(ratio.size), "spread": spread})
    out = pd.DataFrame(rows, columns=["run", "factor", "n", "spread"])
    spreads = out["spread"].dropna()
    out.attrs["constant"] = bool(len(spreads)) and bool((spreads < 1e-6).all())
    out.attrs["label"] = (
        "Median-ratio size factor of each run, derived by the viewer: a precursor's "
        "quantity (peptide_quant) divided by its precursor-level MaxLFQ value; the "
        "engine logs the factors and writes them nowhere."
    )
    return out


# --------------------------------------------------------------------------- heatmap


@dataclass(frozen=True)
class Heatmap:
    """The LFQ matrix arranged for a heatmap (the viewer's order).

    ``groups`` and ``log10`` are in display order (top row first), ``runs`` and
    ``conditions`` in column order. ``relative`` is log2 of the quantity minus the
    row's mean log2 over its runs with a value. ``rows`` gives each displayed row's
    position in the source matrix.
    """

    groups: np.ndarray = field(repr=False)
    runs: tuple[str, ...]
    conditions: tuple[str, ...]
    log10: np.ndarray = field(repr=False)
    relative: np.ndarray = field(repr=False)
    rows: np.ndarray = field(repr=False)
    row_rule: str
    column_rule: str
    dropped: int = 0


def _average_linkage(d: np.ndarray) -> list[int]:
    """Leaf order of average-linkage (UPGMA) agglomerative clustering on distances ``d``.

    Ties are broken by the lowest pair of cluster ids, and the cluster with the lower
    id goes left, so the order is deterministic.
    """
    n = len(d)
    if n <= 2:
        return list(range(n))
    dist = np.array(d, dtype="float64", copy=True)
    np.fill_diagonal(dist, np.inf)
    order = {i: [i] for i in range(n)}
    size = {i: 1 for i in range(n)}
    active = list(range(n))
    while len(active) > 1:
        sub = dist[np.ix_(active, active)]
        flat = int(np.argmin(sub))
        a, b = active[flat // len(active)], active[flat % len(active)]
        a, b = min(a, b), max(a, b)
        na, nb = size[a], size[b]
        merged = (na * dist[a] + nb * dist[b]) / (na + nb)
        dist[a] = merged
        dist[:, a] = merged
        dist[a, a] = np.inf
        dist[b] = np.inf
        dist[:, b] = np.inf
        order[a] = order[a] + order[b]
        size[a] = na + nb
        active.remove(b)
    return order[active[0]]


def _column_order(log10: np.ndarray) -> tuple[list[int] | None, int]:
    """Average linkage on 1 - Pearson r of the columns over complete rows (None if < 3)."""
    complete = np.isfinite(log10).all(axis=1)
    n = int(complete.sum())
    if n < 3 or log10.shape[1] < 3:
        return None, n
    r = np.corrcoef(log10[complete].T)
    r = np.nan_to_num(r, nan=0.0)
    return _average_linkage(1.0 - r), n


K_CLUSTERS = 48


def _row_clusters(rel: np.ndarray) -> list[int]:
    """Row order by profile: k-means on the centred profiles, clusters by average linkage.

    Missing values count as the row mean (0 in a centred profile). k-means starts from
    rows spread along the first principal component and runs 25 iterations, so the
    result is deterministic. Clusters are ordered by average linkage of their centroids,
    rows inside a cluster by the first principal component.
    """
    x = np.nan_to_num(rel, nan=0.0)
    n = len(x)
    if n == 0:
        return []
    centred = x - x.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        pc1 = centred @ vt[0]
    except np.linalg.LinAlgError:
        pc1 = np.zeros(n)
    by_pc = np.argsort(pc1, kind="stable")
    k = min(K_CLUSTERS, n)
    seeds = by_pc[np.linspace(0, n - 1, k).round().astype(int)]
    centroids = x[seeds].copy()
    label = np.full(n, -1, dtype=np.int64)
    sq = (x**2).sum(axis=1)
    for _ in range(25):
        d = sq[:, None] - 2.0 * (x @ centroids.T) + (centroids**2).sum(axis=1)[None, :]
        new = np.argmin(d, axis=1)
        if np.array_equal(new, label):
            break
        label = new
        for c in range(k):
            members = label == c
            if members.any():
                centroids[c] = x[members].mean(axis=0)
    used = np.unique(label)
    cd = np.sqrt(((centroids[used][:, None, :] - centroids[used][None, :, :]) ** 2).sum(axis=2))
    cluster_order = [int(used[i]) for i in _average_linkage(cd)]
    out: list[int] = []
    for c in cluster_order:
        members = np.flatnonzero(label == c)
        out.extend(members[np.argsort(pc1[members], kind="stable")].tolist())
    return out


def heatmap(m: QuantMatrix, conditions: Mapping[str, str], *, cluster: bool = False) -> Heatmap:
    """The rows and columns of ``m`` in heatmap order (keys without any value dropped).

    Without ``cluster``: the columns grouped by condition (conditions in the order of
    their first run, runs in experiment order) and the rows by mean log10 quantity over
    the runs with a value, highest first. With ``cluster``: the columns by average
    linkage on 1 - Pearson r of log10 quantity over the complete rows (in the
    condition order when fewer than three runs or complete rows), and the rows by
    :func:`_row_clusters` on the profiles relative to their row mean.
    """
    have = np.isfinite(m.values).any(axis=1)
    rows = np.flatnonzero(have)
    values = m.values[rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        log10 = np.log10(values)
        log2 = np.log2(values)
    mean2 = np.nanmean(log2, axis=1) if len(rows) else np.zeros(0)
    relative = log2 - mean2[:, None]
    groups_by_cond = condition_groups(conditions, m.runs)
    grouped = [r for runs in groups_by_cond.values() for r in runs]
    grouped += [r for r in m.runs if r not in grouped]
    col = {r: j for j, r in enumerate(m.runs)}
    columns = [col[r] for r in grouped]
    column_rule = "runs grouped by condition, in experiment order"
    if cluster:
        order, n_complete = _column_order(log10)
        if order is not None:
            columns = order
            column_rule = (
                "runs clustered by average linkage on 1 - Pearson r of log10 quantity over "
                f"the {n_complete:,} keys with a value in every run"
            )
        else:
            column_rule += " (too few runs or complete keys to cluster)"
    if cluster:
        row_order = np.asarray(_row_clusters(relative), dtype=np.int64)
        row_rule = (
            f"rows clustered: k-means (k = {min(K_CLUSTERS, len(rows))}) on the profiles "
            "relative to the row mean, clusters by average linkage, rows of a cluster by "
            "the first principal component"
        )
    else:
        row_order = np.argsort(-np.nanmean(log10, axis=1), kind="stable")
        row_rule = "rows by mean log10 quantity over the runs with a value, highest first"
    names = (
        m.keys["protein_group"].astype(str).to_numpy()
        if m.level == "protein"
        else (m.keys["peptidoform"].astype(str) + "/" + m.keys["charge"].astype(str)).to_numpy()
    )
    return Heatmap(
        groups=names[rows][row_order],
        runs=tuple(m.runs[j] for j in columns),
        conditions=tuple(conditions.get(m.runs[j], "") for j in columns),
        log10=log10[row_order][:, columns],
        relative=relative[row_order][:, columns],
        rows=rows[row_order],
        row_rule=row_rule,
        column_rule=column_rule,
        dropped=int((~have).sum()),
    )


# --------------------------------------------------------------------------- one protein


_SPECIES = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")


def species_of(group: str | None) -> tuple[str, ...]:
    """Species of a protein group by the viewer's rule (the identification page's rule).

    The entry-name suffix of each member (``ATLA3_HUMAN`` gives ``HUMAN``), members
    split at ``;``, ``DECOY_`` removed; a suffix of 2 to 6 capitals and digits starting
    with a capital counts. Distinct species in member order.
    """
    out: list[str] = []
    for member in str(group or "").split(";"):
        name = member.removeprefix("DECOY_")
        i = name.rfind("_")
        if i <= 0:
            continue
        sp = name[i + 1 :]
        if _SPECIES.match(sp) and sp not in out:
            out.append(sp)
    return tuple(out)


@dataclass(frozen=True)
class ProteinProfile:
    """One protein group across the runs (values from the engine's tables).

    ``lfq`` and ``lfq_transferred`` are per run (NaN: no LFQ value), None in a single
    run. ``quant`` has one row per run: ``run``, ``quantity`` (NaN when missing),
    ``quant_status``, ``description``, ``n_peptides`` and, with MBR,
    ``n_transferred_precursors``; a run without a row has NA there. ``precursors`` has
    the group's ``peptidoform`` and ``charge`` (from the runs' ``peptide_quant``) with
    the values ``precursor_lfq`` / ``precursor_quant`` (keys by runs). ``pg_q_value`` is
    the group's q (its winning row; experiment-wide in an experiment).
    """

    group: str
    runs: tuple[str, ...]
    lfq: np.ndarray | None = field(repr=False)
    lfq_transferred: np.ndarray | None = field(repr=False)
    n_features: int | None
    quant: pd.DataFrame = field(repr=False)
    precursors: pd.DataFrame = field(repr=False)
    precursor_lfq: np.ndarray | None = field(repr=False)
    precursor_quant: np.ndarray = field(repr=False)
    pg_q_value: float | None
    species: tuple[str, ...]
    in_lfq: bool


def _quant_row(rs: ResultSet, run: Run, group: str) -> dict[str, Any]:
    try:
        row = protein_quant(rs, run, group)
    except ViewerError as exc:
        return {"run": run.name, "error": str(exc)}
    if row is None:
        return {"run": run.name}
    return {
        "run": run.name,
        "quantity": row["quantity"] if row["quantity"] is not None else math.nan,
        "quant_status": row["quant_status"],
        "description": row["description"],
        "n_peptides": row["n_peptides"],
        "n_transferred_precursors": row.get("n_transferred_precursors"),
    }


def protein_profile(rs: ResultSet, group: str) -> ProteinProfile:
    """The quantities of one protein group in every run (exact ``protein_group`` match).

    Memoised on the result set per group (a profile is small).
    """
    group = str(group)
    key = (
        "quantqc.profile",
        group,
        _identity(rs, "protein", "quant"),
        _identity(rs, "precursor", "quant"),
    )
    return _memo(rs, key, lambda: _protein_profile(rs, group))


def _protein_profile(rs: ResultSet, group: str) -> ProteinProfile:
    runs = tuple(r.name for r in rs.runs)
    lfq = lfq_tr = None
    n_features: int | None = None
    in_lfq = False
    precursor_lfq: np.ndarray | None = None
    if rs.is_experiment:
        try:
            m = quant_matrix(rs, "protein", "lfq")
        except ViewerError:
            m = None
        if m is not None:
            hit = m.key_index().get_indexer([group])[0]
            in_lfq = hit >= 0
            lfq = m.values[hit].copy() if in_lfq else np.full(len(runs), np.nan)
            if m.transferred is not None:
                lfq_tr = m.transferred[hit].copy() if in_lfq else np.zeros(len(runs), np.int64)
            if in_lfq and "n_features" in m.keys:
                n_features = int(m.keys["n_features"].iloc[hit])
    quant = pd.DataFrame.from_records([_quant_row(rs, run, group) for run in rs.runs])
    for column in ("quantity", "quant_status", "description", "n_peptides", "error"):
        if column not in quant:
            quant[column] = np.nan if column == "quantity" else None
    quant["quantity"] = quant["quantity"].astype("float64")
    quant["n_peptides"] = pd.array(
        [None if pd.isna(v) else int(v) for v in quant["n_peptides"]], dtype="Int64"
    )
    for column in ("quant_status", "description", "error"):
        quant[column] = quant[column].astype(object).where(quant[column].notna(), None)
    if "n_transferred_precursors" in quant:
        quant["n_transferred_precursors"] = pd.array(
            [None if pd.isna(v) else int(v) for v in quant["n_transferred_precursors"]],
            dtype="Int64",
        )
    union = quant_relation(rs, "peptide_quant", ("peptidoform", "charge", "protein_group"))
    if union is not None:
        sql, params, _ = union
        precursors = rs.duck.df(
            f"SELECT DISTINCT peptidoform, charge FROM ({sql}) u WHERE protein_group = ? "
            "ORDER BY peptidoform, charge",
            [*params, group],
        )
        precursors["charge"] = precursors["charge"].astype("int64")
    else:
        precursors = pd.DataFrame({"peptidoform": [], "charge": []})
    keys = _index(precursors, "precursor") if len(precursors) else None

    def values_of(source: str) -> np.ndarray:
        out = np.full((len(precursors), len(runs)), np.nan)
        if keys is None:
            return out
        mm = quant_matrix(rs, "precursor", source)
        where = mm.key_index().get_indexer(keys)
        ok = where >= 0
        out[ok] = mm.values[where[ok]]
        return out

    try:
        precursor_quant = values_of("quant")
    except ViewerError:
        precursor_quant = np.full((len(precursors), len(runs)), np.nan)
    if rs.is_experiment:
        try:
            precursor_lfq = values_of("lfq")
        except ViewerError:
            precursor_lfq = None
    q = None
    if rs.scored.parquet().has_column("pg_q_value"):
        q = rs.duck.scalar(
            "SELECT min(pg_q_value) FROM read_parquet(?) WHERE protein_group = ? "
            "AND label = 'target'",
            [sql_path(rs.scored.require()), group],
        )
    return ProteinProfile(
        group=group,
        runs=runs,
        lfq=lfq,
        lfq_transferred=lfq_tr,
        n_features=n_features,
        quant=quant,
        precursors=precursors,
        precursor_lfq=precursor_lfq,
        precursor_quant=precursor_quant,
        pg_q_value=None if q is None or (isinstance(q, float) and math.isnan(q)) else float(q),
        species=species_of(group),
        in_lfq=in_lfq,
    )


def find_groups(
    rs: ResultSet, text: str | None, *, limit: int = 30, accepted_at: float | None = None
) -> list[str]:
    """Protein groups whose name contains ``text`` (any case), best matches first.

    The groups are the keys of the protein quantities (MaxLFQ in an experiment, else
    ``protein_group_quant``), restricted to the groups accepted at ``accepted_at`` when
    it is given. An exact match comes first, then names that start with ``text``, then
    the rest; ties by mean log10 quantity, highest first. An empty ``text`` gives the
    most abundant groups.
    """
    source = "lfq" if rs.is_experiment else "quant"
    try:
        m = quant_matrix(rs, "protein", source, accepted_at=accepted_at)
    except ViewerError:
        if source == "quant":
            return []
        m = quant_matrix(rs, "protein", "quant", accepted_at=accepted_at)
    names = m.keys["protein_group"].astype(str)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a key without any value
        mean = np.nanmean(np.log10(m.values), axis=1) if m.n_keys else np.zeros(0)
    mean = np.nan_to_num(mean, nan=-np.inf)
    needle = (text or "").strip().lower()
    lower = names.str.lower()
    if needle:
        hit = lower.str.contains(needle, regex=False).to_numpy()
    else:
        hit = np.ones(len(names), dtype=bool)
    idx = np.flatnonzero(hit)
    rank = np.where(lower.to_numpy()[idx] == needle, 0, 1) if needle else np.ones(idx.size)
    if needle:
        starts = lower.str.startswith(needle).to_numpy()[idx]
        rank = np.where(rank == 0, 0, np.where(starts, 1, 2))
    order = np.lexsort((-mean[idx], rank))
    return names.to_numpy()[idx[order]][:limit].tolist()
