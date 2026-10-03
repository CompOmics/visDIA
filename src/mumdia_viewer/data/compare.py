"""Compare two result sets (P2 view 10): overlap, shared scores and quantities, the
identifications unique to each side, and the differences in provenance.

The two sides can come from different libraries, so candidate ids and
``base_peptide_id`` values mean nothing across them. Every match is made by strings:

* precursors by ``(peptidoform, charge)``;
* peptides by the stripped sequence of the row that carries ``peptide_q_value`` (the
  modifications removed with :func:`strip_sequence`; I and L stay distinct);
* protein groups by the ``protein_group`` string. The two sides can group the same
  proteins differently, so a group of one side can be missing from the other while its
  members pass there in another group (:func:`unique_keys` reports that).

Each side is accepted on its own q column at the threshold: ``precursor_q``,
``peptide_q_value`` and ``pg_q_value`` (experiment-wide in an experiment). These are the
grouped columns, set on each group's winning row only, so the rows that pass are the
winners: one row per key, whose score, run and candidate id the functions return. Only
targets count (in entrapment mode the real targets, as in :mod:`.counts`). No q value is
recomputed; the overlap, the correlations and the pooled quantities are the viewer's.

One scan per side reads the winning rows up to ``q <= 0.1`` (:func:`winner_rows`); every
threshold up to 0.1 is then a filter in memory. Results are memoised on the result sets,
keyed by both scored tables' identities.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from .discovery import ResultSet
from .duck import sql_path
from .entrapment import count_classes
from .quant import quant_relation
from .rescore import precursor_q_is_precursor_unit, rescore_info
from .units import check_threshold, execute_bound, format_threshold

__all__ = [
    "COMPARE_UNITS",
    "FETCH_Q",
    "SPEC",
    "ConfigDiff",
    "Overlap",
    "ProvenanceRow",
    "RunPair",
    "UnitSpec",
    "config_diff",
    "flatten_config",
    "overlap",
    "provenance",
    "quantity_pairs",
    "run_inputs",
    "run_pairs",
    "shared_keys",
    "side_counts",
    "spearman",
    "strip_sequence",
    "unique_keys",
    "unit_keys",
    "winner_rows",
]

COMPARE_UNITS: tuple[str, ...] = ("precursor", "peptide", "protein_group")
# One scan per side reads every winning row up to this q; a larger threshold reads again.
FETCH_Q = 0.1

# A modification tag ([...] or (...)) with the dash that joins a terminal one.
_TAG = re.compile(r"-?(\[[^\]]*\]|\([^)]*\))-?")
_TAG_SQL = r"-?(\[[^\]]*\]|\([^)]*\))-?"
STRIP_SQL = (
    "CASE WHEN contains(peptidoform, '[') OR contains(peptidoform, '(') "
    f"THEN regexp_replace(peptidoform, '{_TAG_SQL}', '', 'g') ELSE peptidoform END"
)


@dataclass(frozen=True)
class UnitSpec:
    """How one unit is matched across the sides."""

    key: str
    singular: str
    plural: str
    q_column: str
    match: str  # what is matched across the sides, in words

    def describe(self, t: float) -> str:
        return f"{self.plural} ({self.match}, {self.q_column} ≤ {format_threshold(t)})"


SPEC: dict[str, UnitSpec] = {
    "precursor": UnitSpec(
        "precursor", "precursor", "precursors", "precursor_q", "unique (peptidoform, charge)"
    ),
    "peptide": UnitSpec(
        "peptide",
        "peptide",
        "peptides",
        "peptide_q_value",
        "unique stripped sequence of the base_peptide_id winners",
    ),
    "protein_group": UnitSpec(
        "protein_group",
        "protein group",
        "protein groups",
        "pg_q_value",
        "unique protein_group string",
    ),
}


def strip_sequence(text: str | None) -> str:
    """The amino-acid sequence of a peptidoform: ``[Acetyl]-PEM[Oxidation]K`` -> ``PEMK``.

    Tags in ``[...]`` or ``(...)`` are removed with the dash that joins a terminal tag.
    A ``DECOY_`` prefix is kept (decoys are never compared).
    """
    return _TAG.sub("", text or "")


def _obj(values: Any) -> pd.Series:
    """Strings as an object column: pandas would otherwise build an Arrow string column,
    which costs more to build than the scan that read them."""
    return pd.Series(np.asarray(values, dtype=object), dtype=object)


def _check_unit(unit: str) -> UnitSpec:
    if unit not in SPEC:
        raise ValueError(f"unknown unit {unit!r}; the compared units are {', '.join(SPEC)}.")
    return SPEC[unit]


def _run_names(rs: ResultSet) -> dict[int, str]:
    return {int(r.index): r.name for r in rs.runs}


# --------------------------------------------------------------------------- one side


def winner_rows(rs: ResultSet, qmax: float = FETCH_Q) -> pd.DataFrame:
    """The target rows of the scored table that pass any grouped q column at ``qmax``.

    Columns: ``peptidoform``, ``charge``, ``protein_group``, ``base_peptide_id``,
    ``score``, ``source``, ``candidate_id``, ``precursor_q``, ``peptide_q_value``,
    ``pg_q_value`` and ``sequence`` (:func:`strip_sequence` of the peptidoform). The rows
    are the winners of their groups (the grouped columns hold 1.0 elsewhere). Targets
    follow :func:`.entrapment.count_classes` (real targets in entrapment mode).
    """
    qmax = max(float(qmax), FETCH_Q)
    cls = count_classes(rs)
    key = ("compare.winners", rs.scored.identity(), qmax, cls.key)

    def make() -> pd.DataFrame:
        sql = (
            "SELECT peptidoform, charge::INTEGER AS charge, protein_group, base_peptide_id, "
            "score, coalesce(source, 0)::INTEGER AS source, candidate_id::BIGINT AS candidate_id, "
            f"precursor_q, peptide_q_value, pg_q_value, {STRIP_SQL} AS sequence "
            f"FROM read_parquet($path) WHERE {cls.target} AND (precursor_q <= $q "
            "OR peptide_q_value <= $q OR pg_q_value <= $q)"
        )
        params = {**cls.params, "path": sql_path(rs.scored.require()), "q": qmax}
        df = execute_bound(rs.duck, sql, params).df()
        rs.duck._release()
        df["protein_group"] = df["protein_group"].fillna("")
        return df

    return rs.memo(key, make)


def _rows_for(rs: ResultSet, t: float) -> pd.DataFrame:
    return winner_rows(rs, t if t > FETCH_Q else FETCH_Q)


def unit_keys(rs: ResultSet, unit: str, t: float) -> pd.DataFrame:
    """One row per key of ``unit`` that passes ``t`` on the unit's q column.

    Columns: ``key`` (the matched string), ``q`` (the engine's value of the unit's q
    column), ``score``, ``run`` (run name, ``''`` in a single run), ``cid`` (the winning
    row's candidate id), ``peptidoform``, ``charge``, ``protein_group``, ``sequence``,
    and ``n_ids`` (the engine's keys behind the matched key: base_peptide_ids for a
    peptide, 1 otherwise). When several rows share a key (several base_peptide_ids
    with one sequence), the row with the smallest q and then the largest score is kept.
    """
    spec = _check_unit(unit)
    t = check_threshold(spec.key, float(t))
    key = ("compare.keys", rs.scored.identity(), unit, t, count_classes(rs).key)

    def make() -> pd.DataFrame:
        rows = _rows_for(rs, t)
        df = rows[rows[spec.q_column] <= t]
        if unit == "precursor":
            keys = df["peptidoform"] + "/" + df["charge"].astype(str)
        elif unit == "peptide":
            keys = df["sequence"]
        else:
            df = df[df["protein_group"] != ""]
            keys = df["protein_group"]
        out = pd.DataFrame(
            {
                "key": _obj(keys),
                "q": df[spec.q_column].to_numpy(dtype=float),
                "score": df["score"].to_numpy(dtype=float),
                "source": df["source"].to_numpy(dtype=np.int64),
                "cid": df["candidate_id"].to_numpy(dtype=np.int64),
                "peptidoform": _obj(df["peptidoform"]),
                "charge": df["charge"].to_numpy(dtype=np.int64),
                "protein_group": _obj(df["protein_group"]),
                "sequence": _obj(df["sequence"]),
                "base_peptide_id": df["base_peptide_id"].to_numpy(dtype=np.int64),
            }
        )
        # The winner of each key: the smallest q, then the largest score (numeric sorts;
        # a string sort of 100,000 keys costs more than the whole scan).
        order = np.lexsort((-out["score"].to_numpy(), out["q"].to_numpy()))
        out = out.iloc[order]
        if unit == "peptide":
            n_ids = out.groupby("key", sort=False)["base_peptide_id"].nunique()
        out = out.drop_duplicates("key", keep="first").reset_index(drop=True)
        out["n_ids"] = out["key"].map(n_ids).to_numpy(dtype=np.int64) if unit == "peptide" else 1
        if rs.is_experiment:
            out["run"] = out["source"].map(_run_names(rs)).fillna("").astype(object)
        else:
            out["run"] = ""
        return out.drop(columns=["base_peptide_id"])

    return rs.memo(key, make)


def side_counts(rs: ResultSet, t: float) -> dict[str, int]:
    """The identification counts of one side at ``t``, as :func:`.counts.unit_counts`
    counts them: PSMs (rows, ``q_value``), precursors (unique (peptidoform, charge),
    ``precursor_q``), peptides (unique ``base_peptide_id``, ``peptide_q_value``) and
    protein groups (unique non-empty ``protein_group``, ``pg_q_value``); targets only.
    The grouped units come from :func:`winner_rows`, the PSMs from one count query."""
    t = float(t)
    for unit in ("psm", *COMPARE_UNITS):
        check_threshold(unit, t)
    cls = count_classes(rs)

    def make() -> dict[str, int]:
        rows = _rows_for(rs, t)
        sql = f"SELECT count(*) FROM read_parquet($path) WHERE {cls.target} AND q_value <= $t"
        params = {**cls.params, "path": sql_path(rs.scored.require()), "t": t}
        row = execute_bound(rs.duck, sql, params).fetchone()
        prec = rows[rows["precursor_q"] <= t]
        pep = rows[rows["peptide_q_value"] <= t]
        pg = rows[(rows["pg_q_value"] <= t) & (rows["protein_group"] != "")]
        return {
            "psm": int(row[0]) if row else 0,
            "precursor": int(prec[["peptidoform", "charge"]].drop_duplicates().shape[0]),
            "peptide": int(pep["base_peptide_id"].dropna().nunique()),
            "protein_group": int(pg["protein_group"].nunique()),
        }

    return dict(rs.memo(("compare.counts", rs.scored.identity(), t, cls.key), make))


# --------------------------------------------------------------------------- overlap


@dataclass(frozen=True)
class Overlap:
    """The overlap of one unit at one threshold.

    ``n_a`` and ``n_b`` are each side's count of matched keys (each on its own q
    column), ``n_both`` the keys in both, ``n_only_a`` and ``n_only_b`` the rest.
    ``label`` names the unit, the match and the q column; ``note`` says what the match
    can and cannot tell.
    """

    unit: str
    threshold: float
    n_a: int
    n_b: int
    n_both: int
    label: str
    note: str

    @property
    def n_only_a(self) -> int:
        return self.n_a - self.n_both

    @property
    def n_only_b(self) -> int:
        return self.n_b - self.n_both

    @property
    def jaccard(self) -> float | None:
        union = self.n_a + self.n_b - self.n_both
        return self.n_both / union if union else None


def _unit_note(a: ResultSet, b: ResultSet, unit: str) -> str:
    if unit == "precursor":
        notes = [
            "Matched by the peptidoform text and the charge, so both sides must write "
            "modifications alike."
        ]
        for name, rs in (("A", a), ("B", b)):
            info = rescore_info(rs)
            if not precursor_q_is_precursor_unit(info):
                notes.append(
                    f"In {name} compete.group_by = {info.group_by} kept about one form per base "
                    "peptide, so its precursor_q count approximates a base-peptide count."
                )
        return " ".join(notes)
    if unit == "peptide":
        return (
            "Each side accepts peptides on peptide_q_value, per base_peptide_id. The "
            "libraries can number peptides differently, so the viewer matches the stripped "
            "sequence of each winning row (modifications removed; I and L kept apart). A "
            "sequence with several base_peptide_ids counts once."
        )
    return (
        "Matched by the protein_group string. The two sides can group proteins "
        "differently, so a group missing from one side can have members that pass there in "
        "another group; the tables of unique groups say so."
    )


def overlap(a: ResultSet, b: ResultSet, t: float) -> list[Overlap]:
    """The overlap of precursors, peptides and protein groups of A and B at ``t``."""
    out = []
    for unit in COMPARE_UNITS:
        ka = set(unit_keys(a, unit, t)["key"].tolist())
        kb = set(unit_keys(b, unit, t)["key"].tolist())
        out.append(
            Overlap(
                unit=unit,
                threshold=float(t),
                n_a=len(ka),
                n_b=len(kb),
                n_both=len(ka & kb),
                label=SPEC[unit].describe(t),
                note=_unit_note(a, b, unit),
            )
        )
    return out


def shared_keys(a: ResultSet, b: ResultSet, unit: str, t: float) -> pd.DataFrame:
    """The keys that pass on both sides, with each side's winning row.

    Columns ``key``, then ``a_`` and ``b_`` versions of ``q``, ``score``, ``run``,
    ``cid``, ``peptidoform``, ``charge`` and ``protein_group``. Sorted by key. The
    frame is memoised on ``a`` (per pair, unit and threshold): do not modify it.
    """
    t = float(t)
    cols = ["key", "q", "score", "run", "cid", "peptidoform", "charge", "protein_group"]

    def make() -> pd.DataFrame:
        ka = unit_keys(a, unit, t)[cols]
        kb = unit_keys(b, unit, t)[cols]
        df = ka.merge(kb, on="key", how="inner", suffixes=("_a", "_b"))
        rename = {f"{c}_a": f"a_{c}" for c in cols[1:]} | {f"{c}_b": f"b_{c}" for c in cols[1:]}
        return df.rename(columns=rename).sort_values("key", kind="stable").reset_index(drop=True)

    key = (
        "compare.shared",
        b.scored.identity(),
        unit,
        t,
        count_classes(a).key,
        count_classes(b).key,
    )
    return a.memo(key, make)


# --------------------------------------------------------------------------- unique keys


def _lookup(rs: ResultSet, unit: str, keys: pd.DataFrame) -> pd.DataFrame:
    """The smallest value of the unit's q column over the target rows of ``keys`` in
    ``rs`` (every row, not only winners), and the number of those rows."""
    spec = SPEC[unit]
    cls = count_classes(rs)
    if keys.empty:
        return pd.DataFrame(
            {
                "key": pd.Series([], dtype=object),
                "other_q": pd.Series([], dtype=float),
                "other_rows": pd.Series([], dtype=np.int64),
            }
        )
    if unit == "precursor":
        table = pa.table(
            {
                "peptidoform": pa.array(keys["peptidoform"].astype(str).tolist(), pa.string()),
                "charge": pa.array([int(c) for c in keys["charge"]], pa.int32()),
            }
        )
        sql = (
            "SELECT s.peptidoform || '/' || CAST(s.charge AS VARCHAR) AS key, "
            "min(s.precursor_q) AS other_q, count(*) AS other_rows "
            "FROM read_parquet($path) s JOIN __keys k "
            "ON s.peptidoform = k.peptidoform AND s.charge = k.charge "
            f"WHERE {cls.target} GROUP BY 1"
        )
    else:
        table = pa.table({"key": pa.array(keys["key"].astype(str).tolist(), pa.string())})
        expr = STRIP_SQL if unit == "peptide" else "protein_group"
        sql = (
            f"SELECT x.key, min(x.q) AS other_q, count(*) AS other_rows FROM (SELECT {expr} "
            f"AS key, {spec.q_column} AS q FROM read_parquet($path) WHERE {cls.target}) x "
            "JOIN __keys k ON x.key = k.key GROUP BY 1"
        )
    name = f"cmp_keys_{uuid.uuid4().hex}"
    cur = rs.duck.cursor()
    cur.register(name, table)
    try:
        params = {**cls.params, "path": sql_path(rs.scored.require())}
        df = execute_bound(rs.duck, sql.replace("__keys", name), params).df()
    finally:
        cur.unregister(name)
        rs.duck._release()
    return df


def _members(group: str) -> list[str]:
    return [m for m in str(group).split(";") if m]


def unique_keys(a: ResultSet, b: ResultSet, unit: str, t: float, side: str = "a") -> pd.DataFrame:
    """The keys that pass on one side only (``side`` ``"a"`` or ``"b"``), with what the
    other side has for them.

    The columns of :func:`unit_keys`, plus ``other_q`` (the smallest value of the
    unit's q column over the other side's target rows of the key; NaN when it has no
    target row of the key) and ``other_rows`` (that number of rows; 0 for none). For
    protein groups, ``other_group`` names a group of the other side that passes at
    ``t`` and contains a member of the group ('' when none does). Sorted by ``q``, then
    by score (largest first).
    """
    if side not in ("a", "b"):
        raise ValueError("side is 'a' or 'b'.")
    own, other = (a, b) if side == "a" else (b, a)
    t = float(t)
    key = (
        "compare.unique",
        other.scored.identity(),
        unit,
        t,
        count_classes(own).key,
        count_classes(other).key,
    )

    def make() -> pd.DataFrame:
        mine = unit_keys(own, unit, t)
        theirs = unit_keys(other, unit, t)
        seen = set(theirs["key"].tolist())
        keep = np.array([k not in seen for k in mine["key"].tolist()], dtype=bool)
        df = mine[keep].copy()
        found = _lookup(other, unit, df)
        found["key"] = found["key"].astype(object)
        df["key"] = df["key"].astype(object)
        df = df.merge(found, on="key", how="left")
        df["other_q"] = df["other_q"].astype(float)
        df["other_rows"] = df["other_rows"].fillna(0).astype(np.int64)
        if unit == "protein_group":
            member_of: dict[str, str] = {}
            for g in theirs["key"]:
                for m in _members(g):
                    member_of.setdefault(m, g)
            df["other_group"] = [
                next((member_of[m] for m in _members(g) if m in member_of), "") for g in df["key"]
            ]
        return df.sort_values(["q", "score"], ascending=[True, False], kind="stable").reset_index(
            drop=True
        )

    return own.memo(key, make)


# --------------------------------------------------------------------------- runs


@dataclass(frozen=True)
class RunInput:
    run: str
    label: str
    path: str | None
    content_hash: str | None

    @property
    def file_name(self) -> str | None:
        return PureWindowsPath(self.path).name if self.path else None


def run_inputs(rs: ResultSet) -> list[RunInput]:
    """The mzML of each run: the manifest's ``mzml`` (single run) or ``mzml[i]`` input
    (run ``i``), else ``params.mzml`` of the run's ``spectra_ms2`` report."""
    inputs = rs.manifest.inputs
    out = []
    for run in rs.runs:
        rec = inputs.get(f"mzml[{run.index}]") or (
            inputs.get("mzml") if not rs.is_experiment else None
        )
        path = rec.path if rec is not None else None
        digest = rec.content_hash if rec is not None else None
        if path is None:
            art = run.artifact("spectra_ms2")
            if art is not None and art.report is not None:
                value = art.report.params.get("mzml")
                path = str(value) if value else None
        out.append(RunInput(run.name, run.label, path, digest))
    return out


@dataclass(frozen=True)
class RunPair:
    """A run of A and a run of B that searched the same mzML. ``how`` says how the
    viewer knows: ``"content hash"`` or ``"file name"``."""

    a: str
    b: str
    a_label: str
    b_label: str
    how: str


def run_pairs(a: ResultSet, b: ResultSet) -> list[RunPair]:
    """Runs of A and B with the same mzML, one to one, in A's run order.

    The recorded content hash decides; among runs with the same hash (copies of one
    file), the same file name is preferred. Without hashes the file name decides.
    """
    ia, ib = run_inputs(a), run_inputs(b)
    used: set[str] = set()
    out: list[RunPair] = []
    passes = (
        (
            "content hash",
            lambda x, y: (
                x.content_hash and x.content_hash == y.content_hash and x.file_name == y.file_name
            ),
        ),
        ("content hash", lambda x, y: x.content_hash and x.content_hash == y.content_hash),
        (
            "file name",
            lambda x, y: (
                not (x.content_hash and y.content_hash)
                and x.file_name
                and x.file_name == y.file_name
            ),
        ),
    )
    paired: dict[str, RunPair] = {}
    for how, same in passes:
        for x in ia:
            if x.run in paired:
                continue
            for y in ib:
                if y.run not in used and same(x, y):
                    paired[x.run] = RunPair(x.run, y.run, x.label, y.label, how)
                    used.add(y.run)
                    break
    for x in ia:
        if x.run in paired:
            out.append(paired[x.run])
    return out


# --------------------------------------------------------------------------- quantities


def _quantities(rs: ResultSet) -> pd.DataFrame:
    """``peptide_quant`` of every run: ``source``, ``candidate_id``, ``peptidoform``,
    ``charge``, ``quantity`` (null when not quantifiable), ``quant_status`` and ``key``."""
    columns = ["candidate_id", "peptidoform", "charge", "quantity", "quant_status"]

    def make() -> pd.DataFrame:
        rel = quant_relation(rs, "peptide_quant", columns)
        if rel is None:
            df = pd.DataFrame({c: [] for c in ["source", *columns]})
        else:
            sql, params, _ = rel
            df = rs.duck.df(sql, params)
        df["key"] = df["peptidoform"].astype(str) + "/" + df["charge"].astype("Int64").astype(str)
        return df

    return rs.memo(("compare.quantities", *(r.name for r in rs.runs)), make)


def quantity_pairs(
    a: ResultSet, b: ResultSet, t: float, pairs: Sequence[RunPair] | None = None
) -> pd.DataFrame:
    """The quantities of the precursors that pass on both sides at ``t``.

    With ``pairs`` (runs that searched the same mzML, :func:`run_pairs`): one point per
    precursor and pair, the ``peptide_quant.quantity`` in that run of A against that run
    of B. Without: pooled, one point per precursor, each side's median of
    ``peptide_quant.quantity`` over its runs with a quantity (the viewer's median).

    Columns ``key``, ``peptidoform``, ``charge``, ``a_run``, ``b_run`` (run names; ''
    when pooled), ``a_cid``, ``b_cid`` (the candidate whose quantity it is; the winning
    row's when pooled), ``a_quantity`` and ``b_quantity`` (NaN when the side has no
    quantity: not quantifiable, or no row in that run). ``attrs`` hold ``mode`` (``"per
    run"`` or ``"pooled"``), ``label``, ``n_shared`` (shared precursors) and ``n_both``
    (points with a positive quantity on both sides). Memoised on ``a``: do not modify it.
    """
    t = float(t)
    names = tuple((p.a, p.b) for p in pairs) if pairs else None
    key = ("compare.quantity", b.scored.identity(), t, names, count_classes(b).key)
    return a.memo(key, lambda: _quantity_pairs(a, b, t, pairs))


def _quantity_pairs(
    a: ResultSet, b: ResultSet, t: float, pairs: Sequence[RunPair] | None
) -> pd.DataFrame:
    shared = shared_keys(a, b, "precursor", t)
    qa, qb = _quantities(a), _quantities(b)
    frames = []
    if pairs:
        for pair in pairs:
            sa, sb = a.run(pair.a).index, b.run(pair.b).index
            va = qa[qa["source"] == sa].drop_duplicates("key").set_index("key")
            vb = qb[qb["source"] == sb].drop_duplicates("key").set_index("key")
            keys = shared["key"]
            frames.append(
                pd.DataFrame(
                    {
                        "key": _obj(keys),
                        "peptidoform": _obj(shared["a_peptidoform"]),
                        "charge": shared["a_charge"].to_numpy(dtype=np.int64),
                        "a_run": pair.a,
                        "b_run": pair.b,
                        "a_cid": keys.map(va["candidate_id"]).to_numpy(dtype=float),
                        "b_cid": keys.map(vb["candidate_id"]).to_numpy(dtype=float),
                        "a_quantity": keys.map(va["quantity"]).to_numpy(dtype=float),
                        "b_quantity": keys.map(vb["quantity"]).to_numpy(dtype=float),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        mode = "per run"
        label = "peptide_quant.quantity in runs that searched the same mzML: " + ", ".join(
            f"A {p.a_label} = B {p.b_label}" for p in pairs
        )
    else:
        va = qa.dropna(subset=["quantity"]).groupby("key")["quantity"].median()
        vb = qb.dropna(subset=["quantity"]).groupby("key")["quantity"].median()
        keys = shared["key"]
        df = pd.DataFrame(
            {
                "key": _obj(keys),
                "peptidoform": _obj(shared["a_peptidoform"]),
                "charge": shared["a_charge"].to_numpy(dtype=np.int64),
                "a_run": _obj(shared["a_run"]),
                "b_run": _obj(shared["b_run"]),
                "a_cid": shared["a_cid"].to_numpy(dtype=float),
                "b_cid": shared["b_cid"].to_numpy(dtype=float),
                "a_quantity": keys.map(va).to_numpy(dtype=float),
                "b_quantity": keys.map(vb).to_numpy(dtype=float),
            }
        )
        mode = "pooled"
        label = (
            "each side's median of peptide_quant.quantity over its runs with a quantity "
            "(the viewer's median; no run of A searched the same mzML as a run of B)"
        )
    ok = (df["a_quantity"] > 0) & (df["b_quantity"] > 0)
    df.attrs.update(
        {"mode": mode, "label": label, "n_shared": len(shared), "n_both": int(ok.sum())}
    )
    return df


def spearman(x: Any, y: Any) -> float | None:
    """The viewer's Spearman rank correlation of two equal-length series (None below 3)."""
    sx, sy = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = sx.notna() & sy.notna()
    if int(ok.sum()) < 3:
        return None
    rx, ry = sx[ok].rank(), sy[ok].rank()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- provenance


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"extract": {"frag_tol_ppm": 10}}`` -> ``{"extract.frag_tol_ppm": 10}``. Lists and
    scalars are leaves; an empty mapping is a leaf ``{}``."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out.update(flatten_config(v, name))
        else:
            out[name] = v
    return out


def _show(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _display(value: Any) -> str:
    """A configuration value as text: a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else _show(value)


def _same_path(x: Any, y: Any) -> bool:
    if not (isinstance(x, str) and isinstance(y, str)) or x == y:
        return False
    if not any(s in x for s in ("/", "\\")):
        return False
    return x.replace("\\", "/").rstrip("/").lower() == y.replace("\\", "/").rstrip("/").lower()


@dataclass(frozen=True)
class ConfigDiff:
    key: str
    a: str  # the value (a string as it is, else JSON); '' when absent
    b: str
    kind: str  # "differs", "only in A", "only in B"
    note: str


def config_diff(a: ResultSet, b: ResultSet) -> list[ConfigDiff]:
    """The keys of the resolved configurations (``config_json``) whose values differ,
    with each side's value as JSON. A path that differs only in its separators or its
    case is marked so in ``note``."""
    fa, fb = flatten_config(a.manifest.config), flatten_config(b.manifest.config)
    out = []
    for k in sorted(set(fa) | set(fb)):
        if k in fa and k in fb:
            if _show(fa[k]) == _show(fb[k]):
                continue
            note = "the same path, written differently" if _same_path(fa[k], fb[k]) else ""
            out.append(ConfigDiff(k, _display(fa[k]), _display(fb[k]), "differs", note))
        elif k in fa:
            out.append(ConfigDiff(k, _display(fa[k]), "", "only in A", ""))
        else:
            out.append(ConfigDiff(k, "", _display(fb[k]), "only in B", ""))
    return out


@dataclass(frozen=True)
class ProvenanceRow:
    """One line of the side-by-side header: ``same`` is None when it is not compared."""

    label: str
    a: str
    b: str
    same: bool | None
    tip: str


def _short(digest: str | None) -> str:
    return digest[:10] if digest else ""


def _input(rs: ResultSet, key: str) -> tuple[str, str | None]:
    rec = rs.manifest.inputs.get(key)
    if rec is None:
        return "", None
    name = PureWindowsPath(rec.path).name if rec.path else ""
    text = f"{name} ({_short(rec.content_hash)})" if rec.content_hash else name
    return text, rec.content_hash or name


def provenance(a: ResultSet, b: ResultSet) -> list[ProvenanceRow]:
    """Versions, commits, models, configuration hashes and inputs of A and B."""
    ma, mb = a.manifest, b.manifest

    def kind(rs: ResultSet) -> str:
        return f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"

    def row(label: str, x: Any, y: Any, tip: str, compare: bool = True) -> ProvenanceRow:
        sx = "" if x is None else str(x)
        sy = "" if y is None else str(y)
        return ProvenanceRow(label, sx, sy, (sx == sy) if compare else None, tip)

    rows = [
        row("Kind", kind(a), kind(b), "single run or experiment, and the number of runs"),
        row("MuMDIA version", ma.mumdia_version, mb.mumdia_version, "manifest mumdia_version"),
        row("Git SHA", ma.git_sha, mb.git_sha, "manifest git_sha: the engine's commit"),
        row("Commit date", ma.commit_date, mb.commit_date, "manifest commit_date"),
        row(
            "Configuration hash",
            _short(ma.config_hash),
            _short(mb.config_hash),
            "manifest config_hash of the resolved configuration (first 10 characters)",
        ),
    ]
    for ident, label in (
        ("rescorer", "Rescorer"),
        ("rt_predictor", "RT predictor"),
        ("fragment_predictor", "Fragment predictor"),
    ):
        rows.append(
            row(
                label,
                ma.model_identities.get(ident),
                mb.model_identities.get(ident),
                f"manifest model_identities.{ident}",
            )
        )
    ra, rb = rescore_info(a), rescore_info(b)
    rows.append(
        row(
            "FDR mode",
            ra.mode.replace("_", "-"),
            rb.mode.replace("_", "-"),
            "target-decoy or entrapment q values (the rescore report's classifier)",
        )
    )
    for key, label in (
        ("lib_precursors", "Library precursors"),
        ("lib_fragments", "Library fragments"),
        ("fasta", "FASTA"),
    ):
        (ta, ha), (tb, hb) = _input(a, key), _input(b, key)
        if not ta and not tb:
            continue
        rows.append(
            ProvenanceRow(
                label,
                ta,
                tb,
                (ha == hb) if ha and hb else None,
                f"manifest inputs.{key}: file name and content hash (first 10 characters). "
                "The same hash means the same file.",
            )
        )
    pairs = run_pairs(a, b)
    ia, ib = run_inputs(a), run_inputs(b)
    rows.append(
        ProvenanceRow(
            "mzML inputs",
            f"{len(ia)} file{'s' if len(ia) != 1 else ''}",
            f"{len(ib)} file{'s' if len(ib) != 1 else ''}",
            len(pairs) == len(ia) == len(ib),
            "The runs' mzML inputs. "
            + (
                f"{len(pairs)} run{'s' if len(pairs) != 1 else ''} of A searched the same mzML "
                "as a run of B: "
                + ", ".join(f"{p.a_label} = {p.b_label}" for p in pairs)
                + f" (by {pairs[0].how})."
                if pairs
                else "No run of A searched the same mzML as a run of B."
            ),
        )
    )
    mbr_a = (ma.experiment or {}).get("mbr", ma.model_identities.get("mbr"))
    mbr_b = (mb.experiment or {}).get("mbr", mb.model_identities.get("mbr"))
    if a.is_experiment or b.is_experiment:
        rows.append(
            row(
                "Match-between-runs",
                mbr_a if a.is_experiment else "single run",
                mbr_b if b.is_experiment else "single run",
                "experiment_manifest experiment.mbr",
            )
        )
    return rows


def describe_q(value: float | None, t: float) -> str:
    """A q value of the other side in words, for the unique tables."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "no target row"
    return f"q {value:.3g} > {format_threshold(t)}"
