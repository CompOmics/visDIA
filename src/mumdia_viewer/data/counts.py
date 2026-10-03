"""Identification counts, per-run counts, identification curves and score histograms.

Every number here counts rows of the engine's own q columns at a threshold. No q value
is recomputed, and every count names its unit, its key and its q column
(:mod:`.units`). The counts run in DuckDB over the pooled scored table with a column
projection: ``psms_scored.parquet`` in a single run, ``scored_combined.parquet`` in an
experiment. ``scored_combined`` is never augmented by match-between-runs, so these are
native identifications; transfers are counted separately in :mod:`.mbr`.

Rules that the functions enforce:

* The grouped q columns (``precursor_q``, ``peptide_q_value``, ``pg_q_value``) are set on
  each group's winning row only and are experiment-wide in an experiment. They are
  never counted per run; per-run counts use ``run_psm_q``. Distinct precursors,
  peptides and protein groups per run are derived from PSM-level acceptance and are
  labelled so.
* A sparse unit needs a threshold below 1 (:func:`.units.check_threshold`).
* Identification counts are target counts. In entrapment mode "target" means real
  target, as in the engine's report statistics: spike-ins are excluded by the engine's
  test (:func:`.entrapment.count_classes`). When the run records no marker strings, the
  viewer's rule decides, and the labels say so. Decoy counts are a diagnostic, not an
  FDR.
* Under ``compete.group_by = base_peptide`` competition kept about one form per base
  peptide, so a count on ``precursor_q`` is labelled as a base-peptide count.

Results are cached in the result set's memo, keyed by the scored table's identity and
the arguments.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .discovery import ResultSet
from .duck import sql_path
from .entrapment import CountClasses, count_classes, entrapment_expr, sql_literal
from .mbr import mbr_ran
from .rescore import precursor_q_is_precursor_unit, rescore_info
from .units import (
    COUNT_UNITS,
    UNITS,
    Unit,
    check_threshold,
    execute_bound,
    format_threshold,
    get_unit,
    run_key,
)

__all__ = [
    "ENGINE_STAT_KEYS",
    "MAX_WINNER_KEYS",
    "Count",
    "EngineCheck",
    "counts_at",
    "engine_check",
    "engine_stats",
    "group_winner_sql",
    "group_winners",
    "id_curve",
    "per_run_counts",
    "score_histogram",
    "unit_counts",
]

# The rescore report statistic that counts each unit at q <= 0.01 (a literal threshold).
ENGINE_STAT_KEYS: dict[str, str] = {
    "psm": "target_psms_at_1pct",
    "precursor": "target_precursors_at_1pct",
    "peptide": "target_peptides_at_1pct",
    "protein_group": "target_protein_groups_at_1pct",
}
ENGINE_THRESHOLD = 0.01

# The engine's group key for each grouped q column (stages/rescore.rs grouped_q). The
# protein-group key is the interned `protein` string; `protein_group` is a byte copy.
_WINNER_PARTITION: dict[str, tuple[str, ...]] = {
    "precursor": ("peptidoform", "charge"),
    "peptide": ("base_peptide_id",),
    "protein_group": ("protein",),
}
# group_winners returns a DataFrame, so it takes at most this many keys. The winners of
# a whole table (1.79 million precursors in a six-run experiment) stay in DuckDB
# (group_winner_sql).
MAX_WINNER_KEYS = 10_000


@dataclass(frozen=True)
class Count:
    """One identification count at one threshold.

    ``n_target`` is the identification count (real targets in entrapment mode);
    ``n_decoy`` counts decoys on the same key and column (a diagnostic);
    ``n_spike_in`` counts spike-ins when the library has entrapment markers.
    ``population`` names the table and the runs; ``sql`` is the counting query with
    its values written in, for display.
    """

    unit: str
    threshold: float
    n_target: int
    n_decoy: int
    population: str
    label: str
    derived: bool
    note: str
    sql: str
    q_column: str = ""
    n_spike_in: int | None = None


class EngineCheck(NamedTuple):
    """The engine's count of a unit at 0.01 beside the viewer's count of the same column."""

    unit: str
    engine: int | None
    viewer: int
    equal: bool | None


# --------------------------------------------------------------------------- helpers


def _real_qualifier(cls: CountClasses) -> str:
    """A label suffix when the real targets follow the viewer's rule, not the run's."""
    if cls.excludes_spike_ins and not cls.markers_recorded:
        return "; real targets by the viewer's spike-in rule, which the run does not record"
    return ""


def _population(rs: ResultSet) -> str:
    name = rs.scored.path.name if rs.scored.path is not None else rs.scored.key
    if not rs.is_experiment:
        return f"{name}: single run"
    labels = [r.label for r in rs.runs]
    shown = ", ".join(labels) if len(labels) <= 8 else f"{labels[0]} ... {labels[-1]}"
    return f"{name}: all {len(labels)} runs pooled ({shown}); experiment-wide"


def _pass(n: int) -> str:
    return "passes" if n == 1 else "pass"


def _path(rs: ResultSet) -> str:
    return sql_path(rs.scored.require())


def _execute(rs: ResultSet, sql: str, params: dict[str, Any]) -> Any:
    """Run ``sql`` on the result set's DuckDB, binding only the parameters it uses."""
    return execute_bound(rs.duck, sql, params)


# --------------------------------------------------------------------------- unit counts


def _unit_aggregates(rs: ResultSet, t: float, cls: CountClasses) -> dict[str, Any]:
    key = ("unit_aggregates", rs.scored.identity(), t, cls.key)
    if key in rs._memo:
        return rs._memo[key]
    columns = [
        "label",
        "q_value",
        "precursor_q",
        "peptide_q_value",
        "pg_q_value",
        "peptidoform",
        "charge",
        "base_peptide_id",
        "protein_group",
    ]
    if cls.params:
        columns.append("protein")
    selects: list[str] = []
    names: list[str] = []
    for k in COUNT_UNITS:
        u = UNITS[k]
        for cls_name, flag in (("t", "is_t"), ("d", "is_d"), ("e", "is_e")):
            selects.append(u.count_sql(f"{flag} AND {u.q_column} <= $t"))
            names.append(f"{k}_{cls_name}")
    selects.append(
        "count(DISTINCT protein_group) FILTER (WHERE is_t AND pg_q_value <= $t "
        "AND protein_group = '')"
    )
    names.append("protein_group_t_empty")
    sql = (
        f"WITH x AS (SELECT {', '.join(columns)}, {cls.target} AS is_t, {cls.decoy} AS is_d, "
        f"{cls.spike} AS is_e FROM read_parquet($path) WHERE q_value <= $t "
        "OR precursor_q <= $t OR peptide_q_value <= $t OR pg_q_value <= $t) "
        f"SELECT {', '.join(selects)} FROM x"
    )
    row = _execute(rs, sql, {**cls.params, "path": _path(rs), "t": t}).fetchone()
    assert row is not None
    raw = {name: int(value) for name, value in zip(names, row, strict=True)}
    rs._memo[key] = raw
    return raw


def _floor_note(rs: ResultSet, u: Unit, t: float, cls: CountClasses) -> str:
    """Why a unit has no identification at ``t``: the smallest q of its column."""
    sql = f"SELECT min({u.q_column}) FROM read_parquet($path) WHERE {u.where_sql(cls.target)}"
    row = _execute(rs, sql, {**cls.params, "path": _path(rs)}).fetchone()
    smallest = row[0] if row is not None else None
    if smallest is None:
        return f"No target row exists for this unit, so no {u.singular} can pass."
    if float(smallest) > t:
        return (
            f"No {u.singular} reaches {u.q_column} <= {format_threshold(t)}: the smallest "
            f"{u.q_column} of a target is {float(smallest):.4g}. A q value is at least 1/T, "
            "with T the number of targets in the estimate, so small data sets cannot reach "
            "low thresholds."
        )
    return ""


def _display_sql(rs: ResultSet, u: Unit, t: float, cls: CountClasses) -> str:
    count = "count(*)" if u.key_sql is None else f"count(DISTINCT {u.key_sql})"
    where = u.where_sql(f"{cls.inline_target} AND {u.q_column} <= {float(t)!r}")
    return f"SELECT {count} FROM read_parquet({sql_literal(_path(rs))}) WHERE {where}"


def unit_counts(rs: ResultSet, t: float = 0.01) -> list[Count]:
    """Target counts of the PSM, precursor, peptide and protein-group units at ``t``.

    Units: PSMs (rows, ``q_value``), precursors (unique ``(peptidoform, charge)``,
    ``precursor_q``), peptides (unique ``base_peptide_id``, ``peptide_q_value``) and
    protein groups (unique non-empty ``protein_group``, ``pg_q_value``). Each count is
    ``COUNT(DISTINCT key) FILTER (WHERE target AND q <= t)`` on the pooled scored
    table. In an experiment the grouped units are experiment-wide.
    """
    t = float(t)
    for k in COUNT_UNITS:
        check_threshold(k, t)
    cls = count_classes(rs)
    key = ("unit_counts", rs.scored.identity(), t, cls.key)
    if key in rs._memo:
        return list(rs._memo[key])
    raw = _unit_aggregates(rs, t, cls)
    info = rescore_info(rs)
    base_peptide_competition = not precursor_q_is_precursor_unit(info)
    population = _population(rs)
    native_only = mbr_ran(rs)
    entrapment_mode = cls.excludes_spike_ins
    out: list[Count] = []
    for k in COUNT_UNITS:
        u = UNITS[k]
        n_t, n_d, n_e = raw[f"{k}_t"], raw[f"{k}_d"], raw[f"{k}_e"]
        label = u.label(n_t, t)
        notes: list[str] = []
        if entrapment_mode and cls.markers_recorded:
            label += "; real targets only, spike-ins excluded (entrapment mode)"
        elif entrapment_mode:
            label += (
                "; real targets only, spike-ins excluded by the viewer's rule (entrapment mode; "
                "the run records no marker strings)"
            )
        if k == "precursor" and base_peptide_competition:
            label += (
                f"; compete.group_by = {info.group_by} kept about one form per base peptide, "
                "so this approximates a base-peptide count, not a count of every precursor"
            )
            notes.append(f"compete.group_by from {info.group_by_source}.")
        if k == "psm":
            notes.append(
                "q_value is the PSM q over all runs of the rescore (experiment-wide); "
                "per-run PSM counts use run_psm_q (per_run_counts)."
                if rs.is_experiment
                else "In a single run q_value equals run_psm_q."
            )
        else:
            scope = " experiment-wide," if rs.is_experiment else ""
            notes.append(
                f"{u.q_column} is{scope} set on the winning row of each {u.group_noun} group "
                "only; every other row holds 1.0."
            )
        if cls.spike_present and n_e:
            notes.append(
                f"{n_e:,} spike-in {u.noun(n_e)} {_pass(n_e)} the same cut"
                + (" and are not in the count." if entrapment_mode else " and are in the count.")
            )
        if cls.mode == "entrapment" and u.sparse:
            notes.append(
                f"In entrapment mode {u.q_column} is 1.0 on every decoy, so no decoy passes."
            )
        elif cls.mode == "entrapment":
            notes.append(
                f"{n_d:,} decoy {u.noun(n_d)} {_pass(n_d)} the same cut. In entrapment mode a "
                "decoy is ranked but counted in neither population of the estimate, so this "
                "is a side diagnostic, not an FDR."
            )
        else:
            notes.append(
                f"{n_d:,} decoy {u.noun(n_d)} {_pass(n_d)} the same cut (a diagnostic, not an FDR)."
            )
        if k == "protein_group":
            notes.append(
                "Rows with an empty protein_group are not counted (the proteins.tsv rule); "
                "the engine's statistic counts them."
            )
            if raw["protein_group_t_empty"]:
                notes.append("An empty protein_group passes this cut and is left out.")
        if native_only:
            notes.append(
                "Native identifications only: match-between-runs transfers are not in this "
                "count (see mbr.transfer_counts)."
            )
        if cls.note:
            notes.append(cls.note)
        if n_t == 0:
            floor = _floor_note(rs, u, t, cls)
            if floor:
                notes.append(floor)
        out.append(
            Count(
                unit=k,
                threshold=t,
                n_target=n_t,
                n_decoy=n_d,
                population=population,
                label=label,
                derived=False,
                note=" ".join(notes),
                sql=_display_sql(rs, u, t, cls),
                q_column=u.q_column,
                n_spike_in=n_e if cls.spike_present else None,
            )
        )
    rs._memo[key] = tuple(out)
    return list(out)


# --------------------------------------------------------------------------- per run


def _run_label(rs: ResultSet, source: int) -> str:
    try:
        return rs.run(source).label
    except KeyError:
        return f"source {source}"


def _sources(rs: ResultSet, found: Any) -> list[int]:
    known = [r.index for r in rs.runs]
    return known + sorted(int(s) for s in found if int(s) not in set(known))


def per_run_counts(rs: ResultSet, t: float = 0.01) -> pd.DataFrame:
    """Counts per run on ``run_psm_q``, the PSM q computed within each run.

    Columns: ``run``, ``source``, ``target_psms``, ``decoy_psms`` (rows with
    ``run_psm_q <= t``), ``spike_in_psms`` when the library has entrapment markers, and
    the derived ``precursors``, ``peptides`` and ``protein_groups``: the distinct keys
    with a target PSM at ``run_psm_q <= t``. The derived counts come from PSM-level
    acceptance within the run; they are not precursor-, peptide- or protein-level FDR
    counts. ``attrs`` holds the labels, the derived flags and the SQL.

    The grouped q columns are never used here: they are experiment-wide, and counting
    one run on them gives the experiment's winners that happen to lie in that run.
    """
    t = check_threshold("run_psm", t)
    cls = count_classes(rs)
    key = ("per_run_counts", rs.scored.identity(), t, cls.key)
    if key in rs._memo:
        return rs._memo[key].copy()
    columns = "source, label, peptidoform, charge, base_peptide_id, protein_group"
    if cls.params:
        columns += ", protein"
    sql = (
        f"WITH x AS (SELECT {columns}, {cls.target} AS is_t, {cls.decoy} AS is_d, "
        f"{cls.spike} AS is_e FROM read_parquet($path) WHERE run_psm_q <= $t) "
        "SELECT source, count(*) FILTER (WHERE is_t), count(*) FILTER (WHERE is_d), "
        "count(*) FILTER (WHERE is_e), "
        "count(DISTINCT (peptidoform, charge)) FILTER (WHERE is_t), "
        "count(DISTINCT base_peptide_id) FILTER (WHERE is_t), "
        "count(DISTINCT protein_group) FILTER (WHERE is_t AND protein_group <> '') "
        "FROM x GROUP BY source ORDER BY source"
    )
    found = {
        int(r[0]): [int(v) for v in r[1:]]
        for r in _execute(rs, sql, {**cls.params, "path": _path(rs), "t": t}).fetchall()
    }
    names = [
        "target_psms",
        "decoy_psms",
        "spike_in_psms",
        "precursors",
        "peptides",
        "protein_groups",
    ]
    records = []
    for source in _sources(rs, found):
        values = found.get(source, [0] * len(names))
        records.append(
            {
                "run": _run_label(rs, source),
                "source": source,
                **dict(zip(names, values, strict=True)),
            }
        )
    df = pd.DataFrame(records, columns=["run", "source", *names])
    if not cls.spike_present:
        df = df.drop(columns=["spike_in_psms"])
    tt = format_threshold(t)
    target_noun = "real target" if cls.excludes_spike_ins else "target"
    within = "PSM-level FDR within the run"
    real = _real_qualifier(cls)
    labels = {
        "target_psms": f"{target_noun} PSMs (rows, run_psm_q <= {tt}){real}",
        "decoy_psms": f"decoy PSMs (rows, run_psm_q <= {tt}); a diagnostic, not an FDR",
        "spike_in_psms": f"spike-in PSMs (rows, run_psm_q <= {tt})",
        "precursors": f"precursors with a {target_noun} PSM at run_psm_q <= {tt} (unique "
        f"(peptidoform, charge); {within}, not precursor-level FDR){real}",
        "peptides": f"peptides with a {target_noun} PSM at run_psm_q <= {tt} (unique "
        f"base_peptide_id; {within}, not peptide-level FDR){real}",
        "protein_groups": f"protein groups with a {target_noun} PSM at run_psm_q <= {tt} "
        f"(unique non-empty protein_group; {within}, not protein-level FDR){real}",
    }
    notes = [
        "Per-run counts use run_psm_q. The grouped q columns (precursor_q, peptide_q_value, "
        "pg_q_value) are experiment-wide and are never counted per run."
    ]
    if mbr_ran(rs):
        notes.append(
            "Native identifications from scored_combined.parquet; match-between-runs "
            "transfers are counted separately (mbr.transfer_counts)."
        )
    if cls.note:
        notes.append(cls.note)
    df.attrs.update(
        {
            "unit": "run_psm",
            "q_column": "run_psm_q",
            "threshold": t,
            "population": _population(rs),
            "labels": {c: labels[c] for c in df.columns if c in labels},
            "derived": {
                c: c in ("precursors", "peptides", "protein_groups")
                for c in df.columns
                if c in labels
            },
            "note": " ".join(notes),
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


# --------------------------------------------------------------------------- engine numbers


def engine_stats(rs: ResultSet) -> dict[str, Any]:
    """The rescore report's own counts: ``target_*_at_1pct`` and every ``decoy_*`` and
    ``entrapment_*`` key of ``stats`` (empty without a report). The threshold of these
    statistics is the literal 0.01, whatever ``quant.q_threshold`` is."""
    report = rs.scored.report
    if report is None:
        return {}
    out: dict[str, Any] = {}
    for k, v in report.stats.items():
        if (k.startswith("target_") and k.endswith("_at_1pct")) or k.startswith(
            ("decoy_", "entrapment_")
        ):
            out[k] = v
    return out


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number == int(number) else None


def engine_check(rs: ResultSet) -> list[EngineCheck]:
    """The engine's counts at 0.01 against the viewer's counts of the same columns.

    One entry per unit (``psm``, ``precursor``, ``peptide``, ``protein_group``); in
    entrapment mode also ``entrapment_peptides`` (spike-in base peptides at
    ``peptide_q_value <= 0.01``). The protein-group entry counts an empty
    ``protein_group`` too, as the engine does. ``engine`` is None when the report lacks
    the statistic; ``equal`` is then None.

    The report has statistics at 0.01 only, so this check says nothing about other
    thresholds. When the run records no marker strings, the viewer's classes can
    differ from the engine's at a higher threshold while this check is all equal. In
    entrapment mode :attr:`.entrapment.FdpRow.matches_q` checks each threshold.
    """
    stats = engine_stats(rs)
    counts = {c.unit: c for c in unit_counts(rs, ENGINE_THRESHOLD)}
    cls = count_classes(rs)
    raw = _unit_aggregates(rs, ENGINE_THRESHOLD, cls)
    out: list[EngineCheck] = []
    for k in COUNT_UNITS:
        viewer = counts[k].n_target
        if k == "protein_group":
            viewer += raw["protein_group_t_empty"]
        engine = _as_int(stats.get(ENGINE_STAT_KEYS[k]))
        out.append(EngineCheck(k, engine, viewer, None if engine is None else engine == viewer))
    if cls.mode == "entrapment" or "entrapment_peptides_at_1pct" in stats:
        engine = _as_int(stats.get("entrapment_peptides_at_1pct"))
        viewer = counts["peptide"].n_spike_in or 0
        out.append(
            EngineCheck(
                "entrapment_peptides", engine, viewer, None if engine is None else engine == viewer
            )
        )
    return out


# --------------------------------------------------------------------------- curves


def _label_predicate(cls: CountClasses, label: str) -> str:
    if label == "target":
        return cls.target
    if label == "decoy":
        return cls.decoy
    if label == "spike_in":
        if not cls.spike_present:
            raise ValueError(
                f"no target protein contains the entrapment marker '{cls.settings.marker}', "
                "so there are no spike-ins to count."
            )
        return cls.spike
    raise ValueError(f"label must be 'target', 'decoy' or 'spike_in', not {label!r}.")


def id_curve(
    rs: ResultSet,
    unit_key: str,
    *,
    q_max: float = 0.05,
    points: int = 200,
    label: str = "target",
    per_run: bool = False,
) -> pd.DataFrame:
    """The count of a unit at ``points`` thresholds ``q_max / points, ..., q_max``.

    Columns ``q`` (the threshold) and ``count``; with ``per_run=True`` also ``run`` and
    ``source``. The count at ``q`` is the number of rows (or distinct keys) whose value
    of the engine's column is at most ``q``: these are counts, not recomputed q values.
    One GROUP BY on ``ceil(q / step)`` with an exact correction at the bin edges, then
    a cumulative sum, gives every point.

    ``per_run=True`` needs the per-run unit ``run_psm`` (``run_psm_q``); the grouped
    columns are experiment-wide. ``label`` is ``'target'`` (real targets in entrapment
    mode), ``'decoy'`` or ``'spike_in'``.
    """
    u = get_unit(unit_key)
    q_max = check_threshold(u, q_max)
    if isinstance(points, bool) or int(points) != points or int(points) < 1:
        raise ValueError(f"points must be a positive integer, not {points!r}.")
    points = int(points)
    if per_run and not u.per_run:
        raise ValueError(
            f"{u.q_column} is not computed per run; a per-run curve uses the unit 'run_psm' "
            "(run_psm_q). The grouped q columns are experiment-wide."
        )
    if u.per_run and not per_run and len(rs.runs) > 1:
        raise ValueError(
            "run_psm_q is computed within each run, so its curve is per run: pass per_run=True."
        )
    cls = count_classes(rs)
    predicate = _label_predicate(cls, label)
    key = ("id_curve", rs.scored.identity(), u.key, q_max, points, label, per_run, cls.key)
    if key in rs._memo:
        return rs._memo[key].copy()
    step = q_max / points
    thresholds = np.arange(1, points + 1, dtype=np.float64) * step
    thresholds[-1] = q_max
    group = "source, " if per_run else ""
    where = u.where_sql(f"{predicate} AND {u.q_column} <= $qmax")
    if u.key_sql is None:
        inner = f"SELECT {group}{u.q_column} AS q FROM read_parquet($path) WHERE {where}"
    else:
        # One value per key: its smallest q, so that a key counts once (COUNT DISTINCT).
        inner = (
            f"SELECT {group}min({u.q_column}) AS q FROM read_parquet($path) WHERE {where} "
            f"GROUP BY {group}{', '.join(u.distinct)}"
        )
    sql = (
        f"WITH x AS ({inner}), "
        f"b AS (SELECT {group}q, least(greatest(CAST(ceil(q / $step) AS BIGINT), 1), $n) AS k0 "
        "FROM x) "
        f"SELECT {group}CASE WHEN k0 > 1 AND q <= (k0 - 1) * $step THEN k0 - 1 "
        "WHEN q > (CASE WHEN k0 >= $n THEN $qmax ELSE k0 * $step END) THEN k0 + 1 "
        "ELSE k0 END AS k, count(*) AS n FROM b GROUP BY ALL"
    )
    params = {**cls.params, "path": _path(rs), "qmax": q_max, "step": step, "n": points}
    rows = _execute(rs, sql, params).fetchall()

    def curve(pairs: list[tuple[int, int]]) -> np.ndarray:
        per_bin = np.zeros(points, dtype=np.int64)
        for k, n in pairs:
            per_bin[int(k) - 1] += int(n)
        return np.cumsum(per_bin)

    if per_run:
        by_source: dict[int, list[tuple[int, int]]] = {}
        for source, k, n in rows:
            by_source.setdefault(int(source), []).append((k, n))
        frames = []
        for source in _sources(rs, by_source):
            frames.append(
                pd.DataFrame(
                    {
                        "q": thresholds,
                        "count": curve(by_source.get(source, [])),
                        "run": _run_label(rs, source),
                        "source": source,
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame({"q": thresholds, "count": curve([(k, n) for k, n in rows])})
    noun = {"target": "target", "decoy": "decoy", "spike_in": "spike-in"}[label]
    if label == "target" and cls.excludes_spike_ins:
        noun = "real target"
    note = "Counts of the engine's column at each threshold; no q value is recomputed."
    if u.sparse:
        note += (
            f" {u.q_column} is 1.0 on every row that is not its group's winner, so the curve "
            "is defined only below q = 1."
        )
    if label in ("target", "spike_in") and cls.note:
        note += " " + cls.note
    df.attrs.update(
        {
            "unit": u.key,
            "q_column": u.q_column,
            "label": label,
            "per_run": per_run,
            "title": f"{noun} {u.plural} ({u.distinct_label}) with {u.q_column} <= q",
            "note": note,
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


def counts_at(
    rs: ResultSet,
    unit_key: str,
    thresholds: Iterable[float],
    *,
    label: str = "target",
) -> pd.DataFrame:
    """The count of a unit at each of ``thresholds`` (any spacing, for example a log grid).

    Columns ``q`` (the sorted, distinct thresholds) and ``count``. The count at ``q`` is
    the :func:`unit_counts` count at ``q``: the rows, or the distinct keys, whose value
    of the engine's column is at most ``q``. One scan computes the smallest q of each key
    below the largest threshold, then one filtered count per threshold. No q value is
    recomputed.
    """
    u = get_unit(unit_key)
    qs = sorted({check_threshold(u, float(q)) for q in thresholds})
    if not qs:
        raise ValueError("thresholds must not be empty.")
    if u.per_run and len(rs.runs) > 1:
        raise ValueError(
            "run_psm_q is computed within each run, so a pooled count of it has no meaning; "
            "use id_curve(..., per_run=True)."
        )
    cls = count_classes(rs)
    predicate = _label_predicate(cls, label)
    key = ("counts_at", rs.scored.identity(), u.key, tuple(qs), label, cls.key)
    if key in rs._memo:
        return rs._memo[key].copy()
    condition = f"{predicate} AND {u.q_column} <= $qmax"
    if len(u.distinct) == 1:
        # count(DISTINCT key) skips a NULL key; the GROUP BY below must too.
        condition += f" AND {u.distinct[0]} IS NOT NULL"
    where = u.where_sql(condition)
    if u.key_sql is None:
        inner = f"SELECT {u.q_column} AS q FROM read_parquet($path) WHERE {where}"
    else:
        inner = (
            f"SELECT min({u.q_column}) AS q FROM read_parquet($path) WHERE {where} "
            f"GROUP BY {', '.join(u.distinct)}"
        )
    filters = ", ".join(f"count(*) FILTER (WHERE q <= $t{i})" for i in range(len(qs)))
    sql = f"WITH x AS ({inner}) SELECT {filters} FROM x"
    params: dict[str, Any] = {**cls.params, "path": _path(rs), "qmax": qs[-1]}
    params.update({f"t{i}": q for i, q in enumerate(qs)})
    row = _execute(rs, sql, params).fetchone()
    assert row is not None
    df = pd.DataFrame({"q": qs, "count": [int(n) for n in row]})
    df.attrs.update(
        {
            "unit": u.key,
            "q_column": u.q_column,
            "label": label,
            "note": "Counts of the engine's column at each threshold; no q value is recomputed.",
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


# --------------------------------------------------------------------------- histograms


def _score_range(rs: ResultSet) -> tuple[float, float, str]:
    handle = rs.scored.parquet()
    stats = handle.column_statistics("score")
    lo = hi = None
    source = "footer statistics"
    try:
        if stats and all(s is not None for s in stats):
            lo = min(float(s[0]) for s in stats)  # type: ignore[index]
            hi = max(float(s[1]) for s in stats)  # type: ignore[index]
    except (TypeError, ValueError):
        lo = hi = None
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        row = _execute(
            rs,
            "SELECT min(score), max(score) FROM read_parquet($path) WHERE isfinite(score)",
            {"path": _path(rs)},
        ).fetchone()
        lo, hi = (row[0], row[1]) if row is not None else (None, None)
        source = "a scan of the score column"
        if lo is None or hi is None:
            return 0.0, 1.0, "no finite score; a placeholder range"
    return float(lo), float(hi), source


def score_histogram(
    rs: ResultSet, bins: int = 100, *, run: int | str | None = None
) -> pd.DataFrame:
    """Row counts of the rescorer ``score`` per class in ``bins`` equal bins.

    Columns ``bin_lo``, ``bin_hi``, ``target`` and ``decoy``, plus ``spike_in`` when the
    library has entrapment markers (then ``target`` holds the real targets). The range
    is the table's score range, from the footer statistics when every row group has
    them, so the bins of different runs line up. ``run`` restricts the rows to one run
    (``source``) of the pooled table.
    """
    if isinstance(bins, bool) or int(bins) != bins or int(bins) < 1:
        raise ValueError(f"bins must be a positive integer, not {bins!r}.")
    bins = int(bins)
    source = None if run is None else rs.run(run_key(run)).index
    cls = count_classes(rs)
    key = ("score_histogram", rs.scored.identity(), bins, source, cls.key)
    if key in rs._memo:
        return rs._memo[key].copy()
    lo, hi, range_source = _score_range(rs)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    target = cls.real if cls.spike_present else cls.target
    where = "isfinite(score)" + (" AND source = $source" if source is not None else "")
    params: dict[str, Any] = {
        **cls.params,
        "path": _path(rs),
        "lo": lo,
        "w": width,
        "nb": bins,
    }
    if source is not None:
        params["source"] = source
    sql = (
        "SELECT least(greatest(CAST(floor((score - $lo) / $w) AS BIGINT), 0), $nb - 1) AS b, "
        f"count(*) FILTER (WHERE {target}), count(*) FILTER (WHERE {cls.decoy}), "
        f"count(*) FILTER (WHERE {cls.spike}) FROM read_parquet($path) WHERE {where} "
        "GROUP BY b ORDER BY b"
    )
    counts = np.zeros((bins, 3), dtype=np.int64)
    for b, n_t, n_d, n_e in _execute(rs, sql, params).fetchall():
        counts[int(b)] += (int(n_t), int(n_d), int(n_e))
    edges = lo + np.arange(bins + 1, dtype=np.float64) * width
    edges[-1] = hi
    df = pd.DataFrame(
        {
            "bin_lo": edges[:-1],
            "bin_hi": edges[1:],
            "target": counts[:, 0],
            "decoy": counts[:, 1],
        }
    )
    if cls.spike_present:
        df["spike_in"] = counts[:, 2]
    info = rescore_info(rs)
    df.attrs.update(
        {
            "run": None if source is None else _run_label(rs, source),
            "range": (lo, hi),
            "range_source": range_source,
            "classifier": info.classifier,
            "labels": {
                "target": "real target rows (spike-ins in their own column)"
                if cls.spike_present
                else "target rows",
                "decoy": "decoy rows",
                "spike_in": "spike-in rows (entrapment targets)",
            },
            "note": (
                f"score is the discriminant of the rescorer {info.classifier or 'unknown'}; "
                "higher is better and its scale depends on the rescorer. The last bin "
                "includes its upper edge."
                + (f" {cls.note}" if cls.spike_present and cls.note else "")
            ),
            "sql": sql,
        }
    )
    rs._memo[key] = df.copy()
    return df


# --------------------------------------------------------------------------- winners


def _grouped_unit(unit_key: Unit | str) -> Unit:
    u = get_unit(unit_key)
    if not u.sparse:
        raise ValueError(
            f"the unit {u.key!r} counts rows; only the grouped units (precursor, peptide, "
            "protein_group) have group winners."
        )
    return u


def _key_filter(u: Unit, keys: list[Any]) -> tuple[str, dict[str, Any]]:
    """A semi-join on the group key: ``key IN (SELECT unnest($keys))``.

    DuckDB runs it as a hash join, so its cost hardly grows with the number of keys
    (0.14 s for 10,000 keys on 3.86 million rows, where ``list_contains`` took 5 to 9 s).
    The lists are cast to a type, so an empty list binds too.
    """
    if u.key == "peptide":
        return (
            "CAST(base_peptide_id AS BIGINT) IN (SELECT unnest(CAST($keys AS BIGINT[])))",
            {"keys": [int(v) for v in keys]},
        )
    if u.key == "protein_group":
        return (
            "protein IN (SELECT unnest(CAST($keys AS VARCHAR[])))",
            {"keys": [str(v) for v in keys]},
        )
    pairs = [(str(p), int(c)) for p, c in keys]
    return (
        "(peptidoform, CAST(charge AS BIGINT)) IN (SELECT (unnest(CAST($key_peptidoforms "
        "AS VARCHAR[])), unnest(CAST($key_charges AS BIGINT[]))))",
        {"key_peptidoforms": [p for p, _ in pairs], "key_charges": [c for _, c in pairs]},
    )


def group_winner_sql(
    rs: ResultSet, unit_key: str, *, keys: Iterable[Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """SQL (and parameters) that returns the winning row of each group of a grouped column.

    The engine's rule (``stages/rescore.rs`` ``grouped_q``): per key the highest
    ``score`` wins; on an exact tie a decoy replaces a target; otherwise the first row
    in file order stays. In entrapment mode decoys do not compete at all, a spike-in
    replaces a real target on a tie, and a group of decoys only has no winner. This is
    for labelling a group's winner on an evidence page; the counts never use it.

    ``keys`` restricts the groups: base_peptide_id values, protein strings or
    ``(peptidoform, charge)`` pairs. Without ``keys`` the SQL covers every group of the
    table. Use that form inside DuckDB (a join, an aggregate or a streamed read). Do
    not load its result whole: it has one row per group, 1.79 million precursors in the
    six-run EXP experiment.
    """
    u = _grouped_unit(unit_key)
    cls = count_classes(rs)
    partition = ", ".join(_WINNER_PARTITION[u.key])
    columns = (
        "file_row_number, source, candidate_id, peptidoform, charge, label, protein, "
        f"protein_group, base_peptide_id, score, {u.q_column}"
    )
    key_where, params = _key_filter(u, list(keys)) if keys is not None else ("", {})
    params["path"] = _path(rs)
    if cls.mode == "entrapment":
        expr, ent_params = entrapment_expr(cls.settings) if cls.spike_present else ("false", {})
        params.update(ent_params)
        where = "label <> 'decoy'" + (f" AND {key_where}" if key_where else "")
        source = (
            f"(SELECT {columns}, {expr} AS is_ent FROM read_parquet($path, "
            f"file_row_number = true) WHERE {where})"
        )
        order = "score DESC, is_ent DESC, file_row_number"
    else:
        where = f" WHERE {key_where}" if key_where else ""
        source = f"(SELECT {columns} FROM read_parquet($path, file_row_number = true){where})"
        order = "score DESC, (label = 'decoy') DESC, file_row_number"
    sql = (
        f"SELECT * EXCLUDE (rk) FROM (SELECT {columns}, row_number() OVER "
        f"(PARTITION BY {partition} ORDER BY {order}) AS rk FROM {source}) WHERE rk = 1"
    )
    return sql, params


def group_winners(rs: ResultSet, unit_key: str, *, keys: Iterable[Any]) -> pd.DataFrame:
    """The engine's winning row of the given groups of a grouped column.

    ``keys`` is required: the base_peptide_id values, protein strings or
    ``(peptidoform, charge)`` pairs of the groups to label, at most
    :data:`MAX_WINNER_KEYS`. The winners of every group of a table do not fit in memory
    at scale (one row per group), so this function refuses to build them. For the
    whole table use :func:`group_winner_sql`, which returns SQL to run inside DuckDB.
    The rule is described there.

    One row per group that has a winner, sorted by file row, with ``run`` added.
    ``attrs['note']`` warns when the table holds non-finite scores, where DuckDB's sort
    order can differ from the engine's.
    """
    u = _grouped_unit(unit_key)
    if keys is None:
        raise ValueError(
            "group_winners needs the keys of the groups to label (base_peptide_id values, "
            "protein strings or (peptidoform, charge) pairs). The winners of every group do "
            "not fit in memory on a large table; use group_winner_sql for SQL that DuckDB "
            "runs without loading the result."
        )
    if isinstance(keys, str | bytes | Mapping):
        raise ValueError(
            "keys must be a collection of keys, not a single string or a mapping; "
            "pass [key] for one group."
        )
    values = list(keys)
    if len(values) > MAX_WINNER_KEYS:
        raise ValueError(
            f"group_winners takes at most {MAX_WINNER_KEYS:,} keys (got {len(values):,}), "
            "because it returns a DataFrame. Use group_winner_sql for more groups; it "
            "returns SQL that DuckDB runs without loading the result."
        )
    sql, params = group_winner_sql(rs, u.key, keys=values)
    df = _execute(rs, sql + " ORDER BY file_row_number", params).df()
    df.insert(1, "run", [_run_label(rs, int(s)) for s in df["source"]])
    nonfinite_key = ("nonfinite_scores", rs.scored.identity())
    if nonfinite_key not in rs._memo:
        row = _execute(
            rs,
            "SELECT count(*) FROM read_parquet($path) WHERE NOT isfinite(score)",
            {"path": _path(rs)},
        ).fetchone()
        rs._memo[nonfinite_key] = int(row[0]) if row is not None else 0
    nonfinite = rs._memo[nonfinite_key]
    cls = count_classes(rs)
    note = "The engine's winner rule applied by the viewer, for labelling only. " + (
        "Entrapment mode: decoys do not compete, and a spike-in wins an exact tie."
        if cls.mode == "entrapment"
        else "A decoy wins an exact score tie; otherwise the first row in file order."
    )
    if nonfinite:
        note += (
            f" {nonfinite:,} rows have a non-finite score; the winner of their groups is "
            "derived and may differ from the engine's."
        )
    df.attrs.update({"unit": u.key, "q_column": u.q_column, "note": note, "sql": sql})
    return df
