"""Entrapment: spike-in classification and the entrapment false discovery proportion.

Some benchmark libraries mark entrapment proteins, from a proteome that is absent
from the sample. A target whose protein string carries the marker is a spike-in; its
identification is false by construction. The entrapment false discovery proportion
(FDP) at a q threshold is

    FDP = (r * E + 1) / max(1, R)

with ``E`` the accepted spike-ins, ``R`` the accepted real targets, both counted on one
unit's key, and ``r`` the library ratio N_real / N_entrapment. It is not capped at 1.
It is computed in float64 in the engine's order of operations
(``fdr.rs``: ``(ratio * ne as f64 + 1.0) / (nr.max(1) as f64)``).

Classification follows the engine (``stages/rescore.rs`` ``classify_entrapment``): a
decoy (by ``label``) is neither class. A target is a spike-in when its protein string
contains the marker, does not contain the exclusion string (when one is set) and
contains none of the contaminant tokens. Every other target is real. The tests are
case-sensitive substring tests on the whole ``protein`` string. The label is tested
first because decoy protein strings keep the marker text (``DECOY_sp|ENTRAP_...``).

What the FDP means depends on how the run was rescored (:func:`rescore.rescore_info`)
and on the settings:

* entrapment mode (``entrapment_gbm`` or ``entrapment_native``): every q column is this
  same estimate. With the engine's own markers and r, the FDP at a cut restates the q
  column: it equals the largest accepted q of that unit, bit for bit, and is the
  engine's own estimate, not a check. With other markers or another r it is a viewer
  computation that does not equal the q column. Every row tests this
  (``FdpRow.matches_q``), and its label says which of the two it is.
* target-decoy mode: the q columns come from decoys, and the FDP is an independent
  check of them.

The engine reports its estimate at the peptide unit (``entrapment_peptides_at_1pct``
beside ``target_peptides_at_1pct``), so the peptide unit is the headline
(:data:`HEADLINE_UNIT`). Per-run FDPs use the PSM unit on ``run_psm_q``; the grouped q
columns are experiment-wide and are never used per run.

:func:`count_classes` gives the SQL predicates of the classes that the identification
counts use (:mod:`.counts`, :mod:`.mbr`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .discovery import ResultSet
from .duck import sql_path
from .rescore import rescore_info
from .units import COUNT_UNITS, UNITS, Unit, check_threshold, execute_bound, format_threshold

__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_MARKER",
    "DEFAULT_RATIO",
    "HEADLINE_UNIT",
    "CountClasses",
    "EntrapmentSettings",
    "FdpRow",
    "class_breakdown",
    "class_sql",
    "count_classes",
    "entrapment_expr",
    "entrapment_fdp",
    "fdp_value",
    "markers_present",
    "markers_recorded",
    "settings_for",
    "sql_literal",
]

DEFAULT_MARKER = "ENTRAP_"
DEFAULT_EXCLUDE = "REAL_"
# N_real / N_entrapment of the group's E. coli / human entrapment library (lib2, built
# from entrap_ecoli_human1to1.fasta: E. coli proteins are real, human proteins are the
# entrapment), at the precursor level: 1,046,870 / 1,867,304 target precursors.
DEFAULT_RATIO = 0.560632
HEADLINE_UNIT = "peptide"

_SOURCE_CONFIG = (
    "config_json rescore.entrapment_marker, entrapment_exclude, "
    "entrapment_contaminant_markers and entrapment_ratio"
)
_SOURCE_REPORT = (
    "r from the rescore report (stats.entrapment_ratio); the marker strings are the "
    "viewer defaults, because the run records none"
)
_SOURCE_DEFAULTS = "viewer defaults; the run records no entrapment settings"
# The source of settings made directly, without settings_for.
_SOURCE_VIEWER_DEFAULTS = "viewer defaults"
_SOURCE_CALLER = "set by the caller"


def _valid_ratio(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and ratio > 0 else None


def _number(value: float) -> str:
    """A float as the shortest text that reads back as the same value (``1`` for 1.0)."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


@dataclass(frozen=True)
class EntrapmentSettings:
    """Marker strings and ratio of an entrapment library, and where they came from.

    The settings are frozen: a changed setting is a new object
    (``dataclasses.replace``). ``source`` names the origin. When it is empty, it
    becomes ``'viewer defaults'`` if every value is a viewer default, else
    ``'set by the caller'``. A ``'viewer defaults'`` source on values that are not the
    defaults is corrected in the same way.
    """

    marker: str = DEFAULT_MARKER
    exclude: str | None = DEFAULT_EXCLUDE
    contaminants: tuple[str, ...] = ()
    ratio: float = DEFAULT_RATIO
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.marker, str) or not self.marker:
            raise ValueError("the entrapment marker must be a non-empty string.")
        if self.exclude is not None and not isinstance(self.exclude, str):
            raise ValueError(
                f"the entrapment exclusion must be a string or None, not {self.exclude!r}."
            )
        if self.exclude == "":
            object.__setattr__(self, "exclude", None)
        tokens = self.contaminants
        if isinstance(tokens, str):
            tokens = (tokens,)
        object.__setattr__(self, "contaminants", tuple(str(c) for c in (tokens or ()) if str(c)))
        ratio = _valid_ratio(self.ratio)
        if ratio is None:
            raise ValueError(
                f"the entrapment ratio r must be a finite number above 0, not {self.ratio!r}."
            )
        object.__setattr__(self, "ratio", ratio)
        if not self.source or self.source == _SOURCE_VIEWER_DEFAULTS:
            defaults = (DEFAULT_MARKER, DEFAULT_EXCLUDE, (), DEFAULT_RATIO)
            source = _SOURCE_VIEWER_DEFAULTS if self.cache_key() == defaults else _SOURCE_CALLER
            object.__setattr__(self, "source", source)

    def cache_key(self) -> tuple[Any, ...]:
        return (self.marker, self.exclude, self.contaminants, self.ratio)

    @property
    def rule(self) -> str:
        """The spike-in rule in words, for labels and footnotes."""
        text = f"target whose protein contains '{self.marker}'"
        if self.exclude:
            text += f", not '{self.exclude}'"
        if self.contaminants:
            text += ", and none of " + ", ".join(self.contaminants)
        return text + " (the engine's classification test; decoys are excluded by label)"


def _rescore_config(rs: ResultSet) -> dict[str, Any]:
    rescore = rs.config_get("rescore") or {}
    return rescore if isinstance(rescore, dict) else {}


def markers_recorded(rs: ResultSet) -> bool:
    """True when the run's ``config_json`` names its spike-in marker.

    Only then are the marker strings of :func:`settings_for` the run's own
    (``rescore.entrapment_marker`` and the keys beside it). Otherwise they are the
    viewer's defaults, and a count that separates the spike-ins follows the viewer's
    rule, not a rule the run recorded.
    """
    marker = _rescore_config(rs).get("entrapment_marker")
    return isinstance(marker, str) and bool(marker)


def _engine_ratio(rs: ResultSet) -> float | None:
    """The r of the rescore report (``stats.entrapment_ratio``, entrapment mode only)."""
    report = rs.scored.report
    return _valid_ratio(report.stats.get("entrapment_ratio")) if report else None


def settings_for(rs: ResultSet) -> EntrapmentSettings:
    """The entrapment settings of a run, with their origin in ``source``.

    1. ``config_json.rescore`` when ``entrapment_marker`` is set: the run's own marker,
       exclusion, contaminant tokens and ratio. The config keys exist in every run with
       defaults (marker null, ratio 1.0), so a ratio without a marker means nothing.
    2. Otherwise the ratio from the rescore report (``stats.entrapment_ratio``, written
       in entrapment mode only), with the viewer's default markers.
    3. Otherwise the viewer defaults: marker ``ENTRAP_``, exclusion ``REAL_``, no
       contaminant tokens, r = 0.560632.
    """
    rescore = _rescore_config(rs)
    marker = rescore.get("entrapment_marker")
    stats_ratio = _engine_ratio(rs)
    if markers_recorded(rs):
        exclude = rescore.get("entrapment_exclude")
        contaminants = rescore.get("entrapment_contaminant_markers") or ()
        if isinstance(contaminants, str):
            contaminants = (contaminants,)
        ratio = _valid_ratio(rescore.get("entrapment_ratio"))
        source = _SOURCE_CONFIG
        if ratio is None:
            ratio = stats_ratio or DEFAULT_RATIO
            source += (
                "; r from the rescore report"
                if stats_ratio
                else "; r is the viewer default (the run records none)"
            )
        return EntrapmentSettings(
            marker=marker,
            exclude=exclude if isinstance(exclude, str) and exclude else None,
            contaminants=tuple(str(c) for c in contaminants),
            ratio=ratio,
            source=source,
        )
    if stats_ratio is not None:
        return EntrapmentSettings(ratio=stats_ratio, source=_SOURCE_REPORT)
    return EntrapmentSettings(source=_SOURCE_DEFAULTS)


def sql_literal(text: str) -> str:
    """A SQL string literal, for SQL that is displayed, never executed."""
    return "'" + str(text).replace("'", "''") + "'"


def entrapment_expr(
    settings: EntrapmentSettings, *, inline: bool = False
) -> tuple[str, dict[str, Any]]:
    """The spike-in test as a SQL boolean over ``label`` and ``protein``, with its parameters.

    The parameters are named ``ent_*``; every string is bound, never spliced into the
    SQL text. One ``NOT contains`` test per contaminant token is the same rule as the
    engine's "matches none of the tokens", and it needs no list parameter. With
    ``inline=True`` the strings are written as literals and no parameter is returned;
    that form is only for SQL shown to the user.
    """
    values: list[tuple[str, str, bool]] = [("ent_marker", settings.marker, False)]
    if settings.exclude:
        values.append(("ent_exclude", settings.exclude, True))
    values += [(f"ent_c{i}", c, True) for i, c in enumerate(settings.contaminants)]
    parts = ["label = 'target'"]
    params: dict[str, Any] = {}
    for name, value, negate in values:
        arg = sql_literal(value) if inline else f"${name}"
        parts.append(("NOT " if negate else "") + f"contains(protein, {arg})")
        if not inline:
            params[name] = value
    return "(" + " AND ".join(parts) + ")", params


def class_sql(
    settings: EntrapmentSettings,
    source: str = "read_parquet($path)",
    columns: str = "*",
    name: str = "cls",
) -> tuple[str, dict[str, Any]]:
    """A CTE that adds the engine's classes to a scored table.

    Returns ``("cls AS (SELECT <columns>, is_decoy, is_ent, ent_ratio FROM <source>)",
    params)``. ``is_decoy`` is ``label = 'decoy'``; ``is_ent`` marks spike-ins; a real
    target is ``NOT is_decoy AND NOT is_ent``. ``ent_ratio`` is r, bound as a DOUBLE
    parameter: a query that computes the FDP in SQL must use it, because a literal such
    as ``0.560632`` is a DECIMAL in DuckDB and changes the last bits of the result. The
    caller binds ``$path`` (or whatever ``source`` uses) next to the returned
    parameters. DuckDB cannot bind parameters inside ``CREATE VIEW``, so this is a
    per-query CTE.
    """
    expr, params = entrapment_expr(settings)
    params["ent_ratio"] = float(settings.ratio)
    sql = (
        f"{name} AS (SELECT {columns}, label = 'decoy' AS is_decoy, {expr} AS is_ent, "
        f"CAST($ent_ratio AS DOUBLE) AS ent_ratio FROM {source})"
    )
    return sql, params


def markers_present(rs: ResultSet, settings: EntrapmentSettings | None = None) -> bool:
    """True when at least one target protein string contains the marker.

    The entrapment FDP is shown only then: without spike-ins the formula prints only
    the pseudocount floor 1/R, which means nothing.
    """
    settings = settings or settings_for(rs)
    key = ("entrapment_markers_present", rs.scored.identity(), settings.marker)
    if key in rs._memo:
        return bool(rs._memo[key])
    row = execute_bound(
        rs.duck,
        "SELECT count(*) FROM (SELECT 1 FROM read_parquet($path) "
        "WHERE label = 'target' AND contains(protein, $marker) LIMIT 1)",
        {"path": sql_path(rs.scored.require()), "marker": settings.marker},
    ).fetchone()
    found = row[0] if row is not None else 0
    rs._memo[key] = bool(found)
    return bool(found)


@dataclass(frozen=True)
class CountClasses:
    """SQL predicates of the classes that the identification counts use.

    ``target`` is the identification population: in entrapment mode the real targets
    (spike-ins excluded, as in the engine's report statistics), otherwise every target.
    ``real`` is the targets that are not spike-ins, ``decoy`` the decoys and ``spike``
    the spike-ins (``false`` when no target protein contains the marker). The
    predicates read ``label`` and ``protein`` and use the bound parameters in
    ``params``, a read-only mapping. ``markers_recorded`` says whether the marker strings
    are the run's own (:func:`markers_recorded`). ``note`` explains the population for
    count labels.
    """

    mode: str
    target: str
    real: str
    decoy: str
    spike: str
    params: Mapping[str, Any]
    spike_present: bool
    settings: EntrapmentSettings
    markers_recorded: bool
    note: str

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.mode, self.spike_present, self.settings.cache_key())

    @property
    def excludes_spike_ins(self) -> bool:
        """True when ``target`` leaves the spike-ins out (entrapment mode with markers)."""
        return self.mode == "entrapment" and self.spike_present

    @property
    def inline_target(self) -> str:
        """The target predicate with literal strings, for displayed SQL."""
        if self.excludes_spike_ins:
            expr, _ = entrapment_expr(self.settings, inline=True)
            return f"(label = 'target' AND NOT {expr})"
        return "label = 'target'"


def count_classes(rs: ResultSet) -> CountClasses:
    """The class predicates for the counts of a result set (memoised per scored table).

    The settings are :func:`settings_for`. In entrapment mode the target population is
    the real targets. When the run records its marker strings, that is the engine's
    population. When it records none, the real targets follow the viewer's spike-in
    rule, and ``note`` says so.
    """
    key = ("count_classes", rs.scored.identity())
    if key in rs._memo:
        return rs._memo[key]
    info = rescore_info(rs)
    settings = settings_for(rs)
    recorded = markers_recorded(rs)
    present = markers_present(rs, settings)
    expr, params = entrapment_expr(settings) if present else ("false", {})
    real = f"(label = 'target' AND NOT {expr})" if present else "label = 'target'"
    if info.mode == "entrapment":
        target = real
        if present and recorded:
            note = (
                "Entrapment mode: the q columns are entrapment estimates, and 'target' means "
                f"real target. Spike-ins ({settings.rule}) are excluded with the run's recorded "
                "marker strings, as the engine's report statistics exclude them."
            )
        elif present:
            note = (
                "Entrapment mode: the q columns are entrapment estimates, and 'target' means "
                "real target. The run records no marker strings, so the real targets follow "
                f"the viewer's spike-in rule ({settings.rule}; {settings.source}), not a rule "
                "the run recorded. The counts equal the engine's statistics only where this "
                "rule gives the engine's classes; engine_check compares them at 0.01."
            )
        else:
            note = (
                "Entrapment mode, but no target protein contains the marker "
                f"'{settings.marker}' ({settings.source}). The spike-ins cannot be separated, "
                "so these counts include them and differ from the engine's statistics."
            )
    else:
        target = "label = 'target'"
        note = (
            "The library has entrapment markers; in target-decoy mode the target counts "
            "include the spike-ins, as the engine's report statistics do."
            if present
            else ""
        )
    classes = CountClasses(
        mode=info.mode,
        target=target,
        real=real,
        decoy="label = 'decoy'",
        spike=expr if present else "false",
        params=MappingProxyType(dict(params)),
        spike_present=present,
        settings=settings,
        markers_recorded=recorded,
        note=note,
    )
    rs._memo[key] = classes
    return classes


def class_breakdown(rs: ResultSet, settings: EntrapmentSettings | None = None) -> dict[str, Any]:
    """Row counts per class, and the protein strings the contaminant tokens carve out.

    The contaminant tokens are substrings, so they can match proteins that are not
    contaminants (``K1C`` inside ``AK1C2_HUMAN``). The carved-out strings are listed so
    that the user can see such matches. Each call returns a new dict and a new list.
    """
    settings = settings or settings_for(rs)
    key = ("entrapment_breakdown", rs.scored.identity(), settings)
    if key in rs._memo:
        return _breakdown_copy(rs._memo[key])
    cte, params = class_sql(settings, columns="label, protein")
    # The same test without the contaminant tokens. It reuses the bound marker and
    # exclusion parameters ($ent_marker, $ent_exclude) of the CTE.
    loose, _ = entrapment_expr(
        EntrapmentSettings(marker=settings.marker, exclude=settings.exclude, ratio=settings.ratio)
    )
    params["path"] = sql_path(rs.scored.require())
    row = execute_bound(
        rs.duck,
        f"WITH {cte} SELECT count(*) FILTER (WHERE is_decoy), count(*) FILTER (WHERE is_ent), "
        "count(*) FILTER (WHERE NOT is_decoy AND NOT is_ent), "
        f"count(*) FILTER (WHERE {loose} AND NOT is_ent) FROM cls",
        params,
    ).fetchone()
    carved: list[str] = []
    if settings.contaminants and row is not None and row[3]:
        carved = [
            r[0]
            for r in execute_bound(
                rs.duck,
                f"WITH {cte} SELECT DISTINCT protein FROM cls WHERE {loose} AND NOT is_ent "
                "ORDER BY protein",
                params,
            ).fetchall()
        ]
    out = {
        "decoy_rows": int(row[0]) if row else 0,
        "spike_in_rows": int(row[1]) if row else 0,
        "real_rows": int(row[2]) if row else 0,
        "carved_out_rows": int(row[3]) if row else 0,
        "carved_out_proteins": tuple(carved),
        "rule": settings.rule,
        "source": settings.source,
    }
    rs._memo[key] = out
    return _breakdown_copy(out)


def _breakdown_copy(memo: dict[str, Any]) -> dict[str, Any]:
    """A caller's copy of a memoised breakdown: a new dict with a new list."""
    out = dict(memo)
    out["carved_out_proteins"] = list(memo["carved_out_proteins"])
    return out


def fdp_value(ratio: float, spike_ins: int, real: int) -> float:
    """``(r * E + 1) / max(1, R)`` in float64, in the engine's order of operations."""
    return (float(ratio) * float(spike_ins) + 1.0) / float(max(int(real), 1))


@dataclass(frozen=True)
class FdpRow:
    """The entrapment FDP of one unit at one threshold (frozen).

    ``largest_accepted_q`` is the largest value of the unit's q column among the
    accepted non-decoy rows (None when nothing is accepted). In entrapment mode
    ``matches_q`` says whether the FDP equals it bit for bit. It does when the markers
    and r are the engine's, and the label then calls the FDP the engine's own estimate.
    When it does not, the FDP is a viewer computation with other settings, and the
    label says so. ``matches_q`` is None in target-decoy mode and when nothing is
    accepted. ``settings_source`` says where the markers and r came from. ``run`` and
    ``source`` are set on the per-run PSM rows.
    """

    unit: str
    threshold: float
    spike_ins: int
    real: int
    ratio: float
    fdp: float
    largest_accepted_q: float | None
    mode: str
    label: str
    q_column: str = ""
    run: str | None = None
    source: int | None = None
    matches_q: bool | None = None
    settings_source: str = ""


@dataclass(frozen=True)
class _LabelContext:
    """What every FDP label of one call needs to know about the settings."""

    entrapment_mode: bool
    rule: str
    provenance: str
    mismatch_cause: str


def _provenance(settings: EntrapmentSettings, run: EntrapmentSettings) -> str:
    """Where the settings came from, checked against the viewer's settings for the run."""
    if settings.cache_key() == run.cache_key():
        return run.source
    if settings.source == run.source:
        # A copy of the run's settings with a changed value keeps the run's source text.
        return f"{_SOURCE_CALLER}; they differ from the viewer's settings for this run"
    return settings.source


def _mismatch_cause(rs: ResultSet, settings: EntrapmentSettings, run: EntrapmentSettings) -> str:
    """Why an entrapment-mode FDP does not restate the q column, as far as it is known."""
    if markers_recorded(rs):
        if settings.cache_key() != run.cache_key():
            return "these settings differ from the run's recorded settings"
        return (
            "the viewer's counts differ from the engine's at this cut, although these are "
            "the run's recorded settings"
        )
    if settings.ratio == _engine_ratio(rs):
        return (
            "the run records no marker strings, and these markers classify some accepted "
            "rows differently from the engine"
        )
    return "the run records no marker strings, and these markers or this r are not the engine's"


def _max_expr(unit: Unit, cond: str) -> str:
    return f"max({unit.q_column}) FILTER (WHERE {unit.where_sql(cond)})"


def _fdp_label(
    unit: Unit,
    t: float,
    e: int,
    r: int,
    ratio: float,
    fdp: float,
    largest: float | None,
    matches_q: bool | None,
    ctx: _LabelContext,
    scope: str | None = None,
) -> str:
    """The label of one FDP row: what the number is, its formula and its settings.

    "The engine's own estimate" is written only in entrapment mode, and only when the
    FDP equals the largest accepted q bit for bit (``matches_q``).
    """
    q = unit.q_column
    where = f"{unit.distinct_label}, {q} <= {format_threshold(t)}"
    if scope is not None:
        where += f"; {scope}"
    formula = f"({_number(ratio)} x {e:,} spike-in + 1) / {r:,} real {unit.noun(r)} ({where})"
    if largest is None:
        kind = "viewer-computed; no spike-in and no real target passes this cut"
        if ctx.entrapment_mode:
            kind += f", so there is no accepted {q} to restate"
    elif ctx.entrapment_mode and matches_q:
        kind = (
            f"the engine's own estimate at the cut: it restates {q} and is not an "
            f"independent check; largest accepted {q} = {largest:.6g}"
        )
    elif ctx.entrapment_mode:
        kind = (
            f"viewer-computed with these settings, not the engine's estimate: it does not "
            f"restate {q} (largest accepted {q} = {largest:.6g}) because {ctx.mismatch_cause}"
        )
    else:
        kind = "viewer-computed: an independent check of the target-decoy q values"
    text = (
        f"Entrapment FDP {fdp * 100:.3f}% ({kind}): {formula}. Spike-in: {ctx.rule}; "
        f"r = {_number(ratio)}; settings: {ctx.provenance}."
    )
    if largest is None:
        text += " The value is only the pseudocount 1 / max(1, R); it estimates nothing."
    elif e == 0:
        text += " No spike-in passes, so the FDP is the pseudocount floor 1/R."
    return text


def entrapment_fdp(
    rs: ResultSet, t: float = 0.01, settings: EntrapmentSettings | None = None
) -> list[FdpRow]:
    """The entrapment FDP at ``t`` for the four units, then one PSM row per run.

    Units: PSM (rows, ``q_value``), precursor (unique ``(peptidoform, charge)``,
    ``precursor_q``), peptide (unique ``base_peptide_id``, ``peptide_q_value``) and
    protein group (unique non-empty ``protein_group``, ``pg_q_value``), counted on the
    pooled scored table (``scored_combined`` in an experiment, never the MBR table).
    The per-run rows count PSMs on ``run_psm_q`` per ``source``.

    ``settings`` defaults to :func:`settings_for`. In entrapment mode a row is labelled
    the engine's own estimate only when its FDP equals the largest accepted q bit for
    bit; otherwise it is labelled a viewer computation with these settings.

    Returns an empty list when no target protein contains the marker.
    """
    run_settings = settings_for(rs)
    settings = settings or run_settings
    t = float(t)
    for key in COUNT_UNITS:
        check_threshold(key, t)
    if not markers_present(rs, settings):
        return []
    info = rescore_info(rs)
    entrapment_mode = info.mode == "entrapment"
    memo_key = ("entrapment_fdp", rs.scored.identity(), t, settings, info.mode)
    if memo_key in rs._memo:
        return list(rs._memo[memo_key])
    ctx = _LabelContext(
        entrapment_mode=entrapment_mode,
        rule=settings.rule,
        provenance=_provenance(settings, run_settings),
        mismatch_cause=_mismatch_cause(rs, settings, run_settings) if entrapment_mode else "",
    )

    columns = (
        "label, protein, source, peptidoform, charge, base_peptide_id, protein_group, "
        "q_value, precursor_q, peptide_q_value, pg_q_value, run_psm_q"
    )
    cte, params = class_sql(settings, columns=columns)
    params.update({"path": sql_path(rs.scored.require()), "t": t})
    selects = []
    for key in COUNT_UNITS:
        u = UNITS[key]
        q = u.q_column
        selects += [
            u.count_sql(f"is_ent AND {q} <= $t"),
            u.count_sql(f"NOT is_decoy AND NOT is_ent AND {q} <= $t"),
            _max_expr(u, f"NOT is_decoy AND {q} <= $t"),
        ]
    row = execute_bound(
        rs.duck, f"WITH {cte} SELECT {', '.join(selects)} FROM cls", params
    ).fetchone()
    assert row is not None
    out: list[FdpRow] = []
    for i, key in enumerate(COUNT_UNITS):
        u = UNITS[key]
        e, r, largest = int(row[3 * i]), int(row[3 * i + 1]), row[3 * i + 2]
        out.append(_row(u, t, e, r, largest, settings.ratio, ctx))

    run_unit = UNITS["run_psm"]
    per_run = execute_bound(
        rs.duck,
        f"WITH {cte} SELECT source, "
        f"{run_unit.count_sql('is_ent AND run_psm_q <= $t')}, "
        f"{run_unit.count_sql('NOT is_decoy AND NOT is_ent AND run_psm_q <= $t')}, "
        f"{_max_expr(run_unit, 'NOT is_decoy AND run_psm_q <= $t')} "
        "FROM cls GROUP BY source ORDER BY source",
        params,
    ).fetchall()
    for source, e, r, largest in per_run:
        name = _run_name(rs, int(source))
        out.append(
            _row(
                run_unit,
                t,
                int(e),
                int(r),
                largest,
                settings.ratio,
                ctx,
                run=name,
                source=int(source),
                scope=f"run {name}" if rs.is_experiment else "within the run",
            )
        )
    rs._memo[memo_key] = tuple(out)
    return list(out)


def _row(
    unit: Unit,
    t: float,
    e: int,
    r: int,
    largest: float | None,
    ratio: float,
    ctx: _LabelContext,
    *,
    run: str | None = None,
    source: int | None = None,
    scope: str | None = None,
) -> FdpRow:
    fdp = fdp_value(ratio, e, r)
    largest_q = float(largest) if largest is not None else None
    matches_q = (fdp == largest_q) if ctx.entrapment_mode and largest_q is not None else None
    return FdpRow(
        unit=unit.key,
        threshold=t,
        spike_ins=e,
        real=r,
        ratio=ratio,
        fdp=fdp,
        largest_accepted_q=largest_q,
        mode="entrapment" if ctx.entrapment_mode else "target_decoy",
        label=_fdp_label(unit, t, e, r, ratio, fdp, largest_q, matches_q, ctx, scope),
        q_column=unit.q_column,
        run=run,
        source=source,
        matches_q=matches_q,
        settings_source=ctx.provenance,
    )


def _run_name(rs: ResultSet, source: int) -> str:
    try:
        return rs.run(source).label
    except KeyError:
        return f"source {source}"
