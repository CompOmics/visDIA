"""Identification units: what a count counts, and on which q column.

A count of identifications is only meaningful as a triple: the row unit, the key that
is counted once, and the engine's q column that decides acceptance. The seven q
columns of the scored table are not interchangeable (docs/15, "q-value column units"):

* ``q_value`` is a PSM-level q over all rows of the rescore; in an experiment it is
  pooled over the runs. ``run_psm_q`` is the same estimate within one run (``source``).
  Both are dense: every row carries its own value.
* ``precursor_q``, ``peptide_q_value`` and ``pg_q_value`` are computed on one winning
  row per group (per ``(peptidoform, charge)``, ``base_peptide_id`` and protein string).
  Only the winner carries the group q; every other row carries 1.0. In an experiment
  the groups span all runs, so these columns are experiment-wide and must never be
  counted per run.

Every label produced here names the unit, the key and the q column, for example
``81,312 peptides (unique base_peptide_id, peptide_q_value <= 0.01)``.

The module also holds the small SQL helpers that the counting modules share
(:func:`bind_params`, :func:`execute_bound`) and :func:`run_key`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "COUNT_UNITS",
    "UNITS",
    "Unit",
    "bind_params",
    "check_threshold",
    "execute_bound",
    "format_threshold",
    "get_unit",
    "run_key",
]

_NAMED_PARAM = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def bind_params(sql: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """The named parameters that ``sql`` uses (``$name``), taken from ``params``.

    DuckDB refuses a named parameter that the statement does not use, so a query built
    from optional parts must bind only the names it contains. A name the SQL uses but
    ``params`` lacks raises KeyError.
    """
    used = set(_NAMED_PARAM.findall(sql))
    missing = sorted(used - set(params))
    if missing:
        raise KeyError(f"SQL parameters without a value: {', '.join(missing)}")
    return {k: v for k, v in params.items() if k in used}


def execute_bound(duck: Any, sql: str, params: Mapping[str, Any]) -> Any:
    """Run ``sql`` on a :class:`~.duck.DuckDB`, binding only the named parameters it uses."""
    return duck.execute(sql, bind_params(sql, params))


def run_key(run: Any) -> int | str:
    """A run given by name or by ``source`` index, with numpy integers made plain ``int``."""
    if isinstance(run, str):
        return run
    if isinstance(run, bool):
        raise ValueError(f"a run is a name or a source index, not {run!r}.")
    try:
        index = int(run)
    except (TypeError, ValueError):
        raise ValueError(f"a run is a name or a source index, not {run!r}.") from None
    if index != run:
        raise ValueError(f"a run is a name or a source index, not {run!r}.")
    return index


def format_threshold(t: float) -> str:
    """A threshold as it appears in labels: ``0.01``, ``0.05``, ``1``."""
    return f"{float(t):.6g}"


@dataclass(frozen=True)
class Unit:
    """One identification unit.

    ``distinct`` lists the columns whose combination is counted once; an empty tuple
    means that rows are counted. ``per_run`` marks the q column that is computed within
    one run. ``sparse`` marks a grouped q column, set on the winning row of each group
    only. ``exclude_empty`` lists key columns whose empty-string value is not counted.
    """

    key: str
    singular: str
    plural: str
    distinct: tuple[str, ...]
    distinct_label: str
    q_column: str
    per_run: bool
    sparse: bool
    exclude_empty: tuple[str, ...] = ()
    group_noun: str = ""

    def noun(self, n: int) -> str:
        """The singular or plural name for ``n`` identifications."""
        return self.singular if n == 1 else self.plural

    def label(self, n: int, t: float) -> str:
        """``'531,720 PSMs (rows, q_value <= 0.01)'``."""
        return (
            f"{n:,} {self.noun(n)} ({self.distinct_label}, {self.q_column} <= "
            f"{format_threshold(t)})"
        )

    def describe(self, t: float) -> str:
        """The unit without a number: ``'peptides (unique base_peptide_id, ...)'``."""
        return f"{self.plural} ({self.distinct_label}, {self.q_column} <= {format_threshold(t)})"

    @property
    def key_sql(self) -> str | None:
        """The counted key as a SQL expression (None for a row unit)."""
        if not self.distinct:
            return None
        if len(self.distinct) == 1:
            return self.distinct[0]
        return "(" + ", ".join(self.distinct) + ")"

    def where_sql(self, condition: str) -> str:
        """``condition`` plus the exclusion of empty key strings."""
        return condition + "".join(f" AND {c} <> ''" for c in self.exclude_empty)

    def count_sql(self, condition: str) -> str:
        """``count(*)`` or ``count(DISTINCT key)`` of the rows that meet ``condition``."""
        where = self.where_sql(condition)
        if self.key_sql is None:
            return f"count(*) FILTER (WHERE {where})"
        return f"count(DISTINCT {self.key_sql}) FILTER (WHERE {where})"


UNITS: dict[str, Unit] = {
    "psm": Unit(
        key="psm",
        singular="PSM",
        plural="PSMs",
        distinct=(),
        distinct_label="rows",
        q_column="q_value",
        per_run=False,
        sparse=False,
    ),
    "precursor": Unit(
        key="precursor",
        singular="precursor",
        plural="precursors",
        distinct=("peptidoform", "charge"),
        distinct_label="unique (peptidoform, charge)",
        q_column="precursor_q",
        per_run=False,
        sparse=True,
        group_noun="(peptidoform, charge)",
    ),
    "peptide": Unit(
        key="peptide",
        singular="peptide",
        plural="peptides",
        distinct=("base_peptide_id",),
        distinct_label="unique base_peptide_id",
        q_column="peptide_q_value",
        per_run=False,
        sparse=True,
        group_noun="base_peptide_id",
    ),
    "protein_group": Unit(
        key="protein_group",
        singular="protein group",
        plural="protein groups",
        distinct=("protein_group",),
        distinct_label="unique protein_group",
        q_column="pg_q_value",
        per_run=False,
        sparse=True,
        exclude_empty=("protein_group",),
        group_noun="protein",
    ),
    "run_psm": Unit(
        key="run_psm",
        singular="PSM",
        plural="PSMs",
        distinct=(),
        distinct_label="rows",
        q_column="run_psm_q",
        per_run=True,
        sparse=False,
    ),
}

# The experiment-wide (or single-run) units that the engine's rescore report counts.
COUNT_UNITS: tuple[str, ...] = ("psm", "precursor", "peptide", "protein_group")


def get_unit(unit: Unit | str) -> Unit:
    """A unit by key (``'psm'``, ``'precursor'``, ``'peptide'``, ``'protein_group'``,
    ``'run_psm'``), or the unit itself."""
    if isinstance(unit, Unit):
        return unit
    try:
        return UNITS[unit]
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; the units are {', '.join(UNITS)}.") from None


def check_threshold(unit: Unit | str, t: float) -> float:
    """Validate a q threshold for a unit and return it as a float.

    Every unit needs ``t > 0``: a q value is at least ``1/T``, never 0. A dense unit
    accepts ``t <= 1``. A sparse unit needs ``t < 1``: its column holds 1.0 on every
    row that lost its group, and a winner can also hold 1.0, so at ``t >= 1`` a winner
    and a loser cannot be told apart and the count would include losing rows.
    """
    u = get_unit(unit)
    if isinstance(t, bool):
        raise ValueError(f"the {u.q_column} threshold must be a number, not {t!r}.")
    try:
        value = float(t)
    except (TypeError, ValueError):
        raise ValueError(f"the {u.q_column} threshold must be a number, not {t!r}.") from None
    if not math.isfinite(value):
        raise ValueError(f"the {u.q_column} threshold must be a finite number, not {t!r}.")
    if value <= 0:
        raise ValueError(
            f"the {u.q_column} threshold must be greater than 0 (got {t!r}). A q value is "
            "at least 1/T, never 0, so nothing is accepted at 0 or below."
        )
    if u.sparse and value >= 1:
        raise ValueError(
            f"the {u.q_column} threshold must be below 1 (got {t!r}). {u.q_column} is set "
            f"on the winning row of each {u.group_noun} group only, and every other row "
            "holds 1.0. At a threshold of 1 or more a winner and a loser cannot be told "
            "apart, so the count would include rows that lost their group."
        )
    if value > 1:
        raise ValueError(
            f"the {u.q_column} threshold must be at most 1 (got {t!r}); q values never exceed 1."
        )
    return value
