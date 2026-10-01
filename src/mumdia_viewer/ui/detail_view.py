"""What the precursor page shows, shaped from the data layer (no Dash components, no figures).

The page links one fragment across its panels: the XIC, the spectrum, the sequence
diagram, the ion table and the fragment list. A fragment is addressed by its ``index``:
its position among the candidate's fragment rows in file order
(``CandidateChromatogram.fragments()``), which is also its row in ``MirrorData.fragments``
and the ``fragment`` of a ``PeakMatch``.

Nothing here recomputes a q value, a score or a quantity. The values come from the data
layer; this module only selects, orders, formats and labels them (:class:`IonLadder`
places the library's fragments on the sequence; :func:`score_bounds` reads the score
column's percentiles for the score bars; :func:`pg_winner` finds the row that holds a
protein group's ``pg_q_value``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.chromatograms import CandidateChromatogram
from mumdia_viewer.data.competition import WINNER_SOURCE, QValue, scored_rows_of, winner_index
from mumdia_viewer.data.detail import EvidenceItem, MirrorData, PrecursorDetail
from mumdia_viewer.data.duck import sql_path
from mumdia_viewer.data.features import EVIDENCE_FEATURES, FeaturePercentile
from mumdia_viewer.data.fragments import PPM_CORRECTED_LABEL, PPM_RAW_LABEL, PeakMatch
from mumdia_viewer.data.spectra import ScanTable

from . import theme
from .widgets import Residue, parse_peptidoform

# --------------------------------------------------------------------------- numbers


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _q_base(v: float, extra: int = 0) -> str:
    """The grid's q form (``window.mvFormat(v, "q")``) with ``extra`` more digits."""
    if v < 0.001:
        mantissa, _, exponent = f"{v:.{2 + extra}e}".partition("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{v:.{4 + extra}f}"


def fmt_q_at(value: Any, threshold: float | None = None) -> str:
    """A q value as text that never reads as the threshold when it differs from it.

    The form is the identification grid's: four decimals from 0.001 up, else three
    significant digits with an exponent (``3.80e-5``). When the value differs from
    ``threshold`` but its text would read as the threshold (0.010003 reads ``0.0100`` at
    q ≤ 0.01), digits are added until the text differs from the threshold and lies on
    the value's side of it (``0.010003``, ``1.0004e-4``). Empty for None and NaN.
    """
    v = _finite(value)
    if v is None:
        return ""
    if v == 0:
        return "0"
    text = _q_base(v)
    t = _finite(threshold)
    if t is None or v == t:
        return text
    for extra in range(1, 14):
        shown = float(text)
        if shown != t and (shown <= t) == (v <= t):
            return text
        text = _q_base(v, extra)
    return repr(v)


def score_digits(values: Iterable[Any], *, least: int = 4, most: int = 8) -> int:
    """The decimals that tell the distinct scores of ``values`` apart (``least`` to ``most``).

    Scores of one competition can agree in their first four decimals (0.999999 and
    0.999996); the rows must still read differently, or the winner cannot be seen.
    """
    vals = sorted({f for f in (_finite(v) for v in values) if f is not None})
    for digits in range(least, most + 1):
        texts = {f"{v:.{digits}f}" for v in vals}
        if len(texts) == len(vals):
            return digits
    return most


# An absolute path: a drive or a leading slash, not inside a word or a relative path.
_PATH = re.compile(r"(?<![\w.\\/])(?:[A-Za-z]:[\\/]|/)(?:[^\s,;()'\"\\/]+[\\/])+([^\s,;()'\"\\/]+)")


def short_paths(text: str) -> str:
    """``text`` with every absolute path shortened to its file name (for one line of text;
    the full message belongs in a tooltip)."""
    return _PATH.sub(lambda m: m.group(1), text)


def wrap_text(text: str, width: int = 72) -> list[str]:
    """``text`` in lines of at most about ``width`` characters, broken at spaces."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines


def fmt_score(value: Any, digits: int = 4) -> str:
    """A score with fixed decimals (the grid's form has four); empty for None and NaN."""
    f = _finite(value)
    return "" if f is None else f"{f:.{digits}f}"


def page_score_digits(d: PrecursorDetail) -> int:
    """The decimals of every score on a precursor page: enough to tell apart this row,
    the rows of its competition and its partner's rows (so the tile, the competition and
    the partner agree)."""
    values: list[Any] = [d.scored.get("score")]
    if d.competition is not None and not d.competition.empty and "score" in d.competition:
        values += list(d.competition["score"])
    rows = d.partner.rows
    if rows is not None and not rows.empty and "score" in rows:
        values += list(rows["score"])
    return score_digits(values)


# The q columns of the verdict strip, in order, with a short unit and the unit colour of
# the overview cards. experiment_psm_q and global_q_value equal q_value (aliases); they are
# listed in the evidence table.
VERDICT_COLUMNS: tuple[str, ...] = (
    "q_value",
    "run_psm_q",
    "precursor_q",
    "peptide_q_value",
    "pg_q_value",
)
SHORT_UNITS: dict[str, tuple[str, str]] = {
    "q_value": ("PSM, pooled", "psm"),
    "experiment_psm_q": ("PSM, pooled", "psm"),
    "global_q_value": ("PSM, pooled", "psm"),
    "run_psm_q": ("PSM, this run", "psm"),
    "precursor_q": ("precursor", "precursor"),
    "peptide_q_value": ("peptide", "peptide"),
    "pg_q_value": ("protein group", "protein_group"),
}
GROUP_NOUNS: dict[str, str] = {
    "precursor_q": "precursor (peptidoform, charge)",
    "peptide_q_value": "base peptide",
    "pg_q_value": "protein group",
}

ION_ORDER = {"b": 0, "y": 1}
# The unit of peptide_quant.quantity (the data layer writes "intensity x s").
QUANT_UNIT = "intensity \u00d7 s"


# --------------------------------------------------------------------------- fragments


@dataclass(frozen=True)
class Fragment:
    """One predicted fragment of the candidate, as the page shows it."""

    index: int
    name: str
    ion: str | None
    ordinal: int | None
    charge: int | None
    mz: float
    obs_mz: float
    predicted: float
    observed: bool
    colour: str

    @property
    def html(self) -> str:
        """The name for Plotly text: ``y7^2`` becomes ``y7<sup>2+</sup>``."""
        if self.ion is None or self.ordinal is None:
            return self.name
        base = f"{self.ion}{self.ordinal}"
        return base if (self.charge or 1) == 1 else f"{base}<sup>{self.charge}+</sup>"

    @property
    def text(self) -> str:
        """The name in plain text: ``y7^2`` becomes ``y7 (2+)``."""
        if self.ion is None or self.ordinal is None:
            return self.name
        base = f"{self.ion}{self.ordinal}"
        return base if (self.charge or 1) == 1 else f"{base} ({self.charge}+)"


def fragments_of(chrom: CandidateChromatogram | None) -> list[Fragment]:
    """The fragment rows of a candidate, ordered b, y, other, then by ordinal and charge.

    The colour of a fragment is its shade in the design system: b ions in blue shades and
    y ions in red shades, one shade per fragment in display order, starting at the second
    shade (the first is too dark on the dark theme).
    """
    if chrom is None:
        return []
    rows = []
    for i, t in enumerate(chrom.fragments()):
        rows.append(
            (
                ION_ORDER.get(t.ion or "", 2),
                t.ordinal if t.ordinal is not None else 10_000,
                t.fragment_charge or 1,
                t.frag_name,
                i,
                t,
            )
        )
    rows.sort(key=lambda r: r[:5])
    seen: dict[str | None, int] = {}
    out = []
    for *_, i, t in rows:
        k = seen.get(t.ion, 0)
        seen[t.ion] = k + 1
        out.append(
            Fragment(
                index=i,
                name=t.frag_name,
                ion=t.ion,
                ordinal=t.ordinal,
                charge=t.fragment_charge,
                mz=float(t.frag_mz),
                obs_mz=float(t.frag_obs_mz),
                predicted=float(t.predicted_intensity),
                observed=bool(t.observed),
                colour=theme.fragment_colour(t.frag_name, t.ion, k + 1),
            )
        )
    return out


# --------------------------------------------------------------------------- scan grid


@dataclass(frozen=True)
class ScanGrid:
    """The MS2 scans of the candidate's XIC grid, one per grid point, in RT order.

    ``rt32`` is ``float32(rt_seconds)``: the chromatogram axis value of each point in
    window-grid mode. An XIC point is matched to its scan by this value.
    """

    rows: np.ndarray
    rt: np.ndarray
    rt32: np.ndarray
    scan_index: np.ndarray
    label: str
    axis_matches: bool | None
    apex_index: int | None

    @property
    def size(self) -> int:
        return int(self.rows.size)

    def index_of_row(self, row: int | None) -> int | None:
        if row is None:
            return None
        hit = np.flatnonzero(self.rows == int(row))
        return int(hit[0]) if hit.size else None

    def index_of_rt(self, x: float) -> int | None:
        """The grid point of an axis value: equal in float32, else the nearest RT."""
        if self.size == 0 or x is None or not math.isfinite(float(x)):
            return None
        x32 = np.float32(x)
        hit = np.flatnonzero(self.rt32 == x32)
        if hit.size:
            return int(hit[0])
        return int(np.argmin(np.abs(self.rt - float(x))))

    def spacing(self) -> float:
        """The median RT step between grid points (seconds); 1.0 for a single point."""
        if self.size < 2:
            return 1.0
        return float(np.median(np.diff(self.rt)))

    def to_store(self) -> dict[str, Any]:
        return {
            "rows": [int(r) for r in self.rows],
            "rt": [float(v) for v in self.rt],
            "rt32": [float(v) for v in self.rt32],
            "scan": [int(s) for s in self.scan_index],
            "apex": self.apex_index,
            "label": self.label,
        }


def scan_grid(rs: ResultSet, d: PrecursorDetail) -> ScanGrid | None:
    """The XIC grid of the candidate (``ScanTable.grid_rows`` over its RT window)."""
    pmz = d.precursor_mz
    if pmz is None or not d.run.has("spectra_ms2"):
        return None
    try:
        scans = ScanTable.for_run(rs, d.run)
    except ViewerError:
        return None
    w = d.window
    lo = w.rt_lo if w is not None and w.rt_lo is not None else -math.inf
    hi = w.rt_hi if w is not None and w.rt_hi is not None else math.inf
    rows = scans.grid_rows(pmz, lo, hi)
    rt = scans.rt[rows].astype(np.float64)
    rt32 = rt.astype(np.float32)
    axis = d.chromatogram.common_axis() if d.chromatogram is not None else None
    matches = None
    if axis is not None:
        matches = bool(axis.size == rt32.size and np.array_equal(axis, rt32))
    apex = d.apex_scan.row if d.apex_scan is not None else None
    hit = np.flatnonzero(rows == apex) if apex is not None else np.zeros(0, dtype=np.int64)
    return ScanGrid(
        rows=rows,
        rt=rt,
        rt32=rt32,
        scan_index=scans.scan_index[rows].astype(np.int64),
        label=scans.grid_label,
        axis_matches=matches,
        apex_index=int(hit[0]) if hit.size else None,
    )


# --------------------------------------------------------------------------- q values


@dataclass(frozen=True)
class GroupWinner:
    """The row that carries a grouped q column's value for this row's group.

    ``how`` says how the row was found (the winner rule, or the stored value).
    """

    column: str
    value: float | None
    run: str
    run_name: str
    candidate_id: int
    peptidoform: str
    charge: int
    label: str
    how: str = WINNER_SOURCE


@dataclass(frozen=True)
class QTile:
    """One q column of the verdict strip.

    ``n_group`` is the number of scored rows in this row's group for a grouped column
    (None when unknown): a group of one row is won by that row by construction.
    """

    column: str
    value: float | None
    unit: str
    short_unit: str
    unit_key: str
    scope: str
    grouped: bool
    winner: bool | None
    winner_source: str | None
    text: str
    group_winner: GroupWinner | None = None
    group_key: str | None = None
    n_group: int | None = None

    @property
    def tested(self) -> float | None:
        """The q tested against the threshold: this row's, or its group winner's."""
        if self.value is not None:
            return self.value
        if self.group_winner is not None:
            return self.group_winner.value
        return None


def _run_name(d: PrecursorDetail, run_label: Any, rs: ResultSet) -> str:
    """The run name for an address: empty in a single run."""
    if not rs.is_experiment:
        return ""
    return str(run_label)


def _winner_of(rs: ResultSet, d: PrecursorDetail, column: str, r: Any, how: str) -> GroupWinner:
    value = r.get(column)
    return GroupWinner(
        column=column,
        value=None if value is None or pd.isna(value) else float(value),
        run=str(r.get("run")),
        run_name=_run_name(d, r.get("run"), rs),
        candidate_id=int(r["candidate_id"]),
        peptidoform=str(r.get("peptidoform")),
        charge=int(r.get("charge")),
        label=str(r.get("label")),
        how=how,
    )


def group_winner(rs: ResultSet, d: PrecursorDetail, column: str) -> GroupWinner | None:
    """The winning row of this row's group for a grouped q column.

    ``precursor_q`` and ``peptide_q_value``: the base-peptide competition
    (``d.competition``) holds every row of the base peptide, so it holds the winner of
    both groups; its winner flags are the engine's rule applied by the data layer.
    ``pg_q_value`` groups span base peptides: see :func:`pg_winner`.
    """
    if column == "pg_q_value":
        return pg_winner(rs, d)
    df = d.competition
    if df is None or df.empty or column not in ("precursor_q", "peptide_q_value"):
        return None
    if column == "peptide_q_value":
        sel = df[df["wins_peptide"].astype(bool)]
    else:
        sel = df[
            df["wins_precursor"].astype(bool)
            & (df["peptidoform"] == d.scored.get("peptidoform"))
            & (df["charge"] == d.scored.get("charge"))
        ]
    if sel.empty:
        return None
    return _winner_of(rs, d, column, sel.iloc[0], WINNER_SOURCE)


PG_WINNER_HOW = (
    "the only row of the protein group whose pg_q_value is below 1.0 (the engine stores the "
    "group's q on its winning row and 1.0 on the others)"
)
PG_WINNER_RULE = (
    "no row of the protein group holds a value below 1.0, so the group's q is 1.0; the row "
    f"shown is the {WINNER_SOURCE.removeprefix('viewer-derived: ')}"
)


def pg_winner(rs: ResultSet, d: PrecursorDetail) -> GroupWinner | None:
    """The row that holds this row's protein group's ``pg_q_value``.

    The engine stores a grouped q on the group's winning row only and 1.0 on the other
    rows, so the group's value is the one row below 1.0. The rows come from the pooled
    scored table (experiment-wide in an experiment). When no row holds a value below 1.0
    the group's q is 1.0, and the row shown is the winner by the engine's rule
    (:func:`~mumdia_viewer.data.competition.winner_index`; in entrapment mode decoys do
    not compete). Memoised per result set and group; None when the group is unknown or
    the table cannot be read.
    """
    group = d.scored.get("protein_group")
    if not group:
        return None
    group = str(group)
    entrapment = d.rescore.mode == "entrapment"

    def load() -> pd.DataFrame | None:
        try:
            df = rs.duck.df(
                "SELECT file_row_number, source, candidate_id, peptidoform, charge, label, "
                "score, pg_q_value FROM read_parquet($p, file_row_number = true) "
                "WHERE protein_group = $g",
                {"p": sql_path(rs.scored.require()), "g": group},
            )
        except (ViewerError, OSError, duckdb.Error):
            return None
        names = {r.index: r.label for r in rs.runs}
        df["run"] = df["source"].map(lambda s: names.get(int(s), str(s)))
        return df

    df = rs.memo(("mumdia_viewer.ui.pg_rows", group), load)
    if df is None or df.empty:
        return None
    held = df[df["pg_q_value"] < 1.0]
    if len(held) == 1:
        return _winner_of(rs, d, "pg_q_value", held.iloc[0], PG_WINNER_HOW)
    flags = pd.Series(False, index=df.index) if entrapment else None
    pool = held if len(held) > 1 else df
    win = winner_index(pool, entrapment=flags.reindex(pool.index) if flags is not None else None)
    if win is None:
        return None
    how = PG_WINNER_HOW if len(held) > 1 else PG_WINNER_RULE
    return _winner_of(rs, d, "pg_q_value", pool.loc[win], how)


def group_size(d: PrecursorDetail, column: str) -> int | None:
    """Scored rows in this row's group for ``precursor_q`` or ``peptide_q_value``."""
    df = d.competition
    if df is None or df.empty:
        return None
    if column == "peptide_q_value":
        return len(df)
    if column == "precursor_q":
        same = (df["peptidoform"] == d.scored.get("peptidoform")) & (
            df["charge"] == d.scored.get("charge")
        )
        return int(same.sum())
    return None


def q_tiles(
    rs: ResultSet, d: PrecursorDetail, columns: Sequence[str] | None = VERDICT_COLUMNS
) -> list[QTile]:
    """The q columns of the row, each with its unit, scope and winner provenance.

    ``columns`` selects and orders them (None: every q column of the row). A grouped
    column that this row does not win carries the winning row of its group
    (:func:`group_winner`), whose value is the one tested against the threshold.
    """
    by_column = {q.column: q for q in d.q_values}
    tiles = []
    for column in columns if columns is not None else list(by_column):
        q: QValue | None = by_column.get(column)
        if q is None:
            continue
        short, key = SHORT_UNITS.get(column, (column, "psm"))
        winner = None
        group_key = None
        if q.grouped and q.winner is False:
            winner = group_winner(rs, d, column)
        if column == "pg_q_value":
            group_key = str(d.scored.get("protein_group") or "")
        tiles.append(
            QTile(
                column=column,
                value=q.display_value,
                unit=q.unit,
                short_unit=short,
                unit_key=key,
                scope=q.scope,
                grouped=q.grouped,
                winner=q.winner,
                winner_source=q.winner_source,
                text=q.text,
                group_winner=winner,
                group_key=group_key,
                n_group=group_size(d, column) if q.grouped else None,
            )
        )
    return tiles


def verdict_tiles(rs: ResultSet, d: PrecursorDetail) -> list[QTile]:
    """The q columns of the verdict strip.

    In a single run ``run_psm_q`` is left out when it equals ``q_value`` (one run: the
    run's PSMs are the pooled PSMs); the q table lists it.
    """
    tiles = q_tiles(rs, d)
    if rs.is_experiment:
        return tiles
    by = {t.column: t for t in tiles}
    psm, run = by.get("q_value"), by.get("run_psm_q")
    if psm is not None and run is not None and psm.value == run.value:
        tiles = [t for t in tiles if t.column != "run_psm_q"]
    return tiles


# --------------------------------------------------------------------------- evidence


def evidence_groups(items: Iterable[EvidenceItem]) -> dict[str, list[EvidenceItem]]:
    out: dict[str, list[EvidenceItem]] = {}
    for e in items:
        out.setdefault(e.group, []).append(e)
    return out


def evidence_value(e: EvidenceItem) -> str:
    """A value with the precision its unit needs; empty for None."""
    return format_value(e.value, e.unit, e.key)


def format_value(value: Any, unit: str = "", key: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("_", " ")
    if isinstance(value, tuple | list):
        parts = [format_value(v, unit, key) for v in value]
        if all(not p or p == "unbounded" for p in parts):
            return "unbounded"
        return "[" + ", ".join(p or "unbounded" for p in parts) + "]"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return ""
    if math.isinf(v):
        return "unbounded"
    u = unit or ""
    if u == "flag":
        return "yes" if v >= 0.5 else "no"
    if u in ("fragments", "count", "scans", "rank") or key in ("n_observations",):
        return f"{v:,.0f}"
    if u in ("counts", "intensity x s"):
        return f"{v:,.0f}"
    if u == "s":
        return f"{v:+.2f}" if key == "rt_error" else f"{v:.2f}"
    if u.startswith("ppm"):
        return (
            f"{v:+.2f}"
            if key in ("signed_mean_frag_ppm", "frag_mass_err_median", "frag_ppm_offset")
            else f"{v:.2f}"
        )
    if u == "fraction of the half-window":
        return f"{v:+.3f}"
    if u.startswith("ln("):
        return f"{v:.2f}"
    if u == "higher is better" or key in ("score", "prelim_score"):
        return f"{v:.4g}"
    if abs(v) >= 1e5:
        return f"{v:,.0f}"
    return f"{v:.3f}"


def unit_text(e: EvidenceItem) -> str:
    """The unit shown after a value (empty where the value says it)."""
    u = e.unit or ""
    if u in ("", "flag", "rank", "higher is better", "q value"):
        return ""
    if u == "intensity x s":
        return QUANT_UNIT
    return u


def percentile_map(rows: Iterable[FeaturePercentile]) -> dict[str, FeaturePercentile]:
    return {r.feature: r for r in rows}


def feature_label(name: str) -> str:
    info = EVIDENCE_FEATURES.get(name)
    return info.label if info is not None else name


def feature_unit(name: str) -> str:
    info = EVIDENCE_FEATURES.get(name)
    return info.unit if info is not None else ""


def percentile_choices(d: PrecursorDetail) -> list[str]:
    """The evidence features this candidate has that can be ranked."""
    if d.features is None:
        return []
    return [
        name
        for name, info in EVIDENCE_FEATURES.items()
        if info.percentile and info.table == "features" and name in d.features
    ]


# --------------------------------------------------------------------------- mirror


@dataclass(frozen=True)
class FragmentRow:
    """One row of the fragment table for the shown scan."""

    fragment: Fragment
    matched: bool
    obs_mz: float | None
    obs_intensity: float | None
    ppm_raw: float | None
    ppm_corrected: float | None


def fragment_rows(m: MirrorData | None, frags: Sequence[Fragment]) -> list[FragmentRow]:
    """The fragments in display order, with this scan's match (if any)."""
    matches = {mt.fragment: mt for mt in m.matches} if m is not None else {}
    out = []
    for f in frags:
        mt = matches.get(f.index)
        out.append(
            FragmentRow(
                fragment=f,
                matched=mt is not None,
                obs_mz=mt.obs_mz if mt is not None else None,
                obs_intensity=mt.obs_intensity if mt is not None else None,
                ppm_raw=mt.ppm_raw if mt is not None else None,
                ppm_corrected=mt.ppm_corrected if mt is not None else None,
            )
        )
    return out


PPM_LABELS = {"raw": PPM_RAW_LABEL, "corrected": PPM_CORRECTED_LABEL}


# --------------------------------------------------------------------------- ion ladder


@dataclass(frozen=True)
class LadderCell:
    """One library fragment on the sequence, with its match in the shown scan (or None)."""

    fragment: Fragment
    match: PeakMatch | None

    @property
    def matched(self) -> bool:
        return self.match is not None


@dataclass(frozen=True)
class IonLadder:
    """The candidate's library fragments placed on its sequence (PeptideShaker's ion table).

    Position ``i`` (1 to n) holds residue ``i``: ``b_i`` holds residues 1 to i and
    ``y_(n-i+1)`` residues i to n, so the cleavage between residues i and i + 1 gives
    ``b_i`` and ``y_(n-i)``. Only the fragments of the library are placed (the
    candidate's chromatogram rows, the only fragments MuMDIA extracts); no other ion is
    computed. ``unplaced`` holds the fragments whose name is not a b or y ion, or whose
    ordinal does not fit the sequence.
    """

    residues: tuple[Residue, ...]
    nterm: tuple[str, ...]
    cterm: tuple[str, ...]
    b_charges: tuple[int, ...]
    y_charges: tuple[int, ...]
    cells: dict[tuple[str, int, int], LadderCell]
    unplaced: tuple[Fragment, ...]
    scan: bool

    @property
    def n(self) -> int:
        return len(self.residues)

    def cell(self, ion: str, ordinal: int, charge: int) -> LadderCell | None:
        return self.cells.get((ion, ordinal, charge))

    @property
    def n_library(self) -> int:
        return len(self.cells)

    @property
    def n_matched(self) -> int:
        return sum(1 for c in self.cells.values() if c.matched)


def ion_ladder(
    peptidoform: str | None, frags: Sequence[Fragment], m: MirrorData | None
) -> IonLadder:
    """The library fragments of the candidate on its sequence, matched against the scan of ``m``."""
    pf = parse_peptidoform(peptidoform)
    n = len(pf.residues)
    matches = {mt.fragment: mt for mt in m.matches} if m is not None else {}
    cells: dict[tuple[str, int, int], LadderCell] = {}
    unplaced: list[Fragment] = []
    for f in frags:
        if f.ion not in ("b", "y") or f.ordinal is None or not 1 <= f.ordinal <= n:
            unplaced.append(f)
            continue
        key = (f.ion, f.ordinal, f.charge or 1)
        if key in cells:
            unplaced.append(f)
            continue
        cells[key] = LadderCell(f, matches.get(f.index))
    return IonLadder(
        residues=pf.residues,
        nterm=pf.nterm,
        cterm=pf.cterm,
        b_charges=tuple(sorted({z for ion, _, z in cells if ion == "b"})),
        y_charges=tuple(sorted({z for ion, _, z in cells if ion == "y"})),
        cells=cells,
        unplaced=tuple(unplaced),
        scan=m is not None,
    )


# --------------------------------------------------------------------------- scales

_SCORE_RANGES: dict[tuple[int, str], tuple[float, float, str]] = {}


def score_range(rs: ResultSet) -> tuple[float, float, str]:
    """The lowest and highest rescorer score of the scored table, and where they come from.

    The footer statistics of ``score`` when every row group has them, else a scan of the
    column. The score bars use this range, so a bar has the same scale on every row.
    """
    art = rs.scored
    key = (id(rs), str(art.path))
    hit = _SCORE_RANGES.get(key)
    if hit is not None:
        return hit
    lo = hi = None
    source = f"footer statistics of {art.path.name if art.path else 'the scored table'}"
    try:
        stats = art.parquet().column_statistics("score")
        if stats and all(s is not None for s in stats):
            lo = min(float(s[0]) for s in stats)  # type: ignore[index]
            hi = max(float(s[1]) for s in stats)  # type: ignore[index]
    except (ViewerError, OSError, TypeError, ValueError):
        lo = hi = None
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        try:
            rows = rs.duck.rows(
                "SELECT min(score), max(score) FROM read_parquet($p) WHERE isfinite(score)",
                {"p": sql_path(art.require())},
            )
            lo, hi = rows[0] if rows else (None, None)
            source = "a scan of the score column"
        except (ViewerError, OSError):
            lo = hi = None
    if lo is None or hi is None or not float(hi) > float(lo):
        out = (0.0, 1.0, "no score range; a placeholder range 0 to 1")
    else:
        out = (float(lo), float(hi), source)
    _SCORE_RANGES[key] = out
    return out


@dataclass(frozen=True)
class Bounds:
    """The fixed scale of the score bars: ``lo`` gives an empty bar, ``hi`` a full one."""

    lo: float
    hi: float
    text: str

    def clamp_note(self, value: Any) -> str:
        v = _finite(value)
        if v is None:
            return ""
        if v < self.lo:
            return " (below the bar's range: drawn empty)"
        if v > self.hi:
            return " (above the bar's range: drawn full)"
        return ""


def score_bounds(rs: ResultSet) -> Bounds:
    """The score bars' scale: the 1st to the 99th percentile of the scored table's score.

    One outlier must not compress every typical score to a full bar (a fixture spans
    -16426 to 9.76), so the bar spans the central 98% of the rows (DuckDB
    ``approx_quantile``) and a score outside it is drawn at the bar's end. The same
    scale serves every row of every page of the result set (memoised). Falls back to
    :func:`score_range` when the percentiles cannot be read.
    """

    def load() -> Bounds:
        name = rs.scored.path.name if rs.scored.path else "the scored table"
        try:
            row = rs.duck.rows(
                "SELECT approx_quantile(score, 0.01), approx_quantile(score, 0.99) "
                "FROM read_parquet($p) WHERE isfinite(score)",
                {"p": sql_path(rs.scored.require())},
            )
            lo, hi = (_finite(v) for v in row[0]) if row else (None, None)
        except (ViewerError, OSError, duckdb.Error):
            lo = hi = None
        if lo is not None and hi is not None and hi > lo:
            return Bounds(
                lo,
                hi,
                f"from the 1st percentile ({lo:.4g}) to the 99th percentile ({hi:.4g}) of "
                f"score in {name} (DuckDB approx_quantile); scores outside are drawn at the "
                "ends",
            )
        a, b, source = score_range(rs)
        return Bounds(a, b, f"from the lowest ({a:.4g}) to the highest ({b:.4g}) score ({source})")

    return rs.memo(("mumdia_viewer.ui.score_bounds", str(rs.scored.path)), load)


def mz_range(m: MirrorData | None) -> list[float] | None:
    """One m/z axis range for every scan of a candidate (computed from the apex scan).

    The range covers the scan's peaks and the library fragments, padded by 3% of the
    span. The mirror of every scan of the candidate uses it, so a zoom survives a scan
    step (Plotly keeps a user zoom only while the figure's own range does not change).
    """
    if m is None:
        return None
    parts = [np.asarray(m.spectrum.mz, dtype=np.float64)]
    if len(m.fragments):
        parts.append(m.fragments["theo_mz"].to_numpy(dtype=np.float64))
    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 1.0)
    return [round(lo - 0.03 * span, 2), round(hi + 0.03 * span, 2)]


def runs_holding(rs: ResultSet, candidate_id: int) -> list[str]:
    """The runs of an experiment with a scored row of ``candidate_id`` (by name)."""
    try:
        df = scored_rows_of(rs, int(candidate_id))
    except (ViewerError, OSError, duckdb.Error, ValueError):
        return []
    names = {r.index: r.name for r in rs.runs}
    return [names[int(s)] for s in df["source"] if int(s) in names]


def in_window(rt: float | None, d: PrecursorDetail) -> bool | None:
    """True when ``rt`` lies inside the candidate's RT window (None when unknown)."""
    w = d.window
    if rt is None or w is None:
        return None
    lo = w.rt_lo if w.rt_lo is not None else -math.inf
    hi = w.rt_hi if w.rt_hi is not None else math.inf
    return lo <= rt <= hi


def transfer_rows(t: dict[str, Any]) -> list[tuple[str, Any, str]]:
    """(label, value, explanation) of a match-between-runs transfer record."""
    labels = (
        ("transfer_q", "q at which the transfer was accepted (permuted-RT null)"),
        ("expected_rt", "RT expected from the donor run, seconds"),
        ("observed_rt", "RT observed in this run, seconds"),
        ("rt_delta", "abs(observed_rt - expected_rt), seconds"),
        ("native_q_value", "pooled q_value in scored_combined.parquet (before MBR)"),
        ("native_run_psm_q", "run_psm_q in scored_combined.parquet (before MBR)"),
        ("q_value_after_mbr", "q_value used by quant and the report after MBR"),
        ("run_psm_q_after_mbr", "run_psm_q used by the report after MBR"),
    )
    return [(k, t.get(k), why) for k, why in labels if k in t]
