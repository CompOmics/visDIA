"""Retention-time and ion-mobility extraction windows of a candidate (``run_windows.parquet``).

``rt-im-train`` writes one row per library candidate, with ``candidate_id`` equal to the row
index: the calibrated RT prediction ``rt_pred_cal`` and the extraction window
``[rt_lo, rt_hi]`` in seconds, and ``im_pred_cal``, ``im_lo``, ``im_hi`` (null on 3D data).
A candidate without a calibrated RT has the row ``(NaN, -inf, +inf)``: extraction searched
an unbounded window. :class:`RtWindow` gives such values as None and says why.

A grouped run has one ``groups/gNN/run_windows.parquet`` per band, with band-local ids:
``local = candidate_id - offset_b``. ``offset_b`` is the number of library precursors with
``precursor_mz < mz_lo`` of the band in the searched precursor table, which is sorted by
m/z with ``candidate_id`` equal to the row index (``Library::precursor_row_span``), and
``n_b`` is the number of rows of the band's table. Under per-group calibration the two bands
of an overlap candidate have different windows, so the window is read from the band whose
chromatograms the run uses.

A precursor table that is not the one the run searched gives offsets that look valid and
point at other candidates' rows. Before its offsets are used for any band, the table is
checked once per run against the run's own tables (:func:`band_offsets`). If one check
fails, the table is used for no band.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import blake3
import numpy as np

from .artifacts import Artifact
from .candidate_index import (
    CandidateIndex,
    DenseIndex,
    concat_parts,
    index_for,
    read_candidate_rows,
)
from .chromatograms import CandidateChromatogram, ChromatogramSource, _index_key
from .discovery import Band, GroupedLayout, ResultSet, Run
from .duck import sql_ident, sql_path
from .errors import ArtifactNotFound, InconsistentData, ViewerError
from .pqio import ParquetHandle
from .reports import load_json

__all__ = [
    "AmbiguousBand",
    "BandOffset",
    "RtWindow",
    "band_offsets",
    "rt_window",
]

RT_COLUMNS = ("candidate_id", "rt_pred_cal", "rt_lo", "rt_hi")
IM_COLUMNS = ("im_pred_cal", "im_lo", "im_hi")
_MEMO_KEY = "mumdia_viewer.data.windows"

#: Most rows sampled from the run-level seed table, and from each band competed table, to
#: check the searched library's ``precursor_mz``.
SEED_SAMPLE_ROWS = 2048
BAND_SAMPLE_ROWS = 64

#: Competed columns compared to find the band whose row a pooled competed table kept. The
#: kept band's row equals the pooled row; the other band's row differs at least in
#: ``prelim_score`` or in the gradient-normalised features (G4 F5, F6.3).
COMPETED_MATCH_COLUMNS = (
    "peak_rank",
    "apex_rt",
    "elution_lo",
    "elution_hi",
    "prelim_score",
    "rt_error_rel",
    "rt_error_signed_norm_gradient",
    "rt_error_abs_norm_gradient",
    "observed_rt_fraction",
    "predicted_rt_fraction",
)


class AmbiguousBand(ViewerError):
    """Several bands hold the candidate in their row spans with different windows.

    The chromatograms do not tell which band the run used (for example a candidate that no
    band accepted). Pass ``band=`` to choose one.
    """


@dataclass(frozen=True)
class RtWindow:
    """The extraction window of one candidate, as ``run_windows.parquet`` records it.

    RT values are seconds. ``rt_pred_cal``, ``rt_lo`` and ``rt_hi`` are None where the
    stored value is not finite: a NaN prediction and infinite bounds mean that no calibrated
    RT exists for the candidate and that the window is unbounded. The ion-mobility values
    are None where they are null (3D data). ``source`` names the table and row read (for a
    band table, the band-local row and how it was found); ``band`` is the band (``gNN``) of
    a grouped run, or None when the band could not be determined because every band that
    holds the candidate gives the same window (``source`` then names them);
    ``calibration_note`` says how the window was calibrated.
    """

    candidate_id: int
    rt_pred_cal: float | None
    rt_lo: float | None
    rt_hi: float | None
    im_pred_cal: float | None
    im_lo: float | None
    im_hi: float | None
    source: str
    band: str | None
    calibration_note: str

    @property
    def half_width(self) -> float | None:
        """``(rt_hi - rt_lo) / 2`` in seconds; None for an unbounded window."""
        if self.rt_lo is None or self.rt_hi is None:
            return None
        return (self.rt_hi - self.rt_lo) / 2.0

    @property
    def bounded(self) -> bool:
        """True when both RT bounds are finite."""
        return self.rt_lo is not None and self.rt_hi is not None


@dataclass(frozen=True)
class BandOffset:
    """Where a band's local ids sit among the library-wide ids: ``offset <= cid < offset + n``.

    ``method`` is ``library`` (counted in the searched precursor table, as the engine does,
    after that table passed the checks of :func:`band_offsets`) or ``inferred`` (the
    seed-join heuristic, used when no library is available or the library failed a check;
    not an engine contract). ``checked`` says what the offset was checked against. It
    unpacks as ``offset, n = band_offset``.
    """

    offset: int
    n: int
    method: Literal["library", "inferred"]
    note: str
    checked: str = ""

    def __iter__(self) -> Iterator[int]:
        yield self.offset
        yield self.n

    def holds(self, cid: int) -> bool:
        return self.offset <= cid < self.offset + self.n


@dataclass(frozen=True)
class _Placement:
    """The band offsets of one run, the reason each other band has none, and the library."""

    offsets: dict[str, BandOffset]
    unplaced: dict[str, str]
    library: str


# --------------------------------------------------------------------------- band offsets


def _plan_bounds(band: Band) -> tuple[float, float] | None:
    plan = band.plan or {}
    lo, hi = plan.get("mz_lo"), plan.get("mz_hi")
    if isinstance(lo, bool) or isinstance(hi, bool):
        return None
    if not isinstance(lo, int | float) or not isinstance(hi, int | float):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return float(lo), float(hi)


def _band_sizes(grouped: GroupedLayout) -> dict[str, int]:
    """Rows of each band's ``run_windows.parquet`` (``n_b``), from the footers."""
    sizes: dict[str, int] = {}
    for band in grouped.bands:
        artifact = band.artifacts.get("run_windows")
        if artifact is None or not artifact.usable:
            continue
        try:
            sizes[band.name] = artifact.parquet().num_rows
        except ViewerError:
            continue
    return sizes


def _searched_library(rs: ResultSet, run: Run) -> Artifact | None:
    """The searched precursor table, when it is readable and its ids are its row numbers."""
    if rs.is_experiment:
        artifact = rs.extra.get("fragment_library_precursors")
    else:
        artifact = run.artifacts.get("fragment_library_precursors")
    if artifact is None or not artifact.usable:
        return None
    try:
        handle = artifact.parquet()
    except ViewerError:
        return None
    if not DenseIndex.applies(handle):
        return None
    return artifact


def _count_rows(
    handle: ParquetHandle,
    column: str,
    thresholds: list[tuple[float, bool]],
) -> list[int]:
    """Exact ``count(column < x)`` (or ``<= x`` when inclusive) for each threshold.

    Row groups whose footer statistics lie wholly on one side are counted from the footer;
    only the groups a threshold falls inside are read, with the one column projected.
    """
    stats = handle.column_statistics(column)
    sizes = np.diff(handle.row_group_offsets()).tolist()
    read: dict[int, np.ndarray] = {}

    def values(g: int) -> np.ndarray:
        if g not in read:
            table = handle.read_row_group(g, [column])
            read[g] = table.column(0).to_numpy()
        return read[g]

    out = []
    for x, inclusive in thresholds:
        total = 0
        for g, (size, stat) in enumerate(zip(sizes, stats, strict=True)):
            if size == 0:
                continue
            if stat is not None:
                lo, hi = float(stat[0]), float(stat[1])
                if (hi <= x) if inclusive else (hi < x):
                    total += size
                    continue
                if (lo > x) if inclusive else (lo >= x):
                    continue
            v = values(g)
            total += int(np.count_nonzero(v <= x if inclusive else v < x))
        out.append(total)
    return out


def _library_counts(
    rs: ResultSet, library: Artifact, bounds: list[tuple[float, float]]
) -> list[tuple[int, int]]:
    """Per band ``(count(precursor_mz < mz_lo), count(precursor_mz <= mz_hi))``, cached."""
    if not bounds:
        return []
    identity = library.identity()
    digest = blake3.blake3(
        json.dumps([[lo.hex(), hi.hex()] for lo, hi in bounds]).encode()
    ).hexdigest()[:24]
    name = f"band_counts_v1_{digest}"
    hit = rs.cache.load_json(identity, name)
    if (
        isinstance(hit, list)
        and len(hit) == len(bounds)
        and all(isinstance(x, list) and len(x) == 2 for x in hit)
    ):
        return [(int(a), int(b)) for a, b in hit]
    thresholds = [t for lo, hi in bounds for t in ((lo, False), (hi, True))]
    counts = _count_rows(library.parquet(), "precursor_mz", thresholds)
    out = [(counts[2 * i], counts[2 * i + 1]) for i in range(len(bounds))]
    rs.cache.save_json(identity, name, [list(p) for p in out])
    return out


def _footer_id_range(artifact: Artifact | None) -> tuple[int, int] | None:
    """``(min, max)`` of ``candidate_id`` from the footer; None when empty or not recorded."""
    if artifact is None or not artifact.usable:
        return None
    try:
        handle = artifact.parquet()
    except ViewerError:
        return None
    stats = handle.column_statistics("candidate_id")
    sizes = np.diff(handle.row_group_offsets()).tolist()
    lo: int | None = None
    hi: int | None = None
    for size, stat in zip(sizes, stats, strict=True):
        if size == 0:
            continue
        if stat is None:
            return None
        lo = int(stat[0]) if lo is None else min(lo, int(stat[0]))
        hi = int(stat[1]) if hi is None else max(hi, int(stat[1]))
    return None if lo is None or hi is None else (lo, hi)


#: Band tables whose ``candidate_id`` is library-wide and lies inside the band's row span:
#: a band's extract, features and compete process only its library slice (G4 F8.4).
_SPAN_TABLES = ("chromatograms", "psms_competed")


def _range_problem(band: Band, offset: int, n: int) -> str | None:
    """Why ``[offset, offset + n)`` cannot be the row span of ``band``; None when it can.

    The footer ``candidate_id`` range of the band's chromatogram and competed tables must
    lie inside the span. A table without rows or statistics is not checked.
    """
    for kind in _SPAN_TABLES:
        artifact = band.artifacts.get(kind)
        bounds = _footer_id_range(artifact)
        if bounds is None or artifact is None or artifact.path is None:
            continue
        lo, hi = bounds
        if lo < offset or hi >= offset + n:
            return (
                f"band {band.name}: {artifact.path.name} holds candidate_id {lo} to {hi}, "
                f"outside the row span [{offset}, {offset + n})"
            )
    return None


def _span_checked(band: Band) -> str:
    kinds = [k for k in _SPAN_TABLES if _footer_id_range(band.artifacts.get(k)) is not None]
    if not kinds:
        return "no band table with candidate_id statistics to check the span against"
    names = " and ".join(f"{k}.parquet" for k in kinds)
    return f"the candidate_id range of the band {names} lies inside the span"


_SAMPLE_SQL = (
    "SELECT candidate_id, precursor_mz FROM read_parquet(?, file_row_number = true) "
    "WHERE file_row_number % ? = 0 AND precursor_mz IS NOT NULL"
)

_LOOKUP_SQL = (
    "SELECT candidate_id, precursor_mz FROM read_parquet(?) "
    "WHERE candidate_id IN (SELECT unnest(CAST(? AS UBIGINT[])))"
)


def _references(rs: ResultSet, run: Run) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``(table, candidate_id, precursor_mz)`` samples of run tables with library-wide ids.

    The run-level ``seed_psms.parquet`` of a grouped run (the pooled seed) and the band
    ``psms_competed.parquet`` tables hold library-wide ids and the searched library's
    ``precursor_mz``. Evenly spaced rows are taken: at most :data:`SEED_SAMPLE_ROWS` of the
    seed and :data:`BAND_SAMPLE_ROWS` of each band table.
    """
    grouped = run.grouped
    tables: list[tuple[str, Artifact | None, int]] = [
        ("seed_psms.parquet", run.artifacts.get("seed_psms"), SEED_SAMPLE_ROWS)
    ]
    if grouped is not None:
        for band in grouped.bands:
            tables.append(
                (
                    f"groups/{band.name}/psms_competed.parquet",
                    band.artifacts.get("psms_competed"),
                    BAND_SAMPLE_ROWS,
                )
            )
    out = []
    for label, artifact, cap in tables:
        if artifact is None or not artifact.usable or artifact.path is None:
            continue
        try:
            handle = artifact.parquet()
        except ViewerError:
            continue
        if not (handle.has_column("candidate_id") and handle.has_column("precursor_mz")):
            continue
        rows = handle.num_rows
        if rows == 0:
            continue
        step = max(1, -(-rows // cap))
        got = rs.duck.arrow(_SAMPLE_SQL, [sql_path(artifact.path), step])
        ids = got.column("candidate_id").to_numpy().astype(np.int64)
        mz = got.column("precursor_mz").to_numpy().astype(np.float64)
        if ids.size:
            out.append((label, ids, mz))
    return out


def _library_mz(rs: ResultSet, library: Artifact, ids: np.ndarray) -> np.ndarray:
    """The library's ``precursor_mz`` at each of the sorted unique ``ids`` (NaN if absent)."""
    identity = library.identity()
    name = f"precursor_mz_v1_{blake3.blake3(ids.astype('<i8').tobytes()).hexdigest()[:24]}"
    hit = rs.cache.load_arrays(identity, name)
    if hit is not None and "ids" in hit and np.array_equal(hit["ids"], ids):
        return hit["mz"]
    assert library.path is not None
    got = rs.duck.arrow(_LOOKUP_SQL, [sql_path(library.path), ids.tolist()])
    found = got.column("candidate_id").to_numpy().astype(np.int64)
    values = got.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
    mz = np.full(ids.size, np.nan)
    if found.size:
        mz[np.searchsorted(ids, found)] = values
    rs.cache.save_arrays(identity, name, ids=ids, mz=mz)
    return mz


def _precursor_check(rs: ResultSet, run: Run, library: Artifact) -> tuple[str | None, str]:
    """Compare the library's ``precursor_mz`` with the run's own tables at library-wide ids.

    Returns ``(problem, checked)``: the mismatch (None when every sampled id agrees) and a
    sentence saying what was compared.
    """
    assert library.path is not None
    refs = _references(rs, run)
    if not refs:
        return None, (
            "precursor_mz not checked: the run has no seed_psms.parquet or band "
            "psms_competed.parquet with library-wide ids"
        )
    ids = np.unique(np.concatenate([r[1] for r in refs]))
    lib_mz = _library_mz(rs, library, ids)
    total = 0
    for label, ref_ids, ref_mz in refs:
        values = lib_mz[np.searchsorted(ids, ref_ids)]
        bad = np.flatnonzero(~(values == ref_mz))
        total += ref_ids.size
        if bad.size:
            k = int(bad[0])
            have = "no row" if math.isnan(float(values[k])) else f"precursor_mz {values[k]!r}"
            return (
                f"precursor_mz differs from {label} at {bad.size} of {ref_ids.size} sampled "
                f"library-wide ids (candidate_id {ref_ids[k]}: {have} in {library.path.name}, "
                f"{ref_mz[k]!r} in {label})"
            ), ""
    names = ", ".join(r[0] for r in refs[:3]) + (" ..." if len(refs) > 3 else "")
    return None, f"precursor_mz agrees at {total} sampled library-wide ids of {names}"


# A full aggregate, not DISTINCT ... LIMIT 2: a query that stops its scans early leaves
# the parquet readers open on the thread's DuckDB cursor until its next statement, and on
# Windows an open file blocks the engine from replacing it.
_SEED_JOIN = """
SELECT count(*) AS n, min(d) AS lo, max(d) AS hi
FROM (
  SELECT CAST(g.candidate_id AS BIGINT) - CAST(b.candidate_id AS BIGINT) AS d
  FROM read_parquet(?) AS b
  JOIN read_parquet(?) AS g
    ON b.scan_index = g.scan_index AND b.score = g.score
   AND b.peptidoform = g.peptidoform AND b.charge = g.charge
)
"""


def _seed_offset(rs: ResultSet, run: Run, band: Band) -> tuple[int | None, str]:
    """The one ``global - local`` id difference of the band seeds joined to the run seeds.

    The band ``seed_psms.parquet`` holds band-local ids, the run-level (pooled) one
    library-wide ids. The join is on ``(scan_index, score, peptidoform, charge)``. Returns
    the offset, or None and the reason: no seed table, no row to join, or a difference
    that is not constant.
    """
    band_seed = band.artifacts.get("seed_psms")
    run_seed = run.artifacts.get("seed_psms")
    if band_seed is None or run_seed is None or not band_seed.usable or not run_seed.usable:
        missing = "band" if band_seed is None or not band_seed.usable else "run-level"
        return None, f"the {missing} seed_psms.parquet is missing or unreadable"
    try:
        band_seed.parquet()
        run_seed.parquet()
    except ViewerError as exc:
        return None, f"a seed_psms.parquet cannot be read ({exc})"
    rows = rs.duck.rows(_SEED_JOIN, [sql_path(band_seed.path), sql_path(run_seed.path)])
    n, lo, hi = rows[0] if rows else (0, None, None)
    if not n or lo is None or hi is None:
        return None, "no band seed row joins a run-level seed row"
    if lo != hi:
        return None, "the band and run-level seed rows do not differ by one constant id offset"
    return int(lo), ""


_SEED_METHOD = (
    "offset inferred from the seed join (global - local candidate_id of band and run "
    "seed_psms on scan_index, score, peptidoform, charge)"
)


def _placement(rs: ResultSet, run: Run, *, library: bool = True) -> _Placement:
    """Compute (once per run) the offset of every band, or why a band has none."""
    grouped = run.grouped
    assert grouped is not None
    key = (_MEMO_KEY, "placement", run.index, str(run.root), library)
    hit = rs._memo.get(key)
    if hit is not None:
        return hit
    sizes = _band_sizes(grouped)
    offsets: dict[str, BandOffset] = {}
    unplaced: dict[str, str] = {}
    lib = _searched_library(rs, run) if library else None
    rejected = ""
    no_bounds: set[str] = set()
    if not library:
        verdict = "the searched library was not used (library=False)"
    elif lib is None or lib.path is None:
        verdict = "no searched library precursor table with candidate_id equal to the row index"
    else:
        planned = []
        for band in grouped.bands:
            bounds = _plan_bounds(band)
            if band.name in sizes and bounds is not None:
                planned.append((band, bounds))
            elif band.name in sizes:
                no_bounds.add(band.name)
        counts = _library_counts(rs, lib, [b for _, b in planned])
        # The precursor m/z comparison is the direct test of the table's identity, so it is
        # listed first; the span and id-range checks follow.
        problem, checked = _precursor_check(rs, run, lib)
        problems: list[str] = [] if problem is None else [problem]
        spans: dict[str, tuple[Band, int, int, float]] = {}
        for (band, (mz_lo, _mz_hi)), (below, upto) in zip(planned, counts, strict=True):
            n = sizes[band.name]
            if upto - below != n:
                problems.append(
                    f"band {band.name}: the span mz_lo <= precursor_mz <= mz_hi holds "
                    f"{upto - below} precursors, but groups/{band.name}/run_windows.parquet "
                    f"has {n} rows"
                )
            spans[band.name] = (band, below, n, mz_lo)
        for band, below, n, _ in spans.values():
            problem = _range_problem(band, below, n)
            if problem is not None:
                problems.append(problem)
        if problems:
            shown = "; ".join(problems[:3]) + (
                f"; and {len(problems) - 3} more" if len(problems) > 3 else ""
            )
            rejected = (
                f"{lib.path.name} is not the precursor table this run searched, or the plan or "
                f"band tables belong to another run ({shown}); it is used for no band"
            )
            verdict = rejected
        else:
            verdict = f"{lib.path.name} passed the checks that could be made ({checked})"
            for name, (band, below, n, mz_lo) in spans.items():
                offsets[name] = BandOffset(
                    below,
                    n,
                    "library",
                    f"offset = count(precursor_mz < mz_lo {mz_lo!r}) in {lib.path.name}; "
                    f"n = rows of groups/{name}/run_windows.parquet",
                    f"the library span holds n = {n} precursors; {_span_checked(band)}; {checked}",
                )
    for band in grouped.bands:
        if band.name in offsets:
            continue
        if band.name not in sizes:
            unplaced[band.name] = (
                f"groups/{band.name}/run_windows.parquet is missing or unreadable, so the band "
                "has no row count"
            )
            continue
        n = sizes[band.name]
        if band.name in no_bounds:
            why_not = f"groups/plan.json gives no finite mz_lo and mz_hi for band {band.name}"
        else:
            why_not = rejected or verdict
        seeded, why = _seed_offset(rs, run, band)
        if seeded is None:
            unplaced[band.name] = f"{why_not}; and the seed join gives no offset: {why}"
            continue
        problem = _range_problem(band, seeded, n)
        if problem is not None:
            unplaced[band.name] = f"{why_not}; and the seed-join offset {seeded} fails: {problem}"
            continue
        offsets[band.name] = BandOffset(
            seeded, n, "inferred", f"{_SEED_METHOD}; {why_not}", _span_checked(band)
        )
    placement = _Placement(offsets, unplaced, verdict)
    rs._memo[key] = placement
    return placement


def band_offsets(rs: ResultSet, run: Run, *, library: bool = True) -> dict[str, BandOffset]:
    """Offset and size of every band of a grouped run, keyed by band name (``gNN``).

    ``offset_b = count(precursor_mz < mz_lo)`` over the searched precursor table (single
    run: ``run.artifacts['fragment_library_precursors']``; experiment:
    ``rs.extra['fragment_library_precursors']``), with ``mz_lo`` from ``groups/plan.json``,
    and ``n_b`` = rows of the band's ``run_windows.parquet``. The counts come from the
    footer statistics plus the row groups a bound falls inside, and are cached under the
    library's identity.

    The table is checked once per run before any band uses it:

    * for every band, ``count(mz_lo <= precursor_mz <= mz_hi)`` equals ``n_b``;
    * for every band, the footer ``candidate_id`` range of the band chromatogram and
      competed tables lies inside ``[offset_b, offset_b + n_b)``;
    * its ``precursor_mz`` equals the ``precursor_mz`` that the run-level
      ``seed_psms.parquet`` and the band ``psms_competed.parquet`` tables record, at a
      sample of their library-wide ids.

    If one check fails, the table is used for no band: every band's offset is inferred
    from the seeds (the constant ``global - local`` id difference of the band and run
    seed tables, labelled ``inferred``; not an engine contract), and such an offset must
    pass the range check too. The same holds without a usable library or with
    ``library=False``. A band without a readable ``run_windows.parquet``, or whose offset
    cannot be found or fails its check, is left out; :func:`rt_window` then raises with the
    reason. An ungrouped run gives an empty dict.
    """
    if run.grouped is None:
        return {}
    return dict(_placement(rs, run, library=library).offsets)


# --------------------------------------------------------------------------- window rows


def _all_null(handle: ParquetHandle, row_group: int, column: str) -> bool:
    """True when the footer says ``column`` is null on every row of the row group."""
    md = handle.metadata()
    schema = md.schema
    leaf = next((j for j in range(len(schema)) if schema.column(j).path == column), None)
    if leaf is None:
        return False
    group = md.row_group(row_group)
    stats = group.column(leaf).statistics
    return bool(stats is not None and stats.has_null_count and stats.null_count == group.num_rows)


def _window_row(rs: ResultSet, artifact: Artifact, row_id: int) -> dict[str, Any] | None:
    """The ``run_windows`` row of ``row_id`` (a band-local id in a band table), or None.

    The row group is read without the row-group cache: a ``run_windows`` group of an Astral
    run is about 30 MB of Arrow memory for one row, and caching it would evict the
    chromatogram groups. Ion-mobility columns that the footer shows to be null are not read.
    """
    handle = artifact.parquet()
    index = index_for(artifact, rs.cache, dense=True)
    key: Any = row_id
    if isinstance(index, CandidateIndex):  # the table is not dense; see _index_key
        key = _index_key(index, row_id)
        if key is None:
            return None
    if key not in index:
        return None
    segments = index.segments(key)
    columns = list(RT_COLUMNS)
    for column in IM_COLUMNS:
        if handle.has_column(column) and not all(
            _all_null(handle, s.row_group, column) for s in segments
        ):
            columns.append(column)
    table = concat_parts(read_candidate_rows(handle, index, key, columns, cached=False))
    if table is None or table.num_rows != 1:
        n = 0 if table is None else table.num_rows
        raise InconsistentData(
            f"{artifact.path}: {n} rows for candidate_id {row_id}; run_windows holds one row "
            "per candidate."
        )
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    if int(row["candidate_id"]) != row_id:
        raise InconsistentData(
            f"{artifact.path}: the row read for candidate_id {row_id} holds candidate_id "
            f"{row['candidate_id']}."
        )
    return row


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:.4f}"


def _calibration_note(row: dict[str, Any], cal_path: Path | None, scope: str) -> str:
    pred, lo, hi = row.get("rt_pred_cal"), row.get("rt_lo"), row.get("rt_hi")
    parts = []
    if all(_finite(v) is not None for v in (pred, lo, hi)):
        parts.append("calibrated RT prediction with a bounded extraction window")
    elif (
        _finite(pred) is None
        and lo is not None
        and hi is not None
        and math.isinf(lo)
        and math.isinf(hi)
        and lo < 0 < hi
    ):
        parts.append(
            f"no calibrated RT for this candidate (rt_pred_cal {_fmt(pred)}) and an unbounded "
            "extraction window (rt_lo -inf, rt_hi +inf)"
        )
    else:
        parts.append(
            f"non-finite values in the row (rt_pred_cal {_fmt(pred)}, rt_lo {_fmt(lo)}, "
            f"rt_hi {_fmt(hi)}); they are given as None"
        )
    cal = load_json(cal_path) if cal_path is not None else None
    if cal is None:
        parts.append(f"no cal.json in {scope}")
    else:
        text = (
            f"cal.json of {scope}: calibration_status {cal.get('calibration_status')}, "
            f"method {cal.get('method')}"
        )
        w_rt = cal.get("w_rt")
        if isinstance(w_rt, int | float) and not isinstance(w_rt, bool) and math.isfinite(w_rt):
            text += f", w_rt {w_rt:.3f} s (half-window)"
        parts.append(text)
    return "; ".join(parts) + "."


def _display(rs: ResultSet, path: Path | None) -> str:
    if path is None:
        return "?"
    try:
        return path.relative_to(rs.root).as_posix()
    except ValueError:
        return str(path)


def _make_window(
    cid: int, row: dict[str, Any], source: str, band: str | None, cal: Path | None, scope: str
) -> RtWindow:
    return RtWindow(
        candidate_id=cid,
        rt_pred_cal=_finite(row.get("rt_pred_cal")),
        rt_lo=_finite(row.get("rt_lo")),
        rt_hi=_finite(row.get("rt_hi")),
        im_pred_cal=_finite(row.get("im_pred_cal")),
        im_lo=_finite(row.get("im_lo")),
        im_hi=_finite(row.get("im_hi")),
        source=source,
        band=band,
        calibration_note=_calibration_note(row, cal, scope),
    )


# --------------------------------------------------------------------------- band choice


@dataclass(frozen=True)
class _BandChoice:
    """The band whose window applies, or the bands that all give the same window.

    ``band`` None with ``identical`` set: the band was not determined, and every band in
    ``identical`` holds the candidate with the same window. ``band`` None with ``identical``
    empty: no band holds the candidate.
    """

    band: str | None
    how: str
    identical: tuple[str, ...] = ()


def _band_named(grouped: GroupedLayout, name: str) -> Band | None:
    return next((b for b in grouped.bands if b.name == name), None)


def _same_chromatogram(a: CandidateChromatogram | None, b: CandidateChromatogram) -> bool:
    if a is None or len(a.traces) != len(b.traces):
        return False
    for x, y in zip(a.traces, b.traces, strict=True):
        if x.frag_name != y.frag_name or x.rt.shape != y.rt.shape:
            return False
        if not np.array_equal(x.rt.view(np.uint32), y.rt.view(np.uint32)):
            return False
        if not np.array_equal(x.intensity.view(np.uint32), y.intensity.view(np.uint32)):
            return False
    return True


def _competed_rows(
    rs: ResultSet, artifact: Artifact, cid: int, columns: list[str]
) -> list[tuple[str, ...]]:
    """The rows of ``cid`` in a competed table (``columns`` only), as exact value strings."""
    assert artifact.path is not None
    cols = ", ".join(sql_ident(c) for c in columns)
    sql = (
        f"SELECT {cols} FROM read_parquet(?) WHERE candidate_id = ? "
        f"ORDER BY {', '.join(sql_ident(c) for c in columns)}"
    )
    return [
        tuple(repr(v) for v in row) for row in rs.duck.rows(sql, [sql_path(artifact.path), cid])
    ]


def _pooled_band(
    rs: ResultSet, run: Run, source: ChromatogramSource, cid: int, spans: list[str]
) -> tuple[str | None, str]:
    """The band a pooled run kept ``cid`` from, when its pooled rows show it.

    First the pooled chromatogram rows are compared with each band table (they differ under
    per-group calibration); then the pooled ``psms_competed`` rows with each band's
    competed rows (the kept band's row equals the pooled row). A band is returned only
    when exactly one band matches.
    """
    pooled = source.read(cid)
    if pooled is not None:
        matches = [
            t.band
            for t in source.band_tables
            if t.band in spans and _same_chromatogram(source.read_table(t, cid), pooled)
        ]
        if len(matches) == 1 and matches[0] is not None:
            return matches[0], "the band table whose rows equal the pooled chromatogram rows"
    grouped = run.grouped
    root = run.artifacts.get("psms_competed")
    if grouped is None or root is None or not root.usable:
        return None, ""
    try:
        root_names = set(root.parquet().schema.names)
    except ViewerError:
        return None, ""
    candidates: list[tuple[str, Artifact, list[str]]] = []
    for name in spans:
        band = _band_named(grouped, name)
        artifact = band.artifacts.get("psms_competed") if band is not None else None
        if artifact is None or not artifact.usable:
            continue
        try:
            names = set(artifact.parquet().schema.names)
        except ViewerError:
            continue
        columns = [c for c in COMPETED_MATCH_COLUMNS if c in names and c in root_names]
        if "prelim_score" in columns:
            candidates.append((name, artifact, columns))
    matches = []
    for name, artifact, columns in candidates:
        want = _competed_rows(rs, root, cid, columns)
        if want and _competed_rows(rs, artifact, cid, columns) == want:
            matches.append(name)
    if len(matches) == 1:
        return matches[0], "the band psms_competed table whose rows equal the pooled competed rows"
    return None, ""


def _check_band_name(band: object) -> str:
    if not isinstance(band, str):
        raise TypeError(
            f"band= takes a band directory name such as 'g01', not {type(band).__name__} "
            f"{band!r}. An integer is ambiguous: it can be a plan index (the NN of gNN) or a "
            "position in the loser file's band list (ChromatogramTable.position), and the two "
            "differ when bands were skipped; pass ChromatogramTable.band instead."
        )
    return band


def _choose_band(rs: ResultSet, run: Run, cid: int, band: str | None) -> _BandChoice:
    """The band whose window applies to ``cid``, and how it was chosen."""
    grouped = run.grouped
    assert grouped is not None
    if band is not None:
        name = _check_band_name(band)
        if _band_named(grouped, name) is None:
            raise ValueError(
                f"{run.label} has no band {name}; bands are {[b.name for b in grouped.bands]}."
            )
        return _BandChoice(name, "band given by the caller")
    # The chromatograms only choose the band. When they cannot be read at all (missing,
    # an unsupported layout, a broken loser file), the row-span rules below still give a
    # correct window or refuse; an error about this candidate's rows is raised.
    source: ChromatogramSource | None
    try:
        source = ChromatogramSource.for_run(rs, run)
    except ViewerError:
        source = None
    if source is not None:
        held = source.band_of(cid)
        if held is not None:
            return _BandChoice(
                held, "the band whose chromatogram table yields the candidate's rows"
            )
    placement = _placement(rs, run)
    offsets = placement.offsets
    spans = [name for name, bo in offsets.items() if bo.holds(cid)]
    if not spans:
        unplaced = list(placement.unplaced)
        if unplaced:
            shown = ", ".join(unplaced[:6]) + (" ..." if len(unplaced) > 6 else "")
            first = placement.unplaced[unplaced[0]]
            raise ArtifactNotFound(
                f"{run.root}: the band of candidate_id {cid} cannot be found: no band table "
                f"yields its chromatogram rows, and the row spans of {len(unplaced)} band(s) "
                f"({shown}) are unknown; a row span needs the band's run_windows.parquet and "
                f"the searched library precursor table or band seed rows (band {unplaced[0]}: "
                f"{first})."
            )
        return _BandChoice(None, "")
    if len(spans) == 1:
        return _BandChoice(spans[0], "the only band whose library row span holds the candidate")
    if source is not None and not source.per_band:
        found, how = _pooled_band(rs, run, source, cid, spans)
        if found is not None:
            return _BandChoice(found, how)
    rows = []
    for name in spans:
        b = _band_named(grouped, name)
        artifact = b.artifacts.get("run_windows") if b is not None else None
        rows.append(
            None if artifact is None else _window_row(rs, artifact, cid - offsets[name].offset)
        )
    compared = (*RT_COLUMNS[1:], *IM_COLUMNS)
    keys = [None if r is None else tuple(repr(r.get(c)) for c in compared) for r in rows]
    if all(k is not None and k == keys[0] for k in keys):
        return _BandChoice(
            None,
            f"band not determined; bands {', '.join(spans)} give identical windows and "
            "the run's tables do not show which band it used",
            tuple(spans),
        )
    raise AmbiguousBand(
        f"{run.root}: candidate_id {cid} lies in the row spans of bands {', '.join(spans)}, "
        "whose windows differ, and its chromatograms do not tell which band the run used; "
        "pass band= to choose one."
    )


# --------------------------------------------------------------------------- entry point


def _offset_text(cid: int, offset: BandOffset) -> str:
    return (
        f"band-local id = candidate_id {cid} - band offset {offset.offset}, offset "
        f"{offset.method}; checked: {offset.checked}"
    )


def rt_window(rs: ResultSet, run: Run, cid: int, *, band: str | None = None) -> RtWindow | None:
    """The RT (and ion-mobility) extraction window of candidate ``cid`` in ``run``.

    Ungrouped run: the row ``cid`` of ``run_windows.parquet``, found through the dense
    index, which checks that ``candidate_id`` equals the row. Grouped run: the band given,
    else the band whose chromatogram table yields the candidate's rows
    (:meth:`ChromatogramSource.band_of`), else the band whose library row span holds it,
    else (a pooled run) the band whose table rows equal the pooled rows; then the
    band-local row ``cid - offset_b`` of that band's ``run_windows.parquet``, with
    ``0 <= local < n_b`` checked. The offsets come from :func:`band_offsets`, which checks
    them against the run's own tables.

    ``band`` is the band directory name (``'g01'``, as :attr:`ChromatogramTable.band`).
    An integer is refused with :class:`TypeError`: it could be a plan index or a loser-file
    position (:attr:`ChromatogramTable.position`), which differ when bands were skipped.

    When several bands hold the candidate with identical windows and the run's tables do
    not show which band it used, the window is returned with ``band`` None and ``source``
    naming every such band.

    Returns None when the candidate has no row (outside the table, or outside the span of
    the band given). Raises :class:`ArtifactNotFound` when the table or the band offset is
    not available (with the reason, for example a library that failed its check), and
    :class:`AmbiguousBand` when the bands give different windows and the band cannot be
    decided.
    """
    cid = int(cid)
    grouped = run.grouped
    if grouped is None:
        if band is not None:
            raise ValueError(f"{run.label} is not a grouped run; band= does not apply.")
        artifact = run.artifacts.get("run_windows")
        if artifact is None:
            raise ArtifactNotFound(f"{run.root}: the run has no run_windows table.")
        artifact.require()
        row = _window_row(rs, artifact, cid)
        if row is None:
            return None
        scope = run.name or "the run"
        return _make_window(
            cid,
            row,
            f"{_display(rs, artifact.path)} row {cid}",
            None,
            run.side_files.get("cal"),
            scope,
        )
    choice = _choose_band(rs, run, cid, band)
    names = [choice.band] if choice.band is not None else list(choice.identical)
    if not names:
        return None
    placement = _placement(rs, run)
    parts: list[str] = []
    first: tuple[dict[str, Any], Band] | None = None
    for name in names:
        chosen = _band_named(grouped, name)
        assert chosen is not None
        artifact = chosen.artifacts.get("run_windows")
        if artifact is None:
            raise ArtifactNotFound(f"{chosen.root}: band {name} has no run_windows table.")
        artifact.require()
        offset = placement.offsets.get(name)
        if offset is None:
            why = placement.unplaced.get(name, "no offset was computed")
            raise ArtifactNotFound(f"{run.root}: the ids of band {name} cannot be placed: {why}.")
        local = cid - offset.offset
        if not 0 <= local < offset.n:
            if band is not None:
                return None
            raise InconsistentData(
                f"{run.root}: candidate_id {cid} is in band {name} ({choice.how}), but outside "
                f"that band's row span [{offset.offset}, {offset.offset + offset.n})."
            )
        row = _window_row(rs, artifact, local)
        if row is None:
            raise InconsistentData(
                f"{artifact.path}: no row for band-local id {local} although the table has "
                f"{offset.n} rows."
            )
        parts.append(f"{_display(rs, artifact.path)} row {local} ({_offset_text(cid, offset)})")
        if first is None:
            first = (row, chosen)
    assert first is not None
    row, chosen = first
    if choice.band is not None:
        source = f"{parts[0]}; band chosen as {choice.how}"
        scope = f"{run.name + '/' if run.name else ''}groups/{choice.band}"
        return _make_window(cid, row, source, choice.band, chosen.side_files.get("cal"), scope)
    source = f"{' and '.join(parts)}; {choice.how}"
    scope = f"{run.name or 'the run'} (the run-level summary; the band was not determined)"
    return _make_window(cid, row, source, None, run.side_files.get("cal"), scope)
