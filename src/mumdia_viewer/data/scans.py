"""The spectrum browser (P2 view 11 of the spec): any scan of a run, and the accepted
identifications whose apex is near it.

The rules:

* **Addressing.** A scan is addressed by its ``scan_index``, the run-global counter that
  MS1 and MS2 scans share (:mod:`.spectra`), or by a retention time: the nearest scan of
  one MS level, and for MS2 optionally of one isolation window. Ties go to the earlier
  scan in RT, then to the lower row.
* **Stepping.** ``window`` steps to the previous or next scan of the same isolation
  window in RT order (:meth:`.spectra.ScanTable.step`); for an MS1 scan it is the
  previous or next MS1 scan in RT order. ``run`` steps in acquisition order: the
  previous or next ``scan_index`` of the run, MS1 or MS2.
* **Accepted identifications** are the scored rows of the run with ``run_psm_q <= t``,
  the PSM q within the run (in a single run ``run_psm_q`` equals ``q_value``), as on the
  run QC page. Grouped q columns are experiment-wide and are not used per run. Targets
  form the list; decoys that pass the same cut can be shown as a diagnostic. In
  entrapment mode the spike-ins are marked (:func:`.entrapment.count_classes`). The
  pooled scored table holds native identifications only: match-between-runs transfers
  are not in it.
* **Near the scan.** The candidate's precursor m/z lies in the scan's isolation window,
  ``lower <= precursor_mz <= upper`` in float64 and inclusive at both ends (the engine's
  covering rule), and either ``|apex_rt - rt| <= delta`` (apex mode) or
  ``elution_lo <= rt <= elution_hi`` (elution mode). For an MS1 scan there is no window,
  so only the RT rule applies. The precursor m/z is the engine's ``precursor_mz`` of the
  candidate's selected peak in ``psms_extracted``; when the run has no readable
  ``psms_extracted`` (a grouped run deletes it after pooling), the searched library's
  ``precursor_mz`` (``fragment_library_precursors``, the same value, joined on
  ``candidate_id``).
* **Fragments of a candidate** are its chromatogram rows (name, library m/z, predicted
  intensity): the fragments extraction used, so the 100-million-row library table is not
  read. They are matched to the shown scan with the extraction's tolerance and mass
  offset and the engine's predicate (:func:`.fragments.match_fragments`). The result is
  a viewer computation with the engine's rule (:attr:`.fragments.Tolerance.match_label`).
* **MS1 isotopes** of a candidate in an MS1 scan: the isotope m/z of extract
  (:func:`.spectra.isotope_mz`) and the peaks within ``extract.prec_tol_ppm``, summed with
  the engine's ``sum_near`` rule. The sum is viewer-recomputed; it equals the engine's
  ``ms1_*`` values only for the MS1 scan nearest ``apex_rt``
  (:data:`.spectra.MS1_SUM_LABEL`).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .chromatograms import CandidateChromatogram, ChromatogramSource
from .discovery import ResultSet, Run
from .duck import sql_path
from .entrapment import count_classes
from .errors import ArtifactNotFound, ViewerError
from .fragments import PeakMatch, Tolerance, extraction_tolerance, match_fragments
from .mbr import mbr_ran
from .spectra import (
    ISOTOPE_SPACING,
    MS1_SUM_LABEL,
    Ms1Table,
    ScanTable,
    Spectrum,
    _resolve_run,
    isotope_mz,
    sum_near,
)
from .units import bind_params, format_threshold

__all__ = [
    "APEX_MODE_LABEL",
    "ELUTION_MODE_LABEL",
    "ISOTOPE_SPACING",
    "LOAD_Q",
    "AcceptedSet",
    "FragmentOverlay",
    "IsotopeOverlay",
    "ScanRef",
    "accepted_set",
    "apex_histogram",
    "candidates_near",
    "default_scan",
    "fragment_overlay",
    "isotope_overlay",
    "locate",
    "matched_counts",
    "ms1_links",
    "nearest_scan",
    "population_label",
    "scan_ref",
    "step",
    "top_peaks",
]

# The loosest q of the header's threshold control: the accepted rows are loaded once at
# this cut and filtered in memory for any stricter threshold.
LOAD_Q = 0.1
# Accepted sets kept in memory per result set (one per run and load cut).
KEEP_SETS = 3

APEX_MODE_LABEL = "apex within {delta:g} s of the scan: |apex_rt - rt| <= {delta:g} s"
ELUTION_MODE_LABEL = "the scan inside the elution bounds: elution_lo <= rt <= elution_hi"

Scope = Literal["window", "run"]
NearMode = Literal["apex", "elution"]

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- scans


@dataclass(frozen=True)
class ScanRef:
    """One scan of a run: MS ``level`` (1 or 2) and ``row`` in that level's spectra table.

    ``scan_index`` is the run-global spectrum counter; ``rt`` is ``rt_seconds``. For an
    MS2 scan ``window_id``, ``lower``, ``upper`` and ``target`` are its isolation window
    (None for MS1).
    """

    level: int
    row: int
    scan_index: int
    rt: float
    window_id: int | None = None
    lower: float | None = None
    upper: float | None = None
    target: float | None = None

    def as_dict(self) -> dict[str, int]:
        return {"level": self.level, "row": self.row}


def _run(rs: ResultSet, run: Run | str | int | None) -> Run:
    if run is None:
        return rs.runs[0]
    return _resolve_run(rs, run)


def _tables(rs: ResultSet, run: Run) -> tuple[ScanTable | None, Ms1Table | None]:
    ms2 = ms1 = None
    if run.has("spectra_ms2"):
        ms2 = ScanTable.for_run(rs, run)
    if run.has("spectra_ms1"):
        ms1 = Ms1Table.for_run(rs, run)
    if ms2 is None and ms1 is None:
        raise ArtifactNotFound(f"{run.label}: the run has no readable spectra tables.")
    return ms2, ms1


def scan_ref(rs: ResultSet, run: Run | str | int | None, level: int, row: int) -> ScanRef:
    """The :class:`ScanRef` of ``row`` of the run's MS``level`` table (IndexError outside it)."""
    r = _run(rs, run)
    row = int(row)
    if int(level) == 1:
        m1 = Ms1Table.for_run(rs, r)
        if not 0 <= row < m1.n:
            raise IndexError(f"{r.label}: MS1 row {row} is outside the table ({m1.n} rows).")
        return ScanRef(1, row, int(m1.scan_index[row]), float(m1.rt[row]))
    st = ScanTable.for_run(rs, r)
    if not 0 <= row < st.n:
        raise IndexError(f"{r.label}: MS2 row {row} is outside the table ({st.n} rows).")
    return ScanRef(
        2,
        row,
        int(st.scan_index[row]),
        float(st.rt[row]),
        int(st.window_id[row]),
        float(st.lower[row]),
        float(st.upper[row]),
        float(st.target[row]),
    )


def locate(rs: ResultSet, run: Run | str | int | None, scan_index: int) -> ScanRef | None:
    """The scan with this ``scan_index`` (MS2 or MS1); None when the run has none."""
    r = _run(rs, run)
    ms2, ms1 = _tables(rs, r)
    si = int(scan_index)
    if ms2 is not None:
        row = ms2.row_of_scan_index(si)
        if row is not None:
            return scan_ref(rs, r, 2, row)
    if ms1 is not None:
        row = ms1.row_of_scan_index(si)
        if row is not None:
            return scan_ref(rs, r, 1, row)
    return None


def _nearest_in(rts: np.ndarray, t: float) -> int:
    """Position of the value nearest ``t`` in ascending ``rts``; ties to the earlier."""
    i = int(np.searchsorted(rts, t, side="left"))
    if i <= 0:
        return 0
    if i >= rts.size:
        return int(rts.size - 1)
    before, after = float(rts[i - 1]), float(rts[i])
    if t - before <= after - t:
        # The first row of an equal-RT run: searchsorted on the earlier value.
        return int(np.searchsorted(rts, before, side="left"))
    return i


def _rt_order(rs: ResultSet, run: Run, level: int) -> np.ndarray:
    """Rows of the MS``level`` table in (rt, row) order (memoised)."""
    if level == 1:
        table: Any = Ms1Table.for_run(rs, run)
    else:
        table = ScanTable.for_run(rs, run)
    key = ("scans.rt_order", run.index, run.name, level, id(table))

    def make() -> np.ndarray:
        order = np.lexsort((np.arange(table.n), table.rt)).astype(np.int64)
        order.setflags(write=False)
        return order

    return rs.memo(key, make)


def nearest_scan(
    rs: ResultSet,
    run: Run | str | int | None,
    rt: float,
    *,
    level: int = 2,
    window_id: int | None = None,
) -> ScanRef | None:
    """The scan nearest in RT to ``rt``: of MS``level``, for MS2 of ``window_id`` when given.

    Ties go to the earlier scan, then to the lower row. None when ``rt`` is not finite,
    or the level (or the window) has no scans. KeyError for an unknown window.
    """
    t = float(rt)
    if not np.isfinite(t):
        return None
    r = _run(rs, run)
    if int(level) == 1:
        m1 = Ms1Table.for_run(rs, r)
        if m1.n == 0:
            return None
        order = _rt_order(rs, r, 1)
        return scan_ref(rs, r, 1, int(order[_nearest_in(m1.rt[order], t)]))
    st = ScanTable.for_run(rs, r)
    rows = st.rows_in_window(window_id) if window_id is not None else _rt_order(rs, r, 2)
    if rows.size == 0:
        return None
    return scan_ref(rs, r, 2, int(rows[_nearest_in(st.rt[rows], t)]))


def _acquisition(rs: ResultSet, run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every scan of the run in ``scan_index`` order: (scan_index, level, row), memoised."""
    ms2, ms1 = _tables(rs, run)
    key = ("scans.acquisition", run.index, run.name, id(ms2), id(ms1))

    def make() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = []
        for level, table in ((1, ms1), (2, ms2)):
            if table is None:
                continue
            n = table.n
            parts.append(
                (
                    np.asarray(table.scan_index, dtype=np.int64),
                    np.full(n, level, dtype=np.int8),
                    np.arange(n, dtype=np.int64),
                )
            )
        si = np.concatenate([p[0] for p in parts])
        lv = np.concatenate([p[1] for p in parts])
        rw = np.concatenate([p[2] for p in parts])
        order = np.lexsort((rw, lv, si))
        out = (si[order], lv[order], rw[order])
        for a in out:
            a.setflags(write=False)
        return out

    return rs.memo(key, make)


def step(
    rs: ResultSet, run: Run | str | int | None, ref: ScanRef, n: int, *, scope: Scope = "window"
) -> ScanRef | None:
    """The ``n``-th neighbour of ``ref`` (``n`` may be negative); None beyond the ends.

    ``scope="window"``: an MS2 scan's neighbour in its isolation window, in (rt, row)
    order; an MS1 scan's neighbour among the MS1 scans in RT order. ``scope="run"``: the
    neighbour in acquisition order (``scan_index``), whatever its level.
    """
    r = _run(rs, run)
    n = int(n)
    if scope == "run":
        si, lv, rw = _acquisition(rs, r)
        pos = int(np.searchsorted(si, ref.scan_index, side="left"))
        if pos >= si.size or int(si[pos]) != ref.scan_index:
            return None
        target = pos + n
        if not 0 <= target < si.size:
            return None
        return scan_ref(rs, r, int(lv[target]), int(rw[target]))
    if scope != "window":
        raise ValueError(f"scope must be 'window' or 'run', not {scope!r}")
    if ref.level == 2:
        pick = ScanTable.for_run(rs, r).step(ref.row, n)
        return None if pick is None else scan_ref(rs, r, 2, pick.row)
    order = _rt_order(rs, r, 1)
    pos = np.flatnonzero(order == ref.row)
    if pos.size == 0:
        return None
    target = int(pos[0]) + n
    if not 0 <= target < order.size:
        return None
    return scan_ref(rs, r, 1, int(order[target]))


def ms1_links(
    rs: ResultSet, run: Run | str | int | None, ref: ScanRef
) -> dict[str, ScanRef | str | None]:
    """The MS1 scans related to an MS2 scan: ``nearest`` in RT (the engine's MS1 sample)
    and ``preceding`` (the acquisition parent, from ``ms2_to_ms1``).

    A missing or refused link table gives ``preceding`` None and its reason in
    ``preceding_note``. An MS1 scan has no links (both None).
    """
    r = _run(rs, run)
    out: dict[str, ScanRef | str | None] = {
        "nearest": None,
        "preceding": None,
        "preceding_note": "",
    }
    if ref.level != 2 or not r.has("spectra_ms1"):
        return out
    m1 = Ms1Table.for_run(rs, r)
    near = m1.nearest(ref.rt)
    out["nearest"] = scan_ref(rs, r, 1, near) if near is not None else None
    try:
        prev = m1.preceding(ref.row)
    except (ViewerError, IndexError) as exc:
        out["preceding_note"] = str(exc)
        return out
    out["preceding"] = scan_ref(rs, r, 1, prev) if prev is not None else None
    return out


def top_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    n: int,
    *,
    lo: float | None = None,
    hi: float | None = None,
    min_gap: float | None = None,
) -> np.ndarray:
    """Indexes of the ``n`` most intense peaks with ``lo <= mz <= hi``, most intense first.

    A peak closer than ``min_gap`` (m/z; default 1.5 % of the range) to a more intense
    chosen peak is skipped, so the labels of a stick plot do not overlap. Ties in
    intensity go to the lower index.
    """
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0 or n <= 0:
        return np.zeros(0, dtype=np.int64)
    a = float(mz.min()) if lo is None else float(lo)
    b = float(mz.max()) if hi is None else float(hi)
    inside = np.flatnonzero((mz >= a) & (mz <= b))
    if inside.size == 0:
        return np.zeros(0, dtype=np.int64)
    gap = 0.015 * max(b - a, 1e-9) if min_gap is None else float(min_gap)
    order = inside[np.lexsort((inside, -inten[inside]))]
    chosen: list[int] = []
    for i in order.tolist():
        if all(abs(mz[i] - mz[j]) >= gap for j in chosen):
            chosen.append(i)
            if len(chosen) >= n:
                break
    return np.asarray(chosen, dtype=np.int64)


# --------------------------------------------------------------------------- accepted rows


@dataclass(frozen=True, eq=False)
class AcceptedSet:
    """The scored rows of one run with ``run_psm_q <= load_q``, with their precursor m/z.

    ``frame`` columns: ``candidate_id``, ``peptidoform``, ``charge``, ``label``,
    ``protein_group``, ``apex_rt``, ``elution_lo``, ``elution_hi``, ``score``,
    ``run_psm_q``, ``precursor_mz``, ``is_target`` (the count population of
    :func:`.entrapment.count_classes`), ``is_entrapment``. ``mz_source`` names where the
    precursor m/z come from.
    """

    run: str
    load_q: float
    frame: pd.DataFrame
    mz_source: str
    spike_present: bool
    note: str = ""
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frame)


def _extracted_usable(run: Run) -> Any:
    art = run.artifact("psms_extracted")
    if art is None or not art.usable:
        return None
    try:
        names = art.parquet().schema.names
    except ViewerError:
        return None
    return art if {"candidate_id", "peak_rank", "precursor_mz"} <= set(names) else None


def _library(rs: ResultSet, run: Run) -> Any:
    art = run.artifacts.get("fragment_library_precursors")
    if art is None:
        art = rs.extra.get("fragment_library_precursors")
    if art is None or not art.usable:
        return None
    try:
        names = art.parquet().schema.names
    except ViewerError:
        return None
    return art if {"candidate_id", "precursor_mz"} <= set(names) else None


def _load_accepted(rs: ResultSet, run: Run, load_q: float) -> AcceptedSet:
    cls = count_classes(rs)
    params: dict[str, Any] = {**cls.params, "scored": sql_path(rs.scored.require()), "q": load_q}
    where = "run_psm_q <= $q"
    if rs.is_experiment:
        where += " AND source = $source"
        params["source"] = int(run.index)
    inner = (
        "SELECT candidate_id, peptidoform, charge, label, protein_group, apex_rt, elution_lo, "
        "elution_hi, score, run_psm_q, selected_peak_rank, "
        f"({cls.target}) AS is_target, ({cls.spike}) AS is_entrapment "
        f"FROM read_parquet($scored) WHERE {where}"
    )
    extracted = _extracted_usable(run)
    if extracted is not None:
        params["src"] = sql_path(extracted.require())
        sql = (
            f"WITH s AS ({inner}) SELECT s.* EXCLUDE (selected_peak_rank), e.precursor_mz "
            "FROM s LEFT JOIN (SELECT candidate_id, peak_rank, precursor_mz FROM "
            "read_parquet($src)) e ON e.candidate_id = s.candidate_id AND "
            "e.peak_rank = coalesce(s.selected_peak_rank, 0)"
        )
        source = "precursor_mz of the selected peak in psms_extracted.parquet"
    else:
        library = _library(rs, run)
        if library is None:
            raise ArtifactNotFound(
                f"{run.label}: no precursor m/z: the run has no readable psms_extracted and "
                "no readable library precursor table."
            )
        params["src"] = sql_path(library.require())
        sql = (
            f"WITH s AS ({inner}) SELECT s.* EXCLUDE (selected_peak_rank), l.precursor_mz "
            "FROM s LEFT JOIN (SELECT candidate_id, precursor_mz FROM read_parquet($src)) l "
            "ON l.candidate_id = s.candidate_id"
        )
        name = library.path.name if library.path is not None else "the library"
        source = (
            f"precursor_mz of the searched library ({name}), joined on candidate_id; "
            "psms_extracted is not available in this run"
        )
    df = rs.duck.df(sql, bind_params(sql, params))
    df = df.sort_values(["apex_rt", "candidate_id"], kind="stable").reset_index(drop=True)
    for col in ("apex_rt", "elution_lo", "elution_hi", "score", "run_psm_q", "precursor_mz"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    df["candidate_id"] = df["candidate_id"].astype(np.int64)
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce").fillna(0).astype(np.int64)
    df["is_target"] = df["is_target"].fillna(False).astype(bool)
    df["is_entrapment"] = df["is_entrapment"].fillna(False).astype(bool)
    arrays = {
        c: df[c].to_numpy()
        for c in (
            "apex_rt",
            "elution_lo",
            "elution_hi",
            "precursor_mz",
            "run_psm_q",
            "is_target",
            "label",
        )
    }
    arrays["is_decoy"] = (df["label"] == "decoy").to_numpy()
    missing = int(df["precursor_mz"].isna().sum())
    note = (
        f"{missing:,} accepted rows have no precursor m/z and are never listed for an MS2 scan."
        if missing
        else ""
    )
    return AcceptedSet(
        run=run.label,
        load_q=load_q,
        frame=df,
        mz_source=source,
        spike_present=cls.spike_present,
        note=note,
        arrays=arrays,
    )


_SETS: dict[int, OrderedDict] = {}


def accepted_set(
    rs: ResultSet, run: Run | str | int | None, threshold: float = LOAD_Q
) -> AcceptedSet:
    """The accepted rows of a run, loaded at ``max(threshold, LOAD_Q)`` and kept in memory.

    One DuckDB query (column projection on the scored table, joined to the precursor m/z
    source) per run; the last :data:`KEEP_SETS` sets of a result set are kept.
    """
    r = _run(rs, run)
    t = float(threshold)
    if not 0 < t <= 1:
        raise ValueError(f"the run_psm_q threshold must be in (0, 1], not {t!r}.")
    load_q = max(t, LOAD_Q)
    key = (rs.scored.identity(), r.index, r.name, load_q)
    with _LOCK:
        sets = _SETS.setdefault(id(rs), OrderedDict())
        hit = sets.get(key)
        if hit is not None:
            sets.move_to_end(key)
            return hit
    made = _load_accepted(rs, r, load_q)
    with _LOCK:
        sets = _SETS.setdefault(id(rs), OrderedDict())
        sets[key] = made
        sets.move_to_end(key)
        while len(sets) > KEEP_SETS:
            sets.popitem(last=False)
    return made


def population_label(rs: ResultSet, run: Run | str | int | None, t: float) -> str:
    """What the candidate list holds, in words (the row unit and the q column)."""
    r = _run(rs, run)
    cls = count_classes(rs)
    noun = "real target" if cls.excludes_spike_ins else "target"
    where = f"run {r.label}" if rs.is_experiment else "the run"
    text = (
        f"accepted {noun} PSMs of {where}: scored rows with run_psm_q <= {format_threshold(t)} "
        "(PSM-level FDR within the run"
        + ("; in a single run run_psm_q equals q_value)" if not rs.is_experiment else ")")
    )
    if mbr_ran(rs):
        text += "; native identifications only, match-between-runs transfers are not included"
    if cls.note:
        text += ". " + cls.note.rstrip(".")
    return text


def candidates_near(
    rs: ResultSet,
    run: Run | str | int | None,
    ref: ScanRef,
    t: float,
    *,
    delta: float = 5.0,
    mode: NearMode = "apex",
    include_decoys: bool = False,
) -> pd.DataFrame:
    """The accepted identifications near scan ``ref`` (see the module docstring).

    Rows with ``run_psm_q <= t`` of the count population (targets; real targets in
    entrapment mode), plus the decoys that pass the cut when ``include_decoys``. For an
    MS2 scan the precursor m/z must lie in the scan's isolation window. ``mode="apex"``
    keeps ``|apex_rt - rt| <= delta``; ``mode="elution"`` keeps
    ``elution_lo <= rt <= elution_hi``. Columns: those of :class:`AcceptedSet` plus
    ``delta_rt`` (``apex_rt - rt``, s), ordered by ``|delta_rt|`` then ``candidate_id``.
    ``attrs`` has ``label`` (the population), ``rule`` (the nearness rule), ``mz_source``
    and ``note``.
    """
    acc = accepted_set(rs, run, t)
    a = acc.arrays
    rt = float(ref.rt)
    keep = a["run_psm_q"] <= float(t)
    cls_mask = a["is_target"] | (a["is_decoy"] if include_decoys else False)
    keep &= cls_mask
    if mode == "apex":
        d = float(delta)
        if not np.isfinite(d) or d < 0:
            raise ValueError(f"delta must be a finite number of seconds >= 0, not {delta!r}")
        with np.errstate(invalid="ignore"):
            keep &= np.abs(a["apex_rt"] - rt) <= d
        rule = APEX_MODE_LABEL.format(delta=d)
    elif mode == "elution":
        with np.errstate(invalid="ignore"):
            keep &= (a["elution_lo"] <= rt) & (rt <= a["elution_hi"])
        rule = ELUTION_MODE_LABEL
    else:
        raise ValueError(f"mode must be 'apex' or 'elution', not {mode!r}")
    if ref.level == 2 and ref.lower is not None and ref.upper is not None:
        pmz = a["precursor_mz"]
        with np.errstate(invalid="ignore"):
            keep &= (float(ref.lower) <= pmz) & (pmz <= float(ref.upper))
        rule += (
            f"; precursor_mz in the scan's isolation window {ref.lower:.4f} to {ref.upper:.4f} "
            "(lower <= precursor_mz <= upper)"
        )
    else:
        rule += "; an MS1 scan has no isolation window, so every precursor m/z is kept"
    out = acc.frame.loc[keep].copy()
    out["delta_rt"] = out["apex_rt"] - rt
    out["_abs"] = out["delta_rt"].abs()
    out = out.sort_values(["_abs", "candidate_id"], kind="stable").drop(columns="_abs")
    out = out.reset_index(drop=True)
    out.attrs.update(
        {
            "label": population_label(rs, run, t),
            "rule": rule,
            "mz_source": acc.mz_source,
            "note": acc.note,
            "threshold": float(t),
            "include_decoys": bool(include_decoys),
        }
    )
    return out


def apex_histogram(
    rs: ResultSet,
    run: Run | str | int | None,
    t: float,
    edges: np.ndarray,
    *,
    window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Accepted targets per RT bin of their ``apex_rt`` (np.histogram's bins).

    The population of :func:`candidates_near` (targets with ``run_psm_q <= t``); with
    ``window`` only the precursors with ``lower <= precursor_mz <= upper``.
    """
    a = accepted_set(rs, run, t).arrays
    keep = (a["run_psm_q"] <= float(t)) & a["is_target"]
    if window is not None:
        pmz = a["precursor_mz"]
        with np.errstate(invalid="ignore"):
            keep &= (float(window[0]) <= pmz) & (pmz <= float(window[1]))
    values = a["apex_rt"][keep]
    counts, _ = np.histogram(values[np.isfinite(values)], bins=np.asarray(edges, dtype=np.float64))
    return counts.astype(np.int64)


def default_scan(
    rs: ResultSet, run: Run | str | int | None, t: float
) -> tuple[ScanRef, int | None]:
    """Where the browser opens without an address: the apex scan of the run's best target.

    The accepted target with the highest score (ties to the lower ``candidate_id``) whose
    precursor m/z is covered by an isolation window, and its apex scan
    (:meth:`.spectra.ScanTable.apex_scan`). Without one, the MS2 scan in the middle of
    the run (or the first MS1 scan when the run has no MS2 table); the candidate is then
    None.
    """
    r = _run(rs, run)
    ms2, ms1 = _tables(rs, r)
    if ms2 is not None:
        try:
            frame = accepted_set(rs, r, t).frame
        except ViewerError:
            frame = None
        if frame is not None and len(frame):
            ok = frame[
                (frame["run_psm_q"] <= float(t))
                & frame["is_target"]
                & frame["precursor_mz"].notna()
            ]
            ok = ok.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
            for row in ok.head(20).itertuples(index=False):
                pick = ms2.apex_scan(float(row.precursor_mz), float(row.apex_rt))
                if pick is not None:
                    return scan_ref(rs, r, 2, pick.row), int(row.candidate_id)
        if ms2.n:
            order = _rt_order(rs, r, 2)
            return scan_ref(rs, r, 2, int(order[order.size // 2])), None
    assert ms1 is not None
    return scan_ref(rs, r, 1, 0), None


# --------------------------------------------------------------------------- overlays


@dataclass(frozen=True, eq=False)
class FragmentOverlay:
    """A candidate's library fragments matched to one MS2 scan.

    ``fragments`` has one row per fragment row of the candidate's chromatogram, in file
    order: ``name``, ``ion``, ``ordinal``, ``charge``, ``theo_mz``,
    ``predicted_intensity``, ``observed_in_xic``. ``matches`` index it
    (:class:`.fragments.PeakMatch`); ``label`` is the tolerance's match label.
    """

    candidate_id: int
    chromatogram: CandidateChromatogram
    fragments: pd.DataFrame
    matches: list[PeakMatch]
    tolerance: Tolerance
    label: str

    @property
    def n_library(self) -> int:
        return len(self.fragments)

    @property
    def n_matched(self) -> int:
        return len(self.matches)


def _fragment_table(chrom: CandidateChromatogram) -> pd.DataFrame:
    frags = chrom.fragments()
    return pd.DataFrame(
        {
            "name": [t.frag_name for t in frags],
            "ion": [t.ion for t in frags],
            "ordinal": [t.ordinal for t in frags],
            "charge": [t.fragment_charge for t in frags],
            "theo_mz": np.asarray([t.frag_mz for t in frags], dtype=np.float64),
            "predicted_intensity": np.asarray(
                [t.predicted_intensity for t in frags], dtype=np.float64
            ),
            "observed_in_xic": [t.observed for t in frags],
        }
    )


def _tolerance(rs: ResultSet, run: Run, chrom: CandidateChromatogram) -> Tolerance:
    key = ("scans.tolerance", run.index, run.name, chrom.band)
    return rs.memo(key, lambda: extraction_tolerance(rs, run, band=chrom.band))


def fragment_overlay(
    rs: ResultSet, run: Run | str | int | None, cid: int, spectrum: Spectrum
) -> FragmentOverlay | None:
    """The fragments of candidate ``cid`` matched to ``spectrum``.

    None when the candidate has no chromatogram rows.
    """
    r = _run(rs, run)
    chrom = ChromatogramSource.for_run(rs, r).read(int(cid))
    if chrom is None:
        return None
    tol = _tolerance(rs, r, chrom)
    table = _fragment_table(chrom)
    matches = match_fragments(
        spectrum.mz, spectrum.intensity, table["theo_mz"].to_numpy(dtype=np.float64), tol
    )
    return FragmentOverlay(
        candidate_id=int(cid),
        chromatogram=chrom,
        fragments=table,
        matches=matches,
        tolerance=tol,
        label=tol.match_label,
    )


def matched_counts(
    rs: ResultSet, run: Run | str | int | None, cids: Any, spectrum: Spectrum
) -> dict[int, tuple[int, int]]:
    """``{cid: (matched, library)}``: how many of each candidate's fragments match ``spectrum``.

    The same matching as :func:`fragment_overlay`; candidates without chromatogram rows
    are left out.
    """
    out: dict[int, tuple[int, int]] = {}
    r = _run(rs, run)
    src = ChromatogramSource.for_run(rs, r)
    for cid in [int(c) for c in cids]:
        chrom = src.read(cid)
        if chrom is None:
            continue
        theo = np.asarray([t.frag_mz for t in chrom.fragments()], dtype=np.float64)
        matches = match_fragments(spectrum.mz, spectrum.intensity, theo, _tolerance(rs, r, chrom))
        out[cid] = (len(matches), int(theo.size))
    return out


@dataclass(frozen=True)
class IsotopeOverlay:
    """A candidate's precursor isotopes in one MS1 scan.

    One row per isotope ``k`` (-1, 0, 1, 2) in ``rows``: ``k``, ``mz`` (extract's isotope
    m/z), ``lo`` and ``hi`` (the ``prec_tol_ppm`` bounds), ``n_peaks`` (peaks inside) and
    ``sum`` (their float32 sum by the engine's ``sum_near`` rule; viewer-recomputed,
    :data:`.spectra.MS1_SUM_LABEL`). ``tol_ppm`` and ``tol_source`` say which tolerance
    was used.
    """

    candidate_id: int
    precursor_mz: float
    charge: int
    tol_ppm: float
    tol_source: str
    rows: tuple[dict[str, float], ...]
    label: str = MS1_SUM_LABEL


def isotope_overlay(
    rs: ResultSet, cid: int, precursor_mz: float, charge: int, spectrum: Spectrum
) -> IsotopeOverlay | None:
    """The isotopes -1 to 2 of a precursor in an MS1 ``spectrum``; None without a charge or m/z."""
    pmz, z = float(precursor_mz), int(charge)
    if not np.isfinite(pmz) or z <= 0:
        return None
    configured = rs.config_get("extract", "prec_tol_ppm")
    try:
        tol = float(configured)
        source = f"config_json extract.prec_tol_ppm = {tol:g} ppm"
    except (TypeError, ValueError):
        tol = 20.0
        source = "engine default extract.prec_tol_ppm = 20 ppm (not recorded in config_json)"
    mz64 = np.asarray(spectrum.mz, dtype=np.float32).astype(np.float64)
    rows = []
    for k in (-1, 0, 1, 2):
        m = isotope_mz(pmz, z, k)
        d = m * tol * 1e-6
        lo, hi = m - d, m + d
        s = int(np.searchsorted(mz64, lo, side="left"))
        e = int(np.searchsorted(mz64, hi, side="right"))
        rows.append(
            {
                "k": float(k),
                "mz": m,
                "lo": lo,
                "hi": hi,
                "n_peaks": float(max(e - s, 0)),
                "sum": sum_near(spectrum.mz, spectrum.intensity, m, tol),
            }
        )
    return IsotopeOverlay(int(cid), pmz, z, tol, source, tuple(rows))
