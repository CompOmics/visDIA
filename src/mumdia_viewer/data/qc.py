"""Run QC (P1 view 6 of the spec): the signal, the acquisition scheme and the accepted
identifications of one run.

Every value is either an engine column or a viewer computation that says how it was
made. The rules:

* **TIC and base peak.** The spectra tables have no TIC column. The viewer sums the peak
  intensities of each scan (``intensity``, float32 values summed in float64) and takes
  the largest one; the base peak m/z is the m/z of the first peak with that intensity
  (the peaks of a scan are in m/z order). One pass over the ``mz`` and ``intensity``
  lists in DuckDB (column projection, streamed) computes every scan of a table; the
  result is kept in the viewer cache (``rs.cache``, never the run directory), keyed by
  the table's recorded content hash, so each table is read once. An empty peak list
  has TIC 0, base peak 0 and no base peak m/z.
* **Peaks per MS2 spectrum** are the list lengths of ``spectra_ms2``. The conversion cap
  is convert's ``--top-peaks-ms2`` (``params.top_peaks_ms2`` of the table's report; 0
  means uncapped). convert sorts a spectrum by intensity and keeps the ``N`` most
  intense peaks when it has more (``peaks_of``, ``stages/convert.rs``), so a truncated
  spectrum holds exactly ``N`` peaks: the share of spectra at the cap is the saturation
  check of docs/20.
* **Scan rate**: MS2 scans per second in RT bins. **Cycle time**: the RT step between
  two consecutive scans of one isolation window (the median per bin, or over a run).
* **Isolation windows**: :func:`.spectra.isolation_scheme`, plus the acquisition order of
  the windows: their order by the RT of their first scan.
* **Identifications** are the target rows of the run with ``run_psm_q <= t``, the PSM
  q computed within the run (in a single run ``run_psm_q`` equals ``q_value``). The
  grouped q columns are experiment-wide and are never used per run. In entrapment mode
  the targets are the real targets (:func:`.entrapment.count_classes`). The pooled
  scored table holds native identifications only: match-between-runs transfers are not
  in it.
* **Missed cleavages** follow the viewer's rule, classic trypsin: a K or R that is not
  followed by P and is not the C-terminal residue. The engine digests with
  ``digest.enzyme`` (default ``trypsin_p``, which also cuts before P) and only in FASTA
  mode; an imported library brings its own peptides.
* **Peptide length** is the number of residues of the peptidoform without its
  modification tags.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .discovery import ResultSet, Run
from .duck import sql_path
from .entrapment import count_classes
from .errors import ViewerError
from .mbr import mbr_ran
from .quant import STRIP_SQL, require_artifact, resolve_run
from .spectra import Ms1Table, ScanTable, isolation_scheme
from .units import bind_params, format_threshold

__all__ = [
    "BASE_PEAK_LABEL",
    "CACHE_NAME",
    "MISSED_CLEAVAGE_RULE",
    "RT_BIN_CHOICES",
    "TIC_LABEL",
    "Acquisition",
    "IdDistributions",
    "PeakCounts",
    "ScanSignals",
    "WindowScheme",
    "acquisition",
    "conversion_cap",
    "counts_by_run",
    "envelope_rows",
    "id_distributions",
    "ids_across_rt",
    "ids_in_rt_range",
    "missed_cleavages",
    "modification_sites",
    "peak_counts",
    "population_label",
    "rt_bin_width",
    "rt_edges",
    "scan_rate",
    "scan_signals",
    "signals_cached",
    "window_scheme",
]

# Bump when the cached arrays change meaning or layout.
CACHE_NAME = "qc_scan_signals_v1"
TIC_LABEL = (
    "TIC: the viewer's sum of the scan's peak intensities (the intensity list of the "
    "spectra table, summed in float64); the engine writes no TIC"
)
BASE_PEAK_LABEL = (
    "base peak: the largest peak intensity of the scan (the viewer's maximum of the "
    "intensity list); its m/z is the m/z of that peak"
)
MISSED_CLEAVAGE_RULE = (
    "the viewer's rule, classic trypsin: a K or R that is not followed by P and is not the "
    "C-terminal residue"
)
# RT bin widths (s) of the binned tracks: the smallest that gives at most MAX_BINS bins.
RT_BIN_CHOICES: tuple[float, ...] = (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 1200)
MAX_BINS = 90
# Percentiles of the peaks per MS2 spectrum (docs/20 reports p25, p50, p95 and max).
PEAK_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p5", 5.0),
    ("p25", 25.0),
    ("p50", 50.0),
    ("p75", 75.0),
    ("p95", 95.0),
)

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


# --------------------------------------------------------------------------- runs


def _run(rs: ResultSet, run: Run | str | int | None) -> Run:
    return resolve_run(rs, run)


def _spectra(rs: ResultSet, run: Run, level: int) -> Any:
    kind = "spectra_ms1" if level == 1 else "spectra_ms2"
    if level not in (1, 2):
        raise ValueError(f"level is 1 (MS1) or 2 (MS2), not {level!r}.")
    name = f"spectra/{kind}.parquet"
    return require_artifact(
        rs, run.artifact(kind), f"the {kind} table of {run.label}", name, run.root / name
    )


def _mzml_name(art: Any) -> str | None:
    report = getattr(art, "report", None)
    value = report.params.get("mzml") if report is not None else None
    return Path(str(value)).name if value else None


# --------------------------------------------------------------------------- scan signals


@dataclass(frozen=True, eq=False)
class ScanSignals:
    """Per-scan TIC and base peak of one spectra table, one element per row.

    ``rt`` (s), ``scan_index`` and the window ids (MS2) are the table's own columns;
    ``tic`` (float64), ``base_peak`` (float32), ``base_peak_mz`` (float32, NaN for an
    empty scan) and ``n_peaks`` (int32) are computed by the viewer (module docstring).
    ``cached`` says whether they came from the viewer cache, ``seconds`` how long the
    computation or the load took.
    """

    level: int
    run: str
    rt: np.ndarray
    scan_index: np.ndarray
    tic: np.ndarray
    base_peak: np.ndarray
    base_peak_mz: np.ndarray
    n_peaks: np.ndarray
    window_id: np.ndarray | None
    source: str
    identity: str
    cached: bool
    seconds: float
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.rt.size)

    def order(self) -> np.ndarray:
        """Rows in RT order (stable): the order in which a trace is drawn."""
        key = "_order"
        hit = self.__dict__.get(key)
        if hit is None:
            hit = np.argsort(self.rt, kind="stable")
            hit.setflags(write=False)
            object.__setattr__(self, key, hit)
        return hit


def _signals_sql() -> str:
    return (
        "SELECT coalesce(list_sum(intensity::DOUBLE[]), 0.0) AS tic, "
        "coalesce(list_max(intensity), 0.0)::FLOAT AS base_peak, "
        "mz[list_position(intensity, list_max(intensity))]::FLOAT AS base_peak_mz, "
        "coalesce(len(intensity), 0)::INTEGER AS n_peaks "
        "FROM read_parquet($path, file_row_number = true) ORDER BY file_row_number"
    )


def _compute_signals(rs: ResultSet, art: Any) -> dict[str, np.ndarray]:
    sql = _signals_sql()
    table = rs.duck.arrow(sql, bind_params(sql, {"path": sql_path(art.require())}))
    out = {
        "tic": np.ascontiguousarray(table.column("tic").to_numpy(), dtype=np.float64),
        "base_peak": np.ascontiguousarray(table.column("base_peak").to_numpy(), dtype=np.float32),
        "base_peak_mz": np.ascontiguousarray(
            table.column("base_peak_mz").to_numpy(zero_copy_only=False), dtype=np.float32
        ),
        "n_peaks": np.ascontiguousarray(table.column("n_peaks").to_numpy(), dtype=np.int32),
    }
    return out


def _cached_arrays(rs: ResultSet, art: Any) -> dict[str, np.ndarray] | None:
    hit = rs.cache.load_arrays(art.identity(), CACHE_NAME)
    if hit is None:
        return None
    rows = art.parquet().num_rows
    needed = ("tic", "base_peak", "base_peak_mz", "n_peaks", "num_rows")
    if any(k not in hit for k in needed) or int(hit["num_rows"][0]) != rows:
        return None
    if any(hit[k].shape != (rows,) for k in needed[:-1]):
        return None
    return hit


def signals_cached(rs: ResultSet, run: Run | str | int | None, level: int) -> bool:
    """True when the scan signals of the run's MS``level`` table are in the viewer cache.

    False also when the table cannot be read. The page uses it to decide whether the
    one-time pass over the peak lists has to run (behind a progress state).
    """
    try:
        art = _spectra(rs, _run(rs, run), level)
        return _cached_arrays(rs, art) is not None
    except (ViewerError, OSError):
        return False


def scan_signals(rs: ResultSet, run: Run | str | int | None, level: int) -> ScanSignals:
    """The per-scan TIC, base peak and peak count of the run's MS1 (1) or MS2 (2) table.

    The first call for a table reads its peak lists once (about 0.9 s for the 293,271
    scans of an Astral MS2 table, 1.4 GB, on a local SSD) and stores the result in the
    viewer cache; later calls, in this process or another, load it. Concurrent calls for
    one table compute it once.
    """
    r = _run(rs, run)
    art = _spectra(rs, r, level)
    t0 = time.perf_counter()
    identity = art.identity()
    arrays = _cached_arrays(rs, art)
    cached = arrays is not None
    if arrays is None:
        with _lock(f"{identity}|{CACHE_NAME}"):
            arrays = _cached_arrays(rs, art)
            cached = arrays is not None
            if arrays is None:
                arrays = _compute_signals(rs, art)
                rs.cache.save_arrays(
                    identity,
                    CACHE_NAME,
                    num_rows=np.array([art.parquet().num_rows], dtype=np.int64),
                    **arrays,
                )
    if level == 1:
        table: Any = Ms1Table.for_run(rs, r)
        window_id = None
    else:
        table = ScanTable.for_run(rs, r)
        window_id = table.window_id
    seconds = time.perf_counter() - t0
    if table.rt.size != arrays["tic"].size:
        raise ViewerError(
            f"{art.key}: the cached scan signals have {arrays['tic'].size} rows, the table "
            f"{table.rt.size}."
        )
    name = art.path.name if art.path is not None else art.key
    return ScanSignals(
        level=level,
        run=r.label,
        rt=table.rt,
        scan_index=table.scan_index,
        tic=arrays["tic"],
        base_peak=arrays["base_peak"],
        base_peak_mz=arrays["base_peak_mz"],
        n_peaks=arrays["n_peaks"],
        window_id=window_id,
        source=name,
        identity=identity,
        cached=cached,
        seconds=seconds,
        labels={"tic": TIC_LABEL, "base_peak": BASE_PEAK_LABEL},
    )


def envelope_rows(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    *,
    n_bins: int = 2000,
    max_points: int = 20_000,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """The rows to draw for a line of ``(x, y)`` over ``lo <= x <= hi``, in x order.

    Every row when the range holds at most ``max_points``; otherwise, per bin of
    ``n_bins`` equal bins, the first, the lowest, the highest and the last point (the
    M4 rule). A line through these rows looks like the full line at a plot width of
    ``n_bins`` pixels or less. ``order`` is ``argsort(x)`` when the caller has it.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if order is None:
        order = np.argsort(x, kind="stable")
    xs = x[order]
    a = int(np.searchsorted(xs, lo, side="left"))
    b = int(np.searchsorted(xs, hi, side="right"))
    if b <= a:
        return np.zeros(0, dtype=np.int64)
    rows = order[a:b].astype(np.int64)
    if rows.size <= max_points or n_bins < 1:
        return rows
    span = float(hi) - float(lo)
    if not span > 0:
        return rows[[0, -1]]
    bins = np.floor((xs[a:b] - float(lo)) / span * n_bins).astype(np.int64)
    np.clip(bins, 0, n_bins - 1, out=bins)
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:], bins.size] - 1
    values = y[rows]
    by_value = np.lexsort((values, bins))  # within each bin: lowest first, highest last
    vstarts = np.searchsorted(bins[by_value], bins[starts], side="left")
    vends = np.searchsorted(bins[by_value], bins[starts], side="right") - 1
    keep = np.unique(np.concatenate([starts, ends, by_value[vstarts], by_value[vends]]))
    return rows[keep]


# --------------------------------------------------------------------------- peaks per MS2


def conversion_cap(rs: ResultSet, run: Run | str | int | None) -> tuple[int | None, str]:
    """convert's ``--top-peaks-ms2`` for the run, and where it was read.

    First ``params.top_peaks_ms2`` of ``spectra_ms2.parquet.report.json`` (convert writes
    it with the table), else a ``--top-peaks-ms2`` in the manifest's command line. 0 means
    uncapped (the default of ``convert`` and ``run``). None when neither records it.
    """
    r = _run(rs, run)
    art = r.artifact("spectra_ms2")
    report = getattr(art, "report", None)
    if report is not None and "top_peaks_ms2" in report.params:
        try:
            return int(report.params["top_peaks_ms2"]), (
                "params.top_peaks_ms2 of spectra_ms2.parquet.report.json"
            )
        except (TypeError, ValueError):
            pass
    args = [str(a) for a in (rs.manifest.cli_args or [])]
    for i, arg in enumerate(args):
        value = None
        if arg == "--top-peaks-ms2" and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith("--top-peaks-ms2="):
            value = arg.split("=", 1)[1]
        if value is not None:
            try:
                return int(value), "--top-peaks-ms2 in the manifest's cli_args"
            except ValueError:
                break
    if args and rs.manifest.cli_args:
        return 0, "the manifest's cli_args have no --top-peaks-ms2 (the default 0, uncapped)"
    return None, "not recorded (no convert report, no command line)"


@dataclass(frozen=True)
class PeakCounts:
    """Peaks per MS2 spectrum of one run: the saturation check of ``--top-peaks-ms2``.

    ``percentiles`` maps ``p5`` ... ``p95`` to a spectrum's peak count (numpy's
    "nearest" method, so every value is the count of a spectrum), plus ``min`` and
    ``max``. ``cap`` is the conversion cap (0: uncapped; None: not recorded) and
    ``n_at_cap`` the spectra with exactly ``cap`` peaks (None without a cap);
    ``n_over_cap`` spectra with more than ``cap`` peaks contradict the cap.
    ``histogram`` has ``bin_lo``, ``bin_hi`` and ``n``.
    """

    run: str
    n_spectra: int
    n_empty: int
    n_peaks_total: int
    mean: float
    percentiles: dict[str, int]
    cap: int | None
    cap_source: str
    n_at_cap: int | None
    n_over_cap: int | None
    histogram: pd.DataFrame
    label: str
    note: str


def _nice_step(span: float, target: int) -> float:
    raw = span / max(1, target)
    if raw <= 1:
        return 1.0
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * power:
            return float(m * power)
    return float(10 * power)


def peak_counts(
    rs: ResultSet, run: Run | str | int | None, *, signals: ScanSignals | None = None
) -> PeakCounts:
    """The peaks per MS2 spectrum of a run as percentiles, a histogram and the cap check."""
    r = _run(rs, run)
    sig = signals if signals is not None else scan_signals(rs, r, 2)
    n = sig.n_peaks.astype(np.int64)
    cap, cap_source = conversion_cap(rs, r)
    if n.size == 0:
        raise ViewerError(f"{r.label}: spectra_ms2 has no spectra.")
    pct = {name: int(np.percentile(n, q, method="nearest")) for name, q in PEAK_PERCENTILES}
    pct["min"], pct["max"] = int(n.min()), int(n.max())
    top = max(int(n.max()), int(cap or 0), 1)
    step = _nice_step(top, 60)
    edges = np.arange(0.0, top + step, step)
    if edges[-1] <= top:
        edges = np.append(edges, edges[-1] + step)
    counts, _ = np.histogram(n, bins=edges)
    hist = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "n": counts})
    n_at = n_over = None
    if cap:
        n_at, n_over = int(np.count_nonzero(n == cap)), int(np.count_nonzero(n > cap))
    label = (
        f"peaks per MS2 spectrum of {r.label}: the length of each scan's peak list in "
        f"{sig.source} ({n.size:,} spectra), counted by the viewer"
    )
    if cap is None:
        note = f"The conversion cap is {cap_source}, so the spectra at a cap cannot be counted."
    elif cap == 0:
        note = (
            f"No conversion cap (top_peaks_ms2 = 0, {cap_source}): convert kept every peak, "
            "so no spectrum was truncated. The distribution is the run's own."
        )
    else:
        note = (
            f"top_peaks_ms2 = {cap} ({cap_source}): convert kept the {cap} most intense peaks "
            f"of every spectrum that had more, so the {n_at:,} spectra with exactly {cap} peaks "
            f"({100.0 * (n_at or 0) / n.size:.1f}%) are the truncated ones (docs/20)."
        )
        if n_over:
            note += (
                f" {n_over:,} spectra have more than {cap} peaks, which a cap of {cap} cannot "
                "leave: the table was not written with this cap."
            )
    return PeakCounts(
        run=r.label,
        n_spectra=int(n.size),
        n_empty=int(np.count_nonzero(n == 0)),
        n_peaks_total=int(n.sum()),
        mean=float(n.mean()),
        percentiles=pct,
        cap=cap,
        cap_source=cap_source,
        n_at_cap=n_at,
        n_over_cap=n_over,
        histogram=hist,
        label=label,
        note=note,
    )


# --------------------------------------------------------------------------- RT bins


def rt_bin_width(span: float) -> float:
    """The smallest width of :data:`RT_BIN_CHOICES` that splits ``span`` s into at most
    :data:`MAX_BINS` bins."""
    span = float(span)
    for w in RT_BIN_CHOICES:
        if span / w <= MAX_BINS:
            return float(w)
    return float(math.ceil(span / MAX_BINS))


def _rt_extent(rs: ResultSet, r: Run) -> tuple[float, str]:
    """The largest RT of the run's scans (else of its identifications) and its source."""
    best = None
    try:
        best = float(np.max(ScanTable.for_run(rs, r).rt))
        source = "the MS2 scans"
    except (ViewerError, ValueError, OSError):
        source = ""
    try:
        ms1 = Ms1Table.for_run(rs, r)
        if ms1.n:
            top = float(np.max(ms1.rt))
            best = top if best is None else max(best, top)
            source = source or "the MS1 scans"
    except (ViewerError, ValueError, OSError):
        pass
    if best is None:
        where, params = _run_filter(rs, r)
        sql = f"SELECT max(apex_rt) FROM read_parquet($path) WHERE isfinite(apex_rt){where}"
        value = rs.duck.scalar(sql, bind_params(sql, {**params, "path": _scored(rs)}))
        best = float(value) if value is not None else 0.0
        source = "the identifications' apex_rt (the run has no spectra tables)"
    return best, source


def rt_edges(rs: ResultSet, run: Run | str | int | None) -> np.ndarray:
    """Bin edges (s) shared by the binned tracks of a run: from 0 to its last scan."""
    r = _run(rs, run)
    top, _ = _rt_extent(rs, r)
    w = rt_bin_width(max(top, 1.0))
    n = max(1, math.ceil(top / w))
    edges = np.arange(n + 1, dtype=np.float64) * w
    if edges[-1] < top:
        edges = np.append(edges, edges[-1] + w)
    return edges


def _bin_of(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The bin of each value, as np.histogram counts them (the last bin is closed)."""
    b = np.searchsorted(edges, values, side="right") - 1
    b[values == edges[-1]] = edges.size - 2
    return b


def scan_rate(rs: ResultSet, run: Run | str | int | None, edges: np.ndarray) -> pd.DataFrame:
    """MS2 and MS1 scans per RT bin and the MS2 cycle time.

    Columns ``bin_lo``, ``bin_hi``, ``ms2_scans``, ``ms2_per_s`` (scans / bin width),
    ``ms1_scans``, ``ms1_per_s`` and ``cycle_s``: the median RT step between two
    consecutive scans of one isolation window, over the steps that end in the bin (NaN
    when none does).
    """
    r = _run(rs, run)
    st = ScanTable.for_run(rs, r)
    edges = np.asarray(edges, dtype=np.float64)
    width = np.diff(edges)
    if st.n:
        # The acquired part of each bin: the first and the last bin are partly covered.
        first, last = float(st.rt.min()), float(st.rt.max())
        width = np.clip(np.minimum(edges[1:], last) - np.maximum(edges[:-1], first), 0.0, None)
    ms2, _ = np.histogram(st.rt, bins=edges)
    steps, ends = [], []
    for w in st.windows["window_id"].to_numpy():
        rows = st.rows_in_window(int(w))
        if rows.size > 1:
            rt = st.rt[rows]
            steps.append(np.diff(rt))
            ends.append(rt[1:])
    cycle = np.full(width.size, np.nan)
    if steps:
        d, at = np.concatenate(steps), np.concatenate(ends)
        inside = (at >= edges[0]) & (at <= edges[-1])
        b = _bin_of(at[inside], edges)
        frame = pd.DataFrame({"b": b, "d": d[inside]})
        med = frame.groupby("b")["d"].median()
        cycle[med.index.to_numpy()] = med.to_numpy()
    try:
        ms1_rt = Ms1Table.for_run(rs, r).rt
    except (ViewerError, OSError):
        ms1_rt = np.zeros(0)
    ms1, _ = np.histogram(ms1_rt, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        ms2_rate = np.where(width > 0, ms2 / width, np.nan)
        ms1_rate = np.where(width > 0, ms1 / width, np.nan)
    df = pd.DataFrame(
        {
            "bin_lo": edges[:-1],
            "bin_hi": edges[1:],
            "acquired_s": width,
            "ms2_scans": ms2.astype(np.int64),
            "ms2_per_s": ms2_rate,
            "ms1_scans": ms1.astype(np.int64),
            "ms1_per_s": ms1_rate,
            "cycle_s": cycle,
        }
    )
    df.attrs["label"] = (
        "MS2 scans per second: the MS2 scans of each RT bin over the acquired part of the bin "
        "(from the first to the last MS2 scan); cycle time: the median RT step between "
        "consecutive scans of one isolation window (viewer-derived from rt_seconds and "
        "window_id of spectra_ms2)"
    )
    return df


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class WindowScheme:
    """The isolation windows of a run with their acquisition order and a summary.

    ``frame`` is :func:`.spectra.isolation_scheme` plus ``order`` (0-based position of
    the window in the acquisition order: windows sorted by the RT of their first scan)
    and ``first_rt``.
    """

    run: str
    frame: pd.DataFrame
    n_windows: int
    mz_lo: float
    mz_hi: float
    width_min: float
    width_median: float
    width_max: float
    n_overlaps: int
    n_gaps: int
    largest_gap: float
    largest_overlap: float
    cycle_time_s: float | None
    source: str
    label: str


def window_scheme(rs: ResultSet, run: Run | str | int | None) -> WindowScheme:
    r = _run(rs, run)
    st = ScanTable.for_run(rs, r)
    df = isolation_scheme(rs, r).copy()
    first = []
    for w in df["window_id"].to_numpy():
        rows = st.rows_in_window(int(w))
        first.append(float(st.rt[rows[0]]) if rows.size else np.inf)
    df["first_rt"] = first
    rank = np.argsort(np.asarray(first), kind="stable")
    order = np.empty(len(df), dtype=np.int64)
    order[rank] = np.arange(len(df))
    df["order"] = order
    gaps = df["overlap_with_next"].to_numpy()
    finite = gaps[np.isfinite(gaps)]
    neg, pos = finite[finite < 0], finite[finite > 0]
    widths = df["width"].to_numpy()
    cycles = df["cycle_time_s"].to_numpy()
    cycles = cycles[np.isfinite(cycles)]
    return WindowScheme(
        run=r.label,
        frame=df,
        n_windows=len(df),
        mz_lo=float(df["lower"].min()) if len(df) else float("nan"),
        mz_hi=float(df["upper"].max()) if len(df) else float("nan"),
        width_min=float(widths.min()) if widths.size else float("nan"),
        width_median=float(np.median(widths)) if widths.size else float("nan"),
        width_max=float(widths.max()) if widths.size else float("nan"),
        n_overlaps=int(pos.size),
        n_gaps=int(neg.size),
        largest_gap=float(-neg.min()) if neg.size else 0.0,
        largest_overlap=float(pos.max()) if pos.size else 0.0,
        cycle_time_s=float(np.median(cycles)) if cycles.size else None,
        source=st.windows_source,
        label=(
            "isolation windows from "
            f"{st.windows_source}; acquisition order: the windows sorted by the RT of their "
            "first MS2 scan (viewer-derived)"
        ),
    )


# --------------------------------------------------------------------------- acquisition


@dataclass(frozen=True)
class Acquisition:
    """What was acquired in a run: scan counts, RT range, windows and cycle time."""

    run: str
    mzml: str | None
    n_ms1: int | None
    n_ms2: int | None
    rt_lo: float | None
    rt_hi: float | None
    n_windows: int | None
    width_median: float | None
    cycle_time_s: float | None
    notes: tuple[str, ...]


def acquisition(rs: ResultSet, run: Run | str | int | None) -> Acquisition:
    """Scan counts, RT range and cycle time of a run, from its spectra tables (if any)."""
    r = _run(rs, run)
    notes: list[str] = []
    n_ms1 = n_ms2 = n_windows = None
    rt_lo = rt_hi = width = cycle = None
    mzml = None
    try:
        art = _spectra(rs, r, 2)
        mzml = _mzml_name(art)
        st = ScanTable.for_run(rs, r)
        n_ms2 = st.n
        if st.n:
            rt_lo, rt_hi = float(st.rt.min()), float(st.rt.max())
        ws = window_scheme(rs, r)
        n_windows, width, cycle = ws.n_windows, ws.width_median, ws.cycle_time_s
    except (ViewerError, OSError) as exc:
        notes.append(str(exc))
    try:
        art1 = _spectra(rs, r, 1)
        mzml = mzml or _mzml_name(art1)
        n_ms1 = Ms1Table.for_run(rs, r).n
    except (ViewerError, OSError) as exc:
        notes.append(str(exc))
    return Acquisition(
        run=r.label,
        mzml=mzml,
        n_ms1=n_ms1,
        n_ms2=n_ms2,
        rt_lo=rt_lo,
        rt_hi=rt_hi,
        n_windows=n_windows,
        width_median=width,
        cycle_time_s=cycle,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- identifications


def _scored(rs: ResultSet) -> str:
    return sql_path(rs.scored.require())


def _run_filter(rs: ResultSet, r: Run) -> tuple[str, dict[str, Any]]:
    if not rs.is_experiment:
        return "", {}
    return " AND source = $source", {"source": int(r.index)}


def _accepted(rs: ResultSet, r: Run, t: float, predicate: str) -> tuple[str, dict[str, Any]]:
    """WHERE text (and parameters) of the rows of ``r`` with ``predicate`` at run_psm_q <= t."""
    t = float(t)
    if not 0 < t <= 1:
        raise ValueError(f"the run_psm_q threshold must be in (0, 1], not {t!r}.")
    cls = count_classes(rs)
    where, params = _run_filter(rs, r)
    return (
        f"({predicate}) AND run_psm_q <= $t{where}",
        {**cls.params, **params, "t": t, "path": _scored(rs)},
    )


def population_label(rs: ResultSet, run: Run | str | int | None, t: float) -> str:
    """What the identification views of a run count, in words."""
    r = _run(rs, run)
    cls = count_classes(rs)
    tt = format_threshold(t)
    noun = "real target" if cls.excludes_spike_ins else "target"
    if rs.is_experiment:
        text = (
            f"accepted {noun} PSMs of run {r.label}: its rows with run_psm_q <= {tt} "
            "(PSM-level FDR within the run). The grouped q columns are experiment-wide and "
            "not used per run"
        )
    else:
        text = (
            f"accepted {noun} PSMs: rows with run_psm_q <= {tt} (PSM-level FDR within the "
            "run; in a single run run_psm_q equals q_value)"
        )
    if mbr_ran(rs):
        text += "; native identifications only, match-between-runs transfers are not included"
    if cls.note:
        text += ". " + cls.note.rstrip(".")
    return text


def ids_across_rt(
    rs: ResultSet, run: Run | str | int | None, t: float, edges: np.ndarray
) -> pd.DataFrame:
    """Accepted identifications of a run per RT bin of their ``apex_rt``.

    Columns ``bin_lo``, ``bin_hi``, ``targets`` (target rows with ``run_psm_q <= t``),
    ``decoys`` (decoy rows that pass the same cut, a diagnostic) and ``spike_ins`` when
    the library has entrapment markers. A bin holds ``bin_lo <= apex_rt < bin_hi`` (the
    last bin also its upper edge), as np.histogram counts. ``attrs`` holds ``total``
    (every accepted target row), ``outside`` (accepted targets with a non-finite apex_rt
    or one outside the edges) and the label.
    """
    r = _run(rs, run)
    edges = np.asarray(edges, dtype=np.float64)
    cls = count_classes(rs)
    key = ("qc.ids_across_rt", rs.scored.identity(), r.index, float(t), edges.tobytes(), cls.key)
    hit = rs._memo.get(key)
    if hit is not None:
        return hit.copy()
    classes = [("targets", cls.target), ("decoys", cls.decoy)]
    if cls.spike_present:
        classes.append(("spike_ins", cls.spike))
    selects = ", ".join(f"({p}) AS is_{name}" for name, p in classes)
    any_class = " OR ".join(f"({p})" for _, p in classes)
    where, params = _accepted(rs, r, t, any_class)
    sql = f"SELECT apex_rt, {selects} FROM read_parquet($path) WHERE {where}"
    table = rs.duck.arrow(sql, bind_params(sql, params))
    apex = table.column("apex_rt").to_numpy(zero_copy_only=False).astype(np.float64)
    out = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:]})
    total = outside = 0
    for name, _ in classes:
        mask = table.column(f"is_{name}").to_numpy(zero_copy_only=False).astype(bool)
        values = apex[mask]
        finite = values[np.isfinite(values)]
        counts, _ = np.histogram(finite, bins=edges)
        out[name] = counts.astype(np.int64)
        if name == "targets":
            total = int(mask.sum())
            outside = total - int(counts.sum())
    out.attrs.update(
        {
            "run": r.label,
            "threshold": float(t),
            "total": total,
            "outside": outside,
            "label": population_label(rs, r, t),
            "q_column": "run_psm_q",
        }
    )
    rs._memo[key] = out.copy()
    return out


def ids_in_rt_range(
    rs: ResultSet,
    run: Run | str | int | None,
    t: float,
    lo: float,
    hi: float,
    *,
    closed: bool = False,
    limit: int = 5000,
) -> pd.DataFrame:
    """The accepted target rows of a run with ``lo <= apex_rt < hi`` (``<= hi`` when
    ``closed``), in apex_rt order, at most ``limit``.

    Columns ``candidate_id``, ``peptidoform``, ``charge``, ``label``, ``apex_rt``,
    ``score``, ``run_psm_q``, ``protein_group`` and ``is_entrapment``. ``attrs['total']``
    is the number of rows in the range (before ``limit``).
    """
    r = _run(rs, run)
    cls = count_classes(rs)
    where, params = _accepted(rs, r, t, cls.target)
    upper = "<=" if closed else "<"
    spike = cls.spike if cls.spike_present else "false"
    sql = (
        "SELECT candidate_id, peptidoform, charge, label, apex_rt, score, run_psm_q, "
        f"protein_group, ({spike}) AS is_entrapment, count(*) OVER () AS total "
        f"FROM read_parquet($path) WHERE {where} AND apex_rt >= $lo AND apex_rt {upper} $hi "
        "ORDER BY apex_rt, candidate_id LIMIT $limit"
    )
    params.update({"lo": float(lo), "hi": float(hi), "limit": int(limit)})
    df = rs.duck.df(sql, bind_params(sql, params))
    total = int(df["total"].iloc[0]) if len(df) else 0
    df = df.drop(columns=["total"])
    df.attrs.update(
        {
            "run": r.label,
            "total": total,
            "lo": float(lo),
            "hi": float(hi),
            "closed": closed,
            "label": population_label(rs, r, t),
        }
    )
    return df


# --------------------------------------------------------------------------- distributions

_TAG = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)")
_CTERM = re.compile(r"-((?:\[[^\]]*\]|\([^)]*\))+)$")
_MISSED = re.compile(r"[KR](?!P)")
# A residue followed by one tag: the common case, read with one findall.
_RESIDUE_TAG = re.compile(r"([^\[\]()])(?:\[([^\]]*)\]|\(([^)]*)\))")


def modification_sites(peptidoform: str) -> list[tuple[str, str]]:
    """``(tag, site)`` for every modification tag of a peptidoform, in order.

    ``site`` is the modified residue, ``"N-term"`` or ``"C-term"``. Tags written before a
    ``-`` at the start are N-terminal (``[Acetyl]-PEPTIDE``), tags after a ``-`` at the
    end C-terminal, a tag after a residue modifies it; a tag at the start without a
    ``-`` is taken as N-terminal. ``DECOY_`` is ignored. ``[...]`` and ``(...)`` tags are
    read alike, and a tag is kept as written (``UNIMOD:35`` stays ``UNIMOD:35``).
    """
    rest = peptidoform[6:] if peptidoform.startswith("DECOY_") else peptidoform
    if (
        rest
        and rest[0] not in "[("
        and "-" not in rest
        and not any(p in rest for p in ("][", ")(", "](", ")["))
    ):
        # No terminal tag and no residue with two tags: every tag follows a residue.
        return [(a if a else b, aa) for aa, a, b in _RESIDUE_TAG.findall(rest)]
    out: list[tuple[str, str]] = []
    while True:
        m = _TAG.match(rest)
        if m is None or not rest[m.end() :].startswith("-"):
            break
        out.append((m.group(1) if m.group(1) is not None else m.group(2), "N-term"))
        rest = rest[m.end() + 1 :]
    cterm: list[str] = []
    cm = _CTERM.search(rest)
    if cm is not None:
        cterm = [a or b for a, b in _TAG.findall(cm.group(1))]
        rest = rest[: cm.start()]
    last: str | None = None
    pos = 0
    while pos < len(rest):
        m = _TAG.match(rest, pos)
        if m is not None:
            tag = m.group(1) if m.group(1) is not None else m.group(2)
            out.append((tag, last if last is not None else "N-term"))
            pos = m.end()
            continue
        last = rest[pos]
        pos += 1
    out.extend((tag, "C-term") for tag in cterm)
    return out


def missed_cleavages(sequence: str) -> int:
    """Missed cleavages of a plain sequence by the viewer's rule (:data:`MISSED_CLEAVAGE_RULE`)."""
    seq = sequence.upper()
    return sum(1 for m in _MISSED.finditer(seq) if m.start() < len(seq) - 1)


def _missed_sql(seq: str) -> str:
    """:func:`missed_cleavages` in SQL: K and R minus KP and RP minus a C-terminal K or R."""

    def count(s: str) -> str:
        return f"(length({seq}) - length(replace({seq}, '{s}', ''))) // {len(s)}"

    return (
        f"({count('K')} + {count('R')} - {count('KP')} - {count('RP')} - "
        f"CASE WHEN right({seq}, 1) IN ('K', 'R') THEN 1 ELSE 0 END)"
    )


@dataclass(frozen=True)
class IdDistributions:
    """Charge, length, missed-cleavage and modification counts of accepted targets.

    Every frame counts PSMs (rows). ``mods`` has ``tag`` (as written), ``site`` (residue,
    ``N-term`` or ``C-term``), ``psms`` (rows with at least one such site), ``sites``
    (sites over all rows) and ``with_site`` (rows that contain the residue: the
    denominator of the share; every row for a terminus).
    """

    run: str
    threshold: float
    n: int
    charge: pd.DataFrame
    length: pd.DataFrame
    missed: pd.DataFrame
    mods: pd.DataFrame
    n_unmodified: int
    label: str
    missed_rule: str
    notes: tuple[str, ...]


def id_distributions(rs: ResultSet, run: Run | str | int | None, t: float) -> IdDistributions:
    """Charge, peptide length, missed cleavages and modifications of a run's accepted targets."""
    r = _run(rs, run)
    cls = count_classes(rs)
    key = ("qc.id_distributions", rs.scored.identity(), r.index, float(t), cls.key)
    hit = rs._memo.get(key)
    if hit is not None:
        return hit
    where, params = _accepted(rs, r, t, cls.target)
    strip = STRIP_SQL.format(col="peptidoform")
    # The modified peptidoforms first: their residues are counted in the main pass.
    sql_mod = (
        f"SELECT peptidoform, count(*) AS n FROM read_parquet($path) WHERE {where} AND "
        "(contains(peptidoform, '[') OR contains(peptidoform, '(')) GROUP BY 1"
    )
    modified = rs.duck.df(sql_mod, bind_params(sql_mod, params))
    psms: dict[tuple[str, str], int] = {}
    sites: dict[tuple[str, str], int] = {}
    n_modified = 0
    for text, count in zip(modified["peptidoform"], modified["n"], strict=True):
        found = modification_sites(str(text))
        if not found:
            continue
        n_modified += int(count)
        for k in set(found):
            psms[k] = psms.get(k, 0) + int(count)
        for k in found:
            sites[k] = sites.get(k, 0) + int(count)
    residues = sorted({s for _, s in psms if len(s) == 1 and s.isalpha() and s.isascii()})
    has = "".join(f", count(*) FILTER (WHERE contains(seq, '{aa}')) AS r_{aa}" for aa in residues)
    sql = (
        f"WITH x AS (SELECT charge, {strip} AS seq FROM read_parquet($path) WHERE {where}) "
        f"SELECT charge, length(seq) AS len, {_missed_sql('seq')} AS mc, count(*) AS n{has} "
        "FROM x GROUP BY ALL"
    )
    combos = rs.duck.df(sql, bind_params(sql, params))
    n = int(combos["n"].sum()) if len(combos) else 0
    with_site: dict[str, int] = {"N-term": n, "C-term": n}
    with_site.update({aa: int(combos[f"r_{aa}"].sum()) for aa in residues})

    def tally(column: str, name: str) -> pd.DataFrame:
        if not len(combos):
            return pd.DataFrame({name: pd.Series([], dtype="Int64"), "n": []})
        g = combos.groupby(column, dropna=False)["n"].sum().reset_index()
        g = g.rename(columns={column: name}).sort_values(name, kind="stable")
        g[name] = g[name].astype("Int64")
        return g.reset_index(drop=True)

    mods = pd.DataFrame(
        [
            {
                "tag": tag,
                "site": site,
                "psms": psms[(tag, site)],
                "sites": sites[(tag, site)],
                "with_site": with_site.get(site, n),
            }
            for tag, site in psms
        ],
        columns=["tag", "site", "psms", "sites", "with_site"],
    )
    if len(mods):
        mods = mods.sort_values(["psms", "tag"], ascending=[False, True]).reset_index(drop=True)
    notes = [
        "Peptide length: residues of the peptidoform without its modification tags.",
        f"Missed cleavages: {MISSED_CLEAVAGE_RULE}.",
    ]
    enzyme = rs.config_get("digest", "enzyme")
    library_mode = "lib_precursors" in rs.manifest.inputs
    if library_mode:
        notes.append(
            "The run searched an imported library (input lib_precursors), so the engine's "
            "digest settings did not choose these peptides."
        )
    elif str(enzyme).lower().replace("/", "_") in ("trypsin_p", "trypsinp"):
        notes.append(
            f"The run's digest.enzyme is {enzyme}, which also cuts before P: the engine "
            "counts a K or R before P as a missed cleavage, the viewer does not."
        )
    elif enzyme:
        notes.append(f"The run's digest.enzyme is {enzyme}.")
    out = IdDistributions(
        run=r.label,
        threshold=float(t),
        n=n,
        charge=tally("charge", "charge"),
        length=tally("len", "length"),
        missed=tally("mc", "missed_cleavages"),
        mods=mods,
        n_unmodified=n - n_modified,
        label=population_label(rs, r, t),
        missed_rule=MISSED_CLEAVAGE_RULE,
        notes=tuple(notes),
    )
    rs._memo[key] = out
    return out


def counts_by_run(rs: ResultSet, t: float) -> dict[str, int]:
    """Accepted target rows of every run at ``run_psm_q <= t`` (the population of the page)."""
    cls = count_classes(rs)
    group = "source" if rs.is_experiment else "0"
    sql = (
        f"SELECT {group} AS s, count(*) FROM read_parquet($path) "
        f"WHERE ({cls.target}) AND run_psm_q <= $t GROUP BY 1"
    )
    rows = rs.duck.rows(sql, bind_params(sql, {**cls.params, "path": _scored(rs), "t": float(t)}))
    found = {int(s): int(n) for s, n in rows}
    return {r.label: found.get(r.index, 0) for r in rs.runs}
