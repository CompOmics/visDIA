"""Spectra of one run: the MS2 scan table, the MS1 scans and the isolation-window scheme.

The engine writes four spectra tables to ``<run>/spectra/`` (convert stage, schema 1 in
MuMDIA 0.5.0): ``spectra_ms2.parquet``, ``spectra_ms1.parquet``,
``isolation_windows.parquet`` and ``ms2_to_ms1.parquet``. The MS2 and MS1 tables are
stored in row groups of 2,048 scans. Their peak lists are ``large_list<float32>`` (older
files may hold ``list<float32>``; both are read).

The rules below are the engine's own (MuMDIA 0.5.0 source, verified on the fixtures and
on real Astral data):

* A precursor is covered by every isolation window with ``lower <= precursor_mz <= upper``,
  compared in float64 and inclusive at both ends. Adjacent windows can touch or overlap,
  so a precursor can be covered by 0, 1 or 2 windows.
* The apex scan of an identification is the covering-window MS2 scan whose
  ``rt_seconds`` equals ``apex_rt`` bit for bit (``demix_apex_scan`` in extract); the
  first in file order when several do.
* The XIC grid of a candidate is the RTs of the covering windows' scans with
  ``rt_lo <= rt <= rt_hi``, sorted and de-duplicated. In window-grid mode
  (``extract.emit_window_grid = true``, the default and the mode of all verified data)
  the chromatogram axis is that grid cast to float32, so point ``k`` of every trace of
  the candidate is scan ``grid_rows(...)[k]``. In sparse mode
  (``extract.emit_window_grid = false``) each fragment's axis holds only the scans where
  the fragment was observed and no MS1 rows are written; the grid is then not the axis.
  :attr:`ScanTable.grid_label` says which mode the run used.
* ``scan_index`` is a run-global counter shared by MS1 and MS2 scans. It is not the row
  number of a table.
* The MS1 scan the engine samples for the ``ms1_*`` columns of ``psms_extracted`` (and,
  in window-grid mode, for the ``ms1_*`` XIC rows) is the MS1 scan nearest in RT, ties
  to the earlier scan. ``ms2_to_ms1`` gives the preceding MS1 (the acquisition parent).
  The two differ for about half of the MS2 scans.

Only scalar columns are loaded whole (about 15 MB for an Astral run). Peak lists are
read one row group at a time. The spectra keep their own small row-group cache
(:data:`SPECTRUM_CACHE`: at most 4 row groups and 96 MiB), so stepping through
neighbouring scans is cheap and an Astral row group (2 to 21 MB decoded) does not fill
the process-wide cache. No file handle stays open between calls.
"""

from __future__ import annotations

import operator
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .artifacts import Artifact
from .discovery import Run
from .errors import ArtifactNotFound, InconsistentData, LayoutError
from .pqio import ParquetHandle, list_values
from .schemas import check_columns

if TYPE_CHECKING:
    from .discovery import ResultSet

__all__ = [
    "APEX_EXACT_LABEL",
    "APEX_NEAREST_LABEL",
    "GRID_AXIS_LABEL",
    "GRID_UNKNOWN_LABEL",
    "ISOTOPE_SPACING",
    "MS1_SUM_LABEL",
    "NEAREST_MS1_LABEL",
    "PRECEDING_MS1_LABEL",
    "SPARSE_GRID_LABEL",
    "SPECTRUM_CACHE",
    "Ms1Table",
    "ScanPick",
    "ScanTable",
    "Spectrum",
    "SpectrumCache",
    "isolation_scheme",
    "isotope_mz",
    "sum_near",
]

APEX_EXACT_LABEL = "apex scan (rt_seconds equals apex_rt exactly; the scan the engine used)"
APEX_NEAREST_LABEL = "nearest scan in RT (approximate: no scan has exactly this RT)"
NEAREST_MS1_LABEL = (
    "nearest MS1 scan in RT (ties to the earlier scan); the scan the engine samples for the "
    "psms_extracted ms1_* columns and, in window-grid mode, the ms1_* XIC rows"
)
PRECEDING_MS1_LABEL = "preceding MS1 (acquisition parent, from ms2_to_ms1)"
# A viewer computation. The engine's own values are the psms_extracted ms1_* columns
# (at apex_rt) and the ms1_mono, ms1_iso1, ms1_iso2 rows of the chromatogram table.
MS1_SUM_LABEL = (
    "viewer-recomputed (engine formula sum_near): MS1 peak intensity within "
    "extract.prec_tol_ppm of the isotope m/z, float32 sum"
)
GRID_AXIS_LABEL = (
    "window grid (extract.emit_window_grid = true): float32 of these scans' RTs is the "
    "candidate's chromatogram axis"
)
SPARSE_GRID_LABEL = (
    "sparse mode (extract.emit_window_grid = false): these are the window-grid scans, not "
    "the chromatogram axis; each fragment's axis holds only the scans where it was observed"
)
GRID_UNKNOWN_LABEL = (
    "window grid; it is the chromatogram axis only under extract.emit_window_grid = true, "
    "which is not known for this table"
)
# mumdia-core constants.rs ISOTOPE_SPACING (Da).
ISOTOPE_SPACING = 1.003354835

_MS2_SCALARS = (
    "scan_index",
    "rt_seconds",
    "window_id",
    "window_target",
    "window_lower",
    "window_upper",
    "precursor_mz",
    "precursor_charge",
)
_WINDOW_COLUMNS = ("window_id", "target", "lower", "upper")
_EMPTY_ROWS = np.zeros(0, dtype=np.int64)
_EMPTY_ROWS.setflags(write=False)
_MEMO_LOCK = threading.Lock()


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class ScanPick:
    """One MS2 scan chosen for a target RT.

    ``row`` is the row in ``spectra_ms2.parquet``; ``scan_index`` is the run-global
    spectrum counter, which differs from the row. ``exact`` is True when the scan's
    ``rt_seconds`` equals the target RT bit for bit. ``delta_rt`` is
    ``rt - target`` in seconds.
    """

    row: int
    scan_index: int
    rt: float
    window_id: int
    window_lower: float
    window_upper: float
    exact: bool
    delta_rt: float

    @property
    def label(self) -> str:
        if self.exact:
            return "rt_seconds equals the requested RT exactly"
        return f"{self.delta_rt:+.3f} s from the requested RT (approximate)"


@dataclass(frozen=True, eq=False)
class Spectrum:
    """The peaks of one MS1 or MS2 scan, as stored by convert.

    ``mz`` and ``intensity`` are float32 arrays of equal length, m/z ascending. Widen
    ``mz`` to float64 before ppm arithmetic. The window and precursor fields are None
    for an MS1 scan; ``precursor_charge`` is also None when the MS2 scan records none
    (every Astral scan). ``native_id`` is the mzML nativeID (MS2 ``id`` column).
    """

    level: int
    row: int
    scan_index: int
    rt: float
    mz: np.ndarray
    intensity: np.ndarray
    native_id: str | None = None
    window_id: int | None = None
    window_lower: float | None = None
    window_upper: float | None = None
    window_target: float | None = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None

    @property
    def n_peaks(self) -> int:
        return int(self.mz.size)

    @property
    def label(self) -> str:
        kind = f"MS{self.level}"
        return f"{kind} scan_index {self.scan_index} (row {self.row}), RT {self.rt:.3f} s"


# --------------------------------------------------------------------------- helpers


def _as_handle(source: ParquetHandle | Path | str) -> ParquetHandle:
    return source if isinstance(source, ParquetHandle) else ParquetHandle(Path(source))


def _as_index(value: object, accepted: str) -> int:
    """``value`` as a Python int. Any integer type is accepted (numpy included); a bool is not.

    ``accepted`` names what the caller may pass, for the error message.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"expected {accepted}, got the bool {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"expected {accepted}, got {type(value).__name__} {value!r}") from None


def _resolve_run(rs: ResultSet, run: Run | str | int) -> Run:
    """A run object from a run, a run name or a ``source`` index.

    The index may be any integer type, for example a ``numpy.uint32`` read from the
    ``source`` column of a pooled scored table.
    """
    if isinstance(run, Run):
        return run
    if isinstance(run, str):
        return rs.run(run)
    return rs.run(_as_index(run, "a Run, a run name (str) or a source index (int)"))


def _require(run: Run, kind: str) -> Artifact:
    artifact = run.artifact(kind)
    if artifact is None:
        raise ArtifactNotFound(f"{run.label}: the run has no {kind} table.")
    artifact.require()
    return artifact


def _numeric(table: pa.Table, name: str, dtype: Any, *, where: str, fill: Any = None) -> np.ndarray:
    col = table.column(name)
    if col.null_count:
        if fill is None:
            raise InconsistentData(f"{where}: column {name} has {col.null_count} null values.")
        col = pc.fill_null(col, pa.scalar(fill, type=col.type))
    values = col.to_numpy()
    return np.ascontiguousarray(values, dtype=dtype)


def _readonly(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


def _peak_list(table: pa.Table, name: str, local: int, *, where: str) -> np.ndarray:
    """One row of a float32 list column as a float32 array (a null list is empty)."""
    col = table.column(name)
    typ = col.type
    if not (pa.types.is_list(typ) or pa.types.is_large_list(typ)):
        raise LayoutError(f"{where}: column {name} is {typ}, not a list of float32.")
    if not pa.types.is_float32(typ.value_type):
        raise LayoutError(f"{where}: column {name} holds {typ.value_type}, not float32.")
    values, _ = list_values(col.slice(local, 1))
    return np.array(values, dtype=np.float32)  # a copy: it holds no reference to the table


def _release_arrow_memory() -> None:
    """Return the Arrow pool's unused memory to the system (T9 R2: about 2 to 10 ms)."""
    pa.default_memory_pool().release_unused()


class SpectrumCache:
    """A small LRU of decoded peak-list row groups, used by the spectra tables only.

    The process-wide row-group cache of :mod:`pqio` (512 MB) suits narrow tables. An
    Astral MS2 row group is 2 to 21 MB decoded and an MS1 row group up to 47 MB, so a
    browsing session would fill that cache with spectra. This cache keeps at most
    ``max_groups`` row groups and ``max_bytes`` bytes (defaults 4 and 96 MiB, the
    spectrum budget of T9 R10 and R15). A row group larger than half of ``max_bytes``
    is read but not kept.

    pyarrow keeps freed memory in its pool. Once ``release_bytes`` of decoded peak data
    have been dropped (evicted, or read without caching), the pool's unused memory is
    returned to the system. The attributes may be changed at run time.
    """

    def __init__(
        self,
        max_groups: int = 4,
        max_bytes: int = 96 * 2**20,
        release_bytes: int = 32 * 2**20,
    ) -> None:
        self.max_groups = max_groups
        self.max_bytes = max_bytes
        self.release_bytes = release_bytes
        self._items: OrderedDict[tuple, pa.Table] = OrderedDict()
        self._bytes = 0
        self._dropped = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        """Decoded bytes held."""
        return self._bytes

    def get(self, key: tuple) -> pa.Table | None:
        with self._lock:
            table = self._items.get(key)
            if table is not None:
                self._items.move_to_end(key)  # most recently used last
            return table

    def keeps(self, nbytes: int) -> bool:
        """True when a row group of ``nbytes`` decoded bytes would be kept."""
        return self.max_groups > 0 and nbytes <= self.max_bytes // 2

    def put(self, key: tuple, table: pa.Table) -> bool:
        """Keep ``table``, evicting the least recently used groups beyond the budget.

        Returns False, and keeps nothing, when the table is too large (:meth:`keeps`);
        the caller then counts it with :meth:`dropped` once it no longer uses it.
        """
        size = table.nbytes
        if not self.keeps(size):
            return False
        evicted: list[pa.Table] = []
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
                evicted.append(old)
            self._items[key] = table
            self._bytes += size
            while self._items and (
                len(self._items) > self.max_groups or self._bytes > self.max_bytes
            ):
                _, oldest = self._items.popitem(last=False)
                self._bytes -= oldest.nbytes
                evicted.append(oldest)
        freed = sum(t.nbytes for t in evicted)
        del evicted, old
        if freed:
            self.dropped(freed)
        return True

    def dropped(self, nbytes: int) -> None:
        """Count decoded bytes that are no longer referenced; release the pool when due."""
        with self._lock:
            self._dropped += int(nbytes)
            due = self._dropped >= self.release_bytes
            if due:
                self._dropped = 0
        if due:
            _release_arrow_memory()

    def clear(self) -> None:
        """Drop every row group and return the freed memory to the system."""
        with self._lock:
            self._items.clear()
            self._bytes = 0
            self._dropped = 0
        _release_arrow_memory()


SPECTRUM_CACHE = SpectrumCache()


def _read_peaks(
    handle: ParquetHandle, row: int, *, with_id: bool, where: str, cached: bool
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """The m/z and intensity arrays of one row, and its ``id`` when ``with_id``.

    The row group comes from :data:`SPECTRUM_CACHE` when ``cached``; otherwise it is
    read, sliced and dropped.
    """
    offsets = handle.row_group_offsets()
    total = int(offsets[-1])
    if not 0 <= row < total:
        raise IndexError(f"{where}: row {row} is outside the table ({total} rows).")
    rg = int(np.searchsorted(offsets, row, side="right") - 1)
    local = row - int(offsets[rg])
    columns = ["mz", "intensity", "id"] if with_id else ["mz", "intensity"]
    key = (str(handle.path), handle.stamp, rg, tuple(columns))
    table = SPECTRUM_CACHE.get(key) if cached else None
    kept = table is not None
    if table is None:
        # Never through the process-wide row-group cache (see SpectrumCache).
        table = handle.read_row_group(rg, columns, cached=False)
        kept = cached and SPECTRUM_CACHE.put(key, table)
    size = table.nbytes
    try:
        mz = _peak_list(table, "mz", local, where=where)
        intensity = _peak_list(table, "intensity", local, where=where)
        native_id = None
        if with_id and "id" in table.column_names:
            value = table.column("id")[local].as_py()
            native_id = None if value is None else str(value)
    finally:
        del table
        if not kept:
            SPECTRUM_CACHE.dropped(size)
    if mz.size != intensity.size:
        raise InconsistentData(
            f"{where}: row {row} has {mz.size} m/z values but {intensity.size} intensities."
        )
    _readonly(mz, intensity)
    return mz, intensity, native_id


class _ScanLookup:
    """``scan_index`` to row.

    The engine writes ``scan_index`` strictly increasing in file order, so a binary
    search on the column suffices; other files get an argsort.
    """

    def __init__(self, scan_index: np.ndarray) -> None:
        if scan_index.size < 2 or bool(np.all(scan_index[1:] > scan_index[:-1])):
            self._order: np.ndarray | None = None
            self._keys = scan_index
            return
        order = np.argsort(scan_index, kind="stable")
        keys = scan_index[order]
        if bool(np.any(keys[1:] == keys[:-1])):
            raise InconsistentData("scan_index values repeat; a scan cannot be addressed by index.")
        self._order, self._keys = order, keys

    def row(self, value: int) -> int | None:
        i = int(np.searchsorted(self._keys, value, side="left"))
        if i >= self._keys.size or int(self._keys[i]) != value:
            return None
        return i if self._order is None else int(self._order[i])


# --------------------------------------------------------------------------- MS2


class ScanTable:
    """The scalar columns of one run's MS2 table, loaded once, and the scan rules.

    Arrays (one element per row of ``spectra_ms2.parquet``, read-only):
    ``scan_index`` (int64), ``rt`` (float64 s), ``window_id`` (int64), ``lower``,
    ``upper``, ``target`` (float64 m/z), ``precursor_mz`` (float64, NaN where null) and
    ``charge`` (int32, 0 where null). ``windows`` is the isolation-window table.

    ``emit_window_grid`` is the run's ``config_json.extract.emit_window_grid``: True when
    the chromatogram axis is the window grid of :meth:`grid_rows`, False in sparse mode,
    None when not known (a table built from files without a result set).

    Build it with :meth:`for_run`, which memoises one table per run and file version.
    """

    def __init__(
        self,
        ms2: ParquetHandle | Path | str,
        windows: ParquetHandle | Path | str | None = None,
        *,
        label: str = "",
        emit_window_grid: bool | None = None,
    ) -> None:
        self._handle = _as_handle(ms2)
        self.label = label or self._handle.path.name
        self.emit_window_grid = emit_window_grid
        where = str(self._handle.path)
        check_columns("spectra_ms2", self._handle.schema, where=where)
        table = self._handle.read(columns=list(_MS2_SCALARS))
        names = set(table.column_names)
        self.scan_index = _numeric(table, "scan_index", np.int64, where=where)
        self.rt = _numeric(table, "rt_seconds", np.float64, where=where)
        self.window_id = _numeric(table, "window_id", np.int64, where=where)
        self.target = _numeric(table, "window_target", np.float64, where=where)
        self.lower = _numeric(table, "window_lower", np.float64, where=where)
        self.upper = _numeric(table, "window_upper", np.float64, where=where)
        n = int(self.rt.size)
        self.precursor_mz = (
            _numeric(table, "precursor_mz", np.float64, where=where, fill=float("nan"))
            if "precursor_mz" in names
            else np.full(n, np.nan)
        )
        self.charge = (
            _numeric(table, "precursor_charge", np.int32, where=where, fill=0)
            if "precursor_charge" in names
            else np.zeros(n, dtype=np.int32)
        )
        del table
        self.n = n
        self._scan_lookup = _ScanLookup(self.scan_index)
        self._load_windows(windows, where)
        self._build_window_order()
        _readonly(
            self.scan_index,
            self.rt,
            self.window_id,
            self.target,
            self.lower,
            self.upper,
            self.precursor_mz,
            self.charge,
        )

    # ----------------------------------------------------------------- construction

    @classmethod
    def for_run(cls, rs: ResultSet, run: Run | str | int) -> ScanTable:
        """The scan table of ``run``, loaded once per file version and kept on ``rs``."""
        run = _resolve_run(rs, run)
        ms2 = _require(run, "spectra_ms2")
        handle = ms2.parquet()
        stamp = handle.stamp
        key = ("spectra.ScanTable", run.index, run.name)
        with _MEMO_LOCK:
            hit = rs._memo.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        windows = run.artifact("isolation_windows")
        # A present table with an unsupported version raises here; an absent one is
        # replaced by the bounds recorded on every MS2 row.
        window_handle = windows.parquet() if windows is not None and windows.present else None
        grid = rs.config_get("extract", "emit_window_grid")
        table = cls(
            handle,
            window_handle,
            label=run.label,
            emit_window_grid=grid if isinstance(grid, bool) else None,
        )
        with _MEMO_LOCK:
            rs._memo[key] = (stamp, table)
        return table

    def _load_windows(self, windows: ParquetHandle | Path | str | None, where: str) -> None:
        """Window bounds per window id, from isolation_windows or else from the scans."""
        if windows is not None:
            handle = _as_handle(windows)
            wwhere = str(handle.path)
            check_columns("isolation_windows", handle.schema, where=wwhere)
            t = handle.read(columns=list(_WINDOW_COLUMNS))
            ids = _numeric(t, "window_id", np.int64, where=wwhere)
            target = _numeric(t, "target", np.float64, where=wwhere)
            lower = _numeric(t, "lower", np.float64, where=wwhere)
            upper = _numeric(t, "upper", np.float64, where=wwhere)
            self.windows_source = "isolation_windows.parquet"
        else:
            ids, first = np.unique(self.window_id, return_index=True)
            target, lower, upper = self.target[first], self.lower[first], self.upper[first]
            self.windows_source = "derived from spectra_ms2 (first scan of each window_id)"
        order = np.argsort(ids, kind="stable")
        ids, target, lower, upper = ids[order], target[order], lower[order], upper[order]
        if ids.size > 1 and bool(np.any(ids[1:] == ids[:-1])):
            raise InconsistentData(f"{self.windows_source}: window_id values repeat.")
        if self.n == 0:
            pos = np.zeros(0, dtype=np.int64)
        else:
            pos = np.searchsorted(ids, self.window_id)
            clipped = np.minimum(pos, max(ids.size - 1, 0))
            if ids.size == 0 or not bool(np.all(ids[clipped] == self.window_id)):
                missing = sorted(set(self.window_id.tolist()) - set(ids.tolist()))[:5]
                raise InconsistentData(
                    f"{where}: MS2 scans name window_id {missing}, which {self.windows_source} "
                    "does not list."
                )
        same = (lower[pos] == self.lower) & (upper[pos] == self.upper)
        if not bool(np.all(same)):
            bad = int(np.flatnonzero(~same)[0])
            raise InconsistentData(
                f"{where}: row {bad} has window bounds ({self.lower[bad]}, {self.upper[bad]}) "
                f"but {self.windows_source} gives window {self.window_id[bad]} the bounds "
                f"({lower[pos[bad]]}, {upper[pos[bad]]})."
            )
        self._w_ids, self._w_target, self._w_lower, self._w_upper = ids, target, lower, upper
        self._w_pos = pos.astype(np.int64)
        _readonly(self._w_ids, self._w_target, self._w_lower, self._w_upper, self._w_pos)

    def _build_window_order(self) -> None:
        """Rows of each window ordered by (rt, row), as the engine's stable RT sort does."""
        n = self.n
        rows = np.arange(n, dtype=np.int64)
        if n < 2 or bool(np.all(self.rt[1:] >= self.rt[:-1])):
            order = np.argsort(self._w_pos, kind="stable")
        else:
            order = np.lexsort((rows, self.rt, self._w_pos))
        self._order = order.astype(np.int64)
        self._starts = np.searchsorted(
            self._w_pos[self._order], np.arange(self._w_ids.size + 1), side="left"
        ).astype(np.int64)
        self._pos_in_order = np.empty(n, dtype=np.int64)
        self._pos_in_order[self._order] = rows
        _readonly(self._order, self._starts, self._pos_in_order)

    # ----------------------------------------------------------------- windows

    @property
    def windows(self) -> pd.DataFrame:
        """The isolation windows: ``window_id``, ``target``, ``lower``, ``upper``."""
        return pd.DataFrame(
            {
                "window_id": self._w_ids,
                "target": self._w_target,
                "lower": self._w_lower,
                "upper": self._w_upper,
            }
        )

    def _covering_positions(self, pmz: float) -> np.ndarray:
        pmz = float(pmz)
        if not np.isfinite(pmz):
            return _EMPTY_ROWS
        hit = np.flatnonzero((self._w_lower <= pmz) & (pmz <= self._w_upper))
        if hit.size > 1:
            hit = hit[np.lexsort((self._w_ids[hit], self._w_lower[hit]))]
        return hit

    def covering_windows(self, pmz: float) -> np.ndarray:
        """Window ids with ``lower <= pmz <= upper`` (float64, inclusive), by lower bound.

        Usually one window. Two when the precursor lies on a bound that two windows
        share or inside an overlap; none when it lies in a gap between windows (such
        a precursor is never extracted).
        """
        return self._w_ids[self._covering_positions(pmz)]

    def rows_in_window(self, window_id: int) -> np.ndarray:
        """Rows of one window, ordered by (rt, row)."""
        w = self._window_position(window_id)
        return self._order[self._starts[w] : self._starts[w + 1]]

    def _window_position(self, window_id: int) -> int:
        w = int(np.searchsorted(self._w_ids, int(window_id)))
        if w >= self._w_ids.size or int(self._w_ids[w]) != int(window_id):
            raise KeyError(f"no isolation window {window_id}")
        return w

    # ----------------------------------------------------------------- scans

    def pick(self, row: int, *, reference_rt: float | None = None) -> ScanPick:
        """A :class:`ScanPick` for ``row``; ``delta_rt`` and ``exact`` refer to ``reference_rt``."""
        row = int(row)
        if not 0 <= row < self.n:
            raise IndexError(f"{self.label}: MS2 row {row} is outside the table ({self.n} rows).")
        rt = float(self.rt[row])
        ref = rt if reference_rt is None else float(reference_rt)
        exact = bool(np.float64(rt).view(np.uint64) == np.float64(ref).view(np.uint64))
        return ScanPick(
            row=row,
            scan_index=int(self.scan_index[row]),
            rt=rt,
            window_id=int(self.window_id[row]),
            window_lower=float(self.lower[row]),
            window_upper=float(self.upper[row]),
            exact=exact,
            delta_rt=rt - ref,
        )

    def apex_scan(self, pmz: float, apex_rt: float) -> ScanPick | None:
        """The MS2 scan behind an apex: the covering-window scan at ``apex_rt``.

        First the engine's rule: a scan of a covering window whose ``rt_seconds``
        equals ``apex_rt`` bit for bit (the lowest row when several do), with
        ``exact=True``. Every scored row and every ``psms_extracted`` row of MuMDIA
        0.5.0 has exactly one such scan. Otherwise the scan nearest in RT, ties to the
        lower RT and then the lower row, with ``exact=False``. None when no window
        covers ``pmz`` or ``apex_rt`` is not finite.
        """
        t = float(apex_rt)
        if not np.isfinite(t):
            return None
        target_bits = np.float64(t).view(np.uint64)
        exact_rows: list[int] = []
        best: tuple[float, float, int] | None = None
        for w in self._covering_positions(pmz):
            rows = self._order[self._starts[w] : self._starts[w + 1]]
            if rows.size == 0:
                continue
            rts = self.rt[rows]
            i = int(np.searchsorted(rts, t, side="left"))
            j = int(np.searchsorted(rts, t, side="right"))
            if j > i:
                same = rows[i:j][rts[i:j].view(np.uint64) == target_bits]
                if same.size:
                    exact_rows.append(int(same.min()))
            if i < rts.size:
                cand = (float(rts[i]) - t, float(rts[i]), int(rows[i]))
                best = cand if best is None or cand < best else best
            if i > 0:
                v = rts[i - 1]
                k = int(np.searchsorted(rts, v, side="left"))
                cand = (t - float(v), float(v), int(rows[k]))
                best = cand if best is None or cand < best else best
        if exact_rows:
            return self.pick(min(exact_rows), reference_rt=t)
        if best is None:
            return None
        return self.pick(best[2], reference_rt=t)

    def step(self, row: int, n: int = 1, *, reference_rt: float | None = None) -> ScanPick | None:
        """The ``n``-th neighbour of ``row`` in the same window, ordered by (rt, row).

        ``n`` may be negative. None beyond the first or last scan of the window.
        ``delta_rt`` and ``exact`` refer to ``reference_rt`` (default: the RT of
        ``row``). The step may leave the candidate's RT window; compare the RT with
        ``rt_lo`` and ``rt_hi`` to flag that.
        """
        row = int(row)
        if not 0 <= row < self.n:
            raise IndexError(f"{self.label}: MS2 row {row} is outside the table ({self.n} rows).")
        w = int(self._w_pos[row])
        target = int(self._pos_in_order[row]) + int(n)
        if not int(self._starts[w]) <= target < int(self._starts[w + 1]):
            return None
        ref = float(self.rt[row]) if reference_rt is None else float(reference_rt)
        return self.pick(int(self._order[target]), reference_rt=ref)

    @property
    def grid_label(self) -> str:
        """What :meth:`grid_rows` is for this run's chromatograms (from ``emit_window_grid``)."""
        if self.emit_window_grid is None:
            return GRID_UNKNOWN_LABEL
        return GRID_AXIS_LABEL if self.emit_window_grid else SPARSE_GRID_LABEL

    def grid_rows(self, pmz: float, rt_lo: float, rt_hi: float) -> np.ndarray:
        """The scans of a candidate's XIC grid, one row per grid point.

        Rows of every covering window with ``rt_lo <= rt <= rt_hi``, sorted by RT and
        de-duplicated on RT (the first row is kept), as extract builds the grid. An
        unbounded window (``-inf``, ``+inf``) selects every scan of the covering
        windows; a NaN bound selects none (extract refuses NaN bounds).

        In window-grid mode (``extract.emit_window_grid = true``, the default)
        ``float32(rt[grid_rows(...)])`` equals the candidate's chromatogram axis, so
        axis point ``k`` is row ``k``. In sparse mode it does not: each fragment's axis
        holds only the scans where the fragment was observed. By the engine source
        (not verified on data) those are scans of this grid, so match an axis value to
        ``float32(rt)`` of these rows by value, not by position. Show
        :attr:`grid_label` with the result.
        """
        lo, hi = float(rt_lo), float(rt_hi)
        if np.isnan(lo) or np.isnan(hi):
            return _EMPTY_ROWS
        parts = []
        for w in self._covering_positions(pmz):
            rows = self._order[self._starts[w] : self._starts[w + 1]]
            rts = self.rt[rows]
            a = int(np.searchsorted(rts, lo, side="left"))
            b = int(np.searchsorted(rts, hi, side="right"))
            if b > a:
                parts.append(rows[a:b])
        if not parts:
            return _EMPTY_ROWS
        rows = np.concatenate(parts)
        if len(parts) > 1:
            rows = rows[np.lexsort((rows, self.rt[rows]))]
        rts = self.rt[rows]
        keep = np.ones(rows.size, dtype=bool)
        keep[1:] = rts[1:] != rts[:-1]
        return rows[keep]

    def row_of_scan_index(self, scan_index: int) -> int | None:
        """The MS2 row holding ``scan_index``; None when it is not an MS2 scan."""
        return self._scan_lookup.row(int(scan_index))

    def spectrum(self, row: int, *, cached: bool = True) -> Spectrum:
        """The peaks of MS2 ``row``, read from its row group.

        With ``cached=True`` the decoded row group stays in :data:`SPECTRUM_CACHE`, so
        neighbouring scans are read from memory. An Astral MS2 row group is 2 to 21 MB
        decoded (median about 10 MB); pass ``cached=False`` for one-off reads, such as
        a spectrum browser jumping between distant scans.
        """
        row = int(row)
        where = str(self._handle.path)
        mz, intensity, native_id = _read_peaks(
            self._handle, row, with_id=True, where=where, cached=cached
        )
        pmz = float(self.precursor_mz[row])
        charge = int(self.charge[row])
        return Spectrum(
            level=2,
            row=row,
            scan_index=int(self.scan_index[row]),
            rt=float(self.rt[row]),
            mz=mz,
            intensity=intensity,
            native_id=native_id,
            window_id=int(self.window_id[row]),
            window_lower=float(self.lower[row]),
            window_upper=float(self.upper[row]),
            window_target=float(self.target[row]),
            precursor_mz=pmz if np.isfinite(pmz) else None,
            precursor_charge=charge if charge != 0 else None,
        )

    # ----------------------------------------------------------------- scheme

    def scheme(self) -> pd.DataFrame:
        """The isolation-window scheme; see :func:`isolation_scheme`."""
        order = np.lexsort((self._w_ids, self._w_upper, self._w_lower))
        lower, upper = self._w_lower[order], self._w_upper[order]
        target = self._w_target[order]
        n_scans = np.diff(self._starts)[order]
        overlap = np.full(order.size, np.nan)
        if order.size > 1:
            overlap[:-1] = upper[:-1] - lower[1:]
        cycle = np.full(order.size, np.nan)
        for k, w in enumerate(order):
            rows = self._order[self._starts[w] : self._starts[w + 1]]
            if rows.size > 1:
                cycle[k] = float(np.median(np.diff(self.rt[rows])))
        return pd.DataFrame(
            {
                "window_id": self._w_ids[order],
                "lower": lower,
                "upper": upper,
                "target": target,
                "width": upper - lower,
                "n_scans": n_scans.astype(np.int64),
                "overlap_with_next": overlap,
                "cycle_time_s": cycle,
                "aif": (target == 0.0) & (lower == 0.0) & (upper == 1.0e6),
            }
        )


def isolation_scheme(rs: ResultSet, run: Run | str | int) -> pd.DataFrame:
    """The isolation-window scheme of a run, one row per window, sorted by (lower, upper).

    Columns: ``window_id``, ``lower``, ``upper``, ``target`` (m/z), ``width``
    (``upper - lower``), ``n_scans`` (MS2 scans acquired in the window),
    ``overlap_with_next`` (Th; ``upper - next.lower``: positive is an overlap with the
    next window, 0 a shared bound, negative a gap; NaN for the last window),
    ``cycle_time_s`` (median RT step between consecutive scans of the window) and
    ``aif`` (the synthesized full-range window ``(0, 1e6)`` of an all-ion scan).
    """
    return ScanTable.for_run(rs, run).scheme()


# --------------------------------------------------------------------------- MS1


def isotope_mz(precursor_mz: float, charge: int, k: int) -> float:
    """The m/z of isotope ``k`` as extract computes it: ``pmz + k * (1.003354835 / z)``.

    ``k = 0, 1, 2`` are the ``ms1_mono``, ``ms1_iso1`` and ``ms1_iso2`` rows; ``k = -1``
    is the ``ms1_isom1`` column of ``psms_extracted``.
    """
    return float(precursor_mz) + float(k) * (ISOTOPE_SPACING / float(charge))


def sum_near(mz: np.ndarray, intensity: np.ndarray, target: float, tol_ppm: float) -> float:
    """Intensity summed over the peaks near ``target``, with the engine's ``sum_near`` rule.

    Peaks with ``target - d <= mz <= target + d``, ``d = target * tol_ppm * 1e-6``
    (float32 m/z widened to float64), summed in m/z order in float32. The result is a
    viewer computation: label it :data:`MS1_SUM_LABEL`. It equals the engine's values
    only for the MS1 scan from :meth:`Ms1Table.nearest` and ``tol_ppm =
    extract.prec_tol_ppm``; then it reproduces the ``ms1_*`` columns of
    ``psms_extracted`` and the MS1 trace values bit for bit. To show the engine's own
    value, read those columns (at ``apex_rt``) or the chromatogram rows ``ms1_mono``,
    ``ms1_iso1`` and ``ms1_iso2``.
    """
    mz64 = np.asarray(mz, dtype=np.float32).astype(np.float64)
    d = float(target) * float(tol_ppm) * 1e-6
    lo, hi = float(target) - d, float(target) + d
    s = int(np.searchsorted(mz64, lo, side="left"))
    e = int(np.searchsorted(mz64, hi, side="right"))
    if e <= s:
        return 0.0
    values = np.asarray(intensity, dtype=np.float32)[s:e]
    return float(np.add.accumulate(values, dtype=np.float32)[-1])


class Ms1Table:
    """The MS1 scans of one run (``scan_index``, ``rt``) and the two MS1 relations.

    :meth:`nearest` is the MS1 scan nearest in RT, the scan the engine samples for the
    ``psms_extracted`` MS1 columns and, in window-grid mode, for its ``ms1_*`` XIC rows.
    :meth:`preceding` is the
    acquisition parent from ``ms2_to_ms1``. They differ for about half of the MS2
    scans, so label which one is shown (:data:`NEAREST_MS1_LABEL`,
    :data:`PRECEDING_MS1_LABEL`).

    ``ms2_to_ms1`` may be a handle, a path or the run's :class:`Artifact`. It is opened
    only by :meth:`preceding` and the two link properties, so a missing or unreadable
    link table (for example an unsupported schema version) refuses those alone.
    ``n_ms2`` is the row count of ``spectra_ms2``, which the link table must match;
    ``ms2`` gives the same check from the ``spectra_ms2`` artifact, read lazily.
    """

    def __init__(
        self,
        ms1: ParquetHandle | Path | str,
        ms2_to_ms1: ParquetHandle | Path | str | Artifact | None = None,
        *,
        n_ms2: int | None = None,
        ms2: Artifact | None = None,
        label: str = "",
    ) -> None:
        self._handle = _as_handle(ms1)
        self.label = label or self._handle.path.name
        where = str(self._handle.path)
        check_columns("spectra_ms1", self._handle.schema, where=where)
        table = self._handle.read(columns=["scan_index", "rt_seconds"])
        self.scan_index = _numeric(table, "scan_index", np.int64, where=where)
        self.rt = _numeric(table, "rt_seconds", np.float64, where=where)
        self.n = int(self.rt.size)
        self._scan_lookup = _ScanLookup(self.scan_index)
        if self.n < 2 or bool(np.all(self.rt[1:] >= self.rt[:-1])):
            self._rt_order: np.ndarray | None = None
            self._rt_sorted = self.rt
        else:
            self._rt_order = np.argsort(self.rt, kind="stable")
            self._rt_sorted = self.rt[self._rt_order]
        _readonly(self.scan_index, self.rt)
        self._m2m_source: ParquetHandle | Artifact | None
        if ms2_to_ms1 is None or isinstance(ms2_to_ms1, Artifact):
            self._m2m_source = ms2_to_ms1
        else:
            self._m2m_source = _as_handle(ms2_to_ms1)
        self._n_ms2 = n_ms2
        self._ms2 = ms2
        self._m2m: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()

    @classmethod
    def for_run(cls, rs: ResultSet, run: Run | str | int) -> Ms1Table:
        """The MS1 table of ``run``, loaded once per file version and kept on ``rs``.

        Only ``spectra_ms1`` is read here. ``ms2_to_ms1`` and the ``spectra_ms2`` row
        count are read on the first :meth:`preceding` call, so :meth:`nearest` and
        :meth:`spectrum` work when the link table is missing or refused.
        """
        run = _resolve_run(rs, run)
        ms1 = _require(run, "spectra_ms1")
        handle = ms1.parquet()
        stamp = handle.stamp
        key = ("spectra.Ms1Table", run.index, run.name)
        with _MEMO_LOCK:
            hit = rs._memo.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        table = cls(
            handle,
            run.artifact("ms2_to_ms1"),
            ms2=run.artifact("spectra_ms2"),
            label=run.label,
        )
        with _MEMO_LOCK:
            rs._memo[key] = (stamp, table)
        return table

    def nearest(self, rt: float) -> int | None:
        """The MS1 row nearest in RT to ``rt``, ties to the earlier scan (the engine's rule).

        None when the run has no MS1 scans or ``rt`` is not finite.
        """
        t = float(rt)
        if self.n == 0 or not np.isfinite(t):
            return None
        return int(self.nearest_rows(np.array([t]))[0])

    def nearest_rows(self, rts: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`nearest` for finite RTs (for example a whole XIC grid)."""
        t = np.asarray(rts, dtype=np.float64)
        if self.n == 0:
            raise ArtifactNotFound(f"{self.label}: the run has no MS1 scans.")
        srt = self._rt_sorted
        p = np.searchsorted(srt, t, side="left")
        lo = np.clip(p - 1, 0, self.n - 1)
        hi = np.clip(p, 0, self.n - 1)
        take_lo = np.abs(t - srt[lo]) <= np.abs(srt[hi] - t)
        j = np.where(p == 0, 0, np.where(p >= self.n, self.n - 1, np.where(take_lo, lo, hi)))
        j = j.astype(np.int64)
        return j if self._rt_order is None else self._rt_order[j].astype(np.int64)

    def _link_handle(self) -> ParquetHandle:
        """The ``ms2_to_ms1`` handle; raises when the table is missing or refused."""
        source = self._m2m_source
        if source is None:
            raise ArtifactNotFound(f"{self.label}: the run has no ms2_to_ms1 table.")
        if isinstance(source, Artifact):
            # ArtifactNotFound when missing, SchemaVersionError for an unsupported
            # version, LayoutError when a column is absent.
            return source.parquet()
        check_columns("ms2_to_ms1", source.schema, where=str(source.path))
        return source

    def _expected_ms2_rows(self) -> int | None:
        if self._n_ms2 is not None:
            return self._n_ms2
        if self._ms2 is not None and self._ms2.usable:
            return self._ms2.parquet().num_rows
        return None

    def _link(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._m2m is None:
                handle = self._link_handle()
                where = str(handle.path)
                t = handle.read(columns=["ms2_scan_index", "ms1_scan_index"])
                ms2 = _numeric(t, "ms2_scan_index", np.int64, where=where)
                ms1 = _numeric(t, "ms1_scan_index", np.int64, where=where)
                n_ms2 = self._expected_ms2_rows()
                if n_ms2 is not None and ms2.size != n_ms2:
                    raise InconsistentData(
                        f"{where}: {ms2.size} rows, but spectra_ms2 has {n_ms2}; the table "
                        "is row-aligned with spectra_ms2."
                    )
                _readonly(ms2, ms1)
                self._m2m = (ms2, ms1)
            return self._m2m

    @property
    def ms2_scan_index(self) -> np.ndarray:
        """``ms2_to_ms1.ms2_scan_index``: the MS2 ``scan_index`` of each MS2 row."""
        return self._link()[0]

    @property
    def ms1_scan_index(self) -> np.ndarray:
        """``ms2_to_ms1.ms1_scan_index``: the preceding MS1 ``scan_index``, -1 for none."""
        return self._link()[1]

    def preceding(self, ms2_row: int) -> int | None:
        """The MS1 row of the MS1 scan acquired before MS2 row ``ms2_row`` (``ms2_to_ms1``).

        None when no MS1 scan precedes it (``-1``). Label it
        :data:`PRECEDING_MS1_LABEL`; it is not the scan the engine's MS1 XIC uses.
        Raises :class:`ArtifactNotFound` or :class:`SchemaVersionError` when the
        ``ms2_to_ms1`` table is missing or refused.
        """
        _, parent = self._link()
        ms2_row = int(ms2_row)
        if not 0 <= ms2_row < parent.size:
            raise IndexError(f"{self.label}: MS2 row {ms2_row} is outside ms2_to_ms1.")
        scan = int(parent[ms2_row])
        if scan < 0:
            return None
        row = self.row_of_scan_index(scan)
        if row is None:
            raise InconsistentData(
                f"{self.label}: ms2_to_ms1 names MS1 scan_index {scan}, which spectra_ms1 does "
                "not hold."
            )
        return row

    def row_of_scan_index(self, scan_index: int) -> int | None:
        """The MS1 row holding ``scan_index``; None when it is not an MS1 scan."""
        return self._scan_lookup.row(int(scan_index))

    def spectrum(self, row: int, *, cached: bool = True) -> Spectrum:
        """The peaks of MS1 ``row``, read from its row group (see :meth:`ScanTable.spectrum`)."""
        row = int(row)
        where = str(self._handle.path)
        mz, intensity, _ = _read_peaks(self._handle, row, with_id=False, where=where, cached=cached)
        return Spectrum(
            level=1,
            row=row,
            scan_index=int(self.scan_index[row]),
            rt=float(self.rt[row]),
            mz=mz,
            intensity=intensity,
        )
