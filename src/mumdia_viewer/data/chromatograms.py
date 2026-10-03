"""Chromatogram traces of one candidate, decoded as the engine decodes them.

A chromatogram table holds one row per predicted fragment of an accepted candidate, in
the library's fragment order, then the three MS1 isotope rows ``ms1_mono``, ``ms1_iso1``
and ``ms1_iso2`` (window-grid mode only). All peak ranks of a candidate share these rows.
Two layouts exist (:class:`~mumdia_viewer.data.schemas.ChromatogramLayout`):

* layout 1 stores each row's axis and full trace (``rt``, ``intensity``);
* layout 2 stores a row's axis in ``rt_axis`` only where no earlier row of the same
  candidate in the same row group wrote it, and the trace from its first to its last value
  that is not ``+0.0`` (``intensity_trimmed`` at ``trace_offset`` of ``trace_len``).

The ion-mobility layouts of the unreleased engine branch (PR #140) add ``im`` to layout 1
(version 3) or ``im_trimmed`` to layout 2 (version 4): the 1/K0 of each point, ``0.0``
where there is no mobility.

:func:`decode_rows` ports ``mumdia::chromatograms::Decoder::row`` with the engine's checks.
The layout 2 axis rule restarts at every row group, so a candidate is decoded one
row-group part at a time, from its first row in that row group, each part with a fresh
decoder. Rows are never filtered before decoding: the row that carries the axis can be any
row of the candidate, an MS1 row included. The decoded arrays are copies, so a kept result
never holds the memory of the row group it was read from.

:class:`ChromatogramSource` finds the tables that hold a run's chromatograms: the run's own
table; the pooled table of a grouped run; or the band tables of a grouped run, in the order
that ``groups/overlap_losers.parquet`` records, each without its overlap losers. A row group
of normal size is read whole through the row-group cache. A row group too large for that
cache (the 1,048,576-row groups of layout 1 tables written before MuMDIA 0.5.0 decode to
about 1.4 GB) is not read whole: only the candidate's rows are read, with a filtered DuckDB
query.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .artifacts import Artifact
from .candidate_index import CandidateIndex, Segment
from .discovery import Band, GroupedLayout, ResultSet, Run
from .duck import sql_ident, sql_path
from .errors import ArtifactNotFound, InconsistentData, LayoutError, ViewerError
from .manifest import band_index
from .paths import split_parts
from .pqio import ROW_GROUP_CACHE, ParquetHandle
from .schemas import ChromatogramLayout, chromatogram_layout

__all__ = [
    "LOSERS_BAND_TABLES_KEY",
    "CandidateChromatogram",
    "ChromatogramSource",
    "ChromatogramTable",
    "DecodedRow",
    "Trace",
    "decode_rows",
    "parse_fragment_name",
    "row_group_bytes",
    "trace_columns",
]

#: Footer key of ``groups/overlap_losers.parquet``: a JSON list of the band tables
#: (``{name, rows, content_hash}``) in pool order. The ``band`` column indexes this list.
LOSERS_BAND_TABLES_KEY = b"mumdia.overlap_losers.band_tables"

#: The per-row columns read beside the trace columns.
SCALAR_COLUMNS = ("candidate_id", "frag_name", "frag_mz", "frag_obs_mz", "predicted_intensity")

#: The engine classifies a row as an MS1 isotope row by this prefix of ``frag_name``.
MS1_PREFIX = "ms1_"

_FRAGMENT_NAME = re.compile(r"^(?P<ion>[by])(?P<ordinal>\d+)(?:\^(?P<charge>\d+))?$")
_MEMO_KEY = "mumdia_viewer.data.chromatograms.ChromatogramSource"

_EMPTY = np.zeros(0, dtype=np.float32)
_EMPTY.flags.writeable = False

#: One decoded row: the axis, the full trace, and the 1/K0 list (ion-mobility layouts) or None.
DecodedRow = tuple[np.ndarray, np.ndarray, np.ndarray | None]


@functools.lru_cache(maxsize=4096)
def parse_fragment_name(name: str) -> tuple[str, int, int] | None:
    """Ion type, ordinal and charge of a b/y fragment name: ``y7^2`` gives ``('y', 7, 2)``.

    The charge is 1 when the name has no ``^z`` part. Any other name (an MS1 row, or a name
    from a foreign library) gives None.
    """
    m = _FRAGMENT_NAME.match(name)
    if m is None:
        return None
    charge = m.group("charge")
    return m.group("ion"), int(m.group("ordinal")), int(charge) if charge is not None else 1


def trace_columns(layout: ChromatogramLayout) -> tuple[str, ...]:
    """The columns that hold a row's traces in ``layout``, in the engine's order."""
    if layout.family == 1:
        cols: tuple[str, ...] = ("rt", "intensity")
        return (*cols, "im") if layout.has_im else cols
    cols = ("rt_axis", "intensity_trimmed", "trace_offset", "trace_len")
    return (*cols, "im_trimmed") if layout.has_im else cols


def _read_columns(layout: ChromatogramLayout) -> list[str]:
    return [*SCALAR_COLUMNS, *trace_columns(layout)]


# --------------------------------------------------------------------------- traces


@dataclass(frozen=True, eq=False)
class Trace:
    """One chromatogram row in its full (layout 1) form.

    ``rt`` holds ``float32(rt_seconds)`` of the scans of the candidate's grid and is empty
    when the fragment was never observed. ``intensity`` is the full trace on that axis:
    for a fragment the highest intensity of its hits in each scan (0.0 where it has none),
    for an MS1 row the summed MS1 peaks of the nearest MS1 scan. ``im`` is the 1/K0 of each
    point on the ion-mobility layouts (0.0 means no mobility) and None otherwise. The
    arrays are read-only. Equality is identity: compare the arrays explicitly.
    """

    frag_name: str
    frag_mz: float
    frag_obs_mz: float
    predicted_intensity: float
    rt: np.ndarray
    intensity: np.ndarray
    im: np.ndarray | None
    trace_len: int

    @property
    def is_ms1(self) -> bool:
        """An MS1 isotope row (``ms1_mono``, ``ms1_iso1``, ``ms1_iso2``), by the engine's rule."""
        return self.frag_name.startswith(MS1_PREFIX)

    @property
    def observed(self) -> bool:
        """False for a predicted fragment that was never observed (an empty trace)."""
        return self.trace_len > 0

    @property
    def ion(self) -> str | None:
        """``'b'`` or ``'y'``; None for a name that is not a b/y fragment name."""
        parsed = parse_fragment_name(self.frag_name)
        return parsed[0] if parsed is not None else None

    @property
    def ordinal(self) -> int | None:
        parsed = parse_fragment_name(self.frag_name)
        return parsed[1] if parsed is not None else None

    @property
    def fragment_charge(self) -> int | None:
        """The fragment charge from ``^z`` (1 when absent); None for other names."""
        parsed = parse_fragment_name(self.frag_name)
        return parsed[2] if parsed is not None else None


@dataclass(frozen=True, eq=False)
class CandidateChromatogram:
    """The chromatogram rows of one candidate, decoded.

    ``traces`` are in file order: the library's fragment order, then ``ms1_mono``,
    ``ms1_iso1`` and ``ms1_iso2``. ``table`` is the file read, ``band`` its band (``gNN``)
    when it is a band table, and ``row_groups`` the row groups the rows came from.
    """

    candidate_id: int
    traces: tuple[Trace, ...]
    layout: ChromatogramLayout
    table: Path
    band: str | None
    row_groups: tuple[int, ...]

    def fragments(self) -> tuple[Trace, ...]:
        """The fragment rows (every row that is not an MS1 row), in file order."""
        return tuple(t for t in self.traces if not t.is_ms1)

    def ms1(self) -> tuple[Trace, ...]:
        """The MS1 isotope rows, in file order."""
        return tuple(t for t in self.traces if t.is_ms1)

    def common_axis(self) -> np.ndarray | None:
        """The axis every observed row shares, compared by bit pattern.

        None when no row was observed, or when the observed rows have different axes (the
        sparse mode, ``extract.emit_window_grid = false``, gives each fragment its own).
        """
        axis: np.ndarray | None = None
        for t in self.traces:
            if not t.observed:
                continue
            if axis is None:
                axis = t.rt
            elif not _same_bits(axis, t.rt):
                return None
        return axis

    def matrix(
        self, include_ms1: bool = False
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray] | None:
        """``(axis, names, values)`` with one float32 row per trace on the common axis.

        Rows are the fragment rows in file order, followed by the MS1 rows when
        ``include_ms1``. A fragment that was never observed had no hit in any scan of the
        grid, so its row is zeros. None when there is no common axis.
        """
        axis = self.common_axis()
        if axis is None:
            return None
        chosen = self.traces if include_ms1 else self.fragments()
        values = np.zeros((len(chosen), axis.size), dtype=np.float32)
        for i, t in enumerate(chosen):
            if t.observed:
                values[i] = t.intensity
        return axis, tuple(t.frag_name for t in chosen), values


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


# --------------------------------------------------------------------------- decoding


def _single(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    if isinstance(column, pa.ChunkedArray):
        return column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    return column


def _ids(table: pa.Table, where: str) -> np.ndarray:
    arr = _single(table.column("candidate_id"))
    if not pa.types.is_integer(arr.type):
        raise LayoutError(f"{where}: candidate_id is {arr.type}, not an integer column.")
    if arr.null_count:
        raise InconsistentData(f"{where}: candidate_id holds a null.")
    return arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _counts(table: pa.Table, name: str, where: str) -> np.ndarray:
    """``trace_offset`` or ``trace_len`` as int64. The engine requires u32 without nulls."""
    arr = _single(table.column(name))
    if not pa.types.is_integer(arr.type):
        raise LayoutError(f"{where}: chromatogram column {name} is {arr.type}, not u32.")
    if arr.null_count:
        raise InconsistentData(f"{where}: chromatogram column {name} holds a null.")
    values = arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if values.size and int(values.min()) < 0:
        raise InconsistentData(f"{where}: chromatogram column {name} holds a negative value.")
    return values


def _float_lists(table: pa.Table, name: str, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Values (float32) and offsets (int64, from 0, length n + 1) of a trace list column.

    A null list is read as an empty list, as the engine's reader does. A null value inside
    a list has no defined value, so it is refused.
    """
    arr = _single(table.column(name))
    typ = arr.type
    if not (pa.types.is_large_list(typ) or pa.types.is_list(typ)) or not pa.types.is_float32(
        typ.value_type
    ):
        raise LayoutError(
            f"{where}: chromatogram column {name} is {typ}; trace columns are lists of float32."
        )
    if arr.null_count:
        arr = pc.fill_null(arr, pa.scalar([], type=typ))
    offsets = np.asarray(arr.offsets, dtype=np.int64)
    flat = arr.flatten()
    if flat.null_count:
        raise InconsistentData(
            f"{where}: chromatogram column {name} holds a null value inside a list; the "
            "engine never writes one."
        )
    values = flat.to_numpy(zero_copy_only=False)
    return values, offsets - offsets[0]


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """``concatenate([arange(s, s + n) for s, n in zip(starts, lengths)])``, vectorised."""
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    out_start = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) + np.repeat(starts - out_start, lengths)


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _owned(values: np.ndarray) -> np.ndarray:
    """A read-only float32 copy of ``values``.

    The values of a list column are a zero-copy view of the Arrow buffer of the whole row
    group, also when the table is a slice of one candidate. A trace that kept such a view
    would keep the whole row group in memory, outside the row-group cache's accounting.
    """
    return _read_only(np.array(values, dtype=np.float32, copy=True))


def decode_rows(
    table: pa.Table, layout: ChromatogramLayout, *, where: str = "chromatograms"
) -> list[DecodedRow]:
    """Decode chromatogram rows into their layout 1 form: ``(rt, intensity, im or None)``.

    ``table`` must be one contiguous slice of a row group that starts at the row group's
    first row or at a candidate's first row in it, the read scopes the engine's decoder
    accepts. A slice that crosses a row-group seam must be split at the seam, because the
    layout 2 rule restarts there.

    Layout 1 and 3 rows are the lists as stored (a null list is empty); ``rt`` and
    ``intensity`` (and ``im``) must have the same length. Layout 2 and 4 rows follow
    ``Decoder::row``: ``trace_len`` 0 means never observed and stores nothing; a
    non-empty ``rt_axis`` opens the axis of its candidate and has ``trace_len`` points; an
    empty one takes the last axis written in this read, which must belong to the same
    candidate and have ``trace_len`` points; the trace is ``trace_len`` zeros with
    ``intensity_trimmed`` copied in at ``trace_offset``; version 4 rebuilds ``im`` from
    ``im_trimmed`` in the same way. A row that breaks a rule raises
    :class:`InconsistentData`, naming the first such row as the engine would.

    The returned arrays are read-only and hold no reference to the buffers of ``table``:
    the values are copied, so a kept result never keeps a row group in memory. Rows that
    share an axis share one copy of it.
    """
    if layout.family == 1:
        return _decode_family1(table, layout, where)
    return _decode_family2(table, layout, where)


def _decode_family1(table: pa.Table, layout: ChromatogramLayout, where: str) -> list[DecodedRow]:
    cid = _ids(table, where)
    rt_v, rt_o = _float_lists(table, "rt", where)
    it_v, it_o = _float_lists(table, "intensity", where)
    rt_v, it_v = _owned(rt_v), _owned(it_v)
    rt_len = np.diff(rt_o)
    bad = np.flatnonzero(rt_len != np.diff(it_o))
    if bad.size:
        k = int(bad[0])
        raise InconsistentData(
            f"{where}: row {k} (candidate_id {cid[k]}) has {rt_len[k]} retention-time points "
            f"but {int(it_o[k + 1] - it_o[k])} intensity points."
        )
    im_v: np.ndarray | None = None
    im_o: np.ndarray | None = None
    if layout.has_im:
        im_v, im_o = _float_lists(table, "im", where)
        im_v = _owned(im_v)
        bad = np.flatnonzero(np.diff(im_o) != rt_len)
        if bad.size:
            k = int(bad[0])
            raise InconsistentData(
                f"{where}: row {k} (candidate_id {cid[k]}) has {rt_len[k]} trace points but "
                f"{int(im_o[k + 1] - im_o[k])} ion-mobility values."
            )
    rt_bounds = rt_o.tolist()
    it_bounds = it_o.tolist()
    im_bounds = im_o.tolist() if im_o is not None else None
    out: list[DecodedRow] = []
    for k in range(table.num_rows):
        rt = rt_v[rt_bounds[k] : rt_bounds[k + 1]]
        it = it_v[it_bounds[k] : it_bounds[k + 1]]
        im = None
        if im_v is not None and im_bounds is not None:
            im = im_v[im_bounds[k] : im_bounds[k + 1]]
        out.append((rt, it, im))
    return out


def _decode_family2(table: pa.Table, layout: ChromatogramLayout, where: str) -> list[DecodedRow]:
    n = table.num_rows
    cid = _ids(table, where)
    ax_v, ax_o = _float_lists(table, "rt_axis", where)
    tr_v, tr_o = _float_lists(table, "intensity_trimmed", where)
    off = _counts(table, "trace_offset", where)
    tl = _counts(table, "trace_len", where)
    a_len = np.diff(ax_o)
    s_len = np.diff(tr_o)
    rows = np.arange(n, dtype=np.int64)
    absent = tl == 0
    has_axis = a_len > 0
    # The axis source of every row: the last row at or before it that wrote an axis.
    src = np.maximum.accumulate(np.where(has_axis, rows, -1)) if n else rows
    src0 = np.clip(src, 0, None)
    inherit = ~absent & ~has_axis
    # The checks of Decoder::resolve, in its order for one row.
    e_absent = absent & ((a_len != 0) | (s_len != 0) | (off != 0))
    e_axis_len = ~absent & has_axis & (a_len != tl)
    e_orphan = inherit & ((src < 0) | (cid[src0] != cid)) if n else inherit
    e_inherit_len = inherit & ~e_orphan & (a_len[src0] != tl) if n else inherit
    e_overflow = ~absent & (off + s_len > tl)
    errors = e_absent | e_axis_len | e_orphan | e_inherit_len | e_overflow
    if errors.any():
        k = int(np.flatnonzero(errors)[0])
        c = int(cid[k])
        if e_absent[k]:
            reason = (
                f"has trace_len 0 (a fragment that was never observed) but {a_len[k]} "
                f"retention-time and {s_len[k]} intensity values at offset {off[k]}"
            )
        elif e_axis_len[k]:
            reason = f"has {a_len[k]} retention-time points but a trace_len of {tl[k]}"
        elif e_orphan[k]:
            reason = (
                f"has a {tl[k]}-point trace but no retention-time axis, and no earlier row of "
                "the candidate in this read carried one; a layout 2 table must be read from "
                "the start of a row group or from a candidate's first row"
            )
        elif e_inherit_len[k]:
            reason = (
                f"has a trace_len of {tl[k]} but the candidate's retention-time axis has "
                f"{a_len[src0[k]]} points"
            )
        else:
            reason = (
                f"stores {s_len[k]} intensity values at offset {off[k]} of a {tl[k]}-point trace"
            )
        raise InconsistentData(f"{where}: chromatogram row {k} (candidate_id {c}) {reason}.")

    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(tl, out=starts[1:])
    dense = np.zeros(int(starts[-1]), dtype=np.float32)
    dst = _ranges(starts[:-1] + off, s_len)
    if dst.size:
        dense[dst] = tr_v[_ranges(tr_o[:-1], s_len)]
    _read_only(dense)
    im_dense: np.ndarray | None = None
    if layout.has_im:
        im_v, im_o = _float_lists(table, "im_trimmed", where)
        bad = np.flatnonzero(np.diff(im_o) != s_len)
        if bad.size:
            k = int(bad[0])
            raise InconsistentData(
                f"{where}: chromatogram row {k} (candidate_id {cid[k]}) stores "
                f"{int(im_o[k + 1] - im_o[k])} ion-mobility values beside {s_len[k]} "
                "intensity values."
            )
        im_dense = np.zeros(int(starts[-1]), dtype=np.float32)
        if dst.size:
            im_dense[dst] = im_v[_ranges(im_o[:-1], s_len)]
        _read_only(im_dense)

    # The intensity (and im) arrays above are new; the axis values are copied here.
    ax_v = _owned(ax_v)
    axis_start = ax_o[:-1][np.where(has_axis, rows, src0)] if n else rows
    lengths = tl.tolist()
    first = starts.tolist()
    ax_first = axis_start.tolist()
    empty_im = _EMPTY if im_dense is not None else None
    out: list[DecodedRow] = []
    for k in range(n):
        length = lengths[k]
        if length == 0:
            out.append((_EMPTY, _EMPTY, empty_im))
            continue
        a = ax_first[k]
        s = first[k]
        im = im_dense[s : s + length] if im_dense is not None else None
        out.append((ax_v[a : a + length], dense[s : s + length], im))
    return out


def _part_traces(part: pa.Table, layout: ChromatogramLayout, cid: int, where: str) -> list[Trace]:
    """Decode one row-group part of candidate ``cid`` and pair it with its per-row columns."""
    ids = _ids(part, where)
    if ids.size and not bool((ids == cid).all()):
        found = sorted({int(x) for x in ids.tolist()})[:4]
        raise InconsistentData(
            f"{where}: the candidate index points at rows of candidate_id {found} for "
            f"candidate_id {cid}; the index does not describe this file."
        )
    decoded = decode_rows(part, layout, where=where)
    names = _single(part.column("frag_name")).to_pylist()
    n = part.num_rows
    # The engine treats both m/z columns as optional: a missing frag_obs_mz falls back to
    # frag_mz (features.rs), a missing frag_mz is not known (NaN).
    names_present = set(part.column_names)
    frag_mz = (
        part.column("frag_mz").to_numpy().tolist() if "frag_mz" in names_present else [math.nan] * n
    )
    frag_obs_mz = (
        part.column("frag_obs_mz").to_numpy().tolist()
        if "frag_obs_mz" in names_present
        else list(frag_mz)
    )
    predicted = part.column("predicted_intensity").to_numpy().tolist()
    return [
        Trace(
            frag_name=names[k] if names[k] is not None else "",
            frag_mz=float(frag_mz[k]),
            frag_obs_mz=float(frag_obs_mz[k]),
            predicted_intensity=float(predicted[k]),
            rt=rt,
            intensity=it,
            im=im,
            trace_len=int(rt.size),
        )
        for k, (rt, it, im) in enumerate(decoded)
    ]


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True, eq=False)
class ChromatogramTable:
    """One chromatogram table of a run.

    ``band`` is the band directory (``gNN``) of a band table and None for a run's own or
    pooled table. ``position`` is the table's position in the loser file's band list (pool
    order), the value of the loser file's ``band`` column; None when no loser file applies.
    ``drop`` holds the candidates this table does not contribute (its overlap losers).
    """

    artifact: Artifact
    path: Path
    layout: ChromatogramLayout
    band: str | None
    position: int | None
    drop: frozenset[int]


def _table_for(
    artifact: Artifact, band: str | None, position: int | None, drop: frozenset[int]
) -> ChromatogramTable:
    """Describe one table: its layout from the columns, cross-checked with the recorded version.

    An artifact that is missing, or whose version is not supported, raises here through
    :meth:`Artifact.require`.
    """
    path = artifact.require()
    handle = artifact.parquet()  # checks the column contract and the recorded version
    layout = chromatogram_layout(handle.schema, where=str(path))
    recorded = artifact.version.version
    if recorded is not None and recorded != layout.schema_version:
        raise LayoutError(
            f"{path}: the recorded chromatograms schema version is {recorded}, but the "
            f"columns are the version {layout.schema_version} layout."
        )
    return ChromatogramTable(artifact, path, layout, band, position, drop)


def _index_key(index: CandidateIndex, cid: int) -> np.integer | None:
    """``cid`` as a scalar of the index's id dtype; None when no id of that dtype equals it.

    ``np.searchsorted`` on the uint32 ids with a Python int promotes and copies the whole
    array on every call (about 1 ms on the 683,297 ids of the Astral run); a key of the
    ids' own dtype does not.
    """
    dtype = index.ids.dtype
    info = np.iinfo(dtype)
    if not info.min <= cid <= info.max:
        return None
    return dtype.type(cid)


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def row_group_bytes(handle: ParquetHandle, row_group: int, columns: Sequence[str]) -> int:
    """Estimated Arrow bytes of one row group read with ``columns``, from the footer.

    The estimate is the uncompressed size of the projected column chunks plus 8 bytes per
    row for the offsets of each list column. It decides whether a row group is read whole.
    """
    md = handle.metadata()
    group = md.row_group(row_group)
    wanted = set(columns)
    total = 0
    for j in range(group.num_columns):
        chunk = group.column(j)
        path = chunk.path_in_schema
        if path.split(".", 1)[0] not in wanted:
            continue
        total += chunk.total_uncompressed_size
        if "." in path:  # a list column's leaf: add its offsets
            total += 8 * group.num_rows
    return total


def _recorded_hash(artifact: Artifact) -> str | None:
    """The content hash the band table's report records, else its manifest record's."""
    if artifact.report is not None and artifact.report.content_hash:
        return artifact.report.content_hash
    if artifact.record is not None and artifact.record.content_hash:
        return artifact.record.content_hash
    return None


def _band_chromatograms(grouped: GroupedLayout) -> list[tuple[Band, Artifact]]:
    out = []
    for band in grouped.bands:
        artifact = band.artifacts.get("chromatograms")
        if artifact is not None:
            out.append((band, artifact))
    return out


def _loser_tables(run: Run, grouped: GroupedLayout, losers: Artifact) -> list[ChromatogramTable]:
    """The band tables named by the loser file, in its order, each with its drop set."""
    loser_path = losers.require()
    handle = losers.parquet()
    where = str(loser_path)
    raw = (handle.metadata().metadata or {}).get(LOSERS_BAND_TABLES_KEY)
    if raw is None:
        raise InconsistentData(
            f"{where}: the loser file does not name the band tables its losers belong to (no "
            f"{LOSERS_BAND_TABLES_KEY.decode()} footer entry)."
        )
    try:
        records = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InconsistentData(f"{where}: the band table list cannot be parsed ({exc}).") from exc
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise InconsistentData(f"{where}: the band table list is not a list of objects.")
    rows = handle.read(["band", "candidate_id"])
    band_col = _single(rows.column("band")).to_numpy(zero_copy_only=False).astype(np.int64)
    cid_col = _single(rows.column("candidate_id")).to_numpy(zero_copy_only=False).astype(np.int64)
    if band_col.size and (int(band_col.max()) >= len(records) or int(band_col.min()) < 0):
        bad = int(band_col[(band_col >= len(records)) | (band_col < 0)][0])
        raise InconsistentData(
            f"{where}: a loser row names band {bad}, but the file lists {len(records)} band "
            "table(s); `band` is a position in that list."
        )
    by_path = {
        _path_key(a.path): (b, a) for b, a in _band_chromatograms(grouped) if a.path is not None
    }
    groups_dir = run.root / "groups"
    tables: list[ChromatogramTable] = []
    for i, rec in enumerate(records):
        name = rec.get("name")
        n_rows = rec.get("rows")
        footer_hash = rec.get("content_hash")
        parts = split_parts(name) if isinstance(name, str) else []
        if not parts or ".." in parts:
            raise InconsistentData(f"{where}: band {i} has no usable table name ({name!r}).")
        path = groups_dir.joinpath(*parts)
        if not path.is_file():
            raise InconsistentData(
                f"{where}: band {i} is {name}, which does not exist in {groups_dir}; the band "
                "tables of this run are incomplete."
            )
        match = by_path.get(_path_key(path))
        if match is None:
            raise InconsistentData(
                f"{where}: band {i} is {name}, which discovery found no band artifact for."
            )
        band, artifact = match
        drop = frozenset(int(c) for c in cid_col[band_col == i].tolist())
        table = _table_for(artifact, band.name, i, drop)
        actual = artifact.parquet().num_rows
        if not isinstance(n_rows, int) or actual != n_rows:
            raise InconsistentData(
                f"{where}: band {i} is {name} with {n_rows} rows, but that table has {actual} "
                "rows; the loser file belongs to other band tables."
            )
        recorded = _recorded_hash(artifact)
        if footer_hash and recorded and footer_hash != recorded:
            raise InconsistentData(
                f"{where}: band {i} is {name} with content hash {footer_hash}, but that "
                f"table records {recorded}; the loser file belongs to other band tables."
            )
        tables.append(table)
    # The pool writes the list in pool order: ascending plan index of the searched bands.
    # A list out of that order would put each loser set on another band's table.
    numbers = [band_index(t.band) for t in tables]
    known = [n for n in numbers if n is not None]
    if len(known) == len(numbers) and any(b <= a for a, b in itertools.pairwise(known)):
        order = ", ".join(str(t.band) for t in tables)
        raise InconsistentData(
            f"{where}: the band table list is not in pool order (ascending band index): "
            f"{order}; its loser sets cannot be assigned to the tables."
        )
    return tables


class ChromatogramSource:
    """The chromatogram tables of one run and the reader of one candidate.

    * An ungrouped run, or a grouped run with a pooled ``chromatograms.parquet``, has one
      table. A pooled table may be unsorted (a later band can win an overlap candidate); the
      candidate index handles that.
    * A grouped run without a pooled table has one table per band, in the order the loser
      file's footer records (pool order), each with its drop set. A candidate must yield rows
      from exactly one table after the drop; two tables are refused, never summed.
    * A grouped run with band tables but no loser file (older layouts) reads the band
      tables in band order without drops, and says so in :attr:`notice`.

    A row group whose estimated Arrow size (:func:`row_group_bytes`) is at most
    :attr:`max_row_group_bytes` is read whole through the row-group cache and sliced.
    A larger one is never read whole: the candidate's rows are read with a DuckDB query on
    ``read_parquet(?)`` filtered by ``candidate_id`` and ``file_row_number``. Such a read
    scans the row group up to the candidate (about 1 s at the end of a 1,048,576-row
    layout 1 group) but holds only the candidate's rows in memory.

    Build it with :meth:`for_run`, which keeps one source per run on the result set.
    """

    #: The largest estimated row group (bytes) read whole. None means the admission limit
    #: of :data:`~mumdia_viewer.data.pqio.ROW_GROUP_CACHE` (a quarter of its size): a
    #: larger row group would not be cached, so it would be decoded whole on every read.
    max_row_group_bytes: int | None = None

    def __init__(self, rs: ResultSet, run: Run) -> None:
        self.rs = rs
        self.run = run
        self.notice: str | None = None
        self._lock = threading.Lock()
        self._indexes: dict[int, CandidateIndex] = {}
        self._id_ranges: dict[int, tuple[int, int] | None] = {}
        self._group_bytes: dict[tuple[int, int], int] = {}
        self._ids: np.ndarray | None = None
        self._band_tables: tuple[ChromatogramTable, ...] | None = None
        self.tables: tuple[ChromatogramTable, ...] = tuple(self._resolve())
        #: True when the tables are band tables (a grouped run without a pooled table).
        self.per_band: bool = bool(self.tables) and self.tables[0].band is not None
        if self.per_band or run.grouped is None:
            self._band_tables = self.tables if self.per_band else ()

    @classmethod
    def for_run(cls, rs: ResultSet, run: Run) -> ChromatogramSource:
        """The source of ``run``, built once per result set."""
        key = (_MEMO_KEY, run.index, str(run.root))
        source = rs._memo.get(key)
        if source is None:
            source = cls(rs, run)
            rs._memo[key] = source
        return source

    def __repr__(self) -> str:
        return f"ChromatogramSource({self.run.label}, {len(self.tables)} table(s))"

    # ------------------------------------------------------------------ resolution

    def _resolve(self) -> list[ChromatogramTable]:
        run = self.run
        own = run.artifacts.get("chromatograms")
        grouped = run.grouped
        if grouped is None:
            if own is None:
                raise ArtifactNotFound(f"{run.root}: the run has no chromatograms table.")
            return [_table_for(own, None, None, frozenset())]
        losers = grouped.losers
        if own is not None and (own.present or losers is None or not losers.present):
            return [_table_for(own, None, None, frozenset())]  # raises when it is missing
        if own is not None:
            self.notice = (
                f"the pooled chromatograms table recorded in the manifest is missing "
                f"({own.recorded_path or own.key}); the band tables are read with "
                f"{losers.path if losers is not None else 'the loser file'}."
            )
        if losers is not None:
            tables = _loser_tables(run, grouped, losers)
            named = {id(t.artifact) for t in tables}
            extra = [
                b.name for b, a in _band_chromatograms(grouped) if a.present and id(a) not in named
            ]
            if extra:
                note = (
                    f"band chromatogram tables not named by the loser file are ignored: "
                    f"{', '.join(extra)}."
                )
                self.notice = f"{self.notice} {note}" if self.notice else note
            return tables
        bands = _band_chromatograms(grouped)
        if not bands:
            raise ArtifactNotFound(
                f"{run.root}: the grouped run has no pooled chromatograms.parquet, no "
                "groups/overlap_losers.parquet and no band chromatogram tables."
            )
        self.notice = (
            "no groups/overlap_losers.parquet: the band tables are read in band order without "
            "dropping overlap losers, which is right only when the bands are disjoint; a "
            "candidate found in two band tables is refused."
        )
        return [_table_for(a, b.name, None, frozenset()) for b, a in bands]

    @property
    def band_tables(self) -> tuple[ChromatogramTable, ...]:
        """Every band chromatogram table of a grouped run, in band order.

        For a run read per band this is :attr:`tables`. For a pooled run it lists the band
        tables that are present, with empty drop sets because no loser file records them;
        use them only as labelled diagnostics. Empty for an ungrouped run.
        """
        if self._band_tables is None:
            grouped = self.run.grouped
            tables = []
            if grouped is not None:
                for band, artifact in _band_chromatograms(grouped):
                    if not artifact.usable:
                        continue
                    try:
                        tables.append(_table_for(artifact, band.name, None, frozenset()))
                    except ViewerError:
                        continue
            self._band_tables = tuple(tables)
        return self._band_tables

    # ------------------------------------------------------------------ lookups

    def _index(self, table: ChromatogramTable) -> CandidateIndex:
        index = self._indexes.get(id(table))
        if index is None:
            index = CandidateIndex.for_artifact(table.artifact, self.rs.cache)
            with self._lock:
                self._indexes[id(table)] = index
        return index

    def _id_range(self, table: ChromatogramTable) -> tuple[int, int] | None:
        """The footer's ``candidate_id`` range of a table; None when it has no rows."""
        key = id(table)
        if key not in self._id_ranges:
            handle = table.artifact.parquet()
            stats = handle.column_statistics("candidate_id")
            sizes = np.diff(handle.row_group_offsets())
            lo, hi, empty = None, None, True
            for size, stat in zip(sizes.tolist(), stats, strict=True):
                if size == 0:
                    continue
                empty = False
                if stat is None:
                    lo, hi = 0, 2**63 - 1
                    break
                lo = int(stat[0]) if lo is None else min(lo, int(stat[0]))
                hi = int(stat[1]) if hi is None else max(hi, int(stat[1]))
            value = None if empty or lo is None or hi is None else (lo, hi)
            with self._lock:
                self._id_ranges[key] = value
        return self._id_ranges[key]

    def _holders(self, cid: int) -> list[ChromatogramTable]:
        """The tables that yield rows for ``cid`` after their drop sets are applied."""
        out = []
        for table in self.tables:
            if cid in table.drop:
                continue
            if len(self.tables) > 1:
                bounds = self._id_range(table)
                if bounds is None or not bounds[0] <= cid <= bounds[1]:
                    continue
            index = self._index(table)
            key = _index_key(index, cid)
            if key is not None and key in index:
                out.append(table)
        return out

    def _holder(self, cid: int) -> ChromatogramTable | None:
        holders = self._holders(cid)
        if len(holders) > 1:
            names = ", ".join(str(t.band or t.path) for t in holders)
            raise InconsistentData(
                f"{self.run.root}: candidate_id {cid} has chromatogram rows in {len(holders)} "
                f"tables ({names}) after the overlap losers are dropped; a candidate must come "
                "from exactly one band table, so the rows are not combined."
            )
        return holders[0] if holders else None

    def read(self, cid: int) -> CandidateChromatogram | None:
        """The decoded chromatogram of ``cid``; None when no table holds it.

        Raises :class:`InconsistentData` when two tables yield rows after the drop.
        """
        cid = int(cid)
        table = self._holder(cid)
        return None if table is None else self.read_table(table, cid)

    def _row_group_limit(self) -> int:
        limit = self.max_row_group_bytes
        return ROW_GROUP_CACHE.max_bytes // 4 if limit is None else int(limit)

    def _is_large(self, table: ChromatogramTable, handle: ParquetHandle, row_group: int) -> bool:
        """True when the row group is too large to be read whole (see the class docstring)."""
        key = (id(table), row_group)
        size = self._group_bytes.get(key)
        if size is None:
            size = row_group_bytes(handle, row_group, _read_columns(table.layout))
            with self._lock:
                self._group_bytes[key] = size
        return size > self._row_group_limit()

    def _query_segment(
        self, table: ChromatogramTable, handle: ParquetHandle, seg: Segment, cid: int
    ) -> pa.Table:
        """The rows of one segment, read with a filtered DuckDB query (large row groups).

        The query filters on ``candidate_id`` (row-group statistics prune the other row
        groups) and on the segment's global row range. The row numbers it returns must be
        exactly that range, else the candidate index does not describe the file.
        """
        base = int(handle.row_group_offsets()[seg.row_group])
        start, stop = base + seg.start, base + seg.stop
        columns = [c for c in _read_columns(table.layout) if handle.has_column(c)]
        sql = (
            f"SELECT file_row_number, {', '.join(sql_ident(c) for c in columns)} "
            "FROM read_parquet(?, file_row_number = true) "
            "WHERE candidate_id = ? AND file_row_number >= ? AND file_row_number < ? "
            "ORDER BY file_row_number"
        )
        part = self.rs.duck.arrow(sql, [sql_path(table.path), cid, start, stop])
        got = part.column("file_row_number").to_numpy()
        if got.size != stop - start or (got.size and (got[0] != start or got[-1] != stop - 1)):
            raise InconsistentData(
                f"{table.path}, row group {seg.row_group}: the candidate index places "
                f"candidate_id {cid} at rows {start} to {stop - 1}, but {got.size} of those "
                "rows hold it; the index does not describe this file."
            )
        return part.drop_columns(["file_row_number"])

    def read_table(self, table: ChromatogramTable, cid: int) -> CandidateChromatogram | None:
        """``cid`` read from one table, ignoring its drop set (for diagnostics).

        The rows must be contiguous in the table. Each row-group part is sliced at the
        candidate's first row in that group and decoded with a fresh decoder. The part comes
        from the row group read whole through the row-group cache, or, for a row group above
        :attr:`max_row_group_bytes`, from a DuckDB query of the candidate's rows only.
        """
        cid = int(cid)
        index = self._index(table)
        key = _index_key(index, cid)
        if key is None or index.rows(key) is None:  # rows() raises when not contiguous
            return None
        handle = table.artifact.parquet()
        columns = _read_columns(table.layout)
        traces: list[Trace] = []
        row_groups: list[int] = []
        for seg in index.segments(key):
            if self._is_large(table, handle, seg.row_group):
                part = self._query_segment(table, handle, seg, cid)
            else:
                group = handle.read_row_group(seg.row_group, columns, cached=True)
                part = group.slice(seg.start, seg.length)
            where = f"{table.path}, row group {seg.row_group}"
            traces.extend(_part_traces(part, table.layout, cid, where))
            row_groups.append(seg.row_group)
        return CandidateChromatogram(
            candidate_id=cid,
            traces=tuple(traces),
            layout=table.layout,
            table=table.path,
            band=table.band,
            row_groups=tuple(row_groups),
        )

    def candidate_ids(self) -> np.ndarray:
        """Every candidate with chromatogram rows, ascending (int64), overlap losers excluded."""
        if self._ids is None:
            parts = []
            for table in self.tables:
                ids = self._index(table).ids.astype(np.int64)
                if table.drop:
                    drop = np.fromiter(table.drop, dtype=np.int64, count=len(table.drop))
                    ids = ids[~np.isin(ids, drop)]
                parts.append(ids)
            self._ids = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        return self._ids

    def band_of(self, cid: int) -> str | None:
        """The band (``gNN``) whose table yields the rows of ``cid``.

        None for an ungrouped run, for a pooled table (it has no band column) and for a
        candidate without chromatogram rows. Raises :class:`InconsistentData` when two band
        tables yield rows.
        """
        if not self.per_band:
            return None
        table = self._holder(int(cid))
        return table.band if table is not None else None

    def describe(self) -> dict[str, Any]:
        """A summary for display: tables, bands, drop counts and the notice."""
        return {
            "run": self.run.label,
            "per_band": self.per_band,
            "tables": [
                {
                    "path": str(t.path),
                    "band": t.band,
                    "position": t.position,
                    "layout": t.layout.schema_version,
                    "dropped": len(t.drop),
                }
                for t in self.tables
            ],
            "notice": self.notice,
        }
