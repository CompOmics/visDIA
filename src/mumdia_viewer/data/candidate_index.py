"""Candidate index: ``candidate_id`` -> the rows of that candidate in a table.

Every candidate-keyed engine table stores the rows of one candidate contiguously, and
in almost every case in ascending ``candidate_id`` order. No writer records that order
in the footer (``sorting_columns`` is empty), so the index is built from the
``candidate_id`` column itself, one row group at a time, and the ordering is checked
while it is built:

* rows of one candidate must be contiguous; a candidate found in two separate runs of
  rows is reported by :meth:`CandidateIndex.ranges` as several ranges;
* the table need not be globally sorted: the pooled chromatograms of overlapping bands
  are not, when a later band wins a candidate.

The index is three arrays (sorted unique ids, first global row, stop row) and is cached
in the viewer's cache under the artifact's content hash. Tables whose ``candidate_id``
equals the row index (``run_windows``, the library precursor tables) use
:class:`DenseIndex`, which needs no build; each read checks the id it returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .artifacts import Artifact
from .cache import Cache
from .errors import InconsistentData
from .pqio import ParquetHandle

INDEX_FORMAT = 1


@dataclass(frozen=True)
class Segment:
    """The rows of one candidate inside one row group (``start``/``stop`` are row-group local)."""

    row_group: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


class CandidateIndex:
    """Sorted unique ids with the global row range of each."""

    def __init__(
        self,
        ids: np.ndarray,
        starts: np.ndarray,
        stops: np.ndarray,
        rg_offsets: np.ndarray,
        *,
        file_sorted: bool,
        contiguous: bool,
    ) -> None:
        self.ids = ids
        self.starts = starts
        self.stops = stops
        self.rg_offsets = rg_offsets
        self.file_sorted = file_sorted
        self.contiguous = contiguous

    def __len__(self) -> int:
        """The number of distinct candidates."""
        return int(np.unique(self.ids).size) if not self.contiguous else int(self.ids.size)

    def _key(self, cid: int) -> np.generic | None:
        """``cid`` as a scalar of the ids' dtype, or None when it cannot be present.

        ``np.searchsorted`` with a Python int promotes and copies the whole ``ids``
        array on every call (about 1 ms on 683,297 ids); a typed key does not.
        """
        value = int(cid)
        if self.ids.dtype.kind in "iu":
            info = np.iinfo(self.ids.dtype)
            if not info.min <= value <= info.max:
                return None
        return self.ids.dtype.type(value)

    def __contains__(self, cid: int) -> bool:
        key = self._key(cid)
        if key is None:
            return False
        i = int(np.searchsorted(self.ids, key, side="left"))
        return i < self.ids.size and int(self.ids[i]) == int(cid)

    @property
    def num_rows(self) -> int:
        return int(self.rg_offsets[-1])

    def ranges(self, cid: int) -> list[tuple[int, int]]:
        """Global ``[start, stop)`` row ranges of ``cid`` (one range when contiguous)."""
        key = self._key(cid)
        if key is None:
            return []
        lo = int(np.searchsorted(self.ids, key, side="left"))
        hi = int(np.searchsorted(self.ids, key, side="right"))
        return [(int(self.starts[i]), int(self.stops[i])) for i in range(lo, hi)]

    def rows(self, cid: int) -> tuple[int, int] | None:
        """The single global row range of ``cid``; None when absent.

        Raises :class:`InconsistentData` when the candidate's rows are not contiguous.
        """
        found = self.ranges(cid)
        if not found:
            return None
        if len(found) > 1:
            raise InconsistentData(
                f"candidate {cid} occupies {len(found)} separate row ranges; its rows are not "
                "contiguous in this table."
            )
        return found[0]

    def segments(self, cid: int) -> list[Segment]:
        """The row-group segments of ``cid``, in file order."""
        out: list[Segment] = []
        for start, stop in self.ranges(cid):
            out.extend(_split(start, stop, self.rg_offsets))
        return out

    # ----------------------------------------------------------------- construction

    @classmethod
    def build(cls, handle: ParquetHandle, column: str = "candidate_id") -> CandidateIndex:
        """Scan ``column`` once, a row group at a time."""
        ids_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        prev: int | None = None
        pos = 0
        with handle.open() as pf:
            for rg in range(pf.num_row_groups):
                arr = pf.read_row_group(rg, columns=[column]).column(0)
                values = _to_numpy_ids(arr)
                if values.size == 0:
                    continue
                change = np.flatnonzero(values[1:] != values[:-1]) + 1
                run_starts = np.concatenate([np.zeros(1, dtype=np.int64), change.astype(np.int64)])
                run_ids = values[run_starts]
                if prev is not None and int(run_ids[0]) == prev:
                    # The first run continues the last run of the previous row group.
                    run_starts = run_starts[1:]
                    run_ids = run_ids[1:]
                ids_parts.append(run_ids)
                start_parts.append(run_starts + pos)
                prev = int(values[-1])
                pos += int(values.size)
        rg_offsets = handle.row_group_offsets()
        if not ids_parts:
            empty = np.zeros(0, dtype=np.int64)
            return cls(
                empty.astype(np.uint32), empty, empty, rg_offsets, file_sorted=True, contiguous=True
            )
        ids = np.concatenate(ids_parts)
        starts = np.concatenate(start_parts).astype(np.int64)
        stops = np.concatenate([starts[1:], np.array([pos], dtype=np.int64)])
        file_sorted = bool(ids.size < 2 or np.all(ids[1:] > ids[:-1]))
        contiguous = True
        if not file_sorted:
            order = np.argsort(ids, kind="stable")
            ids, starts, stops = ids[order], starts[order], stops[order]
            contiguous = bool(ids.size < 2 or np.all(ids[1:] != ids[:-1]))
        pos_type = np.uint32 if pos < 2**32 else np.int64
        return cls(
            ids.astype(np.uint32),
            starts.astype(pos_type),
            stops.astype(pos_type),
            rg_offsets,
            file_sorted=file_sorted,
            contiguous=contiguous,
        )

    @classmethod
    def for_artifact(
        cls, artifact: Artifact, cache: Cache | None = None, column: str = "candidate_id"
    ) -> CandidateIndex:
        """Load the cached index of an artifact, or build and cache it."""
        handle = artifact.parquet()
        name = f"candidate_index_v{INDEX_FORMAT}_{column}"
        identity = artifact.identity()
        # The recorded hash names the file the engine wrote. A file replaced without a
        # manifest or report update keeps that hash, so the entry also stores a digest
        # of the file's size and footer, and a mismatch rebuilds the index.
        digest = handle.footer_digest()
        if cache is not None:
            hit = cache.load_arrays(identity, name)
            if (
                hit is not None
                and int(hit["num_rows"][0]) == handle.num_rows
                and "footer" in hit
                and str(hit["footer"][0]) == digest
            ):
                return cls(
                    hit["ids"],
                    hit["starts"],
                    hit["stops"],
                    handle.row_group_offsets(),
                    file_sorted=bool(hit["flags"][0]),
                    contiguous=bool(hit["flags"][1]),
                )
        index = cls.build(handle, column)
        if cache is not None:
            cache.save_arrays(
                identity,
                name,
                ids=index.ids,
                starts=index.starts,
                stops=index.stops,
                flags=np.array([index.file_sorted, index.contiguous]),
                num_rows=np.array([handle.num_rows], dtype=np.int64),
                footer=np.array([digest]),
            )
        return index


class DenseIndex:
    """For tables whose ``candidate_id`` equals the row index (one row per candidate).

    Detected from footer statistics; every read made through it must check the
    returned ``candidate_id`` (see :func:`read_candidate_rows`).
    """

    def __init__(self, num_rows: int, rg_offsets: np.ndarray) -> None:
        self.n = num_rows
        self.rg_offsets = rg_offsets
        self.file_sorted = True
        self.contiguous = True

    def __len__(self) -> int:
        return self.n

    def __contains__(self, cid: int) -> bool:
        return 0 <= int(cid) < self.n

    def ranges(self, cid: int) -> list[tuple[int, int]]:
        return [(int(cid), int(cid) + 1)] if cid in self else []

    def rows(self, cid: int) -> tuple[int, int] | None:
        return (int(cid), int(cid) + 1) if cid in self else None

    def segments(self, cid: int) -> list[Segment]:
        r = self.rows(cid)
        return [] if r is None else _split(r[0], r[1], self.rg_offsets)

    @staticmethod
    def applies(handle: ParquetHandle, column: str = "candidate_id") -> bool:
        """True when every row group's min/max equals its first/last row index."""
        stats = handle.column_statistics(column)
        offsets = handle.row_group_offsets()
        if not stats or any(s is None for s in stats):
            return False
        for rg, (mn, mx) in enumerate(stats):  # type: ignore[misc]
            if int(mn) != int(offsets[rg]) or int(mx) != int(offsets[rg + 1]) - 1:
                return False
        return True


def index_for(
    artifact: Artifact, cache: Cache | None = None, *, dense: bool | None = None
) -> CandidateIndex | DenseIndex:
    """The index of an artifact: dense when the footer shows ``candidate_id == row``."""
    handle = artifact.parquet()
    if dense is None:
        dense = artifact.kind in ("run_windows", "fragment_library_precursors")
    if dense and DenseIndex.applies(handle):
        return DenseIndex(handle.num_rows, handle.row_group_offsets())
    return CandidateIndex.for_artifact(artifact, cache)


def read_candidate_rows(
    handle: ParquetHandle,
    index: CandidateIndex | DenseIndex,
    cid: int,
    columns: list[str] | None = None,
    *,
    cached: bool = True,
) -> list[tuple[int, pa.Table]]:
    """The rows of ``cid`` as (row group, table) parts, in file order.

    For a :class:`DenseIndex` the returned ``candidate_id`` is checked.
    """
    parts: list[tuple[int, pa.Table]] = []
    cols = None
    if columns is not None:
        cols = list(columns)
        if isinstance(index, DenseIndex) and "candidate_id" not in cols:
            cols.append("candidate_id")
    for seg in index.segments(cid):
        table = handle.read_row_group(seg.row_group, cols, cached=cached)
        parts.append((seg.row_group, table.slice(seg.start, seg.length)))
    if isinstance(index, DenseIndex):
        for _, t in parts:
            got = t.column("candidate_id").to_pylist()
            if got != [cid] * t.num_rows:
                raise InconsistentData(
                    f"{handle.path}: row {cid} holds candidate_id {got}, so candidate_id is not "
                    "the row index in this table."
                )
    return parts


def concat_parts(parts: list[tuple[int, pa.Table]]) -> pa.Table | None:
    """Concatenate the parts returned by :func:`read_candidate_rows` (None when empty)."""
    if not parts:
        return None
    return pa.concat_tables([t for _, t in parts])


def _split(start: int, stop: int, rg_offsets: np.ndarray) -> list[Segment]:
    out = []
    first = int(np.searchsorted(rg_offsets, start, side="right") - 1)
    last = int(np.searchsorted(rg_offsets, stop - 1, side="right") - 1)
    for rg in range(first, last + 1):
        base = int(rg_offsets[rg])
        lo = max(start, base) - base
        hi = min(stop, int(rg_offsets[rg + 1])) - base
        if hi > lo:
            out.append(Segment(rg, lo, hi))
    return out


def _to_numpy_ids(arr: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if arr.null_count:
        raise InconsistentData("candidate_id has null values.")
    return arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
