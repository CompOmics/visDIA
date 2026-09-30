"""Parquet access with cached footers and no long-lived file handles.

On Windows an open file handle blocks the engine from replacing or deleting the file
(it publishes every artifact by rename). A :class:`ParquetHandle` therefore caches the
footer (``pyarrow.parquet.FileMetaData``) and opens the file only for the duration of
one read. The cached footer is dropped when the file's size or modification time
changes.

The smallest unit ``pyarrow`` can read is a row group, so per-candidate reads read one
row group with a column projection and slice it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class RowGroupCache:
    """A process-wide LRU of decoded row groups, bounded by Arrow bytes.

    Neighbouring candidates and neighbouring scans share row groups, so a small cache
    makes stepping through them cheap. Keys include the file's (size, mtime_ns), so a
    replaced file never returns stale rows.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._items: dict[tuple, pa.Table] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> pa.Table | None:
        with self._lock:
            table = self._items.pop(key, None)
            if table is not None:
                self._items[key] = table  # most recently used last
            return table

    def put(self, key: tuple, table: pa.Table) -> None:
        size = table.nbytes
        if size > self.max_bytes // 4:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._items[key] = table
            self._bytes += size
            while self._bytes > self.max_bytes and self._items:
                oldest = next(iter(self._items))  # least recently used first
                self._bytes -= self._items.pop(oldest).nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    @property
    def nbytes(self) -> int:
        return self._bytes


ROW_GROUP_CACHE = RowGroupCache()


class ParquetHandle:
    """Footer-cached access to one parquet file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._metadata: pq.FileMetaData | None = None
        self._offsets: np.ndarray | None = None

    def _current_stamp(self) -> tuple[int, int]:
        st = os.stat(self.path)
        return (st.st_size, st.st_mtime_ns)

    @property
    def stamp(self) -> tuple[int, int]:
        """(size, mtime_ns) of the file when its footer was last read."""
        self.metadata()
        assert self._stamp is not None
        return self._stamp

    def metadata(self) -> pq.FileMetaData:
        stamp = self._current_stamp()
        with self._lock:
            if self._metadata is None or stamp != self._stamp:
                self._metadata = pq.read_metadata(self.path)
                self._stamp = stamp
                self._offsets = None
            return self._metadata

    @property
    def schema(self) -> pa.Schema:
        return self.metadata().schema.to_arrow_schema()

    @property
    def num_rows(self) -> int:
        return self.metadata().num_rows

    @property
    def num_row_groups(self) -> int:
        return self.metadata().num_row_groups

    def row_group_offsets(self) -> np.ndarray:
        """Global index of the first row of each row group, plus the total (length n+1)."""
        md = self.metadata()
        with self._lock:
            if self._offsets is None:
                sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
                self._offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]).astype(
                    np.int64
                )
            return self._offsets

    def has_column(self, name: str) -> bool:
        return name in self.schema.names

    @contextmanager
    def open(self, *, memory_map: bool = False) -> Iterator[pq.ParquetFile]:
        """Open the file for one read; the handle is closed on exit."""
        pf = pq.ParquetFile(self.path, metadata=self.metadata(), memory_map=memory_map)
        try:
            yield pf
        finally:
            pf.close()

    def _present(self, columns: Sequence[str] | None) -> list[str] | None:
        if columns is None:
            return None
        names = set(self.schema.names)
        return [c for c in columns if c in names]

    def read_row_group(
        self, index: int, columns: Sequence[str] | None = None, *, cached: bool = False
    ) -> pa.Table:
        """One row group with the requested columns (absent columns are skipped).

        With ``cached=True`` the decoded row group is kept in :data:`ROW_GROUP_CACHE`.
        """
        present = self._present(columns)
        key = None
        if cached:
            key = (str(self.path), self.stamp, index, tuple(present) if present is not None else None)
            hit = ROW_GROUP_CACHE.get(key)
            if hit is not None:
                return hit
        with self.open() as pf:
            table = pf.read_row_group(index, columns=present)
        if key is not None:
            ROW_GROUP_CACHE.put(key, table)
        return table

    def read_rows(
        self, start: int, stop: int, columns: Sequence[str] | None = None, *, cached: bool = False
    ) -> list[tuple[int, pa.Table]]:
        """Global rows ``[start, stop)`` as (row group, table) parts, one per row group touched.

        The parts are kept separate because some layouts (chromatograms v2) restart their
        encoding at every row group.
        """
        if stop <= start:
            return []
        offsets = self.row_group_offsets()
        first = int(np.searchsorted(offsets, start, side="right") - 1)
        last = int(np.searchsorted(offsets, stop - 1, side="right") - 1)
        parts = []
        for rg in range(first, last + 1):
            lo = max(start, int(offsets[rg])) - int(offsets[rg])
            hi = min(stop, int(offsets[rg + 1])) - int(offsets[rg])
            table = self.read_row_group(rg, columns, cached=cached)
            parts.append((rg, table.slice(lo, hi - lo)))
        return parts

    def read_row_groups(
        self, indices: Sequence[int], columns: Sequence[str] | None = None
    ) -> pa.Table:
        with self.open() as pf:
            return pf.read_row_groups(list(indices), columns=self._present(columns))

    def read(self, columns: Sequence[str] | None = None) -> pa.Table:
        """The whole file with a column projection. Use only on small tables."""
        with self.open() as pf:
            return pf.read(columns=self._present(columns))

    def iter_column(self, column: str, batch_size: int = 1 << 22) -> Iterator[tuple[int, pa.Array]]:
        """Stream one column as (row group, array) pairs, one row group at a time."""
        with self.open() as pf:
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=[column])
                yield rg, table.column(0).combine_chunks()

    def column_statistics(self, column: str) -> list[tuple[Any, Any] | None]:
        """Footer min/max of ``column`` per row group (None where absent)."""
        md = self.metadata()
        # Resolve the parquet leaf column by its path: the arrow field index differs
        # from the leaf index once a nested (list) column precedes it.
        leaf = None
        schema = md.schema
        for j in range(len(schema)):
            if schema.column(j).path == column:
                leaf = j
                break
        if leaf is None:
            return [None] * md.num_row_groups
        out: list[tuple[Any, Any] | None] = []
        for rg in range(md.num_row_groups):
            stats = md.row_group(rg).column(leaf).statistics
            if stats is None or not stats.has_min_max:
                out.append(None)
            else:
                out.append((stats.min, stats.max))
        return out

    def fingerprint(self) -> str:
        """An identity for a file without a recorded content hash.

        blake3 over (size, mtime_ns, footer bytes). It changes whenever the file is
        rewritten or copied, which is safe for caching.
        """
        size, mtime = self._current_stamp()
        h = blake3.blake3()
        h.update(f"{size}:{mtime}:".encode())
        with open(self.path, "rb") as fh:
            if size >= 12:
                fh.seek(size - 8)
                tail = fh.read(8)
                footer_len = int.from_bytes(tail[:4], "little")
                start = max(0, size - 8 - footer_len)
                fh.seek(start)
                h.update(fh.read(size - start))
        return h.hexdigest()


def list_values(array: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Flat values and offsets of a list or large_list array (nulls become empty lists)."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.null_count:
        array = pc.fill_null(array, pa.scalar([], type=array.type))
    offsets = np.asarray(array.offsets, dtype=np.int64)
    values = array.values.to_numpy(zero_copy_only=False)
    if offsets.size and offsets[0] != 0:
        values = values[offsets[0] : offsets[-1]]
        offsets = offsets - offsets[0]
    else:
        values = values[: offsets[-1]] if offsets.size else values[:0]
    return values, offsets
