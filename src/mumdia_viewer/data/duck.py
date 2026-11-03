"""A DuckDB connection with bounded threads and memory and one cursor per thread.

A DuckDB connection shared between threads without cursors returns wrong results, so
every thread (a Dash callback, for example) gets its own cursor of one shared
in-memory database. Spill files go to the viewer's cache directory: the default,
``.tmp`` in the working directory, could be inside a run directory.

The memory limit (:func:`resolve_memory_limit`) is ``MUMDIA_VIEWER_MEMORY_LIMIT`` when it
is set, a size such as ``4GB``, ``1.5GiB`` or ``800MB`` or a share of the physical memory
such as ``40%``; otherwise a quarter of the physical memory, at least 1 GiB and at most
16 GiB. It bounds DuckDB, not the whole process, and every opened result set has its own
database (a comparison opens two). DuckDB runs on at most 8 threads.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import os
import re
import shutil
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from .errors import ViewerError

_GLOB = str.maketrans({"[": "[[]", "*": "[*]", "?": "[?]"})

MEMORY_ENV_VAR = "MUMDIA_VIEWER_MEMORY_LIMIT"
# The default limit: this share of the physical memory, clamped to the floor and ceiling.
MEMORY_SHARE = 0.25
MEMORY_FLOOR = 1 << 30
MEMORY_CEILING = 16 << 30
MAX_THREADS = 8

# DuckDB's units: KB, MB, GB and TB are powers of 1000; KiB, MiB, GiB and TiB of 1024.
_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "tib": 1 << 40,
}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+|%)\s*$")
_DETECT: Any = object()


def sql_path(path: Path | str) -> str:
    """A file path for a ``read_parquet(?)`` parameter.

    DuckDB reads that parameter as a glob pattern: a directory named ``run[1]`` would
    match ``run1`` and silently read another run's table. The glob characters are
    therefore escaped (``[`` as ``[[]``, ``*`` as ``[*]``, ``?`` as ``[?]``), and
    backslashes become forward slashes.
    """
    return str(path).replace("\\", "/").translate(_GLOB)


def sql_paths(paths) -> list[str]:
    return [sql_path(p) for p in paths]


def sql_ident(name: str) -> str:
    """Quote a column name for use in SQL text."""
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------------------------------- limits


def format_size(n: int) -> str:
    """``n`` bytes in binary units, for example ``4.0 GiB``."""
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TiB"  # unreachable


def _cgroup_memory() -> int | None:
    """The memory limit of this process's cgroup (v2, else v1), or None when unlimited."""
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(name).read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            continue
        # cgroup v2 writes "max", v1 a huge number, when there is no limit.
        if text.isdigit() and 0 < int(text) < 1 << 60:
            return int(text)
    return None


class _MemoryStatus(ctypes.Structure):
    """MEMORYSTATUSEX of the Windows API (GlobalMemoryStatusEx)."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def physical_memory() -> int | None:
    """The physical memory in bytes, or the process's smaller cgroup limit; None if unknown."""
    total: int | None = None
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        with contextlib.suppress(AttributeError, OSError):
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = int(status.ullTotalPhys)
    else:
        with contextlib.suppress(AttributeError, ValueError, OSError):
            total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        if sys.platform.startswith("linux"):
            cgroup = _cgroup_memory()
            if cgroup is not None and (total is None or cgroup < total):
                total = cgroup
    return total if total is not None and total > 0 else None


def parse_size(text: str, *, physical: int | None = None) -> int:
    """The bytes of a size such as ``4GB``, ``1.5 GiB``, ``800MB`` or ``40%``.

    The units are DuckDB's, in any case: B; KB, MB, GB, TB (powers of 1000); KiB, MiB,
    GiB, TiB (powers of 1024). A percentage is a share of ``physical``. Raises ValueError.
    """
    match = _SIZE.match(str(text))
    if match is None:
        raise ValueError(f"{text!r} is not a size; use for example 4GB, 1.5GiB, 800MB or 40%")
    number, unit = float(match.group(1)), match.group(2).lower()
    if unit == "%":
        if not 0 < number <= 100:
            raise ValueError(f"{text!r}: a percentage must be above 0 and at most 100")
        if physical is None:
            raise ValueError(
                f"{text!r}: the physical memory of this machine is unknown; give a size such as 4GB"
            )
        value = int(physical * number / 100)
    elif unit in _UNITS:
        value = int(number * _UNITS[unit])
    else:
        raise ValueError(
            f"{text!r}: unknown unit {match.group(2)!r}; use B, KB, MB, GB, TB, KiB, MiB, "
            "GiB, TiB or %"
        )
    if value <= 0:
        raise ValueError(f"{text!r} is not a positive size")
    return value


@dataclass(frozen=True)
class MemoryLimit:
    """DuckDB's memory limit in bytes and how it was chosen (``source``, in words)."""

    bytes: int
    source: str

    def __str__(self) -> str:
        return f"{format_size(self.bytes)} ({self.source})"


def resolve_memory_limit(
    value: str | int | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    physical: Any = _DETECT,
) -> MemoryLimit:
    """The memory limit of a viewer database.

    ``value`` when given (bytes, or a size of :func:`parse_size`); else
    ``MUMDIA_VIEWER_MEMORY_LIMIT``; else :data:`MEMORY_SHARE` of the physical memory,
    clamped to [:data:`MEMORY_FLOOR`, :data:`MEMORY_CEILING`] (the floor when the
    physical memory is unknown). A value that was asked for is not clamped. ``environ``
    and ``physical`` replace the process environment and the detected memory (tests).
    Raises ViewerError on a value that is not a size.
    """
    env = os.environ if environ is None else environ
    total = physical_memory() if physical is _DETECT else physical
    if value is not None and not isinstance(value, str):
        if int(value) <= 0:
            raise ViewerError(f"the memory limit must be positive, not {value!r}.")
        return MemoryLimit(int(value), "given")
    if value is not None:
        try:
            return MemoryLimit(parse_size(value, physical=total), f"given as {value.strip()}")
        except ValueError as exc:
            raise ViewerError(f"invalid memory limit: {exc}.") from None
    text = env.get(MEMORY_ENV_VAR, "").strip()
    if text:
        try:
            return MemoryLimit(parse_size(text, physical=total), f"{MEMORY_ENV_VAR}={text}")
        except ValueError as exc:
            raise ViewerError(f"{MEMORY_ENV_VAR}: {exc}.") from None
    if total is None:
        return MemoryLimit(MEMORY_FLOOR, "the physical memory is unknown")
    share = int(total * MEMORY_SHARE)
    limit = min(max(share, MEMORY_FLOOR), MEMORY_CEILING)
    source = f"a quarter of {format_size(total)} of physical memory"
    if limit != share:
        source += f", clamped to {format_size(MEMORY_FLOOR)}-{format_size(MEMORY_CEILING)}"
    return MemoryLimit(limit, source)


def usable_cpus() -> int:
    """The CPUs this process may run on (its affinity where the platform has one)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_threads() -> int:
    """DuckDB's threads: the usable CPUs, at most :data:`MAX_THREADS`."""
    return min(MAX_THREADS, usable_cpus())


# --------------------------------------------------------------------------- database


class DuckDB:
    """One in-memory DuckDB database; :meth:`cursor` returns the calling thread's cursor."""

    def __init__(
        self,
        *,
        threads: int | None = None,
        memory_limit: str | int | None = None,
        temp_directory: Path | None = None,
    ) -> None:
        """``temp_directory`` must be private to this instance (see ``Cache.temp_dir``).

        None disables spilling: DuckDB then raises an out-of-memory error instead of
        writing spill files into its default location, ``.tmp`` in the working
        directory, which could be a run directory. ``threads`` defaults to
        :func:`default_threads`, ``memory_limit`` (bytes, or a size such as ``4GB``) to
        :func:`resolve_memory_limit`.
        """
        self.limit = resolve_memory_limit(memory_limit)
        self.threads = int(threads) if threads is not None else default_threads()
        config: dict[str, Any] = {
            "threads": self.threads,
            "memory_limit": f"{max(1, self.limit.bytes >> 10)}KiB",
            # Passed as configuration, not as SQL text, so any path is safe.
            "temp_directory": str(temp_directory) if temp_directory is not None else "",
        }
        self.temp_directory = temp_directory
        self._con = duckdb.connect(database=":memory:", config=config)
        self._con.execute("SET enable_progress_bar = false")
        # The next two are GLOBAL: a plain SET applies to this connection only, and every
        # query runs on a cursor (a connection of its own).
        # Keeps parsed footers between queries. It holds no file handle (a rename onto the
        # path succeeds) and re-reads a file whose modification time changed.
        self._con.execute("SET GLOBAL parquet_metadata_cache = true")
        # Every file the viewer reads is local. DuckDB 2 prefetches the projected columns
        # of whole row groups on every thread, and MuMDIA writes about a million rows per
        # row group: on an 8.5M-row scored table those buffers alone filled a 488 MiB
        # limit at 8 threads. DuckDB 1 prefetches only remote files, so this changes
        # nothing there; a version without the setting is left as it is.
        with contextlib.suppress(duckdb.Error):
            self._con.execute("SET GLOBAL disable_parquet_prefetching = true")
        self._local = threading.local()
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def cursor(self) -> duckdb.DuckDBPyConnection:
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            with self._lock:
                cur = self._con.cursor()
            self._local.cursor = cur
        return cur

    def memory_hint(self) -> str:
        """The memory limit and how to raise it, for an out-of-memory message."""
        spill = (
            f"spill files go to {self.temp_directory}"
            if self.temp_directory is not None
            else "spilling to disk is disabled, because no private temporary directory "
            "could be made"
        )
        return (
            f"The viewer's DuckDB memory limit is {self.limit}; {spill}. Raise the limit "
            f"with {MEMORY_ENV_VAR} (for example 8GB or 50%) or mumdia-viewer --memory-limit."
        )

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None):
        """Run ``sql`` on this thread's cursor. An out-of-memory error says how to raise
        the limit (:meth:`memory_hint`) and keeps its type."""
        try:
            return self.cursor().execute(sql, params if params is not None else [])
        except duckdb.OutOfMemoryException as exc:
            raise duckdb.OutOfMemoryException(f"{exc}\n{self.memory_hint()}") from exc

    def _release(self) -> None:
        """End the previous statement on this thread's cursor.

        DuckDB keeps a parquet reader open until the cursor runs its next statement; on
        Windows that open file blocks the engine from replacing the file.
        """
        self.cursor().execute("SELECT 1").fetchall()

    def df(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> pd.DataFrame:
        out = self.execute(sql, params).df()
        self._release()
        return out

    def arrow(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> pa.Table:
        out = self.execute(sql, params).to_arrow_table()
        self._release()
        return out

    def rows(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> list[tuple]:
        out = self.execute(sql, params).fetchall()
        self._release()
        return out

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> Any:
        rows = self.rows(sql, params)
        return rows[0][0] if rows else None

    def close(self) -> None:
        """Close the database and remove this instance's spill directory."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._con.close()
        if self.temp_directory is not None:
            shutil.rmtree(self.temp_directory, ignore_errors=True)
