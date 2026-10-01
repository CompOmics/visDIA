"""A DuckDB connection with bounded threads and memory and one cursor per thread.

A DuckDB connection shared between threads without cursors returns wrong results, so
every thread (a Dash callback, for example) gets its own cursor of one shared
in-memory database. Spill files go to the viewer's cache directory: the default,
``.tmp`` in the working directory, could be inside a run directory.
"""

from __future__ import annotations

import atexit
import contextlib
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

_GLOB = str.maketrans({"[": "[[]", "*": "[*]", "?": "[?]"})


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


class DuckDB:
    """One in-memory DuckDB database; :meth:`cursor` returns the calling thread's cursor."""

    def __init__(
        self,
        *,
        threads: int = 8,
        memory_limit: str = "512MB",
        temp_directory: Path | None = None,
    ) -> None:
        """``temp_directory`` must be private to this instance (see ``Cache.temp_dir``).

        None disables spilling: DuckDB then raises an out-of-memory error instead of
        writing spill files into its default location, ``.tmp`` in the working
        directory, which could be a run directory.
        """
        config: dict[str, Any] = {
            "threads": threads,
            "memory_limit": memory_limit,
            # Passed as configuration, not as SQL text, so any path is safe.
            "temp_directory": str(temp_directory) if temp_directory is not None else "",
        }
        self.temp_directory = temp_directory
        self._con = duckdb.connect(database=":memory:", config=config)
        self._con.execute("SET enable_progress_bar = false")
        # Keeps parsed footers between queries. It holds no file handle (a rename onto the
        # path succeeds) and re-reads a file whose modification time changed.
        self._con.execute("SET parquet_metadata_cache = true")
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

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None):
        return self.cursor().execute(sql, params if params is not None else [])

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
