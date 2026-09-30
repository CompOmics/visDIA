"""A DuckDB connection with bounded threads and memory and one cursor per thread.

A DuckDB connection shared between threads without cursors returns wrong results, so
every thread (a Dash callback, for example) gets its own cursor of one shared
in-memory database. Spill files go to the viewer's cache directory: the default,
``.tmp`` in the working directory, could be inside a run directory.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa


def sql_path(path: Path | str) -> str:
    """A path in the form DuckDB expects in a parameter (forward slashes)."""
    return str(path).replace("\\", "/")


def sql_ident(name: str) -> str:
    """Quote a column name for use in SQL text."""
    return '"' + name.replace('"', '""') + '"'


class DuckDB:
    """One in-memory DuckDB database; :meth:`cursor` returns the calling thread's cursor."""

    def __init__(
        self, *, threads: int = 8, memory_limit: str = "1GB", temp_directory: Path | None = None
    ) -> None:
        config: dict[str, Any] = {"threads": threads, "memory_limit": memory_limit}
        self._con = duckdb.connect(database=":memory:", config=config)
        if temp_directory is not None:
            self._con.execute(f"SET temp_directory = '{sql_path(temp_directory)}'")
        self._con.execute("SET enable_progress_bar = false")
        # Keeps parsed footers between queries. It holds no file handle (a rename onto the
        # path succeeds) and re-reads a file whose modification time changed.
        self._con.execute("SET parquet_metadata_cache = true")
        self._local = threading.local()
        self._lock = threading.Lock()

    def cursor(self) -> duckdb.DuckDBPyConnection:
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            with self._lock:
                cur = self._con.cursor()
            self._local.cursor = cur
        return cur

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None):
        return self.cursor().execute(sql, params if params is not None else [])

    def df(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> pd.DataFrame:
        return self.execute(sql, params).df()

    def arrow(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> pa.Table:
        return self.execute(sql, params).to_arrow_table()

    def rows(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> list[tuple]:
        return self.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> Any:
        row = self.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def close(self) -> None:
        self._con.close()
