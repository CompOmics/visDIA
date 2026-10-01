"""Exports of tables: every row of an identification table, as TSV.

A table export holds every row that passes the table's filters, not only the page on
screen, with the data layer's columns as they are (the engine's values; derived columns
keep the names the table gives them). The rows are read in pages of
:data:`tables.MAX_LIMIT`. :func:`table_export` returns the text and a description that
names the row unit, the q column and the filters, for the file name and the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import pandas as pd

from .discovery import ResultSet
from .errors import ViewerError
from .tables import MAX_LIMIT, TableQuery, identification_table

__all__ = ["EXPORT_MAX_ROWS", "TableExport", "table_export", "table_frame"]

# A larger export is refused; the filters can make it smaller.
EXPORT_MAX_ROWS = 2_000_000
HIDDEN = ("source", "rn", "pos", "file_row_number")


@dataclass(frozen=True)
class TableExport:
    text: str
    rows: int
    description: str
    filename: str


def table_frame(
    rs: ResultSet, query: TableQuery, *, max_rows: int = EXPORT_MAX_ROWS
) -> pd.DataFrame:
    """Every row of ``query`` (its offset and limit are ignored), in the table's order."""
    base = replace(query, offset=0, limit=MAX_LIMIT)
    first = identification_table(rs, base)
    if first.total > max_rows:
        raise ViewerError(
            f"The table has {first.total:,} rows; an export holds at most {max_rows:,}. "
            "Narrow the filters."
        )
    frames = [first.rows]
    offset = len(first.rows)
    while offset < first.total:
        page = identification_table(rs, replace(base, offset=offset))
        if not len(page.rows):
            break
        frames.append(page.rows)
        offset += len(page.rows)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df = df.drop(columns=[c for c in HIDDEN if c in df.columns])
    df.attrs["description"] = first.description
    df.attrs["total"] = first.total
    return df


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:60] or "table"


def table_export(
    rs: ResultSet, query: TableQuery, *, max_rows: int = EXPORT_MAX_ROWS
) -> TableExport:
    """The TSV of every row of ``query``; booleans as true/false, missing values empty."""
    df = table_frame(rs, query, max_rows=max_rows)
    text = df.to_csv(sep="\t", index=False, lineterminator="\n", na_rep="")
    name = _slug(f"{rs.root.name}-{query.unit}")
    if query.threshold is not None:
        name += f"-q{query.threshold:g}"
    return TableExport(
        text=text,
        rows=len(df),
        description=str(df.attrs.get("description", "")),
        filename=f"{name}.tsv",
    )
