"""The data layer: a plain Python API over MuMDIA outputs, independent of the UI.

It returns pandas DataFrames, numpy arrays, dataclasses and dicts, and never writes
to a run directory. Start with :func:`open_results`.
"""

from .artifacts import Artifact, Status
from .cache import Cache
from .discovery import Band, GroupedLayout, Notice, ResultSet, Run, open_results
from .duck import DuckDB
from .errors import (
    ArtifactNotFound,
    InconsistentData,
    LayoutError,
    NotAResultDirectory,
    SchemaVersionError,
    ViewerError,
)

__all__ = [
    "Artifact",
    "ArtifactNotFound",
    "Band",
    "Cache",
    "DuckDB",
    "GroupedLayout",
    "InconsistentData",
    "LayoutError",
    "NotAResultDirectory",
    "Notice",
    "ResultSet",
    "Run",
    "SchemaVersionError",
    "Status",
    "ViewerError",
    "open_results",
]
