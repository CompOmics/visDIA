"""Exceptions raised by the data layer."""

from __future__ import annotations


class ViewerError(Exception):
    """Base class of every error the data layer raises on purpose."""


class NotAResultDirectory(ViewerError):
    """The path is not a MuMDIA run directory or experiment directory."""


class SchemaVersionError(ViewerError):
    """An artifact has a schema version that this viewer does not read."""


class LayoutError(ViewerError):
    """The columns of an artifact match no layout that this viewer knows."""


class ArtifactNotFound(ViewerError):
    """A required artifact is absent from the directory."""


class InconsistentData(ViewerError):
    """Artifacts contradict each other, for example one candidate in two band tables."""
