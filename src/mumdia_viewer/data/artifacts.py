"""One artifact of a run: where it is, what it is, and whether it can be read."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .errors import ArtifactNotFound, SchemaVersionError
from .manifest import ArtifactRecord
from .pqio import ParquetHandle
from .reports import Report
from .schemas import VersionInfo, check_columns, infer_version


class Status(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    # A band intermediate that `groups.delete_band_intermediates` removed after pooling;
    # the manifest still lists it.
    DELETED_AFTER_POOLING = "deleted_after_pooling"


@dataclass
class Artifact:
    """A file the viewer may read.

    ``kind`` is the schema name the viewer reads it as. ``key`` is the manifest key
    when the manifest records the artifact, otherwise a key made up from the run and
    band names (for example ``chromatograms[r0]``).
    """

    kind: str
    key: str
    path: Path | None
    status: Status
    version: VersionInfo
    record: ArtifactRecord | None = None
    report: Report | None = None
    recorded_path: str | None = None
    resolution: str = "found"
    error: str | None = None
    _handle: ParquetHandle | None = field(default=None, repr=False)
    _columns_checked: bool = field(default=False, repr=False)

    @property
    def present(self) -> bool:
        return self.status is Status.PRESENT and self.path is not None

    @property
    def usable(self) -> bool:
        return self.present and self.error is None

    @property
    def rows(self) -> int | None:
        if self.record is not None and self.record.rows is not None:
            return self.record.rows
        return self.report.rows if self.report is not None else None

    @property
    def content_hash(self) -> str | None:
        """The blake3 hash the engine recorded (manifest first, then report)."""
        if self.record is not None and self.record.content_hash:
            return self.record.content_hash
        if self.report is not None and self.report.content_hash:
            return self.report.content_hash
        return None

    @property
    def stage(self) -> str | None:
        if self.record is not None and self.record.producing_stage:
            return self.record.producing_stage
        return self.report.stage if self.report is not None else None

    def require(self) -> Path:
        """The path, or raise when the artifact is absent or unreadable."""
        if not self.present or self.path is None:
            state = {
                Status.MISSING: "is missing",
                Status.DELETED_AFTER_POOLING: (
                    "was deleted after pooling (groups.delete_band_intermediates)"
                ),
            }.get(self.status, "is absent")
            where = self.recorded_path or self.key
            raise ArtifactNotFound(f"{self.key} {state}: {where}")
        if self.error is not None:
            raise SchemaVersionError(self.error)
        return self.path

    def parquet(self) -> ParquetHandle:
        """A footer-cached handle. The column contract is checked on first use."""
        path = self.require()
        if self._handle is None:
            self._handle = ParquetHandle(path)
        if not self._columns_checked:
            check_columns(
                self.kind, self._handle.schema, where=str(path), version=self.version.version
            )
            self._columns_checked = True
        return self._handle

    def identity(self) -> str:
        """A cache key for derived data: the recorded content hash, else a footer fingerprint."""
        if self.content_hash:
            return f"b3:{self.content_hash}"
        return f"fp:{self.parquet().fingerprint()}"

    def infer_version_if_unrecorded(self) -> None:
        """Fill in an inferred version for a file with no manifest record and no report."""
        if self.version.version is not None or not self.present or self.path is None:
            return
        if self._handle is None:
            self._handle = ParquetHandle(self.path)
        inferred = infer_version(self.kind, self._handle.schema)
        if inferred is not None:
            self.version = VersionInfo(self.kind, inferred, "inferred")
