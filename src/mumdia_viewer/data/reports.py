"""Reading of ``<artifact>.report.json`` files and other JSON sidecars.

A report records the artifact's schema, rows, content hash, the parameters the stage
used, summary statistics and the stage wall time. Not every artifact has one (for
example the per-run ``scored.parquet`` of an experiment, the LFQ tables and the
multi-head library table), so every reader here returns None for a missing file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or return None when the file is missing or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def report_path(artifact_path: Path) -> Path:
    """``<artifact>.report.json`` next to an artifact."""
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(artifact_path.name + ".report.json")


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Report:
    """A parsed ``<artifact>.report.json``.

    ``logical_name`` is the schema name, not the manifest key (``scored_combined`` has
    ``logical_name`` ``psms_scored``).
    """

    path: Path
    logical_name: str | None
    schema_name: str | None
    schema_version: int | None
    stage: str | None
    rows: int | None
    content_hash: str | None
    params: dict[str, Any]
    stats: dict[str, Any]
    model_identity: Any
    elapsed_ms: int | None
    raw: dict[str, Any] = field(repr=False)


def load_report(artifact_path: Path) -> Report | None:
    """The report of an artifact, or None when it has none."""
    path = report_path(artifact_path)
    data = load_json(path)
    if data is None:
        return None
    params = data.get("params")
    stats = data.get("stats")
    return Report(
        path=path,
        logical_name=data.get("logical_name"),
        schema_name=data.get("schema_name"),
        schema_version=_int(data.get("schema_version")),
        stage=data.get("stage"),
        rows=_int(data.get("rows")),
        content_hash=data.get("content_hash"),
        params=params if isinstance(params, dict) else {},
        stats=stats if isinstance(stats, dict) else {},
        model_identity=data.get("model_identity"),
        elapsed_ms=_int(data.get("elapsed_ms")),
        raw=data,
    )


def normalise_enum(value: Any) -> str:
    """Compare engine enum spellings across files: ``NnTorch``, ``nn_torch`` -> ``nntorch``."""
    return "".join(ch for ch in str(value) if ch not in "_ -").lower()


@dataclass(frozen=True)
class StageTiming:
    """Wall time of one stage in one directory, as recorded by the engine."""

    directory: str
    stage: str
    elapsed_ms: int
    source: str
    artifacts: tuple[str, ...]


def stage_timings(entries: Iterable[tuple[str, Report]]) -> list[StageTiming]:
    """Group reports by (directory, stage) and take the largest ``elapsed_ms``.

    Every artifact of a multi-artifact stage repeats the stage time, so summing the
    reports would count a stage several times.
    """
    groups: dict[tuple[str, str], list[tuple[str, Report]]] = {}
    for directory, report in entries:
        if report.stage is None or report.elapsed_ms is None:
            continue
        groups.setdefault((directory, report.stage), []).append((directory, report))
    out = []
    for (directory, stage), items in groups.items():
        best = max(r.elapsed_ms or 0 for _, r in items)
        names = tuple(sorted({r.logical_name or r.path.name for _, r in items}))
        out.append(StageTiming(directory, stage, best, "report.json", names))
    out.sort(key=lambda t: (t.directory, -t.elapsed_ms))
    return out
