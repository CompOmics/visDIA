"""Parsing of ``manifest.json`` (single run) and ``experiment_manifest.json`` (experiment).

Both are parsed leniently: unknown keys are kept in ``raw`` and absent keys become None,
because older releases wrote fewer fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import NotAResultDirectory
from .paths import common_prefix, recorded_out_dir

_QUALIFIED = re.compile(r"^(?P<base>.+)\[(?P<qual>[^\]]+)\]$")
_BAND = re.compile(r"^g(\d+)$")


def split_key(key: str) -> tuple[str, str | None]:
    """Split ``'chromatograms[g00]'`` into ``('chromatograms', 'g00')``."""
    m = _QUALIFIED.match(key)
    if m is None:
        return key, None
    return m.group("base"), m.group("qual")


def band_index(qualifier: str | None) -> int | None:
    """The band number of a ``gNN`` qualifier (``g100`` and above included)."""
    if qualifier is None:
        return None
    m = _BAND.match(qualifier)
    return int(m.group(1)) if m else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ArtifactRecord:
    """One entry of a manifest's ``artifacts`` map."""

    key: str
    path: str
    logical_name: str | None
    schema_name: str | None
    schema_version: int | None
    rows: int | None
    content_hash: str | None
    producing_stage: str | None
    config_hash: str | None
    format: str | None

    @property
    def base_key(self) -> str:
        return split_key(self.key)[0]

    @property
    def qualifier(self) -> str | None:
        return split_key(self.key)[1]

    @classmethod
    def from_json(cls, key: str, data: dict[str, Any]) -> ArtifactRecord:
        return cls(
            key=key,
            path=str(data.get("path", "")),
            logical_name=data.get("logical_name"),
            schema_name=data.get("schema_name"),
            schema_version=_int_or_none(data.get("schema_version")),
            rows=_int_or_none(data.get("rows")),
            content_hash=data.get("content_hash"),
            producing_stage=data.get("producing_stage"),
            config_hash=data.get("config_hash"),
            format=data.get("format"),
        )


@dataclass(frozen=True)
class InputRecord:
    """One entry of a manifest's ``inputs`` map (an mzML, a library or a FASTA)."""

    key: str
    path: str
    bytes: int | None
    content_hash: str | None


@dataclass
class Manifest:
    """A parsed manifest. ``kind`` is ``'run'`` or ``'experiment'``."""

    path: Path
    kind: Literal["run", "experiment"]
    mumdia_version: str | None
    git_sha: str | None
    commit_date: str | None
    cli_args: list[str]
    config_json: str | None
    config: dict[str, Any]
    config_hash: str | None
    model_identities: dict[str, Any]
    inputs: dict[str, InputRecord]
    inputs_hashed_at: str | None
    artifacts: dict[str, ArtifactRecord]
    experiment: dict[str, Any] | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def recorded_out_dir(self) -> str | None:
        """The ``--out-dir`` of the recorded command line, or the artifacts' common prefix.

        Early manifests have no ``cli_args``; for them the common directory prefix of
        the artifact paths inside the run is used. Library-input records point outside
        the run and are left out of that prefix.
        """
        out = recorded_out_dir(self.cli_args)
        if out is not None:
            return out
        inside = [
            a.path
            for a in self.artifacts.values()
            if a.path and a.producing_stage not in ("library-input",)
        ]
        return common_prefix(inside)

    def config_get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup in the resolved configuration.

        Example: ``config_get("extract", "frag_tol_ppm")``.
        """
        node: Any = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _parse_config(config_json: Any) -> dict[str, Any]:
    if isinstance(config_json, dict):
        return config_json
    if not isinstance(config_json, str) or not config_json:
        return {}
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_manifest(path: Path) -> Manifest:
    """Read a ``manifest.json`` or ``experiment_manifest.json``."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotAResultDirectory(f"{path}: cannot read the manifest ({exc}).") from exc
    if not isinstance(raw, dict):
        raise NotAResultDirectory(f"{path}: the manifest is not a JSON object.")
    kind: Literal["run", "experiment"] = (
        "experiment" if path.name == "experiment_manifest.json" else "run"
    )
    experiment = raw.get("experiment")
    if kind == "experiment" and not isinstance(experiment, dict):
        # The oldest experiment manifests have no envelope: runs and tables at top level.
        experiment = {k: raw[k] for k in ("runs", "scored_combined", "lfq", "mbr") if k in raw}
    artifacts = {
        str(k): ArtifactRecord.from_json(str(k), v)
        for k, v in (raw.get("artifacts") or {}).items()
        if isinstance(v, dict)
    }
    inputs = {
        str(k): InputRecord(
            key=str(k),
            path=str(v.get("path", "")),
            bytes=_int_or_none(v.get("bytes")),
            content_hash=v.get("content_hash"),
        )
        for k, v in (raw.get("inputs") or {}).items()
        if isinstance(v, dict)
    }
    config_json = raw.get("config_json")
    return Manifest(
        path=path,
        kind=kind,
        mumdia_version=raw.get("mumdia_version"),
        git_sha=raw.get("git_sha"),
        commit_date=raw.get("commit_date"),
        cli_args=[str(a) for a in (raw.get("cli_args") or [])],
        config_json=config_json if isinstance(config_json, str) else None,
        config=_parse_config(config_json),
        config_hash=raw.get("config_hash"),
        model_identities=dict(raw.get("model_identities") or {}),
        inputs=inputs,
        inputs_hashed_at=raw.get("inputs_hashed_at"),
        artifacts=artifacts,
        experiment=experiment if kind == "experiment" else None,
        raw=raw,
    )
