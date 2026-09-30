"""Verification of artifacts against the blake3 content hashes the engine recorded.

Opening a directory never hashes files: the recorded hash is trusted as the
artifact's identity. Verification is run on request. It streams the file through
blake3 (multi-threaded) and caches the verdict by (path, size, mtime_ns), so a file
that changes is verified again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import blake3

from .artifacts import Artifact
from .cache import Cache

Verdict = Literal["match", "mismatch", "no_recorded_hash", "missing"]


@dataclass(frozen=True)
class HashCheck:
    key: str
    path: Path | None
    verdict: Verdict
    recorded: str | None
    computed: str | None


def blake3_file(path: Path, chunk: int = 1 << 22) -> str:
    """blake3 hex digest of the whole file (footer included), as the engine computes it."""
    h = blake3.blake3(max_threads=blake3.blake3.AUTO)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify(artifact: Artifact, cache: Cache | None = None) -> HashCheck:
    """Compare an artifact's bytes with its recorded content hash."""
    if not artifact.present or artifact.path is None:
        return HashCheck(artifact.key, None, "missing", artifact.content_hash, None)
    recorded = artifact.content_hash
    if not recorded:
        return HashCheck(artifact.key, artifact.path, "no_recorded_hash", None, None)
    st = os.stat(artifact.path)
    memo_key = f"verify:{artifact.path}:{st.st_size}:{st.st_mtime_ns}"
    if cache is not None:
        hit = cache.load_json("verdicts", _safe(memo_key))
        if isinstance(hit, dict) and hit.get("recorded") == recorded:
            return HashCheck(
                artifact.key, artifact.path, hit["verdict"], recorded, hit.get("computed")
            )
    computed = blake3_file(artifact.path)
    verdict: Verdict = "match" if computed == recorded else "mismatch"
    if cache is not None:
        cache.save_json(
            "verdicts",
            _safe(memo_key),
            {"verdict": verdict, "recorded": recorded, "computed": computed},
        )
    return HashCheck(artifact.key, artifact.path, verdict, recorded, computed)


def _safe(text: str) -> str:
    return blake3.blake3(text.encode("utf-8")).hexdigest()
