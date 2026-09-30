"""Resolution of the paths recorded in manifests and reports.

The engine records artifact paths as ``<--out-dir as typed>/<relative path>``: absolute
or relative, with mixed separators on Windows. A directory that was moved or copied
therefore holds paths that point at the old location. This module re-roots them:

1. A recorded path that starts with the recorded ``--out-dir`` is re-rooted onto the
   directory that was opened. If the re-rooted file does not exist, the artifact is
   missing. The old location is never used in its place, because it may belong to
   another run.
2. A path outside the recorded ``--out-dir`` (an input library, a FASTA file) is looked
   up through the user's remapped roots, then as recorded when it is absolute.

A bare file name is never searched for, because file names repeat across run
directories and bands.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_DRIVE = re.compile(r"^[A-Za-z]:")
_SEP = re.compile(r"[\\/]+")

Resolution = Literal["rerooted", "remapped", "as_recorded", "missing"]


def split_parts(recorded: str) -> list[str]:
    """Split a recorded path on both separators, dropping empty and ``.`` parts."""
    return [p for p in _SEP.split(recorded) if p not in ("", ".")]


def is_absolute_recorded(recorded: str) -> bool:
    """True for a drive-letter, UNC or POSIX absolute path, on any platform."""
    return bool(_DRIVE.match(recorded)) or recorded.startswith(("/", "\\"))


def is_windows_origin(recorded: str) -> bool:
    """True when a path was written on Windows, where comparisons ignore case."""
    return bool(_DRIVE.match(recorded)) or "\\" in recorded


def recorded_out_dir(cli_args: Sequence[str] | None) -> str | None:
    """The ``--out-dir`` value of a recorded command line (long form, as the engine parses)."""
    if not cli_args:
        return None
    for i, arg in enumerate(cli_args):
        if arg == "--out-dir" and i + 1 < len(cli_args):
            return cli_args[i + 1]
        if arg.startswith("--out-dir="):
            return arg.split("=", 1)[1]
    return None


def common_prefix(paths: Iterable[str]) -> str | None:
    """The longest common directory prefix of recorded paths, or None when there is none."""
    paths = list(paths)
    split = [split_parts(p)[:-1] for p in paths]
    split = [s for s in split if s]
    if not split:
        return None
    fold = any(is_windows_origin(p) for p in paths)
    first = split[0]
    n = len(first)
    for other in split[1:]:
        m = 0
        for a, b in zip(first, other, strict=False):
            if (a.casefold() == b.casefold()) if fold else (a == b):
                m += 1
            else:
                break
        n = min(n, m)
    if n == 0:
        return None
    return "/".join(first[:n])


def _strip(parts: list[str], prefix: list[str], fold: bool) -> list[str] | None:
    if len(parts) < len(prefix):
        return None
    head = parts[: len(prefix)]
    same = (
        [p.casefold() for p in head] == [p.casefold() for p in prefix] if fold else head == prefix
    )
    return parts[len(prefix) :] if same else None


@dataclass(frozen=True)
class ResolvedPath:
    """A recorded path and the file it resolved to, if any."""

    recorded: str
    path: Path | None
    how: Resolution
    inside_root: bool

    @property
    def exists(self) -> bool:
        return self.path is not None


class PathResolver:
    """Re-roots recorded paths onto the directory that was opened."""

    def __init__(
        self,
        root: Path,
        recorded_root: str | None,
        remaps: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.recorded_root = recorded_root
        self._root_parts = split_parts(recorded_root) if recorded_root else None
        self._remaps = [
            (split_parts(old), Path(new), is_windows_origin(old))
            for old, new in (remaps or {}).items()
        ]

    def relative_parts(self, recorded: str) -> list[str] | None:
        """The parts of ``recorded`` below the recorded out-dir, or None when outside it."""
        if self._root_parts is None:
            return None
        fold = is_windows_origin(recorded) or is_windows_origin(self.recorded_root or "")
        return _strip(split_parts(recorded), self._root_parts, fold)

    def resolve(self, recorded: str) -> ResolvedPath:
        rel = self.relative_parts(recorded)
        if rel is not None:
            path = self.root.joinpath(*rel)
            if path.exists():
                return ResolvedPath(recorded, path, "rerooted", True)
            return ResolvedPath(recorded, None, "missing", True)
        for prefix, target, fold in self._remaps:
            tail = _strip(split_parts(recorded), prefix, fold)
            if tail is not None:
                path = target.joinpath(*tail)
                if path.exists():
                    return ResolvedPath(recorded, path, "remapped", False)
                return ResolvedPath(recorded, None, "missing", False)
        if is_absolute_recorded(recorded):
            path = Path(recorded)
            if path.exists():
                return ResolvedPath(recorded, path, "as_recorded", False)
        return ResolvedPath(recorded, None, "missing", False)
