"""The viewer's own cache directory.

Derived data (candidate indexes, hash verdicts, DuckDB spill files) is stored outside
every run directory:

* ``MUMDIA_VIEWER_CACHE_DIR`` when it is set;
* ``%LOCALAPPDATA%\\mumdia-viewer`` on Windows;
* ``~/Library/Caches/mumdia-viewer`` on macOS;
* ``$XDG_CACHE_HOME/mumdia-viewer`` (when absolute) or ``~/.cache/mumdia-viewer`` elsewhere.

Entries are keyed by an artifact identity, normally the blake3 content hash the engine
recorded, so a cache entry can never describe a different file. The engine's own cache
(``%LOCALAPPDATA%\\mumdia\\cache``, ``MUMDIA_CACHE_DIR``) is a different directory.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ENV_VAR = "MUMDIA_VIEWER_CACHE_DIR"
FORMAT_VERSION = 1
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def default_cache_root() -> Path:
    """The platform cache directory for the viewer (see the module docstring)."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "mumdia-viewer"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mumdia-viewer"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and os.path.isabs(xdg):
        return Path(xdg) / "mumdia-viewer"
    return Path.home() / ".cache" / "mumdia-viewer"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


class Cache:
    """Keyed storage for derived data. It refuses to write inside a run directory.

    Nothing is created on disk until the first write, so a cache whose root lies inside
    a run directory is refused (:meth:`forbid`) before it could create anything there.
    When the cache directory cannot be created, the cache degrades to memory only and
    ``writable`` is False.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_root()
        self._forbidden: list[Path] = []
        self._memory: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self._writable: bool | None = None  # decided on first use

    @property
    def writable(self) -> bool:
        """True when entries go to disk. Creates the root on first use, never inside a
        forbidden directory."""
        if self._writable is None:
            if any(_is_within(self.root, d) for d in self._forbidden):
                self._writable = False
            else:
                try:
                    (self.root / f"v{FORMAT_VERSION}").mkdir(parents=True, exist_ok=True)
                    self._writable = True
                except OSError:
                    self._writable = False
        return self._writable

    def forbid(self, directory: Path) -> None:
        """Never write inside ``directory`` (a run or experiment directory)."""
        self._forbidden.append(Path(directory))
        if _is_within(self.root, Path(directory)):
            self._writable = False

    def is_forbidden(self, path: Path) -> bool:
        return any(_is_within(path, d) for d in self._forbidden)

    def entry_dir(self, identity: str) -> Path:
        """``<root>/v1/<identity[:2]>/<identity>/`` for a sanitised identity."""
        safe = _SAFE.sub("_", identity)
        kind, _, digest = safe.partition("_")
        key = digest or kind
        return self.root / f"v{FORMAT_VERSION}" / kind / key[:2] / key

    def temp_dir(self) -> Path | None:
        """A new spill directory for one DuckDB instance, or None when none can be made.

        DuckDB spill files are not safe to share between instances, so every call returns
        a directory of its own: under the cache root when it is writable, else a private
        directory in the system's temporary directory. Never inside a forbidden directory.
        """
        name = f"{os.getpid()}-{uuid.uuid4().hex}"
        candidates = []
        if self.writable:
            candidates.append(self.root / "tmp" / name)
        candidates.append(Path(tempfile.gettempdir()) / "mumdia-viewer" / name)
        for path in candidates:
            if self.is_forbidden(path):
                continue
            try:
                path.mkdir(parents=True, exist_ok=False)
            except OSError:
                continue
            return path
        return None

    def _target(self, identity: str, name: str) -> Path:
        path = self.entry_dir(identity) / name
        if any(_is_within(path, d) for d in self._forbidden):
            raise PermissionError(f"refusing to write cache data inside a run directory: {path}")
        return path

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def save_arrays(self, identity: str, name: str, **arrays: np.ndarray) -> None:
        """Store arrays as an ``.npz`` entry (in memory only when not writable)."""
        with self._lock:
            self._memory[(identity, name)] = {k: np.asarray(v) for k, v in arrays.items()}
        if not self.writable:
            return
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        with contextlib.suppress(OSError):
            self._atomic_write(self._target(identity, name + ".npz"), buf.getvalue())

    def load_arrays(self, identity: str, name: str) -> dict[str, np.ndarray] | None:
        with self._lock:
            hit = self._memory.get((identity, name))
        if hit is not None:
            return hit
        if not self.writable:
            return None
        path = self.entry_dir(identity) / (name + ".npz")
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                arrays = {k: npz[k] for k in npz.files}
        except (OSError, ValueError):
            return None
        with self._lock:
            self._memory[(identity, name)] = arrays
        return arrays

    def save_json(self, identity: str, name: str, value: Any) -> None:
        with self._lock:
            self._memory[(identity, name + ".json")] = value
        if not self.writable:
            return
        try:
            data = json.dumps(value, sort_keys=True).encode("utf-8")
            self._atomic_write(self._target(identity, name + ".json"), data)
        except (OSError, TypeError):
            pass

    def load_json(self, identity: str, name: str) -> Any:
        with self._lock:
            if (identity, name + ".json") in self._memory:
                return self._memory[(identity, name + ".json")]
        if not self.writable:
            return None
        path = self.entry_dir(identity) / (name + ".json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        with self._lock:
            self._memory[(identity, name + ".json")] = value
        return value
