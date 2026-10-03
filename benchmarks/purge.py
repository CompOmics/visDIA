"""Evict files from the operating system's file cache, for cold-read measurements.

Windows: opening a file with ``FILE_FLAG_NO_BUFFERING`` and closing it drops its pages
from the standby cache (the method ``fio`` uses). Linux: ``posix_fadvise(DONTNEED)``.
Elsewhere this is a no-op and the measurement is labelled accordingly.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path


def purge_file(path: Path) -> bool:
    """Drop one file's cached pages; True when the platform supports it."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        GENERIC_READ = 0x80000000
        SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004
        OPEN_EXISTING = 3
        FILE_FLAG_NO_BUFFERING = 0x20000000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        handle = kernel32.CreateFileW(
            str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, None
        )
        if handle in (None, wintypes.HANDLE(-1).value):
            return False
        kernel32.CloseHandle(handle)
        return True
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    return False


def purge_tree(roots: Iterable[Path]) -> int:
    """Purge every file under the given directories; returns the number purged."""
    n = 0
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.is_file() and purge_file(path):
                n += 1
    return n
