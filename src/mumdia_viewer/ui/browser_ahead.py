"""Work the server starts ahead while the browser draws a new identification page.

A new page reaches the browser with its first table's first block. Drawing it takes the
browser a few hundred milliseconds, and only then does it ask for the preview of the
page's first selection. Meanwhile the server reads that precursor's detail (the
preview's slow part) in a background thread; the preview callback waits for it instead
of reading it again. The detail read does not take the data layer's table lock, so it
does not hold up the first table. (Starting the child panels' queries ahead as well
was measured: it made the six-run experiment's first table 0.26 s later, so the
browser asks for them once that table shows its rows.) Nothing is kept beyond a few
pages, and a failure here only means that the preview reads the detail itself.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

log = logging.getLogger(__name__)

# Off in the tests that count the work a layout does.
ENABLED = True
# How long the preview waits for a read that is on its way (seconds).
WAIT_S = 10.0
_MAX = 16

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ib-ahead")
_JOBS: OrderedDict[tuple[Any, ...], Future] = OrderedDict()
_LOCK = threading.Lock()


def _key(rs: Any, run: Any, cid: Any) -> tuple[Any, ...] | None:
    try:
        return (id(rs), str(run or ""), int(cid))
    except (TypeError, ValueError):
        return None


def start_detail(rs: Any, run: str, cid: Any, warm: Callable[[Any, str, Any], bool]) -> None:
    """Read the detail of the precursor (``run``, ``cid``) in the background."""
    key = _key(rs, run, cid)
    if not ENABLED or key is None:
        return
    with _LOCK:
        if key in _JOBS:
            _JOBS.move_to_end(key)
            return
        try:
            _JOBS[key] = _POOL.submit(warm, rs, str(run or ""), cid)
        except RuntimeError:  # the pool is shut down (the interpreter exits)
            return
        while len(_JOBS) > _MAX:
            _JOBS.popitem(last=False)


def wait_detail(rs: Any, run: str, cid: Any) -> None:
    """Wait for the detail read started ahead for this precursor, if there is one."""
    key = _key(rs, run, cid)
    if key is None:
        return
    with _LOCK:
        future = _JOBS.get(key)
    if future is None:
        return
    try:
        future.result(timeout=WAIT_S)
    except Exception:  # the preview reads it itself
        log.debug("detail read ahead failed", exc_info=True)
