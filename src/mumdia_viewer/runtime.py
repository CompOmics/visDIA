"""Process-level settings that keep memory and thread use predictable.

Measured on the development machine (32 threads, Windows):

* importing numpy with the default OpenBLAS thread pool commits about 740 MB of
  private memory; ``OPENBLAS_NUM_THREADS=1`` avoids that. It only takes effect when
  set before numpy is first imported, so :func:`configure_environment` must run first;
* pyarrow imports pandas lazily on its first conversion (220-280 ms); a warm-up call
  moves that cost to start-up;
* Arrow's default allocator (mimalloc) keeps freed memory; the system allocator returns
  it (see :func:`configure_environment`), and :func:`release_memory` releases what the
  pool holds.

The command-line entry point calls :func:`configure_environment` before any other
import. A notebook may call it too, before importing numpy.
"""

from __future__ import annotations

import os

ARROW_THREADS = 8


def configure_environment() -> None:
    """Set thread and allocator environment variables; call before numpy or pyarrow.

    ``ARROW_DEFAULT_MEMORY_POOL=system`` returns freed memory to the system. On the
    Astral single run it halved the viewer's private memory (1.33-1.49 GB to
    0.64-0.72 GB after 21 precursor details) for about 17% slower warm details
    (184 to 216 ms). Existing values are kept, so a user can choose otherwise.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")


def configure_arrow(threads: int = ARROW_THREADS) -> None:
    """Bound Arrow's CPU and I/O thread pools and take the lazy pandas import now."""
    import pyarrow as pa
    import pyarrow.compute as pc

    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    pc.equal(pa.array([0]), 0)
    pa.array([0]).to_pandas()


def release_memory() -> None:
    """Return memory that Arrow's allocator holds after wide reads to the system."""
    import pyarrow as pa

    pa.default_memory_pool().release_unused()
