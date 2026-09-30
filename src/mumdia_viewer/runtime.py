"""Process-level settings that keep memory and thread use predictable.

Measured on the development machine (32 threads, Windows):

* importing numpy with the default OpenBLAS thread pool commits about 740 MB of
  private memory; ``OPENBLAS_NUM_THREADS=1`` avoids that. It only takes effect when
  set before numpy is first imported, so :func:`configure_environment` must run first;
* pyarrow imports pandas lazily on its first conversion (220-280 ms); a warm-up call
  moves that cost to start-up;
* Arrow's allocator keeps freed memory; :func:`release_memory` returns it after wide
  reads.

The command-line entry point calls :func:`configure_environment` before any other
import. A notebook may call it too, before importing numpy.
"""

from __future__ import annotations

import os

ARROW_THREADS = 8


def configure_environment() -> None:
    """Set thread-count environment variables; call before numpy is imported."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


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
