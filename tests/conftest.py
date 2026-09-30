"""Shared pytest fixtures.

Committed fixture outputs live in ``tests/fixtures`` (see its README). Tests that need
real MuMDIA outputs are marked ``real_data`` and read their paths from environment
variables; they are skipped when the variable is unset:

* ``MUMDIA_VIEWER_REAL_SINGLE``: a single-run directory (the Astral run);
* ``MUMDIA_VIEWER_REAL_EXPERIMENT``: an experiment directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_DIRS = {
    "single": FIXTURES / "smoke" / "out",
    "chrom_v1": FIXTURES / "smoke" / "out_chrom_v1",
    "chrom_v2_rg1": FIXTURES / "smoke" / "out_chrom_v2_rg1",
    "grouped": FIXTURES / "smoke" / "out_grouped",
    "grouped_pool": FIXTURES / "smoke" / "out_grouped_pool",
    "experiment": FIXTURES / "smoke" / "exp",
    "topk": FIXTURES / "topk" / "out_dbl_topk3",
    "mbr": FIXTURES / "mbr" / "exp_mbr_c_tr_qt004",
    "ovl_bp": FIXTURES / "overlap" / "out_ovl_bp",
    "ovl_bp_pool": FIXTURES / "overlap" / "out_ovl_bp_pool",
    "ovl_rg50": FIXTURES / "overlap" / "out_ovl_pg_rg50",
    "ovl_rg50_pool": FIXTURES / "overlap" / "out_ovl_pg_pool_rg50",
    "ovl128": FIXTURES / "overlap" / "out_ovl128",
    "ovl128_pool": FIXTURES / "overlap" / "out_ovl128_pool",
    "entrapment": FIXTURES / "entrapment" / "entrap_mode",
}


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep the tests' derived data out of the user's viewer cache."""
    os.environ["MUMDIA_VIEWER_CACHE_DIR"] = str(tmp_path_factory.mktemp("viewer_cache"))


@pytest.fixture(scope="session")
def fixture_dir():
    """Path of a named fixture directory: ``fixture_dir("single")``."""

    def get(name: str) -> Path:
        path = FIXTURE_DIRS[name]
        assert path.is_dir(), f"fixture {name} is missing at {path}"
        return path

    return get


@pytest.fixture(scope="session")
def open_fixture(fixture_dir):
    """Open a named fixture as a ResultSet (cached per session)."""
    from mumdia_viewer.data import open_results

    opened: dict[str, object] = {}

    def get(name: str):
        if name not in opened:
            opened[name] = open_results(fixture_dir(name))
        return opened[name]

    return get


def _real(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        pytest.skip(f"{var} is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{var}={value} is not a directory")
    return path


@pytest.fixture(scope="session")
def real_single() -> Path:
    return _real("MUMDIA_VIEWER_REAL_SINGLE")


@pytest.fixture(scope="session")
def real_experiment() -> Path:
    return _real("MUMDIA_VIEWER_REAL_EXPERIMENT")
