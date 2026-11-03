"""DuckDB's memory limit, and the identification tables of a large table under a small one.

A desktop user's protein-group table failed with "could not allocate block ... (487.6
MiB/488.2 MiB used)" on a scored table of 8.5M rows: the limit was a fixed 512 MB. The
limit now follows the machine (a quarter of the physical memory, 1 to 16 GiB, or
``MUMDIA_VIEWER_MEMORY_LIMIT``), and the grouped tables count protein groups under
integer ids and pick each group's row with an aggregate over narrow columns.

The regression test writes a synthetic single run of several million rows whose protein
groups are 150 characters long (a group of shared-peptide proteins) and queries it under
a limit far below what the queries used before.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.cache import Cache
from mumdia_viewer.data.duck import (
    MAX_THREADS,
    MEMORY_CEILING,
    MEMORY_ENV_VAR,
    MEMORY_FLOOR,
    DuckDB,
    default_threads,
    format_size,
    parse_size,
    physical_memory,
    resolve_memory_limit,
)
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.tables import TableQuery, identification_table

GIB = 1 << 30

# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4GB", 4 * 10**9),
        ("4gb", 4 * 10**9),
        (" 1.5 GiB ", int(1.5 * GIB)),
        ("800MB", 800 * 10**6),
        ("512KiB", 512 * 1024),
        ("2TiB", 2 << 40),
        ("1000000B", 10**6),
        ("40%", 4 * GIB),
        ("100%", 10 * GIB),
    ],
)
def test_sizes_are_parsed_in_duckdb_units(text, expected):
    assert parse_size(text, physical=10 * GIB) == expected


@pytest.mark.parametrize("text", ["", "lots", "4", "4 XB", "0GB", "0%", "150%", "-1GB", "4GB!"])
def test_a_bad_size_is_refused(text):
    with pytest.raises(ValueError):
        parse_size(text, physical=10 * GIB)


def test_a_percentage_needs_the_physical_memory():
    with pytest.raises(ValueError, match="physical memory"):
        parse_size("40%", physical=None)


@pytest.mark.parametrize(
    ("physical", "expected"),
    [
        (16 * GIB, 4 * GIB),  # a quarter
        (2 * GIB, MEMORY_FLOOR),  # a quarter would be 512 MiB: the floor
        (755 * GIB, MEMORY_CEILING),  # a quarter would be 189 GiB: the ceiling
        (None, MEMORY_FLOOR),  # unknown
    ],
)
def test_the_default_is_a_clamped_quarter_of_the_physical_memory(physical, expected):
    limit = resolve_memory_limit(environ={}, physical=physical)
    assert limit.bytes == expected
    if physical is None:
        assert "unknown" in limit.source
    else:
        assert format_size(physical) in limit.source
        assert ("clamped" in limit.source) == (expected != physical // 4)


def test_the_environment_variable_overrides_the_default_unclamped():
    limit = resolve_memory_limit(environ={MEMORY_ENV_VAR: "600MB"}, physical=16 * GIB)
    assert limit.bytes == 600 * 10**6
    assert limit.source == f"{MEMORY_ENV_VAR}=600MB"
    share = resolve_memory_limit(environ={MEMORY_ENV_VAR: "50%"}, physical=16 * GIB)
    assert share.bytes == 8 * GIB
    # An empty variable is unset.
    assert resolve_memory_limit(environ={MEMORY_ENV_VAR: " "}, physical=16 * GIB).bytes == 4 * GIB


def test_a_given_value_wins_over_the_environment():
    env = {MEMORY_ENV_VAR: "600MB"}
    assert resolve_memory_limit("2GB", environ=env, physical=16 * GIB).bytes == 2 * 10**9
    assert resolve_memory_limit(123_456_789, environ=env).bytes == 123_456_789


def test_a_bad_environment_variable_names_itself():
    with pytest.raises(ViewerError, match=MEMORY_ENV_VAR):
        resolve_memory_limit(environ={MEMORY_ENV_VAR: "plenty"}, physical=16 * GIB)
    with pytest.raises(ViewerError, match="invalid memory limit"):
        resolve_memory_limit("plenty", environ={}, physical=16 * GIB)
    with pytest.raises(ViewerError, match="positive"):
        resolve_memory_limit(0)


def test_this_machine_reports_its_memory_and_bounded_threads():
    total = physical_memory()
    assert total is not None and total > 256 << 20
    assert 1 <= default_threads() <= MAX_THREADS


# --------------------------------------------------------------------------- database


def test_the_database_uses_the_limit_on_every_cursor(tmp_path):
    duck = DuckDB(memory_limit="64MB", threads=2, temp_directory=tmp_path / "spill")
    try:
        assert duck.limit.bytes == 64 * 10**6
        assert duck.threads == 2
        sql = (
            "SELECT current_setting('memory_limit'), current_setting('threads'), "
            "current_setting('parquet_metadata_cache'), "
            "current_setting('disable_parquet_prefetching')"
        )
        got: list = []
        other = threading.Thread(target=lambda: got.append(duck.rows(sql)[0]))
        other.start()
        other.join()
        # Every thread has its own cursor; the settings are the database's, not one cursor's.
        for row in (duck.rows(sql)[0], got[0]):
            assert row[0] == "61.0 MiB" and int(row[1]) == 2
            assert row[2] is True or row[2] == "true"
            assert row[3] is True or row[3] == "true"
    finally:
        duck.close()


def test_open_results_reads_the_environment_variable(fixture_dir, monkeypatch):
    monkeypatch.setenv(MEMORY_ENV_VAR, "700MB")
    rs = open_results(fixture_dir("single"))
    try:
        assert rs.duck.limit.bytes == 700 * 10**6
        assert rs.duck.scalar("SELECT current_setting('memory_limit')") == "667.5 MiB"
    finally:
        rs.duck.close()


def test_an_out_of_memory_error_says_how_to_raise_the_limit():
    duck = DuckDB(memory_limit="16MB", threads=1, temp_directory=None)
    try:
        with pytest.raises(duckdb.OutOfMemoryException) as info:
            duck.rows(
                "SELECT count(DISTINCT i::VARCHAR || repeat('x', 40)) FROM range(3000000) t(i)"
            )
    finally:
        duck.close()
    text = str(info.value)
    assert "Out of Memory" in text
    assert MEMORY_ENV_VAR in text and "--memory-limit" in text
    assert "15.3 MiB (given as 16MB)" in text and "spilling to disk is disabled" in text


def test_the_command_line_refuses_a_bad_memory_limit(fixture_dir, monkeypatch, capsys):
    from mumdia_viewer.cli import main

    monkeypatch.delenv(MEMORY_ENV_VAR, raising=False)
    assert main([str(fixture_dir("single")), "--memory-limit", "plenty"]) == 2
    assert "--memory-limit" in capsys.readouterr().err
    assert MEMORY_ENV_VAR not in os.environ
    monkeypatch.setenv(MEMORY_ENV_VAR, "plenty")
    assert main([str(fixture_dir("single"))]) == 2
    assert MEMORY_ENV_VAR in capsys.readouterr().err


# --------------------------------------------------------------------------- large table

N_ROWS = 3_000_000
N_GROUPS = 4_000
GROUP_LEN = 150


def _grouped_q(key, decoy, score, accept, *, by_label: bool) -> np.ndarray:
    """1.0 everywhere but on the best row of each group, the engine's layout.

    A group is a key, or a (key, label) with ``by_label``. Its best row gets 0.001 when
    it is a target with ``accept``, else 0.5.
    """
    k = key.astype(np.int64) * 2 + decoy if by_label else key.astype(np.int64)
    order = np.lexsort((-score, k))
    first = np.ones(k.size, dtype=bool)
    first[1:] = k[order][1:] != k[order][:-1]
    q = np.ones(k.size)
    win = order[first]
    q[win] = np.where(decoy[win] | ~accept[win], 0.5, 0.001)
    return q


def _strings(indices: np.ndarray, values: list[str]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(indices, pa.int32()), pa.array(values))


def _large_run(directory: Path) -> dict[str, Any]:
    """A single run of N_ROWS scored rows; returns the expected tables in brief."""
    rng = np.random.default_rng(7)
    n = N_ROWS
    n_pep = n // 3
    pep_group = rng.integers(0, N_GROUPS, size=n_pep)
    base = rng.integers(0, n_pep, size=n).astype(np.uint32)
    group = pep_group[base]
    decoy = rng.random(n) < 0.5
    score = np.round(rng.normal(size=n), 4) + np.where(decoy, 0.0, 1.0)
    accepted_groups = rng.random(N_GROUPS) < 0.8
    accepted_peptides = rng.random(n_pep) < 0.3
    pad = "PROTEIN_HUMAN;" * (GROUP_LEN // 14 + 1)
    names = [f"sp|P{g:05d}|{pad}"[:GROUP_LEN] for g in range(N_GROUPS)]
    names += ["DECOY_" + s for s in names]
    group_index = group.astype(np.int64) + np.where(decoy, N_GROUPS, 0)
    groups = _strings(group_index, names)
    peptides = _strings(base, [f"PEPTIDE{i}K" for i in range(n_pep)])
    pg_q = _grouped_q(group, decoy, score, accepted_groups[group], by_label=True)
    pep_q = _grouped_q(base, decoy, score, accepted_peptides[base], by_label=False)
    q = np.where(decoy, 0.5, 0.001)
    table = pa.table(
        {
            "candidate_id": pa.array(np.arange(n, dtype=np.uint32)),
            "peptidoform": peptides,
            "charge": pa.array(2 + (base % 2).astype(np.int32)),
            "label": _strings(decoy.astype(np.int64), ["target", "decoy"]),
            "protein": groups,
            "base_peptide_id": pa.array(base),
            "apex_rt": pa.array(np.full(n, 600.0)),
            "elution_lo": pa.array(np.full(n, 590.0)),
            "elution_hi": pa.array(np.full(n, 610.0)),
            "score": pa.array(score),
            "q_value": pa.array(q),
            "peptide_q_value": pa.array(pep_q),
            "protein_group": groups,
            "pg_q_value": pa.array(pg_q),
            "global_q_value": pa.array(q),
            "prelim_score": pa.array(score),
            "source": pa.array(np.zeros(n, dtype=np.uint32)),
            "run_psm_q": pa.array(q),
            "experiment_psm_q": pa.array(q),
            "precursor_q": pa.array(q),
            "selected_peak_rank": pa.array(np.zeros(n, dtype=np.int32)),
        }
    )
    directory.mkdir(parents=True)
    path = directory / "psms_scored.parquet"
    # About a million rows per row group, as the engine writes; without the Arrow schema
    # the dictionary columns read back as plain strings.
    pq.write_table(table, path, row_group_size=1 << 20, store_schema=False)
    report = {
        "logical_name": "psms_scored",
        "schema_name": "psms_scored",
        "schema_version": 4,
        "stage": "rescore",
        "rows": n,
        "content_hash": "0" * 64,
        "params": {"classifier": "native_tda"},
        "stats": {},
    }
    (directory / "psms_scored.parquet.report.json").write_text(json.dumps(report))
    record = {k: report[k] for k in ("schema_name", "schema_version", "rows", "content_hash")}
    manifest = {
        "mumdia_version": "0.9.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", "/recorded/large"],
        "config_json": "{}",
        "artifacts": {
            "psms_scored": {
                **record,
                "path": "/recorded/large/psms_scored.parquet",
                "producing_stage": "rescore",
            }
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    targets = ~decoy
    accepted = np.unique(group[targets & (pg_q <= 0.01)])
    return {
        "protein_group": int(accepted.size),
        "peptide": int(np.unique(base[targets & (pep_q <= 0.01)]).size),
        "protein_group_all": int(np.unique(group[targets]).size),
        "peptide_all": int(np.unique(base[targets]).size),
        "accepted_groups": {names[i] for i in accepted},
    }


@pytest.fixture(scope="module")
def large_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("large") / "run"
    return directory, _large_run(directory)


def _open_large(directory: Path, tmp_path: Path, limit: str, *, spill: bool):
    cache = Cache(tmp_path / "cache")
    duck = DuckDB(memory_limit=limit, threads=4, temp_directory=cache.temp_dir() if spill else None)
    return open_results(directory, cache=cache, duck=duck)


@pytest.mark.parametrize("unit", ["protein_group", "peptide"])
def test_grouped_tables_of_a_large_table_fit_a_small_limit(large_run, tmp_path, unit):
    """The grouped tables of 3M rows under a 256 MB limit, spilling to the private directory.

    The reported query (the unit's own q at 0.01, sorted by score) and the table of every
    group, which ranks the rows of each group.
    """
    directory, expected = large_run
    rs = _open_large(directory, tmp_path, "256MB", spill=True)
    try:
        query = TableQuery(unit=unit, threshold=0.01, sort_by="score", limit=100)
        page = identification_table(rs, query)
        assert page.total == expected[unit]
        assert len(page.rows) == 100 and page.rows["is_winner"].all()
        assert page.rows["score"].is_monotonic_decreasing
        everything = identification_table(rs, TableQuery(unit=unit, threshold=None, limit=50))
        assert everything.total == expected[f"{unit}_all"]
        if unit == "protein_group":
            whole = identification_table(rs, TableQuery(unit=unit, limit=10_000))
            assert set(whole.rows["protein_group"]) == expected["accepted_groups"]
            assert whole.rows["n_peptides"].min() >= 1
    finally:
        rs.duck.close()


_DUCKDB = tuple(int(x) for x in duckdb.__version__.split(".")[:2])


@pytest.mark.skipif(_DUCKDB < (1, 3), reason="DuckDB 1.2 needs more than 512 MB here")
def test_protein_groups_are_ranked_without_holding_every_row(large_run, tmp_path):
    """Every protein group of 3M rows in 256 MB without spilling.

    The rows of each group were ranked by a window over every passing row with all its
    columns, which did not fit 512 MB here (DuckDB 1.5.6); the aggregate over the
    narrow columns does. Without spill files a limit is a hard bound on the working set.
    """
    directory, expected = large_run
    rs = _open_large(directory, tmp_path, "256MB", spill=False)
    try:
        page = identification_table(rs, TableQuery(unit="protein_group", threshold=None))
        assert page.total == expected["protein_group_all"]
    finally:
        rs.duck.close()
