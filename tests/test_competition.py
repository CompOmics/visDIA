"""Base-peptide competition, the engine's winner rule, and the exact decoy partner."""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import Cache, open_results
from mumdia_viewer.data.competition import (
    competition,
    exact_partner,
    partner_map,
    q_values,
)


@pytest.mark.parametrize("name", ["single", "experiment", "grouped", "topk", "mbr"])
def test_winner_rule_reproduces_the_sparse_q_columns(open_fixture, name):
    rs = open_fixture(name)
    table = pq.read_table(
        rs.scored.path,
        columns=["source", "candidate_id", "base_peptide_id", "peptide_q_value", "precursor_q"],
    ).to_pylist()
    seen: set[int] = set()
    for row in table:
        if row["base_peptide_id"] in seen:
            continue
        seen.add(row["base_peptide_id"])
        df, flags = competition(rs, row["source"], row["candidate_id"], row["base_peptide_id"])
        # Every group has exactly one winner, and it is the row whose grouped q is below 1.
        assert df["wins_peptide"].sum() == 1
        assert flags["peptide_q_value"] == (row["peptide_q_value"] < 1.0)
        assert flags["precursor_q"] == (row["precursor_q"] < 1.0)


def test_q_values_carry_their_units(open_fixture):
    rs = open_fixture("experiment")
    row = pq.read_table(rs.scored.path).slice(0, 1).to_pylist()[0]
    qs = {q.column: q for q in q_values(rs, row)}
    assert set(qs) >= {"q_value", "run_psm_q", "precursor_q", "peptide_q_value", "pg_q_value"}
    assert qs["run_psm_q"].scope == "this run"
    assert qs["peptide_q_value"].scope == "experiment-wide" and qs["peptide_q_value"].grouped
    assert "picked target-decoy" in qs["peptide_q_value"].unit


def test_fasta_library_has_no_exact_partner(open_fixture):
    rs = open_fixture("single")
    pm = partner_map(rs)
    assert not pm.valid and "FASTA" in pm.reason
    assert exact_partner(rs, 0) is None


def test_exact_partner_on_a_paired_library(fixture_dir, tmp_path: Path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    lib = copy / "fragment_library_precursors.parquet"
    original = pq.read_table(lib)
    n = original.num_rows - original.num_rows % 2
    table = original.slice(0, n)
    pairs = [i // 2 for i in range(n)]
    labels = ["target" if i % 2 == 0 else "decoy" for i in range(n)]
    table = table.set_column(
        table.schema.get_field_index("peptidoform_id"),
        "peptidoform_id",
        pa.array(pairs, pa.uint32()),
    )
    table = table.set_column(table.schema.get_field_index("label"), "label", pa.array(labels))
    pq.write_table(table, lib)
    rs = open_results(copy, cache=Cache(tmp_path / "cache"))
    pm = partner_map(rs)
    assert pm.valid
    assert exact_partner(rs, 0) == 1 and exact_partner(rs, 1) == 0 and exact_partner(rs, 5) == 4
    assert exact_partner(rs, n + 10) is None


@pytest.mark.real_data
def test_real_library_pairs_targets_and_decoys(real_single):
    rs = open_results(real_single)
    pm = partner_map(rs)
    assert pm.valid
