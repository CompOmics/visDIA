"""Base-peptide competition, the engine's winner rule, and the exact decoy partner."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import Cache, open_results
from mumdia_viewer.data.competition import competition, exact_partner, q_values
from mumdia_viewer.data.entrapment import entrapment_expr, settings_for
from mumdia_viewer.data.rescore import rescore_info

TOOLS = Path(__file__).parent / "fixtures" / "tools" / "entrapment"


def _check_every_row(rs, *, entrapment=None, max_keys: int | None = None) -> int:
    """Compare the winner flags of every row of each checked base peptide with the engine's
    sparse q columns (all base peptides, or a fixed sample of ``max_keys`` of them)."""
    table = pq.read_table(
        rs.scored.path,
        columns=["source", "candidate_id", "base_peptide_id", "peptide_q_value", "precursor_q"],
    ).to_pandas()
    checked = 0
    groups = list(table.groupby("base_peptide_id"))
    if max_keys is not None and len(groups) > max_keys:
        step = len(groups) / max_keys
        groups = [groups[int(i * step)] for i in range(max_keys)]
    for bpid, rows in groups:
        first = rows.iloc[0]
        df, _ = competition(
            rs, int(first.source), int(first.candidate_id), int(bpid), entrapment=entrapment
        )
        merged = df.merge(
            rows, on=["source", "candidate_id"], suffixes=("", "_stored"), validate="one_to_one"
        )
        assert len(merged) == len(rows)
        # The winner of each key is the only row whose grouped q is below 1.
        assert (merged["wins_peptide"] == (merged["peptide_q_value_stored"] < 1.0)).all(), bpid
        assert (merged["wins_precursor"] == (merged["precursor_q_stored"] < 1.0)).all(), bpid
        checked += len(merged)
    return checked


@pytest.mark.parametrize("name", ["single", "experiment", "grouped", "topk", "mbr", "ovl_bp"])
def test_winner_flags_of_every_row_match_the_sparse_q_columns(open_fixture, name):
    rs = open_fixture(name)
    assert _check_every_row(rs) == rs.scored.rows


def _entrapment_run(fixture_dir, dst: Path) -> Path:
    dst.mkdir(parents=True)
    src = fixture_dir("entrapment")
    for f in ("psms_scored.parquet", "psms_scored.parquet.report.json"):
        shutil.copy2(src / f, dst / f)
    report = json.loads((dst / "psms_scored.parquet.report.json").read_text(encoding="utf-8"))
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", str(dst)],
        "config_json": json.dumps(config),
        "artifacts": {
            "psms_scored": {
                "path": str(dst / "psms_scored.parquet"),
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": report["content_hash"],
                "producing_stage": "rescore",
            }
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dst


def test_entrapment_mode_winner_rule(fixture_dir, tmp_path: Path):
    rs = open_results(_entrapment_run(fixture_dir, tmp_path / "run"))
    info = rescore_info(rs)
    assert info.mode == "entrapment"
    test = entrapment_expr(settings_for(rs))
    # Decoys do not compete and a spike-in wins a tie: the flags still equal the engine's.
    assert _check_every_row(rs, entrapment=test, max_keys=400) > 400
    row = pq.read_table(rs.scored.path).slice(0, 1).to_pylist()[0]
    units = {q.column: q.unit for q in q_values(rs, row, info=info)}
    assert all("entrapment estimate" in u for u in units.values())


def test_q_values_carry_their_units(open_fixture):
    rs = open_fixture("experiment")
    row = pq.read_table(rs.scored.path).slice(0, 1).to_pylist()[0]
    qs = {q.column: q for q in q_values(rs, row)}
    assert set(qs) >= {"q_value", "run_psm_q", "precursor_q", "peptide_q_value", "pg_q_value"}
    assert qs["run_psm_q"].scope == "this run"
    assert qs["peptide_q_value"].scope == "experiment-wide" and qs["peptide_q_value"].grouped
    assert "picked target-decoy" in qs["peptide_q_value"].unit
    loser = next(q for q in qs.values() if q.grouped and q.value == 1.0)
    assert loser.winner is False and loser.display_value is None


def test_base_peptide_grouping_labels_precursor_q(open_fixture):
    rs = open_fixture("ovl_bp")
    row = pq.read_table(rs.scored.path).slice(0, 1).to_pylist()[0]
    units = {q.column: q.unit for q in q_values(rs, row, info=rescore_info(rs))}
    assert "base-peptide unit" in units["precursor_q"]


def test_fasta_library_has_no_exact_partner(open_fixture):
    rs = open_fixture("single")
    row = pq.read_table(rs.scored.path).slice(0, 1).to_pylist()[0]
    lookup = exact_partner(
        rs, row["candidate_id"], peptidoform=row["peptidoform"], charge=row["charge"]
    )
    assert lookup.candidate_id is None and "FASTA" in lookup.reason


def _paired_copy(fixture_dir, tmp_path: Path) -> tuple[Path, int]:
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
    return copy, n


def test_exact_partner_on_a_paired_library(fixture_dir, tmp_path: Path):
    copy, n = _paired_copy(fixture_dir, tmp_path)
    rs = open_results(copy, cache=Cache(tmp_path / "cache"))
    assert exact_partner(rs, 0).candidate_id == 1
    assert exact_partner(rs, 1).candidate_id == 0
    assert exact_partner(rs, 5).candidate_id == 4
    assert exact_partner(rs, n + 10).candidate_id is None


def test_partner_refuses_a_library_that_does_not_match_the_scored_row(fixture_dir, tmp_path):
    copy, _ = _paired_copy(fixture_dir, tmp_path)
    rs = open_results(copy, cache=Cache(tmp_path / "cache"))
    lookup = exact_partner(rs, 0, peptidoform="NOTTHEPEPTIDE", charge=2)
    assert lookup.candidate_id is None and "not the library the run searched" in lookup.reason


@pytest.mark.real_data
def test_real_library_pairs_targets_and_decoys(real_single):
    rs = open_results(real_single)
    row = pq.read_table(rs.scored.path).slice(1000, 1).to_pylist()[0]
    lookup = exact_partner(
        rs, row["candidate_id"], peptidoform=row["peptidoform"], charge=row["charge"]
    )
    assert lookup.candidate_id is not None, lookup.reason
