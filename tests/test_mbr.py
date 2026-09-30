"""Match-between-runs: detection, transfers, native counts and the n_runs decomposition.

The MBR fixture (``tests/fixtures/mbr/exp_mbr_c_tr_qt004``) is a two-run experiment
(runs a and c) with 48 accepted transfers (16 into a, 32 into c) and
``quant.q_threshold`` 0.004.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.counts import per_run_counts, unit_counts
from mumdia_viewer.data.mbr import (
    mbr_info,
    mbr_ran,
    n_runs_decomposition,
    transfer_counts,
    transfer_of,
    transfers_for_run,
)

TRANSFER_Q = 7 / 48  # (6 + 1) / 48, the transfer q of every fixture transfer


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    df["n_runs"] = df["n_runs"].astype(int)
    if "charge" in df.columns:
        df["charge"] = df["charge"].astype(int)
    return df


def _per_source(table: pa.Table, mask) -> dict[int, int]:
    part = table.filter(mask)
    counts = pc.value_counts(part["source"]).to_pylist()
    return {int(d["values"]): int(d["counts"]) for d in counts}


# --------------------------------------------------------------------------- detection


def test_mbr_info_of_the_fixture(open_fixture):
    rs = open_fixture("mbr")
    info = mbr_info(rs)
    assert mbr_ran(rs)
    assert info.ran and info.strategy == "RtTransfer"
    assert info.n_transfers == 48
    assert info.scored_for_quant is not None
    assert info.scored_for_quant.path.name == "scored_mbr.parquet"
    assert info.transfers is not None and info.transfers.path.name == "mbr_transferred.parquet"
    text = " ".join(info.notes)
    assert "scored_combined.parquet" in text
    assert "no manifest record" in text
    assert "48 transfers were accepted at transfer_q <= 0.15" in text
    assert "No transfer is possible" not in text  # mbr.min_anchor_runs is 1 here


def test_the_two_run_trap_is_noted(fixture_dir, tmp_path):
    root = _copy(fixture_dir("mbr"), tmp_path / "exp")
    path = root / "experiment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads(manifest["config_json"])
    config["mbr"]["min_anchor_runs"] = 2  # the default
    manifest["config_json"] = json.dumps(config)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    notes = mbr_info(open_results(root)).notes
    assert any("No transfer is possible" in n and "min_anchor_runs = 2" in n for n in notes)


def test_without_mbr(open_fixture):
    rs = open_fixture("experiment")
    info = mbr_info(rs)
    assert not info.ran and info.strategy == "None" and info.n_transfers == 0
    assert info.scored_for_quant is None and info.transfers is None and info.notes == ()
    df = transfer_counts(rs, 0.01)
    assert list(df["native_psms"]) == [273, 273] and list(df["transfers"]) == [0, 0]
    assert list(df["native_or_transferred"]) == [273, 273]
    assert transfers_for_run(rs, "a").empty
    assert transfer_of(rs, "a", 1) is None
    single = mbr_info(open_fixture("single"))
    assert not single.ran and single.strategy is None


def test_stale_mbr_files_are_reported_and_not_used(fixture_dir, tmp_path):
    root = _copy(fixture_dir("experiment"), tmp_path / "exp")
    for name in ("scored_mbr.parquet", "mbr_transferred.parquet"):
        shutil.copy2(fixture_dir("mbr") / name, root / name)
    rs = open_results(root)
    info = mbr_info(rs)
    assert not info.ran and info.transfers is None and info.n_transfers == 0
    stale = [n for n in info.notes if "left over from an earlier run" in n]
    assert len(stale) == 2
    assert list(transfer_counts(rs, 0.01)["transfers"]) == [0, 0]


# --------------------------------------------------------------------------- counts


def test_native_counts_come_from_scored_combined(open_fixture):
    rs = open_fixture("mbr")
    t = 0.004
    df = transfer_counts(rs, t)
    assert df[["run", "native_psms", "transfers", "added_by_mbr", "native_or_transferred"]].to_dict(
        "list"
    ) == {
        "run": ["a", "c"],
        "native_psms": [0, 281],
        "transfers": [16, 32],
        "added_by_mbr": [16, 0],
        "native_or_transferred": [16, 281],
    }
    assert "match no row" not in df.attrs["note"]
    combined = pq.read_table(rs.scored.path)
    augmented = pq.read_table(rs.scored_for_quant.path)
    target = pc.equal(combined["label"], "target")
    native = _per_source(combined, pc.and_(target, pc.less_equal(combined["run_psm_q"], t)))
    assert native.get(0, 0) == 0 and native.get(1, 0) == 281
    # The engine's report rule on scored_mbr: run_psm_q <= t or transferred.
    rule = pc.and_(
        pc.equal(augmented["label"], "target"),
        pc.or_(
            pc.less_equal(augmented["run_psm_q"], t),
            pc.fill_null(augmented["is_transferred"], False),
        ),
    )
    engine_rule = _per_source(augmented, rule)
    assert engine_rule == {0: 16, 1: 281}
    assert list(df["native_or_transferred"]) == [engine_rule[0], engine_rule[1]]
    assert engine_rule[0] != native.get(0, 0)  # counting on scored_mbr would add transfers


def test_unit_counts_are_native_not_the_report_rule(open_fixture):
    rs = open_fixture("mbr")
    t = 0.004
    counts = {c.unit: c for c in unit_counts(rs, t)}
    combined = pq.read_table(rs.scored.path)
    augmented = pq.read_table(rs.scored_for_quant.path)
    mask = pc.and_(
        pc.equal(combined["label"], "target"), pc.less_equal(combined["peptide_q_value"], t)
    )
    native = pc.count_distinct(combined.filter(mask)["base_peptide_id"]).as_py()
    report_mask = pc.and_(
        pc.equal(augmented["label"], "target"),
        pc.or_(
            pc.less_equal(augmented["peptide_q_value"], t),
            pc.fill_null(augmented["is_transferred"], False),
        ),
    )
    report_rule = pc.count_distinct(augmented.filter(report_mask)["base_peptide_id"]).as_py()
    assert counts["peptide"].n_target == native
    assert report_rule > native


# --------------------------------------------------------------------------- transfers


def test_transfers_for_run(open_fixture):
    rs = open_fixture("mbr")
    a = transfers_for_run(rs, "a")
    c = transfers_for_run(rs, 1)
    assert len(a) == 16 and len(c) == 32
    assert a["candidate_id"].is_monotonic_increasing
    assert np.allclose(a["transfer_q"], TRANSFER_Q) and np.allclose(c["transfer_q"], TRANSFER_Q)
    assert (a["label"] == "target").all() and a["peptidoform"].notna().all()
    combined = pq.read_table(rs.scored.path).to_pandas()
    native = combined[combined["source"] == 0].set_index("candidate_id")
    for _, row in a.iterrows():
        ref = native.loc[int(row["candidate_id"])]
        assert row["native_q_value"] == ref["q_value"]
        assert row["native_run_psm_q"] == ref["run_psm_q"]
        assert row["peptidoform"] == ref["peptidoform"]
        assert row["observed_rt"] == ref["apex_rt"]
    assert a["q_value_after_mbr"].notna().all() and a["rt_delta"].notna().all()
    first = int(a["candidate_id"].iloc[0])
    record = transfer_of(rs, "a", first)
    assert record is not None and record["run"] == "a" and record["source"] == 0
    assert record["transfer_q"] == pytest.approx(TRANSFER_Q)
    untransferred = sorted(set(native.index) - set(a["candidate_id"]))[0]
    assert transfer_of(rs, "a", untransferred) is None
    assert transfer_of(rs, np.int64(0), first) == record  # a numpy index is accepted


def test_null_and_nan_transfer_q_are_alike(fixture_dir, tmp_path):
    root = _copy(fixture_dir("mbr"), tmp_path / "exp")
    path = root / "mbr_transferred.parquet"
    table = pq.read_table(path)
    values = table["transfer_q"].to_pylist()
    values[0], values[1] = float("nan"), None
    index = table.schema.get_field_index("transfer_q")
    table = table.set_column(index, table.schema.field(index), pa.array(values, pa.float64()))
    pq.write_table(table, path)
    rs = open_results(root)
    source = int(table["source"][0].as_py())
    cids = [int(table["candidate_id"][i].as_py()) for i in (0, 1, 2)]
    df = transfers_for_run(rs, source).set_index("candidate_id")
    assert np.isnan(df.loc[cids[0], "transfer_q"]) and np.isnan(df.loc[cids[1], "transfer_q"])
    assert df.loc[cids[2], "transfer_q"] == pytest.approx(TRANSFER_Q)
    assert transfer_of(rs, source, cids[0])["transfer_q"] is None
    assert transfer_of(rs, source, cids[1])["transfer_q"] is None


def test_empty_transfer_table_with_null_typed_strings(fixture_dir, tmp_path):
    root = _copy(fixture_dir("mbr"), tmp_path / "exp")
    schema = pa.schema(
        [
            ("candidate_id", pa.uint32()),
            ("source", pa.uint32()),
            ("peptidoform", pa.null()),
            ("charge", pa.int32()),
            ("protein_group", pa.null()),
            ("label", pa.null()),
            ("expected_rt", pa.float64()),
            ("observed_rt", pa.float64()),
            ("rt_delta", pa.float64()),
            ("transfer_q", pa.float64()),
        ]
    )
    pq.write_table(schema.empty_table(), root / "mbr_transferred.parquet")
    rs = open_results(root)
    info = mbr_info(rs)
    assert info.ran and info.n_transfers == 0
    # scored_mbr.parquet of the copy still flags 48 rows: the mismatch is reported.
    assert any("48 rows" in n for n in info.notes)
    assert list(transfer_counts(rs, 0.004)["transfers"]) == [0, 0]
    assert transfers_for_run(rs, "a").empty
    assert int(n_runs_decomposition(rs)["n_runs_transfer_only"].sum()) == 0


# --------------------------------------------------------------------------- n_runs


def test_n_runs_decomposition_reproduces_the_tsv(open_fixture, fixture_dir):
    rs = open_fixture("mbr")
    dec = n_runs_decomposition(rs)
    assert dec.attrs["threshold"] == 0.004  # experiment.report.q_threshold
    peptides = _tsv(fixture_dir("mbr") / "peptides.tsv")
    merged = peptides.merge(
        dec, left_on=["precursor", "charge"], right_on=["peptidoform", "charge"], how="left"
    )
    assert len(merged) == len(peptides) == 48 and merged["n_runs_engine"].notna().all()
    assert (merged["n_runs"] == merged["n_runs_engine"]).all()
    # 16 precursors were transferred into run a, where nothing passes 0.004 natively.
    assert int(merged["n_runs_transfer_only"].sum()) == 16
    assert (dec["n_runs_engine"] == dec["n_runs_native"] + dec["n_runs_transfer_only"]).all()
    proteins = _tsv(fixture_dir("mbr") / "proteins.tsv")
    groups = n_runs_decomposition(rs, level="protein_group")
    merged = proteins.merge(groups, on="protein_group", how="left")
    assert len(merged) == 13 and (merged["n_runs"] == merged["n_runs_engine"]).all()
    with pytest.raises(ValueError, match="level"):
        n_runs_decomposition(rs, level="peptide")


def test_n_runs_without_mbr_is_native(open_fixture, fixture_dir):
    rs = open_fixture("experiment")
    dec = n_runs_decomposition(rs, 0.01)
    assert (dec["n_runs_transfer_only"] == 0).all()
    peptides = _tsv(fixture_dir("experiment") / "peptides.tsv")
    merged = peptides.merge(
        dec, left_on=["precursor", "charge"], right_on=["peptidoform", "charge"], how="left"
    )
    assert (merged["n_runs"] == merged["n_runs_native"]).all()
    everything = n_runs_decomposition(rs, 0.01, include_zero=True)
    assert len(everything) >= len(dec) and (everything["n_runs_engine"] >= 0).all()


def test_n_runs_label_names_the_report_threshold(open_fixture, fixture_dir):
    rs = open_fixture("mbr")
    at_report = n_runs_decomposition(rs)
    label = at_report.attrs["labels"]["n_runs_engine"]
    assert "(the TSV's n_runs rule)" in label
    assert "0.004 is the report threshold (experiment.report.q_threshold)" in label
    assert at_report.attrs["report_threshold"] == 0.004
    other = n_runs_decomposition(rs, 0.05)
    label = other.attrs["labels"]["n_runs_engine"]
    assert "the n_runs of the TSV report" not in label
    assert (
        "it equals peptides.tsv n_runs only at the report threshold "
        "(0.004, experiment.report.q_threshold)"
    ) in label
    # The TSV was written at 0.004, so at 0.05 the rule differs from it on 32 of 48 rows.
    peptides = _tsv(fixture_dir("mbr") / "peptides.tsv")
    merged = peptides.merge(
        other, left_on=["precursor", "charge"], right_on=["peptidoform", "charge"], how="left"
    )
    assert int((merged["n_runs"] != merged["n_runs_engine"]).sum()) == 32
    groups = n_runs_decomposition(rs, 0.05, level="protein_group")
    assert "equals proteins.tsv n_runs only" in groups.attrs["labels"]["n_runs_engine"]


def test_mbr_info_is_frozen(open_fixture):
    """The memoised MbrInfo cannot be changed by a caller."""
    rs = open_fixture("mbr")
    info = mbr_info(rs)
    assert isinstance(info.notes, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.notes = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        info.notes.append("CALLER EDIT")  # type: ignore[attr-defined]
    assert mbr_info(rs) is info and "CALLER EDIT" not in mbr_info(rs).notes


# --------------------------------------------------------------------------- entrapment

ENTRAP_TOOLS = Path(__file__).parent / "fixtures" / "tools" / "entrapment"
CONTAMINANTS = ("KRT", "K1C", "K2C", "ALBU", "TRYP")


def _entrapment_experiment(src: Path, dst: Path, transfers: pa.Table) -> Path:
    """A one-run MBR experiment made from the entrapment rescore, in a temporary copy.

    The rescore's psms_scored.parquet becomes scored_combined.parquet (every row has
    source 0) with its report, and ``transfers`` becomes mbr_transferred.parquet.
    """
    dst.mkdir(parents=True)
    shutil.copy2(src / "psms_scored.parquet", dst / "scored_combined.parquet")
    shutil.copy2(
        src / "psms_scored.parquet.report.json", dst / "scored_combined.parquet.report.json"
    )
    pq.write_table(transfers, dst / "mbr_transferred.parquet")
    report = json.loads((dst / "scored_combined.parquet.report.json").read_text(encoding="utf-8"))
    config = json.loads((ENTRAP_TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    config["mbr"] = {"strategy": "rt_transfer", "min_anchor_runs": 1, "q_transfer": 0.01}
    scored = str(dst / "scored_combined.parquet")
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "run-experiment", "--out-dir", str(dst)],
        "config_json": json.dumps(config),
        "model_identities": {"rescorer": report["model_identity"], "mbr": "RtTransfer"},
        "experiment": {
            "runs": ["r0"],
            "n_runs": 1,
            "scored_combined": scored,
            "rescorer": report["params"]["classifier"],
            "mbr": "RtTransfer",
        },
        "artifacts": {
            "scored_combined": {
                "logical_name": "psms_scored",
                "path": scored,
                "format": "parquet",
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": report["content_hash"],
                "producing_stage": "rescore",
            }
        },
    }
    (dst / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dst


def test_entrapment_native_counts_match_the_per_run_counts(fixture_dir, tmp_path):
    """In entrapment mode native_psms counts real targets, as per_run_counts does.

    Three real targets and two spike-ins that fail run_psm_q <= 0.01 are made
    transfers. The engine's report rule (native_or_transferred) keeps every
    label = 'target' row, spike-ins included.
    """
    src = fixture_dir("entrapment")
    table = pq.read_table(
        src / "psms_scored.parquet",
        columns=["candidate_id", "source", "label", "protein", "run_psm_q"],
    )
    target = pc.equal(table["label"], "target")
    spike = pc.and_(target, pc.match_substring(table["protein"], "ENTRAP_"))
    spike = pc.and_(spike, pc.invert(pc.match_substring(table["protein"], "REAL_")))
    for token in CONTAMINANTS:
        spike = pc.and_(spike, pc.invert(pc.match_substring(table["protein"], token)))
    failing = pc.greater(table["run_psm_q"], 0.01)
    real_rows = table.filter(pc.and_(pc.and_(target, pc.invert(spike)), failing)).slice(0, 3)
    spike_rows = table.filter(pc.and_(spike, failing)).slice(0, 2)
    moved = pa.concat_tables([real_rows, spike_rows])
    transfers = pa.table(
        {
            "candidate_id": moved["candidate_id"],
            "source": moved["source"],
            "transfer_q": pa.array([0.005] * moved.num_rows, pa.float64()),
        }
    )
    rs = open_results(_entrapment_experiment(src, tmp_path / "exp", transfers))
    per_run = per_run_counts(rs, 0.01)
    counts = transfer_counts(rs, 0.01)
    assert list(counts.columns) == [
        "run",
        "source",
        "native_psms",
        "spike_in_psms",
        "transfers",
        "added_by_mbr",
        "native_or_transferred",
    ]
    assert list(counts["native_psms"]) == list(per_run["target_psms"]) == [8582]
    assert list(counts["spike_in_psms"]) == list(per_run["spike_in_psms"]) == [151]
    assert list(counts["transfers"]) == [5] and list(counts["added_by_mbr"]) == [5]
    assert list(counts["native_or_transferred"]) == [8582 + 151 + 5]
    labels = counts.attrs["labels"]
    assert labels["native_psms"].startswith("real target PSMs (rows, run_psm_q <= 0.01)")
    assert "spike-ins are in spike_in_psms" in labels["native_psms"]
    assert "not in native_psms" in labels["spike_in_psms"]
    assert "spike-ins included" in labels["native_or_transferred"]
    assert "Every transfer matches" in counts.attrs["note"]
    assert "spike-ins are included" in n_runs_decomposition(rs, 0.01).attrs["note"]


def test_decoy_mode_native_counts_have_no_spike_in_column(open_fixture):
    counts = transfer_counts(open_fixture("mbr"), 0.004)
    assert "spike_in_psms" not in counts.columns
    assert "spike_in_psms" not in counts.attrs["labels"]
    assert counts.attrs["labels"]["native_psms"].startswith("target PSMs (rows, run_psm_q")


# --------------------------------------------------------------------------- real data


@pytest.mark.real_data
def test_real_experiment_native_counts_and_n_runs(real_experiment):
    rs = open_results(real_experiment)
    info = mbr_info(rs)
    start = time.perf_counter()
    counts = transfer_counts(rs, 0.01)
    counts_time = time.perf_counter() - start
    start = time.perf_counter()
    dec = n_runs_decomposition(rs, 0.01)
    dec_time = time.perf_counter() - start
    print(
        f"\n{real_experiment}: MBR ran {info.ran}; native PSMs per run "
        f"{list(counts['native_psms'])}; transfer_counts {counts_time:.3f} s, "
        f"n_runs_decomposition {dec_time:.3f} s ({len(dec):,} precursors)"
    )
    peptides = _tsv(Path(real_experiment) / "peptides.tsv")
    merged = peptides.merge(
        dec, left_on=["precursor", "charge"], right_on=["peptidoform", "charge"], how="left"
    )
    assert (merged["n_runs"] == merged["n_runs_engine"]).all()
