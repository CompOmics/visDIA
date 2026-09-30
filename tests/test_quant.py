"""Quant semantics: the gate, quant states, protein quant, the status breakdown and LFQ.

Synthetic copies (in ``tmp_path``) cover what no fixture shows: unsupported schema
versions, a winner that is not quantifiable, a winner whose q is at the cap of 1.0 and
the gate columns of other ``quant.q_filter`` values.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.errors import ArtifactNotFound, SchemaVersionError, ViewerError
from mumdia_viewer.data.quant import (
    N_FEATURES_LABEL,
    PEPTIDE_STATUS,
    PROTEIN_STATUS,
    describe_status,
    engine_q_filter,
    lfq_matrix,
    protein_quant,
    quant_gate,
    quant_state,
    quant_states,
    quant_status_breakdown,
)
from mumdia_viewer.data.tables import TableQuery, identification_table


def _read(path, columns=None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def _copy(tmp_path: Path, fixture_dir, name: str) -> Path:
    out = tmp_path / name
    shutil.copytree(fixture_dir(name), out)
    return out


def _rewrite(path: Path, change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Rewrite a parquet file of a tmp copy through pandas, keeping its schema."""
    table = pq.read_table(path)
    df = change(table.to_pandas())
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), path)


def _edit_json(path: Path, change: Callable[[dict], None]) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_status_vocabulary():
    assert set(PEPTIDE_STATUS) == {
        "quantified",
        "no_fragment_traces",
        "no_positive_fragment_area",
        "no_fragments_selected",
        "nonfinite_quantity",
    }
    assert set(PROTEIN_STATUS) == {"quantified", "no_quantifiable_peptide"}
    assert describe_status("no_quantifiable_peptide", "protein_group_quant").startswith(
        "not quantifiable"
    )
    assert describe_status("made_up_status") == "made_up_status"
    assert engine_q_filter("run_psm_q") == "RunPsmQ"
    assert engine_q_filter("PsmQ") == "PsmQ"
    assert engine_q_filter("new_filter") == "new_filter"


# --------------------------------------------------------------------------- single run


def test_single_run_gate(open_fixture):
    gate = quant_gate(open_fixture("single"))
    assert gate.q_filter == "PeptideQ"
    assert gate.q_column == "peptide_q_value"
    assert gate.threshold == 0.01
    assert (gate.configured, gate.effective) == ("PeptideQ", "PeptideQ")
    assert not gate.transfers_admitted
    assert "peptide_quant.parquet.report.json" in gate.source
    assert "one precursor per accepted base peptide" in gate.note
    assert gate.label == "target rows with peptide_q_value <= 0.01"
    assert gate.describe(False) == (
        "the target rows with peptide_q_value <= 0.01 (PeptideQ: the winning precursor of "
        "each accepted base peptide)"
    )


def test_single_run_states(open_fixture):
    """150 quantified winners; the 130 target siblings of accepted peptides are not selected."""
    rs = open_fixture("single")
    scored = _read(rs.scored.path)
    states = quant_states(rs, None, scored["candidate_id"].to_numpy())
    assert list(states["candidate_id"]) == list(scored["candidate_id"])  # order kept
    merged = scored.merge(states, on="candidate_id")
    quantified = merged[merged["state"] == "quantified"]
    expected = scored[(scored["label"] == "target") & (scored["peptide_q_value"] <= 0.01)]
    assert set(quantified["candidate_id"]) == set(expected["candidate_id"])
    assert len(quantified) == 150
    accepted = set(expected["base_peptide_id"])
    siblings = merged[
        (merged["label"] == "target")
        & merged["base_peptide_id"].isin(accepted)
        & (merged["state"] == "not_selected")
    ]
    assert len(siblings) == 130
    assert (
        siblings["reason"].str.startswith("not selected: another precursor of this base peptide (")
    ).all()
    assert (
        siblings["reason"]
        .str.contains("is its winning row and was selected for quant (quantified)", regex=False)
        .all()
    )
    decoys = merged[merged["label"] == "decoy"]
    assert (decoys["state"] == "not_selected").all()
    assert decoys["reason"].str.contains("never selects decoy").all()
    rest = merged[
        (merged["label"] == "target")
        & ~merged["base_peptide_id"].isin(accepted)
        & (merged["state"] == "not_selected")
    ].set_index("candidate_id")
    assert len(rest) == 2
    assert rest["reason"].str.contains("fails the quant gate peptide_q_value <= 0.01").all()
    # Base peptide 792: its winner (228) fails the gate; the sibling (240) names it.
    q228 = f"{rest.loc[228, 'peptide_q_value']:.6g}"
    assert rest.loc[228, "reason"] == (
        f"not selected: peptide_q_value = {q228} fails the quant gate peptide_q_value <= 0.01"
    )
    assert rest.loc[240, "reason"].endswith(
        "the winning row of this base peptide is candidate 228, and every other row holds 1.0 "
        "(the winning row is not selected either)"
    )
    # gate_q is the gate column that quant read; nothing else is reported for this run.
    assert np.allclose(merged["gate_q"], merged["peptide_q_value"])
    assert merged["native_q"].isna().all()
    assert merged["window_also_covers"].isna().all()
    assert "peak_rank 0 only" in states.attrs["peaks"]
    # A missing quantity is NaN, never 0; the quantified rows equal peptide_quant.
    assert merged.loc[merged["state"] != "quantified", "quantity"].isna().all()
    assert not (merged["quantity"] == 0).any()
    pqt = _read(rs.runs[0].artifact("peptide_quant").path)
    check = quantified.merge(pqt, on="candidate_id", suffixes=("", "_pq"))
    assert np.allclose(check["quantity"], check["quantity_pq"])
    assert np.allclose(check["integration_lo_rt"], check["integration_lo_rt_pq"])
    assert (check["n_fragments_used"] == check["n_fragments_used_pq"]).all()
    assert (check["status"] == "quantified").all()


def test_single_candidate_state(open_fixture):
    rs = open_fixture("single")
    winner = quant_state(rs, None, 4)  # HHALPAR charge 3, the base peptide's winner
    assert winner.state == "quantified"
    assert winner.quantity is not None and winner.quantity > 0
    assert winner.integration_lo_rt <= winner.integration_apex_rt <= winner.integration_hi_rt
    sibling = quant_state(rs, "", 138)  # HHALPAR charge 2, peptide_q_value 1.0
    assert sibling.state == "not_selected"
    assert sibling.quantity is None and sibling.status is None
    assert sibling.gate_q == 1.0 and sibling.window_also_covers == ()
    assert sibling.reason == (
        "not selected: another precursor of this base peptide (candidate 4) is its winning row "
        "and was selected for quant (quantified); peptide_q_value is set on the winning row "
        "only (peptide_q_value = 1 on this row), so quant takes one precursor per base peptide"
    )
    missing = quant_state(rs, None, 999_999_999)
    assert missing.state == "not_selected"
    assert "has no scored row" in missing.reason


def test_not_quantifiable_is_never_zero(tmp_path, fixture_dir):
    """A synthetic copy of peptide_quant with a null quantity for every status."""
    copy = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), copy)
    path = copy / "peptide_quant.parquet"
    table = pq.read_table(path)
    df = table.to_pandas()
    statuses = [
        "no_fragment_traces",
        "no_positive_fragment_area",
        "no_fragments_selected",
        "nonfinite_quantity",
        "status_of_a_future_engine",
    ]
    rows = list(range(len(statuses)))
    ids = df["candidate_id"].iloc[rows].tolist()
    df["quantity"] = df["quantity"].astype(object)
    df.loc[rows, "quantity"] = None
    df.loc[rows, "quant_status"] = statuses
    used = df["n_fragments_used"].to_numpy().copy()
    used[rows] = [0, 0, 0, 3, 0]
    df["n_fragments_used"] = used
    df.loc[rows[0], ["integration_apex_rt", "integration_lo_rt", "integration_hi_rt"]] = None
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), path)

    rs = open_results(copy)
    states = quant_states(rs, None, ids)
    assert (states["state"] == "not_quantifiable").all()
    assert states["status"].tolist() == statuses
    assert states["quantity"].isna().all()
    assert states["n_fragments_used"].tolist() == [0, 0, 0, 3, 0]
    for status, reason in zip(statuses, states["reason"], strict=True):
        assert reason.startswith(describe_status(status))
        assert "missing, not zero" in reason
    assert states["reason"].iloc[-1].startswith("status_of_a_future_engine")
    one = quant_state(rs, None, ids[0])
    assert one.state == "not_quantifiable" and one.quantity is None
    assert one.integration_apex_rt is None

    page = identification_table(
        rs, TableQuery(threshold=None, quant_status="not_quantifiable", limit=100)
    )
    assert page.total == 5
    assert set(page.rows["candidate_id"]) == set(ids)
    assert page.rows["quantity"].isna().all()
    raw = identification_table(rs, TableQuery(threshold=None, quant_status="no_fragment_traces"))
    assert raw.total == 1 and "quant_status 'no_fragment_traces'" in raw.description
    whole = identification_table(rs, TableQuery(threshold=None, limit=1000, include_decoys=True))
    assert not (whole.rows["quantity"] == 0).any()
    assert (whole.rows["quant_state"] == "quantified").sum() == 145

    breakdown = quant_status_breakdown(rs)
    pep = breakdown[breakdown["table"] == "peptide_quant"].set_index("status")["n"].to_dict()
    assert pep == {"quantified": 145, **dict.fromkeys(statuses, 1)}
    unknown = breakdown[breakdown["status"] == "status_of_a_future_engine"]
    assert unknown["description"].iloc[0] == "status_of_a_future_engine"


def test_quant_status_breakdown_equals_the_files(open_fixture):
    for name in ("single", "experiment", "mbr"):
        rs = open_fixture(name)
        breakdown = quant_status_breakdown(rs)
        assert list(breakdown.columns[:4]) == ["run", "table", "status", "n"]
        for run in rs.runs:
            for kind in ("peptide_quant", "protein_group_quant"):
                counts = _read(run.artifact(kind).path)["quant_status"].value_counts().to_dict()
                sub = breakdown[(breakdown["run"] == run.name) & (breakdown["table"] == kind)]
                assert dict(zip(sub["status"], sub["n"], strict=True)) == counts


def test_protein_quant_single(open_fixture):
    rs = open_fixture("single")
    pgq = _read(rs.runs[0].artifact("protein_group_quant").path)
    for row in pgq.itertuples():
        got = protein_quant(rs, None, row.protein_group)
        assert got is not None
        assert got["quantity"] == pytest.approx(row.quantity)
        assert got["n_peptides"] == row.n_peptides
        assert got["state"] == "quantified"
        assert "n_transferred_precursors" not in got
    assert protein_quant(rs, None, "sp|NOPE|NOPE_TEST") is None


# --------------------------------------------------------------------------- experiments


def test_experiment_gate_configured_against_effective(open_fixture):
    rs = open_fixture("experiment")
    for run in (None, "a", "b", 1):
        gate = quant_gate(rs, run)
        assert (gate.q_filter, gate.q_column, gate.threshold) == ("PsmQ", "q_value", 0.01)
        assert (gate.configured, gate.effective) == ("PeptideQ", "PsmQ")
        assert "run-experiment gates every run's quant on the pooled q_value" in gate.note
        assert not gate.transfers_admitted
        assert gate.describe(True) == (
            "the target rows with q_value <= 0.01 (PsmQ: the pooled PSM-level q, "
            "experiment-wide, not run_psm_q; run-experiment forces PsmQ, the configured "
            "PeptideQ is not applied)"
        )


def test_experiment_quant_rows_are_the_pooled_q_gate(open_fixture):
    """Each run's peptide_quant is its target rows with the pooled q_value <= 0.01."""
    rs = open_fixture("experiment")
    pooled = _read(rs.scored.path)
    for run in rs.runs:
        own = pooled[pooled["source"] == run.index]
        expected = set(own[(own["label"] == "target") & (own["q_value"] <= 0.01)]["candidate_id"])
        pqt = set(_read(run.artifact("peptide_quant").path)["candidate_id"])
        assert pqt == expected
        states = quant_states(rs, run.name, own["candidate_id"].to_numpy())
        assert set(states.loc[states["state"] == "quantified", "candidate_id"]) == expected
        rest = states[states["state"] == "not_selected"]
        assert rest["reason"].str.contains("decoy|fails the quant gate q_value").all()
        page = identification_table(rs, TableQuery(run=run.name, threshold=None, limit=1000))
        quantified = page.rows[page.rows["quant_state"] == "quantified"]
        assert set(quantified["candidate_id"]) == expected
    with pytest.raises(ViewerError, match="name one"):
        quant_state(rs, None, 4)


def test_mbr_gate_and_flagged_transfers(open_fixture):
    rs = open_fixture("mbr")
    gate = quant_gate(rs, "a")
    assert (gate.q_filter, gate.threshold) == ("PsmQ", 0.004)
    assert (gate.configured, gate.effective) == ("PeptideQ", "PsmQ")
    assert gate.transfers_admitted
    assert "transfers" in gate.label
    transfers = _read(rs.artifact("mbr_transferred").path)
    native = _read(rs.scored.path)
    for run, n_transfers in (("a", 16), ("c", 32)):
        r = rs.run(run)
        split = _read(r.artifact("psms_scored").path)
        gated = split[
            (split["label"] == "target") & ((split["q_value"] <= 0.004) | split["is_transferred"])
        ]
        pqt = _read(r.artifact("peptide_quant").path)
        assert set(pqt["candidate_id"]) == set(gated["candidate_id"])
        states = quant_states(rs, run, pqt["candidate_id"].to_numpy())
        flagged = set(states.loc[states["from_transfer"], "candidate_id"])
        expected = set(transfers.loc[transfers["source"] == r.index, "candidate_id"])
        assert flagged == expected and len(flagged) == n_transfers
        assert (states["state"] == "quantified").all()
        # Their native q_value (scored_combined) fails 0.004: quantified only by the transfer.
        # On this fixture the split q equals the native q (G3 F7.2); test_tables covers a
        # copy whose split q was lowered, which tells the two apart.
        own = native[(native["source"] == r.index) & native["candidate_id"].isin(expected)]
        assert (own["q_value"] > 0.004).all()
        moved = states[states["from_transfer"]].merge(own, on="candidate_id")
        assert np.allclose(moved["native_q"], moved["q_value"])
        assert np.allclose(moved["gate_q"], moved["q_value"])
        assert (
            states.loc[states["from_transfer"], "reason"]
            .str.contains("quantified only through the transfer")
            .all()
        )
        assert np.allclose(states.loc[states["from_transfer"], "transfer_q"], 0.145833, atol=1e-6)
        assert not states.loc[~states["from_transfer"], "reason"].str.contains("transfer").any()
        summed = 0
        for group in pqt["protein_group"].unique():
            got = protein_quant(rs, run, group)
            summed += got["n_transferred_precursors"]
        assert summed == n_transfers


def test_mbr_without_accepted_transfers(tmp_path, fixture_dir):
    """G3 F5.2: 0 accepted transfers give a 0-row table with null-typed string columns."""
    copy = tmp_path / "exp"
    shutil.copytree(fixture_dir("mbr"), copy)
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
    pq.write_table(schema.empty_table(), copy / "mbr_transferred.parquet")
    rs = open_results(copy)
    assert quant_gate(rs, "a").transfers_admitted  # MBR ran, with no transfer
    page = identification_table(rs, TableQuery(threshold=None, include_decoys=True, limit=1000))
    assert not page.rows["is_transferred"].any()
    assert "(0 of the 565 rows)" in page.description
    states = quant_states(rs, "c", page.rows.loc[page.rows["run"] == "c", "candidate_id"])
    assert not states["from_transfer"].any()
    assert int(lfq_matrix(rs, "precursor", wide=False)["n_transferred"].sum()) == 0


# --------------------------------------------------------------------------- LFQ


def test_lfq_zero_means_missing(open_fixture):
    rs = open_fixture("mbr")
    raw = _read(rs.artifact("lfq_maxlfq_precursor").path)
    n_zero = int((raw["quantity"] == 0).sum())
    assert n_zero == 77
    long = lfq_matrix(rs, "precursor", wide=False)
    assert len(long) == len(raw)
    assert int(long["quantity"].isna().sum()) == n_zero
    assert not (long["quantity"] == 0).any()
    assert set(long["run"]) == {"a", "c"}
    check = long.merge(
        raw.rename(columns={"run": "source"}),
        on=["group", "charge", "source"],
        suffixes=("", "_raw"),
    )
    positive = check["quantity_raw"] > 0
    assert np.allclose(check.loc[positive, "quantity"], check.loc[positive, "quantity_raw"])
    wide = lfq_matrix(rs, "precursor")
    assert len(wide) == raw[["group", "charge"]].drop_duplicates().shape[0]
    assert int(wide[["a", "c"]].isna().sum().sum()) == n_zero
    assert wide.attrs["run_columns"] == ["a", "c"]
    # Every transfer is one precursor cell (G3: 48 cells went from missing to a value).
    assert int(long["n_transferred"].sum()) == 48
    assert int(wide[["n_transferred_a", "n_transferred_c"]].sum().sum()) == 48


def test_lfq_levels_and_labels(open_fixture):
    rs = open_fixture("mbr")
    protein = lfq_matrix(rs, "protein")
    assert list(protein.columns[:4]) == ["protein_group", "n_features", "a", "c"]
    assert protein.attrs["column_labels"]["n_features"] == N_FEATURES_LABEL
    assert N_FEATURES_LABEL == "precursors in any run"
    raw = _read(rs.artifact("lfq_maxlfq").path)
    assert len(protein) == raw["protein_group"].nunique()
    cells = lfq_matrix(rs, "protein", wide=False)
    # G3 F8.3: 18 protein cells contain at least one transferred precursor.
    assert int((cells["n_transferred"] > 0).sum()) == 18
    peptide = lfq_matrix(rs, "peptide")
    assert (peptide["charge"] == -1).all()
    assert "missing" in peptide.attrs["description"]
    assert "not FDR-controlled" in peptide.attrs["description"]


def test_lfq_accepted_at(open_fixture):
    rs = open_fixture("experiment")
    scored = _read(rs.scored.path)
    accepted = scored[(scored["label"] == "target") & (scored["precursor_q"] <= 0.01)]
    precursor = lfq_matrix(rs, "precursor", accepted_at=0.01)
    assert len(precursor) == accepted[["peptidoform", "charge"]].drop_duplicates().shape[0]
    assert len(lfq_matrix(rs, "protein", accepted_at=0.01)) == 0  # protein floor 1/16
    assert len(lfq_matrix(rs, "protein", accepted_at=1.0)) == 16


def test_lfq_needs_an_experiment(open_fixture):
    with pytest.raises(ViewerError, match="only across the runs of an experiment"):
        lfq_matrix(open_fixture("single"))
    with pytest.raises(ViewerError, match="unknown LFQ level"):
        lfq_matrix(open_fixture("experiment"), "fragment")  # type: ignore[arg-type]


def test_missing_quant_table_raises(tmp_path, fixture_dir):
    """An absent table is refused with the location where it was expected."""
    copy = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), copy)
    (copy / "peptide_quant.parquet").unlink()
    rs = open_results(copy)
    expected = str(copy / "peptide_quant.parquet")
    with pytest.raises(ArtifactNotFound) as err:
        quant_states(rs, None, [4])
    message = str(err.value)
    assert message.startswith("the peptide_quant table of run cannot be read: ")
    assert f"{expected} does not exist (recorded as " in message
    page = identification_table(rs, TableQuery())
    assert page.rows["quant_state"].isna().all()
    assert f"The quant columns are empty: peptide_quant.parquet is missing: {expected}" in (
        page.description
    )
    notes = quant_status_breakdown(rs).attrs["notes"]
    assert len(notes) == 1 and notes[0].startswith("run peptide_quant is not counted: ")


def test_unsupported_versions_are_refused(tmp_path, fixture_dir):
    """Hard rule 5: an unknown schema version is refused with the version error (issue 5)."""
    single = _copy(tmp_path, fixture_dir, "single")
    _edit_json(
        single / "manifest.json",
        lambda m: m["artifacts"]["peptide_quant"].__setitem__("schema_version", 99),
    )
    rs = open_results(single)
    with pytest.raises(
        SchemaVersionError, match="peptide_quant schema version 99 is not supported"
    ):
        quant_states(rs, None, [4])
    with pytest.raises(SchemaVersionError, match="schema version 99"):
        quant_state(rs, None, 4)
    page = identification_table(rs, TableQuery())
    assert "The quant columns are empty: " in page.description
    assert "peptide_quant schema version 99 is not supported" in page.description
    breakdown = quant_status_breakdown(rs)
    assert set(breakdown["table"]) == {"protein_group_quant"}
    assert any("schema version 99" in n for n in breakdown.attrs["notes"])

    exp = _copy(tmp_path, fixture_dir, "experiment")

    def versions(m: dict) -> None:
        m["artifacts"]["lfq_maxlfq"]["schema_version"] = 2
        m["artifacts"]["protein_group_quant[a]"]["schema_version"] = 3

    _edit_json(exp / "experiment_manifest.json", versions)
    rs = open_results(exp)
    with pytest.raises(SchemaVersionError, match="lfq_maxlfq schema version 2 is not supported"):
        lfq_matrix(rs, "protein")
    with pytest.raises(
        SchemaVersionError, match="protein_group_quant schema version 3 is not supported"
    ):
        protein_quant(rs, "a", "sp|FIXT05|FIX05_TEST")
    assert protein_quant(rs, "b", "sp|FIXT05|FIX05_TEST") is not None
    assert identification_table(rs, TableQuery(run="a")).total == 273


def test_sibling_of_an_unquantifiable_winner(tmp_path, fixture_dir):
    """The sibling reason gives the winner's own state, not 'is quantified' (issue 7)."""
    out = _copy(tmp_path, fixture_dir, "single")

    def change(df: pd.DataFrame) -> pd.DataFrame:
        i = df.index[df["candidate_id"] == 4][0]  # HHALPAR/3, the base peptide's winner
        df["quantity"] = df["quantity"].astype(object)
        df.loc[i, "quantity"] = None
        df.loc[i, "quant_status"] = "no_fragment_traces"
        df.loc[i, "n_fragments_used"] = 0
        df.loc[i, ["integration_apex_rt", "integration_lo_rt", "integration_hi_rt"]] = None
        return df

    _rewrite(out / "peptide_quant.parquet", change)
    rs = open_results(out)
    winner = quant_state(rs, None, 4)
    assert winner.state == "not_quantifiable" and winner.quantity is None
    sibling = quant_state(rs, None, 138)
    assert sibling.state == "not_selected"
    assert (
        "(candidate 4) is its winning row and was selected for quant (not quantifiable: "
        "no_fragment_traces)" in sibling.reason
    )
    assert "is quantified" not in sibling.reason


def test_winner_at_the_cap_is_not_called_a_loser(tmp_path, fixture_dir):
    """A 1.0 on a grouped gate column does not mean 'not the winning row' (issue 8).

    Under a PrecursorQ gate in a single run every row is the winner of its own
    (peptidoform, charge) group. One target row gets precursor_q 1.0, a winner at the
    cap (T1 F3.8).
    """
    out = _copy(tmp_path, fixture_dir, "single")
    _edit_json(
        out / "peptide_quant.parquet.report.json",
        lambda r: r["params"].__setitem__("q_filter", "PrecursorQ"),
    )
    selected = set(_read(out / "peptide_quant.parquet", ["candidate_id"])["candidate_id"])
    scored = _read(out / "psms_scored.parquet")
    free = scored[(scored["label"] == "target") & ~scored["candidate_id"].isin(selected)]
    cid = int(free.sort_values("q_value").iloc[-1]["candidate_id"])
    _rewrite(
        out / "psms_scored.parquet",
        lambda df: df.assign(precursor_q=df["precursor_q"].where(df["candidate_id"] != cid, 1.0)),
    )
    rs = open_results(out)
    gate = quant_gate(rs)
    assert (gate.q_filter, gate.q_column) == ("PrecursorQ", "precursor_q")
    state = quant_state(rs, None, cid)
    assert state.state == "not_selected" and state.gate_q == 1.0
    assert state.reason == (
        "not selected: precursor_q = 1 fails the quant gate precursor_q <= 0.01 (this row is "
        "the winning row of its precursor (peptidoform, charge); the q of a winning row can be "
        "at the cap of 1.0)"
    )
    states = quant_states(rs, None, scored["candidate_id"].to_numpy())
    assert not states["reason"].str.contains("winning row of this").any()


def test_window_covering_another_peak(open_fixture):
    """G1 R10: a promoted peak's integration window that also covers another peak rank."""
    rs = open_fixture("topk")
    scored = _read(rs.scored.path, ["candidate_id", "selected_peak_rank"])
    peaks = _read(
        rs.runs[0].artifact("psms_extracted").path, ["candidate_id", "peak_rank", "apex_rt"]
    )
    pqt = _read(rs.runs[0].artifact("peptide_quant").path)
    states = quant_states(rs, None, scored["candidate_id"].to_numpy())
    # Independent: the other ranks whose float32 apex lies inside [lo, hi] (G1 R10 SQL).
    m = peaks.merge(scored, on="candidate_id").merge(pqt, on="candidate_id")
    m = m[m["peak_rank"] != m["selected_peak_rank"]]
    apex = m["apex_rt"].astype(np.float32).astype(np.float64)
    inside = m[(apex >= m["integration_lo_rt"]) & (apex <= m["integration_hi_rt"])]
    expected = inside.groupby("candidate_id")["peak_rank"].apply(
        lambda r: ", ".join(str(int(v)) for v in sorted(r))
    )
    got = states.dropna(subset=["window_also_covers"]).set_index("candidate_id")
    assert got["window_also_covers"].to_dict() == expected.to_dict()
    both = states.merge(scored, on="candidate_id")
    promoted = both[(both["selected_peak_rank"] == 1) & (both["state"] == "quantified")]
    assert len(promoted) == 16
    assert (promoted["window_also_covers"] == "0").all()  # G1 F10.2: 16 of 16
    assert (
        promoted["reason"]
        .str.contains(
            "; the integration window also covers the apex of another peak of this candidate "
            "(peak_rank 0 at ",
            regex=False,
        )
        .all()
    )
    alternates = set(peaks.loc[peaks["peak_rank"] > 0, "candidate_id"])
    kept = both[(both["selected_peak_rank"] == 0) & both["candidate_id"].isin(alternates)]
    assert len(kept) and kept["window_also_covers"].isna().all()  # G1 F10.2
    one = quant_state(rs, None, int(promoted["candidate_id"].iloc[0]))
    assert one.window_also_covers == (0,)
    assert "psms_extracted.parquet" in states.attrs["peaks"]
