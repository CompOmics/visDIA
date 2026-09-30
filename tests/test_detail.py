"""Precursor detail assembly: every part checked against a direct read of the artifacts."""

import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.detail import detail_percentiles, mirror, precursor_detail


def _accepted(rs, run, *, rank: int | None = None):
    t = pq.read_table(
        run.artifact("psms_scored").path,
        columns=["candidate_id", "label", "q_value", "selected_peak_rank", "source"],
    ).to_pandas()
    t = t[(t.label == "target") & (t.q_value <= 0.01)]
    if rank is not None:
        t = t[t.selected_peak_rank == rank]
    return [int(c) for c in t.candidate_id]


@pytest.mark.parametrize("name", ["single", "grouped", "experiment", "topk", "mbr", "ovl_bp"])
def test_detail_parts_equal_direct_reads(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[-1]
    scored = pq.read_table(run.artifact("psms_scored").path).to_pandas().set_index("candidate_id")
    for cid in _accepted(rs, run)[:25]:
        d = precursor_detail(rs, run, cid)
        row = scored.loc[cid]
        assert d.scored["score"] == row["score"] and d.scored["label"] == row["label"]
        assert d.selected_peak_rank == int(row["selected_peak_rank"])
        # The chromatogram: fragment rows then the three MS1 rows.
        names = [t.frag_name for t in d.chromatogram.traces]
        assert names[-3:] == ["ms1_mono", "ms1_iso1", "ms1_iso2"]
        # The identification apex is a point of the axis (compared in float32).
        axis = d.chromatogram.common_axis()
        assert np.float32(d.markers["apex_rt"]) in axis
        # The apex scan is the exact scan at apex_rt in a window covering the precursor.
        assert d.apex_scan is not None and d.apex_scan.exact
        assert d.apex_scan.window_lower <= d.precursor_mz <= d.apex_scan.window_upper
        # Grouped q columns: the winner flag equals "q below 1" on this data.
        for q in d.q_values:
            if q.grouped:
                assert q.winner == (q.value < 1.0), (cid, q.column)
        # Every evidence line names its source; derived values say how.
        for item in d.evidence:
            assert item.label and item.source
            if item.derived:
                assert item.source.startswith("viewer-derived")


def test_alternative_peak_selected(open_fixture):
    rs = open_fixture("topk")
    run = rs.runs[0]
    cids = _accepted(rs, run, rank=1)
    assert cids, "the top-K fixture has identifications on peak rank 1"
    d = precursor_detail(rs, run, cids[0])
    assert d.selected_peak_rank == 1 and len(d.peaks) >= 2
    selected = [p for p in d.markers["peaks"] if p["selected"]]
    assert len(selected) == 1 and selected[0]["peak_rank"] == 1
    assert selected[0]["apex_rt"] == pytest.approx(d.markers["apex_rt"])
    assert d.features is not None and int(d.features["peak_rank"]) == 1


def test_mirror_uses_the_chromatogram_fragments(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    d = precursor_detail(rs, run, _accepted(rs, run)[0])
    m = mirror(rs, d)
    frags = d.chromatogram.fragments()
    assert list(m.fragments["name"]) == [t.frag_name for t in frags]
    assert m.pick.row == d.apex_scan.row
    tol = d.tolerance.tol_ppm
    for match in m.matches:
        assert abs(match.ppm_corrected) <= tol + 1e-6
    assert m.previous_row is not None or m.next_row is not None
    # Stepping to a neighbouring scan keeps the window.
    other = mirror(rs, d, row=m.next_row if m.next_row is not None else m.previous_row)
    assert other.pick.window_id == m.pick.window_id


def test_transfer_record_in_an_mbr_experiment(open_fixture):
    rs = open_fixture("mbr")
    tr = pq.read_table(rs.extra["mbr_transferred"].path).to_pandas()
    first = tr.iloc[0]
    run = rs.run(int(first["source"]))
    d = precursor_detail(rs, run, int(first["candidate_id"]))
    assert d.transfer is not None and d.transfer["candidate_id"] == int(first["candidate_id"])


def test_percentiles_rank_the_selected_peak(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    d = precursor_detail(rs, run, _accepted(rs, run)[3])
    ranked = detail_percentiles(rs, d)
    assert ranked and all(p.feature in d.features for p in ranked)
    assert all(p.pct_target is None or 0 <= p.pct_target <= 100 for p in ranked)


def test_decoy_row_and_its_competition(open_fixture):
    rs = open_fixture("grouped")
    run = rs.runs[0]
    t = pq.read_table(run.artifact("psms_scored").path, columns=["candidate_id", "label"])
    decoys = [r["candidate_id"] for r in t.to_pylist() if r["label"] == "decoy"]
    d = precursor_detail(rs, run, decoys[0])
    assert d.is_decoy and "this row is a decoy" in d.notes
    assert d.competition["is_this_row"].sum() == 1
    assert d.competition["wins_peptide"].sum() == 1


@pytest.mark.real_data
def test_real_detail_is_fast_warm(real_single):
    rs = open_results(real_single)
    run = rs.runs[0]
    cids = _accepted(rs, run)
    precursor_detail(rs, run, cids[0])  # first call: indexes, scan table, partner map
    times = []
    for cid in cids[1000:1020]:
        t0 = time.perf_counter()
        precursor_detail(rs, run, cid)
        times.append(time.perf_counter() - t0)
    print(
        f"[astral] warm precursor detail median {1000 * np.median(times):.0f} ms, "
        f"max {1000 * max(times):.0f} ms"
    )
    assert np.median(times) < 1.0
