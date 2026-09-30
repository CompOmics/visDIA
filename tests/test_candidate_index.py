"""The candidate index against a brute-force scan of the candidate_id column."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import Cache, InconsistentData
from mumdia_viewer.data.candidate_index import (
    CandidateIndex,
    DenseIndex,
    concat_parts,
    index_for,
    read_candidate_rows,
)
from mumdia_viewer.data.pqio import ParquetHandle


def _brute(path: Path) -> dict[int, list[int]]:
    ids = pq.read_table(path, columns=["candidate_id"]).column(0).to_numpy()
    rows: dict[int, list[int]] = {}
    for i, c in enumerate(ids):
        rows.setdefault(int(c), []).append(i)
    return rows


def _assert_matches(index: CandidateIndex, path: Path) -> None:
    expected = _brute(path)
    assert len(index) == len(expected)
    for cid, rows in expected.items():
        assert index.ranges(cid) == [(rows[0], rows[-1] + 1)], cid
        assert rows[-1] - rows[0] + 1 == len(rows)


@pytest.mark.parametrize(
    ("fixture", "kind"),
    [
        ("single", "chromatograms"),
        ("single", "psms_extracted"),
        ("single", "features"),
        ("chrom_v2_rg1", "chromatograms"),
        ("topk", "psms_extracted"),
        ("topk", "features"),
        ("ovl128_pool", "chromatograms"),
    ],
)
def test_index_matches_brute_force(open_fixture, kind, fixture):
    artifact = open_fixture(fixture).runs[0].artifact(kind)
    index = CandidateIndex.build(artifact.parquet())
    _assert_matches(index, artifact.path)
    assert index.file_sorted and index.contiguous


def test_unsorted_pooled_table_of_overlapping_bands(open_fixture):
    # A later band that wins a candidate breaks the global order; each candidate stays contiguous.
    artifact = open_fixture("ovl_bp_pool").runs[0].artifact("chromatograms")
    index = CandidateIndex.build(artifact.parquet())
    assert not index.file_sorted and index.contiguous
    _assert_matches(index, artifact.path)


def test_segments_follow_row_groups(open_fixture):
    artifact = open_fixture("chrom_v2_rg1").runs[0].artifact("chromatograms")
    handle = artifact.parquet()
    index = CandidateIndex.build(handle)
    cid = int(index.ids[5])
    start, stop = index.rows(cid)
    segments = index.segments(cid)
    # One row per row group in this fixture: one segment per row.
    assert len(segments) == stop - start
    assert all(s.length == 1 and s.start == 0 for s in segments)
    parts = read_candidate_rows(handle, index, cid, ["candidate_id", "frag_name"])
    table = concat_parts(parts)
    assert table.column("candidate_id").to_pylist() == [cid] * (stop - start)


def test_index_is_cached_by_content_hash(open_fixture, tmp_path: Path):
    artifact = open_fixture("single").runs[0].artifact("chromatograms")
    cache = Cache(tmp_path / "cache")
    first = CandidateIndex.for_artifact(artifact, cache)
    files = list((tmp_path / "cache").rglob("candidate_index_*.npz"))
    assert len(files) == 1 and artifact.content_hash in str(files[0])
    fresh = Cache(tmp_path / "cache")  # a new process: no memory hit
    second = CandidateIndex.for_artifact(artifact, fresh)
    assert np.array_equal(first.ids, second.ids) and np.array_equal(first.starts, second.starts)


def test_dense_index_for_run_windows(open_fixture):
    run = open_fixture("single").runs[0]
    artifact = run.artifact("run_windows")
    index = index_for(artifact)
    assert isinstance(index, DenseIndex)
    table = concat_parts(read_candidate_rows(artifact.parquet(), index, 17, ["rt_pred_cal"]))
    direct = pq.read_table(artifact.path, columns=["candidate_id", "rt_pred_cal"]).slice(17, 1)
    assert table.column("rt_pred_cal").to_pylist() == direct.column("rt_pred_cal").to_pylist()
    assert 10**9 not in index and index.rows(10**9) is None


def _write(path: Path, ids: list[int], row_group_size: int) -> ParquetHandle:
    table = pa.table({"candidate_id": pa.array(ids, pa.uint32()), "v": list(range(len(ids)))})
    pq.write_table(table, path, row_group_size=row_group_size)
    return ParquetHandle(path)


def test_dense_read_checks_the_returned_id(tmp_path: Path):
    # Footer statistics cannot see a permutation inside one row group ([3, 2] has min 2 and
    # max 3), so the dense index applies and every read checks the id it returns.
    handle = _write(tmp_path / "w.parquet", [0, 1, 3, 2], row_group_size=2)
    assert DenseIndex.applies(handle)
    index = DenseIndex(4, handle.row_group_offsets())
    with pytest.raises(InconsistentData, match="not the row index"):
        read_candidate_rows(handle, index, 2, ["v"])
    assert not DenseIndex.applies(_write(tmp_path / "g.parquet", [0, 2, 3, 4], row_group_size=2))


def test_non_contiguous_candidate_is_reported(tmp_path: Path):
    handle = _write(tmp_path / "t.parquet", [5, 5, 7, 5, 9], row_group_size=2)
    index = CandidateIndex.build(handle)
    assert not index.file_sorted and not index.contiguous
    assert index.ranges(5) == [(0, 2), (3, 4)]
    with pytest.raises(InconsistentData, match="not contiguous"):
        index.rows(5)
    assert index.rows(7) == (2, 3)
    assert index.rows(8) is None


def test_candidate_crossing_a_row_group_seam(tmp_path: Path):
    handle = _write(tmp_path / "s.parquet", [1, 1, 1, 2, 2, 3], row_group_size=2)
    index = CandidateIndex.build(handle)
    assert index.rows(1) == (0, 3) and index.rows(2) == (3, 5)
    assert [(s.row_group, s.start, s.stop) for s in index.segments(2)] == [(1, 1, 2), (2, 0, 1)]
