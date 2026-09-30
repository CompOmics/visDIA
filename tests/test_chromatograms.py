"""Chromatogram decoding (layouts 1 to 4) and the chromatogram sources of a run.

The oracles are read with pyarrow directly: the layout 1 table of the same search
(``smoke/out_chrom_v1``), the pooled twin of each grouped fixture, and the loser files.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import (
    ArtifactNotFound,
    InconsistentData,
    LayoutError,
    SchemaVersionError,
    open_results,
)
from mumdia_viewer.data.candidate_index import CandidateIndex
from mumdia_viewer.data.chromatograms import (
    LOSERS_BAND_TABLES_KEY,
    CandidateChromatogram,
    ChromatogramSource,
    Trace,
    decode_rows,
    parse_fragment_name,
    row_group_bytes,
)
from mumdia_viewer.data.pqio import ROW_GROUP_CACHE
from mumdia_viewer.data.schemas import ChromatogramLayout, chromatogram_layout

V1 = ChromatogramLayout(family=1, has_im=False)
V2 = ChromatogramLayout(family=2, has_im=False)
V3 = ChromatogramLayout(family=1, has_im=True)
V4 = ChromatogramLayout(family=2, has_im=True)
LIST = pa.large_list(pa.float32())
SCALARS = ["candidate_id", "frag_name", "frag_mz", "frag_obs_mz", "predicted_intensity"]


# --------------------------------------------------------------------------- helpers


def bits32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def bits64(x: float) -> int:
    return int(np.array([x], dtype=np.float64).view(np.uint64)[0])


def same32(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(bits32(a), bits32(b))


def list_rows(column: pa.ChunkedArray) -> list[np.ndarray]:
    """Per-row float32 arrays of a list column, bit for bit (null lists are empty)."""
    arr = column.combine_chunks()
    if arr.null_count:
        arr = arr.fill_null([])
    offsets = arr.offsets.to_numpy()
    values = arr.flatten().to_numpy(zero_copy_only=False)
    base = offsets[0]
    return [values[a - base : b - base] for a, b in itertools.pairwise(offsets)]


def v1_oracle(path: Path) -> dict[int, list[dict]]:
    """The rows of a layout 1 table grouped by candidate, in file order."""
    t = pq.read_table(path, columns=[*SCALARS, "rt", "intensity"])
    rt = list_rows(t.column("rt"))
    it = list_rows(t.column("intensity"))
    out: dict[int, list[dict]] = defaultdict(list)
    cols = {c: t.column(c).to_pylist() for c in SCALARS}
    for k in range(t.num_rows):
        out[int(cols["candidate_id"][k])].append(
            {
                "frag_name": cols["frag_name"][k],
                "frag_mz": cols["frag_mz"][k],
                "frag_obs_mz": cols["frag_obs_mz"][k],
                "predicted_intensity": cols["predicted_intensity"][k],
                "rt": rt[k],
                "intensity": it[k],
            }
        )
    return dict(out)


def assert_trace_equals_row(trace: Trace, row: dict) -> None:
    assert trace.frag_name == row["frag_name"]
    assert bits64(trace.frag_mz) == bits64(row["frag_mz"])
    assert bits64(trace.frag_obs_mz) == bits64(row["frag_obs_mz"])
    assert bits32(np.float32(trace.predicted_intensity)) == bits32(
        np.float32(row["predicted_intensity"])
    )
    assert same32(trace.rt, row["rt"]), trace.frag_name
    assert same32(trace.intensity, row["intensity"]), trace.frag_name
    assert trace.trace_len == len(row["rt"])


def assert_same_chromatogram(a: CandidateChromatogram, b: CandidateChromatogram) -> None:
    assert a.candidate_id == b.candidate_id
    assert len(a.traces) == len(b.traces)
    for x, y in zip(a.traces, b.traces, strict=True):
        assert x.frag_name == y.frag_name
        assert bits64(x.frag_mz) == bits64(y.frag_mz)
        assert bits64(x.frag_obs_mz) == bits64(y.frag_obs_mz)
        assert x.predicted_intensity == y.predicted_intensity
        assert same32(x.rt, y.rt) and same32(x.intensity, y.intensity)
        assert x.trace_len == y.trace_len


def encode_v2(cids, rows, row_group_rows, ims=None):
    """A port of the engine's ``Encoder::encode`` (and ``encode_im``) for synthetic tables."""
    out = {"rt_axis": [], "intensity_trimmed": [], "trace_offset": [], "trace_len": []}
    out_im = []
    open_cid, axis = None, None
    for k, (cid, (rt, it)) in enumerate(zip(cids, rows, strict=True)):
        rt = np.asarray(rt, np.float32)
        it = np.asarray(it, np.float32)
        im = None if ims is None else np.asarray(ims[k], np.float32)
        if k % row_group_rows == 0:
            open_cid = None
        n = len(rt)
        if n == 0:
            out["rt_axis"].append([])
            out["intensity_trimmed"].append([])
            out["trace_offset"].append(0)
            out["trace_len"].append(0)
            out_im.append([])
            continue
        if open_cid == cid and axis is not None and same32(axis, rt):
            out["rt_axis"].append([])
        else:
            open_cid, axis = cid, rt
            out["rt_axis"].append(rt)
        nz = np.flatnonzero(it.view(np.uint32) != 0)
        lo, hi = (0, 0) if nz.size == 0 else (int(nz[0]), int(nz[-1]) + 1)
        out["intensity_trimmed"].append(it[lo:hi])
        out["trace_offset"].append(lo)
        out["trace_len"].append(n)
        out_im.append([] if im is None else im[lo:hi])
    return out, out_im


def scalar_arrays(cids, names=None):
    n = len(cids)
    names = names or [f"y{3 + (k % 5)}" for k in range(n)]
    return {
        "candidate_id": pa.array(cids, pa.uint32()),
        "frag_name": pa.array(names, pa.string()),
        "frag_mz": pa.array([500.0 + k for k in range(n)], pa.float64()),
        "frag_obs_mz": pa.array([500.0 + k for k in range(n)], pa.float64()),
        "predicted_intensity": pa.array([1.0] * n, pa.float32()),
    }


def lists(values) -> pa.Array:
    return pa.array([np.asarray(v, np.float32).tolist() for v in values], type=LIST)


def v2_table(cids, rows, row_group_rows=1 << 16, ims=None, names=None) -> pa.Table:
    enc, enc_im = encode_v2(cids, rows, row_group_rows, ims)
    cols = scalar_arrays(cids, names)
    cols["rt_axis"] = lists(enc["rt_axis"])
    cols["intensity_trimmed"] = lists(enc["intensity_trimmed"])
    cols["trace_offset"] = pa.array(enc["trace_offset"], pa.uint32())
    cols["trace_len"] = pa.array(enc["trace_len"], pa.uint32())
    if ims is not None:
        cols["im_trimmed"] = lists(enc_im)
    return pa.table(cols)


def raw_v2(cids, rt_axis, trimmed, offsets, lengths) -> pa.Table:
    cols = scalar_arrays(cids)
    cols["rt_axis"] = lists(rt_axis)
    cols["intensity_trimmed"] = lists(trimmed)
    cols["trace_offset"] = pa.array(offsets, pa.uint32())
    cols["trace_len"] = pa.array(lengths, pa.uint32())
    return pa.table(cols)


def engine_fixture():
    """Rows in the spirit of the engine's ``rows_fixture``: absent rows first and inside,
    all-zero traces, -0.0 and NaN at the edges, a candidate whose rows change axes (sparse
    mode), an axis that differs from the previous one only in the sign of a zero, one-row
    and absent-only candidates, and a long grid."""
    a = np.array([10.0, 11.0, 12.0, 13.0], np.float32)
    b = np.array([20.0, 21.5, 23.0], np.float32)
    z = np.array([0.0, 1.0, 2.0], np.float32)
    nz = np.array([-0.0, 1.0, 2.0], np.float32)
    long_axis = np.arange(40, dtype=np.float32) * 0.5
    e = np.zeros(0, np.float32)
    rows = [
        (1, e, e),
        (1, a, [0, 5, 7, 0]),
        (1, e, e),
        (1, a, [3, 0, 0, 9]),
        (1, a, [0, 0, 0, 0]),
        (1, a, [-0.0, 2, 0, 0]),
        (1, a, [0, 0, 1, np.nan]),
        (2, b, [1, 2, 3]),
        (3, e, e),
        (3, e, e),
        (4, a, [0, 1, 0, 0]),
        (4, b, [0, 0, 8]),
        (4, a, [2, 0, 0, 0]),
        (5, z, [0, 4, 0]),
        (5, nz, [0, 5, 0]),
        (6, long_axis, np.where(np.arange(40) % 7 == 0, 0, np.arange(40)).astype(np.float32)),
        (6, long_axis, np.zeros(40, np.float32)),
        (7, e, e),
        (7, b, [0, 6, 0]),
    ]
    cids = [r[0] for r in rows]
    dense = [(np.asarray(r[1], np.float32), np.asarray(r[2], np.float32)) for r in rows]
    ims = [
        np.where(it.view(np.uint32) != 0, 0.8 + 0.01 * k, 0.0) for k, (_, it) in enumerate(dense)
    ]
    return cids, dense, [np.asarray(m, np.float32) for m in ims]


def synthetic_rows(
    n_candidates: int, rows_per: int, points: int, *, seed: int
) -> tuple[list[int], list[tuple[np.ndarray, np.ndarray]]]:
    """Rows with an axis of their own each (as in sparse mode), so every layout stores all."""
    rng = np.random.default_rng(seed)
    cids, dense = [], []
    for c in range(1, n_candidates + 1):
        for _ in range(rows_per):
            axis = np.sort(rng.random(points, dtype=np.float32) * np.float32(1000))
            cids.append(c)
            dense.append((axis, rng.random(points, dtype=np.float32) + np.float32(1)))
    return cids, dense


def v1_table(cids, dense, ims=None) -> pa.Table:
    cols = scalar_arrays(cids)
    cols["rt"] = lists([r for r, _ in dense])
    cols["intensity"] = lists([i for _, i in dense])
    if ims is not None:
        cols["im"] = lists(ims)
    return pa.table(cols)


def copy_fixture(fixture_dir, name: str, tmp_path: Path) -> Path:
    dst = tmp_path / name
    shutil.copytree(fixture_dir(name), dst)
    return dst


def replace_chromatograms(
    run_dir: Path, table: pa.Table, version: int | None, row_group_size: int | None = None
) -> None:
    """Write a new chromatograms table into a copied run and update its record.

    The recorded content hash is removed, so the viewer keys derived data by a footer
    fingerprint and never reuses an index built for the original file.
    """
    pq.write_table(table, run_dir / "chromatograms.parquet", row_group_size=row_group_size)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    record = manifest["artifacts"]["chromatograms"]
    record["content_hash"] = None
    record["rows"] = table.num_rows
    record["schema_version"] = version
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report = run_dir / "chromatograms.parquet.report.json"
    if report.exists():
        report.unlink()


def rewrite_losers(run_dir: Path, *, records=None, rows: pa.Table | None = None) -> list[dict]:
    """Rewrite ``groups/overlap_losers.parquet`` of a copied run (footer and/or rows)."""
    path = run_dir / "groups" / "overlap_losers.parquet"
    pf = pq.ParquetFile(path)
    recs = json.loads(pf.metadata.metadata[LOSERS_BAND_TABLES_KEY])
    table = pf.read() if rows is None else rows
    new = records(recs) if callable(records) else recs
    meta = {LOSERS_BAND_TABLES_KEY: json.dumps(new).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)
    return new


def band_table_ids(run_dir: Path, name: str) -> set[int]:
    p = run_dir / "groups" / name
    return set(pq.read_table(p, columns=["candidate_id"]).column(0).to_pylist())


# --------------------------------------------------------------------------- names


def test_fragment_names():
    assert parse_fragment_name("b3") == ("b", 3, 1)
    assert parse_fragment_name("y7^2") == ("y", 7, 2)
    assert parse_fragment_name("y10^3") == ("y", 10, 3)
    for name in ("ms1_mono", "ms1_iso2", "", "b", "x3", "b3-H2O", "y3^", "Y3", "b3^2^2"):
        assert parse_fragment_name(name) is None, name
    e = np.zeros(0, np.float32)
    t = Trace("y7^2", 700.0, 700.0, 0.5, e, e, None, 0)
    assert (t.ion, t.ordinal, t.fragment_charge) == ("y", 7, 2)
    assert not t.is_ms1 and not t.observed
    m = Trace(
        "ms1_iso1", 500.5, 500.5, 0.0, np.ones(2, np.float32), np.zeros(2, np.float32), None, 2
    )
    assert m.is_ms1 and m.observed and m.ion is None and m.fragment_charge is None


# --------------------------------------------------------------------------- v2 == v1


@pytest.mark.parametrize("name", ["single", "chrom_v2_rg1"])
def test_v2_source_equals_v1_for_every_candidate(open_fixture, fixture_dir, name):
    oracle = v1_oracle(fixture_dir("chrom_v1") / "chromatograms.parquet")
    rs = open_fixture(name)
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert src.tables[0].layout == V2 and not src.per_band and src.notice is None
    assert src.candidate_ids().tolist() == sorted(oracle)
    points = 0
    for cid, rows in oracle.items():
        chrom = src.read(cid)
        assert chrom is not None and chrom.candidate_id == cid and chrom.band is None
        assert len(chrom.traces) == len(rows)
        for trace, row in zip(chrom.traces, rows, strict=True):
            assert_trace_equals_row(trace, row)
            points += trace.trace_len
    assert points == 29_832
    if name == "chrom_v2_rg1":
        # One row per row group: a candidate's nine rows are nine row groups.
        assert all(len(src.read(c).row_groups) == 9 for c in list(oracle)[:20])


def test_v1_source_reads_the_stored_lists(open_fixture, fixture_dir):
    oracle = v1_oracle(fixture_dir("chrom_v1") / "chromatograms.parquet")
    rs = open_fixture("chrom_v1")
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert src.tables[0].layout == V1
    for cid, rows in oracle.items():
        chrom = src.read(cid)
        for trace, row in zip(chrom.traces, rows, strict=True):
            assert_trace_equals_row(trace, row)


@pytest.mark.parametrize("name", ["single", "chrom_v2_rg1"])
def test_decode_rows_per_row_group_equals_v1(fixture_dir, name):
    oracle = v1_oracle(fixture_dir("chrom_v1") / "chromatograms.parquet")
    expected = [row for rows in oracle.values() for row in rows]
    pf = pq.ParquetFile(fixture_dir(name) / "chromatograms.parquet")
    layout = chromatogram_layout(pf.schema_arrow)
    got = []
    for g in range(pf.metadata.num_row_groups):
        got.extend(decode_rows(pf.read_row_group(g), layout))
    assert len(got) == len(expected) == 2_556
    for (rt, it, im), row in zip(got, expected, strict=True):
        assert im is None
        assert same32(rt, row["rt"]) and same32(it, row["intensity"])
        assert not rt.flags.writeable and not it.flags.writeable


def test_scalar_columns_equal_v1(fixture_dir):
    v1 = pq.read_table(fixture_dir("chrom_v1") / "chromatograms.parquet", columns=SCALARS)
    for name in ("single", "chrom_v2_rg1"):
        v2 = pq.read_table(fixture_dir(name) / "chromatograms.parquet", columns=SCALARS)
        assert v2.equals(v1)


# --------------------------------------------------------------------------- row content


def test_rows_per_candidate_and_ms1_last(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    ex = pq.read_table(
        run.artifacts["psms_extracted"].path,
        columns=["candidate_id", "n_predicted_fragments", "n_matched_fragments", "apex_rt"],
    ).to_pylist()
    assert len(ex) == 284 == len(src.candidate_ids())
    for row in ex:
        chrom = src.read(row["candidate_id"])
        assert len(chrom.traces) == row["n_predicted_fragments"] + 3
        assert [t.frag_name for t in chrom.traces[-3:]] == ["ms1_mono", "ms1_iso1", "ms1_iso2"]
        assert all(t.predicted_intensity == 0.0 for t in chrom.ms1())
        assert not any(t.is_ms1 for t in chrom.fragments())
        assert len(chrom.fragments()) == row["n_predicted_fragments"]
        assert sum(t.observed for t in chrom.fragments()) == row["n_matched_fragments"]
        axis = chrom.common_axis()
        assert axis is not None
        assert int((bits32(axis) == bits32(np.float32(row["apex_rt"]))).sum()) == 1
        for t in chrom.traces:
            assert not t.observed or same32(t.rt, axis)


def test_matrix_and_axis_views(open_fixture):
    rs = open_fixture("single")
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    chrom = next(
        src.read(c)
        for c in src.candidate_ids().tolist()
        if not all(t.observed for t in src.read(c).fragments())
    )
    axis, names, values = chrom.matrix()
    frags = chrom.fragments()
    assert names == tuple(t.frag_name for t in frags)
    assert values.shape == (len(frags), axis.size) and values.dtype == np.float32
    for i, t in enumerate(frags):
        expected = t.intensity if t.observed else np.zeros(axis.size, np.float32)
        assert same32(values[i], expected)
    axis2, names2, values2 = chrom.matrix(include_ms1=True)
    assert same32(axis2, axis) and names2[-3:] == ("ms1_mono", "ms1_iso1", "ms1_iso2")
    assert values2.shape[0] == len(chrom.traces)
    # Different axes (sparse mode): no common axis and no matrix.
    e = np.zeros(0, np.float32)
    sparse = CandidateChromatogram(
        1,
        (
            Trace(
                "b3", 1.0, 1.0, 1.0, np.array([1, 2], np.float32), np.ones(2, np.float32), None, 2
            ),
            Trace(
                "y3", 1.0, 1.0, 1.0, np.array([1, 3], np.float32), np.ones(2, np.float32), None, 2
            ),
            Trace("y4", 1.0, 1.0, 1.0, e, e, None, 0),
        ),
        V2,
        Path("x"),
        None,
        (0,),
    )
    assert sparse.common_axis() is None and sparse.matrix() is None
    nothing = CandidateChromatogram(
        2, (Trace("y4", 1.0, 1.0, 1.0, e, e, None, 0),), V2, Path("x"), None, (0,)
    )
    assert nothing.common_axis() is None


def test_source_is_memoised_and_read_misses(open_fixture):
    rs = open_fixture("single")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    assert ChromatogramSource.for_run(rs, run) is src
    assert src.read(0) is None and src.read(10**9) is None and src.read(-1) is None
    assert src.band_of(4) is None
    assert src.describe()["tables"][0]["layout"] == 2


def test_experiment_runs_read_their_own_tables(open_fixture):
    rs = open_fixture("experiment")
    for run in rs.runs:
        src = ChromatogramSource.for_run(rs, run)
        assert src.tables[0].path.parent == run.root
        cid = int(src.candidate_ids()[0])
        chrom = src.read(cid)
        assert chrom.table.parent == run.root and len(chrom.traces) == 9
    a, b = (ChromatogramSource.for_run(rs, r) for r in rs.runs)
    assert a is not b


# --------------------------------------------------------------------------- memory, large groups


@pytest.mark.parametrize("layout", [1, 2])
def test_kept_chromatograms_do_not_hold_row_groups(fixture_dir, tmp_path, layout):
    """A kept result holds only its own points, never the row group it was read from.

    The table is one row group of 15,000 rows with 40 points each (about 2.4 MB per list
    column). Before the fix, 40 kept candidates held the whole group's list buffers
    (4.8 MB in layout 1, the 2.4 MB rt_axis buffer in layout 2) after the cache was cleared.
    """
    cids, dense = synthetic_rows(1500, 10, 40, seed=5)
    table = v1_table(cids, dense) if layout == 1 else v2_table(cids, dense)
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, table, layout)
    rs = open_results(run_dir)
    src = ChromatogramSource(rs, rs.runs[0])
    assert src.read(1) is not None  # builds the candidate index
    ROW_GROUP_CACHE.clear()
    gc.collect()
    base = pa.total_allocated_bytes()
    kept = [src.read(c) for c in range(1, 41)]
    assert ROW_GROUP_CACHE.nbytes > 2_000_000  # read whole through the row-group cache
    ROW_GROUP_CACHE.clear()
    gc.collect()
    held = pa.total_allocated_bytes() - base
    assert held < 64 * 1024, f"{held} bytes of Arrow memory held by 40 kept results"
    rows = iter(dense)
    for chrom in kept:
        for trace in chrom.traces:
            rt, it = next(rows)
            assert same32(trace.rt, rt) and same32(trace.intensity, it)
            for values in (trace.rt, trace.intensity):
                owner = values if values.base is None else values.base
                assert isinstance(owner, np.ndarray) and owner.base is None  # numpy memory


def test_row_group_bytes_estimate(open_fixture):
    rs = open_fixture("chrom_v1")
    handle = rs.runs[0].artifacts["chromatograms"].parquet()
    full = row_group_bytes(handle, 0, [*SCALARS, "rt", "intensity"])
    assert full >= handle.metadata().row_group(0).total_byte_size
    assert 0 < row_group_bytes(handle, 0, SCALARS) < full
    src = ChromatogramSource(rs, rs.runs[0])
    assert src._row_group_limit() == ROW_GROUP_CACHE.max_bytes // 4
    assert not src._is_large(src.tables[0], handle, 0)


@pytest.mark.parametrize(("name", "every"), [("single", 1), ("chrom_v1", 1), ("chrom_v2_rg1", 142)])
def test_large_row_groups_are_read_by_a_filtered_query(open_fixture, fixture_dir, name, every):
    """With a limit of 0 bytes every row group counts as too large to read whole: each
    candidate's rows come from a DuckDB query, bit for bit equal to the layout 1 table,
    and nothing enters the row-group cache. (A query on the 2,556-row-group file takes
    about 90 ms, mostly footer parsing, so only every 142nd candidate is read there.)"""
    oracle = v1_oracle(fixture_dir("chrom_v1") / "chromatograms.parquet")
    rs = open_fixture(name)
    src = ChromatogramSource(rs, rs.runs[0])
    src.max_row_group_bytes = 0
    ROW_GROUP_CACHE.clear()
    checked = 0
    for cid in list(oracle)[::every]:
        chrom = src.read(cid)
        assert chrom is not None and len(chrom.traces) == len(oracle[cid])
        for trace, row in zip(chrom.traces, oracle[cid], strict=True):
            assert_trace_equals_row(trace, row)
        checked += 1
        if name == "chrom_v2_rg1":
            assert len(chrom.row_groups) == 9  # nine one-row groups, nine queries
    assert checked >= 2
    assert ROW_GROUP_CACHE.nbytes == 0


@pytest.mark.parametrize("with_im", [False, True])
@pytest.mark.parametrize("layout", [1, 2])
def test_filtered_query_decodes_the_engine_fixture(fixture_dir, tmp_path, with_im, layout):
    """Seams every 3 rows, absent rows, all-zero traces, -0.0 and NaN: the query path and
    the whole-row-group path give the same bits, ion mobility included."""
    cids, dense, ims = engine_fixture()
    if layout == 1:
        table = v1_table(cids, dense, ims if with_im else None)
    else:
        table = v2_table(cids, dense, 3, ims if with_im else None)
    version = {(1, False): 1, (2, False): 2, (1, True): 3, (2, True): 4}[(layout, with_im)]
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, table, version, row_group_size=3)
    rs = open_results(run_dir, allow_unreleased=with_im)
    whole = ChromatogramSource(rs, rs.runs[0])
    query = ChromatogramSource(rs, rs.runs[0])
    query.max_row_group_bytes = 0
    for cid in sorted(set(cids)):
        a, b = whole.read(cid), query.read(cid)
        assert a.row_groups == b.row_groups
        assert_same_chromatogram(a, b)
        for x, y in zip(a.traces, b.traces, strict=True):
            assert (x.im is None) == (y.im is None) == (not with_im)
            if with_im:
                assert same32(x.im, y.im)
    first = [k for k, c in enumerate(cids) if c == 1]
    got = query.read(1)
    for trace, k in zip(got.traces, first, strict=True):
        assert same32(trace.intensity, dense[k][1])  # -0.0 and NaN kept by bits


def test_filtered_query_refuses_an_index_that_does_not_describe_the_file(open_fixture):
    rs = open_fixture("single")
    src = ChromatogramSource(rs, rs.runs[0])
    src.max_row_group_bytes = 0
    table = src.tables[0]
    index = src._index(table)
    src._indexes[id(table)] = CandidateIndex(
        index.ids,
        index.starts + 1,
        index.stops + 1,
        index.rg_offsets,
        file_sorted=index.file_sorted,
        contiguous=index.contiguous,
    )
    with pytest.raises(InconsistentData, match="does not describe this file"):
        src.read(int(index.ids[0]))


# --------------------------------------------------------------------------- engine fixture


@pytest.mark.parametrize("rows_per_group", [1, 2, 3, 4, 5, 7, 64])
@pytest.mark.parametrize("with_im", [False, True])
def test_encoded_fixture_decodes_at_every_row_group_size(tmp_path, rows_per_group, with_im):
    cids, dense, ims = engine_fixture()
    table = v2_table(cids, dense, rows_per_group, ims if with_im else None)
    path = tmp_path / "chrom.parquet"
    pq.write_table(table, path, row_group_size=rows_per_group)
    pf = pq.ParquetFile(path)
    layout = chromatogram_layout(pf.schema_arrow)
    assert layout == (V4 if with_im else V2)
    decoded = []
    for g in range(pf.metadata.num_row_groups):
        group = pf.read_row_group(g)
        decoded.extend(decode_rows(group, layout))
        # A read that starts at a candidate's first row in the group is exact too.
        ids = group.column("candidate_id").to_pylist()
        for start in [k for k in range(len(ids)) if k == 0 or ids[k] != ids[k - 1]]:
            part = decode_rows(group.slice(start), layout)
            assert all(
                same32(x[0], y[0]) and same32(x[1], y[1])
                for x, y in zip(part, decoded[len(decoded) - len(ids) + start :], strict=True)
            )
    assert len(decoded) == len(dense)
    for k, ((rt, it, im), (want_rt, want_it)) in enumerate(zip(decoded, dense, strict=True)):
        assert same32(rt, want_rt) and same32(it, want_it), k
        if with_im:
            assert same32(im, ims[k] if len(want_rt) else np.zeros(0, np.float32)), k
        else:
            assert im is None


def test_v3_layout_decodes_im_as_stored():
    cids, dense, ims = engine_fixture()
    cols = scalar_arrays(cids)
    cols["rt"] = lists([r for r, _ in dense])
    cols["intensity"] = lists([i for _, i in dense])
    cols["im"] = lists(ims)
    table = pa.table(cols)
    assert chromatogram_layout(table.schema) == V3
    for (rt, it, im), (want_rt, want_it), want_im in zip(
        decode_rows(table, V3), dense, ims, strict=True
    ):
        assert same32(rt, want_rt) and same32(it, want_it) and same32(im, want_im)
    cols["im"] = lists([m[:-1] if len(m) else m for m in ims])
    with pytest.raises(InconsistentData, match="ion-mobility"):
        decode_rows(pa.table(cols), V3)


def test_v4_im_length_must_match_trimmed_intensity():
    cids, dense, ims = engine_fixture()
    table = v2_table(cids, dense, 1 << 16, ims)
    trimmed = table.column("im_trimmed").to_pylist()
    k = next(i for i, v in enumerate(trimmed) if len(v) > 1)
    trimmed[k] = trimmed[k][:-1]
    bad = table.set_column(
        table.schema.get_field_index("im_trimmed"), "im_trimmed", pa.array(trimmed, LIST)
    )
    with pytest.raises(InconsistentData, match="ion-mobility"):
        decode_rows(bad, V4)


# --------------------------------------------------------------------------- refusals


def test_mixed_layout_columns_are_refused(fixture_dir, tmp_path):
    t = pq.read_table(fixture_dir("single") / "chromatograms.parquet")
    v1 = pq.read_table(fixture_dir("chrom_v1") / "chromatograms.parquet")
    mixed = t.append_column("rt", v1.column("rt")).append_column(
        "intensity", v1.column("intensity")
    )
    with pytest.raises(LayoutError):
        chromatogram_layout(mixed.schema)
    partial = t.drop_columns(["trace_offset"])
    with pytest.raises(LayoutError):
        chromatogram_layout(partial.schema)
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, mixed, 1)
    rs = open_results(run_dir)
    with pytest.raises(LayoutError, match="neither chromatogram layout"):
        ChromatogramSource.for_run(rs, rs.runs[0])
    # Without a recorded version, discovery infers one from the columns and records the
    # refusal on the artifact; require() raises it.
    replace_chromatograms(run_dir, mixed, None)
    rs = open_results(run_dir)
    with pytest.raises(SchemaVersionError, match="neither chromatogram layout"):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_borrowed_axis_of_another_candidate_is_refused():
    t = raw_v2([1, 2], [[1, 2], []], [[5, 6], [7]], [0, 0], [2, 2])
    with pytest.raises(InconsistentData, match="no earlier row of the candidate"):
        decode_rows(t, V2)


def test_orphan_row_is_refused():
    t = raw_v2([1, 1], [[], [1, 2]], [[5], [7]], [0, 0], [2, 2])
    with pytest.raises(InconsistentData, match=r"row 0 \(candidate_id 1\).*no retention-time axis"):
        decode_rows(t, V2)


def test_axis_length_must_equal_trace_len():
    t = raw_v2([1], [[1, 2, 3]], [[5]], [0], [2])
    with pytest.raises(InconsistentData, match="3 retention-time points but a trace_len of 2"):
        decode_rows(t, V2)
    inherited = raw_v2([1, 1], [[1, 2], []], [[5], [7]], [0, 0], [2, 3])
    with pytest.raises(InconsistentData, match=r"row 1 .*trace_len of 3 but the candidate's"):
        decode_rows(inherited, V2)


def test_values_past_trace_len_are_refused():
    t = raw_v2([1], [[1, 2, 3]], [[5, 6]], [2], [3])
    with pytest.raises(
        InconsistentData, match="stores 2 intensity values at offset 2 of a 3-point"
    ):
        decode_rows(t, V2)


def test_absent_row_must_store_nothing():
    t = raw_v2([1], [[]], [[5]], [0], [0])
    with pytest.raises(InconsistentData, match="trace_len 0"):
        decode_rows(t, V2)
    t = raw_v2([1], [[]], [[]], [1], [0])
    with pytest.raises(InconsistentData, match="trace_len 0"):
        decode_rows(t, V2)


def test_first_failing_row_is_reported():
    # Row 1 borrows another candidate's axis; row 2 overflows. The engine stops at row 1.
    t = raw_v2([1, 2, 2], [[1, 2], [], []], [[5], [7], [1, 2, 3]], [0, 0, 0], [2, 2, 2])
    with pytest.raises(InconsistentData, match=r"row 1 \(candidate_id 2\)"):
        decode_rows(t, V2)


def test_null_trace_len_and_v1_length_mismatch_are_refused():
    t = raw_v2([1], [[1, 2]], [[5]], [0], [2])
    nulls = t.set_column(
        t.schema.get_field_index("trace_len"), "trace_len", pa.array([None], pa.uint32())
    )
    with pytest.raises(InconsistentData, match="null"):
        decode_rows(nulls, V2)
    cols = scalar_arrays([1])
    cols["rt"] = lists([[1, 2]])
    cols["intensity"] = lists([[1]])
    with pytest.raises(InconsistentData, match="2 retention-time points but 1 intensity"):
        decode_rows(pa.table(cols), V1)


def test_null_lists_are_empty():
    cols = scalar_arrays([1, 1])
    cols["rt_axis"] = pa.array([[1.0, 2.0], None], LIST)
    cols["intensity_trimmed"] = pa.array([None, [4.0]], LIST)
    cols["trace_offset"] = pa.array([0, 1], pa.uint32())
    cols["trace_len"] = pa.array([2, 2], pa.uint32())
    rows = decode_rows(pa.table(cols), V2)
    assert same32(rows[0][1], np.zeros(2, np.float32))
    assert same32(rows[1][0], np.array([1, 2], np.float32))
    assert same32(rows[1][1], np.array([0, 4], np.float32))


def test_slice_inside_a_candidate_is_refused(fixture_dir):
    pf = pq.ParquetFile(fixture_dir("single") / "chromatograms.parquet")
    group = pf.read_row_group(0)
    tl = group.column("trace_len").to_numpy()
    axis = np.array([len(v) for v in group.column("rt_axis").to_pylist()])
    # A row that inherits its axis from the row before it: start the read there.
    k = int(np.flatnonzero((axis == 0) & (tl > 0))[0])
    with pytest.raises(InconsistentData, match="no earlier row of the candidate"):
        decode_rows(group.slice(k), V2)


def test_unreleased_im_versions_need_opt_in(fixture_dir, tmp_path):
    cids, dense, ims = engine_fixture()
    table = v2_table(cids, dense, 1 << 16, ims)
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, table, 4)
    with pytest.raises(SchemaVersionError, match="unreleased"):
        rs = open_results(run_dir)
        ChromatogramSource.for_run(rs, rs.runs[0])
    rs = open_results(run_dir, allow_unreleased=True)
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert src.tables[0].layout == V4
    chrom = src.read(1)
    want = [k for k, c in enumerate(cids) if c == 1]
    assert [t.trace_len for t in chrom.traces] == [len(dense[k][0]) for k in want]
    for t, k in zip(chrom.traces, want, strict=True):
        assert same32(t.intensity, dense[k][1])
        assert same32(t.im, ims[k] if len(dense[k][0]) else np.zeros(0, np.float32))


def test_v3_table_through_a_source(fixture_dir, tmp_path):
    v1 = pq.read_table(fixture_dir("chrom_v1") / "chromatograms.parquet")
    intensity = list_rows(v1.column("intensity"))
    im = [
        np.where(bits32(v) != 0, np.float32(0.9), np.float32(0.0)).astype(np.float32)
        for v in intensity
    ]
    v3 = v1.append_column("im", lists(im))
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, v3, 3)
    rs = open_results(run_dir, allow_unreleased=True)
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert src.tables[0].layout == V3
    cid = int(src.candidate_ids()[0])
    chrom = src.read(cid)
    first = v1.column("candidate_id").to_pylist().index(cid)
    for j, t in enumerate(chrom.traces):
        assert t.im is not None and same32(t.im, im[first + j])


def test_recorded_version_must_match_the_columns(fixture_dir, tmp_path):
    table = pq.read_table(fixture_dir("single") / "chromatograms.parquet")
    run_dir = copy_fixture(fixture_dir, "chrom_v1", tmp_path)
    replace_chromatograms(run_dir, table, 1)  # layout 2 columns recorded as version 1
    rs = open_results(run_dir)
    with pytest.raises(LayoutError, match="version 2 layout"):
        ChromatogramSource.for_run(rs, rs.runs[0])


# --------------------------------------------------------------------------- grouped runs


GROUPED_PAIRS = [
    ("grouped", "grouped_pool"),
    ("ovl_bp", "ovl_bp_pool"),
    ("ovl_rg50", "ovl_rg50_pool"),
    ("ovl128", "ovl128_pool"),
]


@pytest.mark.parametrize(("bands", "pooled"), GROUPED_PAIRS)
def test_band_tables_minus_losers_equal_the_pooled_table(open_fixture, bands, pooled):
    rs_b, rs_p = open_fixture(bands), open_fixture(pooled)
    run_b, run_p = rs_b.runs[0], rs_p.runs[0]
    src_b = ChromatogramSource.for_run(rs_b, run_b)
    src_p = ChromatogramSource.for_run(rs_p, run_p)
    assert src_b.per_band and not src_p.per_band and src_b.notice is None
    assert len(src_p.tables) == 1 and src_p.tables[0].path == run_p.root / "chromatograms.parquet"

    # The drop sets are the loser file's rows, by position in its band list.
    loser_file = run_b.root / "groups" / "overlap_losers.parquet"
    losers = pq.read_table(loser_file).to_pylist()
    records = json.loads(pq.read_metadata(loser_file).metadata[LOSERS_BAND_TABLES_KEY])
    assert [t.position for t in src_b.tables] == list(range(len(records)))
    assert [f"{t.band}/chromatograms.parquet" for t in src_b.tables] == [r["name"] for r in records]
    for i, table in enumerate(src_b.tables):
        assert table.drop == {r["candidate_id"] for r in losers if r["band"] == i}

    ids = src_b.candidate_ids()
    assert ids.tolist() == src_p.candidate_ids().tolist()
    held = [
        (t.band, band_table_ids(run_b.root, f"{t.band}/chromatograms.parquet"), t.drop)
        for t in src_b.tables
    ]
    for cid in ids.tolist():
        a, b = src_b.read(cid), src_p.read(cid)
        assert_same_chromatogram(a, b)
        yielding = [band for band, table_ids, drop in held if cid in table_ids and cid not in drop]
        assert len(yielding) == 1, cid
        assert a.band == yielding[0] == src_b.band_of(cid)
        assert src_p.band_of(cid) is None and b.band is None
    total = sum(len(src_b.read(c).traces) for c in ids.tolist())
    assert total == pq.read_metadata(run_p.root / "chromatograms.parquet").num_rows


def test_ovl128_band_positions_are_not_gnn(open_fixture):
    rs = open_fixture("ovl128")
    run = rs.runs[0]
    src = ChromatogramSource.for_run(rs, run)
    assert len(src.tables) == 120 and run.grouped.skipped == [83, 114, 115, 122, 123, 124, 125, 126]
    by_position = {t.position: t for t in src.tables}
    assert by_position[83].band == "g84" and by_position[84].band == "g85"
    assert 3229 in by_position[84].drop
    assert src.band_of(3229) != "g85"
    chrom = src.read(3229)
    assert chrom is not None and chrom.band == src.band_of(3229)
    # The band table g85 holds the loser; read from it only as a labelled diagnostic.
    assert src.read_table(by_position[84], 3229) is not None
    assert any(t.band == "g100" for t in src.tables)
    empty = [t.band for t in src.tables if t.artifact.parquet().num_rows == 0]
    assert empty == ["g01", "g51"]
    mismatched = sum(1 for t in src.tables for _ in t.drop if t.band != f"g{t.position:02d}")
    assert mismatched == 32


def test_disjoint_bands_have_empty_drops(open_fixture):
    rs = open_fixture("grouped")
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert [t.band for t in src.tables] == ["g00", "g01", "g02"]
    assert all(not t.drop for t in src.tables)
    assert src.band_of(4) == "g00" and src.band_of(1806) == "g01" and src.band_of(3799) == "g02"
    assert src.band_tables == src.tables


def test_pooled_run_band_tables_are_diagnostics(open_fixture):
    rs = open_fixture("ovl_bp_pool")
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert [t.band for t in src.band_tables] == ["g00", "g01", "g02"]
    assert all(t.position is None and not t.drop for t in src.band_tables)
    assert src.band_tables[0].path.parent.name == "g00"
    # 1740 is extracted in g00 and g01; the pooled table holds one band's copy.
    copies = [src.read_table(t, 1740) for t in src.band_tables]
    assert sum(c is not None for c in copies) == 2


def _grouped_copy(fixture_dir, name, tmp_path) -> Path:
    return copy_fixture(fixture_dir, name, tmp_path)


def test_loser_footer_with_wrong_rows_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)

    def wrong_rows(recs):
        recs[1]["rows"] += 1
        return recs

    rewrite_losers(run_dir, records=wrong_rows)
    rs = open_results(run_dir)
    with pytest.raises(
        InconsistentData, match=re.escape("band 1 is g01/chromatograms.parquet with 1072 rows")
    ):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_loser_footer_with_wrong_hash_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)

    def wrong_hash(recs):
        recs[2]["content_hash"] = "0" * 64
        return recs

    rewrite_losers(run_dir, records=wrong_hash)
    rs = open_results(run_dir)
    with pytest.raises(InconsistentData, match=r"band 2 .* content hash"):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_loser_footer_with_swapped_tables_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)

    def swapped(recs):
        recs[0], recs[1] = recs[1], recs[0]
        return recs

    rewrite_losers(run_dir, records=swapped)
    rs = open_results(run_dir)
    with pytest.raises(InconsistentData, match=r"not in pool order .*: g01, g00, g02"):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_missing_band_table_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)
    os.remove(run_dir / "groups" / "g01" / "chromatograms.parquet")
    rs = open_results(run_dir)
    with pytest.raises(
        InconsistentData, match=re.escape("g01/chromatograms.parquet, which does not exist")
    ):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_loser_band_outside_the_list_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)
    rows = pa.table(
        {"band": pa.array([3], pa.uint32()), "candidate_id": pa.array([1740], pa.uint32())}
    )
    rewrite_losers(run_dir, rows=rows)
    rs = open_results(run_dir)
    with pytest.raises(InconsistentData, match="names band 3"):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_loser_file_without_footer_is_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)
    path = run_dir / "groups" / "overlap_losers.parquet"
    pq.write_table(pq.read_table(path).replace_schema_metadata({}), path)
    rs = open_results(run_dir)
    with pytest.raises(InconsistentData, match="does not name the band tables"):
        ChromatogramSource.for_run(rs, rs.runs[0])


def test_two_tables_yielding_rows_are_refused(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "ovl_bp", tmp_path)
    empty = pa.table({"band": pa.array([], pa.uint32()), "candidate_id": pa.array([], pa.uint32())})
    rewrite_losers(run_dir, rows=empty)
    rs = open_results(run_dir)
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert all(not t.drop for t in src.tables)
    assert src.read(4) is not None
    with pytest.raises(
        InconsistentData, match="candidate_id 1740 has chromatogram rows in 2 tables"
    ):
        src.read(1740)
    with pytest.raises(InconsistentData):
        src.band_of(1740)


def test_zero_row_loser_file_is_valid(open_fixture):
    rs = open_fixture("grouped")
    losers = rs.runs[0].grouped.losers
    md = pq.read_metadata(losers.path)
    assert md.num_rows == 0 and md.num_row_groups == 0
    assert ChromatogramSource.for_run(rs, rs.runs[0]).candidate_ids().size == 305


def test_band_tables_without_loser_file(fixture_dir, open_fixture, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "grouped", tmp_path)
    os.remove(run_dir / "groups" / "overlap_losers.parquet")
    os.remove(run_dir / "groups" / "overlap_losers.parquet.report.json")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    del manifest["artifacts"]["overlap_losers"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rs = open_results(run_dir)
    src = ChromatogramSource.for_run(rs, rs.runs[0])
    assert src.per_band and src.notice and "without dropping overlap losers" in src.notice
    assert [t.position for t in src.tables] == [None, None, None]
    pooled = ChromatogramSource.for_run(
        open_fixture("grouped_pool"), open_fixture("grouped_pool").runs[0]
    )
    for cid in pooled.candidate_ids().tolist():
        assert_same_chromatogram(src.read(cid), pooled.read(cid))


def test_recorded_loser_file_that_is_missing(fixture_dir, tmp_path):
    run_dir = _grouped_copy(fixture_dir, "grouped", tmp_path)
    os.remove(run_dir / "groups" / "overlap_losers.parquet")
    rs = open_results(run_dir)
    with pytest.raises(ArtifactNotFound, match="overlap_losers"):
        ChromatogramSource.for_run(rs, rs.runs[0])


# --------------------------------------------------------------------------- real data


@pytest.mark.real_data
def test_real_single_random_candidates(real_single):
    rs = open_results(real_single)
    run = rs.runs[0]
    t0 = time.perf_counter()
    src = ChromatogramSource.for_run(rs, run)
    ids = src.candidate_ids()
    t_index = time.perf_counter() - t0
    ex = pq.read_table(
        run.artifacts["psms_extracted"].path,
        columns=["candidate_id", "peak_rank", "apex_rt", "n_predicted_fragments"],
    )
    cid = ex.column("candidate_id").to_numpy()
    apex = ex.column("apex_rt").to_numpy()
    n_frag = ex.column("n_predicted_fragments").to_numpy()
    sample = np.random.default_rng(20260930).choice(np.unique(cid), 200, replace=False)
    ROW_GROUP_CACHE.clear()
    cold, warm = [], []
    for c in sample.tolist():
        t0 = time.perf_counter()
        chrom = src.read(c)
        cold.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        src.read(c)
        warm.append(time.perf_counter() - t0)
        rows = np.flatnonzero(cid == c)
        assert len(chrom.traces) == int(n_frag[rows[0]]) + 3
        assert [t.frag_name for t in chrom.traces[-3:]] == ["ms1_mono", "ms1_iso1", "ms1_iso2"]
        axis = chrom.common_axis()
        assert axis is not None
        for k in rows.tolist():  # every peak rank shares the candidate's axis
            assert int((bits32(axis) == bits32(np.float32(apex[k]))).sum()) == 1

    def ms(a: list[float]) -> str:
        return (
            f"median {1e3 * np.median(a):.1f} ms, p90 {1e3 * np.percentile(a, 90):.1f} ms, "
            f"max {1e3 * np.max(a):.1f} ms"
        )

    print(
        f"\n{len(ids)} candidates; source and index {t_index:.2f} s; "
        f"cold (row group read) {ms(cold)}; warm (row group cached) {ms(warm)}"
    )
