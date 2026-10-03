"""Run QC data (data.qc): scan signals, peaks per spectrum, scan rate, windows and the
accepted identifications of a run.

Every reference is computed independently here, with pyarrow, numpy and plain Python
(no DuckDB, no code of the module under test).
"""

from __future__ import annotations

import itertools
import re
from types import SimpleNamespace

import duckdb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import Run, qc
from mumdia_viewer.data.counts import per_run_counts, unit_counts

SPECTRA_FIXTURES = ["single", "experiment", "mbr", "topk"]
SCORED_FIXTURES = ["single", "experiment", "mbr", "topk", "grouped"]


def _runs(rs):
    return list(rs.runs)


def _peaks(path):
    """Per-scan sum, max, m/z of the first max and length, with pyarrow and numpy."""
    t = pq.read_table(path, columns=["mz", "intensity"])
    sums, maxs, mzs, ns = [], [], [], []
    for mz, it in zip(t.column("mz").to_pylist(), t.column("intensity").to_pylist(), strict=True):
        it = np.asarray(it or [], dtype=np.float32)
        mz = np.asarray(mz or [], dtype=np.float32)
        ns.append(it.size)
        sums.append(float(np.sum(it.astype(np.float64))))
        if it.size:
            k = int(np.argmax(it))
            maxs.append(float(it[k]))
            mzs.append(float(mz[k]))
        else:
            maxs.append(0.0)
            mzs.append(np.nan)
    return np.array(sums), np.array(maxs), np.array(mzs), np.array(ns)


# --------------------------------------------------------------------------- scan signals


@pytest.mark.parametrize("name", SPECTRA_FIXTURES)
@pytest.mark.parametrize("level", [1, 2])
def test_scan_signals_equal_an_independent_sum(open_fixture, name, level):
    rs = open_fixture(name)
    for run in _runs(rs):
        kind = "spectra_ms1" if level == 1 else "spectra_ms2"
        path = run.artifact(kind).path
        sums, maxs, mzs, ns = _peaks(path)
        sig = qc.scan_signals(rs, run, level)
        rt = pq.read_table(path, columns=["rt_seconds"]).column(0).to_numpy()
        assert sig.n == rt.size
        np.testing.assert_array_equal(sig.rt, rt)
        np.testing.assert_allclose(sig.tic, sums, rtol=1e-12)
        np.testing.assert_array_equal(sig.base_peak, maxs.astype(np.float32))
        np.testing.assert_array_equal(sig.base_peak_mz, mzs.astype(np.float32))
        np.testing.assert_array_equal(sig.n_peaks, ns)


def test_scan_signals_are_cached_by_content(open_fixture):
    rs = open_fixture("single")
    first = qc.scan_signals(rs, None, 2)
    again = qc.scan_signals(rs, None, 2)
    assert again.cached
    assert again.identity == rs.runs[0].artifact("spectra_ms2").identity()
    assert qc.signals_cached(rs, None, 2)
    np.testing.assert_array_equal(first.tic, again.tic)


def test_scan_signals_of_a_run_without_spectra(open_fixture):
    rs = open_fixture("chrom_v1")
    assert not qc.signals_cached(rs, None, 2)
    with pytest.raises(Exception, match="spectra_ms2"):
        qc.scan_signals(rs, None, 2)


def test_envelope_keeps_every_row_or_the_extremes_of_each_bin():
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 100, 50_000))
    y = rng.normal(size=x.size)
    every = qc.envelope_rows(x, y, 10.0, 20.0, max_points=10_000)
    inside = np.flatnonzero((x >= 10.0) & (x <= 20.0))
    np.testing.assert_array_equal(every, inside)
    rows = qc.envelope_rows(x, y, 0.0, 100.0, n_bins=200, max_points=1000)
    assert rows.size <= 4 * 200 and np.all(np.diff(x[rows]) >= 0)
    bins = np.clip(np.floor(x / 100.0 * 200).astype(int), 0, 199)
    for b in (0, 57, 199):
        members = np.flatnonzero(bins == b)
        for want in (
            members[0],
            members[-1],
            members[np.argmin(y[members])],
            members[np.argmax(y[members])],
        ):
            assert want in rows
    assert qc.envelope_rows(x, y, 200.0, 300.0).size == 0


# --------------------------------------------------------------------------- peaks per MS2


@pytest.mark.parametrize("name", ["single", "topk"])
def test_peak_counts_are_percentiles_of_the_list_lengths(open_fixture, name):
    rs = open_fixture(name)
    path = rs.runs[0].artifact("spectra_ms2").path
    n = np.array(
        pc.list_value_length(pq.read_table(path, columns=["mz"]).column(0)).to_numpy(
            zero_copy_only=False
        )
    )
    p = qc.peak_counts(rs, None)
    assert p.n_spectra == n.size and p.n_peaks_total == int(n.sum())
    for key, q in (("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)):
        assert p.percentiles[key] == int(np.percentile(n, q, method="nearest"))
    assert p.percentiles["max"] == int(n.max()) and p.percentiles["min"] == int(n.min())
    assert int(p.histogram["n"].sum()) == n.size
    # The fixtures were converted without a cap; the report says so.
    assert p.cap == 0 and p.n_at_cap is None and "No conversion cap" in p.note


def test_peak_counts_count_the_spectra_at_the_cap(open_fixture, monkeypatch):
    rs = open_fixture("single")
    path = rs.runs[0].artifact("spectra_ms2").path
    n = np.array(
        pc.list_value_length(pq.read_table(path, columns=["mz"]).column(0)).to_numpy(
            zero_copy_only=False
        )
    )
    cap = int(np.median(n))
    monkeypatch.setattr(qc, "conversion_cap", lambda rs, run: (cap, "a test value"))
    p = qc.peak_counts(rs, None)
    assert p.cap == cap
    assert p.n_at_cap == int(np.count_nonzero(n == cap))
    assert p.n_over_cap == int(np.count_nonzero(n > cap))
    assert "cannot leave" in p.note  # spectra above the cap contradict it


def test_conversion_cap_from_the_report_or_the_command_line(open_fixture):
    rs = open_fixture("single")
    cap, source = qc.conversion_cap(rs, None)
    assert cap == 0 and "report.json" in source

    def fake(args):
        return SimpleNamespace(manifest=SimpleNamespace(cli_args=args))

    r = Run(name="", index=0, root=rs.root, artifacts={}, side_files={})
    assert qc.conversion_cap(fake(["mumdia", "run", "--top-peaks-ms2", "300"]), r)[0] == 300
    assert qc.conversion_cap(fake(["mumdia", "run", "--top-peaks-ms2=150"]), r)[0] == 150
    assert qc.conversion_cap(fake(["mumdia", "run"]), r)[0] == 0
    assert qc.conversion_cap(fake([]), r)[0] is None


# --------------------------------------------------------------------------- scan rate, windows


def test_rt_bins():
    assert qc.rt_bin_width(1611) == 20 and qc.rt_bin_width(120) == 2
    assert qc.rt_bin_width(10_000) == 120
    for span in (5, 120, 1611, 7000):
        assert span / qc.rt_bin_width(span) <= qc.MAX_BINS


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_scan_rate_and_cycle_time(open_fixture, name):
    rs = open_fixture(name)
    for run in _runs(rs):
        edges = qc.rt_edges(rs, run)
        t = pq.read_table(run.artifact("spectra_ms2").path, columns=["rt_seconds", "window_id"])
        rt, win = t.column(0).to_numpy(), t.column(1).to_numpy()
        assert edges[0] == 0 and edges[-1] >= rt.max()
        df = qc.scan_rate(rs, run, edges)
        counts, _ = np.histogram(rt, bins=edges)
        np.testing.assert_array_equal(df["ms2_scans"].to_numpy(), counts)
        # Cycle: steps between consecutive scans of one window, by the bin of the later.
        steps: dict[int, list[float]] = {}
        for w in np.unique(win):
            r = np.sort(rt[win == w])
            for a, b in itertools.pairwise(r):
                k = min(int(np.searchsorted(edges, b, side="right") - 1), edges.size - 2)
                steps.setdefault(k, []).append(b - a)
        for k, values in steps.items():
            assert df["cycle_s"].iloc[k] == pytest.approx(float(np.median(values)))
        acquired = df["acquired_s"].to_numpy()
        assert acquired.sum() == pytest.approx(rt.max() - rt.min())


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_window_scheme_order_is_the_first_scan_order(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    ws = qc.window_scheme(rs, run)
    t = pq.read_table(run.artifact("spectra_ms2").path, columns=["rt_seconds", "window_id"])
    rt, win = t.column(0).to_numpy(), t.column(1).to_numpy()
    first = {int(w): float(rt[win == w].min()) for w in np.unique(win)}
    expected = sorted(first, key=lambda w: (first[w], w))
    by_order = ws.frame.sort_values("order")["window_id"].tolist()
    assert by_order == expected
    iw = pq.read_table(run.artifact("isolation_windows").path).to_pandas()
    assert ws.n_windows == len(iw)
    assert ws.mz_lo == pytest.approx(iw["lower"].min()) and ws.mz_hi == pytest.approx(
        iw["upper"].max()
    )
    srt = iw.sort_values(["lower", "upper"])
    gaps = srt["upper"].to_numpy()[:-1] - srt["lower"].to_numpy()[1:]
    assert ws.n_gaps == int((gaps < 0).sum()) and ws.n_overlaps == int((gaps > 0).sum())


# --------------------------------------------------------------------------- identifications


def _accepted(rs, run, t):
    """Accepted target rows of one run (pyarrow): label target, run_psm_q <= t."""
    table = pq.read_table(
        rs.scored.path,
        columns=[
            "source",
            "label",
            "run_psm_q",
            "apex_rt",
            "peptidoform",
            "charge",
            "candidate_id",
        ],
    )
    mask = pc.and_(pc.equal(table["label"], "target"), pc.less_equal(table["run_psm_q"], t))
    if rs.is_experiment:
        mask = pc.and_(mask, pc.equal(table["source"], run.index))
    return table.filter(mask)


@pytest.mark.parametrize("name", SCORED_FIXTURES)
@pytest.mark.parametrize("t", [0.01, 0.05])
def test_ids_across_rt_equal_a_histogram_of_apex_rt(open_fixture, name, t):
    rs = open_fixture(name)
    for run in _runs(rs):
        edges = qc.rt_edges(rs, run)
        df = qc.ids_across_rt(rs, run, t, edges)
        acc = _accepted(rs, run, t)
        apex = acc["apex_rt"].to_numpy()
        counts, _ = np.histogram(apex[np.isfinite(apex)], bins=edges)
        np.testing.assert_array_equal(df["targets"].to_numpy(), counts)
        assert df.attrs["total"] == acc.num_rows
        assert df.attrs["total"] - df.attrs["outside"] == int(counts.sum())
    # The population is the per-run count of the overview.
    if rs.is_experiment:
        per_run = per_run_counts(rs, t).set_index("run")["target_psms"]
        for run in _runs(rs):
            assert qc.ids_across_rt(rs, run, t, qc.rt_edges(rs, run)).attrs["total"] == int(
                per_run[run.label]
            )
    else:
        psm = next(c for c in unit_counts(rs, t) if c.unit == "psm")
        assert qc.ids_across_rt(rs, None, t, qc.rt_edges(rs, None)).attrs["total"] == psm.n_target


@pytest.mark.parametrize("name", ["single", "experiment"])
def test_ids_in_rt_range_are_the_rows_of_the_bin(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[-1]
    edges = qc.rt_edges(rs, run)
    acc = _accepted(rs, run, 0.01)
    apex = acc["apex_rt"].to_numpy()
    cids = acc["candidate_id"].to_numpy()
    for k in (0, len(edges) // 2, len(edges) - 2):
        lo, hi = edges[k], edges[k + 1]
        closed = k == len(edges) - 2
        rows = qc.ids_in_rt_range(rs, run, 0.01, lo, hi, closed=closed)
        inside = (apex >= lo) & ((apex <= hi) if closed else (apex < hi))
        assert sorted(rows["candidate_id"].tolist()) == sorted(cids[inside].tolist())
        assert rows.attrs["total"] == int(inside.sum())
        assert list(rows["apex_rt"]) == sorted(rows["apex_rt"])
    few = qc.ids_in_rt_range(rs, run, 0.01, 0.0, edges[-1], closed=True, limit=3)
    assert len(few) == min(3, acc.num_rows) and few.attrs["total"] == acc.num_rows


# --------------------------------------------------------------------------- distributions

_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _sequence(text: str) -> str:
    text = text[6:] if text.startswith("DECOY_") else text
    return "".join(c for c in _TAG.sub("", text) if c.isalpha())


def _missed(seq: str) -> int:
    n = 0
    for i, aa in enumerate(seq[:-1]):
        if aa in "KR" and seq[i + 1] != "P":
            n += 1
    return n


@pytest.mark.parametrize("name", SCORED_FIXTURES)
def test_distributions_equal_an_independent_count(open_fixture, name):
    rs = open_fixture(name)
    for run in _runs(rs):
        d = qc.id_distributions(rs, run, 0.01)
        acc = _accepted(rs, run, 0.01)
        peps = acc["peptidoform"].to_pylist()
        charges = acc["charge"].to_pylist()
        assert d.n == len(peps)
        assert dict(zip(d.charge["charge"], d.charge["n"], strict=True)) == {
            c: charges.count(c) for c in set(charges)
        }
        seqs = [_sequence(p) for p in peps]
        lengths = [len(s) for s in seqs]
        assert dict(zip(d.length["length"], d.length["n"], strict=True)) == {
            v: lengths.count(v) for v in set(lengths)
        }
        missed = [_missed(s) for s in seqs]
        assert dict(zip(d.missed["missed_cleavages"], d.missed["n"], strict=True)) == {
            v: missed.count(v) for v in set(missed)
        }
        # Residue modifications, counted from the text.
        for r in d.mods.to_dict("records"):
            if len(r["site"]) != 1:
                continue
            pattern = re.compile(re.escape(r["site"]) + r"\[" + re.escape(r["tag"]) + r"\]")
            carriers = [p for p in peps if pattern.search(p)]
            assert r["psms"] == len(carriers)
            assert r["sites"] == sum(len(pattern.findall(p)) for p in peps)
            assert r["with_site"] == sum(1 for s in seqs if r["site"] in s)
        unmodified = sum(1 for p in peps if not _TAG.search(p))
        assert d.n_unmodified == unmodified


def test_missed_cleavages_rule():
    assert qc.missed_cleavages("PEPTIDEK") == 0
    assert qc.missed_cleavages("PEPKTIDER") == 1
    assert qc.missed_cleavages("PEPKPTIDER") == 0  # K before P
    assert qc.missed_cleavages("KKAR") == 2
    assert qc.missed_cleavages("AKP") == 0
    assert qc.missed_cleavages("RPKR") == 1


def test_missed_cleavages_in_sql_equal_the_python_rule():
    rng = np.random.default_rng(3)
    seqs = ["".join(rng.choice(list("KRPAG"), size=rng.integers(1, 12))) for _ in range(500)]
    seqs += ["K", "R", "KP", "PK", "RKPP", "KRP", "KPRP"]
    con = duckdb.connect()
    expr = qc._missed_sql("s")
    got = con.execute(
        f"SELECT s, {expr} FROM (SELECT unnest(?::VARCHAR[]) AS s)", [seqs]
    ).fetchall()
    for s, n in got:
        assert n == _missed(s) == qc.missed_cleavages(s), s


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PEPTIDE", []),
        ("PEM[Oxidation]TIDE", [("Oxidation", "M")]),
        ("C[Carbamidomethyl]PEM(Oxidation)K", [("Carbamidomethyl", "C"), ("Oxidation", "M")]),
        ("[Acetyl]-PEPTIDE", [("Acetyl", "N-term")]),
        ("PEPTIDE-[Amidated]", [("Amidated", "C-term")]),
        ("DECOY_PEM[Oxidation]K", [("Oxidation", "M")]),
        ("PEM[Oxidation][Phospho]K", [("Oxidation", "M"), ("Phospho", "M")]),
        ("[Acetyl]PEPTIDE", [("Acetyl", "N-term")]),
        ("PEM[UNIMOD:35]K", [("UNIMOD:35", "M")]),
    ],
)
def test_modification_sites(text, expected):
    assert qc.modification_sites(text) == expected


@pytest.mark.parametrize("name", SCORED_FIXTURES)
def test_modification_fast_path_equals_the_general_parser(open_fixture, name):
    rs = open_fixture(name)
    peps = set(pq.read_table(rs.scored.path, columns=["peptidoform"]).column(0).to_pylist())
    for p in peps:
        # A leading "[x]-" tag forces the general parser; drop it from its answer.
        slow = qc.modification_sites("[x]-" + p if not p.startswith("DECOY_") else p)
        if not p.startswith("DECOY_"):
            slow = slow[1:]
        assert qc.modification_sites(p) == slow, p


def test_population_label_names_the_unit_and_the_column(open_fixture):
    single = qc.population_label(open_fixture("single"), None, 0.01)
    assert "run_psm_q <= 0.01" in single and "equals q_value" in single
    exp = open_fixture("experiment")
    text = qc.population_label(exp, exp.runs[1].name, 0.05)
    assert f"run {exp.runs[1].label}" in text and "experiment-wide" in text
    mbr = open_fixture("mbr")
    assert "match-between-runs" in qc.population_label(mbr, mbr.runs[0].name, 0.01)


def test_acquisition_summary(open_fixture):
    rs = open_fixture("single")
    a = qc.acquisition(rs, None)
    ms1 = pq.read_metadata(rs.runs[0].artifact("spectra_ms1").path).num_rows
    ms2 = pq.read_metadata(rs.runs[0].artifact("spectra_ms2").path).num_rows
    assert (a.n_ms1, a.n_ms2) == (ms1, ms2)
    assert a.mzml and a.mzml.endswith(".mzML")
    assert a.n_windows == qc.window_scheme(rs, None).n_windows
    none = qc.acquisition(open_fixture("chrom_v1"), None)
    assert none.n_ms2 is None and none.notes


def test_counts_by_run(open_fixture):
    rs = open_fixture("experiment")
    got = qc.counts_by_run(rs, 0.01)
    per_run = per_run_counts(rs, 0.01).set_index("run")["target_psms"]
    assert got == {r.label: int(per_run[r.label]) for r in rs.runs}
