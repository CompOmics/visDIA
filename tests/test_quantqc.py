"""Quant QC data (data.quantqc), each checked against an independent computation.

The matrices are rebuilt from the parquet files with pandas, the summaries (percentiles,
histograms, missing values, CVs) with numpy on those values, and the accepted
identifications from the scored table. Synthetic copies in ``tmp_path`` cover what no
fixture shows: a null quantity.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data import quantqc as Q
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.manifest import InputRecord
from mumdia_viewer.data.quant import quant_status_breakdown

FIXTURES = ["single", "experiment", "mbr"]


def _read(path, columns=None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def _positive(values: pd.Series) -> pd.Series:
    v = values.astype("float64")
    return v.where(np.isfinite(v) & (v > 0))


def _brute_quant(rs, level: str) -> pd.DataFrame:
    """Keys by run names from each run's own table, with pandas."""
    frames = []
    for run in rs.runs:
        if level == "precursor":
            df = _read(run.artifact("peptide_quant").path)
            df["q"] = _positive(df["quantity"])
            g = df.groupby(["peptidoform", "charge"], dropna=False)["q"].max()
        else:
            df = _read(run.artifact("protein_group_quant").path)
            df["q"] = _positive(df["quantity"])
            g = df.groupby("protein_group", dropna=False)["q"].max()
        frames.append(g.rename(run.name))
    return pd.concat(frames, axis=1)


def _brute_lfq(rs, level: str) -> pd.DataFrame:
    name = "lfq_maxlfq.parquet" if level == "protein" else "lfq_maxlfq.parquet.precursor.parquet"
    df = _read(rs.root / name)
    df["q"] = df["quantity"].where(df["quantity"] > 0)
    keys = ["protein_group"] if level == "protein" else ["group", "charge"]
    wide = df.pivot_table(index=keys, columns="run", values="q", aggfunc="max", dropna=False)
    wide.columns = [rs.run(int(c)).name for c in wide.columns]
    if level == "precursor":
        wide.index = wide.index.set_names(["peptidoform", "charge"])
    return wide


def _as_frame(m: Q.QuantMatrix) -> pd.DataFrame:
    keys = list(Q.KEY_COLUMNS[m.level])
    return m.frame().set_index(keys)[list(m.runs)]


def _same(got: pd.DataFrame, want: pd.DataFrame) -> None:
    want = want.reindex(got.index)
    assert set(got.index) == set(want.index)
    a, b = got.to_numpy(dtype="float64"), want[list(got.columns)].to_numpy(dtype="float64")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- conditions


def _stub(names: list[str | None], experiment: bool = True):
    inputs = {}
    for i, name in enumerate(names):
        if name is not None:
            key = f"mzml[{i}]" if experiment else "mzml"
            inputs[key] = InputRecord(key=key, path=name, bytes=None, content_hash=None)
    runs = [
        SimpleNamespace(
            name=f"r{i}" if experiment else "", index=i, label=f"r{i}" if experiment else "run"
        )
        for i in range(len(names))
    ]
    return SimpleNamespace(
        manifest=SimpleNamespace(inputs=inputs), runs=runs, is_experiment=experiment
    )


def test_run_files_and_suggestions_of_the_fixtures(open_fixture):
    assert Q.run_files(open_fixture("single")) == {"": "fixture.mzML"}
    assert Q.suggest_conditions(open_fixture("single")) == {"": "fixture"}
    assert Q.suggest_conditions(open_fixture("experiment")) == {"a": "fixture", "b": "b"}
    assert Q.suggest_conditions(open_fixture("mbr")) == {"a": "fixture", "c": "c"}


@pytest.mark.parametrize(
    ("names", "want"),
    [
        (
            [
                "C:\\data\\LFQ_Astral_DIA_15min_50ng_Condition_A_REP1.mzML",
                "C:/data/LFQ_Astral_DIA_15min_50ng_Condition_A_REP2.mzML",
                "/data/LFQ_Astral_DIA_15min_50ng_Condition_A_REP3.mzML",
                "LFQ_Astral_DIA_15min_50ng_Condition_B_REP1.mzML",
                "LFQ_Astral_DIA_15min_50ng_Condition_B_REP2.mzML",
                "LFQ_Astral_DIA_15min_50ng_Condition_B_REP3.mzML",
            ],
            ["A", "A", "A", "B", "B", "B"],
        ),
        (
            ["HeLa_ctrl_01.mzML", "HeLa_ctrl_02.mzML", "HeLa_treat_01.mzML", "HeLa_treat_02.mzML"],
            ["ctrl", "ctrl", "treat", "treat"],
        ),
        (["s-condX-R1.mzML.gz", "s-condY-R1.mzML.gz"], ["X", "Y"]),
        (["x_rep_1.mzML", "x_rep_2.mzML", "y_rep_1.mzML"], ["x", "x", "y"]),
        (["same_01.mzML", "same_02.mzML"], ["same", "same"]),
        (["a_run1.raw", None], ["a", "r1"]),
        (["dose_10_inj1.mzML", "dose_20_inj1.mzML"], ["10", "20"]),
    ],
)
def test_suggest_conditions_rule(names, want):
    rs = _stub(names)
    assert list(Q.suggest_conditions(rs).values()) == want


def test_suggest_conditions_single_run():
    rs = _stub(["LFQ_Astral_DIA_15min_50ng_Condition_A_REP1.mzML"], experiment=False)
    assert Q.suggest_conditions(rs) == {"": "A"}
    assert Q.suggest_conditions(_stub(["sample_07.mzML"], experiment=False)) == {"": "sample"}
    assert Q.suggest_conditions(_stub([None], experiment=False)) == {"": "run"}


def test_resolve_conditions_keeps_the_stored_choice(open_fixture):
    rs = open_fixture("experiment")
    assert Q.resolve_conditions(rs, None) == Q.suggest_conditions(rs)
    assert Q.resolve_conditions(rs, {}) == Q.suggest_conditions(rs)
    got = Q.resolve_conditions(rs, {"a": "  ctrl ", "b": "", "zz": "other"})
    assert got == {"a": "ctrl", "b": "b"}
    assert Q.resolve_conditions(rs, {"a": 3, "b": None}) == Q.suggest_conditions(rs)
    assert Q.condition_groups({"a": "X", "b": "Y", "c": "X"}, ["c", "b", "a"]) == {
        "X": ["c", "a"],
        "Y": ["b"],
    }


# --------------------------------------------------------------------------- matrices


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("level", Q.LEVELS)
def test_quant_matrix_equals_the_files(open_fixture, name, level):
    rs = open_fixture(name)
    m = Q.quant_matrix(rs, level, "quant")
    assert m.runs == tuple(r.name for r in rs.runs)
    assert m.values.shape == (m.n_keys, len(rs.runs))
    _same(_as_frame(m), _brute_quant(rs, level))
    assert not (m.values == 0).any()
    # A key is sorted, unique, and has a value in at least one run.
    assert not m.key_index().duplicated().any()


@pytest.mark.parametrize("name", ["experiment", "mbr"])
@pytest.mark.parametrize("level", Q.LEVELS)
def test_lfq_matrix_equals_the_files(open_fixture, name, level):
    rs = open_fixture(name)
    m = Q.quant_matrix(rs, level, "lfq")
    _same(_as_frame(m), _brute_lfq(rs, level))
    assert "n_features" in m.keys


def test_lfq_source_needs_an_experiment(open_fixture):
    with pytest.raises(ViewerError, match="experiment"):
        Q.quant_matrix(open_fixture("single"), "protein", "lfq")
    with pytest.raises(ViewerError, match="level"):
        Q.quant_matrix(open_fixture("single"), "peptide", "quant")


def test_transfers_of_the_cells(open_fixture):
    rs = open_fixture("mbr")
    transfers = _read(rs.root / "mbr_transferred.parquet", ["source", "candidate_id"])
    pairs = set(
        zip(transfers["source"].astype(int), transfers["candidate_id"].astype(int), strict=True)
    )
    m = Q.quant_matrix(rs, "precursor", "quant")
    assert m.transferred is not None
    want: dict[tuple[str, int, str], int] = {}
    for run in rs.runs:
        df = _read(run.artifact("peptide_quant").path)
        df = df[_positive(df["quantity"]).notna()]
        for row in df.itertuples():
            if (int(run.index), int(row.candidate_id)) in pairs:
                k = (row.peptidoform, int(row.charge), run.name)
                want[k] = want.get(k, 0) + 1
    got = {}
    keys = m.keys
    for i, j in zip(*np.nonzero(m.transferred), strict=True):
        got[(keys["peptidoform"].iloc[i], int(keys["charge"].iloc[i]), m.runs[j])] = int(
            m.transferred[i, j]
        )
    assert got == want and sum(want.values()) > 0
    lfq = Q.quant_matrix(rs, "protein", "lfq")
    assert lfq.transferred is not None and lfq.transferred.sum() > 0
    assert Q.quant_matrix(open_fixture("experiment"), "precursor", "quant").transferred is None


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("level", Q.LEVELS)
@pytest.mark.parametrize("t", [0.01, 0.1, 1.0])
def test_accepted_keys_and_the_restricted_matrix(open_fixture, name, level, t):
    rs = open_fixture(name)
    scored = _read(rs.scored.path)
    target = scored[scored["label"] == "target"]
    if level == "precursor":
        acc = target[target["precursor_q"] <= t][["peptidoform", "charge"]].drop_duplicates()
        want = set(zip(acc["peptidoform"], acc["charge"].astype(int), strict=True))
    else:
        want = set(target[target["pg_q_value"] <= t]["protein_group"].astype(str))
    keys = Q.accepted_keys(rs, level, t)
    got = (
        set(zip(keys["peptidoform"], keys["charge"], strict=True))
        if level == "precursor"
        else set(keys["protein_group"])
    )
    assert got == want
    base = Q.quant_matrix(rs, level, "quant")
    m = Q.quant_matrix(rs, level, "quant", accepted_at=t)
    assert m.accepted_at == t and m.n_keys == len(want)
    full = _as_frame(base)
    sub = _as_frame(m)
    inside = sub.index.isin(full.index)
    _same(sub[inside], full)
    # An accepted key without any quantity is a row of NaN (missing, not zero).
    assert np.isnan(sub[~inside].to_numpy(dtype="float64")).all()


def test_accepted_precursors_without_quantity_in_a_peptide_q_run(open_fixture):
    """PeptideQ quantifies one precursor per base peptide: the others are missing."""
    rs = open_fixture("single")
    m = Q.quant_matrix(rs, "precursor", "quant", accepted_at=0.01)
    miss = Q.missing_by_run(m)
    assert miss["keys"].iloc[0] == 273
    assert miss["present"].iloc[0] == 150 and miss["missing"].iloc[0] == 123
    assert miss["missing_pct"].iloc[0] == pytest.approx(100 * 123 / 273)


# --------------------------------------------------------------------------- summaries


@pytest.mark.parametrize("name", FIXTURES)
def test_distribution_against_numpy(open_fixture, name):
    rs = open_fixture(name)
    m = Q.quant_matrix(rs, "precursor", "quant")
    d = Q.quantity_distribution(m, bins=20)
    assert d.counts.shape == (len(m.runs), 20) and len(d.edges) == 21
    every = np.log10(m.values[np.isfinite(m.values)])
    assert d.edges[0] == pytest.approx(every.min()) and d.edges[-1] == pytest.approx(every.max())
    for j, run in enumerate(m.runs):
        x = np.log10(m.values[:, j][np.isfinite(m.values[:, j])])
        row = d.table.iloc[j]
        assert row["run"] == run and row["n"] == x.size
        assert row["missing"] == m.n_keys - x.size
        for p in Q.PERCENTILES:
            assert row[f"p{p}"] == pytest.approx(np.percentile(x, p))
        assert np.array_equal(d.counts[j], np.histogram(x, bins=d.edges)[0])
        assert d.counts[j].sum() == x.size


def test_missing_and_runs_with_value(open_fixture):
    rs = open_fixture("mbr")
    m = Q.quant_matrix(rs, "precursor", "lfq")
    ok = np.isfinite(m.values)
    miss = Q.missing_by_run(m)
    assert miss["present"].tolist() == ok.sum(axis=0).tolist()
    assert (miss["present"] + miss["missing"] == m.n_keys).all()
    per = Q.runs_with_value(m)
    assert per["n_runs"].tolist() == list(range(len(m.runs) + 1))
    assert per["keys"].tolist() == [int((ok.sum(axis=1) == n).sum()) for n in per["n_runs"]]
    assert per["keys"].sum() == m.n_keys and per["keys"].iloc[0] == 0
    assert per["pct"].sum() == pytest.approx(100.0)


def _matrix(values, runs) -> Q.QuantMatrix:
    values = np.asarray(values, dtype="float64")
    keys = pd.DataFrame({"protein_group": [f"G{i}" for i in range(len(values))]})
    return Q.QuantMatrix("protein", "quant", tuple(runs), keys, values)


def test_condition_cvs_against_numpy():
    rng = np.random.default_rng(7)
    values = rng.lognormal(10, 1, size=(200, 7))
    values[rng.random(values.shape) < 0.25] = np.nan
    runs = ["a1", "a2", "a3", "b1", "b2", "c1", "x"]
    conditions = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"}
    m = _matrix(values, runs)
    for all_runs in (True, False):
        res = Q.condition_cvs(m, conditions, all_runs=all_runs)
        assert [c.condition for c in res.conditions] == ["A", "B", "C"]
        for c, cols in zip(res.conditions[:2], ([0, 1, 2], [3, 4]), strict=True):
            need = len(cols) if all_runs else 2
            assert c.need == need and c.runs == tuple(runs[j] for j in cols)
            want_rows, want_cv = [], []
            for i in range(len(values)):
                x = values[i, cols]
                x = x[np.isfinite(x)]
                if x.size >= need:
                    want_rows.append(i)
                    want_cv.append(100 * np.std(x, ddof=1) / np.mean(x))
            assert c.rows.tolist() == want_rows
            assert np.allclose(c.cv, want_cv, rtol=1e-12)
            assert c.n_cv == len(want_rows)
            assert c.median == pytest.approx(np.median(want_cv))
            assert c.share_below(20) == pytest.approx(100 * np.mean(np.asarray(want_cv) <= 20))
            assert c.n_with_value == int(np.isfinite(values[:, cols]).any(axis=1).sum())
        one = res.conditions[2]
        assert one.n_cv == 0 and one.median is None and "one run" in one.reason
        assert ("every run" in res.rule) == all_runs and "n - 1" in res.rule


def test_condition_cvs_of_the_mbr_fixture(open_fixture):
    rs = open_fixture("mbr")
    m = Q.quant_matrix(rs, "precursor", "lfq")
    res = Q.condition_cvs(m, {"a": "X", "c": "X"})
    (c,) = res.conditions
    both = np.isfinite(m.values).all(axis=1)
    want = 100 * np.std(m.values[both], axis=1, ddof=1) / np.mean(m.values[both], axis=1)
    assert c.rows.tolist() == np.flatnonzero(both).tolist()
    assert np.allclose(c.cv, want)
    # The experiment fixture's two runs are copies: every CV is 0.
    e = Q.quant_matrix(open_fixture("experiment"), "precursor", "lfq")
    (z,) = Q.condition_cvs(e, {"a": "X", "b": "X"}).conditions
    assert z.n_cv == e.n_keys and np.allclose(z.cv, 0.0)


# --------------------------------------------------------------------------- per run


def _brute_states(rs, t: float) -> pd.DataFrame:
    scored = _read(rs.scored.path)
    rows = []
    for run in rs.runs:
        part = scored[scored["source"] == run.index] if "source" in scored else scored
        acc = set(part[(part["label"] == "target") & (part["run_psm_q"] <= t)]["candidate_id"])
        pqt = _read(run.artifact("peptide_quant").path)
        pqt["ok"] = _positive(pqt["quantity"]).notna()
        ok = set(pqt[pqt["ok"]]["candidate_id"])
        rows_q = set(pqt["candidate_id"])
        rows.append(
            {
                "run": run.name,
                "accepted": len(acc),
                "quantified": len(acc & ok),
                "not_quantifiable": len((acc & rows_q) - ok),
                "not_selected": len(acc - rows_q),
                "quantified_not_accepted": len(ok - acc),
                "quant_rows": len(rows_q),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("t", [0.001, 0.01, 0.05])
def test_accepted_quant_states_against_pandas(open_fixture, name, t):
    rs = open_fixture(name)
    got = Q.accepted_quant_states(rs, t)
    want = _brute_states(rs, t)
    for column in want.columns:
        assert got[column].tolist() == want[column].tolist(), column
    assert set(got.attrs["labels"]) >= set(want.columns[1:])
    assert ("transfers" in got) == (name == "mbr")


def test_mbr_transfers_among_the_quantified_rows(open_fixture):
    rs = open_fixture("mbr")
    got = Q.accepted_quant_states(rs, 0.01)
    transfers = _read(rs.root / "mbr_transferred.parquet", ["source", "candidate_id"])
    for run, n in zip(got["run"], got["transfers"], strict=True):
        r = rs.run(run)
        pqt = _read(r.artifact("peptide_quant").path)
        ok = set(pqt[_positive(pqt["quantity"]).notna()]["candidate_id"])
        moved = set(transfers[transfers["source"] == r.index]["candidate_id"])
        assert n == len(ok & moved) and n > 0


def _null_quantities(tmp_path: Path, fixture_dir) -> tuple[object, list[int]]:
    copy = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), copy)
    path = copy / "peptide_quant.parquet"
    table = pq.read_table(path)
    df = table.to_pandas()
    rows = [0, 1, 2]
    df["quantity"] = df["quantity"].astype(object)
    df.loc[rows, "quantity"] = None
    df.loc[rows, "quant_status"] = "no_positive_fragment_area"
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), path)
    return open_results(copy), df["candidate_id"].iloc[rows].tolist()


def test_a_null_quantity_is_missing_and_not_quantifiable(tmp_path, fixture_dir):
    rs, ids = _null_quantities(tmp_path, fixture_dir)
    states = Q.accepted_quant_states(rs, 1.0)
    want = _brute_states(rs, 1.0)
    assert states["not_quantifiable"].iloc[0] == want["not_quantifiable"].iloc[0] == 3
    m = Q.quant_matrix(rs, "precursor", "quant")
    pqt = _read(rs.runs[0].artifact("peptide_quant").path)
    gone = pqt[pqt["candidate_id"].isin(ids)]
    frame = _as_frame(m)
    for row in gone.itertuples():
        assert np.isnan(frame.loc[(row.peptidoform, row.charge)].iloc[0])
    status = Q.status_matrix(rs)
    sub = status[status["table"] == "peptide_quant"].set_index("status")
    assert sub.loc["no_positive_fragment_area", ""] == 3
    assert sub.loc["quantified", ""] == len(pqt) - 3
    assert sub.loc["no_positive_fragment_area", "description"].startswith("not quantifiable")


@pytest.mark.parametrize("name", FIXTURES)
def test_status_matrix_equals_the_breakdown(open_fixture, name):
    rs = open_fixture(name)
    got = Q.status_matrix(rs)
    b = quant_status_breakdown(rs)
    assert list(got.columns[:3]) == ["table", "status", "description"]
    for row in b.itertuples():
        hit = got[(got["table"] == row.table) & (got["status"] == row.status)]
        assert int(hit[row.run].iloc[0]) == row.n
        if "n_transferred" in b.columns:
            assert int(hit[f"n_transferred_{row.run}"].iloc[0]) == row.n_transferred
    assert got["table"].iloc[0] == "peptide_quant" and got["status"].iloc[0] == "quantified"


def test_size_factors(open_fixture):
    assert Q.size_factors(open_fixture("single")) is None
    rs = open_fixture("mbr")
    sf = Q.size_factors(rs)
    assert sf is not None and sf.attrs["constant"]
    quant = _brute_quant(rs, "precursor")
    lfq = _brute_lfq(rs, "precursor").reindex(quant.index)
    for row in sf.itertuples():
        ratio = (quant[row.run] / lfq[row.run]).dropna()
        assert row.n == len(ratio)
        assert row.factor == pytest.approx(float(np.median(ratio)), rel=1e-12)
        assert np.allclose(ratio, row.factor, rtol=1e-9)
    assert not np.isclose(sf["factor"], 1.0).all()


# --------------------------------------------------------------------------- heatmap


def test_average_linkage_order():
    d = np.array(
        [
            [0, 9, 1, 9],
            [9, 0, 9, 2],
            [1, 9, 0, 9],
            [9, 2, 9, 0],
        ],
        dtype=float,
    )
    order = Q._average_linkage(d)
    assert sorted(order) == [0, 1, 2, 3]
    assert abs(order.index(0) - order.index(2)) == 1
    assert abs(order.index(1) - order.index(3)) == 1
    assert Q._average_linkage(d) == order  # deterministic


def test_heatmap_default_order(open_fixture):
    rs = open_fixture("mbr")
    m = Q.quant_matrix(rs, "protein", "lfq")
    h = Q.heatmap(m, {"a": "Y", "c": "X"})
    assert h.runs == ("a", "c") and h.conditions == ("Y", "X")
    means = np.nanmean(h.log10, axis=1)
    assert (np.diff(means) <= 1e-12).all()
    frame = _as_frame(m)
    for i, g in enumerate(h.groups[:5]):
        assert np.allclose(np.log10(frame.loc[g].to_numpy(dtype=float)), h.log10[i], equal_nan=True)
    rel = h.relative
    assert np.allclose(np.nanmean(rel, axis=1), 0.0)
    swapped = Q.heatmap(m, {"a": "X", "c": "Y", "zz": "Z"})
    assert swapped.runs == ("a", "c")


def test_heatmap_clusters_rows_and_columns():
    rng = np.random.default_rng(3)
    base = rng.uniform(5, 8, size=(120, 1))
    up = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    profiles = np.where(np.arange(120)[:, None] < 60, up, -up)
    log10 = base + profiles * 0.6 + rng.normal(0, 0.02, size=(120, 6))
    # Columns given in a mixed order: r0 r3 r1 r4 r2 r5.
    perm = [0, 3, 1, 4, 2, 5]
    values = 10 ** log10[:, perm]
    runs = [f"r{j}" for j in perm]
    m = _matrix(values, runs)
    cond = {f"r{j}": "A" if j < 3 else "B" for j in range(6)}
    plain = Q.heatmap(m, cond)
    assert plain.runs == ("r0", "r1", "r2", "r3", "r4", "r5")
    h = Q.heatmap(m, cond, cluster=True)
    first = {r for r in h.runs[:3]}
    assert first in ({"r0", "r1", "r2"}, {"r3", "r4", "r5"})
    group = np.array([int(g[1:]) < 60 for g in h.groups])
    assert np.count_nonzero(np.diff(group.astype(int))) == 1  # two contiguous blocks
    again = Q.heatmap(m, cond, cluster=True)
    assert again.groups.tolist() == h.groups.tolist() and again.runs == h.runs
    assert "k-means" in h.row_rule and "average linkage" in h.column_rule


def test_heatmap_drops_keys_without_values():
    m = _matrix([[1.0, np.nan], [np.nan, np.nan], [3.0, 4.0]], ["a", "b"])
    h = Q.heatmap(m, {"a": "X", "b": "X"})
    assert h.dropped == 1 and h.groups.tolist() == ["G2", "G0"] and h.rows.tolist() == [2, 0]


# --------------------------------------------------------------------------- one protein


def test_species_of():
    assert Q.species_of("ATLA3_HUMAN") == ("HUMAN",)
    assert Q.species_of("1433B_HUMAN;1433Z_BOVIN;BMH1_YEAST;1433E_HUMAN") == (
        "HUMAN",
        "BOVIN",
        "YEAST",
    )
    assert Q.species_of("DECOY_P12345_ECOLI") == ("ECOLI",)
    assert Q.species_of("sp|FIXT01|FIX01_TEST") == ("TEST",)
    assert Q.species_of("noseparator") == () and Q.species_of(None) == ()
    assert Q.species_of("x_lower") == ()


@pytest.mark.parametrize("name", FIXTURES)
def test_protein_profile_equals_the_tables(open_fixture, name):
    rs = open_fixture(name)
    group = Q.find_groups(rs, "")[0]
    p = Q.protein_profile(rs, group)
    assert p.group == group and p.runs == tuple(r.name for r in rs.runs)
    for run, row in zip(rs.runs, p.quant.itertuples(), strict=True):
        pgq = _read(run.artifact("protein_group_quant").path)
        hit = pgq[pgq["protein_group"] == group]
        assert row.run == run.name
        if len(hit):
            assert row.quantity == pytest.approx(hit["quantity"].iloc[0])
            assert row.quant_status == hit["quant_status"].iloc[0]
            assert row.n_peptides == hit["n_peptides"].iloc[0]
        else:
            assert np.isnan(row.quantity) and row.quant_status is None
    pairs = set()
    for run in rs.runs:
        pqt = _read(run.artifact("peptide_quant").path)
        sub = pqt[pqt["protein_group"] == group]
        pairs |= set(zip(sub["peptidoform"], sub["charge"].astype(int), strict=True))
    assert set(zip(p.precursors["peptidoform"], p.precursors["charge"], strict=True)) == pairs
    quant = _brute_quant(rs, "precursor")
    for i, (pf, z) in enumerate(
        zip(p.precursors["peptidoform"], p.precursors["charge"], strict=True)
    ):
        want = quant.loc[(pf, z)].to_numpy(dtype=float)
        assert np.allclose(p.precursor_quant[i], want, equal_nan=True)
    scored = _read(rs.scored.path)
    q = scored[(scored["protein_group"] == group) & (scored["label"] == "target")]["pg_q_value"]
    assert p.pg_q_value == pytest.approx(q.min())
    if rs.is_experiment:
        lfq = _brute_lfq(rs, "protein").loc[group].to_numpy(dtype=float)
        assert p.in_lfq and np.allclose(p.lfq, lfq, equal_nan=True)
        assert p.precursor_lfq is not None
        assert (p.lfq_transferred is not None) == (name == "mbr")
    else:
        assert p.lfq is None and p.precursor_lfq is None


def test_profile_of_an_unknown_group(open_fixture):
    p = Q.protein_profile(open_fixture("experiment"), "NOT_A_GROUP")
    assert not p.in_lfq and np.isnan(p.lfq).all() and len(p.precursors) == 0
    assert p.quant["quant_status"].isna().all() and p.pg_q_value is None


def test_find_groups_ranks_exact_and_prefix_matches(open_fixture):
    rs = open_fixture("experiment")
    every = Q.find_groups(rs, "", limit=100)
    m = Q.quant_matrix(rs, "protein", "lfq")
    assert len(every) == m.n_keys
    means = _as_frame(m).loc[every].apply(lambda r: np.nanmean(np.log10(r.to_numpy(float))), axis=1)
    assert (np.diff(means.to_numpy()) <= 1e-12).all()
    exact = every[3]
    assert Q.find_groups(rs, exact.upper())[0] == exact
    hits = Q.find_groups(rs, "fix0", limit=100)
    assert hits and all("fix0" in h.lower() for h in hits)
    assert Q.find_groups(rs, "FIXT01|FIX01_TEST") == ["sp|FIXT01|FIX01_TEST"]
    assert Q.find_groups(rs, "no such protein") == []
    assert Q.find_groups(rs, "", accepted_at=0.01) == []
