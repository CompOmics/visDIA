"""One precursor across runs and condition ratios (data.across), each checked against an
independent computation with pyarrow and pandas on the fixtures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import across as A
from mumdia_viewer.data.chromatograms import ChromatogramSource
from mumdia_viewer.data.errors import ViewerError

FIXTURES = ["single", "experiment", "mbr", "grouped"]


def _scored(rs) -> pd.DataFrame:
    return pq.read_table(rs.scored.path).to_pandas()


def _some_precursors(rs, n: int = 4) -> list[tuple[str, int]]:
    """Target precursors scored in the most runs (and one scored in fewer, if any)."""
    df = _scored(rs)
    df = df[df["label"] == "target"]
    counts = df.groupby(["peptidoform", "charge"])["source"].nunique().sort_values()
    keys = list(counts.index[-n:])
    if counts.iloc[0] < counts.iloc[-1]:
        keys.append(counts.index[0])
    return [(str(p), int(z)) for p, z in keys]


def _quant(run) -> pd.DataFrame:
    art = run.artifact("peptide_quant")
    return pq.read_table(art.path, columns=["candidate_id", "quantity"]).to_pandas()


@pytest.mark.parametrize("name", FIXTURES)
def test_rows_match_the_scored_and_quant_tables(open_fixture, name):
    rs = open_fixture(name)
    scored = _scored(rs)
    if "source" not in scored:
        scored["source"] = 0
    for pep, z in _some_precursors(rs):
        a = A.precursor_across(rs, pep, z)
        assert list(a.rows["run"]) == [r.name for r in rs.runs]
        mine = scored[(scored["peptidoform"] == pep) & (scored["charge"] == z)]
        for run, row in zip(rs.runs, a.rows.to_dict("records"), strict=True):
            want = mine[mine["source"] == run.index]
            assert row["scored"] == (not want.empty)
            if want.empty:
                assert row["state"] == "not_scored" and math.isnan(row["quantity"])
                continue
            best = want.sort_values("score", ascending=False).iloc[0]
            assert row["candidate_id"] == best["candidate_id"]
            assert row["score"] == pytest.approx(best["score"], rel=0, abs=0)
            assert row["run_psm_q"] == pytest.approx(best["run_psm_q"], rel=0, abs=0)
            assert row["apex_rt"] == pytest.approx(best["apex_rt"], rel=0, abs=0)
            q = _quant(run)
            hit = q[q["candidate_id"] == best["candidate_id"]]["quantity"]
            value = float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else math.nan
            if math.isfinite(value) and value > 0:
                assert row["state"] == "quantified"
                assert row["quantity"] == pytest.approx(value)
            else:
                # Never 0: a missing quantity stays NaN with a state that says why.
                assert math.isnan(row["quantity"])
                assert row["state"] in ("not_quantifiable", "not_selected")
        assert a.n_scored == mine["source"].nunique()


@pytest.mark.parametrize("name", ["experiment", "mbr"])
def test_grouped_values_are_the_winning_rows(open_fixture, name):
    rs = open_fixture(name)
    scored = _scored(rs)
    for pep, z in _some_precursors(rs):
        a = A.precursor_across(rs, pep, z)
        mine = scored[(scored["peptidoform"] == pep) & (scored["charge"] == z)]
        g = a.group("precursor_q")
        # The sparse column: the group's q is on one row, the others hold 1.0.
        assert g.value == pytest.approx(mine["precursor_q"].min())
        assert g.this_precursor
        bp = scored[scored["base_peptide_id"] == mine["base_peptide_id"].iloc[0]]
        assert a.group("peptide_q_value").value == pytest.approx(bp["peptide_q_value"].min())
        pg = scored[scored["protein_group"] == mine["protein_group"].iloc[0]]
        gq = a.group("pg_q_value")
        assert gq.value == pytest.approx(pg["pg_q_value"].min())
        winner = pg[(pg["candidate_id"] == gq.winner_cid)]
        assert not winner.empty and winner["pg_q_value"].min() == pytest.approx(gq.value)
        # Siblings: the other precursors of the base peptide.
        want = {(p, int(c)) for p, c in zip(bp["peptidoform"], bp["charge"], strict=True)}
        want.discard((pep, z))
        got = set(zip(a.siblings["peptidoform"], a.siblings["charge"].astype(int), strict=True))
        assert got == want


def test_lfq_quantity_is_the_precursor_maxlfq(open_fixture):
    rs = open_fixture("experiment")
    lfq = pq.read_table(rs.artifact("lfq_maxlfq_precursor").path).to_pandas()
    for pep, z in _some_precursors(rs):
        a = A.precursor_across(rs, pep, z)
        sub = lfq[(lfq["group"] == pep) & (lfq["charge"] == z)]
        for run, got in zip(rs.runs, a.rows["lfq_quantity"], strict=True):
            hit = sub[(sub["run"] == run.index) & (sub["quantity"] > 0)]["quantity"]
            if hit.empty:
                assert math.isnan(got)
            else:
                assert got == pytest.approx(float(hit.iloc[0]))


def test_transfers_are_marked(open_fixture):
    rs = open_fixture("mbr")
    tr = pq.read_table(rs.artifact("mbr_transferred").path).to_pandas()
    scored = _scored(rs)
    row = tr.iloc[0]
    pr = scored[(scored["candidate_id"] == row["candidate_id"])].iloc[0]
    a = A.precursor_across(rs, str(pr["peptidoform"]), int(pr["charge"]))
    assert a.mbr
    flagged = {
        (r["source"], r["candidate_id"]) for r in a.rows.to_dict("records") if r["transferred"]
    }
    want = {
        (int(s), int(c))
        for s, c in zip(tr["source"], tr["candidate_id"], strict=True)
        if c in set(a.rows["candidate_id"].dropna().astype(int))
    }
    assert flagged == want and flagged


def test_unknown_precursor(open_fixture):
    rs = open_fixture("experiment")
    with pytest.raises(ViewerError, match="no scored row"):
        A.precursor_across(rs, "NOTAPEPTIDE", 2)


@pytest.mark.parametrize("name", ["single", "experiment", "grouped"])
def test_run_xics(open_fixture, name):
    rs = open_fixture(name)
    pep, z = _some_precursors(rs)[0]
    a = A.precursor_across(rs, pep, z)
    xics = A.run_xics(rs, a)
    assert [x.run for x in xics] == [r.name for r in rs.runs]
    for run, x, row in zip(rs.runs, xics, a.rows.to_dict("records"), strict=True):
        if not row["scored"]:
            assert x.chromatogram is None and x.note == "not scored in this run"
            continue
        want = ChromatogramSource.for_run(rs, run).read(int(row["candidate_id"]))
        assert (want is None) == (x.chromatogram is None)
        assert x.apex_rt == row["apex_rt"]
        s = x.summed()
        if s is not None:
            axis, total = s
            m = want.matrix()
            assert np.array_equal(axis, m[0].astype("float64"))
            assert np.allclose(total, m[2].astype("float64").sum(axis=0))


def test_find_and_default_precursor(open_fixture):
    rs = open_fixture("experiment")
    df = _scored(rs)
    t = df[df["label"] == "target"]
    got = A.find_precursors(rs, "fk", limit=500)
    want = t[t["peptidoform"].str.lower().str.contains("fk")]
    assert set(zip(got["peptidoform"], got["charge"], strict=True)) == set(
        zip(want["peptidoform"], want["charge"], strict=True)
    )
    assert list(got["precursor_q"]) == sorted(got["precursor_q"])
    acc = t[t["precursor_q"] <= 0.01].sort_values(
        ["score", "peptidoform", "charge"], ascending=[False, True, True]
    )
    pool = acc.head(50)
    n = t.groupby(["peptidoform", "charge"])["source"].nunique()
    pool = pool.assign(
        n=[n[(p, c)] for p, c in zip(pool["peptidoform"], pool["charge"], strict=True)]
    )
    best = pool.sort_values(["n", "score"], ascending=[False, False], kind="mergesort").iloc[0]
    assert A.default_precursor(rs, 0.01) == (best["peptidoform"], int(best["charge"]))
    # Nothing accepted: the best-scoring target row.
    top = t.sort_values("score", ascending=False).iloc[0]
    assert A.default_precursor(rs, 1e-12) == (top["peptidoform"], int(top["charge"]))


# --------------------------------------------------------------------------- ratios


def test_species_and_ratio_parsing():
    s = A.DEFAULT_SUFFIXES
    assert A.species_by_suffix("ALBU_HUMAN", s) == "_HUMAN"
    assert A.species_by_suffix("ALBU_HUMAN;ALBU_BOVIN", s) == "_HUMAN"
    assert A.species_by_suffix("A_HUMAN;B_YEAST", s) == A.MIXED
    assert A.species_by_suffix("DECOY_A_ECOLI", s) == "_ECOLI"
    assert A.species_by_suffix("sp|P1|X_MOUSE", s) == A.OTHER
    assert A.species_by_suffix(None, s) == A.OTHER
    assert A.parse_ratio("2:1") == 1.0
    assert A.parse_ratio("1:4") == -2.0
    assert A.parse_ratio("1/4") == -2.0
    assert A.parse_ratio("0.5") == -1.0
    assert A.parse_ratio(" 1 : 1 ") == 0.0
    for bad in ("", "0:1", "a:b", "-1", None, "1:0"):
        assert A.parse_ratio(bad) is None


def _brute_ratio(rs, groups, a, b, summary, need, level="precursor"):
    frames = []
    for run in rs.runs:
        if level == "precursor":
            df = pq.read_table(run.artifact("peptide_quant").path).to_pandas()
            df["q"] = df["quantity"].where(np.isfinite(df["quantity"]) & (df["quantity"] > 0))
            frames.append(df.groupby(["peptidoform", "charge"])["q"].max().rename(run.name))
    wide = pd.concat(frames, axis=1)

    def summ(cols):
        v = wide[cols]
        n = v.notna().sum(axis=1)
        need_n = {"one": 1, "two": min(2, len(cols)), "all": len(cols)}[need]
        s = v.median(axis=1) if summary == "median" else v.mean(axis=1)
        return s.where(n >= need_n), n

    sa, na = summ(groups[a])
    sb, nb = summ(groups[b])
    return np.log2(sa / sb), na, nb


@pytest.mark.parametrize("summary", ["median", "mean"])
@pytest.mark.parametrize("need", ["one", "two", "all"])
def test_condition_ratios_against_pandas(open_fixture, summary, need):
    rs = open_fixture("mbr")
    stored = {"a": "X", "c": "Y"}
    r = A.condition_ratios(
        rs,
        "precursor",
        "quant",
        stored,
        summary=summary,
        need=need,
        accepted_at=None,
        suffixes=("_TEST",),
    )
    assert (r.a, r.b, r.runs_a, r.runs_b) == ("X", "Y", ("a",), ("c",))
    want, na, nb = _brute_ratio(rs, {"X": ["a"], "Y": ["c"]}, "X", "Y", summary, need)
    got = r.table.set_index(["peptidoform", "charge"])
    want = want.reindex(got.index)
    a, w = got["log2_ratio"].to_numpy(), want.to_numpy(dtype="float64")
    assert np.array_equal(np.isnan(a), np.isnan(w))
    assert np.allclose(a[~np.isnan(a)], w[~np.isnan(w)])
    assert np.array_equal(got["n_a"].to_numpy(), na.reindex(got.index).fillna(0).to_numpy())
    assert np.array_equal(got["n_b"].to_numpy(), nb.reindex(got.index).fillna(0).to_numpy())
    sp = r.species.set_index("species")
    total = sp["keys"].sum()
    assert total == len(r.table)
    assert sp["both"].sum() == r.n_both
    assert (sp["both"] + sp["only_a"] + sp["only_b"] + sp["neither"] == sp["keys"]).all()
    x = a[np.isfinite(a)]
    if x.size:
        both = r.table[np.isfinite(r.table["log2_ratio"])]
        test = both[both["species"] == "_TEST"]["log2_ratio"]
        if len(test):
            assert sp.loc["_TEST", "median"] == pytest.approx(float(np.median(test)))


def test_condition_ratios_on_several_runs_per_condition(open_fixture):
    rs = open_fixture("experiment")
    # One condition with both runs and one empty would be one condition: refused.
    with pytest.raises(ViewerError, match="two conditions"):
        A.condition_ratios(rs, "precursor", "quant", {"a": "X", "b": "X"}, accepted_at=None)
    r = A.condition_ratios(rs, "precursor", "lfq", {"a": "B", "b": "A"}, "A", "B", accepted_at=None)
    assert (r.a, r.b, r.runs_a, r.runs_b) == ("A", "B", ("b",), ("a",))
    # A over B reversed is minus the ratio.
    rev = A.condition_ratios(
        rs, "precursor", "lfq", {"a": "B", "b": "A"}, "B", "A", accepted_at=None
    )
    x, y = r.table["log2_ratio"].to_numpy(), rev.table["log2_ratio"].to_numpy()
    ok = np.isfinite(x)
    assert np.array_equal(ok, np.isfinite(y)) and np.allclose(x[ok], -y[ok])


def test_single_run_has_no_ratio(open_fixture):
    with pytest.raises(ViewerError, match="single run"):
        A.condition_ratios(open_fixture("single"), "precursor", "quant")


def test_precursor_groups_and_species(open_fixture):
    rs = open_fixture("experiment")
    r = A.condition_ratios(rs, "precursor", "quant", None, accepted_at=None, suffixes=("_TEST",))
    df = _scored(rs)
    t = df[df["label"] == "target"].sort_values("precursor_q")
    first = t.drop_duplicates(["peptidoform", "charge"]).set_index(["peptidoform", "charge"])
    got = r.table.set_index(["peptidoform", "charge"])
    assert (got["protein_group"] == first["protein_group"].reindex(got.index)).all()
    assert set(got["species"]) <= {"_TEST", A.MIXED, A.OTHER}


def test_ratio_histogram(open_fixture):
    rs = open_fixture("mbr")
    r = A.condition_ratios(rs, "precursor", "quant", None, accepted_at=None, suffixes=("_TEST",))
    edges, counts, clipped = A.ratio_histogram(r, width=0.1, clip=6.0)
    assert np.allclose(np.diff(edges), 0.1)
    sp = r.species.set_index("species")
    for name, c in counts.items():
        assert int(c.sum()) == sp.loc[name, "both"]
        assert clipped[name] >= 0
