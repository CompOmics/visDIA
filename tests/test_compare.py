"""The compare data layer (data.compare) against a brute force over the fixtures' scored
tables (pyarrow and pandas), with the stripped sequence from the UI's own peptidoform
parser (widgets.parse_peptidoform) as the independent reference."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import compare as cd
from mumdia_viewer.data.counts import unit_counts
from mumdia_viewer.ui.widgets import parse_peptidoform

# Pairs with identifications only on one side (grouped and ovl_bp differ from the
# smoke run) and pairs from one library and one mzML (single, experiment, mbr).
PAIRS = [
    ("single", "grouped"),
    ("single", "ovl_bp"),
    ("experiment", "mbr"),
    ("single", "experiment"),
]
THRESHOLDS = [0.01, 0.05, 0.5]
Q = {"precursor": "precursor_q", "peptide": "peptide_q_value", "protein_group": "pg_q_value"}


def _table(rs) -> pd.DataFrame:
    df = pq.read_table(rs.scored.path).to_pandas()
    df["protein_group"] = df["protein_group"].fillna("")
    return df


def _keyed(rs, unit: str) -> pd.DataFrame:
    """Every target row with its matched key (brute force)."""
    df = _table(rs)
    df = df[df["label"] == "target"].copy()
    if unit == "precursor":
        df["key"] = df["peptidoform"] + "/" + df["charge"].astype(str)
    elif unit == "peptide":
        df["key"] = [parse_peptidoform(p).sequence for p in df["peptidoform"]]
    else:
        df["key"] = df["protein_group"]
        df = df[df["key"] != ""]
    return df


def _brute_keys(rs, unit: str, t: float) -> pd.DataFrame:
    df = _keyed(rs, unit)
    df = df[df[Q[unit]] <= t]
    best = []
    for key, g in df.groupby("key"):
        g = g.sort_values([Q[unit], "score"], ascending=[True, False])
        r = g.iloc[0]
        best.append({"key": key, "q": r[Q[unit]], "score": r["score"], "cid": r["candidate_id"]})
    return pd.DataFrame(best, columns=["key", "q", "score", "cid"])


# --------------------------------------------------------------------------- strings


def test_strip_sequence_cases():
    assert cd.strip_sequence("PEPTIDEK") == "PEPTIDEK"
    assert cd.strip_sequence("PEM[Oxidation]C[Carbamidomethyl]K") == "PEMCK"
    assert cd.strip_sequence("[Acetyl]-PEM[UNIMOD:35]K") == "PEMK"
    assert cd.strip_sequence("PEM(Oxidation)K-[Amidated]") == "PEMK"
    assert cd.strip_sequence("[Acetyl]-[Formyl]-AK") == "AK"
    assert cd.strip_sequence("") == "" and cd.strip_sequence(None) == ""


@pytest.mark.parametrize("name", ["single", "experiment", "mbr", "grouped", "topk", "ovl_bp"])
def test_strip_sequence_matches_the_parser(open_fixture, name):
    peptidoforms = set(_table(open_fixture(name))["peptidoform"])
    for text in peptidoforms:
        expected = parse_peptidoform(text).sequence
        assert cd.strip_sequence(text) == (
            ("DECOY_" + expected) if text.startswith("DECOY_") else expected
        )


def test_sql_strip_equals_python(open_fixture):
    rs = open_fixture("single")
    rows = cd.winner_rows(rs)
    assert len(rows) > 0
    assert rows["sequence"].tolist() == [cd.strip_sequence(p) for p in rows["peptidoform"]]


# --------------------------------------------------------------------------- one side


@pytest.mark.parametrize("name", ["single", "experiment", "mbr", "grouped"])
@pytest.mark.parametrize("unit", cd.COMPARE_UNITS)
@pytest.mark.parametrize("t", THRESHOLDS)
def test_unit_keys_equal_brute_force(open_fixture, name, unit, t):
    rs = open_fixture(name)
    got = cd.unit_keys(rs, unit, t).sort_values("key").reset_index(drop=True)
    want = _brute_keys(rs, unit, t).sort_values("key").reset_index(drop=True)
    assert got["key"].tolist() == want["key"].tolist()
    np.testing.assert_allclose(got["q"].to_numpy(float), want["q"].to_numpy(float))
    np.testing.assert_allclose(got["score"].to_numpy(float), want["score"].to_numpy(float))
    assert got["cid"].astype(int).tolist() == want["cid"].astype(int).tolist()
    if rs.is_experiment:
        names = {r.index: r.name for r in rs.runs}
        table = _table(rs).set_index("candidate_id")
        for cid, run in zip(got["cid"].head(20), got["run"].head(20), strict=True):
            sources = table.loc[[cid], "source"]
            assert run in {names[int(s)] for s in sources}
    else:
        assert set(got["run"]) <= {""}


@pytest.mark.parametrize("name", ["single", "experiment", "mbr", "grouped", "ovl_bp"])
@pytest.mark.parametrize("t", THRESHOLDS)
def test_side_counts_equal_unit_counts(open_fixture, name, t):
    rs = open_fixture(name)
    got = cd.side_counts(rs, t)
    want = {c.unit: c.n_target for c in unit_counts(rs, t)}
    assert got == want


def test_threshold_above_the_fetch_reads_again(open_fixture):
    rs = open_fixture("single")
    keys = cd.unit_keys(rs, "precursor", 0.3)
    want = _brute_keys(rs, "precursor", 0.3)
    assert len(keys) == len(want) and cd.FETCH_Q < 0.3


def test_unknown_unit_is_refused(open_fixture):
    with pytest.raises(ValueError):
        cd.unit_keys(open_fixture("single"), "psm", 0.01)


# --------------------------------------------------------------------------- overlap


@pytest.mark.parametrize(("x", "y"), PAIRS)
@pytest.mark.parametrize("t", THRESHOLDS)
def test_overlap_equals_set_operations(open_fixture, x, y, t):
    a, b = open_fixture(x), open_fixture(y)
    for o in cd.overlap(a, b, t):
        ka = set(_brute_keys(a, o.unit, t)["key"])
        kb = set(_brute_keys(b, o.unit, t)["key"])
        assert (o.n_a, o.n_b, o.n_both) == (len(ka), len(kb), len(ka & kb))
        assert o.n_only_a == len(ka - kb) and o.n_only_b == len(kb - ka)
        assert Q[o.unit] in o.label
        if ka | kb:
            assert o.jaccard == pytest.approx(len(ka & kb) / len(ka | kb))
        else:
            assert o.jaccard is None


def test_some_pairs_differ(open_fixture):
    o = cd.overlap(open_fixture("single"), open_fixture("grouped"), 0.01)[0]
    assert o.n_only_a > 0 and o.n_only_b > 0


@pytest.mark.parametrize(("x", "y"), PAIRS)
@pytest.mark.parametrize("unit", cd.COMPARE_UNITS)
def test_shared_keys_carry_both_winners(open_fixture, x, y, unit):
    a, b = open_fixture(x), open_fixture(y)
    t = 0.5
    got = cd.shared_keys(a, b, unit, t)
    wa = _brute_keys(a, unit, t).set_index("key")
    wb = _brute_keys(b, unit, t).set_index("key")
    assert got["key"].tolist() == sorted(set(wa.index) & set(wb.index))
    for r in got.itertuples():
        assert r.a_score == pytest.approx(wa.loc[r.key, "score"])
        assert r.b_score == pytest.approx(wb.loc[r.key, "score"])
        assert r.a_q == pytest.approx(wa.loc[r.key, "q"])
        assert r.b_q == pytest.approx(wb.loc[r.key, "q"])


@pytest.mark.parametrize(("x", "y"), PAIRS[:3])
@pytest.mark.parametrize("unit", cd.COMPARE_UNITS)
@pytest.mark.parametrize("side", ["a", "b"])
@pytest.mark.parametrize("t", [0.01, 0.5])
def test_unique_keys_and_the_other_side(open_fixture, x, y, unit, side, t):
    a, b = open_fixture(x), open_fixture(y)
    own, other = (a, b) if side == "a" else (b, a)
    got = cd.unique_keys(a, b, unit, t, side)
    mine = set(_brute_keys(own, unit, t)["key"])
    theirs = set(_brute_keys(other, unit, t)["key"])
    assert set(got["key"]) == mine - theirs
    assert got["q"].is_monotonic_increasing
    rows = _keyed(other, unit)
    for r in got.itertuples():
        hits = rows[rows["key"] == r.key]
        assert r.other_rows == len(hits)
        if len(hits):
            assert r.other_q == pytest.approx(hits[Q[unit]].min())
            assert r.other_q > t
        else:
            assert math.isnan(r.other_q)
    if unit == "protein_group":
        members = {m: g for g in theirs for m in g.split(";")}
        for r in got.itertuples():
            found = [m for m in r.key.split(";") if m in members]
            assert (r.other_group != "") == bool(found)
            if r.other_group:
                assert set(r.key.split(";")) & set(r.other_group.split(";"))


def test_unique_keys_are_memoised_per_pair(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    first = cd.unique_keys(a, b, "precursor", 0.01, "a")
    assert cd.unique_keys(a, b, "precursor", 0.01, "a") is first
    other = cd.unique_keys(a, open_fixture("ovl_bp"), "precursor", 0.01, "a")
    assert other is not first
    with pytest.raises(ValueError):
        cd.unique_keys(a, b, "precursor", 0.01, "c")


# --------------------------------------------------------------------------- runs and quant


def test_run_pairs_one_to_one(open_fixture):
    single, exp = open_fixture("single"), open_fixture("experiment")
    pairs = cd.run_pairs(single, exp)
    assert [(p.a, p.b) for p in pairs] == [("", "a")]
    assert pairs[0].how == "content hash"
    # Runs a and b of the experiment have one content hash (copies of one file): the
    # file name decides, and no run is used twice.
    pairs = cd.run_pairs(exp, exp)
    assert [(p.a, p.b) for p in pairs] == [("a", "a"), ("b", "b")]
    assert cd.run_pairs(single, open_fixture("topk")) == []


def test_run_inputs(open_fixture):
    exp = open_fixture("experiment")
    inputs = cd.run_inputs(exp)
    assert [i.run for i in inputs] == ["a", "b"]
    assert inputs[0].file_name == "fixture.mzML" and inputs[1].file_name == "fixture_b.mzML"
    assert inputs[0].content_hash == exp.manifest.inputs["mzml[0]"].content_hash


def _peptide_quant(rs, run_name: str) -> pd.DataFrame:
    run = rs.run(run_name)
    df = pq.read_table(run.artifact("peptide_quant").path).to_pandas()
    df["key"] = df["peptidoform"] + "/" + df["charge"].astype(str)
    return df.drop_duplicates("key").set_index("key")


def test_quantity_pairs_per_run(open_fixture):
    a, b = open_fixture("single"), open_fixture("grouped")
    pairs = cd.run_pairs(a, b)
    assert pairs
    got = cd.quantity_pairs(a, b, 0.01, pairs)
    shared = cd.shared_keys(a, b, "precursor", 0.01)
    assert got.attrs["mode"] == "per run" and got.attrs["n_shared"] == len(shared)
    assert len(got) == len(shared) * len(pairs)
    qa, qb = _peptide_quant(a, pairs[0].a), _peptide_quant(b, pairs[0].b)
    for r in got.itertuples():
        want_a = qa["quantity"].get(r.key, np.nan)
        want_b = qb["quantity"].get(r.key, np.nan)
        assert (math.isnan(r.a_quantity) and pd.isna(want_a)) or r.a_quantity == pytest.approx(
            want_a
        )
        assert (math.isnan(r.b_quantity) and pd.isna(want_b)) or r.b_quantity == pytest.approx(
            want_b
        )
    ok = (got["a_quantity"] > 0) & (got["b_quantity"] > 0)
    assert got.attrs["n_both"] == int(ok.sum())


def test_quantity_pairs_pooled_is_the_median(open_fixture):
    a, b = open_fixture("experiment"), open_fixture("topk")
    got = cd.quantity_pairs(a, b, 0.01, None)
    assert got.attrs["mode"] == "pooled" and "median" in got.attrs["label"]
    frames = [_peptide_quant(a, r.name) for r in a.runs]
    pooled = pd.concat([f["quantity"] for f in frames]).dropna().groupby(level=0).median()
    for r in got.head(50).itertuples():
        want = pooled.get(r.key, np.nan)
        assert (math.isnan(r.a_quantity) and pd.isna(want)) or r.a_quantity == pytest.approx(want)


def test_spearman():
    assert cd.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cd.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert cd.spearman([1, 2], [1, 2]) is None
    assert cd.spearman([1, 1, 1], [1, 2, 3]) is None


# --------------------------------------------------------------------------- provenance


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            out.update(_flat(v, name))
        else:
            out[name] = v
    return out


@pytest.mark.parametrize(
    ("x", "y"), [("single", "experiment"), ("single", "mbr"), ("single", "topk")]
)
def test_config_diff_equals_brute_force(open_fixture, x, y):
    a, b = open_fixture(x), open_fixture(y)
    fa = _flat(json.loads(a.manifest.config_json))
    fb = _flat(json.loads(b.manifest.config_json))
    want = {
        k
        for k in set(fa) | set(fb)
        if k not in fa
        or k not in fb
        or json.dumps(fa[k], sort_keys=True) != json.dumps(fb[k], sort_keys=True)
    }
    got = cd.config_diff(a, b)
    assert {d.key for d in got} == want
    for d in got:
        assert d.kind == (
            "only in A" if d.key not in fb else "only in B" if d.key not in fa else "differs"
        )


def test_config_diff_marks_path_spelling():
    assert cd._same_path("C:/a/b.exe", "C:\\a\\b.exe")
    assert not cd._same_path("C:/a/b.exe", "C:/a/c.exe")
    assert not cd._same_path("auto", "AUTO")


def test_provenance(open_fixture):
    a, b = open_fixture("single"), open_fixture("experiment")
    rows = {r.label: r for r in cd.provenance(a, b)}
    assert rows["MuMDIA version"].a == a.manifest.mumdia_version and rows["MuMDIA version"].same
    assert rows["Git SHA"].a == a.manifest.git_sha
    assert rows["Kind"].same is False and "2 runs" in rows["Kind"].b
    assert rows["mzML inputs"].same is False and "run = a" in rows["mzML inputs"].tip
    same = {r.label: r for r in cd.provenance(a, a)}
    assert all(r.same in (True, None) for r in same.values())
