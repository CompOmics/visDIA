"""The protein-group data of the protein page, against an independent computation.

The reference reads the fixtures' parquet files with pyarrow and computes every number
with pandas (no DuckDB, no code of mumdia_viewer.data). It checks the group's row, its
peptides and precursors, the quantity per run, the peptides-by-runs matrix and the
search, and that the viewer's per-peptide quantities give the engine's protein quantity
(the top-N sum of protein_group_quant) on every group of every run.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import protein as P
from mumdia_viewer.data.errors import ViewerError

FIXTURES = ["single", "experiment", "mbr", "grouped", "topk", "ovl_bp"]
SCORED_COLUMNS = [
    "source",
    "candidate_id",
    "peptidoform",
    "charge",
    "label",
    "protein_group",
    "base_peptide_id",
    "score",
    "run_psm_q",
    "pg_q_value",
    "peptide_q_value",
]


# --------------------------------------------------------------------------- reference


def _scored(rs) -> pd.DataFrame:
    df = pq.read_table(rs.scored.path, columns=SCORED_COLUMNS).to_pandas()
    df["rn"] = np.arange(len(df))  # the file order
    return df


def _label(group: str) -> str:
    return "decoy" if group.startswith("DECOY_") else "target"


def _groups(rs) -> list[str]:
    df = _scored(rs)
    return sorted(set(df["protein_group"]))


def _transfers(rs) -> set[tuple[int, int]]:
    """Accepted transfers (source, candidate_id), read from the file the engine wrote."""
    path = rs.root / "mbr_transferred.parquet"
    exp = rs.manifest.experiment or {}
    if not path.is_file() or str(exp.get("mbr", "None")).lower() == "none":
        return set()
    t = pq.read_table(path, columns=["source", "candidate_id"]).to_pandas().dropna()
    return {(int(s), int(c)) for s, c in zip(t["source"], t["candidate_id"], strict=True)}


def _peptide_quant(run) -> pd.DataFrame:
    return pq.read_table(run.artifact("peptide_quant").path).to_pandas()


def _protein_quant(run) -> pd.DataFrame:
    return pq.read_table(run.artifact("protein_group_quant").path).to_pandas()


def _winner(rows: pd.DataFrame) -> pd.Series:
    """The engine's winner of a group: the row below 1.0, else highest score, decoy first."""
    below = rows[rows["pg_q_value"] < 1.0]
    if len(below):
        return below.iloc[0]
    order = rows.assign(_d=(rows["label"] == "decoy").astype(int))
    return order.sort_values(["score", "_d", "rn"], ascending=[False, False, True]).iloc[0]


def _strip(peptidoform: str) -> str:
    import re

    text = re.sub(r"^DECOY_", "", peptidoform)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    return re.sub(r"[^A-Za-z]", "", text)


# --------------------------------------------------------------------------- names


def test_species_rule():
    assert P.species_of("ATLA3_HUMAN") == ("HUMAN",)
    assert P.species_of("K0754_HUMAN;MACF1_HUMAN") == ("HUMAN",)
    assert P.species_of("DECOY_ADH1_YEAST;BGAL_ECOLI;X_HUMAN") == ("YEAST", "ECOLI", "HUMAN")
    assert P.species_of("sp|P12345|ABC_HUMAN") == ("HUMAN",)
    for none in ("PLAIN1", "abc_human", "A_TOOLONGX", "_HUMAN", ""):
        assert P.species_of(none) == (), none
    assert "viewer's rule" in P.SPECIES_RULE


def test_members_and_decoys():
    assert P.group_members("A_HUMAN; B_HUMAN;") == ("A_HUMAN", "B_HUMAN")
    assert P.group_members("DECOY_A;B") == ("DECOY_A", "B")
    assert P.is_decoy_group("DECOY_K0754_HUMAN;MACF1_HUMAN")
    assert not P.is_decoy_group("K0754_HUMAN;DECOY_X")


# --------------------------------------------------------------------------- the group


@pytest.mark.parametrize("name", FIXTURES)
def test_group_row_is_the_winning_row(open_fixture, name):
    rs = open_fixture(name)
    scored = _scored(rs)
    for group in _groups(rs):
        rows = scored[scored["protein_group"] == group]
        pg = P.protein_group(rs, group)
        assert pg.found and pg.members == P.group_members(group)
        assert pg.label == _label(group) and pg.decoy == (pg.label == "decoy")
        assert set(rows["label"]) == {pg.label}
        win = _winner(rows)
        assert pg.get("candidate_id") == int(win["candidate_id"])
        assert pg.get("source") == int(win["source"])
        assert pg.get("pg_q_value") == pytest.approx(float(rows["pg_q_value"].min()), abs=0)
        assert pg.get("n_peptides") == rows["base_peptide_id"].nunique()
        assert pg.get("is_winner") is True
        assert "protein group" in pg.description


def test_an_unknown_group_is_not_found(open_fixture):
    pg = P.protein_group(open_fixture("single"), "NOPE_HUMAN")
    assert not pg.found and pg.row is None and pg.get("pg_q_value") is None
    assert pg.members == ("NOPE_HUMAN",) and pg.label == "target"


@pytest.mark.parametrize("name", FIXTURES)
def test_peptides_of_a_group(open_fixture, name):
    rs = open_fixture(name)
    scored = _scored(rs)
    for group in _groups(rs):
        rows = scored[(scored["protein_group"] == group) & (scored["label"] == _label(group))]
        page = P.group_peptides(rs, group)
        got = page.rows
        assert page.total == len(got) == rows["base_peptide_id"].nunique()
        assert set(got["base_peptide_id"]) == set(rows["base_peptide_id"])
        assert set(got["label"]) == {_label(group)}
        assert got["score"].is_monotonic_decreasing
        for bp, part in rows.groupby("base_peptide_id"):
            hit = got[got["base_peptide_id"] == bp].iloc[0]
            assert hit["sequence"] == _strip(part["peptidoform"].iloc[0])
            # The row shown is the peptide's best row of this label.
            assert hit["score"] == pytest.approx(float(part["score"].max()), abs=0)
            assert hit["n_precursors"] == len(
                set(zip(part["peptidoform"], part["charge"], strict=True))
            )


def test_peptides_load_in_blocks(open_fixture, monkeypatch):
    rs = open_fixture("experiment")
    group = "sp|FIXT14|FIX14_TEST"
    whole = P.group_peptides(rs, group).rows
    monkeypatch.setattr(P, "MAX_LIMIT", 4)
    paged = P.group_peptides(rs, group)
    assert paged.total == len(whole) > 4
    assert paged.rows["base_peptide_id"].tolist() == whole["base_peptide_id"].tolist()


@pytest.mark.parametrize("name", ["single", "experiment", "mbr", "topk"])
def test_precursors_of_a_peptide(open_fixture, name):
    rs = open_fixture(name)
    scored = _scored(rs)
    runs = {r.index: r.name for r in rs.runs}
    for group in _groups(rs)[:6]:
        label = _label(group)
        rows = scored[(scored["protein_group"] == group) & (scored["label"] == label)]
        for bp, part in list(rows.groupby("base_peptide_id"))[:4]:
            page = P.peptide_precursors(rs, group, int(bp))
            got = page.rows
            want = {
                (int(s), int(c)) for s, c in zip(part["source"], part["candidate_id"], strict=True)
            }
            assert {
                (int(s), int(c)) for s, c in zip(got["source"], got["candidate_id"], strict=True)
            } == want
            assert page.total == len(want)
            assert set(got["label"]) == {label}
            assert got["score"].is_monotonic_decreasing
            if rs.is_experiment:
                assert all(got["run"] == got["source"].map(runs))


# --------------------------------------------------------------------------- search


@pytest.mark.parametrize("name", ["single", "experiment"])
@pytest.mark.parametrize("text", ["fixt0", "FIX1", "fix14_test", "nothing-like-this"])
def test_search_finds_groups_by_substring(open_fixture, name, text):
    rs = open_fixture(name)
    scored = _scored(rs)
    targets = scored[scored["label"] == "target"]
    want = sorted({g for g in targets["protein_group"] if text.lower() in g.lower()})
    page = P.find_groups(rs, text, threshold=0.1)
    assert page.total == len(want)
    assert sorted(page.rows["protein_group"]) == want
    assert page.rows["score"].is_monotonic_decreasing
    best = targets.groupby("protein_group")["score"].max()
    for g, s in zip(page.rows["protein_group"], page.rows["score"], strict=True):
        assert s == pytest.approx(float(best[g]), abs=0)


def test_search_without_text_lists_the_passing_groups(open_fixture):
    rs = open_fixture("single")
    scored = _scored(rs)
    targets = scored[scored["label"] == "target"]
    for t in (0.01, 0.1):
        want = {
            g for g, q in targets.groupby("protein_group")["pg_q_value"].min().items() if q <= t
        }
        page = P.find_groups(rs, "  ", threshold=t, limit=5)
        assert page.total == len(want)
        assert set(page.rows["protein_group"]) <= want and len(page.rows) == min(5, len(want))
    with pytest.raises(ViewerError):
        P.find_groups(rs, "x", limit=0)


# --------------------------------------------------------------------------- quantities


@pytest.mark.parametrize("name", FIXTURES)
def test_quantity_per_run_is_the_engine_tables(open_fixture, name):
    rs = open_fixture(name)
    transfers = _transfers(rs)
    lfq = None
    if rs.is_experiment:
        lfq = pq.read_table(rs.root / "lfq_maxlfq.parquet").to_pandas()
    for group in _groups(rs):
        df = P.quant_by_run(rs, group)
        assert df["run"].tolist() == [r.name for r in rs.runs]
        for run in rs.runs:
            got = df[df["source"] == run.index].iloc[0]
            pg = _protein_quant(run)
            hit = pg[pg["protein_group"] == group]
            if not len(hit):
                assert got["state"] == "not_selected" and math.isnan(got["quantity"])
            else:
                want = hit.iloc[0]
                if pd.isna(want["quantity"]):
                    assert math.isnan(got["quantity"]) and got["state"] == "not_quantifiable"
                else:
                    assert got["quantity"] == pytest.approx(float(want["quantity"]), rel=1e-12)
                    assert got["state"] == "quantified"
                assert got["n_peptides"] == int(want["n_peptides"])
                assert got["quant_status"] == want["quant_status"]
            if lfq is not None:
                cell = lfq[(lfq["protein_group"] == group) & (lfq["run"] == run.index)]
                if len(cell) and float(cell["quantity"].iloc[0]) != 0.0:
                    assert got["lfq"] == pytest.approx(float(cell["quantity"].iloc[0]), rel=1e-12)
                    assert got["lfq_n_features"] == int(cell["n_features"].iloc[0])
                else:
                    assert pd.isna(got["lfq"])
            if transfers:
                q = _peptide_quant(run)
                q = q[(q["protein_group"] == group) & (q["quantity"] > 0)]
                n = sum((run.index, int(c)) in transfers for c in q["candidate_id"])
                assert got["n_transferred"] == n
        assert "quantity" in df.attrs["labels"]
    if not rs.is_experiment:
        assert "lfq" not in df.columns


def _reference_matrix(rs, group: str, t: float) -> dict[tuple[int, int], dict]:
    scored = _scored(rs)
    rows = scored[(scored["protein_group"] == group) & (scored["label"] == _label(group))]
    transfers = _transfers(rs)
    cells: dict[tuple[int, int], dict] = {}
    for (src, bp), part in rows.groupby(["source", "base_peptide_id"]):
        best = part.sort_values(["score", "rn"], ascending=[False, True]).iloc[0]
        cells[(int(src), int(bp))] = {
            "n_rows": len(part),
            "best_run_psm_q": float(part["run_psm_q"].min()),
            "n_identified": int((part["run_psm_q"] <= t).sum()),
            "best_cid": int(best["candidate_id"]),
            "n_transferred": sum((int(src), int(c)) in transfers for c in part["candidate_id"]),
        }
    for run in rs.runs:
        q = _peptide_quant(run)
        q = q[q["protein_group"] == group]
        for bp, part in q.groupby("base_peptide_id"):
            cell = cells.setdefault((run.index, int(bp)), {})
            pos = part[(part["quantity"] > 0) & np.isfinite(part["quantity"])]
            cell["n_quant_rows"] = len(part)
            cell["n_quantified"] = len(pos)
            if len(pos):
                top = pos.loc[pos["quantity"].idxmax()]
                cell["quantity"] = float(top["quantity"])
                cell["quantity_cid"] = int(top["candidate_id"])
                cell["quantity_from_transfer"] = (run.index, int(top["candidate_id"])) in transfers
    return cells


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("t", [0.01, 0.1])
def test_peptide_run_matrix_equals_the_reference(open_fixture, name, t):
    rs = open_fixture(name)
    for group in _groups(rs):
        m = P.peptide_run_matrix(rs, group, t)
        peptides = P.group_peptides(rs, group).rows
        assert len(m) == len(peptides) * len(rs.runs)
        # Peptides in the peptide table's order, the runs in run order within each.
        assert (
            m["base_peptide_id"].drop_duplicates().tolist() == peptides["base_peptide_id"].tolist()
        )
        assert m["run"].tolist()[: len(rs.runs)] == [r.name for r in rs.runs]
        ref = _reference_matrix(rs, group, t)
        for row in m.itertuples(index=False):
            want = ref.get((int(row.source), int(row.base_peptide_id)), {})
            assert row.n_rows == want.get("n_rows", 0)
            assert row.n_identified == want.get("n_identified", 0)
            assert row.identified == (want.get("n_identified", 0) > 0)
            if "best_run_psm_q" in want:
                assert row.best_run_psm_q == pytest.approx(want["best_run_psm_q"], abs=0)
                assert row.best_cid == want["best_cid"]
            else:
                assert math.isnan(row.best_run_psm_q) and pd.isna(row.best_cid)
            assert row.n_quant_rows == want.get("n_quant_rows", 0)
            assert row.n_quantified == want.get("n_quantified", 0)
            if "quantity" in want:
                assert row.quantity == pytest.approx(want["quantity"], rel=1e-12)
                assert row.quantity_cid == want["quantity_cid"]
                assert row.quantity_from_transfer == want["quantity_from_transfer"]
            else:
                assert math.isnan(row.quantity) and not row.in_rollup
            assert row.n_transferred == want.get("n_transferred", 0)
        assert list(m.columns) == P.MATRIX_COLUMNS
        assert "viewer-derived" in m.attrs["labels"]["quantity"]


@pytest.mark.parametrize("name", FIXTURES)
def test_rollup_gives_the_engine_protein_quantity(open_fixture, name):
    """The top-N sum of the per-peptide maxima is protein_group_quant.quantity."""
    rs = open_fixture(name)
    checked = 0
    for group in _groups(rs):
        if group.startswith("DECOY_"):
            continue
        m = P.peptide_run_matrix(rs, group, 0.01)
        assert m.attrs["rollup"] == "TopNSum" and m.attrs["top_n"] == 3
        for run in rs.runs:
            part = m[m["source"] == run.index]
            pg = _protein_quant(run)
            hit = pg[pg["protein_group"] == group]
            quantified = part["quantity"].notna()
            if not len(hit):
                assert not quantified.any()
                continue
            assert int(quantified.sum()) == int(hit["n_peptides"].iloc[0])
            assert int(part["in_rollup"].sum()) == min(3, int(quantified.sum()))
            total = float(part.loc[part["in_rollup"], "quantity"].sum())
            assert total == pytest.approx(float(hit["quantity"].iloc[0]), rel=1e-9)
            # The rollup takes the largest per-peptide quantities.
            if part["in_rollup"].any() and (quantified & ~part["in_rollup"]).any():
                assert (
                    part.loc[part["in_rollup"], "quantity"].min()
                    >= part.loc[quantified & ~part["in_rollup"], "quantity"].max()
                )
            checked += 1
    assert checked


def test_transfers_are_marked_in_the_matrix(open_fixture):
    rs = open_fixture("mbr")
    marked = 0
    for group in _groups(rs):
        m = P.peptide_run_matrix(rs, group, 0.01)
        marked += int(m["n_transferred"].sum())
        assert m.attrs["mbr"] is True
    scored = _scored(rs)
    rows = {(int(s), int(c)) for s, c in zip(scored["source"], scored["candidate_id"], strict=True)}
    assert marked == len(_transfers(rs) & rows) > 0


def test_matrix_refuses_a_bad_threshold(open_fixture):
    with pytest.raises(ViewerError):
        P.peptide_run_matrix(open_fixture("single"), "sp|FIXT01|FIX01_TEST", 0.0)
