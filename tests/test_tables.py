"""identification_table against an independent pandas computation on the fixtures.

The brute force reads the (small) fixture tables whole with pyarrow and applies the
table rules in pandas. Precursor rows are the scored rows. For the peptide and
protein-group tables the filters select scored rows first; a key is listed when one of
its rows passes, and the row shown is the first passing row under "grouped q below 1.0
first, then score DESC, decoy first, file order". The winning row of a group is its row
below 1.0, or the rule's first row when no row is below 1.0. Counts are counts of
distinct keys; the quant state comes from each run's peptide_quant joined on
``(source, candidate_id)``.

Synthetic copies (in ``tmp_path``) cover what no fixture shows: a base peptide won by a
decoy, exact target/decoy and spike-in score ties, and match-between-runs tables whose
q values were lowered on transferred rows.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data import tables as T
from mumdia_viewer.data.counts import group_winners, unit_counts
from mumdia_viewer.data.duck import DuckDB, sql_path
from mumdia_viewer.data.entrapment import (
    EntrapmentSettings,
    count_classes,
    entrapment_expr_positional,
    spike_in_condition,
)
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.hashing import blake3_file
from mumdia_viewer.data.quant import quant_gate, quant_state, quant_states
from mumdia_viewer.data.tables import TableQuery, identification_table

FIXTURES = ["single", "experiment", "mbr", "topk"]
UNITS = ["precursor", "peptide", "protein_group"]
STATES = ("quantified", "not_quantifiable", "not_selected")
TIEBREAK = {
    "precursor": ["source", "candidate_id"],
    "peptide": ["base_peptide_id", "label"],
    "protein_group": ["protein_group", "label"],
}
UNIT_Q = {"precursor": "precursor_q", "peptide": "peptide_q_value", "protein_group": "pg_q_value"}
KEY = {"peptide": "base_peptide_id", "protein_group": "protein_group"}


def _strip(peptidoform: str) -> str:
    text = re.sub(r"^DECOY_", "", peptidoform)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    return re.sub(r"[^A-Za-z]", "", text)


def _rewrite(path: Path, change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Rewrite a parquet file of a tmp copy through pandas, keeping its schema."""
    table = pq.read_table(path)
    df = change(table.to_pandas())
    pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), path)


def _copy(tmp_path: Path, fixture_dir, name: str) -> Path:
    out = tmp_path / name
    shutil.copytree(fixture_dir(name), out)
    return out


class Brute:
    """The expected tables, computed in pandas from whole-file reads."""

    def __init__(self, rs) -> None:
        self.rs = rs
        s = pq.read_table(rs.scored.path).to_pandas()
        s["file_row_number"] = np.arange(len(s), dtype=np.int64)
        s["source"] = s["source"].astype(np.int64)
        names = {int(r.index): r.name for r in rs.runs}
        s["run"] = s["source"].map(names)
        self.scored = s
        parts = []
        for run in rs.runs:
            art = run.artifact("peptide_quant")
            d = pq.read_table(art.path, columns=["candidate_id", "quantity", "quant_status"])
            d = d.to_pandas()
            d["source"] = int(run.index)
            parts.append(d)
        self.pq = pd.concat(parts, ignore_index=True)
        self.pq["candidate_id"] = self.pq["candidate_id"].astype(np.int64)
        tr_art = rs.artifact("mbr_transferred")
        self.mbr = tr_art is not None
        if self.mbr:
            tr = pq.read_table(tr_art.path, columns=["source", "candidate_id", "transfer_q"])
            tr = tr.to_pandas().astype({"source": np.int64, "candidate_id": np.int64})
            self.tr = tr
        self.experiment = rs.is_experiment
        if not self.experiment:
            art = rs.runs[0].artifact("protein_group_quant")
            self.pgq = pq.read_table(art.path).to_pandas()

    # ------------------------------------------------------------------ relations

    def _with_quant(self, df: pd.DataFrame) -> pd.DataFrame:
        q = self.pq.assign(_has=True)
        m = df.astype({"candidate_id": np.int64}).merge(
            q, on=["source", "candidate_id"], how="left"
        )
        m["quant_state"] = np.where(
            m["_has"].isna(),
            "not_selected",
            np.where(m["quantity"].isna(), "not_quantifiable", "quantified"),
        )
        return m.drop(columns="_has")

    def _transferred(self, df: pd.DataFrame) -> pd.Series:
        if not self.mbr:
            return pd.Series(False, index=df.index)
        keys = set(zip(self.tr["source"], self.tr["candidate_id"], strict=True))
        return pd.Series(
            [
                (int(a), int(b)) in keys
                for a, b in zip(df["source"], df["candidate_id"], strict=True)
            ],
            index=df.index,
        )

    def precursor(self) -> pd.DataFrame:
        m = self._with_quant(self.scored)
        if self.mbr:
            m = m.merge(self.tr, on=["source", "candidate_id"], how="left")
            m["is_transferred"] = self._transferred(m)
        return m

    def winner_rows(self, key: str, q_column: str) -> set[int]:
        """File rows of the winning row of each group.

        The row whose grouped q is below 1.0; for a group without one, the first row
        under score DESC, decoy first, file order.
        """
        s = self.scored
        below = s[s[q_column] < 1.0]
        rows = set(below["file_row_number"])
        capped = s[~s[key].isin(set(below[key]))]
        if len(capped):
            ranked = capped.assign(_d=capped["label"] == "decoy").sort_values(
                ["score", "_d", "file_row_number"], ascending=[False, False, True], kind="mergesort"
            )
            rows |= set(ranked.drop_duplicates(key)["file_row_number"])
        return rows

    def _runs(self, key: str, t: float | None) -> pd.DataFrame:
        s = self.scored
        native = (s["run_psm_q"] <= t) if t is not None else pd.Series(True, index=s.index)
        x = s.assign(native=native, tr=self._transferred(s))
        per = x.groupby([key, "label", "source"]).agg(native=("native", "any"), tr=("tr", "any"))
        per = per.reset_index()
        n_runs = per[per["native"]].groupby([key, "label"]).size().rename("n_runs")
        only = per[per["tr"] & ~per["native"]].groupby([key, "label"]).size()
        out = pd.concat([n_runs, only.rename("n_runs_transfer_only")], axis=1).fillna(0)
        return out.reset_index()

    def grouped_rows(self, unit: str, t: float | None) -> pd.DataFrame:
        """Every scored row with the per-group columns of the peptide or protein table."""
        key = KEY[unit]
        s = self.scored
        if unit == "peptide":
            n = (
                s.drop_duplicates([key, "label", "peptidoform", "charge"])
                .groupby([key, "label"])
                .size()
                .rename("n_precursors")
                .reset_index()
            )
        else:
            sel = s if t is None else s[s["peptide_q_value"] <= t]
            n = (
                sel.groupby([key, "label"])["base_peptide_id"]
                .nunique()
                .rename("n_peptides")
                .reset_index()
            )
        w = s.merge(n, on=[key, "label"], how="left")
        if unit == "protein_group":
            w["n_peptides"] = w["n_peptides"].fillna(0)
        if self.experiment:
            w = w.merge(self._runs(key, t), on=[key, "label"], how="left")
            w[["n_runs", "n_runs_transfer_only"]] = w[["n_runs", "n_runs_transfer_only"]].fillna(0)
        elif unit == "peptide":
            w = self._with_quant(w)
        else:
            g = self.pgq.rename(columns={"n_peptides": "quant_n_peptides"}).assign(_has=True)
            w = w.merge(g, on="protein_group", how="left")
            w["quant_state"] = np.where(
                w["_has"].isna(),
                "not_selected",
                np.where(w["quantity"].isna(), "not_quantifiable", "quantified"),
            )
            w = w.drop(columns="_has")
        if unit == "peptide":
            w["sequence"] = w["peptidoform"].map(_strip)
        w["is_winner"] = w["file_row_number"].isin(self.winner_rows(key, UNIT_Q[unit]))
        return w

    # ------------------------------------------------------------------ query

    def page(self, query: TableQuery) -> tuple[pd.DataFrame, int, str]:
        run_idx = None
        if query.run is not None and self.experiment:
            run_idx = int(self.rs.run(query.run).index)
        q_col = query.q_column
        if q_col is None:
            q_col = "run_psm_q" if run_idx is not None else UNIT_Q[query.unit]
        t = query.threshold
        df = self.precursor() if query.unit == "precursor" else self.grouped_rows(query.unit, t)
        if t is not None:
            df = df[df[q_col] <= t]
        if not query.include_decoys:
            df = df[df["label"] == "target"]
        if run_idx is not None:
            df = df[df["source"] == run_idx]
        if query.charge is not None:
            df = df[df["charge"] == query.charge]
        if query.protein:
            df = df[df["protein"].str.lower().str.contains(query.protein.lower(), regex=False)]
        if query.modification:
            df = df[df["peptidoform"].str.contains(query.modification, regex=False)]
        if query.search:
            needle = query.search.lower()
            df = df[
                df["peptidoform"].str.lower().str.contains(needle, regex=False)
                | df["protein"].str.lower().str.contains(needle, regex=False)
            ]
        if query.quant_status:
            col = "quant_state" if query.quant_status in STATES else "quant_status"
            df = df[df[col] == query.quant_status]
        if query.unit != "precursor":
            uq = UNIT_Q[query.unit]
            df = (
                df.assign(_w=df[uq] < 1.0, _d=df["label"] == "decoy")
                .sort_values(
                    ["_w", "score", "_d", "file_row_number"],
                    ascending=[False, False, False, True],
                    kind="mergesort",
                )
                .drop_duplicates(KEY[query.unit])
                .drop(columns=["_w", "_d"])
            )
        by = [query.sort_by] + [c for c in TIEBREAK[query.unit] if c != query.sort_by]
        asc = [not query.descending] + [True] * (len(by) - 1)
        df = df.sort_values(by, ascending=asc, na_position="last", kind="mergesort")
        total = len(df)
        return df.iloc[query.offset : query.offset + query.limit], total, q_col


def _compare(rs, brute: Brute, query: TableQuery) -> T.TablePage:
    page = identification_table(rs, query)
    expected, total, q_col = brute.page(query)
    assert page.total == total, query
    assert page.q_column == q_col
    exp = expected[list(page.rows.columns)].reset_index(drop=True)
    got = page.rows.reset_index(drop=True)
    for col in ("n_runs", "n_runs_transfer_only", "n_peptides", "n_precursors"):
        if col in got.columns:
            got[col] = got[col].astype("float64")
            exp[col] = exp[col].astype("float64")
    for frame in (got, exp):
        for col in frame.columns:
            if frame[col].dtype == object or pd.api.types.is_string_dtype(frame[col]):
                values = frame[col].astype(object)
                frame[col] = values.where(values.notna(), None)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False, obj=repr(query))
    return page


def _queries(rs, unit: str) -> list[TableQuery]:
    """Every filter, sorts in both directions, pages and the q column choices."""
    base = [
        TableQuery(unit=unit),
        TableQuery(unit=unit, threshold=0.05),
        TableQuery(unit=unit, threshold=None),
        TableQuery(unit=unit, threshold=1.0),
        TableQuery(unit=unit, threshold=0.0),
        TableQuery(unit=unit, threshold=None, include_decoys=True),
        TableQuery(unit=unit, threshold=0.05, include_decoys=True),
        TableQuery(unit=unit, charge=2),
        TableQuery(unit=unit, charge=3, threshold=None),
        TableQuery(unit=unit, charge=2, threshold=None, include_decoys=True),
        TableQuery(unit=unit, protein="fix1", threshold=None),
        TableQuery(unit=unit, protein="FIXT0", threshold=None, include_decoys=True),
        TableQuery(unit=unit, modification="Oxidation", threshold=None),
        TableQuery(unit=unit, modification="oxidation", threshold=None),
        TableQuery(unit=unit, search="yk", threshold=None),
        TableQuery(unit=unit, search="FIX05", threshold=0.05),
        TableQuery(unit=unit, offset=7, limit=5),
        TableQuery(unit=unit, offset=10_000, limit=5),
        TableQuery(unit=unit, limit=0),
        TableQuery(unit=unit, q_column="q_value"),
        TableQuery(unit=unit, q_column="q_value", threshold=None, include_decoys=True),
        TableQuery(unit=unit, q_column="run_psm_q", threshold=0.004),
        TableQuery(unit=unit, q_column="precursor_q", threshold=0.02),
        TableQuery(unit=unit, q_column="global_q_value", include_decoys=True),
    ]
    columns = list(identification_table(rs, TableQuery(unit=unit, limit=1)).rows.columns)
    sortable = [
        "score",
        "apex_rt",
        "peptidoform",
        "candidate_id",
        "label",
        "charge",
        "q_value",
        "quantity",
        "quant_state",
        "n_runs",
        "n_precursors",
        "n_peptides",
        "is_winner",
        "is_transferred",
        "transfer_q",
        "sequence",
        "run",
        "protein",
    ]
    for col in sortable:
        if col in columns:
            base.append(TableQuery(unit=unit, threshold=None, sort_by=col, limit=40))
            base.append(
                TableQuery(
                    unit=unit, threshold=None, include_decoys=True, sort_by=col, descending=False
                )
            )
    if "quant_state" in columns:
        for status in (*STATES, "no_fragment_traces"):
            base.append(TableQuery(unit=unit, threshold=None, quant_status=status, limit=500))
    if rs.is_experiment and unit == "precursor":
        for run in rs.runs:
            base.append(TableQuery(unit=unit, run=run.name))
            base.append(TableQuery(unit=unit, run=int(run.index), include_decoys=True, limit=500))
            base.append(TableQuery(unit=unit, run=run.name, q_column="q_value", sort_by="apex_rt"))
    return base


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("name", FIXTURES)
def test_table_matches_brute_force(open_fixture, name, unit):
    rs = open_fixture(name)
    brute = Brute(rs)
    for query in _queries(rs, unit):
        _compare(rs, brute, query)


@pytest.mark.parametrize("path", ["wide", "order", "direct"])
@pytest.mark.parametrize("name", FIXTURES)
def test_materialised_pages_match_brute_force(fixture_dir, monkeypatch, name, path):
    """The large-table paths give the same pages as the brute force.

    ``wide`` materialises the whole result, ``order`` only its order (the rows of a
    page are then read by row number) and ``direct`` runs the query per page because
    the order table would exceed the cell budget.
    """
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    if path in ("order", "direct"):
        monkeypatch.setattr(T, "MAX_MATERIALISED_ROWS", 0)
    if path == "direct":
        monkeypatch.setattr(T, "MAX_CACHED_CELLS", 0)
    rs = open_results(fixture_dir(name))
    brute = Brute(rs)
    for unit in UNITS:
        queries = _queries(rs, unit)
        for query in [*queries[:14], *queries[-6:]]:
            page = _compare(rs, brute, query)
            if path == "direct" and page.total:
                assert "more than the table cache holds" in page.description
    cache = T._cache(rs)
    kinds = {k[0] for k in cache.entries}
    if path == "wide":
        assert kinds == {"filtered"}
    elif path == "order":
        assert "order" in kinds
    else:
        assert not cache.entries or kinds == {"filtered"}  # only empty results are cached
    names = {e.name for e in cache.entries.values()}
    tables = {r[0] for r in rs.duck.rows("SELECT table_name FROM duckdb_tables()")}
    assert names <= tables
    T.drop_table_cache(rs)
    tables = {r[0] for r in rs.duck.rows("SELECT table_name FROM duckdb_tables()")}
    assert not names & tables


ORDER_SORTS = {
    "precursor": ("score", "peptidoform", "apex_rt", "run", "quant_state", "is_transferred"),
    "peptide": ("score", "sequence", "n_precursors", "n_runs", "is_winner", "run"),
    "protein_group": ("score", "protein_group", "n_peptides", "n_runs", "is_winner"),
}


@pytest.mark.parametrize("name", ["experiment", "mbr"])
def test_order_pages_at_every_offset(fixture_dir, monkeypatch, name):
    """Paging through an order table returns every row once, in the brute-force order.

    The sorts include the per-group columns, which the order table computes over all
    keys while a page computes them for its own keys only.
    """
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    monkeypatch.setattr(T, "MAX_MATERIALISED_ROWS", 0)
    rs = open_results(fixture_dir(name))
    brute = Brute(rs)
    for unit in UNITS:
        columns = identification_table(rs, TableQuery(unit=unit, limit=0)).rows.columns
        for sort_by in ORDER_SORTS[unit]:
            if sort_by not in columns:
                continue
            for descending in (True, False):
                query = TableQuery(
                    unit=unit,
                    threshold=None,
                    include_decoys=True,
                    sort_by=sort_by,
                    descending=descending,
                )
                total = brute.page(query)[1]
                for offset in range(0, total + 60, 60):
                    _compare(rs, brute, replace(query, offset=offset, limit=60))
    assert any(k[0] == "order" for k in T._cache(rs).entries)


def test_materialised_table_is_reused_for_pages_and_sorts(fixture_dir, monkeypatch):
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    rs = open_results(fixture_dir("experiment"))
    identification_table(rs, TableQuery(threshold=None))
    n = len(T._cache(rs).entries)
    identification_table(rs, TableQuery(threshold=None, offset=50))
    identification_table(rs, TableQuery(threshold=None, sort_by="apex_rt", descending=False))
    assert len(T._cache(rs).entries) == n
    identification_table(rs, TableQuery(threshold=0.05))
    assert len(T._cache(rs).entries) == n + 1


def _spy_creates(monkeypatch) -> list[str]:
    """Record the kind of every table the cache creates."""
    created: list[str] = []
    original = T._TableCache.create

    def spy(self, rs, key, query, n_columns):
        if self.entries.get(key) is None:
            created.append(key[0])
        return original(self, rs, key, query, n_columns)

    monkeypatch.setattr(T._TableCache, "create", spy)
    return created


def test_cache_hit_creates_no_table(fixture_dir, monkeypatch):
    """A page of a cached result reads that one table and builds nothing (review issue 3)."""
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    monkeypatch.setattr(T, "MAX_CACHED_TABLES", 4)
    created = _spy_creates(monkeypatch)
    rs = open_results(fixture_dir("experiment"))
    identification_table(rs, TableQuery(unit="peptide"))
    assert created == ["filtered"]
    created.clear()
    identification_table(rs, TableQuery(unit="peptide", offset=50))
    identification_table(rs, TableQuery(unit="protein_group", threshold=None))
    assert created == ["filtered"]
    created.clear()
    identification_table(rs, TableQuery(unit="peptide", offset=100))
    identification_table(rs, TableQuery(unit="protein_group", threshold=None, sort_by="label"))
    assert created == []
    monkeypatch.setattr(T, "MAX_MATERIALISED_ROWS", 0)
    identification_table(rs, TableQuery(unit="peptide", threshold=None))
    assert created == ["order"]
    created.clear()
    identification_table(rs, TableQuery(unit="peptide", threshold=None, offset=40))
    assert created == []


@pytest.mark.parametrize("cells", [0, 1, 100])
def test_cache_budget_never_breaks_a_query(fixture_dir, monkeypatch, cells):
    """A cell budget below the size of one table still answers every query (review issue 2).

    Each cached table is self-contained, so the eviction that follows a new table can
    never remove a table that the running query reads.
    """
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    monkeypatch.setattr(T, "MAX_CACHED_CELLS", cells)
    monkeypatch.setattr(T, "MAX_CACHED_TABLES", 1)
    rs = open_results(fixture_dir("experiment"))
    brute = Brute(rs)
    for unit in UNITS:
        for query in (
            TableQuery(unit=unit, threshold=None),
            TableQuery(unit=unit, threshold=None, include_decoys=True, offset=5, limit=20),
            TableQuery(unit=unit),
            TableQuery(unit=unit, threshold=None, sort_by="apex_rt"),
        ):
            _compare(rs, brute, query)
            assert len(T._cache(rs).entries) <= 1


def test_cache_eviction_drops_tables(fixture_dir, monkeypatch):
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    monkeypatch.setattr(T, "MAX_CACHED_TABLES", 2)
    rs = open_results(fixture_dir("single"))
    for t in (0.01, 0.02, 0.03, 0.04):
        identification_table(rs, TableQuery(threshold=t))
    cache = T._cache(rs)
    assert len(cache.entries) <= 2
    tables = {r[0] for r in rs.duck.rows("SELECT table_name FROM duckdb_tables()")}
    assert {e.name for e in cache.entries.values()} == tables


@pytest.mark.parametrize("name", FIXTURES)
def test_counts_equal_engine_report(open_fixture, name):
    """At 0.01 the table totals equal the rescore report's own counts."""
    rs = open_fixture(name)
    stats = rs.scored.report.stats
    for unit, stat in (
        ("precursor", "target_precursors_at_1pct"),
        ("peptide", "target_peptides_at_1pct"),
        ("protein_group", "target_protein_groups_at_1pct"),
    ):
        assert identification_table(rs, TableQuery(unit=unit, limit=0)).total == stats[stat]


def _distinct_keys(duck, path: Path, key: str, where: str) -> int:
    sql = f"SELECT count(DISTINCT {key}) FROM read_parquet(?) WHERE label = 'target' AND {where}"
    return int(duck.scalar(sql, [sql_path(path)]))


@pytest.mark.parametrize("name", FIXTURES)
def test_grouped_totals_count_distinct_keys(open_fixture, name):
    """Every total is COUNT(DISTINCT key) over the passing target rows (review issue 1).

    That includes a threshold of 1.0 and no q filter on the sparse unit column, and the
    other q columns (CRITIC V-5).
    """
    rs = open_fixture(name)
    for unit in ("peptide", "protein_group"):
        key = KEY[unit]
        cases = [
            (UNIT_Q[unit], 1.0),
            (UNIT_Q[unit], None),
            (UNIT_Q[unit], 0.02),
            ("q_value", 0.01),
            ("run_psm_q", 0.004),
            ("precursor_q", 0.02),
        ]
        for column, t in cases:
            page = identification_table(
                rs, TableQuery(unit=unit, q_column=column, threshold=t, limit=0)
            )
            where = "true" if t is None else f"{column} <= {t!r}"
            assert page.total == _distinct_keys(rs.duck, rs.scored.path, key, where), (column, t)


def _decoy_won_copy(tmp_path: Path, fixture_dir) -> Path:
    """The single fixture where a decoy wins the HHALPAR base peptide (id 222).

    Decoy candidate 1998 joins base peptide 222 with a higher score and takes the group
    q (0.005); the former target winner (candidate 4) gets 1.0, as the engine writes.
    Candidate 1983, left alone in base peptide 296, becomes that group's winner.
    """
    out = _copy(tmp_path, fixture_dir, "single")

    def change(df: pd.DataFrame) -> pd.DataFrame:
        cid = df["candidate_id"]
        df.loc[cid == 1998, ["base_peptide_id", "score", "peptide_q_value"]] = [222, 5.0, 0.005]
        df.loc[cid == 4, "peptide_q_value"] = 1.0
        df.loc[cid == 1983, "peptide_q_value"] = 0.02
        return df

    _rewrite(out / "psms_scored.parquet", change)
    return out


def test_peptide_won_by_a_decoy(tmp_path, fixture_dir):
    """A target base peptide whose winner is a decoy stays in the table without a q filter.

    The old rule (filter the winning rows) dropped it at threshold None and 1.0.
    """
    rs = open_results(_decoy_won_copy(tmp_path, fixture_dir))
    brute = Brute(rs)
    for query in _queries(rs, "peptide"):
        _compare(rs, brute, query)
    scored = pq.read_table(rs.scored.path).to_pandas()
    targets = scored[scored["label"] == "target"]
    for t in (None, 1.0):
        page = identification_table(
            rs, TableQuery(unit="peptide", threshold=t, sort_by="base_peptide_id", limit=1000)
        )
        assert page.total == targets["base_peptide_id"].nunique() == 151
        row = page.rows[page.rows["base_peptide_id"] == 222].iloc[0]
        assert int(row["candidate_id"]) == 4 and not row["is_winner"]
        assert row["peptide_q_value"] == 1.0
        assert "On 1 of the 151 rows the group's winning row does not pass" in page.description
    accepted = identification_table(rs, TableQuery(unit="peptide", limit=1000))
    assert 222 not in set(accepted.rows["base_peptide_id"])
    assert (
        accepted.total
        == targets.loc[targets["peptide_q_value"] <= 0.01, "base_peptide_id"].nunique()
    )
    both = identification_table(
        rs, TableQuery(unit="peptide", threshold=None, include_decoys=True, limit=1000)
    )
    row = both.rows[both.rows["base_peptide_id"] == 222].iloc[0]
    assert int(row["candidate_id"]) == 1998 and row["is_winner"] and row["label"] == "decoy"
    assert both.rows["is_winner"].all()
    # The quant reason of the losing target sibling names the decoy winner.
    reason = quant_state(rs, None, 138).reason
    assert "the winning row of this base peptide is a decoy row (candidate 1998)" in reason


def _tie_copy(tmp_path: Path, fixture_dir) -> Path:
    """The single fixture with an exact target/decoy score tie in base peptide 222.

    Decoy candidate 1998 (file row 153) gets the score of target candidate 4 (file row
    0). Every row of the group holds peptide_q_value 1.0, so no row marks the winner and
    the engine rule decides: a decoy wins an exact tie, before file order.
    """
    out = _copy(tmp_path, fixture_dir, "single")

    def change(df: pd.DataFrame) -> pd.DataFrame:
        cid = df["candidate_id"]
        score = float(df.loc[cid == 4, "score"].iloc[0])
        df.loc[cid == 1998, ["base_peptide_id", "score"]] = [222, score]
        df.loc[cid.isin([4, 138, 1998]), "peptide_q_value"] = 1.0
        df.loc[cid == 1983, "peptide_q_value"] = 0.02
        return df

    _rewrite(out / "psms_scored.parquet", change)
    return out


def test_decoy_wins_an_exact_tie(tmp_path, fixture_dir):
    """T1 F4.1: on an exact score tie a decoy replaces a target (no fixture has a tie)."""
    rs = open_results(_tie_copy(tmp_path, fixture_dir))
    path = rs.scored.path
    rows = pq.read_table(path, columns=["candidate_id"]).column(0).to_pylist()
    winners = _winner_rows(rs.duck, path, "w.base_peptide_id", entrapment=False)
    assert rows.index(1998) in winners and rows.index(4) not in winners
    both = identification_table(
        rs, TableQuery(unit="peptide", threshold=None, include_decoys=True, limit=1000)
    )
    row = both.rows[both.rows["base_peptide_id"] == 222].iloc[0]
    assert int(row["candidate_id"]) == 1998 and row["is_winner"]
    targets = identification_table(rs, TableQuery(unit="peptide", threshold=None, limit=1000))
    row = targets.rows[targets.rows["base_peptide_id"] == 222].iloc[0]
    assert int(row["candidate_id"]) == 4 and not row["is_winner"]
    # The search filter selects rows: only the HHALPAR rows pass, so the target is shown.
    found = identification_table(
        rs, TableQuery(unit="peptide", threshold=None, include_decoys=True, search="HHALPAR")
    )
    row = found.rows[found.rows["base_peptide_id"] == 222]
    assert row["candidate_id"].tolist() == [4]
    assert not row["is_winner"].iloc[0]
    brute = Brute(rs)
    for query in _queries(rs, "peptide")[:12]:
        _compare(rs, brute, query)


def test_single_run_units_and_descriptions(open_fixture):
    rs = open_fixture("single")
    page = identification_table(rs, TableQuery())
    assert page.description.startswith("273 precursors (candidate_id, precursor_q <= 0.01)")
    page = identification_table(rs, TableQuery(unit="peptide"))
    assert page.description.startswith(
        "150 peptides (unique base_peptide_id, peptide_q_value <= 0.01), targets only."
    )
    assert "Each row is the winning row of its base peptide" in page.description
    assert "a decoy wins an exact score tie" in page.description
    assert page.column_labels["n_precursors"].startswith("scored precursors of this peptide")
    assert page.rows["is_winner"].all()
    # Under the PeptideQ gate the winning precursor of every accepted peptide is quantified.
    assert (page.rows["quant_state"] == "quantified").all()
    assert "the winning precursor of each accepted base peptide" in page.column_labels["quantity"]
    page = identification_table(rs, TableQuery(unit="protein_group", threshold=None))
    assert page.total == 16
    assert "no q filter" in page.description
    assert "Every row shown is its group's winning row." in page.description
    pgq = pq.read_table(rs.runs[0].artifact("protein_group_quant").path).to_pandas()
    merged = page.rows.merge(pgq, on="protein_group", suffixes=("", "_pgq"))
    assert len(merged) == 16
    assert np.allclose(merged["quantity"], merged["quantity_pgq"])


def test_peptide_winners_and_counts(open_fixture):
    """The HHALPAR base peptide: charge 3 wins, charge 2 is a sibling (1.0)."""
    rs = open_fixture("single")
    page = identification_table(rs, TableQuery(unit="peptide", search="HHALPAR", threshold=None))
    rows = page.rows[page.rows["sequence"] == "HHALPAR"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert int(row["candidate_id"]) == 4 and int(row["charge"]) == 3
    assert int(row["n_precursors"]) == 2
    # Charge 2 passes the charge filter: its row is shown, and it is not the winner.
    page = identification_table(
        rs, TableQuery(unit="peptide", search="HHALPAR", threshold=None, charge=2)
    )
    row = page.rows.iloc[0]
    assert int(row["candidate_id"]) == 138 and not row["is_winner"]
    pre = identification_table(rs, TableQuery(search="HHALPAR", threshold=None))
    siblings = pre.rows[pre.rows["peptidoform"] == "HHALPAR"]
    assert set(siblings["quant_state"]) == {"quantified", "not_selected"}


def test_experiment_descriptions_and_run_rules(open_fixture):
    rs = open_fixture("experiment")
    page = identification_table(rs, TableQuery())
    assert "experiment-wide precursor_q <= 0.01" in page.description
    assert "in the run of its winning row" in page.description
    page = identification_table(rs, TableQuery(run="b"))
    assert page.q_column == "run_psm_q"
    assert page.total == 273
    assert "run b" in page.description
    # The quant gate is named: per-run quant uses the pooled q_value (T2 R10).
    assert (
        "each run's peptide_quant holds the target rows with q_value <= 0.01 (PsmQ: the pooled "
        "PSM-level q, experiment-wide, not run_psm_q; run-experiment forces PsmQ, the "
        "configured PeptideQ is not applied)" in page.description
    )
    assert "(run_psm_q) is not the quant gate column (q_value)" in page.description
    assert page.column_labels["quant_state"].endswith(
        "the row is outside the quant gate, target rows with q_value <= 0.01)"
    )
    page = identification_table(rs, TableQuery(unit="peptide"))
    assert "experiment-wide peptide_q_value" in page.description
    assert "n_runs" in page.column_labels
    assert set(page.rows["n_runs"]) == {2}
    with pytest.raises(ViewerError, match="experiment-wide"):
        identification_table(rs, TableQuery(run="a", q_column="peptide_q_value"))
    with pytest.raises(ViewerError, match="experiment-wide"):
        identification_table(rs, TableQuery(unit="peptide", run="a"))
    with pytest.raises(ViewerError, match="no quant columns"):
        identification_table(rs, TableQuery(unit="protein_group", quant_status="quantified"))
    with pytest.raises(ViewerError):
        identification_table(rs, TableQuery(run="zzz"))


def test_quantity_label_names_the_recorded_gate(tmp_path, fixture_dir):
    """A single run gated on RunPsmQ is not described as PeptideQ (review issue 6)."""
    out = _copy(tmp_path, fixture_dir, "single")
    report = out / "peptide_quant.parquet.report.json"
    data = json.loads(report.read_text())
    data["params"]["q_filter"] = "RunPsmQ"
    report.write_text(json.dumps(data))
    rs = open_results(out)
    page = identification_table(rs, TableQuery(unit="peptide", limit=1))
    label = page.column_labels["quantity"]
    assert "run_psm_q <= 0.01 (RunPsmQ: the PSM-level q within the run)" in label
    assert "PeptideQ" not in label
    assert "peptide_quant holds the target rows with run_psm_q <= 0.01" in page.description


def test_mbr_transfers_are_flagged(open_fixture):
    rs = open_fixture("mbr")
    page = identification_table(rs, TableQuery(threshold=None, include_decoys=True, limit=1000))
    assert page.total == 565
    assert int(page.rows["is_transferred"].sum()) == 48
    assert "(48 of the 565 rows)" in page.description
    by_run = page.rows[page.rows["is_transferred"]].groupby("run").size().to_dict()
    assert by_run == {"a": 16, "c": 32}
    # Every transfer is quantified, and its transfer_q is kept.
    tr = page.rows[page.rows["is_transferred"]]
    assert (tr["quant_state"] == "quantified").all()
    assert np.allclose(tr["transfer_q"], 0.145833, atol=1e-6)
    assert "plus match-between-runs transfers" in page.description
    page = identification_table(rs, TableQuery(unit="peptide", threshold=None))
    assert "n_runs_transfer_only" in page.rows.columns
    assert "n_runs_transfer_only" in page.column_labels


def _lowered_mbr_copy(tmp_path: Path, fixture_dir) -> Path:
    """The mbr fixture with q_value and run_psm_q lowered on the transferred rows.

    MBR writes ``min(q, transfer_q)`` to ``q_value``, ``run_psm_q`` and
    ``experiment_psm_q`` of ``scored_mbr.parquet`` and the per-run split tables (G3
    F7.1). On the fixture transfer_q exceeds the native q, so nothing changed (G3 F7.2);
    here the lowered value is 0.0001.
    """
    out = _copy(tmp_path, fixture_dir, "mbr")

    def lower(df: pd.DataFrame) -> pd.DataFrame:
        moved = df["is_transferred"].fillna(False).astype(bool)
        for col in ("q_value", "run_psm_q", "experiment_psm_q"):
            df.loc[moved, col] = 0.0001
        return df

    _rewrite(out / "scored_mbr.parquet", lower)
    for run in ("a", "c"):
        _rewrite(out / run / "scored.parquet", lower)
    return out


def test_mbr_tables_use_native_q(tmp_path, fixture_dir):
    """The table and the native q of quant_states read scored_combined (review issue 10)."""
    rs = open_results(_lowered_mbr_copy(tmp_path, fixture_dir))
    native = pq.read_table(rs.scored.path).to_pandas()
    transfers = pq.read_table(rs.artifact("mbr_transferred").path).to_pandas()
    moved = native.merge(transfers[["source", "candidate_id"]], on=["source", "candidate_id"])
    assert len(moved) == 48 and (moved["run_psm_q"] > 0.004).any()
    page = identification_table(rs, TableQuery(threshold=None, include_decoys=True, limit=1000))
    shown = page.rows[page.rows["is_transferred"]].merge(
        moved, on=["source", "candidate_id"], suffixes=("", "_native")
    )
    assert len(shown) == 48
    for col in ("q_value", "run_psm_q", "experiment_psm_q"):
        assert np.allclose(shown[col], shown[f"{col}_native"])
    assert (shown["q_value"] > 0.001).all()
    # A per-run q filter at 0.004 admits no transfer whose native run_psm_q fails it.
    for run in ("a", "c"):
        idx = rs.run(run).index
        expected = native[
            (native["source"] == idx)
            & (native["label"] == "target")
            & (native["run_psm_q"] <= 0.004)
        ]
        filtered = identification_table(rs, TableQuery(run=run, threshold=0.004, limit=1000))
        assert filtered.total == len(expected)
        assert set(filtered.rows["candidate_id"]) == set(expected["candidate_id"])
        own = moved[moved["source"] == idx]
        states = quant_states(rs, run, own["candidate_id"].to_numpy())
        assert np.allclose(states["gate_q"], 0.0001)  # the lowered q that quant compared
        assert np.allclose(states["native_q"], own["q_value"])  # scored_combined
        assert states["from_transfer"].all()
        for reason, value in zip(states["reason"], own["q_value"], strict=True):
            assert (
                f"its native q_value {value:.6g} (scored_combined.parquet) fails the gate" in reason
            )
            assert "quantified only through the transfer" in reason


def test_validation_errors(open_fixture):
    rs = open_fixture("single")
    with pytest.raises(ViewerError, match="unknown table unit"):
        identification_table(rs, TableQuery(unit="psm"))  # type: ignore[arg-type]
    with pytest.raises(ViewerError, match="not in the scored table"):
        identification_table(rs, TableQuery(q_column="transfer_q"))
    with pytest.raises(ViewerError, match="cannot sort"):
        identification_table(rs, TableQuery(sort_by="score; DROP TABLE x"))
    with pytest.raises(ViewerError, match="cannot sort"):
        identification_table(rs, TableQuery(sort_by="rn"))
    with pytest.raises(ViewerError, match="no run filter"):
        identification_table(rs, TableQuery(run="a"))
    with pytest.raises(ViewerError, match="limit"):
        identification_table(rs, TableQuery(limit=-1))
    with pytest.raises(ViewerError, match="offset"):
        identification_table(rs, TableQuery(offset=-3))
    with pytest.raises(ViewerError, match="NaN"):
        identification_table(rs, TableQuery(threshold=float("nan")))
    # A run filter of 0 or '' on a single run is accepted.
    assert identification_table(rs, TableQuery(run=0)).total == 273


def _winner_rows(duck: DuckDB, path: Path, key: str, *, entrapment: bool, tie=None) -> set[int]:
    sql = T.winner_sql(key, sql_path(path), entrapment=entrapment, is_entrapment=tie)
    return {int(r[0]) for r in duck.rows(sql.text, sql.params)}


def _q_below_one(duck: DuckDB, path: Path, column: str) -> set[int]:
    rows = duck.rows(
        f"SELECT file_row_number FROM read_parquet(?, file_row_number = true) WHERE {column} < 1",
        [sql_path(path)],
    )
    return {int(r[0]) for r in rows}


@pytest.mark.parametrize("name", FIXTURES)
def test_winner_rule_selects_the_engine_winners(open_fixture, name):
    """Only the engine's winning row of a group carries a grouped q below 1.0."""
    rs = open_fixture(name)
    path = rs.scored.path
    for column, key in (
        ("peptide_q_value", "w.base_peptide_id"),
        ("pg_q_value", "w.protein_group"),
        ("precursor_q", "w.peptidoform, w.charge"),
    ):
        winners = _winner_rows(rs.duck, path, key, entrapment=False)
        assert winners == _q_below_one(rs.duck, path, column), column


def _entrapment_tie(config: dict) -> T.SqlFragment:
    settings = EntrapmentSettings(
        marker=config["entrapment_marker"],
        exclude=config["entrapment_exclude"],
        contaminants=tuple(config["entrapment_contaminant_markers"]),
    )
    text, params = entrapment_expr_positional(settings, alias="w")
    return T.SqlFragment(text, *params)


def test_winner_rule_in_entrapment_mode(fixture_dir):
    """Entrapment mode: decoys do not compete (psms_scored of an entrapment_native rescore)."""
    directory = fixture_dir("entrapment")
    path = directory / "psms_scored.parquet"
    config = json.loads((directory.parent / "config.entrap_mode.json").read_text())["rescore"]
    report = json.loads((directory / "psms_scored.parquet.report.json").read_text())
    assert report["params"]["classifier"] == "entrapment_native"
    tie = _entrapment_tie(config)
    duck = DuckDB()
    try:
        for column, key in (
            ("peptide_q_value", "w.base_peptide_id"),
            ("pg_q_value", "w.protein_group"),
            ("precursor_q", "w.peptidoform, w.charge"),
        ):
            expected = _q_below_one(duck, path, column)
            assert _winner_rows(duck, path, key, entrapment=True, tie=tie) == expected, column
        # The decoy-mode rule (decoys compete) gives other peptide winners here.
        decoy_rule = _winner_rows(duck, path, "w.base_peptide_id", entrapment=False)
        assert decoy_rule != _q_below_one(duck, path, "peptide_q_value")
        # The report counts real targets (not entrapment) and entrapment peptides at 0.01.
        count = (
            "SELECT count(DISTINCT base_peptide_id) FROM read_parquet(?) w "
            "WHERE w.label = 'target' AND w.peptide_q_value <= 0.01 AND "
        )
        real = duck.scalar(count + "NOT " + tie.text, [sql_path(path), *tie.params])
        spike = duck.scalar(count + tie.text, [sql_path(path), *tie.params])
        assert real == report["stats"]["target_peptides_at_1pct"]
        assert spike == report["stats"]["entrapment_peptides_at_1pct"]
    finally:
        duck.close()


def _entrapment_result(tmp_path: Path, fixture_dir):
    """The entrapment fixture with a minimal manifest.json (it was written without one)."""
    src = fixture_dir("entrapment")
    config = json.loads((src.parent / "config.entrap_mode.json").read_text())
    report = json.loads((src / "psms_scored.parquet.report.json").read_text())
    out = tmp_path / "entrap_mode"
    out.mkdir()
    shutil.copy2(src / "psms_scored.parquet", out)
    shutil.copy2(src / "psms_scored.parquet.report.json", out)
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", "/recorded/entrap_mode"],
        "config_json": json.dumps(config),
        "artifacts": {
            "psms_scored": {
                "path": "/recorded/entrap_mode/psms_scored.parquet",
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": report["content_hash"],
                "producing_stage": "rescore",
            }
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return open_results(out), config["rescore"], report


def test_entrapment_mode_tables(tmp_path, fixture_dir):
    rs, config, report = _entrapment_result(tmp_path, fixture_dir)
    stats = report["stats"]
    scored = pq.read_table(rs.scored.path).to_pandas()
    # Peptide rows: decoys do not compete, so every row is a non-decoy winner.
    page = identification_table(
        rs, TableQuery(unit="peptide", threshold=None, include_decoys=True, limit=0)
    )
    non_decoy = scored[scored["label"] != "decoy"]
    assert page.total == non_decoy["base_peptide_id"].nunique()
    winners = scored.loc[scored["peptide_q_value"] < 1.0, "candidate_id"]
    rows = []
    for offset in range(0, page.total, 10_000):
        rows.append(
            identification_table(
                rs,
                TableQuery(
                    unit="peptide",
                    threshold=None,
                    include_decoys=True,
                    offset=offset,
                    limit=10_000,
                    sort_by="base_peptide_id",
                ),
            ).rows
        )
    got = pd.concat(rows)
    assert set(got["candidate_id"]) == set(winners)
    assert (got["label"] != "decoy").all()
    assert got["is_winner"].all()
    # The spike-ins are flagged; the engine's target counts exclude them.
    for unit, stat in (
        ("peptide", "target_peptides_at_1pct"),
        ("precursor", "target_precursors_at_1pct"),
    ):
        page = identification_table(rs, TableQuery(unit=unit, limit=5))
        assert "is_entrapment" in page.rows.columns
        assert "Entrapment mode" in page.description
        match = re.search(
            r"is_entrapment flags them \(([\d,]+) of the ([\d,]+) rows\)", page.description
        )
        assert match is not None
        spikes, total = (int(g.replace(",", "")) for g in match.groups())
        assert total == page.total
        assert total - spikes == stats[stat]
        if unit == "peptide":
            assert spikes == stats["entrapment_peptides_at_1pct"]
    assert config["entrapment_marker"] in page.column_labels["is_entrapment"]


def test_spike_in_wins_an_exact_tie(tmp_path, fixture_dir):
    """G5 F4.3: in entrapment mode a spike-in replaces a real target on an exact tie.

    A real target winner R and a later spike-in row E share one base peptide and one
    score, and a decoy D of the same base peptide scores higher. Every row of the group
    holds peptide_q_value 1.0, so the rule decides: D does not compete, and E wins.
    """
    rs, config, _ = _entrapment_result(tmp_path, fixture_dir)
    path = rs.scored.path
    df = pq.read_table(path).to_pandas()
    df["frn"] = np.arange(len(df))
    protein = df["protein"]
    spike = (
        (df["label"] == "target")
        & protein.str.contains(config["entrapment_marker"], regex=False)
        & ~protein.str.contains(config["entrapment_exclude"], regex=False)
    )
    for token in config["entrapment_contaminant_markers"]:
        spike &= ~protein.str.contains(token, regex=False)
    real = (df["label"] == "target") & ~spike & (df["peptide_q_value"] < 1.0)
    r = df[real].iloc[0]
    e = df[spike & (df["frn"] > r["frn"]) & (df["base_peptide_id"] != r["base_peptide_id"])].iloc[0]
    d = df[(df["label"] == "decoy") & (df["frn"] > e["frn"])].iloc[0]
    group = int(r["base_peptide_id"])

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        cid = frame["candidate_id"]
        frame.loc[cid == e["candidate_id"], ["base_peptide_id", "score"]] = [group, r["score"]]
        frame.loc[cid == d["candidate_id"], ["base_peptide_id", "score"]] = [group, r["score"] + 10]
        frame.loc[frame["base_peptide_id"] == group, "peptide_q_value"] = 1.0
        return frame

    _rewrite(path, change)
    rs = open_results(path.parent)
    winners = _winner_rows(
        rs.duck, path, "w.base_peptide_id", entrapment=True, tie=_entrapment_tie(config)
    )
    assert int(e["frn"]) in winners and int(r["frn"]) not in winners
    page = identification_table(
        rs,
        TableQuery(
            unit="peptide",
            threshold=None,
            include_decoys=True,
            limit=10_000,
            sort_by="base_peptide_id",
        ),
    )
    rows = page.rows
    while len(rows) < page.total:
        more = identification_table(rs, replace(page.query, offset=len(rows)))
        rows = pd.concat([rows, more.rows])
    row = rows[rows["base_peptide_id"] == group].iloc[0]
    assert int(row["candidate_id"]) == int(e["candidate_id"])
    assert row["is_winner"] and row["is_entrapment"]


@pytest.mark.parametrize("path", ["wide", "order"])
def test_entrapment_pages_on_every_path(tmp_path, fixture_dir, monkeypatch, path):
    """In entrapment mode the large-table paths give the pages of the direct query."""
    rs, _, _ = _entrapment_result(tmp_path, fixture_dir)
    queries = [
        TableQuery(unit=unit, threshold=t, include_decoys=decoys, sort_by=sort_by, offset=off)
        for unit in UNITS
        for t in (0.01, None)
        for decoys in (False, True)
        for sort_by in ("score", "base_peptide_id")
        for off in (0, 3_000)
    ]
    expected = [identification_table(rs, q) for q in queries]
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    if path == "order":
        monkeypatch.setattr(T, "MAX_MATERIALISED_ROWS", 0)
    T.drop_table_cache(rs)
    for query, want in zip(queries, expected, strict=True):
        got = identification_table(rs, query)
        assert got.total == want.total, query
        assert got.description == want.description, query
        pd.testing.assert_frame_equal(
            got.rows.reset_index(drop=True), want.rows.reset_index(drop=True), obj=repr(query)
        )
    kinds = {k[0] for k in T._cache(rs).entries}
    assert ("order" in kinds) == (path == "order")


def _entrapment_quant_result(tmp_path: Path, fixture_dir, *, recorded: bool, change=None):
    """The entrapment fixture with a manifest and a synthetic peptide_quant table.

    ``recorded`` writes the rescore's configuration, with its marker strings. Without
    it the run records no marker strings, and the spike-ins follow the viewer's default
    rule (entrapment.settings_for). ``change`` edits psms_scored first. peptide_quant
    holds the target rows with peptide_q_value <= 0.01 (a single run's PeptideQ gate).
    """
    src = fixture_dir("entrapment")
    config = json.loads((src.parent / "config.entrap_mode.json").read_text())
    report = json.loads((src / "psms_scored.parquet.report.json").read_text())
    out = tmp_path / "entrap_quant"
    out.mkdir()
    scored_path = out / "psms_scored.parquet"
    shutil.copy2(src / "psms_scored.parquet", scored_path)
    shutil.copy2(src / "psms_scored.parquet.report.json", out)
    if change is not None:
        _rewrite(scored_path, change)
    scored = pq.read_table(scored_path).to_pandas()
    gated = scored[(scored["label"] == "target") & (scored["peptide_q_value"] <= 0.01)]
    n = len(gated)
    quant = pa.table(
        {
            "candidate_id": pa.array(gated["candidate_id"], pa.uint32()),
            "base_peptide_id": pa.array(gated["base_peptide_id"], pa.uint32()),
            "peptidoform": pa.array(gated["peptidoform"], pa.string()),
            "charge": pa.array(gated["charge"], pa.int32()),
            "protein_group": pa.array(gated["protein_group"], pa.string()),
            "quantity": pa.array(np.ones(n), pa.float64()),
            "quant_status": pa.array(["quantified"] * n, pa.string()),
            "n_fragments_used": pa.array(np.full(n, 3), pa.int32()),
            "integration_apex_rt": pa.array(gated["apex_rt"], pa.float64()),
            "integration_lo_rt": pa.array(gated["elution_lo"], pa.float64()),
            "integration_hi_rt": pa.array(gated["elution_hi"], pa.float64()),
        }
    )
    pq.write_table(quant, out / "peptide_quant.parquet")
    quant_report = {
        "logical_name": "peptide_quant",
        "schema_name": "peptide_quant",
        "schema_version": 2,
        "stage": "quant",
        "rows": n,
        "content_hash": blake3_file(out / "peptide_quant.parquet"),
        "params": {"q_filter": "PeptideQ", "q_threshold": 0.01, "top_n_fragments": 3},
        "stats": {},
        "model_identity": None,
        "elapsed_ms": 1,
    }
    (out / "peptide_quant.parquet.report.json").write_text(json.dumps(quant_report))
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", "/recorded/entrap_quant"],
        "config_json": json.dumps(config) if recorded else None,
        "artifacts": {
            "psms_scored": {
                "path": "/recorded/entrap_quant/psms_scored.parquet",
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": blake3_file(scored_path),
                "producing_stage": "rescore",
            },
            "peptide_quant": {
                "path": "/recorded/entrap_quant/peptide_quant.parquet",
                "schema_name": "peptide_quant",
                "schema_version": 2,
                "rows": n,
                "content_hash": quant_report["content_hash"],
                "producing_stage": "quant",
            },
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return open_results(out)


def _all_pages(rs, query: TableQuery) -> pd.DataFrame:
    page = identification_table(rs, replace(query, limit=T.MAX_LIMIT, offset=0))
    rows = [page.rows]
    while sum(len(r) for r in rows) < page.total:
        offset = sum(len(r) for r in rows)
        rows.append(identification_table(rs, replace(page.query, offset=offset)).rows)
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("recorded", [True, False])
def test_tables_counts_and_quant_flag_the_same_spike_ins(tmp_path, fixture_dir, recorded):
    """Review consistency #5: tables, counts and quant use one spike-in test.

    The test is entrapment.count_classes (the settings of settings_for). Without
    recorded marker strings the tables used to flag nothing, while the counts used the
    viewer's default rule. A tie makes the rule visible in every module: a real target
    R and a later spike-in E share a base peptide and a score, a decoy D of the group
    scores higher, and every row of the group holds peptide_q_value 1.0. In entrapment
    mode D does not compete and E wins the tie.
    """
    src = pq.read_table(fixture_dir("entrapment") / "psms_scored.parquet").to_pandas()
    src["frn"] = np.arange(len(src))
    protein = src["protein"]
    target = src["label"] == "target"
    spike = (
        target
        & protein.str.contains("ENTRAP_", regex=False)
        & ~protein.str.contains("REAL_", regex=False)
    )
    for token in ("KRT", "K1C", "K2C", "ALBU", "TRYP"):
        spike &= ~protein.str.contains(token, regex=False)
    real = target & ~protein.str.contains("ENTRAP_", regex=False) & (src["peptide_q_value"] < 1)
    r = src[real].iloc[0]
    later = spike & (src["frn"] > r["frn"]) & (src["base_peptide_id"] != r["base_peptide_id"])
    e = src[later].iloc[0]
    d = src[(src["label"] == "decoy") & (src["frn"] > e["frn"])].iloc[0]
    group = int(r["base_peptide_id"])

    def change(frame: pd.DataFrame) -> pd.DataFrame:
        cid = frame["candidate_id"]
        frame.loc[cid == e["candidate_id"], ["base_peptide_id", "score"]] = [group, r["score"]]
        frame.loc[cid == d["candidate_id"], ["base_peptide_id", "score"]] = [group, r["score"] + 10]
        frame.loc[frame["base_peptide_id"] == group, "peptide_q_value"] = 1.0
        return frame

    rs = _entrapment_quant_result(tmp_path, fixture_dir, recorded=recorded, change=change)
    cls = count_classes(rs)
    assert cls.mode == "entrapment" and cls.spike_present
    assert cls.markers_recorded == recorded
    path = sql_path(rs.scored.path)
    # counts: the rows of count_classes(rs).spike.
    sql = f"SELECT source, candidate_id FROM read_parquet($path) WHERE {cls.spike}"
    params = {k: v for k, v in cls.params.items() if f"${k}" in sql}
    counted = {(int(a), int(b)) for a, b in rs.duck.rows(sql, {**params, "path": path})}
    # quant: the predicate of the quant states' winner rule.
    text, values = spike_in_condition(rs, "p")
    quant_sql = f"SELECT p.source, p.candidate_id FROM read_parquet(?) p WHERE {text}"
    in_quant = {(int(a), int(b)) for a, b in rs.duck.rows(quant_sql, [path, *values])}
    # tables: is_entrapment of every precursor row.
    rows = _all_pages(rs, TableQuery(threshold=None, include_decoys=True, sort_by="candidate_id"))
    assert len(rows) == len(src)
    flagged = rows[rows["is_entrapment"].astype(bool)]
    pairs = zip(flagged["source"], flagged["candidate_id"], strict=True)
    in_table = {(int(a), int(b)) for a, b in pairs}
    assert counted == in_quant == in_table
    assert len(counted) == (17_590 if recorded else 17_594)
    # The peptide table states the spike-ins and the real targets; the real targets are
    # the identification count.
    counts = {c.unit: c for c in unit_counts(rs, 0.01)}
    page = identification_table(rs, TableQuery(unit="peptide", limit=0))
    n_real = counts["peptide"].n_target
    assert f"is_entrapment flags them ({counts['peptide'].n_spike_in:,} of the" in page.description
    assert f"the other {n_real:,} rows are real targets" in page.description
    assert page.total == n_real + counts["peptide"].n_spike_in
    # The tie: counts, tables and quant all name E as the winner of R's base peptide.
    winner = group_winners(rs, "peptide", keys=[group])
    assert list(winner["candidate_id"]) == [int(e["candidate_id"])]
    peptides = _all_pages(rs, TableQuery(unit="peptide", threshold=None, sort_by="base_peptide_id"))
    row = peptides[peptides["base_peptide_id"] == group].iloc[0]
    assert int(row["candidate_id"]) == int(e["candidate_id"])
    assert row["is_winner"] and row["is_entrapment"]
    reason = quant_state(rs, None, int(r["candidate_id"])).reason
    assert f"the winning row of this base peptide is candidate {int(e['candidate_id'])}" in reason


def test_base_peptide_competition_is_named_in_the_tables(open_fixture):
    """Review semantics #6: under compete.group_by = BasePeptide, precursor_q counts are a
    base-peptide unit, as the overview's count says (counts.unit_counts)."""
    rs = open_fixture("ovl_bp")
    caveat = "compete.group_by = BasePeptide kept about one form per base peptide"
    count = {c.unit: c for c in unit_counts(rs, 0.01)}["precursor"]
    assert caveat in count.label and "approximates a base-peptide count" in count.label
    page = identification_table(rs, TableQuery(limit=0))
    assert page.total == count.n_target
    assert caveat in page.description
    assert "a count on precursor_q approximates a base-peptide count" in page.description
    assert caveat in page.column_labels["precursor_q"]
    # Another q column, or no q filter: the rows themselves approximate base peptides.
    for query in (TableQuery(q_column="q_value", limit=0), TableQuery(threshold=None, limit=0)):
        other = identification_table(rs, query)
        assert "these rows approximate base peptides, not every precursor" in other.description
        assert caveat in other.column_labels["precursor_q"]
    peptide = identification_table(rs, TableQuery(unit="peptide", limit=0))
    assert caveat in peptide.column_labels["precursor_q"]
    single = identification_table(open_fixture("single"), TableQuery(limit=0))
    assert "compete.group_by" not in single.description
    assert "base-peptide count" not in single.column_labels["precursor_q"]


@pytest.mark.parametrize("large", [False, True])
def test_duckdb_failure_is_a_viewer_error(fixture_dir, monkeypatch, large):
    """Review scale #5: a DuckDB error (here a simulated out-of-memory) names the result."""
    if large:
        monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    rs = open_results(fixture_dir("single"))
    real_execute = rs.duck.execute

    def execute(sql, params=None):
        if sql.startswith("CREATE TABLE"):
            raise duckdb.OutOfMemoryException("Out of Memory Error: failed to allocate")
        return real_execute(sql, params)

    def df(sql, params=None):
        raise duckdb.OutOfMemoryException("Out of Memory Error: failed to allocate")

    monkeypatch.setattr(rs.duck, "execute", execute)
    monkeypatch.setattr(rs.duck, "df", df)
    rows = pq.read_metadata(rs.scored.path).num_rows
    query = TableQuery(threshold=None, include_decoys=True, offset=100)
    with pytest.raises(ViewerError) as err:
        identification_table(rs, query)
    text = str(err.value)
    assert text.startswith("The precursor table query failed on a result of ")
    assert f"a result of {rows:,} rows, from a scored table of {rows:,} rows" in text
    assert "OutOfMemoryException" in text
    assert f"Query: {query!r}." in text


def test_topk_selected_peaks(open_fixture):
    """G1: 26 scored targets chose an alternative peak; 16 of them are quantified."""
    rs = open_fixture("topk")
    page = identification_table(rs, TableQuery(threshold=None, include_decoys=True, limit=1000))
    promoted = page.rows[page.rows["selected_peak_rank"] == 1]
    assert len(promoted) == 26
    assert (promoted["label"] == "target").all()
    assert (promoted["q_value"] <= 0.01).all()
    assert (promoted["quant_state"] == "quantified").sum() == 16
    ranked = identification_table(rs, TableQuery(sort_by="selected_peak_rank", limit=30))
    assert (ranked.rows["selected_peak_rank"].iloc[:26] == 1).all()


def test_concurrent_pages_with_eviction(fixture_dir, monkeypatch):
    """Threads page while the table cache evicts; every page is still correct."""
    monkeypatch.setattr(T, "MATERIALISE_MIN_ROWS", 0)
    monkeypatch.setattr(T, "MAX_CACHED_TABLES", 3)
    monkeypatch.setattr(T, "MAX_MATERIALISED_ROWS", 200)
    rs = open_results(fixture_dir("experiment"))
    brute = Brute(rs)
    queries = [
        TableQuery(unit=unit, threshold=t, offset=offset, limit=20)
        for unit in UNITS
        for t in (0.01, 0.05, None)
        for offset in (0, 20)
    ]
    expected = {q: brute.page(q)[1] for q in queries}

    def work(query: TableQuery) -> tuple[TableQuery, int, int]:
        page = identification_table(rs, query)
        return query, page.total, len(page.rows)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(work, queries * 3))
    for query, total, n in results:
        assert total == expected[query]
        assert n == min(query.limit, max(0, total - query.offset))
    assert len(T._cache(rs).entries) <= 3


def test_page_helpers(open_fixture):
    rs = open_fixture("single")
    page = identification_table(rs, TableQuery(limit=50))
    assert T.page_count(page) == 6
    nxt = T.with_page(page.query, 50)
    assert nxt.offset == 50 and nxt.signature() == page.query.signature()


# --------------------------------------------------------------------------- real data


def _time_pages(rs, queries: list[TableQuery]) -> list[float]:
    out = []
    for query in queries:
        t0 = time.perf_counter()
        page = identification_table(rs, query)
        out.append(time.perf_counter() - t0)
        assert len(page.rows) == min(query.limit, max(0, page.total - query.offset))
    return out


def _check_engine_counts(rs) -> None:
    """Unit totals equal the rescore report at 0.01 and COUNT DISTINCT elsewhere.

    The quant gate of every run equals its peptide_quant rows.
    """
    stats = rs.scored.report.stats
    for unit, stat in (
        ("precursor", "target_precursors_at_1pct"),
        ("peptide", "target_peptides_at_1pct"),
        ("protein_group", "target_protein_groups_at_1pct"),
    ):
        assert identification_table(rs, TableQuery(unit=unit, limit=0)).total == stats[stat]
    for column, t in (("peptide_q_value", 1.0), ("peptide_q_value", None), ("q_value", 0.01)):
        page = identification_table(
            rs, TableQuery(unit="peptide", q_column=column, threshold=t, limit=0)
        )
        where = "true" if t is None else f"{column} <= {t!r}"
        assert page.total == _distinct_keys(rs.duck, rs.scored.path, "base_peptide_id", where)
    for run in rs.runs:
        gate = quant_gate(rs, run)
        query = TableQuery(q_column=gate.q_column, threshold=gate.threshold, limit=0)
        if rs.is_experiment:
            query = replace(query, run=run.name)
        gated = identification_table(rs, query)
        missing = identification_table(rs, replace(query, quant_status="not_selected"))
        assert gated.total == run.artifact("peptide_quant").rows
        assert missing.total == 0


def _report(label: str, rs, timings: list[float], later: list[float]) -> None:
    print(
        f"\n{label} ({rs.scored.parquet().num_rows:,} scored rows): first page "
        f"{timings[0]:.3f} s; later pages "
        + ", ".join(f"{t:.3f}" for t in later)
        + f" s (median {statistics.median(later):.3f} s, max {max(later):.3f} s)"
    )


@pytest.mark.real_data
def test_real_single_precursor_page(real_single):
    t0 = time.perf_counter()
    rs = open_results(real_single)
    opened = time.perf_counter() - t0
    timings = _time_pages(
        rs,
        [
            TableQuery(),
            TableQuery(offset=50),
            TableQuery(offset=80_000),
            TableQuery(sort_by="apex_rt", descending=False),
            TableQuery(sort_by="quantity", offset=500),
            TableQuery(search="LVNELTEFAK", threshold=0.01),
        ],
    )
    print(f"\nopen {opened:.3f} s")
    _report("Astral single, precursor table", rs, timings, timings[1:])
    # The whole table without a q filter (above the wide cap: order table and fetch).
    q = TableQuery(threshold=None, include_decoys=True)
    big = _time_pages(rs, [q, *(replace(q, offset=o) for o in (50, 5_000, 400_000, 680_000))])
    _report("Astral single, unfiltered precursor table", rs, big, big[1:])
    q = TableQuery(unit="peptide", threshold=None)
    pep = _time_pages(rs, [q, *(replace(q, offset=o) for o in (50, 100_000, 300_000))])
    _report("Astral single, peptide table without q filter", rs, pep, pep[1:])
    _check_engine_counts(rs)
    assert statistics.median(timings[1:]) < 0.3
    assert statistics.median(big[1:]) < 0.3
    assert statistics.median(pep[1:]) < 0.3


@pytest.mark.real_data
def test_real_experiment_precursor_page(real_experiment):
    rs = open_results(real_experiment)
    run = rs.runs[-1].name
    timings = _time_pages(
        rs,
        [
            TableQuery(),
            TableQuery(offset=50),
            TableQuery(offset=100_000),
            TableQuery(sort_by="apex_rt", descending=False),
            TableQuery(run=run),
            TableQuery(run=run, offset=50),
            TableQuery(run=run, sort_by="quantity"),
        ],
    )
    later = [timings[1], timings[2], timings[3], timings[5], timings[6]]
    _report("experiment, precursor table", rs, timings, later)
    print(f"first page of run {run}: {timings[4]:.3f} s")
    q = TableQuery(threshold=None, include_decoys=True)
    big = _time_pages(rs, [q, *(replace(q, offset=o) for o in (50, 5_000, 400_000))])
    _report("experiment, unfiltered precursor table", rs, big, big[1:])
    q = TableQuery(unit="peptide", threshold=None, include_decoys=True)
    pep = _time_pages(rs, [q, *(replace(q, offset=o) for o in (50, 100, 5_000, 400_000))])
    _report("experiment, peptide table without q filter", rs, pep, pep[1:])
    _check_engine_counts(rs)
    assert statistics.median(later) < 0.3
    assert statistics.median(big[1:]) < 0.3
    assert statistics.median(pep[1:]) < 0.3
