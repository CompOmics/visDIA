"""Feature columns, per-candidate feature rows and target/decoy percentiles."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import blake3
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import (
    ArtifactNotFound,
    InconsistentData,
    SchemaVersionError,
    ViewerError,
    open_results,
)
from mumdia_viewer.data import features as F
from mumdia_viewer.data.features import (
    BOOKKEEPING,
    CONTEXT_COLUMNS,
    EVIDENCE_FEATURES,
    candidate_feature_rows,
    candidate_features,
    feature_columns,
    feature_percentiles,
    feature_source,
)

SUBSET = ["frag_corr", "coelution_mean", "spectral_angle_matched", "rt_error_abs"]

# pandas forms of the validity rules, keyed by their SQL (an independent oracle).
ORACLE_RULES = {
    "(n_matched_b + n_matched_y) >= 2": lambda d: (d["n_matched_b"] + d["n_matched_y"]) >= 2,
    "(n_matched_b + n_matched_y) >= 1": lambda d: (d["n_matched_b"] + d["n_matched_y"]) >= 1,
    "n_observations >= 3": lambda d: d["n_observations"] >= 3,
    "predicted_rt_raw <> 0": lambda d: d["predicted_rt_raw"] != 0,
    "rt_error_over_peak_width > 0": lambda d: d["rt_error_over_peak_width"] > 0,
    "log_apex_intensity > 0": lambda d: d["log_apex_intensity"] > 0,
    "has_ms1 = 1": lambda d: d["has_ms1"] == 1,
    "ms1_isotope_cosine_apex > 0": lambda d: d["ms1_isotope_cosine_apex"] > 0,
}

# The validity inputs that the smaller feature sets (minimal, rich) lack.
MINIMAL_DROP = [
    "n_observations",
    "predicted_rt_raw",
    "n_matched_b",
    "n_matched_y",
    "has_ms1",
    "n_peak_scans",
    "peak_window_degenerate",
]


def _rehash(copy: Path, key: str, path: Path) -> None:
    """Record the blake3 hash of a rewritten artifact in the manifest and its report.

    The candidate index cache is keyed by the recorded hash, so a rewritten table must
    not keep the hash of the original.
    """
    digest = blake3.blake3(path.read_bytes()).hexdigest()
    report = path.with_name(path.name + ".report.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    data["content_hash"] = digest
    report.write_text(json.dumps(data), encoding="utf-8")
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["artifacts"][key]["content_hash"] = digest
    manifest.write_text(json.dumps(data), encoding="utf-8")


def _set_version(copy: Path, key: str, path: Path, version: int) -> None:
    """Record another schema version for an artifact, in the manifest and its report."""
    report = path.with_name(path.name + ".report.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    data["schema_version"] = version
    report.write_text(json.dumps(data), encoding="utf-8")
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["artifacts"][key]["schema_version"] = version
    manifest.write_text(json.dumps(data), encoding="utf-8")


def _write_like(df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path)


def _table(paths: list[Path] | Path) -> pa.Table:
    if isinstance(paths, Path):
        return pq.read_table(paths)
    return pa.concat_tables([pq.read_table(p) for p in paths])


def _row(table: pa.Table, cid: int, rank: int) -> dict:
    mask = pc.and_(pc.equal(table["candidate_id"], cid), pc.equal(table["peak_rank"], rank))
    rows = table.filter(mask).to_pylist()
    assert len(rows) == 1
    return rows[0]


def _scored(path: Path, source: int | None = None) -> pd.DataFrame:
    df = pq.read_table(
        path, columns=["candidate_id", "selected_peak_rank", "source", "label"]
    ).to_pandas()
    return df if source is None else df[df["source"] == source]


def _pandas_percentile(
    features: pd.DataFrame, scored: pd.DataFrame, column: str, value: float, valid=None
) -> dict[str, tuple[float | None, int]]:
    """Independent oracle: rows of the scored peaks, validity filter, share <= value."""
    keys = scored[["candidate_id", "selected_peak_rank"]].rename(
        columns={"selected_peak_rank": "peak_rank"}
    )
    pop = features.merge(keys.drop_duplicates(), on=["candidate_id", "peak_rank"], how="inner")
    if valid is not None:
        pop = pop[valid(pop)]
    out = {}
    for label in ("target", "decoy"):
        sub = pop.loc[pop["label"] == label, column].astype("float64")
        pct = None if sub.empty else 100.0 * float((sub <= value).mean())
        out[label] = (pct, int(sub.size))
    return out


def _first_cid(rs, run="") -> int:
    table = pq.read_table(feature_source(rs, run).paths[0], columns=["candidate_id"])
    return int(table["candidate_id"][0].as_py())


def _frames(root: Path, features_rel: str, scored_rel: str, source=None):
    return pq.read_table(root / features_rel).to_pandas(), _scored(root / scored_rel, source)


# --------------------------------------------------------------------------- columns


def test_feature_columns_single_experiment_grouped(open_fixture, fixture_dir):
    listed = json.loads(
        (fixture_dir("single") / "features.parquet.schema.json").read_text(encoding="utf-8")
    )["feature_columns"]
    rs = open_fixture("single")
    cols = feature_columns(rs, rs.runs[0])
    assert cols == listed and len(cols) == 387 and cols[0] == "rt_error_abs"
    assert not set(cols) & set(BOOKKEEPING)
    exp = open_fixture("experiment")
    assert feature_columns(exp, "a") == listed and feature_columns(exp, 1) == listed
    grouped = open_fixture("grouped")
    band_listed = json.loads(
        (fixture_dir("grouped") / "groups" / "g00" / "psms_competed.parquet.schema.json").read_text(
            encoding="utf-8"
        )
    )["feature_columns"]
    assert feature_columns(grouped, "") == band_listed


def test_feature_columns_without_schema_json(fixture_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    (copy / "features.parquet.schema.json").unlink()
    rs = open_results(copy)
    names = pq.read_schema(copy / "features.parquet").names
    assert feature_columns(rs, "") == [n for n in names if n not in BOOKKEEPING]
    assert names[: len(BOOKKEEPING)] == list(BOOKKEEPING)


def test_feature_source_per_layout(open_fixture, fixture_dir, tmp_path):
    single = feature_source(open_fixture("single"), "")
    assert single.kind == "features" and single.artifacts[0].path.name == "features.parquet"
    grouped = feature_source(open_fixture("grouped"), "")
    assert grouped.kind == "band_psms_competed" and grouped.bands == ("g00", "g01", "g02")
    assert all(p.name == "psms_competed.parquet" for p in grouped.paths)
    pooled = feature_source(open_fixture("grouped_pool"), "")
    assert pooled.kind == "psms_competed" and pooled.paths[0].parent == fixture_dir("grouped_pool")
    # Overlapping bands: the engine pools the competed table even without pool_competed.
    assert feature_source(open_fixture("ovl_bp"), "").kind == "psms_competed"
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    (copy / "features.parquet").unlink()
    fallback = feature_source(open_results(copy), "")
    assert fallback.kind == "psms_competed" and "absent" in fallback.description
    (copy / "psms_competed.parquet").unlink()
    with pytest.raises(ArtifactNotFound, match="feature rows"):
        feature_source(open_results(copy), "")


# --------------------------------------------------------------------------- one candidate


def test_candidate_features_equal_pyarrow_rows(open_fixture, fixture_dir):
    rs = open_fixture("single")
    table = _table(fixture_dir("single") / "features.parquet")
    cids = table["candidate_id"].to_pylist()
    for cid in [*cids[::23], cids[0], cids[-1]]:
        expected = _row(table, cid, 0)
        assert candidate_features(rs, "", cid) == expected
        sub = candidate_features(rs, rs.runs[0], cid, 0, SUBSET)
        assert list(sub) == [
            c for c in dict.fromkeys([*BOOKKEEPING, *CONTEXT_COLUMNS, *SUBSET]) if c in expected
        ]
        assert all(sub[k] == expected[k] for k in sub)
    assert candidate_features(rs, "", 10**9) is None
    assert candidate_features(rs, "", cids[0], peak_rank=1) is None
    # A requested column the table lacks is left out of the row.
    assert "contested_frac" not in candidate_features(rs, "", cids[0], 0, ["contested_frac"])


def test_candidate_features_pick_the_selected_alternative_peak(open_fixture, fixture_dir):
    rs = open_fixture("topk")
    root = fixture_dir("topk")
    table = _table(root / "features.parquet")
    scored = pq.read_table(root / "psms_scored.parquet").to_pandas()
    promoted = scored[scored["selected_peak_rank"] == 1]
    assert len(promoted) == 26
    for s in promoted.itertuples():
        row = candidate_features(rs, "", s.candidate_id, peak_rank=1)
        assert row == _row(table, s.candidate_id, 1)
        # The selected peak's apex and bounds are the identification's.
        assert (row["apex_rt"], row["elution_lo"], row["elution_hi"]) == (
            s.apex_rt,
            s.elution_lo,
            s.elution_hi,
        )
        rank0 = candidate_features(rs, "", s.candidate_id, peak_rank=0)
        assert rank0 == _row(table, s.candidate_id, 0) and rank0["apex_rt"] != s.apex_rt
        ranks = [r["peak_rank"] for r in candidate_feature_rows(rs, "", s.candidate_id, SUBSET)]
        assert ranks == [0, 1]
        assert candidate_features(rs, "", s.candidate_id, peak_rank=2) is None


def test_candidate_features_grouped_read_band_competed_rows(open_fixture, fixture_dir):
    rs = open_fixture("grouped")
    root = fixture_dir("grouped")
    bands = {
        b: _table(root / "groups" / b / "psms_competed.parquet") for b in ("g00", "g01", "g02")
    }
    scored = _scored(root / "psms_scored.parquet")
    seen_bands = set()
    for cid in scored["candidate_id"].tolist()[::17]:
        holders = [b for b, t in bands.items() if cid in set(t["candidate_id"].to_pylist())]
        assert len(holders) == 1
        seen_bands.add(holders[0])
        assert candidate_features(rs, "", cid) == _row(bands[holders[0]], cid, 0)
    assert seen_bands == {"g00", "g01", "g02"}
    # The pooled twin reads the same rows from its root table.
    pooled = open_fixture("grouped_pool")
    cid = scored["candidate_id"].iloc[3]
    assert candidate_features(pooled, "", cid) == candidate_features(rs, "", cid)


def test_candidate_features_experiment_runs(open_fixture, fixture_dir):
    rs = open_fixture("mbr")
    for name in ("a", "c"):
        table = _table(fixture_dir("mbr") / name / "features.parquet")
        cid = table["candidate_id"].to_pylist()[7]
        assert candidate_features(rs, name, cid) == _row(table, cid, 0)
        assert candidate_features(rs, rs.run(name).index, cid) == _row(table, cid, 0)


def test_duplicate_band_rows_are_refused(open_fixture, fixture_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("grouped"), copy)
    g00 = copy / "groups" / "g00" / "psms_competed.parquet"
    g01 = copy / "groups" / "g01" / "psms_competed.parquet"
    first = pq.read_table(g00).slice(0, 1)
    cid = first["candidate_id"][0].as_py()
    other = pq.read_table(g01)["candidate_id"][5].as_py()
    pq.write_table(pa.concat_tables([first, pq.read_table(g01)]), g01)
    _rehash(copy, "psms_competed[g01]", g01)
    rs = open_results(copy)
    # Without a pooled table the band id ranges must be disjoint; a shared candidate
    # would be counted twice in a percentile population, so the whole run is refused.
    for read in (
        lambda: candidate_features(rs, "", cid),
        lambda: candidate_features(rs, "", other),
        lambda: feature_percentiles(rs, "", {"frag_corr": 0.5}),
    ):
        with pytest.raises(InconsistentData, match="g00 and g01 hold overlapping candidate_id"):
            read()
    # The per-candidate check behind the range check names both tables.
    source = F.FeatureSource(
        "",
        "band_psms_competed",
        tuple(b.artifact("psms_competed") for b in rs.runs[0].grouped.bands),
        ("g00", "g01", "g02"),
        (),
        "3 band tables",
    )
    with pytest.raises(InconsistentData, match=r"2 band tables \(g00, g01\) of 3 band tables"):
        F._locate(rs, source, cid)


# --------------------------------------------------------------------------- percentiles


def test_percentiles_equal_pandas_on_the_scored_population(open_fixture, fixture_dir):
    rs = open_fixture("single")
    root = fixture_dir("single")
    features, scored = _frames(root, "features.parquet", "psms_scored.parquet")
    cid = int(features["candidate_id"].iloc[40])
    values = candidate_features(rs, "", cid)
    cols = ["frag_corr", "coelution_mean", "spectral_angle", "rt_error_abs", "log_ms1_mono"]
    rules = {
        "coelution_mean": lambda d: d["n_observations"] >= 3,
        "rt_error_abs": lambda d: d["predicted_rt_raw"] != 0,
        "log_ms1_mono": lambda d: d["has_ms1"] == 1,
    }
    results = {p.feature: p for p in feature_percentiles(rs, "", values, cols)}
    assert list(results) == cols
    for col in cols:
        expected = _pandas_percentile(features, scored, col, values[col], rules.get(col))
        p = results[col]
        assert p.valid is True and p.note is None
        assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
        assert p.pct_decoy == pytest.approx(expected["decoy"][0], abs=1e-9)
        assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
        assert "not FDR-filtered" in p.population and "selected_peak_rank" in p.population
        # Only a population filtered by a rule is called valid.
        assert p.rule == EVIDENCE_FEATURES[col].validity
        assert p.population.startswith("valid ") is (p.rule is not None)
    assert "every value of this feature is a measurement" in results["frag_corr"].population


def test_shortcut_and_join_give_the_same_population(open_fixture, monkeypatch):
    rs = open_fixture("single")
    source = feature_source(rs, "")
    assert F._population_is_whole_table(rs, source, rs.scored, 0)
    values = candidate_features(rs, "", _first_cid(rs))
    cols = [c for c in EVIDENCE_FEATURES if c in values]
    fast = feature_percentiles(rs, "", values, cols)
    monkeypatch.setattr(F, "_population_is_whole_table", lambda *a, **k: False)
    joined = feature_percentiles(rs, "", values, cols)
    assert fast == joined


def test_topk_population_excludes_unselected_alternative_peaks(open_fixture, fixture_dir):
    rs = open_fixture("topk")
    root = fixture_dir("topk")
    source = feature_source(rs, "")
    assert not F._population_is_whole_table(rs, source, rs.scored, 0)
    features, scored = _frames(root, "features.parquet", "psms_scored.parquet")
    assert len(features) == 360 and len(scored) == 288
    promoted = scored[scored["selected_peak_rank"] == 1]["candidate_id"].iloc[0]
    values = candidate_features(rs, "", int(promoted), peak_rank=1)
    results = feature_percentiles(rs, "", values, ["frag_corr", "log_apex_intensity"])
    for p in results:
        assert p.n_target + p.n_decoy == 288
        rule = (
            (lambda d: d["log_apex_intensity"] > 0) if p.feature == "log_apex_intensity" else None
        )
        expected = _pandas_percentile(features, scored, p.feature, values[p.feature], rule)
        assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
        assert p.pct_decoy == pytest.approx(expected["decoy"][0], abs=1e-9)
    # A table without the join would rank against all 360 extracted peaks.
    assert (features["peak_rank"] > 0).sum() == 72


def test_validity_filters_are_applied(open_fixture, fixture_dir):
    rs = open_fixture("single")
    features = pq.read_table(fixture_dir("single") / "features.parquet").to_pandas()
    zero = features[features["ms1_isotope_cosine_apex"] == 0].iloc[0]
    positive = features[features["ms1_isotope_cosine_apex"] > 0].iloc[0]
    valid = features[features["ms1_isotope_cosine_apex"] > 0]
    expected_n = (int((valid.label == "target").sum()), int((valid.label == "decoy").sum()))
    assert sum(expected_n) == 192 < len(features)
    for row, ok in ((zero, False), (positive, True)):
        values = candidate_features(rs, "", int(row.candidate_id))
        (p,) = feature_percentiles(rs, "", values, ["ms1_isotope_cosine_apex"])
        assert (p.n_target, p.n_decoy) == expected_n
        assert p.valid is ok
        assert (p.pct_target is not None) is ok
        assert "ms1_isotope_cosine_apex > 0" in p.population
    # Every band row of the grouped fixture has no RT calibration (predicted_rt_raw = 0).
    grouped = open_fixture("grouped")
    cid = int(
        pq.read_table(fixture_dir("grouped") / "psms_scored.parquet")["candidate_id"][0].as_py()
    )
    values = candidate_features(grouped, "", cid)
    rt, corr = feature_percentiles(grouped, "", values, ["rt_error_abs", "frag_corr"])
    assert rt.valid is False and rt.pct_target is None and (rt.n_target, rt.n_decoy) == (0, 0)
    assert "predicted_rt_raw <> 0" in rt.note
    assert corr.valid is True and corr.n_target + corr.n_decoy == 305


def test_validity_of_the_candidate_value_from_given_inputs(open_fixture):
    rs = open_fixture("single")
    one_fragment = {"spectral_angle_matched": 1.0, "n_matched_b": 1.0, "n_matched_y": 0.0}
    # Without columns, every key that is not a bookkeeping column is ranked.
    assert [r.feature for r in feature_percentiles(rs, "", one_fragment)] == list(one_fragment)
    (p,) = feature_percentiles(rs, "", one_fragment, ["spectral_angle_matched"])
    assert p.valid is False and p.pct_target is None and "fails" in p.note
    (q,) = feature_percentiles(rs, "", {"coelution_mean": 0.5})
    assert q.valid is None and q.pct_target is None and "not checked" in q.note
    # With the candidate named, missing validity inputs are read from the feature table.
    cid = _first_cid(rs)
    row = candidate_features(rs, "", cid, 0, ["coelution_mean"])
    (r,) = feature_percentiles(
        rs, "", {"candidate_id": cid, "peak_rank": 0, "coelution_mean": row["coelution_mean"]}
    )
    assert r.valid is True and r.pct_target is not None


def test_flags_uncomputed_and_absent_features(open_fixture):
    rs = open_fixture("single")
    values = candidate_features(rs, "", _first_cid(rs))
    wanted = ["has_ms1", "peak_contested_frac", "contested_frac", "peptidoform"]
    flag, contested, absent, text = feature_percentiles(rs, "", values, wanted)
    assert flag.valid is True and flag.pct_target is None and "flag" in flag.note
    assert flag.n_target + flag.n_decoy == 284
    assert contested.valid is False and contested.pct_target is None
    assert "not computed" in contested.note and "peak_claim = none" in contested.note
    assert absent.value is None and absent.n_target == 0 and "psms_extracted" in absent.note
    assert text.pct_target is None and "not numeric" in text.note


def test_contested_features_are_ranked_when_computed(fixture_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    config = json.loads(data["config_json"])
    config["extract"]["emit_contested_features"] = True
    data["config_json"] = json.dumps(config)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    rs = open_results(copy)
    assert F.contested_computed(rs)[0]
    values = candidate_features(rs, "", _first_cid(rs))
    (p,) = feature_percentiles(rs, "", values, ["peak_contested_frac"])
    assert p.valid is True and p.pct_target == 100.0


def test_experiment_population_is_the_runs_own_rows(open_fixture, fixture_dir):
    rs = open_fixture("mbr")
    root = fixture_dir("mbr")
    for name in ("a", "c"):
        run = rs.run(name)
        features = pq.read_table(root / name / "features.parquet").to_pandas()
        scored = _scored(root / "scored_combined.parquet", run.index)
        values = candidate_features(rs, name, int(features["candidate_id"].iloc[11]))
        (p,) = feature_percentiles(rs, name, values, ["frag_corr"])
        expected = _pandas_percentile(features, scored, "frag_corr", values["frag_corr"])
        assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
        assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
        assert f"run {name}" in p.population and f"source = {run.index}" in p.population
    # Run c has no decoy rows: no decoy percentage, and the note says so.
    values = candidate_features(rs, "c", _first_cid(rs, "c"))
    (p,) = feature_percentiles(rs, "c", values, ["frag_corr"])
    assert p.n_decoy == 0 and p.pct_decoy is None and "no decoy rows" in p.note
    (q,) = feature_percentiles(rs, "c", values, ["log_apex_intensity"])
    assert q.rule == "log_apex_intensity > 0" and "no valid decoy rows" in q.note


def test_grouped_population_is_the_union_of_band_tables(open_fixture, fixture_dir):
    rs = open_fixture("grouped")
    root = fixture_dir("grouped")
    features = pd.concat(
        [
            pq.read_table(root / "groups" / b / "psms_competed.parquet").to_pandas()
            for b in ("g00", "g01", "g02")
        ]
    )
    scored = _scored(root / "psms_scored.parquet")
    values = candidate_features(rs, "", int(scored["candidate_id"].iloc[100]))
    cols = ["frag_corr", "ms1_isotope_cosine_apex", "coelution_best"]
    rules = {"ms1_isotope_cosine_apex": lambda d: d["ms1_isotope_cosine_apex"] > 0}
    for p in feature_percentiles(rs, "", values, cols):
        expected = _pandas_percentile(
            features, scored, p.feature, values[p.feature], rules.get(p.feature)
        )
        assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
        if p.valid:
            assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
            assert p.pct_decoy == pytest.approx(expected["decoy"][0], abs=1e-9)
        assert "3 band tables" in p.population


def _break_every_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Make each validity rule fail on a known block of rows (the fixture has none)."""
    idx = np.arange(len(df))
    df.loc[idx % 7 == 0, ["n_matched_b", "n_matched_y"]] = [0.0, 1.0]  # one fragment at apex
    df.loc[idx % 11 == 0, ["n_matched_b", "n_matched_y"]] = 0.0  # none at the apex scan
    df.loc[idx % 5 == 1, "n_observations"] = 2.0
    df.loc[idx % 6 == 2, ["predicted_rt_raw", "rt_error_abs", "rt_error_signed"]] = 0.0
    df.loc[idx % 9 == 3, "rt_error_over_peak_width"] = 0.0
    df.loc[idx % 13 == 4, "log_apex_intensity"] = 0.0
    df.loc[idx % 4 == 0, "has_ms1"] = 0.0
    return df


def test_every_validity_rule_on_rows_that_fail_it(fixture_dir, tmp_path, monkeypatch):
    """Each rule excludes known rows; both population paths equal the pandas oracle."""
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    path = copy / "features.parquet"
    schema = pq.read_schema(path)
    features = _break_every_rule(pq.read_table(path).to_pandas())
    _write_like(features, path, schema)
    _rehash(copy, "features", path)
    rs = open_results(copy)
    scored = _scored(copy / "psms_scored.parquet")
    ruled = [n for n, info in EVIDENCE_FEATURES.items() if info.validity and n in schema.names]
    cols = [*ruled, "frag_corr"]
    for sql in ORACLE_RULES:
        failing = int((~ORACLE_RULES[sql](features)).sum())
        assert 20 <= failing < len(features), sql  # every rule excludes rows
    for forced_join in (False, True):
        if forced_join:
            monkeypatch.setattr(F, "_population_is_whole_table", lambda *a, **k: False)
        for pos in (3, 50, 101, 200):
            values = candidate_features(rs, "", int(features["candidate_id"].iloc[pos]))
            results = {p.feature: p for p in feature_percentiles(rs, "", values, cols)}
            for col in cols:
                rule = EVIDENCE_FEATURES[col].validity
                oracle = ORACLE_RULES[rule] if rule else None
                own = True if oracle is None else bool(oracle(features.iloc[[pos]]).iloc[0])
                expected = _pandas_percentile(features, scored, col, values[col], oracle)
                p = results[col]
                assert p.valid is own and p.rule == rule, (col, pos)
                assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
                if own:
                    assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
                    assert p.pct_decoy == pytest.approx(expected["decoy"][0], abs=1e-9)
                else:
                    assert p.pct_target is None and p.pct_decoy is None and "fails" in p.note


def _minimal_copy(fixture_dir, tmp_path: Path) -> tuple[Path, set[int]]:
    """smoke/out without the validity inputs of the smaller feature sets.

    Every fifth candidate has no RT calibration: rt_pred_cal is NaN in psms_extracted
    and rt_error_abs is 0 in the features, as the engine writes it.
    """
    copy = tmp_path / "minimal"
    shutil.copytree(fixture_dir("single"), copy)
    ext_path = copy / "psms_extracted.parquet"
    ext = pq.read_table(ext_path)
    uncalibrated = set(ext["candidate_id"].to_pylist()[::5])
    # NaN, not null: the engine writes a non-finite rt_pred_cal into a non-nullable column.
    hit = pc.is_in(ext["candidate_id"], value_set=pa.array(sorted(uncalibrated), pa.uint32()))
    nan = pa.scalar(math.nan, pa.float64())
    column = pc.if_else(hit, nan, ext["rt_pred_cal"])
    ext = ext.set_column(
        ext.schema.get_field_index("rt_pred_cal"), ext.schema.field("rt_pred_cal"), column
    )
    pq.write_table(ext, ext_path)
    _rehash(copy, "psms_extracted", ext_path)
    path = copy / "features.parquet"
    table = pq.read_table(path)
    table = table.select([c for c in table.column_names if c not in MINIMAL_DROP])
    df = table.to_pandas()
    df.loc[df["candidate_id"].isin(uncalibrated), "rt_error_abs"] = 0.0
    _write_like(df, path, table.schema)
    _rehash(copy, "features", path)
    listed = path.with_name(path.name + ".schema.json")
    data = json.loads(listed.read_text(encoding="utf-8"))
    data["feature_columns"] = [c for c in data["feature_columns"] if c not in MINIMAL_DROP]
    listed.write_text(json.dumps(data), encoding="utf-8")
    return copy, uncalibrated


def test_rules_without_their_inputs_in_the_feature_table(fixture_dir, tmp_path):
    copy, uncalibrated = _minimal_copy(fixture_dir, tmp_path)
    rs = open_results(copy)
    assert not set(MINIMAL_DROP) & set(feature_columns(rs, ""))
    features = pq.read_table(copy / "features.parquet").to_pandas()
    ext = pq.read_table(copy / "psms_extracted.parquet").to_pandas()
    scored = _scored(copy / "psms_scored.parquet")
    calibrated = features.merge(
        ext[["candidate_id", "peak_rank", "rt_pred_cal"]], on=["candidate_id", "peak_rank"]
    )

    def finite(d: pd.DataFrame) -> pd.Series:
        return np.isfinite(d["rt_pred_cal"])

    # rt_error_abs: the rule is applied in its psms_extracted form.
    good = next(c for c in features["candidate_id"] if c not in uncalibrated)
    bad = min(uncalibrated)
    for cid, ok in ((good, True), (bad, False)):
        values = candidate_features(rs, "", int(cid))
        (p,) = feature_percentiles(rs, "", values, ["rt_error_abs"])
        expected = _pandas_percentile(
            calibrated, scored, "rt_error_abs", values["rt_error_abs"], finite
        )
        assert p.valid is ok and "isfinite(rt_pred_cal)" in p.rule and "psms_extracted" in p.rule
        assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
        assert p.n_target + p.n_decoy == len(features) - len(uncalibrated)
        assert (
            p.population.startswith("valid target/decoy rows")
            and "lacks predicted_rt_raw" in p.note
        )
        if ok:
            assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
        else:
            assert p.pct_target is None and "fails the validity rule" in p.note
    # The rt_pred_cal of a value given without a candidate is checked as given.
    rt = ["rt_error_abs"]
    (nan,) = feature_percentiles(rs, "", {"rt_error_abs": 3.0, "rt_pred_cal": math.nan}, rt)
    (given,) = feature_percentiles(rs, "", {"rt_error_abs": 3.0, "rt_pred_cal": 1500.0}, rt)
    assert nan.valid is False and nan.pct_target is None
    assert given.valid is True and given.pct_target is not None

    # Rules without an equivalent: no percentage, no row count, the rule is named.
    values = candidate_features(rs, "", int(good))
    wanted = ["coelution_mean", "spectral_angle_matched", "median_abs_frag_ppm", "log_ms1_mono"]
    for p in feature_percentiles(rs, "", values, wanted):
        info = EVIDENCE_FEATURES[p.feature]
        assert p.pct_target is None and p.pct_decoy is None and (p.n_target, p.n_decoy) == (0, 0)
        assert p.valid is None and p.rule is None
        assert f"the validity rule {info.validity} cannot be evaluated" in p.note
        assert "not ranked" in p.population and not p.population.startswith("valid")
    # On request, ranked against every row, labelled as unfiltered.
    (p,) = feature_percentiles(rs, "", values, ["coelution_mean"], rank_unfiltered=True)
    everything = _pandas_percentile(features, scored, "coelution_mean", values["coelution_mean"])
    assert p.valid is None and p.rule is None
    assert (p.n_target, p.n_decoy) == (everything["target"][1], everything["decoy"][1])
    assert p.pct_target == pytest.approx(everything["target"][0], abs=1e-9)
    assert "not filtered by a validity rule" in p.population
    assert "sentinel values are included" in p.population and not p.population.startswith("valid")
    # A rule that the population cannot apply is still checked on inputs the caller gives.
    given_inputs = {"coelution_mean": 0.5, "n_observations": 2.0}
    (q,) = feature_percentiles(rs, "", given_inputs, ["coelution_mean"])
    assert q.valid is False and q.pct_target is None
    # No handle on psms_extracted.parquet stays open (the engine publishes by rename).
    for name in ("psms_extracted.parquet", "features.parquet"):
        path = copy / name
        os.replace(path, path.with_name(name + ".moved"))
        os.replace(path.with_name(name + ".moved"), path)


def test_rt_rule_without_predicted_rt_raw_or_psms_extracted(fixture_dir, tmp_path):
    """Band tables without predicted_rt_raw: every band row holds the no-calibration 0."""
    copy = tmp_path / "grouped"
    shutil.copytree(fixture_dir("grouped"), copy)
    for band in ("g00", "g01", "g02"):
        path = copy / "groups" / band / "psms_competed.parquet"
        table = pq.read_table(path)
        pq.write_table(table.select([c for c in table.column_names if c not in MINIMAL_DROP]), path)
        _rehash(copy, f"psms_competed[{band}]", path)
    rs = open_results(copy)
    cid = int(pq.read_table(copy / "psms_scored.parquet")["candidate_id"][0].as_py())
    values = candidate_features(rs, "", cid)
    assert values["rt_error_abs"] == 0.0
    rt, corr = feature_percentiles(rs, "", values, ["rt_error_abs", "frag_corr"])
    assert rt.pct_target is None and rt.valid is None and (rt.n_target, rt.n_decoy) == (0, 0)
    assert "no readable psms_extracted.parquet with rt_pred_cal" in rt.note
    assert corr.valid is True and corr.n_target + corr.n_decoy == 305


def test_overlapping_bands_need_their_pooled_table(open_fixture, fixture_dir, tmp_path):
    src = fixture_dir("ovl_bp")
    # With the pooled table every scored row counts once (the bands hold 10 rows twice).
    rs = open_fixture("ovl_bp")
    bands = pd.concat(
        [
            pq.read_table(src / "groups" / b / "psms_competed.parquet").to_pandas()
            for b in ("g00", "g01", "g02")
        ]
    )
    twice = bands.loc[bands["candidate_id"].duplicated(), "candidate_id"]
    assert len(twice) == 5
    values = candidate_features(rs, "", int(twice.iloc[0]))
    (p,) = feature_percentiles(rs, "", values, ["frag_corr"])
    assert p.n_target + p.n_decoy == 161 == pq.read_metadata(src / "psms_scored.parquet").num_rows
    # The pooled table is deleted, but the manifest still records it: its own error.
    copy = tmp_path / "deleted"
    shutil.copytree(src, copy)
    for suffix in ("", ".report.json", ".schema.json"):
        (copy / f"psms_competed.parquet{suffix}").unlink()
    rs = open_results(copy)
    for read in (
        lambda: feature_source(rs, ""),
        lambda: candidate_features(rs, "", int(twice.iloc[0])),
        lambda: feature_percentiles(rs, "", {"frag_corr": 0.5}),
    ):
        with pytest.raises(ArtifactNotFound, match=r"pooled psms_competed\.parquet, which cannot"):
            read()
    # No record either: the loser rows show that the bands overlap.
    manifest = copy / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["artifacts"]["psms_competed"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ArtifactNotFound, match=r"overlap_losers\.parquet has 11 rows"):
        feature_source(open_results(copy), "")
    # A pooled table of an unknown schema version is refused, not bypassed.
    copy = tmp_path / "v99"
    shutil.copytree(src, copy)
    _set_version(copy, "psms_competed", copy / "psms_competed.parquet", 99)
    with pytest.raises(SchemaVersionError, match="version 99"):
        feature_source(open_results(copy), "")


def test_a_band_table_of_an_unknown_version_is_refused(fixture_dir, tmp_path):
    copy = tmp_path / "grouped"
    shutil.copytree(fixture_dir("grouped"), copy)
    _set_version(copy, "psms_competed[g01]", copy / "groups" / "g01" / "psms_competed.parquet", 99)
    with pytest.raises(SchemaVersionError, match="version 99"):
        feature_source(open_results(copy), "")


def test_a_full_row_is_ranked_in_batches(open_fixture, fixture_dir, monkeypatch):
    rs = open_fixture("single")
    values = candidate_features(rs, "", _first_cid(rs))
    names = feature_columns(rs, "")
    results = feature_percentiles(rs, "", values)
    assert [p.feature for p in results] == names and len(names) == 387
    sizes: list[int] = []
    batch = F._percentile_batch

    def counted(*args, **kwargs):
        sizes.append(len(args[4]))
        return batch(*args, **kwargs)

    monkeypatch.setattr(F, "_percentile_batch", counted)
    monkeypatch.setattr(F, "_BATCH", 50)
    assert feature_percentiles(rs, "", values) == results
    assert sizes == [50] * 7 + [37]
    # Features without a known rule are ranked against every row, and say so.
    features, scored = _frames(fixture_dir("single"), "features.parquet", "psms_scored.parquet")
    by_name = {p.feature: p for p in results}
    plain = [n for n in names if n not in EVIDENCE_FEATURES and n not in CONTEXT_COLUMNS]
    for name in (plain[0], plain[len(plain) // 2], plain[-1]):
        expected = _pandas_percentile(features, scored, name, values[name])
        p = by_name[name]
        assert p.pct_target == pytest.approx(expected["target"][0], abs=1e-9)
        assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])
        assert "no validity rule is defined" in p.population and p.rule is None


def test_duckdb_errors_name_the_run_and_the_features(open_fixture, monkeypatch):
    rs = open_fixture("single")
    values = candidate_features(rs, "", _first_cid(rs))

    def fail(*args, **kwargs):
        raise duckdb.OutOfMemoryException("Out of Memory Error: failed to allocate 512.0 KiB")

    monkeypatch.setattr(F, "_percentile_batch", fail)
    with pytest.raises(ViewerError, match=r"run out: DuckDB failed .* 64 of 387 feature\(s\)"):
        feature_percentiles(rs, "", values)


def test_nullable_columns_are_counted_per_feature(fixture_dir, tmp_path):
    """Columns with nulls do not share the row count of the columns without nulls."""
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    path = copy / "features.parquet"
    table = pq.read_table(path)
    schema = table.schema.set(
        table.schema.get_field_index("frag_corr"), pa.field("frag_corr", pa.float32())
    )
    df = table.to_pandas()
    df.loc[df.index[::10], "frag_corr"] = None
    _write_like(df, path, schema)
    _rehash(copy, "features", path)
    rs = open_results(copy)
    source = feature_source(rs, "")
    non_null = F._non_null_columns(source.artifacts)
    assert "frag_corr" not in non_null and "frag_cosine" in non_null
    scored = _scored(copy / "psms_scored.parquet")
    values = candidate_features(rs, "", int(df["candidate_id"].iloc[1]))
    for p in feature_percentiles(rs, "", values, ["frag_corr", "frag_cosine"]):
        expected = _pandas_percentile(df, scored, p.feature, values[p.feature])
        present = df[p.feature].notna()
        assert (p.n_target, p.n_decoy) == (
            int((present & (df["label"] == "target")).sum()),
            int((present & (df["label"] == "decoy")).sum()),
        )
        sub = df.loc[present & (df["label"] == "target"), p.feature].astype("float64")
        assert p.pct_target == pytest.approx(100.0 * float((sub <= values[p.feature]).mean()))
        if p.feature == "frag_cosine":
            assert (p.n_target, p.n_decoy) == (expected["target"][1], expected["decoy"][1])


def test_predicate_columns_skip_function_names():
    assert F.predicate_columns("isfinite(rt_pred_cal)") == ("rt_pred_cal",)
    assert F.predicate_columns("(n_matched_b + n_matched_y) >= 2") == ("n_matched_b", "n_matched_y")
    assert (
        F._qualify("isfinite(rt_pred_cal) and x > 0", "e")
        == 'isfinite(e."rt_pred_cal") and e."x" > 0'
    )


def test_evidence_feature_definitions(open_fixture, fixture_dir):
    rs = open_fixture("single")
    names = set(pq.read_schema(feature_source(rs, "").paths[0]).names)
    extracted = set(pq.read_schema(fixture_dir("single") / "psms_extracted.parquet").names)
    assert len(EVIDENCE_FEATURES) == 29
    for name, info in EVIDENCE_FEATURES.items():
        assert info.name == name and info.label and info.unit and info.description and info.note
        assert chr(0x2014) not in info.description + info.note  # no em dashes
        if info.table == "features":
            assert name in names, name
        assert set(info.validity_columns) <= names, name
        assert set(info.extracted_validity_columns) <= extracted, name
        # Every rule has an independent pandas form in the tests.
        assert info.validity is None or info.validity in ORACLE_RULES, name
    assert EVIDENCE_FEATURES["rt_error_abs"].extracted_validity == "isfinite(rt_pred_cal)"
    assert EVIDENCE_FEATURES["rt_error_signed"].extracted_validity_columns == ("rt_pred_cal",)
    width = EVIDENCE_FEATURES["rt_error_over_peak_width"].note
    assert "fewer than 3 window points" in width and "stays on one point" in width
    assert EVIDENCE_FEATURES["contested_frac"].table == "psms_extracted"
    assert not EVIDENCE_FEATURES["has_ms1"].percentile
    assert EVIDENCE_FEATURES["has_ms1"].label == "MS1 data available"
    assert "0 ppm" in EVIDENCE_FEATURES["mean_mass_error"].note
    assert EVIDENCE_FEATURES["spectral_angle_matched"].validity_columns == (
        "n_matched_b",
        "n_matched_y",
    )
    for info in EVIDENCE_FEATURES.values():
        # Every validity input other than the feature itself is read with any projection.
        assert set(info.validity_columns) - {info.name} <= set(CONTEXT_COLUMNS), info.name


def test_reads_do_not_write_to_the_run_directory(fixture_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("topk"), copy)
    before = {p: os.stat(p).st_mtime_ns for p in copy.rglob("*") if p.is_file()}
    rs = open_results(copy)
    values = candidate_features(rs, "", _first_cid(rs))
    feature_percentiles(rs, "", values, ["frag_corr", "coelution_mean"])
    after = {p: os.stat(p).st_mtime_ns for p in copy.rglob("*") if p.is_file()}
    assert before == after


def test_no_file_handle_stays_open(fixture_dir, tmp_path):
    """The engine publishes by rename; an open handle would block it on Windows."""
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    rs = open_results(copy)
    values = candidate_features(rs, "", _first_cid(rs))
    feature_percentiles(rs, "", values, ["frag_corr", "coelution_mean"])
    feature_columns(rs, "")
    for name in ("features.parquet", "psms_competed.parquet", "psms_scored.parquet"):
        path = copy / name
        moved = path.with_name(name + ".moved")
        os.replace(path, moved)
        os.replace(moved, path)


# --------------------------------------------------------------------------- real data


TWENTY = [
    "frag_corr",
    "frag_cosine",
    "spectral_angle",
    "spectral_angle_matched",
    "coelution_mean",
    "coelution_best",
    "coelution_run",
    "n_matched_fragments",
    "median_abs_frag_ppm",
    "signed_mean_frag_ppm",
    "frag_mass_err_median",
    "weighted_mass_error",
    "mean_mass_error",
    "log_ms1_mono",
    "ms1_isotope_cosine_apex",
    "n_interfered_fragments",
    "interference_apex_residual_fraction",
    "rt_error_signed",
    "rt_error_abs",
    "rt_error_over_peak_width",
]


@pytest.mark.real_data
def test_real_single_features_and_percentiles(real_single):
    rs = open_results(real_single)
    scored = pq.read_table(
        rs.scored.path, columns=["candidate_id", "selected_peak_rank"]
    ).to_pandas()
    sample = scored.iloc[:: max(1, len(scored) // 7)].head(7)
    source = feature_source(rs, "")
    t0 = time.perf_counter()
    first = candidate_features(rs, "", int(sample.candidate_id.iloc[0]), columns=TWENTY)
    cold_row = time.perf_counter() - t0
    timings_row, timings_pct = [], []
    for s in sample.itertuples():
        t0 = time.perf_counter()
        row = candidate_features(rs, "", int(s.candidate_id), int(s.selected_peak_rank), TWENTY)
        timings_row.append(time.perf_counter() - t0)
        expected = rs.duck.execute(
            "SELECT * FROM read_parquet(?) WHERE candidate_id = ? AND peak_rank = ?",
            [
                str(source.paths[0]).replace("\\", "/"),
                int(s.candidate_id),
                int(s.selected_peak_rank),
            ],
        ).fetchdf()
        assert len(expected) == 1
        assert all(row[k] == pytest.approx(expected[k].iloc[0]) for k in TWENTY)
        t0 = time.perf_counter()
        results = feature_percentiles(rs, "", row, TWENTY)
        timings_pct.append(time.perf_counter() - t0)
        assert len(results) == 20
        assert all(p.n_target > 0 and p.n_decoy > 0 for p in results)
    assert first is not None
    # The default ranks every feature of a full row: 387 on Astral, in batches that stay
    # inside the DuckDB memory limit (one query over all of them ran out of memory).
    full = candidate_features(rs, "", int(sample.candidate_id.iloc[1]))
    t0 = time.perf_counter()
    everything = feature_percentiles(rs, "", full)
    full_time = time.perf_counter() - t0
    assert [p.feature for p in everything] == feature_columns(rs, "")
    print(
        f"\nAstral: first candidate_features {cold_row * 1e3:.1f} ms; warm "
        f"{min(timings_row) * 1e3:.1f}-{max(timings_row) * 1e3:.1f} ms; 20-feature percentiles "
        f"{min(timings_pct) * 1e3:.1f}-{max(timings_pct) * 1e3:.1f} ms; all "
        f"{len(everything)} features {full_time * 1e3:.0f} ms"
    )
    assert max(timings_row) < 1.0 and max(timings_pct[1:]) < 1.0 and full_time < 3.0
