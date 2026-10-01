"""Identification counts, per-run counts, curves, histograms and group winners.

The reference numbers are computed independently here with pyarrow.parquet and
pyarrow.compute (no DuckDB).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.counts import (
    ENGINE_STAT_KEYS,
    MAX_WINNER_KEYS,
    engine_check,
    engine_stats,
    group_winner_sql,
    group_winners,
    id_curve,
    per_run_counts,
    score_histogram,
    unit_counts,
)
from mumdia_viewer.data.hashing import blake3_file
from mumdia_viewer.data.units import (
    COUNT_UNITS,
    UNITS,
    bind_params,
    check_threshold,
    format_threshold,
    get_unit,
)

COUNT_FIXTURES = ["single", "grouped", "experiment", "topk", "mbr", "ovl_bp"]
REPORT_FIXTURES = [
    "single",
    "chrom_v1",
    "chrom_v2_rg1",
    "grouped",
    "grouped_pool",
    "experiment",
    "topk",
    "mbr",
    "ovl_bp",
    "ovl_bp_pool",
    "ovl_rg50",
    "ovl_rg50_pool",
    "ovl128",
    "ovl128_pool",
]
COLUMNS = [
    "label",
    "source",
    "q_value",
    "run_psm_q",
    "precursor_q",
    "peptide_q_value",
    "pg_q_value",
    "peptidoform",
    "charge",
    "base_peptide_id",
    "protein_group",
]


# --------------------------------------------------------------------------- reference

# The units, written out here and not read from mumdia_viewer.data.units, so that the
# reference shares no definition with the code under test: (q column, counted key
# columns; no key column means rows). A protein group is a non-empty protein_group.
REFERENCE_UNITS: dict[str, tuple[str, tuple[str, ...]]] = {
    "psm": ("q_value", ()),
    "precursor": ("precursor_q", ("peptidoform", "charge")),
    "peptide": ("peptide_q_value", ("base_peptide_id",)),
    "protein_group": ("pg_q_value", ("protein_group",)),
}
REFERENCE_THRESHOLDS = [0.01, 0.05, 0.1, 0.5]


def _table(rs) -> pa.Table:
    return pq.read_table(rs.scored.path, columns=COLUMNS)


def _n_keys(table: pa.Table, unit: str) -> int:
    """Rows or distinct keys of an already filtered table (pyarrow only)."""
    _, keys = REFERENCE_UNITS[unit]
    if unit == "protein_group":
        table = table.filter(pc.not_equal(table["protein_group"], ""))
    if not keys:
        return table.num_rows
    return table.group_by(list(keys)).aggregate([]).num_rows


def _reference_mask(table: pa.Table, unit: str, rows, t: float) -> int:
    """Keys of ``unit`` among the rows of a boolean mask that pass its q column at ``t``."""
    q_column, _ = REFERENCE_UNITS[unit]
    return _n_keys(table.filter(pc.and_(rows, pc.less_equal(table[q_column], t))), unit)


def _reference(table: pa.Table, unit: str, label: str, t: float) -> int:
    return _reference_mask(table, unit, pc.equal(table["label"], label), t)


# --------------------------------------------------------------------------- units


def test_unit_labels_are_exact():
    assert (
        UNITS["peptide"].label(81312, 0.01)
        == "81,312 peptides (unique base_peptide_id, peptide_q_value <= 0.01)"
    )
    assert UNITS["psm"].label(531720, 0.01) == "531,720 PSMs (rows, q_value <= 0.01)"
    assert (
        UNITS["precursor"].label(127386, 0.01)
        == "127,386 precursors (unique (peptidoform, charge), precursor_q <= 0.01)"
    )
    assert (
        UNITS["protein_group"].label(12335, 0.05)
        == "12,335 protein groups (unique protein_group, pg_q_value <= 0.05)"
    )
    assert UNITS["run_psm"].label(1, 0.01) == "1 PSM (rows, run_psm_q <= 0.01)"
    assert format_threshold(0.01) == "0.01" and format_threshold(1.0) == "1"
    assert get_unit("peptide") is UNITS["peptide"]
    with pytest.raises(ValueError, match="unknown unit"):
        get_unit("stripped_sequence")


def test_check_threshold():
    assert check_threshold("psm", 1.0) == 1.0
    assert check_threshold("run_psm", 0.5) == 0.5
    assert check_threshold("peptide", 0.999999) == 0.999999
    for unit in ("precursor", "peptide", "protein_group"):
        with pytest.raises(ValueError, match="cannot be told") as err:
            check_threshold(unit, 1.0)
        assert UNITS[unit].q_column in str(err.value)
    for bad in (0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            check_threshold("psm", bad)
    with pytest.raises(ValueError, match="at most 1"):
        check_threshold("psm", 1.5)
    with pytest.raises(ValueError, match="number"):
        check_threshold("psm", "0.01x")
    with pytest.raises(ValueError, match="number"):
        check_threshold("psm", True)


def test_bind_params_keeps_only_used_names():
    assert bind_params("SELECT $a + $a", {"a": 1, "b": 2}) == {"a": 1}
    with pytest.raises(KeyError, match="c"):
        bind_params("SELECT $c", {"a": 1})


# --------------------------------------------------------------------------- counts


@pytest.mark.parametrize("name", COUNT_FIXTURES)
@pytest.mark.parametrize("t", REFERENCE_THRESHOLDS)
def test_counts_equal_an_independent_pyarrow_count(open_fixture, name, t):
    """The SPEC acceptance test: each unit's count equals a pyarrow count.

    The smallest target pg_q_value of these fixtures is 1/16, so protein groups pass only
    from t = 0.1 on; 0.1 and 0.5 make the protein-group comparison non-trivial.
    """
    rs = open_fixture(name)
    table = _table(rs)
    counts = {c.unit: c for c in unit_counts(rs, t)}
    assert list(counts) == list(COUNT_UNITS)
    for unit in COUNT_UNITS:
        c = counts[unit]
        assert c.n_target == _reference(table, unit, "target", t), (name, unit)
        assert c.n_decoy == _reference(table, unit, "decoy", t), (name, unit)
        assert c.label.startswith(UNITS[unit].label(c.n_target, t)), c.label
        assert c.q_column == REFERENCE_UNITS[unit][0] and c.q_column in c.sql
        assert not c.derived
        assert c.n_spike_in is None  # no entrapment markers in these fixtures
    if t >= 0.1:
        assert counts["protein_group"].n_target > 0, name


ENTRAPMENT_TOOLS = Path(__file__).parent / "fixtures" / "tools" / "entrapment"


@pytest.fixture(scope="module")
def entrapment_rs(fixture_dir, tmp_path_factory):
    """The entrapment fixture (no manifest) with a minimal manifest and its configuration."""
    src = fixture_dir("entrapment")
    dst = tmp_path_factory.mktemp("counts_entrapment") / "run"
    dst.mkdir()
    for name in ("psms_scored.parquet", "psms_scored.parquet.report.json"):
        shutil.copy2(src / name, dst / name)
    report = json.loads((dst / "psms_scored.parquet.report.json").read_text(encoding="utf-8"))
    config = (ENTRAPMENT_TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8")
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", str(dst)],
        "config_json": json.dumps(json.loads(config)),
        "artifacts": {
            "psms_scored": {
                "path": str(dst / "psms_scored.parquet"),
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": report["content_hash"],
                "producing_stage": "rescore",
            }
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return open_results(dst)


@pytest.mark.parametrize("t", REFERENCE_THRESHOLDS)
def test_entrapment_counts_equal_an_independent_pyarrow_count(entrapment_rs, t):
    """Entrapment mode: real targets, decoys and spike-ins per unit against pyarrow.

    The spike-in rule of the run's configuration is written out here: a target whose
    protein contains ENTRAP_, not REAL_, and none of the contaminant tokens.
    """
    rs = entrapment_rs
    table = pq.read_table(rs.scored.path, columns=[*COLUMNS, "protein"])
    target = pc.equal(table["label"], "target")
    spike = pc.and_(target, pc.match_substring(table["protein"], "ENTRAP_"))
    for token in ("REAL_", "KRT", "K1C", "K2C", "ALBU", "TRYP"):
        spike = pc.and_(spike, pc.invert(pc.match_substring(table["protein"], token)))
    real = pc.and_(target, pc.invert(spike))
    decoy = pc.equal(table["label"], "decoy")
    counts = {c.unit: c for c in unit_counts(rs, t)}
    for unit in COUNT_UNITS:
        c = counts[unit]
        assert c.n_target == _reference_mask(table, unit, real, t), (unit, t)
        assert c.n_decoy == _reference_mask(table, unit, decoy, t), (unit, t)
        assert c.n_spike_in == _reference_mask(table, unit, spike, t), (unit, t)
    assert counts["protein_group"].n_target > 0 and counts["protein_group"].n_spike_in > 0


def test_an_empty_protein_group_is_not_counted(fixture_dir, tmp_path):
    """A winning row with protein_group '' is not a protein group (no fixture has one)."""
    root = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), root)
    path = root / "psms_scored.parquet"
    table = pq.read_table(path)
    winners = pc.and_(pc.equal(table["label"], "target"), pc.less(table["pg_q_value"], 1.0))
    i = int(pc.indices_nonzero(winners)[0].as_py())
    groups = table["protein_group"].to_pylist()
    groups[i] = ""
    index = table.schema.get_field_index("protein_group")
    table = table.set_column(index, table.schema.field(index), pa.array(groups, pa.string()))
    pq.write_table(table, path)
    digest = blake3_file(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["psms_scored"]["content_hash"] = digest
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rs = open_results(root)
    counts = {c.unit: c for c in unit_counts(rs, 0.1)}
    # 16 protein groups pass 0.1 in the fixture; the one whose winner lost its string is
    # not counted, and '' is not a group.
    reference = _table(rs)
    assert counts["protein_group"].n_target == _reference(reference, "protein_group", "target", 0.1)
    assert counts["protein_group"].n_target == 15
    # The per-run count leaves the empty string out as well.
    target = pc.equal(reference["label"], "target")
    accepted = reference.filter(pc.and_(target, pc.less_equal(reference["run_psm_q"], 0.5)))
    assert "" in accepted["protein_group"].to_pylist()
    per_run = int(per_run_counts(rs, 0.5)["protein_groups"].iloc[0])
    assert (
        per_run
        == _n_keys(accepted, "protein_group")
        == len(set(accepted["protein_group"].to_pylist()) - {""})
    )


@pytest.mark.parametrize("name", REPORT_FIXTURES)
def test_engine_check_is_all_equal(open_fixture, name):
    rs = open_fixture(name)
    checks = engine_check(rs)
    assert [c.unit for c in checks] == list(COUNT_UNITS)
    assert all(c.equal for c in checks), checks
    stats = engine_stats(rs)
    assert set(ENGINE_STAT_KEYS.values()) <= set(stats)
    assert "classifier" not in stats and "psms" not in stats


def test_single_run_numbers(open_fixture):
    rs = open_fixture("single")
    checks = {c.unit: (c.engine, c.viewer) for c in engine_check(rs)}
    assert checks == {
        "psm": (273, 273),
        "precursor": (273, 273),
        "peptide": (150, 150),
        "protein_group": (0, 0),
    }
    at5 = {c.unit: c.n_target for c in unit_counts(rs, 0.05)}
    assert at5 == {"psm": 282, "precursor": 282, "peptide": 151, "protein_group": 0}
    counts = {c.unit: c for c in unit_counts(rs, 0.01)}
    assert counts["psm"].label == "273 PSMs (rows, q_value <= 0.01)"
    assert counts["peptide"].label == (
        "150 peptides (unique base_peptide_id, peptide_q_value <= 0.01)"
    )
    assert counts["psm"].population == "psms_scored.parquet: single run"
    # 16 target protein groups make the floor 1/16 = 0.0625, above 0.01.
    assert "0.0625" in counts["protein_group"].note
    assert "empty protein_group" in counts["protein_group"].note


def test_experiment_counts_are_pooled(open_fixture):
    rs = open_fixture("experiment")
    counts = {c.unit: c for c in unit_counts(rs, 0.01)}
    assert counts["psm"].n_target == 564  # pooled q_value over both copies
    assert counts["peptide"].n_target == 150
    assert "all 2 runs pooled (a, b)" in counts["psm"].population
    assert "experiment-wide" in counts["peptide"].note


def test_counts_are_memoised(open_fixture):
    rs = open_fixture("single")
    first = unit_counts(rs, 0.01)
    keys = {k[:3] for k in rs._memo if isinstance(k, tuple) and k[0] == "unit_counts"}
    assert ("unit_counts", rs.scored.identity(), 0.01) in keys
    again = unit_counts(rs, 0.01)
    assert again == first and all(a is b for a, b in zip(again, first, strict=True))
    frame = per_run_counts(rs, 0.01)
    frame.loc[0, "target_psms"] = -1  # a caller's change does not reach the memo
    assert int(per_run_counts(rs, 0.01)["target_psms"].iloc[0]) == 273


def test_base_peptide_competition_relabels_the_precursor_count(open_fixture):
    counts = {c.unit: c for c in unit_counts(open_fixture("ovl_bp"), 0.01)}
    assert "base peptide" in counts["precursor"].label
    assert "group_by" in counts["precursor"].label
    single = {c.unit: c for c in unit_counts(open_fixture("single"), 0.01)}
    assert "base peptide" not in single["precursor"].label


def test_mbr_counts_are_native(open_fixture):
    rs = open_fixture("mbr")
    counts = {c.unit: c for c in unit_counts(rs, 0.01)}
    assert counts["psm"].population.startswith("scored_combined.parquet")
    assert "transfers are not in this count" in counts["psm"].note


def test_thresholds_are_validated(open_fixture):
    rs = open_fixture("single")
    with pytest.raises(ValueError, match="cannot be told"):
        unit_counts(rs, 1.0)
    with pytest.raises(ValueError):
        unit_counts(rs, 0.0)


# --------------------------------------------------------------------------- per run


def test_experiment_per_run_counts_use_run_psm_q(open_fixture):
    rs = open_fixture("experiment")
    df = per_run_counts(rs, 0.01)
    assert list(df["run"]) == ["a", "b"] and list(df["source"]) == [0, 1]
    assert dict(zip(df["run"], df["target_psms"], strict=True)) == {"a": 273, "b": 273}
    table = _table(rs)
    for _, row in df.iterrows():
        part = table.filter(pc.equal(table["source"], int(row["source"])))
        target = pc.equal(part["label"], "target")
        accepted = part.filter(pc.and_(target, pc.less_equal(part["run_psm_q"], 0.01)))
        assert row["target_psms"] == accepted.num_rows
        assert row["precursors"] == _n_keys(accepted, "precursor")
        assert row["peptides"] == _n_keys(accepted, "peptide")
        assert row["protein_groups"] == _n_keys(accepted, "protein_group") > 0
        decoys = part.filter(
            pc.and_(pc.equal(part["label"], "decoy"), pc.less_equal(part["run_psm_q"], 0.01))
        )
        assert row["decoy_psms"] == decoys.num_rows
    labels = df.attrs["labels"]
    for column in ("precursors", "peptides", "protein_groups"):
        assert df.attrs["derived"][column] is True
        assert "PSM-level FDR within the run" in labels[column]
    assert df.attrs["derived"]["target_psms"] is False
    assert "run_psm_q <= 0.01" in labels["target_psms"]
    assert "never counted per run" in df.attrs["note"]


def test_per_run_count_on_a_grouped_column_would_be_wrong(open_fixture):
    """Guard: why per_run_counts never counts a run on peptide_q_value.

    The two runs of the fixture are identical copies. Every experiment-wide winner is
    the first copy in file order, so a per-run count on peptide_q_value gives
    {a: 150, b: 0}, while each run holds 150 peptides with an accepted PSM.
    """
    rs = open_fixture("experiment")
    table = _table(rs)
    wrong = {}
    for source in (0, 1):
        part = table.filter(pc.equal(table["source"], source))
        mask = pc.and_(
            pc.equal(part["label"], "target"), pc.less_equal(part["peptide_q_value"], 0.01)
        )
        wrong[source] = _n_keys(part.filter(mask), "peptide")
    assert wrong == {0: 150, 1: 0}
    assert list(per_run_counts(rs, 0.01)["peptides"]) == [150, 150]
    assert "peptide_q_value" not in per_run_counts(rs, 0.01).attrs["sql"]


def test_single_run_per_run_counts(open_fixture):
    df = per_run_counts(open_fixture("single"), 0.01)
    assert list(df["run"]) == ["run"] and int(df["target_psms"].iloc[0]) == 273
    assert "spike_in_psms" not in df.columns


# --------------------------------------------------------------------------- curves


def _reference_curve(table: pa.Table, unit: str, label: str, thresholds: np.ndarray) -> list:
    return [_reference(table, unit, label, float(q)) for q in thresholds]


@pytest.mark.parametrize("name", ["single", "experiment", "mbr"])
def test_id_curves_are_exact_monotone_and_end_at_the_count(open_fixture, name):
    rs = open_fixture(name)
    table = _table(rs)
    for unit in COUNT_UNITS:
        curve = id_curve(rs, unit, q_max=0.05, points=40)
        assert list(curve.columns) == ["q", "count"] and len(curve) == 40
        assert curve["q"].iloc[-1] == 0.05
        counts = curve["count"].to_numpy()
        assert np.all(np.diff(counts) >= 0)
        end = {c.unit: c.n_target for c in unit_counts(rs, 0.05)}[unit]
        assert counts[-1] == end
        assert list(counts) == _reference_curve(table, unit, "target", curve["q"].to_numpy())
    decoys = id_curve(rs, "psm", q_max=0.05, points=10, label="decoy")
    assert decoys["count"].iloc[-1] == _reference(table, "psm", "decoy", 0.05)


def test_id_curve_is_exact_on_bin_edges(open_fixture):
    """Thresholds that equal a stored q value exactly must count that value."""
    for name in ("single", "grouped", "mbr"):
        rs = open_fixture(name)
        table = _table(rs)
        for unit in ("psm", "peptide"):
            values = np.unique(table[UNITS[unit].q_column].to_numpy())
            values = values[values < 1.0]
            smallest, middle, largest = (float(v) for v in values[[0, len(values) // 2, -1]])
            cases = ((smallest, 1), (2 * smallest, 2), (middle, 3), (largest, 7), (largest, 200))
            for q_max, points in cases:
                curve = id_curve(rs, unit, q_max=q_max, points=points)
                expected = _reference_curve(table, unit, "target", curve["q"].to_numpy())
                assert list(curve["count"]) == expected, (name, unit, q_max, points)
            first = id_curve(rs, unit, q_max=smallest, points=1)["count"].iloc[0]
            assert first == _reference(table, unit, "target", smallest)


def test_id_curve_bin_correction_on_synthetic_values(fixture_dir, tmp_path):
    """q values on a threshold and one ulp either side of it, where ceil(q / step) errs."""
    root = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), root)
    path = root / "psms_scored.parquet"
    table = pq.read_table(path)
    step = 0.1
    edges = np.arange(1, 11, dtype=np.float64) * step
    edges[-1] = 1.0
    pool = np.concatenate(
        [edges, np.nextafter(edges, 0.0), np.nextafter(edges, 2.0), [0.3, 0.7, 0.6, 0.9]]
    )
    pool = pool[(pool > 0) & (pool <= 1.0)]
    values = np.resize(pool, table.num_rows)
    index = table.schema.get_field_index("q_value")
    table = table.set_column(index, table.schema.field(index), pa.array(values))
    pq.write_table(table, path)
    rs = open_results(root)
    curve = id_curve(rs, "psm", q_max=1.0, points=10)
    assert list(curve["q"]) == list(edges)
    target = table.filter(pc.equal(table["label"], "target"))["q_value"].to_numpy()
    assert list(curve["count"]) == [int((target <= q).sum()) for q in edges]


def test_per_run_curves(open_fixture):
    rs = open_fixture("experiment")
    curve = id_curve(rs, "run_psm", q_max=0.05, points=20, per_run=True)
    assert set(curve.columns) == {"q", "count", "run", "source"}
    ends = curve.groupby("run")["count"].last().to_dict()
    per_run = per_run_counts(rs, 0.05)
    assert ends == dict(zip(per_run["run"], per_run["target_psms"], strict=True))
    with pytest.raises(ValueError, match="experiment-wide"):
        id_curve(rs, "peptide", per_run=True)
    with pytest.raises(ValueError, match="per_run=True"):
        id_curve(rs, "run_psm")
    with pytest.raises(ValueError, match="below 1"):
        id_curve(rs, "peptide", q_max=1.0)
    with pytest.raises(ValueError, match="spike-ins"):
        id_curve(rs, "psm", label="spike_in")
    with pytest.raises(ValueError, match="points"):
        id_curve(rs, "psm", points=0)
    single = id_curve(open_fixture("single"), "run_psm", q_max=0.05, points=5)
    assert list(single.columns) == ["q", "count"]


# --------------------------------------------------------------------------- histograms


@pytest.mark.parametrize("name", ["single", "experiment", "grouped", "mbr"])
def test_score_histogram_totals_equal_row_counts(open_fixture, name):
    rs = open_fixture(name)
    hist = score_histogram(rs, bins=50)
    assert list(hist.columns) == ["bin_lo", "bin_hi", "target", "decoy"]
    assert len(hist) == 50
    table = pq.read_table(rs.scored.path, columns=["label", "source", "score"])
    assert int(hist["target"].sum() + hist["decoy"].sum()) == table.num_rows
    assert int(hist["target"].sum()) == int(pc.sum(pc.equal(table["label"], "target")).as_py())
    assert hist["bin_lo"].iloc[0] == pc.min(table["score"]).as_py()
    assert hist["bin_hi"].iloc[-1] == pc.max(table["score"]).as_py()
    assert hist.attrs["range_source"] == "footer statistics"
    for run in rs.runs:
        part = score_histogram(rs, bins=50, run=run.name if run.name else run.index)
        rows = int(pc.sum(pc.equal(table["source"], run.index)).as_py())
        assert int(part["target"].sum() + part["decoy"].sum()) == rows
        assert list(part["bin_lo"]) == list(hist["bin_lo"])  # the bins line up across runs


# --------------------------------------------------------------------------- winners

GROUPED = {"peptide": "peptide_q_value", "precursor": "precursor_q", "protein_group": "pg_q_value"}
WINNER_KEYS = {
    "peptide": lambda df: [int(v) for v in df["base_peptide_id"]],
    "precursor": lambda df: [
        (p, int(c)) for p, c in zip(df["peptidoform"], df["charge"], strict=True)
    ],
    "protein_group": lambda df: list(df["protein"]),
}


def _all_winners(rs, unit: str):
    """Every group's winner, from the whole-table SQL run inside DuckDB."""
    sql, params = group_winner_sql(rs, unit)
    return rs.duck.execute(sql, bind_params(sql, params)).df()


def _row_ids(df) -> set:
    return set(zip(df["source"].tolist(), df["candidate_id"].tolist(), strict=True))


def test_group_winners_follow_the_engine_rule(open_fixture):
    for name in ("single", "experiment", "grouped"):
        rs = open_fixture(name)
        table = pq.read_table(rs.scored.path, columns=["source", "candidate_id", *GROUPED.values()])
        for unit, column in GROUPED.items():
            winners = _all_winners(rs, unit)
            below = table.filter(pc.less(table[column], 1.0))
            expected = set(
                zip(below["source"].to_pylist(), below["candidate_id"].to_pylist(), strict=True)
            )
            assert _row_ids(winners) == expected, (name, unit)
            # The DataFrame helper with every key gives the same rows (small fixtures).
            keyed = group_winners(rs, unit, keys=WINNER_KEYS[unit](winners))
            assert _row_ids(keyed) == expected, (name, unit)
    # The experiment's runs are identical copies: every group ties, and the first copy
    # in file order (run a) wins.
    rs = open_fixture("experiment")
    winners = _all_winners(rs, "peptide")
    keyed = group_winners(rs, "peptide", keys=WINNER_KEYS["peptide"](winners))
    assert set(keyed["run"]) == {"a"}


def test_group_winners_for_selected_keys(open_fixture):
    rs = open_fixture("experiment")
    all_winners = _all_winners(rs, "peptide")
    some = [int(v) for v in all_winners["base_peptide_id"].iloc[:3]]
    picked = group_winners(rs, "peptide", keys=some)
    assert sorted(picked["base_peptide_id"]) == sorted(some)
    assert picked["file_row_number"].is_monotonic_increasing
    row = all_winners.iloc[0]
    pair = group_winners(rs, "precursor", keys=[(row["peptidoform"], int(row["charge"]))])
    assert len(pair) == 1 and pair["peptidoform"].iloc[0] == row["peptidoform"]
    protein = group_winners(rs, "protein_group", keys=[row["protein"]])
    assert list(protein["protein"]) == [row["protein"]]
    empty = group_winners(rs, "peptide", keys=[])
    assert empty.empty and "run" in empty.columns
    sql, params = group_winner_sql(rs, "peptide")
    assert "(label = 'decoy') DESC" in sql and "file_row_number" in sql
    assert set(params) == {"path"}
    with pytest.raises(ValueError, match="group winners"):
        group_winners(rs, "psm", keys=[1])


def test_group_winners_refuse_the_whole_table(open_fixture):
    """The whole-table winners do not fit in memory at scale: keys are required and bounded."""
    rs = open_fixture("experiment")
    with pytest.raises(TypeError, match="keys"):
        group_winners(rs, "peptide")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="group_winner_sql"):
        group_winners(rs, "peptide", keys=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=f"at most {MAX_WINNER_KEYS:,} keys"):
        group_winners(rs, "peptide", keys=range(MAX_WINNER_KEYS + 1))
    with pytest.raises(ValueError, match="single string"):
        group_winners(rs, "protein_group", keys="sp|P12345")
    # At the bound the helper still answers; keys without a group give no row.
    at_bound = group_winners(rs, "peptide", keys=range(10**9, 10**9 + MAX_WINNER_KEYS))
    assert at_bound.empty


# --------------------------------------------------------------------------- real data


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start


EXP_TOTALS = {"psm": 531720, "precursor": 127386, "peptide": 115016, "protein_group": 12335}
EXP_PER_RUN = [88849, 87331, 88624, 89205, 88911, 88781]


@pytest.mark.real_data
def test_real_experiment(real_experiment):
    rs = open_results(real_experiment)
    counts, cold = _timed(unit_counts, rs, 0.01)
    _, warm = _timed(unit_counts, rs, 0.01)
    at5, other = _timed(unit_counts, rs, 0.05)
    checks = engine_check(rs)
    per_run, per_run_time = _timed(per_run_counts, rs, 0.01)
    curve, curve_time = _timed(id_curve, rs, "peptide")
    run_curve, run_curve_time = _timed(id_curve, rs, "run_psm", per_run=True)
    hist, hist_time = _timed(score_histogram, rs)
    print(f"\n{real_experiment}: {rs.scored.rows:,} scored rows")
    for c in counts:
        print("  ", c.label)
    print("   per run:", list(per_run["target_psms"]))
    print(
        f"   unit_counts cold {cold:.3f} s, memoised {warm:.4f} s, other threshold {other:.3f} s; "
        f"per_run_counts {per_run_time:.3f} s; id_curve {curve_time:.3f} s; per-run id_curve "
        f"{run_curve_time:.3f} s; score_histogram {hist_time:.3f} s"
    )
    assert all(c.equal for c in checks), checks
    assert curve["count"].iloc[-1] == {c.unit: c.n_target for c in at5}["peptide"]
    assert int(hist["target"].sum() + hist["decoy"].sum()) == rs.scored.parquet().num_rows
    assert len(run_curve) == 200 * len(rs.runs)
    if {c.unit: c.n_target for c in counts} == EXP_TOTALS:
        assert list(per_run["target_psms"]) == EXP_PER_RUN
    if os.environ.get("MUMDIA_VIEWER_STRICT_TIMING"):
        assert cold < 0.3


@pytest.mark.real_data
def test_real_single(real_single):
    rs = open_results(real_single)
    counts, cold = _timed(unit_counts, rs, 0.01)
    checks = engine_check(rs)
    per_run, per_run_time = _timed(per_run_counts, rs, 0.01)
    print(f"\n{real_single}: {rs.scored.rows:,} scored rows")
    for c in counts:
        print("  ", c.label)
    print(f"   unit_counts {cold:.3f} s; per_run_counts {per_run_time:.3f} s")
    assert all(c.equal for c in checks), checks
    assert int(per_run["target_psms"].iloc[0]) == counts[0].n_target  # single run
