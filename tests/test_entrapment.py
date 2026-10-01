"""Entrapment classification, settings and the entrapment FDP.

The entrapment fixture (``tests/fixtures/entrapment/entrap_mode``) is a MuMDIA 0.5.0
rescore in entrapment mode (``entrapment_native``) without a manifest. The tests give
it a minimal ``manifest.json`` in a temporary copy, with the configuration the rescore
used (``tests/fixtures/tools/entrapment/config.entrap_mode.json``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.counts import (
    engine_check,
    engine_stats,
    group_winner_sql,
    group_winners,
    per_run_counts,
    score_histogram,
    unit_counts,
)
from mumdia_viewer.data.duck import sql_path
from mumdia_viewer.data.entrapment import (
    DEFAULT_RATIO,
    EntrapmentSettings,
    class_breakdown,
    class_sql,
    count_classes,
    entrapment_expr,
    entrapment_expr_positional,
    entrapment_fdp,
    fdp_value,
    markers_present,
    markers_recorded,
    settings_for,
    spike_in_condition,
)
from mumdia_viewer.data.hashing import blake3_file
from mumdia_viewer.data.rescore import rescore_info
from mumdia_viewer.data.units import COUNT_UNITS, UNITS, bind_params

TOOLS = Path(__file__).parent / "fixtures" / "tools" / "entrapment"
CONTAMINANTS = ("KRT", "K1C", "K2C", "ALBU", "TRYP")
STATS_KEYS = {
    "psm": "target_psms_at_1pct",
    "precursor": "target_precursors_at_1pct",
    "peptide": "target_peptides_at_1pct",
    "protein_group": "target_protein_groups_at_1pct",
}


def _manifest_dir(src: Path, dst: Path, *, config: dict | None) -> Path:
    """Copy psms_scored.parquet and its report into ``dst`` and add a minimal manifest."""
    dst.mkdir(parents=True)
    for name in ("psms_scored.parquet", "psms_scored.parquet.report.json"):
        shutil.copy2(src / name, dst / name)
    report = json.loads((dst / "psms_scored.parquet.report.json").read_text(encoding="utf-8"))
    manifest = {
        "mumdia_version": "0.5.0",
        "cli_args": ["mumdia", "rescore", "--out-dir", str(dst)],
        "config_json": json.dumps(config) if config is not None else None,
        "model_identities": {"rescorer": report["model_identity"]},
        "artifacts": {
            "psms_scored": {
                "logical_name": "psms_scored",
                "path": str(dst / "psms_scored.parquet"),
                "format": "parquet",
                "schema_name": "psms_scored",
                "schema_version": 4,
                "rows": report["rows"],
                "content_hash": report["content_hash"],
                "producing_stage": "rescore",
            }
        },
    }
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dst


@pytest.fixture(scope="module")
def entrap(fixture_dir, tmp_path_factory):
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    root = _manifest_dir(
        fixture_dir("entrapment"), tmp_path_factory.mktemp("e") / "run", config=config
    )
    return open_results(root)


@pytest.fixture(scope="module")
def entrap_table(entrap) -> pa.Table:
    return pq.read_table(entrap.scored.path)


def _classes(table: pa.Table, contaminants=CONTAMINANTS):
    """The engine's classes with pyarrow only: (is_decoy, is_spike_in, is_real)."""
    protein = table["protein"]
    target = pc.equal(table["label"], "target")
    spike = pc.and_(target, pc.match_substring(protein, "ENTRAP_"))
    spike = pc.and_(spike, pc.invert(pc.match_substring(protein, "REAL_")))
    for token in contaminants:
        spike = pc.and_(spike, pc.invert(pc.match_substring(protein, token)))
    decoy = pc.equal(table["label"], "decoy")
    real = pc.and_(target, pc.invert(spike))
    return decoy, spike, real


def _n(table: pa.Table, mask, unit: str) -> int:
    u = UNITS[unit]
    part = table.filter(mask)
    for column in u.exclude_empty:
        part = part.filter(pc.not_equal(part[column], ""))
    if not u.distinct:
        return part.num_rows
    return part.group_by(list(u.distinct)).aggregate([]).num_rows


# --------------------------------------------------------------------------- settings


def test_mode_and_settings_from_the_config(entrap):
    info = rescore_info(entrap)
    assert info.mode == "entrapment" and info.classifier == "entrapment_native"
    settings = settings_for(entrap)
    assert settings.marker == "ENTRAP_" and settings.exclude == "REAL_"
    assert settings.contaminants == CONTAMINANTS
    assert settings.ratio == 0.560632
    assert settings.source.startswith("config_json")
    assert markers_recorded(entrap)
    assert markers_present(entrap, settings)


def test_settings_fall_back_to_the_report_then_to_defaults(fixture_dir, tmp_path, open_fixture):
    rs = open_results(_manifest_dir(fixture_dir("entrapment"), tmp_path / "run", config=None))
    settings = settings_for(rs)
    assert settings.ratio == 0.560632 and settings.contaminants == ()
    assert "stats.entrapment_ratio" in settings.source
    assert not markers_recorded(rs)
    defaults = settings_for(open_fixture("single"))  # marker null in the config
    assert defaults.marker == "ENTRAP_" and defaults.exclude == "REAL_"
    assert defaults.ratio == DEFAULT_RATIO and defaults.source.startswith("viewer defaults")
    with pytest.raises(ValueError, match="ratio"):
        EntrapmentSettings(ratio=0.0)
    with pytest.raises(ValueError, match="marker"):
        EntrapmentSettings(marker="")


def test_settings_made_directly_name_their_source():
    assert EntrapmentSettings().source == "viewer defaults"
    assert EntrapmentSettings(ratio=1.0).source == "set by the caller"
    # A copy with a changed value does not keep the 'viewer defaults' source.
    assert dataclasses.replace(EntrapmentSettings(), ratio=1.0).source == "set by the caller"
    assert EntrapmentSettings(source="user input").source == "user input"
    assert EntrapmentSettings(contaminants="KRT").contaminants == ("KRT",)
    assert EntrapmentSettings(exclude="").exclude is None
    settings = EntrapmentSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.ratio = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- classes


def test_classification_matches_the_engine_counts(entrap, entrap_table):
    breakdown = class_breakdown(entrap)
    # out2_mh0 competed table: 17,590 spike-in, 19,299 real and 27,666 decoy rows.
    assert breakdown["spike_in_rows"] == 17590
    assert breakdown["real_rows"] == 19299
    assert breakdown["decoy_rows"] == 27666
    decoy, spike, real = _classes(entrap_table)
    assert pc.sum(spike).as_py() == 17590 and pc.sum(real).as_py() == 19299
    # The contaminant tokens only match K1C inside human spike-in accessions here.
    assert breakdown["carved_out_rows"] == 4
    assert all("K1C" in p for p in breakdown["carved_out_proteins"])
    # Decoy proteins keep the marker text; the label is tested first.
    decoys_with_marker = pc.and_(decoy, pc.match_substring(entrap_table["protein"], "ENTRAP_"))
    assert pc.sum(decoys_with_marker).as_py() > 0


def _spike_ins_by_sql(rs, settings: EntrapmentSettings) -> int:
    cte, params = class_sql(settings, columns="label, protein")
    assert settings.marker not in cte  # every string is bound, never spliced
    sql = f"WITH {cte} SELECT count(*) FILTER (WHERE is_ent) FROM cls"
    params["path"] = str(rs.scored.path).replace("\\", "/")
    return int(rs.duck.scalar(sql, bind_params(sql, params)))


def test_class_sql_composes_with_bound_parameters(entrap, entrap_table):
    assert _spike_ins_by_sql(entrap, settings_for(entrap)) == 17590
    # Without the contaminant tokens the 4 accidental K1C matches count as spike-ins.
    assert _spike_ins_by_sql(entrap, EntrapmentSettings(contaminants=())) == 17594
    # Without the exclusion every target with the marker is a spike-in.
    with_marker = pc.and_(
        pc.equal(entrap_table["label"], "target"),
        pc.match_substring(entrap_table["protein"], "ENTRAP_"),
    )
    no_exclude = EntrapmentSettings(exclude=None)
    assert _spike_ins_by_sql(entrap, no_exclude) == pc.sum(with_marker).as_py()


# --------------------------------------------------------------------------- FDP


def test_sql_fdp_with_the_bound_ratio_is_bitwise_the_python_fdp(entrap):
    settings = settings_for(entrap)
    cte, params = class_sql(settings, columns="label, protein, base_peptide_id, peptide_q_value")
    params["path"] = str(entrap.scored.path).replace("\\", "/")
    sql = (
        f"WITH {cte}, c AS (SELECT any_value(ent_ratio) AS r, "
        "count(DISTINCT base_peptide_id) FILTER (WHERE is_ent AND peptide_q_value <= 0.01) AS e, "
        "count(DISTINCT base_peptide_id) FILTER (WHERE NOT is_decoy AND NOT is_ent "
        "AND peptide_q_value <= 0.01) AS n FROM cls) "
        "SELECT (r * CAST(e AS DOUBLE) + 1.0) / CAST(greatest(n, 1) AS DOUBLE), e, n FROM c"
    )
    fdp, e, n = entrap.duck.execute(sql, bind_params(sql, params)).fetchone()
    assert (e, n) == (138, 7846)
    assert fdp == fdp_value(settings.ratio, 138, 7846)


def test_fdp_arithmetic():
    # out2_mh0 peptide unit at 0.01 (facts G5): (0.560632 * 138 + 1) / 7,879.
    assert fdp_value(0.560632, 138, 7879) == 0.009946340398527731
    assert fdp_value(0.560632, 131, 7730) == 0.009630374126778784
    assert fdp_value(0.5, 3, 0) == 2.5  # R = 0: the denominator is 1, and nothing is capped


@pytest.mark.parametrize("t", [0.01, 0.05])
def test_entrapment_mode_fdp_restates_the_q_column(entrap, t):
    rows = entrapment_fdp(entrap, t)
    assert [r.unit for r in rows] == [*COUNT_UNITS, "run_psm"]
    for r in rows:
        assert r.mode == "entrapment"
        assert r.largest_accepted_q is not None
        assert r.fdp == r.largest_accepted_q, (r.unit, r.fdp, r.largest_accepted_q)
        assert r.matches_q is True
        assert "engine's own estimate" in r.label and "not an independent check" in r.label
        assert f"largest accepted {r.q_column}" in r.label
        assert "does not restate" not in r.label
        assert r.ratio == 0.560632 and r.threshold == t
        assert r.settings_source.startswith("config_json")
        assert "settings: config_json rescore.entrapment_marker" in r.label
        assert "'ENTRAP_', not 'REAL_', and none of KRT, K1C, K2C, ALBU, TRYP" in r.label
    assert rows[-1].run == "run" and rows[-1].source == 0


def test_fdp_counts_are_the_engine_statistics(entrap, entrap_table):
    stats = engine_stats(entrap)
    rows = {r.unit: r for r in entrapment_fdp(entrap, 0.01)}
    for unit, key in STATS_KEYS.items():
        assert rows[unit].real == stats[key], unit
    assert rows["peptide"].spike_ins == stats["entrapment_peptides_at_1pct"] == 138
    assert rows["peptide"].real == 7846
    _, spike, real = _classes(entrap_table)
    for unit in COUNT_UNITS:
        accepted = pc.less_equal(entrap_table[UNITS[unit].q_column], 0.01)
        assert rows[unit].spike_ins == _n(entrap_table, pc.and_(spike, accepted), unit)
        assert rows[unit].real == _n(entrap_table, pc.and_(real, accepted), unit)
    assert rows["peptide"].fdp == fdp_value(0.560632, 138, 7846)


def test_counts_in_entrapment_mode_are_real_targets(entrap):
    checks = engine_check(entrap)
    assert [c.unit for c in checks] == [*COUNT_UNITS, "entrapment_peptides"]
    assert all(c.equal for c in checks), checks
    counts = {c.unit: c for c in unit_counts(entrap, 0.01)}
    assert counts["peptide"].n_target == 7846 and counts["peptide"].n_spike_in == 138
    for c in counts.values():
        assert "real targets only" in c.label and "spike-ins excluded" in c.label
        assert "ENTRAP_" in c.sql  # the displayed SQL writes the rule out
    # Grouped q columns are 1.0 on every decoy in entrapment mode.
    assert counts["peptide"].n_decoy == 0 and counts["psm"].n_decoy > 0
    hist = score_histogram(entrap, bins=20)
    assert list(hist.columns) == ["bin_lo", "bin_hi", "target", "decoy", "spike_in"]
    assert int(hist[["target", "decoy", "spike_in"]].to_numpy().sum()) == entrap.scored.rows


def test_thresholds_are_validated(entrap):
    with pytest.raises(ValueError, match="cannot be told"):
        entrapment_fdp(entrap, 1.0)


# --------------------------------------------------------------------------- other settings


def test_fdp_with_another_ratio_is_a_viewer_computation(entrap):
    """User-set r in entrapment mode: the FDP does not restate the q column, and says so."""
    own = {r.unit: r for r in entrapment_fdp(entrap, 0.01)}
    rows = entrapment_fdp(entrap, 0.01, settings=EntrapmentSettings(ratio=1.0))
    assert [r.unit for r in rows] == [*COUNT_UNITS, "run_psm"]
    for r in rows:
        ref = own[r.unit]
        assert (r.spike_ins, r.real) == (ref.spike_ins, ref.real)  # the same classes
        assert r.fdp == fdp_value(1.0, r.spike_ins, r.real)
        assert r.fdp != r.largest_accepted_q and r.matches_q is False
        assert "engine's own estimate" not in r.label
        assert "viewer-computed with these settings, not the engine's estimate" in r.label
        assert f"does not restate {r.q_column}" in r.label
        assert "because these settings differ from the run's recorded settings" in r.label
        assert "(1 x " in r.label and "r = 1;" in r.label
        assert r.settings_source == "set by the caller"
    psm = rows[0]
    assert f"largest accepted q_value = {psm.largest_accepted_q:.6g}" in psm.label
    # A copy of the run's settings with another r keeps the run's source text; the label
    # does not repeat it as if the settings were the run's.
    copied = dataclasses.replace(settings_for(entrap), ratio=1.0)
    first = entrapment_fdp(entrap, 0.01, settings=copied)[0]
    assert first.matches_q is False
    assert first.settings_source.startswith("set by the caller; they differ")
    # The run's own values given again are the run's settings, whatever their source says.
    again = dataclasses.replace(settings_for(entrap), source="typed in by the user")
    assert all(r.matches_q for r in entrapment_fdp(entrap, 0.01, settings=again))


def test_fdp_without_recorded_markers(fixture_dir, tmp_path):
    """No config: r from the report, the viewer's default markers, no contaminant tokens.

    At 0.01 the viewer's counts equal the engine's statistics. At 0.09 an accepted K1C
    row (a human spike-in accession that the engine's contaminant tokens carve out) is
    a spike-in for the viewer, so the PSM and precursor FDPs no longer restate q.
    """
    rs = open_results(_manifest_dir(fixture_dir("entrapment"), tmp_path / "run", config=None))
    assert all(c.equal for c in engine_check(rs))
    rows = entrapment_fdp(rs, 0.09)
    mismatched = {r.unit for r in rows if r.matches_q is False}
    assert {"psm", "precursor", "run_psm"} <= mismatched
    for r in rows:
        assert r.settings_source.startswith("r from the rescore report")
        if r.matches_q is False:
            assert "engine's own estimate" not in r.label
            assert f"does not restate {r.q_column}" in r.label
            assert "the run records no marker strings" in r.label
            assert "classify some accepted rows differently" in r.label
        else:
            assert "engine's own estimate" in r.label
    psm = {r.unit: r for r in rows}["psm"]
    assert (psm.spike_ins, psm.real) == (1706, 10636)
    # With the engine's contaminant tokens the FDP restates q again, and the label follows
    # the bitwise check, not the source of the settings.
    engine_rule = EntrapmentSettings(contaminants=CONTAMINANTS, ratio=0.560632)
    fixed = entrapment_fdp(rs, 0.09, settings=engine_rule)
    assert all(r.matches_q for r in fixed)
    assert all("engine's own estimate" in r.label for r in fixed)
    assert fixed[0].settings_source == "set by the caller"
    # The counts say that the real targets follow the viewer's rule.
    for c in unit_counts(rs, 0.09):
        assert "spike-ins excluded by the viewer's rule" in c.label
        assert "the run records no marker strings" in c.label
        assert "not a rule the run recorded" in c.note
        assert "as the engine's report statistics exclude them" not in c.note
    labels = per_run_counts(rs, 0.09).attrs["labels"]
    assert "viewer's spike-in rule" in labels["target_psms"]
    assert "viewer's spike-in rule" in labels["peptides"]


def test_counts_with_recorded_markers_name_the_run_rule(entrap):
    for c in unit_counts(entrap, 0.09):
        assert c.label.endswith("; real targets only, spike-ins excluded (entrapment mode)")
        assert "run's recorded marker strings" in c.note
    assert "viewer's spike-in rule" not in per_run_counts(entrap, 0.09).attrs["labels"]["peptides"]


def test_nothing_accepted_is_not_called_an_estimate(entrap):
    # 1e-9 is below every q value of the table: no row passes.
    rows = entrapment_fdp(entrap, 1e-9)
    for r in rows:
        assert (r.spike_ins, r.real) == (0, 0) and r.fdp == 1.0
        assert r.largest_accepted_q is None and r.matches_q is None
        assert "engine's own estimate" not in r.label
        assert "no spike-in and no real target passes this cut" in r.label
        assert "it estimates nothing" in r.label


# --------------------------------------------------------------------------- decoy mode


def _with_markers(src: Path, dst: Path) -> Path:
    """A copy of a decoy-mode single run whose proteins carry entrapment markers.

    Targets with base_peptide_id divisible by 3 become spike-ins; the other targets
    become real. Decoys keep DECOY_ in front of the marked string, as in real data. The
    manifest and report hashes are updated to the rewritten file.
    """
    shutil.copytree(src, dst)
    path = dst / "psms_scored.parquet"
    table = pq.read_table(path)
    rows = table.to_pydict()
    marked = []
    for label, protein, bpid in zip(
        rows["label"], rows["protein"], rows["base_peptide_id"], strict=True
    ):
        bare = protein.removeprefix("DECOY_").removeprefix("sp|")
        mark = "ENTRAP_" if bpid % 3 == 0 else "REAL_"
        text = f"sp|{mark}{bare}"
        marked.append(f"DECOY_{text}" if label == "decoy" else text)
    for column in ("protein", "protein_group"):
        index = table.schema.get_field_index(column)
        table = table.set_column(index, table.schema.field(index), pa.array(marked))
    pq.write_table(table, path)
    digest = blake3_file(path)
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["psms_scored"]["content_hash"] = digest
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report_path = dst / "psms_scored.parquet.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["content_hash"] = digest
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return dst


def test_decoy_mode_fdp_is_an_independent_check(fixture_dir, tmp_path):
    rs = open_results(_with_markers(fixture_dir("single"), tmp_path / "marked"))
    assert rescore_info(rs).mode == "target_decoy"
    assert markers_present(rs)
    table = pq.read_table(rs.scored.path)
    _, spike, real = _classes(table, contaminants=())
    rows = {r.unit: r for r in entrapment_fdp(rs, 0.01)}
    for unit in COUNT_UNITS:
        r = rows[unit]
        accepted = pc.less_equal(table[UNITS[unit].q_column], 0.01)
        e = _n(table, pc.and_(spike, accepted), unit)
        n_real = _n(table, pc.and_(real, accepted), unit)
        assert (r.spike_ins, r.real) == (e, n_real), unit
        assert r.fdp == fdp_value(DEFAULT_RATIO, e, n_real)
        assert r.mode == "target_decoy" and r.matches_q is None
        if r.largest_accepted_q is None:
            # Nothing passes (the protein-group floor here is 1/16): not a check at all.
            assert (e, n_real) == (0, 0) and "no real target passes this cut" in r.label
        else:
            assert "independent check" in r.label
    assert rows["psm"].spike_ins > 0 and rows["psm"].fdp != rows["psm"].largest_accepted_q
    # The identification counts keep the spike-ins in target-decoy mode, as the engine does.
    counts = {c.unit: c for c in unit_counts(rs, 0.01)}
    assert counts["psm"].n_target == 273
    assert counts["psm"].n_spike_in == rows["psm"].spike_ins
    assert all(c.equal for c in engine_check(rs))


def test_no_markers_means_no_fdp(open_fixture):
    rs = open_fixture("single")
    assert not markers_present(rs)
    assert entrapment_fdp(rs, 0.01) == []
    assert class_breakdown(rs)["spike_in_rows"] == 0
    assert spike_in_condition(rs, "w") == ("false", [])


def test_markers_present_leaves_no_reader_open(fixture_dir, tmp_path):
    """Review constraints #7: the marker test is a full aggregate, not ``LIMIT 1``.

    A scan that stops early leaves the parquet reader open on the thread's DuckDB cursor
    until its next statement, and on Windows the engine then cannot replace the file by
    rename (PermissionError). The engine's replace must succeed right after the call.
    """
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    root = _manifest_dir(fixture_dir("entrapment"), tmp_path / "run", config=config)
    rs = open_results(root)
    assert markers_present(rs)
    target = root / "psms_scored.parquet"
    spare = tmp_path / "psms_scored.replacement.parquet"
    shutil.copy2(target, spare)
    os.replace(spare, target)
    assert not markers_present(rs, EntrapmentSettings(marker="NO_SUCH_MARKER_"))
    shutil.copy2(target, spare)
    os.replace(spare, target)


@pytest.mark.parametrize("recorded", [True, False])
def test_every_form_of_the_spike_in_test_selects_the_same_rows(
    fixture_dir, tmp_path, entrap_table, recorded
):
    """The named, aliased and positional forms flag the rows of count_classes (pyarrow)."""
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    rs = open_results(
        _manifest_dir(
            fixture_dir("entrapment"), tmp_path / "run", config=config if recorded else None
        )
    )
    cls = count_classes(rs)
    assert cls.spike_present and cls.markers_recorded == recorded
    _, spike, _ = _classes(entrap_table, contaminants=CONTAMINANTS if recorded else ())
    expected = set(pc.indices_nonzero(spike).to_pylist())
    path = sql_path(rs.scored.path)

    def rows(where: str, params) -> set[int]:
        sql = f"SELECT file_row_number FROM read_parquet(?, file_row_number = true) w WHERE {where}"
        if isinstance(params, Mapping):
            sql = sql.replace("read_parquet(?", "read_parquet($path")
            return {r[0] for r in rs.duck.rows(sql, bind_params(sql, {**params, "path": path}))}
        return {r[0] for r in rs.duck.rows(sql, [path, *params])}

    assert rows(cls.spike, cls.params) == expected
    named, named_params = entrapment_expr(cls.settings, alias="w")
    assert "w.protein" in named and rows(named, named_params) == expected
    text, params = entrapment_expr_positional(cls.settings, alias="w")
    assert "$" not in text and rows(text, params) == expected
    assert spike_in_condition(rs, "w") == (text, params)


# --------------------------------------------------------------------------- winners

GROUPED = {"peptide": "peptide_q_value", "precursor": "precursor_q", "protein_group": "pg_q_value"}


def _all_winners(rs, unit: str):
    sql, params = group_winner_sql(rs, unit)
    return rs.duck.execute(sql, bind_params(sql, params)).df()


def _row_ids(df) -> set:
    return set(zip(df["source"].tolist(), df["candidate_id"].tolist(), strict=True))


def test_group_winners_in_entrapment_mode(entrap, entrap_table):
    """Decoys do not compete: the winners are the non-decoy rows with a grouped q below 1."""
    non_decoy = entrap_table.filter(pc.not_equal(entrap_table["label"], "decoy"))
    for unit, column in GROUPED.items():
        winners = _all_winners(entrap, unit)
        assert (winners["label"] != "decoy").all(), unit
        below = non_decoy.filter(pc.less(non_decoy[column], 1.0))
        expected = set(
            zip(below["source"].to_pylist(), below["candidate_id"].to_pylist(), strict=True)
        )
        assert _row_ids(winners) == expected, unit
        # The DataFrame helper on a subset of the groups gives the same rows.
        subset = winners.iloc[:500]
        if unit == "peptide":
            keys = [int(v) for v in subset["base_peptide_id"]]
        elif unit == "precursor":
            keys = [
                (p, int(c)) for p, c in zip(subset["peptidoform"], subset["charge"], strict=True)
            ]
        else:
            keys = list(subset["protein"])
        keyed = group_winners(entrap, unit, keys=keys)
        assert _row_ids(keyed) == _row_ids(subset), unit
        assert "decoys do not compete" in keyed.attrs["note"]
    # The decoy of a real target shares its base_peptide_id; a group of decoys only has
    # no winner.
    sql, _ = group_winner_sql(entrap, "peptide")
    assert "label <> 'decoy'" in sql and "is_ent DESC" in sql


def test_entrapment_winner_rule_prefers_the_spike_in_on_a_tie(entrap_table, fixture_dir, tmp_path):
    """A spike-in wins an exact score tie in its group, also when it is later in the file.

    The data have no mixed group (every key holds one class), so a copy is edited: a
    spike-in row after a real target's winning row takes that row's peptide and
    precursor keys and its score.
    """
    _, spike, real = _classes(entrap_table)
    winner_mask = pc.and_(real, pc.less(entrap_table["peptide_q_value"], 1.0))
    w = int(pc.indices_nonzero(winner_mask)[0].as_py())
    s = next(i.as_py() for i in pc.indices_nonzero(spike) if i.as_py() > w)
    columns = {name: entrap_table[name].to_pylist() for name in entrap_table.column_names}
    for name in ("base_peptide_id", "peptidoform", "charge", "score"):
        columns[name][s] = columns[name][w]
    table = pa.table(
        {
            name: pa.array(values, entrap_table.schema.field(name).type)
            for name, values in columns.items()
        },
        schema=entrap_table.schema,
    )
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(table, src / "psms_scored.parquet")
    report_path = fixture_dir("entrapment") / "psms_scored.parquet.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["content_hash"] = blake3_file(src / "psms_scored.parquet")
    (src / "psms_scored.parquet.report.json").write_text(json.dumps(report), encoding="utf-8")
    config = json.loads((TOOLS / "config.entrap_mode.json").read_text(encoding="utf-8"))
    rs = open_results(_manifest_dir(src, tmp_path / "run", config=config))
    spike_id = columns["candidate_id"][s]
    peptide = group_winners(rs, "peptide", keys=[columns["base_peptide_id"][w]])
    assert list(peptide["candidate_id"]) == [spike_id]
    precursor = group_winners(
        rs, "precursor", keys=[(columns["peptidoform"][w], columns["charge"][w])]
    )
    assert list(precursor["candidate_id"]) == [spike_id]


# --------------------------------------------------------------------------- memo


def test_results_are_frozen_or_copied(entrap):
    """A caller's change to a result never reaches the memoised result."""
    row = entrapment_fdp(entrap, 0.01)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.fdp = -1.0  # type: ignore[misc]
    breakdown = class_breakdown(entrap)
    breakdown["carved_out_proteins"].append("CALLER EDIT")
    breakdown["spike_in_rows"] = -1
    again = class_breakdown(entrap)
    assert "CALLER EDIT" not in again["carved_out_proteins"]
    assert again["spike_in_rows"] == 17590
    classes = count_classes(entrap)
    with pytest.raises(TypeError):
        classes.params["ent_marker"] = "X"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        classes.settings.marker = "X"  # type: ignore[misc]
