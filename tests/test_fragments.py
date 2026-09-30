"""Tests of fragment names, the extraction tolerance and the mirror-plot matcher.

The matcher is checked against scalar pure-Python ports of the engine predicates
(``within_ppm``, ``ppm_bounds`` with float32 bounds, ``MassOffset::factor_at``) and
against the engine's own output: the trace value of each fragment at the apex scan is
the most intense peak the engine matched there.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.fragments import (
    MATCH_LABEL,
    FragmentName,
    Tolerance,
    extraction_tolerance,
    is_ms1_row,
    match_fragments,
    parse_fragment_name,
    ppm_bounds_match,
    within_ppm,
)
from mumdia_viewer.data.spectra import ScanTable

# --------------------------------------------------------------------------- oracles


def _f32(x: float) -> float:
    """Round a float to float32 without numpy (C cast, round to nearest even)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _within_py(a: float, b: float, tol: float) -> bool:
    """constants.rs ``within_ppm``, scalar."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    lo, hi = min(a, b), max(a, b)
    return hi - lo <= tol * 1e-6 * lo


def _bucketed_py(q: float, frag: float, tol: float) -> bool:
    """index.rs ``page_search`` entry test: ``ppm_bounds`` cast to float32."""
    d = q * tol * 1e-6
    return _f32(q - d) <= _f32(frag) <= _f32(q + d)


def _factor_py(gx: list[float], gy: list[float], offset: float, mz: float) -> float:
    """extract.rs ``MassOffset::factor_at``, scalar."""
    if len(gx) >= 2:
        i = bisect_left(gx, mz)
        if i < len(gx) and gx[i] == mz:
            ppm = gy[i]
        elif i == 0:
            ppm = gy[0]
        elif i >= len(gx):
            ppm = gy[-1]
        else:
            x0, x1, y0, y1 = gx[i - 1], gx[i], gy[i - 1], gy[i]
            ppm = y0 + (y1 - y0) * (mz - x0) / (x1 - x0)
    else:
        ppm = offset
    return 1.0 + ppm * 1e-6


def _next32(x: float, direction: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(direction)))


def _edge(t: float, tol: float, up: bool, accept) -> tuple[float, float]:
    """The last accepted float32 peak on one side of ``t`` and the first rejected one."""
    delta = tol * 1e-6
    out = math.inf if up else -math.inf
    p = _f32(t * (1 + delta) if up else t / (1 + delta))
    while not accept(p):
        p = _next32(p, -out)
    while accept(_next32(p, out)):
        p = _next32(p, out)
    return p, _next32(p, out)


# --------------------------------------------------------------------------- names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("b3", FragmentName("b", 3, 1)),
        ("y12", FragmentName("y", 12, 1)),
        ("y7^2", FragmentName("y", 7, 2)),
        ("b26^2", FragmentName("b", 26, 2)),
        ("ms1_mono", None),
        ("ms1_iso2", None),
        ("y3-H2O", None),
        ("Y3", None),
        ("a2", None),
        ("b0", None),
        ("y5^0", None),
        ("y5^", None),
        ("y^2", None),
        (" y5", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_fragment_name(name, expected):
    got = parse_fragment_name(name)
    assert got == expected
    if got is not None:
        assert got.label == name


def test_is_ms1_row():
    assert is_ms1_row("ms1_mono") and is_ms1_row("ms1_iso1") and is_ms1_row("ms1_iso2")
    assert not is_ms1_row("y3") and not is_ms1_row("") and not is_ms1_row(None)


def test_fragment_names_agree_with_the_library_columns(open_fixture):
    rs = open_fixture("single")
    lib = rs.artifact("fragment_library_fragments")
    t = pq.read_table(lib.path, columns=["name", "ion_type", "ordinal", "frag_charge"])
    names = t["name"].to_pylist()
    for name, ion, ordinal, charge in zip(
        names,
        t["ion_type"].to_pylist(),
        t["ordinal"].to_pylist(),
        t["frag_charge"].to_pylist(),
        strict=True,
    ):
        assert parse_fragment_name(name) == FragmentName(ion, ordinal, charge)
    chrom = pq.read_table(rs.runs[0].artifact("chromatograms").path, columns=["frag_name"])
    for name in set(chrom["frag_name"].to_pylist()):
        assert is_ms1_row(name) != (parse_fragment_name(name) is not None), name


# --------------------------------------------------------------------------- tolerance


def _params(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["params"]


def _edited_copy(tmp_path: Path, source: Path, edit=None) -> Path:
    """A copy of a fixture; ``edit(config)`` changes the recorded configuration."""
    root = tmp_path / source.name
    shutil.copytree(source, root)
    if edit is not None:
        name = "manifest.json" if (root / "manifest.json").is_file() else "experiment_manifest.json"
        manifest = root / name
        m = json.loads(manifest.read_text(encoding="utf-8"))
        cfg = json.loads(m["config_json"])
        edit(cfg)
        m["config_json"] = json.dumps(cfg)
        manifest.write_text(json.dumps(m), encoding="utf-8")
    return root


def _set_params(report: Path, **params) -> None:
    data = json.loads(report.read_text(encoding="utf-8"))
    data["params"].update(params)
    report.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("name", ["single", "topk", "experiment", "mbr"])
def test_extraction_tolerance_equals_the_report(open_fixture, name):
    rs = open_fixture(name)
    for run in rs.runs:
        tol = extraction_tolerance(rs, run)
        report = run.artifact("psms_extracted").path.with_name("psms_extracted.parquet.report.json")
        params = _params(report)
        assert tol.tol_ppm == params["effective_frag_tol_ppm"]
        assert tol.offset_ppm == params["frag_ppm_offset"]
        assert tol.config_fallback_ppm == rs.config_get("extract", "frag_tol_ppm") == 20.0
        assert tol.matcher == "fragindex"
        prefix = f"{run.name}/" if run.name else ""
        assert tol.source.startswith(f"{prefix}psms_extracted.parquet.report.json")
        assert not tol.uses_grid and not tol.from_config and not tol.assumed
        assert tol.grid_mz is not None and tol.grid_mz.size == 0  # the masscal grid is empty
        assert "fallback, not used" in tol.label
        assert extraction_tolerance(rs, run.name or 0) == tol


@pytest.mark.parametrize("name", ["grouped", "grouped_pool", "ovl_bp", "ovl_rg50"])
def test_extraction_tolerance_of_grouped_bands(open_fixture, name):
    rs = open_fixture(name)
    run = rs.runs[0]
    assert rs.config_get("groups", "calibration") == "per_group"
    for band in run.grouped.bands:
        params = _params(band.root / "chromatograms.parquet.report.json")
        for key in (band.index, band.name, band):
            tol = extraction_tolerance(rs, run, band=key)
            assert tol.tol_ppm == params["effective_frag_tol_ppm"]
            assert tol.offset_ppm == params["frag_ppm_offset"]
            assert tol.source.startswith(f"groups/{band.name}/chromatograms.parquet.report.json")
            # per_group calibration: the band's own mass calibration file
            assert tol.grid_source.startswith(f"groups/{band.name}/seed_psms.parquet.masscal.json")
    common = extraction_tolerance(rs, run)
    assert common.tol_ppm == 20.0 and common.offset_ppm == 0.0 and not common.assumed
    assert common.source.endswith(f"the same in all {len(run.grouped.bands)} bands")
    # the pooled run-level calibration (5 ppm) is not what the per-group bands used
    pooled = json.loads((run.root / "seed_psms.parquet.masscal.json").read_text())
    assert pooled["frag_tol_ppm"] != common.tol_ppm
    with pytest.raises(KeyError):
        extraction_tolerance(rs, run, band="g99")


def test_band_argument_needs_a_grouped_run(open_fixture):
    rs = open_fixture("single")
    with pytest.raises(ViewerError, match="not a grouped run"):
        extraction_tolerance(rs, 0, band=0)


def test_grouped_bands_that_disagree(tmp_path, fixture_dir):
    root = tmp_path / "out_grouped"
    shutil.copytree(fixture_dir("grouped"), root)
    report = root / "groups" / "g01" / "chromatograms.parquet.report.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data["params"]["effective_frag_tol_ppm"] = 7.5
    data["params"]["frag_ppm_offset"] = -1.25
    report.write_text(json.dumps(data), encoding="utf-8")
    rs = open_results(root)
    with pytest.raises(ViewerError, match="different fragment tolerances"):
        extraction_tolerance(rs, 0)
    tol = extraction_tolerance(rs, 0, band="g01")
    assert (tol.tol_ppm, tol.offset_ppm) == (7.5, -1.25)
    # the band's mass calibration now records other values: say which ones were used
    assert tol.notes and "the report values are the ones extract used" in tol.notes[0]
    assert extraction_tolerance(rs, 0, band=0).tol_ppm == 20.0


def test_global_calibration_bands_read_the_pooled_masscal(tmp_path, fixture_dir):
    root = tmp_path / "out_grouped"
    shutil.copytree(fixture_dir("grouped"), root)
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    cfg = json.loads(m["config_json"])
    cfg["groups"]["calibration"] = "global"
    m["config_json"] = json.dumps(cfg)
    manifest.write_text(json.dumps(m), encoding="utf-8")
    tol = extraction_tolerance(open_results(root), 0, band=1)
    assert tol.grid_source.startswith("seed_psms.parquet.masscal.json")


def test_extraction_tolerance_fallbacks(tmp_path, fixture_dir):
    root = tmp_path / "out"
    shutil.copytree(fixture_dir("single"), root)
    # 1. without the psms_extracted report: the chromatograms report
    (root / "psms_extracted.parquet.report.json").unlink()
    tol = extraction_tolerance(open_results(root), 0)
    assert tol.source.startswith("chromatograms.parquet.report.json")
    assert (tol.tol_ppm, tol.offset_ppm) == (5.0, 0.0)
    # 2. without extract reports: the mass calibration, here with an m/z grid
    (root / "chromatograms.parquet.report.json").unlink()
    masscal = root / "seed_psms.parquet.masscal.json"
    data = json.loads(masscal.read_text(encoding="utf-8"))
    data.update(
        frag_tol_ppm=6.5,
        frag_ppm_offset=-1.5,
        mz_cal_grid_mz=[300.0, 800.0, None, 1300.0],  # a non-number is skipped, as extract does
        mz_cal_grid_ppm=[-2.0, 0.0, 3.0],
    )
    masscal.write_text(json.dumps(data), encoding="utf-8")
    tol = extraction_tolerance(open_results(root), 0)
    assert tol.source == "seed_psms.parquet.masscal.json (frag_tol_ppm, frag_ppm_offset)"
    assert (tol.tol_ppm, tol.offset_ppm) == (6.5, -1.5)
    assert tol.uses_grid and tol.grid_mz.tolist() == [300.0, 800.0, 1300.0]
    assert tol.ppm_at(550.0) == -1.0
    assert "grid" in tol.label and "fallback, not used" in tol.label
    # 3. without any record: the values extract used are unknown. The configured
    # tolerance with offset 0 is returned as an assumption, never as the extraction's.
    masscal.unlink()
    tol = extraction_tolerance(open_results(root), 0)
    assert tol.assumed and tol.from_config
    assert tol.source.startswith("assumed: configured extract.frag_tol_ppm")
    assert (tol.tol_ppm, tol.offset_ppm, tol.grid_mz) == (20.0, 0.0, None)
    assert "not used" not in tol.label and "mass offset +0 ppm (assumed)" in tol.label
    for name in (
        "psms_extracted.parquet.report.json",
        "chromatograms.parquet.report.json",
        "seed_psms.parquet.masscal.json",
    ):
        assert name in tol.notes[0]  # the missing files are named
    assert "unknown" in tol.notes[1] and "extract then uses" not in " ".join(tol.notes)
    # 4. a config without the key: the engine default, still an assumption
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    cfg = json.loads(m["config_json"])
    del cfg["extract"]["frag_tol_ppm"]
    cfg["extract"]["matcher"] = "Bucketed"
    m["config_json"] = json.dumps(cfg)
    manifest.write_text(json.dumps(m), encoding="utf-8")
    tol = extraction_tolerance(open_results(root), 0)
    assert tol.tol_ppm == 20.0 and tol.config_fallback_ppm is None and tol.assumed
    assert tol.source.startswith("assumed: engine default 20 ppm")
    assert tol.matcher == "bucketed"
    assert any("engine default" in n for n in tol.notes)


def test_a_report_without_the_mass_calibration_file(tmp_path, fixture_dir):
    """The report gives the scalar values; the grid is unknown only under mass_cal_loess."""

    def loess(cfg):
        cfg["search_seed"]["mass_cal_loess"] = True

    for edit, warned in ((None, False), (loess, True)):
        base = tmp_path / ("loess" if warned else "plain")
        base.mkdir()
        root = _edited_copy(base, fixture_dir("single"), edit)
        (root / "seed_psms.parquet.masscal.json").unlink()
        tol = extraction_tolerance(open_results(root), 0)
        assert not tol.assumed and tol.source.startswith("psms_extracted.parquet.report.json")
        assert (tol.tol_ppm, tol.offset_ppm, tol.grid_mz) == (5.0, 0.0, None)
        assert any("mass_cal_loess" in n for n in tol.notes) is warned


def test_missing_band_records_are_an_assumption(open_fixture):
    """ovl128_pool keeps no band files: the configured tolerance is an assumption.

    Under per_group calibration each band read its own mass calibration, so the
    configured value is not known to be what the bands used.
    """
    rs = open_fixture("ovl128_pool")
    run = rs.runs[0]
    assert rs.config_get("groups", "calibration") == "per_group"
    assert not (run.root / "groups").exists()
    n = len(run.grouped.bands)
    tol = extraction_tolerance(rs, run)
    assert tol.assumed and tol.from_config
    assert (tol.tol_ppm, tol.offset_ppm) == (rs.config_get("extract", "frag_tol_ppm"), 0.0)
    assert tol.source.startswith("assumed: configured extract.frag_tol_ppm")
    assert tol.source.endswith(f"for all {n} bands")
    assert "(assumed)" in tol.label and "(fallback, not used)" not in tol.label
    notes = " ".join(tol.notes)
    assert f"none of the {n} bands has an extraction record" in notes
    assert "groups/g00/seed_psms.parquet.masscal.json" in notes and "unknown" in notes
    assert "extract then uses" not in notes  # no inference stated as a fact
    band = extraction_tolerance(rs, run, band=np.uint32(100))
    assert band.assumed and "groups/g100/chromatograms.parquet.report.json" in band.notes[0]
    # the trimmed non-pooled twin keeps every band report: a record, not an assumption
    kept = open_fixture("ovl128")
    tol = extraction_tolerance(kept, 0)
    assert not tol.assumed
    assert tol.source.endswith(f"the same in all {len(kept.runs[0].grouped.bands)} bands")


def test_bands_without_a_record_stay_out_of_the_agreement(tmp_path, fixture_dir):
    root = _edited_copy(tmp_path, fixture_dir("grouped"))
    for name in ("g00", "g02"):
        report = root / "groups" / name / "chromatograms.parquet.report.json"
        _set_params(report, effective_frag_tol_ppm=7.5, frag_ppm_offset=-1.25)
    # g01 loses its record; its assumed 20 ppm would disagree with the other bands
    (root / "groups" / "g01" / "chromatograms.parquet.report.json").unlink()
    (root / "groups" / "g01" / "seed_psms.parquet.masscal.json").unlink()
    rs = open_results(root)
    tol = extraction_tolerance(rs, 0)
    assert (tol.tol_ppm, tol.offset_ppm) == (7.5, -1.25) and not tol.assumed
    assert tol.source.endswith("the same in all 2 of 3 bands that have an extraction record")
    assert any("1 of 3 bands have no extraction record" in n and "g01" in n for n in tol.notes)
    g01 = extraction_tolerance(rs, 0, band="g01")
    assert g01.assumed and (g01.tol_ppm, g01.offset_ppm) == (20.0, 0.0)
    # the run-level mass calibration is the pooled fit, not what a per_group band read
    assert any("pooled fit" in n for n in g01.notes)
    assert not extraction_tolerance(rs, 0, band="g02").assumed


@pytest.mark.parametrize("calibration", ["per_group", "global"])
def test_a_grouped_experiment_without_band_tables(tmp_path, fixture_dir, calibration):
    """groups.window_groups > 1 but no groups/ directory: the pooled masscal stands in for
    the bands only under global calibration."""

    def grouped(cfg):
        cfg["groups"]["window_groups"] = 3
        cfg["groups"]["calibration"] = calibration

    rs = open_results(_edited_copy(tmp_path, fixture_dir("experiment"), grouped))
    run = rs.run("a")
    assert run.grouped is None
    tol = extraction_tolerance(rs, np.uint32(run.index))
    if calibration == "global":
        assert not tol.assumed
        assert tol.source == "a/seed_psms.parquet.masscal.json (frag_tol_ppm, frag_ppm_offset)"
    else:
        assert tol.assumed
        notes = " ".join(tol.notes)
        assert "a/groups/gNN/seed_psms.parquet.masscal.json" in notes
        assert "pooled fit" in notes
    with pytest.raises(KeyError, match="no band"):
        extraction_tolerance(rs, "a", band="g00")


def test_band_and_run_keys_of_any_integer_type(open_fixture):
    rs = open_fixture("grouped")
    want = extraction_tolerance(rs, 0, band="g01")
    for key in (np.uint32(1), np.int64(1), 1):
        assert extraction_tolerance(rs, np.uint32(0), band=key) == want
    for bad in (True, np.True_, 1.0):
        with pytest.raises(TypeError, match="plan index"):
            extraction_tolerance(rs, 0, band=bad)
    with pytest.raises(TypeError, match="source index"):
        extraction_tolerance(rs, True)
    exp = open_fixture("experiment")
    source = pq.read_table(exp.scored.path, columns=["source"])["source"].to_numpy()
    for value in np.unique(source):
        assert extraction_tolerance(exp, value) == extraction_tolerance(exp, int(value))


def test_peak_claim_decides_whether_a_match_is_the_trace_value(tmp_path, open_fixture, fixture_dir):
    tol = extraction_tolerance(open_fixture("single"), 0)
    assert tol.peak_claim == "none" and tol.match_is_trace_value
    assert tol.match_label == MATCH_LABEL and "peak_claim" not in tol.label

    def winner(cfg):
        cfg["extract"]["peak_claim"] = "winner_predicted_intensity"

    tol = extraction_tolerance(
        open_results(_edited_copy(tmp_path, fixture_dir("single"), winner)), 0
    )
    assert tol.peak_claim == "winner_predicted_intensity" and not tol.match_is_trace_value
    assert tol.match_label.startswith(f"{MATCH_LABEL}; predicate only")
    assert "extract.peak_claim = winner_predicted_intensity" in tol.match_label
    assert "extract.peak_claim winner_predicted_intensity" in tol.label
    for mode in ("proportional", "coelution_winner", "CoelutionDemix", "coelution_shadow"):
        assert not Tolerance(5.0, peak_claim=mode).match_is_trace_value
    assert Tolerance(5.0).match_is_trace_value
    assert Tolerance(5.0, peak_claim="None").match_is_trace_value  # the Rust Debug spelling


def test_ppm_at_scalar_and_grid():
    scalar = Tolerance(5.0, offset_ppm=-1.5)
    assert scalar.ppm_at(700.0) == -1.5
    assert scalar.ppm_at(np.array([1.0, 2.0])).tolist() == [-1.5, -1.5]
    grid = Tolerance(5.0, offset_ppm=9.0, grid_mz=[300.0, 800.0, 1300.0], grid_ppm=[-2.0, 0.0, 3.0])
    assert grid.uses_grid
    probe = [100.0, 300.0, 550.0, 800.0, 1050.0, 1300.0, 2000.0]
    assert [grid.ppm_at(x) for x in probe] == [-2.0, -2.0, -1.0, 0.0, 1.5, 3.0, 3.0]
    assert grid.ppm_at(np.array(probe)).tolist() == [-2.0, -2.0, -1.0, 0.0, 1.5, 3.0, 3.0]
    # unusable grids fall back to the scalar offset, as extract does
    assert Tolerance(5.0, offset_ppm=1.0, grid_mz=[300.0], grid_ppm=[2.0]).ppm_at(300.0) == 1.0
    uneven = Tolerance(5.0, offset_ppm=1.0, grid_mz=[300.0, 400.0], grid_ppm=[2.0])
    assert not uneven.uses_grid and uneven.ppm_at(350.0) == 1.0


def test_query_matches_the_engine_factor():
    rng = np.random.default_rng(11)
    gx = sorted(rng.uniform(200, 1800, 60).tolist())
    gy = rng.normal(-1.5, 2.0, 60).tolist()
    tol = Tolerance(8.0, offset_ppm=-1.0, grid_mz=gx, grid_ppm=gy)
    mz = np.concatenate([rng.uniform(100, 2000, 2000), np.asarray(gx)]).astype(np.float32)
    mz64 = mz.astype(np.float64)
    q = tol.query_mz(mz64)
    want = [float(x) / _factor_py(gx, gy, -1.0, float(x)) for x in mz64]
    assert q.tolist() == want
    flat = Tolerance(8.0, offset_ppm=-1.0)
    assert flat.query_mz(mz64).tolist() == [float(x) / _factor_py([], [], -1.0, x) for x in mz64]


def test_tolerance_validation_equality_and_matcher_names():
    with pytest.raises(ValueError):
        Tolerance(float("nan"))
    with pytest.raises(ValueError):
        Tolerance(-1.0)
    a = Tolerance(5.0, grid_mz=[1.0, 2.0], grid_ppm=[0.0, 1.0], source="x")
    b = Tolerance(5.0, grid_mz=[1.0, 2.0], grid_ppm=[0.0, 1.0], source="x")
    c = Tolerance(5.0, grid_mz=[1.0, 2.0], grid_ppm=[0.0, 2.0], source="x")
    assert a == b and hash(a) == hash(b) and a != c and not a.same_values(c)
    assert not a.grid_mz.flags.writeable
    assert Tolerance(5.0, matcher="Bucketed").matcher == "bucketed"
    assert Tolerance(5.0, matcher="FragIndex").matcher == "fragindex"
    with pytest.raises(ViewerError, match=r"unknown extract\.matcher"):
        match_fragments(np.array([500.0]), np.array([1.0]), [500.0], Tolerance(5.0, matcher="xyz"))


# --------------------------------------------------------------------------- predicates


def test_predicates_equal_the_scalar_engine_ports():
    rng = np.random.default_rng(5)
    t = rng.uniform(150, 2000, 3000)
    q = t * (1 + rng.uniform(-3e-5, 3e-5, 3000))
    tol = 8.4217
    assert within_ppm(t, q, tol).tolist() == [
        _within_py(a, b, tol) for a, b in zip(t, q, strict=True)
    ]
    assert ppm_bounds_match(q, t, tol).tolist() == [
        _bucketed_py(a, b, tol) for a, b in zip(q, t, strict=True)
    ]
    assert not within_ppm(np.nan, 500.0, tol) and not within_ppm(500.0, np.inf, tol)


@pytest.mark.parametrize("up", [True, False], ids=["above", "below"])
@pytest.mark.parametrize("theo", [412.2301, 1000.0, 1873.9])
def test_fragindex_edge(theo, up):
    tol = Tolerance(8.421748911071566)
    t = _f32(theo)
    inside, outside = _edge(t, tol.tol_ppm, up, lambda p: _within_py(t, p, tol.tol_ppm))
    got = match_fragments(np.array([inside, outside], np.float32), np.array([1.0, 2.0]), [t], tol)
    assert [(m.fragment, m.peak) for m in got] == [(0, 0)]
    assert got[0].obs_mz == inside and got[0].theo_mz == t
    assert got[0].ppm_raw == 1e6 * (inside - t) / t
    assert match_fragments(np.array([outside], np.float32), np.array([1.0]), [t], tol) == []


@pytest.mark.parametrize("up", [True, False], ids=["above", "below"])
def test_bucketed_edge(up):
    tol = Tolerance(8.421748911071566, matcher="bucketed")
    t = _f32(1000.0)
    inside, outside = _edge(t, tol.tol_ppm, up, lambda p: _bucketed_py(p, t, tol.tol_ppm))
    got = match_fragments(np.array([inside, outside], np.float32), np.array([1.0, 1.0]), [t], tol)
    assert [(m.fragment, m.peak) for m in got] == [(0, 0)]


def test_the_two_matchers_differ_above_the_fragment():
    # For a peak above the fragment, ppm_bounds admits (q - t) <= tol*q and within_ppm
    # only (q - t) <= tol*t. A wide tolerance makes the difference larger than float32.
    t, peak = 1000.0, np.float32(1010.05)
    frag = Tolerance(10_000.0)
    buck = Tolerance(10_000.0, matcher="bucketed")
    assert match_fragments(np.array([peak]), np.array([1.0]), [t], frag) == []
    assert len(match_fragments(np.array([peak]), np.array([1.0]), [t], buck)) == 1


def test_theoretical_mz_is_rounded_to_float32():
    tol = Tolerance(8.421748911071566)
    flips = 0
    for k in range(200):
        t64 = 1000.0 + 1.3e-5 * k  # mostly not float32-representable
        t32 = _f32(t64)
        inside, outside = _edge(
            t32, tol.tol_ppm, True, lambda p, t=t32: _within_py(t, p, tol.tol_ppm)
        )
        got = match_fragments(np.array([inside, outside], np.float32), np.ones(2), [t64], tol)
        assert [(m.peak, m.theo_mz) for m in got] == [(0, t32)]
        flips += _within_py(t64, inside, tol.tol_ppm) != _within_py(t32, inside, tol.tol_ppm)
        flips += _within_py(t64, outside, tol.tol_ppm) != _within_py(t32, outside, tol.tol_ppm)
    assert flips > 0  # the rounding decides some edge cases


def test_most_intense_peak_per_fragment():
    tol = Tolerance(5.0)
    mz = np.array([999.999, 1000.0, 1000.001, 1500.0], np.float32)
    inten = np.array([5.0, 7.0, 7.0, 3.0], np.float32)
    theo = [1500.0, 1000.0, 1000.002, 2000.0]
    got = match_fragments(mz, inten, theo, tol)
    assert [(m.fragment, m.peak, m.obs_intensity) for m in got] == [
        (0, 3, 3.0),
        (1, 1, 7.0),  # tie between peaks 1 and 2: the lower index
        (2, 1, 7.0),  # one peak may match two fragments
    ]
    assert match_fragments(np.array([], np.float32), np.array([]), theo, tol) == []
    assert match_fragments(mz, inten, [], tol) == []
    with pytest.raises(ValueError):
        match_fragments(mz, inten[:2], theo, tol)


def test_mass_offset_moves_the_query():
    t = _f32(800.0)
    obs = _f32(t * (1 - 3e-6))  # 3 ppm low
    shifted = Tolerance(1.0, offset_ppm=-3.0)
    got = match_fragments(np.array([obs], np.float32), np.array([1.0]), [t], shifted)
    assert len(got) == 1
    m = got[0]
    q = obs / (1.0 + -3.0 * 1e-6)
    assert m.query_mz == q
    assert m.ppm_raw == 1e6 * (obs - t) / t and m.ppm_raw == pytest.approx(-3.0, abs=0.1)
    assert m.ppm_corrected == 1e6 * (q - t) / t and abs(m.ppm_corrected) < 0.1
    assert match_fragments(np.array([obs], np.float32), np.array([1.0]), [t], Tolerance(1.0)) == []


# --------------------------------------------------------------------------- engine output


def _decode_v2(table) -> list[tuple[list[float], list[float]]]:
    """Axis and dense trace of every row of one v2 row group (axis inherited per candidate)."""
    cid = table["candidate_id"].to_pylist()
    axes = table["rt_axis"].to_pylist()
    trimmed = table["intensity_trimmed"].to_pylist()
    offset = table["trace_offset"].to_pylist()
    length = table["trace_len"].to_pylist()
    last: dict[int, list[float]] = {}
    out = []
    for i in range(len(cid)):
        if length[i] == 0:
            out.append(([], []))
            continue
        if axes[i]:
            last[cid[i]] = axes[i]
        trace = [0.0] * length[i]
        trace[offset[i] : offset[i] + len(trimmed[i])] = trimmed[i]
        out.append((last[cid[i]], trace))
    return out


def _extracted(run) -> list[tuple[int, float, float]]:
    e = pq.read_table(
        run.artifact("psms_extracted").path, columns=["candidate_id", "precursor_mz", "apex_rt"]
    )
    return list(
        zip(
            e["candidate_id"].to_pylist(),
            e["precursor_mz"].to_pylist(),
            e["apex_rt"].to_pylist(),
            strict=True,
        )
    )


def _check_apex_traces(rs, run, rows_of, candidates) -> int:
    """Viewer matches at the apex scan equal the engine's trace values at the apex.

    ``rows_of(cid)`` gives the candidate's chromatogram rows as (name, frag_mz, axis,
    dense trace); ``candidates`` are (candidate_id, precursor_mz, apex_rt).
    """
    st = ScanTable.for_run(rs, run)
    tol = extraction_tolerance(rs, run)
    # the equality below holds under peak_claim none, which the property states
    assert tol.match_is_trace_value and tol.match_label == MATCH_LABEL
    assert st.emit_window_grid is True
    checked = 0
    for cid, pmz, apex in candidates:
        frags = [r for r in rows_of(cid) if not is_ms1_row(r[0])]
        pick = st.apex_scan(pmz, apex)
        assert pick is not None and pick.exact
        sp = st.spectrum(pick.row)
        matches = match_fragments(sp.mz, sp.intensity, [r[1] for r in frags], tol)
        got = {m.fragment: m for m in matches}
        apex32 = np.float32(apex)
        for k, (_, _, axis, trace) in enumerate(frags):
            value = 0.0
            if axis:
                hit = np.flatnonzero(np.asarray(axis, np.float32) == apex32)
                assert hit.size == 1
                value = float(np.float32(trace[hit[0]]))
            if value > 0:
                assert k in got and got[k].obs_intensity == value, (cid, k)
            else:
                assert k not in got, (cid, k)
            checked += 1
    return checked


def _rows_by_candidate(cids, names, mzs, axes, traces) -> dict[int, list]:
    out: dict[int, list] = {}
    for c, n, m, a, t in zip(cids, names, mzs, axes, traces, strict=True):
        out.setdefault(c, []).append((n, m, a, t))
    return out


def test_mirror_matches_reproduce_the_v1_apex_traces(open_fixture, fixture_dir):
    rs = open_fixture("single")
    assert rs.config_get("extract", "peak_claim") == "none"
    v1 = pq.read_table(
        fixture_dir("chrom_v1") / "chromatograms.parquet",
        columns=["candidate_id", "frag_name", "frag_mz", "rt", "intensity"],
    )
    rows = _rows_by_candidate(
        v1["candidate_id"].to_pylist(),
        v1["frag_name"].to_pylist(),
        v1["frag_mz"].to_pylist(),
        v1["rt"].to_pylist(),
        v1["intensity"].to_pylist(),
    )
    run = rs.runs[0]
    assert _check_apex_traces(rs, run, rows.__getitem__, _extracted(run)) == 1704


def test_mirror_matches_reproduce_the_v2_traces_at_every_peak_rank(open_fixture):
    rs = open_fixture("topk")
    run = rs.runs[0]
    pf = pq.ParquetFile(run.artifact("chromatograms").path)
    assert pf.metadata.num_row_groups == 1
    t = pf.read_row_group(0)
    decoded = _decode_v2(t)
    rows = _rows_by_candidate(
        t["candidate_id"].to_pylist(),
        t["frag_name"].to_pylist(),
        t["frag_mz"].to_pylist(),
        [axis for axis, _ in decoded],
        [trace for _, trace in decoded],
    )
    candidates = _extracted(run)
    assert len(candidates) == 360  # every peak rank
    assert _check_apex_traces(rs, run, rows.__getitem__, candidates) == 2160


@pytest.mark.real_data
def test_astral_mirror_matches_and_tolerance(real_single, capsys):
    """On the Astral run: the report tolerance, and the engine's apex trace values."""
    from mumdia_viewer.data.candidate_index import CandidateIndex, read_candidate_rows

    rs = open_results(real_single)
    run = rs.runs[0]
    tol = extraction_tolerance(rs, run)
    params = _params(
        run.artifact("psms_extracted").path.with_name("psms_extracted.parquet.report.json")
    )
    assert tol.tol_ppm == params["effective_frag_tol_ppm"]
    assert tol.offset_ppm == params["frag_ppm_offset"]
    assert tol.config_fallback_ppm == 20.0 and tol.matcher == "fragindex"
    chrom = run.artifact("chromatograms")
    handle = chrom.parquet()
    index = CandidateIndex.for_artifact(chrom, rs.cache)
    cols = ["candidate_id", "frag_name", "frag_mz", "rt_axis", "intensity_trimmed"]
    cols += ["trace_offset", "trace_len"]

    def rows_of(cid: int) -> list:
        out = []
        for _, part in read_candidate_rows(handle, index, cid, cols):
            decoded = _decode_v2(part)  # a part starts at the candidate's first row
            out += [
                (n, m, a, t)
                for n, m, (a, t) in zip(
                    part["frag_name"].to_pylist(),
                    part["frag_mz"].to_pylist(),
                    decoded,
                    strict=True,
                )
            ]
        return out

    candidates = _extracted(run)
    rng = np.random.default_rng(4)
    sample = [candidates[i] for i in rng.choice(len(candidates), 300, replace=False)]
    checked = _check_apex_traces(rs, run, rows_of, sample)
    with capsys.disabled():
        print(f"\n[astral] {checked} fragment rows at the apex scan agree with the engine traces")
