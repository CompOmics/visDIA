"""The calibration data functions against independent computations on the fixtures.

The anchors are rebuilt here a second way: a plain loop over the seed rows, written from
``rt_im_train.rs`` (``fit_anchors``) with a dict, as the engine does it, and Rust's
nearest-rank percentile. Both rebuilds must give cal.json's numbers.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import calibration as cal
from mumdia_viewer.data.windows import band_offsets

UNGROUPED = ["single", "experiment", "topk", "mbr"]
GROUPED = ["grouped", "ovl_bp"]


def rust_percentile(values, p):
    """calibrate.rs percentile: sort, rank = (p * (n - 1)).round() (half away from zero)."""
    v = sorted(values)
    x = min(max(p, 0.0), 1.0) * (len(v) - 1)
    rank = math.floor(x + 0.5)  # x >= 0, so this is Rust's round
    return v[rank]


def engine_anchors(seed_path, irt_of, q_train):
    """rt_im_train.rs fit_anchors: best (score, irt, rt) per base_peptide_id, row order."""
    t = pq.read_table(
        seed_path,
        columns=["candidate_id", "base_peptide_id", "spectrum_q", "score", "observed_rt", "label"],
    ).to_pydict()
    best: dict[int, tuple[float, float, float, int]] = {}
    for i in range(len(t["candidate_id"])):
        q, s, rt = t["spectrum_q"][i], t["score"][i], t["observed_rt"][i]
        if not (math.isfinite(q) and math.isfinite(s) and math.isfinite(rt)) or q >= q_train:
            continue
        if t["label"][i] != "target":
            continue
        irt = irt_of(t["candidate_id"][i])
        if irt is None or not math.isfinite(irt):
            continue
        b = t["base_peptide_id"][i]
        e = best.get(b, (-math.inf, 0.0, 0.0, -1))
        if s > e[0]:
            best[b] = (s, irt, rt, t["candidate_id"][i])
    return {b: best[b] for b in sorted(best)}


def _scopes(rs):
    for run in rs.runs:
        for s in cal.scopes(rs, run):
            if s.mode != "global":
                yield run, s


def _irt_lookup(rs, run, s):
    lib = rs.extra.get("fragment_library_precursors") if s.band is not None else None
    lib = lib or (run.artifacts["rt_library"] if s.band is None else lib)
    irt = pq.read_table(lib.path, columns=["predicted_irt"]).column(0).to_numpy()

    def get(cid):
        row = cid + s.offset
        if s.size is not None and cid >= s.size:
            return None
        return float(irt[row]) if 0 <= row < irt.size else None

    return get


@pytest.mark.parametrize("name", UNGROUPED + GROUPED)
def test_anchors_equal_the_engine_rule_and_cal_json(open_fixture, name):
    rs = open_fixture(name)
    q_train = rs.config_get("rt_im_train", "q_train", default=0.01)
    seen = 0
    for run, s in _scopes(rs):
        seed = s.artifact("seed_psms").path
        mine = engine_anchors(seed, _irt_lookup(rs, run, s), q_train)
        a = cal.rt_anchors(rs, run, s.key or None)
        assert a.error is None, a.error
        assert list(a.frame["base_peptide_id"]) == list(mine)
        assert list(a.frame["local_id"]) == [v[3] for v in mine.values()]
        assert np.allclose(a.frame["irt"], [v[1] for v in mine.values()], equal_nan=True)
        assert list(a.frame["observed_rt"]) == [v[2] for v in mine.values()]
        record = json.loads(s.side("cal").read_text(encoding="utf-8"))
        assert a.n == record["n_train"]
        if a.n >= 2:
            windows = pq.read_table(s.artifact("run_windows").path, columns=["rt_pred_cal"])
            pred = windows.column(0).to_numpy()[a.frame["local_id"].to_numpy()]
            res = [rt - p for rt, p in zip(a.frame["observed_rt"], pred, strict=True)]
            med = rust_percentile(res, 0.5)
            assert med == pytest.approx(record["rt_residual_median_s"], abs=1e-9)
            assert rust_percentile([abs(r) for r in res], 0.5) == pytest.approx(
                record["rt_residual_abs_median_s"], abs=1e-9
            )
            assert rust_percentile([abs(r - med) for r in res], 0.5) == pytest.approx(
                record["rt_residual_mad_s"], abs=1e-9
            )
            w = max(
                rust_percentile([abs(r) for r in res], record["p_rt"]) * record["multiplier"], 1
            )
            assert w == pytest.approx(record["w_rt"], abs=1e-9)
        assert a.agrees is True, [(c.name, c.recorded, c.rebuilt) for c in a.checks]
        seen += 1
    assert seen


@pytest.mark.parametrize("name", UNGROUPED + GROUPED)
def test_the_funnel_counts_the_rule_step_by_step(open_fixture, name):
    rs = open_fixture(name)
    q_train = rs.config_get("rt_im_train", "q_train", default=0.01)
    for _run, s in _scopes(rs):
        a = cal.rt_anchors(rs, s.run, s.key or None)
        df = pq.read_table(s.artifact("seed_psms").path).to_pandas()
        finite = np.isfinite(df[["spectrum_q", "score", "observed_rt"]]).all(axis=1)
        confident = finite & (df["spectrum_q"] < q_train)
        target = confident & (df["label"] == "target")
        counts = [len(df), int(finite.sum()), int(confident.sum()), int(target.sum())]
        assert [n for _, n in a.funnel[:4]] == counts
        assert a.funnel[-1][1] == a.n
        assert a.n <= a.funnel[4][1] <= counts[3]


def test_rust_rounding_of_the_nearest_rank():
    # 110 values: (n - 1) / 2 = 54.5 rounds to 55 in Rust (Python's round gives 54).
    values = np.arange(110, dtype=float)
    assert cal.nearest_rank(values, 0.5) == 55.0
    assert cal.nearest_rank(np.arange(11.0), 0.5) == 5.0
    assert cal.nearest_rank(np.arange(101.0), 0.95) == 95.0
    assert math.isnan(cal.nearest_rank([], 0.5))
    med, absmed, mad = cal.residual_stats([-3.0, -1.0, 2.0, 4.0])
    assert (med, absmed, mad) == (2.0, 3.0, 3.0)


def test_curve_rows_are_run_windows_rows(open_fixture):
    rs = open_fixture("single")
    a = cal.rt_anchors(rs, rs.runs[0])
    lib = pq.read_table(rs.runs[0].artifacts["rt_library"].path, columns=["predicted_irt"])
    win = pq.read_table(rs.runs[0].artifacts["run_windows"].path).to_pandas()
    rows = set(
        zip(
            lib.column(0).to_numpy().astype(float),
            win["rt_pred_cal"],
            win["rt_lo"],
            win["rt_hi"],
            strict=True,
        )
    )
    curve = list(a.curve[["irt", "rt_pred_cal", "rt_lo", "rt_hi"]].itertuples(index=False))
    assert curve and all(tuple(r) in rows for r in curve)
    assert a.curve["irt"].is_monotonic_increasing


def test_anchors_are_cached_and_read_back_equal(open_fixture, fixture_dir):
    from mumdia_viewer.data import open_results

    rs = open_fixture("experiment")
    first = cal.rt_anchors(rs, rs.runs[1])
    again = open_results(fixture_dir("experiment"))  # a new result set: the disk cache
    second = cal.rt_anchors(again, again.runs[1])
    cols = [c for c in first.frame.columns if c not in ("peptidoform",)]
    pd.testing.assert_frame_equal(first.frame[cols], second.frame[cols])
    assert list(first.frame["peptidoform"]) == list(second.frame["peptidoform"])
    pd.testing.assert_frame_equal(first.curve, second.curve)
    assert first.funnel == second.funnel


@pytest.mark.parametrize("name", UNGROUPED + GROUPED)
def test_accepted_errors_are_the_target_rows_with_their_features(open_fixture, name):
    rs = open_fixture(name)
    t = 0.01
    for run in rs.runs:
        e = cal.accepted_errors(rs, run, t)
        assert e.error is None, e.error
        scored = pq.read_table(rs.scored.path).to_pandas()
        q = "run_psm_q" if rs.is_experiment else "q_value"
        want = scored[(scored["label"] == "target") & (scored[q] <= t)]
        if rs.is_experiment:
            want = want[want["source"] == run.index]
        if run.grouped is not None:
            s = cal.scopes(rs, run)[0]
            if s.band is not None:
                e = cal.accepted_errors(rs, run, t, s.key)
                want = want[(want["candidate_id"] >= s.offset)]
                want = want[want["candidate_id"] < s.offset + s.size]
        assert sorted(e.frame["candidate_id"]) == sorted(want["candidate_id"])
        assert (e.frame["q"] <= t).all()
        from mumdia_viewer.data.features import feature_source

        src = feature_source(rs, run)
        feats = pd.concat(
            [
                pq.read_table(
                    p,
                    columns=[
                        "candidate_id",
                        "peak_rank",
                        "rt_error_signed",
                        "predicted_rt_raw",
                        "frag_mass_err_median",
                        "n_matched_b",
                        "n_matched_y",
                    ],
                ).to_pandas()
                for p in src.paths
            ]
        )
        joined = want.merge(
            feats,
            left_on=["candidate_id", "selected_peak_rank"],
            right_on=["candidate_id", "peak_rank"],
            how="left",
        ).set_index("candidate_id")
        got = e.frame.set_index("candidate_id")
        for cid, row in got.iterrows():
            ref = joined.loc[cid]
            if ref["predicted_rt_raw"] != 0 and np.isfinite(ref["rt_error_signed"]):
                assert row["rt_error"] == pytest.approx(float(ref["rt_error_signed"]))
            else:
                assert math.isnan(row["rt_error"])
            if ref["n_matched_b"] + ref["n_matched_y"] >= 1:
                assert row["frag_mass_err_median"] == pytest.approx(
                    float(ref["frag_mass_err_median"])
                )
            else:
                assert math.isnan(row["frag_mass_err_median"])
        if run.grouped is not None:
            break


def test_accepted_errors_follow_the_threshold_and_the_half_window(open_fixture):
    rs = open_fixture("single")
    record = cal.cal_record(rs, rs.runs[0])
    small = cal.accepted_errors(rs, rs.runs[0], 0.001)
    large = cal.accepted_errors(rs, rs.runs[0], 0.05)
    assert small.n <= large.n
    assert set(small.frame["candidate_id"]) <= set(large.frame["candidate_id"])
    assert np.allclose(large.frame["half_width"], record.w_rt)
    rel = large.frame["rt_error"] / record.w_rt
    assert np.allclose(large.frame["rt_error_rel"], rel, equal_nan=True)
    with pytest.raises(ValueError):
        cal.accepted_errors(rs, rs.runs[0], 0.0)


def test_masscal_and_cal_records_are_the_files(open_fixture):
    rs = open_fixture("ovl_bp")
    for s in cal.scopes(rs, rs.runs[0]):
        m = cal.masscal_record(rs, s.run, s.key)
        raw = json.loads(s.side("masscal").read_text(encoding="utf-8"))
        assert m.frag_ppm_offset == raw["frag_ppm_offset"]
        assert m.frag_tol_ppm == raw["frag_tol_ppm"]
        assert m.n_dev == raw["n_dev"]
        assert m.uses_grid is (len(raw["mz_cal_grid_mz"]) >= 2)
        c = cal.cal_record(rs, s.run, s.key)
        rec = json.loads(s.side("cal").read_text(encoding="utf-8"))
        assert c.n_train == rec["n_train"] and c.w_rt == rec["w_rt"]
        assert c.status == rec["calibration_status"]


def test_grouped_scopes_are_the_bands_at_their_offsets(open_fixture):
    rs = open_fixture("ovl_bp")
    run = rs.runs[0]
    found = cal.scopes(rs, run)
    offsets = band_offsets(rs, run)
    assert [s.key for s in found] == [b.name for b in run.grouped.bands]
    assert all(s.mode == "band" and s.offset == offsets[s.key].offset for s in found)
    assert cal.scope(rs, run, "g01").key == "g01"
    assert cal.scope(rs, run, "nope").key == found[0].key
    assert cal.scopes(rs, open_fixture("single").runs[0])[0].mode == "run"


def test_no_calibration_gives_no_anchors_and_says_so(open_fixture):
    rs = open_fixture("grouped")  # no band reached a confident seed: unbounded windows
    for s in cal.scopes(rs, rs.runs[0]):
        a = cal.rt_anchors(rs, s.run, s.key)
        assert a.n == 0 and a.error is None
        assert a.funnel[2][1] == 0
        c = cal.cal_record(rs, s.run, s.key)
        assert c.status == "insufficient_anchors_unbounded" and c.w_rt is None
        e = cal.accepted_errors(rs, s.run, 0.01, s.key)
        assert e.frame["rt_error"].isna().all()  # the engine writes 0 without calibration


def test_rt_model_names_the_library(open_fixture):
    rs = open_fixture("single")
    m = cal.rt_model(rs, rs.runs[0])
    assert m.library_note == "the searched library" and m.summary is None
    assert m.library_label.endswith("fragment_library_precursors.parquet")
    band = cal.rt_model(open_fixture("ovl_bp"), "", "g01")
    assert band.library_note.startswith("the searched library, rows")


def test_fragment_mz_range_is_the_footer_range(open_fixture):
    rs = open_fixture("single")
    lo, hi, source = cal.fragment_mz_range(rs, rs.runs[0])
    col = pq.read_table(rs.runs[0].artifacts["chromatograms"].path, columns=["frag_mz"])
    values = col.column(0).to_numpy()
    assert lo == pytest.approx(float(np.nanmin(values))) and hi == pytest.approx(
        float(np.nanmax(values))
    )
    assert "frag_mz" in source


def test_binned_quantiles_are_numpy_quantiles_per_bin():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, 5000)
    y = rng.normal(0, 1, 5000) + x / 50
    df = cal.binned_quantiles(x, y, bins=10, min_count=20)
    assert len(df) == 10 and int(df["n"].sum()) == 5000
    edges = np.linspace(x.min(), x.max(), 11)
    k = 3
    inside = (x >= edges[k]) & (x < edges[k + 1])
    assert df["q50"].iloc[k] == pytest.approx(np.quantile(y[inside], 0.5))
    assert cal.binned_quantiles([], [], bins=4).empty


@pytest.mark.real_data
def test_real_runs_rebuild_cal_json(real_single, real_experiment):
    from mumdia_viewer.data import open_results

    for root in (real_single, real_experiment):
        rs = open_results(root)
        for run in rs.runs:
            a = cal.rt_anchors(rs, run)
            assert a.error is None and a.agrees is True, (run.name, a.checks)
            e = cal.accepted_errors(rs, run, 0.01)
            assert e.n > 0 and e.frame["rt_error"].notna().mean() > 0.99
