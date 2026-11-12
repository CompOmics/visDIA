"""A synthetic SCIEX ZT Scan DIA run, written for the tests (``build_ztscan_run``).

The run is small but has the layout MuMDIA 0.12.0 writes for a ZT Scan run: ``spectra_ms2``
v3 with the Q1 sweep (``q1_cycle``, ``q1_step``), ``cal.json`` with ``q1_calibration``,
``psms_extracted`` v6 with the per-fragment Q1 decisions (``q1f_*``, the model in the report's
``params.q1_fragments``) and the chromatograms of the candidates: one sample per covering
step (layout 1, version 1), or with ``cycle_traces=True`` the engine's cycle-level traces of
``extract.q1_traces`` (layout 1 plus ``q1_class``, version 5, and ``psms_extracted`` v9 with
two of the separable-model columns).

The signal is planted with known values. The quadrupole sweeps 60 steps 0.7614 m/z apart
(targets 500.0 to 544.9) in 1 s cycles, each step read as +-1.9 m/z. Every fragment of a
candidate is placed at its exact library m/z with the intensity ``A_f * E_c * w_f(target)``:
``A_f`` its amplitude, ``E_c`` a Gaussian elution over the cycles and ``w_f`` its calibrated
Q1 profile (:class:`mumdia_viewer.data.ztscan.CandQ1`) at the expected centre, so the
transmission-weighted collapse of a cycle's covering steps is ``A_f * E_c`` exactly. One
fragment of candidate 0 is centred four steps away (``mu = 4``, the engine's class
"interfered"), one is absent; the precursor's isotopes are planted in MS2 through their own
windows. Noise peaks keep at least 60 ppm from every planted m/z.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mumdia_viewer.data import ztscan as zt

STEP = 0.7614
N_STEPS = 60
TARGET0 = 500.0
HALF = 1.9
N_CYCLES = 40
CYCLE_S = 1.0
STEP_S = 0.001345
RT0 = 100.0
TOL_PPM = 20.0
PREC_TOL_PPM = 3.0
ELUTION_SD = 1.6
WINDOW_S = 8.0

MODEL_BLOCK = {
    "step_mz": STEP,
    "half_width_mz": 2.58,
    "edge_width_mz": 4.1,
    "margin_steps": 2.0,
    "patch_cycles": 7,
    "profile_cycles": 5,
    "reach_mz": 2.58 + 2.05,
    "q1_repick": True,
    "repick_peaks": 5,
    "offset_terms": [{"var": "precursor_mz", "grid": [400.0, 900.0], "offset_mz": [0.12, 0.02]}],
    "tolerances": {"offset_steps": 1.4, "centre_steps": 0.76, "corr2d_min": 0.44},
}
CAL_BLOCK = {
    "status": "fitted",
    "anchors": 1000,
    "profiles": 6000,
    "half_width_mz": 2.58,
    "edge_width_mz": 4.1,
    "sweep": {"step_mz": STEP},
    "offset": {"terms": {"precursor_mz": {"grid": [400.0, 900.0], "offset_mz": [0.12, 0.02]}}},
    "fragment_decision": {
        "tolerances": {"offset_steps": 1.4, "centre_steps": 0.76, "corr2d_min": 0.44},
        "patch_cycles": 7,
        "profile_cycles": 5,
        "margin_steps": 2.0,
    },
}


@dataclass
class Cand:
    cid: int
    peptidoform: str
    label: str
    pm: float
    z: int
    apex_cycle: int
    names: list[str]
    mz: list[float]
    amp: list[float]
    mu: list[float]
    classes: list[int]
    bpid: int
    score: float
    q: float
    protein: str = "sp|P00001|TEST_HUMAN"
    apex_rt: float = 0.0
    model: zt.CandQ1 | None = field(default=None, repr=False)


def candidates() -> list[Cand]:
    """The planted candidates: two accepted targets and one decoy."""
    return [
        Cand(
            0,
            "LGEDNINVVEGNEQFISASK",
            "target",
            520.30,
            2,
            20,
            ["b3", "b4", "y3", "y4", "y5", "y6", "y7", "y8"],
            [301.1503, 414.2343, 347.2036, 476.2462, 589.3303, 702.4143, 815.4984, 928.5824],
            [8000.0, 5000.0, 12000.0, 9000.0, 15000.0, 7000.0, 11000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [1, 1, 1, 1, 1, 2, 1, 0],
            10,
            3.2,
            0.0004,
        ),
        Cand(
            1,
            "KSASIFQENGEVVNINDEGL",
            "decoy",
            525.80,
            2,
            22,
            ["b3", "y3", "y4", "y5"],
            [312.1660, 360.2240, 489.2666, 602.3507],
            [900.0, 1500.0, 0.0, 700.0],
            [0.0, 2.5, 0.0, 0.0],
            [1, 2, 0, 1],
            11,
            0.2,
            0.6,
        ),
        Cand(
            2,
            "AVLDEMNPSHR",
            "target",
            510.55,
            3,
            15,
            ["y2", "y3", "y4", "y5^2", "y6"],
            [312.1773, 449.2362, 536.2683, 333.6620, 650.3112],
            [6000.0, 4000.0, 9000.0, 3000.0, 5000.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1, 1, 1, 1, 1],
            12,
            2.5,
            0.002,
        ),
    ]


@dataclass
class Truth:
    """What the builder planted, for the tests."""

    root: Path
    cands: list[Cand]
    model: zt.Q1Model
    targets: np.ndarray
    cycle_traces: bool
    # (cycle, step) -> {mz: intensity} of the planted fragment and precursor peaks.
    planted: dict[tuple[int, int], dict[float, float]]

    def elution(self, c: Cand, cycle: int) -> float:
        return _elution(c, cycle)


def _elution(c: Cand, cycle: int) -> float:
    d = cycle - c.apex_cycle
    return math.exp(-0.5 * (d / ELUTION_SD) ** 2) if abs(d) <= 6 else 0.0


def ms2_rt(cycle: int, step: int) -> float:
    return RT0 + cycle * CYCLE_S + 0.01 + step * STEP_S


def ms1_rt(cycle: int) -> float:
    return RT0 + cycle * CYCLE_S


def _write(
    table: pa.Table, path: Path, schema: str, version: int, stage: str, params=None, rgs=None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=rgs or max(table.num_rows, 1), compression="snappy")
    report = {
        "logical_name": schema,
        "schema_name": schema,
        "schema_version": version,
        "stage": stage,
        "rows": table.num_rows,
        "content_hash": None,
        "params": params or {},
        "stats": {},
        "model_identity": None,
        "elapsed_ms": 1,
    }
    path.with_name(path.name + ".report.json").write_text(json.dumps(report), encoding="utf-8")
    return {
        "logical_name": schema,
        "path": str(path),
        "format": "parquet",
        "schema_name": schema,
        "schema_version": version,
        "rows": table.num_rows,
        "content_hash": None,
        "producing_stage": stage,
        "config_hash": "synthetic",
    }


def _list(values: list[np.ndarray]) -> pa.Array:
    return pa.array(
        [np.asarray(v, dtype=np.float32) for v in values], type=pa.large_list(pa.float32())
    )


def build_ztscan_run(root: Path, *, cycle_traces: bool = False, seed: int = 7) -> Truth:
    """Write the synthetic ZT Scan run into ``root`` (see the module docstring)."""
    root = Path(root)
    (root / "spectra").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = zt.model_from_block(MODEL_BLOCK, "synthetic")
    targets = TARGET0 + STEP * np.arange(N_STEPS)
    cands = candidates()
    for c in cands:
        c.model = zt.CandQ1(model, c.pm, c.z, 0.0, c.names, c.mz)
        # The apex: the cycle's step nearest the expected Q1 centre.
        s = int(np.argmin(np.abs(targets - c.model.centre)))
        c.apex_rt = ms2_rt(c.apex_cycle, s)
        # The offset depends on RT only through an "rt" term, which this model has not.
        c.model = zt.CandQ1(model, c.pm, c.z, c.apex_rt, c.names, c.mz)
    planted_mz: list[float] = []
    for c in cands:
        planted_mz += c.mz
        sp = zt.ISOTOPE_SPACING / c.z
        planted_mz += [c.pm + k * sp for k in range(3)]

    def noise(n: int) -> list[tuple[float, float]]:
        out = []
        while len(out) < n:
            mz = float(rng.uniform(200.0, 1000.0))
            if all(abs(mz - p) / p * 1e6 > 60 for p in planted_mz):
                out.append((mz, float(rng.uniform(40.0, 300.0))))
        return out

    # MS2 spectra.
    planted: dict[tuple[int, int], dict[float, float]] = {}
    rows = {
        k: []
        for k in (
            "scan_index",
            "id",
            "rt",
            "wid",
            "tgt",
            "lo",
            "hi",
            "pmz",
            "mz",
            "it",
            "cyc",
            "stp",
        )
    }
    ms1 = {k: [] for k in ("scan_index", "id", "rt", "mz", "it")}
    scan = 0
    for cyc in range(N_CYCLES):
        scan += 1
        peaks = noise(15)
        for c in cands:
            e = _elution(c, cyc)
            sp = zt.ISOTOPE_SPACING / c.z
            for k, w in enumerate(c.model.pre[:3]):
                if e > 0:
                    peaks.append((c.pm + k * sp, 1e5 * e * float(w)))
        peaks.sort()
        ms1["scan_index"].append(scan)
        ms1["id"].append(f"cycle={cyc + 1} experiment=1")
        ms1["rt"].append(ms1_rt(cyc))
        ms1["mz"].append([p[0] for p in peaks])
        ms1["it"].append([p[1] for p in peaks])
        for s in range(N_STEPS):
            scan += 1
            t = float(targets[s])
            mine: dict[float, float] = {}
            for c in cands:
                e = _elution(c, cyc)
                if e <= 0:
                    continue
                for f in range(len(c.names)):
                    if c.amp[f] <= 0:
                        continue
                    w = float(c.model.expected(f, t, c.mu[f]))
                    v = c.amp[f] * e * w
                    if v > 0.5:
                        mine[c.mz[f]] = v
                if c.cid == 0:
                    prec = zt.CandQ1.precursor(model, c.pm, c.z, c.apex_rt)
                    for k in range(3):
                        v = (
                            30000.0
                            * e
                            * float(prec.frags[k].weights[k])
                            * float(prec.expected(k, t))
                            * float(c.model.pre[k])
                        )
                        if v > 0.5:
                            mine[prec.frags[k].mz] = v
            planted[(cyc, s)] = dict(mine)
            peaks = sorted(noise(25) + list(mine.items()))
            rows["scan_index"].append(scan)
            rows["id"].append(f"cycle={cyc + 1} experiment={s + 2}")
            rows["rt"].append(ms2_rt(cyc, s))
            rows["wid"].append(s)
            rows["tgt"].append(t)
            rows["lo"].append(t - HALF)
            rows["hi"].append(t + HALF)
            rows["pmz"].append(t)
            rows["mz"].append([p[0] for p in peaks])
            rows["it"].append([p[1] for p in peaks])
            rows["cyc"].append(cyc)
            rows["stp"].append(s)
    records: dict[str, dict] = {}
    ms2 = pa.table(
        {
            "scan_index": pa.array(rows["scan_index"], pa.uint32()),
            "id": pa.array(rows["id"], pa.string()),
            "rt_seconds": pa.array(rows["rt"], pa.float64()),
            "window_id": pa.array(rows["wid"], pa.uint32()),
            "window_target": pa.array(rows["tgt"], pa.float64()),
            "window_lower": pa.array(rows["lo"], pa.float64()),
            "window_upper": pa.array(rows["hi"], pa.float64()),
            "precursor_mz": pa.array(rows["pmz"], pa.float64()),
            "precursor_charge": pa.array([None] * len(rows["rt"]), pa.int32()),
            "mz": _list(rows["mz"]),
            "intensity": _list(rows["it"]),
            "q1_cycle": pa.array(rows["cyc"], pa.uint32()),
            "q1_step": pa.array(rows["stp"], pa.uint32()),
        }
    )
    records["spectra_ms2"] = _write(
        ms2,
        root / "spectra" / "spectra_ms2.parquet",
        "spectra_ms2",
        3,
        "convert",
        {
            "isolation_half_width": HALF,
            "q1_sweep": {"cycles": N_CYCLES, "max_steps": N_STEPS},
            "instrument": {"accession": "MS:1003443", "name": "ZenoTOF 8600"},
        },
        rgs=128,
    )
    ms1t = pa.table(
        {
            "scan_index": pa.array(ms1["scan_index"], pa.uint32()),
            "id": pa.array(ms1["id"], pa.string()),
            "rt_seconds": pa.array(ms1["rt"], pa.float64()),
            "mz": _list(ms1["mz"]),
            "intensity": _list(ms1["it"]),
        }
    )
    records["spectra_ms1"] = _write(
        ms1t, root / "spectra" / "spectra_ms1.parquet", "spectra_ms1", 1, "convert"
    )
    iso = pa.table(
        {
            "window_id": pa.array(range(N_STEPS), pa.uint32()),
            "target": pa.array(targets, pa.float64()),
            "lower": pa.array(targets - HALF, pa.float64()),
            "upper": pa.array(targets + HALF, pa.float64()),
        }
    )
    records["isolation_windows"] = _write(
        iso, root / "spectra" / "isolation_windows.parquet", "isolation_windows", 1, "convert"
    )

    # Chromatograms: the covering-step grid of each candidate in its RT window.
    ms2_rt_arr = np.asarray(rows["rt"])
    ms2_cyc = np.asarray(rows["cyc"])
    ms2_stp = np.asarray(rows["stp"])
    ms1_rt_arr = np.asarray(ms1["rt"])
    chrom = {k: [] for k in ("cid", "name", "mz", "pred", "rt", "it", "q1")}
    for c in cands:
        lo, hi = c.apex_rt - WINDOW_S, c.apex_rt + WINDOW_S
        cover = (targets - HALF <= c.pm) & (c.pm <= targets + HALF)
        grid = np.flatnonzero(cover[ms2_stp] & (ms2_rt_arr >= lo) & (ms2_rt_arr <= hi))
        grid = grid[np.argsort(ms2_rt_arr[grid], kind="stable")]

        def ms1_at(rt: float, mz: float) -> float:
            j = int(np.argmin(np.abs(ms1_rt_arr - rt)))
            ps, its = np.asarray(ms1["mz"][j]), np.asarray(ms1["it"][j])
            d = mz * PREC_TOL_PPM * 1e-6
            return float(np.float32(its[(ps >= mz - d) & (ps <= mz + d)].sum()))

        sp = zt.ISOTOPE_SPACING / c.z
        if not cycle_traces:
            axis = ms2_rt_arr[grid].astype(np.float32)
            for f, name in enumerate(c.names):
                vals = np.array(
                    [planted[(int(ms2_cyc[g]), int(ms2_stp[g]))].get(c.mz[f], 0.0) for g in grid],
                    dtype=np.float32,
                )
                observed = bool(np.any(vals > 0))
                chrom["cid"].append(c.cid)
                chrom["name"].append(name)
                chrom["mz"].append(c.mz[f])
                chrom["pred"].append(1.0 - 0.05 * f)
                chrom["rt"].append(axis if observed else np.zeros(0, np.float32))
                chrom["it"].append(vals if observed else np.zeros(0, np.float32))
                chrom["q1"].append(float(c.classes[f]))
            for k, name in enumerate(("ms1_mono", "ms1_iso1", "ms1_iso2")):
                chrom["cid"].append(c.cid)
                chrom["name"].append(name)
                chrom["mz"].append(c.pm + k * sp)
                chrom["pred"].append(0.0)
                chrom["rt"].append(axis)
                chrom["it"].append(
                    np.array(
                        [ms1_at(float(r), c.pm + k * sp) for r in ms2_rt_arr[grid]], np.float32
                    )
                )
                chrom["q1"].append(float("nan"))
        else:
            cycles = np.unique(ms2_cyc[grid])
            rep_step = int(np.argmin(np.abs(targets - c.model.centre)))
            cyc_rt = np.array([ms2_rt(int(k), rep_step) for k in cycles])
            axis = cyc_rt.astype(np.float32)
            for f, name in enumerate(c.names):
                vals = []
                for k in cycles:
                    g = grid[ms2_cyc[grid] == k]
                    x = np.array([planted[(int(k), int(ms2_stp[i]))].get(c.mz[f], 0.0) for i in g])
                    w = c.model.expected(f, targets[ms2_stp[g]])
                    sww = float(np.sum(w * w))
                    vals.append(float(np.sum(w * x)) / sww if sww > 0 else 0.0)
                vals = np.asarray(vals, dtype=np.float32)
                observed = bool(np.any(vals > 0))
                chrom["cid"].append(c.cid)
                chrom["name"].append(name)
                chrom["mz"].append(c.mz[f])
                chrom["pred"].append(1.0 - 0.05 * f)
                chrom["rt"].append(axis if observed else np.zeros(0, np.float32))
                chrom["it"].append(vals if observed else np.zeros(0, np.float32))
                chrom["q1"].append(float(c.classes[f]))
            for k, name in enumerate(("ms1_mono", "ms1_iso1", "ms1_iso2")):
                chrom["cid"].append(c.cid)
                chrom["name"].append(name)
                chrom["mz"].append(c.pm + k * sp)
                chrom["pred"].append(0.0)
                chrom["rt"].append(axis)
                chrom["it"].append(
                    np.array([ms1_at(float(r), c.pm + k * sp) for r in cyc_rt], np.float32)
                )
                chrom["q1"].append(float("nan"))
    ct = {
        "candidate_id": pa.array(chrom["cid"], pa.uint32()),
        "frag_name": pa.array(chrom["name"], pa.string()),
        "frag_mz": pa.array(chrom["mz"], pa.float64()),
        "frag_obs_mz": pa.array(chrom["mz"], pa.float64()),
        "predicted_intensity": pa.array(chrom["pred"], pa.float32()),
        "rt": _list(chrom["rt"]),
        "intensity": _list(chrom["it"]),
    }
    params = {
        "effective_frag_tol_ppm": TOL_PPM,
        "frag_ppm_offset": 0.0,
        "frag_tol_ppm": TOL_PPM,
        "q1_fragments": MODEL_BLOCK,
    }
    if cycle_traces:
        ct["q1_class"] = pa.array(chrom["q1"], pa.float32())
        params = {**params, "q1_traces": {"apex_on_cycles": True, "gate_cycles": 3}}
    records["chromatograms"] = _write(
        pa.table(ct),
        root / "chromatograms.parquet",
        "chromatograms",
        5 if cycle_traces else 1,
        "extract",
        params,
    )

    # psms_extracted.
    n = len(cands)
    nan = float("nan")

    def lists(fn) -> pa.Array:
        return pa.array(
            [[float(fn(c, f)) for f in range(len(c.names))] for c in cands], pa.list_(pa.float32())
        )

    pe = {
        "candidate_id": pa.array([c.cid for c in cands], pa.uint32()),
        "peak_rank": pa.array([0] * n, pa.int32()),
        "apex_rt": pa.array([c.apex_rt for c in cands], pa.float64()),
        "apex_im": pa.array([None] * n, pa.float64()),
        "apex_intensity": pa.array([max(c.amp) for c in cands], pa.float32()),
        "n_matched_fragments": pa.array([sum(a > 0 for a in c.amp) for c in cands], pa.int32()),
        "n_predicted_fragments": pa.array([len(c.names) for c in cands], pa.int32()),
        "coelution_run": pa.array([4] * n, pa.int32()),
        "rt_pred_cal": pa.array([c.apex_rt + 0.3 for c in cands], pa.float64()),
        "precursor_mz": pa.array([c.pm for c in cands], pa.float64()),
        "charge": pa.array([c.z for c in cands], pa.int32()),
        "label": pa.array([c.label for c in cands], pa.string()),
        "base_peptide_id": pa.array([c.bpid for c in cands], pa.uint32()),
        "peptidoform": pa.array([c.peptidoform for c in cands], pa.string()),
        "protein": pa.array([c.protein for c in cands], pa.string()),
        "predicted_irt": pa.array([50.0] * n, pa.float32()),
        "contested_frac": pa.array([0.0] * n, pa.float64()),
        "rt_lo": pa.array([c.apex_rt - WINDOW_S for c in cands], pa.float64()),
        "rt_hi": pa.array([c.apex_rt + WINDOW_S for c in cands], pa.float64()),
        "q1f_n_present": pa.array([float(sum(k > 0 for k in c.classes)) for c in cands]),
        "q1f_n_belong": pa.array([float(sum(k == 1 for k in c.classes)) for c in cands]),
        "q1f_n_interfered": pa.array([float(sum(k == 2 for k in c.classes)) for c in cands]),
        "q1f_belong_share": pa.array([0.8] * n),
        "q1f_interfered_share": pa.array([0.1] * n),
        "q1f_offset_median": pa.array([0.05] * n),
        "q1f_offset_mad": pa.array([0.3] * n),
        "q1f_corr2d": pa.array([0.9] * n),
        "q1f_fit_resid": pa.array([0.1] * n),
        "q1f_centre": pa.array([0.1] * n),
        "q1f_width": pa.array([1.0] * n),
        "q1f_apex_shift_cycles": pa.array([0.0] * n),
        "q1f_apex_moved_s": pa.array([0.0] * n),
        "q1f_frag_class": lists(lambda c, f: c.classes[f]),
        "q1f_frag_offset": lists(lambda c, f: nan if c.classes[f] == 0 else c.mu[f]),
        "q1f_frag_resid": lists(lambda c, f: nan if c.classes[f] == 0 else 0.1),
        "q1f_frag_corr2d": lists(lambda c, f: nan if c.classes[f] == 0 else 0.9),
    }
    version = 6
    if cycle_traces:
        pe["q1s_r2"] = pa.array([0.8] * n)
        pe["q1s_elution"] = pa.array(
            [[_elution(c, c.apex_cycle + k) for k in range(-3, 4)] for c in cands],
            pa.list_(pa.float32()),
        )
        pe["q1s_cycle_rt"] = pa.array(
            [
                [
                    ms2_rt(c.apex_cycle + k, int(np.argmin(np.abs(targets - c.model.centre))))
                    for k in range(-3, 4)
                ]
                for c in cands
            ],
            pa.list_(pa.float32()),
        )
        pe["q1s_frag_amp"] = lists(lambda c, f: c.amp[f] if c.mu[f] == 0 else c.amp[f] / 3)
        pe["q1s_frag_r2"] = lists(lambda c, f: 0.9 if c.classes[f] == 1 else 0.2)
        pe["q1s_frag_interf"] = lists(lambda c, f: 0.05 if c.classes[f] == 1 else 0.7)
        version = 9
    records["psms_extracted"] = _write(
        pa.table(pe), root / "psms_extracted.parquet", "psms_extracted", version, "extract", params
    )

    # The scored table.
    sc = {
        "candidate_id": pa.array([c.cid for c in cands], pa.uint32()),
        "peptidoform": pa.array([c.peptidoform for c in cands], pa.string()),
        "charge": pa.array([c.z for c in cands], pa.int32()),
        "label": pa.array([c.label for c in cands], pa.string()),
        "protein": pa.array([c.protein for c in cands], pa.string()),
        "base_peptide_id": pa.array([c.bpid for c in cands], pa.uint32()),
        "apex_rt": pa.array([c.apex_rt for c in cands], pa.float64()),
        "elution_lo": pa.array([float(np.float32(c.apex_rt - 2.0)) for c in cands], pa.float64()),
        "elution_hi": pa.array([float(np.float32(c.apex_rt + 2.0)) for c in cands], pa.float64()),
        "score": pa.array([c.score for c in cands], pa.float64()),
        "q_value": pa.array([c.q for c in cands], pa.float64()),
        "peptide_q_value": pa.array([c.q for c in cands], pa.float64()),
        "protein_group": pa.array([c.protein for c in cands], pa.string()),
        "pg_q_value": pa.array([c.q for c in cands], pa.float64()),
        "prelim_score": pa.array([c.score for c in cands], pa.float64()),
        "source": pa.array([0] * n, pa.uint32()),
        "run_psm_q": pa.array([c.q for c in cands], pa.float64()),
        "precursor_q": pa.array([c.q for c in cands], pa.float64()),
        "selected_peak_rank": pa.array([0] * n, pa.int32()),
    }
    records["psms_scored"] = _write(
        pa.table(sc),
        root / "psms_scored.parquet",
        "psms_scored",
        4,
        "rescore",
        {"classifier": "native_tda"},
    )
    rw = pa.table(
        {
            "candidate_id": pa.array([c.cid for c in cands], pa.uint32()),
            "rt_pred_cal": pa.array([c.apex_rt + 0.3 for c in cands], pa.float64()),
            "rt_lo": pa.array([c.apex_rt - WINDOW_S for c in cands], pa.float64()),
            "rt_hi": pa.array([c.apex_rt + WINDOW_S for c in cands], pa.float64()),
        }
    )
    records["run_windows"] = _write(
        rw, root / "run_windows.parquet", "run_windows", 1, "rt-im-train"
    )
    (root / "cal.json").write_text(
        json.dumps({"method": "loess", "q1_calibration": CAL_BLOCK}), encoding="utf-8"
    )
    config = {
        "extract": {"emit_window_grid": True, "prec_tol_ppm": PREC_TOL_PPM, "q1_fragments": "auto"},
        "convert": {"isolation_half_width": HALF},
    }
    manifest = {
        "mumdia_version": "0.12.0",
        "git_sha": "synthetic",
        "commit_date": "unknown",
        "cli_args": ["mumdia", "run", "--out-dir", str(root)],
        "config_json": json.dumps(config),
        "config_hash": "synthetic",
        "model_identities": {},
        "inputs": {},
        "artifacts": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return Truth(root, cands, model, targets, cycle_traces, planted)
