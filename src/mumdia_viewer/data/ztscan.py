"""SCIEX ZT Scan DIA runs: the Q1 sweep, the calibrated Q1 model and the evidence the
precursor page draws from them.

In a ZT Scan DIA run the quadrupole sweeps the precursor range once a cycle in steps far
narrower than its window (on the ZenoTOF method 663 steps 0.7614 m/z apart, a 3.8 m/z
window, one cycle a second), so a precursor is transmitted through about five to seven
consecutive steps of every cycle, each step an MS2 spectrum of its own at a different
transmission. MuMDIA converts such a run into ``spectra_ms2`` v3, which records the
sweep: ``q1_cycle`` and ``q1_step`` of every spectrum (docs/37_ztscan.md of the engine).
The chromatogram table holds one sample per covering spectrum, so a trace drawn through
every sample is a sawtooth: five samples 1.3 ms apart, each at its own transmission, then
a second to the next cycle.

The rules below are the engine's (MuMDIA 0.12.0, ``stages/q1frag.rs`` and
``stages/q1quant.rs``):

* **The Q1 model** (:class:`Q1Model`) is the calibration ``rt_im_train.q1_calibrate``
  fitted on the seed: a trapezoid transmission of half-width ``h`` and edge width ``e``
  (whole within ``h - e/2`` of the window centre, nothing beyond ``h + e/2``), centred
  ``offset(precursor m/z, fragment m/z, RT)`` above each step's target. Extract records
  the model it decided with in ``params.q1_fragments`` of the ``psms_extracted`` report,
  which is read first; ``cal.json``'s ``q1_calibration`` otherwise.
* **The expected Q1 profile** of a fragment (:class:`CandQ1`) is that transmission summed
  over the precursor's isotopes, each weighted by its Poisson-averagine abundance times
  the chance ``(1 - f)^i`` that the fragment stays monoisotopic (``f`` its share of the
  precursor mass). Its centre in target units is ``pm + spacing * mean isotope -
  offset``.
* **One value per cycle** (:func:`cycle_traces`): every sample of a trace is located in
  the sweep (the nearest spectrum within 0.45 of the median scan spacing whose window
  covers the precursor), and the samples of one cycle collapse to
  ``sum(w x) / sum(w^2)``, the least-squares amplitude under the expected profile ``w``
  at the calibrated expected centre, at the mean retention time of the cycle's samples.
  It is the estimator of ``quant.q1_fragments`` (``q1quant::collapse_trace``), so it is
  the fragment's intensity at full transmission. MS1 rows (one MS1 scan a cycle) take the
  mean of the cycle's samples.
* **The patch** (:func:`q1_patch`): the cycles around the apex cycle by the steps whose
  target lies within the window's reach (``h + e/2``) of the candidate's expected centre,
  plus two steps either side; a cell holds the intensity of the peak nearest the
  offset-corrected fragment m/z within the extraction's fragment tolerance, in the
  spectrum of that step in that cycle (``q1frag::read_patch``). The engine's decision
  reads three cycles either side.

Everything here is a viewer computation with the engine's rules and is labelled so; the
engine's own numbers are the ``q1f_*`` columns of ``psms_extracted`` (:func:`q1f_row`).
Peaks are read only for the cells of a patch, one row group of ``spectra_ms2`` at a time,
and the cell spectra are kept in a small cache, so a patch on a run of billions of peaks
reads a few row groups.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .chromatograms import CandidateChromatogram, Q1TraceSource
from .duck import sql_ident, sql_path
from .errors import ViewerError
from .pqio import list_values
from .reports import load_json
from .spectra import ScanTable

if TYPE_CHECKING:
    from .detail import PrecursorDetail
    from .discovery import ResultSet, Run
    from .fragments import Tolerance

__all__ = [
    "ABSENT",
    "BELONGS",
    "CELL_CACHE",
    "CLASS_NAMES",
    "CYCLE_LABEL",
    "ENGINE_CYCLE_LABEL",
    "INTERFERED",
    "PATCH_LABEL",
    "UNIFORM_LABEL",
    "CandQ1",
    "CycleTrace",
    "CycleTraces",
    "Q1Model",
    "Q1Patch",
    "Q1Profile",
    "ZtRun",
    "ZtSweep",
    "averagine_isotopes",
    "cycle_traces",
    "isotope_weights",
    "model_from_block",
    "q1_patch",
    "q1_profile",
    "q1f_row",
    "separable_model",
    "summary_items",
    "transmission",
    "ztscan_run",
]

# mumdia-core constants.rs
PROTON = 1.007276466812
ISOTOPE_SPACING = 1.003354835
AVERAGINE_LAMBDA_PER_DA = 0.000594
# stages/q1frag.rs
ISOTOPES = 6
PATCH_CYCLES = 3
PROFILE_CYCLES = 2
MARGIN_STEPS = 2.0
# stages/q1score.rs: the unfragmented precursor's isotopes read in MS2.
PREC_ISOTOPES = 3

# The fragment classes of q1f_frag_class.
ABSENT, BELONGS, INTERFERED = 0, 1, 2
CLASS_NAMES = {ABSENT: "absent", BELONGS: "belongs", INTERFERED: "interfered"}
CLASS_TEXT = {
    ABSENT: "absent: fewer than three coherent cells within the window's reach",
    BELONGS: (
        "belongs: its own Q1 centre lies within the offset tolerance of the shared centre "
        "and its 2D correlation reaches the floor"
    ),
    INTERFERED: (
        "interfered: present, but centred elsewhere in Q1 or not co-eluting with the others "
        "(another precursor's signal at this m/z)"
    ),
}

CYCLE_LABEL = (
    "viewer-derived: one value per Q1 cycle, the transmission-weighted collapse "
    "sum(w x)/sum(w^2) of the cycle's covering steps under the calibrated Q1 profile at the "
    "expected Q1 centre (the estimator of the engine's quant.q1_fragments), at the mean RT "
    "of the cycle's samples"
)
UNIFORM_LABEL = (
    "viewer-derived: one value per Q1 cycle, the mean of the cycle's covering steps (no Q1 "
    "model is recorded for this run, so the steps are not weighted by transmission)"
)
MS1_CYCLE_LABEL = "viewer-derived: the mean of the cycle's samples (one MS1 scan a cycle)"
STEP_LABEL = (
    "the engine's trace: one sample per covering MS2 spectrum (step) of the chromatogram table"
)
ENGINE_CYCLE_LABEL = (
    "the engine's cycle-level trace (extract.q1_traces, chromatograms_q1.parquet, schema 5/6): "
    "one point per Q1 "
    "cycle, the least-squares amplitude sum(w x)/sum(w^2) of the cycle's covering steps under "
    "the calibrated Q1 transmission, at the RT of the cycle's step nearest the expected Q1 "
    "centre"
)
PATCH_LABEL = (
    "viewer read of spectra_ms2: per cell the nearest peak within the extraction's fragment "
    "tolerance of the offset-corrected fragment m/z (the engine's patch rule, "
    "q1frag::read_patch)"
)

# The cell spectra kept between patch reads (decoded m/z and intensity of single rows).
CELL_CACHE_BYTES = 96 * 2**20
# Row groups of spectra_ms2 read at once for a patch.
READ_THREADS = 4

_MEMO_LOCK = threading.Lock()


# --------------------------------------------------------------------------- physics


def averagine_isotopes(mass: float, n: int = ISOTOPES) -> np.ndarray:
    """Poisson-averagine abundances of the isotopes M+0 .. M+(n-1) of a peptide of neutral
    mass ``mass`` (Da), normalised to sum 1 (``constants::averagine_isotopes``)."""
    lam = max(AVERAGINE_LAMBDA_PER_DA * float(mass), 0.0)
    v = np.empty(n, dtype=np.float64)
    t = math.exp(-lam)
    for k in range(n):
        if k > 0:
            t *= lam / k
        v[k] = t
    s = float(v.sum())
    return v / s if s > 0 else v


def _neutral(mz: float, z: int) -> float:
    return (float(mz) - PROTON) * int(z)


def transmission(x: np.ndarray | float, h: float, e: float) -> np.ndarray:
    """The share of an ion at distance ``x`` (m/z) from a window's centre that the window
    transmits: whole within ``h - e/2``, nothing beyond ``h + e/2``, linear in between."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    if e <= 0:
        return (ax <= h).astype(np.float64)
    return np.clip((h + 0.5 * e - ax) / e, 0.0, 1.0)


def isotope_weights(pm: float, z: int, fmz: float, fz: int) -> np.ndarray:
    """The precursor isotopes' weights for a fragment: Poisson-averagine abundance times the
    chance ``(1 - f)^i`` that the fragment stays monoisotopic, ``f`` its share of the
    precursor mass, normalised (``q1frag::isotope_weights``)."""
    m = _neutral(pm, z)
    frac = min(max(_neutral(fmz, fz) / m, 0.0), 1.0) if m != 0 else 0.0
    w = averagine_isotopes(m, ISOTOPES)
    keep = 1.0
    for i in range(w.size):
        w[i] *= keep
        keep *= 1.0 - frac
    s = float(w.sum())
    return w / s if s > 0 else w


def _interp_held(grid: np.ndarray, values: np.ndarray, x: float) -> float:
    """Linear interpolation held at the ends (``config::interp_held``)."""
    n = min(grid.size, values.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    if math.isnan(x) or x <= grid[0]:
        return float(values[0])
    if x >= grid[n - 1]:
        return float(values[n - 1])
    j = int(np.searchsorted(grid[:n], x, side="left"))
    x0, x1, y0, y1 = grid[j - 1], grid[j], values[j - 1], values[j]
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0)) if x1 > x0 else float(y0)


def fragment_charge(name: str) -> int:
    """The charge of a fragment named like ``y7`` or ``b3^2`` (1 without a suffix)."""
    _, sep, tail = str(name).partition("^")
    if sep:
        try:
            z = int(tail)
        except ValueError:
            return 1
        return z if z >= 1 else 1
    return 1


# --------------------------------------------------------------------------- the model


@dataclass(frozen=True, eq=False)
class Q1Model:
    """The calibrated Q1 window of a run (``Q1FragModel`` of the engine).

    ``offset_terms`` are ``(variable, grid, offset_mz)``, the variable ``precursor_mz``,
    ``fragment_mz`` or ``rt``; the window is centred their sum above each step's target.
    ``tolerances`` are the decision's (``offset_steps``, ``centre_steps``, ``corr2d_min``)
    when recorded. ``source`` names the file and key the model was read from.
    """

    step_mz: float
    half_width_mz: float
    edge_width_mz: float
    offset_terms: tuple[tuple[str, np.ndarray, np.ndarray], ...]
    tolerances: dict[str, float] | None
    source: str
    margin_steps: float = MARGIN_STEPS
    patch_cycles: int = 2 * PATCH_CYCLES + 1
    profile_cycles: int = 2 * PROFILE_CYCLES + 1

    @property
    def reach_mz(self) -> float:
        """How far from a window's centre it still transmits anything: ``h + e/2``."""
        return self.half_width_mz + 0.5 * self.edge_width_mz

    def offset_mz(self, pm: float, fmz: float | None, rt: float) -> float:
        """The window-centre offset (m/z) at precursor m/z ``pm``, fragment m/z ``fmz``
        (None leaves the fragment term out) and retention time ``rt``."""
        total = 0.0
        for var, grid, off in self.offset_terms:
            if var == "precursor_mz":
                total += _interp_held(grid, off, pm)
            elif var == "fragment_mz":
                total += 0.0 if fmz is None else _interp_held(grid, off, fmz)
            elif var == "rt":
                total += _interp_held(grid, off, rt)
        return total

    @property
    def label(self) -> str:
        return (
            f"calibrated Q1 window: half-width {self.half_width_mz:.3f} m/z, soft edge "
            f"{self.edge_width_mz:.3f} m/z, step {self.step_mz:.4f} m/z ({self.source})"
        )


def _floats(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("a Q1 model value is not finite")
    return arr


def model_from_block(block: dict[str, Any], source: str) -> Q1Model:
    """The model of a ``params.q1_fragments`` block of the psms_extracted report, or of a
    ``q1_calibration`` block of ``cal.json``. Raises :class:`ViewerError` with the reason
    when the block is not a fitted window."""
    if not isinstance(block, dict):
        raise ViewerError(f"{source}: no Q1 model")
    if "status" in block and block.get("status") != "fitted":
        raise ViewerError(
            f"{source}: the Q1 calibration has status {block.get('status')!r}, so there is no "
            "calibrated Q1 window"
        )
    try:
        step = block.get("step_mz")
        if step is None:
            step = (block.get("sweep") or {}).get("step_mz")
        step = float(step)
        h = float(block["half_width_mz"])
        e = float(block["edge_width_mz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError(f"{source}: the Q1 model lacks a value ({exc})") from None
    if not (step > 0 and h > 0 and e >= 0) or not all(map(math.isfinite, (step, h, e))):
        raise ViewerError(f"{source}: step {step}, half-width {h} and edge {e} are not a window")
    terms: list[tuple[str, np.ndarray, np.ndarray]] = []
    raw = block.get("offset_terms")
    try:
        if isinstance(raw, list):
            for t in raw:
                terms.append((str(t["var"]), _floats(t["grid"]), _floats(t["offset_mz"])))
        else:
            named = ((block.get("offset") or {}).get("terms")) or {}
            for var in ("precursor_mz", "fragment_mz", "rt"):
                if var in named:
                    t = named[var]
                    terms.append((var, _floats(t["grid"]), _floats(t["offset_mz"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ViewerError(f"{source}: a Q1 offset term is malformed ({exc})") from None
    for var, grid, off in terms:
        if grid.size != off.size or (grid.size > 1 and bool(np.any(np.diff(grid) < 0))):
            raise ViewerError(f"{source}: the Q1 offset term {var} has a malformed grid")
    tol = block.get("tolerances")
    if tol is None:
        tol = (block.get("fragment_decision") or {}).get("tolerances")
    tolerances = None
    if isinstance(tol, dict):
        try:
            tolerances = {k: float(tol[k]) for k in ("offset_steps", "centre_steps", "corr2d_min")}
        except (KeyError, TypeError, ValueError):
            tolerances = None
    decision = block.get("fragment_decision") or {}

    def _int(key: str, default: int) -> int:
        for value in (block.get(key), decision.get(key)):
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
        return default

    margin = block.get("margin_steps", decision.get("margin_steps", MARGIN_STEPS))
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = MARGIN_STEPS
    return Q1Model(
        step_mz=step,
        half_width_mz=h,
        edge_width_mz=e,
        offset_terms=tuple(terms),
        tolerances=tolerances,
        source=source,
        margin_steps=margin,
        patch_cycles=_int("patch_cycles", 2 * PATCH_CYCLES + 1),
        profile_cycles=_int("profile_cycles", 2 * PROFILE_CYCLES + 1),
    )


@dataclass(frozen=True, eq=False)
class FragQ1:
    """One ion as the Q1 model reads it: the m/z matched in the spectra (library m/z times
    the mass-offset factor), its isotope weights and mean isotope index, and its expected
    Q1 centre in target units (m/z of the step targets)."""

    name: str
    mz: float
    target_mz: float
    weights: np.ndarray
    ibar: float
    centre: float


class CandQ1:
    """A candidate as the Q1 model reads it (``q1frag::CandQ1``): the precursor-level
    expected centre and one :class:`FragQ1` per fragment."""

    def __init__(
        self,
        model: Q1Model,
        pm: float,
        z: int,
        rt: float,
        names: list[str],
        frag_mz: list[float] | np.ndarray,
        *,
        factor=None,
    ) -> None:
        z = max(int(z), 1)
        self.model = model
        self.pm = float(pm)
        self.z = z
        self.sp = ISOTOPE_SPACING / z
        pre = averagine_isotopes(_neutral(pm, z), ISOTOPES)
        self.pre = pre
        ipre = float(np.dot(np.arange(ISOTOPES), pre))
        self.centre = self.pm + self.sp * ipre - model.offset_mz(pm, None, rt)
        frags = []
        for name, fmz in zip(names, frag_mz, strict=True):
            fmz = float(fmz)
            w = isotope_weights(pm, z, fmz, fragment_charge(name))
            ibar = float(np.dot(np.arange(w.size), w))
            f = 1.0 if factor is None else float(factor(fmz))
            frags.append(
                FragQ1(
                    name=name,
                    mz=fmz,
                    target_mz=fmz * f,
                    weights=w,
                    ibar=ibar,
                    centre=self.pm + self.sp * ibar - model.offset_mz(pm, fmz, rt),
                )
            )
        self.frags: tuple[FragQ1, ...] = tuple(frags)

    @classmethod
    def precursor(
        cls, model: Q1Model, pm: float, z: int, rt: float, n_iso: int = PREC_ISOTOPES, factor=None
    ) -> CandQ1:
        """The unfragmented precursor as the decision reads it (``CandQ1::precursor``): its
        first ``n_iso`` isotopes as ions of their own, each matched at ``pm + k spacing``
        and transmitted at its own Q1 position (one window centred on it), with its
        Poisson-averagine abundance."""
        self = cls(model, pm, z, rt, [], [])
        frags = []
        for k in range(min(int(n_iso), ISOTOPES)):
            mz = self.pm + self.sp * k
            w = np.zeros(ISOTOPES)
            w[k] = 1.0
            f = 1.0 if factor is None else float(factor(mz))
            frags.append(
                FragQ1(
                    name="M0" if k == 0 else f"M+{k}",
                    mz=mz,
                    target_mz=mz * f,
                    weights=w,
                    ibar=float(k),
                    centre=mz - model.offset_mz(pm, mz, rt),
                )
            )
        self.frags = tuple(frags)
        return self

    def expected(
        self, f: int, t: np.ndarray | float, mu: float = 0.0, omega: float = 1.0
    ) -> np.ndarray:
        """The expected profile of ion ``f`` at step targets ``t`` (m/z), shifted by ``mu``
        steps and scaled in width by ``omega`` (``CandQ1::expected``)."""
        fr = self.frags[f]
        m = self.model
        u = np.asarray(t, dtype=np.float64) - fr.centre - mu * m.step_mz
        h, e = omega * m.half_width_mz, omega * m.edge_width_mz
        out = np.zeros(u.shape, dtype=np.float64)
        for i, wi in enumerate(fr.weights):
            if wi > 0:
                out += wi * transmission(self.sp * (i - fr.ibar) - u, h, e)
        return out

    def envelope(self, t: np.ndarray, mu: float = 0.0, omega: float = 1.0) -> np.ndarray:
        """The precursor's whole isotope envelope through the window, at step targets ``t``:
        ``sum_i pre_i transmission(spacing (i - mean isotope) - u)`` about :attr:`centre`.
        Each fragment's profile is this envelope reweighted by its own isotope weights."""
        m = self.model
        ipre = float(np.dot(np.arange(self.pre.size), self.pre))
        u = np.asarray(t, dtype=np.float64) - self.centre - mu * m.step_mz
        h, e = omega * m.half_width_mz, omega * m.edge_width_mz
        out = np.zeros(u.shape, dtype=np.float64)
        for i, wi in enumerate(self.pre):
            out += wi * transmission(self.sp * (i - ipre) - u, h, e)
        return out


# --------------------------------------------------------------------------- the sweep


class ZtSweep:
    """The Q1 sweep of a ``spectra_ms2`` v3 table: per row its cycle and step, the scan of
    every (cycle, step) cell, the target of every step and the scans' matching tolerance.

    Rows are the rows of the table (the rows of :class:`.spectra.ScanTable`). ``cell(c, s)``
    is the row of step ``s`` in cycle ``c`` (-1 where the cycle has none). A step is the rank
    of its target in the sweep (the engine refuses a sweep where it is not), so
    ``step_target`` ascends; ``notes`` says when the table breaks that.
    """

    def __init__(self, scans: ScanTable, cycle: np.ndarray, step: np.ndarray) -> None:
        if cycle.size != scans.n or step.size != scans.n:
            raise ViewerError("the Q1 sweep columns and the MS2 table differ in length")
        # int32 (a run has far fewer than 2^31 spectra): 4 bytes per spectrum each.
        self.cycle = np.ascontiguousarray(cycle, dtype=np.int32)
        self.step = np.ascontiguousarray(step, dtype=np.int32)
        self.notes: list[str] = []
        n = scans.n
        self.n_cycles = int(self.cycle.max()) + 1 if n else 0
        self.n_steps = int(self.step.max()) + 1 if n else 0
        key = self.cycle.astype(np.int64) * max(self.n_steps, 1) + self.step
        cells = np.full(self.n_cycles * self.n_steps, -1, dtype=np.int32)
        rows = np.arange(n, dtype=np.int32)
        # The first row of a cell wins: assign in reverse.
        cells[key[::-1]] = rows[::-1]
        dup = n - int(np.count_nonzero(cells >= 0))
        if dup:
            self.notes.append(f"{dup} spectra repeat a (cycle, step) cell; the first is used")
        self._cells = cells.reshape(self.n_cycles, self.n_steps)
        first = np.full(self.n_steps, -1, dtype=np.int64)
        uniq, idx = np.unique(self.step, return_index=True)
        first[uniq] = idx
        self.step_target = np.where(first >= 0, scans.target[np.maximum(first, 0)], np.nan)
        fin = self.step_target[np.isfinite(self.step_target)]
        if fin.size > 1 and bool(np.any(np.diff(fin) <= 0)):
            self.notes.append("the recorded steps are not the ranks of their targets")
        diffs = np.diff(fin) if fin.size > 1 else np.zeros(0)
        self.step_mz = float(np.median(diffs)) if diffs.size else float("nan")
        rt = np.sort(scans.rt)
        gaps = np.diff(rt)
        gaps = gaps[gaps > 0]
        self.tol = 0.45 * float(np.median(gaps)) if gaps.size else 0.0
        # Cycle duration: the median spacing of the first step's scans.
        c0 = self._cells[:, 0] if self.n_steps else np.zeros(0, dtype=np.int64)
        c0 = c0[c0 >= 0]
        self.cycle_s = float(np.median(np.diff(scans.rt[c0]))) if c0.size > 1 else float("nan")

    def cell(self, cycles: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Rows of the cells ``cycles x steps`` (shape ``(len(cycles), len(steps))``), -1
        outside the sweep or where a cycle lacks the step."""
        c = np.asarray(cycles, dtype=np.int64)
        s = np.asarray(steps, dtype=np.int64)
        out = np.full((c.size, s.size), -1, dtype=np.int64)
        okc = (c >= 0) & (c < self.n_cycles)
        oks = (s >= 0) & (s < self.n_steps)
        if okc.any() and oks.any():
            sub = self._cells[np.ix_(c[okc], s[oks])]
            out[np.ix_(np.flatnonzero(okc), np.flatnonzero(oks))] = sub
        return out

    def steps_within(self, x: float, reach: float) -> np.ndarray:
        """The steps whose target lies within ``reach`` of ``x`` (m/z), ascending."""
        t = self.step_target
        return np.flatnonzero(np.isfinite(t) & (t >= x - reach) & (t <= x + reach))

    def nearest_step(self, x: float) -> int:
        """The step whose target is nearest ``x`` (the lower on a tie)."""
        t = np.where(np.isfinite(self.step_target), self.step_target, np.inf)
        d = np.abs(t - x)
        return int(np.argmin(d))


@dataclass(frozen=True, eq=False)
class ZtRun:
    """A ZT Scan run: its MS2 table, its sweep and its Q1 model (None when no model is
    recorded; ``model_note`` says why)."""

    run: Run
    scans: ScanTable
    sweep: ZtSweep
    model: Q1Model | None
    model_note: str | None
    half_width: float | None
    instrument: str | None

    @property
    def method_label(self) -> str:
        parts = [
            f"{self.sweep.n_steps} Q1 steps",
            f"{self.sweep.step_mz:.4f} m/z apart" if math.isfinite(self.sweep.step_mz) else "",
            f"{self.sweep.n_cycles} cycles",
            f"{self.sweep.cycle_s:.3f} s a cycle" if math.isfinite(self.sweep.cycle_s) else "",
        ]
        if self.half_width is not None:
            parts.append(f"each step read as +-{self.half_width:g} m/z")
        return ", ".join(p for p in parts if p)


def _report_params(run: Run, kind: str) -> dict[str, Any]:
    a = run.artifact(kind)
    if a is None or a.report is None:
        return {}
    return a.report.params or {}


def _load_model(run: Run) -> tuple[Q1Model | None, str | None]:
    """The Q1 model of a run: the extract report's ``params.q1_fragments``, else
    ``cal.json``'s ``q1_calibration``."""
    why: list[str] = []
    block = _report_params(run, "psms_extracted").get("q1_fragments")
    if isinstance(block, dict):
        try:
            return (
                model_from_block(block, "psms_extracted report, params.q1_fragments"),
                None,
            )
        except ViewerError as exc:
            why.append(str(exc))
    cal_path = run.side_files.get("cal")
    cal = load_json(cal_path) if cal_path is not None else None
    if cal is not None and isinstance(cal.get("q1_calibration"), dict):
        try:
            return model_from_block(cal["q1_calibration"], "cal.json, q1_calibration"), None
        except ViewerError as exc:
            why.append(str(exc))
    elif cal_path is None:
        why.append("the run has no cal.json")
    else:
        why.append("cal.json records no q1_calibration (rt_im_train.q1_calibrate was off)")
    return None, "; ".join(why) or "no Q1 model recorded"


def ztscan_run(rs: ResultSet, run: Run | str | int) -> ZtRun | None:
    """The ZT Scan view of ``run``; None for a run that is not a ZT Scan run read with its
    sweep (any other run). Loaded once per spectra file version and kept on ``rs``."""
    run = rs.run(run)
    if not run.is_ztscan:
        return None
    art = run.artifact("spectra_ms2")
    assert art is not None
    handle = art.parquet()
    stamp = handle.stamp
    key = ("ztscan.ZtRun", run.index, run.name)
    with _MEMO_LOCK:
        hit = rs._memo.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    scans = ScanTable.for_run(rs, run)
    t = handle.read(columns=["q1_cycle", "q1_step"])
    sweep = ZtSweep(scans, t.column("q1_cycle").to_numpy(), t.column("q1_step").to_numpy())
    del t
    model, why = _load_model(run)
    params = art.report.params if art.report is not None else {}
    hw = params.get("isolation_half_width")
    try:
        half_width = float(hw) if hw is not None else None
    except (TypeError, ValueError):
        half_width = None
    inst = params.get("instrument")
    instrument = inst.get("name") if isinstance(inst, dict) else None
    zr = ZtRun(
        run=run,
        scans=scans,
        sweep=sweep,
        model=model,
        model_note=why,
        half_width=half_width,
        instrument=instrument,
    )
    with _MEMO_LOCK:
        rs._memo[key] = (stamp, zr)
    return zr


# --------------------------------------------------------------------------- the q1f row


Q1F_LISTS = ("q1f_frag_class", "q1f_frag_offset", "q1f_frag_resid", "q1f_frag_corr2d")


_ROWS: OrderedDict[tuple, dict[str, Any] | None] = OrderedDict()
_ROWS_MAX = 256
# The Q1 columns the page shows: the decision's summaries and lists, and the separable
# model's (psms_extracted v9). The other Q1 feature families stay in the file.
_ROW_PREFIXES = ("q1f_", "q1s_")


def q1f_row(rs: ResultSet, d: PrecursorDetail) -> dict[str, Any] | None:
    """The ``psms_extracted`` row of the candidate's selected peak with its Q1 columns (the
    ``q1f_*`` summaries and per-fragment lists, which are in library order, the
    chromatogram's fragment row order, and the ``q1s_*`` separable model when present).
    None when the run has no readable ``psms_extracted`` or the row is absent.

    Read with DuckDB filtered on ``candidate_id``: a ZT Scan ``psms_extracted`` holds row
    groups of a million rows with three list columns, which a row-group read decodes whole.
    """
    run = d.run
    art = run.artifact("psms_extracted")
    if art is None or not art.usable:
        return None
    key = (id(rs), run.index, int(d.candidate_id), int(d.selected_peak_rank))
    with _MEMO_LOCK:
        if key in _ROWS:
            _ROWS.move_to_end(key)
            return _ROWS[key]
    names = art.parquet().schema.names
    wanted = ["candidate_id", "peak_rank", "precursor_mz", "charge", "apex_rt", "rt_lo", "rt_hi"]
    wanted += [n for n in names if n.startswith(_ROW_PREFIXES)]
    cols = [c for c in wanted if c in names]
    table = rs.duck.arrow(
        "SELECT "
        + ", ".join(sql_ident(c) for c in cols)
        + " FROM read_parquet($p) WHERE candidate_id = $c",
        {"p": sql_path(art.require()), "c": int(d.candidate_id)},
    )
    row = None
    if table.num_rows:
        rows = table.to_pylist()
        row = next(
            (r for r in rows if int(r.get("peak_rank") or 0) == int(d.selected_peak_rank)),
            rows[0],
        )
    with _MEMO_LOCK:
        _ROWS[key] = row
        while len(_ROWS) > _ROWS_MAX:
            _ROWS.popitem(last=False)
    return row


def _classes(row: dict[str, Any] | None, n: int) -> np.ndarray | None:
    if row is None or row.get("q1f_frag_class") is None:
        return None
    cls = np.asarray(row["q1f_frag_class"], dtype=np.float64)
    if cls.size != n:
        return None
    return cls.astype(np.int64)


def _list(row: dict[str, Any] | None, key: str, n: int) -> np.ndarray | None:
    if row is None or row.get(key) is None:
        return None
    v = np.asarray([np.nan if x is None else x for x in row[key]], dtype=np.float64)
    return v if v.size == n else None


# --------------------------------------------------------------------------- cycle traces


@dataclass(frozen=True, eq=False)
class CycleTrace:
    """One trace collapsed to one value per Q1 cycle.

    ``index`` is the fragment's index among the chromatogram's fragment rows (-1 for an MS1
    row). ``rt`` is the mean RT of the cycle's samples (float32, as the engine), ``value``
    the collapsed intensity, ``cycles`` the cycle numbers, ``n`` the samples of each cycle.
    ``weights`` are the per-sample weights on the trace's axis (NaN where a sample is not
    in the sweep). ``q1_class`` is the engine's ``q1f_frag_class`` (None when not recorded).
    """

    name: str
    index: int
    kind: str
    rt: np.ndarray
    value: np.ndarray
    cycles: np.ndarray
    n: np.ndarray
    weights: np.ndarray
    q1_class: int | None = None


@dataclass(frozen=True, eq=False)
class CycleTraces:
    """The cycle traces of a candidate, with what they were computed on."""

    fragments: tuple[CycleTrace, ...]
    ms1: tuple[CycleTrace, ...]
    sample_cycle: np.ndarray
    sample_step: np.ndarray
    sample_target: np.ndarray
    collapsed: bool
    label: str
    source: str
    model: Q1Model | None
    centre_mz: float | None
    apex_cycle: int | None
    row: dict[str, Any] | None
    notes: tuple[str, ...] = ()
    cycle_s: float = float("nan")
    #: The engine's cycle traces came from ``chromatograms_q1.parquet``, so the candidate's
    #: ``chromatograms.parquet`` rows still hold the per-step samples.
    steps_in_chromatogram: bool = False

    @property
    def derived(self) -> bool:
        """True when the viewer collapsed the steps to cycles; False for the engine's own
        cycle traces."""
        return self.source == "viewer"

    @property
    def has_steps(self) -> bool:
        """True when the candidate's chromatogram (``chromatograms.parquet``) holds one
        sample per Q1 step, which is what quant integrates and what the XIC can show behind
        the cycles; False only for a development build of the engine that wrote the cycle
        traces into ``chromatograms.parquet`` itself, whose steps are in the spectra only."""
        return self.source == "viewer" or self.steps_in_chromatogram

    def fragment(self, index: int) -> CycleTrace | None:
        return next((t for t in self.fragments if t.index == index), None)

    @property
    def classes(self) -> dict[int, int]:
        """Fragment index to its Q1 class, for the fragments with a recorded class."""
        return {t.index: t.q1_class for t in self.fragments if t.q1_class is not None}


def _locate(rt_cov: np.ndarray, rows_cov: np.ndarray, x: np.ndarray, tol: float) -> np.ndarray:
    """For each axis value ``x`` the covering row nearest in RT within ``tol`` (ties to the
    earlier), -1 where none (``SweepScans::locate``)."""
    out = np.full(x.size, -1, dtype=np.int64)
    if rt_cov.size == 0 or x.size == 0:
        return out
    j = np.searchsorted(rt_cov, x, side="left")
    lo = np.clip(j - 1, 0, rt_cov.size - 1)
    hi = np.clip(j, 0, rt_cov.size - 1)
    dlo = np.abs(rt_cov[lo] - x)
    dhi = np.abs(rt_cov[hi] - x)
    pick = np.where(dlo <= dhi, lo, hi)
    d = np.minimum(dlo, dhi)
    ok = d <= tol
    out[ok] = rows_cov[pick[ok]]
    return out


def _collapse(
    cyc: np.ndarray, rt: np.ndarray, it: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive samples of one cycle to ``sum(w x) / sum(w^2)`` (0 where every weight
    is 0) at the float32 mean RT of the samples (``q1quant::collapse_trace``)."""
    if cyc.size == 0:
        e = np.zeros(0)
        return e, e, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, cyc[1:] != cyc[:-1]])
    ends = np.r_[starts[1:], cyc.size]
    rt64 = rt.astype(np.float64)
    x = it.astype(np.float64)
    swx = np.add.reduceat(w * x, starts)
    sww = np.add.reduceat(w * w, starts)
    srt = np.add.reduceat(rt64, starts)
    n = ends - starts
    value = np.where(sww > 0, swx / np.where(sww > 0, sww, 1.0), 0.0)
    mean_rt = (srt / n).astype(np.float32).astype(np.float64)
    return mean_rt, value, cyc[starts].astype(np.int64), n.astype(np.int64)


def _row_values(d: PrecursorDetail, row: dict[str, Any] | None) -> tuple[float, int, float]:
    """Precursor m/z, charge and apex RT of the candidate's selected peak."""
    peak = d.selected_peak or {}
    pm = (row or {}).get("precursor_mz", peak.get("precursor_mz", d.precursor_mz))
    z = (row or {}).get("charge", peak.get("charge", d.scored.get("charge")))
    rt = (row or {}).get("apex_rt", peak.get("apex_rt", d.scored.get("apex_rt")))
    if pm is None or z is None or rt is None:
        raise ViewerError("the candidate's precursor m/z, charge or apex is unknown")
    return float(pm), int(z), float(rt)


def candidate_model(
    zr: ZtRun, d: PrecursorDetail, row: dict[str, Any] | None, tol: Tolerance | None = None
) -> CandQ1:
    """The candidate's fragments (the chromatogram's fragment rows, library order) as the Q1
    model reads them; with ``tol`` the matched m/z carry the extraction's mass offset."""
    if zr.model is None:
        raise ViewerError(f"no Q1 model for this run: {zr.model_note}")
    if d.chromatogram is None:
        raise ViewerError("no chromatogram rows for this candidate, so its fragments are unknown")
    pm, z, rt = _row_values(d, row)
    frags = d.chromatogram.fragments()
    mzs = [float(np.float32(t.frag_mz)) for t in frags]
    factor = None if tol is None else (lambda mz: 1.0 + float(tol.ppm_at(mz)) * 1e-6)
    return CandQ1(zr.model, pm, z, rt, [t.frag_name for t in frags], mzs, factor=factor)


def cycle_traces(rs: ResultSet, d: PrecursorDetail) -> CycleTraces:
    """The candidate's fragment and MS1 traces collapsed to one value per Q1 cycle (module
    docstring). Raises :class:`ViewerError` on a run that is not a ZT Scan run or a
    candidate without chromatogram rows."""
    zr = ztscan_run(rs, d.run)
    if zr is None:
        raise ViewerError(f"{d.run.label} is not a ZT Scan run read with its Q1 sweep")
    chrom = d.chromatogram
    if chrom is None:
        raise ViewerError("no chromatogram rows for this candidate")
    row = q1f_row(rs, d)
    if chrom.layout.q1_traces:
        # A development build of the engine wrote the cycle traces into chromatograms.parquet.
        return _engine_cycle_traces(zr, d, row, chrom, steps_in_chromatogram=False)
    q1t, why = engine_cycle_chromatogram(rs, d)
    if q1t is not None:
        return _engine_cycle_traces(zr, d, row, q1t, steps_in_chromatogram=True)
    pm, _, _ = _row_values(d, row)
    notes: list[str] = list(zr.sweep.notes)
    if why:
        notes.append(why)
    cand: CandQ1 | None = None
    if zr.model is not None:
        cand = candidate_model(zr, d, row)
    else:
        notes.append(f"no Q1 model ({zr.model_note}): the steps are averaged, not weighted")
    scans = zr.scans
    rows_cov = scans.grid_rows(pm, -math.inf, math.inf)
    rt_cov = scans.rt[rows_cov]
    frags = chrom.fragments()
    classes = _classes(row, len(frags))
    if row is not None and row.get("q1f_frag_class") is not None and classes is None:
        notes.append(
            "q1f_frag_class does not have one entry per chromatogram fragment row; the "
            "fragment classes are not shown"
        )
    axis = chrom.common_axis()
    collapsed = True
    reason = ""

    def locate(rt32: np.ndarray) -> tuple[np.ndarray, bool]:
        r = _locate(rt_cov, rows_cov, rt32.astype(np.float64), zr.sweep.tol)
        ok = bool(r.size) and bool(np.all(r >= 0))
        if ok:
            cyc = zr.sweep.cycle[r]
            ok = bool(np.all(np.diff(cyc) >= 0))
        return r, ok

    if axis is not None:
        shared_rows, shared_ok = locate(np.asarray(axis))
    else:
        shared_rows, shared_ok = np.zeros(0, dtype=np.int64), False

    out_f: list[CycleTrace] = []
    for i, t in enumerate(frags):
        r, ok = (shared_rows, shared_ok) if axis is not None else locate(np.asarray(t.rt))
        q = int(classes[i]) if classes is not None else None
        if not t.observed:
            e = np.zeros(0)
            out_f.append(
                CycleTrace(
                    t.frag_name, i, "fragment", e, e, e.astype(np.int64), e.astype(np.int64), e, q
                )
            )
            continue
        if not ok:
            collapsed = False
            reason = (
                "a sample of the trace is not a spectrum of the sweep, or its cycles do not ascend"
            )
            w = np.full(t.rt.size, np.nan)
            out_f.append(
                CycleTrace(
                    t.frag_name,
                    i,
                    "fragment",
                    t.rt.astype(np.float64),
                    t.intensity.astype(np.float64),
                    np.zeros(t.rt.size, dtype=np.int64),
                    np.ones(t.rt.size, dtype=np.int64),
                    w,
                    q,
                )
            )
            continue
        target = scans.target[r]
        w = cand.expected(i, target) if cand is not None else np.ones(r.size)
        mrt, val, cyc, n = _collapse(
            zr.sweep.cycle[r], np.asarray(t.rt), np.asarray(t.intensity), w
        )
        out_f.append(CycleTrace(t.frag_name, i, "fragment", mrt, val, cyc, n, w, q))

    out_m: list[CycleTrace] = []
    for t in chrom.ms1():
        if not t.observed:
            continue
        r, ok = (shared_rows, shared_ok) if axis is not None else locate(np.asarray(t.rt))
        if not ok:
            continue
        w = np.ones(r.size)
        mrt, val, cyc, n = _collapse(
            zr.sweep.cycle[r], np.asarray(t.rt), np.asarray(t.intensity), w
        )
        out_m.append(CycleTrace(t.frag_name, -1, "ms1", mrt, val, cyc, n, w))

    if axis is not None and shared_ok:
        s_cyc = zr.sweep.cycle[shared_rows]
        s_step = zr.sweep.step[shared_rows]
        s_tgt = scans.target[shared_rows]
    else:
        s_cyc = s_step = np.zeros(0, dtype=np.int64)
        s_tgt = np.zeros(0)
    apex_cycle = None
    if d.apex_scan is not None:
        apex_cycle = int(zr.sweep.cycle[d.apex_scan.row])
    label = CYCLE_LABEL if cand is not None else UNIFORM_LABEL
    if not collapsed:
        notes.append(f"shown per step: {reason}")
    return CycleTraces(
        fragments=tuple(out_f),
        ms1=tuple(out_m),
        sample_cycle=s_cyc,
        sample_step=s_step,
        sample_target=s_tgt,
        collapsed=collapsed,
        label=label,
        source="viewer",
        model=zr.model,
        centre_mz=cand.centre if cand is not None else None,
        apex_cycle=apex_cycle,
        row=row,
        notes=tuple(notes),
        cycle_s=zr.sweep.cycle_s,
    )


def engine_cycle_chromatogram(
    rs: ResultSet, d: PrecursorDetail
) -> tuple[CandidateChromatogram | None, str | None]:
    """The candidate's rows of the run's cycle-trace table (``chromatograms_q1.parquet``,
    :class:`~mumdia_viewer.data.chromatograms.Q1TraceSource`) and, when they cannot be used,
    why (None when the run has no such table, which is the usual case).

    The rows are used only when their fragment rows are those of the candidate's
    ``chromatograms.parquet`` rows, in the same order (the engine writes both from the same
    library rows); otherwise the viewer derives the cycle traces from the per-step rows.
    """
    try:
        source = Q1TraceSource.for_run(rs, d.run)
    except ViewerError as exc:
        return None, f"the cycle-trace table is not used: {exc}"
    art = d.run.artifacts.get("chromatograms_q1")
    if source is None:
        if art is not None and art.error:
            return None, f"the cycle-trace table is not used: {art.error}"
        return None, None
    try:
        q1t = source.read(d.candidate_id)
    except ViewerError as exc:
        return None, f"the cycle-trace table is not used: {exc}"
    if q1t is None:
        return None, (
            "chromatograms_q1.parquet has no rows for this candidate (the engine found no Q1 "
            "cycles for it); the cycle traces are derived from the per-step traces"
        )
    chrom = d.chromatogram
    if chrom is not None:
        names = [t.frag_name for t in chrom.traces]
        if [t.frag_name for t in q1t.traces] != names:
            return None, (
                "the rows of chromatograms_q1.parquet are not the candidate's chromatogram rows; "
                "the cycle traces are derived from the per-step traces"
            )
    return q1t, None


def _engine_cycle_traces(
    zr: ZtRun,
    d: PrecursorDetail,
    row: dict[str, Any] | None,
    chrom: CandidateChromatogram,
    *,
    steps_in_chromatogram: bool,
) -> CycleTraces:
    """The engine's cycle-level traces (``chromatograms_q1.parquet``, schema 5/6) as
    :class:`CycleTraces`: each point is already one Q1 cycle; its cycle is the cycle of the
    spectrum at its RT."""
    frags = chrom.fragments()
    classes = _classes(row, len(frags))
    order = np.argsort(zr.scans.rt, kind="stable")
    rt_sorted = zr.scans.rt[order]

    def cycles_of(rt32: np.ndarray) -> np.ndarray:
        r = _locate(rt_sorted, order, np.asarray(rt32, dtype=np.float64), zr.sweep.tol)
        return np.where(r >= 0, zr.sweep.cycle[np.maximum(r, 0)], -1).astype(np.int64)

    def trace(t: Any, i: int, kind: str, q: int | None) -> CycleTrace:
        rt = np.asarray(t.rt, dtype=np.float64)
        return CycleTrace(
            t.frag_name,
            i,
            kind,
            rt,
            np.asarray(t.intensity, dtype=np.float64),
            cycles_of(t.rt) if rt.size else np.zeros(0, dtype=np.int64),
            np.zeros(rt.size, dtype=np.int64),
            np.zeros(0),
            q,
        )

    out_f = []
    for i, t in enumerate(frags):
        q = t.q1_class
        if q is None and classes is not None:
            q = int(classes[i])
        out_f.append(trace(t, i, "fragment", q))
    out_m = [trace(t, -1, "ms1", None) for t in chrom.ms1() if t.observed]
    apex_cycle = int(zr.sweep.cycle[d.apex_scan.row]) if d.apex_scan is not None else None
    centre = None
    if zr.model is not None:
        try:
            centre = candidate_model(zr, d, row).centre
        except ViewerError:
            centre = None
    e = np.zeros(0, dtype=np.int64)
    return CycleTraces(
        fragments=tuple(out_f),
        ms1=tuple(out_m),
        sample_cycle=e,
        sample_step=e,
        sample_target=np.zeros(0),
        collapsed=True,
        label=ENGINE_CYCLE_LABEL,
        source="engine",
        model=zr.model,
        centre_mz=centre,
        apex_cycle=apex_cycle,
        row=row,
        notes=tuple(zr.sweep.notes),
        cycle_s=zr.sweep.cycle_s,
        steps_in_chromatogram=steps_in_chromatogram,
    )


# --------------------------------------------------------------------------- the patch


class _CellCache:
    """Decoded peak lists of single spectra rows, least recently used out, by bytes."""

    def __init__(self, max_bytes: int = CELL_CACHE_BYTES) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
            return hit

    def put(self, key: tuple, value: tuple[np.ndarray, np.ndarray]) -> None:
        size = value[0].nbytes + value[1].nbytes
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[0].nbytes + old[1].nbytes
            self._items[key] = value
            self._bytes += size
            while self._items and self._bytes > self.max_bytes:
                _, ev = self._items.popitem(last=False)
                self._bytes -= ev[0].nbytes + ev[1].nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


CELL_CACHE = _CellCache()


@dataclass
class ReadStats:
    cells: int = 0
    cached: int = 0
    row_groups: int = 0
    ms: float = 0.0


def read_cells(zr: ZtRun, rows: np.ndarray, stats: ReadStats | None = None) -> dict[int, tuple]:
    """The peak lists (float32 m/z ascending, intensity) of ``rows`` of ``spectra_ms2``,
    reading only the row groups that hold rows not in :data:`CELL_CACHE`."""
    t0 = time.perf_counter()
    stats = stats if stats is not None else ReadStats()
    art = zr.run.artifact("spectra_ms2")
    assert art is not None
    handle = art.parquet()
    stamp = handle.stamp
    path = str(handle.path)
    out: dict[int, tuple] = {}
    missing: list[int] = []
    for r in np.unique(rows[rows >= 0]).tolist():
        hit = CELL_CACHE.get((path, stamp, r))
        if hit is not None:
            out[r] = hit
            stats.cached += 1
        else:
            missing.append(r)
    if missing:
        offsets = handle.row_group_offsets()
        miss = np.asarray(missing, dtype=np.int64)
        rgs = np.searchsorted(offsets, miss, side="right") - 1

        def read(rg: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
            """The wanted rows of one row group, copied out of it."""
            local = miss[rgs == rg] - int(offsets[rg])
            table = handle.read_row_group(int(rg), ["mz", "intensity"], cached=False)
            mzv, mzo = list_values(table.column("mz"))
            itv, ito = list_values(table.column("intensity"))
            got = {}
            for lr in local.tolist():
                got[int(offsets[rg]) + int(lr)] = (
                    np.array(mzv[mzo[lr] : mzo[lr + 1]], dtype=np.float32),
                    np.array(itv[ito[lr] : ito[lr + 1]], dtype=np.float32),
                )
            return got

        groups = np.unique(rgs).tolist()
        # The row groups are read in parallel: on a busy disk the reads overlap.
        with ThreadPoolExecutor(max_workers=max(1, min(READ_THREADS, len(groups)))) as pool:
            for got in pool.map(read, groups):
                for g, value in got.items():
                    out[g] = value
                    CELL_CACHE.put((path, stamp, g), value)
        stats.row_groups += len(groups)
    stats.cells += len(out)
    stats.ms += (time.perf_counter() - t0) * 1000.0
    return out


def nearest_peaks(
    mz: np.ndarray, intensity: np.ndarray, targets: np.ndarray, tol_ppm: float
) -> np.ndarray:
    """For each target the intensity of the peak nearest it within ``tol_ppm`` (relative
    to the target; ties to the lower index), 0 where none (``spectra::nearest_peak``)."""
    out = np.zeros(targets.size, dtype=np.float64)
    if mz.size == 0:
        return out
    mz64 = mz.astype(np.float64)
    d = targets * tol_ppm * 1e-6
    a = np.searchsorted(mz64, targets - d, side="left")
    b = np.searchsorted(mz64, targets + d, side="right")
    for i in np.flatnonzero(b > a).tolist():
        seg = np.abs(mz64[a[i] : b[i]] - targets[i])
        out[i] = float(intensity[a[i] + int(np.argmin(seg))])
    return out


@dataclass(frozen=True, eq=False)
class Q1Patch:
    """A candidate's RT x Q1 patch: ``values[f, c, s]`` is fragment ``f`` in cycle
    ``cycles[c]`` at step ``steps[s]`` (NaN where the cycle has no spectrum of the step);
    ``precursor[k, c, s]`` the unfragmented precursor's isotope ``k`` in the same cells.

    ``cycle_rt`` is the RT of each cycle's spectrum at the step nearest the expected
    centre; ``targets`` the m/z of each step's target. ``expected_centre`` (per fragment)
    and ``centre_mz`` (the precursor) are the calibrated expected Q1 centres in target
    units; ``fitted_centre_steps`` and ``width`` the engine's shared fit (``q1f_centre``,
    ``q1f_width``), ``frag_offset_steps`` each fragment's own centre (``q1f_frag_offset``),
    all in steps against the expected centre. ``apex_cycle`` is the cycle of the apex scan
    and ``apex2d_cycle`` the 2D apex of the decision (``apex_cycle + q1f_apex_shift_cycles``).
    ``in_window`` marks the cycles whose centre spectrum lies inside the candidate's RT
    window (the cycles the decision reads).
    """

    candidate_id: int
    cycles: np.ndarray
    cycle_rt: np.ndarray
    steps: np.ndarray
    targets: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray
    precursor_names: tuple[str, ...]
    precursor_mz: np.ndarray
    precursor: np.ndarray
    expected_centre: np.ndarray
    precursor_centre: np.ndarray
    centre_mz: float
    step_mz: float
    apex_cycle: int
    apex2d_cycle: int | None
    fitted_centre_steps: float | None
    width: float | None
    frag_offset_steps: np.ndarray | None
    classes: np.ndarray | None
    in_window: np.ndarray
    cand: CandQ1 = field(repr=False)
    prec_cand: CandQ1 = field(repr=False)
    tol_ppm: float = 0.0
    read: ReadStats = field(default_factory=ReadStats)
    label: str = PATCH_LABEL
    notes: tuple[str, ...] = ()

    @property
    def n_cycles(self) -> int:
        return int(self.cycles.size)

    def cycle_index(self, cycle: int) -> int | None:
        hit = np.flatnonzero(self.cycles == int(cycle))
        return int(hit[0]) if hit.size else None

    def steps_from_centre(self) -> np.ndarray:
        """Each step's target against the precursor's expected centre, in steps."""
        return (self.targets - self.centre_mz) / self.step_mz


_PATCHES: OrderedDict[tuple, Q1Patch] = OrderedDict()
_PATCH_LOCK = threading.Lock()
_PATCH_MAX = 32
_INFLIGHT: dict[tuple, threading.Lock] = {}


def q1_patch(
    rs: ResultSet,
    d: PrecursorDetail,
    *,
    half_cycles: int = PATCH_CYCLES,
    centre_cycle: int | None = None,
) -> Q1Patch:
    """The candidate's patch: ``half_cycles`` cycles either side of ``centre_cycle`` (default
    the apex scan's cycle) by the steps within the calibrated window's reach of its
    expected centre plus the model's margin (module docstring). Kept in a small LRU."""
    zr = ztscan_run(rs, d.run)
    if zr is None:
        raise ViewerError(f"{d.run.label} is not a ZT Scan run read with its Q1 sweep")
    if zr.model is None:
        raise ViewerError(f"no Q1 model for this run: {zr.model_note}")
    if d.tolerance is None:
        raise ViewerError("the extraction's fragment tolerance is unknown")
    if d.apex_scan is None:
        raise ViewerError("the apex scan of the candidate is unknown")
    apex_cycle = int(zr.sweep.cycle[d.apex_scan.row])
    c0 = apex_cycle if centre_cycle is None else int(centre_cycle)
    k = max(int(half_cycles), 0)
    key = (id(rs), d.run.name, int(d.candidate_id), int(d.selected_peak_rank), c0, k)
    with _PATCH_LOCK:
        hit = _PATCHES.get(key)
        if hit is not None:
            _PATCHES.move_to_end(key)
            return hit
        # One read per patch: a second caller (the card's callback while the page's
        # prefetch reads) waits for the first and takes its result.
        gate = _INFLIGHT.setdefault(key, threading.Lock())
    with gate:
        with _PATCH_LOCK:
            hit = _PATCHES.get(key)
        if hit is not None:
            return hit
        try:
            patch = _read_patch(rs, d, zr, apex_cycle, c0, k)
        finally:
            with _PATCH_LOCK:
                _INFLIGHT.pop(key, None)
        with _PATCH_LOCK:
            _PATCHES[key] = patch
            while len(_PATCHES) > _PATCH_MAX:
                _PATCHES.popitem(last=False)
        return patch


def _read_patch(
    rs: ResultSet, d: PrecursorDetail, zr: ZtRun, apex_cycle: int, c0: int, k: int
) -> Q1Patch:
    """The patch of :func:`q1_patch`, read."""
    assert zr.model is not None and d.tolerance is not None
    row = q1f_row(rs, d)
    pm, z, rt = _row_values(d, row)
    tol = d.tolerance
    model = zr.model
    cand = candidate_model(zr, d, row, tol)
    factor = lambda mz: 1.0 + float(tol.ppm_at(mz)) * 1e-6  # noqa: E731
    prec = CandQ1.precursor(model, pm, z, rt, PREC_ISOTOPES, factor=factor)
    reach = model.reach_mz + model.margin_steps * model.step_mz
    steps = zr.sweep.steps_within(cand.centre, reach)
    if steps.size == 0:
        raise ViewerError(
            f"no Q1 step of the sweep lies within {reach:.2f} m/z of the expected centre "
            f"{cand.centre:.3f}"
        )
    first = max(c0 - k, 0)
    last = min(c0 + k, zr.sweep.n_cycles - 1)
    cycles = np.arange(first, last + 1, dtype=np.int64)
    rows = zr.sweep.cell(cycles, steps)
    stats = ReadStats()
    peaks = read_cells(zr, rows, stats)
    targets = np.array([f.target_mz for f in cand.frags] + [f.target_mz for f in prec.frags])
    nf, npc = len(cand.frags), len(prec.frags)
    vals = np.full((nf + npc, cycles.size, steps.size), np.nan, dtype=np.float32)
    for ci in range(cycles.size):
        for si in range(steps.size):
            r = int(rows[ci, si])
            if r < 0:
                continue
            mz, it = peaks[r]
            vals[:, ci, si] = nearest_peaks(mz, it, targets, tol.tol_ppm)
    mid = zr.sweep.nearest_step(cand.centre)
    mid_rows = zr.sweep.cell(cycles, np.array([mid]))[:, 0]
    cycle_rt = np.where(mid_rows >= 0, zr.scans.rt[np.maximum(mid_rows, 0)], np.nan)
    lo = (row or {}).get("rt_lo")
    hi = (row or {}).get("rt_hi")
    if lo is None and d.window is not None:
        lo, hi = d.window.rt_lo, d.window.rt_hi
    lo = -math.inf if lo is None else float(lo)
    hi = math.inf if hi is None else float(hi)
    in_window = np.isfinite(cycle_rt) & (cycle_rt >= lo) & (cycle_rt <= hi)
    in_window |= cycles == apex_cycle
    shift = (row or {}).get("q1f_apex_shift_cycles")
    apex2d = None
    if shift is not None and math.isfinite(float(shift)):
        apex2d = apex_cycle + round(float(shift))
    centre = (row or {}).get("q1f_centre")
    width = (row or {}).get("q1f_width")
    # The decision writes 0 and 1 (and a residual of 1) without a shared fit.
    has_fit = (
        row is not None
        and centre is not None
        and (
            float(centre) != 0.0
            or float(width or 1.0) != 1.0
            or float(row.get("q1f_fit_resid", 1.0) or 1.0) < 1.0
        )
    )
    notes = list(zr.sweep.notes)
    if k != PATCH_CYCLES:
        notes.append(f"{2 * k + 1} cycles read; the engine's decision reads {2 * PATCH_CYCLES + 1}")
    patch = Q1Patch(
        candidate_id=int(d.candidate_id),
        cycles=cycles,
        cycle_rt=cycle_rt,
        steps=steps,
        targets=zr.sweep.step_target[steps].astype(np.float64),
        names=tuple(f.name for f in cand.frags),
        values=vals[:nf],
        precursor_names=tuple(f.name for f in prec.frags),
        precursor_mz=np.array([f.mz for f in prec.frags]),
        precursor=vals[nf:],
        expected_centre=np.array([f.centre for f in cand.frags]),
        precursor_centre=np.array([f.centre for f in prec.frags]),
        centre_mz=float(cand.centre),
        step_mz=float(model.step_mz),
        apex_cycle=apex_cycle,
        apex2d_cycle=apex2d,
        fitted_centre_steps=float(centre) if has_fit and centre is not None else None,
        width=float(width) if has_fit and width is not None else None,
        frag_offset_steps=_list(row, "q1f_frag_offset", nf),
        classes=_classes(row, nf),
        in_window=in_window,
        cand=cand,
        prec_cand=prec,
        tol_ppm=float(tol.tol_ppm),
        read=stats,
        notes=tuple(notes),
    )
    return patch


# --------------------------------------------------------------------------- the Q1 profile


@dataclass(frozen=True, eq=False)
class Q1Profile:
    """Each ion's intensity across the patch's steps in the cycles ``cycles`` (summed),
    with its expected profile and its least-squares amplitude under it.

    ``observed[f]`` is the summed intensity at each step (NaN where no spectrum),
    ``expected[f]`` the calibrated profile at the expected centre (0 to 1),
    ``amplitude[f]`` = ``sum(w x) / sum(w^2)`` over the steps, so ``observed / amplitude``
    lies on ``expected`` for an ion that is where the model puts it. ``fitted[f]`` is the
    profile at the engine's shared fit (``q1f_centre``, ``q1f_width``) when recorded.
    The ``precursor_*`` arrays are the same for the unfragmented precursor's isotopes.
    """

    cycles: np.ndarray
    targets: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    fitted: np.ndarray | None
    amplitude: np.ndarray
    precursor_observed: np.ndarray
    precursor_expected: np.ndarray
    precursor_amplitude: np.ndarray
    envelope: np.ndarray
    envelope_fitted: np.ndarray | None


def _amplitude(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    xw = np.where(ok, x * w, 0.0).sum(axis=-1)
    ww = np.where(ok, w * w, 0.0).sum(axis=-1)
    return np.where(ww > 0, xw / np.where(ww > 0, ww, 1.0), 0.0)


def q1_profile(patch: Q1Patch, cycle: int | None = None, half: int = 0) -> Q1Profile:
    """The Q1 profile of ``patch`` at ``cycle`` (default: the decision's 2D apex, else the
    apex cycle), summed over ``half`` cycles either side."""
    c = cycle
    if c is None:
        c = patch.apex2d_cycle if patch.apex2d_cycle is not None else patch.apex_cycle
    ci = patch.cycle_index(c)
    if ci is None:
        ci = int(np.argmin(np.abs(patch.cycles - int(c))))
    lo, hi = max(ci - half, 0), min(ci + half, patch.n_cycles - 1)
    sel = slice(lo, hi + 1)

    def summed(v: np.ndarray) -> np.ndarray:
        block = v[:, sel, :].astype(np.float64)
        allnan = np.all(np.isnan(block), axis=1)
        out = np.nansum(block, axis=1)
        out[allnan] = np.nan
        return out

    obs = summed(patch.values)
    pobs = summed(patch.precursor)
    t = patch.targets
    nf = len(patch.names)
    exp = np.vstack([patch.cand.expected(f, t) for f in range(nf)]) if nf else np.zeros((0, t.size))
    fitted = None
    env_fit = None
    if patch.fitted_centre_steps is not None:
        mu = patch.fitted_centre_steps
        om = patch.width if patch.width is not None and patch.width > 0 else 1.0
        fitted = (
            np.vstack([patch.cand.expected(f, t, mu, om) for f in range(nf)])
            if nf
            else np.zeros((0, t.size))
        )
        env_fit = patch.cand.envelope(t, mu, om)
    np_ = len(patch.precursor_names)
    pexp = np.vstack([patch.prec_cand.expected(k, t) for k in range(np_)])
    return Q1Profile(
        cycles=patch.cycles[sel],
        targets=t,
        observed=obs,
        expected=exp,
        fitted=fitted,
        amplitude=_amplitude(obs, exp),
        precursor_observed=pobs,
        precursor_expected=pexp,
        precursor_amplitude=_amplitude(pobs, pexp),
        envelope=patch.cand.envelope(t),
        envelope_fitted=env_fit,
    )


def separable_model(
    row: dict[str, Any] | None, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The engine's separable RT x Q1 model of the row (``features.q1_separable``,
    psms_extracted v9): the cycle RTs, the elution profile ``E`` (maximum 1) and each
    fragment's amplitude ``A_f`` (library order), so ``A_f * E`` is the model's trace of
    fragment ``f``. None when the row has no fit."""
    if row is None or row.get("q1s_elution") is None or row.get("q1s_cycle_rt") is None:
        return None
    e = np.asarray([np.nan if v is None else v for v in row["q1s_elution"]], dtype=np.float64)
    rt = np.asarray([np.nan if v is None else v for v in row["q1s_cycle_rt"]], dtype=np.float64)
    amp = _list(row, "q1s_frag_amp", n)
    if amp is None or e.size == 0 or e.size != rt.size or not np.nanmax(e, initial=0.0) > 0:
        return None
    return rt, e, amp


def summary_items(row: dict[str, Any] | None) -> list[tuple[str, str, Any]]:
    """The engine's per-candidate Q1 columns of ``psms_extracted`` as (column, meaning,
    value), the ones present in the row."""
    if row is None:
        return []
    meaning = [
        ("q1f_n_present", "fragments present in Q1"),
        ("q1f_n_belong", "fragments that belong"),
        ("q1f_n_interfered", "fragments interfered"),
        ("q1f_belong_share", "predicted-intensity share of the fragments that belong"),
        ("q1f_interfered_share", "observed-intensity share of the interfered fragments"),
        ("q1f_centre", "shared Q1 centre against the expected one (steps)"),
        ("q1f_width", "shared width scale of the window"),
        ("q1f_offset_median", "median of the fragments' own centres (steps)"),
        ("q1f_offset_mad", "their median absolute deviation (steps)"),
        ("q1f_corr2d", "mean 2D correlation of the fragments that belong"),
        ("q1f_fit_resid", "mean 1 - R^2 of the fitted set"),
        ("q1f_apex_shift_cycles", "2D apex cycle minus the apex scan's cycle"),
        ("q1f_apex_moved_s", "how far the Q1 re-pick moved the apex (s)"),
        ("q1s_r2", "R^2 of the separable RT x Q1 model over the fitted fragments"),
        ("q1s_r2_pred", "prediction-weighted R^2 of the separable model"),
        ("q1s_interf_pred", "prediction-weighted share of the patch intensity above the model"),
        ("q1s_explained", "share of the patch intensity the model explains"),
        ("q1s_bg_share", "share of the patch intensity in the fragments' backgrounds"),
        ("q1s_amp_cos", "cosine of the model's amplitudes against the predictions"),
        ("q1s_amp_corr", "correlation of the model's amplitudes with the predictions"),
        ("q1s_n_fit", "fragments the separable model fitted"),
        ("q1s_fwhm", "FWHM of the model's elution profile (cycles)"),
        ("q1s_apex_offset", "the model's elution apex against the row's apex (cycles)"),
    ]
    out = []
    for col, text in meaning:
        if col in row and row[col] is not None:
            out.append((col, text, row[col]))
    return out
