"""Mobilograms of a diaPASEF candidate: intensity against 1/K0, per fragment and MS1 isotope.

No MuMDIA artifact holds a mobility profile, so the viewer offers two sources:

* **centroids** (:func:`centroid_mobilogram`): the peaks of the run's own spectra tables.
  The MS2 frames are the frame of the shown scan (the apex scan by default) and up to
  ``frames`` frames either side, over every quad slot whose isolation window holds the
  precursor m/z (one ``spectra_ms2`` row per slot); the MS1 frames are the nearest MS1 scan
  and its neighbours. A fragment's sticks are every peak within the extraction's fragment
  tolerance and mass offset (the engine's predicate, :func:`fragments.within_ppm`); an MS1
  isotope's every peak within ``extract.prec_tol_ppm`` (the ``sum_near`` rule), inside the
  1/K0 range of the slots. Convert keeps one intensity-weighted 1/K0 per centroid, so this
  is a set of sticks, not a profile (:data:`CENTROID_LABEL`). Two slots of one frame whose
  1/K0 ranges overlap hold the same raw events, so a peak in the overlap is taken from the
  first slot only.
* **raw** (:func:`engine_mobilogram`): ``mumdia mobilogram``, which bins the raw timsTOF
  events of the same frames in 1/K0 (docs/35_diapasef.md section 17 of the MuMDIA
  repository). It needs the engine binary (``mumdia-viewer --engine`` or ``MUMDIA_BIN``)
  and the ``.d`` the run was converted from. Results are kept in the viewer cache, keyed
  by the ``.d`` and input files' stamps, the candidate, the centre RT, the frames, the
  bin width and the engine version.

Both return a :class:`Mobilogram` with the frames read, so a figure can show the centre
frame alone or the sum of one or two frames either side.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import blake3
import numpy as np

from .errors import ViewerError
from .fragments import Tolerance, ppm_bounds_match, within_ppm
from .spectra import Ms1Table, ScanTable, Spectrum

if TYPE_CHECKING:
    from .detail import PrecursorDetail
    from .discovery import ResultSet, Run

ENGINE_ENV = "MUMDIA_BIN"
TIMEOUT_ENV = "MUMDIA_VIEWER_MOBILOGRAM_TIMEOUT"
DEFAULT_TIMEOUT_S = 120.0
# Frames read either side of the centre frame; the page shows 0, 1 or 2 of them.
FRAMES = 2
# Bin width of the raw profile in 1/K0 (about two TIMS scans), the engine's default.
BIN = 0.002
ENGINE_FORMAT = "mumdia-mobilogram"
ENGINE_FORMAT_VERSION = 1

CENTROID_LABEL = "convert centroids, one 1/K0 per peak, not a profile"
RAW_LABEL = "raw timsTOF events binned in 1/K0 by mumdia mobilogram"

_ENGINE: dict[str, Path | None] = {"path": None}
_VERSIONS: dict[tuple[str, int], str] = {}
_LOCK = threading.Lock()


def set_engine(path: str | os.PathLike[str] | None) -> None:
    """The engine binary the raw source runs (``--engine``); None falls back to
    ``MUMDIA_BIN``."""
    _ENGINE["path"] = Path(path) if path else None


def engine_binary() -> Path | None:
    """The configured engine binary, else ``MUMDIA_BIN``; None when neither is a file."""
    for value in (_ENGINE["path"], os.environ.get(ENGINE_ENV)):
        if value and Path(value).is_file():
            return Path(value)
    return None


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class MobFrame:
    """One frame a mobilogram read: its position relative to the centre frame, its RT and
    a label (``frame=N`` for MS2, the MS1 ``scan_index`` or frame for MS1)."""

    offset: int
    rt: float
    label: str


@dataclass(frozen=True, eq=False)
class IonMobilogram:
    """One ion: ``name`` is the chromatogram row's ``frag_name`` (``ms1_mono``, ``ms1_iso1``,
    ``ms1_iso2`` for MS1), ``kind`` ``"fragment"`` or ``"ms1"``. ``frames[i]`` holds the
    (1/K0, intensity) arrays read in frame ``i`` of the mobilogram's ``ms2_frames``
    (fragments) or ``ms1_frames`` (MS1): the matched centroids, or the raw bins."""

    name: str
    kind: str
    mz: float
    frames: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True, eq=False)
class Mobilogram:
    """The mobilograms of one candidate from one source (``"centroids"`` or ``"raw"``)."""

    source: str
    candidate_id: int
    centre_rt: float
    ms2_frames: tuple[MobFrame, ...]
    ms1_frames: tuple[MobFrame, ...]
    ions: tuple[IonMobilogram, ...]
    slot_im: tuple[float, float] | None
    label: str
    bin: float | None = None
    notes: tuple[str, ...] = ()

    def ion(self, name: str) -> IonMobilogram | None:
        return next((i for i in self.ions if i.name == name), None)

    def frames_of(self, ion: IonMobilogram) -> tuple[MobFrame, ...]:
        return self.ms1_frames if ion.kind == "ms1" else self.ms2_frames

    def within(self, k: int) -> tuple[int, int]:
        """The MS2 and MS1 frames within ``k`` of the centre."""
        return (
            sum(1 for f in self.ms2_frames if abs(f.offset) <= k),
            sum(1 for f in self.ms1_frames if abs(f.offset) <= k),
        )

    def profile(self, ion: IonMobilogram, k: int) -> tuple[np.ndarray, np.ndarray]:
        """``ion`` over the frames within ``k`` of the centre: the raw bins summed (on the
        shared bin axis), or the centroid sticks of those frames, sorted by 1/K0."""
        keep = [i for i, f in enumerate(self.frames_of(ion)) if abs(f.offset) <= int(k)]
        if not keep:
            return np.zeros(0), np.zeros(0)
        if self.source == "raw":
            im = ion.frames[keep[0]][0]
            total = np.sum([ion.frames[i][1] for i in keep], axis=0)
            return im, total
        im = np.concatenate([ion.frames[i][0] for i in keep])
        inten = np.concatenate([ion.frames[i][1] for i in keep])
        order = np.argsort(im, kind="stable")
        return im[order], inten[order]


# --------------------------------------------------------------------------- centroids


def _nearest(sorted_values: np.ndarray, t: float) -> int:
    """Index of the value nearest ``t`` in ascending ``sorted_values``, ties to the earlier
    one (the engine's ``nearest_index``)."""
    p = int(np.searchsorted(sorted_values, t, side="left"))
    if p == 0:
        return 0
    if p >= sorted_values.size:
        return int(sorted_values.size - 1)
    return p - 1 if abs(t - sorted_values[p - 1]) <= abs(sorted_values[p] - t) else p


def _around(sorted_values: np.ndarray, t: float, n: int) -> list[tuple[int, int]]:
    """Positions ``j - n .. j + n`` (clipped) of ``sorted_values``, ``j`` nearest ``t``, with
    their offsets from ``j``."""
    if sorted_values.size == 0:
        return []
    j = _nearest(sorted_values, t)
    a, b = max(0, j - n), min(sorted_values.size - 1, j + n)
    return [(k, k - j) for k in range(a, b + 1)]


def fragment_hits(mz: np.ndarray, theo_mz: float, tol: Tolerance) -> np.ndarray:
    """Indices of every peak of ``mz`` that matches the fragment at ``theo_mz`` under the
    extraction's tolerance, mass offset and matcher (the predicate of
    :func:`fragments.match_fragments`, without keeping only the most intense peak)."""
    mz64 = np.asarray(mz).astype(np.float32, copy=False).astype(np.float64)
    if mz64.size == 0 or not np.isfinite(theo_mz) or theo_mz <= 0:
        return np.zeros(0, dtype=np.int64)
    theo = float(np.float32(theo_mz))
    q = mz64 / (1.0 + tol.ppm_at(mz64) * 1e-6)
    if tol.matcher == "fragindex":
        ok = within_ppm(np.full(q.shape, theo), q, tol.tol_ppm)
    else:
        ok = ppm_bounds_match(q, np.full(q.shape, theo), tol.tol_ppm)
    return np.flatnonzero(ok)


def ms1_hits(mz: np.ndarray, target: float, tol_ppm: float) -> np.ndarray:
    """Indices of the peaks with ``target - d <= mz <= target + d``, ``d = target * tol_ppm
    * 1e-6`` (float32 m/z widened to float64; the engine's ``sum_near`` rule)."""
    mz64 = np.asarray(mz, dtype=np.float32).astype(np.float64)
    d = float(target) * float(tol_ppm) * 1e-6
    return np.flatnonzero((mz64 >= target - d) & (mz64 <= target + d))


def _frame_label(native_id: str | None, scan_index: int) -> str:
    if native_id and native_id.startswith("frame="):
        return native_id.split()[0]
    return f"scan_index {scan_index}"


def merge_slots(spectra: Sequence[Spectrum]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The peaks (m/z, intensity, 1/K0) of the slot spectra of one frame, each raw event
    once: the slots in order of their lower 1/K0 bound, and a peak of a later slot whose
    1/K0 lies in the range of an earlier one dropped (convert centroided both slots from
    the same events there). A spectrum without 1/K0 is skipped."""
    ordered = sorted(
        spectra,
        key=lambda s: (s.window_im_lower if s.window_im_lower is not None else np.inf, s.row),
    )
    covered: list[tuple[float, float]] = []
    mz_parts, int_parts, im_parts = [], [], []
    for s in ordered:
        if s.im is None:
            continue
        keep = np.ones(s.mz.size, dtype=bool)
        for a, b in covered:
            keep &= ~((s.im >= a) & (s.im <= b))
        mz_parts.append(s.mz[keep])
        int_parts.append(s.intensity[keep])
        im_parts.append(s.im[keep])
        if s.window_im_lower is not None and s.window_im_upper is not None:
            covered.append((s.window_im_lower, s.window_im_upper))
    if not mz_parts:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    return np.concatenate(mz_parts), np.concatenate(int_parts), np.concatenate(im_parts)


def _centre(d: PrecursorDetail, centre_rt: float | None) -> float:
    if centre_rt is not None and np.isfinite(centre_rt):
        return float(centre_rt)
    peak = d.selected_peak or {}
    for value in (peak.get("apex_rt"), d.scored.get("apex_rt")):
        if value is not None and np.isfinite(float(value)):
            return float(value)
    raise ViewerError("the identification has no apex retention time")


def prec_tol_ppm(rs: ResultSet) -> float:
    """``extract.prec_tol_ppm`` of the run (3 ppm, the engine default, when not recorded)."""
    value = rs.config_get("extract", "prec_tol_ppm", default=None)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 3.0
    return v if np.isfinite(v) and v > 0 else 3.0


def centroid_mobilogram(
    rs: ResultSet,
    d: PrecursorDetail,
    *,
    centre_rt: float | None = None,
    frames: int = FRAMES,
) -> Mobilogram:
    """The centroid mobilogram of ``d`` (module docstring), centred on the MS2 frame
    nearest ``centre_rt`` (default: the apex). Raises :class:`ViewerError` with the reason
    when the run has no 1/K0 or a part it needs is missing."""
    run = d.run
    if not run.has_im:
        raise ViewerError(f"{run.label} has no ion mobility: a mobilogram needs a diaPASEF run")
    if d.chromatogram is None:
        raise ViewerError("no chromatogram rows for this candidate, so its ions are unknown")
    if d.tolerance is None:
        raise ViewerError("the extraction's fragment tolerance is unknown")
    pmz = d.precursor_mz
    if pmz is None:
        raise ViewerError("the precursor m/z is unknown")
    centre = _centre(d, centre_rt)
    scans = ScanTable.for_run(rs, run)
    windows = scans.covering_windows(pmz)
    if windows.size == 0:
        raise ViewerError(f"no isolation window covers the precursor m/z {pmz:.4f}")
    rows = np.concatenate([scans.rows_in_window(int(w)) for w in windows])
    rts = scans.rt[rows]
    frame_rts = np.unique(rts)
    chosen = _around(frame_rts, centre, int(frames))
    tol = d.tolerance
    frag_traces = [t for t in d.chromatogram.fragments() if t.predicted_intensity > 0]
    ms2_frames: list[MobFrame] = []
    frag_hits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        t.frag_name: [] for t in frag_traces
    }
    lo_im, hi_im = np.inf, -np.inf
    for pos, offset in chosen:
        t = frame_rts[pos]
        here = rows[rts == t]
        spectra = [scans.spectrum(int(r)) for r in here]
        mz, inten, im = merge_slots(spectra)
        for s in spectra:
            if s.window_im_lower is not None and s.window_im_upper is not None:
                lo_im, hi_im = min(lo_im, s.window_im_lower), max(hi_im, s.window_im_upper)
        first = min(spectra, key=lambda s: s.row)
        ms2_frames.append(
            MobFrame(int(offset), float(t), _frame_label(first.native_id, first.scan_index))
        )
        for tr in frag_traces:
            hit = fragment_hits(mz, tr.frag_mz, tol)
            frag_hits[tr.frag_name].append(
                (im[hit].astype(np.float64), inten[hit].astype(np.float64))
            )
    if not ms2_frames:
        raise ViewerError("no MS2 frame of the covering windows")
    slot_im = (float(lo_im), float(hi_im)) if np.isfinite(lo_im) and lo_im < hi_im else None

    # MS1: the nearest MS1 scan and its neighbours, cut to the slots' 1/K0 range.
    ms1_frames: list[MobFrame] = []
    ms1_traces = [t for t in d.chromatogram.ms1() if np.isfinite(t.frag_mz)]
    ms1_hits_by: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        t.frag_name: [] for t in ms1_traces
    }
    notes: list[str] = []
    if ms1_traces and run.has("spectra_ms1"):
        ms1 = Ms1Table.for_run(rs, run)
        order = np.argsort(ms1.rt, kind="stable")
        ms1_tol = prec_tol_ppm(rs)
        for pos, offset in _around(ms1.rt[order], centre, int(frames)):
            row = int(order[pos])
            s = ms1.spectrum(row)
            mz, inten, im = s.mz, s.intensity, s.im
            if im is not None and slot_im is not None:
                keep = (im >= slot_im[0]) & (im <= slot_im[1])
                mz, inten, im = mz[keep], inten[keep], im[keep]
            ms1_frames.append(MobFrame(int(offset), float(s.rt), f"MS1 scan_index {s.scan_index}"))
            for tr in ms1_traces:
                hit = ms1_hits(mz, tr.frag_mz, ms1_tol)
                values = im[hit] if im is not None else np.zeros(hit.size)
                ms1_hits_by[tr.frag_name].append(
                    (values.astype(np.float64), inten[hit].astype(np.float64))
                )
    elif ms1_traces:
        notes.append("the run has no readable spectra_ms1 table: no MS1 mobilogram")
    ions = [
        IonMobilogram(t.frag_name, "fragment", float(t.frag_mz), tuple(frag_hits[t.frag_name]))
        for t in frag_traces
    ]
    if ms1_frames:
        ions += [
            IonMobilogram(t.frag_name, "ms1", float(t.frag_mz), tuple(ms1_hits_by[t.frag_name]))
            for t in ms1_traces
        ]
    return Mobilogram(
        source="centroids",
        candidate_id=int(d.candidate_id),
        centre_rt=centre,
        ms2_frames=tuple(ms2_frames),
        ms1_frames=tuple(ms1_frames),
        ions=tuple(ions),
        slot_im=slot_im,
        label=CENTROID_LABEL,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- engine


def raw_input(rs: ResultSet, run: Run) -> Path | None:
    """The diaPASEF ``.d`` the run's spectra were read from natively, where it is now
    (through the result set's path resolver and ``--remap``); None when the spectra were
    not read from a ``.d`` or it is not found."""
    art = run.artifact("spectra_ms2")
    report = getattr(art, "report", None)
    if report is None:
        return None
    params = report.params or {}
    if params.get("reader") != "timsrust" or not params.get("mzml"):
        return None
    found = rs.resolver.resolve(str(params["mzml"])).path
    return found if found is not None and Path(found).is_dir() else None


def engine_status(rs: ResultSet, run: Run) -> tuple[bool, str]:
    """Whether the raw source can run for ``run``, and why not when it cannot."""
    if not run.has_im:
        return False, "not a diaPASEF run"
    engine = engine_binary()
    if engine is None:
        return False, (
            "no MuMDIA engine: start the viewer with --engine <mumdia> or set MUMDIA_BIN"
        )
    if raw_input(rs, run) is None:
        return False, (
            "the .d the run was converted from is not found (give its new location with "
            "--remap OLD=NEW)"
        )
    for kind in ("psms_extracted", "chromatograms"):
        if not run.has(kind):
            return False, f"the run has no readable {kind} table"
    return True, ""


def _stamp(path: Path | None) -> list[Any]:
    if path is None:
        return [None]
    try:
        st = path.stat()
    except OSError:
        return [str(path), None]
    return [str(path), st.st_size, st.st_mtime_ns]


def engine_version(engine: Path, timeout: float = 30.0) -> str:
    """``mumdia --version`` of ``engine`` (memoised per binary and modification time)."""
    key = (str(engine), engine.stat().st_mtime_ns)
    with _LOCK:
        hit = _VERSIONS.get(key)
    if hit is not None:
        return hit
    out = subprocess.run(
        [str(engine), "--version"],
        capture_output=True,
        text=True,
        timeout=timeout,
        **_no_window(),
    )
    version = (out.stdout or out.stderr).strip() or "unknown"
    with _LOCK:
        _VERSIONS[key] = version
    return version


def _no_window() -> dict[str, Any]:
    """On Windows, keep the engine from opening a console window."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _timeout() -> float:
    try:
        v = float(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return v if v > 0 else DEFAULT_TIMEOUT_S


def engine_command(
    rs: ResultSet,
    d: PrecursorDetail,
    *,
    engine: Path,
    raw: Path,
    centre_rt: float | None,
    frames: int,
    bin_width: float,
) -> list[str]:
    """The ``mumdia mobilogram`` command for ``d``: every input named explicitly, as the
    viewer resolved it."""
    run = d.run
    psms = run.artifact("psms_extracted")
    chrom_path = d.chromatogram.table if d.chromatogram is not None else None
    if chrom_path is None:
        chrom = run.artifact("chromatograms")
        chrom_path = chrom.path if chrom is not None else None
    cmd = [
        str(engine),
        "mobilogram",
        "-q",
        "--threads",
        str(max(1, min(8, os.cpu_count() or 1))),
        "--run-dir",
        str(run.root),
        "--candidate",
        str(int(d.candidate_id)),
        "--peak-rank",
        str(int(d.selected_peak_rank)),
        "--frames",
        str(int(frames)),
        "--bin",
        repr(float(bin_width)),
        "--raw",
        str(raw),
        "--psms-extracted",
        str(psms.path),
        "--chromatograms",
        str(chrom_path),
        "--json",
    ]
    windows = run.artifact("run_windows")
    if windows is not None and windows.usable:
        cmd += ["--run-windows", str(windows.path)]
    masscal = run.side_files.get("masscal")
    if masscal is not None and Path(masscal).is_file():
        cmd += ["--mass-cal", str(masscal)]
    if centre_rt is not None and np.isfinite(centre_rt):
        cmd += ["--rt", repr(float(centre_rt))]
    return cmd


def engine_mobilogram(
    rs: ResultSet,
    d: PrecursorDetail,
    *,
    centre_rt: float | None = None,
    frames: int = FRAMES,
    bin_width: float = BIN,
    timeout: float | None = None,
) -> Mobilogram:
    """The raw mobilogram of ``d`` from ``mumdia mobilogram`` (module docstring), from the
    viewer cache when the same request was answered before. Raises :class:`ViewerError`
    with the reason when the engine cannot run or fails."""
    ok, why = engine_status(rs, d.run)
    if not ok:
        raise ViewerError(why)
    engine = engine_binary()
    raw = raw_input(rs, d.run)
    assert engine is not None and raw is not None
    cmd = engine_command(
        rs, d, engine=engine, raw=raw, centre_rt=centre_rt, frames=frames, bin_width=bin_width
    )
    version = engine_version(engine)
    key = {
        "raw": _stamp(raw / "analysis.tdf_bin") + _stamp(raw / "analysis.tdf"),
        "engine": version,
        "candidate": int(d.candidate_id),
        "rank": int(d.selected_peak_rank),
        "rt": None if centre_rt is None else round(float(centre_rt), 4),
        "frames": int(frames),
        "bin": float(bin_width),
        "inputs": [_stamp(Path(a)) for a in cmd if a.endswith((".parquet", ".json"))],
    }
    identity = "mobilogram_" + blake3.blake3(json.dumps(key, sort_keys=True).encode()).hexdigest()
    cached = rs.cache.load_json(identity, "mobilogram")
    if cached is None:
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or _timeout(),
                **_no_window(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ViewerError(
                f"mumdia mobilogram did not answer within {exc.timeout:.0f} s"
            ) from None
        except OSError as exc:
            raise ViewerError(f"mumdia mobilogram did not start: {exc}") from None
        if out.returncode != 0:
            said = [ln.strip() for ln in (out.stderr or "").splitlines() if ln.strip()]
            raise ViewerError(
                "mumdia mobilogram failed" + (f": {said[-1]}" if said else f" ({out.returncode})")
            )
        try:
            cached = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            raise ViewerError(f"mumdia mobilogram printed no JSON ({exc})") from None
        rs.cache.save_json(identity, "mobilogram", cached)
    return parse_engine_json(cached, int(d.candidate_id))


def parse_engine_json(obj: Any, candidate_id: int) -> Mobilogram:
    """The :class:`Mobilogram` of ``candidate_id`` in a ``mumdia mobilogram --json``
    answer. Raises :class:`ViewerError` for another format or version, or when the
    engine reports the candidate as an error or does not list it."""
    if not isinstance(obj, dict) or obj.get("format") != ENGINE_FORMAT:
        raise ViewerError("the engine's answer is not a mumdia-mobilogram document")
    version = obj.get("format_version")
    if version != ENGINE_FORMAT_VERSION:
        raise ViewerError(
            f"mumdia-mobilogram format version {version}; this viewer reads {ENGINE_FORMAT_VERSION}"
        )
    for err in obj.get("errors") or []:
        if int(err.get("candidate_id", -1)) == int(candidate_id):
            raise ViewerError(f"the engine could not profile this candidate: {err.get('error')}")
    c = next(
        (c for c in obj.get("candidates") or [] if int(c["candidate_id"]) == int(candidate_id)),
        None,
    )
    if c is None:
        raise ViewerError("the engine's answer does not list this candidate")
    axis = np.asarray(c["im"], dtype=np.float64)

    def frames_of(rows: list[dict], ms1: bool) -> tuple[MobFrame, ...]:
        out = []
        for f in rows:
            fid = int(f.get("frame_id", int(f["index"]) + 1))
            label = f"MS1 frame={fid}" if ms1 else f"frame={fid}"
            out.append(MobFrame(int(f["offset"]), float(f["rt"]), label))
        return tuple(out)

    ions = []
    for ion in c["ions"]:
        per = tuple((axis, np.asarray(v, dtype=np.float64)) for v in ion["per_frame"])
        if any(v.size != axis.size for _, v in per):
            raise ViewerError(f"the engine's profile of {ion['name']} does not match its bins")
        ions.append(IonMobilogram(str(ion["name"]), str(ion["kind"]), float(ion["mz"]), per))
    slot = c.get("slot_im")
    notes = []
    if obj.get("ms1_denoise") is False:
        notes.append("MS1 events without convert's MS1 noise filter")
    return Mobilogram(
        source="raw",
        candidate_id=int(c["candidate_id"]),
        centre_rt=float(c["centre_rt"]),
        ms2_frames=frames_of(c["ms2_frames"], False),
        ms1_frames=frames_of(c["ms1_frames"], True),
        ions=tuple(ions),
        slot_im=(float(slot[0]), float(slot[1])) if slot else None,
        label=RAW_LABEL,
        bin=float(c.get("bin", obj.get("bin"))),
        notes=tuple(notes),
    )
