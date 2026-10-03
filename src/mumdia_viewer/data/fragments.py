"""Fragment names, the fragment tolerance extraction used, and peak matching for the mirror plot.

The tolerance is not the configured ``extract.frag_tol_ppm``. In every orchestrated run
extract reads the learned tolerance and offset of the mass calibration
(``seed_psms.parquet.masscal.json``) and records the values it used in its reports as
``params.effective_frag_tol_ppm`` and ``params.frag_ppm_offset``. search-seed always
writes that file, also when the calibration fails, and ``mumdia run`` always passes it
to extract, so the configured value applies only to a hand-run extract without
``--mass-cal``. :func:`extraction_tolerance` reads the recorded values and names their
source. When a directory has lost both the reports and the mass calibration file, the
values extract used are unknown: the configured value is then returned as an
assumption (:attr:`Tolerance.assumed`), never as the extraction's value.

Extract matches a recalibrated query ``q = peak_mz / (1 + ppm(peak_mz) * 1e-6)`` against
the library fragment m/z held as float32. ``ppm`` is the scalar offset, or the linear
interpolation of the mass-calibration grid when that grid has two or more points. The
default ``fragindex`` matcher accepts a pair when
``max(q, t) - min(q, t) <= tol * 1e-6 * min(q, t)`` (``within_ppm``); the ``bucketed``
matcher accepts ``f32(q - q*tol*1e-6) <= f32(t) <= f32(q + q*tol*1e-6)``
(``ppm_bounds``). The two differ only at the tolerance edge.

:func:`match_fragments` applies the same predicate to one observed spectrum. Its result
is a viewer computation with the engine's rule: label it :attr:`Tolerance.match_label`.
Under ``extract.peak_claim = none`` (the default) each kept peak is also the value the
engine wrote into the fragment's trace for that scan. Under the other claim modes the
engine gives a shared peak to one candidate or splits its intensity, so only the
predicate is the engine's. Its ppm errors are single-scan errors, unlike the engine's
features, which average the observed m/z over the RT window.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from .discovery import SIDE_FILES, Band
from .errors import ViewerError
from .reports import load_json, normalise_enum
from .spectra import _as_index, _resolve_run

if TYPE_CHECKING:
    from .artifacts import Artifact
    from .discovery import ResultSet, Run

__all__ = [
    "ENGINE_DEFAULT_FRAG_TOL_PPM",
    "FRAGMENT_NAME",
    "MATCHERS",
    "MATCH_LABEL",
    "PPM_CORRECTED_LABEL",
    "PPM_RAW_LABEL",
    "FragmentName",
    "PeakMatch",
    "Tolerance",
    "extraction_tolerance",
    "is_ms1_row",
    "match_fragments",
    "parse_fragment_name",
    "ppm_bounds_match",
    "within_ppm",
]

# b/y fragment names of both engine paths: FASTA mode (``Fragment::name``) and the
# DIA-NN library importer write ``b3``, ``y7``, ``y7^2``; charge 1 has no suffix.
FRAGMENT_NAME = re.compile(r"^(?P<ion>[by])(?P<ordinal>\d+)(?:\^(?P<charge>\d+))?$")
MS1_PREFIX = "ms1_"
MATCHERS = ("fragindex", "bucketed")
ENGINE_DEFAULT_FRAG_TOL_PPM = 20.0

MATCH_LABEL = "viewer match (engine predicate)"
PPM_RAW_LABEL = "ppm, raw: 1e6*(observed - theoretical)/theoretical in this scan"
PPM_CORRECTED_LABEL = (
    "ppm, viewer-derived: 1e6*(q - theoretical)/theoretical with q = observed/(1 + "
    "offset*1e-6), the query extract matched"
)
_REPORT_KEYS = "params.effective_frag_tol_ppm, params.frag_ppm_offset"
_MISSING_RECORD = "the extraction record (extract report, mass calibration) is missing"
_WHY_MISSING = (
    "search-seed always writes the mass calibration file and mumdia run passes it to "
    "extract, so files are missing from this directory. The tolerance and offset extract "
    "used are unknown."
)
_MASSCAL = SIDE_FILES["masscal"]
_REPORT_FILES = ("psms_extracted.parquet.report.json", "chromatograms.parquet.report.json")


# --------------------------------------------------------------------------- names


@dataclass(frozen=True)
class FragmentName:
    """A parsed b/y fragment name: ion type, ordinal and fragment charge."""

    ion: str
    ordinal: int
    charge: int

    @property
    def label(self) -> str:
        suffix = "" if self.charge == 1 else f"^{self.charge}"
        return f"{self.ion}{self.ordinal}{suffix}"


def parse_fragment_name(name: str | None) -> FragmentName | None:
    """``'y7^2'`` -> ``FragmentName('y', 7, 2)``; None for any other name.

    MS1 pseudo-rows (``ms1_mono``, ``ms1_iso1``, ``ms1_iso2``) and names of a foreign
    library return None; show those verbatim.
    """
    if not name:
        return None
    m = FRAGMENT_NAME.match(name)
    if m is None:
        return None
    ordinal = int(m.group("ordinal"))
    charge = int(m.group("charge")) if m.group("charge") is not None else 1
    if ordinal < 1 or charge < 1:
        return None
    return FragmentName(m.group("ion"), ordinal, charge)


def is_ms1_row(name: str | None) -> bool:
    """True for the MS1 pseudo-rows of a chromatogram table (the engine's prefix rule)."""
    return bool(name) and str(name).startswith(MS1_PREFIX)


# --------------------------------------------------------------------------- tolerance


def _grid_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    arr.setflags(write=False)
    return arr


def _same_array(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool(np.array_equal(a, b))


@dataclass(frozen=True, eq=False)
class Tolerance:
    """The fragment tolerance and mass offset one extraction used.

    ``tol_ppm`` is the effective tolerance and ``offset_ppm`` the scalar mass offset.
    ``grid_mz`` and ``grid_ppm`` are the m/z-dependent offset of the mass calibration;
    extract uses them only when both have the same length of at least 2
    (:attr:`uses_grid`). ``matcher`` is ``config_json.extract.matcher`` (``fragindex``
    by default, or ``bucketed``). ``config_fallback_ppm`` is the configured
    ``extract.frag_tol_ppm``, which extract does not use when a mass calibration file
    exists. ``source`` names the file and keys the values come from.

    ``assumed`` is True when no record of the extraction was found: no extract report
    with the effective values and no mass calibration file. The values are then the
    configured ``extract.frag_tol_ppm`` (or the engine default) with offset 0, which
    extract did not necessarily use; show them as an assumption. ``peak_claim`` is
    ``config_json.extract.peak_claim``; it decides whether a matched peak is also the
    engine's trace value (:attr:`match_is_trace_value`).
    """

    tol_ppm: float
    offset_ppm: float = 0.0
    grid_mz: np.ndarray | None = None
    grid_ppm: np.ndarray | None = None
    matcher: str = "fragindex"
    config_fallback_ppm: float | None = None
    source: str = ""
    grid_source: str | None = None
    notes: tuple[str, ...] = ()
    assumed: bool = False
    peak_claim: str = "none"

    def __post_init__(self) -> None:
        tol = float(self.tol_ppm)
        if not math.isfinite(tol) or tol < 0:
            raise ValueError(f"fragment tolerance must be finite and >= 0 ppm, got {tol}")
        object.__setattr__(self, "tol_ppm", tol)
        object.__setattr__(self, "offset_ppm", float(self.offset_ppm))
        object.__setattr__(self, "grid_mz", _grid_array(self.grid_mz))
        object.__setattr__(self, "grid_ppm", _grid_array(self.grid_ppm))
        object.__setattr__(self, "matcher", _normalise_matcher(self.matcher))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "assumed", bool(self.assumed))
        object.__setattr__(self, "peak_claim", str(self.peak_claim))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tolerance):
            return NotImplemented
        return self._key() == other._key() and self.same_values(other)

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple:
        return (
            self.tol_ppm,
            self.offset_ppm,
            self.matcher,
            self.config_fallback_ppm,
            self.source,
            self.grid_source,
            self.notes,
            self.assumed,
            self.peak_claim,
        )

    def same_values(self, other: Tolerance) -> bool:
        """True when both describe the same matching (tolerance, offset, grid, matcher)."""
        return (
            self.tol_ppm == other.tol_ppm
            and self.offset_ppm == other.offset_ppm
            and self.matcher == other.matcher
            and self.uses_grid == other.uses_grid
            and (
                not self.uses_grid
                or (
                    _same_array(self.grid_mz, other.grid_mz)
                    and _same_array(self.grid_ppm, other.grid_ppm)
                )
            )
        )

    @property
    def uses_grid(self) -> bool:
        """True when extract applied the m/z-dependent grid instead of the scalar offset."""
        return (
            self.grid_mz is not None
            and self.grid_ppm is not None
            and self.grid_mz.size >= 2
            and self.grid_mz.size == self.grid_ppm.size
        )

    @property
    def from_config(self) -> bool:
        """True when the values are the configured tolerance (or the engine default), offset 0.

        :func:`extraction_tolerance` returns them only when the extraction record is
        missing, so this equals :attr:`assumed`: the values are not known to be the
        ones extract used.
        """
        return self.assumed

    @property
    def match_is_trace_value(self) -> bool:
        """True under ``extract.peak_claim = none``, the default.

        Every candidate that matches a peak then receives its full intensity, so the
        peak :func:`match_fragments` keeps is the value extract wrote into the
        fragment's trace for that scan. Under the other claim modes extract may give a
        shared peak to another candidate or split its intensity.
        """
        return normalise_enum(self.peak_claim) == "none"

    @property
    def match_label(self) -> str:
        """The label of a :func:`match_fragments` result under this extraction's claim mode."""
        if self.match_is_trace_value:
            return MATCH_LABEL
        return (
            f"{MATCH_LABEL}; predicate only: extract.peak_claim = {self.peak_claim} may give "
            "a shared peak to another candidate or split its intensity, so the trace value "
            "can differ"
        )

    @overload
    def ppm_at(self, mz: float) -> float: ...

    @overload
    def ppm_at(self, mz: np.ndarray) -> np.ndarray: ...

    def ppm_at(self, mz: float | np.ndarray) -> float | np.ndarray:
        """The mass offset (ppm) extract applied at observed m/z ``mz``.

        The scalar offset, or with a grid the engine's ``MassOffset::factor_at``: the
        node value on an exact node, the first or last node value outside the grid
        (clamped), else ``y0 + (y1 - y0) * (mz - x0) / (x1 - x0)`` between the
        neighbouring nodes.
        """
        arr = np.asarray(mz, dtype=np.float64)
        if not self.uses_grid:
            out = np.full(arr.shape, self.offset_ppm, dtype=np.float64)
        else:
            assert self.grid_mz is not None and self.grid_ppm is not None
            gx, gy = self.grid_mz, self.grid_ppm
            n = gx.size
            i = np.searchsorted(gx, arr, side="left")
            ic = np.minimum(i, n - 1)
            hit = (i < n) & (gx[ic] == arr)
            below = (i == 0) & ~hit
            above = i >= n
            mid = ~(hit | below | above)
            out = np.empty(arr.shape, dtype=np.float64)
            out[hit] = gy[ic[hit]]
            out[below] = gy[0]
            out[above] = gy[n - 1]
            j = i[mid]
            x0, x1, y0, y1 = gx[j - 1], gx[j], gy[j - 1], gy[j]
            out[mid] = y0 + (y1 - y0) * (arr[mid] - x0) / (x1 - x0)
        if np.ndim(mz) == 0:
            return float(out)
        return out

    def query_mz(self, mz: np.ndarray) -> np.ndarray:
        """The recalibrated query ``mz / (1 + ppm_at(mz) * 1e-6)`` extract matched (float64)."""
        mz64 = np.asarray(mz, dtype=np.float64)
        return mz64 / (1.0 + self.ppm_at(mz64) * 1e-6)

    @property
    def label(self) -> str:
        """Plain text for the UI: the values used, their source and the unused fallback."""
        parts = [f"fragment tolerance {self.tol_ppm:.4g} ppm ({self.source})"]
        if self.uses_grid:
            assert self.grid_mz is not None
            parts.append(
                f"m/z-dependent mass offset, {self.grid_mz.size}-point grid ({self.grid_source})"
            )
        else:
            offset = f"mass offset {self.offset_ppm:+.4g} ppm"
            parts.append(f"{offset} (assumed)" if self.assumed else offset)
        parts.append(f"matcher {self.matcher}")
        if not self.match_is_trace_value:
            parts.append(f"extract.peak_claim {self.peak_claim}")
        if self.config_fallback_ppm is not None and not self.assumed:
            parts.append(
                f"configured extract.frag_tol_ppm {self.config_fallback_ppm:g} ppm "
                "(fallback, not used)"
            )
        return "; ".join(parts)


def _normalise_matcher(value: Any) -> str:
    """``fragindex`` or ``bucketed`` for the known spellings; other values lower-cased."""
    key = normalise_enum(value if value is not None else "fragindex")
    return key if key in MATCHERS else str(value).lower()


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rel(rs: ResultSet, path: Path) -> str:
    try:
        return Path(path).relative_to(rs.root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _masscal_grid(data: dict[str, Any], key: str) -> np.ndarray:
    """A grid list as the engine reads it: numeric entries only (``filter_map(as_f64)``)."""
    raw = data.get(key)
    if not isinstance(raw, list):
        return np.zeros(0, dtype=np.float64)
    vals = [float(v) for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return np.asarray(vals, dtype=np.float64)


def _listing(items: list[str], limit: int = 12) -> str:
    """``items`` joined by commas, cut after ``limit`` entries."""
    text = ", ".join(items[:limit])
    return text if len(items) <= limit else f"{text}, ... ({len(items)} in all)"


def _looked_for(rs: ResultSet, root: Path, masscal: Path) -> str:
    """The files an extraction record is read from: the extract reports and the masscal."""
    return ", ".join(_rel(rs, p) for p in (*(root / f for f in _REPORT_FILES), masscal))


def _assumed(
    *, looked_for: str, matcher: str, config_tol: float | None, peak_claim: str, scope: str
) -> Tolerance:
    """The configured tolerance with offset 0, marked as an assumption (no record found)."""
    notes = [
        f"{scope}: no extract report records the effective values and no mass calibration "
        f"file was found (looked for {looked_for}).",
        _WHY_MISSING,
    ]
    if config_tol is None:
        tol = ENGINE_DEFAULT_FRAG_TOL_PPM
        source = (
            f"assumed: engine default {tol:g} ppm (offset 0); extract.frag_tol_ppm is not in "
            f"config_json and {_MISSING_RECORD}"
        )
        notes.append(f"extract.frag_tol_ppm is not in config_json; engine default {tol:g} ppm")
    else:
        tol = config_tol
        source = f"assumed: configured extract.frag_tol_ppm (offset 0); {_MISSING_RECORD}"
    return Tolerance(
        tol_ppm=tol,
        offset_ppm=0.0,
        matcher=matcher,
        config_fallback_ppm=config_tol,
        source=source,
        notes=tuple(notes),
        assumed=True,
        peak_claim=peak_claim,
    )


def _from_sources(
    rs: ResultSet,
    reports: list[Artifact | None],
    masscal_path: Path | None,
    *,
    looked_for: str,
    matcher: str,
    config_tol: float | None,
    peak_claim: str,
    scope: str,
) -> Tolerance:
    """The first extract report with the effective values, else the masscal, else an assumption.

    ``looked_for`` names the files of the record, for the note of an assumption.
    """
    masscal = load_json(masscal_path) if masscal_path is not None else None
    grid_mz = grid_ppm = None
    grid_source = None
    if masscal is not None and masscal_path is not None:
        grid_mz = _masscal_grid(masscal, "mz_cal_grid_mz")
        grid_ppm = _masscal_grid(masscal, "mz_cal_grid_ppm")
        grid_source = f"{_rel(rs, masscal_path)} (mz_cal_grid_mz, mz_cal_grid_ppm)"
    for artifact in reports:
        if artifact is None or artifact.report is None:
            continue
        params = artifact.report.params
        tol = _finite(params.get("effective_frag_tol_ppm"))
        offset = _finite(params.get("frag_ppm_offset"))
        if tol is None or offset is None:
            continue
        notes: list[str] = []
        fallback = config_tol if config_tol is not None else _finite(params.get("frag_tol_ppm"))
        if masscal is None and rs.config_get("search_seed", "mass_cal_loess") is True:
            # The report records the scalar offset only; a grid exists only under
            # search_seed.mass_cal_loess (masscal.rs), and it lives in the missing file.
            notes.append(
                f"{scope}: the mass calibration file is missing and "
                "search_seed.mass_cal_loess = true, so extract may have applied an "
                "m/z-dependent offset grid, which is unknown here. The scalar offset of the "
                "report is used."
            )
        if masscal is not None:
            m_tol = _finite(masscal.get("frag_tol_ppm"))
            m_off = _finite(masscal.get("frag_ppm_offset"))
            if (m_tol is not None and abs(m_tol - tol) > 1e-9) or (
                m_off is not None and abs(m_off - offset) > 1e-9
            ):
                notes.append(
                    f"{_rel(rs, masscal_path)} now records frag_tol_ppm {m_tol} and "
                    f"frag_ppm_offset {m_off}; the report values are the ones extract used."
                )
        return Tolerance(
            tol_ppm=tol,
            offset_ppm=offset,
            grid_mz=grid_mz,
            grid_ppm=grid_ppm,
            matcher=matcher,
            config_fallback_ppm=fallback,
            source=f"{_rel(rs, artifact.report.path)} ({_REPORT_KEYS})",
            grid_source=grid_source,
            notes=tuple(notes),
            peak_claim=peak_claim,
        )
    if masscal is not None and masscal_path is not None:
        notes = [f"no extract report of {scope} records the effective values"]
        tol = _finite(masscal.get("frag_tol_ppm"))
        if tol is None:
            tol = config_tol if config_tol is not None else ENGINE_DEFAULT_FRAG_TOL_PPM
            notes.append("the mass calibration has no frag_tol_ppm; extract then uses the config")
        offset = _finite(masscal.get("frag_ppm_offset"))
        return Tolerance(
            tol_ppm=tol,
            offset_ppm=0.0 if offset is None else offset,
            grid_mz=grid_mz,
            grid_ppm=grid_ppm,
            matcher=matcher,
            config_fallback_ppm=config_tol,
            source=f"{_rel(rs, masscal_path)} (frag_tol_ppm, frag_ppm_offset)",
            grid_source=grid_source,
            notes=tuple(notes),
            peak_claim=peak_claim,
        )
    return _assumed(
        looked_for=looked_for,
        matcher=matcher,
        config_tol=config_tol,
        peak_claim=peak_claim,
        scope=scope,
    )


def _resolve_band(run: Run, band: int | str | Band) -> Band:
    """A band of a grouped run by plan index, name or object.

    An integer (any integer type) is the plan index :attr:`Band.index`, the ``NN`` of
    ``groups/gNN``. It is not the ``band`` column of ``overlap_losers``, which counts
    the searched bands in pool order and differs from ``NN`` once a band was skipped.
    """
    if isinstance(band, Band):
        return band
    bands = run.grouped.bands if run.grouped is not None else []
    if isinstance(band, str):
        found = next((b for b in bands if b.name == band), None)
    else:
        index = _as_index(band, "a Band, a band name (str) or a plan index (int)")
        found = next((b for b in bands if b.index == index), None)
    if found is None:
        raise KeyError(f"{run.label}: no band {band!r}; bands are {[b.name for b in bands]}")
    return found


def _grouped_calibration(rs: ResultSet, run: Run) -> str:
    """``global`` or ``pergroup``: which mass calibration the band extractions read."""
    value = rs.config_get("groups", "calibration")
    if value is None and run.grouped is not None:
        value = run.grouped.plan.get("calibration")
    return normalise_enum(value) if value is not None else "global"


def _window_groups(rs: ResultSet) -> int:
    """``config_json.groups.window_groups``: more than 1 means the runs were searched in bands."""
    value = rs.config_get("groups", "window_groups", default=1)
    if isinstance(value, bool):
        return 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _band_tolerance(
    rs: ResultSet,
    run: Run,
    band: Band,
    *,
    matcher: str,
    config_tol: float | None,
    peak_claim: str,
) -> Tolerance:
    # run_groups: a band extraction reads the pooled mass calibration under
    # `groups.calibration = global`, else the band's own.
    per_group = _grouped_calibration(rs, run) != "global"
    if per_group:
        masscal, expected = band.side_files.get("masscal"), band.root / _MASSCAL
    else:
        masscal, expected = run.side_files.get("masscal"), run.root / _MASSCAL
    tol = _from_sources(
        rs,
        [band.artifact("psms_extracted"), band.artifact("chromatograms")],
        masscal,
        looked_for=_looked_for(rs, band.root, expected),
        matcher=matcher,
        config_tol=config_tol,
        peak_claim=peak_claim,
        scope=f"band {band.name}",
    )
    return _note_pooled_masscal(rs, run, tol) if tol.assumed and per_group else tol


def _note_pooled_masscal(rs: ResultSet, run: Run, tol: Tolerance) -> Tolerance:
    """Say why the run-level mass calibration does not stand in for a band's own."""
    if run.side_files.get("masscal") is None:
        return tol
    note = (
        f"{_rel(rs, run.root / _MASSCAL)} is the pooled fit of the run; under "
        "groups.calibration = per_group each band read its own, so it is not used here."
    )
    return dataclasses.replace(tol, notes=(*tol.notes, note))


def _without_bands(
    rs: ResultSet, run: Run, *, matcher: str, config_tol: float | None, peak_claim: str
) -> Tolerance:
    """A grouped run whose band tables were not found."""
    scope = (
        f"{run.label}: grouped run (groups.window_groups = {_window_groups(rs)}) without "
        "band tables"
    )
    kw: dict[str, Any] = {"matcher": matcher, "config_tol": config_tol, "peak_claim": peak_claim}
    if _grouped_calibration(rs, run) == "global":
        # Every band extraction read the run's pooled mass calibration.
        looked_for = _rel(rs, run.root / _MASSCAL)
        return _from_sources(
            rs, [], run.side_files.get("masscal"), looked_for=looked_for, scope=scope, **kw
        )
    names = (*_REPORT_FILES, _MASSCAL)
    looked_for = ", ".join(_rel(rs, run.root / "groups" / "gNN" / f) for f in names)
    return _note_pooled_masscal(
        rs, run, _from_sources(rs, [], None, looked_for=looked_for, scope=scope, **kw)
    )


def extraction_tolerance(
    rs: ResultSet, run: Run | str | int, *, band: int | str | Band | None = None
) -> Tolerance:
    """The fragment tolerance, mass offset and matcher the run's extraction used.

    Order of sources: the ``psms_extracted`` report, else the ``chromatograms`` report
    (``params.effective_frag_tol_ppm``, ``params.frag_ppm_offset``); else
    ``seed_psms.parquet.masscal.json`` (``frag_tol_ppm``, ``frag_ppm_offset``). The grid
    always comes from the mass calibration file extract read. ``run`` is a
    :class:`Run`, a run name or a ``source`` index of any integer type.

    A ``mumdia run`` output always holds that record. When it is missing (a trimmed or
    damaged copy), the values extract used are unknown: the result is then the
    configured ``extract.frag_tol_ppm`` with offset 0, marked :attr:`Tolerance.assumed`,
    and its notes name the missing files.

    A grouped run extracts each band separately (``groups/gNN``), with the band's own
    mass calibration under ``groups.calibration = per_group``. Pass ``band`` for the
    band that holds the candidate: a :class:`Band`, a name ``gNN`` or the plan index
    :attr:`Band.index` (any integer type). The plan index is not the ``band`` column of
    ``overlap_losers``, which is a position in pool order. Without ``band``, every band
    that has an extraction record must agree; otherwise :class:`ViewerError` is raised.
    Bands without a record do not take part: their names are listed in the notes, and
    the result is an assumption only when no band has a record.
    """
    run = _resolve_run(rs, run)
    matcher = _normalise_matcher(rs.config_get("extract", "matcher", default="fragindex"))
    config_tol = _finite(rs.config_get("extract", "frag_tol_ppm"))
    peak_claim = str(rs.config_get("extract", "peak_claim", default="none"))
    kw: dict[str, Any] = {"matcher": matcher, "config_tol": config_tol, "peak_claim": peak_claim}
    layout = run.grouped
    if layout is None and _window_groups(rs) <= 1:
        if band is not None:
            raise ViewerError(f"{run.label} is not a grouped run; band={band!r} does not apply.")
        return _from_sources(
            rs,
            [run.artifact("psms_extracted"), run.artifact("chromatograms")],
            run.side_files.get("masscal"),
            looked_for=_looked_for(rs, run.root, run.root / _MASSCAL),
            scope=f"run {run.name}" if run.name else "the run",
            **kw,
        )
    if band is not None:
        return _band_tolerance(rs, run, _resolve_band(run, band), **kw)
    bands = layout.bands if layout is not None else []
    if not bands:
        return _without_bands(rs, run, **kw)
    per_band = [(b, _band_tolerance(rs, run, b, **kw)) for b in bands]
    known = [(b, t) for b, t in per_band if not t.assumed]
    unknown = [b.name for b, t in per_band if t.assumed]
    if not known:
        first_band, first = per_band[0]
        if len(per_band) == 1:
            return first
        summary = (
            f"{run.label}: none of the {len(per_band)} bands has an extraction record; "
            f"the notes of band {first_band.name} follow."
        )
        return dataclasses.replace(
            first,
            source=f"{first.source} for all {len(per_band)} bands",
            notes=(summary, *first.notes),
        )
    ref = known[0][1]
    if not all(t.same_values(ref) for _, t in known[1:]):
        listed = _listing(
            [f"{b.name} {t.tol_ppm:.4g} ppm (offset {t.offset_ppm:+.4g})" for b, t in known]
        )
        raise ViewerError(
            f"{run.label}: the bands were extracted with different fragment tolerances "
            f"({listed}). Pass band= for the band that holds the candidate."
        )
    if len(per_band) == 1:
        return ref
    if not unknown:
        return dataclasses.replace(ref, source=f"{ref.source}; the same in all {len(known)} bands")
    missing = (
        f"{len(unknown)} of {len(per_band)} bands have no extraction record (extract report, "
        f"mass calibration): {_listing(unknown)}. Their tolerance is unknown; pass band= for "
        "a candidate of one of them."
    )
    return dataclasses.replace(
        ref,
        source=(
            f"{ref.source}; the same in all {len(known)} of {len(per_band)} bands that have "
            "an extraction record"
        ),
        notes=(*ref.notes, missing),
    )


# --------------------------------------------------------------------------- matching


def within_ppm(a: np.ndarray | float, b: np.ndarray | float, tol_ppm: float) -> np.ndarray:
    """The engine's ``within_ppm``: ``max - min <= tol*1e-6*min``; False for non-finite input."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    lo = np.minimum(a64, b64)
    hi = np.maximum(a64, b64)
    with np.errstate(invalid="ignore"):
        ok = hi - lo <= tol_ppm * 1e-6 * lo
    return ok & np.isfinite(a64) & np.isfinite(b64)


def ppm_bounds_match(
    query: np.ndarray | float, frag_mz: np.ndarray | float, tol_ppm: float
) -> np.ndarray:
    """The bucketed probe: ``f32(q - q*tol*1e-6) <= f32(frag_mz) <= f32(q + q*tol*1e-6)``."""
    q = np.asarray(query, dtype=np.float64)
    d = q * tol_ppm * 1e-6
    with np.errstate(invalid="ignore", over="ignore"):
        lo32 = (q - d).astype(np.float32)
        hi32 = (q + d).astype(np.float32)
        m32 = np.asarray(frag_mz, dtype=np.float64).astype(np.float32)
        return (m32 >= lo32) & (m32 <= hi32)


@dataclass(frozen=True)
class PeakMatch:
    """One predicted fragment matched to one observed peak.

    ``fragment`` indexes the theoretical list, ``peak`` the spectrum. ``theo_mz`` is the
    fragment m/z rounded to float32 (as extract holds it), ``obs_mz`` the raw peak m/z
    and ``query_mz`` the recalibrated query ``q``. ``ppm_raw`` is
    ``1e6*(obs_mz - theo_mz)/theo_mz`` (:data:`PPM_RAW_LABEL`); ``ppm_corrected`` is
    ``1e6*(q - theo_mz)/theo_mz`` (:data:`PPM_CORRECTED_LABEL`, viewer-derived).
    """

    fragment: int
    peak: int
    obs_mz: float
    obs_intensity: float
    ppm_raw: float
    ppm_corrected: float
    theo_mz: float
    query_mz: float


def match_fragments(
    spec_mz: np.ndarray,
    spec_intensity: np.ndarray,
    theo_mz: np.ndarray,
    tol: Tolerance,
) -> list[PeakMatch]:
    """Match predicted fragments to the peaks of one spectrum with extract's rule.

    Peak m/z are float32 in the artifact (other dtypes are rounded to float32 first);
    ``theo_mz`` is rounded to float32, as the engine's library holds it. Each peak
    becomes the query ``q = mz / (1 + tol.ppm_at(mz) * 1e-6)``, and a pair is accepted
    by ``within_ppm(theo, q, tol)`` (``fragindex``) or by the float32 ``ppm_bounds``
    test (``bucketed``). Per fragment the most intense accepted peak is kept; a tie
    goes to the lower peak index. One peak may match several fragments. The result is
    sorted by fragment and is a viewer computation: label it ``tol.match_label``.

    Under ``extract.peak_claim = none`` (the default, :attr:`Tolerance.match_is_trace_value`)
    the kept peak's intensity is the value extract wrote into the fragment's trace for
    that scan. Under the other claim modes extract gives a shared peak to one candidate
    (``winner_predicted_intensity``, the ``coelution_*`` winners) or splits its
    intensity (``proportional`` and the other redistribution modes), so only the
    predicate is the engine's.
    """
    if tol.matcher not in MATCHERS:
        raise ViewerError(
            f"unknown extract.matcher {tol.matcher!r}; this viewer knows {', '.join(MATCHERS)}."
        )
    mz32 = np.asarray(spec_mz).astype(np.float32, copy=False).reshape(-1)
    inten = np.asarray(spec_intensity, dtype=np.float64).reshape(-1)
    if mz32.shape != inten.shape:
        raise ValueError(f"{mz32.size} m/z values but {inten.size} intensities")
    theo64 = np.asarray(theo_mz, dtype=np.float64).reshape(-1).astype(np.float32).astype(np.float64)
    if mz32.size == 0 or theo64.size == 0:
        return []
    mz64 = mz32.astype(np.float64)
    q = mz64 / (1.0 + tol.ppm_at(mz64) * 1e-6)
    order = np.argsort(q, kind="stable")
    qs = q[order]
    delta = tol.tol_ppm * 1e-6
    usable = np.isfinite(theo64) & (theo64 > 0)
    safe = np.where(usable, theo64, 1.0)
    # Candidate window per fragment, widened past both predicates' edges; every pair in
    # it is then tested with the exact predicate.
    lo = safe / (1.0 + delta) * (1.0 - 1e-6)
    hi = safe / (1.0 - delta) * (1.0 + 1e-6) if delta < 1.0 else np.full(safe.shape, np.inf)
    hi = np.maximum(hi, safe * (1.0 + delta) * (1.0 + 1e-6))
    a = np.searchsorted(qs, lo, side="left")
    b = np.searchsorted(qs, hi, side="right")
    counts = np.where(usable, b - a, 0).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return []
    frag = np.repeat(np.arange(theo64.size, dtype=np.int64), counts)
    first = np.cumsum(counts) - counts
    pos = np.repeat(a.astype(np.int64) - first, counts) + np.arange(total, dtype=np.int64)
    peak = order[pos]
    t = theo64[frag]
    qq = q[peak]
    if tol.matcher == "fragindex":
        ok = within_ppm(t, qq, tol.tol_ppm)
    else:
        ok = ppm_bounds_match(qq, t, tol.tol_ppm)
    frag, peak = frag[ok], peak[ok]
    if frag.size == 0:
        return []
    best = np.lexsort((peak, -inten[peak], frag))
    frag, peak = frag[best], peak[best]
    keep = np.ones(frag.size, dtype=bool)
    keep[1:] = frag[1:] != frag[:-1]
    out = []
    for f, p in zip(frag[keep].tolist(), peak[keep].tolist(), strict=True):
        theo = float(theo64[f])
        obs = float(mz64[p])
        query = float(q[p])
        out.append(
            PeakMatch(
                fragment=f,
                peak=p,
                obs_mz=obs,
                obs_intensity=float(inten[p]),
                ppm_raw=1e6 * (obs - theo) / theo,
                ppm_corrected=1e6 * (query - theo) / theo,
                theo_mz=theo,
                query_mz=query,
            )
        )
    return out
