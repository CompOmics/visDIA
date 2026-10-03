"""RT and fragment mass calibration of a run (P1 views 4 and 5).

What the engine records, read as written:

* ``cal.json`` (rt-im-train, ``rt_im_train.rs``): ``method``, ``calibration_status``,
  the anchor count ``n_train``, the half-window ``w_rt``, ``p_rt``, the ``multiplier``
  and three residual statistics over the anchors (signed median, absolute median and
  MAD; nearest-rank percentiles of ``calibrate.rs``). The residuals are in-sample: they
  are evaluated on the anchors that trained the fit, and under LOESS on the points the
  smoother was fitted through, so they are fit diagnostics, not error estimates
  (docs/08, "The reported RT residuals are in-sample").
* ``run_windows.parquet``: ``rt_pred_cal`` (the fitted map at the candidate's library
  ``predicted_irt``) and the window ``[rt_lo, rt_hi]`` of every library candidate.
* ``seed_psms.parquet.masscal.json`` (search-seed, ``masscal.rs``): the fragment mass
  offset (the median calibrant deviation), the tolerance ``1.5 * p95(|dev - offset|)``
  floored at 5 ppm, ``n_dev`` and the optional m/z grid.
* ``<rt library>.summary.json`` (the DeepLC worker, ``deeplc_finetune.py``): the heads of
  the multi-head calibration.
* ``features.parquet``: ``rt_error_signed`` (``apex_rt - rt_pred_cal``) and the fragment
  mass errors of every scored row.

What the viewer rebuilds, labelled so:

* The anchors (:func:`rt_anchors`). No artifact lists them. The viewer selects them with
  the engine's rule (``rt_im_train.rs``, ``fit_anchors``): seed rows with finite
  ``spectrum_q``, ``score`` and ``observed_rt``, ``spectrum_q < rt_im_train.q_train``,
  label ``target`` and a finite library ``predicted_irt`` (joined by ``candidate_id``
  from the precursor table the calibration read), then the highest ``score`` per
  ``base_peptide_id`` (the first such row in file order on a tie). The fitted curve at
  each anchor is its ``run_windows`` row. The rebuild is checked against ``cal.json``:
  ``n_train``, the three residual statistics and ``w_rt`` are recomputed with the
  engine's nearest-rank percentile (Rust's ``round``: half away from zero) and compared.
* Accepted identifications (:func:`accepted_errors`): the target rows of the run at a q
  threshold (``q_value`` in a single run, ``run_psm_q`` in an experiment), joined to their
  feature row (``peak_rank = selected_peak_rank``).

Grouped runs: under ``groups.calibration = per_group`` each band fits its own anchors
(docs/33, section 4); a band is then a scope of its own (:func:`scopes`), read from its
``groups/gNN`` files with band-local ids and the searched library's rows at the band's
offset (:func:`~mumdia_viewer.data.windows.band_offsets`). Under ``global`` the bands share
one fit on the pooled seed; the run-level record is shown and the anchors are not rebuilt.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import blake3
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .artifacts import Artifact
from .candidate_index import DenseIndex
from .discovery import Band, ResultSet, Run
from .duck import sql_path
from .errors import InconsistentData, ViewerError
from .features import EVIDENCE_FEATURES, feature_source, predicate_columns
from .fragments import extraction_tolerance
from .mbr import mbr_ran
from .pqio import ParquetHandle
from .reports import load_json, normalise_enum
from .units import check_threshold
from .windows import band_offsets

__all__ = [
    "ANCHOR_RULE",
    "IN_SAMPLE",
    "LOAD_CEILING",
    "MASS_COLUMNS",
    "AcceptedErrors",
    "Anchors",
    "CalCheck",
    "CalRecord",
    "MassCalRecord",
    "RtModel",
    "Scope",
    "accepted_errors",
    "binned_quantiles",
    "cal_record",
    "extraction_text",
    "fragment_mz_range",
    "masscal_record",
    "nearest_rank",
    "residual_stats",
    "rt_anchors",
    "rt_model",
    "scope",
    "scopes",
]

_MEMO = "mumdia_viewer.data.calibration"
CACHE_VERSION = 1
# Besides the anchors, the curve holds at most about this many evenly spaced rows of
# run_windows (the fitted map outside the anchors' iRT range).
CURVE_SAMPLE = 20_000
# Accepted rows are read once up to this q (the header's largest stop) and filtered in
# memory for any threshold at or below it.
LOAD_CEILING = 0.1
# The signed per-candidate fragment mass errors of the feature table (raw ppm).
MASS_COLUMNS = ("frag_mass_err_median", "signed_mean_frag_ppm")
# Equal up to this many seconds: the rebuild against the engine's record.
TOLERANCE = 1e-9

ANCHOR_RULE = (
    "seed rows with finite spectrum_q, score and observed_rt, spectrum_q < q_train, label "
    "target and a finite library predicted_irt (joined by candidate_id), then the "
    "highest-scoring row per base_peptide_id (rt_im_train.rs, fit_anchors)"
)
IN_SAMPLE = (
    "In-sample: evaluated on the anchors that trained the fit (under LOESS, the points the "
    "smoother was fitted through). A fit diagnostic, not an error estimate (docs/08)."
)


# --------------------------------------------------------------------------- numbers


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def nearest_rank(values: Any, p: float) -> float:
    """The engine's percentile (``calibrate.rs``): the sorted value at ``round(p * (n - 1))``.

    Rust's ``f64::round`` rounds half away from zero, so ``k.5`` gives ``k + 1``, unlike
    Python's ``round`` (half to even). Empty input gives NaN; the engine never reports
    a percentile of no values.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size == 0:
        return float("nan")
    rank = math.floor(min(max(float(p), 0.0), 1.0) * (v.size - 1) + 0.5)
    return float(v[rank])


def residual_stats(residuals: Any) -> tuple[float, float, float]:
    """(signed median, absolute median, MAD) as ``rt_im_train.rs`` writes them to
    ``cal.json``: nearest-rank medians; the MAD is the median of ``|r - median|``."""
    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        nan = float("nan")
        return nan, nan, nan
    med = nearest_rank(r, 0.5)
    return med, nearest_rank(np.abs(r), 0.5), nearest_rank(np.abs(r - med), 0.5)


def binned_quantiles(
    x: Any,
    y: Any,
    *,
    bins: int = 40,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    min_count: int = 20,
    lo: float | None = None,
    hi: float | None = None,
) -> pd.DataFrame:
    """Quantiles of ``y`` in equal-width bins of ``x`` (a viewer-derived summary line).

    Columns ``x_lo``, ``x_hi``, ``x_mid``, ``n`` and ``q05``, ``q50``, ... (numpy's
    linearly interpolated quantiles). Bins with fewer than ``min_count`` values are left
    out; non-finite pairs are ignored. The last bin includes its upper edge.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    names = [f"q{round(100 * q):02d}" for q in quantiles]
    columns = ["x_lo", "x_hi", "x_mid", "n", *names]
    if xa.size == 0:
        return pd.DataFrame(columns=columns)
    a = float(np.min(xa)) if lo is None else float(lo)
    b = float(np.max(xa)) if hi is None else float(hi)
    if not b > a:
        b = a + 1.0
    nb = max(1, int(bins))
    edges = np.linspace(a, b, nb + 1)
    which = np.clip(np.searchsorted(edges, xa, side="right") - 1, 0, nb - 1)
    keep = (xa >= a) & (xa <= b)
    rows = []
    for k in range(nb):
        values = ya[keep & (which == k)]
        if values.size < min_count:
            continue
        qs = np.quantile(values, list(quantiles))
        rows.append([edges[k], edges[k + 1], (edges[k] + edges[k + 1]) / 2, values.size, *qs])
    return pd.DataFrame(rows, columns=columns)


def _rel(rs: ResultSet, path: Path | None) -> str:
    if path is None:
        return "?"
    try:
        return Path(path).relative_to(rs.root).as_posix()
    except ValueError:
        return Path(path).as_posix()


# --------------------------------------------------------------------------- scopes


@dataclass(frozen=True)
class Scope:
    """Where one RT and mass calibration lives: a run, or one band of a grouped run.

    ``offset`` maps the scope's candidate ids to library-wide ids (``local + offset``);
    it is 0 for a run and the band's row offset for a band, whose library rows are
    ``[offset, offset + size)``. ``mode`` is ``run`` (ungrouped), ``band``
    (``groups.calibration = per_group``) or ``global`` (a grouped run whose bands share
    one fit; ``note`` says what is shown instead).
    """

    run: Run
    band: Band | None
    offset: int
    size: int | None
    mode: str
    note: str = ""

    @property
    def key(self) -> str:
        """The band name (``gNN``), or ``""`` for the run itself."""
        return self.band.name if self.band is not None else ""

    def label(self, rs: ResultSet) -> str:
        where = f"run {self.run.name}" if rs.is_experiment else "the run"
        return f"band {self.band.name} of {where}" if self.band is not None else where

    def side(self, name: str) -> Path | None:
        files = self.band.side_files if self.band is not None else self.run.side_files
        return files.get(name)

    def artifact(self, kind: str) -> Artifact | None:
        if self.band is not None:
            return self.band.artifacts.get(kind)
        return self.run.artifacts.get(kind)


def _groups_calibration(rs: ResultSet, run: Run) -> str:
    value = rs.config_get("groups", "calibration")
    if value is None and run.grouped is not None:
        value = run.grouped.plan.get("calibration")
    return normalise_enum(value) if value is not None else "global"


def scopes(rs: ResultSet, run: Run | str | int) -> list[Scope]:
    """The calibration scopes of a run: the run itself, or its bands under
    ``groups.calibration = per_group`` (each band whose row offset is known)."""
    run = rs.run(run)
    if run.grouped is None:
        return [Scope(run, None, 0, None, "run")]
    if _groups_calibration(rs, run) == "global":
        return [
            Scope(
                run,
                None,
                0,
                None,
                "global",
                "Grouped run under groups.calibration = global: the bands share one fit on "
                "the pooled seed (docs/33, section 4). The run-level cal.json is shown; the "
                "viewer does not rebuild these anchors.",
            )
        ]
    try:
        offsets = band_offsets(rs, run)
    except ViewerError:
        offsets = {}
    out = []
    for band in run.grouped.bands:
        bo = offsets.get(band.name)
        if bo is not None:
            out.append(
                Scope(
                    run,
                    band,
                    bo.offset,
                    bo.n,
                    "band",
                    f"Band {band.name}: band-local candidate ids; library-wide id = local id "
                    f"+ {bo.offset} ({bo.method} offset).",
                )
            )
    if out:
        return out
    return [
        Scope(
            run,
            None,
            0,
            None,
            "global",
            "Grouped run whose band offsets are unknown: the run-level cal.json is shown; the "
            "viewer does not rebuild the anchors.",
        )
    ]


def scope(rs: ResultSet, run: Run | str | int, band: str | None = None) -> Scope:
    """The scope of band ``band`` (``gNN``) of :func:`scopes`, else the first scope."""
    found = scopes(rs, run)
    for s in found:
        if band and s.key == band:
            return s
    return found[0]


# --------------------------------------------------------------------------- cal.json


@dataclass(frozen=True)
class CalRecord:
    """``cal.json`` of one scope, as rt-im-train wrote it (``rt_im_train.rs``).

    ``groups`` is the per-band list of a grouped run's run-level record (docs/33,
    section 7). ``raw`` keeps every field.
    """

    path: Path | None
    method: str | None
    status: str | None
    n_train: int | None
    w_rt: float | None
    p_rt: float | None
    multiplier: float | None
    slope: float | None
    intercept: float | None
    residual_median_s: float | None
    residual_abs_median_s: float | None
    residual_mad_s: float | None
    w_rt_sizing: str | None
    holdout_frac: float | None
    groups: tuple[dict[str, Any], ...] = ()
    groups_calibration: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def p_rt_width(self) -> float | None:
        """``w_rt / multiplier``: the residual percentile behind ``w_rt`` (it can be
        floored at 1 s, then this is the floor over the multiplier)."""
        if self.w_rt is None or not self.multiplier:
            return None
        return self.w_rt / self.multiplier


def cal_record(rs: ResultSet, run: Run | str | int, band: str | None = None) -> CalRecord | None:
    """``cal.json`` of the run (or of band ``gNN``); None when the file is absent."""
    s = scope(rs, run, band)
    path = s.side("cal")
    data = load_json(path) if path is not None else None
    if not isinstance(data, dict):
        return None
    return CalRecord(
        path=path,
        method=data.get("method"),
        status=data.get("calibration_status"),
        n_train=_int(data.get("n_train")),
        w_rt=_finite(data.get("w_rt")),
        p_rt=_finite(data.get("p_rt")),
        multiplier=_finite(data.get("multiplier")),
        slope=_finite(data.get("slope")),
        intercept=_finite(data.get("intercept")),
        residual_median_s=_finite(data.get("rt_residual_median_s")),
        residual_abs_median_s=_finite(data.get("rt_residual_abs_median_s")),
        residual_mad_s=_finite(data.get("rt_residual_mad_s")),
        w_rt_sizing=data.get("w_rt_sizing"),
        holdout_frac=_finite(data.get("window_holdout_frac")),
        groups=tuple(g for g in data.get("groups") or () if isinstance(g, dict)),
        groups_calibration=data.get("groups_calibration"),
        raw=dict(data),
    )


# --------------------------------------------------------------------------- masscal


@dataclass(frozen=True)
class MassCalRecord:
    """``seed_psms.parquet.masscal.json`` of one scope (search-seed, ``masscal.rs``).

    ``frag_ppm_offset`` is the median calibrant deviation and ``frag_tol_ppm`` the
    learned tolerance ``1.5 * p95(|dev - offset|)``, floored at 5 ppm (with fewer than 20
    calibrants the configured ``search_seed.fragment_tol_ppm`` is kept and the offset is
    0). ``frag_ppm_sigma`` is the same value as the tolerance. The residual median and MAD
    are of the deviations after the offset correction (diagnostics). The grid is the
    optional m/z-dependent offset (``search_seed.mass_cal_loess``); extract applies it only
    when both lists have the same length of at least 2.
    """

    path: Path | None
    frag_ppm_offset: float | None
    frag_tol_ppm: float | None
    frag_ppm_sigma: float | None
    n_dev: int | None
    cal_passes: int | None
    ppm_residual_median: float | None
    ppm_residual_mad: float | None
    grid_mz: np.ndarray
    grid_ppm: np.ndarray
    source: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def uses_grid(self) -> bool:
        return self.grid_mz.size >= 2 and self.grid_mz.size == self.grid_ppm.size


def _grid(data: dict[str, Any], key: str) -> np.ndarray:
    """A grid list as the engine reads it: its numeric entries only."""
    raw = data.get(key)
    if not isinstance(raw, list):
        return np.zeros(0, dtype=np.float64)
    vals = [float(v) for v in raw if isinstance(v, int | float) and not isinstance(v, bool)]
    return np.asarray(vals, dtype=np.float64)


def masscal_record(
    rs: ResultSet, run: Run | str | int, band: str | None = None
) -> MassCalRecord | None:
    """The fragment mass calibration of the run (or of band ``gNN``); None when absent."""
    s = scope(rs, run, band)
    path = s.side("masscal")
    data = load_json(path) if path is not None else None
    if not isinstance(data, dict):
        return None
    return MassCalRecord(
        path=path,
        frag_ppm_offset=_finite(data.get("frag_ppm_offset")),
        frag_tol_ppm=_finite(data.get("frag_tol_ppm")),
        frag_ppm_sigma=_finite(data.get("frag_ppm_sigma")),
        n_dev=_int(data.get("n_dev")),
        cal_passes=_int(data.get("cal_passes")),
        ppm_residual_median=_finite(data.get("ppm_residual_median")),
        ppm_residual_mad=_finite(data.get("ppm_residual_mad")),
        grid_mz=_grid(data, "mz_cal_grid_mz"),
        grid_ppm=_grid(data, "mz_cal_grid_ppm"),
        source=data.get("masscal_source"),
        raw=dict(data),
    )


def extraction_text(rs: ResultSet, run: Run | str | int, band: str | None = None) -> str:
    """What extract used (:func:`~mumdia_viewer.data.fragments.extraction_tolerance`)."""
    s = scope(rs, run, band)
    try:
        tol = extraction_tolerance(rs, s.run, band=s.band)
    except (ViewerError, KeyError, ValueError) as exc:
        return f"not known: {exc}"
    return tol.label


def fragment_mz_range(
    rs: ResultSet, run: Run | str | int, band: str | None = None
) -> tuple[float, float, str] | None:
    """The m/z range of the extracted fragments, for a display axis.

    The footer minimum and maximum of ``frag_mz`` in the chromatogram table (which also
    holds the MS1 pseudo-rows), else of ``mz`` in the library fragment table; None when
    neither has statistics. The third item names the source.
    """
    s = scope(rs, run, band)
    tables: list[tuple[Artifact | None, str]] = [
        (s.artifact("chromatograms"), "frag_mz"),
        (rs.extra.get("fragment_library_fragments"), "mz"),
    ]
    for artifact, column in tables:
        if artifact is None or not artifact.usable:
            continue
        try:
            stats = artifact.parquet().column_statistics(column)
        except (ViewerError, OSError):
            continue
        if not stats or any(st is None for st in stats):
            continue
        lo = min(float(st[0]) for st in stats if st is not None)
        hi = max(float(st[1]) for st in stats if st is not None)
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi, f"footer min and max of {column} in {_rel(rs, artifact.path)}"
    return None


# --------------------------------------------------------------------------- RT model


@dataclass(frozen=True)
class RtModel:
    """The precursor table whose ``predicted_irt`` the calibration read, and its model.

    ``library_note`` says whose table it is: the run's own adapted table, the first run's
    (``experiment.rt_library_scope = first_run_only``), the experiment's DeepLC
    re-prediction or the searched library. ``summary`` is the DeepLC worker's
    ``<table>.summary.json`` when it exists; :attr:`multihead` is its ``multihead``
    record (``deeplc_finetune.py``, ``multihead_record``). ``rt_predictor`` is the
    manifest's ``model_identities.rt_predictor``.
    """

    library: Artifact | None
    library_label: str
    library_note: str
    owner: str | None
    rt_predictor: str | None
    summary_path: Path | None
    summary: dict[str, Any] | None
    config: dict[str, Any]

    @property
    def multihead(self) -> dict[str, Any] | None:
        mh = (self.summary or {}).get("multihead")
        return mh if isinstance(mh, dict) else None

    @property
    def heads(self) -> tuple[int, ...]:
        heads = (self.multihead or {}).get("heads") or ()
        return tuple(int(h) for h in heads if isinstance(h, int | float))

    @property
    def best_head(self) -> int | None:
        return _int((self.multihead or {}).get("best_head"))


def _band_table(band: Band) -> Path | None:
    """A band's own re-predicted precursor table (``groups/gNN/lib_precursors*.parquet``)."""
    for path in sorted(band.root.glob("lib_precursors*.parquet")):
        if path.is_file():
            return path
    return None


def _scope_library(rs: ResultSet, s: Scope) -> Artifact | None:
    """The precursor table whose ``predicted_irt`` a scope's calibration read.

    A run: its ``rt_library`` (discovery: its own multi-head or fine-tuned table, the
    first run's under ``first_run_only``, the experiment's DeepLC table, the searched
    library). A band: the searched library, read at the band's row offset; a band with a
    table of its own (an RT model rewrote it) is not supported: None.
    """
    if s.band is not None:
        if _band_table(s.band) is not None:
            return None
        return rs.extra.get("fragment_library_precursors") or s.run.artifacts.get(
            "fragment_library_precursors"
        )
    return s.run.artifacts.get("rt_library")


def _table_kind(name: str) -> str:
    if "_multihead" in name:
        return "the multi-head table adapted to"
    if "_ft" in name:
        return "the DeepLC fine-tuned table of"
    return "the table of"


def rt_model(rs: ResultSet, run: Run | str | int, band: str | None = None) -> RtModel:
    """The RT library of the scope and its DeepLC summary (multi-head heads, timings)."""
    s = scope(rs, run, band)
    lib = _scope_library(rs, s)
    config = {
        "multihead_calibration": rs.config_get("rt_im_train", "multihead_calibration"),
        "library_irt": rs.config_get("rt_im_train", "library_irt"),
        "finetune_deeplc": rs.config_get("rt_im_train", "finetune_deeplc"),
        "rt_library_scope": rs.config_get("experiment", "rt_library_scope"),
    }
    predictor = rs.manifest.model_identities.get("rt_predictor")
    if lib is None or lib.path is None:
        if s.band is not None and _band_table(s.band) is not None:
            note = "a band table of its own (groups/gNN/lib_precursors*.parquet), not read yet"
        else:
            note = "no precursor table was found"
        return RtModel(lib, "", note, None, predictor, None, None, config)
    owner = next((r.name for r in rs.runs if r.name and lib.path.parent == r.root), None)
    name = lib.path.name
    adapted = "_multihead" in name or "_ft" in name
    if s.band is not None:
        last = s.offset + (s.size or 0) - 1
        note = f"the searched library, rows {s.offset} to {last} (the band's rows)"
    elif owner is not None and owner != s.run.name:
        note = (
            f"{_table_kind(name)} run {owner}, reused under experiment.rt_library_scope = "
            f"{config['rt_library_scope'] or 'first_run_only'}; this run fits its own LOESS "
            "on its predicted_irt"
        )
    elif adapted:
        note = f"{_table_kind(name)} this run"
    elif lib is rs.extra.get("fragment_library_precursors_deeplc"):
        note = "the experiment's DeepLC re-prediction (one table for every run)"
    else:
        note = "the searched library"
    summary_path = lib.path.with_name(name + ".summary.json")
    summary = load_json(summary_path) if summary_path.is_file() else None
    ok = isinstance(summary, dict)
    return RtModel(
        lib,
        _rel(rs, lib.path),
        note,
        owner,
        predictor,
        summary_path if ok else None,
        summary if ok else None,
        config,
    )


# --------------------------------------------------------------------------- dense reads


def _dense(artifact: Artifact) -> ParquetHandle:
    """The handle of a table whose ``candidate_id`` equals the row index, or raise."""
    handle = artifact.parquet()
    if not DenseIndex.applies(handle):
        raise InconsistentData(
            f"{artifact.path}: candidate_id is not the row index, so its rows cannot be "
            "joined by position (the engine writes and loads this table that way)."
        )
    return handle


def _take(handle: ParquetHandle, rows: np.ndarray, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """The values of ``columns`` at global ``rows`` of a dense table.

    One row group at a time; every row's ``candidate_id`` is checked against its row.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = handle.num_rows
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n):
        raise InconsistentData(f"{handle.path}: rows outside the table's {n} rows were asked for.")
    offsets = handle.row_group_offsets()
    groups = np.searchsorted(offsets, rows, side="right") - 1
    out: dict[str, np.ndarray] = {}
    for g in np.unique(groups):
        sel = np.flatnonzero(groups == g)
        table = handle.read_row_group(int(g), ["candidate_id", *columns])
        local = rows[sel] - int(offsets[g])
        ids = table.column("candidate_id").to_numpy().astype(np.int64)[local]
        if not np.array_equal(ids, rows[sel]):
            raise InconsistentData(f"{handle.path}: candidate_id differs from the row index.")
        for c in columns:
            values = table.column(c).to_numpy(zero_copy_only=False)
            if c not in out:
                out[c] = np.empty(rows.size, dtype=values.dtype)
            out[c][sel] = values[local]
    for c in columns:
        out.setdefault(c, np.empty(0, dtype=np.float64))
    return out


# --------------------------------------------------------------------------- anchors


@dataclass(frozen=True)
class CalCheck:
    """One number of ``cal.json`` against the viewer's rebuild from the anchors.

    ``equal`` is None when the number was not checked (``note`` says why).
    """

    name: str
    recorded: float | int | None
    rebuilt: float | int | None
    equal: bool | None
    note: str = ""


ANCHOR_COLUMNS = (
    "candidate_id",
    "local_id",
    "base_peptide_id",
    "peptidoform",
    "charge",
    "precursor_mz",
    "score",
    "spectrum_q",
    "observed_rt",
    "scan_index",
    "irt",
    "seed_irt",
    "rt_pred_cal",
    "rt_lo",
    "rt_hi",
)
CURVE_COLUMNS = ("irt", "rt_pred_cal", "rt_lo", "rt_hi")
FUNNEL_STEPS = (
    "seed rows",
    "finite spectrum_q, score and observed_rt",
    "spectrum_q < q_train",
    "label target",
    "finite library predicted_irt",
    "anchors: best score per base_peptide_id",
)


@dataclass
class Anchors:
    """The RT calibration anchors of one scope, rebuilt by the viewer (see the module).

    ``frame``: one row per anchor, in ``base_peptide_id`` order (the engine's order):
    ``candidate_id`` (library-wide), ``local_id``, ``base_peptide_id``, ``peptidoform``,
    ``charge``, ``precursor_mz``, ``score``, ``spectrum_q``, ``observed_rt``,
    ``scan_index``, ``irt`` (the library ``predicted_irt`` the fit read), ``seed_irt``
    (the seed row's own ``predicted_irt``), ``rt_pred_cal``, ``rt_lo``, ``rt_hi``
    (``run_windows``), ``residual`` (``observed_rt - rt_pred_cal``), ``in_window``, and
    ``scored`` (the run has a scored row of the candidate) with that row's ``q``
    (``q_column``).

    ``curve``: ``irt``, ``rt_pred_cal``, ``rt_lo`` and ``rt_hi`` at the anchors and at
    evenly spaced rows of ``run_windows``, sorted by ``irt``: the fitted map as the engine
    wrote it. ``funnel``: the seed rows left after each step of the rule. ``checks``: the
    rebuild against ``cal.json``. ``error`` says why the anchors could not be rebuilt.
    """

    scope: Scope
    frame: pd.DataFrame
    curve: pd.DataFrame
    funnel: list[tuple[str, int]]
    checks: list[CalCheck]
    q_train: float
    q_column: str
    irt_source: str = ""
    windows_source: str = ""
    seed_source: str = ""
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def agrees(self) -> bool | None:
        """True when every checked number equals ``cal.json``; None when none was checked."""
        checked = [c for c in self.checks if c.equal is not None]
        if not checked:
            return None
        return all(c.equal for c in checked)


def _q_train(rs: ResultSet) -> float:
    value = _finite(rs.config_get("rt_im_train", "q_train"))
    return 0.01 if value is None else value


def _q_column(rs: ResultSet) -> str:
    return "run_psm_q" if rs.is_experiment else "q_value"


def _empty_anchors(s: Scope, rs: ResultSet, error: str, **sources: str) -> Anchors:
    return Anchors(
        scope=s,
        frame=pd.DataFrame(columns=[*ANCHOR_COLUMNS, "residual", "in_window", "scored", "q"]),
        curve=pd.DataFrame(columns=list(CURVE_COLUMNS)),
        funnel=[],
        checks=[],
        q_train=_q_train(rs),
        q_column=_q_column(rs),
        error=error,
        **sources,
    )


_SEED_COLUMNS = (
    "candidate_id",
    "base_peptide_id",
    "label",
    "score",
    "spectrum_q",
    "observed_rt",
    "predicted_irt",
    "peptidoform",
    "charge",
    "precursor_mz",
    "scan_index",
)


def _read_seed(artifact: Artifact) -> dict[str, np.ndarray]:
    path = artifact.require()
    artifact.parquet()  # checks the column contract
    table = pq.read_table(path, columns=list(_SEED_COLUMNS))
    out: dict[str, np.ndarray] = {}
    for name in _SEED_COLUMNS:
        col = table.column(name)
        if name in ("label", "peptidoform"):
            out[name] = np.asarray(col.to_numpy(zero_copy_only=False), dtype=object)
        else:
            out[name] = col.to_numpy()
    for name in ("spectrum_q", "score", "observed_rt"):
        out[name] = out[name].astype(np.float64)
    out["candidate_id"] = out["candidate_id"].astype(np.int64)
    return out


def _select(
    seed: dict[str, np.ndarray], irt: np.ndarray, q_train: float
) -> tuple[np.ndarray, list[int]]:
    """Row indexes of the anchors (in base_peptide_id order) and the funnel counts."""
    q, score, rt = seed["spectrum_q"], seed["score"], seed["observed_rt"]
    finite = np.isfinite(q) & np.isfinite(score) & np.isfinite(rt)
    confident = finite & (q < q_train)
    target = confident & (seed["label"] == "target")
    with_irt = target & np.isfinite(irt)
    idx = np.flatnonzero(with_irt)
    base = seed["base_peptide_id"]
    # base_peptide_id ascending, score descending, file order ascending: the first row of
    # each base is the one the engine's `score > best` update keeps.
    order = np.lexsort((idx, -score[idx], base[idx]))
    sel = idx[order]
    first = np.ones(sel.size, dtype=bool)
    if sel.size:
        first[1:] = base[sel][1:] != base[sel][:-1]
    anchors = sel[first]
    counts = [
        int(q.size),
        int(finite.sum()),
        int(confident.sum()),
        int(target.sum()),
        int(with_irt.sum()),
        int(anchors.size),
    ]
    return anchors, counts


def _compute(
    seed_artifact: Artifact, lib: ParquetHandle, win: ParquetHandle, s: Scope, q_train: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """The anchor table, the curve and the funnel (the arrays the cache keeps)."""
    seed = _read_seed(seed_artifact)
    local = seed["candidate_id"]
    rows = local + int(s.offset)
    inside = (rows >= 0) & (rows < lib.num_rows)
    if s.size is not None:
        inside &= local < int(s.size)
    # Only confident targets can become anchors: read the library iRT of those rows, and
    # of the evenly spaced rows that sample the curve, in one pass over the library.
    q, score, rt = seed["spectrum_q"], seed["score"], seed["observed_rt"]
    wanted = inside & np.isfinite(q) & np.isfinite(score) & np.isfinite(rt) & (q < q_train)
    wanted &= seed["label"] == "target"
    n_win = win.num_rows
    stride = max(1, n_win // CURVE_SAMPLE)
    sample = np.arange(0, n_win, stride, dtype=np.int64)
    sample_rows = sample + int(s.offset)
    sample = sample[sample_rows < lib.num_rows]
    sample_rows = sample + int(s.offset)
    lib_rows = np.unique(np.concatenate([rows[wanted], sample_rows]))
    lib_irt = _take(lib, lib_rows, ["predicted_irt"])["predicted_irt"].astype(np.float64)
    irt = np.full(local.size, np.nan)
    if wanted.any():
        irt[wanted] = lib_irt[np.searchsorted(lib_rows, rows[wanted])]
    anchors, counts = _select(seed, irt, q_train)
    a_local = local[anchors]
    win_rows = np.unique(np.concatenate([a_local, sample]))
    w = _take(win, win_rows, ["rt_pred_cal", "rt_lo", "rt_hi"])
    pos = np.searchsorted(win_rows, a_local)
    frame = pd.DataFrame(
        {
            "candidate_id": a_local + int(s.offset),
            "local_id": a_local,
            "base_peptide_id": seed["base_peptide_id"][anchors].astype(np.int64),
            "peptidoform": seed["peptidoform"][anchors],
            "charge": seed["charge"][anchors].astype(np.int64),
            "precursor_mz": seed["precursor_mz"][anchors].astype(np.float64),
            "score": seed["score"][anchors],
            "spectrum_q": seed["spectrum_q"][anchors],
            "observed_rt": seed["observed_rt"][anchors],
            "scan_index": seed["scan_index"][anchors].astype(np.int64),
            "irt": irt[anchors],
            "seed_irt": seed["predicted_irt"][anchors].astype(np.float64),
            "rt_pred_cal": w["rt_pred_cal"][pos].astype(np.float64),
            "rt_lo": w["rt_lo"][pos].astype(np.float64),
            "rt_hi": w["rt_hi"][pos].astype(np.float64),
        }
    )
    _derive(frame)
    s_pos = np.searchsorted(win_rows, sample)
    curve = pd.DataFrame(
        {
            "irt": np.concatenate(
                [frame["irt"].to_numpy(), lib_irt[np.searchsorted(lib_rows, sample_rows)]]
            ),
            **{
                c: np.concatenate([frame[c].to_numpy(), w[c][s_pos].astype(np.float64)])
                for c in ("rt_pred_cal", "rt_lo", "rt_hi")
            },
        }
    )
    curve = curve[np.isfinite(curve["irt"]) & np.isfinite(curve["rt_pred_cal"])]
    curve = curve.sort_values("irt", kind="mergesort").drop_duplicates("irt").reset_index(drop=True)
    return frame, curve, counts


def _derive(frame: pd.DataFrame) -> None:
    frame["residual"] = frame["observed_rt"] - frame["rt_pred_cal"]
    frame["in_window"] = (frame["observed_rt"] >= frame["rt_lo"]) & (
        frame["observed_rt"] <= frame["rt_hi"]
    )


def _to_cache(frame: pd.DataFrame, curve: pd.DataFrame, counts: list[int]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for c in ANCHOR_COLUMNS:
        values = frame[c].to_numpy()
        if c == "peptidoform":
            values = values.astype(str) if values.size else np.zeros(0, dtype="<U1")
        arrays[f"a_{c}"] = values
    for c in CURVE_COLUMNS:
        arrays[f"c_{c}"] = curve[c].to_numpy(dtype=np.float64)
    arrays["funnel"] = np.asarray(counts, dtype=np.int64)
    return arrays


def _from_cache(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    frame = pd.DataFrame({c: arrays[f"a_{c}"] for c in ANCHOR_COLUMNS})
    frame["peptidoform"] = frame["peptidoform"].astype(object)
    _derive(frame)
    curve = pd.DataFrame({c: arrays[f"c_{c}"] for c in CURVE_COLUMNS})
    return frame, curve, [int(v) for v in arrays["funnel"]]


def _checks(
    rs: ResultSet, cal: CalRecord | None, frame: pd.DataFrame
) -> tuple[list[CalCheck], list[str]]:
    """The rebuild against cal.json: n_train, the residual statistics and w_rt."""
    if cal is None:
        return [], ["No cal.json in this scope: the rebuild cannot be checked."]
    notes: list[str] = []
    n = len(frame)
    out = [CalCheck("n_train", cal.n_train, n, None if cal.n_train is None else cal.n_train == n)]
    res = frame["residual"].to_numpy(dtype=np.float64)
    usable = n >= 2 and bool(np.isfinite(res).all())
    recorded = (
        ("rt_residual_median_s", cal.residual_median_s),
        ("rt_residual_abs_median_s", cal.residual_abs_median_s),
        ("rt_residual_mad_s", cal.residual_mad_s),
    )
    if usable:
        for (name, value), rebuilt in zip(recorded, residual_stats(res), strict=True):
            equal = None if value is None else abs(value - rebuilt) <= TOLERANCE
            out.append(CalCheck(name, value, rebuilt, equal))
    else:
        why = "fewer than two anchors" if n < 2 else "a non-finite rt_pred_cal at an anchor"
        out += [CalCheck(name, value, None, None, why) for name, value in recorded]
    status = cal.status or ""
    sizing = cal.w_rt_sizing or "in_sample"
    if status in ("loess", "linear") and sizing in ("in_sample", "holdout_fallback_in_sample"):
        if usable and cal.p_rt is not None and cal.multiplier is not None:
            w = max(nearest_rank(np.abs(res), cal.p_rt) * cal.multiplier, 1.0)
            equal = None if cal.w_rt is None else abs(cal.w_rt - w) <= TOLERANCE
            text = (
                f"max(percentile(|residual|, p_rt {cal.p_rt:g}) x multiplier "
                f"{cal.multiplier:g}, 1 s)"
            )
            out.append(CalCheck("w_rt", cal.w_rt, w, equal, text))
    elif status == "fallback_fixed":
        fixed = _finite(rs.config_get("rt_im_train", "fallback_rt_window_s"))
        equal = None if fixed is None or cal.w_rt is None else abs(cal.w_rt - fixed) <= TOLERANCE
        text = "fallback_fixed: rt_im_train.fallback_rt_window_s (too few anchors)"
        out.append(CalCheck("w_rt", cal.w_rt, fixed, equal, text))
    elif sizing == "holdout":
        text = "sized on held-out anchors (w_rt_sizing holdout), a refit cal.json does not record"
        out.append(CalCheck("w_rt", cal.w_rt, None, None, text))
    if bool(rs.config_get("rt_im_train", "adaptive_rt_window", default=False)):
        notes.append(
            "rt_im_train.adaptive_rt_window = true: each candidate's half-width is the width of "
            "its calibrated-RT bin; w_rt is the global width."
        )
    return out, notes


def _scored_lookup(rs: ResultSet, s: Scope, cids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For library-wide ``cids``: whether the run has a scored row, and its q (NaN if not).

    The q is ``q_value`` in a single run and ``run_psm_q`` in an experiment (the rows of
    the run's ``source``).
    """
    has = np.zeros(cids.size, dtype=bool)
    q = np.full(cids.size, np.nan)
    if cids.size == 0:
        return has, q
    q_column = _q_column(rs)
    params: list[Any] = [sql_path(rs.scored.require())]
    where = ""
    if rs.is_experiment:
        where = "source = ? AND "
        params.append(int(s.run.index))
    params.append(np.unique(cids).astype(np.int64).tolist())
    sql = (
        f"SELECT candidate_id, min({q_column}) FROM read_parquet(?) WHERE {where}"
        "candidate_id IN (SELECT unnest(CAST(? AS UBIGINT[]))) GROUP BY candidate_id"
    )
    found = {int(c): float(v) for c, v in rs.duck.rows(sql, params)}
    for i, c in enumerate(cids.tolist()):
        v = found.get(int(c))
        if v is not None:
            has[i] = True
            q[i] = v
    return has, q


def _cache_identity(*parts: str) -> str:
    return "b3:" + blake3.blake3("|".join(parts).encode()).hexdigest()


def rt_anchors(rs: ResultSet, run: Run | str | int, band: str | None = None) -> Anchors:
    """The RT calibration anchors of a run (or band), rebuilt with the engine's rule.

    See the module docstring for the rule and the check against ``cal.json``. The result
    is memoised on the result set; its arrays are cached in the viewer's cache under the
    identities of the seed, library and windows tables. A scope whose anchors cannot be
    rebuilt (a missing table, a grouped run under ``global`` calibration) gives an empty
    result with ``error`` set.
    """
    s = scope(rs, run, band)
    return rs.memo((_MEMO, "anchors", s.run.index, s.key), lambda: _rt_anchors(rs, s))


def _rt_anchors(rs: ResultSet, s: Scope) -> Anchors:
    q_train = _q_train(rs)
    if s.mode == "global":
        return _empty_anchors(s, rs, s.note)
    seed = s.artifact("seed_psms")
    windows = s.artifact("run_windows")
    library = _scope_library(rs, s)
    missing = [
        name
        for name, a in (
            ("seed_psms.parquet", seed),
            ("run_windows.parquet", windows),
            ("the precursor table with predicted_irt", library),
        )
        if a is None or not a.usable
    ]
    if missing or seed is None or windows is None or library is None:
        return _empty_anchors(
            s, rs, f"The anchors cannot be rebuilt: {', '.join(missing)} not found."
        )
    sources = {
        "seed_source": _rel(rs, seed.path),
        "windows_source": f"{_rel(rs, windows.path)} (rt_pred_cal, rt_lo, rt_hi)",
        "irt_source": f"{_rel(rs, library.path)} (predicted_irt by candidate_id"
        + (f" + {s.offset})" if s.offset else ")"),
    }
    try:
        lib_handle = _dense(library)
        win_handle = _dense(windows)
        identity = _cache_identity(
            seed.identity(),
            library.identity(),
            windows.identity(),
            str(s.offset),
            repr(q_train),
            f"v{CACHE_VERSION}",
        )
        cached = rs.cache.load_arrays(identity, "calibration_anchors")
        if cached is not None and "funnel" in cached:
            frame, curve, counts = _from_cache(cached)
        else:
            frame, curve, counts = _compute(seed, lib_handle, win_handle, s, q_train)
            rs.cache.save_arrays(identity, "calibration_anchors", **_to_cache(frame, curve, counts))
        has, q = _scored_lookup(rs, s, frame["candidate_id"].to_numpy(dtype=np.int64))
    except (ViewerError, OSError, ValueError, duckdb.Error) as exc:
        return _empty_anchors(s, rs, f"The anchors cannot be rebuilt: {exc}", **sources)
    frame["scored"] = has
    frame["q"] = q
    checks, notes = _checks(rs, cal_record(rs, s.run, s.key or None), frame)
    if s.band is not None:
        notes.append(s.note)
    return Anchors(
        scope=s,
        frame=frame,
        curve=curve,
        funnel=list(zip(FUNNEL_STEPS, counts, strict=True)),
        checks=checks,
        q_train=q_train,
        q_column=_q_column(rs),
        notes=notes,
        **sources,
    )


# --------------------------------------------------------------------------- accepted


@dataclass
class AcceptedErrors:
    """The accepted target rows of a scope at a threshold, with their RT and mass errors.

    ``frame``: ``candidate_id``, ``peptidoform``, ``charge``, ``apex_rt``, ``q`` (the
    ``q_column``), ``precursor_mz``, ``rt_error`` (the feature ``rt_error_signed``,
    ``apex_rt - rt_pred_cal``, seconds), the mass columns of :data:`MASS_COLUMNS` (raw
    ppm), ``half_width`` (the candidate's RT half-window, seconds) and ``rt_error_rel``
    (viewer-derived: ``rt_error / half_width``). A value whose validity rule
    (:data:`~mumdia_viewer.data.features.EVIDENCE_FEATURES`) fails is NaN: the engine
    writes 0 there as a sentinel. ``counts`` holds the rows and the NaN values per
    column; ``labels`` says what every column is.
    """

    scope: Scope
    threshold: float
    q_column: str
    frame: pd.DataFrame
    counts: dict[str, int]
    labels: dict[str, str]
    half_width_source: str
    population: str
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def n(self) -> int:
        return len(self.frame)


_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b(?!\s*\()")
_SQL_WORDS = frozenset({"and", "or", "not", "is", "null", "true", "false", "in", "between"})
_ERROR_COLUMNS = ("rt_error_signed", *MASS_COLUMNS)
_FRAME_COLUMNS = (
    "candidate_id",
    "peptidoform",
    "charge",
    "apex_rt",
    "q",
    "precursor_mz",
    "rt_error",
    *MASS_COLUMNS,
    "half_width",
    "rt_error_rel",
)


def _qualified(predicate: str, alias: str) -> str:
    """A validity predicate with its column names written ``alias."name"``."""

    def column(m: re.Match[str]) -> str:
        word = m.group(0)
        return word if word.lower() in _SQL_WORDS else f'{alias}."{word}"'

    return _WORD.sub(column, predicate)


def _feature_names(rs: ResultSet, s: Scope) -> tuple[list[str], set[str]]:
    source = feature_source(rs, s.run)
    names: set[str] = set()
    for a in source.artifacts:
        names |= set(a.parquet().schema.names)
    return [sql_path(p) for p in source.paths], names


def _accepted_sql(
    rs: ResultSet, s: Scope, paths: list[str], names: set[str], ceiling: float
) -> tuple[str, list[Any], dict[str, str]]:
    """The query of the accepted rows (up to ``ceiling``) joined to their feature row."""
    q_column = _q_column(rs)
    select = [
        "s.candidate_id",
        "s.peptidoform",
        "s.charge",
        "s.apex_rt",
        f"s.{q_column} AS q",
        'f."precursor_mz" AS precursor_mz' if "precursor_mz" in names else "NULL AS precursor_mz",
    ]
    rules: dict[str, str] = {}
    for c in _ERROR_COLUMNS:
        if c not in names:
            select += [f'NULL AS "{c}"', f'FALSE AS "valid_{c}"']
            continue
        select.append(f'f."{c}" AS "{c}"')
        info = EVIDENCE_FEATURES.get(c)
        rule = info.validity if info is not None else None
        if rule and all(n in names for n in predicate_columns(rule)):
            select.append(f'coalesce(({_qualified(rule, "f")}), FALSE) AS "valid_{c}"')
            rules[c] = rule
        else:
            select.append(f'(f."{c}" IS NOT NULL) AS "valid_{c}"')
    scored = rs.scored.parquet()
    s_rank = "s.selected_peak_rank" if scored.has_column("selected_peak_rank") else "0"
    f_rank = "f.peak_rank" if "peak_rank" in names else "0"
    where = ["s.label = 'target'", f"s.{q_column} <= ?"]
    params: list[Any] = [sql_path(rs.scored.require()), paths[0] if len(paths) == 1 else paths]
    params.append(float(ceiling))
    if rs.is_experiment:
        where.append("s.source = ?")
        params.append(int(s.run.index))
    if s.band is not None and s.size is not None:
        where.append(
            f"s.candidate_id >= {int(s.offset)} AND s.candidate_id < {int(s.offset + s.size)}"
        )
    sql = (
        f"SELECT {', '.join(select)} FROM read_parquet(?) s LEFT JOIN read_parquet(?) f "
        f"ON f.candidate_id = s.candidate_id AND {f_rank} = {s_rank} "
        f"WHERE {' AND '.join(where)} ORDER BY s.apex_rt, s.candidate_id"
    )
    return sql, params, rules


def _half_widths(rs: ResultSet, s: Scope, cids: np.ndarray) -> tuple[np.ndarray, str]:
    """Each accepted candidate's RT half-window, and where it comes from."""
    cal = cal_record(rs, s.run, s.key or None)
    adaptive = bool(rs.config_get("rt_im_train", "adaptive_rt_window", default=False))
    if cal is not None and cal.w_rt is not None and not adaptive:
        return (
            np.full(cids.size, cal.w_rt),
            f"w_rt {cal.w_rt:.4g} s of cal.json ({s.label(rs)}): the half-width of every "
            "candidate's window (rt_im_train.adaptive_rt_window = false)",
        )
    windows = s.artifact("run_windows")
    if windows is None or not windows.usable or cids.size == 0:
        return np.full(cids.size, np.nan), "not available: no readable run_windows table"
    try:
        handle = _dense(windows)
        local = cids - int(s.offset)
        ok = (local >= 0) & (local < handle.num_rows)
        hw = np.full(cids.size, np.nan)
        if ok.any():
            got = _take(handle, local[ok], ["rt_lo", "rt_hi"])
            hw[ok] = (got["rt_hi"].astype(np.float64) - got["rt_lo"].astype(np.float64)) / 2.0
        hw[~np.isfinite(hw)] = np.nan
    except (ViewerError, OSError) as exc:
        return np.full(cids.size, np.nan), f"not available: {exc}"
    return hw, f"(rt_hi - rt_lo) / 2 of each candidate's row of {_rel(rs, windows.path)}"


def accepted_errors(
    rs: ResultSet, run: Run | str | int, threshold: float = 0.01, band: str | None = None
) -> AcceptedErrors:
    """The accepted target identifications of a run (or band) with their RT and mass errors.

    Accepted: label ``target`` and ``q_value <= threshold`` in a single run;
    ``run_psm_q <= threshold`` on the run's rows of the pooled table in an experiment
    (the PSM-level q within the run; the grouped q columns are experiment-wide). In a
    band: the accepted rows whose candidate lies in the band's library rows. Each row is
    joined to its feature row (``candidate_id``, ``peak_rank = selected_peak_rank``).
    """
    s = scope(rs, run, band)
    q_column = _q_column(rs)
    t = check_threshold("run_psm" if rs.is_experiment else "psm", threshold)
    ceiling = max(LOAD_CEILING, t)
    base = rs.memo((_MEMO, "accepted", s.run.index, s.key, ceiling), lambda: _load(rs, s, ceiling))
    frame = base.frame
    if base.error is None and len(frame):
        frame = frame[frame["q"].to_numpy() <= t].reset_index(drop=True)
    counts = {
        "rows": len(frame),
        "no_feature_row": int(frame["no_row"].sum()) if "no_row" in frame else 0,
    }
    for c in ("rt_error", *MASS_COLUMNS):
        counts[f"nan_{c}"] = int(frame[c].isna().sum()) if c in frame else 0
    population = f"target rows with {q_column} ≤ {t:g}"
    if rs.is_experiment:
        population += f" in run {s.run.name} (its rows of {rs.scored.path.name})"
    if s.band is not None:
        population += f", candidates of band {s.band.name}"
    return AcceptedErrors(
        scope=s,
        threshold=t,
        q_column=q_column,
        frame=frame,
        counts=counts,
        labels=base.labels,
        half_width_source=base.half_width_source,
        population=population,
        notes=list(base.notes),
        error=base.error,
    )


def _labels() -> dict[str, str]:
    labels = {
        "apex_rt": "apex_rt of the scored row (seconds)",
        "rt_error": "rt_error_signed of the feature table: apex_rt - rt_pred_cal (seconds)",
        "half_width": "the candidate's RT half-window (seconds)",
        "rt_error_rel": "viewer-derived: rt_error_signed / the candidate's half-window",
    }
    for c in MASS_COLUMNS:
        info = EVIDENCE_FEATURES.get(c)
        text = info.label if info is not None else c
        labels[c] = (
            f"{c} of the feature table ({text}): raw ppm, centred on the run's frag_ppm_offset, "
            "not on 0"
        )
    return labels


def _load(rs: ResultSet, s: Scope, ceiling: float) -> AcceptedErrors:
    labels = _labels()
    q_column = _q_column(rs)
    try:
        paths, names = _feature_names(rs, s)
        sql, params, rules = _accepted_sql(rs, s, paths, names, ceiling)
        df = rs.duck.df(sql, params)
    except (ViewerError, OSError, ValueError, duckdb.Error) as exc:
        empty = pd.DataFrame(columns=list(_FRAME_COLUMNS))
        return AcceptedErrors(s, ceiling, q_column, empty, {}, labels, "", "", [], str(exc))
    notes = []
    absent = [c for c in _ERROR_COLUMNS if c not in names]
    if absent:
        notes.append(f"The feature table has no {', '.join(absent)} (a smaller features.set).")
    for c, rule in rules.items():
        labels[f"valid_{c}"] = rule
    frame = pd.DataFrame(
        {
            "candidate_id": df["candidate_id"].to_numpy(dtype=np.int64),
            "peptidoform": df["peptidoform"].astype(object).to_numpy(),
            "charge": df["charge"].to_numpy(dtype=np.int64),
            "apex_rt": df["apex_rt"].to_numpy(dtype=np.float64),
            "q": df["q"].to_numpy(dtype=np.float64),
            "precursor_mz": pd.to_numeric(df["precursor_mz"], errors="coerce").to_numpy(np.float64),
        }
    )
    frame["no_row"] = (
        df["rt_error_signed"].isna().to_numpy() if "rt_error_signed" in names else True
    )
    for c, out in (("rt_error_signed", "rt_error"), *((m, m) for m in MASS_COLUMNS)):
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        valid = df[f"valid_{c}"].fillna(False).to_numpy(dtype=bool)
        frame[out] = np.where(valid, values, np.nan)
    hw, hw_source = _half_widths(rs, s, frame["candidate_id"].to_numpy(dtype=np.int64))
    frame["half_width"] = hw
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["rt_error_rel"] = frame["rt_error"].to_numpy() / hw
    if rs.is_experiment and mbr_ran(rs):
        notes.append(
            "Native identifications of scored_combined.parquet; match-between-runs transfers "
            "are not included."
        )
    return AcceptedErrors(s, ceiling, q_column, frame, {}, labels, hw_source, "", notes)
