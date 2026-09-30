"""Feature rows of a run: the column list, one candidate's values and percentile ranks.

The features stage writes one row per ``(candidate_id, peak_rank)`` to
``features.parquet``: 11 bookkeeping columns (:data:`BOOKKEEPING`) followed by the
active feature set, listed in ``features.parquet.schema.json``. ``psms_competed.parquet``
has the same columns (it is often a hard link of the features table). In a grouped run
the band ``features.parquet`` files are deleted after pooling. The feature rows are then
read from the root ``psms_competed.parquet`` when the run has one (the engine always
writes it when the bands overlap), else from the band ``groups/gNN/psms_competed.parquet``
tables, which must then hold disjoint candidates.

Rules this module follows:

* A candidate's rows are found with :class:`CandidateIndex`; the rank is selected by an
  explicit ``peak_rank`` equality, never by position.
* Percentiles are computed in DuckDB over the rows of the scored peaks only: feature
  rows joined to the same run's scored rows on ``candidate_id`` and
  ``peak_rank = selected_peak_rank``, so unselected alternative peaks are excluded. They
  are not FDR-filtered, and every result names its population.
* Sentinel values are not ranked: each evidence feature carries a validity predicate
  (:data:`EVIDENCE_FEATURES`), taken from the engine definitions. When the feature table
  lacks the inputs of a rule (``features.set`` minimal or rich), an equivalent rule over
  ``psms_extracted.parquet`` is used where one exists. Otherwise the value is not ranked,
  unless the caller asks for a percentage over the unfiltered rows, which is then
  labelled as such.
"""

from __future__ import annotations

import itertools
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pyarrow as pa

from .artifacts import Artifact
from .candidate_index import CandidateIndex, concat_parts, read_candidate_rows
from .discovery import GroupedLayout, ResultSet, Run
from .duck import sql_ident, sql_path
from .errors import ArtifactNotFound, InconsistentData, ViewerError
from .reports import load_json, normalise_enum

# The columns before the feature columns, in file order (features v2, psms_competed v4).
BOOKKEEPING: tuple[str, ...] = (
    "candidate_id",
    "peak_rank",
    "label",
    "base_peptide_id",
    "peptidoform",
    "protein",
    "apex_rt",
    "elution_lo",
    "elution_hi",
    "precursor_mz",
    "prelim_score",
)

# Feature columns always read with a projection: the inputs of the validity rules and
# the context the evidence panel shows beside the values.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "n_observations",
    "n_peak_scans",
    "peak_window_degenerate",
    "n_matched_b",
    "n_matched_y",
    "predicted_rt_raw",
    "has_ms1",
)

_SQL_WORDS = frozenset({"and", "or", "not", "is", "null", "true", "false", "in", "between"})
# A name in a predicate; a name followed by "(" is a function (isfinite), not a column.
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b(?!\s*\()")


def predicate_columns(predicate: str | None) -> tuple[str, ...]:
    """The column names a validity predicate reads (function names are skipped)."""
    if not predicate:
        return ()
    names = [w for w in _IDENT.findall(predicate) if w.lower() not in _SQL_WORDS]
    return tuple(dict.fromkeys(names))


def _qualify(predicate: str, alias: str) -> str:
    """``predicate`` with every column name written as ``alias."name"``."""

    def column(match: re.Match[str]) -> str:
        word = match.group(0)
        return word if word.lower() in _SQL_WORDS else f"{alias}.{sql_ident(word)}"

    return _IDENT.sub(column, predicate)


@dataclass(frozen=True)
class FeatureInfo:
    """Definition of one evidence feature, as the MuMDIA 0.5.0 engine computes it.

    ``validity`` is a DuckDB predicate over the feature table that is true where the
    value is a measurement and not a sentinel (None: every value is valid). Percentiles
    rank only the rows where it holds. ``extracted_validity`` is the same rule over the
    columns of ``psms_extracted.parquet``, used when the feature table lacks the inputs
    of ``validity`` (the smaller feature sets). ``percentile`` is False for flags, which
    have no meaningful percentile. ``higher_is_better`` is None for columns that are not
    a monotone quality score; show those without good or bad colouring. ``table`` names
    the artifact that holds the column.
    """

    name: str
    label: str
    unit: str
    description: str
    validity: str | None
    note: str
    table: str = "features"
    higher_is_better: bool | None = None
    percentile: bool = True
    extracted_validity: str | None = None

    @property
    def validity_columns(self) -> tuple[str, ...]:
        """The columns the validity predicate reads."""
        return predicate_columns(self.validity)

    @property
    def extracted_validity_columns(self) -> tuple[str, ...]:
        """The ``psms_extracted`` columns the fallback predicate reads."""
        return predicate_columns(self.extracted_validity)


_APEX = "the apex scan (the grid scan nearest apex_rt)"
_RAW_PPM = (
    "Uncorrected (raw) ppm: the run's values centre on its frag_ppm_offset (extract report "
    "params), not on 0. "
)
_NO_APEX_ZERO = (
    "The engine writes 0 when no fragment is observed at the apex scan; real zeros also exist."
)
_COELUTION_NOTE = (
    "The window holds n_observations grid scans (1 to 4 on Astral data). The engine writes 0 "
    "when n_observations < 2, which collides with a real 0. With 2 points every non-constant "
    "pair has r = +1 or -1, so windows with n_observations <= 2 or peak_window_degenerate = 1 "
    "are degenerate. Percentiles rank rows with n_observations >= 3."
)
_RT_NOTE = (
    "The engine writes 0 when rt_pred_cal is not finite (no RT calibration, for example a band "
    "with calibration_status insufficient_anchors_unbounded); such rows fail the validity rule. "
    "Without predicted_rt_raw (features.set minimal or rich) the rule reads rt_pred_cal of "
    "psms_extracted.parquet. Extraction searches only inside [rt_lo, rt_hi], so the error is "
    "truncated at the window."
)
_RT_FINITE = "isfinite(rt_pred_cal)"
_CONTESTED_NOTE = (
    "Computed only on the two-pass extraction path: extract.peak_claim = coelution_* or "
    "extract.emit_contested_features = true. Otherwise the engine writes 0 on every row, which "
    "means 'not computed', never '0% interference'. Even when computed, 0 also means that no "
    "intensity was lost."
)
_APEX_OBSERVED = "(n_matched_b + n_matched_y) >= 1"

_EVIDENCE: tuple[FeatureInfo, ...] = (
    FeatureInfo(
        "frag_corr",
        "Apex intensity correlation with library",
        "Pearson r",
        f"Pearson correlation between the observed intensity of each predicted fragment at "
        f"{_APEX} and its predicted intensity, over all predicted fragments. A fragment not "
        "observed there counts as 0.",
        None,
        "The extraction gate (extract.gate_mode = apex_pearson, extract.gate_min_score, default "
        "0.2) rejects candidates on this quantity, so the distribution starts at the gate. The "
        "engine writes 0 with fewer than 2 fragments or zero variance.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "frag_cosine",
        "Apex intensity cosine with library",
        "cosine",
        f"Cosine similarity between the observed fragment intensities at {_APEX} and the "
        "predicted intensities, over all predicted fragments. A fragment not observed there "
        "counts as 0. Range 0 to 1.",
        None,
        "0 only when no fragment is observed at the apex scan.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "spectral_angle",
        "Spectral angle, all predicted fragments",
        "normalized angle [0, 1]",
        "Normalized spectral contrast angle, 1 - 2*acos(cosine)/pi, between the observed "
        f"fragment intensities at {_APEX} and the predicted intensities, over all predicted "
        "fragments. A fragment not observed there counts as 0. 1 means identical, 0 means "
        "orthogonal.",
        None,
        "Not radians and not 1 - angle. Unobserved strong predicted ions lower it. 0 when no "
        "fragment is observed at the apex scan.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "spectral_angle_matched",
        "Spectral angle, fragments observed at apex",
        "normalized angle [0, 1]",
        "The same angle as spectral_angle, over only the fragments observed at the apex scan "
        "(observed intensity > 0), paired with their predicted intensities.",
        "(n_matched_b + n_matched_y) >= 2",
        "Exactly 1 when one fragment is observed at the apex scan (a one-element vector is "
        "parallel) and 0 when none is, so it is n/a with fewer than 2 fragments at the apex scan.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "coelution_mean",
        "Mean fragment co-elution (elution window)",
        "Pearson r",
        "Mean, over all pairs of predicted fragments, of the population Pearson correlation of "
        "their traces inside the feature elution window [elution_lo, elution_hi]. Every predicted "
        "fragment enters; an unobserved or constant trace gives r = 0 against every partner.",
        "n_observations >= 3",
        _COELUTION_NOTE,
        higher_is_better=True,
    ),
    FeatureInfo(
        "coelution_best",
        "Best fragment-pair co-elution (elution window)",
        "Pearson r",
        "The largest pairwise Pearson correlation of fragment traces inside the feature elution "
        "window [elution_lo, elution_hi], floored at 0.",
        "n_observations >= 3",
        _COELUTION_NOTE + " A value of 1 is common in small windows (63% of targets and 60% of "
        "decoys on Astral data): it is a small-window artefact, not evidence.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "coelution_run",
        "Longest co-elution run (scans with >= 2 fragments)",
        "scans",
        "The longest run of consecutive grid scans in the whole RT window [rt_lo, rt_hi] that "
        "each have at least extract.presence_min_coelution (default 2) distinct matched "
        "fragments.",
        None,
        "At least extract.fixed_scan_window (default 3) by acceptance. Multiply by the scan "
        "spacing for seconds. On peak_rank >= 1 rows it is the candidate-level (rank 0) value.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "n_matched_fragments",
        "Matched fragments (in RT window)",
        "count",
        "The number of distinct predicted fragments with at least one matched peak in any scan "
        "of the RT window [rt_lo, rt_hi].",
        None,
        "Counted over the whole RT window, not at the apex scan (that count is n_matched_b + "
        "n_matched_y). On peak_rank >= 1 rows it counts the fragments inside the alternative "
        "peak's envelope.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "n_matched_b",
        "b fragments observed at apex scan",
        "count",
        "The number of b-ion fragments observed at the apex scan (observed intensity > 0).",
        None,
        "n_matched_b + n_matched_y is the number of fragments observed at the apex scan.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "n_matched_y",
        "Non-b fragments observed at apex scan",
        "count",
        "The number of fragments other than b ions (y ions) observed at the apex scan (observed "
        "intensity > 0).",
        None,
        "n_matched_b + n_matched_y is the number of fragments observed at the apex scan.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "median_abs_frag_ppm",
        "Median |fragment mass error| (uncorrected)",
        "ppm (raw)",
        "The median absolute fragment mass error over the fragments observed at the apex scan. "
        "Each fragment's error is 1e6 * (frag_obs_mz - frag_mz) / frag_mz, where frag_obs_mz is "
        "the intensity-weighted mean m/z of its matched peaks over the whole RT window.",
        _APEX_OBSERVED,
        _RAW_PPM + _NO_APEX_ZERO + " An even count takes the mean of the two middle values.",
        higher_is_better=False,
    ),
    FeatureInfo(
        "signed_mean_frag_ppm",
        "Mean fragment mass error (uncorrected)",
        "ppm (raw)",
        "The mean signed fragment mass error over the fragments observed at the apex scan "
        "(per-fragment m/z averaged over the RT window).",
        _APEX_OBSERVED,
        _RAW_PPM + _NO_APEX_ZERO,
    ),
    FeatureInfo(
        "frag_mass_err_median",
        "Median fragment mass error (uncorrected)",
        "ppm (raw)",
        "The median signed fragment mass error over the fragments observed at the apex scan "
        "(per-fragment m/z averaged over the RT window; linear-interpolated percentile 0.5).",
        _APEX_OBSERVED,
        _RAW_PPM + _NO_APEX_ZERO + " Closest to the run's frag_ppm_offset is best.",
    ),
    FeatureInfo(
        "weighted_mass_error",
        "Apex-weighted |fragment mass error| (uncorrected)",
        "ppm (raw)",
        "The absolute fragment mass error weighted by the apex-scan intensity: "
        "sum(|ppm| * apex intensity) / sum(apex intensity) over all predicted fragments, so "
        "unobserved fragments have weight 0.",
        _APEX_OBSERVED,
        _RAW_PPM + _NO_APEX_ZERO + " Equals the feature intensity_weighted_abs_ppm.",
        higher_is_better=False,
    ),
    FeatureInfo(
        "mean_mass_error",
        "Mean |ppm| over all predicted fragments (unobserved = 0)",
        "ppm (raw)",
        "The mean absolute fragment mass error over all K predicted fragments, sum(|ppm|) / K, "
        "where an unobserved fragment counts as 0 ppm.",
        None,
        "Biased low: unobserved fragments count as 0 ppm. It is not the mass accuracy of the "
        "matched fragments; do not show it as the mass error. " + _RAW_PPM.strip(),
        higher_is_better=False,
    ),
    FeatureInfo(
        "has_ms1",
        "MS1 data available",
        "flag",
        "1 when the candidate's MS1 values (ms1_mono, ms1_iso1, ms1_iso2) are non-null, that is "
        "when the run has MS1 spectra; else 0.",
        None,
        "It flags MS1 data availability, not MS1 signal: it is 1 on every Astral row, also where "
        "ms1_mono = 0. Use has_ms1_signal or log_ms1_mono for the signal. A flag has no "
        "percentile.",
        percentile=False,
    ),
    FeatureInfo(
        "has_ms1_signal",
        "MS1 monoisotopic signal at apex",
        "flag",
        "1 when ms1_mono > 0: at least one MS1 peak within extract.prec_tol_ppm of the "
        "monoisotopic m/z in the MS1 scan nearest apex_rt.",
        None,
        "A flag has no percentile.",
        higher_is_better=True,
        percentile=False,
    ),
    FeatureInfo(
        "log_ms1_mono",
        "MS1 monoisotopic intensity at apex",
        "ln(1 + intensity)",
        "ln(1 + ms1_mono), where ms1_mono is the summed intensity of the MS1 peaks within "
        "extract.prec_tol_ppm of the monoisotopic m/z in the MS1 scan nearest apex_rt (null read "
        "as 0).",
        "has_ms1 = 1",
        "0 means no monoisotopic peak within the tolerance, or no MS1 data (has_ms1 = 0, which "
        "fails the validity rule). It equals the feature log_mono_ms1.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "ms1_isotope_cosine_apex",
        "Isotope pattern vs averagine (apex MS1 scan)",
        "cosine",
        "The cosine between the MS1 intensities [monoisotopic, +1, +2] in the MS1 scan nearest "
        "apex_rt and a Poisson averagine isotope envelope for the precursor mass.",
        "ms1_isotope_cosine_apex > 0",
        "0 if and only if all three MS1 intensities are 0: no isotope signal at the apex. "
        "Percentiles rank rows with a value > 0.",
        higher_is_better=True,
    ),
    FeatureInfo(
        "n_interfered_fragments",
        "Fragments above 2x their profile share at apex",
        "count",
        "The number of fragments whose apex-scan intensity exceeds twice their share of the "
        "shared elution profile (the least-squares scale of the fragment's trace onto the "
        "predicted-weighted sum of traces, inside the elution window).",
        _APEX_OBSERVED,
        "0 both for 'no outlier' and for 'no fragment at the apex scan'. Not directional on "
        "Astral data: do not colour it as good or bad.",
    ),
    FeatureInfo(
        "interference_apex_residual_fraction",
        "Apex intensity not explained by the shared profile",
        "fraction",
        "The share of the apex-scan fragment intensity that the scaled shared elution profile "
        "does not explain: sum(max(0, x_f - r_f * R)) / sum(x_f) at the apex point of the elution "
        "window.",
        _APEX_OBSERVED,
        "0 when there is no apex signal, and also a real 0 when every trace is proportional to the "
        "profile (one contributing fragment, or a one-point window). Not directional on Astral "
        "data: do not colour it as good or bad.",
    ),
    FeatureInfo(
        "rt_error_signed",
        "RT error (apex - predicted)",
        "s",
        "apex_rt - rt_pred_cal, in seconds; positive when the peptide eluted later than predicted.",
        "predicted_rt_raw <> 0",
        _RT_NOTE,
        extracted_validity=_RT_FINITE,
    ),
    FeatureInfo(
        "rt_error_abs",
        "|RT error|",
        "s",
        "|apex_rt - rt_pred_cal|, in seconds.",
        "predicted_rt_raw <> 0",
        _RT_NOTE,
        higher_is_better=False,
        extracted_validity=_RT_FINITE,
    ),
    FeatureInfo(
        "rt_error_over_peak_width",
        "|RT error| / feature-window profile width",
        "ratio",
        "|apex_rt - rt_pred_cal| divided by the width of the feature-window profile at 10% of its "
        "apex height. This width is not a chromatographic peak width.",
        "rt_error_over_peak_width > 0",
        "The engine writes 0 when the width cannot be computed, that is with fewer than 3 window "
        "points or when the walk down to 10% of the apex height stays on one point (25% to 26% of "
        "the rows of the Astral runs checked), and when rt_pred_cal is not finite.",
        higher_is_better=False,
    ),
    FeatureInfo(
        "n_observations",
        "Scans in the elution window",
        "scans",
        "The number of grid scans in the feature elution window [elution_lo, elution_hi].",
        None,
        "1 to 4 on Astral data. The window half-widths are learned per run from confident anchors "
        "and are not recorded in any artifact.",
    ),
    FeatureInfo(
        "peak_window_degenerate",
        "Degenerate elution window",
        "flag",
        "1 when fewer than 3 scans of the elution window carry fragment signal (n_peak_scans < 3).",
        None,
        "A flag has no percentile.",
        percentile=False,
    ),
    FeatureInfo(
        "contested_frac",
        "Matched intensity lost to co-eluting candidates",
        "fraction",
        "lost / (won + lost): the fraction of the candidate's matched peak intensity in its RT "
        "window that a better-eluting co-claimant took in the two-pass extraction. Every matched "
        "peak counts; a peak with one claimant is won.",
        None,
        _CONTESTED_NOTE,
        table="psms_extracted",
        higher_is_better=False,
    ),
    FeatureInfo(
        "peak_contested_frac",
        "Matched intensity lost to co-eluting candidates",
        "fraction",
        "The contested_frac of psms_extracted, copied into the feature table.",
        None,
        _CONTESTED_NOTE,
        higher_is_better=False,
    ),
    FeatureInfo(
        "log_apex_intensity",
        "Apex fragment intensity",
        "ln(1 + intensity)",
        "ln(1 + apex_intensity), where apex_intensity is the summed intensity of the fragments "
        "observed at the apex scan.",
        "log_apex_intensity > 0",
        "0 marks the apex fallback: no scan qualified, apex_rt is the first scan of the RT window "
        "and apex_intensity is 0. exp(value) - 1 recovers apex_intensity.",
        higher_is_better=True,
    ),
)

# The evidence summary set: definitions from the engine source, verified on data.
EVIDENCE_FEATURES: dict[str, FeatureInfo] = {f.name: f for f in _EVIDENCE}

_CONTESTED = frozenset({"contested_frac", "peak_contested_frac"})

SourceKind = Literal["features", "psms_competed", "band_psms_competed"]


@dataclass(frozen=True)
class FeatureSource:
    """The table or tables that hold a run's feature rows.

    ``kind`` is ``features`` (``features.parquet``), ``psms_competed`` (the run's
    ``psms_competed.parquet``: the pooled table of a grouped run, or the fallback when
    ``features.parquet`` is absent) or ``band_psms_competed`` (one table per band, used
    only when the run has no pooled table and no overlap losers, and checked to hold
    disjoint ``candidate_id`` ranges). ``bands`` names the band of each table.
    """

    run: str
    kind: SourceKind
    artifacts: tuple[Artifact, ...]
    bands: tuple[str, ...]
    missing_bands: tuple[str, ...]
    description: str

    @property
    def paths(self) -> list[Path]:
        return [a.require() for a in self.artifacts]


@dataclass(frozen=True)
class FeaturePercentile:
    """One candidate value ranked against the target and decoy rows of its run.

    ``pct_target`` and ``pct_decoy`` are percentages (0 to 100): the share of the
    population rows of that label whose value is <= ``value``. They are None when the
    value is not ranked (``note`` says why). ``n_target`` and ``n_decoy`` count the rows
    of the population (0 when nothing was counted). ``population`` names these rows:
    when ``rule`` is set they are the valid rows, the rows that pass that validity rule;
    otherwise no rule was applied and ``population`` says why. ``valid`` says whether
    the candidate's own value passes the feature's validity rule: None when the rule
    could not be checked (its inputs were not given, or the table lacks them).
    """

    feature: str
    value: float | None
    pct_target: float | None
    pct_decoy: float | None
    n_target: int
    n_decoy: int
    valid: bool | None
    population: str
    note: str | None = None
    rule: str | None = None


# --------------------------------------------------------------------------- sources


def _run(rs: ResultSet, run: Run | str | int) -> Run:
    return run if isinstance(run, Run) else rs.run(run)


def _run_name(rs: ResultSet, run: Run) -> str:
    return run.name or rs.root.name


_SCHEMAS: dict[tuple[str, int, int], pa.Schema] = {}


def _schema(artifact: Artifact) -> pa.Schema:
    """The Arrow schema of an artifact, cached by (path, size, mtime_ns)."""
    handle = artifact.parquet()
    key = (str(handle.path), *handle.stamp)
    schema = _SCHEMAS.get(key)
    if schema is None:
        schema = handle.schema
        _SCHEMAS[key] = schema
    return schema


_RANGES: dict[tuple[str, int, int], tuple[tuple[int, int] | None, ...]] = {}


def _id_ranges(artifact: Artifact) -> tuple[tuple[int, int] | None, ...]:
    """Footer min/max of ``candidate_id`` per row group (None where not recorded), memoised."""
    handle = artifact.parquet()
    key = (str(handle.path), *handle.stamp)
    ranges = _RANGES.get(key)
    if ranges is None:
        ranges = tuple(
            None if s is None else (int(s[0]), int(s[1]))
            for s in handle.column_statistics("candidate_id")
        )
        _RANGES[key] = ranges
    return ranges


def _loser_rows(layout: GroupedLayout) -> int | None:
    """Rows of ``groups/overlap_losers.parquet``; None when the run has none or it is unreadable."""
    losers = layout.losers
    if losers is None or not losers.present:
        return None
    if losers.rows is not None:
        return int(losers.rows)
    try:
        return int(losers.parquet().num_rows)
    except (ViewerError, OSError, pa.ArrowException):
        return None


def _check_disjoint(rs: ResultSet, label: str, tables: list[Artifact], bands: list[str]) -> None:
    """Refuse band tables whose ``candidate_id`` ranges overlap.

    Band ids are library-wide and lie inside the band's library row span, and the
    engine pools the competed table whenever two spans overlap. Band tables of a run
    without a pooled table therefore hold disjoint id ranges; overlapping ranges would
    count the shared candidates twice.
    """
    spans: list[tuple[int, int, str]] = []
    for artifact, band in zip(tables, bands, strict=True):
        ranges = _id_ranges(artifact)
        if not ranges:
            continue  # an empty band table has no row group
        known = [r for r in ranges if r is not None]
        if len(known) == len(ranges):
            spans.append((min(r[0] for r in known), max(r[1] for r in known), band))
            continue
        ids = CandidateIndex.for_artifact(artifact, rs.cache).ids
        if ids.size:
            spans.append((int(ids.min()), int(ids.max()), band))
    spans.sort()
    for (lo1, hi1, b1), (lo2, hi2, b2) in itertools.pairwise(spans):
        if lo2 <= hi1:
            raise InconsistentData(
                f"run {label}: the band tables {b1} and {b2} hold overlapping candidate_id ranges "
                f"({lo1} to {hi1} and {lo2} to {hi2}), but the run has no pooled "
                "psms_competed.parquet and no overlap losers. A candidate may then have feature "
                "rows in two bands, so the band tables are not read."
            )


def _grouped_source(rs: ResultSet, run: Run, layout: GroupedLayout, label: str) -> FeatureSource:
    """The feature source of a grouped run (see :func:`feature_source`)."""
    pooled = run.artifact("psms_competed")
    if pooled is not None:
        if pooled.usable:
            return FeatureSource(
                run.name,
                "psms_competed",
                (pooled,),
                (),
                (),
                "psms_competed.parquet (pooled over the bands)",
            )
        # Recorded in the manifest or found on disk, but not readable. The band tables
        # are no substitute: under overlap they hold some candidates twice.
        try:
            pooled.require()
        except ViewerError as exc:
            raise type(exc)(
                f"run {label}: the feature rows of this grouped run are in its pooled "
                f"psms_competed.parquet, which cannot be read ({exc}). The band tables are not "
                "read in its place."
            ) from exc
    n_losers = _loser_rows(layout)
    if n_losers:
        raise ArtifactNotFound(
            f"run {label}: the bands overlap (groups/overlap_losers.parquet has {n_losers} rows). "
            "The engine then always writes a pooled psms_competed.parquet, but the run has none. "
            "The band tables hold the overlap candidates twice, so they are not read in its place."
        )
    tables: list[Artifact] = []
    bands: list[str] = []
    missing: list[str] = []
    for band in layout.bands:
        a = band.artifact("psms_competed")
        if a is not None and a.usable:
            tables.append(a)
            bands.append(band.name)
        elif a is not None and a.present:
            a.require()  # raises the band table's own error (for example its schema version)
        else:
            missing.append(band.name)
    if not tables:
        raise ArtifactNotFound(
            f"run {label}: the run has no pooled psms_competed.parquet and no band has a "
            "readable groups/gNN/psms_competed.parquet, so the feature rows cannot be read."
        )
    _check_disjoint(rs, label, tables, bands)
    text = (
        f"groups/gNN/psms_competed.parquet ({len(tables)} band tables; the run has no pooled "
        "psms_competed.parquet"
    )
    if missing:
        text += f"; band tables not found: {', '.join(missing)}"
    return FeatureSource(
        run.name, "band_psms_competed", tuple(tables), tuple(bands), tuple(missing), text + ")"
    )


def feature_source(rs: ResultSet, run: Run | str | int) -> FeatureSource:
    """The artifact or artifacts holding the feature rows of ``run``.

    Ungrouped run: ``features.parquet``, else ``psms_competed.parquet`` (same columns).
    Grouped run: the root ``psms_competed.parquet`` when the run has one; the engine
    always writes it when the bands overlap. When it is recorded or present but not
    readable (missing file, unsupported schema version), its own error is raised. When
    the run has no pooled table, ``groups/overlap_losers.parquet`` must have no rows,
    and the band ``psms_competed.parquet`` tables are read after a check that their
    ``candidate_id`` ranges are disjoint.
    """
    run = _run(rs, run)
    label = _run_name(rs, run)
    if run.grouped is not None:
        return _grouped_source(rs, run, run.grouped, label)
    features = run.artifact("features")
    if features is not None and features.usable:
        return FeatureSource(run.name, "features", (features,), (), (), "features.parquet")
    competed = run.artifact("psms_competed")
    if competed is not None and competed.usable:
        why = "absent"
        if features is not None and features.error is not None:
            why = f"not readable: {features.error}"
        return FeatureSource(
            run.name,
            "psms_competed",
            (competed,),
            (),
            (),
            f"psms_competed.parquet (features.parquet is {why})",
        )
    for a in (features, competed):
        if a is not None and a.present:
            a.require()  # raises the artifact's own error (version, layout)
    raise ArtifactNotFound(
        f"run {label}: neither features.parquet nor psms_competed.parquet exists, so the feature "
        "rows cannot be read."
    )


def _schema_json_columns(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    data = load_json(path.with_name(path.name + ".schema.json"))
    if data is None:
        return None
    cols = data.get("feature_columns")
    if isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols):
        return list(cols)
    return None


def feature_columns(rs: ResultSet, run: Run | str | int) -> list[str]:
    """The feature column names of ``run``, in file order.

    From ``<table>.schema.json`` ``feature_columns`` (for band tables, the first band
    that has one), restricted to the columns the parquet footer has; without a schema
    file, the footer columns minus :data:`BOOKKEEPING`.
    """
    source = feature_source(rs, run)
    names = _schema(source.artifacts[0]).names
    present = set(names)
    for artifact in source.artifacts:
        listed = _schema_json_columns(artifact.path)
        if listed is not None:
            return [c for c in listed if c in present]
    return [c for c in names if c not in BOOKKEEPING]


# --------------------------------------------------------------------------- one candidate


def _may_hold(artifact: Artifact, cid: int) -> bool:
    """False when the footer statistics show that no row group holds ``cid``."""
    ranges = _id_ranges(artifact)
    if not ranges:
        return False
    return any(r is None or r[0] <= cid <= r[1] for r in ranges)


def _index_key(index: CandidateIndex, cid: int) -> Any | None:
    """``cid`` as a scalar of the index's own dtype, or None when it cannot be an id.

    numpy converts a whole uint32 array to int64 when it is searched with a Python int
    (about 1 ms per lookup on 700,000 ids); a scalar of the array's dtype avoids that.
    """
    dtype = index.ids.dtype
    if not np.issubdtype(dtype, np.integer):
        return cid
    info = np.iinfo(dtype)
    if not int(info.min) <= cid <= int(info.max):
        return None
    return dtype.type(cid)


def _locate(
    rs: ResultSet, source: FeatureSource, cid: int
) -> tuple[Artifact, CandidateIndex, Any, str | None] | None:
    """The table that holds ``cid``: artifact, index, typed index key, band name."""
    candidates = list(zip(source.artifacts, source.bands or (None,), strict=True))
    hits: list[tuple[Artifact, CandidateIndex, Any, str | None]] = []
    for artifact, band in candidates:
        if band is not None and not _may_hold(artifact, cid):
            continue
        index = CandidateIndex.for_artifact(artifact, rs.cache)
        key = _index_key(index, cid)
        if key is not None and key in index:
            hits.append((artifact, index, key, band))
    if len(hits) > 1:
        raise InconsistentData(
            f"candidate {cid} has feature rows in {len(hits)} band tables "
            f"({', '.join(str(h[3]) for h in hits)}) of {source.description}; without a pooled "
            "table each candidate must be in one band table."
        )
    return hits[0] if hits else None


def _projection(names: Sequence[str], columns: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    present = set(names)
    wanted = [*BOOKKEEPING, *CONTEXT_COLUMNS, *columns]
    return [c for c in dict.fromkeys(wanted) if c in present]


def candidate_feature_rows(
    rs: ResultSet, run: Run | str | int, cid: int, columns: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Every feature row of candidate ``cid`` (one per peak rank), in file order.

    ``columns=None`` reads every column. Otherwise the rows hold :data:`BOOKKEEPING`,
    :data:`CONTEXT_COLUMNS` and the requested columns, where the table has them; a
    column the table lacks is left out of the dict (test with ``name in row``). An
    empty list means the candidate has no feature row in this run.
    """
    source = feature_source(rs, run)
    cid = int(cid)
    hit = _locate(rs, source, cid)
    if hit is None:
        return []
    artifact, index, key, _ = hit
    handle = artifact.parquet()
    proj = _projection(_schema(artifact).names, columns)
    table = concat_parts(read_candidate_rows(handle, index, key, proj, cached=False))
    if table is None or table.num_rows == 0:
        return []
    rows = table.to_pylist()
    wrong = {r.get("candidate_id") for r in rows} - {cid}
    if wrong:
        raise InconsistentData(
            f"{artifact.path}: the candidate index of {cid} points at rows of candidate(s) "
            f"{sorted(wrong)}."
        )
    return rows


def candidate_features(
    rs: ResultSet,
    run: Run | str | int,
    cid: int,
    peak_rank: int = 0,
    columns: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """The feature row of ``(cid, peak_rank)``, or None when the run has no such row.

    For the identification's own peak pass ``peak_rank = selected_peak_rank`` of the
    scored row. The row is picked by an explicit ``peak_rank`` equality (a table
    without the column has rank 0 only). See :func:`candidate_feature_rows` for
    ``columns``.
    """
    rows = candidate_feature_rows(rs, run, cid, columns)
    found = [r for r in rows if int(r.get("peak_rank") or 0) == int(peak_rank)]
    if len(found) > 1:
        raise InconsistentData(
            f"candidate {cid} has {len(found)} feature rows with peak_rank {peak_rank}."
        )
    return found[0] if found else None


# --------------------------------------------------------------------------- percentiles


def contested_computed(rs: ResultSet) -> tuple[bool, str]:
    """Whether the engine computed ``contested_frac`` in this result set, with the reason."""
    emit = rs.config_get("extract", "emit_contested_features", default=False)
    claim = rs.config_get("extract", "peak_claim", default="none")
    setting = (
        f"extract.peak_claim = {claim}, extract.emit_contested_features = {str(bool(emit)).lower()}"
    )
    if bool(emit) or normalise_enum(claim).startswith("coelution"):
        return True, f"computed ({setting})"
    return False, f"not computed ({setting}): the engine writes 0 on every row"


def _scored_for_run(rs: ResultSet, run: Run) -> Artifact:
    """The scored table whose rows define the run's population (the run's own when usable)."""
    own = run.artifact("psms_scored")
    if own is not None and own.usable:
        return own
    return rs.scored


def _stats_all_equal(artifact: Artifact, column: str, value: int) -> bool:
    """True when the footer statistics prove that every row holds ``value`` in ``column``.

    A table without the column counts as holding 0 (the engine's default rank).
    """
    handle = artifact.parquet()
    if not handle.has_column(column):
        return value == 0
    if handle.num_rows == 0:
        return True
    stats = handle.column_statistics(column)
    return bool(stats) and all(
        s is not None and int(s[0]) == value and int(s[1]) == value for s in stats
    )


_WHOLE_TABLE: dict[tuple[str, str, int], bool] = {}
_WHOLE_TABLE_LOCK = threading.Lock()


def _population_is_whole_table(
    rs: ResultSet, source: FeatureSource, scored: Artifact, run_index: int
) -> bool:
    """True when the scored-peak rows are provably every row of the feature table.

    That holds when every feature row has ``peak_rank`` 0 and there is one row per
    candidate, every scored row belongs to the run with ``selected_peak_rank`` 0, and
    both tables hold the same candidate ids. The join can then be skipped, which halves
    the cost of the percentile query on a large run. The verdict is memoised by the
    content hashes of the two tables.
    """
    if source.kind == "band_psms_competed":
        return False
    features = source.artifacts[0]
    key = (features.identity(), scored.identity(), int(run_index))
    with _WHOLE_TABLE_LOCK:
        if key in _WHOLE_TABLE:
            return _WHOLE_TABLE[key]
    verdict = False
    if (
        _stats_all_equal(features, "peak_rank", 0)
        and _stats_all_equal(scored, "selected_peak_rank", 0)
        and _stats_all_equal(scored, "source", int(run_index))
    ):
        f_index = CandidateIndex.for_artifact(features, rs.cache)
        s_index = CandidateIndex.for_artifact(scored, rs.cache)
        verdict = bool(
            f_index.contiguous
            and s_index.contiguous
            and f_index.num_rows == f_index.ids.size
            and s_index.num_rows == s_index.ids.size
            and np.array_equal(f_index.ids, s_index.ids)
        )
    with _WHOLE_TABLE_LOCK:
        _WHOLE_TABLE[key] = verdict
    return verdict


def _number(value: Any) -> float | None:
    """``value`` as a finite float, or None."""
    if value is None or isinstance(value, str | bytes):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _input(value: Any) -> float | None:
    """A validity-rule input as a float (NaN and infinities kept), or None when not given."""
    if value is None or isinstance(value, str | bytes):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric(pa_field: pa.Field) -> bool:
    t = pa_field.type
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)


_NON_NULL: dict[tuple[str, int, int], frozenset[str]] = {}


def _non_null_one(artifact: Artifact) -> frozenset[str]:
    """Top-level columns whose footer statistics record no null in any row group."""
    handle = artifact.parquet()
    key = (str(handle.path), *handle.stamp)
    hit = _NON_NULL.get(key)
    if hit is not None:
        return hit
    md = handle.metadata()
    leaves = {j: md.schema.column(j).path for j in range(md.num_columns)}
    nulls: dict[str, int | None] = dict.fromkeys(leaves.values(), 0)
    for rg in range(md.num_row_groups):
        group = md.row_group(rg)
        for j, name in leaves.items():
            count = nulls[name]
            if count is None:
                continue
            stats = group.column(j).statistics
            nulls[name] = (
                None if stats is None or not stats.has_null_count else count + int(stats.null_count)
            )
    result = frozenset(n for n, c in nulls.items() if c == 0 and "." not in n)
    _NON_NULL[key] = result
    return result


def _non_null_columns(artifacts: Sequence[Artifact]) -> frozenset[str]:
    """Columns without a null value in every one of ``artifacts`` (from the footers)."""
    out: frozenset[str] | None = None
    for artifact in artifacts:
        cols = _non_null_one(artifact)
        out = cols if out is None else out & cols
    return out or frozenset()


@dataclass(frozen=True)
class _Rule:
    """A validity rule as applied: the table whose columns it reads, its SQL, its label."""

    table: Literal["features", "psms_extracted"]
    sql: str
    text: str


@dataclass
class _Spec:
    """Working state of one feature in :func:`feature_percentiles`."""

    name: str
    value: float | None
    count: bool = True  # count the population (the column exists and is numeric)
    rank: bool = True  # compute the percentage (a rankable feature that is computed)
    rule: _Rule | None = None  # the validity rule applied to the population and the value
    missing_rule: str | None = None  # why the feature's validity rule cannot be applied
    valid: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ranked(self) -> bool:
        if not (self.count and self.rank) or self.value is None:
            return False
        if self.rule is not None:
            return self.valid is True
        return self.valid is not False


def _extracted_for(run: Run) -> Artifact | None:
    """The run's readable ``psms_extracted.parquet``, or None."""
    extracted = run.artifact("psms_extracted")
    return extracted if extracted is not None and extracted.usable else None


def _extracted_rule(info: FeatureInfo, run: Run) -> _Rule | None:
    """The ``psms_extracted`` form of the feature's validity rule, when the run can apply it."""
    extracted = _extracted_for(run)
    if info.extracted_validity is None or extracted is None:
        return None
    try:
        names = set(_schema(extracted).names)
    except ViewerError:
        return None
    if "candidate_id" not in names or not set(info.extracted_validity_columns) <= names:
        return None
    return _Rule(
        "psms_extracted",
        info.extracted_validity,
        f"{info.extracted_validity} in the psms_extracted.parquet row of the same candidate_id "
        "and peak_rank",
    )


def _extracted_row(
    rs: ResultSet, extracted: Artifact, cid: int, rank: int, columns: Sequence[str]
) -> dict[str, Any] | None:
    """The ``psms_extracted`` row of ``(cid, rank)`` with ``columns``, or None."""
    index = CandidateIndex.for_artifact(extracted, rs.cache)
    key = _index_key(index, cid)
    if key is None or key not in index:
        return None
    names = set(_schema(extracted).names)
    proj = [c for c in dict.fromkeys(["candidate_id", "peak_rank", *columns]) if c in names]
    table = concat_parts(read_candidate_rows(extracted.parquet(), index, key, proj, cached=False))
    rows = [] if table is None else table.to_pylist()
    found = [
        r for r in rows if r.get("candidate_id") == cid and int(r.get("peak_rank") or 0) == rank
    ]
    if len(found) > 1:
        raise InconsistentData(
            f"{extracted.path}: candidate {cid} has {len(found)} rows with peak_rank {rank}."
        )
    return found[0] if found else None


def _check_candidate(
    rs: ResultSet, run: Run, values: Mapping[str, Any], specs: list[_Spec], names: set[str]
) -> None:
    """Evaluate each validity rule on the candidate's own values (sets ``valid``).

    Inputs missing from ``values`` are read from the feature table, or from
    ``psms_extracted.parquet`` for a rule over its columns, when ``values`` names the
    candidate (``candidate_id``, ``peak_rank``). A rule that the population cannot
    apply is still checked on the value when ``values`` holds its inputs.
    """
    checks: list[tuple[_Spec, _Rule]] = []
    for s in specs:
        if s.valid is not None:
            continue
        if s.rule is not None:
            checks.append((s, s.rule))
        elif s.missing_rule is not None:
            info = EVIDENCE_FEATURES[s.name]
            if info.validity is not None:
                checks.append((s, _Rule("features", info.validity, info.validity)))
    if not checks:
        return
    need: dict[str, list[str]] = {"features": [], "psms_extracted": []}
    for _, rule in checks:
        need[rule.table].extend(predicate_columns(rule.sql))
    given = {t: {c: _input(values.get(c)) for c in dict.fromkeys(cols)} for t, cols in need.items()}
    cid = _number(values.get("candidate_id"))
    if cid is not None:
        rank = int(_number(values.get("peak_rank")) or 0)
        missing = [c for c, v in given["features"].items() if v is None and c in names]
        if missing:
            row = candidate_features(rs, run, int(cid), rank, missing)
            if row is not None:
                given["features"].update({c: _input(row.get(c)) for c in missing})
        missing = [c for c, v in given["psms_extracted"].items() if v is None]
        extracted = _extracted_for(run)
        if missing and extracted is not None:
            row = _extracted_row(rs, extracted, int(cid), rank, missing)
            if row is not None:
                given["psms_extracted"].update({c: _input(row.get(c)) for c in missing})
    alias = {"features": "f", "psms_extracted": "e"}
    relations: list[str] = []
    params: list[Any] = []
    for table, inputs in given.items():
        if inputs:
            cols = ", ".join(f"CAST(? AS DOUBLE) AS {sql_ident(c)}" for c in inputs)
            relations.append(f"(SELECT {cols}) AS {alias[table]}")
            params.extend(inputs.values())
    select = ", ".join(
        f"({_qualify(rule.sql, alias[rule.table])}) AS v{i}" for i, (_, rule) in enumerate(checks)
    )
    tables = f" FROM {', '.join(relations)}" if relations else ""
    result = rs.duck.execute(f"SELECT {select}{tables}", params).fetchone()
    for i, (spec, rule) in enumerate(checks):
        verdict = None if result is None else result[i]
        if verdict is None:
            if spec.missing_rule is None:
                spec.notes.append(
                    f"the validity rule {rule.text} was not checked on this value: its inputs "
                    "were not given"
                )
        else:
            spec.valid = bool(verdict)
            if not spec.valid:
                spec.notes.append(f"this value fails the validity rule {rule.text}")


def _population_text(
    rs: ResultSet, run: Run, source: FeatureSource, scored: Artifact, spec: _Spec
) -> str:
    scored_name = scored.path.name if scored.path is not None else scored.key
    where = f", source = {run.index}" if rs.is_experiment else ""
    rows = (
        f"{source.description} rows of the scored peaks (candidate_id and peak_rank = "
        f"selected_peak_rank of {scored_name}{where})"
    )
    of_run = f"target/decoy rows of run {_run_name(rs, run)}, not FDR-filtered"
    if spec.rule is not None:
        text = f"valid {of_run}: {rows}, rows with {spec.rule.text}"
    elif spec.missing_rule is not None and not spec.count:
        return f"{of_run}: {rows}; this feature is not ranked, because {spec.missing_rule}"
    elif spec.missing_rule is not None:
        text = (
            f"{of_run} and not filtered by a validity rule ({spec.missing_rule}, so sentinel "
            f"values are included): {rows}"
        )
    elif spec.name in EVIDENCE_FEATURES:
        text = f"{of_run}: {rows}; every value of this feature is a measurement, so no rule applies"
    else:
        text = (
            f"{of_run}: {rows}; no validity rule is defined for this feature, so sentinel values, "
            "if any, are included"
        )
    return text + "; percentile = % of these rows with a value <= this value"


def _prepare(
    rs: ResultSet,
    run: Run,
    name: str,
    value: float | None,
    schema: pa.Schema,
    source: FeatureSource,
    rank_unfiltered: bool,
) -> _Spec:
    """Decide how one feature is handled: counted, ranked, which validity rule applies."""
    spec = _Spec(name, value)
    info = EVIDENCE_FEATURES.get(name)
    if name not in schema.names:
        where = f" (it is a {info.table} column)" if info and info.table != "features" else ""
        spec.notes.append(f"{name} is not a column of {source.description}{where}")
        spec.count = spec.rank = False
        spec.valid = False
        return spec
    if not _numeric(schema.field(name)):
        spec.notes.append(f"{name} is not numeric")
        spec.count = spec.rank = False
        spec.valid = False
        return spec
    if info is not None and info.validity:
        absent = [c for c in info.validity_columns if c not in schema.names]
        if not absent:
            spec.rule = _Rule("features", info.validity, info.validity)
        else:
            lacks = f"the feature table lacks {', '.join(absent)}"
            fallback = _extracted_rule(info, run)
            if fallback is not None:
                spec.rule = fallback
                spec.notes.append(
                    f"{lacks}, so the validity rule {info.validity} is applied in its "
                    f"psms_extracted.parquet form, {info.extracted_validity}"
                )
            else:
                spec.missing_rule = (
                    f"the validity rule {info.validity} cannot be evaluated: {lacks}"
                )
                if info.extracted_validity is not None:
                    spec.missing_rule += (
                        f", and the run has no readable psms_extracted.parquet with "
                        f"{', '.join(info.extracted_validity_columns)}"
                    )
                if rank_unfiltered:
                    spec.notes.append(
                        f"{spec.missing_rule}; the value is ranked against every row, sentinel "
                        "values included"
                    )
                else:
                    spec.notes.append(
                        f"{spec.missing_rule}; the value is not ranked, because its sentinel "
                        "values cannot be told apart from measurements"
                    )
                    spec.count = spec.rank = False
    if name in _CONTESTED:
        computed, reason = contested_computed(rs)
        if not computed:
            spec.notes.append(reason)
            spec.rank = False
            spec.valid = False
            return spec
    if info is not None and not info.percentile:
        spec.notes.append("a flag has no percentile")
        spec.rank = False
    if value is None:
        spec.notes.append("no finite value given")
        spec.valid = False
    elif spec.rule is None and spec.missing_rule is None:
        spec.valid = True
    return spec


def feature_percentiles(
    rs: ResultSet,
    run: Run | str | int,
    values: Mapping[str, Any],
    columns: Sequence[str] | None = None,
    *,
    rank_unfiltered: bool = False,
) -> list[FeaturePercentile]:
    """Rank a candidate's feature values against the target and decoy rows of its run.

    ``values`` maps feature names to the candidate's values, normally the row from
    :func:`candidate_features`. ``columns`` selects the features to rank, in that order
    (default: every key of ``values`` that is not a bookkeeping column; with a full row
    that is all 387 features, which took 0.40 to 0.48 s on the Astral single run, against
    30 to 40 ms for 20 features).

    The population is the run's feature rows of the scored peaks: joined to the run's
    scored rows on ``candidate_id`` and ``peak_rank = selected_peak_rank`` (unselected
    alternative peaks are excluded; in an experiment, the run's own rows,
    ``source = run.index``), restricted to the rows that pass each feature's validity
    rule. The rows are not FDR-filtered. DuckDB ranks up to 64 features per query:
    ``avg((col <= value)::DOUBLE)`` per label. Flags, features the engine did not
    compute and values that fail their validity rule get no percentage.

    When the table lacks the inputs of a validity rule (``features.set`` minimal or
    rich), the rule is applied in its ``psms_extracted.parquet`` form where one exists
    (``isfinite(rt_pred_cal)`` for the RT errors). Otherwise the value is not ranked;
    with ``rank_unfiltered=True`` it is ranked against every row, sentinel values
    included, ``valid`` stays None and ``population`` says so.
    """
    run = _run(rs, run)
    source = feature_source(rs, run)
    scored = _scored_for_run(rs, run)
    schema = _schema(source.artifacts[0])
    if columns is None:
        wanted = [k for k in dict.fromkeys(values) if k not in BOOKKEEPING]
    else:
        wanted = list(dict.fromkeys(columns))
    specs = [
        _prepare(rs, run, name, _number(values.get(name)), schema, source, rank_unfiltered)
        for name in wanted
    ]
    names = set(schema.names)
    _check_candidate(rs, run, values, specs, names)
    counted = [s for s in specs if s.count]
    results: dict[str, tuple[float | None, float | None, int, int]] = {}
    if counted:
        results = _percentile_query(rs, run, source, scored, counted, names)
    out = []
    for s in specs:
        pct_t, pct_d, n_t, n_d = results.get(s.name, (None, None, 0, 0))
        if s.ranked:
            for label, pct, n in (("target", pct_t, n_t), ("decoy", pct_d, n_d)):
                if pct is None and n == 0:
                    s.notes.append(f"no {'valid ' if s.rule else ''}{label} rows")
        else:
            pct_t = pct_d = None
        out.append(
            FeaturePercentile(
                feature=s.name,
                value=s.value,
                pct_target=pct_t,
                pct_decoy=pct_d,
                n_target=n_t,
                n_decoy=n_d,
                valid=s.valid,
                population=_population_text(rs, run, source, scored, s),
                note="; ".join(s.notes) or None,
                rule=s.rule.text if s.rule is not None else None,
            )
        )
    return out


def _when(flag: str | None, expr: str) -> str:
    return f"CASE WHEN {flag} THEN {expr} END" if flag else expr


# Features ranked per DuckDB query. All 387 features of an Astral run in one aggregate
# exceed the 1 GB DuckDB memory limit; 64 per query stay far below it.
_BATCH = 64


def _percentile_query(
    rs: ResultSet,
    run: Run,
    source: FeatureSource,
    scored: Artifact,
    specs: list[_Spec],
    names: set[str],
) -> dict[str, tuple[float | None, float | None, int, int]]:
    """Per feature and label: % of the population rows <= value, and the row count.

    The features are ranked in batches of :data:`_BATCH`, one DuckDB aggregate each. A
    DuckDB failure is raised as a :class:`ViewerError` that names the run and the batch.
    """
    out: dict[str, tuple[float | None, float | None, int, int]] = {}
    for start in range(0, len(specs), _BATCH):
        batch = specs[start : start + _BATCH]
        try:
            out.update(_percentile_batch(rs, run, source, scored, batch, names))
        except duckdb.Error as exc:
            first = (str(exc).splitlines() or [""])[0]
            raise ViewerError(
                f"run {_run_name(rs, run)}: DuckDB failed on the percentile query for "
                f"{len(batch)} of {len(specs)} feature(s) of {source.description} "
                f"({type(exc).__name__}: {first})"
            ) from exc
    return out


def _percentile_batch(
    rs: ResultSet,
    run: Run,
    source: FeatureSource,
    scored: Artifact,
    specs: list[_Spec],
    names: set[str],
) -> dict[str, tuple[float | None, float | None, int, int]]:
    """One aggregate over the population for up to :data:`_BATCH` features.

    Each distinct validity rule is evaluated once per row, as a flag column of the
    population; ``CASE WHEN flag`` then keeps only the valid rows (avg and count skip
    the NULL). A rule over ``psms_extracted`` columns tests whether the row's
    ``(candidate_id, peak_rank)`` is among the ``psms_extracted`` rows that pass it
    (``IN``, so a duplicate row there cannot count a feature row twice). A column that
    the footers show to be free of nulls shares one row count per rule with every such
    column, which makes the 20-feature query about a quarter faster on an Astral run.
    """
    flags: dict[_Rule, str] = {}
    for s in specs:
        if s.rule is not None and s.rule not in flags:
            flags[s.rule] = f"__valid_{len(flags)}"
    non_null = _non_null_columns(source.artifacts)
    needed = [s.name for s in specs if s.ranked or s.name not in non_null]
    f_rank = "f.peak_rank" if "peak_rank" in names else "0"
    proj = ["f.label AS label", *(f"f.{sql_ident(c)} AS {sql_ident(c)}" for c in needed)]
    params: list[Any] = []
    for rule, flag in flags.items():
        if rule.table == "features":
            proj.append(f"({_qualify(rule.sql, 'f')}) AS {flag}")
            continue
        extracted = _extracted_for(run)
        if extracted is None:  # the rule was chosen because the table is readable
            raise ArtifactNotFound(f"run {_run_name(rs, run)}: psms_extracted.parquet is absent.")
        e_rank = "e.peak_rank" if "peak_rank" in _schema(extracted).names else "0"
        proj.append(
            f"(f.candidate_id, {f_rank}) IN (SELECT (e.candidate_id, {e_rank}) FROM "
            f"read_parquet(?) AS e WHERE ({_qualify(rule.sql, 'e')})) AS {flag}"
        )
        params.append(sql_path(extracted.require()))
    paths = [sql_path(p) for p in source.paths]
    table: str | list[str] = paths[0] if len(paths) == 1 else paths
    if _population_is_whole_table(rs, source, scored, run.index):
        pop = f"SELECT {', '.join(proj)} FROM read_parquet(?) AS f"
        params.append(table)
    else:
        s_rank = "selected_peak_rank" if scored.parquet().has_column("selected_peak_rank") else "0"
        pop = (
            f"SELECT {', '.join(proj)} FROM read_parquet(?) AS f SEMI JOIN "
            f"(SELECT candidate_id, {s_rank} AS spr FROM read_parquet(?) WHERE source = ?) AS s "
            f"ON f.candidate_id = s.candidate_id AND {f_rank} = s.spr"
        )
        params.extend([table, sql_path(scored.require()), int(run.index)])
    items: list[str] = []
    shared: dict[str | None, str] = {}
    count_of: dict[str, str] = {}
    for i, s in enumerate(specs):
        col = sql_ident(s.name)
        flag = flags.get(s.rule) if s.rule is not None else None
        if s.ranked:
            items.append(f"avg({_when(flag, f'({col} <= ?)::DOUBLE')}) * 100.0 AS p{i}")
            params.append(float(s.value))  # type: ignore[arg-type]
        if s.name in non_null:
            if flag not in shared:
                shared[flag] = f"n{len(shared)}"
                items.append(f"count({_when(flag, '1')}) AS {shared[flag]}")
            count_of[s.name] = shared[flag]
        else:
            count_of[s.name] = f"m{i}"
            items.append(f"count({_when(flag, col)}) AS m{i}")
    sql = f"WITH pop AS ({pop}) SELECT label, {', '.join(items)} FROM pop GROUP BY label"
    cursor = rs.duck.execute(sql, params)
    out_names = [d[0] for d in cursor.description]
    by_label = {str(r[0]): dict(zip(out_names, r, strict=True)) for r in cursor.fetchall()}
    unknown = set(by_label) - {"target", "decoy"}
    if unknown:
        raise InconsistentData(
            f"{source.description}: label values other than target and decoy: {sorted(unknown)}"
        )
    out: dict[str, tuple[float | None, float | None, int, int]] = {}
    for i, s in enumerate(specs):
        pct: list[float | None] = []
        count: list[int] = []
        for label in ("target", "decoy"):
            row = by_label.get(label)
            p = None if row is None else row.get(f"p{i}")
            pct.append(None if p is None else float(p))
            count.append(0 if row is None else int(row[count_of[s.name]]))
        out[s.name] = (pct[0], pct[1], count[0], count[1])
    return out
