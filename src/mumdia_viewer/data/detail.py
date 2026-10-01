"""Everything the precursor detail page shows about one identification.

:func:`precursor_detail` assembles, for one candidate in one run:

* the scored row, every q value with its unit, the score and the rescorer;
* every extracted peak of the candidate (``peak_rank``) and the one the scored row
  selected (``selected_peak_rank``);
* the decoded chromatogram (fragment and MS1 isotope traces) and the markers of the
  XIC panel: identification apex and elution bounds, quant's integration bounds, the
  calibrated RT prediction and the extraction window, and the alternative peaks;
* the extraction window, the extraction tolerance and the MS2 scan at the apex;
* the quant state, the base-peptide competition, the exact decoy partner and, when
  match-between-runs ran, the transfer record;
* an evidence summary in which every value names the column it comes from, and every
  viewer-derived value says how it was computed.

:func:`mirror` gives the observed spectrum against the predicted fragments, and
:func:`detail_percentiles` ranks the candidate's features against the run's targets
and decoys. Each part is read with the per-candidate machinery of the data layer; no
table is loaded whole.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import Artifact
from .candidate_index import CandidateIndex, concat_parts, read_candidate_rows
from .chromatograms import CandidateChromatogram, ChromatogramSource
from .competition import QValue, competition, exact_partner, q_values, scored_rows_of
from .discovery import ResultSet, Run
from .duck import sql_path
from .entrapment import entrapment_expr, markers_present, settings_for
from .errors import ArtifactNotFound, InconsistentData, ViewerError
from .features import (
    EVIDENCE_FEATURES,
    FeaturePercentile,
    candidate_feature_rows,
    contested_computed,
    feature_percentiles,
)
from .fragments import PeakMatch, Tolerance, extraction_tolerance, match_fragments
from .mbr import mbr_ran, transfer_of
from .quant import QuantState, quant_state
from .rescore import RescoreInfo, rescore_info
from .spectra import ScanPick, ScanTable, Spectrum
from .windows import RtWindow, rt_window

EXTRACTED_COLUMNS = (
    "candidate_id",
    "peak_rank",
    "apex_rt",
    "apex_im",
    "apex_intensity",
    "n_matched_fragments",
    "n_predicted_fragments",
    "coelution_run",
    "rt_pred_cal",
    "precursor_mz",
    "charge",
    "label",
    "predicted_irt",
    "contested_frac",
    "ms1_isom1",
    "ms1_mono",
    "ms1_iso1",
    "ms1_iso2",
)


@dataclass(frozen=True)
class EvidenceItem:
    """One line of the evidence summary.

    ``source`` names the table and column the value was read from. A value the viewer
    computed has ``derived=True`` and a ``source`` that says how.
    """

    group: str
    key: str
    label: str
    value: Any
    unit: str
    source: str
    note: str | None = None
    derived: bool = False


@dataclass(frozen=True)
class DecoyPartner:
    """The exact library decoy (or target) partner of the candidate, and its scored rows."""

    candidate_id: int | None
    reason: str
    rows: pd.DataFrame


@dataclass
class PrecursorDetail:
    run: Run
    candidate_id: int
    scored: dict[str, Any]
    q_values: list[QValue]
    rescore: RescoreInfo
    peaks: pd.DataFrame
    peaks_source: str
    selected_peak_rank: int
    features: dict[str, Any] | None
    window: RtWindow | None
    chromatogram: CandidateChromatogram | None
    band: str | None
    apex_scan: ScanPick | None
    tolerance: Tolerance | None
    quant: QuantState | None
    competition: pd.DataFrame
    partner: DecoyPartner
    transfer: dict[str, Any] | None
    evidence: list[EvidenceItem]
    markers: dict[str, Any]
    # The row quant and the report read after match-between-runs (q lowered on
    # transfers), when it differs from the native row in ``scored``.
    scored_after_mbr: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def is_decoy(self) -> bool:
        return self.scored.get("label") == "decoy"

    @property
    def precursor_mz(self) -> float | None:
        row = self.selected_peak
        if row is not None and row.get("precursor_mz") is not None:
            return float(row["precursor_mz"])
        if self.features is not None and self.features.get("precursor_mz") is not None:
            return float(self.features["precursor_mz"])
        return None

    @property
    def selected_peak(self) -> dict[str, Any] | None:
        if self.peaks.empty or "peak_rank" not in self.peaks:
            return None if self.peaks.empty else self.peaks.iloc[0].to_dict()
        hit = self.peaks[self.peaks["peak_rank"] == self.selected_peak_rank]
        return None if hit.empty else hit.iloc[0].to_dict()


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def run(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.ms[name] = round((time.perf_counter() - t0) * 1000.0, 2)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# --------------------------------------------------------------------------- rows


def _point_row(rs: ResultSet, artifact: Artifact, cid: int, source: int) -> dict[str, Any] | None:
    """One row of a scored table by (source, candidate_id), with a DuckDB point query.

    A single run's scored table is one row group of up to a million rows; a point query
    reads the filter column and the matching row only.
    """
    df = rs.duck.df(
        "SELECT * FROM read_parquet($path) WHERE candidate_id = $cid AND source = $source",
        {"path": sql_path(artifact.require()), "cid": int(cid), "source": int(source)},
    )
    if df.empty:
        return None
    if len(df) > 1:
        raise InconsistentData(
            f"candidate {cid} has {len(df)} rows with source {source} in {artifact.path}."
        )
    row = df.iloc[0].to_dict()
    return {
        k: (None if not isinstance(v, list | np.ndarray) and pd.isna(v) else v)
        for k, v in row.items()
    }


def _scored_rows(rs: ResultSet, run: Run, cid: int) -> tuple[dict[str, Any], dict | None]:
    """(native scored row, row after match-between-runs or None).

    After MBR a run's ``scored.parquet`` is the split of ``scored_mbr.parquet``, whose
    ``q_value``, ``run_psm_q`` and ``experiment_psm_q`` are lowered on transferred rows.
    The native values, the rescorer's, stay in ``scored_combined.parquet``.
    """
    if rs.is_experiment and mbr_ran(rs):
        native = _point_row(rs, rs.scored, cid, run.index)
        split = run.artifact("psms_scored")
        after = (
            _point_row(rs, split, cid, run.index) if split is not None and split.usable else None
        )
    else:
        art = run.artifact("psms_scored")
        native = _point_row(
            rs, art if art is not None and art.usable else rs.scored, cid, run.index
        )
        after = None
    if native is None:
        raise ArtifactNotFound(f"candidate {cid} has no scored row in run {run.label}.")
    return native, after


def _peak_rows(rs: ResultSet, run: Run, cid: int) -> tuple[pd.DataFrame, str]:
    """Every extracted peak of ``cid`` (one row per ``peak_rank``) and where it came from."""
    art = run.artifact("psms_extracted")
    refused = None
    if art is not None and art.present and not art.usable:
        refused = art.error
    if art is not None and art.usable:
        index = CandidateIndex.for_artifact(art, rs.cache)
        handle = art.parquet()
        cols = [c for c in EXTRACTED_COLUMNS if c in handle.schema.names]
        table = concat_parts(read_candidate_rows(handle, index, cid, cols, cached=True))
        df = table.to_pandas() if table is not None else pd.DataFrame(columns=cols)
        return df, "psms_extracted.parquet"
    # Grouped runs delete band psms_extracted after pooling: the competed rows carry the
    # peak's apex and bounds, but not the extraction-only columns.
    rows = candidate_feature_rows(
        rs,
        run,
        cid,
        [
            "apex_rt",
            "elution_lo",
            "elution_hi",
            "precursor_mz",
            "prelim_score",
            "n_matched_fragments",
            "coelution_run",
        ],
    )
    df = pd.DataFrame(rows)
    why = f"refused: {refused}" if refused else "not available in this run"
    return df, f"psms_competed rows (psms_extracted.parquet {why})"


def _entrapment_test(rs: ResultSet, info: RescoreInfo) -> tuple[str, dict] | None:
    if info.mode != "entrapment":
        return None
    settings = settings_for(rs)
    if not markers_present(rs, settings):
        return None
    return entrapment_expr(settings)


# --------------------------------------------------------------------------- evidence


LABELS: dict[str, str] = {
    "score": "Rescorer score",
    "prelim_score": "Preliminary score",
    "n_predicted_fragments": "Predicted fragments",
    "n_apex": "Fragments observed at the apex scan",
    "frag_ppm_offset": "Fragment mass offset of the run",
    "frag_tol_ppm": "Fragment tolerance used by extraction",
    "apex_rt": "Identification apex",
    "rt_pred_cal": "Calibrated RT prediction",
    "rt_window": "Extraction window",
    "rt_error": "RT error (apex - prediction)",
    "rt_error_rel": "RT error relative to the half-window",
    "ms1_mono": "MS1 monoisotopic intensity",
    "ms1_iso1": "MS1 +1 isotope intensity",
    "ms1_iso2": "MS1 +2 isotope intensity",
    "selected_peak_rank": "Selected peak rank",
    "quant_state": "Quant state",
    "quantity": "Quantity",
}


def _label(key: str) -> str:
    info = EVIDENCE_FEATURES.get(key)
    if info is not None:
        return info.label
    return LABELS.get(key, key)


def _item(
    group: str,
    key: str,
    value: Any,
    *,
    unit: str,
    source: str,
    note: str | None = None,
    derived: bool = False,
) -> EvidenceItem:
    return EvidenceItem(
        group=group,
        key=key,
        label=_label(key),
        value=value,
        unit=unit,
        source=source,
        note=note,
        derived=derived,
    )


def _feature_item(
    features: dict[str, Any] | None, name: str, group: str, *, note: str | None = None
) -> EvidenceItem | None:
    info = EVIDENCE_FEATURES.get(name)
    if info is None or features is None or name not in features:
        return None
    return _item(
        group,
        name,
        features[name],
        unit=info.unit,
        source=f"features: {name}",
        note=note or info.note or None,
    )


def _evidence(rs: ResultSet, d: PrecursorDetail) -> list[EvidenceItem]:
    s, f, peak = d.scored, d.features, d.selected_peak or {}
    items: list[EvidenceItem] = []

    def add(item: EvidenceItem | None) -> None:
        if item is not None:
            items.append(item)

    add(
        _item(
            "score",
            "score",
            s.get("score"),
            unit="higher is better",
            source="psms_scored: score",
            note=f"rescorer: {d.rescore.label}",
        )
    )
    add(
        _item(
            "score",
            "prelim_score",
            s.get("prelim_score"),
            unit="",
            source="psms_scored: prelim_score",
            note="competition uses it to pick the winner of a key; it is not a classifier input",
        )
    )
    for q in d.q_values:
        # The value is the engine's; only a grouped column's winner flag is the viewer's.
        source = f"psms_scored: {q.column}"
        if q.grouped and q.winner_source:
            source += f"; winner flag: {q.winner_source}"
        add(EvidenceItem("q", q.column, q.column, q.display_value, "q value", source, q.text))
    if d.scored_after_mbr is not None:
        for column in ("q_value", "run_psm_q", "experiment_psm_q"):
            after = d.scored_after_mbr.get(column)
            native = d.scored.get(column)
            if after is None or native is None or float(after) == float(native):
                continue
            add(
                EvidenceItem(
                    "q",
                    f"{column}_after_mbr",
                    f"{column} after match-between-runs",
                    after,
                    "q value",
                    f"{d.run.label}/scored.parquet: {column}",
                    "the value quant and the report used after MBR: min(native q, transfer_q); "
                    "not the rescorer's estimate",
                )
            )

    # Fragments.
    n_matched = peak.get("n_matched_fragments")
    if n_matched is None and f is not None:
        n_matched = f.get("n_matched_fragments")
    envelope = "; for an alternative peak, inside its envelope" if d.selected_peak_rank else ""
    add(
        _item(
            "fragments",
            "n_matched_fragments",
            n_matched,
            unit="fragments",
            source=f"{d.peaks_source}: n_matched_fragments",
            note="distinct predicted fragments matched anywhere in the RT window" + envelope,
        )
    )
    if peak.get("n_predicted_fragments") is not None:
        add(
            _item(
                "fragments",
                "n_predicted_fragments",
                peak.get("n_predicted_fragments"),
                unit="fragments",
                source=f"{d.peaks_source}: n_predicted_fragments",
            )
        )
    n_apex = None
    if f is not None and f.get("n_matched_b") is not None and f.get("n_matched_y") is not None:
        n_apex = int(f["n_matched_b"]) + int(f["n_matched_y"])
        add(
            _item(
                "fragments",
                "n_apex",
                n_apex,
                unit="fragments",
                source="viewer-derived: features n_matched_b + n_matched_y",
                derived=True,
            )
        )
    for name in ("frag_corr", "frag_cosine", "spectral_angle"):
        add(_feature_item(f, name, "fragments"))
    if f is not None and "spectral_angle_matched" in f:
        if n_apex is not None and n_apex < 2:
            add(
                _item(
                    "fragments",
                    "spectral_angle_matched",
                    None,
                    unit="normalized angle [0, 1]",
                    source="features: spectral_angle_matched",
                    note=f"not shown: {n_apex} fragment(s) observed at the apex scan",
                )
            )
        else:
            add(_feature_item(f, "spectral_angle_matched", "fragments"))

    # Co-elution.
    coel = peak.get("coelution_run")
    if coel is None and f is not None:
        coel = f.get("coelution_run")
    info = EVIDENCE_FEATURES.get("coelution_run")
    add(
        _item(
            "coelution",
            "coelution_run",
            coel,
            unit="scans",
            source=f"{d.peaks_source}: coelution_run",
            note=info.note if info else None,
        )
    )
    for name in ("coelution_mean", "coelution_best", "n_observations"):
        add(_feature_item(f, name, "coelution"))

    # Mass error.
    for name in ("median_abs_frag_ppm", "signed_mean_frag_ppm", "frag_mass_err_median"):
        add(_feature_item(f, name, "mass"))
    if d.tolerance is not None:
        assumed = " (assumed: the extraction record is missing)" if d.tolerance.assumed else ""
        add(
            _item(
                "mass",
                "frag_ppm_offset",
                d.tolerance.offset_ppm,
                unit="ppm",
                source=d.tolerance.source,
                note="systematic fragment mass offset of the run",
            )
        )
        add(
            _item(
                "mass",
                "frag_tol_ppm",
                d.tolerance.tol_ppm,
                unit="ppm",
                source=d.tolerance.source,
                note="the tolerance the extraction used" + assumed,
            )
        )

    # Retention time.
    apex = _finite(s.get("apex_rt"))
    add(_item("rt", "apex_rt", apex, unit="s", source="psms_scored: apex_rt"))
    w = d.window
    if w is not None:
        add(
            _item(
                "rt",
                "rt_pred_cal",
                w.rt_pred_cal,
                unit="s",
                source=w.source,
                note=w.calibration_note or None,
            )
        )
        add(
            _item(
                "rt",
                "rt_window",
                (w.rt_lo, w.rt_hi),
                unit="s",
                source=w.source,
                note=None if w.bounded else "unbounded: no RT calibration for this candidate",
            )
        )
        if apex is not None and w.rt_pred_cal is not None:
            err = apex - w.rt_pred_cal
            add(
                _item(
                    "rt",
                    "rt_error",
                    err,
                    unit="s",
                    source="viewer-derived: apex_rt - rt_pred_cal",
                    derived=True,
                )
            )
            if w.rt_hi is not None and w.rt_hi > w.rt_pred_cal:
                add(
                    _item(
                        "rt",
                        "rt_error_rel",
                        err / (w.rt_hi - w.rt_pred_cal),
                        unit="fraction of the half-window",
                        source="viewer-derived: (apex_rt - rt_pred_cal) / (rt_hi - rt_pred_cal)",
                        note="the error is truncated at the window by construction",
                        derived=True,
                    )
                )

    # MS1.
    for name in ("ms1_mono", "ms1_iso1", "ms1_iso2"):
        if name in peak:
            value = _finite(peak.get(name))
            add(
                _item(
                    "ms1",
                    name,
                    value,
                    unit="counts",
                    source=f"{d.peaks_source}: {name}",
                    note=("summed MS1 peaks near the isotope m/z in the MS1 scan nearest the apex")
                    if value is not None
                    else "no MS1 data",
                )
            )
    for name in ("has_ms1", "log_ms1_mono", "ms1_isotope_cosine_apex"):
        add(_feature_item(f, name, "ms1"))

    # Interference.
    computed, why = contested_computed(rs)
    add(
        _item(
            "interference",
            "contested_frac",
            peak.get("contested_frac") if computed else None,
            unit="fraction",
            source=f"{d.peaks_source}: contested_frac",
            note=None if computed else f"not computed: {why}",
        )
    )
    for name in ("n_interfered_fragments", "interference_apex_residual_fraction"):
        add(_feature_item(f, name, "interference"))

    # Peaks.
    add(
        _item(
            "peaks",
            "selected_peak_rank",
            d.selected_peak_rank,
            unit="rank",
            source="psms_scored: selected_peak_rank",
            note=(
                f"{len(d.peaks)} extracted peak(s); the rescorer selected rank "
                f"{d.selected_peak_rank}; the scores of the other peaks are not stored"
            ),
        )
    )

    # Quant.
    if d.quant is not None:
        add(
            _item(
                "quant",
                "quant_state",
                d.quant.state,
                unit="",
                source="peptide_quant",
                note=d.quant.reason,
            )
        )
        note = None
        if d.quant.quantity is None and d.quant.state == "not_quantifiable":
            note = f"not quantifiable: {d.quant.status}"
        add(
            _item(
                "quant",
                "quantity",
                d.quant.quantity,
                unit="intensity x s",
                source="peptide_quant: quantity",
                note=note,
            )
        )
    return items


def _markers(d: PrecursorDetail) -> dict[str, Any]:
    s = d.scored
    peaks = []
    if not d.peaks.empty and "apex_rt" in d.peaks:
        for _, row in d.peaks.iterrows():
            rank = int(row["peak_rank"]) if "peak_rank" in row and pd.notna(row["peak_rank"]) else 0
            peaks.append(
                {
                    "peak_rank": rank,
                    "apex_rt": _finite(row["apex_rt"]),
                    "selected": rank == d.selected_peak_rank,
                }
            )
    q = d.quant
    w = d.window
    return {
        "apex_rt": _finite(s.get("apex_rt")),
        "elution_lo": _finite(s.get("elution_lo")),
        "elution_hi": _finite(s.get("elution_hi")),
        "integration_apex_rt": q.integration_apex_rt if q else None,
        "integration_lo_rt": q.integration_lo_rt if q else None,
        "integration_hi_rt": q.integration_hi_rt if q else None,
        "rt_pred_cal": w.rt_pred_cal if w else None,
        "rt_lo": w.rt_lo if w else None,
        "rt_hi": w.rt_hi if w else None,
        "peaks": peaks,
        "note": "elution bounds are float32 axis points: compare them with the axis in float32",
    }


# --------------------------------------------------------------------------- entry point


def _guard(notes: list[str], what: str, timer: _Timer, name: str, fn, *args, **kwargs):
    """Run an optional part; a ViewerError becomes a note and the part is None."""
    try:
        return timer.run(name, fn, *args, **kwargs)
    except ViewerError as exc:
        notes.append(f"{what} unavailable: {exc}")
        return None


def precursor_detail(rs: ResultSet, run: Run | str | int, candidate_id: int) -> PrecursorDetail:
    """Assemble the precursor detail of ``candidate_id`` in ``run``.

    Only the scored row is required. Every other part is optional: when its artifact is
    missing or refused, the part is None (or empty) and ``notes`` says why.
    """
    run = rs.run(run)
    cid = int(candidate_id)
    timer = _Timer()
    notes: list[str] = []

    scored, after_mbr = timer.run("scored", _scored_rows, rs, run, cid)
    info = timer.run("rescore", rescore_info, rs)
    found = _guard(notes, "extracted peaks", timer, "peaks", _peak_rows, rs, run, cid)
    peaks, peaks_source = found if found is not None else (pd.DataFrame(), "none")
    selected = scored.get("selected_peak_rank")
    if selected is None:
        notes.append("the scored table has no selected_peak_rank (schema v3): peak rank 0 assumed")
        selected = 0
    selected = int(selected)

    source = _guard(
        notes, "chromatograms", timer, "chromatogram_source", ChromatogramSource.for_run, rs, run
    )
    chrom = (
        _guard(notes, "chromatograms", timer, "chromatogram", source.read, cid) if source else None
    )
    band = chrom.band if chrom is not None else None
    if chrom is None and source is not None:
        notes.append("no chromatogram rows for this candidate")

    rows = (
        _guard(
            notes,
            "features",
            timer,
            "features",
            candidate_feature_rows,
            rs,
            run,
            cid,
            list(EVIDENCE_FEATURES),
        )
        or []
    )
    feats = next((r for r in rows if int(r.get("peak_rank") or 0) == selected), None)
    if feats is None:
        notes.append("no feature row for the selected peak")

    window = _guard(notes, "extraction window", timer, "window", rt_window, rs, run, cid, band=band)
    if window is not None and not window.bounded:
        notes.append("the extraction window is unbounded: the run has no RT calibration here")
    tol = _guard(
        notes, "extraction tolerance", timer, "tolerance", extraction_tolerance, rs, run, band=band
    )

    apex_pick = None
    pmz = None
    sel = peaks[peaks["peak_rank"] == selected] if "peak_rank" in peaks else peaks
    if not sel.empty and "precursor_mz" in sel:
        pmz = _finite(sel.iloc[0]["precursor_mz"])
    if pmz is None and feats is not None:
        pmz = _finite(feats.get("precursor_mz"))
    apex = _finite(scored.get("apex_rt"))
    if pmz is not None and apex is not None and run.has("spectra_ms2"):
        scans = _guard(notes, "spectra", timer, "scan_table", ScanTable.for_run, rs, run)
        if scans is not None:
            apex_pick = timer.run("apex_scan", scans.apex_scan, pmz, apex)
            if apex_pick is None:
                notes.append("no isolation window covers the precursor m/z")
            elif not apex_pick.exact:
                notes.append(f"apex scan approximate: {apex_pick.label}")
    elif not run.has("spectra_ms2"):
        notes.append("the run has no readable spectra_ms2 table")

    quant = _guard(notes, "quant state", timer, "quant", quant_state, rs, run, cid)

    bpid = int(scored["base_peptide_id"])
    found = _guard(
        notes,
        "competition",
        timer,
        "competition",
        competition,
        rs,
        run.index,
        cid,
        bpid,
        entrapment=_entrapment_test(rs, info),
    )
    comp, winners = found if found is not None else (pd.DataFrame(), {})
    qvals = q_values(rs, scored, winners, info=info)

    lookup = _guard(
        notes,
        "decoy partner",
        timer,
        "partner",
        exact_partner,
        rs,
        cid,
        peptidoform=scored.get("peptidoform"),
        charge=scored.get("charge"),
    )
    if lookup is not None and lookup.candidate_id is not None:
        partner_rows = _guard(
            notes,
            "decoy partner rows",
            timer,
            "partner_rows",
            scored_rows_of,
            rs,
            lookup.candidate_id,
        )
        partner_rows = partner_rows if partner_rows is not None else pd.DataFrame()
        partner = DecoyPartner(lookup.candidate_id, lookup.reason, partner_rows)
        if partner_rows.empty:
            notes.append(
                f"the exact library partner (candidate {lookup.candidate_id}) was not scored"
            )
    else:
        partner = DecoyPartner(None, lookup.reason if lookup else "not looked up", pd.DataFrame())

    transfer = None
    if rs.is_experiment and mbr_ran(rs):
        transfer = _guard(notes, "transfer", timer, "transfer", transfer_of, rs, run.index, cid)
        if transfer is not None:
            tq = transfer.get("transfer_q")
            notes.append(
                "match-between-runs transfer into this run"
                + (f" (transfer_q {float(tq):.6g})" if tq is not None else "")
                + ": the q values shown are the rescorer's native values; the lowered values "
                "that quant and the report used are listed separately"
            )

    peak = sel.iloc[0].to_dict() if not sel.empty else {}
    if _finite(peak.get("apex_intensity")) == 0.0:
        notes.append(
            "apex fallback: no scan met the apex rule, apex_rt is the first grid scan and "
            "apex_intensity is not recorded"
        )
    if scored.get("label") == "decoy":
        notes.append("this row is a decoy")
    if info.mode == "entrapment":
        notes.append("entrapment mode: the q columns are entrapment estimates")

    detail = PrecursorDetail(
        run=run,
        candidate_id=cid,
        scored=scored,
        q_values=qvals,
        rescore=info,
        peaks=peaks,
        peaks_source=peaks_source,
        selected_peak_rank=selected,
        features=feats,
        window=window,
        chromatogram=chrom,
        band=band,
        apex_scan=apex_pick,
        tolerance=tol,
        quant=quant,
        competition=comp,
        partner=partner,
        transfer=transfer,
        evidence=[],
        markers={},
        notes=notes,
        scored_after_mbr=after_mbr,
    )
    detail.evidence = timer.run("evidence", _evidence, rs, detail)
    detail.markers = _markers(detail)
    detail.timings_ms = timer.ms
    return detail


# --------------------------------------------------------------------------- mirror plot


@dataclass(frozen=True)
class MirrorData:
    """An observed MS2 spectrum against the candidate's predicted fragments."""

    spectrum: Spectrum
    pick: ScanPick
    fragments: pd.DataFrame
    matches: list[PeakMatch]
    tolerance: Tolerance
    label: str
    previous_row: int | None
    next_row: int | None


def mirror(rs: ResultSet, detail: PrecursorDetail, row: int | None = None) -> MirrorData | None:
    """The mirror plot of the apex scan, or of MS2 row ``row`` of the same run.

    The predicted fragments are the candidate's chromatogram fragment rows (name, library
    m/z, predicted intensity), so the 100-million-row library table is not read. The
    matches use the extraction's tolerance and mass offset with the engine's predicate.
    """
    if detail.tolerance is None or detail.chromatogram is None:
        return None
    run = detail.run
    scans = ScanTable.for_run(rs, run)
    apex = _finite(detail.scored.get("apex_rt"))
    if row is None:
        if detail.apex_scan is None:
            return None
        pick = detail.apex_scan
    else:
        pick = scans.pick(int(row), reference_rt=apex)
    spectrum = scans.spectrum(pick.row)
    frags = detail.chromatogram.fragments()
    table = pd.DataFrame(
        {
            "name": [t.frag_name for t in frags],
            "ion": [t.ion for t in frags],
            "ordinal": [t.ordinal for t in frags],
            "charge": [t.fragment_charge for t in frags],
            "theo_mz": [t.frag_mz for t in frags],
            "predicted_intensity": [t.predicted_intensity for t in frags],
            "observed_in_xic": [t.observed for t in frags],
        }
    )
    theo = table["theo_mz"].to_numpy(dtype=np.float64)
    matches = match_fragments(spectrum.mz, spectrum.intensity, theo, detail.tolerance)
    prev = scans.step(pick.row, -1, reference_rt=apex)
    nxt = scans.step(pick.row, 1, reference_rt=apex)
    return MirrorData(
        spectrum=spectrum,
        pick=pick,
        fragments=table,
        matches=matches,
        tolerance=detail.tolerance,
        label=detail.tolerance.match_label if hasattr(detail.tolerance, "match_label") else "",
        previous_row=prev.row if prev is not None else None,
        next_row=nxt.row if nxt is not None else None,
    )


# --------------------------------------------------------------------------- features


DEFAULT_PERCENTILE_FEATURES = (
    "frag_corr",
    "frag_cosine",
    "spectral_angle",
    "spectral_angle_matched",
    "coelution_mean",
    "coelution_best",
    "median_abs_frag_ppm",
    "signed_mean_frag_ppm",
    "log_ms1_mono",
    "ms1_isotope_cosine_apex",
    "n_interfered_fragments",
    "interference_apex_residual_fraction",
    "rt_error_abs",
    "rt_error_over_peak_width",
    "log_apex_intensity",
)


def detail_percentiles(
    rs: ResultSet, detail: PrecursorDetail, columns: Sequence[str] | None = None
) -> list[FeaturePercentile]:
    """Rank the candidate's features (selected peak) against the run's targets and decoys."""
    if detail.features is None:
        return []
    names = [c for c in (columns or DEFAULT_PERCENTILE_FEATURES) if c in detail.features]
    return feature_percentiles(rs, detail.run, detail.features, names)
