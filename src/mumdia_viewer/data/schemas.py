"""Schema-version registry, column contracts and chromatogram layouts.

Versions come from ``rust/mumdia/crates/mumdia-core/src/schema.rs`` of each MuMDIA
release. A parquet footer carries no schema name or version, so the version of an
artifact is taken from its manifest record or its ``<file>.report.json``. For the few
files the engine writes without either, the version is inferred from the columns and
labelled as inferred.

The viewer refuses a version it does not know, with a message that names the file,
the artifact, the version found and the versions supported. It does not guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pyarrow as pa

from .errors import LayoutError, SchemaVersionError

# Versions written by the tagged releases v0.1.0 to v0.5.0.
RELEASED: dict[str, frozenset[int]] = {
    "spectra_ms1": frozenset({1}),
    "spectra_ms2": frozenset({1}),
    "isolation_windows": frozenset({1}),
    "ms2_to_ms1": frozenset({1}),
    "peptides": frozenset({1}),
    "peptidoforms": frozenset({1}),
    "fragment_library_precursors": frozenset({1}),
    "fragment_library_fragments": frozenset({1}),
    "prescan_survivors": frozenset({1}),
    "seed_psms": frozenset({1}),
    "run_windows": frozenset({1}),
    "psms_extracted": frozenset({2}),
    "chromatograms": frozenset({1, 2}),
    "features": frozenset({1, 2}),
    "psms_competed": frozenset({3, 4}),
    "psms_scored": frozenset({4}),
    "peptide_quant": frozenset({2}),
    "protein_group_quant": frozenset({2}),
    "fragment_quant": frozenset({1}),
    "lfq_maxlfq": frozenset({1}),
    "overlap_losers": frozenset({1}),
    "alignment": frozenset({1}),
    "candidate_audit": frozenset({1}),
}

# Versions written only by pre-release development builds that report version 0.1.0.
DEV_BUILD: dict[str, frozenset[int]] = {
    "psms_scored": frozenset({3}),
    "psms_competed": frozenset({2}),
    "psms_extracted": frozenset({1}),
}

# Versions on the unreleased ion-mobility branch (PR #140, origin/IM).
UNRELEASED_IM: dict[str, frozenset[int]] = {
    "spectra_ms1": frozenset({2, 3}),
    "spectra_ms2": frozenset({2, 3}),
    "isolation_windows": frozenset({2}),
    "fragment_library_precursors": frozenset({2}),
    "seed_psms": frozenset({2}),
    "psms_extracted": frozenset({3, 4, 5}),
    "chromatograms": frozenset({3, 4}),
}

VersionSource = Literal["manifest", "report", "inferred", "unrecorded"]


@dataclass(frozen=True)
class VersionInfo:
    """The schema version of one artifact and where it was read from."""

    schema_name: str
    version: int | None
    source: VersionSource

    @property
    def label(self) -> str:
        if self.version is None:
            return f"{self.schema_name}, version not recorded"
        suffix = {"manifest": "", "report": "", "inferred": " (inferred from columns)"}.get(
            self.source, ""
        )
        return f"{self.schema_name} v{self.version}{suffix}"


def supported_versions(schema_name: str, *, allow_unreleased: bool = False) -> frozenset[int]:
    """Versions of ``schema_name`` this viewer reads."""
    versions = set(RELEASED.get(schema_name, frozenset()))
    versions |= DEV_BUILD.get(schema_name, frozenset())
    if allow_unreleased:
        versions |= UNRELEASED_IM.get(schema_name, frozenset())
    return frozenset(versions)


def check_version(
    schema_name: str, version: int | None, *, where: str, allow_unreleased: bool = False
) -> None:
    """Raise :class:`SchemaVersionError` when ``version`` is not readable.

    ``version=None`` (nothing recorded) is accepted here; callers label such files.
    Schema names outside the registry are not checked, because the viewer does not
    read them.
    """
    if version is None or schema_name not in RELEASED:
        return
    supported = supported_versions(schema_name, allow_unreleased=allow_unreleased)
    if version in supported:
        return
    listed = ", ".join(str(v) for v in sorted(supported))
    if version in UNRELEASED_IM.get(schema_name, frozenset()):
        raise SchemaVersionError(
            f"{where}: {schema_name} schema version {version} is written by the unreleased "
            f"ion-mobility branch of MuMDIA (PR #140). This viewer reads versions {listed}. "
            "Ion-mobility layouts can be enabled with allow_unreleased=True, but they are not "
            "validated on real data yet."
        )
    newest = max(supported) if supported else None
    hint = (
        " It was probably written by a newer MuMDIA than this viewer knows; update the viewer."
        if newest is not None and version > newest
        else ""
    )
    raise SchemaVersionError(
        f"{where}: {schema_name} schema version {version} is not supported "
        f"(this viewer reads versions {listed}).{hint}"
    )


# Columns the viewer reads from each schema. Extra columns are ignored. Nullability
# flags are never checked, because pandas-written tables declare every field nullable.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "psms_scored": (
        "candidate_id",
        "peptidoform",
        "charge",
        "label",
        "protein",
        "base_peptide_id",
        "apex_rt",
        "elution_lo",
        "elution_hi",
        "score",
        "q_value",
        "peptide_q_value",
        "protein_group",
        "pg_q_value",
        "prelim_score",
        "source",
        "run_psm_q",
        "precursor_q",
    ),
    "psms_extracted": (
        "candidate_id",
        "apex_rt",
        "apex_intensity",
        "n_matched_fragments",
        "n_predicted_fragments",
        "coelution_run",
        "rt_pred_cal",
        "precursor_mz",
        "charge",
        "label",
        "base_peptide_id",
        "peptidoform",
        "protein",
    ),
    "features": (
        "candidate_id",
        "label",
        "base_peptide_id",
        "peptidoform",
        "protein",
        "apex_rt",
        "elution_lo",
        "elution_hi",
        "precursor_mz",
        "prelim_score",
    ),
    "psms_competed": (
        "candidate_id",
        "label",
        "base_peptide_id",
        "peptidoform",
        "protein",
        "apex_rt",
        "elution_lo",
        "elution_hi",
        "precursor_mz",
        "prelim_score",
    ),
    # frag_mz and frag_obs_mz are optional for the engine's reader (frag_obs_mz falls
    # back to frag_mz), so the decoder handles their absence.
    "chromatograms": ("candidate_id", "frag_name", "predicted_intensity"),
    "run_windows": ("candidate_id", "rt_pred_cal", "rt_lo", "rt_hi"),
    "seed_psms": (
        "candidate_id",
        "peptidoform",
        "charge",
        "base_peptide_id",
        "label",
        "score",
        "spectrum_q",
        "observed_rt",
        "predicted_irt",
        "scan_index",
    ),
    "spectra_ms2": (
        "scan_index",
        "rt_seconds",
        "window_id",
        "window_target",
        "window_lower",
        "window_upper",
        "mz",
        "intensity",
    ),
    "spectra_ms1": ("scan_index", "rt_seconds", "mz", "intensity"),
    "isolation_windows": ("window_id", "target", "lower", "upper"),
    "ms2_to_ms1": ("ms2_scan_index", "ms1_scan_index"),
    "peptide_quant": (
        "candidate_id",
        "base_peptide_id",
        "peptidoform",
        "charge",
        "protein_group",
        "quantity",
        "quant_status",
        "n_fragments_used",
        "integration_apex_rt",
        "integration_lo_rt",
        "integration_hi_rt",
    ),
    "protein_group_quant": ("protein_group", "quantity", "quant_status", "n_peptides"),
    "fragment_quant": (
        "candidate_id",
        "peptidoform",
        "charge",
        "protein_group",
        "fragment_name",
        "quantity",
    ),
    "lfq_maxlfq": ("protein_group", "run", "quantity", "n_features"),
    "lfq_maxlfq_sibling": ("group", "charge", "run", "quantity", "n_features"),
    "overlap_losers": ("band", "candidate_id"),
    "fragment_library_precursors": (
        "candidate_id",
        "base_peptide_id",
        "peptidoform",
        "charge",
        "precursor_mz",
        "predicted_irt",
        "label",
        "protein",
    ),
    "fragment_library_fragments": ("candidate_id", "mz", "predicted_intensity", "name"),
    "mbr_transferred": ("source", "candidate_id"),
}

# Columns that a given version adds and the viewer relies on.
VERSION_COLUMNS: dict[tuple[str, int], tuple[str, ...]] = {
    ("psms_scored", 4): ("selected_peak_rank",),
    ("psms_extracted", 2): ("peak_rank",),
}


def check_columns(
    schema_name: str, schema: pa.Schema, *, where: str, version: int | None = None
) -> None:
    """Raise :class:`LayoutError` when a column the viewer reads is absent."""
    names = set(schema.names)
    required = list(REQUIRED_COLUMNS.get(schema_name, ()))
    if version is not None:
        required += VERSION_COLUMNS.get((schema_name, version), ())
    missing = [c for c in required if c not in names]
    if missing:
        label = f"{schema_name} v{version}" if version is not None else schema_name
        raise LayoutError(f"{where}: {label} lacks the column(s) {', '.join(missing)}.")
    if schema_name == "chromatograms":
        layout = chromatogram_layout(schema, where=where)
        if version is not None and version != layout.schema_version:
            raise LayoutError(
                f"{where}: the recorded chromatograms schema version is {version}, but the "
                f"columns are the version {layout.schema_version} layout."
            )


# --------------------------------------------------------------------------- layouts

_CHROM_V1 = frozenset({"rt", "intensity"})
_CHROM_V2 = frozenset({"rt_axis", "intensity_trimmed", "trace_offset", "trace_len"})


@dataclass(frozen=True)
class ChromatogramLayout:
    """The on-disk layout of a chromatogram table.

    ``family`` 1 stores every row's axis and full trace (``rt``, ``intensity``).
    ``family`` 2 stores the axis once per candidate per row group and a trimmed trace
    (``rt_axis``, ``intensity_trimmed``, ``trace_offset``, ``trace_len``). The
    ion-mobility layouts add one list parallel to the trace: ``im`` (version 3, on the
    family-1 layout) or ``im_trimmed`` (version 4, on the family-2 layout).
    """

    family: int
    has_im: bool

    @property
    def schema_version(self) -> int:
        return {(1, False): 1, (2, False): 2, (1, True): 3, (2, True): 4}[
            (self.family, self.has_im)
        ]


def chromatogram_layout(schema: pa.Schema, *, where: str = "chromatograms") -> ChromatogramLayout:
    """Detect the chromatogram layout from the column names, as the engine does.

    A table with some but not all of the version-2 columns, or with version-2 columns
    beside ``rt`` or ``intensity``, is neither layout and is refused.
    """
    names = set(schema.names)
    v1 = names & _CHROM_V1
    v2 = names & _CHROM_V2
    if v2 == _CHROM_V2 and not v1:
        if "im" in names:
            raise LayoutError(f"{where}: an `im` column beside the version-2 layout is not valid.")
        return ChromatogramLayout(family=2, has_im="im_trimmed" in names)
    if v1 == _CHROM_V1 and not v2:
        if "im_trimmed" in names:
            raise LayoutError(
                f"{where}: an `im_trimmed` column beside the version-1 layout is not valid."
            )
        return ChromatogramLayout(family=1, has_im="im" in names)
    raise LayoutError(
        f"{where}: the columns are neither chromatogram layout (found "
        f"{sorted(v1 | v2) or 'none'} of rt, intensity, rt_axis, intensity_trimmed, "
        "trace_offset, trace_len)."
    )


def infer_version(schema_name: str, schema: pa.Schema) -> int | None:
    """Infer the version of a file that has neither a manifest record nor a report."""
    names = set(schema.names)
    if schema_name == "chromatograms":
        return chromatogram_layout(schema).schema_version
    if schema_name == "psms_scored":
        return 4 if "selected_peak_rank" in names else 3
    if schema_name == "psms_extracted":
        return 2 if "peak_rank" in names else 1
    if all(c in names for c in REQUIRED_COLUMNS.get(schema_name, ())) and schema_name in RELEASED:
        released = RELEASED[schema_name]
        if len(released) == 1:
            return next(iter(released))
    return None
