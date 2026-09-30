"""Provenance of a result set: summary, inputs, artifacts, stage timings, hash checks.

Everything here comes from what the engine recorded: the manifest, the reports and
the JSON sidecars. Opening never hashes a file; :func:`verify_hashes` does, on request.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import Artifact, Status
from .discovery import Notice, ResultSet
from .hashing import verify
from .manifest import ArtifactRecord
from .paths import is_absolute_recorded
from .reports import Report, StageTiming, load_json, normalise_enum, stage_timings
from .rescore import RescoreInfo, rescore_info
from .schemas import VersionInfo

# Stages that write no report.json (their wall time is not recorded anywhere).
_UNTIMED_EXPERIMENT = ("split-by-source", "quant-lfq")
_UNTIMED_ALWAYS = ("report",)
_UNTIMED_GROUPED = ("seed-pool",)

# DeepLC library rewrites: their stage has no report; the time is in <table>.summary.json.
_DEEPLC_SUFFIXES = (
    ("_multihead", "deeplc-multihead"),
    ("_ft", "deeplc-finetune"),
    ("_deeplc", "deeplc-repredict"),
)
# timings_s phases that run one after another. featurisation and forward are part of
# predict (process-seconds, summed over prediction shards).
_SEQUENTIAL_PHASES = ("read_library", "reference", "fit", "unique", "predict", "rewrite", "write")


@dataclass
class RunSummary:
    """Provenance of a result set, from its manifest.

    ``runs`` holds the run names (``''`` for a single run). ``grouped`` maps each
    grouped run's name to its band layout, or is None when no run is grouped.
    ``mbr_strategy`` is the recorded match-between-runs strategy of an experiment
    (the string ``"None"`` when MBR did not run; None for a single run).
    ``quant_q_filter`` holds the q column that quant was configured to gate on and the
    one it used (they differ in experiments, where per-run quant is gated on the pooled
    ``q_value``), and where each value was read.
    """

    kind: str
    root: Path
    mumdia_version: str | None
    git_sha: str | None
    commit_date: str | None
    cli_args: list[str]
    config: dict[str, Any]
    config_hash: str | None
    model_identities: dict[str, Any]
    rescore: RescoreInfo
    n_runs: int
    runs: list[str]
    grouped: dict[str, dict[str, Any]] | None
    mbr_strategy: str | None
    quant_q_filter: dict[str, Any] | None
    notices: list[Notice]

    @property
    def mbr_ran(self) -> bool:
        """True when match-between-runs ran (an experiment whose strategy is not None)."""
        return self.mbr_strategy is not None and normalise_enum(self.mbr_strategy) != "none"


def _grouped_summary(rs: ResultSet) -> dict[str, dict[str, Any]] | None:
    out: dict[str, dict[str, Any]] = {}
    for run in rs.runs:
        g = run.grouped
        if g is None:
            continue
        out[run.name] = {
            "bands": [b.name for b in g.bands],
            "n_bands": len(g.bands),
            "skipped": list(g.skipped),
            "pooled_chromatograms": g.pooled_chromatograms,
            "pooled_competed": g.pooled_competed,
            "overlap_losers_rows": g.losers.rows if g.losers is not None else None,
            "window_groups_configured": g.plan.get("window_groups"),
            "calibration": g.plan.get("calibration"),
        }
    return out or None


def _quant_q_filter(rs: ResultSet) -> dict[str, Any] | None:
    threshold = rs.config_get("quant", "q_threshold")
    exp = rs.manifest.experiment or {}
    recorded = exp.get("quant_q_filter")
    if isinstance(recorded, dict):
        return {
            "configured": recorded.get("configured"),
            "effective": recorded.get("effective"),
            "q_threshold": threshold,
            "source": "experiment_manifest.json experiment.quant_q_filter; "
            "config_json quant.q_threshold",
        }
    configured = rs.config_get("quant", "q_filter")
    effective = None
    quant = rs.runs[0].artifact("peptide_quant") if rs.runs else None
    if quant is not None and quant.report is not None:
        effective = quant.report.params.get("q_filter")
        if threshold is None:
            threshold = quant.report.params.get("q_threshold")
    if configured is None and effective is None:
        return None
    return {
        "configured": configured,
        "effective": effective,
        "q_threshold": threshold,
        "source": "config_json quant.q_filter; peptide_quant.parquet.report.json params.q_filter",
    }


def run_summary(rs: ResultSet) -> RunSummary:
    """Version, configuration, rescorer, runs and layout of a result set."""
    m = rs.manifest
    mbr = None
    if rs.is_experiment:
        exp = m.experiment or {}
        value = exp.get("mbr", m.model_identities.get("mbr"))
        mbr = None if value is None else str(value)
    return RunSummary(
        kind=rs.kind,
        root=rs.root,
        mumdia_version=m.mumdia_version,
        git_sha=m.git_sha,
        commit_date=m.commit_date,
        cli_args=list(m.cli_args),
        config=m.config,
        config_hash=m.config_hash,
        model_identities=dict(m.model_identities),
        rescore=rescore_info(rs),
        n_runs=len(rs.runs),
        runs=[r.name for r in rs.runs],
        grouped=_grouped_summary(rs),
        mbr_strategy=mbr,
        quant_q_filter=_quant_q_filter(rs),
        notices=list(rs.notices),
    )


# --------------------------------------------------------------------------- inputs

INPUT_COLUMNS = [
    "key",
    "recorded_path",
    "resolved_path",
    "bytes",
    "content_hash",
    "status",
    "resolution",
    "size_on_disk",
    "note",
]


def inputs_table(rs: ResultSet) -> pd.DataFrame:
    """The recorded inputs (mzML, library or FASTA) and where they are now.

    Paths are resolved through the result set's resolver: an input outside the run
    directory is found as recorded when it still exists, or through a remapped root.
    ``status`` is ``found`` (the file size equals the recorded ``bytes``),
    ``size_mismatch`` (a file exists but has another size, so it is not the recorded
    input), ``needs_remap`` (not found outside the directory: remap its root) or
    ``missing`` (not found inside the directory). The content hash is not computed
    here; use :func:`verify_hashes` with ``include_inputs=True``.
    """
    rows = []
    for key, inp in rs.manifest.inputs.items():
        resolved = rs.resolver.resolve(inp.path)
        size = None
        if resolved.path is not None:
            try:
                size = os.stat(resolved.path).st_size
            except OSError:
                size = None
        relative = not is_absolute_recorded(inp.path)
        if resolved.path is None or size is None:
            status = "missing" if resolved.inside_root else "needs_remap"
            if relative:
                note = (
                    "recorded relative to the engine's working directory, which the manifest "
                    "does not record; remap its first directory to use it"
                )
            elif resolved.inside_root:
                note = "not found inside the directory"
            else:
                note = "not found outside the directory; remap its root to use it"
        elif inp.bytes is not None and size != inp.bytes:
            status = "size_mismatch"
            note = f"the file has {size} bytes, the recorded input {inp.bytes}; it is another file"
        else:
            status = "found"
            note = "size matches the recorded bytes; the content hash is checked on request"
        rows.append(
            {
                "key": key,
                "recorded_path": inp.path,
                "resolved_path": str(resolved.path) if resolved.path is not None else None,
                "bytes": inp.bytes,
                "content_hash": inp.content_hash,
                "status": status,
                "resolution": resolved.how,
                "size_on_disk": size,
                "note": note,
            }
        )
    table = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    return table.astype({"bytes": "Int64", "size_on_disk": "Int64"})


# --------------------------------------------------------------------------- artifacts

ARTIFACT_COLUMNS = [
    "scope",
    "key",
    "kind",
    "version",
    "rows",
    "status",
    "resolution",
    "path",
    "content_hash_short",
    "stage",
    "error",
]


def _scope_dir(rs: ResultSet, scope: str) -> str:
    """The directory of a ResultSet scope, relative to the opened root (``.`` for the root).

    ``all_artifacts`` yields each artifact's directory relative to the root: ``""``,
    ``<run>``, ``groups/gNN`` or ``<run>/groups/gNN``.
    """
    return "." if scope in ("", ".") else scope


def _unique_artifacts(rs: ResultSet) -> list[tuple[str, Artifact]]:
    """Every discovered artifact once (an artifact listed under two names is one row)."""
    seen: set[int] = set()
    out = []
    for scope, a in rs.all_artifacts():
        if id(a) in seen:
            continue
        seen.add(id(a))
        out.append((_scope_dir(rs, scope), a))
    return out


def _display_path(rs: ResultSet, path: Path | None, recorded: str | None) -> str | None:
    if path is None:
        return recorded
    try:
        return Path(path).relative_to(rs.root).as_posix()
    except ValueError:
        return str(path)


def artifact_table(rs: ResultSet) -> pd.DataFrame:
    """Every artifact the viewer found or the manifest lists, with its version and status.

    ``status`` is ``present``, ``missing`` or ``deleted_after_pooling`` (a band
    intermediate that ``groups.delete_band_intermediates`` removed; the manifest still
    lists it). ``error`` holds the reason an artifact cannot be read (for example an
    unsupported schema version). ``path`` is relative to the opened directory when it
    lies inside it; for an absent artifact it is the recorded path.
    """
    rows = []
    for scope, a in _unique_artifacts(rs):
        rows.append(
            {
                "scope": scope,
                "key": a.key,
                "kind": a.kind,
                "version": a.version.label,
                "rows": a.rows,
                "status": a.status.value,
                "resolution": a.resolution,
                "path": _display_path(rs, a.path, a.recorded_path),
                "content_hash_short": a.content_hash[:12] if a.content_hash else None,
                "stage": a.stage,
                "error": a.error,
            }
        )
    return pd.DataFrame(rows, columns=ARTIFACT_COLUMNS).astype({"rows": "Int64"})


# --------------------------------------------------------------------------- timings

TIMING_COLUMNS = ["directory", "stage", "elapsed_s", "source", "artifacts", "note"]


def _summary_files(rs: ResultSet) -> list[tuple[str, Path, str]]:
    """``(directory, path, stage)`` of every DeepLC ``.summary.json`` in the result set."""
    found: list[tuple[str, Path, str]] = []
    places: list[tuple[str, Path, str]] = []
    exp_prefix = "fragment_library_precursors"
    if rs.is_experiment:
        places.append((".", rs.root, exp_prefix))
    for run in rs.runs:
        run_dir = "." if not rs.is_experiment else run.name
        places.append((run_dir, run.root, exp_prefix))
        if run.grouped is not None:
            for band in run.grouped.bands:
                band_dir = (
                    f"groups/{band.name}" if run_dir == "." else f"{run_dir}/groups/{band.name}"
                )
                places.append((band_dir, band.root, "lib_precursors"))
    seen: set[Path] = set()
    for directory, root, prefix in places:
        for suffix, stage in _DEEPLC_SUFFIXES:
            path = root / f"{prefix}{suffix}.parquet.summary.json"
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append((directory, path, stage))
    return found


def _deeplc_rows(rs: ResultSet) -> list[dict[str, Any]]:
    """One timing row per DeepLC library rewrite, from ``timings_s`` of its summary.

    The phases ``read_library``, ``reference``, ``fit``, ``unique``, ``predict``,
    ``rewrite`` and ``write`` run one after another; ``model_load`` too in a multi-head
    calibration (in the other modes the model load is inside ``fit`` or ``predict``).
    ``featurisation`` and ``forward`` are part of ``predict``. Band tables of one call
    share a single timing record, which is reported once.
    """
    rows: list[dict[str, Any]] = []
    shared: dict[str, dict[str, Any]] = {}
    for directory, path, stage in _summary_files(rs):
        data = load_json(path) or {}
        timings = data.get("timings_s")
        name = path.name[: -len(".summary.json")]
        if not isinstance(timings, dict):
            rows.append(
                {
                    "directory": directory,
                    "stage": stage,
                    "elapsed_s": None,
                    "source": f"{path.name}: no timings_s",
                    "artifacts": name,
                    "note": "the summary records no phase timings",
                }
            )
            continue
        phases = list(_SEQUENTIAL_PHASES)
        if "multihead" in data:
            phases.insert(1, "model_load")
        values = {p: timings.get(p) for p in phases}
        numbers = [float(v) for v in values.values() if isinstance(v, int | float)]
        total = sum(numbers) if numbers else None
        detail = ", ".join(
            f"{p} {float(v):.1f} s" for p, v in values.items() if isinstance(v, int | float)
        )
        source = (
            f"{path.name} timings_s: sum of {', '.join(phases)} (featurisation and forward are "
            "part of predict)"
        )
        row = {
            "directory": directory,
            "stage": stage,
            "elapsed_s": total,
            "source": source,
            "artifacts": name,
            "note": detail,
        }
        if isinstance(data.get("bands"), dict):
            key = f"{stage}:{json.dumps(timings, sort_keys=True)}"
            if key in shared:
                shared[key]["_bands"].append(directory)
                continue
            row["_bands"] = [directory]
            shared[key] = row
        rows.append(row)
    for row in rows:
        bands = row.pop("_bands", None)
        if bands and len(bands) > 1:
            row["directory"] = f"{bands[0]} (+{len(bands) - 1} bands)"
            row["note"] = f"one call for {len(bands)} band tables; " + str(row["note"])
    return rows


def _lost_band_stages(rs: ResultSet, timed: set[tuple[str, str]]) -> list[str]:
    """Band stages whose reports were deleted after pooling, as ``stage (N band ...)``.

    With ``groups.delete_band_intermediates`` (the default) the engine deletes the band
    ``features.parquet`` and ``psms_extracted.parquet`` with their reports. The extract
    time survives in the band chromatograms report; the features stage writes no other
    artifact, so its band time is lost. A single-run manifest still lists the deleted
    files; an experiment records no band artifact, so there an absent band features
    table under that setting counts as deleted.
    """
    deletes = bool(rs.config_get("groups", "delete_band_intermediates", default=True))
    lost: dict[str, set[str]] = {}
    for run in rs.runs:
        if run.grouped is None:
            continue
        prefix = f"{run.name}/" if rs.is_experiment else ""
        for band in run.grouped.bands:
            band_dir = f"{prefix}groups/{band.name}"
            stages = {
                a.record.producing_stage
                for a in band.artifacts.values()
                if a.status is Status.DELETED_AFTER_POOLING
                and a.record is not None
                and a.record.producing_stage
            }
            features = band.artifact("features")
            if deletes and (features is None or not features.present):
                stages.add("features")
            for stage in stages:
                if (band_dir, stage) not in timed:
                    lost.setdefault(stage, set()).add(band_dir)
    return [
        f"{stage} ({len(dirs)} band report{'s' if len(dirs) > 1 else ''} deleted after pooling)"
        for stage, dirs in lost.items()
    ]


def _untimed_stages(rs: ResultSet, timed: set[tuple[str, str]]) -> tuple[list[str], bool]:
    """The stages without a recorded time, and whether some of them lost their reports."""
    stages: list[str] = []
    if rs.is_experiment:
        stages.extend(_UNTIMED_EXPERIMENT)
        exp = rs.manifest.experiment or {}
        mbr = exp.get("mbr")
        if mbr is not None and normalise_enum(mbr) != "none":
            stages.append("mbr")
    if any(run.grouped is not None for run in rs.runs):
        stages.extend(_UNTIMED_GROUPED)
    lost = _lost_band_stages(rs, timed)
    stages.extend(lost)
    stages.extend(_UNTIMED_ALWAYS)
    return stages, bool(lost)


# (scope, producing stage, report stage, artifact, report) of a report written by another stage
_Borrowed = tuple[str, str, str, Artifact, Report]


def _report_entries(rs: ResultSet) -> tuple[list[StageTiming], list[_Borrowed]]:
    """Stage timings from the reports, and the reports that another stage wrote.

    A report whose ``stage`` differs from the ``producing_stage`` that the manifest
    records for its artifact did not time that artifact in this run. On a
    ``library-cache`` hit the engine copies the stored library's reports: their stage
    stays ``predict-frag`` and their ``elapsed_ms`` is the time of the run that stored
    the library. Such reports are kept out of the timings and returned apart.
    """
    entries: list[tuple[str, Report]] = []
    borrowed: list[_Borrowed] = []
    seen: set[Path] = set()
    for scope, a in rs.all_artifacts():
        report = a.report
        if report is None or report.path in seen:
            continue
        seen.add(report.path)
        produced = a.record.producing_stage if a.record is not None else None
        if produced and report.stage and normalise_enum(produced) != normalise_enum(report.stage):
            borrowed.append((scope or ".", produced, report.stage, a, report))
        else:
            entries.append((scope or ".", report))
    return stage_timings(entries), borrowed


def _borrowed_rows(rs: ResultSet, borrowed: list[_Borrowed]) -> list[dict[str, Any]]:
    """One row per (directory, producing stage) whose reports another stage wrote."""
    groups: dict[tuple[str, str, str], list[tuple[Artifact, Report]]] = {}
    for scope, produced, reported, a, report in borrowed:
        key = (_scope_dir(rs, scope), produced, reported)
        groups.setdefault(key, []).append((a, report))
    rows: list[dict[str, Any]] = []
    for (directory, produced, reported), items in groups.items():
        times = [r.elapsed_ms for _, r in items if r.elapsed_ms is not None]
        recorded = f" ({max(times) / 1000.0:.3f} s)" if times else ""
        if normalise_enum(produced) == "librarycache":
            note = (
                "library-cache hit: the engine reused a stored library and copied its reports. "
                f"Their elapsed_ms{recorded} is the {reported} time of the run that stored the "
                "library, not a time of this run; the library-cache time is not recorded"
            )
        else:
            note = (
                f"the reports record stage {reported}{recorded}, but the manifest records that "
                f"{produced} produced these artifacts in this run; that elapsed_ms is not shown "
                f"as the {produced} time"
            )
        rows.append(
            {
                "directory": directory,
                "stage": produced,
                "elapsed_s": None,
                "source": f"manifest producing_stage {produced}; report.json stage {reported}",
                "artifacts": ", ".join(sorted({r.logical_name or a.key for a, r in items})),
                "note": note,
            }
        )
    return rows


def stage_timings_table(rs: ResultSet) -> pd.DataFrame:
    """Wall time per (directory, stage), in seconds.

    From the reports: every artifact of a stage repeats the stage's ``elapsed_ms``, so
    the largest value per (directory, stage) is the stage time; values are never
    summed, also not across bands that ran in parallel. A report copied from another
    run (a ``library-cache`` hit, whose reports keep stage ``predict-frag``) gives a
    row under the manifest's producing stage with no time. DeepLC library rewrites (for
    example the multi-head calibration) write no report; their time comes from
    ``timings_s`` of the table's ``.summary.json``, labelled as such. The last row lists
    the stages whose time the engine does not record, including band stages whose
    reports were deleted after pooling.
    """
    timings, borrowed = _report_entries(rs)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for t in timings:
        directory = _scope_dir(rs, t.directory)
        entry = merged.setdefault(
            (directory, t.stage), {"elapsed_ms": t.elapsed_ms, "artifacts": set()}
        )
        entry["elapsed_ms"] = max(entry["elapsed_ms"], t.elapsed_ms)
        entry["artifacts"].update(t.artifacts)
    rows: list[dict[str, Any]] = []
    for (directory, stage), entry in merged.items():
        rows.append(
            {
                "directory": directory,
                "stage": stage,
                "elapsed_s": entry["elapsed_ms"] / 1000.0,
                "source": "report.json elapsed_ms (largest over the stage's reports)",
                "artifacts": ", ".join(sorted(entry["artifacts"])),
                "note": None,
            }
        )
    rows.extend(_borrowed_rows(rs, borrowed))
    rows.extend(_deeplc_rows(rs))
    rows.sort(key=lambda r: (r["directory"] != ".", r["directory"], -(r["elapsed_s"] or 0.0)))
    untimed, lost = _untimed_stages(rs, set(merged))
    rows.append(
        {
            "directory": ".",
            "stage": "not recorded",
            "elapsed_s": None,
            "source": "these stages write no report.json"
            + (", or their band reports were deleted after pooling" if lost else ""),
            "artifacts": ", ".join(untimed),
            "note": "their wall time is only in the engine log",
        }
    )
    return pd.DataFrame(rows, columns=TIMING_COLUMNS)


# --------------------------------------------------------------------------- hashes

HASH_COLUMNS = ["scope", "key", "path", "verdict", "recorded", "computed", "seconds"]


def _input_artifacts(rs: ResultSet) -> list[tuple[str, Artifact]]:
    out = []
    for role, inp in rs.manifest.inputs.items():
        resolved = rs.resolver.resolve(inp.path)
        record = ArtifactRecord(
            key=role,
            path=inp.path,
            logical_name=None,
            schema_name=None,
            schema_version=None,
            rows=None,
            content_hash=inp.content_hash,
            producing_stage=None,
            config_hash=None,
            format=None,
        )
        out.append(
            (
                "input",
                Artifact(
                    kind="input",
                    key=role,
                    path=resolved.path,
                    status=Status.PRESENT if resolved.path is not None else Status.MISSING,
                    version=VersionInfo("input", None, "unrecorded"),
                    record=record,
                    recorded_path=inp.path,
                    resolution=resolved.how,
                ),
            )
        )
    return out


def verify_hashes(
    rs: ResultSet, keys: Iterable[str] | None = None, *, include_inputs: bool = False
) -> pd.DataFrame:
    """Hash artifacts with blake3 and compare them with the recorded content hashes.

    This reads every selected file in full, so it runs only on request, never on open.
    ``keys`` selects artifacts by key (``artifact_table`` column ``key``) or inputs by
    role (``inputs_table`` column ``key``); None selects every artifact, and the inputs
    too when ``include_inputs`` is True (mzML inputs can be several GB). Verdicts are
    ``match``, ``mismatch``, ``no_recorded_hash`` and ``missing``, and are cached by
    (path, size, modification time) in the viewer cache.
    """
    targets = _unique_artifacts(rs)
    inputs = _input_artifacts(rs)
    if keys is None:
        chosen = targets + (inputs if include_inputs else [])
    else:
        wanted = list(dict.fromkeys(keys))
        by_key: dict[str, list[tuple[str, Artifact]]] = {}
        for scope, a in targets + inputs:
            by_key.setdefault(a.key, []).append((scope, a))
        unknown = [k for k in wanted if k not in by_key]
        if unknown:
            raise KeyError(f"no artifact or input with key {', '.join(unknown)}")
        chosen = [item for k in wanted for item in by_key[k]]
    rows = []
    for scope, a in chosen:
        started = time.perf_counter()
        check = verify(a, rs.cache)
        rows.append(
            {
                "scope": scope,
                "key": a.key,
                "path": _display_path(rs, check.path, a.recorded_path),
                "verdict": check.verdict,
                "recorded": check.recorded,
                "computed": check.computed,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    return pd.DataFrame(rows, columns=HASH_COLUMNS)
