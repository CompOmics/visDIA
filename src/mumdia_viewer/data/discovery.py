"""Discovery of the artifacts of a single run or an experiment directory.

A directory is classified by the manifest it holds:

* ``experiment_manifest.json``: an experiment. Its runs are ``<root>/<name>`` for each
  name in ``experiment.runs``; the name at position ``i`` is the run whose rows carry
  ``source == i``. The experiment manifest records only the pooled scored tables, the
  per-run ``scored``, ``peptide_quant`` and ``protein_group_quant`` tables and the LFQ
  matrix. Every other per-run artifact is found by its fixed file name and described
  by its own ``report.json``.
* ``manifest.json``: a single run. It is grouped when ``groups/plan.json`` exists or
  a manifest key ends in ``[gNN]``.

Nothing here writes to the directory.
"""

from __future__ import annotations

import operator
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import Artifact, Status
from .cache import Cache
from .duck import DuckDB
from .errors import LayoutError, NotAResultDirectory, SchemaVersionError, ViewerError
from .manifest import ArtifactRecord, Manifest, band_index, load_manifest, split_key
from .paths import PathResolver
from .reports import Report, StageTiming, load_json, load_report, stage_timings
from .schemas import VersionInfo, check_version

# Fixed file names of the artifacts in a run directory (relative to the run root).
RUN_FILES: dict[str, str] = {
    "psms_scored": "psms_scored.parquet",
    "psms_extracted": "psms_extracted.parquet",
    "features": "features.parquet",
    "psms_competed": "psms_competed.parquet",
    "chromatograms": "chromatograms.parquet",
    "run_windows": "run_windows.parquet",
    "seed_psms": "seed_psms.parquet",
    "peptide_quant": "peptide_quant.parquet",
    "protein_group_quant": "protein_group_quant.parquet",
    "fragment_quant": "fragment_quant.parquet",
    "spectra_ms2": "spectra/spectra_ms2.parquet",
    "spectra_ms1": "spectra/spectra_ms1.parquet",
    "isolation_windows": "spectra/isolation_windows.parquet",
    "ms2_to_ms1": "spectra/ms2_to_ms1.parquet",
    "overlap_losers": "groups/overlap_losers.parquet",
}

# Fixed file names inside a band directory ``groups/gNN``.
BAND_FILES: dict[str, str] = {
    "chromatograms": "chromatograms.parquet",
    "psms_competed": "psms_competed.parquet",
    "run_windows": "run_windows.parquet",
    "seed_psms": "seed_psms.parquet",
    "features": "features.parquet",
    "psms_extracted": "psms_extracted.parquet",
}

# JSON and parquet sidecars the engine writes without a manifest record.
SIDE_FILES: dict[str, str] = {
    "cal": "cal.json",
    "masscal": "seed_psms.parquet.masscal.json",
    "features_schema": "features.parquet.schema.json",
    "competed_schema": "psms_competed.parquet.schema.json",
    "multihead_summary": "fragment_library_precursors_multihead.parquet.summary.json",
    "plan": "groups/plan.json",
}

PEAKS_SIDECAR = "psms_extracted.parquet.peaks.parquet"
_TMP = re.compile(r"\.tmp-\d+-\d+")


@dataclass(frozen=True)
class Notice:
    """Something a user should know about the directory. ``code`` is stable for tests."""

    code: str
    message: str


@dataclass
class Band:
    """One window group ``groups/gNN`` of a grouped run."""

    index: int
    name: str
    root: Path
    artifacts: dict[str, Artifact]
    plan: dict[str, Any] | None
    side_files: dict[str, Path] = field(default_factory=dict)

    def artifact(self, kind: str) -> Artifact | None:
        return self.artifacts.get(kind)


@dataclass
class GroupedLayout:
    """The window-group structure of a grouped run (docs/33)."""

    plan: dict[str, Any]
    bands: list[Band]
    skipped: list[int]
    losers: Artifact | None
    pooled_chromatograms: bool
    pooled_competed: bool

    def band(self, index: int) -> Band | None:
        for b in self.bands:
            if b.index == index:
                return b
        return None


@dataclass
class Run:
    """One LC-MS run. ``index`` is the ``source`` value of its rows in pooled tables."""

    name: str
    index: int
    root: Path
    artifacts: dict[str, Artifact]
    side_files: dict[str, Path]
    grouped: GroupedLayout | None = None

    @property
    def label(self) -> str:
        return self.name or "run"

    def artifact(self, kind: str) -> Artifact | None:
        return self.artifacts.get(kind)

    def has(self, kind: str) -> bool:
        a = self.artifacts.get(kind)
        return a is not None and a.usable


class ResultSet:
    """A single run or an experiment, with its runs and the pooled tables."""

    def __init__(
        self,
        *,
        root: Path,
        kind: str,
        manifest: Manifest,
        runs: list[Run],
        scored: Artifact,
        scored_for_quant: Artifact | None,
        extra: dict[str, Artifact],
        notices: list[Notice],
        resolver: PathResolver,
        cache: Cache,
        duck: DuckDB,
        allow_unreleased: bool,
    ) -> None:
        self.root = root
        self.kind = kind
        self.manifest = manifest
        self.runs = runs
        self.scored = scored
        self.scored_for_quant = scored_for_quant
        self.extra = extra
        self.notices = notices
        self.resolver = resolver
        self.cache = cache
        self.duck = duck
        self.allow_unreleased = allow_unreleased
        self._memo: dict[Any, Any] = {}
        self._memo_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"ResultSet({self.kind}, {self.root}, runs={[r.label for r in self.runs]})"

    @property
    def is_experiment(self) -> bool:
        return self.kind == "experiment"

    @property
    def config(self) -> dict[str, Any]:
        return self.manifest.config

    def config_get(self, *keys: str, default: Any = None) -> Any:
        return self.manifest.config_get(*keys, default=default)

    def run(self, key: str | int | Run) -> Run:
        """A run by name, by ``source`` index (any integer type) or the Run itself."""
        if isinstance(key, Run):
            return key
        if isinstance(key, str):
            for r in self.runs:
                if r.name == key:
                    return r
        elif not isinstance(key, bool | np.bool_):
            try:
                index = operator.index(key)
            except TypeError:
                index = None
            if index is not None:
                for r in self.runs:
                    if r.index == index:
                        return r
        raise KeyError(f"no run {key!r}; runs are {[r.label for r in self.runs]}")

    def memo(self, key: Any, factory: Callable[[], Any]) -> Any:
        """A value memoised on this result set, computed once under a lock.

        The memo lives as long as the ResultSet; open the directory again to see files
        the engine has rewritten since.
        """
        with self._memo_lock:
            if key in self._memo:
                return self._memo[key]
        value = factory()
        with self._memo_lock:
            return self._memo.setdefault(key, value)

    def artifact(self, kind: str) -> Artifact | None:
        """An experiment-level artifact (``lfq_maxlfq``, ``mbr_transferred``, libraries...)."""
        return self.extra.get(kind)

    def all_artifacts(self) -> Iterator[tuple[str, Artifact]]:
        """Every discovered artifact once, with its directory relative to the root.

        The directory is ``""`` for the result-set root (a single run's own directory),
        the run name for an experiment run, and ``groups/gNN`` or ``<run>/groups/gNN``
        for a band.
        """
        seen: set[int] = set()

        def fresh(a: Artifact) -> bool:
            if id(a) in seen:
                return False
            seen.add(id(a))
            return True

        for a in (self.scored, self.scored_for_quant, *self.extra.values()):
            if a is not None and fresh(a):
                yield "", a
        for run in self.runs:
            prefix = f"{run.name}/" if run.name else ""
            for a in run.artifacts.values():
                if fresh(a):
                    yield run.name, a
            if run.grouped is not None:
                for band in run.grouped.bands:
                    for a in band.artifacts.values():
                        if fresh(a):
                            yield f"{prefix}groups/{band.name}", a

    def stage_timings(self) -> list[StageTiming]:
        """Stage wall times from the reports, grouped by (directory, stage)."""
        entries: list[tuple[str, Report]] = []
        seen: set[Path] = set()
        for scope, a in self.all_artifacts():
            if a.report is not None and a.report.path not in seen:
                seen.add(a.report.path)
                entries.append((scope or ".", a.report))
        return stage_timings(entries)

    def notice_codes(self) -> list[str]:
        return [n.code for n in self.notices]


# --------------------------------------------------------------------------- helpers


class _Builder:
    """Shared state while one directory is being discovered."""

    def __init__(
        self,
        root: Path,
        manifest: Manifest,
        remaps: Mapping[str, Any] | None,
        allow_unreleased: bool,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.allow_unreleased = allow_unreleased
        self.resolver = PathResolver(root, manifest.recorded_out_dir, remaps)
        self.notices: list[Notice] = []
        self.missing: list[tuple[str, str, bool]] = []
        self.moved = False

    def notice(self, code: str, message: str) -> None:
        self.notices.append(Notice(code, message))

    def flush(self) -> list[Notice]:
        """Close discovery: summarise the missing artifacts in one notice."""
        if self.missing:
            inside = [k for k, _, i in self.missing if i]
            outside = [f"{k} ({p})" for k, p, i in self.missing if not i]
            parts = []
            if inside:
                parts.append(
                    f"{len(inside)} inside the directory: "
                    + ", ".join(inside[:8])
                    + (" ..." if len(inside) > 8 else "")
                )
            if outside:
                parts.append(
                    f"{len(outside)} outside it (remap their root to use them): "
                    + ", ".join(outside[:4])
                    + (" ..." if len(outside) > 4 else "")
                )
            self.notice(
                "missing_artifact",
                f"{len(self.missing)} recorded artifact(s) not found; " + "; ".join(parts),
            )
            self.missing = []
        return self.notices

    def _version(
        self, kind: str, record: ArtifactRecord | None, report: Report | None
    ) -> VersionInfo:
        """The recorded version: the report's when it disagrees with the manifest.

        The report is written together with the file (a single-stage re-run rewrites the
        table and its report, not manifest.json), so it describes the file on disk.
        """
        rec_v = record.schema_version if record is not None else None
        rep_v = report.schema_version if report is not None else None
        if rep_v is not None and rec_v is not None and rep_v != rec_v:
            return VersionInfo(report.schema_name or kind, rep_v, "report")
        if rec_v is not None:
            return VersionInfo(record.schema_name or kind, rec_v, "manifest")
        if rep_v is not None:
            return VersionInfo(report.schema_name or kind, rep_v, "report")
        return VersionInfo(kind, None, "unrecorded")

    def finish(self, artifact: Artifact) -> Artifact:
        """Cross-check record and report, infer an unrecorded version, check it."""
        rec, rep = artifact.record, artifact.report
        where = str(artifact.path or artifact.recorded_path or artifact.key)
        if rec is not None and rep is not None:
            if rec.content_hash and rep.content_hash and rec.content_hash != rep.content_hash:
                self.notice(
                    "hash_mismatch",
                    f"{artifact.key}: the manifest and {rep.path.name} record different content "
                    "hashes; one of them describes another file.",
                )
            if (
                rec.schema_version is not None
                and rep.schema_version is not None
                and rec.schema_version != rep.schema_version
            ):
                self.notice(
                    "version_mismatch",
                    f"{artifact.key}: the manifest records version {rec.schema_version}, the "
                    f"report next to the file version {rep.schema_version}; the report's version "
                    "is used, because the report is written with the file.",
                )
                # Refuse when the manifest's version is not readable either, so that a stale
                # manifest cannot hide an unsupported file or the reverse.
                try:
                    check_version(
                        artifact.kind,
                        rec.schema_version,
                        where=where,
                        allow_unreleased=self.allow_unreleased,
                    )
                except SchemaVersionError as exc:
                    artifact.error = str(exc)
                    self.notice("unsupported_version", str(exc))
        if artifact.present:
            try:
                artifact.infer_version_if_unrecorded()
            except LayoutError as exc:  # the footer was read; the columns match no layout
                artifact.error = str(exc)
                artifact.error_class = LayoutError
                self.notice("layout", artifact.error)
                return artifact
            except Exception as exc:  # an unreadable footer: keep the artifact, record why
                artifact.error = f"{where}: cannot read the parquet footer ({exc})."
                artifact.error_class = ViewerError
                self.notice("unreadable", artifact.error)
                return artifact
            try:
                check_version(
                    artifact.kind,
                    artifact.version.version,
                    where=where,
                    allow_unreleased=self.allow_unreleased,
                )
            except SchemaVersionError as exc:
                artifact.error = str(exc)
                self.notice("unsupported_version", str(exc))
        return artifact

    def from_record(
        self, record: ArtifactRecord, *, kind: str | None = None, deleted_ok: bool = False
    ) -> Artifact:
        kind = kind or record.schema_name or record.base_key
        resolved = self.resolver.resolve(record.path)
        if resolved.how == "rerooted" and resolved.path is not None:
            try:
                same = Path(record.path).exists() and os.path.samefile(record.path, resolved.path)
            except OSError:
                same = False
            if not same:
                self.moved = True
        if resolved.path is not None:
            status = Status.PRESENT
        elif deleted_ok:
            status = Status.DELETED_AFTER_POOLING
        else:
            status = Status.MISSING
        report = load_report(resolved.path) if resolved.path is not None else None
        artifact = Artifact(
            kind=kind,
            key=record.key,
            path=resolved.path,
            status=status,
            version=self._version(kind, record, report),
            record=record,
            report=report,
            recorded_path=record.path,
            resolution=resolved.how,
            expected_path=resolved.expected,
        )
        if status is Status.MISSING:
            self.missing.append((record.key, record.path, resolved.inside_root))
        return self.finish(artifact)

    def from_path(
        self,
        kind: str,
        key: str,
        path: Path,
        *,
        record: ArtifactRecord | None = None,
        resolution: str = "fixed name",
    ) -> Artifact | None:
        """An artifact found by its fixed file name; None when the file does not exist."""
        if not path.is_file():
            return None
        report = load_report(path)
        artifact = Artifact(
            kind=kind,
            key=key,
            path=path,
            status=Status.PRESENT,
            version=self._version(kind, record, report),
            record=record,
            report=report,
            recorded_path=record.path if record is not None else None,
            resolution=resolution,
        )
        return self.finish(artifact)

    def side_files(self, directory: Path) -> dict[str, Path]:
        out = {}
        for name, rel in SIDE_FILES.items():
            p = directory / rel
            if p.is_file():
                out[name] = p
        return out


def _delete_band_intermediates(manifest: Manifest) -> bool:
    value = manifest.config_get("groups", "delete_band_intermediates", default=True)
    return bool(value)


def _in_progress(directories: Iterable[Path]) -> list[str]:
    """Temporary files of an unfinished engine write, in the given directories."""
    hits = []
    for d in directories:
        try:
            for entry in os.scandir(d):
                if entry.is_file() and _TMP.search(entry.name):
                    hits.append(os.path.join(d, entry.name))
        except OSError:
            continue
    return hits


def _grouped_layout(
    builder: _Builder,
    run_root: Path,
    band_records: dict[str, dict[str, ArtifactRecord]],
    run_key: str,
    root_artifacts: dict[str, Artifact],
) -> GroupedLayout | None:
    plan_path = run_root / "groups" / "plan.json"
    plan = load_json(plan_path) if plan_path.is_file() else None
    if plan is None and not band_records:
        return None
    plan = plan or {}
    plan_bands = {int(b.get("index", i)): b for i, b in enumerate(plan.get("bands") or [])}
    names: dict[int, str] = {}
    for qual in band_records:
        idx = band_index(qual)
        if idx is not None:
            names[idx] = qual
    groups_dir = run_root / "groups"
    if groups_dir.is_dir():
        for entry in os.scandir(groups_dir):
            idx = band_index(entry.name)
            if entry.is_dir() and idx is not None:
                names.setdefault(idx, entry.name)
    delete_ok = _delete_band_intermediates(builder.manifest)
    bands = []
    for idx in sorted(names):
        name = names[idx]
        band_root = groups_dir / name
        artifacts: dict[str, Artifact] = {}
        for base, record in band_records.get(name, {}).items():
            deleted_ok = delete_ok and base in ("features", "psms_extracted")
            artifacts[base] = builder.from_record(
                record, kind=record.schema_name or base, deleted_ok=deleted_ok
            )
        for kind, rel in BAND_FILES.items():
            if kind not in artifacts:
                found = builder.from_path(kind, f"{kind}[{run_key}{name}]", band_root / rel)
                if found is not None:
                    artifacts[kind] = found
        bands.append(
            Band(
                index=idx,
                name=name,
                root=band_root,
                artifacts=artifacts,
                plan=plan_bands.get(idx),
                side_files=builder.side_files(band_root),
            )
        )
    skipped = sorted(i for i in plan_bands if i not in names)
    deleted = [
        f"{b.name}/{k}"
        for b in bands
        for k, a in b.artifacts.items()
        if a.status is Status.DELETED_AFTER_POOLING
    ]
    if deleted:
        builder.notice(
            "deleted_after_pooling",
            f"{len(deleted)} band intermediate(s) listed in the manifest were deleted after "
            "pooling (groups.delete_band_intermediates): "
            + ", ".join(sorted(deleted)[:6])
            + (" ..." if len(deleted) > 6 else ""),
        )
    if skipped:
        builder.notice(
            "skipped_bands", f"plan bands without a directory (skipped at run time): {skipped}"
        )
    losers = root_artifacts.get("overlap_losers")
    return GroupedLayout(
        plan=plan,
        bands=bands,
        skipped=skipped,
        losers=losers,
        pooled_chromatograms=bool(
            root_artifacts.get("chromatograms") and root_artifacts["chromatograms"].present
        ),
        pooled_competed=bool(
            root_artifacts.get("psms_competed") and root_artifacts["psms_competed"].present
        ),
    )


def _peaks_sidecar(builder: _Builder, directory: Path, key: str) -> Artifact | None:
    path = directory / PEAKS_SIDECAR
    if not path.is_file():
        return None
    return builder.from_path("psms_extracted_peaks", key, path)


# --------------------------------------------------------------------------- entry points


def open_results(
    path: str | os.PathLike[str],
    *,
    remaps: Mapping[str, str | os.PathLike[str]] | None = None,
    allow_unreleased: bool = False,
    cache: Cache | None = None,
    duck: DuckDB | None = None,
) -> ResultSet:
    """Open a MuMDIA run directory or experiment directory, read-only.

    ``remaps`` maps a recorded root (for example ``C:\\Users\\old``) to where those
    files are now; it is used for inputs outside the run directory, such as the
    library. ``allow_unreleased`` accepts the ion-mobility schema versions of the
    unreleased MuMDIA branch.
    """
    root = Path(os.path.abspath(path))
    if not root.is_dir():
        raise NotAResultDirectory(f"{root} is not a directory.")
    cache = cache or Cache()
    cache.forbid(root)
    duck = duck or DuckDB(temp_directory=cache.temp_dir())
    if (root / "experiment_manifest.json").is_file():
        manifest = load_manifest(root / "experiment_manifest.json")
        rs = _open_experiment(root, manifest, remaps, allow_unreleased, cache, duck)
    elif (root / "manifest.json").is_file():
        manifest = load_manifest(root / "manifest.json")
        rs = _open_single(root, manifest, remaps, allow_unreleased, cache, duck)
    else:
        raise NotAResultDirectory(_explain_not_a_result(root))
    # Without a readable scored table nothing can be shown: raise its own error.
    rs.scored.require()
    return rs


def _explain_not_a_result(root: Path) -> str:
    parent = root.parent
    exp = parent / "experiment_manifest.json"
    if exp.is_file():
        return (
            f"{root} is a run directory of the experiment {parent}. Open the experiment "
            "directory; its runs are listed there."
        )
    if list(root.glob("*.report.json")):
        return (
            f"{root} has stage reports but no manifest.json: it is an unfinished or failed run, "
            "or the output of single stages. Open a directory written by `mumdia run` or "
            "`mumdia run-experiment`."
        )
    if (root / "peptides.tsv").is_file():
        return (
            f"{root} holds only TSV reports, which are presentation files; open the run directory."
        )
    return (
        f"{root} has no manifest.json or experiment_manifest.json; "
        "it is not a MuMDIA result directory."
    )


def _open_single(
    root: Path, manifest: Manifest, remaps, allow_unreleased: bool, cache: Cache, duck: DuckDB
) -> ResultSet:
    b = _Builder(root, manifest, remaps, allow_unreleased)
    artifacts: dict[str, Artifact] = {}
    band_records: dict[str, dict[str, ArtifactRecord]] = {}
    for key, record in manifest.artifacts.items():
        base, qual = split_key(key)
        if qual is not None and band_index(qual) is not None:
            band_records.setdefault(qual, {})[base] = record
            continue
        kind = record.schema_name or base
        artifacts[base] = b.from_record(record, kind=kind)
    peaks = _peaks_sidecar(b, root, "psms_extracted_peaks")
    if peaks is not None:
        artifacts["psms_extracted_peaks"] = peaks
    grouped = _grouped_layout(b, root, band_records, "", artifacts)
    run = Run(
        name="",
        index=0,
        root=root,
        artifacts=artifacts,
        side_files=b.side_files(root),
        grouped=grouped,
    )
    if "psms_scored" not in artifacts:
        raise NotAResultDirectory(f"{root}: the manifest records no psms_scored table.")
    scored = artifacts["psms_scored"]
    extra: dict[str, Artifact] = {}
    for name in ("fragment_library_precursors", "fragment_library_fragments"):
        if name in artifacts:
            extra[name] = artifacts[name]
    if "fragment_library_precursors" in artifacts:
        artifacts.setdefault("rt_library", artifacts["fragment_library_precursors"])
    tmp = _in_progress([root, root / "spectra", root / "groups"])
    if tmp:
        b.notice(
            "in_progress",
            f"unfinished engine writes found ({len(tmp)} temporary files); "
            "the results may be incomplete.",
        )
    if b.moved:
        b.notice(
            "moved",
            f"the directory was written as {manifest.recorded_out_dir}; paths were "
            f"re-rooted onto {root}.",
        )
    return ResultSet(
        root=root,
        kind="run",
        manifest=manifest,
        runs=[run],
        scored=scored,
        scored_for_quant=None,
        extra=extra,
        notices=b.flush(),
        resolver=b.resolver,
        cache=cache,
        duck=duck,
        allow_unreleased=allow_unreleased,
    )


def _open_experiment(
    root: Path, manifest: Manifest, remaps, allow_unreleased: bool, cache: Cache, duck: DuckDB
) -> ResultSet:
    b = _Builder(root, manifest, remaps, allow_unreleased)
    exp = manifest.experiment or {}
    records = manifest.artifacts
    run_names = [str(r) for r in (exp.get("runs") or [])]
    if not run_names:
        run_names = sorted(
            {q for k in records for base, q in [split_key(k)] if base == "scored" and q}
        )
    per_run: dict[str, dict[str, ArtifactRecord]] = {}
    top: dict[str, ArtifactRecord] = {}
    for key, record in records.items():
        base, qual = split_key(key)
        if qual is not None:
            per_run.setdefault(qual, {})[base] = record
        else:
            top[base] = record

    def top_artifact(
        kind: str, key: str, recorded: str | None, fixed: Path | None, schema: str | None = None
    ) -> Artifact | None:
        record = top.get(key)
        if record is not None:
            return b.from_record(record, kind=schema or record.schema_name or kind)
        if recorded:
            resolved = b.resolver.resolve(recorded)
            if resolved.path is not None:
                return b.from_path(schema or kind, key, resolved.path, resolution=resolved.how)
        if fixed is not None:
            return b.from_path(schema or kind, key, fixed)
        return None

    scored = top_artifact(
        "psms_scored",
        "scored_combined",
        exp.get("scored_combined"),
        root / "scored_combined.parquet",
        schema="psms_scored",
    )
    if scored is None:
        raise NotAResultDirectory(f"{root}: the experiment has no scored_combined table.")
    # With match-between-runs, quant and the report read scored_mbr.parquet (recorded as
    # `scored_for_quant`); without it, `experiment.scored_for_quant` names scored_combined.
    scored_for_quant = None
    sfq = exp.get("scored_for_quant")
    if "scored_for_quant" in top:
        scored_for_quant = b.from_record(top["scored_for_quant"], kind="psms_scored")
    elif sfq:
        resolved = b.resolver.resolve(sfq)
        if resolved.path is not None and resolved.path != scored.path:
            scored_for_quant = b.from_path(
                "psms_scored", "scored_for_quant", resolved.path, resolution=resolved.how
            )

    extra: dict[str, Artifact] = {}
    lfq = top_artifact("lfq_maxlfq", "lfq_maxlfq", exp.get("lfq"), root / "lfq_maxlfq.parquet")
    if lfq is not None:
        extra["lfq_maxlfq"] = lfq
        if lfq.path is not None:
            for level in ("peptide", "precursor"):
                sib = lfq.path.with_name(lfq.path.name + f".{level}.parquet")
                found = b.from_path("lfq_maxlfq_sibling", f"lfq_maxlfq.{level}", sib)
                if found is not None:
                    extra[f"lfq_maxlfq_{level}"] = found
    mbr_strategy = str(exp.get("mbr", "None"))
    if mbr_strategy != "None":
        mbr_table = b.from_path(
            "mbr_transferred", "mbr_transferred", root / "mbr_transferred.parquet"
        )
        if mbr_table is not None:
            extra["mbr_transferred"] = mbr_table
    stale = [
        name
        for name in ("mbr_transferred.parquet", "scored_mbr.parquet")
        if (root / name).is_file()
    ]
    if mbr_strategy == "None" and stale:
        b.notice(
            "stale_mbr",
            f"{', '.join(stale)} exist although the manifest records no match-between-runs "
            "(experiment.mbr = None): they are left over from an earlier run and are not used.",
        )
    # The searched library: FASTA mode builds it at the root; library mode records inputs.
    for name, input_key in (
        ("fragment_library_precursors", "lib_precursors"),
        ("fragment_library_fragments", "lib_fragments"),
    ):
        found = b.from_path(name, name, root / f"{name}.parquet")
        if found is None and input_key in manifest.inputs:
            resolved = b.resolver.resolve(manifest.inputs[input_key].path)
            if resolved.path is not None:
                found = b.from_path(name, name, resolved.path, resolution=resolved.how)
            else:
                b.notice(
                    "missing_input",
                    f"input {input_key} recorded at "
                    f"{manifest.inputs[input_key].path} was not found; remap its root to use it.",
                )
        if found is not None:
            extra[name] = found
    deeplc_lib = b.from_path(
        "fragment_library_precursors",
        "fragment_library_precursors_deeplc",
        root / "fragment_library_precursors_deeplc.parquet",
    )
    if deeplc_lib is not None:
        extra["fragment_library_precursors_deeplc"] = deeplc_lib

    runs: list[Run] = []
    for index, name in enumerate(run_names):
        run_root = root / name
        recs = per_run.get(name, {})
        artifacts: dict[str, Artifact] = {}
        if "scored" in recs:
            artifacts["psms_scored"] = b.from_record(recs["scored"], kind="psms_scored")
        for base in ("peptide_quant", "protein_group_quant"):
            if base in recs:
                artifacts[base] = b.from_record(recs[base], kind=base)
        for kind, rel in RUN_FILES.items():
            if kind in artifacts:
                continue
            fixed = run_root / ("scored.parquet" if kind == "psms_scored" else rel)
            found = b.from_path(kind, f"{kind}[{name}]", fixed)
            if found is not None:
                artifacts[kind] = found
        peaks = _peaks_sidecar(b, run_root, f"psms_extracted_peaks[{name}]")
        if peaks is not None:
            artifacts["psms_extracted_peaks"] = peaks
        grouped = _grouped_layout(b, run_root, {}, f"{name}/", artifacts)
        runs.append(
            Run(
                name=name,
                index=index,
                root=run_root,
                artifacts=artifacts,
                side_files=b.side_files(run_root),
                grouped=grouped,
            )
        )
        if not run_root.is_dir():
            b.notice("missing_run", f"run directory {run_root} does not exist.")

    _attach_rt_libraries(b, root, runs, extra)
    dirs = [root] + [r.root for r in runs] + [r.root / "spectra" for r in runs]
    tmp = _in_progress(dirs)
    if tmp:
        b.notice(
            "in_progress",
            f"unfinished engine writes found ({len(tmp)} temporary files); "
            "the results may be incomplete.",
        )
    if b.moved:
        b.notice(
            "moved",
            f"the directory was written as {manifest.recorded_out_dir}; paths were "
            f"re-rooted onto {root}.",
        )
    return ResultSet(
        root=root,
        kind="experiment",
        manifest=manifest,
        runs=runs,
        scored=scored,
        scored_for_quant=scored_for_quant,
        extra=extra,
        notices=b.flush(),
        resolver=b.resolver,
        cache=cache,
        duck=duck,
        allow_unreleased=allow_unreleased,
    )


def _attach_rt_libraries(
    b: _Builder, root: Path, runs: list[Run], extra: dict[str, Artifact]
) -> None:
    """Find the precursor table each run's RT calibration read (docs/08).

    Order: the run's own multi-head or fine-tuned table; under
    ``experiment.rt_library_scope = first_run_only`` (the default) the first run's; the
    experiment-level DeepLC table; the searched library.
    """
    scope = b.manifest.config_get("experiment", "rt_library_scope", default="first_run_only")
    scope = str(scope).lower()
    names = (
        "fragment_library_precursors_multihead.parquet",
        "fragment_library_precursors_ft.parquet",
    )
    first: Artifact | None = None
    for i, run in enumerate(runs):
        own = None
        for fname in names:
            own = b.from_path(
                "fragment_library_precursors", f"rt_library[{run.name}]", run.root / fname
            )
            if own is not None:
                own.resolution = f"own {fname}"
                break
        if i == 0:
            first = own
        chosen = own
        if chosen is None and scope in ("first_run_only", "firstrunonly") and first is not None:
            chosen = first
        if chosen is None:
            chosen = extra.get("fragment_library_precursors_deeplc") or extra.get(
                "fragment_library_precursors"
            )
        if chosen is not None:
            run.artifacts["rt_library"] = chosen
