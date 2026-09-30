"""Which rescorer produced the scored table, and in which FDR mode.

The scored table's report records the classifier path that actually ran
(``params.classifier``: ``native_tda``, ``mokapot``, ``nn_torch``, ``entrapment_gbm``,
``entrapment_native``, ``not_run_empty``) next to the configured one
(``params.classifier_requested``). A requested entrapment rescorer that fell back to
``native_tda`` runs in target-decoy mode. In entrapment mode every q column is an
entrapment estimate and the engine's target counts exclude the spike-ins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .discovery import ResultSet
from .reports import normalise_enum

Mode = Literal["target_decoy", "entrapment"]


@dataclass(frozen=True)
class RescoreInfo:
    classifier: str | None
    classifier_requested: str | None
    model_identity: str | None
    mode: Mode
    fallback: bool
    group_by: str | None
    group_by_source: str | None

    @property
    def label(self) -> str:
        ran = self.classifier or "unknown"
        text = f"{ran} ({self.mode.replace('_', '-')} q values)"
        if self.fallback:
            text += f"; configured {self.classifier_requested}, which did not run"
        return text


def _requested_matches(requested: str | None, ran: str | None) -> bool:
    if not requested or not ran:
        return True
    req, got = normalise_enum(requested), normalise_enum(ran)
    if req == got:
        return True
    return req == "entrapment" and got.startswith("entrapment")


def rescore_info(rs: ResultSet) -> RescoreInfo:
    """Rescorer identity and FDR mode from the pooled scored table's report."""
    report = rs.scored.report
    params = report.params if report is not None else {}
    classifier = params.get("classifier")
    requested = params.get("classifier_requested")
    identity = report.model_identity if report is not None else None
    if identity is None:
        identity = rs.manifest.model_identities.get("rescorer")
    if classifier is None and rs.manifest.experiment:
        classifier = rs.manifest.experiment.get("rescorer")
    mode: Mode = (
        "entrapment" if classifier and str(classifier).startswith("entrapment") else "target_decoy"
    )
    group_by, source = None, None
    for run in rs.runs:
        competed = run.artifact("psms_competed")
        if competed is not None and competed.report is not None:
            value = competed.report.params.get("group_by")
            if value is not None:
                group_by, source = str(value), f"{competed.report.path.name} params.group_by"
                break
    if group_by is None:
        value = rs.config_get("compete", "group_by")
        if value is not None:
            group_by, source = str(value), "config_json compete.group_by"
    return RescoreInfo(
        classifier=str(classifier) if classifier is not None else None,
        classifier_requested=str(requested) if requested is not None else None,
        model_identity=str(identity) if identity is not None else None,
        mode=mode,
        fallback=not _requested_matches(requested, classifier),
        group_by=group_by,
        group_by_source=source,
    )


def precursor_q_is_precursor_unit(info: RescoreInfo) -> bool:
    """False when competition collapsed charge and modification siblings (group_by base_peptide).

    Under ``compete.group_by = base_peptide`` the rows reaching rescore hold about one
    form per peptide, so counts on ``precursor_q`` are base-peptide counts.
    """
    if info.group_by is None:
        return True
    return normalise_enum(info.group_by) not in ("basepeptide", "precursor")
