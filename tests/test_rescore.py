import json
import shutil
from pathlib import Path

from mumdia_viewer.data import open_results
from mumdia_viewer.data.reports import normalise_enum
from mumdia_viewer.data.rescore import precursor_q_is_precursor_unit, rescore_info


def test_target_decoy_runs(open_fixture):
    for name in ("single", "experiment", "grouped", "mbr"):
        info = rescore_info(open_fixture(name))
        assert info.mode == "target_decoy" and not info.fallback, name
        assert info.classifier == "native_tda"
        assert normalise_enum(info.group_by) == "peptidoformcharge", name
        assert precursor_q_is_precursor_unit(info)


def test_base_peptide_grouping_changes_the_precursor_unit(open_fixture):
    info = rescore_info(open_fixture("ovl_bp"))
    assert normalise_enum(info.group_by) == "basepeptide"
    assert info.group_by_source.endswith("params.group_by")
    assert not precursor_q_is_precursor_unit(info)


def _with_report_params(src: Path, dst: Path, **params) -> Path:
    shutil.copytree(src, dst)
    path = dst / "psms_scored.parquet.report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["params"].update(params)
    path.write_text(json.dumps(report), encoding="utf-8")
    return dst


def test_entrapment_mode_and_fallback(fixture_dir, tmp_path: Path):
    ent = _with_report_params(
        fixture_dir("single"),
        tmp_path / "ent",
        classifier="entrapment_native",
        classifier_requested="Entrapment",
    )
    info = rescore_info(open_results(ent))
    assert info.mode == "entrapment" and not info.fallback
    fell_back = _with_report_params(
        fixture_dir("single"),
        tmp_path / "fb",
        classifier="native_tda",
        classifier_requested="Entrapment",
    )
    info = rescore_info(open_results(fell_back))
    assert info.mode == "target_decoy" and info.fallback
    assert "did not run" in info.label
