import json
from pathlib import Path

import pytest

from mumdia_viewer.data.errors import NotAResultDirectory
from mumdia_viewer.data.manifest import band_index, load_manifest, split_key


def test_split_key_and_band_index():
    assert split_key("chromatograms[g00]") == ("chromatograms", "g00")
    assert split_key("scored[r0]") == ("scored", "r0")
    assert split_key("psms_scored") == ("psms_scored", None)
    assert band_index("g07") == 7 and band_index("g127") == 127
    assert band_index("r0") is None and band_index(None) is None


def test_single_run_manifest(fixture_dir):
    m = load_manifest(fixture_dir("single") / "manifest.json")
    assert m.kind == "run"
    assert m.mumdia_version == "0.5.0" and m.git_sha == "80d4874318cb"
    assert m.recorded_out_dir.endswith("smoke/out")
    rec = m.artifacts["psms_scored"]
    assert rec.schema_name == "psms_scored" and rec.schema_version == 4 and rec.rows == 284
    assert m.artifacts["chromatograms"].schema_version == 2
    assert m.config_get("compete", "group_by") == "peptidoform_charge"
    assert m.config_get("no", "such", default=5) == 5
    assert set(m.inputs) == {"fasta", "mzml"}


def test_experiment_manifest(fixture_dir):
    m = load_manifest(fixture_dir("experiment") / "experiment_manifest.json")
    assert m.kind == "experiment"
    assert m.experiment["runs"] == ["a", "b"]
    assert m.experiment["mbr"] == "None"
    assert {"scored_combined", "scored[a]", "peptide_quant[b]", "lfq_maxlfq"} <= set(m.artifacts)


def test_legacy_experiment_manifest_without_envelope(tmp_path: Path):
    path = tmp_path / "experiment_manifest.json"
    path.write_text(json.dumps({"runs": ["x", "y"], "scored_combined": "s.parquet"}))
    m = load_manifest(path)
    assert m.experiment == {"runs": ["x", "y"], "scored_combined": "s.parquet"}
    assert m.cli_args == [] and m.artifacts == {}


def test_unreadable_manifest(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    with pytest.raises(NotAResultDirectory):
        load_manifest(path)
