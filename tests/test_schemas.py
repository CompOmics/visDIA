import pyarrow as pa
import pytest

from mumdia_viewer.data.errors import LayoutError, SchemaVersionError
from mumdia_viewer.data.schemas import (
    check_columns,
    check_version,
    chromatogram_layout,
    infer_version,
    supported_versions,
)


def _schema(*names: str) -> pa.Schema:
    return pa.schema([(n, pa.float32()) for n in names])


def test_released_and_dev_versions_are_accepted():
    check_version("psms_scored", 4, where="x")
    check_version("psms_scored", 3, where="x")  # pre-release development builds
    check_version("chromatograms", 1, where="x")
    check_version("chromatograms", 2, where="x")
    check_version("features", 1, where="x")
    check_version("psms_competed", 3, where="x")


def test_unknown_future_version_is_refused_with_a_clear_message():
    with pytest.raises(SchemaVersionError) as err:
        check_version("psms_scored", 5, where="run/psms_scored.parquet")
    text = str(err.value)
    assert "run/psms_scored.parquet" in text and "version 5" in text
    assert "3, 4" in text and "newer MuMDIA" in text


def test_ion_mobility_versions_need_opt_in():
    with pytest.raises(SchemaVersionError, match="PR #140"):
        check_version("chromatograms", 4, where="c.parquet")
    check_version("chromatograms", 4, where="c.parquet", allow_unreleased=True)
    assert 5 in supported_versions("psms_extracted", allow_unreleased=True)
    assert 5 not in supported_versions("psms_extracted")


def test_unrecorded_version_and_unknown_schema_are_not_checked():
    check_version("psms_scored", None, where="x")
    check_version("something_new", 99, where="x")


def test_chromatogram_layouts():
    v1 = chromatogram_layout(_schema("candidate_id", "rt", "intensity"))
    assert (v1.family, v1.has_im, v1.schema_version) == (1, False, 1)
    v2 = chromatogram_layout(_schema("rt_axis", "intensity_trimmed", "trace_offset", "trace_len"))
    assert v2.schema_version == 2
    v3 = chromatogram_layout(_schema("rt", "intensity", "im"))
    assert v3.schema_version == 3
    v4 = chromatogram_layout(
        _schema("rt_axis", "intensity_trimmed", "trace_offset", "trace_len", "im_trimmed")
    )
    assert v4.schema_version == 4


@pytest.mark.parametrize(
    "names",
    [
        ("rt", "intensity", "trace_len"),  # a v2 column beside the v1 lists
        ("rt_axis", "intensity_trimmed", "trace_offset"),  # incomplete v2
        ("rt_axis", "intensity_trimmed", "trace_offset", "trace_len", "im"),
        ("rt", "intensity", "im_trimmed"),
        ("candidate_id",),
    ],
)
def test_chromatogram_mixtures_are_refused(names):
    with pytest.raises(LayoutError):
        chromatogram_layout(_schema(*names))


def test_column_contract():
    schema = pa.schema([("candidate_id", pa.uint32()), ("rt_pred_cal", pa.float64())])
    with pytest.raises(LayoutError, match="rt_lo, rt_hi"):
        check_columns("run_windows", schema, where="w")
    scored = pa.schema(
        [
            (c, pa.float64())
            for c in (
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
            )
        ]
    )
    check_columns("psms_scored", scored, where="s", version=3)
    with pytest.raises(LayoutError, match="selected_peak_rank"):
        check_columns("psms_scored", scored, where="s", version=4)
    # Extra columns are ignored.
    check_columns("psms_scored", scored.append(pa.field("is_transferred", pa.bool_())), where="s")


def test_recorded_chromatogram_version_must_match_the_columns():
    v2 = pa.schema(
        [
            (c, pa.float32())
            for c in (
                "candidate_id",
                "frag_name",
                "frag_mz",
                "frag_obs_mz",
                "predicted_intensity",
                "rt_axis",
                "intensity_trimmed",
                "trace_offset",
                "trace_len",
            )
        ]
    )
    check_columns("chromatograms", v2, where="c", version=2)
    with pytest.raises(LayoutError, match="recorded chromatograms schema version is 1"):
        check_columns("chromatograms", v2, where="c", version=1)


def test_infer_version():
    assert infer_version("psms_scored", _schema("selected_peak_rank")) == 4
    assert infer_version("psms_scored", _schema("score")) == 3
    assert infer_version("psms_extracted", _schema("peak_rank")) == 2
    assert infer_version("chromatograms", _schema("rt", "intensity")) == 1
    assert (
        infer_version("lfq_maxlfq", _schema("protein_group", "run", "quantity", "n_features")) == 1
    )
