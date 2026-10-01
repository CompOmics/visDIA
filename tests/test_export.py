"""Table exports: every row of the filtered table, as the pages show them."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.export import table_export, table_frame
from mumdia_viewer.data.tables import TableQuery, identification_table


@pytest.mark.parametrize("name", ["single", "experiment"])
@pytest.mark.parametrize("unit", ["precursor", "peptide", "protein_group"])
def test_the_export_is_every_page_together(open_fixture, monkeypatch, name, unit):
    rs = open_fixture(name)
    import mumdia_viewer.data.export as E

    monkeypatch.setattr(E, "MAX_LIMIT", 7)  # force several pages
    query = TableQuery(unit=unit, threshold=0.05, include_decoys=True, offset=3, limit=2)
    df = table_frame(rs, query)
    whole = identification_table(
        rs, TableQuery(unit=unit, threshold=0.05, include_decoys=True, limit=10_000)
    )
    assert len(df) == whole.total == df.attrs["total"]
    expected = whole.rows.drop(columns=[c for c in E.HIDDEN if c in whole.rows.columns])

    # Joining pages can change a column's type (object against string); the values and the
    # TSV are the same.
    def plain(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reset_index(drop=True).astype(object)
        return frame.where(frame.notna(), None)

    pd.testing.assert_frame_equal(plain(df), plain(expected))


def test_tsv_text(open_fixture):
    rs = open_fixture("single")
    out = table_export(rs, TableQuery(unit="precursor", threshold=0.01))
    back = pd.read_csv(io.StringIO(out.text), sep="\t")
    assert len(back) == out.rows > 0
    assert "candidate_id" in back.columns and "source" not in back.columns
    assert out.filename.endswith("-precursor-q0.01.tsv")
    assert "precursor" in out.description


def test_too_large_is_refused(open_fixture):
    rs = open_fixture("single")
    with pytest.raises(ViewerError, match="at most"):
        table_frame(
            rs, TableQuery(unit="precursor", threshold=None, include_decoys=True), max_rows=10
        )
