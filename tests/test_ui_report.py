"""The static overview report: self-contained, and the same numbers as the page."""

from __future__ import annotations

import re

import pytest

from mumdia_viewer.data.counts import unit_counts
from mumdia_viewer.ui.report import overview_report


@pytest.mark.parametrize("name", ["single", "experiment", "grouped"])
def test_the_report_is_one_offline_file(open_fixture, name):
    rs = open_fixture(name)
    text = overview_report(rs, 0.05)
    head, _, body = text.partition("</head>")
    # No file is fetched: no external script, stylesheet or image.
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=", text, flags=re.I)
    assert "<script>" in head and "Plotly" in head
    for c in unit_counts(rs, 0.05):
        assert f"{c.n_target:,}" in body
    assert "q &le; 0.05" in body and rs.root.name in body
    assert body.count('class="plotly-graph-div"') >= 3
    assert "Every number is MuMDIA's own column" in body
