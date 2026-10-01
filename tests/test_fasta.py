"""FASTA index, the recorded FASTA of a run, and protein coverage against a brute force.

The brute force reads the fixture's scored table with pyarrow, strips the peptidoforms
in Python, and locates each target peptide of a protein group in the sequence with
``re`` (overlapping occurrences). It shares no code with ``data.fasta``.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from mumdia_viewer.data import open_results
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.fasta import (
    Fasta,
    FastaSource,
    find_recorded_fasta,
    group_members,
    protein_coverage,
)

FIXTURE_FASTA = Path(__file__).parent / "fixtures" / "smoke" / "test_data" / "fixture.fasta"


def _strip(peptidoform: str) -> str:
    text = re.sub(r"^DECOY_", "", peptidoform)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    return re.sub(r"[^A-Za-z]", "", text)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- index


def test_headers_keys_and_sequences(tmp_path):
    fa = _write(
        tmp_path / "a.fasta",
        ">sp|P12345|ABC_HUMAN Some protein OS=Homo sapiens\n"
        "MKT\nAYI\n\nakq*\n"
        ">PLAIN1 a protein without pipes\n"
        "PEPTIDEK\n"
        ">tr|Q99999|ABC_HUMAN the same entry name again\n"
        "MMMM\n",
    )
    index = Fasta.read([fa])
    assert index.n_entries == 3
    first = index.get("ABC_HUMAN")
    assert first is not None and first.sequence == "MKTAYIAKQ" and first.accession == "P12345"
    assert index.get("P12345") is first and index.get("sp|P12345|ABC_HUMAN") is first
    assert index.get(" P12345 ") is first
    assert index.get("Q99999").sequence == "MMMM"  # its accession is its own key
    assert index.duplicates == 1  # ABC_HUMAN is shared; the first entry keeps it
    plain = index.get("PLAIN1")
    assert (
        plain is not None and plain.name is None and plain.description == "a protein without pipes"
    )
    assert index.get("nope") is None
    assert "a.fasta (3 proteins)" in index.label


def test_bad_files_are_refused(tmp_path):
    with pytest.raises(ViewerError, match="not found"):
        Fasta.read([tmp_path / "missing.fasta"])
    with pytest.raises(ViewerError, match="no FASTA entry"):
        Fasta.read([_write(tmp_path / "empty.fasta", "just text\n")])


def test_two_files_are_one_index(tmp_path):
    a = _write(tmp_path / "a.fasta", ">sp|A1|AA_HUMAN\nMAAA\n")
    b = _write(tmp_path / "b.fasta", ">sp|B1|BB_YEAST\nMBBB\n")
    index = Fasta.read([FastaSource(a, "given"), b])
    assert index.get("AA_HUMAN").source == "a.fasta" and index.get("BB_YEAST").source == "b.fasta"
    assert index.n_entries == 2 and [s.how for s in index.sources] == ["given", "given"]


def test_group_members():
    assert group_members("A_HUMAN;B_HUMAN") == ("A_HUMAN", "B_HUMAN")
    assert group_members(" A ; ") == ("A",)


# --------------------------------------------------------------------------- recorded FASTA


def test_the_recorded_fasta_is_found_beside_the_run(open_fixture):
    found = find_recorded_fasta(open_fixture("single"))
    assert [s.path.resolve() for s in found] == [FIXTURE_FASTA.resolve()]
    assert "relative to" in found[0].how and "size" in found[0].how
    exp = find_recorded_fasta(open_fixture("experiment"))
    assert [s.path.resolve() for s in exp] == [FIXTURE_FASTA.resolve()]


def test_a_file_of_another_size_is_not_the_recorded_fasta(tmp_path, fixture_dir):
    root = tmp_path / "smoke" / "out"
    shutil.copytree(fixture_dir("single"), root)
    (tmp_path / "smoke" / "test_data").mkdir()
    _write(tmp_path / "smoke" / "test_data" / "fixture.fasta", ">sp|X|Y\nMK\n")
    assert find_recorded_fasta(open_results(root)) == []
    shutil.copy2(FIXTURE_FASTA, tmp_path / "smoke" / "test_data" / "fixture.fasta")
    assert len(find_recorded_fasta(open_results(root))) == 1


# --------------------------------------------------------------------------- coverage


def _brute(rs, sequence: str, group: str, t: float):
    table = pq.read_table(
        rs.scored.path,
        columns=["label", "protein_group", "base_peptide_id", "peptidoform", "peptide_q_value"],
    )
    rows = table.filter(
        pc.and_(pc.equal(table["label"], "target"), pc.equal(table["protein_group"], group))
    ).to_pandas()
    peptides = {}
    for key, part in rows.groupby("base_peptide_id"):
        stripped = {_strip(p) for p in part["peptidoform"]}
        assert len(stripped) == 1, (key, stripped)
        peptides[int(key)] = (stripped.pop(), float(part["peptide_q_value"].min()) <= t)
    spans, unmatched = set(), []
    states = np.zeros(len(sequence), dtype=np.int8)
    for key, (seq, passes) in peptides.items():
        starts = [m.start() for m in re.finditer(f"(?={re.escape(seq)})", sequence)]
        if not starts:
            unmatched.append(seq)
        for s in starts:
            spans.add((key, s, s + len(seq), passes))
    for _key, a, b, passes in spans:
        if not passes:
            states[a:b] = np.maximum(states[a:b], 1)
    for _key, a, b, passes in spans:
        if passes:
            states[a:b] = 2
    return peptides, spans, sorted(unmatched), states


@pytest.mark.parametrize("name", ["single", "experiment", "mbr", "grouped"])
@pytest.mark.parametrize("t", [0.01, 0.05])
def test_coverage_equals_the_brute_force(open_fixture, name, t):
    rs = open_fixture(name)
    index = Fasta.read([FIXTURE_FASTA])
    groups = pq.read_table(rs.scored.path, columns=["label", "protein_group"]).to_pandas()
    targets = sorted(set(groups.loc[groups["label"] == "target", "protein_group"]))
    assert targets
    for group in targets:
        cov = protein_coverage(rs, index, group, threshold=t)
        entry = index.get(group)
        assert entry is not None and cov.entry is entry and cov.member == group
        peptides, spans, unmatched, states = _brute(rs, entry.sequence, group, t)
        assert cov.n_peptides == len(peptides)
        assert cov.n_passing == sum(p for _, p in peptides.values())
        got = {(s.base_peptide_id, s.start, s.end, s.passes) for s in cov.spans}
        assert got == spans, group
        assert sorted(cov.unmatched) == unmatched
        np.testing.assert_array_equal(cov.states(), states)
        assert cov.covered_any == int(np.count_nonzero(states))
        assert cov.covered_passing == int(np.count_nonzero(states == 2))
        assert cov.fraction_any == pytest.approx(np.count_nonzero(states) / len(entry.sequence))
        runs = cov.runs()
        assert runs[0][0] == 0 and runs[-1][1] == len(entry.sequence)
        assert all(a < b for a, b, _ in runs)
        assert all(runs[i][1] == runs[i + 1][0] for i in range(len(runs) - 1))
        assert all(states[a] == st for a, _, st in runs)
        assert "Computed by the viewer" in cov.note and f"{t:g}" in cov.note


def test_coverage_refuses_decoys_and_strangers(open_fixture):
    rs = open_fixture("single")
    index = Fasta.read([FIXTURE_FASTA])
    with pytest.raises(ViewerError, match="decoy"):
        protein_coverage(rs, index, "DECOY_sp|FIXT14|FIX14_TEST")
    with pytest.raises(ViewerError, match="not a member"):
        protein_coverage(rs, index, "sp|FIXT14|FIX14_TEST", member="sp|FIXT01|FIX01_TEST")


def test_a_protein_missing_from_the_fasta_is_named(open_fixture, tmp_path):
    rs = open_fixture("single")
    other = Fasta.read([_write(tmp_path / "other.fasta", ">sp|Q1|OTHER_HUMAN\nMKKK\n")])
    cov = protein_coverage(rs, other, "sp|FIXT14|FIX14_TEST")
    assert cov.entry is None and cov.length == 0 and cov.fraction_any is None
    assert cov.spans == () and len(cov.unmatched) == cov.n_peptides
    assert "is not in other.fasta" in cov.note


def test_an_unmatched_peptide_is_reported(open_fixture, tmp_path):
    rs = open_fixture("single")
    entry = Fasta.read([FIXTURE_FASTA]).get("sp|FIXT14|FIX14_TEST")
    cut = entry.sequence[: len(entry.sequence) // 2]
    half = Fasta.read([_write(tmp_path / "half.fasta", f">sp|FIXT14|FIX14_TEST\n{cut}\n")])
    cov = protein_coverage(rs, half, "sp|FIXT14|FIX14_TEST")
    assert cov.unmatched and all(seq not in cut for seq in cov.unmatched)
    assert cov.length == len(cut)


@pytest.mark.real_data
def test_real_coverage_with_the_hye_fasta(real_single):
    path = os.environ.get("MUMDIA_VIEWER_REAL_FASTA")
    if not path:
        pytest.skip("MUMDIA_VIEWER_REAL_FASTA is not set")
    rs = open_results(real_single)
    index = Fasta.read([path])
    cov = protein_coverage(rs, index, "ATLA3_HUMAN")
    assert cov.entry is not None and cov.length > 100 and not cov.unmatched
    assert 0 < cov.fraction_passing <= cov.fraction_any <= 1
    two = protein_coverage(rs, index, "K0754_HUMAN;MACF1_HUMAN", member="MACF1_HUMAN")
    assert two.member == "MACF1_HUMAN" and two.members == ("K0754_HUMAN", "MACF1_HUMAN")
