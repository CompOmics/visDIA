"""Protein sequences from FASTA files, and the sequence coverage of a protein group.

MuMDIA's outputs name proteins but do not hold their sequences. A run searched from a
FASTA (``mumdia run --fasta``) records that file in its manifest; a library search does
not, and the viewer is then given the FASTA the library was built from.

Protein names are matched exactly against three keys of each FASTA entry: the first
token of the header (``sp|Q6DD88|ATLA3_HUMAN``, the protein name of a FASTA search),
the accession (``Q6DD88``) and the UniProt entry name (``ATLA3_HUMAN``, the protein
name of the HYE library). A key that two entries share keeps the first entry and is
counted in :attr:`Fasta.duplicates`.

Coverage is computed by the viewer, not by the engine: each peptide's sequence (its
peptidoform without modifications, the peptide table's ``sequence``) is located in the
protein sequence by exact string search, and every occurrence counts. A peptide passes
when ``peptide_q_value`` of its winning row is at most the threshold (the engine's
column; experiment-wide in an experiment).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .discovery import ResultSet
from .errors import ViewerError
from .paths import is_absolute_recorded
from .tables import MAX_LIMIT, TableQuery, identification_table

__all__ = [
    "Coverage",
    "Fasta",
    "FastaSource",
    "PeptideSpan",
    "ProteinEntry",
    "find_recorded_fasta",
    "group_members",
    "protein_coverage",
]

# Residue states of Coverage.states(), in drawing order.
NOT_COVERED, COVERED, PASSING = 0, 1, 2


@dataclass(frozen=True)
class ProteinEntry:
    """One FASTA entry. ``accession`` and ``name`` are set for UniProt-style headers."""

    key: str
    accession: str | None
    name: str | None
    description: str
    sequence: str
    source: str

    @property
    def length(self) -> int:
        return len(self.sequence)


def _parse_header(line: str) -> tuple[str, str | None, str | None, str]:
    text = line[1:].strip()
    key, _, description = text.partition(" ")
    parts = key.split("|")
    if len(parts) >= 3 and parts[1] and parts[2]:
        return key, parts[1], parts[2], description.strip()
    return key, None, None, description.strip()


def _entries(path: Path) -> Iterator[ProteinEntry]:
    source = path.name
    header: str | None = None
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _entry(header, chunks, source)
                header, chunks = line, []
            elif header is not None:
                chunks.append(line)
    if header is not None:
        yield _entry(header, chunks, source)


def _entry(header: str, chunks: list[str], source: str) -> ProteinEntry:
    key, accession, name, description = _parse_header(header)
    sequence = "".join(chunks).replace(" ", "").rstrip("*").upper()
    return ProteinEntry(key, accession, name, description, sequence, source)


@dataclass(frozen=True)
class FastaSource:
    """Where a FASTA file came from, in words for the UI."""

    path: Path
    how: str


@dataclass
class Fasta:
    """An index of the proteins of one or more FASTA files (exact keys, see the module)."""

    sources: tuple[FastaSource, ...]
    n_entries: int
    duplicates: int
    _index: dict[str, ProteinEntry] = field(repr=False)

    @classmethod
    def read(cls, sources: Iterable[FastaSource | str | Path]) -> Fasta:
        """Read every entry of the files; raises ViewerError for a missing or empty file."""
        resolved: list[FastaSource] = []
        for s in sources:
            resolved.append(s if isinstance(s, FastaSource) else FastaSource(Path(s), "given"))
        index: dict[str, ProteinEntry] = {}
        n = dup = 0
        for src in resolved:
            if not src.path.is_file():
                raise ViewerError(f"FASTA file not found: {src.path}")
            found = 0
            for entry in _entries(src.path):
                found += 1
                for k in {entry.key, entry.accession, entry.name}:
                    if not k:
                        continue
                    if k in index:
                        if index[k] is not entry:
                            dup += 1
                        continue
                    index[k] = entry
            if found == 0:
                raise ViewerError(f"no FASTA entry in {src.path} (no line starts with '>').")
            n += found
        return cls(tuple(resolved), n, dup, index)

    def get(self, protein: str) -> ProteinEntry | None:
        """The entry of a protein name (header token, accession or entry name), or None."""
        return self._index.get(protein.strip())

    @property
    def label(self) -> str:
        names = ", ".join(s.path.name for s in self.sources)
        return f"{names} ({self.n_entries:,} proteins)"

    @property
    def identity(self) -> tuple:
        return tuple((str(s.path), s.path.stat().st_mtime_ns) for s in self.sources)


def find_recorded_fasta(rs: ResultSet) -> list[FastaSource]:
    """The FASTA inputs the run recorded, where they are now.

    An input is used when the result set's resolver finds it, or, for a path recorded
    relative to the engine's unrecorded working directory, when the path exists relative
    to the run directory or one of its two parents. Either way its size must equal the
    recorded size; otherwise it is another file and is not used.
    """
    found: list[FastaSource] = []
    for key, inp in rs.manifest.inputs.items():
        if not key.lower().startswith("fasta"):
            continue
        candidates: list[tuple[Path, str]] = []
        resolved = rs.resolver.resolve(inp.path)
        if resolved.path is not None:
            candidates.append((Path(resolved.path), f"recorded input {key} ({resolved.how})"))
        if not is_absolute_recorded(inp.path):
            for base in (rs.root, rs.root.parent, rs.root.parent.parent):
                candidates.append(
                    (base / inp.path, f"recorded input {key}, found relative to {base}")
                )
        for path, how in candidates:
            try:
                size = os.stat(path).st_size
            except OSError:
                continue
            if inp.bytes is None or size == inp.bytes:
                found.append(FastaSource(path, how + "; the size equals the recorded size"))
                break
    return found


def group_members(protein_group: str) -> tuple[str, ...]:
    """The proteins of a protein group string (``A;B``)."""
    return tuple(p.strip() for p in protein_group.split(";") if p.strip())


@dataclass(frozen=True)
class PeptideSpan:
    """One occurrence of a peptide in the protein sequence (0-based, ``end`` exclusive)."""

    base_peptide_id: int
    sequence: str
    peptidoform: str
    start: int
    end: int
    passes: bool
    q: float | None


@dataclass(frozen=True)
class Coverage:
    """The coverage of one member of a protein group by the group's peptides."""

    protein_group: str
    member: str
    members: tuple[str, ...]
    entry: ProteinEntry | None
    spans: tuple[PeptideSpan, ...]
    unmatched: tuple[str, ...]
    n_peptides: int
    n_passing: int
    threshold: float
    note: str
    fasta_label: str

    @property
    def length(self) -> int:
        return self.entry.length if self.entry is not None else 0

    def states(self) -> np.ndarray:
        """Per residue: 0 not covered, 1 covered by peptides that do not pass, 2 passing."""
        out = np.zeros(self.length, dtype=np.int8)
        for s in self.spans:
            if not s.passes:
                out[s.start : s.end] = np.maximum(out[s.start : s.end], COVERED)
        for s in self.spans:
            if s.passes:
                out[s.start : s.end] = PASSING
        return out

    @property
    def covered_any(self) -> int:
        return int(np.count_nonzero(self.states()))

    @property
    def covered_passing(self) -> int:
        return int(np.count_nonzero(self.states() == PASSING))

    @property
    def fraction_any(self) -> float | None:
        return self.covered_any / self.length if self.length else None

    @property
    def fraction_passing(self) -> float | None:
        return self.covered_passing / self.length if self.length else None

    def runs(self) -> list[tuple[int, int, int]]:
        """``(start, end, state)`` runs of equal residue state over the whole sequence."""
        st = self.states()
        if st.size == 0:
            return []
        edges = np.flatnonzero(np.diff(st)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [st.size]])
        return [(int(a), int(b), int(st[a])) for a, b in zip(starts, ends, strict=True)]

    def spans_of(self, base_peptide_id: int) -> list[PeptideSpan]:
        return [s for s in self.spans if s.base_peptide_id == base_peptide_id]


def _occurrences(sequence: str, peptide: str) -> list[int]:
    out: list[int] = []
    if not peptide:
        return out
    i = sequence.find(peptide)
    while i >= 0:
        out.append(i)
        i = sequence.find(peptide, i + 1)
    return out


def _group_peptides(rs: ResultSet, protein_group: str) -> list[dict]:
    """Every target peptide of the group (one row per base_peptide_id, any q)."""
    rows: list[dict] = []
    offset = 0
    while True:
        page = identification_table(
            rs,
            TableQuery(
                unit="peptide",
                protein_group=protein_group,
                threshold=None,
                include_decoys=False,
                sort_by="score",
                offset=offset,
                limit=MAX_LIMIT,
            ),
        )
        rows += page.rows.to_dict("records")
        offset += MAX_LIMIT
        if offset >= page.total:
            return rows


def protein_coverage(
    rs: ResultSet,
    fasta: Fasta,
    protein_group: str,
    *,
    member: str | None = None,
    threshold: float = 0.01,
) -> Coverage:
    """The coverage of ``member`` (default: the first member in the FASTA) by the group's peptides.

    The peptides are the target rows of the peptide table with this exact protein group
    and no q filter. Raises ViewerError for a decoy group (its peptides are not in the
    FASTA) and for a member that is not in the group.
    """
    if protein_group.startswith("DECOY_"):
        raise ViewerError("A decoy protein group has no sequence in the FASTA, so no coverage.")
    members = group_members(protein_group)
    if not members:
        raise ViewerError("The protein group is empty, so it has no coverage.")
    if member is not None and member not in members:
        raise ViewerError(f"{member!r} is not a member of the protein group {protein_group!r}.")
    t = float(threshold)
    key = ("protein_coverage", rs.scored.identity(), fasta.identity, protein_group, member, t)
    if key in rs._memo:
        return rs._memo[key]
    if member is None:
        member = next((m for m in members if fasta.get(m) is not None), members[0])
    entry = fasta.get(member)
    peptides = _group_peptides(rs, protein_group)
    spans: list[PeptideSpan] = []
    unmatched: list[str] = []
    n_passing = 0
    for row in peptides:
        q = row.get("peptide_q_value")
        q = None if q is None or q != q else float(q)
        passes = q is not None and q <= t
        n_passing += passes
        seq = str(row.get("sequence") or "")
        starts = _occurrences(entry.sequence, seq) if entry is not None else []
        if not starts:
            unmatched.append(seq)
        for start in starts:
            spans.append(
                PeptideSpan(
                    base_peptide_id=int(row["base_peptide_id"]),
                    sequence=seq,
                    peptidoform=str(row.get("peptidoform") or seq),
                    start=start,
                    end=start + len(seq),
                    passes=passes,
                    q=q,
                )
            )
    spans.sort(key=lambda s: (s.start, s.end, s.base_peptide_id))
    scope = ", experiment-wide" if rs.is_experiment else ""
    note = (
        "Computed by the viewer: each peptide's sequence (its peptidoform without "
        f"modifications) is located by exact search in the sequence of {member} from "
        f"{fasta.label}; every occurrence counts. A peptide passes when peptide_q_value of "
        f"its winning row is at most {t:g} (the engine's column{scope}). The peptides are "
        "the target peptides of this protein group, at any q."
    )
    if entry is None:
        note = f"{member} is not in {fasta.label}, so its coverage cannot be computed."
    cov = Coverage(
        protein_group=protein_group,
        member=member,
        members=members,
        entry=entry,
        spans=tuple(spans),
        unmatched=tuple(unmatched),
        n_peptides=len(peptides),
        n_passing=int(n_passing),
        threshold=t,
        note=note,
        fasta_label=fasta.label,
    )
    rs._memo[key] = cov
    return cov
