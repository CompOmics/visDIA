"""Sequence coverage views, in the manner of PeptideShaker's coverage panel.

The bar shows a protein's sequence from residue 1 to its length: residues covered by
peptides that pass the threshold, by peptides that do not, and by none. The selected
peptide is outlined. The numbers come from :func:`data.fasta.protein_coverage`, which
the viewer computes from a FASTA; every view says so.

A protein can have tens of thousands of residues and hundreds of peptides, and each Dash
component costs the browser time when it mounts. So the server sends one element per
view with its data in ``data-*`` attributes (the residue states as a string of 0, 1 and
2, the peptide spans, the sequence), and ``assets/coverage.js`` draws the bar, the lanes
and the sequence text as plain HTML. The script also gives the hover read-out, outlines
the selected peptide, and turns a click into a ``mv-coverage-pick`` event (a peptide) or
a ``mv-coverage-member`` event (another member of the group) that the page acts on.
"""

from __future__ import annotations

import json
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import html

from mumdia_viewer.data.fasta import Coverage

from .icons import icon


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def summary_text(cov: Coverage) -> str:
    """The short line: coverage by passing peptides and by all of the group's peptides."""
    return f"{_pct(cov.fraction_passing)} passing · {_pct(cov.fraction_any)} all"


def counts_text(cov: Coverage) -> str:
    noun = "peptide" if cov.n_passing == 1 else "peptides"
    return (
        f"{_pct(cov.fraction_passing)} of the {cov.length:,} residues are covered by the "
        f"{cov.n_passing:,} passing {noun}, {_pct(cov.fraction_any)} by all {cov.n_peptides:,} "
        "peptides of the group."
    )


def help_text(cov: Coverage) -> str:
    parts = [counts_text(cov), cov.note]
    if cov.unmatched:
        shown = ", ".join(cov.unmatched[:5]) + (" ..." if len(cov.unmatched) > 5 else "")
        parts.append(
            f"{len(cov.unmatched):,} of {cov.n_peptides:,} peptides are not in this sequence "
            f"({shown}); they may belong to another member of the group."
        )
    if cov.entry is not None and cov.entry.description:
        parts.append(f"{cov.entry.key}: {cov.entry.description}")
    parts.append(
        "Green: residues covered by passing peptides; grey: covered only by peptides that do "
        "not pass; outline: the selected peptide. Hover the bar to read a residue; click it "
        "to select the peptide there."
    )
    return " ".join(parts)


def data_attrs(cov: Coverage, selected: int | None) -> dict[str, str]:
    """The data the script draws from (the same for the bar, the lanes and the text)."""
    states = (cov.states().astype(np.uint8) + 48).tobytes().decode("ascii")
    spans = [
        [
            s.start,
            s.end,
            s.base_peptide_id,
            1 if s.passes else 0,
            s.sequence,
            None if s.q is None else float(f"{s.q:.4g}"),
        ]
        for s in cov.spans
    ]
    return {
        "data-length": str(cov.length),
        "data-states": states,
        "data-spans": json.dumps(spans, separators=(",", ":")),
        "data-sequence": cov.entry.sequence if cov.entry is not None else "",
        "data-selected": "" if selected is None else str(selected),
        "data-group": cov.protein_group,
        "data-member": cov.member,
        "data-threshold": f"{cov.threshold:g}",
    }


def bar(cov: Coverage, *, selected: int | None = None) -> html.Div:
    return html.Div(className="mvc-bar", **data_attrs(cov, selected))


def lanes(cov: Coverage, *, selected: int | None = None) -> html.Div:
    return html.Div(className="mvc-lanes", **data_attrs(cov, selected))


def sequence_view(cov: Coverage, *, selected: int | None = None) -> html.Div:
    return html.Div(className="mvc-seq", **data_attrs(cov, selected))


def members_switch(cov: Coverage) -> Any:
    """The members of a group as small toggles; the shown member is pressed."""
    if len(cov.members) < 2:
        return html.Span(cov.member, className="mvc-member-one", title=cov.member)
    return html.Span(
        [
            html.Button(
                m,
                className="mvc-member" + (" is-on" if m == cov.member else ""),
                title=f"Show the coverage of {m}",
                **{"data-member": m, "data-group": cov.protein_group},
            )
            for m in cov.members
        ],
        className="mvc-members",
    )


def legend() -> html.Span:
    return html.Span(
        [
            html.Span(className="mvc-key mvc-s2"),
            html.Span("passing", className="mvc-key-text"),
            html.Span(className="mvc-key mvc-s1"),
            html.Span("not passing", className="mvc-key-text"),
            html.Span(className="mvc-key mvc-key-sel"),
            html.Span("selected", className="mvc-key-text"),
        ],
        className="mvc-legend",
    )


def head(cov: Coverage, *, title: str = "Coverage") -> html.Div:
    return html.Div(
        [
            html.Span(title, className="mvc-title"),
            members_switch(cov),
            html.Span(summary_text(cov), className="mvc-summary", title=counts_text(cov)),
            legend(),
            dmc.Tooltip(
                html.Span(icon("info", 13), className="mv-help"),
                label=help_text(cov),
                w=460,
                multiline=True,
                position="bottom-end",
            ),
        ],
        className="mvc-head",
    )


def strip(cov: Coverage, *, selected: int | None = None) -> html.Div:
    """The compact coverage view of the identification page: one line and the bar."""
    if cov.entry is None:
        return message(cov.note, warn=True)
    return html.Div([head(cov), bar(cov, selected=selected)], className="mvc-strip")


def card_body(cov: Coverage, *, selected: int | None = None) -> html.Div:
    """The full view of the precursor page: the bar, the peptide lanes and the sequence."""
    if cov.entry is None:
        return message(cov.note, warn=True)
    return html.Div(
        [
            head(cov, title="Coverage of"),
            bar(cov, selected=selected),
            lanes(cov, selected=selected),
            sequence_view(cov, selected=selected),
        ],
        className="mvc-full",
    )


def message(text: str, *, warn: bool = False) -> html.Div:
    return html.Div(
        [icon("alert" if warn else "info", 14), html.Span(text)],
        className="mvc-strip mvc-message" + (" mvc-warn" if warn else ""),
    )


def no_fasta() -> html.Div:
    return html.Div(
        [
            icon("info", 14),
            html.Span(
                [
                    "Sequence coverage needs the protein sequences: start the viewer with ",
                    html.Code("--fasta <file>"),
                    " (the FASTA the library was built from).",
                ]
            ),
        ],
        className="mvc-strip mvc-message",
    )
