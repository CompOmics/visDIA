"""The fragment ions of one candidate on its sequence, as PeptideShaker draws them.

* :func:`sequence_diagram`: the residues in a row; between residue i and i + 1 a mark
  above the letters for ``b_i`` and one below for ``y_(n-i)``, one per fragment charge.
  A filled mark is a fragment matched in the shown scan, an outlined mark a library
  fragment that is not matched there; no mark means the library has no such fragment.
* :func:`ladder_table`: the ion table in ladder form, one row per residue position with
  the b ions (by charge), the residue and the y ions (by charge). A cell holds the
  library m/z of a fragment the library has; a matched cell is highlighted with its raw
  ppm error, and a dot gives the fragment's XIC colour.

Only the library's fragments are shown (the candidate's chromatogram rows, the only
fragments MuMDIA extracts); no other ion is computed. Every fragment element carries
``data-frag`` (its index, see :mod:`.detail_view`), so ``assets/detail.js`` links it to
the XIC, the spectrum and the fragment list, and ``data-tip`` (its library m/z and its
match), which the page's one tooltip shows (``assets/detail.js``): a Mantine tooltip on
each of these elements would make every scan step slower to draw.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import dash_mantine_components as dmc
from dash import html

from .detail_view import IonLadder, LadderCell
from .icons import icon
from .theme import mod_name, mod_style

NO_SCAN = "no scan is shown"


def cell_text(c: LadderCell, *, scan: bool, ref: float | None = None) -> str:
    """The tooltip of one fragment: library m/z and its match in the shown scan."""
    f = c.fragment
    head = f"{f.text}: library m/z {f.mz:.4f}, predicted intensity {f.predicted:.3f}"
    if not scan:
        return f"{head}. {NO_SCAN.capitalize()}."
    if c.match is None:
        return f"{head}. Not matched in the shown scan."
    mt = c.match
    rel = f" ({100.0 * mt.obs_intensity / ref:.0f}% of the highest matched peak)" if ref else ""
    return (
        f"{head}. Matched in the shown scan: observed m/z {mt.obs_mz:.4f}, "
        f"{mt.ppm_raw:+.2f} ppm raw, intensity {mt.obs_intensity:,.0f}{rel}."
    )


def _ref(ladder: IonLadder) -> float | None:
    top = max((c.match.obs_intensity for c in ladder.cells.values() if c.match), default=0.0)
    return top if top > 0 else None


def _mods(r_mods: tuple[str, ...], extra: tuple[str, ...] = ()) -> tuple[str, str, str] | None:
    """(short tags, colour, full names) of a residue's modifications, or None."""
    mods = (*extra, *r_mods)
    if not mods:
        return None
    styles = [mod_style(m) for m in mods]
    return "+".join(s for s, _ in styles), styles[0][1], ", ".join(mod_name(m) for m in mods)


# --------------------------------------------------------------------------- diagram


def _mark(c: LadderCell, ion: str, scan: bool, ref: float | None, hidden: Collection[int]) -> Any:
    f = c.fragment
    state = "on" if c.matched else "lib"
    cls = f"pd-mark pd-mark-{ion} pd-mark-{state}" + (" pd-off" if f.index in hidden else "")
    return html.Div(
        className=cls,
        style={"--pd-c": f.colour},
        **{
            "data-frag": str(f.index),
            "aria-label": f.text,
            "data-tip": cell_text(c, scan=scan, ref=ref),
        },
    )


def _num(ordinal: int, ion: str, on: bool) -> Any:
    cls = f"pd-seq-num pd-seq-num-{ion}" + (" pd-seq-num-on" if on else "")
    return html.Span(str(ordinal), className=cls)


def sequence_diagram(
    ladder: IonLadder,
    *,
    hidden: Collection[int] = (),
    compact: bool = False,
    legend: bool = True,
    match: Any = None,
) -> Any:
    """The sequence with the library's b ions above and y ions below (see the module).

    ``match`` is shown in the legend: the badge that says the matches are the viewer's.
    """
    n = ladder.n
    if n == 0:
        return html.Div("No sequence to draw.", className="pd-seq-empty")
    ref = _ref(ladder)
    bz = ladder.b_charges or (1,)
    yz = ladder.y_charges or (1,)
    nb, ny = len(bz), len(yz)
    letters_row = 2 + nb
    num_row = "0px" if compact else "auto"
    items: list[Any] = []

    def place(el: Any, row: int, col: int) -> Any:
        return html.Div(el, className="pd-seq-slot", style={"gridRow": row, "gridColumn": col})

    for i, r in enumerate(ladder.residues, start=1):
        extra = ladder.nterm if i == 1 else ()
        extra = (*extra, *ladder.cterm) if i == n else extra
        mods = _mods(r.mods, extra)
        kids: list[Any] = [r.aa]
        style: dict[str, Any] = {}
        title = f"{r.aa}{i}"
        if mods is not None:
            short, colour, names = mods
            kids.append(html.Span(short, className="pd-seq-tag"))
            style["color"] = colour
            title += f": {names}"
        items.append(
            html.Div(
                kids,
                className="pd-seq-aa" + (" pd-seq-mod" if mods else ""),
                style={**style, "gridRow": letters_row, "gridColumn": 2 * i - 1},
                **{"data-tip": title},
            )
        )
    for g in range(1, n):
        col = 2 * g
        b_cells = [(z, ladder.cell("b", g, z)) for z in bz]
        y_cells = [(z, ladder.cell("y", n - g, z)) for z in yz]
        b_lib = [c for _, c in b_cells if c is not None]
        y_lib = [c for _, c in y_cells if c is not None]
        if b_lib and not compact:
            items.append(place(_num(g, "b", any(c.matched for c in b_lib)), 1, col))
        for k, (_, c) in enumerate(b_cells):
            if c is not None:
                # The highest charge on top: charge 1 is next to the letters.
                items.append(place(_mark(c, "b", ladder.scan, ref, hidden), 1 + nb - k, col))
        for k, (_, c) in enumerate(y_cells):
            if c is not None:
                items.append(
                    place(_mark(c, "y", ladder.scan, ref, hidden), letters_row + 1 + k, col)
                )
        if y_lib and not compact:
            items.append(
                place(_num(n - g, "y", any(c.matched for c in y_lib)), letters_row + ny + 1, col)
            )
    # Residue and gap columns shrink from their size to a minimum when the card is narrow.
    res_col = "minmax(var(--pd-seq-wmin), var(--pd-seq-w))"
    gap_col = "minmax(var(--pd-seq-gapmin), var(--pd-seq-gap))"
    grid = html.Div(
        items,
        className="pd-seq-grid",
        style={
            "maxWidth": f"calc({n} * var(--pd-seq-w) + {n - 1} * var(--pd-seq-gap))",
            "gridTemplateColumns": " ".join(
                [res_col if k % 2 == 0 else gap_col for k in range(2 * n - 1)]
            ),
            "gridTemplateRows": " ".join(
                [num_row]
                + ["var(--pd-mark-h)"] * nb
                + ["auto"]
                + ["var(--pd-mark-h)"] * ny
                + [num_row]
            ),
        },
    )
    parts: list[Any] = [html.Div(grid, className="pd-seq-scroll")]
    if legend and not compact:
        parts.append(diagram_legend(ladder, match=match))
    return html.Div(parts, className="pd-seq" + (" pd-seq-compact" if compact else ""))


def diagram_legend(ladder: IonLadder, *, match: Any = None) -> Any:
    counts = (
        f"{ladder.n_matched} of {ladder.n_library} library fragments matched"
        if ladder.scan
        else f"{ladder.n_library} library fragments; {NO_SCAN}"
    )
    more = [
        "Each mark is a fragment of the library (the only fragments MuMDIA extracts; no other "
        "ion is computed): b ions above the letters, y ions below, at the cleavage that "
        "makes them. The numbers are the ion ordinals."
    ]
    if len(ladder.b_charges) > 1 or len(ladder.y_charges) > 1:
        more.append("One mark per fragment charge; charge 1 is next to the letters.")
    if ladder.unplaced:
        more.append(
            f"{len(ladder.unplaced)} fragment(s) are not b or y ions of this sequence; the "
            "fragment list shows them."
        )
    return html.Div(
        [
            html.Span(
                [html.Span(className="pd-mark pd-mark-on pd-mark-key"), "matched in this scan"],
                className="pd-seq-key",
            ),
            html.Span(
                [html.Span(className="pd-mark pd-mark-lib pd-mark-key"), "not matched"],
                className="pd-seq-key",
            ),
            html.Span(f"b above, y below · {counts}", className="pd-seq-note"),
            match if ladder.scan else None,
            dmc.Tooltip(
                html.Span(icon("info", 12), className="mv-help"), label=" ".join(more), w=340
            ),
        ],
        className="pd-seq-legend",
    )


# --------------------------------------------------------------------------- ladder


def _ion_head(ion: str, z: int) -> Any:
    return html.Th(
        [ion, html.Sup(f"{z}+") if z > 1 else None],
        className=f"pd-lad-ion pd-lad-{ion}",
        title=f"{ion} ions, fragment charge {z}: library m/z",
    )


def _cell(
    c: LadderCell | None,
    *,
    scan: bool,
    ref: float | None,
    hidden: Collection[int],
    compact: bool,
    digits: int,
) -> Any:
    if c is None:
        return html.Td("", className="pd-lc pd-lc-none")
    f = c.fragment
    state = "on" if c.matched else "lib"
    # The compact table has no dot (its cells must stay narrow): a matched cell's left
    # edge carries the fragment's colour instead (detail.css).
    kids: list[Any] = (
        [] if compact else [html.Span(className="pd-lc-dot", style={"background": f.colour})]
    )
    kids.append(html.Span(f"{f.mz:.{digits}f}", className="pd-lc-mz"))
    if c.match is not None and not compact:
        kids.append(html.Span(f"{c.match.ppm_raw:+.1f} ppm", className="pd-lc-ppm"))
    return html.Td(
        html.Div(kids, className="pd-lc-in"),
        className=f"pd-lc pd-lc-{state}" + (" pd-off" if f.index in hidden else ""),
        style={"--pd-c": f.colour},
        **{
            "data-frag": str(f.index),
            "data-tip": cell_text(c, scan=scan, ref=ref),
            "data-tip-side": "left",
        },
    )


def ladder_table(ladder: IonLadder, *, hidden: Collection[int] = (), compact: bool = False) -> Any:
    """The ion table in ladder form (see the module)."""
    n = ladder.n
    if n == 0:
        return html.Div("No sequence.", className="pd-seq-empty")
    ref = _ref(ladder)
    bz = ladder.b_charges or (1,)
    yz = ladder.y_charges or (1,)
    # Four decimals when the table is narrow enough; the tooltip always has them.
    digits = 2 if compact or len(bz) + len(yz) > 2 else 4
    head = html.Thead(
        html.Tr(
            [
                html.Th("#", className="pd-lad-num", title="b ion ordinal (residues 1 to #)"),
                *[_ion_head("b", z) for z in bz],
                html.Th("", className="pd-lad-res"),
                *[_ion_head("y", z) for z in yz],
                html.Th("#", className="pd-lad-num", title="y ion ordinal (residues # to the end)"),
            ]
        )
    )
    rows = []
    for i, r in enumerate(ladder.residues, start=1):
        extra = ladder.nterm if i == 1 else ()
        extra = (*extra, *ladder.cterm) if i == n else extra
        mods = _mods(r.mods, extra)
        res: list[Any] = [r.aa]
        style: dict[str, Any] = {}
        title = f"residue {i}: {r.aa}"
        if mods is not None:
            short, colour, names = mods
            res.append(html.Sup(short, className="mv-pep-tag"))
            style["color"] = colour
            title += f" ({names})"
        kw: dict[str, Any] = {
            "scan": ladder.scan,
            "ref": ref,
            "hidden": hidden,
            "compact": compact,
            "digits": digits,
        }
        rows.append(
            html.Tr(
                [
                    html.Td(str(i), className="pd-lad-num"),
                    *[_cell(ladder.cell("b", i, z), **kw) for z in bz],
                    html.Td(res, className="pd-lad-res", style=style, **{"data-tip": title}),
                    *[_cell(ladder.cell("y", n - i + 1, z), **kw) for z in yz],
                    html.Td(str(n - i + 1), className="pd-lad-num"),
                ]
            )
        )
    return html.Table(
        [head, html.Tbody(rows)], className="pd-lad" + (" pd-lad-compact" if compact else "")
    )


def ladder_caption(ladder: IonLadder) -> str:
    """The ion table's one visible line: how many library fragments the shown scan matches."""
    if not ladder.scan:
        return f"{ladder.n_library} library fragments; {NO_SCAN}"
    return f"{ladder.n_matched} of {ladder.n_library} library fragments matched"


def ladder_help(ladder: IonLadder, *, compact: bool = False) -> str:
    """What the ion table shows (its help tooltip)."""
    where = "the raw ppm error is in the tooltip" if compact else "with the raw ppm error"
    what = (
        f"{ladder.n_matched} of {ladder.n_library} matched in the shown scan (highlighted; "
        f"{where}; the match is the viewer's, with the engine's rule)"
        if ladder.scan
        else f"{NO_SCAN}, so nothing is matched"
    )
    extra = f"; {len(ladder.unplaced)} not on the sequence" if ladder.unplaced else ""
    colour = "The left edge of a matched cell" if compact else "The dot"
    return (
        f"The {ladder.n_library} library fragments of this candidate on its sequence, one row "
        f"per residue: b ions on the left, y ions on the right, by fragment charge; {what}"
        f"{extra}. A cell holds the library m/z. MuMDIA extracts only these fragments; no "
        f"other ion is computed. {colour} is the fragment's XIC colour."
    )


__all__ = [
    "cell_text",
    "diagram_legend",
    "ladder_caption",
    "ladder_help",
    "ladder_table",
    "sequence_diagram",
]
