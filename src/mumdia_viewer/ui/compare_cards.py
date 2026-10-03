"""The cards of the compare page (components only; the page and its callbacks are in
:mod:`.compare`). Values come from :mod:`mumdia_viewer.data.compare`; the viewer's own
numbers (the overlap, the Jaccard index, the correlations, the pooled medians) say so
in their tooltips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash import dcc, html

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data import compare as cd

from . import compare_figures as cf
from .icons import icon
from .state import href, stop_label
from .widgets import graph, peptidoform, section

UNIT_LABELS = {"precursor": "Precursors", "peptide": "Peptides", "protein_group": "Protein groups"}
COUNT_LABELS = {
    "psm": ("PSMs", "rows, q_value"),
    "precursor": ("Precursors", "unique (peptidoform, charge), precursor_q"),
    "peptide": ("Peptides", "unique base_peptide_id, peptide_q_value"),
    "protein_group": ("Protein groups", "unique protein_group, pg_q_value"),
}
GRID_LIMIT = 1_000
SCORE_GRAPH = "cmp-score"
QUANT_GRAPH = "cmp-quant"
Q_BAR_FULL = 1e-4


# --------------------------------------------------------------------------- helpers


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def side_badge(side: str, *, size: str = "md") -> Any:
    """The A or B mark, in the side's colour."""
    return html.Span(side.upper(), className=f"cmp-side cmp-side-{side.lower()} cmp-side-{size}")


def derived(text: str) -> Any:
    return tip(html.Span("viewer", className="cmp-derived"), f"Viewer-derived: {text}")


def t_text(t: float) -> str:
    return f"q ≤ {stop_label(t)}"


def where(rs: ResultSet) -> str:
    return f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"


def signed(n: int) -> str:
    return f"+{n:,}" if n > 0 else f"{n:,}" if n < 0 else "0"


def percent(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


# --------------------------------------------------------------------------- header


def verdict_chips(a: ResultSet, b: ResultSet, rows: Sequence[cd.ProvenanceRow], n_diff: int) -> Any:
    """One-line verdicts: engine, configuration, library and runs."""
    by = {r.label: r for r in rows}

    def chip(text: str, ok: bool | None, label: str) -> Any:
        colour = "green" if ok else "yellow" if ok is False else "gray"
        mark = icon("check", 12) if ok else icon("alert", 12) if ok is False else None
        return tip(
            dmc.Badge(
                text,
                color=colour,
                variant="light",
                leftSection=mark,
                size="md",
                style={"textTransform": "none"},
            ),
            label,
        )

    version, sha = by["MuMDIA version"], by["Git SHA"]
    same_engine = bool(version.same and sha.same)
    engine = (
        f"same engine {version.a} · {sha.a[:7]}"
        if same_engine
        else f"engine {version.a or '?'} · {sha.a[:7]} vs {version.b or '?'} · {sha.b[:7]}"
    )
    chips = [
        chip(engine, same_engine, "manifest mumdia_version and git_sha of A and B"),
        chip(
            "configuration identical"
            if n_diff == 0
            else f"{n_diff} configuration difference{'s' if n_diff != 1 else ''}",
            n_diff == 0,
            "Keys of the resolved configurations (config_json) whose values differ; the "
            "list is in the Configuration card.",
        ),
    ]
    lib = by.get("Library precursors")
    if lib is not None:
        chips.append(
            chip(
                "same library" if lib.same else "different library",
                lib.same,
                "manifest inputs.lib_precursors, by content hash. With different libraries "
                "the candidate ids differ; the viewer compares by strings.",
            )
        )
    elif "FASTA" in by:
        fasta = by["FASTA"]
        chips.append(chip("same FASTA" if fasta.same else "different FASTA", fasta.same, fasta.tip))
    mz = by["mzML inputs"]
    pairs = cd.run_pairs(a, b)
    chips.append(
        chip(
            f"{len(pairs)} of {len(a.runs)} run{'s' if len(a.runs) != 1 else ''} of A "
            "share an mzML with B"
            if pairs
            else "no shared mzML",
            True if pairs and mz.same else None,
            mz.tip,
        )
    )
    return dmc.Group(chips, gap=6)


def hero(a: ResultSet, b: ResultSet, *, base: str, b_base: str | None, chips: Any) -> Any:
    def name_line(side: str, rs: ResultSet) -> Any:
        return dmc.Group(
            [
                side_badge(side, size="lg"),
                html.Div(rs.root.name, className="mv-title cmp-name"),
                dmc.Badge(where(rs), size="sm", color="gray", variant="light"),
            ],
            gap=10,
            wrap="nowrap",
            align="center",
        )

    title = html.Div(
        [
            dmc.Text("Compare · two result sets", className="mv-eyebrow"),
            dmc.Group(
                [name_line("a", a), html.Span("vs", className="cmp-vs"), name_line("b", b)],
                gap="md",
                align="center",
                className="cmp-names",
            ),
            html.Div(
                [
                    html.Div([html.Span("A ", className="cmp-path-side"), str(a.root)]),
                    html.Div([html.Span("B ", className="cmp-path-side"), str(b.root)]),
                ],
                className="mv-path cmp-paths",
            ),
        ]
    )
    buttons: list[Any] = []
    if b_base:
        buttons = [
            tip(
                dmc.Anchor(
                    dmc.Button(
                        "Open B",
                        size="xs",
                        variant="light",
                        leftSection=icon("external", 13),
                    ),
                    href=b_base,
                    id="cmp-open-b",
                ),
                "The full viewer of B, served beside this one",
            ),
            tip(
                dmc.Anchor(
                    dmc.Button(
                        "Compare from B",
                        size="xs",
                        variant="default",
                        leftSection=icon("compare", 13),
                    ),
                    href=href(b_base, "compare"),
                    id="cmp-swap",
                ),
                "The same comparison with the sides swapped (B's viewer)",
            ),
        ]
    return dmc.Group(
        [
            dmc.Stack([title, chips], gap="sm"),
            dmc.Group(buttons, gap="xs") if buttons else html.Div(),
        ],
        justify="space-between",
        align="flex-end",
        className="cmp-hero",
    )


def _mark(same: bool | None) -> Any:
    if same is None:
        return html.Span("", className="cmp-mark")
    if same:
        return tip(html.Span(icon("check", 12), className="cmp-mark cmp-mark-same"), "the same")
    return tip(html.Span("≠", className="cmp-mark cmp-mark-diff"), "different")


def _value(text: str, *, mono: bool = False) -> Any:
    if not text:
        return html.Span("not recorded", className="cmp-dim")
    return html.Span(text, className="cmp-mono" if mono else None)


MONO_ROWS = {"Git SHA", "Configuration hash", "Library precursors", "Library fragments", "FASTA"}


def count_rows(ca: Mapping[str, int], cb: Mapping[str, int], t: float) -> list[Any]:
    """The rows of the identification counts, A and B side by side."""
    rows = []
    for unit, (label, column) in COUNT_LABELS.items():
        na, nb = int(ca.get(unit, 0)), int(cb.get(unit, 0))
        delta = nb - na
        rows.append(
            html.Tr(
                [
                    html.Td(
                        tip(
                            html.Span(label, className="cmp-help-text"),
                            f"{label} ({column} ≤ {stop_label(t)}); targets only, each side on "
                            "its own column, as on its overview",
                        )
                    ),
                    html.Td(f"{na:,}", className="mv-num"),
                    html.Td(f"{nb:,}", className="mv-num"),
                    html.Td(
                        tip(
                            html.Span(
                                signed(delta),
                                className="cmp-delta "
                                + (
                                    "cmp-delta-up"
                                    if delta > 0
                                    else "cmp-delta-down"
                                    if delta < 0
                                    else ""
                                ),
                            ),
                            f"B minus A: {signed(delta)} "
                            f"({percent(delta, na) if na else 'n/a'} of A)",
                        ),
                        className="mv-num",
                    ),
                ]
            )
        )
    return rows


def provenance_card(
    a: ResultSet,
    b: ResultSet,
    rows: Sequence[cd.ProvenanceRow],
    counts: tuple[Mapping[str, int], Mapping[str, int]],
    t: float,
) -> Any:
    def head(label: str) -> Any:
        return html.Tr(html.Td(label, colSpan=4, className="cmp-group"))

    body: list[Any] = [head("Engine")]
    for r in rows:
        if r.label in ("Library precursors", "Library fragments", "FASTA", "mzML inputs"):
            continue
        body.append(_prov_row(r))
    body.append(head("Inputs"))
    for r in rows:
        if r.label in ("Library precursors", "Library fragments", "FASTA", "mzML inputs"):
            body.append(_prov_row(r))
    body.append(
        html.Tr(
            html.Td(
                html.Span(f"Identifications at {t_text(t)}", id="cmp-counts-title"),
                colSpan=4,
                className="cmp-group",
            )
        )
    )
    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(""),
                        html.Th(
                            dmc.Group(
                                [side_badge("a", size="sm"), a.root.name], gap=6, wrap="nowrap"
                            )
                        ),
                        html.Th(
                            dmc.Group(
                                [side_badge("b", size="sm"), b.root.name], gap=6, wrap="nowrap"
                            )
                        ),
                        html.Th("", className="cmp-mark-col"),
                    ]
                )
            ),
            html.Tbody(body),
            html.Tbody(count_rows(*counts, t), id="cmp-counts"),
        ],
        className="cmp-prov",
    )
    return section(
        "Result sets",
        html.Div(table, className="cmp-prov-box"),
        help="From each side's manifest: versions, the engine's commit, the models, the "
        "configuration hash and the inputs (file name and content hash). The counts are each "
        "side's targets at the header threshold on its own q column, as on its overview; in "
        "an experiment the grouped columns are experiment-wide.",
        id="cmp-prov-card",
    )


def _prov_row(r: cd.ProvenanceRow) -> Any:
    mono = r.label in MONO_ROWS
    return html.Tr(
        [
            html.Td(tip(html.Span(r.label, className="cmp-help-text"), r.tip)),
            html.Td(_value(r.a, mono=mono)),
            html.Td(_value(r.b, mono=mono)),
            html.Td(_mark(r.same), className="cmp-mark-col"),
        ],
        className="cmp-row-diff" if r.same is False else None,
    )


def config_sections(a: ResultSet, b: ResultSet, diffs: Sequence[cd.ConfigDiff]) -> Any:
    """The top-level sections of the configurations, each with its number of keys and of
    differing keys (green when none differs)."""
    fa = cd.flatten_config(a.manifest.config)
    fb = cd.flatten_config(b.manifest.config)
    keys = set(fa) | set(fb)
    sections: dict[str, int] = {}
    for k in keys:
        sections[k.split(".", 1)[0]] = sections.get(k.split(".", 1)[0], 0) + 1
    differing: dict[str, int] = {}
    for d in diffs:
        top = d.key.split(".", 1)[0]
        differing[top] = differing.get(top, 0) + 1
    chips = []
    for name in sorted(sections):
        n_diff = differing.get(name, 0)
        chips.append(
            tip(
                dmc.Badge(
                    f"{name} {n_diff}/{sections[name]}" if n_diff else name,
                    size="sm",
                    variant="light",
                    color="yellow" if n_diff else "green",
                    style={"textTransform": "none"},
                ),
                f"{name}: {sections[name]} keys"
                + (f", {n_diff} differ" if n_diff else ", all equal"),
                multiline=False,
                w="auto",
            )
        )
    return html.Div(
        [
            html.Div(
                f"{len(keys):,} keys in {len(sections)} sections compared",
                className="cmp-note",
            ),
            dmc.Group(chips, gap=4, mt=6),
        ],
        className="cmp-sections",
    )


def config_card(a: ResultSet, b: ResultSet, diffs: Sequence[cd.ConfigDiff]) -> Any:
    ha, hb = a.manifest.config_hash, b.manifest.config_hash
    if not diffs:
        body: Any = dmc.Stack(
            [
                dmc.ThemeIcon(
                    icon("check", 18), color="green", variant="light", size=34, radius="xl"
                ),
                dmc.Text("The resolved configurations are identical.", size="sm", fw=600),
                dmc.Text(
                    "Every key of config_json has the same value"
                    + (" (and config_hash is equal)." if ha and ha == hb else "."),
                    size="xs",
                    c="dimmed",
                    ta="center",
                ),
            ],
            align="center",
            gap=6,
            py="xl",
        )
    else:
        rows = []
        for d in diffs:
            note = (
                tip(
                    dmc.Badge(
                        "same path",
                        size="xs",
                        color="gray",
                        variant="light",
                        style={"textTransform": "none"},
                    ),
                    d.note,
                )
                if d.note
                else None
            )
            kind = (
                dmc.Badge(
                    d.kind,
                    size="xs",
                    color="gray",
                    variant="outline",
                    style={"textTransform": "none"},
                )
                if d.kind != "differs"
                else None
            )
            rows.append(
                html.Tr(
                    [
                        html.Td(
                            [
                                html.Div(d.key, className="cmp-key"),
                                dmc.Group([x for x in (kind, note) if x], gap=4),
                            ]
                            if (kind or note)
                            else html.Div(d.key, className="cmp-key")
                        ),
                        html.Td(
                            html.Div(
                                d.a or "absent", className="cmp-cfg" + ("" if d.a else " cmp-dim")
                            )
                        ),
                        html.Td(
                            html.Div(
                                d.b or "absent", className="cmp-cfg" + ("" if d.b else " cmp-dim")
                            )
                        ),
                    ]
                )
            )
        body = html.Div(
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("key"),
                                html.Th(side_badge("a", size="sm")),
                                html.Th(side_badge("b", size="sm")),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="cmp-cfg-table",
            ),
            className="cmp-cfg-box",
        )
    return section(
        "Configuration",
        body,
        config_sections(a, b, diffs),
        count=len(diffs) if diffs else None,
        subtitle="Keys of config_json that differ" if diffs else None,
        help="The resolved configurations of the manifests (config_json), flattened to dotted "
        "keys and compared value by value. A path written with other separators is marked "
        "'same path'.",
        id="cmp-config-card",
    )


# --------------------------------------------------------------------------- overlap


def overlap_rows(overlaps: Sequence[cd.Overlap], unit: str) -> list[Any]:
    """One row per unit: A's count, the bar (only A | both | only B), B's count."""
    out = []
    for o in overlaps:
        total = o.n_a + o.n_b - o.n_both
        parts = [
            ("a", o.n_only_a, f"only in A: {o.n_only_a:,} {cd.SPEC[o.unit].plural}"),
            (
                "both",
                o.n_both,
                f"in both: {o.n_both:,} ({percent(o.n_both, o.n_a)} of A, "
                f"{percent(o.n_both, o.n_b)} of B)",
            ),
            ("b", o.n_only_b, f"only in B: {o.n_only_b:,} {cd.SPEC[o.unit].plural}"),
        ]
        segments = []
        for kind, n, label in parts:
            if not n or not total:
                continue
            # A native title, not a Mantine tooltip: the tooltip's wrapper would take the
            # segment out of the flex row.
            segments.append(
                html.Div(
                    className=f"cmp-seg cmp-seg-{kind}",
                    style={"flexGrow": n, "flexBasis": 0},
                    title=label,
                )
            )
        bar = html.Div(
            [
                html.Div(
                    segments or [html.Div(className="cmp-seg cmp-seg-none", style={"flexGrow": 1})],
                    className="cmp-bar",
                ),
                html.Div(
                    [
                        html.Span(
                            [
                                html.B(f"{o.n_only_a:,}"),
                                f" only in A ({percent(o.n_only_a, o.n_a)})",
                            ],
                            className="cmp-lab-a",
                        ),
                        html.Span(
                            [html.B(f"{o.n_both:,}"), " in both"],
                            className="cmp-lab-both",
                        ),
                        html.Span(
                            [
                                html.B(f"{o.n_only_b:,}"),
                                f" only in B ({percent(o.n_only_b, o.n_b)})",
                            ],
                            className="cmp-lab-b",
                        ),
                    ],
                    className="cmp-bar-labels",
                ),
            ],
            className="cmp-bar-cell",
        )
        jac = o.jaccard
        out.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(UNIT_LABELS[o.unit], className="cmp-ov-unit"),
                            tip(
                                html.Div(cd.SPEC[o.unit].q_column, className="cmp-ov-q"),
                                f"{o.label}. {o.note}",
                            ),
                        ],
                        className="cmp-ov-name",
                    ),
                    html.Div(f"{o.n_a:,}", className="cmp-ov-n cmp-ov-na"),
                    bar,
                    html.Div(f"{o.n_b:,}", className="cmp-ov-n cmp-ov-nb"),
                    tip(
                        html.Div(f"{jac:.3f}" if jac is not None else "n/a", className="cmp-ov-j"),
                        "Viewer-derived: the Jaccard index, shared / (A + B - shared)",
                    ),
                ],
                id={"type": "cmp-ov-row", "unit": o.unit},
                n_clicks=0,
                className="cmp-ov-row" + (" cmp-ov-row-active" if o.unit == unit else ""),
                title=f"Show the shared and unique {cd.SPEC[o.unit].plural} below",
            )
        )
    return out


def overlap_body(overlaps: Sequence[cd.Overlap], unit: str, t: float) -> Any:
    head = html.Div(
        [
            html.Div("unit", className="cmp-ov-name cmp-ov-head"),
            html.Div(side_badge("a", size="sm"), className="cmp-ov-n cmp-ov-head"),
            html.Div(
                [
                    html.Span([html.I(className="cmp-key-sw cmp-seg-a"), "only A"]),
                    html.Span([html.I(className="cmp-key-sw cmp-seg-both"), "both"]),
                    html.Span([html.I(className="cmp-key-sw cmp-seg-b"), "only B"]),
                ],
                className="cmp-ov-legend",
            ),
            html.Div(side_badge("b", size="sm"), className="cmp-ov-n cmp-ov-head"),
            html.Div(
                ["Jaccard ", derived("shared / (A + B - shared)")], className="cmp-ov-j cmp-ov-head"
            ),
        ],
        className="cmp-ov-row cmp-ov-header",
    )
    return html.Div([head, *overlap_rows(overlaps, unit)], className="cmp-ov")


def overlap_card(overlaps: Sequence[cd.Overlap], unit: str, t: float) -> Any:
    return section(
        "Identification overlap",
        html.Div(overlap_body(overlaps, unit, t), id="cmp-overlap"),
        subtitle=html.Span(
            [
                "Each side on its own q column at ",
                html.Span(t_text(t), id="cmp-overlap-t"),
                "; matched by strings, never by ids. Click a unit to list it below.",
            ]
        ),
        help="Precursors are matched by (peptidoform, charge), peptides by the stripped "
        "sequence of the row that carries peptide_q_value (the libraries can number peptides "
        "differently), protein groups by the protein_group string (the sides can group "
        "proteins differently). Targets only. The overlap is the viewer's set operation; no "
        "q value is recomputed.",
        id="cmp-overlap-card",
    )


def unit_bar(unit: str, o: cd.Overlap | None) -> Any:
    control = dmc.SegmentedControl(
        id="cmp-unit",
        data=[{"value": u, "label": UNIT_LABELS[u]} for u in cd.COMPARE_UNITS],
        value=unit,
        size="sm",
        radius="xl",
    )
    return dmc.Paper(
        dmc.Group(
            [
                dmc.Group(
                    [html.Div("Shared and unique", className="mv-section-title"), control],
                    gap="md",
                ),
                html.Div(unit_summary(o), id="cmp-unit-summary", className="cmp-unit-summary"),
            ],
            justify="space-between",
            wrap="wrap",
        ),
        withBorder=True,
        p="sm",
        px="md",
        className="cmp-unitbar",
    )


def unit_summary(o: cd.Overlap | None) -> Any:
    if o is None:
        return ""
    return dmc.Group(
        [
            html.Span([html.I(className="cmp-key-sw cmp-seg-both"), f"{o.n_both:,} shared"]),
            html.Span([html.I(className="cmp-key-sw cmp-seg-a"), f"{o.n_only_a:,} only in A"]),
            html.Span([html.I(className="cmp-key-sw cmp-seg-b"), f"{o.n_only_b:,} only in B"]),
        ],
        gap="md",
    )


# --------------------------------------------------------------------------- scatters


def score_card(
    fig: Any,
    *,
    unit: str,
    n_shared: int,
    n_drawn: int,
    rho: float | None,
    a: ResultSet,
    b: ResultSet,
) -> Any:
    ra = a.manifest.model_identities.get("rescorer") or "rescorer not recorded"
    rb = b.manifest.model_identities.get("rescorer") or "rescorer not recorded"
    same = ra == rb
    scale = (
        "Both sides used the same rescorer, but each trained its own model, so the scales "
        "can still differ: read the rank order."
        if same
        else f"Different rescorers ({ra} and {rb}): the scales are not comparable; read the "
        "rank order."
    )
    noun = cd.SPEC[unit].plural
    stats = dmc.Group(
        [
            tip(
                dmc.Badge(
                    f"Spearman \u03c1 {rho:.3f}" if rho is not None else "Spearman \u03c1 n/a",
                    size="sm",
                    variant="light",
                    color="indigo",
                    style={"textTransform": "none"},
                ),
                "Viewer-derived: the rank correlation of A's and B's scores over every shared "
                f"{cd.SPEC[unit].singular} (not only the drawn ones)",
            ),
            html.Span(cf.drawn_note(n_drawn, n_shared, noun), className="cmp-note"),
        ],
        gap="sm",
    )
    return section(
        f"Scores of shared {noun}",
        graph(SCORE_GRAPH, fig),
        stats,
        html.Div(scale, className="cmp-note"),
        count=n_shared,
        subtitle="Each side's score of the row that carries its q value (the winner)",
        help="score of each side's winning row: x is A, y is B; the dashed line is y = x. "
        "Click a point to open it in A or B.",
        id="cmp-score-card",
    )


def pair_options(pairs: Sequence[cd.RunPair]) -> list[dict[str, str]]:
    opts = []
    if len(pairs) > 1:
        opts.append({"value": "all", "label": f"All {len(pairs)} run pairs"})
    for i, p in enumerate(pairs):
        opts.append({"value": str(i), "label": f"A {p.a_label} = B {p.b_label}"})
    if not opts:
        opts.append({"value": "pooled", "label": "Pooled (median over runs)"})
    return opts


def quant_card(
    fig: Any,
    df: pd.DataFrame,
    *,
    pairs: Sequence[cd.RunPair],
    pick: str,
) -> Any:
    mode = df.attrs.get("mode", "pooled")
    ratio = cf.log2_median_ratio(df)
    rho = None
    if len(df):
        ok = (df["a_quantity"] > 0) & (df["b_quantity"] > 0)
        if ok.any():
            rho = cd.spearman(df.loc[ok, "a_quantity"], df.loc[ok, "b_quantity"])
    selector = dmc.Select(
        id="cmp-pair",
        data=pair_options(pairs),
        value=pick,
        size="xs",
        radius="xl",
        w=200,
        allowDeselect=False,
        disabled=not pairs,
        comboboxProps={"shadow": "md"},
    )
    mode_text = (
        "Per run: each point is one precursor in one pair of runs that searched the same mzML."
        if mode == "per run"
        else "Pooled: the runs of A and B do not pair (no shared mzML), so each side shows its "
        "median quantity over its runs with a quantity (the viewer's median)."
    )
    stats = dmc.Group(
        [
            tip(
                dmc.Badge(
                    f"median log2 B/A {ratio:+.3f}" if ratio is not None else "median log2 B/A n/a",
                    size="sm",
                    variant="light",
                    color="indigo",
                    style={"textTransform": "none"},
                ),
                "Viewer-derived: the median of log2(B quantity / A quantity) over the points "
                "with a quantity on both sides",
            ),
            tip(
                dmc.Badge(
                    f"Spearman \u03c1 {rho:.3f}" if rho is not None else "Spearman \u03c1 n/a",
                    size="sm",
                    variant="light",
                    color="gray",
                    style={"textTransform": "none"},
                ),
                "Viewer-derived: the rank correlation of the quantities",
            ),
            html.Span(
                f"{df.attrs.get('n_both', 0):,} points with a quantity on both sides; "
                + cf.drawn_note(
                    min(cf.MAX_POINTS, df.attrs.get("n_both", 0)),
                    df.attrs.get("n_both", 0),
                    "points",
                ).lower(),
                className="cmp-note",
            ),
        ],
        gap="sm",
    )
    notes = [html.Div(mode_text, className="cmp-note")]
    return section(
        "Quantity of shared precursors",
        graph(QUANT_GRAPH, fig),
        stats,
        *notes,
        right=selector,
        subtitle="Per precursor, whatever the unit above; peptide_quant.quantity, log "
        "scale; not quantifiable is left out, never 0",
        help="Each side's peptide_quant.quantity of the precursors that pass on both sides. "
        "A null quantity (not quantifiable) is not drawn. Runs pair when they searched the "
        "same mzML (recorded content hash, else file name). Click a point to open it.",
        id="cmp-quant-card",
    )


def pick_strip(item: Mapping[str, Any] | None) -> Any:
    """The point clicked last in a scatter, with links into A and B."""
    if not item:
        return None
    label: Any = (
        peptidoform(item["text"], size="0.92em") if item.get("pep") else html.B(item["text"])
    )
    links = []
    for side in ("a", "b"):
        url = item.get(f"{side}_href")
        if not url:
            continue
        button = dmc.Button(
            f"Open in {side.upper()}",
            size="xs",
            variant="light" if side == "a" else "default",
            leftSection=icon("external", 12),
        )
        # A's pages open inside this app; B is another Dash app, so a full navigation.
        links.append(
            dcc.Link(button, href=url, id=f"cmp-pick-{side}")
            if side == "a"
            else html.A(button, href=url, id=f"cmp-pick-{side}")
        )
    return dmc.Group(
        [
            dmc.Group(
                [
                    dmc.Text("Selected", size="xs", c="dimmed", fw=600, tt="uppercase"),
                    label,
                    html.Span(item.get("detail", ""), className="cmp-note"),
                ],
                gap="sm",
            ),
            dmc.Group(links, gap="xs"),
        ],
        justify="space-between",
        className="cmp-pick",
    )


# --------------------------------------------------------------------------- unique tables


GRID_OPTIONS: dict[str, Any] = {
    "theme": {"function": "mvxTheme(themeQuartz)"},
    "rowHeight": 26,
    "headerHeight": 28,
    "animateRows": False,
    "suppressCellFocus": True,
    "tooltipShowDelay": 350,
    "enableCellTextSelection": True,
}


def unique_columns(
    unit: str, side: str, *, t: float, experiment: bool, score_range: tuple[float, float]
) -> list[dict[str, Any]]:
    other = "B" if side == "a" else "A"
    spec = cd.SPEC[unit]
    first: dict[str, Any]
    if unit == "precursor":
        first = {
            "field": "key_text",
            "headerName": "peptidoform",
            "cellRenderer": "CmpLink",
            "cellRendererParams": {"peptidoform": True},
            "minWidth": 150,
            "flex": 2,
            "headerTooltip": "peptidoform of the precursor; click it to open its precursor page in "
            + side.upper(),
        }
    elif unit == "peptide":
        first = {
            "field": "key_text",
            "headerName": "sequence",
            "cellRenderer": "CmpLink",
            "minWidth": 140,
            "flex": 2,
            "tooltipField": "peptidoform",
            "headerTooltip": "stripped sequence; click it to open the precursor page of the "
            "row that carries peptide_q_value in " + side.upper(),
        }
    else:
        first = {
            "field": "key_text",
            "headerName": "protein group",
            "cellRenderer": "CmpLink",
            "minWidth": 150,
            "flex": 2,
            "tooltipField": "key_text",
            "headerTooltip": "protein_group; click it to open its protein page in " + side.upper(),
        }
    cols: list[dict[str, Any]] = [first]
    if unit == "precursor":
        cols.append(
            {
                "field": "charge",
                "headerName": "z",
                "width": 44,
                "maxWidth": 54,
                "type": "rightAligned",
                "headerTooltip": "charge",
            }
        )
    if experiment:
        cols.append(
            {
                "field": "run",
                "headerName": "run",
                "width": 52,
                "minWidth": 48,
                "maxWidth": 70,
                "headerTooltip": "the run of the winning row (the grouped q columns are "
                "experiment-wide)",
            }
        )
    lo, hi = score_range
    cols += [
        {
            "field": "q",
            "headerName": spec.q_column,
            "cellRenderer": "MvBar",
            "cellRendererParams": {
                "scale": "neglog10",
                "min": 1.0,
                "max": Q_BAR_FULL,
                "passColour": "var(--mantine-color-green-6)",
                "failColour": "var(--mantine-color-gray-5)",
                "threshold": t,
                "format": "q",
                "width": 26,
                "tip": "bar: -log10(q) from 1 to 1e-4",
            },
            "width": 100,
            "minWidth": 96,
            "headerTooltip": f"{spec.q_column} of {side.upper()} (the engine's value); the bar is "
            "-log10(q) from 1 (empty) to 1e-4 (full)",
        },
        {
            "field": "score",
            "headerName": "score",
            "cellRenderer": "MvBar",
            "cellRendererParams": {
                "scale": "linear",
                "min": lo,
                "max": hi,
                "colour": "var(--cmp-bar-score)",
                "format": "score",
                "width": 26,
                "tip": f"bar: score from {lo:.3g} to {hi:.3g} ({side.upper()}'s accepted range)",
            },
            "width": 96,
            "minWidth": 90,
            "headerTooltip": f"score of {side.upper()}'s winning row; the bar spans the scores of "
            f"{side.upper()}'s accepted {spec.plural}",
        },
        {
            "field": "other_q",
            "headerName": f"in {other}",
            "cellRenderer": "CmpOther",
            "cellRendererParams": {"side": other, "threshold": t},
            "width": 104,
            "minWidth": 96,
            "headerTooltip": f"What {other} has: the smallest {spec.q_column} over its target rows "
            f"of this key (it does not pass), or 'no target row' when {other} scored none "
            "(not in its library, or not scored)",
        },
    ]
    if unit == "protein_group":
        cols.append(
            {
                "field": "other_group",
                "headerName": f"members pass in {other} as",
                "minWidth": 140,
                "flex": 2,
                "tooltipField": "other_group",
                "headerTooltip": f"A protein group of {other} that passes and shares a member with "
                "this group: the proteins are found, but grouped differently",
            }
        )
    else:
        cols.append(
            {
                "field": "protein_group",
                "headerName": "protein group",
                "minWidth": 100,
                "flex": 1,
                "tooltipField": "protein_group",
                "headerTooltip": f"protein_group of {side.upper()}'s winning row",
            }
        )
    return cols


def unique_records(
    df: pd.DataFrame, unit: str, base: str | None, *, limit: int = GRID_LIMIT
) -> list[dict[str, Any]]:
    """Grid rows: the first ``limit`` keys, each with the address of its page (None when
    the side has no viewer to link to)."""
    out = []
    for r in df.head(limit).itertuples(index=False):
        if base is None:
            url = None
        elif unit == "protein_group":
            url = href(base, "protein", {"group": r.key})
        else:
            url = href(base, "precursor", {"run": r.run, "cid": int(r.cid)})
        other_q = float(r.other_q)
        out.append(
            {
                "key_text": r.peptidoform if unit == "precursor" else r.key,
                "peptidoform": r.peptidoform,
                "charge": int(r.charge),
                "run": r.run,
                "q": float(r.q),
                "score": float(r.score),
                "other_q": other_q if math.isfinite(other_q) else None,
                "other_rows": int(r.other_rows),
                "protein_group": r.protein_group,
                "other_group": getattr(r, "other_group", ""),
                "_href": url,
            }
        )
    return out


def unique_summary(df: pd.DataFrame, unit: str, other: str, t: float) -> Any:
    n = len(df)
    fails = int(df["other_q"].notna().sum()) if n else 0
    none = n - fails
    parts = [
        tip(
            dmc.Badge(
                f"{fails:,} above {t_text(t)} in {other}",
                size="sm",
                color="gray",
                variant="light",
                style={"textTransform": "none"},
            ),
            f"{other} has target rows of these keys, but its {cd.SPEC[unit].q_column} is "
            "above the threshold",
        ),
        tip(
            dmc.Badge(
                f"{none:,} no target row in {other}",
                size="sm",
                color="gray",
                variant="outline",
                style={"textTransform": "none"},
            ),
            f"{other} scored no target row of these keys: not in its library, or not scored",
        ),
    ]
    if unit == "protein_group" and n:
        regrouped = int((df["other_group"] != "").sum())
        parts.append(
            tip(
                dmc.Badge(
                    f"{regrouped:,} regrouped in {other}",
                    size="sm",
                    color="yellow",
                    variant="light",
                    style={"textTransform": "none"},
                ),
                f"A member of the group passes in {other} in another protein group",
            )
        )
    return dmc.Group(parts, gap=6)


def unique_card(
    side: str,
    rs: ResultSet,
    df: pd.DataFrame,
    *,
    unit: str,
    t: float,
    base: str | None,
    score_range: tuple[float, float],
) -> Any:
    other = "B" if side == "a" else "A"
    spec = cd.SPEC[unit]
    rows = unique_records(df, unit, base)
    grid = dag.AgGrid(
        id=f"cmp-grid-{side}",
        rowData=rows,
        columnDefs=unique_columns(
            unit, side, t=t, experiment=rs.is_experiment, score_range=score_range
        ),
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "suppressHeaderMenuButton": True,
            "sortingOrder": ["asc", "desc"],
        },
        dashGridOptions={
            **GRID_OPTIONS,
            "context": {"external": side == "b", "threshold": t},
            "localeText": {
                "noRowsToShow": f"Every {spec.singular} of {side.upper()} also passes in {other}."
            },
        },
        style={"height": "100%", "width": "100%"},
        className="cmp-grid",
    )
    more = (
        f"The first {GRID_LIMIT:,} by {spec.q_column} are listed; the TSV holds all {len(df):,}."
        if len(df) > GRID_LIMIT
        else ""
    )
    title = html.Span(
        [f"Only in {side.upper()} · ", html.Span(rs.root.name, className="cmp-title-name")]
    )
    export = tip(
        dmc.Button(
            "TSV",
            id=f"cmp-export-{side}",
            n_clicks=0,
            size="xs",
            variant="subtle",
            leftSection=icon("table", 13),
        ),
        f"Download every {spec.singular} only in {side.upper()} as TSV",
    )
    return section(
        title,
        unique_summary(df, unit, other, t),
        html.Div(grid, className="cmp-gridbox"),
        html.Div(more, className="cmp-note"),
        dcc.Download(id=f"cmp-download-{side}"),
        count=len(df),
        right=export,
        help=f"{spec.plural.capitalize()} that pass in {side.upper()} ({spec.q_column} ≤ "
        f"{stop_label(t)}) and not in {other}, sorted by q. 'in {other}' is the smallest "
        f"{spec.q_column} of {other}'s target rows of the key"
        + (" (B's pages open in B's viewer)." if side == "b" else "."),
        id=f"cmp-unique-{side}",
    )


def export_frame(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    cols = [
        "key",
        "q",
        "score",
        "run",
        "cid",
        "peptidoform",
        "charge",
        "protein_group",
        "other_q",
        "other_rows",
    ]
    if unit == "protein_group":
        cols.append("other_group")
    out = df[cols].copy()
    out = out.rename(
        columns={
            "q": cd.SPEC[unit].q_column,
            "cid": "candidate_id",
            "other_q": f"other_side_min_{cd.SPEC[unit].q_column}",
            "other_rows": "other_side_target_rows",
        }
    )
    return out


# --------------------------------------------------------------------------- without B


def start_card(rs: ResultSet) -> Any:
    command = f'mumdia-viewer "{rs.root}" --compare "<other result directory>"'
    steps = [
        "Start the viewer with --compare and a second run or experiment directory: two "
        "MuMDIA versions, two configurations, or two libraries.",
        "This result set is A. B is served as a full viewer of its own, at this address "
        "plus b/, so every identification links to its page on either side.",
        "The page shows the overlap of precursors, peptides and protein groups at the header "
        "threshold, the scores and quantities of the shared identifications, and the ones "
        "unique to each side.",
        "The sides may use different libraries: the viewer matches by peptidoform and "
        "charge, stripped sequence and protein group string, never by candidate id.",
    ]
    return dmc.Card(
        dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.ThemeIcon(
                            icon("compare", 22),
                            size=44,
                            radius="md",
                            variant="light",
                            color="indigo",
                        ),
                        html.Div(
                            [
                                dmc.Text("Start a comparison", fw=700, size="lg"),
                                dmc.Text(
                                    "No second result set was given, so there is nothing "
                                    "to compare yet.",
                                    size="sm",
                                    c="dimmed",
                                ),
                            ]
                        ),
                    ],
                    gap="md",
                ),
                html.Div(
                    [
                        dmc.Code(command, block=True, className="cmp-command", id="cmp-command"),
                        dcc.Clipboard(
                            target_id="cmp-command", title="Copy the command", className="cmp-copy"
                        ),
                    ],
                    className="cmp-command-box",
                ),
                dmc.List(
                    [dmc.ListItem(dmc.Text(s, size="sm")) for s in steps],
                    spacing=6,
                    size="sm",
                ),
                dmc.Text(
                    "Add --port, --fasta or --remap as for one result set. The command needs a "
                    "restart of the viewer; this page cannot open a second directory itself.",
                    size="xs",
                    c="dimmed",
                ),
            ],
            gap="md",
        ),
        p="xl",
        maw=860,
        className="cmp-start",
    )


def detail_skeleton() -> Any:
    """The lower part while it is computed (the unique keys need a pass over each side's
    scored table, cached afterwards)."""

    def card(h: int) -> Any:
        return dmc.Card(
            [dmc.Skeleton(h=12, w=180, radius="sm"), dmc.Skeleton(h=h, mt="md", radius="md")],
            p="md",
        )

    return dmc.Stack(
        [
            dmc.Grid(
                [
                    dmc.GridCol(card(380), span={"base": 12, "lg": 6}),
                    dmc.GridCol(card(380), span={"base": 12, "lg": 6}),
                ],
                gutter="lg",
            ),
            dmc.Group(
                [
                    dmc.Loader(size="xs", type="dots"),
                    dmc.Text("Comparing the identifications of A and B...", size="xs", c="dimmed"),
                ],
                gap="xs",
            ),
        ],
        gap="lg",
    )
