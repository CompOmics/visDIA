"""The cards of the spectrum browser (components only; the page and its callbacks are in
:mod:`.spectra`). Values come from :mod:`mumdia_viewer.data.scans` and
:mod:`mumdia_viewer.data.spectra`; a viewer computation carries a "derived" mark and
says in its tooltip how it was made.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import dcc, html

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data.qc import BASE_PEAK_LABEL, TIC_LABEL
from mumdia_viewer.data.scans import FragmentOverlay, IsotopeOverlay, ScanRef
from mumdia_viewer.data.spectra import NEAREST_MS1_LABEL, PRECEDING_MS1_LABEL, Spectrum

from .icons import _URIS, icon
from .state import href, stop_label
from .widgets import empty, graph, peptidoform, section

GRID_ID = "sp-grid"
SPECTRUM_GRAPH = "sp-spectrum"
NAV_GRAPH = "sp-nav"
Q_BAR_FULL = 1e-4
NEAR_OPTIONS = [
    {"value": "apex:2", "label": "apex ± 2 s"},
    {"value": "apex:5", "label": "apex ± 5 s"},
    {"value": "apex:10", "label": "apex ± 10 s"},
    {"value": "apex:20", "label": "apex ± 20 s"},
    {"value": "elution", "label": "in elution bounds"},
]
DEFAULT_NEAR = "apex:5"
LABEL_OPTIONS = [
    {"value": "10", "label": "Top 10"},
    {"value": "25", "label": "Top 25"},
    {"value": "0", "label": "None"},
]


# --------------------------------------------------------------------------- helpers


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def derived(text: str) -> Any:
    """The small "derived" mark of a viewer computation; ``text`` says how."""
    return tip(html.Span("derived", className="sp-derived"), f"Viewer-derived: {text}")


def compact(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and not math.isfinite(n)):
        return "n/a"
    n = float(n)
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n:.4g}"


def parse_near(value: Any) -> tuple[str, float]:
    """``"apex:5"`` -> ("apex", 5.0); ``"elution"`` -> ("elution", 0.0); else the default."""
    text = str(value or DEFAULT_NEAR)
    if text == "elution":
        return "elution", 0.0
    mode, _, num = text.partition(":")
    try:
        d = float(num)
    except ValueError:
        d = float("nan")
    if mode != "apex" or not math.isfinite(d) or d < 0:
        return parse_near(DEFAULT_NEAR)
    return "apex", d


def threshold_text(t: float) -> str:
    return f"run_psm_q ≤ {stop_label(t)}"


def fact(label: str, value: Any, text: str, *, sub: Any = None) -> Any:
    """A key value of the shown scan (an uppercase label over a large number)."""
    body: list[Any] = [
        html.Div(label, className="sp-fact-label"),
        html.Div(value, className="sp-fact-value"),
    ]
    if sub is not None:
        body.append(html.Div(sub, className="sp-fact-sub"))
    return tip(dmc.Paper(body, withBorder=True, className="sp-fact"), text)


def scan_title(ref: ScanRef | None) -> str:
    if ref is None:
        return "No scan"
    return f"MS{ref.level} scan {ref.scan_index:,}"


# --------------------------------------------------------------------------- header


def run_selector(rs: ResultSet, current: str) -> Any:
    """The run picker of an experiment; the browser keeps the RT when the run changes."""
    data = [{"value": r.name, "label": r.label} for r in rs.runs]
    if len(rs.runs) <= 8:
        control: Any = dmc.SegmentedControl(
            id="sp-run", data=data, value=current, size="sm", radius="xl", className="sp-runs"
        )
    else:
        control = dmc.Select(
            id="sp-run",
            data=data,
            value=current,
            w=220,
            radius="xl",
            size="sm",
            allowDeselect=False,
            searchable=True,
        )
    return tip(
        control,
        "Each run has its own scans. A run change opens the nearest scan at the same RT "
        "(same MS level and the same isolation window when the run has it).",
    )


def facts(spec: Spectrum | None, ref: ScanRef | None, tic: float | None) -> list[Any]:
    """The key values of the shown scan (they follow every step)."""
    if spec is None or ref is None:
        return []
    out = [
        fact("RT", f"{spec.rt:.2f} s", "rt_seconds of the scan", sub=f"row {spec.row:,}"),
    ]
    if ref.level == 2 and ref.lower is not None and ref.upper is not None:
        out.append(
            fact(
                "Isolation window",
                f"{ref.lower:.2f} to {ref.upper:.2f}",
                "window_lower and window_upper of the scan (m/z); window_target is the "
                "centre the instrument recorded",
                sub=f"window {ref.window_id} · target {ref.target:.2f}",
            )
        )
    else:
        out.append(
            fact("Isolation window", "none", "An MS1 scan has no isolation window", sub="MS1")
        )
    out.append(
        fact("Peaks", f"{spec.n_peaks:,}", "Length of the scan's peak list (as convert stored it)")
    )
    if spec.n_peaks:
        k = int(np.argmax(spec.intensity))
        out.append(
            fact(
                "Base peak",
                compact(float(spec.intensity[k])),
                BASE_PEAK_LABEL,
                sub=f"at {float(spec.mz[k]):.4f}",
            )
        )
    out.append(
        fact("TIC", compact(tic), TIC_LABEL, sub=html.Span("derived", className="sp-derived"))
    )
    return out


def header(rs: ResultSet, run_name: str, run_label: str, ref: ScanRef | None, sub: Any) -> Any:
    where = f"run {run_label} of {rs.root.name}" if rs.is_experiment else "single run"
    left = dmc.Stack(
        [
            dmc.Text(f"Spectrum browser · {where}", className="mv-eyebrow"),
            dmc.Group(
                [html.Div(scan_title(ref), id="sp-title", className="mv-title")]
                + ([run_selector(rs, run_name)] if rs.is_experiment else []),
                gap="md",
                align="center",
            ),
            html.Div(sub, id="sp-sub", className="sp-sub"),
        ],
        gap=6,
    )
    return dmc.Group(
        [left, dmc.Group(id="sp-facts", gap="sm", wrap="wrap", className="sp-facts")],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="sp-hero",
    )


def sub_line(spec: Spectrum | None, ref: ScanRef | None) -> Any:
    if spec is None or ref is None:
        return ""
    parts: list[Any] = []
    if spec.native_id:
        parts.append(
            tip(
                dmc.Group(
                    [icon("spectrum", 13), html.Span(spec.native_id, className="sp-mono")],
                    gap=6,
                    wrap="nowrap",
                ),
                "The mzML nativeID of the scan (spectra_ms2 column id)",
            )
        )
    else:
        parts.append(
            html.Span(
                f"scan_index {ref.scan_index:,} (run-global counter of MS1 and MS2 scans)",
                className="sp-mono",
            )
        )
    return parts


# --------------------------------------------------------------------------- navigation


def window_options(windows: pd.DataFrame) -> list[dict[str, str]]:
    out = []
    for w in windows.sort_values(["lower", "upper", "window_id"]).itertuples(index=False):
        out.append(
            {
                "value": str(int(w.window_id)),
                "label": f"{w.lower:.2f} to {w.upper:.2f} · w{int(w.window_id)}",
            }
        )
    return out


def nav_card(
    *,
    ref: ScanRef,
    windows: pd.DataFrame | None,
    window_value: str | None,
    rt_max: float,
    nav_fig: Any,
    has_ms1: bool,
    has_ms2: bool,
    nav_note: str,
) -> Any:
    def arrow(id_: str, name: str, label: str, key: str) -> Any:
        return tip(
            dmc.ActionIcon(
                icon(name, 15),
                id=id_,
                variant="default",
                size="lg",
                radius="xl",
                n_clicks=0,
                **{"aria-label": label},
            ),
            f"{label} ({key}); the step follows the Step control",
            multiline=False,
            w="auto",
        )

    levels = [
        {"value": "2", "label": "MS2", "disabled": not has_ms2},
        {"value": "1", "label": "MS1", "disabled": not has_ms1},
    ]
    controls = dmc.Group(
        [
            tip(
                dmc.NumberInput(
                    id="sp-scan-input",
                    value=ref.scan_index,
                    min=0,
                    step=1,
                    allowDecimal=False,
                    hideControls=True,
                    debounce=True,
                    w=150,
                    size="sm",
                    radius="xl",
                    leftSection=dmc.Text("scan", size="xs", c="dimmed"),
                    leftSectionWidth=46,
                ),
                "Go to a scan by its scan_index (the run-global counter of MS1 and MS2 "
                "scans); press Enter",
            ),
            tip(
                dmc.NumberInput(
                    id="sp-rt-input",
                    value=round(ref.rt, 2),
                    min=0,
                    step=1,
                    decimalScale=2,
                    hideControls=True,
                    debounce=True,
                    w=132,
                    size="sm",
                    radius="xl",
                    leftSection=dmc.Text("RT", size="xs", c="dimmed"),
                    leftSectionWidth=34,
                    rightSection=dmc.Text("s", size="xs", c="dimmed"),
                ),
                "Go to the scan nearest this retention time (seconds): of the shown MS level, "
                "and for MS2 of the chosen isolation window; press Enter",
            ),
            dmc.Divider(orientation="vertical"),
            tip(
                dmc.SegmentedControl(
                    id="sp-level", data=levels, value=str(ref.level), size="sm", radius="xl"
                ),
                "MS level. A switch opens the nearest scan of that level at the same RT.",
            ),
            tip(
                dmc.Select(
                    id="sp-window",
                    data=window_options(windows) if windows is not None else [],
                    value=window_value,
                    searchable=True,
                    allowDeselect=False,
                    w=250,
                    size="sm",
                    radius="xl",
                    leftSection=dmc.Text("m/z", size="xs", c="dimmed"),
                    leftSectionWidth=38,
                    disabled=not has_ms2,
                    comboboxProps={"shadow": "md"},
                    maxDropdownHeight=320,
                ),
                "Isolation window (window_lower to window_upper). A change opens the nearest "
                "MS2 scan of that window at the same RT.",
            ),
            dmc.Divider(orientation="vertical"),
            tip(
                dmc.SegmentedControl(
                    id="sp-scope",
                    data=[
                        {"value": "window", "label": "In window"},
                        {"value": "run", "label": "In run"},
                    ],
                    value="window",
                    size="sm",
                    radius="xl",
                ),
                "Step: 'In window' goes to the previous or next scan of the same isolation "
                "window (for MS1: the previous or next MS1 scan); 'In run' to the previous or "
                "next scan in acquisition order (scan_index), MS1 or MS2.",
            ),
            dmc.Group(
                [
                    arrow("scan-prev", "left", "Previous scan", "←"),
                    arrow("scan-next", "right", "Next scan", "→"),
                ],
                gap=4,
            ),
        ],
        gap="sm",
        wrap="wrap",
        className="sp-controls",
    )
    slider = html.Div(
        dmc.Slider(
            id="sp-slider",
            min=0,
            max=round(float(rt_max), 1),
            step=0.1,
            value=round(ref.rt, 1),
            updatemode="mouseup",
            size="sm",
            color="orange",
            label={"function": "spRtLabel"},
            thumbLabel="Retention time",
        ),
        className="sp-slider",
    )
    return section(
        "Navigation",
        controls,
        html.Div(graph(NAV_GRAPH, nav_fig, config={"displayModeBar": False}), className="sp-nav"),
        slider,
        html.Div(nav_note, id="sp-nav-note", className="sp-note"),
        help=(
            "The strip is the run's RT axis: green bars count the accepted targets per RT "
            "bin of their apex_rt (for MS2 only the precursors of the shown window), the grey "
            "area is the TIC of the scans shown (the viewer's sum of intensities; drawn once "
            "the run QC page has computed it), the orange line is the shown scan and the blue "
            "band the nearness rule of the list. Click the strip or drag the slider to go to "
            "an RT. ← and → step scans."
        ),
        id="sp-nav-card",
    )


# --------------------------------------------------------------------------- spectrum


def spectrum_card(fig: Any, title: Any, seq: Any, foot: Any) -> Any:
    right = dmc.Group(
        [
            tip(
                dmc.SegmentedControl(
                    id="sp-labels", data=LABEL_OPTIONS, value="10", size="xs", radius="xl"
                ),
                "m/z labels on the most intense peaks in view (zoom to label others)",
            ),
            tip(
                dmc.SegmentedControl(
                    id="sp-scale",
                    data=[
                        {"value": "base", "label": "Base peak"},
                        {"value": "window", "label": "Outside window"},
                        {"value": "matched", "label": "Matched"},
                    ],
                    value="window",
                    size="xs",
                    radius="xl",
                ),
                "Intensity scale: % of the base peak; % of the highest peak outside the "
                "isolation window (MS2: the unfragmented precursor is often the base peak; MS1: "
                "the base peak, or with a selected candidate the highest peak of the shown "
                "precursor region); or % of the highest peak matched by the selected candidate. "
                "Taller peaks are clipped at 110 %.",
            ),
        ],
        gap=6,
    )
    return section(
        html.Span(title, id="sp-spec-title"),
        html.Div(seq, id="sp-seq", className="sp-seq"),
        html.Div(graph(SPECTRUM_GRAPH, fig), className="sp-spectrum"),
        html.Div(foot, id="sp-spec-foot"),
        right=right,
        help=(
            "The peaks of the scan as convert stored them (spectra_ms1 or spectra_ms2), as "
            "stems. Drag to zoom; double click to reset. For an MS2 scan the orange band is "
            "its isolation window. Select a candidate in the list to overlay its library "
            "fragments (MS2: matched peaks in colour, the library mirrored below) or its "
            "precursor isotopes (MS1)."
        ),
        id="sp-spec-card",
    )


def spectrum_title(ref: ScanRef | None, spec: Spectrum | None) -> Any:
    if ref is None or spec is None:
        return "Spectrum"
    return f"Spectrum · MS{ref.level} scan {ref.scan_index:,}"


def tolerance_chips(ov: FragmentOverlay) -> Any:
    tol = ov.tolerance
    chips = [
        tip(
            dmc.Badge(f"tolerance {tol.tol_ppm:.4g} ppm", size="sm", color="blue", variant="light"),
            tol.label,
        ),
    ]
    if not tol.uses_grid:
        chips.append(
            dmc.Badge(f"offset {tol.offset_ppm:+.3g} ppm", size="sm", color="gray", variant="light")
        )
    chips.append(dmc.Badge(f"matcher {tol.matcher}", size="sm", color="gray", variant="light"))
    chips.append(
        tip(
            dmc.Badge(
                f"{ov.n_matched} of {ov.n_library} matched",
                size="sm",
                color="green" if ov.n_matched else "gray",
                variant="light",
            ),
            f"{ov.label}: the candidate's library fragments (its chromatogram rows) matched "
            "to this scan",
        )
    )
    chips.append(html.Span("viewer match", className="sp-derived"))
    return dmc.Group(
        [dmc.Badge(c, size="sm") if isinstance(c, str) else c for c in chips],
        gap=6,
        className="sp-chips",
    )


# --------------------------------------------------------------------------- candidates


def candidate_columns(threshold: float, score_lo: float, score_hi: float) -> list[dict[str, Any]]:
    """Column definitions of the candidate list (the shared renderers of clientside.js)."""
    return [
        {
            "colId": "_valid",
            "headerName": "",
            "field": "q",
            "cellRenderer": "MvValidation",
            "cellRendererParams": {
                "qField": "q",
                "threshold": threshold,
                "labelField": "label",
                "spikeField": "is_entrapment",
            },
            "width": 30,
            "minWidth": 30,
            "maxWidth": 30,
            "resizable": False,
            "sortable": False,
            "headerTooltip": f"Validation: a check when run_psm_q ≤ {stop_label(threshold)}",
        },
        {
            "field": "peptidoform",
            "headerName": "peptidoform",
            "cellRenderer": "MvPeptidoform",
            "minWidth": 84,
            "flex": 2,
            "tooltipValueGetter": {
                "function": "'candidate ' + params.data.cid + '; precursor m/z ' + "
                "d3.format('.4f')(params.data.mz) + '; protein group ' + (params.data.pg || 'none')"
            },
            "headerTooltip": "psms_scored.peptidoform. Click a row to overlay its fragments on "
            "the spectrum; hover for the candidate, its precursor m/z and its protein group.",
        },
        {
            "field": "charge",
            "headerName": "z",
            "width": 30,
            "minWidth": 30,
            "maxWidth": 40,
            "type": "rightAligned",
            "headerTooltip": "psms_scored.charge",
        },
        {
            "field": "delta",
            "headerName": "Δ apex",
            "width": 56,
            "minWidth": 54,
            "type": "rightAligned",
            "valueFormatter": {"function": "d3.format('+.2f')(params.value)"},
            "headerTooltip": "apex_rt - RT of the shown scan, in seconds (psms_scored.apex_rt)",
        },
        {
            "field": "matched",
            "headerName": "frags",
            "cellRenderer": "SpMatched",
            "width": 68,
            "minWidth": 66,
            "headerTooltip": "Library fragments of the candidate (its chromatogram rows) that "
            "match a peak of the shown scan with the extraction tolerance and the engine's "
            "predicate (viewer match). MS1 scans: not computed.",
        },
        {
            "field": "score",
            "headerName": "score",
            "cellRenderer": "MvBar",
            "cellRendererParams": {
                "scale": "linear",
                "min": score_lo,
                "max": score_hi,
                "colour": "var(--sp-bar-score)",
                "format": "score",
                "width": 16,
                "tip": f"bar: score from {score_lo:.3g} to {score_hi:.3g} (the run's range)",
            },
            "width": 78,
            "minWidth": 76,
            "headerTooltip": "psms_scored.score, the rescorer's score; the bar spans the "
            "scored table's score range",
        },
        {
            "field": "q",
            "headerName": "run_psm_q",
            "cellRenderer": "MvBar",
            "cellRendererParams": {
                "scale": "neglog10",
                "min": 1.0,
                "max": Q_BAR_FULL,
                "passColour": "var(--mantine-color-green-6)",
                "failColour": "var(--mantine-color-gray-5)",
                "threshold": threshold,
                "format": "q",
                "width": 16,
                "tip": "bar: -log10(q) from 1 to 1e-4",
            },
            "width": 86,
            "minWidth": 84,
            "headerTooltip": "psms_scored.run_psm_q: the PSM q within the run; the bar is "
            "-log10(q) from 1 (empty) to 1e-4 (full), green at or below the threshold",
        },
        {
            "colId": "_open",
            "headerName": "",
            "field": "cid",
            "cellRenderer": "SpOpen",
            "width": 30,
            "minWidth": 30,
            "maxWidth": 30,
            "resizable": False,
            "sortable": False,
            "headerTooltip": "Open the precursor page",
        },
    ]


def candidate_records(
    df: pd.DataFrame, selected: int | None, counts: Mapping[int, tuple[int, int]]
) -> list[dict[str, Any]]:
    out = []
    for r in df.to_dict("records"):
        cid = int(r["candidate_id"])
        row: dict[str, Any] = {
            "_key": str(cid),
            "cid": cid,
            "peptidoform": str(r["peptidoform"]),
            "charge": int(r["charge"]),
            "label": str(r["label"]),
            "mz": round(float(r["precursor_mz"]), 4) if pd.notna(r["precursor_mz"]) else None,
            "apex_rt": round(float(r["apex_rt"]), 2),
            "delta": round(float(r["delta_rt"]), 2),
            "score": round(float(r["score"]), 4),
            "q": float(r["run_psm_q"]),
            "pg": str(r["protein_group"] or ""),
            "_sel": cid == selected,
        }
        hit = counts.get(cid)
        if hit is not None:
            row["matched"], row["n_lib"] = int(hit[0]), int(hit[1])
        if r.get("is_entrapment"):
            row["is_entrapment"] = True
        out.append(row)
    return out


GRID_OPTIONS: dict[str, Any] = {
    "theme": {"function": "mvsTheme(themeQuartz)"},
    "rowHeight": 27,
    "headerHeight": 28,
    "animateRows": False,
    "suppressCellFocus": True,
    "tooltipShowDelay": 350,
    "rowClassRules": {"sp-row-selected": "params.data._sel"},
    "localeText": {"noRowsToShow": "No accepted identification is near this scan."},
}


def candidates_card(
    *,
    base: str,
    run_name: str,
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    count: int,
    note: Any,
    decoys: bool = False,
    scroll: Any = None,
) -> Any:
    extra = {"scrollTo": scroll} if scroll else {}
    grid = dag.AgGrid(
        id=GRID_ID,
        rowData=rows,
        columnDefs=columns,
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "suppressHeaderMenuButton": True,
            "sortingOrder": ["asc", "desc"],
        },
        getRowId="params.data._key",
        columnSize="responsiveSizeToFit",
        dashGridOptions={
            **GRID_OPTIONS,
            "context": {"base": base, "run": run_name, "icon": _URIS["external"]},
        },
        style={"height": "100%", "width": "100%"},
        className="sp-grid",
        **extra,
    )
    title = html.Span(
        [
            html.Span("Identifications near the scan"),
            dmc.Badge(
                f"{count:,}",
                id="sp-cand-count",
                size="sm",
                color="gray",
                variant="light",
                style={"textTransform": "none"},
            ),
        ],
        className="sp-title-count",
    )
    right = dmc.Group(
        [
            tip(
                dmc.Select(
                    id="sp-near",
                    data=NEAR_OPTIONS,
                    value=DEFAULT_NEAR,
                    allowDeselect=False,
                    w=150,
                    size="xs",
                    radius="xl",
                    comboboxProps={"shadow": "md"},
                ),
                "Which accepted identifications are near the scan: apex_rt within a few seconds "
                "of the scan, or the scan inside their elution bounds (elution_lo to "
                "elution_hi). For MS2 their precursor m/z must also lie in the scan's window.",
            ),
            tip(
                dmc.Switch(id="sp-decoys", label="decoys", size="xs", checked=decoys),
                "Also list the decoys that pass the same run_psm_q cut (a diagnostic)",
            ),
        ],
        gap=8,
    )
    return section(
        title,
        html.Div(grid, className="sp-gridbox"),
        html.Div(note, id="sp-cand-note", className="sp-note"),
        right=right,
        help=(
            "Accepted identifications of the run (run_psm_q at or below the header threshold) "
            "whose apex is near the shown scan and, for an MS2 scan, whose precursor m/z lies "
            "in its isolation window. Click a row to overlay its fragments; the arrow at the "
            "end opens its precursor page. TableQuery has no RT or m/z filter, so the list is "
            "made here (data.scans.candidates_near)."
        ),
        id="sp-cand-card",
    )


def candidate_note(attrs: Mapping[str, Any], n: int) -> Any:
    rule = attrs.get("rule", "")
    label = attrs.get("label", "")
    parts: list[Any] = [
        tip(
            html.Span(f"{n:,} {'row' if n == 1 else 'rows'}: {label}.", className="sp-note-main"),
            f"Precursor m/z: {attrs.get('mz_source', '')}.",
        ),
        html.Span(f" Near: {rule}."),
    ]
    if attrs.get("note"):
        parts.append(html.Span(" " + str(attrs["note"])))
    return parts


# --------------------------------------------------------------------------- fragments


def fragments_card(title: Any, body: Any) -> Any:
    return section(
        html.Span(title, id="sp-frag-title"),
        html.Div(body, id="sp-frag-body"),
        help=(
            "The selected candidate's library fragments on its sequence (PeptideShaker's ion "
            "table): b ions left, y ions right, by fragment charge. A cell holds the library "
            "m/z; a matched cell is highlighted, with the raw ppm error of the matched peak in "
            "the shown scan. Only the library's fragments are placed (MuMDIA extracts only "
            "those). For an MS1 scan: the candidate's precursor isotopes."
        ),
        id="sp-frag-card",
    )


def isotope_table(iso: IsotopeOverlay) -> Any:
    head = html.Thead(
        html.Tr(
            [
                html.Th("isotope"),
                html.Th("m/z", className="mv-num"),
                html.Th("± ppm window", className="mv-num"),
                html.Th("peaks", className="mv-num"),
                html.Th(["sum ", derived(iso.label)], className="mv-num"),
            ]
        )
    )
    names = {-1: "M-1", 0: "M (mono)", 1: "M+1", 2: "M+2"}
    rows = []
    for r in iso.rows:
        rows.append(
            html.Tr(
                [
                    html.Td(names.get(int(r["k"]), str(int(r["k"])))),
                    html.Td(f"{r['mz']:.4f}", className="mv-num"),
                    html.Td(f"{r['lo']:.4f} to {r['hi']:.4f}", className="mv-num"),
                    html.Td(f"{int(r['n_peaks'])}", className="mv-num"),
                    html.Td(f"{r['sum']:,.0f}", className="mv-num"),
                ]
            )
        )
    return html.Div(
        [
            html.Table([head, html.Tbody(rows)], className="sp-iso"),
            dmc.Text(
                f"Isotope m/z: precursor_mz + k * 1.003354835 / {iso.charge} (extract's "
                f"rule); tolerance {iso.tol_source}. The engine's own MS1 values are the "
                "ms1_* columns of psms_extracted, sampled in the MS1 scan nearest apex_rt "
                "(see the precursor page); the sums here equal them only in that scan.",
                size="xs",
                c="dimmed",
                mt=6,
            ),
        ]
    )


def no_candidate(text: str) -> Any:
    return empty(text, "spectrum")


# --------------------------------------------------------------------------- scan details


def _kv(label: str, value: Any, text: str | None = None) -> Any:
    cell = html.Div(
        [html.Div(label, className="sp-kv-label"), html.Div(value, className="sp-kv-value")],
        className="sp-kv",
    )
    return tip(cell, text) if text else cell


def jump_button(label: str, ref: ScanRef, text: str) -> Any:
    return tip(
        # spectra.js turns a click into sp-goto (the scan's level and row).
        html.Button(
            [icon("spectrum", 12), html.Span(label)],
            className="sp-jump",
            **{"data-level": str(ref.level), "data-row": str(ref.row)},
        ),
        text,
    )


def details_card(body: Any) -> Any:
    return section(
        "Scan",
        html.Div(body, id="sp-details"),
        help=(
            "The shown scan's values from the spectra table, and the related scans: for an "
            "MS2 scan the MS1 scan nearest in RT (the scan the engine samples for its MS1 "
            "features) and the preceding MS1 scan (the acquisition parent, ms2_to_ms1)."
        ),
        id="sp-details-card",
    )


def details_body(
    spec: Spectrum | None,
    ref: ScanRef | None,
    links: Mapping[str, Any],
    *,
    ms2_here: ScanRef | None = None,
) -> list[Any]:
    if spec is None or ref is None:
        return [empty("No scan.")]
    kvs = [
        _kv("scan_index", f"{ref.scan_index:,}", "Run-global counter of MS1 and MS2 scans"),
        _kv(
            "table row",
            f"{ref.row:,}",
            f"Row of spectra_ms{ref.level}.parquet (not the scan_index)",
        ),
        _kv("RT", f"{spec.rt:.3f} s", "rt_seconds"),
        _kv("level", f"MS{ref.level}"),
    ]
    if ref.level == 2:
        kvs.append(
            _kv(
                "precursor_mz",
                f"{spec.precursor_mz:.4f}" if spec.precursor_mz is not None else "not recorded",
                "The precursor m/z the MS2 scan records (null on Astral: DIA scans name a window)",
            )
        )
        kvs.append(
            _kv(
                "charge",
                str(spec.precursor_charge) if spec.precursor_charge else "not recorded",
                "precursor_charge of the MS2 scan",
            )
        )
    buttons: list[Any] = []
    near = links.get("nearest")
    prev = links.get("preceding")
    if isinstance(near, ScanRef):
        buttons.append(
            jump_button(
                f"Nearest MS1 · {near.scan_index:,} ({near.rt - ref.rt:+.2f} s)",
                near,
                NEAREST_MS1_LABEL,
            )
        )
    if isinstance(prev, ScanRef):
        buttons.append(
            jump_button(
                f"Preceding MS1 · {prev.scan_index:,} ({prev.rt - ref.rt:+.2f} s)",
                prev,
                PRECEDING_MS1_LABEL,
            )
        )
    elif links.get("preceding_note"):
        buttons.append(dmc.Text(f"Preceding MS1: {links['preceding_note']}", size="xs", c="dimmed"))
    if ms2_here is not None:
        buttons.append(
            jump_button(
                f"MS2 in window {ms2_here.window_id} · {ms2_here.scan_index:,} "
                f"({ms2_here.rt - ref.rt:+.2f} s)",
                ms2_here,
                "The MS2 scan of the chosen isolation window nearest this MS1 scan in RT",
            )
        )
    out: list[Any] = [html.Div(kvs, className="sp-kvs")]
    if buttons:
        out.append(dmc.Group(buttons, gap=6, mt="sm"))
    return out


def footer(timings: Mapping[str, float], notes: Sequence[str] = ()) -> Any:
    parts = [f"Page built in {timings.get('page', 0.0):,.0f} ms"]
    for key, label in (
        ("accepted", "accepted rows"),
        ("spectrum", "spectrum"),
        ("candidates", "candidates"),
        ("matches", "fragment matches"),
    ):
        if key in timings:
            parts.append(f"{label} {timings[key]:,.0f} ms")
    return dmc.Group(
        [
            icon("clock", 13),
            dmc.Text("; ".join(parts) + ".", size="xs", c="dimmed", id="sp-timings"),
        ]
        + [dmc.Text(n, size="xs", c="dimmed") for n in notes],
        gap=6,
        className="sp-footer",
    )


def candidate_header(row: Mapping[str, Any] | None, ref: ScanRef) -> str:
    """The fragments card's title."""
    return "Precursor isotopes" if ref.level == 1 else "Fragment ions"


def candidate_line(row: Mapping[str, Any], base: str, run_name: str) -> Any:
    """The selected candidate above its fragments: peptidoform, charge, ids and a link."""
    cid = int(row["candidate_id"])
    pmz = row.get("precursor_mz")
    bits = [f"{int(row['charge'])}+"]
    if pmz is not None and math.isfinite(float(pmz)):
        bits.append(f"m/z {float(pmz):.4f}")
    bits.append(f"apex {float(row['apex_rt']):.2f} s")
    return dmc.Group(
        [
            html.Span(peptidoform(str(row["peptidoform"]), size="1.02em"), className="sp-frag-pep"),
            dmc.Text(" · ".join(bits), size="xs", c="dimmed"),
            dcc.Link(
                dmc.Group([html.Span("Precursor page"), icon("external", 12)], gap=4),
                href=href(base, "precursor", {"run": run_name, "cid": cid}),
                className="sp-link",
            ),
        ],
        gap=8,
        mb=8,
        wrap="wrap",
    )
