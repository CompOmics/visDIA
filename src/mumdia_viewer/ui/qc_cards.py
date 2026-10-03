"""The cards of the run QC page (components only; the page and its callbacks are in
:mod:`.qc`). Values come from :mod:`mumdia_viewer.data.qc`; a viewer computation carries
a "derived" mark and says in its tooltip how it was made.
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
from mumdia_viewer.data.qc import (
    BASE_PEAK_LABEL,
    MISSED_CLEAVAGE_RULE,
    TIC_LABEL,
    Acquisition,
    IdDistributions,
    PeakCounts,
    WindowScheme,
)
from mumdia_viewer.data.spectra import Spectrum

from . import qc_figures as qf
from .icons import icon
from .state import PageContext, href, stop_label
from .widgets import empty, graph, section

GRID_ID = "qc-bin-grid"
RT_GRAPH = "qc-rt"
SPECTRUM_GRAPH = "qc-spectrum"
WINDOWS_GRAPH = "qc-windows"
PEAKS_GRAPH = "qc-peaks"
Q_BAR_FULL = 1e-4


# --------------------------------------------------------------------------- helpers


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def derived(text: str) -> Any:
    """The small "derived" mark of a viewer computation; ``text`` says how."""
    return tip(html.Span("derived", className="qc-derived"), f"Viewer-derived: {text}")


def compact(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and not math.isfinite(n)):
        return "n/a"
    n = float(n)
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n:.4g}"


def fact(label: str, value: Any, text: str, *, id_: str | None = None, sub: Any = None) -> Any:
    """A key value in the page header (an uppercase label over a large number)."""
    body: list[Any] = [
        html.Div(label, className="qc-fact-label"),
        html.Div(value, className="qc-fact-value", id=id_)
        if id_
        else html.Div(value, className="qc-fact-value"),
    ]
    if sub is not None:
        body.append(html.Div(sub, className="qc-fact-sub"))
    return tip(dmc.Paper(body, withBorder=True, className="qc-fact"), text)


def run_where(rs: ResultSet, run_label: str) -> str:
    return f"run {run_label}" if rs.is_experiment else "single run"


def threshold_text(t: float) -> str:
    return f"run_psm_q ≤ {stop_label(t)}"


# --------------------------------------------------------------------------- header


def run_selector(rs: ResultSet, current: str) -> Any:
    """The run picker of an experiment (each run is its own QC page: qc?run=<name>)."""
    data = [{"value": r.name, "label": r.label} for r in rs.runs]
    if len(rs.runs) <= 8:
        control: Any = dmc.SegmentedControl(
            id="qc-run",
            data=data,
            value=current,
            size="sm",
            radius="xl",
            className="qc-runs",
        )
    else:
        control = dmc.Select(
            id="qc-run",
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
        "Each run has its own QC page. The identification views count each run on "
        "run_psm_q (PSM-level FDR within the run); the grouped q columns are experiment-wide.",
    )


def header(
    ctx: PageContext,
    run_name: str,
    run_label: str,
    acq: Acquisition,
    n_accepted: int | None,
    counts: Mapping[str, int] | None,
) -> Any:
    rs = ctx.rs
    where = run_where(rs, run_label)
    eyebrow = f"Run QC · {where}"
    if rs.is_experiment:
        eyebrow += f" of {rs.root.name}"
    title = run_label if rs.is_experiment else rs.root.name
    lines: list[Any] = []
    if acq.mzml:
        lines.append(
            tip(
                dmc.Group(
                    [icon("spectrum", 13), dmc.Text(acq.mzml, className="qc-mzml")],
                    gap=6,
                    wrap="nowrap",
                ),
                "params.mzml of the run's spectra_ms2.parquet.report.json (the converted input)",
            )
        )
    left_children: list[Any] = [
        dmc.Text(eyebrow, className="mv-eyebrow"),
        dmc.Group(
            [html.Div(title, className="mv-title")]
            + ([run_selector(rs, run_name)] if rs.is_experiment else []),
            gap="md",
            align="center",
        ),
        *lines,
    ]
    width = acq.width_median
    width_text = f"{width:g} Th wide" if width is not None and math.isfinite(width) else None
    facts = [
        fact(
            "MS1 scans",
            f"{acq.n_ms1:,}" if acq.n_ms1 is not None else "n/a",
            "Rows of spectra_ms1.parquet (the run's MS1 scans)",
        ),
        fact(
            "MS2 scans",
            f"{acq.n_ms2:,}" if acq.n_ms2 is not None else "n/a",
            "Rows of spectra_ms2.parquet (the run's MS2 scans)",
        ),
        fact(
            "Cycle time",
            f"{acq.cycle_time_s:.3f} s" if acq.cycle_time_s is not None else "n/a",
            "Viewer-derived: the median over the isolation windows of the median RT step "
            "between consecutive scans of one window (isolation_scheme, cycle_time_s)",
            sub=html.Span("derived", className="qc-derived"),
        ),
        fact(
            "Isolation windows",
            f"{acq.n_windows:,}" if acq.n_windows is not None else "n/a",
            "Windows of isolation_windows.parquet and their median width (upper - lower)",
            sub=width_text,
        ),
        fact(
            "Accepted PSMs",
            f"{n_accepted:,}" if n_accepted is not None else "n/a",
            "Target rows of the run with run_psm_q at or below the header threshold "
            "(PSM-level FDR within the run). It follows the threshold.",
            id_="qc-fact-accepted",
            sub=html.Span(threshold_text(ctx.threshold), id="qc-fact-accepted-q"),
        ),
    ]
    return dmc.Group(
        [
            dmc.Stack(left_children, gap=6),
            dmc.Group(facts, gap="sm", wrap="wrap", className="qc-facts"),
        ],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="qc-hero",
    )


# --------------------------------------------------------------------------- RT card


def window_options(ws: WindowScheme | None) -> list[dict[str, str]]:
    options = [{"value": "all", "label": "All windows"}]
    if ws is None:
        return options
    df = ws.frame.sort_values(["lower", "upper"])
    for r in df.itertuples(index=False):
        options.append(
            {
                "value": str(int(r.window_id)),
                "label": f"{r.lower:.2f} to {r.upper:.2f} m/z (window {int(r.window_id)})",
            }
        )
    return options


def rt_help(population: str) -> str:
    return (
        f"{TIC_LABEL}. {BASE_PEAK_LABEL}. Identifications: {population}, counted per RT bin "
        "of apex_rt (green), with the decoys that pass the same cut (orange). MS2 scans per "
        "second: the MS2 scans of each bin over the acquired part of the bin; the hover gives "
        "the cycle time (median RT step between consecutive scans of one window). The tracks "
        "share the RT axis: zoom one to zoom all."
    )


def rt_card(
    figure: Any,
    *,
    level: int,
    window: str,
    ws: WindowScheme | None,
    has_ms2: bool,
    note: str,
    population: str,
) -> Any:
    level_control = tip(
        dmc.SegmentedControl(
            id="qc-level",
            data=[
                {"value": "1", "label": "MS1"},
                {"value": "2", "label": "MS2", "disabled": not has_ms2},
            ],
            value=str(level),
            size="xs",
            radius="xl",
        ),
        "The MS level of the TIC and base-peak tracks. The first MS2 view reads the peak "
        "lists of every MS2 scan once; the result is cached for later visits.",
    )
    window_control = dmc.Select(
        id="qc-window",
        data=window_options(ws),
        value=window,
        size="xs",
        radius="xl",
        w=250,
        allowDeselect=False,
        searchable=True,
        comboboxProps={"shadow": "md"},
        maxDropdownHeight=320,
    )
    right = dmc.Group(
        [
            html.Div(
                tip(
                    window_control,
                    "Every window, or the scans of one isolation window (its MS2 TIC is a "
                    "chromatogram). A click on the isolation-window chart picks one.",
                ),
                id="qc-window-box",
                style=window_style(level),
            ),
            level_control,
        ],
        gap="xs",
        wrap="nowrap",
    )
    body = dcc.Loading(
        graph(RT_GRAPH, figure),
        delay_show=400,
        overlay_style={"visibility": "visible", "opacity": 0.5},
        custom_spinner=dmc.Stack(
            [dmc.Loader(type="dots"), dmc.Text("Reading the scans", size="xs", c="dimmed")],
            align="center",
            gap=4,
        ),
    )
    return section(
        "Signal across retention time",
        body,
        html.Div(note, id="qc-rt-note", className="qc-note"),
        right=right,
        subtitle=("Viewer sums and maxima of the peak lists. Click a scan or a bar."),
        help=rt_help(population),
        id="qc-rt-card",
    )


def window_style(level: int) -> dict[str, str]:
    """The window picker is shown in the MS2 view only."""
    return {} if level == 2 else {"display": "none"}


def rt_note(level: int, n_view: int, n_drawn: int, envelope: bool, window: str | None) -> str:
    what = f"MS{level}, {window}" if window else f"MS{level}"
    if envelope:
        return (
            f"{what}: {n_view:,} scans in view, drawn as the first, lowest, highest and last "
            f"scan of each of 2,000 RT bins ({n_drawn:,} points). Zoom in for every scan "
            "(12,000 or fewer in view)."
        )
    return f"{what}: every scan in view is drawn ({n_drawn:,} scans)."


# --------------------------------------------------------------------------- scan card


def scan_nav() -> Any:
    def arrow(id_: str, name: str, label: str, key: str) -> Any:
        return tip(
            dmc.ActionIcon(
                icon(name, 15),
                id=id_,
                variant="default",
                size="md",
                radius="xl",
                n_clicks=0,
                **{"aria-label": label},
            ),
            f"{label} ({key})",
            multiline=False,
            w="auto",
        )

    return dmc.Group(
        [
            arrow("scan-prev", "left", "Previous scan", "←"),
            arrow("scan-next", "right", "Next scan", "→"),
        ],
        gap=4,
    )


def _info_row(label: str, value: Any, text: str | None = None) -> Any:
    head = dmc.Text(label, className="qc-kv-label")
    cell = html.Div([head, html.Div(value, className="qc-kv-value")], className="qc-kv")
    return tip(cell, text) if text else cell


def scan_body(
    spec: Spectrum | None,
    *,
    tic: float | None,
    scheme: str,
    error: str | None = None,
    step_note: str = "",
) -> list[Any]:
    """The spectrum of the selected scan and its key values."""
    if spec is None:
        return [empty(error or "Click a point of the TIC or the base peak to see its scan.")]
    bp_mz = bp = None
    if spec.n_peaks:
        k = int(np.argmax(spec.intensity))
        bp_mz, bp = float(spec.mz[k]), float(spec.intensity[k])
    window = (
        (spec.window_lower, spec.window_upper)
        if spec.level == 2 and spec.window_lower is not None and spec.window_upper is not None
        else None
    )
    fig = qf.spectrum_figure(
        spec.mz,
        spec.intensity,
        level=spec.level,
        scheme=scheme,
        window=window,
        base_peak=(bp_mz, bp) if bp_mz is not None else None,
    )
    rows = [
        _info_row("RT", f"{spec.rt:.2f} s", "rt_seconds of the scan"),
        _info_row(
            "scan_index",
            f"{spec.scan_index:,}",
            f"scan_index (run-global counter of MS1 and MS2 scans); table row {spec.row:,}",
        ),
        _info_row("peaks", f"{spec.n_peaks:,}", "Length of the scan's peak list"),
        _info_row("TIC", compact(tic), f"{TIC_LABEL}"),
        _info_row(
            "base peak",
            f"{compact(bp)} at {bp_mz:.4f}" if bp_mz is not None else "n/a",
            BASE_PEAK_LABEL,
        ),
    ]
    if window is not None:
        rows.append(
            _info_row(
                "window",
                f"{window[0]:.2f} to {window[1]:.2f} m/z",
                f"isolation window {spec.window_id} (window_lower, window_upper); the band in "
                "the spectrum",
            )
        )
    if spec.native_id:
        rows.append(_info_row("native id", spec.native_id, "The mzML nativeID (column id)"))
    out: list[Any] = [graph(SPECTRUM_GRAPH, fig), html.Div(rows, className="qc-kvs")]
    if step_note:
        out.append(dmc.Text(step_note, size="xs", c="dimmed", mt=4))
    return out


def scan_title(spec: Spectrum | None) -> Any:
    if spec is None:
        return "Scan"
    return f"MS{spec.level} scan {spec.scan_index:,}"


def scan_card(title: Any, body: list[Any]) -> Any:
    return section(
        html.Span(title, id="qc-scan-title"),
        html.Div(body, id="qc-scan-body"),
        right=scan_nav(),
        help=(
            "The peaks of one scan as stored by convert (spectra_ms1 or spectra_ms2). Click "
            "a point of the TIC or the base peak to pick a scan; ← and → step to the "
            "neighbouring scan (MS1: the previous or next MS1 scan; MS2: the previous or next "
            "scan of the same isolation window)."
        ),
        id="qc-scan-card",
    )


# --------------------------------------------------------------------------- bin card


def bin_title(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "Identifications in an RT bin"
    return f"Identifications at {lo:,.0f} to {hi:,.0f} s"


def bin_columns(threshold: float) -> list[dict[str, Any]]:
    """Column definitions of the RT-bin list (the shared renderers of clientside.js)."""
    return [
        {
            "colId": "_valid",
            "headerName": "",
            "field": "run_psm_q",
            "cellRenderer": "MvValidation",
            "cellRendererParams": {
                "qField": "run_psm_q",
                "threshold": threshold,
                "labelField": "label",
                "spikeField": "is_entrapment",
            },
            "width": 38,
            "minWidth": 38,
            "maxWidth": 38,
            "resizable": False,
            "sortable": False,
            "cellClass": "qc-cell-valid",
            "headerTooltip": f"Validation: a check when run_psm_q ≤ {stop_label(threshold)}",
        },
        {
            "field": "peptidoform",
            "headerName": "peptidoform",
            "cellRenderer": "QcPeptidoform",
            "minWidth": 100,
            "tooltipValueGetter": {
                "function": "'score ' + d3.format('.4f')(params.data.score) + "
                "'; protein group ' + (params.data.pg || 'none')"
            },
            "headerTooltip": "psms_scored.peptidoform; click it to open the precursor page. "
            "Hover a row for its score and protein group.",
        },
        {
            "field": "charge",
            "headerName": "z",
            "width": 38,
            "minWidth": 34,
            "type": "rightAligned",
            "headerTooltip": "psms_scored.charge",
        },
        {
            "field": "apex_rt",
            "headerName": "RT (s)",
            "width": 66,
            "type": "rightAligned",
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
            "headerTooltip": "psms_scored.apex_rt, seconds",
        },
        {
            "field": "run_psm_q",
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
                "width": 22,
                "tip": "bar: -log10(q) from 1 to 1e-4",
            },
            "width": 98,
            "minWidth": 90,
            "headerTooltip": "psms_scored.run_psm_q: the PSM q within the run; the bar is "
            "-log10(q) from 1 (empty) to 1e-4 (full), green at or below the threshold",
        },
    ]


def bin_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Grid rows of :func:`mumdia_viewer.data.qc.ids_in_rt_range` (short rows: a bin can
    hold thousands). The browser builds each precursor link from the grid context
    (:func:`precursor_link`, the twin of the QcPeptidoform renderer)."""
    out = []
    for r in df.to_dict("records"):
        cid = int(r["candidate_id"])
        row = {
            "_key": str(cid),
            "cid": cid,
            "peptidoform": str(r["peptidoform"]),
            "charge": int(r["charge"]),
            "apex_rt": round(float(r["apex_rt"]), 2),
            "score": round(float(r["score"]), 4),
            "run_psm_q": float(r["run_psm_q"]),
            "pg": str(r["protein_group"] or ""),
        }
        if r.get("is_entrapment"):
            row["is_entrapment"] = True
        out.append(row)
    return out


def precursor_link(base: str, run_name: str, cid: int) -> str:
    """The precursor page of a row, as the QcPeptidoform renderer builds it (qc.js)."""
    return href(base, "precursor", {"run": run_name, "cid": int(cid)})


GRID_OPTIONS: dict[str, Any] = {
    "theme": {"function": "mvqcTheme(themeQuartz)"},
    "rowHeight": 26,
    "headerHeight": 28,
    "animateRows": False,
    "suppressCellFocus": True,
    "tooltipShowDelay": 350,
    "localeText": {"noRowsToShow": "No accepted identification has its apex in this bin."},
}


def bin_card(
    *,
    base: str,
    run_name: str,
    lo: float | None,
    hi: float | None,
    total: int,
    rows: list[dict[str, Any]],
    threshold: float,
    population: str,
    limit: int,
) -> Any:
    grid = dag.AgGrid(
        id=GRID_ID,
        rowData=rows,
        columnDefs=bin_columns(threshold),
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "suppressHeaderMenuButton": True,
            "sortingOrder": ["asc", "desc"],
        },
        getRowId="params.data._key",
        columnSize="responsiveSizeToFit",
        dashGridOptions={**GRID_OPTIONS, "context": {"base": base, "run": run_name}},
        style={"height": "100%", "width": "100%"},
        className="qc-grid",
    )
    shown = html.Div(
        f"The first {limit:,} by apex RT are listed." if total > limit else "",
        id="qc-bin-more",
        className="qc-note",
    )
    title = html.Span(
        [
            html.Span(bin_title(lo, hi), id="qc-bin-title"),
            dmc.Badge(
                f"{total:,}",
                id="qc-bin-count",
                size="sm",
                color="gray",
                variant="light",
                style={"textTransform": "none"},
                className="qc-title-badge",
            ),
        ],
        className="qc-title-count",
    )
    return section(
        title,
        html.Div(grid, className="qc-gridbox"),
        shown,
        help=(
            f"{population}, with lo ≤ apex_rt < hi (the last bin includes its upper edge). "
            "Click a bar of the identification track to list another bin; click a "
            "peptidoform to open its precursor page. TableQuery has no RT filter, so the "
            "list is made here."
        ),
        id="qc-bin-card",
    )


# --------------------------------------------------------------------------- windows card


def windows_card(ws: WindowScheme | None, figure: Any, error: str | None = None) -> Any:
    if ws is None:
        return section("Isolation windows", empty(error or "No isolation windows."))
    gap_text = (
        f"{ws.n_gaps:,} gaps between adjacent windows (the largest {ws.largest_gap:.4f} Th)"
        if ws.n_gaps
        else "no gap between adjacent windows"
    )
    overlap_text = (
        f"{ws.n_overlaps:,} overlaps (the largest {ws.largest_overlap:.4f} Th)"
        if ws.n_overlaps
        else "no overlap"
    )
    subtitle = (
        f"{ws.mz_lo:.2f} to {ws.mz_hi:.2f} m/z, width {ws.width_median:g} Th"
        + (f" ({ws.width_min:g} to {ws.width_max:g})" if ws.width_max - ws.width_min > 1e-9 else "")
        + (f", cycle {ws.cycle_time_s:.3f} s" if ws.cycle_time_s is not None else "")
    )
    return section(
        "Isolation windows",
        graph(WINDOWS_GRAPH, figure),
        html.Div(
            f"{gap_text}; {overlap_text}. Click a window to show its MS2 TIC.",
            className="qc-note",
        ),
        count=ws.n_windows,
        subtitle=subtitle,
        help=(
            f"{ws.label}. Each bar is the m/z range of one window (lower to upper); its row is "
            "its position in the acquisition order. Gap or overlap: upper of a window minus "
            "lower of the next one in m/z order (isolation_scheme, overlap_with_next)."
        ),
        id="qc-windows-card",
    )


# --------------------------------------------------------------------------- peaks card


def _pct_tile(label: str, value: Any, text: str, *, warn: bool = False) -> Any:
    return tip(
        html.Div(
            [
                html.Div(label, className="qc-mini-label"),
                html.Div(value, className="qc-mini-value" + (" qc-warn" if warn else "")),
            ],
            className="qc-mini",
        ),
        text,
    )


def peaks_body(pc: PeakCounts, scheme: str) -> list[Any]:
    p = pc.percentiles
    if pc.cap is None:
        at_cap = _pct_tile("at the cap", "n/a", pc.note)
    elif pc.cap == 0:
        at_cap = _pct_tile("at the cap", "uncapped", pc.note)
    else:
        share = 100.0 * (pc.n_at_cap or 0) / max(1, pc.n_spectra)
        at_cap = _pct_tile(
            f"at cap {pc.cap:,}",
            f"{share:.1f}%",
            pc.note,
            warn=share >= 10.0,
        )
    tiles = html.Div(
        [
            _pct_tile("p5", f"{p['p5']:,}", "5th percentile of the peaks per MS2 spectrum"),
            _pct_tile("p25", f"{p['p25']:,}", "25th percentile"),
            _pct_tile("median", f"{p['p50']:,}", "50th percentile (the median)"),
            _pct_tile("p95", f"{p['p95']:,}", "95th percentile"),
            _pct_tile("max", f"{p['max']:,}", "The largest peak list"),
            at_cap,
        ],
        className="qc-minis",
    )
    extra = []
    if pc.n_empty:
        extra.append(f"{pc.n_empty:,} spectra have no peak.")
    return [
        tiles,
        graph(PEAKS_GRAPH, qf.peaks_figure(pc, scheme)),
        html.Div(
            " ".join([pc.note, *extra]),
            className="qc-note",
        ),
    ]


def peaks_card(body: list[Any], *, n_spectra: int | None, label: str | None = None) -> Any:
    return section(
        "Peaks per MS2 spectrum",
        html.Div(body, id="qc-peaks-body"),
        count=f"{n_spectra:,} spectra" if n_spectra is not None else None,
        subtitle="The --top-peaks-ms2 saturation check (docs/20)",
        help=(
            (label + ". " if label else "")
            + "Percentiles use numpy's 'nearest' method, so each value is the count of a "
            "spectrum. convert keeps the N most intense peaks of a spectrum with more than "
            "N (--top-peaks-ms2 N; 0 is uncapped), so a capped spectrum holds exactly N "
            "peaks and the share at the cap is the share of truncated spectra."
        ),
        id="qc-peaks-card",
    )


def peaks_pending(n_ms2: int | None, size: int | None) -> list[Any]:
    """The progress state of the one-time pass over the MS2 peak lists."""
    what = f"{n_ms2:,} MS2 spectra" if n_ms2 is not None else "the MS2 spectra"
    where = f" ({size / 1e9:.1f} GB)" if size else ""
    return [
        dmc.Stack(
            [
                dmc.Loader(type="bars", size="sm"),
                dmc.Text(f"Reading the peak lists of {what}{where} once.", size="sm"),
                dmc.Text(
                    "The counts, the TIC and the base peak of every scan are then cached "
                    "outside the run directory, so later visits are immediate.",
                    size="xs",
                    c="dimmed",
                    ta="center",
                    maw=360,
                ),
            ],
            align="center",
            gap=6,
            py="xl",
            className="qc-pending",
        )
    ]


# --------------------------------------------------------------------------- distributions


def _share(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total else ""


def mods_table(d: IdDistributions) -> Any:
    """Modifications of the accepted PSMs: PSMs with the modification, and its share of
    the PSMs whose sequence holds the residue (in-cell bars on fixed scales)."""
    rows = qf.mods_rows(d)
    if d.n == 0:
        return empty("No accepted identification.")
    head = html.Tr(
        [
            html.Th("modification"),
            html.Th("site"),
            html.Th(
                tip(html.Span("PSMs"), "Accepted PSMs with at least one such site"),
                className="qc-num",
            ),
            html.Th(
                tip(
                    html.Span("share"),
                    "PSMs with this modification over the PSMs whose sequence holds the "
                    "residue (all PSMs for a terminus); the bar runs from 0 to 100%",
                ),
                className="qc-num",
            ),
        ]
    )
    body = []
    for r in rows:
        share = r["psms"] / r["with_site"] if r["with_site"] else 0.0
        holds = f" with {r['site']}" if r["site"] not in ("N-term", "C-term") else ""
        body.append(
            html.Tr(
                [
                    html.Td(
                        tip(
                            html.Span(
                                [
                                    html.Span(
                                        r["short"],
                                        className="qc-mod-tag",
                                        style={"background": r["colour"]},
                                    ),
                                    html.Span(r["name"], className="qc-mod-name"),
                                ],
                                className="qc-mod",
                            ),
                            f"{r['tag']} as written in the peptidoform",
                        )
                    ),
                    html.Td(r["site"], className="qc-mono"),
                    html.Td(
                        tip(
                            html.Span(f"{r['psms']:,}"),
                            f"{r['psms']:,} PSMs carry it; {r['sites']:,} modified sites in all",
                        ),
                        className="qc-num",
                    ),
                    html.Td(
                        tip(
                            _bar_cell(share, 1.0, f"{100 * share:.1f}%", colour="teal"),
                            f"{r['psms']:,} of {r['with_site']:,} PSMs{holds}",
                        ),
                        className="qc-num",
                    ),
                ]
            )
        )
    body.append(
        html.Tr(
            [
                html.Td(html.Span("unmodified", className="qc-mod-name qc-dim")),
                html.Td(""),
                html.Td(f"{d.n_unmodified:,}", className="qc-num"),
                html.Td(
                    _bar_cell(
                        d.n_unmodified / d.n if d.n else 0.0,
                        1.0,
                        _share(d.n_unmodified, d.n),
                        colour="gray",
                    ),
                    className="qc-num",
                ),
            ]
        )
    )
    return html.Table([html.Thead(head), html.Tbody(body)], className="qc-table")


def _bar_cell(value: float, top: float, text: str, colour: str = "green") -> Any:
    frac = 0.0 if not top else max(0.0, min(1.0, float(value) / float(top)))
    return html.Div(
        [
            html.Div(
                html.Div(
                    className="qc-bar-fill",
                    style={
                        "width": f"{100 * frac:.1f}%",
                        "background": f"var(--mantine-color-{colour}-6)",
                    },
                ),
                className="qc-bar",
            ),
            html.Span(text, className="qc-bar-text"),
        ],
        className="qc-bar-cell",
    )


def dist_cards(d: IdDistributions | None, scheme: str, error: str | None = None) -> Any:
    if d is None:
        return section("Accepted identifications", empty(error or "Not available."))
    total = d.n
    common = f"{d.label}."
    cards = [
        section(
            "Charge",
            graph(
                "qc-charge",
                qf.count_bars(
                    d.charge,
                    "charge",
                    scheme=scheme,
                    xtitle="precursor charge",
                    noun="charge",
                    total=total,
                    tick_suffix="+",
                ),
            ),
            count=total,
            help=f"{common} psms_scored.charge of each accepted PSM.",
        ),
        section(
            "Peptide length",
            graph(
                "qc-length",
                qf.count_bars(
                    d.length,
                    "length",
                    scheme=scheme,
                    xtitle="residues",
                    noun="length",
                    total=total,
                ),
            ),
            count=total,
            help=f"{common} {d.notes[0]}",
            right=derived("residues of the peptidoform without its modification tags"),
        ),
        section(
            "Missed cleavages",
            graph(
                "qc-missed",
                qf.count_bars(
                    d.missed,
                    "missed_cleavages",
                    scheme=scheme,
                    xtitle="missed cleavages",
                    noun="missed cleavages",
                    total=total,
                ),
            ),
            count=total,
            help=f"{common} " + " ".join(d.notes[1:]),
            right=derived(MISSED_CLEAVAGE_RULE),
        ),
        section(
            "Modifications",
            mods_table(d),
            count=total,
            help=(
                f"{common} Tags as written in psms_scored.peptidoform: a tag after a residue "
                "modifies it; a tag before a '-' at the start is N-terminal, after a '-' at "
                "the end C-terminal. PSMs: rows with at least one such site."
            ),
            right=derived("modification tags counted from the peptidoform text"),
        ),
    ]
    # Charge and missed cleavages have a few bars; length and the modification table need
    # the room.
    spans = ({"xl": 2}, {"xl": 3}, {"xl": 3}, {"xl": 4})
    return dmc.Grid(
        [
            dmc.GridCol(card, span={"base": 12, "md": 6, **extra})
            for card, extra in zip(cards, spans, strict=True)
        ],
        gutter="lg",
    )


def footer(timings: Mapping[str, float], notes: Sequence[str]) -> Any:
    parts = [f"Page built in {timings.get('page', 0.0):,.0f} ms"]
    for key, label in (
        ("ms1", "MS1 scan signals"),
        ("ms2", "MS2 scan signals"),
        ("ids", "identifications"),
        ("dist", "distributions"),
    ):
        if key in timings:
            parts.append(f"{label} {timings[key]:,.0f} ms")
    text = "; ".join(parts) + "."
    return dmc.Group(
        [icon("clock", 13), dmc.Text(text, size="xs", c="dimmed", id="qc-timings")]
        + [dmc.Text(n, size="xs", c="dimmed") for n in notes],
        gap=6,
        className="qc-footer",
    )
