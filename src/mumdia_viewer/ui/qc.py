"""Run QC page (P1 view 6): the signal, the acquisition and the identifications of one run.

Address: ``qc`` for a single run, ``qc?run=<name>`` in an experiment (the first run when
none is given). Each run of an experiment has its own page; the identification views
count the run on ``run_psm_q``.

The page follows PeptideShaker's linked panels. The RT tracks (TIC and base peak of the
chosen MS level, the accepted identifications per RT bin, the MS2 scan rate) share one
RT axis. A click on a scan shows its spectrum; a click on a bar lists the
identifications of that bin, each linked to its precursor page. Both panels start with
a selection (the scan with the largest TIC, the bin with the most identifications), so
no panel is empty. Below: the isolation windows (a click shows one window's MS2 TIC),
the peaks per MS2 spectrum (the ``--top-peaks-ms2`` saturation check) and the charge,
length, missed-cleavage and modification distributions.

The MS2 scan signals need one pass over every MS2 peak list (about a second for an
Astral run on a local SSD). The page never waits for it: until the viewer cache holds
them, the peaks card shows a progress state and fills itself, and the MS2 option of the
tracks computes them on first use. The MS2 tracks draw an M4 envelope of the scans in
view and redraw at full resolution when 12,000 or fewer scans are in view.
"""

from __future__ import annotations

import time
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import ClientsideFunction, Input, Output, Patch, State, dcc, html, no_update
from dash import ctx as dash_ctx

from mumdia_viewer.data import ResultSet, Run, ViewerError
from mumdia_viewer.data import qc as qcd
from mumdia_viewer.data.spectra import Ms1Table, ScanTable, Spectrum

from . import qc_cards as cards
from . import qc_figures as qf
from .state import PageContext, parse_threshold, stop_label
from .widgets import empty, section

BIN_LIMIT = 5000
# Views with more scans are drawn as an M4 envelope (data.qc.envelope_rows).
ENVELOPE_ABOVE = 12_000
ENVELOPE_BINS = 2_000
RT_FIG = {"type": "fig", "name": cards.RT_GRAPH}
WIN_FIG = {"type": "fig", "name": cards.WINDOWS_GRAPH}


# --------------------------------------------------------------------------- helpers


def _scheme(value: Any) -> str:
    return "dark" if value == "dark" else "light"


def run_of(rs: ResultSet, query: dict[str, str]) -> tuple[Run, str | None]:
    """The run of the page: ``run`` of the address, else the first run."""
    wanted = (query or {}).get("run")
    if not rs.is_experiment:
        return rs.runs[0], None
    if wanted:
        for r in rs.runs:
            if r.name == wanted:
                return r, None
        return rs.runs[0], f"No run {wanted!r}; showing {rs.runs[0].label}."
    return rs.runs[0], None


def _memo(rs: ResultSet, key: tuple, make: Any) -> Any:
    hit = rs._memo.get(key)
    if hit is None:
        hit = make()
        rs._memo[key] = hit
    return hit


def edges_of(rs: ResultSet, run: Run) -> np.ndarray:
    return _memo(rs, ("qc.page.edges", run.index), lambda: qcd.rt_edges(rs, run))


def rate_of(rs: ResultSet, run: Run) -> pd.DataFrame | None:
    def make() -> pd.DataFrame | None:
        try:
            return qcd.scan_rate(rs, run, edges_of(rs, run))
        except (ViewerError, OSError):
            return None

    return _memo(rs, ("qc.page.rate", run.index), make)


def scheme_of(rs: ResultSet, run: Run) -> qcd.WindowScheme | None:
    def make() -> qcd.WindowScheme | None:
        try:
            return qcd.window_scheme(rs, run)
        except (ViewerError, OSError):
            return None

    return _memo(rs, ("qc.page.windows", run.index), make)


def bin_label(edges: np.ndarray) -> str:
    w = float(edges[1] - edges[0]) if edges.size > 1 else 0.0
    return f"PSMs / {w:g} s"


def t_label(t: float) -> str:
    return f"run_psm_q ≤ {stop_label(t)}"


def _window_value(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "", "all") else None
    except (TypeError, ValueError):
        return None


def _window_name(ws: qcd.WindowScheme | None, window: int | None) -> str | None:
    if window is None:
        return None
    if ws is not None:
        hit = ws.frame[ws.frame["window_id"] == window]
        if len(hit):
            r = hit.iloc[0]
            return f"window {window}, {r['lower']:.2f} to {r['upper']:.2f} m/z"
    return f"window {window}"


def series(
    rs: ResultSet,
    run: Run,
    level: int,
    window: int | None,
    x_range: list[float] | None,
) -> tuple[qf.SignalTrace | None, qf.SignalTrace | None, int, bool]:
    """The TIC and base-peak traces of a view: every scan, or the M4 envelope."""
    sig = qcd.scan_signals(rs, run, level)
    if level == 2 and window is not None:
        # One window has a scan per cycle (about a thousand): every scan is drawn.
        rows = ScanTable.for_run(rs, run).rows_in_window(window)
        n_view = int(rows.size)
        picks = (rows, rows)
        envelope = False
    else:
        lo, hi = (
            (float(x_range[0]), float(x_range[1]))
            if x_range is not None
            else (float(np.min(sig.rt)) if sig.n else 0.0, float(np.max(sig.rt)) if sig.n else 0.0)
        )
        order = sig.order()
        rt_sorted = sig.rt[order]
        n_view = int(
            np.searchsorted(rt_sorted, hi, side="right") - np.searchsorted(rt_sorted, lo, "left")
        )
        picks = (
            qcd.envelope_rows(
                sig.rt,
                sig.tic,
                lo,
                hi,
                order=order,
                max_points=ENVELOPE_ABOVE,
                n_bins=ENVELOPE_BINS,
            ),
            qcd.envelope_rows(
                sig.rt,
                sig.base_peak,
                lo,
                hi,
                order=order,
                max_points=ENVELOPE_ABOVE,
                n_bins=ENVELOPE_BINS,
            ),
        )
        envelope = picks[0].size < n_view
    name = _window_name(scheme_of(rs, run), window)
    out = []
    for rows in picks:
        rows = np.asarray(rows, dtype=np.int64)
        out.append(
            qf.SignalTrace(
                level=level,
                rows=rows,
                rt=sig.rt[rows],
                tic=sig.tic[rows],
                base_peak=sig.base_peak[rows],
                base_peak_mz=sig.base_peak_mz[rows],
                scan_index=sig.scan_index[rows],
                n_total=n_view,
                envelope=envelope,
                window=name,
            )
        )
    return out[0], out[1], n_view, envelope


def spectrum_of(rs: ResultSet, run: Run, level: int, row: int) -> Spectrum:
    if level == 1:
        return Ms1Table.for_run(rs, run).spectrum(int(row))
    return ScanTable.for_run(rs, run).spectrum(int(row))


def default_scan(rs: ResultSet, run: Run, level: int) -> dict[str, int] | None:
    """The scan with the largest TIC (the panel's first selection)."""
    try:
        sig = qcd.scan_signals(rs, run, level)
    except (ViewerError, OSError):
        return None
    if sig.n == 0:
        return None
    return {"level": level, "row": int(np.argmax(sig.tic))}


def bin_of(ids: pd.DataFrame | None, index: int | None) -> dict[str, Any] | None:
    if ids is None or index is None or not 0 <= index < len(ids):
        return None
    return {
        "index": int(index),
        "lo": float(ids["bin_lo"].iloc[index]),
        "hi": float(ids["bin_hi"].iloc[index]),
        "closed": bool(index == len(ids) - 1),
    }


def default_bin(ids: pd.DataFrame | None) -> dict[str, Any] | None:
    if ids is None or not len(ids) or int(ids["targets"].max()) == 0:
        return None
    return bin_of(ids, int(np.argmax(ids["targets"].to_numpy())))


def bin_rows(
    rs: ResultSet, run: Run, t: float, sel: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], int]:
    if not sel:
        return [], 0
    df = qcd.ids_in_rt_range(
        rs, run, t, sel["lo"], sel["hi"], closed=bool(sel.get("closed")), limit=BIN_LIMIT
    )
    return cards.bin_records(df), int(df.attrs.get("total", len(df)))


def scan_rt(rs: ResultSet, run: Run, scan: dict[str, Any] | None) -> float | None:
    if not scan:
        return None
    try:
        sig = qcd.scan_signals(rs, run, int(scan["level"]))
        return float(sig.rt[int(scan["row"])])
    except (ViewerError, OSError, IndexError, KeyError, ValueError):
        return None


def scan_parts(
    rs: ResultSet, run: Run, scan: dict[str, Any] | None, scheme: str, note: str = ""
) -> tuple[Any, list[Any]]:
    if not scan:
        return cards.scan_title(None), cards.scan_body(None, tic=None, scheme=scheme)
    level, row = int(scan["level"]), int(scan["row"])
    try:
        spec = spectrum_of(rs, run, level, row)
        sig = qcd.scan_signals(rs, run, level)
        tic = float(sig.tic[row])
    except (ViewerError, OSError, IndexError) as exc:
        return cards.scan_title(None), cards.scan_body(
            None, tic=None, scheme=scheme, error=str(exc)
        )
    return cards.scan_title(spec), cards.scan_body(spec, tic=tic, scheme=scheme, step_note=note)


def ids_frame(rs: ResultSet, run: Run, t: float) -> pd.DataFrame | None:
    try:
        return qcd.ids_across_rt(rs, run, t, edges_of(rs, run))
    except (ViewerError, ValueError):
        return None


def rt_figure_for(
    rs: ResultSet,
    run: Run,
    *,
    level: int,
    window: int | None,
    x_range: list[float] | None,
    t: float,
    scan: dict[str, Any] | None,
    sel: dict[str, Any] | None,
    scheme: str,
) -> tuple[Any, str]:
    """The RT figure of a view and the note under it."""
    edges = edges_of(rs, run)
    ids = ids_frame(rs, run, t)
    message = None
    tic = bp = None
    note = ""
    try:
        tic, bp, n_view, envelope = series(rs, run, level, window, x_range)
        note = cards.rt_note(level, n_view, int(tic.rows.size), envelope, tic.window)
    except (ViewerError, OSError) as exc:
        message = f"No MS{level} signal: {exc}"
        note = message
    fig = qf.rt_figure(
        tic,
        bp,
        ids,
        rate_of(rs, run),
        t_label=t_label(t),
        bin_label=bin_label(edges),
        scheme=scheme,
        scan_rt=scan_rt(rs, run, scan),
        band=(sel["lo"], sel["hi"]) if sel else None,
        selected_bin=sel["index"] if sel else None,
        x_range=x_range,
        message=message,
    )
    return fig, note


def _signal_patch(patch: Patch, tic: qf.SignalTrace, bp: qf.SignalTrace) -> None:
    for index, s, values, with_mz in (
        (qf.TRACE["tic"], tic, tic.tic, False),
        (qf.TRACE["bp"], bp, bp.base_peak, True),
    ):
        patch["data"][index]["x"] = qf.typed(s.rt)
        patch["data"][index]["y"] = qf.typed(values)
        patch["data"][index]["customdata"] = qf.signal_customdata(s, with_mz=with_mz)


def _ids_patch(patch: Patch, ids: pd.DataFrame, t: float, selected: int | None) -> None:
    targets, decoys = qf._id_traces(ids, t_label(t), selected)
    patch["data"][qf.TRACE["targets"]]["y"] = ids["targets"].astype(int).tolist()
    patch["data"][qf.TRACE["targets"]]["customdata"] = qf.plain(qf.bin_customdata(ids))
    patch["data"][qf.TRACE["targets"]]["hovertemplate"] = targets.hovertemplate
    patch["data"][qf.TRACE["targets"]]["marker"]["opacity"] = qf.bin_opacity(len(ids), selected)
    patch["data"][qf.TRACE["decoys"]]["y"] = ids["decoys"].astype(int).tolist()
    patch["data"][qf.TRACE["decoys"]]["hovertemplate"] = decoys.hovertemplate


# --------------------------------------------------------------------------- layout


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs, t, scheme = ctx.rs, ctx.threshold, ctx.scheme
    run, notice = run_of(rs, ctx.query)
    timings: dict[str, float] = {}

    acq = qcd.acquisition(rs, run)
    has_ms1, has_ms2 = acq.n_ms1 is not None, acq.n_ms2 is not None
    level = 1 if has_ms1 or not has_ms2 else 2
    tick = time.perf_counter()
    scan = default_scan(rs, run, 1) if has_ms1 else None
    timings["ms1"] = (time.perf_counter() - tick) * 1000.0

    tick = time.perf_counter()
    ids = ids_frame(rs, run, t)
    sel = default_bin(ids)
    population = qcd.population_label(rs, run, t)
    try:
        counts = qcd.counts_by_run(rs, t)
    except ViewerError:
        counts = {}
    n_accepted = ids.attrs.get("total") if ids is not None else counts.get(run.label)
    timings["ids"] = (time.perf_counter() - tick) * 1000.0

    ms2_ready = has_ms2 and qcd.signals_cached(rs, run, 2)
    if level == 2 and not ms2_ready:
        rt_fig = qf.rt_figure(
            None,
            None,
            ids,
            rate_of(rs, run),
            t_label=t_label(t),
            bin_label=bin_label(edges_of(rs, run)),
            scheme=scheme,
            message="This run has no MS1 table. Pick MS2 to read the MS2 scans once.",
        )
        note = ""
    else:
        rt_fig, note = rt_figure_for(
            rs,
            run,
            level=level,
            window=None,
            x_range=None,
            t=t,
            scan=scan,
            sel=sel,
            scheme=scheme,
        )
    ws = scheme_of(rs, run)
    rt = cards.rt_card(
        rt_fig,
        level=level,
        window="all",
        ws=ws,
        has_ms2=has_ms2,
        note=note,
        population=population,
    )
    title, body = scan_parts(rs, run, scan, scheme)
    scan_card = cards.scan_card(title, body)

    try:
        rows, total = bin_rows(rs, run, t, sel)
    except ViewerError:
        rows, total = [], 0
    bin_card = cards.bin_card(
        base=ctx.base,
        run_name=run.name,
        lo=sel["lo"] if sel else None,
        hi=sel["hi"] if sel else None,
        total=total,
        rows=rows,
        threshold=t,
        population=population,
        limit=BIN_LIMIT,
    )

    win_card = cards.windows_card(
        ws,
        qf.windows_figure(ws, scheme) if ws is not None else None,
        error=next(iter(acq.notes), None),
    )
    if not has_ms2:
        peaks = cards.peaks_card(
            [empty(next(iter(acq.notes), "This run has no MS2 table."))], n_spectra=None
        )
        need = None
    elif ms2_ready:
        tick = time.perf_counter()
        pc = qcd.peak_counts(rs, run)
        timings["ms2"] = (time.perf_counter() - tick) * 1000.0
        peaks = cards.peaks_card(
            cards.peaks_body(pc, scheme), n_spectra=pc.n_spectra, label=pc.label
        )
        need = None
    else:
        art = run.artifact("spectra_ms2")
        size = art.path.stat().st_size if art is not None and art.path is not None else None
        peaks = cards.peaks_card(cards.peaks_pending(acq.n_ms2, size), n_spectra=acq.n_ms2)
        need = {"run": run.name}

    tick = time.perf_counter()
    try:
        dist = cards.dist_cards(qcd.id_distributions(rs, run, t), scheme)
    except (ViewerError, ValueError) as exc:
        dist = cards.dist_cards(None, scheme, error=str(exc))
    timings["dist"] = (time.perf_counter() - tick) * 1000.0

    timings["page"] = (time.perf_counter() - t0) * 1000.0
    notes = [notice] if notice else []
    stores = [
        dcc.Store(id="qc-key", data={"run": run.name, "base": ctx.base}),
        dcc.Store(id="qc-view", data={"level": level, "window": None}),
        dcc.Store(id="qc-xrange", data=None),
        dcc.Store(id="qc-scan", data=scan),
        dcc.Store(id="qc-bin", data=sel),
        dcc.Store(id="qc-need", data=need),
    ]
    return dmc.Stack(
        [
            cards.header(ctx, run.name, run.label, acq, n_accepted, counts),
            *([dmc.Alert(notice, color="yellow", variant="light", p="xs")] if notice else []),
            dmc.Grid(
                [
                    dmc.GridCol(rt, span={"base": 12, "lg": 8}),
                    dmc.GridCol(
                        dmc.Stack([scan_card, bin_card], gap="lg"),
                        span={"base": 12, "lg": 4},
                    ),
                ],
                gutter="lg",
            ),
            dmc.Grid(
                [
                    dmc.GridCol(win_card, span={"base": 12, "lg": 6}),
                    dmc.GridCol(html.Div(peaks, id="qc-peaks-slot"), span={"base": 12, "lg": 6}),
                ],
                gutter="lg",
            ),
            html.Div(dist, id="qc-dist"),
            cards.footer(timings, notes),
            *stores,
        ],
        gap="lg",
        className="qc-page",
    )


# --------------------------------------------------------------------------- callbacks


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""

    app.clientside_callback(
        ClientsideFunction("mvqc", "run"),
        Output("url", "href", allow_duplicate=True),
        Input("qc-run", "value"),
        State("qc-key", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvqc", "xrange"),
        Output("qc-xrange", "data"),
        Input(RT_FIG, "relayoutData"),
        State("qc-xrange", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvqc", "pickWindow"),
        Output("qc-level", "value"),
        Output("qc-window", "value"),
        Input(WIN_FIG, "clickData"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output(RT_FIG, "figure", allow_duplicate=True),
        Output("qc-view", "data"),
        Output("qc-rt-note", "children"),
        Output("qc-window-box", "style"),
        Output(WIN_FIG, "figure", allow_duplicate=True),
        Input("qc-level", "value"),
        Input("qc-window", "value"),
        State("qc-key", "data"),
        State("qc-xrange", "data"),
        State("qc-scan", "data"),
        State("qc-bin", "data"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _view(level_value, window_value, key, xr, scan, sel, t, scheme):
        if not key:
            return (no_update,) * 5
        rs = get_rs()
        try:
            run = rs.run(key["run"]) if rs.is_experiment else rs.runs[0]
        except KeyError:
            return (no_update,) * 5
        level = 2 if str(level_value) == "2" else 1
        window = _window_value(window_value) if level == 2 else None
        x_range = (xr or {}).get("range")
        fig, note = rt_figure_for(
            rs,
            run,
            level=level,
            window=window,
            x_range=x_range,
            t=parse_threshold(t),
            scan=scan,
            sel=sel,
            scheme=_scheme(scheme),
        )
        ws = scheme_of(rs, run)
        windows = (
            qf.windows_figure(ws, _scheme(scheme), selected=window) if ws is not None else no_update
        )
        return fig, {"level": level, "window": window}, note, cards.window_style(level), windows

    @app.callback(
        Output(RT_FIG, "figure", allow_duplicate=True),
        Output("qc-rt-note", "children", allow_duplicate=True),
        Input("qc-xrange", "data"),
        State("qc-key", "data"),
        State("qc-view", "data"),
        prevent_initial_call=True,
    )
    def _zoom(xr, key, view):
        if not key or not view:
            return no_update, no_update
        rs = get_rs()
        try:
            run = rs.run(key["run"]) if rs.is_experiment else rs.runs[0]
            level = int(view.get("level") or 1)
            window = view.get("window")
            sig = qcd.scan_signals(rs, run, level)
        except (KeyError, ViewerError, OSError):
            return no_update, no_update
        x_range = (xr or {}).get("range")
        if window is not None or sig.n <= ENVELOPE_ABOVE:
            return no_update, no_update  # every scan is drawn already
        tic, bp, n_view, envelope = series(rs, run, level, window, x_range)
        patch = Patch()
        _signal_patch(patch, tic, bp)
        return patch, cards.rt_note(level, n_view, int(tic.rows.size), envelope, tic.window)

    @app.callback(
        Output("qc-scan", "data"),
        Output("qc-scan-title", "children"),
        Output("qc-scan-body", "children"),
        Output("qc-bin", "data"),
        Output("qc-bin-title", "children"),
        Output("qc-bin-count", "children"),
        Output("qc-bin-more", "children"),
        Output(cards.GRID_ID, "rowData"),
        Output(RT_FIG, "figure", allow_duplicate=True),
        Input(RT_FIG, "clickData"),
        Input("scan-prev", "n_clicks"),
        Input("scan-next", "n_clicks"),
        State("qc-key", "data"),
        State("qc-view", "data"),
        State("qc-scan", "data"),
        State("qc-bin", "data"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _select(click, _prev, _next, key, view, scan, sel, t, scheme):
        nothing = (no_update,) * 9
        if not key:
            return nothing
        rs = get_rs()
        try:
            run = rs.run(key["run"]) if rs.is_experiment else rs.runs[0]
        except KeyError:
            return nothing
        threshold = parse_threshold(t)
        trigger = dash_ctx.triggered_id
        out = list(nothing)
        new_scan = None
        note = ""
        if trigger in ("scan-prev", "scan-next"):
            if not scan:
                return nothing
            step = -1 if trigger == "scan-prev" else 1
            level, row = int(scan["level"]), int(scan["row"])
            if level == 1:
                n = Ms1Table.for_run(rs, run).n
                target = row + step
                if not 0 <= target < n:
                    return nothing
                new_scan = {"level": 1, "row": target}
            else:
                pick = ScanTable.for_run(rs, run).step(row, step)
                if pick is None:
                    return nothing
                new_scan = {"level": 2, "row": int(pick.row)}
                note = "Stepped within the isolation window of the scan."
        elif isinstance(click, dict) and click.get("points"):
            point = click["points"][0]
            curve = int(point.get("curveNumber", -1))
            if curve in (qf.TRACE["tic"], qf.TRACE["bp"]):
                custom = point.get("customdata")
                if not custom:
                    return nothing
                level = int((view or {}).get("level") or 1)
                new_scan = {"level": level, "row": int(custom[0])}
            elif curve in (qf.TRACE["targets"], qf.TRACE["decoys"], qf.TRACE["rate"]):
                ids = ids_frame(rs, run, threshold)
                if ids is None:
                    return nothing
                x = float(point.get("x"))
                edges = edges_of(rs, run)
                index = int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(ids) - 1))
                new_sel = bin_of(ids, index)
                rows, total = bin_rows(rs, run, threshold, new_sel)
                out[3] = new_sel
                out[4] = cards.bin_title(new_sel["lo"], new_sel["hi"])
                out[5] = f"{total:,}"
                out[6] = (
                    f"The first {BIN_LIMIT:,} by apex RT are listed." if total > BIN_LIMIT else ""
                )
                out[7] = rows
                sel = new_sel
                patch = Patch()
                patch["layout"]["shapes"] = qf.selection_shapes(
                    scan_rt(rs, run, scan), (sel["lo"], sel["hi"])
                )
                patch["data"][qf.TRACE["targets"]]["marker"]["opacity"] = qf.bin_opacity(
                    len(ids), sel["index"]
                )
                out[8] = patch
                return tuple(out)
            else:
                return nothing
        else:
            return nothing
        if new_scan is not None:
            title, body = scan_parts(rs, run, new_scan, _scheme(scheme), note)
            out[0], out[1], out[2] = new_scan, title, body
            patch = Patch()
            patch["layout"]["shapes"] = qf.selection_shapes(
                scan_rt(rs, run, new_scan), (sel["lo"], sel["hi"]) if sel else None
            )
            out[8] = patch
        return tuple(out)

    @app.callback(
        Output(RT_FIG, "figure", allow_duplicate=True),
        Output(cards.GRID_ID, "rowData", allow_duplicate=True),
        Output(cards.GRID_ID, "columnDefs"),
        Output("qc-bin-count", "children", allow_duplicate=True),
        Output("qc-bin-more", "children", allow_duplicate=True),
        Output("qc-dist", "children"),
        Output("qc-fact-accepted", "children"),
        Output("qc-fact-accepted-q", "children"),
        Input("threshold", "data"),
        State("qc-key", "data"),
        State("qc-bin", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _threshold(t, key, sel, scheme):
        if not key:
            return (no_update,) * 8
        rs = get_rs()
        try:
            run = rs.run(key["run"]) if rs.is_experiment else rs.runs[0]
        except KeyError:
            return (no_update,) * 8
        threshold = parse_threshold(t)
        ids = ids_frame(rs, run, threshold)
        patch = Patch()
        if ids is not None:
            _ids_patch(patch, ids, threshold, sel["index"] if sel else None)
        try:
            rows, total = bin_rows(rs, run, threshold, sel)
        except ViewerError:
            rows, total = [], 0
        try:
            dist = cards.dist_cards(qcd.id_distributions(rs, run, threshold), _scheme(scheme))
        except (ViewerError, ValueError) as exc:
            dist = cards.dist_cards(None, _scheme(scheme), error=str(exc))
        n = ids.attrs.get("total") if ids is not None else None
        return (
            patch,
            rows,
            cards.bin_columns(threshold),
            f"{total:,}",
            f"The first {BIN_LIMIT:,} by apex RT are listed." if total > BIN_LIMIT else "",
            dist,
            f"{n:,}" if n is not None else "n/a",
            cards.threshold_text(threshold),
        )

    @app.callback(
        Output("qc-peaks-slot", "children"),
        Input("qc-need", "data"),
        State("scheme", "data"),
    )
    def _peaks(need, scheme):
        if not need:
            return no_update
        rs = get_rs()
        try:
            run = rs.run(need["run"]) if rs.is_experiment else rs.runs[0]
            pc = qcd.peak_counts(rs, run)
        except (KeyError, ViewerError, OSError) as exc:
            return section("Peaks per MS2 spectrum", empty(str(exc)))
        return cards.peaks_card(
            cards.peaks_body(pc, _scheme(scheme)), n_spectra=pc.n_spectra, label=pc.label
        )
