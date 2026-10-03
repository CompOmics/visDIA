"""The spectrum browser (P2 view 11 of the spec): any scan of a run, by ``scan_index`` or
by retention time, with its isolation window and the accepted identifications whose apex
is near it.

Address: ``spectra?run=<name>&scan=<scan_index>`` (or ``rt=<s>`` with optional
``level=1|2`` and ``window=<window_id>``), and ``cid=<candidate_id>`` for the selected
candidate. The page rewrites ``scan`` and ``cid`` into the address as the user steps, so
a link restores the view.

Layout: the scan header (key values of the shown scan), the navigation card (scan and RT
boxes, MS level, isolation window, step scope, previous and next, and the RT strip with
its slider), the spectrum with the selected candidate's overlay, the list of nearby
identifications, the candidate's fragment ladder and the scan's details. Data come from
:mod:`mumdia_viewer.data.scans`; see its docstring for the rules.

Callbacks: ``navigate`` turns any navigation input into the shown scan (``sp-scan``);
``show`` builds everything that depends on the scan, the threshold, the list settings
and the selected candidate (``sp-cid``). The arrow keys and the step buttons reach
``navigate`` through ``sp-step`` (spectra.js), so no other page's click handler can
swallow them.
"""

from __future__ import annotations

import math
import time
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import ClientsideFunction, Input, Output, State, dcc, html, no_update
from dash import callback_context as dash_ctx

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data import qc as qcd
from mumdia_viewer.data import scans as sc
from mumdia_viewer.data.detail import MirrorData
from mumdia_viewer.data.discovery import Run
from mumdia_viewer.data.errors import ViewerError
from mumdia_viewer.data.spectra import Ms1Table, ScanTable, Spectrum

from . import spectra_cards as cards
from . import spectra_figures as sf
from .detail_ions import ladder_caption, ladder_table, sequence_diagram
from .detail_view import fragments_of, ion_ladder, score_range
from .state import PageContext, parse_threshold
from .widgets import empty, section

SPEC_FIG = {"type": "fig", "name": cards.SPECTRUM_GRAPH}
NAV_FIG = {"type": "fig", "name": cards.NAV_GRAPH}
# Fragment matches are computed for at most this many listed candidates (the column says
# so beyond it).
MATCH_LIMIT = 80


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


def _int(value: Any) -> int | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) and f == int(f) else None


def _float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def start_of(
    rs: ResultSet, run: Run, query: dict[str, str], t: float
) -> tuple[sc.ScanRef, int | None, str | None]:
    """The scan (and candidate) the address asks for; a notice when it cannot be honoured."""
    cid = _int(query.get("cid"))
    if "scan" in query:
        si = _int(query.get("scan"))
        ref = sc.locate(rs, run, si) if si is not None else None
        if ref is not None:
            return ref, cid, None
        ref, best = sc.default_scan(rs, run, t)
        return ref, best, f"No scan with scan_index {query.get('scan')!r} in this run."
    rt = _float(query.get("rt"))
    if rt is not None:
        level = 1 if query.get("level") == "1" and run.has("spectra_ms1") else 2
        if level == 2 and not run.has("spectra_ms2"):
            level = 1
        window = _int(query.get("window"))
        notice = None
        if level == 2:
            st = ScanTable.for_run(rs, run)
            if window is not None and window not in set(st.windows["window_id"].tolist()):
                notice = f"No isolation window {window} in this run; showing the nearest MS2 scan."
                window = None
            if window is None and cid is not None:
                row = _candidate_row(rs, run, cid, t)
                if row is not None and pd.notna(row.get("precursor_mz")):
                    cover = st.covering_windows(float(row["precursor_mz"]))
                    window = int(cover[0]) if cover.size else None
        ref = sc.nearest_scan(rs, run, rt, level=level, window_id=window)
        if ref is not None:
            return ref, cid, notice
    ref, best = sc.default_scan(rs, run, t)
    return ref, cid if cid is not None else best, None


def _candidate_row(rs: ResultSet, run: Run, cid: int, t: float) -> dict[str, Any] | None:
    try:
        frame = sc.accepted_set(rs, run, t).frame
    except ViewerError:
        return None
    hit = frame[frame["candidate_id"] == int(cid)]
    return None if hit.empty else hit.iloc[0].to_dict()


def _spectrum(rs: ResultSet, run: Run, ref: sc.ScanRef) -> Spectrum:
    if ref.level == 1:
        return Ms1Table.for_run(rs, run).spectrum(ref.row)
    return ScanTable.for_run(rs, run).spectrum(ref.row)


def rt_max_of(rs: ResultSet, run: Run) -> float:
    def make() -> float:
        top = 0.0
        if run.has("spectra_ms2"):
            top = max(top, float(np.max(ScanTable.for_run(rs, run).rt)))
        if run.has("spectra_ms1"):
            m1 = Ms1Table.for_run(rs, run)
            if m1.n:
                top = max(top, float(np.max(m1.rt)))
        return math.ceil(top / 10.0) * 10.0 if top > 0 else 1.0

    return rs.memo(("spectra.page.rt_max", run.index, run.name), make)


def nav_edges(rt_max: float) -> np.ndarray:
    """RT bins of the strip: about 160 bins of a round width (1, 2, 5, 10, 20 or 30 s)."""
    raw = rt_max / 160.0
    width = next((w for w in (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0) if w >= raw), 120.0)
    n = max(1, math.ceil(rt_max / width))
    return np.arange(n + 1, dtype=np.float64) * width


def tic_trace(rs: ResultSet, run: Run, ref: sc.ScanRef) -> tuple[np.ndarray, np.ndarray] | None:
    """The TIC of the scans the strip covers, when run QC has computed it (never here)."""
    if not qcd.signals_cached(rs, run, ref.level):
        return None
    try:
        sig = rs.memo(
            ("spectra.page.signals", run.index, run.name, ref.level),
            lambda: qcd.scan_signals(rs, run, ref.level),
        )
    except (ViewerError, OSError):
        return None
    if ref.level == 1:
        order = sig.order()
        return sig.rt[order], sig.tic[order]
    rows = ScanTable.for_run(rs, run).rows_in_window(int(ref.window_id))
    return sig.rt[rows], sig.tic[rows]


def scan_store(ref: sc.ScanRef, window: int | None = None) -> dict[str, Any]:
    """The shown scan; ``window`` is the isolation window of an MS2 scan, or for an MS1 scan
    the window the window control holds (the MS2 window to return to)."""
    return {
        "level": ref.level,
        "row": ref.row,
        "rt": ref.rt,
        "scan_index": ref.scan_index,
        "window": ref.window_id if ref.window_id is not None else window,
    }


def ref_of(rs: ResultSet, run: Run, data: dict[str, Any] | None) -> sc.ScanRef | None:
    if not data:
        return None
    try:
        return sc.scan_ref(rs, run, int(data["level"]), int(data["row"]))
    except (KeyError, TypeError, ValueError, IndexError, ViewerError):
        return None


# --------------------------------------------------------------------------- the view


def view(
    rs: ResultSet,
    run: Run,
    ref: sc.ScanRef,
    *,
    t: float,
    near: str,
    decoys: bool,
    cid: int | None,
    n_labels: int,
    scale: str,
    scheme: str,
    base: str = "/",
    window_hint: int | None = None,
) -> dict[str, Any]:
    """Every part of the page that depends on the scan, the list settings and the candidate."""
    timings: dict[str, float] = {}
    tick = time.perf_counter()
    spec = _spectrum(rs, run, ref)
    timings["spectrum"] = (time.perf_counter() - tick) * 1000.0
    tic = float(np.sum(spec.intensity.astype(np.float64))) if spec.n_peaks else 0.0
    mode, delta = cards.parse_near(near)

    tick = time.perf_counter()
    try:
        df = sc.candidates_near(rs, run, ref, t, delta=delta, mode=mode, include_decoys=decoys)
        cand_error = None
    except ViewerError as exc:
        df, cand_error = None, str(exc)
    timings["candidates"] = (time.perf_counter() - tick) * 1000.0

    selected: dict[str, Any] | None = None
    if df is not None and len(df):
        ids = df["candidate_id"].tolist()
        chosen = cid if cid is not None and cid in ids else int(ids[0])
        selected = df[df["candidate_id"] == chosen].iloc[0].to_dict()
    elif cid is not None:
        selected = None
    sel_cid = int(selected["candidate_id"]) if selected is not None else None

    counts: dict[int, tuple[int, int]] = {}
    overlay = None
    isotopes = None
    frags: list[Any] = []
    tick = time.perf_counter()
    if df is not None and ref.level == 2 and len(df):
        try:
            counts = sc.matched_counts(rs, run, df["candidate_id"].head(MATCH_LIMIT), spec)
        except ViewerError:
            counts = {}
    if selected is not None:
        if ref.level == 2:
            try:
                overlay = sc.fragment_overlay(rs, run, sel_cid, spec)
            except ViewerError:
                overlay = None
            if overlay is not None:
                frags = fragments_of(overlay.chromatogram)
        else:
            isotopes = sc.isotope_overlay(
                rs, sel_cid, float(selected["precursor_mz"]), int(selected["charge"]), spec
            )
    timings["matches"] = (time.perf_counter() - tick) * 1000.0

    fig = sf.spectrum_figure(
        spec,
        scheme,
        n_labels=n_labels,
        overlay=overlay,
        frags=frags,
        isotopes=isotopes,
        scale=scale,
        precursor_mz=float(selected["precursor_mz"]) if selected is not None else None,
        candidate=sel_cid,
    )

    # The sequence diagram above the spectrum and the ladder (MS2), or the isotopes (MS1).
    seq: Any = None
    foot: Any = None
    if selected is None:
        frag_body: Any = cards.no_candidate(
            "No accepted identification is near this scan."
            if df is not None
            else (cand_error or "The candidate list is not available.")
        )
    elif ref.level == 2 and overlay is not None:
        pick = ScanTable.for_run(rs, run).pick(ref.row, reference_rt=float(selected["apex_rt"]))
        mdata = MirrorData(
            spectrum=spec,
            pick=pick,
            fragments=overlay.fragments,
            matches=overlay.matches,
            tolerance=overlay.tolerance,
            label=overlay.label,
            previous_row=None,
            next_row=None,
        )
        ladder = ion_ladder(str(selected["peptidoform"]), frags, mdata)
        seq = sequence_diagram(ladder, compact=True, legend=False)
        foot = cards.tolerance_chips(overlay)
        frag_body = [
            cards.candidate_line(selected, base, run.name),
            dmc.Text(ladder_caption(ladder), size="xs", c="dimmed", mb=6),
            html.Div(ladder_table(ladder), className="sp-ladder"),
        ]
    elif ref.level == 2:
        frag_body = cards.no_candidate("This candidate has no chromatogram rows to match.")
    else:
        frag_body = [
            cards.candidate_line(selected, base, run.name),
            cards.isotope_table(isotopes)
            if isotopes is not None
            else cards.no_candidate("No precursor m/z or charge for this candidate."),
        ]

    # The RT strip.
    rt_max = rt_max_of(rs, run)
    edges = nav_edges(rt_max)
    window = (ref.lower, ref.upper) if ref.level == 2 else None
    try:
        nav_counts = sc.apex_histogram(rs, run, t, edges, window=window)  # type: ignore[arg-type]
    except ViewerError:
        nav_counts = np.zeros(edges.size - 1, dtype=np.int64)
    band = (ref.rt - delta, ref.rt + delta) if mode == "apex" else None
    what = "accepted targets in this window" if ref.level == 2 else "accepted targets"
    tic_line = tic_trace(rs, run, ref)
    nav_fig = sf.nav_figure(
        edges,
        nav_counts,
        rt=ref.rt,
        band=band,
        rt_max=rt_max,
        tic=tic_line,
        what=what,
        scheme=scheme,
    )
    width = float(edges[1] - edges[0]) if edges.size > 1 else 0.0
    nav_note = (
        f"{int(nav_counts.sum()):,} {what} (run_psm_q ≤ {t:g}) in {width:g} s bins of apex_rt."
        + ("" if tic_line is not None else " TIC: open Run QC once to compute it.")
    )

    links = sc.ms1_links(rs, run, ref) if ref.level == 2 else {}
    ms2_here = None
    if ref.level == 1 and run.has("spectra_ms2"):
        w = window_hint
        ms2_here = sc.nearest_scan(rs, run, ref.rt, level=2, window_id=w) if w is not None else None

    rows = cards.candidate_records(df, sel_cid, counts) if df is not None else []
    note: Any = cards.candidate_note(df.attrs, len(df)) if df is not None else (cand_error or "")
    if df is not None and ref.level == 2 and len(df) > MATCH_LIMIT:
        note = [*note, html.Span(f" Fragments are matched for the first {MATCH_LIMIT}.")]
    lo, hi, _ = score_range(rs)
    return {
        "spec": spec,
        "ref": ref,
        "title": cards.scan_title(ref),
        "sub": cards.sub_line(spec, ref),
        "facts": cards.facts(spec, ref, tic),
        "fig": fig,
        "spec_title": cards.spectrum_title(ref, spec),
        "seq": seq,
        "foot": foot,
        "rows": rows,
        "columns": cards.candidate_columns(t, lo, hi),
        "count": f"{len(rows):,}",
        "note": note,
        "nav_fig": nav_fig,
        "nav_note": nav_note,
        "frag_title": cards.candidate_header(selected, ref),
        "frag_body": frag_body,
        "details": cards.details_body(spec, ref, links, ms2_here=ms2_here),
        "cid": sel_cid,
        "scroll": {"rowId": str(sel_cid), "rowPosition": "middle"} if sel_cid is not None else None,
        "timings": timings,
    }


# --------------------------------------------------------------------------- layout


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs, t, scheme = ctx.rs, ctx.threshold, ctx.scheme
    run, notice = run_of(rs, ctx.query)
    has_ms2, has_ms1 = run.has("spectra_ms2"), run.has("spectra_ms1")
    if not has_ms2 and not has_ms1:
        why = []
        for kind in ("spectra_ms2", "spectra_ms1"):
            art = run.artifact(kind)
            why.append(f"{kind}: {art.error if art is not None and art.error else 'missing'}")
        return section(
            "Spectrum browser",
            empty("This run's spectra tables cannot be read. " + "; ".join(why), "alert"),
        )
    try:
        tick = time.perf_counter()
        sc.accepted_set(rs, run, t)
        accepted_ms = (time.perf_counter() - tick) * 1000.0
    except ViewerError as exc:
        accepted_ms = 0.0
        notice = ((notice or "") + f" The accepted identifications cannot be read: {exc}").strip()
    ref, cid, where = start_of(rs, run, ctx.query, t)
    notice = " ".join(n for n in (notice, where) if n) or None
    windows = ScanTable.for_run(rs, run).windows if has_ms2 else None
    window_value = str(ref.window_id) if ref.window_id is not None else None
    if window_value is None and windows is not None and len(windows):
        # An MS1 start: the window control holds the window of the address, else the
        # window in the middle of the scheme.
        wanted = _int(ctx.query.get("window"))
        ids = windows["window_id"].tolist()
        window_value = str(wanted if wanted in ids else int(ids[len(ids) // 2]))
    v = view(
        rs,
        run,
        ref,
        t=t,
        near=cards.DEFAULT_NEAR,
        decoys=False,
        cid=cid,
        n_labels=10,
        scale="window",
        scheme=scheme,
        base=ctx.base,
        window_hint=_int(window_value),
    )
    nav = cards.nav_card(
        ref=ref,
        windows=windows,
        window_value=window_value,
        rt_max=rt_max_of(rs, run),
        nav_fig=v["nav_fig"],
        has_ms1=has_ms1,
        has_ms2=has_ms2,
        nav_note=v["nav_note"],
    )
    hero = cards.header(rs, run.name, run.label, ref, v["sub"])
    # The facts are the hero's right side; fill them in place.
    hero.children[1].children = v["facts"]
    spec_card = cards.spectrum_card(v["fig"], v["spec_title"], v["seq"], v["foot"])
    cand_card = cards.candidates_card(
        base=ctx.base,
        run_name=run.name,
        rows=v["rows"],
        columns=v["columns"],
        count=len(v["rows"]),
        note=v["note"],
        scroll=v["scroll"],
    )
    frag_card = cards.fragments_card(v["frag_title"], v["frag_body"])
    det_card = cards.details_card(v["details"])
    timings = {**v["timings"], "accepted": accepted_ms}
    timings["page"] = (time.perf_counter() - t0) * 1000.0
    stores = [
        dcc.Store(
            id="sp-key",
            data={"run": run.name, "base": ctx.base, "rt_max": rt_max_of(rs, run)},
        ),
        dcc.Store(id="sp-scan", data=scan_store(ref, _int(window_value))),
        dcc.Store(id="sp-cid", data=v["cid"]),
        dcc.Store(id="sp-step", data=None),
        dcc.Store(id="sp-goto", data=None),
        dcc.Store(id="sp-url", data=None),
    ]
    return dmc.Stack(
        [
            hero,
            *([dmc.Alert(notice, color="yellow", variant="light", p="xs")] if notice else []),
            nav,
            # Two columns from 1200 px; below, one column in the order spectrum, list,
            # fragments, scan (spectra.css).
            html.Div(
                [
                    html.Div([spec_card, det_card], className="sp-col sp-col-left"),
                    html.Div([cand_card, frag_card], className="sp-col sp-col-right"),
                ],
                className="sp-main",
            ),
            cards.footer(timings),
            *stores,
        ],
        gap="lg",
        className="sp-page",
    )


# --------------------------------------------------------------------------- callbacks


def navigate_to(
    rs: ResultSet,
    run: Run,
    cur: sc.ScanRef,
    trigger: Any,
    *,
    step: Any = None,
    scope: str = "window",
    scan_value: Any = None,
    rt_value: Any = None,
    level_value: Any = None,
    window_value: Any = None,
    slider_value: Any = None,
    click: Any = None,
    goto: Any = None,
) -> sc.ScanRef | None:
    """The scan a navigation input asks for (None: stay)."""
    window = _int(window_value)
    if window is None:
        window = cur.window_id
    level = cur.level

    def at(rt: Any, lvl: int | None = None) -> sc.ScanRef | None:
        value = _float(rt)
        if value is None:
            return None
        lv = level if lvl is None else lvl
        try:
            return sc.nearest_scan(rs, run, value, level=lv, window_id=window if lv == 2 else None)
        except KeyError:
            return sc.nearest_scan(rs, run, value, level=lv)

    if trigger == "sp-step":
        n = _int((step or {}).get("n")) if isinstance(step, dict) else None
        if not n:
            return None
        return sc.step(rs, run, cur, n, scope="run" if scope == "run" else "window")
    if trigger == "sp-scan-input":
        si = _int(scan_value)
        return sc.locate(rs, run, si) if si is not None else None
    if trigger == "sp-rt-input":
        return at(rt_value)
    if trigger == "sp-slider":
        return at(slider_value)
    if trigger == "sp-level":
        lv = _int(level_value)
        if lv not in (1, 2) or lv == cur.level:
            return None
        return at(cur.rt, lv)
    if trigger == "sp-window":
        if window is None:
            return None
        return at(cur.rt, 2)
    if trigger == NAV_FIG or (isinstance(trigger, dict) and trigger.get("name") == cards.NAV_GRAPH):
        points = (click or {}).get("points") or []
        if not points:
            return None
        return at(points[0].get("x"))
    if trigger == "sp-goto":
        if not isinstance(goto, dict):
            return None
        level, row = _int(goto.get("level")), _int(goto.get("row"))
        if level not in (1, 2) or row is None:
            return None
        try:
            return sc.scan_ref(rs, run, level, row)
        except IndexError:
            return None
    return None


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""

    app.clientside_callback(
        ClientsideFunction("mvs", "run"),
        Output("url", "href", allow_duplicate=True),
        Input("sp-run", "value"),
        State("sp-key", "data"),
        State("sp-scan", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvs", "pick"),
        Output("sp-cid", "data", allow_duplicate=True),
        Input(cards.GRID_ID, "cellClicked"),
        State("sp-cid", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvs", "address"),
        Output("sp-url", "data"),
        Input("sp-scan", "data"),
        Input("sp-cid", "data"),
        State("sp-key", "data"),
    )
    app.clientside_callback(
        ClientsideFunction("mvs", "relabel"),
        Output("sp-url", "data", allow_duplicate=True),
        Input(SPEC_FIG, "relayoutData"),
        Input(SPEC_FIG, "figure"),
        prevent_initial_call=True,
    )

    def _run(key: dict[str, Any] | None) -> Run | None:
        if not key:
            return None
        rs = get_rs()
        try:
            return rs.run(key["run"]) if rs.is_experiment else rs.runs[0]
        except KeyError:
            return None

    @app.callback(
        Output("sp-scan", "data"),
        Output("sp-scan-input", "value"),
        Output("sp-rt-input", "value"),
        Output("sp-level", "value"),
        Output("sp-window", "value"),
        Output("sp-slider", "value"),
        Input("sp-step", "data"),
        Input("sp-scan-input", "value"),
        Input("sp-rt-input", "value"),
        Input("sp-level", "value"),
        Input("sp-window", "value"),
        Input("sp-slider", "value"),
        Input(NAV_FIG, "clickData"),
        Input("sp-goto", "data"),
        State("sp-key", "data"),
        State("sp-scan", "data"),
        State("sp-scope", "value"),
        prevent_initial_call=True,
    )
    def _navigate(step, scan_v, rt_v, level_v, window_v, slider_v, click, goto, key, cur, scope):
        nothing = (no_update,) * 6
        run = _run(key)
        if run is None:
            return nothing
        rs = get_rs()
        here = ref_of(rs, run, cur)
        if here is None:
            return nothing
        trigger = dash_ctx.triggered_id
        try:
            new = navigate_to(
                rs,
                run,
                here,
                trigger,
                step=step,
                scope=scope,
                scan_value=scan_v,
                rt_value=rt_v,
                level_value=level_v,
                window_value=window_v,
                slider_value=slider_v,
                click=click,
                goto=goto,
            )
        except ViewerError:
            new = None
        if new is None:
            # Put the boxes back to the shown scan (an unknown scan_index, a click beside).
            return (
                no_update,
                here.scan_index,
                round(here.rt, 2),
                str(here.level),
                no_update,
                round(here.rt, 1),
            )
        return (
            scan_store(new, _int(window_v) if window_v is not None else here.window_id),
            new.scan_index,
            round(new.rt, 2),
            str(new.level),
            str(new.window_id) if new.window_id is not None else no_update,
            round(new.rt, 1),
        )

    @app.callback(
        Output("sp-title", "children"),
        Output("sp-sub", "children"),
        Output("sp-facts", "children"),
        Output(SPEC_FIG, "figure", allow_duplicate=True),
        Output("sp-spec-title", "children"),
        Output("sp-seq", "children"),
        Output("sp-spec-foot", "children"),
        Output(cards.GRID_ID, "rowData"),
        Output(cards.GRID_ID, "columnDefs"),
        Output("sp-cand-count", "children"),
        Output("sp-cand-note", "children"),
        Output(NAV_FIG, "figure", allow_duplicate=True),
        Output("sp-nav-note", "children"),
        Output("sp-frag-title", "children"),
        Output("sp-frag-body", "children"),
        Output("sp-details", "children"),
        Output("sp-cid", "data"),
        Output("sp-timings", "children"),
        Output(cards.GRID_ID, "scrollTo"),
        Input("sp-scan", "data"),
        Input("threshold", "data"),
        Input("sp-near", "value"),
        Input("sp-decoys", "checked"),
        Input("sp-cid", "data"),
        Input("sp-labels", "value"),
        Input("sp-scale", "value"),
        State("sp-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _show(scan, t, near, decoys, cid, labels, scale, key, scheme):
        nothing = (no_update,) * 19
        run = _run(key)
        if run is None:
            return nothing
        rs = get_rs()
        ref = ref_of(rs, run, scan)
        if ref is None:
            return nothing
        t0 = time.perf_counter()
        v = view(
            rs,
            run,
            ref,
            t=parse_threshold(t),
            near=near,
            decoys=bool(decoys),
            cid=_int(cid),
            n_labels=_int(labels) or 0,
            scale=scale if scale in ("base", "window", "matched") else "window",
            scheme=_scheme(scheme),
            base=str(key.get("base") or base),
            window_hint=_int((scan or {}).get("window")),
        )
        ms = (time.perf_counter() - t0) * 1000.0
        parts = [f"Updated in {ms:,.0f} ms"] + [
            f"{label} {v['timings'][k]:,.0f} ms"
            for k, label in (
                ("spectrum", "spectrum"),
                ("candidates", "candidates"),
                ("matches", "fragment matches"),
            )
            if k in v["timings"]
        ]
        return (
            v["title"],
            v["sub"],
            v["facts"],
            v["fig"],
            v["spec_title"],
            v["seq"],
            v["foot"],
            v["rows"],
            v["columns"],
            v["count"],
            v["note"],
            v["nav_fig"],
            v["nav_note"],
            v["frag_title"],
            v["frag_body"],
            v["details"],
            v["cid"] if v["cid"] != _int(cid) else no_update,
            "; ".join(parts) + ".",
            v["scroll"] if v["scroll"] is not None else no_update,
        )
