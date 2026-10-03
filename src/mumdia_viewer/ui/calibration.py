"""RT and mass calibration page (P1 views 4 and 5): how the run's retention times and
fragment masses were calibrated, and how the accepted identifications sit in the result.

Per run (a run selector in experiments; a band selector for a grouped run calibrated per
band), the page shows:

* the RT calibration anchors (the viewer's rebuild with the engine's rule, checked
  against ``cal.json``) with the fitted map and the window of ``run_windows``, and their
  in-sample residuals;
* the RT error of the accepted identifications across the gradient (the feature
  ``rt_error_signed``), at the header threshold on ``q_value`` (single run) or
  ``run_psm_q`` (experiment);
* how the anchors were selected, and the RT model that wrote the library iRT (the
  multi-head summary);
* the fragment mass calibration (``seed_psms.parquet.masscal.json``) against m/z, and the
  fragment mass errors of the accepted identifications.

Every number comes from :mod:`mumdia_viewer.data.calibration`. A hovered point fills the
card's point bar (server callback); a click opens its precursor page when it is a scored
row. The RT plots share their retention-time axis: a zoom in one moves the others
(``assets/calibration.js``). The address holds ``run`` (experiments) and ``band``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import dash_mantine_components as dmc
from dash import ClientsideFunction, Input, Output, State, ctx, dcc, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import calibration as data
from mumdia_viewer.data.calibration import MASS_COLUMNS

from . import calibration_cards as cards
from . import calibration_figures as cfig
from .state import PageContext, href, parse_threshold, stop_label
from .widgets import section

log = logging.getLogger(__name__)

FIT = {"type": "fig", "name": "cal-fit"}
ERR = {"type": "fig", "name": "cal-err"}
MASS = {"type": "fig", "name": "cal-mass"}
DEFAULT_MASS = MASS_COLUMNS[0]

_WARMING: set[tuple[int, str]] = set()
_WARM_LOCK = threading.Lock()


# --------------------------------------------------------------------------- address


def target(ctx: PageContext) -> tuple[str, str | None]:
    """(run name, band) of the address: an experiment's ``run`` (the first run when it
    names none or an unknown one), and the ``band`` of a grouped run (``gNN``)."""
    rs = ctx.rs
    run = rs.runs[0].name
    if rs.is_experiment:
        wanted = str(ctx.query.get("run") or "")
        if any(r.name == wanted for r in rs.runs):
            run = wanted
    band = str(ctx.query.get("band") or "") or None
    return run, band


def calibration_href(base: str, run: str = "", band: str | None = None) -> str:
    """The address of this page for one run (and band)."""
    return href(base, "calibration", {"run": run, "band": band})


def _scheme(scheme: Any) -> str:
    return "dark" if scheme == "dark" else "light"


def _run_of(rs: ResultSet, key: dict[str, Any] | None) -> tuple[str, str | None]:
    key = key or {}
    run = str(key.get("run") or (rs.runs[0].name if rs.runs else ""))
    band = key.get("band") or None
    return run, band


# --------------------------------------------------------------------------- warm-up


def _warm_runs(rs: ResultSet, threshold: float) -> None:
    """Read the anchors and the accepted rows of the other runs of an experiment, one run
    at a time in a background thread, so that switching runs answers at once."""
    todo = []
    with _WARM_LOCK:
        for r in rs.runs:
            k = (id(rs), r.name)
            if k not in _WARMING:
                _WARMING.add(k)
                todo.append(r)
    if not todo:
        return

    def work() -> None:
        for r in todo:
            try:
                for s in data.scopes(rs, r):
                    data.rt_anchors(rs, r, s.key or None)
                    data.accepted_errors(rs, r, threshold, s.key or None)
            except Exception:  # a guess ahead of the user: a failure only means a cold visit
                log.debug("calibration warm-up of %s failed", r.name, exc_info=True)

    threading.Thread(target=work, name="calibration-warm", daemon=True).start()


# --------------------------------------------------------------------------- layout


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs = ctx.rs
    run, band = target(ctx)
    try:
        scopes = data.scopes(rs, run)
        s = data.scope(rs, run, band)
    except (ViewerError, KeyError) as exc:
        return section("Calibration", dmc.Text(str(exc), c="red", size="sm"))
    key = s.key or None
    anchors = data.rt_anchors(rs, run, key)
    cal = data.cal_record(rs, run, key)
    masscal = data.masscal_record(rs, run, key)
    model = data.rt_model(rs, run, key)
    errors = data.accepted_errors(rs, run, ctx.threshold, key)
    mz = data.fragment_mz_range(rs, run, key)
    extract = data.extraction_text(rs, run, key)
    seed_tol = rs.config_get("search_seed", "fragment_tol_ppm")
    notes = [n for n in (anchors.error, *anchors.notes, *errors.notes) if n]
    if s.mode == "global" and s.note and s.note not in notes:
        notes.insert(0, s.note)
    scheme = ctx.scheme
    page = [
        cards.header(rs, s, scopes, anchors, cal, model),
        cards.facts(anchors, cal, masscal, errors),
        cards.notes_card(notes),
        dmc.Grid(
            [
                dmc.GridCol(cards.fit_card(anchors, cal, scheme), span={"base": 12, "lg": 8}),
                dmc.GridCol(cards.residual_card(anchors, cal, scheme), span={"base": 12, "lg": 4}),
            ],
            gutter="lg",
        ),
        dmc.Grid(
            [
                dmc.GridCol(cards.error_card(errors, scheme), span={"base": 12, "lg": 8}),
                dmc.GridCol(
                    dmc.Stack([cards.funnel_card(anchors), cards.model_card(model)], gap="lg"),
                    span={"base": 12, "lg": 4},
                ),
            ],
            gutter="lg",
        ),
        dmc.Grid(
            [
                dmc.GridCol(
                    cards.masscal_card(
                        masscal,
                        mz,
                        extract,
                        float(seed_tol) if isinstance(seed_tol, int | float) else None,
                        scheme,
                    ),
                    span={"base": 12, "lg": 5},
                ),
                dmc.GridCol(cards.mass_card(errors, masscal, scheme), span={"base": 12, "lg": 7}),
            ],
            gutter="lg",
        ),
    ]
    stores = [
        dcc.Store(id="cal-key", data={"run": run, "band": key}),
        dcc.Store(id="cal-err-meta", data={"t": ctx.threshold, "view": "seconds"}),
        dcc.Store(
            id="cal-mass-meta",
            data={"t": ctx.threshold, "column": DEFAULT_MASS, "view": "distribution"},
        ),
        dcc.Store(id="cal-link"),
    ]
    if rs.is_experiment:
        _warm_runs(rs, ctx.threshold)
    elapsed = (time.perf_counter() - t0) * 1000.0
    log.debug("calibration layout in %.0f ms", elapsed)
    footer = dmc.Text(
        f"Page built in {elapsed:.0f} ms. RT values in seconds; mass errors in ppm.",
        size="xs",
        c="dimmed",
        className="cal-footer",
    )
    return dmc.Stack(
        [*[p for p in page if p is not None], footer, *stores],
        gap="lg",
        className="cal-page",
        id="cal-root",
    )


# --------------------------------------------------------------------------- answers


def _point(event: dict[str, Any] | None) -> dict[str, Any] | None:
    points = (event or {}).get("points") or []
    return points[0] if points and isinstance(points[0], dict) else None


def anchor_row(rs: ResultSet, a: data.Anchors, event: dict[str, Any] | None) -> int | None:
    """The anchor row of a hovered or clicked point of the fit figure.

    ``customdata`` when the event has it, else the point's ``(curveNumber, pointIndex)``
    through :func:`calibration_figures.anchor_traces` (Dash leaves ``customdata`` out of
    the events of WebGL traces).
    """
    p = _point(event)
    if p is None or a.n == 0:
        return None
    cd = p.get("customdata")
    if isinstance(cd, list):
        cd = cd[0] if cd else None
    if cd is not None:
        try:
            return int(cd)
        except (TypeError, ValueError):
            return None
    try:
        curve, k = int(p.get("curveNumber")), int(p.get("pointIndex", p.get("pointNumber")))
    except (TypeError, ValueError):
        return None
    traces = rs.memo((data.__name__, "fit-traces", id(a)), lambda: cfig.anchor_traces(a))
    rows = traces.get(curve)
    return int(rows[k]) if rows is not None and 0 <= k < rows.size else None


def id_row(event: dict[str, Any] | None, e: data.AcceptedErrors, column: str) -> int | None:
    """The row of ``e.frame`` of a hovered or clicked identification.

    The identifications are the figure's first trace, without ``customdata``: point ``k``
    is row ``calibration_figures.id_rows(e, column)[k]``. Other traces give None.
    """
    p = _point(event)
    if p is None or p.get("curveNumber") != 0:
        return None
    k = p.get("pointIndex", p.get("pointNumber"))
    try:
        k = int(k)
    except (TypeError, ValueError):
        return None
    rows = cfig.id_rows(e, column)
    return int(rows[k]) if 0 <= k < rows.size else None


def _err_column(meta: dict[str, Any] | None) -> str:
    return cfig.error_column(str((meta or {}).get("view") or "seconds"))


def _mass_column(meta: dict[str, Any] | None) -> str | None:
    """The column the mass figure draws as points, or None in the histogram view."""
    meta = meta or {}
    if meta.get("view") != "gradient":
        return None
    column = meta.get("column")
    return column if column in MASS_COLUMNS else DEFAULT_MASS


def error_answer(
    rs: ResultSet, key: dict[str, Any] | None, t: Any, view: str, scheme: Any
) -> tuple[Any, ...]:
    """The RT error card at a threshold: figure, count, q chip, help, foot, fact, meta,
    key."""
    run, band = _run_of(rs, key)
    threshold = parse_threshold(t)
    view = view if view in cfig.ERROR_VIEWS else "seconds"
    e = data.accepted_errors(rs, run, threshold, band)
    med, sub = cards.accepted_rt_text(e)
    return (
        cfig.error_figure(e, _scheme(scheme), view=view),
        f"{e.n:,}",
        f"{e.q_column} ≤ {stop_label(threshold)}",
        cards.error_help(e),
        cards.error_foot(e),
        med,
        sub,
        {"t": threshold, "view": view},
        cards.error_keys(e),
    )


def mass_answer(
    rs: ResultSet, key: dict[str, Any] | None, t: Any, column: str, view: str, scheme: Any
) -> tuple[Any, ...]:
    """The mass error card at a threshold: figure, count, q chip, help, foot, meta."""
    run, band = _run_of(rs, key)
    threshold = parse_threshold(t)
    column = column if column in MASS_COLUMNS else DEFAULT_MASS
    view = view if view in cfig.MASS_VIEWS else "distribution"
    e = data.accepted_errors(rs, run, threshold, band)
    m = data.masscal_record(rs, run, band)
    return (
        cfig.mass_figure(e, m, _scheme(scheme), column=column, view=view),
        f"{e.n:,}",
        f"{e.q_column} ≤ {stop_label(threshold)}",
        cards.mass_help(e, column),
        cards.mass_foot(e, column),
        {"t": threshold, "column": column, "view": view},
    )


def anchor_point(rs: ResultSet, base: str, key: dict[str, Any] | None, event: Any, t: Any) -> Any:
    """The point bar of a hovered anchor (no_update for a point that is not an anchor)."""
    run, band = _run_of(rs, key)
    a = data.rt_anchors(rs, run, band)
    row = anchor_row(rs, a, event)
    if row is None:
        return no_update
    return cards.anchor_bar(rs, base, a, row, parse_threshold(t))


def id_point(
    rs: ResultSet,
    base: str,
    key: dict[str, Any] | None,
    event: Any,
    meta: dict[str, Any] | None,
    which: str,
    mass_column: str | None = None,
) -> Any:
    """The point bar of a hovered identification (``which``: ``err`` or ``mass``)."""
    run, band = _run_of(rs, key)
    meta = meta or {}
    e = data.accepted_errors(rs, run, parse_threshold(meta.get("t")), band)
    if which == "err":
        column = _err_column(meta)
    else:
        column = _mass_column(meta)
        if column is None:
            return no_update
    row = id_row(event, e, column)
    if row is None:
        return no_update
    shown = (
        mass_column
        if mass_column in MASS_COLUMNS
        else (column if column in MASS_COLUMNS else DEFAULT_MASS)
    )
    return cards.id_bar(rs, base, e, row, shown)


def click_target(
    rs: ResultSet,
    base: str,
    key: dict[str, Any] | None,
    which: str,
    event: Any,
    meta: dict[str, Any] | None,
) -> str | None:
    """The precursor page of a clicked point; None when it is not a scored row."""
    run, band = _run_of(rs, key)
    page_run = run if rs.is_experiment else ""
    if which == "fit":
        a = data.rt_anchors(rs, run, band)
        row = anchor_row(rs, a, event)
        if row is None or not 0 <= row < a.n or not bool(a.frame["scored"].iloc[row]):
            return None
        cid = int(a.frame["candidate_id"].iloc[row])
    else:
        e = data.accepted_errors(rs, run, parse_threshold((meta or {}).get("t")), band)
        column = _err_column(meta) if which == "err" else _mass_column(meta)
        row = id_row(event, e, column) if column else None
        if row is None:
            return None
        cid = int(e.frame["candidate_id"].iloc[row])
    return href(base, "precursor", {"run": page_run, "cid": cid})


# --------------------------------------------------------------------------- callbacks


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (``app.callback`` and ``app.clientside_callback`` only)."""

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input("cal-run", "value"),
        State("cal-key", "data"),
        prevent_initial_call=True,
    )
    def _choose_run(value, key):
        if not value or value == (key or {}).get("run"):
            return no_update
        return calibration_href(base, value)

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input("cal-band", "value"),
        State("cal-key", "data"),
        prevent_initial_call=True,
    )
    def _choose_band(value, key):
        if not value or value == (key or {}).get("band"):
            return no_update
        return calibration_href(base, (key or {}).get("run") or "", value)

    @app.callback(
        Output(ERR, "figure", allow_duplicate=True),
        Output("cal-err-count", "children"),
        Output("cal-err-q", "children"),
        Output("cal-err-help", "label"),
        Output("cal-err-foot", "children"),
        Output("cal-fact-acc-value", "children"),
        Output("cal-fact-acc-sub", "children"),
        Output("cal-err-meta", "data"),
        Output("cal-err-keys", "children"),
        Input("threshold", "data"),
        Input("cal-err-view", "value"),
        State("cal-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _errors(t, view, key, scheme):
        if not key:
            return (no_update,) * 9
        return error_answer(get_rs(), key, t, view, scheme)

    @app.callback(
        Output(MASS, "figure", allow_duplicate=True),
        Output("cal-mass-count", "children"),
        Output("cal-mass-q", "children"),
        Output("cal-mass-help", "label"),
        Output("cal-mass-foot", "children"),
        Output("cal-mass-meta", "data"),
        Input("threshold", "data"),
        Input("cal-mass-col", "value"),
        Input("cal-mass-view", "value"),
        State("cal-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _mass(t, column, view, key, scheme):
        if not key:
            return (no_update,) * 6
        return mass_answer(get_rs(), key, t, column, view, scheme)

    @app.callback(
        Output("cal-pt-fit", "children"),
        Input(FIT, "hoverData"),
        State("cal-key", "data"),
        State("threshold", "data"),
        prevent_initial_call=True,
    )
    def _hover_fit(hover, key, t):
        return anchor_point(get_rs(), base, key, hover, t)

    @app.callback(
        Output("cal-pt-err", "children"),
        Input(ERR, "hoverData"),
        State("cal-key", "data"),
        State("cal-err-meta", "data"),
        State("cal-mass-meta", "data"),
        prevent_initial_call=True,
    )
    def _hover_err(hover, key, meta, mass_meta):
        column = (mass_meta or {}).get("column")
        return id_point(get_rs(), base, key, hover, meta, "err", column)

    @app.callback(
        Output("cal-pt-mass", "children"),
        Input(MASS, "hoverData"),
        State("cal-key", "data"),
        State("cal-mass-meta", "data"),
        prevent_initial_call=True,
    )
    def _hover_mass(hover, key, meta):
        return id_point(get_rs(), base, key, hover, meta, "mass")

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input(FIT, "clickData"),
        Input(ERR, "clickData"),
        Input(MASS, "clickData"),
        State("cal-key", "data"),
        State("cal-err-meta", "data"),
        State("cal-mass-meta", "data"),
        prevent_initial_call=True,
    )
    def _click(fit_click, err_click, mass_click, key, err_meta, mass_meta):
        trig = ctx.triggered_id
        name = trig.get("name") if isinstance(trig, dict) else None
        rs = get_rs()
        if name == "cal-fit":
            out = click_target(rs, base, key, "fit", fit_click, None)
        elif name == "cal-err":
            out = click_target(rs, base, key, "err", err_click, err_meta)
        elif name == "cal-mass":
            out = click_target(rs, base, key, "mass", mass_click, mass_meta)
        else:
            out = None
        return out if out else no_update

    # The RT plots share the retention-time axis: a zoom in one moves the others.
    app.clientside_callback(
        ClientsideFunction("mvc", "link"),
        Output(FIT, "figure", allow_duplicate=True),
        Output(ERR, "figure", allow_duplicate=True),
        Output(MASS, "figure", allow_duplicate=True),
        Input(FIT, "relayoutData"),
        Input(ERR, "relayoutData"),
        Input(MASS, "relayoutData"),
        State(FIT, "figure"),
        State(ERR, "figure"),
        State(MASS, "figure"),
        prevent_initial_call=True,
    )


__all__ = ["calibration_href", "layout", "register", "target"]
