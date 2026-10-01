"""Precursor detail page (P0 view 3): why one identification was accepted.

The page links its panels around one scan: the XIC (fragment and MS1 traces with the
identification, quant and window markers), the spectrum mirror of the shown scan with the
sequence fragmentation diagram above it, and the fragments of the scan as a list or as
the ion table in ladder form (:mod:`.detail_ions`). A click on the XIC, the scan slider,
the scan buttons and the arrow keys choose the scan; hovering a fragment highlights it in
every panel and a legend click hides it in both plots (``assets/detail.js``). The verdict
strip, the q table, the competition and the partner carry validation marks at the
header's threshold. Every number comes from the data layer; values the viewer derives
say so. The cards are built in :mod:`.detail_cards`.

:func:`preview` is the compact, static version of the page for the identification page.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import (
    ClientsideFunction,
    Input,
    Output,
    State,
    dcc,
    html,
    no_update,
)

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.detail import (
    PrecursorDetail,
    detail_percentiles,
    mirror,
    precursor_detail,
)
from mumdia_viewer.data.errors import ArtifactNotFound
from mumdia_viewer.data.fasta import Fasta, protein_coverage

from . import coverage
from . import detail_cards as cards
from . import detail_figures as dfig
from . import detail_ions as ions
from . import detail_preview as pv
from . import detail_view as view
from .state import PageContext, href, parse_threshold
from .widgets import section

_CACHE: OrderedDict[tuple, PrecursorDetail] = OrderedDict()
_LOCK = threading.Lock()
_MAX = 32
# The apex mirror and the default percentiles of the cached details (same keys).
_PARTS: OrderedDict[tuple, Any] = OrderedDict()
_MAX_PARTS = 2 * _MAX

XIC = {"type": "fig", "name": "pd-xic"}
MIRROR = {"type": "fig", "name": "pd-mirror"}
COMP = {"type": "fig", "name": "pd-comp"}
PCT = {"type": "fig", "name": "pd-pct"}


def get_detail(rs: ResultSet, run: str, cid: int) -> PrecursorDetail:
    """The assembled detail, kept for the page's callbacks (a small LRU)."""
    key = (id(rs), run, int(cid))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    d = precursor_detail(rs, run, cid)
    with _LOCK:
        _CACHE[key] = d
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return d


def _is_cached(rs: ResultSet, run: str, cid: int) -> bool:
    with _LOCK:
        return (id(rs), run, int(cid)) in _CACHE


def _part(kind: str, rs: ResultSet, d: PrecursorDetail, make: Callable[[], Any]) -> Any:
    """A part of a page computed once per candidate (the apex mirror, the percentiles)."""
    key = (kind, id(rs), d.run.name, int(d.candidate_id))
    with _LOCK:
        if key in _PARTS:
            _PARTS.move_to_end(key)
            return _PARTS[key]
    value = make()
    with _LOCK:
        _PARTS[key] = value
        while len(_PARTS) > _MAX_PARTS:
            _PARTS.popitem(last=False)
    return value


def _target(ctx: PageContext) -> tuple[str, int] | None:
    try:
        cid = int(ctx.query.get("cid", ""))
    except ValueError:
        return None
    return (ctx.query.get("run", ""), cid) if cid >= 0 else None


def recent(ctx: PageContext) -> dict | None:
    target = _target(ctx)
    if target is None:
        return None
    try:
        d = get_detail(ctx.rs, *target)
    except (ViewerError, KeyError, ValueError):
        return None
    s = d.scored
    return {
        "run": target[0],
        "cid": target[1],
        "peptidoform": s.get("peptidoform"),
        "charge": s.get("charge"),
        "label": s.get("label"),
    }


def _why_no_mirror(d: PrecursorDetail) -> str:
    if d.tolerance is None:
        return "No spectrum match: the extraction tolerance is unknown."
    if d.chromatogram is None:
        return "No predicted fragments: the candidate has no chromatogram rows."
    if d.apex_scan is None:
        return "No MS2 scan of an isolation window covering the precursor at the apex."
    return "No spectrum for this candidate."


def _apex_mirror(rs: ResultSet, d: PrecursorDetail) -> tuple[Any, str]:
    """The mirror of the apex scan and the reason when there is none (cached)."""

    def make() -> tuple[Any, str]:
        try:
            return mirror(rs, d), _why_no_mirror(d)
        except ViewerError as exc:
            return None, f"Spectrum unavailable: {exc}"

    return _part("mirror", rs, d, make)


def _percentiles(rs: ResultSet, d: PrecursorDetail) -> list:
    def make() -> list:
        try:
            return detail_percentiles(rs, d)
        except ViewerError:
            return []

    return _part("percentiles", rs, d, make)


# --------------------------------------------------------------------------- addresses

_CID = re.compile(r"\d+(?:\.0*)?")


def parse_cid(value: Any) -> tuple[int | None, str | None]:
    """A candidate id from an address value: (id, None), or (None, why it is not one)."""
    text = str(value if value is not None else "").strip()
    if not text:
        return None, "No candidate id in the address. Open a row of the identification browser."
    if not _CID.fullmatch(text):
        return None, f"The candidate id {text!r} is not a whole number of 0 or more."
    return int(text.split(".")[0]), None


def _run_links(ctx: PageContext, cid: int) -> list[tuple[str, str]]:
    """Links to the runs of an experiment that hold a scored row of ``cid``."""
    return [
        (f"run {name}", href(ctx.base, "precursor", {"run": name, "cid": cid}))
        for name in view.runs_holding(ctx.rs, cid)
    ]


def _resolve(ctx: PageContext) -> tuple[str, int | None, PrecursorDetail | None, Any, bool]:
    """(run, cid, detail, page to show instead, detail was cached) of the page's address.

    In a single run the address's run is ignored. In an experiment an address without a
    run, with an unknown run or with a run that does not hold the candidate offers the
    runs that do.
    """
    rs = ctx.rs
    cid, why = parse_cid(ctx.query.get("cid"))
    if cid is None:
        return "", None, None, cards.missing(ctx, why or ""), False
    if not rs.is_experiment:
        run = ""
    else:
        run = str(ctx.query.get("run") or "").strip()
        if not run:
            links = _run_links(ctx, cid)
            text = (
                f"The address names no run. Candidate {cid} has scored rows in "
                f"{len(links)} run{'s' if len(links) != 1 else ''} of this experiment; choose one."
                if links
                else f"Candidate {cid} has no scored row in any run of this experiment."
            )
            return run, cid, None, cards.missing(ctx, text, links), False
    cached = _is_cached(rs, run, cid)
    try:
        return run, cid, get_detail(rs, run, cid), None, cached
    except KeyError as exc:
        reason = str(exc.args[0]) if exc.args else str(exc)
        links = _run_links(ctx, cid)
        text = f"{reason[0].upper()}{reason[1:]}." if reason else f"No run {run!r}."
        return run, cid, None, cards.missing(ctx, text, links), False
    except ArtifactNotFound:
        where = f"run {run}" if rs.is_experiment else "this run"
        links = _run_links(ctx, cid) if rs.is_experiment else []
        text = f"Candidate {cid} is not a scored row of {where}."
        if links:
            text += " It is scored in the runs below."
        return run, cid, None, cards.missing(ctx, text, links), False
    except (ViewerError, ValueError) as exc:
        return run, cid, None, cards.missing(ctx, f"Candidate {cid}: {exc}"), False


# --------------------------------------------------------------------------- preview


def preview(ctx: PageContext, run: str, cid: int) -> Any:
    """A compact, static summary of one precursor for the identification page.

    ``run`` is the run name (``""`` in a single run) and ``cid`` the candidate id, as in
    the precursor page's address. The result is one ``dmc.Card`` (id ``pv-card``, about
    320 px high): place it as a panel of its own, not inside another card. It shows the
    peptidoform, charge, label, the key q values with validation marks at
    ``ctx.threshold``, the score and the quant state; the compact XIC and MS1 figure; the
    sequence fragmentation diagram over the annotated spectrum of the apex scan; the ion
    table; and a link that opens the precursor page. It has no callbacks; its graphs are
    named ``pv-xic`` and ``pv-spec``. The detail comes from :func:`get_detail`, so the
    precursor page then opens warm. A candidate that cannot be read gives a card with
    the reason.
    """
    rs = ctx.rs
    run = (run or "") if rs.is_experiment else ""
    number, why = parse_cid(cid)
    if number is None:
        return pv.preview_missing(ctx, run, cid, why or "")
    if rs.is_experiment and not run:
        runs = view.runs_holding(rs, number)
        held = f"; it is scored in {', '.join(runs)}" if runs else ""
        return pv.preview_missing(ctx, run, number, f"No run given for candidate {number}{held}.")
    try:
        d = get_detail(rs, run, number)
    except KeyError as exc:
        reason = str(exc.args[0]) if exc.args else str(exc)
        return pv.preview_missing(ctx, run, number, f"Candidate {number}: {reason}.")
    except ArtifactNotFound:
        where = f"run {run}" if rs.is_experiment else "this run"
        return pv.preview_missing(
            ctx, run, number, f"Candidate {number} is not a scored row of {where}."
        )
    except (ViewerError, ValueError, TypeError) as exc:
        return pv.preview_missing(ctx, run, number, f"Candidate {number}: {exc}")
    m, why = _apex_mirror(rs, d)
    grid = view.scan_grid(rs, d)
    frags = view.fragments_of(d.chromatogram)
    return pv.preview_card(ctx, d, run, frags, grid, m, why)


# --------------------------------------------------------------------------- coverage


def coverage_card(ctx: PageContext, d: PrecursorDetail) -> Any:
    """The protein coverage of this precursor's group (filled by a callback after the page)."""
    group = str(d.scored.get("protein_group") or "")
    request = {
        "group": group,
        "peptide": d.scored.get("base_peptide_id"),
        "member": None,
        "t": ctx.threshold,
    }
    body = (
        coverage.no_fasta()
        if ctx.fasta is None
        else coverage.message("The coverage of this protein group follows.")
    )
    return section(
        "Protein coverage",
        html.Div(
            body,
            id="pd-cov",
            className="pd-cov",
            **{"data-cov-req": "pd-cov-req", "data-cov-href": f"{ctx.base}identifications"},
        ),
        dcc.Store(id="pd-cov-req", data=request),
        subtitle="Where this peptide (outlined) sits in its protein, beside the group's "
        "other peptides. Computed by the viewer from the FASTA; click a peptide to open it "
        "in the identifications.",
    )


def coverage_body(rs: ResultSet, fasta: Fasta | None, request: dict | None, t: Any) -> Any:
    if fasta is None:
        return coverage.no_fasta()
    group = str((request or {}).get("group") or "")
    if not group:
        return coverage.message("This row has no protein group, so it has no coverage.")
    try:
        cov = protein_coverage(
            rs,
            fasta,
            group,
            member=(request or {}).get("member") or None,
            threshold=parse_threshold(t if t is not None else (request or {}).get("t")),
        )
    except ViewerError as exc:
        return coverage.message(str(exc), warn=True)
    peptide = (request or {}).get("peptide")
    return coverage.card_body(cov, selected=int(peptide) if peptide is not None else None)


# --------------------------------------------------------------------------- layout


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs = ctx.rs
    run, cid, d, instead, cached = _resolve(ctx)
    if d is None or cid is None:
        return instead
    extra: dict[str, float] = {}
    t = time.perf_counter()
    m, why = _apex_mirror(rs, d)
    extra["mirror"] = (time.perf_counter() - t) * 1000.0
    t = time.perf_counter()
    pct = _percentiles(rs, d)
    extra["percentiles"] = (time.perf_counter() - t) * 1000.0
    grid = view.scan_grid(rs, d)
    frags = view.fragments_of(d.chromatogram)
    ladder = view.ion_ladder(d.scored.get("peptidoform"), frags, m)
    tiles = view.verdict_tiles(rs, d)
    pct_map = view.percentile_map(pct)
    scan_rt = float(np.float32(m.pick.rt)) if m is not None else None
    xr = dfig.x_range(d, grid)
    views = cards.xic_views(d, grid)
    mz = view.mz_range(m)
    linked = html.Div(
        [
            html.Div(
                [
                    cards.xic_card(ctx, d, frags, grid, key=f"{run}|{int(cid)}", views=views),
                    cards.rt_card(d, xr, pct_map),
                ],
                className="pd-col",
            ),
            html.Div(
                [
                    cards.mirror_card(ctx, d, frags, m, why, ladder, x_range=mz),
                    cards.fragment_card(m, frags, ladder),
                ],
                className="pd-col",
            ),
        ],
        className="pd-linked-grid",
    )
    side = [c for c in (cards.transfer_card(d), cards.partner_card(ctx, d)) if c is not None]
    competition = cards.competition_card(ctx, d)
    if rs.is_experiment:
        # The competition spans every run: its table needs the full width. The partner
        # (two lines of text) sits next to the q values.
        rows = [
            competition,
            dmc.Grid(
                [
                    dmc.GridCol(cards.q_card(ctx, d), span={"base": 12, "lg": 6}),
                    dmc.GridCol(dmc.Stack(side, gap="lg"), span={"base": 12, "lg": 6}),
                ],
                gutter="lg",
            ),
            cards.features_card(ctx, d, pct),
        ]
    else:
        rows = [
            dmc.Grid(
                [
                    dmc.GridCol(competition, span={"base": 12, "lg": 8}),
                    dmc.GridCol(dmc.Stack(side, gap="lg"), span={"base": 12, "lg": 4}),
                ],
                gutter="lg",
            ),
            dmc.Grid(
                [
                    dmc.GridCol(cards.q_card(ctx, d), span={"base": 12, "lg": 6}),
                    dmc.GridCol(cards.features_card(ctx, d, pct), span={"base": 12, "lg": 6}),
                ],
                gutter="lg",
            ),
        ]
    page = [
        cards.hero(ctx, d, frags),
        cards.verdict(ctx, d, tiles),
        cards.notes(d),
        linked,
        cards.evidence_grid(d, pct_map),
        coverage_card(ctx, d),
        *rows,
    ]
    index = grid.index_of_row(m.pick.row) if grid is not None and m is not None else None
    stores = [
        dcc.Store(id="pd-key", data={"run": run, "cid": int(cid), "mz": mz}),
        dcc.Store(id="pd-grid", data=grid.to_store() if grid is not None else {"rows": []}),
        dcc.Store(
            id="pd-scan",
            data={"row": m.pick.row if m is not None else None, "index": index, "rt": scan_rt},
        ),
        dcc.Store(id="pd-nav", data=cards.nav_data(rs, d, m)),
        dcc.Store(id="pd-hidden", data=[]),
        dcc.Store(id="pd-init"),
        dcc.Store(id="pd-settled"),
        dcc.Store(id="pd-mirror-next"),
    ]
    extra["page"] = (time.perf_counter() - t0) * 1000.0
    return dmc.Stack(
        [p for p in page if p is not None] + [cards.footer(d, extra, cached=cached), *stores],
        gap="lg",
        className="pd-page pd-linked",
    )


# --------------------------------------------------------------------------- callbacks


def _scheme(scheme: Any) -> str:
    return "dark" if scheme == "dark" else "light"


def show_scan(
    rs: ResultSet,
    key: dict,
    scan: dict | None,
    scale: str | None,
    hidden: list[int] | None,
    scheme: str,
) -> tuple[Any, ...]:
    """The mirror, the fragment list, the scan badges, the tolerance row, the scan
    neighbours, the sequence diagram and the ion table of the scan in ``scan``.

    The mirror keeps the candidate's m/z range (``key["mz"]``, from the apex scan), so a
    zoom survives the step.
    """
    d = get_detail(rs, key.get("run", ""), int(key["cid"]))
    row = (scan or {}).get("row")
    m = mirror(rs, d, row=int(row)) if row is not None else _apex_mirror(rs, d)[0]
    frags = view.fragments_of(d.chromatogram)
    off = {int(h) for h in (hidden or [])}
    ladder = view.ion_ladder(d.scored.get("peptidoform"), frags, m)
    outside = view.in_window(m.pick.rt, d) is False if m is not None else None
    mz = key.get("mz") or view.mz_range(_apex_mirror(rs, d)[0]) or view.mz_range(m)
    fig = dfig.mirror_figure(
        m,
        frags,
        scheme,
        hidden=off,
        scale=scale or "matched",
        outside=outside,
        why_empty=_why_no_mirror(d),
        x_range=mz,
    )
    badge = cards.match_badge(m)
    return (
        fig,
        cards.fragment_table(m, frags, off),
        cards.scan_info(rs, d, m),
        cards.tolerance_row(d, m),
        cards.nav_data(rs, d, m),
        ions.sequence_diagram(ladder, hidden=off, match=badge),
        cards.ladder_panel(ladder, off, m),
    )


def follow_threshold(rs: ResultSet, base: str, key: dict, threshold: float) -> tuple:
    """The parts of the page that test q values against the threshold, rebuilt at ``threshold``:
    the verdict, the q table, the competition table and the partner."""
    d = get_detail(rs, key.get("run", ""), int(key["cid"]))
    ctx = PageContext(rs=rs, base=base, threshold=threshold)
    return (
        cards.verdict_body(ctx, d, view.verdict_tiles(rs, d)),
        cards.q_table(ctx, d),
        cards.comp_table(ctx, d),
        cards.partner_body(ctx, d),
    )


def features(rs: ResultSet, key: dict, chosen: list[str] | None, scheme: str) -> tuple:
    """The percentile figure, its notes and the chooser label for the chosen features."""
    d = get_detail(rs, key.get("run", ""), int(key["cid"]))
    rows = detail_percentiles(rs, d, list(chosen)) if chosen else []
    n = len(chosen or [])
    return (
        dfig.percentile_figure(rows, scheme),
        cards.pct_notes(rows),
        f"{n} feature{'s' if n != 1 else ''}",
    )


def register(app, get_rs, base: str) -> None:
    """Callbacks of the precursor page."""
    get_fasta = getattr(app, "mv_fasta", lambda: None)

    @app.callback(
        Output("pd-cov", "children"),
        Input("pd-cov-req", "data"),
        Input("threshold", "data"),
    )
    def coverage_of(request, t):
        if not request:
            return no_update
        return coverage_body(get_rs(), get_fasta(), request, t)

    app.clientside_callback(
        ClientsideFunction("mvd", "init"),
        Output("pd-init", "data"),
        Input("pd-key", "data"),
        State("pd-grid", "data"),
        State("pd-scan", "data"),
        State("pd-xview", "data"),
        State("pd-nav", "data"),
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "scan"),
        Output("pd-scan", "data"),
        Output("pd-scrub", "value"),
        Input(XIC, "clickData"),
        Input("pd-scrub", "value"),
        Input("scan-prev", "n_clicks"),
        Input("scan-apex", "n_clicks"),
        Input("scan-next", "n_clicks"),
        State("pd-grid", "data"),
        State("pd-scan", "data"),
        State("pd-nav", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "hidden"),
        Output(XIC, "figure", allow_duplicate=True),
        Output(MIRROR, "figure", allow_duplicate=True),
        Input("pd-hidden", "data"),
        State(XIC, "figure"),
        State(MIRROR, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "axis"),
        Output(XIC, "figure", allow_duplicate=True),
        Input("pd-xic-axis", "value"),
        State(XIC, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "view"),
        Output(XIC, "figure", allow_duplicate=True),
        Output("pd-scrub", "min"),
        Output("pd-scrub", "max"),
        Output("pd-scrub", "marks"),
        Output("pd-scrub-box", "style"),
        Input("pd-xic-view", "value"),
        State(XIC, "figure"),
        State("pd-xview", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "settled"),
        Output("pd-settled", "data"),
        Input("pd-nav", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvd", "open"),
        Output("url", "href", allow_duplicate=True),
        Input(COMP, "clickData"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction("mvd", "fragView"),
        Output("pd-frag-table", "style"),
        Output("pd-ladder", "style"),
        Input("pd-frag-view", "value"),
    )

    # The server's mirror goes through a store: Dash writes a user's zoom into the
    # graph's figure, so Plotly cannot keep it when a new figure arrives; detail.js puts
    # the zoom on the new figure (mvd.mirror).
    app.clientside_callback(
        ClientsideFunction("mvd", "mirror"),
        Output(MIRROR, "figure", allow_duplicate=True),
        Input("pd-mirror-next", "data"),
        State(MIRROR, "figure"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("pd-mirror-next", "data"),
        Output("pd-frag-table", "children"),
        Output("pd-scan-info", "children"),
        Output("pd-tol", "children"),
        Output("pd-nav", "data"),
        Output("pd-seq", "children"),
        Output("pd-ladder", "children"),
        Input("pd-scan", "data"),
        Input("pd-scale", "value"),
        State("pd-hidden", "data"),
        State("pd-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _show_scan(scan, scale, hidden, key, scheme):
        if not key:
            return (no_update,) * 7
        try:
            return show_scan(get_rs(), key, scan, scale, hidden, _scheme(scheme))
        except (ViewerError, IndexError, KeyError, ValueError):
            return (no_update,) * 7

    @app.callback(
        Output("pd-verdict", "children"),
        Output("pd-q-table", "children"),
        Output("pd-comp-table", "children"),
        Output("pd-partner-body", "children"),
        Input("threshold", "data"),
        State("pd-key", "data"),
        prevent_initial_call=True,
    )
    def _threshold(t, key):
        if not key:
            return (no_update,) * 4
        try:
            return follow_threshold(get_rs(), base, key, parse_threshold(t))
        except (ViewerError, KeyError, ValueError):
            return (no_update,) * 4

    @app.callback(
        Output(PCT, "figure", allow_duplicate=True),
        Output("pd-pct-notes", "children"),
        Output("pd-feat-btn", "children"),
        Input("pd-feat-select", "value"),
        State("pd-key", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _features(chosen, key, scheme):
        if not key:
            return no_update, no_update, no_update
        try:
            return features(get_rs(), key, chosen, _scheme(scheme))
        except ViewerError as exc:
            return no_update, [dmc.Text(str(exc), size="xs", c="red")], no_update


__all__ = [
    "features",
    "follow_threshold",
    "get_detail",
    "layout",
    "parse_cid",
    "preview",
    "recent",
    "register",
    "show_scan",
]
