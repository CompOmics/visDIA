"""One precursor across the runs of an experiment (P2 view 9).

The address is ``runs?peptidoform=..&charge=..``. The page shows, for every run, the
precursor's XIC on the run's own RT axis (side by side, or overlaid on the apex), its
scored row with run_psm_q, its quant state and quantity (MBR transfers marked; a missing
quantity is never 0), and the experiment-wide grouped q values with the run that holds
them. Every run links to its precursor page. A single run shows its one run and says so.

The data are :func:`mumdia_viewer.data.across.precursor_across` and
:func:`~mumdia_viewer.data.across.run_xics` (memoised on the result set, so every
callback of the page is fast after the first build). Ids start with ``xr-``; the browser
code is ``window.dash_clientside.mvr`` (``assets/runs.js``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import dash_mantine_components as dmc
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import across as A
from mumdia_viewer.data.quantqc import suggest_conditions

from . import runs_cards as cards
from . import runs_figures as rf
from .icons import icon
from .quant_cards import loading, qq_section
from .state import PageContext, href, parse_threshold
from .widgets import empty, graph

log = logging.getLogger(__name__)

CONDITIONS_STORE = "mv-conditions"
MODES = [("grid", "Side by side"), ("overlay", "Overlaid on the apex")]
SCALES = [("own", "Each run's scale"), ("shared", "One scale")]


def _scheme(value: Any) -> str:
    return "dark" if value == "dark" else "light"


def _charge(value: Any) -> int | None:
    try:
        z = int(str(value))
    except (TypeError, ValueError):
        return None
    return z if 0 < z < 100 else None


def wanted(ctx: PageContext) -> tuple[tuple[str, int] | None, str | None]:
    """The precursor of the address, or the default one with a note saying so."""
    pep = (ctx.query.get("peptidoform") or "").strip()
    z = _charge(ctx.query.get("charge"))
    if pep and z is not None:
        return (pep, z), None
    pick = A.default_precursor(ctx.rs, ctx.threshold)
    if pick is None:
        return None, None
    return pick, (
        "No precursor in the address: of the 50 best-scoring precursors accepted at the "
        "header threshold (precursor_q), this one is scored in the most runs. Find another "
        "one on the right, or open this page from a precursor page."
    )


def xic_figure(
    a: A.PrecursorAcross, xics: list[A.RunXic], mode: str, scale: str, t: float, scheme: str
) -> Any:
    shared = scale == "shared"
    if mode == "overlay":
        return rf.xic_overlay_figure(a, xics, t, scheme, shared=shared)
    return rf.xic_grid_figure(a, xics, t, scheme, shared=shared)


def _options(a: A.PrecursorAcross) -> list[dict[str, str]]:
    out = [
        {
            "value": cards.pick_value(a.peptidoform, a.charge),
            "label": f"{a.peptidoform} {a.charge}+",
        }
    ]
    for row in a.siblings.head(10).to_dict("records"):
        if row["label"] != "target":
            continue
        out.append(
            {
                "value": cards.pick_value(row["peptidoform"], row["charge"]),
                "label": cards.pick_label(
                    row["peptidoform"], row["charge"], row["n_runs"], row["precursor_q"]
                ),
            }
        )
    return out


def search_options(rs: ResultSet, text: Any, value: Any) -> list[dict[str, str]]:
    """Target precursors containing ``text``, with the current value kept first."""
    df = A.find_precursors(rs, str(text or ""), limit=30)
    out: list[dict[str, str]] = []
    cur = cards.parse_pick(value)
    if cur is not None:
        out.append({"value": cards.pick_value(*cur), "label": f"{cur[0]} {cur[1]}+"})
    for row in df.to_dict("records"):
        v = cards.pick_value(row["peptidoform"], row["charge"])
        if any(o["value"] == v for o in out):
            continue
        out.append(
            {
                "value": v,
                "label": cards.pick_label(
                    row["peptidoform"], row["charge"], row["n_runs"], row["precursor_q"]
                ),
            }
        )
    return out


def _marker_key() -> Any:
    def item(cls: str, text: str) -> Any:
        return html.Span([html.Span(className=f"xr-key {cls}"), text], className="xr-key-item")

    return html.Div(
        [
            item("xr-key-elution", "elution bounds"),
            item("xr-key-apex", "apex_rt"),
            item("xr-key-int", "integration (peptide_quant)"),
            item("xr-key-pred", "rt_pred_cal"),
        ],
        className="xr-keys",
    )


def threshold_parts(
    ctx: PageContext, a: A.PrecursorAcross, xics: list[A.RunXic], suggestion: dict[str, str]
) -> tuple[Any, Any, Any]:
    """The parts that follow the threshold: the strip, the table and the q figure."""
    return (
        cards.q_strip(ctx, a),
        cards.run_table(ctx, a, xics, suggestion),
        rf.q_figure(a, ctx.threshold, ctx.scheme),
    )


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs = ctx.rs
    key, note = wanted(ctx)
    if key is None:
        return dmc.Stack(
            [
                cards.picker(None, []),
                dmc.Card(empty("This result set has no target precursor.")),
            ]
        )
    try:
        a = A.precursor_across(rs, key[0], key[1])
    except ViewerError as exc:
        return dmc.Stack(
            [
                dmc.Group(
                    [
                        html.Div(
                            [
                                dmc.Text("Across runs", className="mv-eyebrow"),
                                html.Div("Precursor not found", className="mv-title"),
                            ]
                        ),
                        cards.picker(None, []),
                    ],
                    justify="space-between",
                    align="flex-end",
                ),
                dmc.Card(empty(str(exc), "alert")),
            ],
            gap="lg",
        )
    xics = A.run_xics(rs, a)
    t_data = time.perf_counter() - t0
    suggestion = suggest_conditions(rs)
    strip, table, qfig = threshold_parts(ctx, a, xics, suggestion)
    notes = [note] if note else []
    notes += a.notes
    if not rs.is_experiment:
        notes.insert(
            0,
            "This is a single run: the page shows its one run. Open an experiment to see a "
            "precursor across runs.",
        )
    alerts = (
        [
            dmc.Alert(
                dmc.Stack([dmc.Text(n, size="sm") for n in notes], gap=2),
                color="blue",
                variant="light",
                icon=icon("info", 16),
                className="xr-alert",
            )
        ]
        if notes
        else []
    )
    n_chrom = sum(1 for x in xics if x.chromatogram is not None)
    mode = "grid"
    controls = dmc.Group(
        [
            dmc.SegmentedControl(
                id="xr-mode",
                data=[{"value": v, "label": label} for v, label in MODES],
                value=mode,
                size="xs",
                radius="md",
            ),
            dmc.Tooltip(
                dmc.SegmentedControl(
                    id="xr-scale",
                    data=[{"value": v, "label": label} for v, label in SCALES],
                    value="own",
                    size="xs",
                    radius="md",
                    color="teal",
                ),
                label="Each run's scale: every panel on its own intensity axis (overlaid: "
                "each run's summed trace divided by its maximum). One scale: every run on "
                "the same intensity axis.",
            ),
        ],
        gap="xs",
        wrap="nowrap",
    )
    xic_card = qq_section(
        "XICs across runs",
        _marker_key(),
        loading(graph("xr-xic", xic_figure(a, xics, mode, "own", ctx.threshold, ctx.scheme))),
        dmc.Text(
            "Click a trace to open that run's precursor page. Click a fragment in the legend "
            "to hide it in every run.",
            size="xs",
            c="dimmed",
            mt=4,
        ),
        count=f"{n_chrom} of {len(xics)} run{'s' if len(xics) != 1 else ''}",
        help="The fragment traces of the candidate in each run (chromatograms.parquet, decoded "
        "per run), b ions blue and y ions red as on the precursor page. Each run is on its own "
        "RT axis: no alignment. Overlaid: the viewer's sum of each run's fragment traces, "
        "shifted so that the run's apex_rt is at 0.",
        subtitle="Each run on its own retention-time axis, with its apex and elution bounds",
        right=controls,
        id="xr-xic-card",
    )
    table_card = qq_section(
        "Per run",
        html.Div(table, id="xr-table-box"),
        count=f"{a.n_scored} of {a.n_runs} scored",
        help="One row per run. The validation mark tests run_psm_q (this run's own q) at the "
        "header threshold; q_value is pooled over the whole rescore. A run without a row did "
        "not score the precursor. Quantity: peptide_quant of the run's row; a missing "
        "quantity says why and is never 0. Click a run to open its precursor page.",
        subtitle="Scored rows, q values and quantities of the precursor in each run",
    )
    quant_card = qq_section(
        "Quantity per run",
        graph("xr-quant", rf.quantity_figure(a, ctx.scheme)),
        count=f"{a.n_quantified} quantified",
        help=f"Bars: {A.ROW_LABELS['quantity']}. Diamonds: {A.ROW_LABELS['lfq_quantity']}. "
        "A run without a quantity has no bar and its state is written at the base. "
        "Match-between-runs transfers are hatched in lime.",
        subtitle="peptide_quant per run" + (", with the MaxLFQ precursor" if a.experiment else ""),
    )
    q_card = qq_section(
        "q values per run",
        graph("xr-q", qfig),
        help="run_psm_q: this run's own target-decoy re-run (filled, green when it passes). "
        "q_value: PSM rows pooled over the whole rescore (open diamond). The grouped columns "
        "are experiment-wide, in the strip at the top.",
        subtitle="run_psm_q and the pooled q_value, log scale",
    )
    sib_card = qq_section(
        "Other precursors of this peptide",
        cards.siblings(ctx, a),
        count=len(a.siblings),
        help="The other (peptidoform, charge) rows of the same base_peptide_id in the pooled "
        "scored table, targets and decoys. The mark tests precursor_q (on the precursor's "
        "winning row) at the header threshold. Click one to see it across runs.",
    )
    rows: list[Any] = [
        dcc.Store(id="xr-key", data={"peptidoform": a.peptidoform, "charge": a.charge}),
        dcc.Store(id="xr-suggest", data=suggestion),
        cards.hero(ctx, a, xics, _options(a)),
        *alerts,
        html.Div(strip, id="xr-strip-box"),
        xic_card,
        table_card,
        dmc.Grid(
            [
                dmc.GridCol(quant_card, span={"base": 12, "md": 6}),
                dmc.GridCol(q_card, span={"base": 12, "md": 6}),
            ],
            gutter="lg",
        ),
        sib_card,
    ]
    build = (time.perf_counter() - t0) * 1000.0
    rows.append(
        dmc.Text(
            f"Page built in {build:.0f} ms (data {t_data * 1000:.0f} ms).",
            size="xs",
            c="dimmed",
            className="xr-foot",
        )
    )
    log.debug("runs layout in %.0f ms", build)
    return html.Div(dmc.Stack(rows, gap="lg"), className="xr-root")


# --------------------------------------------------------------------------- callbacks


def _across(rs: ResultSet, key: Any) -> A.PrecursorAcross | None:
    if not isinstance(key, dict):
        return None
    z = _charge(key.get("charge"))
    pep = key.get("peptidoform")
    if not pep or z is None:
        return None
    try:
        return A.precursor_across(rs, str(pep), z)
    except ViewerError:
        return None


def click_target(base: str, a: A.PrecursorAcross, click: Any) -> str | None:
    """The precursor page of the run a click on a figure points at."""
    try:
        point = click["points"][0]
    except (TypeError, KeyError, IndexError):
        return None
    custom = point.get("customdata")
    run = custom[0] if isinstance(custom, list) and custom else custom
    if not isinstance(run, str):
        return None
    rows = a.rows[(a.rows["run"] == str(run)) & a.rows["scored"]]
    if rows.empty:
        return None
    return cards.run_href(base, str(run), rows["candidate_id"].iloc[0])


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""
    app.clientside_callback(
        ClientsideFunction("mvr", "conditions"),
        Output({"type": "xr-cond", "run": ALL}, "children"),
        Input("xr-suggest", "data"),
        State(CONDITIONS_STORE, "data"),
        State({"type": "xr-cond", "run": ALL}, "id"),
    )

    @app.callback(
        Output({"type": "fig", "name": "xr-xic"}, "figure", allow_duplicate=True),
        Input("xr-mode", "value"),
        Input("xr-scale", "value"),
        State("xr-key", "data"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def follow_mode(mode, scale, key, t, scheme):
        rs = get_rs()
        a = _across(rs, key)
        if a is None:
            return no_update
        return xic_figure(a, A.run_xics(rs, a), mode, scale, parse_threshold(t), _scheme(scheme))

    @app.callback(
        Output("xr-strip-box", "children"),
        Output("xr-table-box", "children"),
        Output({"type": "fig", "name": "xr-q"}, "figure", allow_duplicate=True),
        Output({"type": "fig", "name": "xr-xic"}, "figure", allow_duplicate=True),
        Input("threshold", "data"),
        State("xr-key", "data"),
        State("xr-mode", "value"),
        State("xr-scale", "value"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def follow_threshold(t, key, mode, scale, scheme):
        rs = get_rs()
        a = _across(rs, key)
        if a is None:
            return no_update, no_update, no_update, no_update
        ctx = PageContext(rs=rs, base=base, threshold=parse_threshold(t), scheme=_scheme(scheme))
        xics = A.run_xics(rs, a)
        strip, table, qfig = threshold_parts(ctx, a, xics, suggest_conditions(rs))
        return strip, table, qfig, xic_figure(a, xics, mode, scale, ctx.threshold, ctx.scheme)

    @app.callback(
        Output("xr-pick", "data"),
        Input("xr-pick", "searchValue"),
        State("xr-pick", "value"),
        State("xr-pick", "data"),
        prevent_initial_call=True,
    )
    def search(text, value, data):
        needle = str(text or "").strip()
        # Selecting an option writes its label into the search box: nothing to search.
        if len(needle) < 2 or any(o.get("label") == needle for o in data or []):
            return no_update
        return search_options(get_rs(), needle, value)

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input("xr-pick", "value"),
        State("xr-key", "data"),
        prevent_initial_call=True,
    )
    def pick(value, key):
        target = cards.parse_pick(value)
        if target is None:
            return no_update
        if isinstance(key, dict) and (key.get("peptidoform"), _charge(key.get("charge"))) == target:
            return no_update
        return cards.runs_href(base, *target)

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input({"type": "fig", "name": "xr-xic"}, "clickData"),
        Input({"type": "fig", "name": "xr-quant"}, "clickData"),
        State("xr-key", "data"),
        prevent_initial_call=True,
    )
    def open_run(click_xic, click_quant, key):
        from dash import ctx as dash_ctx

        a = _across(get_rs(), key)
        if a is None:
            return no_update
        trig = dash_ctx.triggered_id
        click = (
            click_quant if isinstance(trig, dict) and trig.get("name") == "xr-quant" else click_xic
        )
        return click_target(base, a, click) or no_update


def page_href(base: str, peptidoform_text: str, charge: int) -> str:
    return href(base, "runs", {"peptidoform": peptidoform_text, "charge": charge})
