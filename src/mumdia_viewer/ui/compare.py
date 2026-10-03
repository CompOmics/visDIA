"""Compare two result sets (P2 view 10): how we validate engine changes.

Address: ``compare`` (``compare?unit=peptide`` selects the unit of the lower part). The
page needs a second result set: ``mumdia-viewer A --compare B`` serves A here and a full
viewer of B at ``<base>b/`` on the same server (:func:`.app.create_compare_apps`).
Without it, the page shows the exact command.

From the top:

* the two result sets side by side (versions, the engine's commit, models,
  configuration hash, inputs, and each side's identification counts at the header
  threshold) and the configuration keys that differ;
* the identification overlap of precursors, peptides and protein groups, each side on
  its own q column, as bars (only A | both | only B);
* for the selected unit: the scores of the shared identifications (A against B), the
  quantities of the shared precursors (per run when runs searched the same mzML, else
  pooled), and the identifications unique to each side, each row linked to its page in
  A or in B.

All numbers come from :mod:`mumdia_viewer.data.compare`, which matches the sides by
strings (the libraries can differ) and memoises per pair of result sets.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html, no_update
from dash import ctx as dash_ctx

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import compare as cd

from . import compare_cards as cards
from . import compare_figures as cf
from .state import PageContext, href, parse_threshold
from .widgets import empty, section

SCORE_FIG = {"type": "fig", "name": cards.SCORE_GRAPH}
QUANT_FIG = {"type": "fig", "name": cards.QUANT_GRAPH}
UNIT_COLOURS = {"precursor": "#7950f2", "peptide": "#12b886", "protein_group": "#e64980"}

# Each app registers the base of the other side's viewer here (keyed by its own base),
# so layout() can link into B without the app object.
_B_BASES: dict[str, Callable[[], str | None]] = {}


def _scheme(value: Any) -> str:
    return "dark" if value == "dark" else "light"


def b_base_of(base: str) -> str | None:
    get = _B_BASES.get(base)
    return get() if get is not None else None


def unit_of(query: dict[str, str] | None) -> str:
    unit = (query or {}).get("unit")
    return unit if unit in cd.COMPARE_UNITS else "precursor"


def default_pick(pairs: list[cd.RunPair]) -> str:
    if not pairs:
        return "pooled"
    return "all" if len(pairs) > 1 else "0"


def chosen_pairs(pairs: list[cd.RunPair], pick: str | None) -> list[cd.RunPair] | None:
    """The run pairs of the quantity scatter: all, one, or none (pooled)."""
    if not pairs or pick == "pooled":
        return None
    if pick in (None, "all"):
        return pairs
    try:
        return [pairs[int(pick)]]
    except (ValueError, IndexError):
        return pairs


def score_range(df: pd.DataFrame) -> tuple[float, float]:
    v = df["score"].to_numpy(float) if len(df) else np.array([])
    v = v[np.isfinite(v)]
    if not v.size:
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    return (lo, hi if hi > lo else lo + 1.0)


def _quant_parts(a: ResultSet, b: ResultSet, t: float, pick: str, scheme: str) -> Any:
    pairs = cd.run_pairs(a, b)
    try:
        df = cd.quantity_pairs(a, b, t, chosen_pairs(pairs, pick))
    except (ViewerError, OSError, KeyError) as exc:
        return section("Quantity of shared precursors", empty(f"No quantities: {exc}"))
    fig = cf.quantity_figure(df, scheme=scheme, a_label=a.root.name, b_label=b.root.name)
    return cards.quant_card(fig, df, pairs=pairs, pick=pick)


def score_part(a: ResultSet, b: ResultSet, *, unit: str, t: float, scheme: str) -> Any:
    """The score scatter of the shared keys of ``unit``."""
    shared = cd.shared_keys(a, b, unit, t)
    fig = cf.score_figure(
        shared,
        unit_noun=cd.SPEC[unit].plural,
        a_label=a.manifest.model_identities.get("rescorer") or a.root.name,
        b_label=b.manifest.model_identities.get("rescorer") or b.root.name,
        scheme=scheme,
        colour=UNIT_COLOURS[unit],
    )
    rho = cd.spearman(shared["a_score"], shared["b_score"]) if len(shared) else None
    return cards.score_card(
        fig,
        unit=unit,
        n_shared=len(shared),
        n_drawn=min(len(shared), cf.MAX_POINTS),
        rho=rho,
        a=a,
        b=b,
    )


def tables_part(
    a: ResultSet, b: ResultSet, *, unit: str, t: float, base: str, b_base: str | None
) -> Any:
    """The keys of ``unit`` unique to each side, side by side."""
    ua = cd.unique_keys(a, b, unit, t, "a")
    ub = cd.unique_keys(a, b, unit, t, "b")
    ka, kb = cd.unit_keys(a, unit, t), cd.unit_keys(b, unit, t)
    return dmc.Grid(
        [
            dmc.GridCol(
                cards.unique_card(
                    "a", a, ua, unit=unit, t=t, base=base, score_range=score_range(ka)
                ),
                span={"base": 12, "lg": 6},
            ),
            dmc.GridCol(
                cards.unique_card(
                    "b", b, ub, unit=unit, t=t, base=b_base, score_range=score_range(kb)
                ),
                span={"base": 12, "lg": 6},
            ),
        ],
        gutter="lg",
    )


def detail(
    a: ResultSet,
    b: ResultSet,
    *,
    unit: str,
    t: float,
    scheme: str,
    base: str,
    b_base: str | None,
    pick: str | None = None,
) -> Any:
    """The lower part of the page for one unit: scatters, the pick strip, the tables.

    The quantity scatter does not depend on the unit, so a change of unit replaces only
    the score scatter and the tables (their slots: ``cmp-score-slot``, ``cmp-tables``).
    """
    pairs = cd.run_pairs(a, b)
    quant = _quant_parts(a, b, t, pick or default_pick(pairs), scheme)
    return dmc.Stack(
        [
            dmc.Grid(
                [
                    dmc.GridCol(
                        html.Div(
                            score_part(a, b, unit=unit, t=t, scheme=scheme), id="cmp-score-slot"
                        ),
                        span={"base": 12, "md": 6},
                    ),
                    dmc.GridCol(html.Div(quant, id="cmp-quant-slot"), span={"base": 12, "md": 6}),
                ],
                gutter="lg",
            ),
            html.Div(cards.pick_strip(None), id="cmp-pick"),
            html.Div(tables_part(a, b, unit=unit, t=t, base=base, b_base=b_base), id="cmp-tables"),
        ],
        gap="lg",
    )


# --------------------------------------------------------------------------- layout


def _no_compare(ctx: PageContext) -> Any:
    return dmc.Stack(
        [
            html.Div(
                [
                    dmc.Text("Compare · two result sets", className="mv-eyebrow"),
                    html.Div("Compare with another result set", className="mv-title"),
                    dmc.Text(
                        "Validate an engine change: two MuMDIA versions, two configurations or two "
                        "libraries on the same data.",
                        size="sm",
                        c="dimmed",
                        mt=4,
                    ),
                ]
            ),
            cards.start_card(ctx.rs),
        ],
        gap="lg",
        className="cmp-page",
    )


def layout(ctx: PageContext) -> Any:
    a, b = ctx.rs, ctx.compare
    if b is None:
        return _no_compare(ctx)
    t0 = time.perf_counter()
    t, base = ctx.threshold, ctx.base
    unit = unit_of(ctx.query)
    b_base = b_base_of(base)
    rows = cd.provenance(a, b)
    diffs = cd.config_diff(a, b)
    counts = (cd.side_counts(a, t), cd.side_counts(b, t))
    overlaps = cd.overlap(a, b, t)
    top_ms = (time.perf_counter() - t0) * 1000.0
    return dmc.Stack(
        [
            cards.hero(
                a, b, base=base, b_base=b_base, chips=cards.verdict_chips(a, b, rows, len(diffs))
            ),
            # The overlap first: the verdict chips above say whether engine, configuration
            # and library differ; the two cards below say how.
            cards.overlap_card(overlaps, unit, t),
            dmc.Grid(
                [
                    dmc.GridCol(
                        cards.provenance_card(a, b, rows, counts, t), span={"base": 12, "lg": 7}
                    ),
                    dmc.GridCol(cards.config_card(a, b, diffs), span={"base": 12, "lg": 5}),
                ],
                gutter="lg",
            ),
            cards.unit_bar(unit, next(o for o in overlaps if o.unit == unit)),
            # Filled by a callback right after the page shows (_detail): the scatters and
            # the tables need the other side's full table for the unique keys.
            html.Div(cards.detail_skeleton(), id="cmp-detail"),
            dcc.Store(id="cmp-top-ms", data=round(top_ms)),
            dcc.Store(id="cmp-addr"),
        ],
        gap="lg",
        className="cmp-page",
    )


def footer(top_ms: float | None, detail_ms: float) -> Any:
    top = f"provenance, counts and overlap {top_ms:.0f} ms; " if top_ms is not None else ""
    return html.Div(
        f"Built in {top}scatters and tables {detail_ms:.0f} ms (server time).",
        className="cmp-note cmp-footer",
        id="cmp-footer",
    )


# --------------------------------------------------------------------------- callbacks


def _pick_item(
    base: str,
    b_base: str | None,
    unit: str,
    row: Any,
    *,
    a_run: str,
    a_cid: Any,
    b_run: str,
    b_cid: Any,
    detail_text: str,
) -> dict[str, Any]:
    def link(side_base: str | None, run: str, cid: Any) -> str | None:
        if side_base is None:
            return None
        if unit == "protein_group":
            return href(side_base, "protein", {"group": row["key"]})
        try:
            value = float(cid)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return href(side_base, "precursor", {"run": run, "cid": int(value)})

    pep = unit == "precursor"
    return {
        "text": str(row["a_peptidoform"] if pep and "a_peptidoform" in row else row["key"]),
        "pep": pep,
        "detail": detail_text,
        "a_href": link(base, a_run, a_cid),
        "b_href": link(b_base, b_run, b_cid),
    }


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""
    _B_BASES[base] = getattr(app, "mv_compare_base", lambda: None)

    def other() -> ResultSet | None:
        get = getattr(app, "mv_compare", None)
        return get() if get is not None else None

    app.clientside_callback(
        ClientsideFunction("mvx", "rowUnit"),
        Output("cmp-unit", "value"),
        Input({"type": "cmp-ov-row", "unit": ALL}, "n_clicks"),
        State("cmp-unit", "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvx", "address"),
        Output("cmp-addr", "data"),
        Input("cmp-unit", "value"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction("mvx", "activeRow"),
        Output({"type": "cmp-ov-row", "unit": ALL}, "className"),
        Input("cmp-unit", "value"),
        State({"type": "cmp-ov-row", "unit": ALL}, "id"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("cmp-overlap", "children"),
        Output("cmp-overlap-t", "children"),
        Output("cmp-counts", "children"),
        Output("cmp-counts-title", "children"),
        Input("threshold", "data"),
        State("cmp-unit", "value"),
        prevent_initial_call=True,
    )
    def _top(t_value, unit_value):
        b = other()
        if b is None:
            return (no_update,) * 4
        a = get_rs()
        t = parse_threshold(t_value)
        unit = unit_value if unit_value in cd.COMPARE_UNITS else "precursor"
        overlaps = cd.overlap(a, b, t)
        counts = (cd.side_counts(a, t), cd.side_counts(b, t))
        return (
            cards.overlap_body(overlaps, unit, t),
            cards.t_text(t),
            cards.count_rows(*counts, t),
            f"Identifications at {cards.t_text(t)}",
        )

    @app.callback(
        Output("cmp-detail", "children"),
        Output("cmp-unit-summary", "children"),
        Input("threshold", "data"),
        State("cmp-unit", "value"),
        State("scheme", "data"),
        State("cmp-top-ms", "data"),
    )
    def _detail(t_value, unit_value, scheme, top_ms):
        """The lower part: on the first view of the page and when the threshold moves."""
        b = other()
        if b is None:
            return no_update, no_update
        tick = time.perf_counter()
        a = get_rs()
        t = parse_threshold(t_value)
        unit = unit_value if unit_value in cd.COMPARE_UNITS else "precursor"
        try:
            lower = detail(
                a, b, unit=unit, t=t, scheme=_scheme(scheme), base=base, b_base=b_base_of(base)
            )
            overlap = next(o for o in cd.overlap(a, b, t) if o.unit == unit)
        except (ViewerError, OSError) as exc:
            return section("Shared and unique", empty(f"The comparison failed: {exc}")), ""
        ms = (time.perf_counter() - tick) * 1000.0
        first = dash_ctx.triggered_id is None
        return (
            html.Div([lower, footer(top_ms if first else None, ms)]),
            cards.unit_summary(overlap),
        )

    @app.callback(
        Output("cmp-score-slot", "children"),
        Output("cmp-tables", "children"),
        Output("cmp-unit-summary", "children", allow_duplicate=True),
        Output("cmp-pick", "children", allow_duplicate=True),
        Input("cmp-unit", "value"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _unit(unit_value, t_value, scheme):
        """Another unit: the score scatter and the tables (the quantities stay)."""
        b = other()
        if b is None:
            return (no_update,) * 4
        a = get_rs()
        t = parse_threshold(t_value)
        unit = unit_value if unit_value in cd.COMPARE_UNITS else "precursor"
        overlap = next(o for o in cd.overlap(a, b, t) if o.unit == unit)
        return (
            score_part(a, b, unit=unit, t=t, scheme=_scheme(scheme)),
            tables_part(a, b, unit=unit, t=t, base=base, b_base=b_base_of(base)),
            cards.unit_summary(overlap),
            None,
        )

    @app.callback(
        Output("cmp-quant-slot", "children"),
        Input("cmp-pair", "value"),
        State("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def _pair(pick, t_value, scheme):
        b = other()
        if b is None or not pick:
            return no_update
        return _quant_parts(get_rs(), b, parse_threshold(t_value), str(pick), _scheme(scheme))

    @app.callback(
        Output("cmp-pick", "children"),
        Input(SCORE_FIG, "clickData"),
        Input(QUANT_FIG, "clickData"),
        State("threshold", "data"),
        State("cmp-unit", "value"),
        State("cmp-pair", "value"),
        prevent_initial_call=True,
    )
    def _pick(score_click, quant_click, t_value, unit_value, pick):
        b = other()
        if b is None:
            return no_update
        a = get_rs()
        t = parse_threshold(t_value)
        unit = unit_value if unit_value in cd.COMPARE_UNITS else "precursor"
        trigger = dash_ctx.triggered_id
        click = quant_click if trigger == QUANT_FIG else score_click
        try:
            index = int(click["points"][0]["customdata"])
        except (TypeError, KeyError, IndexError, ValueError):
            return no_update
        b_base = b_base_of(base)
        if trigger == QUANT_FIG:
            df = cd.quantity_pairs(a, b, t, chosen_pairs(cd.run_pairs(a, b), pick))
            if not 0 <= index < len(df):
                return no_update
            r = df.iloc[index]
            row = {"key": r["key"], "a_peptidoform": r["peptidoform"]}
            item = _pick_item(
                base,
                b_base,
                "precursor",
                row,
                a_run=r["a_run"],
                a_cid=r["a_cid"],
                b_run=r["b_run"],
                b_cid=r["b_cid"],
                detail_text=f"A {r['a_quantity']:.4g} · B {r['b_quantity']:.4g}",
            )
        else:
            shared = cd.shared_keys(a, b, unit, t)
            if not 0 <= index < len(shared):
                return no_update
            r = shared.iloc[index]
            item = _pick_item(
                base,
                b_base,
                unit,
                r,
                a_run=r["a_run"],
                a_cid=r["a_cid"],
                b_run=r["b_run"],
                b_cid=r["b_cid"],
                detail_text=(
                    f"A score {r['a_score']:.4f} (q {r['a_q']:.3g}) · "
                    f"B score {r['b_score']:.4f} (q {r['b_q']:.3g})"
                ),
            )
        return cards.pick_strip(item)

    def _export(side: str):
        @app.callback(
            Output(f"cmp-download-{side}", "data"),
            Input(f"cmp-export-{side}", "n_clicks"),
            State("threshold", "data"),
            State("cmp-unit", "value"),
            prevent_initial_call=True,
        )
        def run(n, t_value, unit_value):
            b = other()
            if not n or b is None:
                return no_update
            unit = unit_value if unit_value in cd.COMPARE_UNITS else "precursor"
            t = parse_threshold(t_value)
            df = cards.export_frame(cd.unique_keys(get_rs(), b, unit, t, side), unit)
            name = f"only_in_{side.upper()}_{cd.SPEC[unit].plural.replace(' ', '_')}_q{t:g}.tsv"
            return dcc.send_string(df.to_csv(sep="\t", index=False), name)

        return run

    _export("a")
    _export("b")
