"""Quant QC page (P1 view 7): are the quantities of this result set sound?

The page shows, per run, the distribution of the quantities and the missing values, the
engine's quant states, the CVs within conditions the user defines, and for experiments
the MaxLFQ matrix with the profile of one protein group. Every summary is the viewer's
(:mod:`mumdia_viewer.data.quantqc`), and every card says what it summarizes.

The run-to-condition mapping lives in the browser: the store ``mv-conditions``
(localStorage) holds ``{run name: condition}``; an empty or missing value means the
suggestion from the mzML file names (:func:`.quantqc.suggest_conditions`). The cards
that depend on it, or on the toolbar, are filled by callbacks. Ids start with ``qq-``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html, no_update

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import quantqc as Q

from . import quant_cards as cards
from . import quant_figures as qf
from .figures import empty_figure
from .icons import icon
from .state import PageContext, href, parse_threshold, stop_label
from .theme import SERIES
from .widgets import graph

log = logging.getLogger(__name__)

CONDITIONS_STORE = "mv-conditions"
GROUP_OPTIONS = 30


def protein_href(base: str, group: str | None) -> str:
    return href(base, "protein", {"group": group}) if group else href(base, "protein")


def _scheme(value: Any) -> str:
    return "dark" if value == "dark" else "light"


def _accepted(keys: Any) -> bool:
    return keys != "all"


def _source(rs: ResultSet, value: Any) -> str:
    return "lfq" if rs.is_experiment and value != "quant" else "quant"


def matrices(
    rs: ResultSet, source: str, accepted: bool, t: float
) -> dict[str, Q.QuantMatrix | str]:
    """The matrix of each level, or the data layer's message when it cannot be read."""
    out: dict[str, Q.QuantMatrix | str] = {}
    for level in Q.LEVELS:
        try:
            out[level] = Q.quant_matrix(rs, level, source, accepted_at=t if accepted else None)
        except ViewerError as exc:
            out[level] = str(exc)
    return out


def _keys_text(accepted: bool, t: float) -> str:
    if accepted:
        return (
            f"accepted keys (precursors at precursor_q ≤ {stop_label(t)}, protein groups at "
            f"pg_q_value ≤ {stop_label(t)})"
        )
    return "every key with a value in any run"


# --------------------------------------------------------------------------- answers


def dist_answer(
    rs: ResultSet, stored: Any, source: Any, keys: Any, t: float, scheme: str
) -> tuple[Any, Any, Any, Any]:
    """(distribution figure, its subtitle, the missing-values body, its subtitle)."""
    src = _source(rs, source)
    accepted = _accepted(keys)
    conditions = Q.resolve_conditions(rs, stored)
    ms = matrices(rs, src, accepted, t)
    good = {k: v for k, v in ms.items() if isinstance(v, Q.QuantMatrix)}
    errors = [f"{qf.LEVEL_TITLES[k]}: {v}" for k, v in ms.items() if isinstance(v, str)]
    dists = {k: Q.quantity_distribution(m) for k, m in good.items()}
    fig = qf.distribution_figure(dists, conditions, scheme)
    label = next(iter(good.values())).source_name if good else Q.SOURCE_NAMES[src]
    sub = (
        f"log10 quantity per run ({label}), over the {_keys_text(accepted, t)} with a value. "
        "Violins: the viewer's histograms; bar: 25th to 75th percentile; dot: median."
    )
    if errors:
        sub += " " + " ".join(errors)
    missing = {k: Q.missing_by_run(m) for k, m in good.items()}
    if not missing:
        body: Any = cards.note(" ".join(errors) or "No quantities.")
    elif rs.is_experiment:
        per = {k: Q.runs_with_value(m) for k, m in good.items()}
        body = graph("qq-missing", qf.missing_figure(missing, per, scheme, accepted=accepted))
    else:
        body = cards.missing_single(missing, accepted, t)
    miss_sub = f"Keys without a value in each run ({label}): of the {_keys_text(accepted, t)}." + (
        " An accepted key without a quantity is missing, never zero."
        if accepted
        else " Every key here has a value in at least one run."
    )
    return fig, sub, body, miss_sub


def cv_results(
    rs: ResultSet, stored: Any, source: Any, keys: Any, need: Any, t: float
) -> tuple[dict[str, Q.CvResult | None], dict[str, str], dict[str, str], list[str]]:
    """(CVs per level, conditions, condition colours, messages)."""
    src = _source(rs, source)
    conditions = Q.resolve_conditions(rs, stored)
    colours = qf.condition_colours(conditions, [r.name for r in rs.runs])
    out: dict[str, Q.CvResult | None] = {}
    errors: list[str] = []
    for level, m in matrices(rs, src, _accepted(keys), t).items():
        if isinstance(m, str):
            out[level] = None
            errors.append(f"{qf.LEVEL_TITLES[level]}: {m}")
        else:
            out[level] = Q.condition_cvs(m, conditions, all_runs=need != "two")
    return out, conditions, colours, errors


def cv_answer(
    rs: ResultSet, stored: Any, source: Any, keys: Any, need: Any, t: float, scheme: str
) -> tuple[Any, Any, Any]:
    """(CV figure, the CV table, the subtitle)."""
    results, _, colours, errors = cv_results(rs, stored, source, keys, need, t)
    if len(rs.runs) < 2:
        fig = empty_figure(
            "A CV needs at least two runs in a condition; this result set has one run.",
            scheme,
            height=200,
        )
        return fig, None, "The CV is the viewer's, over the runs of a condition."
    fig, above = qf.cv_figure(results, colours, scheme)
    first = next((r for r in results.values() if r is not None), None)
    src_name = Q.SOURCE_NAMES[_source(rs, source)]
    sub = f"{Q.CV_RULE} ({src_name}); " + (
        "each CV needs a value in every run of its condition."
        if need != "two"
        else "each CV needs at least two values."
    )
    outside = sum(above.values())
    if outside:
        sub += f" {outside:,} CVs above the plotted range are not drawn (the table counts them)."
    if errors:
        sub += " " + " ".join(errors)
    table = cards.cv_table(results, colours) if first is not None else None
    return fig, table, sub


def heat_answer(
    rs: ResultSet,
    stored: Any,
    keys: Any,
    t: float,
    scale: Any,
    cluster: Any,
    selected: str | None,
    scheme: str,
) -> tuple[Any, str, str]:
    """(LFQ heatmap figure, the count, the subtitle)."""
    accepted = _accepted(keys)
    try:
        m = Q.quant_matrix(rs, "protein", "lfq", accepted_at=t if accepted else None)
    except ViewerError as exc:
        return empty_figure(str(exc), scheme, height=200), "", str(exc)
    conditions = Q.resolve_conditions(rs, stored)
    colours = qf.condition_colours(conditions, list(m.runs))
    h = Q.heatmap(m, conditions, cluster=bool(cluster))
    relative = scale != "log10"
    fig = qf.heatmap_figure(h, colours, scheme, relative=relative, selected=selected)
    what = (
        "log2 MaxLFQ relative to the row's mean over its runs with a value (the viewer's)"
        if relative
        else "log10 MaxLFQ"
    )
    sub = (
        f"{what}; protein groups: {_keys_text(accepted, t)}. {h.row_rule[0].upper()}"
        f"{h.row_rule[1:]}; {h.column_rule}. Grey: no value in the run."
    )
    if h.dropped:
        sub += f" {h.dropped:,} accepted groups without any value are not drawn."
    count = f"{len(h.groups):,} protein groups"
    return fig, count, sub


def group_options(
    rs: ResultSet, text: str | None, value: str | None, keys: Any, t: float
) -> list[dict[str, str]]:
    """The protein groups the select offers: matches of ``text``, and the current value."""
    accepted = _accepted(keys)
    names = Q.find_groups(rs, text, limit=GROUP_OPTIONS, accepted_at=t if accepted else None)
    if value and value not in names:
        names = [value, *names]
    return [{"value": n, "label": n if len(n) <= 60 else n[:57] + "..."} for n in names]


def default_group(rs: ResultSet, wanted: str | None, t: float) -> str | None:
    if wanted:
        return wanted
    for at in (t, None):
        try:
            hits = Q.find_groups(rs, "", limit=1, accepted_at=at)
        except ViewerError:
            return None
        if hits:
            return hits[0]
    return None


def profile_answer(
    rs: ResultSet,
    group: str | None,
    stored: Any,
    source: Any,
    need: Any,
    t: float,
    base: str,
    scheme: str,
) -> tuple[Any, Any, Any, str, str]:
    """(profile figure, header, table, subtitle, protein page address)."""
    if not group:
        fig = empty_figure("Pick a protein group, or click a row of the matrix.", scheme)
        return fig, None, None, "", protein_href(base, None)
    try:
        p = Q.protein_profile(rs, group)
    except ViewerError as exc:
        return empty_figure(str(exc), scheme), None, None, str(exc), protein_href(base, group)
    conditions = Q.resolve_conditions(rs, stored)
    colours = qf.condition_colours(conditions, list(p.runs))
    fig, shown = qf.profile_figure(p, conditions, colours, scheme)
    src = _source(rs, source)
    values = (
        p.lfq
        if src == "lfq" and p.lfq is not None
        else p.quant.set_index("run")["quantity"].reindex(list(p.runs)).to_numpy(dtype=float)
    )
    col = {r: j for j, r in enumerate(p.runs)}
    cvs: dict[str, float | None] = {}
    for cond, runs in Q.condition_groups(conditions, p.runs).items():
        if len(runs) < 2:
            continue
        x = values[[col[r] for r in runs]]
        ok = x[np.isfinite(x)]
        enough = ok.size >= (len(runs) if need != "two" else 2)
        cvs[cond] = float(100.0 * np.std(ok, ddof=1) / np.mean(ok)) if enough else None
    order = [r for runs in Q.condition_groups(conditions, p.runs).values() for r in runs]
    order += [r for r in p.runs if r not in order]
    head = cards.profile_head(p, t, base, cvs)
    table = cards.profile_table(p, conditions, order)
    n_prec = len(p.precursors)
    sub = (
        f"MaxLFQ of the group (line) and its protein_group_quant quantity (dashed, not "
        f"normalized); thin lines: {shown:,} of its {n_prec:,} precursors "
        f"({'MaxLFQ' if p.precursor_lfq is not None else 'quantity'})."
        + (
            " A ringed point has match-between-runs transfers among its features."
            if p.lfq_transferred is not None
            else ""
        )
        + f" CV chips: {Q.SOURCE_NAMES[src]}."
    )
    if not p.in_lfq and rs.is_experiment:
        sub += " The group is not in the LFQ table."
    return fig, head, table, sub, protein_href(base, group)


def states_answer(rs: ResultSet, t: float) -> tuple[Any, str]:
    """(the quant states table, its subtitle)."""
    notes: list[str] = []
    try:
        states = Q.accepted_quant_states(rs, t)
        notes += states.attrs.get("notes") or []
    except ViewerError as exc:
        states = None
        notes.append(str(exc))
    try:
        status = Q.status_matrix(rs)
        notes += status.attrs.get("notes") or []
    except ViewerError as exc:
        status = None
        notes.append(str(exc))
    column = states.attrs.get("q_column", "run_psm_q") if states is not None else "run_psm_q"
    sub = (
        f"Per run: its accepted identifications (target rows with {column} ≤ {stop_label(t)}, "
        "distinct candidate_id"
        + (", native: scored_combined.parquet" if rs.is_experiment else "")
        + ") by their row in the run's peptide_quant, then the engine's rows by quant_status."
    )
    body: list[Any] = []
    if states is not None or status is not None:
        body.append(cards.states_table(states, status, t))
    if notes:
        body.append(cards.note(" ".join(notes), "orange"))
    return html.Div(body), sub


# --------------------------------------------------------------------------- layout


def _quantified_rows(rs: ResultSet) -> dict[str, int]:
    """Rows with a quantity of each quant table, summed over the runs (for the header)."""
    try:
        status = Q.status_matrix(rs)
    except ViewerError:
        return {}
    names = [r.name for r in rs.runs]
    rows: dict[str, int] = {}
    for level, table in (("precursor", "peptide_quant"), ("protein", "protein_group_quant")):
        sub = status[(status["table"] == table) & (status["status"] == "quantified")]
        if len(sub):
            rows[level] = int(sub[names].to_numpy().sum())
    return rows


def hidden_parts(rs: ResultSet, scheme: str) -> Any:
    """Hidden stand-ins for the components this result set does not show.

    Every callback of the page is registered for every result set; a callback whose
    inputs are partly in the layout would fail in the browser, so the editor (a single
    run) and the LFQ cards (a single run) have stand-ins here. Their callbacks answer
    with an empty figure.
    """
    parts: list[Any] = []
    if len(rs.runs) < 2:
        parts += [html.Button(id="qq-cond-reset", n_clicks=0), html.Div(id="qq-cond-state")]
    if not rs.is_experiment:
        parts += [
            graph("qq-heat", empty_figure("", scheme, height=40)),
            graph("qq-profile", empty_figure("", scheme, height=40)),
            dmc.SegmentedControl(id="qq-heat-scale", data=["relative"], value="relative"),
            dmc.Switch(id="qq-heat-cluster", checked=False),
            dmc.Select(id="qq-group", data=[], value=None),
            *[
                html.Div(id=i)
                for i in (
                    "qq-heat-count",
                    "qq-heat-sub",
                    "qq-profile-head",
                    "qq-profile-table",
                    "qq-profile-sub",
                )
            ],
            dcc.Link(id="qq-open", href=protein_href("/", None)),
        ]
    return html.Div(parts, style={"display": "none"}) if parts else None


def single_cv_card(
    rs: ResultSet, suggestion: dict[str, str], files: dict[str, str | None], scheme: str
) -> Any:
    """A single run: why there is no CV, and the condition suggested for the run."""
    run = rs.runs[0].name
    name = files.get(run) or "no mzML recorded"
    return cards.qq_section(
        "CVs within conditions",
        dmc.Text(
            f"A CV needs at least two runs in a condition; this result set has one run. Its "
            f"suggested condition is {suggestion.get(run, '')!r}, from {name}.",
            size="sm",
        ),
        html.Div(
            [
                graph("qq-cv", empty_figure("", scheme, height=60)),
                html.Div(id="qq-cv-table"),
                html.Div(id="qq-cv-sub"),
            ],
            style={"display": "none"},
        ),
        help=f"{Q.CV_RULE}.",
    )


def layout(ctx: PageContext) -> Any:
    t0 = time.perf_counter()
    rs = ctx.rs
    t = ctx.threshold
    experiment = rs.is_experiment
    suggestion = Q.suggest_conditions(rs)
    files = Q.run_files(rs)
    quant_rows = _quantified_rows(rs)
    lfq_keys: dict[str, int] = {}
    if experiment:
        with contextlib.suppress(ViewerError):
            lfq_keys["protein"] = Q.quant_matrix(rs, "protein", "lfq").n_keys
    states_body, states_sub = states_answer(rs, t)
    stores = [
        dcc.Store(id=CONDITIONS_STORE, storage_type="local"),
        dcc.Store(id="qq-suggest", data=suggestion),
        dcc.Store(id="qq-palette", data=list(SERIES)),
        dcc.Store(id="qq-addr"),
    ]
    dist_card = cards.qq_section(
        "Quantity distributions",
        cards.loading(graph("qq-dist", empty_figure("", ctx.scheme, height=340))),
        count=f"{len(rs.runs)} run{'s' if len(rs.runs) != 1 else ''}",
        help="The quantities of each run, by level. Precursors: peptide_quant.quantity (the "
        "top-N fragment area sum), or the precursor MaxLFQ; protein groups: "
        "protein_group_quant.quantity, or the protein MaxLFQ. A missing quantity is never "
        "drawn as zero. Hover a violin for its numbers.",
        subtitle="",
        subtitle_id="qq-dist-sub",
    )
    missing_card = cards.qq_section(
        "Missing values",
        cards.loading(html.Div(id="qq-missing-body", className="qq-missing-body")),
        help="A key is missing in a run when the run has no value for it: no row in the "
        "quant table, a null quantity (not quantifiable), or 0.0 in an LFQ table. With "
        "Accepted keys, an accepted precursor or protein group without a quantity in a run "
        "is missing there.",
        subtitle="",
        subtitle_id="qq-missing-sub",
    )
    cv_card = cards.qq_section(
        "CVs within conditions",
        cards.loading(
            html.Div(
                [
                    graph("qq-cv", empty_figure("", ctx.scheme, height=330)),
                    html.Div(id="qq-cv-table", className="qq-cv-table"),
                ]
            )
        ),
        help=f"{Q.CV_RULE}. The conditions come from the Conditions card. A precursor is a "
        "(peptidoform, charge), a protein group a protein_group string.",
        subtitle="",
        subtitle_id="qq-cv-sub",
    )
    rows: list[Any] = [
        *stores,
        cards.header(ctx, quant_rows, lfq_keys),
        cards.toolbar(ctx),
        dmc.Grid(
            [
                dmc.GridCol(dist_card, span={"base": 12, "lg": 7}),
                dmc.GridCol(missing_card, span={"base": 12, "lg": 5}),
            ],
            gutter="lg",
        ),
    ]
    if len(rs.runs) > 1:
        rows.append(
            dmc.Grid(
                [
                    dmc.GridCol(
                        cards.conditions_card(rs, suggestion, files), span={"base": 12, "lg": 4}
                    ),
                    dmc.GridCol(cv_card, span={"base": 12, "lg": 8}),
                ],
                gutter="lg",
            )
        )
    if experiment:
        group = default_group(rs, ctx.query.get("group"), t)
        heat_card = cards.qq_section(
            "LFQ matrix",
            cards.loading(graph("qq-heat", empty_figure("", ctx.scheme, height=520))),
            count="",
            count_id="qq-heat-count",
            help="lfq_maxlfq.parquet: one row per protein group, one column per run. Click a "
            "row to see its profile. Clustering is off by default; with it on, the rows and "
            "the runs are reordered by the viewer (k-means and average linkage, numpy).",
            subtitle="",
            subtitle_id="qq-heat-sub",
            right=dmc.Group(
                [
                    dmc.SegmentedControl(
                        id="qq-heat-scale",
                        data=[
                            {"value": "relative", "label": "Relative"},
                            {"value": "log10", "label": "log10"},
                        ],
                        value="relative",
                        size="xs",
                        radius="md",
                    ),
                    dmc.Switch(id="qq-heat-cluster", label="Cluster", checked=False, size="sm"),
                ],
                gap="sm",
                wrap="nowrap",
            ),
        )
        profile_card = cards.qq_section(
            "Protein profile",
            dmc.Select(
                id="qq-group",
                data=group_options(rs, None, group, "accepted", t),
                value=group,
                searchable=True,
                searchValue=None,
                nothingFoundMessage="No protein group contains this text",
                placeholder="Find a protein group",
                leftSection=icon("search", 14),
                size="xs",
                radius="md",
                allowDeselect=False,
                limit=GROUP_OPTIONS,
                maxDropdownHeight=300,
                comboboxProps={"shadow": "md"},
                className="qq-group",
            ),
            cards.loading(
                html.Div(
                    [
                        html.Div(id="qq-profile-head", className="qq-profile-head"),
                        graph("qq-profile", empty_figure("", ctx.scheme, height=300)),
                        html.Div(id="qq-profile-table"),
                    ]
                )
            ),
            help="One protein group across the runs: its MaxLFQ, its per-run "
            "protein_group_quant quantity and the MaxLFQ of its precursors. Find a group by "
            "name, or click a row of the LFQ matrix.",
            subtitle="",
            subtitle_id="qq-profile-sub",
            right=cards.open_button(ctx.base, group),
        )
        rows.append(
            dmc.Grid(
                [
                    dmc.GridCol(heat_card, span={"base": 12, "lg": 7}),
                    dmc.GridCol(profile_card, span={"base": 12, "lg": 5}),
                ],
                gutter="lg",
            )
        )
    states_card = cards.qq_section(
        "Quant states",
        html.Div(states_body, id="qq-states-body"),
        count=f"{len(rs.runs)} run{'s' if len(rs.runs) != 1 else ''}",
        help="Why a quantity is missing. An accepted identification can be not selected (the "
        "quant gate uses its own q column: in an experiment the pooled q_value, not "
        "run_psm_q) or not quantifiable (a quant row with a null quantity; quant_status "
        "says why). Hover a header or a count for what it counts.",
        subtitle=states_sub,
        subtitle_id="qq-states-sub",
    )
    rows.append(states_card)
    hidden = hidden_parts(rs, ctx.scheme)
    if hidden is not None:
        rows.append(hidden)
    if len(rs.runs) < 2:
        rows.append(single_cv_card(rs, suggestion, files, ctx.scheme))
    log.debug("quant layout in %.0f ms", (time.perf_counter() - t0) * 1000.0)
    return html.Div(dmc.Stack(rows, gap="lg"), id="qq-root", className="qq-root")


# --------------------------------------------------------------------------- callbacks


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""
    fig = {"type": "fig", "name": "qq-dist"}

    app.clientside_callback(
        ClientsideFunction("mvq", "conditions"),
        Output(CONDITIONS_STORE, "data"),
        Output({"type": "qq-cond", "run": ALL}, "value"),
        Output({"type": "qq-dot", "run": ALL}, "style"),
        Output("qq-cond-state", "children"),
        Input(CONDITIONS_STORE, "data"),
        Input({"type": "qq-cond", "run": ALL}, "value"),
        Input("qq-cond-reset", "n_clicks"),
        State("qq-suggest", "data"),
        State({"type": "qq-cond", "run": ALL}, "id"),
        State("qq-palette", "data"),
    )
    app.clientside_callback(
        ClientsideFunction("mvq", "pill"),
        Output("qq-t", "children"),
        Input("threshold", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvq", "markRow"),
        Output({"type": "fig", "name": "qq-heat"}, "figure", allow_duplicate=True),
        Input("qq-group", "value"),
        State({"type": "fig", "name": "qq-heat"}, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvq", "heatScheme"),
        Output({"type": "fig", "name": "qq-heat"}, "figure", allow_duplicate=True),
        Input("scheme", "data"),
        State({"type": "fig", "name": "qq-heat"}, "figure"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mvq", "address"),
        Output("qq-addr", "data"),
        Input("qq-group", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output(fig, "figure", allow_duplicate=True),
        Output("qq-dist-sub", "children"),
        Output("qq-missing-body", "children"),
        Output("qq-missing-sub", "children"),
        Input(CONDITIONS_STORE, "data"),
        Input("qq-source", "value"),
        Input("qq-keys", "value"),
        Input("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow_dist(stored, source, keys, t, scheme):
        return dist_answer(get_rs(), stored, source, keys, parse_threshold(t), _scheme(scheme))

    @app.callback(
        Output({"type": "fig", "name": "qq-cv"}, "figure", allow_duplicate=True),
        Output("qq-cv-table", "children"),
        Output("qq-cv-sub", "children"),
        Input(CONDITIONS_STORE, "data"),
        Input("qq-source", "value"),
        Input("qq-keys", "value"),
        Input("qq-need", "value"),
        Input("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow_cv(stored, source, keys, need, t, scheme):
        return cv_answer(get_rs(), stored, source, keys, need, parse_threshold(t), _scheme(scheme))

    @app.callback(
        Output({"type": "fig", "name": "qq-heat"}, "figure", allow_duplicate=True),
        Output("qq-heat-count", "children"),
        Output("qq-heat-sub", "children"),
        Input(CONDITIONS_STORE, "data"),
        Input("qq-keys", "value"),
        Input("threshold", "data"),
        Input("qq-heat-scale", "value"),
        Input("qq-heat-cluster", "checked"),
        State("qq-group", "value"),
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow_heat(stored, keys, t, scale, cluster, group, scheme):
        return heat_answer(
            get_rs(), stored, keys, parse_threshold(t), scale, cluster, group, _scheme(scheme)
        )

    @app.callback(
        Output({"type": "fig", "name": "qq-profile"}, "figure", allow_duplicate=True),
        Output("qq-profile-head", "children"),
        Output("qq-profile-table", "children"),
        Output("qq-profile-sub", "children"),
        Output("qq-open", "href"),
        Input("qq-group", "value"),
        Input(CONDITIONS_STORE, "data"),
        Input("qq-source", "value"),
        Input("qq-need", "value"),
        Input("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def follow_profile(group, stored, source, need, t, scheme):
        return profile_answer(
            get_rs(), group, stored, source, need, parse_threshold(t), base, _scheme(scheme)
        )

    @app.callback(
        Output("qq-group", "data"),
        Input("qq-group", "searchValue"),
        State("qq-group", "value"),
        State("qq-keys", "value"),
        State("threshold", "data"),
        prevent_initial_call=True,
    )
    def search_groups(text, value, keys, t):
        return group_options(get_rs(), text, value, keys, parse_threshold(t))

    @app.callback(
        Output("qq-group", "value"),
        Output("qq-group", "data", allow_duplicate=True),
        Input({"type": "fig", "name": "qq-heat"}, "clickData"),
        State("qq-group", "data"),
        prevent_initial_call=True,
    )
    def click_row(click, data):
        try:
            group = str(click["points"][0]["y"])
        except (TypeError, KeyError, IndexError):
            return no_update, no_update
        options = list(data or [])
        if not any(o.get("value") == group for o in options):
            options = [
                {"value": group, "label": group[:57] + "..." if len(group) > 60 else group},
                *options,
            ]
        return group, options

    @app.callback(
        Output("qq-states-body", "children"),
        Output("qq-states-sub", "children"),
        Input("threshold", "data"),
        prevent_initial_call=True,
    )
    def follow_states(t):
        return states_answer(get_rs(), parse_threshold(t))
