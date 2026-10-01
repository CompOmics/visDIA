"""Run overview page (P0 view 1): what was run, and how many identifications it gave."""

from __future__ import annotations

import json
import math
from typing import Any

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import ClientsideFunction, Input, Output, State, dcc, html

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data import counts as counts_mod
from mumdia_viewer.data import overview as overview_mod
from mumdia_viewer.data.entrapment import entrapment_fdp
from mumdia_viewer.data.mbr import mbr_info, transfer_counts
from mumdia_viewer.data.units import UNITS

from . import figures
from .icons import icon
from .state import THRESHOLD_STOPS, PageContext, href, stop_label
from .widgets import (
    chip,
    data_table,
    fmt,
    fmt_bytes,
    frame_table,
    graph,
    section,
    status_badge,
    unit_colour,
)

CARD_UNITS = ("psm", "precursor", "peptide", "protein_group")
UNIT_ICONS = {"psm": "spectrum", "precursor": "peak", "peptide": "flask", "protein_group": "layers"}
# The identification curves: exact counts at 10 points per decade, plus the stops.
CURVE_GRID = tuple(
    sorted({float(f"{q:.6g}") for q in np.logspace(-4, -1, 31)} | set(THRESHOLD_STOPS))
)
# Where a card leads in the identification browser.
CARD_LINKS = {
    "psm": {"unit": "precursor", "q": "q_value"},
    "precursor": {"unit": "precursor"},
    "peptide": {"unit": "peptide"},
    "protein_group": {"unit": "protein_group"},
}


# --------------------------------------------------------------------------- data


def _curves(rs: ResultSet) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for unit in CARD_UNITS:
        try:
            out[unit] = counts_mod.counts_at(rs, unit, CURVE_GRID)
        except (ViewerError, ValueError):
            continue
    return out


def _at(df: pd.DataFrame | None, t: float) -> int | None:
    if df is None or df.empty:
        return None
    hit = df[np.isclose(df["q"].to_numpy(), t, rtol=1e-9, atol=0)]
    return int(hit["count"].iloc[0]) if len(hit) else None


def _slider_data(rs: ResultSet, curves: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Exact target and decoy counts of every card unit at every stop (for the slider)."""
    targets: dict[str, list[int | None]] = {}
    decoys: dict[str, list[int | None]] = {}
    for unit in CARD_UNITS:
        targets[unit] = [_at(curves.get(unit), t) for t in THRESHOLD_STOPS]
        try:
            d = counts_mod.counts_at(rs, unit, THRESHOLD_STOPS, label="decoy")
            decoys[unit] = [int(n) for n in d["count"]]
        except (ViewerError, ValueError):
            decoys[unit] = [None] * len(THRESHOLD_STOPS)
    return {
        "stops": list(THRESHOLD_STOPS),
        "values": [repr(t) for t in THRESHOLD_STOPS],
        "units": list(CARD_UNITS),
        "columns": {u: UNITS[u].q_column for u in CARD_UNITS},
        "targets": targets,
        "decoys": decoys,
    }


def _nearest_stop(t: float) -> int:
    return int(np.argmin([abs(math.log(s) - math.log(t)) for s in THRESHOLD_STOPS]))


# --------------------------------------------------------------------------- header


def _identity(rs: ResultSet, summary: overview_mod.RunSummary) -> Any:
    r = summary.rescore
    models = summary.model_identities or {}
    chips: list[Any] = [
        chip(
            f"MuMDIA {summary.mumdia_version or 'version not recorded'}",
            "indigo",
            tip=f"git {summary.git_sha or 'not recorded'}, committed "
            f"{summary.commit_date or 'date not recorded'}",
            left=icon("bolt", 12),
        ),
        chip(
            f"rescorer {r.classifier or 'not recorded'}",
            "violet",
            tip=f"model {r.model_identity or 'not recorded'}; mode {r.mode}; requested "
            f"{r.classifier_requested or 'not recorded'}"
            + ("; the rescorer fell back to another classifier" if r.fallback else ""),
        ),
        chip(
            f"competition {r.group_by_display or 'not recorded'}",
            "grape",
            tip=f"compete.group_by, from {r.group_by_source or 'no record'}",
        ),
    ]
    if r.fallback:
        chips.append(chip("rescorer fallback", "red", tip="The requested classifier was not used."))
    if models.get("rt_predictor"):
        chips.append(
            chip(f"RT {models['rt_predictor']}", "teal", tip="model_identities.rt_predictor")
        )
    if models.get("fragment_predictor"):
        chips.append(
            chip(
                f"fragments {models['fragment_predictor']}",
                "cyan",
                tip="model_identities.fragment_predictor",
            )
        )
    qf = summary.quant_q_filter
    if qf:
        differs = (
            qf.get("configured")
            and qf.get("effective")
            and str(qf["configured"]).replace("_", "").lower()
            != str(qf["effective"]).replace("_", "").lower()
        )
        text = (
            f"quant filter {qf.get('effective') or qf.get('configured')} ≤ {qf.get('q_threshold')}"
        )
        tip = (
            f"configured {qf.get('configured')}, effective {qf.get('effective')}; "
            f"from {qf.get('source')}"
        )
        chips.append(chip(text, "yellow" if differs else "gray", tip=tip))
    if summary.grouped:
        chips.append(chip(f"grouped extraction, {len(summary.grouped)} groups", "orange"))
    if summary.mbr_strategy and summary.mbr_strategy != "None":
        chips.append(chip(f"MBR {summary.mbr_strategy}", "lime"))

    kind = f"Experiment, {summary.n_runs} runs" if rs.is_experiment else "Single run"
    title = html.Div(
        [
            dmc.Text(kind, className="mv-eyebrow"),
            html.Div(rs.root.name, className="mv-title"),
            dmc.Group(
                [
                    dmc.Text(str(rs.root), className="mv-path", c="dimmed"),
                    dcc.Clipboard(
                        content=str(rs.root),
                        title="Copy the path",
                        style={"cursor": "pointer", "opacity": 0.6},
                    ),
                ],
                gap=6,
                mt=2,
            ),
        ]
    )
    return dmc.Stack([title, dmc.Group(chips, gap=6)], gap="sm")


def _run_stats(rs: ResultSet) -> Any:
    try:
        table = overview_mod.artifact_table(rs)
    except ViewerError:
        table = pd.DataFrame(columns=["key", "rows"])

    def rows_of(prefix: str) -> int | None:
        sel = table[table["key"].astype(str).str.startswith(prefix)]["rows"].dropna()
        return int(sel.sum()) if len(sel) else None

    items = [
        ("scored rows", rs.scored.rows, "Rows of the scored table (every scored candidate)"),
        ("MS2 spectra", rows_of("spectra_ms2"), "Rows of spectra_ms2 (all runs)"),
        (
            "library precursors",
            rows_of("fragment_library_precursors"),
            "Rows of the fragment library precursor table",
        ),
    ]
    tiles = []
    for label, value, tip in items:
        if value is None:
            continue
        tiles.append(
            dmc.Tooltip(
                dmc.Paper(
                    [
                        dmc.Text(
                            label,
                            size="xs",
                            c="dimmed",
                            fw=600,
                            tt="uppercase",
                            style={"letterSpacing": "0.04em"},
                        ),
                        dmc.Text(
                            _compact(value),
                            fw=700,
                            size="xl",
                            style={"fontVariantNumeric": "tabular-nums"},
                        ),
                    ],
                    withBorder=True,
                    px="md",
                    py=8,
                    radius="md",
                ),
                label=f"{tip}: {value:,}",
            )
        )
    return dmc.Group(tiles, gap="sm", wrap="nowrap")


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 10_000:
        return f"{n / 1e3:.0f} k"
    return f"{n:,}"


# --------------------------------------------------------------------------- cards


def _kpi_cards(ctx: PageContext, curves: dict[str, pd.DataFrame]) -> Any:
    rs, t = ctx.rs, ctx.threshold
    try:
        counts = {c.unit: c for c in counts_mod.unit_counts(rs, t)}
        error = None
    except (ViewerError, ValueError) as exc:
        counts, error = {}, str(exc)
    cards = []
    for unit in CARD_UNITS:
        u = UNITS[unit]
        c = counts.get(unit)
        colour = unit_colour(unit)
        curve = curves.get(unit)
        spark = [int(v) for v in curve["count"]] if curve is not None and len(curve) else [0, 0]
        decoys = (
            f"{c.n_decoy:,} {'decoy passes' if c.n_decoy == 1 else 'decoys pass'} the same cut"
            if c is not None
            else ""
        )
        tip = (c.label + ". " + c.note) if c is not None else (error or "not available")
        card = dmc.Card(
            [
                dmc.Group(
                    [
                        dmc.Group(
                            [
                                dmc.ThemeIcon(
                                    icon(UNIT_ICONS[unit], 18),
                                    variant="light",
                                    color=colour,
                                    size=34,
                                    radius="md",
                                ),
                                dmc.Text(u.plural, fw=650, size="sm"),
                            ],
                            gap=10,
                        ),
                        dmc.Tooltip(
                            dmc.ThemeIcon(
                                icon("info", 15), variant="subtle", color="gray", size="sm"
                            ),
                            label=tip,
                            position="bottom-end",
                        ),
                    ],
                    justify="space-between",
                ),
                html.Div(
                    fmt(c.n_target) if c is not None else "-",
                    id=f"kpi-n-{unit}",
                    className="mv-kpi-value",
                    style={"marginTop": "12px"},
                ),
                dmc.Badge(
                    f"{u.q_column} ≤ {stop_label(t)}",
                    id=f"kpi-q-{unit}",
                    color=colour,
                    variant="light",
                    className="mv-kpi-q",
                    style={"textTransform": "none"},
                    mt=6,
                ),
                dmc.Text(u.distinct_label, size="xs", c="dimmed", mt=8),
                dmc.Text(decoys, id=f"kpi-d-{unit}", size="xs", c="dimmed"),
                dmc.Sparkline(
                    data=spark,
                    color=f"{colour}.6",
                    h=46,
                    curveType="monotone",
                    fillOpacity=0.35,
                    strokeWidth=2,
                    withGradient=True,
                    mt="sm",
                ),
            ],
            className="mv-kpi",
            p="lg",
        )
        cards.append(
            dcc.Link(
                card,
                href=href(ctx.base, "identifications", CARD_LINKS[unit]),
                style={"textDecoration": "none", "color": "inherit"},
            )
        )
    return dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 4}, spacing="lg")


def _slider(ctx: PageContext, data: dict[str, Any]) -> Any:
    index = _nearest_stop(ctx.threshold)
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.Group(
                        [
                            dmc.Text("Explore the threshold", fw=650, size="sm"),
                            dmc.Badge(
                                f"q ≤ {stop_label(THRESHOLD_STOPS[index])}",
                                id="ov-slider-label",
                                variant="filled",
                                size="lg",
                                style={"textTransform": "none"},
                            ),
                        ],
                        gap="sm",
                    ),
                    dmc.Text(
                        "Drag: the cards show the exact counts at each stop. The page "
                        "follows when you let go.",
                        size="xs",
                        c="dimmed",
                    ),
                ],
                justify="space-between",
            ),
            dmc.Slider(
                id="ov-slider",
                min=0,
                max=len(THRESHOLD_STOPS) - 1,
                step=1,
                value=index,
                marks=[{"value": i, "label": stop_label(s)} for i, s in enumerate(THRESHOLD_STOPS)],
                restrictToMarks=True,
                updatemode="drag",
                size="lg",
                mt="md",
                mb="md",
                styles={"label": {"display": "none"}},
            ),
            dcc.Store(id="ov-slider-data", data=data),
        ],
        className="mv-slider-strip",
    )


def _engine_card(rs: ResultSet) -> Any:
    try:
        checks = counts_mod.engine_check(rs)
    except ViewerError as exc:
        return section("Engine agreement", dmc.Text(str(exc), size="sm", c="dimmed"))
    rows = []
    for c in checks:
        name = UNITS[c.unit].plural if c.unit in UNITS else c.unit.replace("_", " ")
        if c.engine is None:
            state = dmc.Tooltip(
                dmc.ThemeIcon(
                    icon("info", 14), color="gray", variant="light", size="sm", radius="xl"
                ),
                label="The report has no statistic for this unit.",
            )
        elif c.equal:
            state = dmc.ThemeIcon(
                icon("check", 14), color="green", variant="light", size="sm", radius="xl"
            )
        else:
            state = dmc.ThemeIcon(
                icon("x", 14), color="red", variant="light", size="sm", radius="xl"
            )
        rows.append([state, name, c.viewer, c.engine if c.engine is not None else ""])
    all_equal = bool(checks) and all(c.equal for c in checks if c.engine is not None)
    summary = dmc.Alert(
        "The viewer's counts at q ≤ 0.01 equal the rescore report's statistics."
        if all_equal
        else "At least one count differs from the rescore report. Check the notes of that "
        "card before you use it.",
        color="green" if all_equal else "red",
        variant="light",
        icon=icon("check" if all_equal else "alert", 18),
        mb="sm",
        p="xs",
    )
    return section(
        "Engine agreement",
        summary,
        data_table(rows, ["", "unit", "viewer", "engine report"], numeric=[2, 3]),
        subtitle="Counts at q ≤ 0.01 (the report's fixed threshold)",
    )


# --------------------------------------------------------------------------- sections


def _threshold_sections(ctx: PageContext) -> list[Any]:
    """The sections that depend on the threshold (rebuilt when it changes)."""
    rs, t, scheme = ctx.rs, ctx.threshold, ctx.scheme
    out: list[Any] = []
    if rs.is_experiment:
        try:
            per_run = counts_mod.per_run_counts(rs, t)
            labels = per_run.attrs.get("labels", {})
            shown = per_run.drop(columns=[c for c in ("source",) if c in per_run])
            out.append(
                section(
                    "Per run",
                    graph("ov-per-run", figures.per_run_figure(per_run, labels, scheme)),
                    frame_table(
                        shown,
                        labels={"run": "run", "decoy_psms": "decoy PSMs", **figures.PER_RUN_NAMES},
                    ),
                    dmc.Text(per_run.attrs.get("note", ""), size="xs", c="dimmed", mt="xs"),
                    subtitle=f"run_psm_q ≤ {stop_label(t)}: PSM-level FDR within each run. "
                    "Distinct precursors, peptides and protein groups per run are derived "
                    "from the accepted PSMs.",
                )
            )
        except (ViewerError, ValueError) as exc:
            out.append(section("Per run", dmc.Text(str(exc), size="sm", c="dimmed")))
    try:
        fdp = entrapment_fdp(rs, t)
    except ViewerError as exc:
        fdp = []
        out.append(section("Entrapment FDP", dmc.Text(str(exc), size="sm", c="dimmed")))
    if fdp:
        rows = [
            [r.unit, r.run, r.spike_ins, r.real, r.ratio, r.fdp, r.largest_accepted_q] for r in fdp
        ]
        out.append(
            section(
                "Entrapment FDP",
                data_table(
                    rows,
                    [
                        "unit",
                        "run",
                        "spike-ins (E)",
                        "real targets (R)",
                        "ratio r",
                        "FDP",
                        "largest accepted q",
                    ],
                    numeric=[2, 3, 4, 5, 6],
                ),
                dmc.Text(fdp[0].label, size="xs", c="dimmed", mt="xs"),
                subtitle="FDP = (r * E + 1) / R at the threshold",
            )
        )
    if rs.is_experiment:
        info = mbr_info(rs)
        if info.ran:
            out.append(
                section(
                    "Match-between-runs",
                    frame_table(transfer_counts(rs, t)),
                    subtitle=f"strategy {info.strategy}; {fmt(info.n_transfers)} transfers",
                )
            )
    return out


def _timings_card(rs: ResultSet, scheme: str) -> Any:
    try:
        df = overview_mod.stage_timings_table(rs)
    except ViewerError as exc:
        return section("Stage timings", dmc.Text(str(exc), size="sm", c="dimmed"))
    timed = df[df["elapsed_s"].notna()]
    total = float(timed["elapsed_s"].sum()) if len(timed) else 0.0
    untimed = df[df["elapsed_s"].isna()]
    note = "; ".join(
        f"Not timed: {r['artifacts']} ({r['source']}; {r['note']})"
        if r.get("note")
        else f"Not timed: {r['artifacts']} ({r['source']})"
        for r in untimed.to_dict("records")
    )
    return section(
        "Stage timings",
        graph("ov-timings", figures.stage_timings_figure(df, scheme)),
        dmc.Text(note, size="xs", c="dimmed") if note else None,
        subtitle=f"{total:,.0f} s recorded in the stage reports",
    )


def _inputs_tab(rs: ResultSet) -> Any:
    df = overview_mod.inputs_table(rs)
    rows = []
    for r in df.to_dict("records"):
        path = r.get("resolved_path") or r.get("recorded_path") or ""
        rows.append(
            [
                dmc.Text(r.get("key"), fw=600, size="sm"),
                dmc.Tooltip(
                    dmc.Text(str(path), className="mv-path"),
                    label=f"recorded as {r.get('recorded_path')}",
                ),
                fmt_bytes(r.get("bytes")),
                status_badge(r.get("status")),
                dmc.Tooltip(
                    dmc.Text(str(r.get("content_hash") or "")[:10], className="mv-path"),
                    label=f"blake3 {r.get('content_hash') or 'not recorded'}",
                ),
                dmc.Text(str(r.get("note") or ""), size="xs", c="dimmed"),
            ]
        )
    return data_table(
        rows,
        ["input", "path", "size", "status", "content hash", "note"],
        numeric=[2],
        min_width=900,
    )


def _artifacts_tab(rs: ResultSet) -> Any:
    df = overview_mod.artifact_table(rs)
    rows = []
    for r in df.to_dict("records"):
        rows.append(
            [
                dmc.Text(str(r.get("scope")), className="mv-path"),
                dmc.Text(str(r.get("key")), fw=600, size="sm"),
                str(r.get("version") or ""),
                "" if pd.isna(r.get("rows")) else int(r["rows"]),
                status_badge(r.get("status")),
                str(r.get("resolution") or ""),
                str(r.get("stage") or ""),
                dmc.Text(str(r.get("path") or ""), className="mv-path"),
                dmc.Text(str(r.get("error") or ""), size="xs", c="red") if r.get("error") else "",
            ]
        )
    return data_table(
        rows,
        ["scope", "artifact", "schema", "rows", "status", "resolution", "stage", "path", ""],
        numeric=[3],
        max_height=520,
        min_width=1100,
    )


def _report_tab(rs: ResultSet) -> Any:
    numbers = overview_mod.engine_report_numbers(rs)
    if not numbers:
        return dmc.Text("No report numbers were found.", c="dimmed", size="sm")
    rows = [
        [
            dmc.Text(n.key, className="mv-path"),
            n.value if isinstance(n.value, int | float) else str(n.value),
            n.label,
            dmc.Text(n.source, size="xs", c="dimmed"),
        ]
        for n in numbers
    ]
    return data_table(rows, ["key", "value", "meaning", "source"], numeric=[1])


def _tabs(rs: ResultSet, summary: overview_mod.RunSummary) -> Any:
    config = json.dumps(summary.config, indent=2, sort_keys=True) if summary.config else "{}"
    cli = " \\\n    ".join(summary.cli_args) if summary.cli_args else "not recorded"
    panels = [
        ("inputs", "Inputs", _inputs_tab),
        ("artifacts", "Artifacts", _artifacts_tab),
        ("report", "Engine report", _report_tab),
    ]
    tabs = [dmc.TabsTab(label, value=key) for key, label, _ in panels]
    tabs += [dmc.TabsTab("Configuration", value="config"), dmc.TabsTab("Command line", value="cli")]
    content = []
    for key, _, build in panels:
        try:
            body = build(rs)
        except ViewerError as exc:
            body = dmc.Text(str(exc), c="dimmed", size="sm")
        content.append(dmc.TabsPanel(body, value=key, pt="md"))
    content.append(
        dmc.TabsPanel(
            [
                dmc.Text(
                    f"config_json of the manifest; hash {summary.config_hash or 'not recorded'}",
                    size="xs",
                    c="dimmed",
                    mb="xs",
                ),
                dmc.ScrollArea(
                    dmc.CodeHighlight(code=config, language="json", withCopyButton=True),
                    h=480,
                    type="auto",
                ),
            ],
            value="config",
            pt="md",
        )
    )
    content.append(
        dmc.TabsPanel(
            dmc.CodeHighlight(code=cli, language="bash", withCopyButton=True), value="cli", pt="md"
        )
    )
    return dmc.Card(
        dmc.Tabs([dmc.TabsList(tabs), *content], value="inputs", variant="outline", radius="md"),
        p="md",
    )


# --------------------------------------------------------------------------- page


def _axis_switch(id_: str) -> Any:
    return dmc.SegmentedControl(
        id=id_,
        data=[{"value": "linear", "label": "Linear"}, {"value": "log", "label": "Log"}],
        value="linear",
        size="xs",
        radius="xl",
    )


def layout(ctx: PageContext) -> Any:
    rs, scheme = ctx.rs, ctx.scheme
    summary = overview_mod.run_summary(rs)
    curves = _curves(rs)
    labels = {u: f"{UNITS[u].plural} ({UNITS[u].q_column})" for u in curves}
    curve_fig = figures.id_curves_figure(
        curves,
        THRESHOLD_STOPS[_nearest_stop(ctx.threshold)],
        labels,
        scheme,
        marker_label=f"q ≤ {stop_label(THRESHOLD_STOPS[_nearest_stop(ctx.threshold)])}",
    )
    try:
        hist = counts_mod.score_histogram(rs)
    except ViewerError:
        hist = None
    hist_card = section(
        "Score distribution",
        graph("ov-hist", figures.score_histogram_figure(hist, scheme)),
        subtitle="All scored rows by label, not filtered by q",
        right=_axis_switch("ov-hist-axis"),
    )
    curves_card = section(
        "Identifications against the threshold",
        graph("ov-curves", curve_fig),
        right=_axis_switch("ov-curves-axis"),
        subtitle="Exact counts of the engine's q columns at 37 thresholds; no q value is "
        "recomputed. Click a legend entry to hide a unit.",
    )
    return dmc.Stack(
        [
            dmc.Group(
                [_identity(rs, summary), _run_stats(rs)], justify="space-between", align="flex-end"
            ),
            html.Div(_kpi_cards(ctx, curves), id="ov-kpis"),
            _slider(ctx, _slider_data(rs, curves)),
            dmc.Grid(
                [
                    dmc.GridCol(curves_card, span={"base": 12, "lg": 7}),
                    dmc.GridCol(hist_card, span={"base": 12, "lg": 5}),
                ],
                gutter="lg",
            ),
            html.Div(dmc.Stack(_threshold_sections(ctx), gap="lg"), id="ov-threshold-sections"),
            dmc.Grid(
                [
                    dmc.GridCol(_timings_card(rs, scheme), span={"base": 12, "lg": 7}),
                    dmc.GridCol(_engine_card(rs), span={"base": 12, "lg": 5}),
                ],
                gutter="lg",
            ),
            _tabs(rs, summary),
        ],
        gap="lg",
    )


def register(app, get_rs, base: str) -> None:
    app.clientside_callback(
        ClientsideFunction("mv", "slide"),
        *[Output(f"kpi-n-{u}", "children") for u in CARD_UNITS],
        *[Output(f"kpi-q-{u}", "children") for u in CARD_UNITS],
        *[Output(f"kpi-d-{u}", "children") for u in CARD_UNITS],
        Output("ov-slider-label", "children"),
        Output({"type": "fig", "name": "ov-curves"}, "figure", allow_duplicate=True),
        Input("ov-slider", "value"),
        State("ov-slider-data", "data"),
        State({"type": "fig", "name": "ov-curves"}, "figure"),
        State("q-select", "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mv", "sliderIndex"),
        Output("ov-slider", "value"),
        Input("threshold", "data"),
        State("ov-slider-data", "data"),
        prevent_initial_call=True,
    )
    for name in ("ov-hist", "ov-curves"):
        app.clientside_callback(
            ClientsideFunction("mv", "axisType"),
            Output({"type": "fig", "name": name}, "figure", allow_duplicate=True),
            Input(f"{name}-axis", "value"),
            State({"type": "fig", "name": name}, "figure"),
            prevent_initial_call=True,
        )

    @app.callback(
        Output("ov-kpis", "children"),
        Output("ov-threshold-sections", "children"),
        Input("threshold", "data"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    def follow_threshold(t, scheme):
        from .state import parse_threshold

        ctx = PageContext(
            rs=get_rs(),
            base=base,
            threshold=parse_threshold(t),
            scheme="dark" if scheme == "dark" else "light",
        )
        return _kpi_cards(ctx, _curves(ctx.rs)), dmc.Stack(_threshold_sections(ctx), gap="lg")
