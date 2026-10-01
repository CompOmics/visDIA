"""Components of the identification page: the toolbar, the linked panels, the preview.

The panels follow PeptideShaker's Overview tab: a title bar with the row count and an
info tooltip (one plain line, then the data layer's description), a dense grid, and the
preview of the selected precursor (:func:`mumdia_viewer.ui.detail.preview`) at the
bottom. Every number shown comes from the data layer; this module only lays it out.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import dcc, html

from mumdia_viewer.data import ResultSet, ViewerError
from mumdia_viewer.data.errors import ArtifactNotFound
from mumdia_viewer.data.tables import QUANT_STATES

from .icons import icon
from .state import PageContext, href, stop_label
from .widgets import chip, fmt, fmt_q, parse_peptidoform, peptidoform, section, unit_colour

log = logging.getLogger(__name__)

TITLES = {"precursor": "Precursors", "peptide": "Peptides", "protein_group": "Protein groups"}
NOUNS = {"precursor": "precursor", "peptide": "peptide", "protein_group": "protein group"}
UNIT_ICONS = {"precursor": "peak", "peptide": "flask", "protein_group": "layers"}
LEVELS = ("protein_group", "peptide", "precursor")
# Sentinels of the Select controls (a Mantine Select value is a non-empty string).
ANY = "any"
ALL_RUNS = "all"
# Filters that select scored rows (at a grouped level with the unit's own q column they
# test each group's winning row; see fast_note).
ROW_FILTERS = ("search", "charge", "protein", "mod", "quant")
FAST_NOTE = (
    "At this level, with the unit's own q column, each row is its group's winning row "
    "and the filters test that row."
)


def help_body(line: str, *more: str | None, small: str | None = None) -> Any:
    """An info tooltip's content: one plain line first, then details in smaller text."""
    return html.Div(
        [
            html.Div(line, className="ib-help-line"),
            *[html.Div(m, className="ib-help-more") for m in more if m],
            html.Div(small, className="ib-help-small") if small else None,
        ],
        className="ib-help",
    )


# --------------------------------------------------------------------------- toolbar


def _label(text: str, tip: str) -> Any:
    return dmc.Tooltip(
        dmc.Group([dmc.Text(text, size="xs", fw=600), icon("info", 11)], gap=3, c="dimmed"),
        label=tip,
        position="top-start",
    )


def q_options(experiment: bool, run: str, unit: str, default: str) -> list[dict[str, Any]]:
    """The seven q columns, PSM-level and grouped; the default one is marked."""
    per_run = experiment and bool(run)

    def item(c: str, grouped: bool) -> dict[str, Any]:
        label = c + (" (default)" if c == default else "")
        return {"value": c, "label": label, "disabled": grouped and per_run}

    grouped = "Grouped: one winning row per group" + (", experiment-wide" if experiment else "")
    if per_run:
        grouped += " (not with a run)"
    return [
        {
            "group": "PSM level",
            "items": [
                item(c, False)
                for c in ("q_value", "run_psm_q", "experiment_psm_q", "global_q_value")
            ],
        },
        {
            "group": grouped,
            "items": [item(c, True) for c in ("precursor_q", "peptide_q_value", "pg_q_value")],
        },
    ]


def quant_options(statuses: Sequence[str]) -> list[dict[str, Any]]:
    items = [{"value": ANY, "label": "any"}] + [
        {"value": s, "label": s.replace("_", " ")} for s in QUANT_STATES
    ]
    if statuses:
        return [
            {"group": "Quant state", "items": items},
            {"group": "quant_status", "items": [{"value": s, "label": s} for s in statuses]},
        ]
    return items


def toolbar(
    rs: ResultSet,
    f: Any,
    *,
    default_q: str,
    charges: Sequence[int],
    statuses: Sequence[str],
    chips: list[Any],
    n_filters: int,
    fast: bool = False,
) -> Any:
    """One compact row: level, search, filters, threshold, decoys, chips, reset.

    The chips stay on that row: they scroll sideways when they do not fit, and Reset
    keeps its place at the end. ``fast`` says that the row filters test each group's
    winning row (see :data:`FAST_NOTE`).
    """
    experiment = rs.is_experiment
    grouped_exp = experiment and f.unit != "precursor"
    level = dmc.SegmentedControl(
        id="ib-unit",
        value=f.unit,
        data=[
            {
                "value": u,
                "label": dmc.Group(
                    [icon(UNIT_ICONS[u], 15), TITLES[u]], gap=6, wrap="nowrap", justify="center"
                ),
            }
            for u in LEVELS
        ],
        color=unit_colour(f.unit),
        radius="md",
        size="sm",
        className="ib-unit",
    )
    search = dmc.TextInput(
        id="ib-search",
        value=f.search,
        placeholder="Peptide or protein",
        leftSection=icon("search", 16),
        debounce=350,
        radius="md",
        size="sm",
        className="ib-search",
        **{"aria-label": "Search a peptidoform or protein (substring, any case)"},
    )
    no_portal = {"shadow": "md", "withinPortal": False}
    q_select = dmc.Select(
        id="ib-q",
        label=_label(
            "q column",
            "The engine column the threshold applies to in the first table. The seven q "
            "columns are not interchangeable: hover a column header for its unit.",
        ),
        data=q_options(experiment, f.run, f.unit, default_q),
        value=f.q or default_q,
        allowDeselect=False,
        size="xs",
        comboboxProps={**no_portal, "width": 280, "position": "bottom-start"},
        checkIconPosition="right",
        maxDropdownHeight=320,
    )
    charge = dmc.Select(
        id="ib-charge",
        label=dmc.Text("charge", size="xs", fw=600, c="dimmed"),
        data=[{"value": ANY, "label": "any"}]
        + [{"value": str(z), "label": f"{z}+"} for z in charges],
        value=f.charge or ANY,
        allowDeselect=False,
        size="xs",
        comboboxProps=no_portal,
    )
    protein = dmc.TextInput(
        id="ib-protein",
        label=dmc.Text("protein contains", size="xs", fw=600, c="dimmed"),
        value=f.protein,
        placeholder="e.g. _YEAST",
        debounce=350,
        size="xs",
    )
    mod = dmc.Autocomplete(
        id="ib-mod",
        label=_label(
            "modification",
            "A substring of the peptidoform, matched with case (for example Oxidation).",
        ),
        value=f.mod,
        data=[f.mod] if f.mod else [],
        placeholder="e.g. Oxidation",
        debounce=350,
        size="xs",
        comboboxProps=no_portal,
        clearable=True,
    )
    quant_tip = (
        f"The {NOUNS[f.unit]} table of an experiment has no quant columns; start from "
        "precursors to filter by quant state."
        if grouped_exp
        else "quantified; not_quantifiable (a quant row with a null quantity, never 0); "
        "not_selected (no quant row: outside the quant gate)"
    )
    quant = dmc.Select(
        id="ib-quant",
        label=_label("quant state", quant_tip),
        data=quant_options(statuses),
        value=f.quant or ANY,
        allowDeselect=False,
        size="xs",
        disabled=grouped_exp and not f.quant,
        comboboxProps=no_portal,
    )
    run_tip = (
        f"The {NOUNS[f.unit]} table of an experiment is experiment-wide (its q column is "
        "experiment-wide). Start from precursors for the rows of one run (run_psm_q)."
        if grouped_exp
        else "Rows of one run. The default q column is then run_psm_q (PSM-level q within "
        "the run); the grouped q columns are experiment-wide and are not offered."
    )
    run = dmc.Select(
        id="ib-run",
        label=_label("run", run_tip),
        data=[{"value": ALL_RUNS, "label": "all runs"}]
        + [{"value": r.name, "label": r.label} for r in rs.runs if r.name],
        value=f.run or ALL_RUNS,
        allowDeselect=False,
        size="xs",
        disabled=grouped_exp and not f.run,
        comboboxProps=no_portal,
        style=None if experiment else {"display": "none"},
    )
    filters = dmc.Popover(
        [
            dmc.PopoverTarget(
                dmc.Button(
                    [
                        html.Span("Filters"),
                        dmc.Badge(
                            str(n_filters) if n_filters else "",
                            id="ib-nfilters",
                            size="xs",
                            circle=True,
                            variant="filled",
                            className="ib-nfilters" + ("" if n_filters else " ib-hidden"),
                        ),
                    ],
                    leftSection=icon("table", 15),
                    variant="default",
                    size="sm",
                    radius="md",
                    className="ib-filters-btn",
                )
            ),
            dmc.PopoverDropdown(
                dmc.Stack(
                    [
                        dmc.Text("Filters of the first table", fw=650, size="sm"),
                        dmc.Text(
                            "The child tables show every row of their parent; the decoy "
                            "switch applies to all tables.",
                            size="xs",
                            c="dimmed",
                        ),
                        dmc.Text(
                            FAST_NOTE if fast else "",
                            id="ib-fast-note",
                            size="xs",
                            c="orange",
                            className="ib-fast-note" + ("" if fast else " ib-hidden"),
                        ),
                        q_select,
                        dmc.SimpleGrid([charge, quant], cols=2, spacing="xs"),
                        protein,
                        mod,
                        run,
                    ],
                    gap=8,
                ),
                className="ib-filters-dropdown",
            ),
        ],
        position="bottom-start",
        shadow="md",
        radius="md",
        width=300,
        withArrow=True,
        trapFocus=False,
        keepMounted=True,
    )
    pill = dmc.Tooltip(
        html.Div(
            [icon("target", 14), html.Span(f"q ≤ {stop_label(f.t)}", id="ib-t")],
            className="ib-t",
            role="button",
            tabIndex="0",
        ),
        label="The header threshold, for every table: the first table keeps the rows that "
        "pass it on its q column, and the validation marks of every table test it. Click "
        "to change it.",
        position="bottom",
    )
    decoys = dmc.Switch(
        id="ib-decoys",
        checked=f.decoys,
        label="Decoys",
        size="sm",
        color="orange",
        className="ib-decoys",
    )
    reset = dmc.Button(
        "Reset",
        id="ib-reset",
        leftSection=icon("x", 13),
        variant="subtle",
        color="gray",
        size="compact-sm",
        radius="md",
        n_clicks=0,
        className="ib-reset",
        **{"aria-label": "Reset the filters and the sort"},
    )
    keys = dmc.Group(
        [
            dmc.Group([dmc.Kbd("↑", size="xs"), dmc.Kbd("↓", size="xs")], gap=2),
            dmc.Text("select", size="xs", c="dimmed"),
            dmc.Kbd("Enter", size="xs"),
            dmc.Text("open", size="xs", c="dimmed"),
        ],
        gap=5,
        wrap="nowrap",
        className="ib-keys",
    )
    return dmc.Card(
        html.Div(
            [
                level,
                search,
                filters,
                pill,
                decoys,
                html.Div(
                    html.Div(chips, id="ib-chips", className="ib-chips"),
                    className="ib-chips-wrap",
                ),
                html.Div(
                    [
                        dmc.Tooltip(reset, label="Reset the filters and the sort", w="auto"),
                        keys,
                    ],
                    className="ib-tools-end",
                ),
            ],
            className="ib-tools",
        ),
        className="ib-toolbar",
        p=6,
    )


def _chip(key: str, text: str, colour: str = "indigo", tip: str | None = None) -> Any:
    return html.Button(
        [html.Span(text, className="ib-chip-text"), icon("x", 11)],
        id={"type": "ib-chip", "key": key},
        n_clicks=0,
        className=f"ib-chip ib-chip-{colour}",
        title=f"{tip or text}. Click to remove this filter.",
    )


def chip_row(f: Any, *, fast: bool = False) -> list[Any]:
    """Removable chips of the active filters (the threshold is the header's).

    ``fast``: the row filters test each group's winning row, and their chips say so.
    """
    where = " (winning row)" if fast else ""
    tip = f". {FAST_NOTE}" if fast else ""
    # The filters of the Filters popover first (their controls are hidden there), then
    # the search and the decoy switch (their controls are on the row).
    chips = []
    if f.q:
        chips.append(_chip("q", f"q column {f.q}"))
    if f.charge:
        chips.append(_chip("charge", f"charge {f.charge}+{where}", tip=f"charge {f.charge}+{tip}"))
    if f.protein:
        chips.append(
            _chip("protein", f"protein “{f.protein}”{where}", tip=f"protein “{f.protein}”{tip}")
        )
    if f.mod:
        chips.append(_chip("mod", f"modification {f.mod}{where}", tip=f"modification {f.mod}{tip}"))
    if f.quant:
        state = f.quant.replace("_", " ")
        chips.append(_chip("quant", f"quant {state}{where}", tip=f"quant {state}{tip}"))
    if f.run:
        chips.append(_chip("run", f"run {f.run}"))
    if f.search:
        chips.append(
            _chip("search", f"contains “{f.search}”{where}", tip=f"contains “{f.search}”{tip}")
        )
    if f.decoys:
        chips.append(_chip("decoys", "decoys shown", "orange"))
    return chips


def n_popover_filters(f: Any) -> int:
    return sum(bool(getattr(f, k)) for k in ("q", "charge", "protein", "mod", "quant", "run"))


# --------------------------------------------------------------------------- panels


def panel(
    key: str,
    title: str,
    body: Any,
    *,
    count: Any,
    help_text: Any,
    subject: Any = None,
    extra: Any = None,
    right: Any = None,
    loading: bool = False,
    above: Any = None,
) -> Any:
    """A linked panel: a :func:`widgets.section` card with a dense grid.

    ``above`` sits between the title bar and the grid (the coverage strip of the peptides
    panel).

    The title holds the row count and the info tooltip with ids (``ib-<key>-count``,
    ``ib-<key>-help``), so the callbacks update them when the selection or the filters
    change; ``subject`` (``ib-<key>-subject``) names the parent of a child table. The
    count sits in ``ib-<key>-countbox``, whose title says what it counts (a child panel
    shows "passing of all", counted by the browser from the rows' engine q values).
    """
    title_row: list[Any] = [html.Span(title, className="ib-ptitle-text")]
    if subject is not None:
        title_row.append(html.Span(subject, id=f"ib-{key}-subject", className="ib-subject"))
    title_row.append(
        html.Span(
            dmc.Badge(
                count,
                id=f"ib-{key}-count",
                size="sm",
                color="gray",
                variant="light",
                className="ib-count",
                style={"textTransform": "none"},
            ),
            id=f"ib-{key}-countbox",
            className="ib-countbox",
        )
    )
    if extra is not None:
        title_row.append(extra)
    title_row.append(
        dmc.Tooltip(
            html.Span(icon("info", 13), className="mv-help"),
            label=help_text,
            id=f"ib-{key}-help",
            w=440,
            position="bottom-start",
            multiline=True,
        )
    )
    card = section(
        html.Span(title_row, className="ib-ptitle"),
        *([above] if above is not None else []),
        html.Div(body, className="ib-gridbox"),
        right=right if right is not None else html.Span(),
        p="0",
        id=f"ib-p-{key}",
    )
    return html.Div(
        card,
        id=f"ib-panel-{key}",
        className=f"ib-panel ib-panel-{key}" + (" ib-loading" if loading else ""),
    )


# Rows and headers of the dense grids (also mvbTheme in browser.js and browser.css).
ROW_HEIGHT = 26
HEADER_HEIGHT = 28

GRID_OPTIONS: dict[str, Any] = {
    "theme": {"function": "mvbTheme(themeQuartz)"},
    "rowHeight": ROW_HEIGHT,
    "headerHeight": HEADER_HEIGHT,
    "suppressMultiSort": True,
    "rowSelection": {"mode": "singleRow", "checkboxes": False, "enableClickSelection": True},
    "tooltipShowDelay": 350,
    "tooltipHideDelay": 15000,
    "animateRows": False,
    "suppressCellFocus": False,
    "maintainColumnOrder": True,
    # A match-between-runs transfer is ringed on its validation mark (browser.css).
    "rowClassRules": {"ib-row-transfer": "params.data && params.data.is_transferred"},
}


def _events(panel_key: str) -> dict[str, list[str]]:
    return {
        "rowClicked": [f"mvbRowClicked(params, '{panel_key}')"],
        "rowDoubleClicked": [f"mvbRowDoubleClicked(params, '{panel_key}')"],
        "cellKeyDown": [f"mvbKeyDown(params, '{panel_key}')"],
        # Every focus move (arrows, page keys, Home, End) moves the selection with it.
        "cellFocused": [f"mvbFocused(params, '{panel_key}')"],
        "modelUpdated": [f"mvbModelUpdated(params, '{panel_key}')"],
        "viewportChanged": [f"mvbModelUpdated(params, '{panel_key}')"],
        "rowDataUpdated": [f"mvbRowDataUpdated(params, '{panel_key}')"],
        "firstDataRendered": [f"mvbFirstData(params, '{panel_key}')"],
        "newColumnsLoaded": [f"mvbColumnsLoaded(params, '{panel_key}')"],
        "sortChanged": [f"mvbSortChanged(params, '{panel_key}')"],
    }


DEFAULT_COL = {
    "sortable": True,
    "resizable": True,
    "suppressHeaderMenuButton": True,
    "suppressHeaderFilterButton": True,
    "sortingOrder": ["desc", "asc"],
    # Space would select the focused row without the panels below (browser.js).
    "suppressKeyboardEvent": {"function": "mvbSuppressKey(params)"},
}


def top_grid(defs: list[dict[str, Any]], base: str, total: int | None, threshold: float) -> Any:
    """The first table: AG Grid's infinite row model, sorted and paged by the server."""
    return dag.AgGrid(
        id="ib-grid",
        rowModelType="infinite",
        columnDefs=defs,
        defaultColDef=DEFAULT_COL,
        getRowId="params.data._key",
        dashGridOptions={
            **GRID_OPTIONS,
            "cacheBlockSize": 100,
            "maxBlocksInCache": 40,
            "maxConcurrentDatasourceRequests": 1,
            "infiniteInitialRowCount": max(1, min(int(total or 0), 20)),
            "blockLoadDebounceMillis": 40,
            "rowBuffer": 10,
            "suppressScrollOnNewData": True,
            "context": {"base": base, "threshold": threshold, "panel": "top"},
            "navigateToNextCell": {"function": "mvbNavigate(params, 'top')"},
        },
        eventListeners=_events("top"),
        style={"height": "100%", "width": "100%"},
        className="ib-grid",
    )


def child_grid(
    grid_id: str,
    panel_key: str,
    defs: list[dict[str, Any]],
    base: str,
    threshold: float,
    empty_text: str,
) -> Any:
    """A child table: every row of the selected parent, sorted in the browser.

    It starts without rows; the browser asks for them once the first table has its
    first block (``ib-need``), or takes them from its cache.
    """
    return dag.AgGrid(
        id=grid_id,
        rowData=[],
        columnDefs=defs,
        defaultColDef=DEFAULT_COL,
        getRowId="params.data._key",
        dashGridOptions={
            **GRID_OPTIONS,
            "rowBuffer": 12,
            "context": {"base": base, "threshold": threshold, "panel": panel_key},
            "navigateToNextCell": {"function": f"mvbNavigate(params, '{panel_key}')"},
            "localeText": {"noRowsToShow": empty_text},
        },
        eventListeners=_events(panel_key),
        style={"height": "100%", "width": "100%"},
        className="ib-grid ib-grid-child",
    )


def selected_rows(key: str | None) -> Any:
    """The ``selectedRows`` of a grid: the row with this id, or none."""
    return {"ids": [key]} if key else []


def columns_menu(
    columns: list[dict[str, str]], shown: list[str], *, prefix: str = "ib-cols"
) -> Any:
    """The Columns menu of a grid (``prefix``: ``ib-cols``, ``ib-pep-cols``, ``ib-pre-cols``).

    The first table's menu has a Defaults button. The menu's list is mounted only while
    it is open.
    """
    head: list[Any] = [dmc.Text("Columns", fw=650, size="sm")]
    if prefix == "ib-cols":
        head.append(
            dmc.Button(
                "Defaults",
                id="ib-cols-default",
                variant="subtle",
                size="compact-xs",
                n_clicks=0,
            )
        )
    return dmc.Popover(
        [
            dmc.PopoverTarget(
                dmc.Button(
                    "Columns",
                    id=f"{prefix}-btn",
                    leftSection=icon("table", 13),
                    variant="subtle",
                    color="gray",
                    size="compact-xs",
                    radius="md",
                    className="ib-cols-btn",
                )
            ),
            dmc.PopoverDropdown(
                [
                    dmc.Group(head, justify="space-between", mb=6),
                    dmc.Text(
                        "Hover a column header for what it holds.", size="xs", c="dimmed", mb=8
                    ),
                    dmc.ScrollArea(
                        dmc.CheckboxGroup(
                            dmc.Stack(
                                [
                                    dmc.Checkbox(value=c["value"], label=c["label"], size="xs")
                                    for c in columns
                                ],
                                gap=7,
                            ),
                            id=prefix,
                            value=shown,
                        ),
                        mah=340,
                        type="auto",
                        offsetScrollbars=True,
                    ),
                ],
                className="ib-cols-dropdown",
            ),
        ],
        position="bottom-end",
        shadow="md",
        radius="md",
        width=250,
        withArrow=True,
        trapFocus=False,
    )


def locate_action() -> Any:
    """'Selected row not in view': shown by the browser when the selection is off the list."""
    return html.Button(
        [icon("search", 12), html.Span("", id="ib-locate-text")],
        id="ib-locate",
        className="ib-locate ib-hidden",
        disable_n_clicks=True,
        title="The selected row is not in the rows on screen. Click to scroll to it.",
        type="button",
    )


def notice_line() -> Any:
    """A one-line notice above the panels (a link that names a row this table lacks)."""
    return html.Div(
        [
            dmc.ThemeIcon(icon("alert", 13), size=20, radius="xl", color="orange", variant="light"),
            html.Span("", id="ib-notice-text", className="ib-notice-text"),
            html.Button(
                icon("x", 12),
                id="ib-notice-close",
                className="ib-notice-close",
                disable_n_clicks=True,
                title="Close",
                type="button",
            ),
        ],
        id="ib-notice",
        className="ib-notice ib-hidden",
        role="status",
    )


def q_chip(text: str, unit: str) -> Any:
    return dmc.Badge(
        text,
        id="ib-top-q",
        color=unit_colour(unit),
        variant="light",
        size="sm",
        className="mv-kpi-q ib-qchip",
        style={"textTransform": "none"},
    )


def group_subject(group: str | None) -> Any:
    """The parent of the peptides panel: the group's first member, ``+n`` for the others."""
    if not group:
        return ""
    members = [m for m in str(group).split(";") if m]
    more = f" +{len(members) - 1}" if len(members) > 1 else ""
    return html.Span([members[0] if members else str(group), more], title=str(group))


def peptide_subject(peptidoform_text: str | None, peptide_id: Any) -> Any:
    """The parent of the precursors panel: the peptide's sequence (its base peptide)."""
    if not peptidoform_text:
        return ""
    pf = parse_peptidoform(peptidoform_text)
    seq = ("DECOY_" if pf.decoy else "") + pf.sequence
    return html.Span(
        seq,
        className="ib-subject-seq",
        title=f"base_peptide_id {peptide_id}; the precursors are every scored row of this "
        "base peptide",
    )


def empty_state(
    text: str,
    sub: str | None = None,
    *,
    icon_name: str = "info",
    red: bool = False,
    detail: str | None = None,
) -> Any:
    """A centred message; ``detail`` is a longer note in small print (the data layer's).

    The note is clamped to three lines with a "more" toggle (browser.js).
    """
    return dmc.Stack(
        [
            dmc.ThemeIcon(
                icon(icon_name, 20),
                color="red" if red else "gray",
                variant="light",
                size=40,
                radius="xl",
            ),
            dmc.Text(text, fw=650, size="sm", ta="center"),
            dmc.Text(sub, size="sm", c="dimmed", ta="center", maw=640, className="ib-empty-sub")
            if sub
            else None,
            html.Div(
                [
                    html.Div(detail, className="ib-clamp-text"),
                    html.Button(
                        "more",
                        className="ib-clamp-more",
                        disable_n_clicks=True,
                        type="button",
                    ),
                ],
                className="ib-clamp",
            )
            if detail
            else None,
        ],
        align="center",
        gap=6,
    )


# --------------------------------------------------------------------------- preview


def preview_slot() -> Any:
    """The bottom panel: the preview of the selected precursor, with its collapse handle.

    The server writes the card into ``ib-preview-data``; the browser inserts it when the
    selection has settled and the child panels have their rows (``assets/browser.js``).
    The handle hides the preview's figures and tables (remembered per browser); the
    header line with the precursor's q values and the open link stays.
    """
    return html.Div(
        [
            dcc.Store(id="ib-preview-data"),
            html.Button(
                [
                    html.Span(className="ib-pvt-icon"),
                    html.Span("Hide preview", className="ib-pvt-hide"),
                    html.Span("Show preview", className="ib-pvt-show"),
                ],
                id="ib-pv-toggle",
                className="ib-pv-toggle",
                disable_n_clicks=True,
                type="button",
                title="Hide or show the preview's figures and ion table (the header line "
                "stays). The tables get the room.",
            ),
            html.Div(
                dmc.Card(
                    dmc.Center(dmc.Loader(type="dots"), h="100%"),
                    className="ib-preview-wait",
                    p="sm",
                ),
                id="ib-preview",
            ),
        ],
        id="ib-panel-prev",
        className="ib-panel ib-panel-prev ib-loading",
    )


def warm_detail(rs: ResultSet, run: str, cid: Any) -> bool:
    """Read a precursor's detail into the detail page's cache (its preview then opens warm).

    False when it cannot be read; nothing is shown either way.
    """
    try:
        from . import detail as detail_page

        get_detail = detail_page.get_detail
        get_detail(rs, run if rs.is_experiment else "", int(cid))
    except Exception:  # a guess ahead of the user: any failure only means a cold preview
        return False
    return True


def _missing(exc: BaseException) -> str:
    reason = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    return reason or type(exc).__name__


def preview(ctx: PageContext, prec: Mapping[str, Any] | None) -> tuple[Any, bool]:
    """The bottom panel's card for the selected precursor, and whether that precursor exists.

    The card is :func:`ui.detail.preview` when the detail module provides it, else a
    summary of the selected row with the link to the precursor page. ``False`` says that
    the run and candidate id name no scored row of this result set (:func:`detail.get_detail`
    refused them), so the page can fall back to its first row.
    """
    if not prec or prec.get("cid") is None:
        return (
            dmc.Card(
                empty_state(
                    "No precursor is selected",
                    "Select a row above: its precursor's XIC, spectrum and ion table appear here.",
                ),
                className="ib-preview-empty",
                p="lg",
            ),
            True,
        )
    run = str(prec.get("run") or "")
    try:
        cid = int(prec["cid"])
    except (TypeError, ValueError):
        return placeholder(ctx, prec, note=f"Not a candidate id: {prec.get('cid')!r}"), False
    try:
        from . import detail as detail_page
    except Exception as exc:  # the detail page is another module; keep this page usable
        return placeholder(ctx, prec, note=f"The precursor preview is not available: {exc}"), True
    exists = True
    get_detail = getattr(detail_page, "get_detail", None)
    if get_detail is not None:
        try:
            get_detail(ctx.rs, run if ctx.rs.is_experiment else "", cid)
        except (KeyError, ArtifactNotFound) as exc:
            log.debug("no scored row for run %r, candidate %s: %s", run, cid, _missing(exc))
            exists = False
        except (ViewerError, ValueError, TypeError):
            pass  # the preview shows the reason
    if not hasattr(detail_page, "preview"):
        note = "The precursor preview (ui.detail.preview) is not available."
        return placeholder(ctx, prec, note=note), exists
    try:
        return detail_page.preview(ctx, run, cid), exists
    except (ViewerError, KeyError, ValueError, TypeError) as exc:
        return placeholder(ctx, prec, note=f"The precursor preview failed: {exc}"), exists
    except Exception as exc:  # a fault of the preview must not stop this page
        log.exception("ui.detail.preview failed for run %r, candidate %s", run, cid)
        note = f"The precursor preview failed: {type(exc).__name__}: {exc}"
        return placeholder(ctx, prec, note=note), exists


def placeholder(ctx: PageContext, prec: Mapping[str, Any], *, note: str) -> Any:
    """A summary of the selected precursor from its grid row, and the open link."""
    row = dict(prec.get("row") or {})
    run = str(prec.get("run") or "")
    cid = prec.get("cid")
    label = str(row.get("label") or "")
    items: list[Any] = [
        html.Span(peptidoform(row.get("peptidoform"), size="1.02rem"), className="ib-pv-pep")
        if row.get("peptidoform")
        else dmc.Text(f"candidate {cid}", fw=650),
    ]
    if row.get("charge") is not None:
        items.append(
            dmc.Badge(
                f"{row.get('charge')}+",
                size="sm",
                variant="outline",
                color="gray",
                style={"textTransform": "none"},
            )
        )
    if label:
        items.append(chip(label, "orange" if label == "decoy" else "indigo", size="sm"))
    if run:
        items.append(chip(f"run {run}", "gray", size="sm"))
    qs = [
        (c, row.get(c))
        for c in ("q_value", "run_psm_q", "precursor_q", "peptide_q_value", "pg_q_value")
        if row.get(c) is not None
    ]
    q_text = " · ".join(f"{c} {fmt_q(v)}" for c, v in qs)
    link = dcc.Link(
        dmc.Button(
            "Open precursor page",
            rightSection=icon("right", 14),
            size="compact-sm",
            radius="md",
        ),
        href=href(ctx.base, "precursor", {"run": run, "cid": cid}),
    )
    return dmc.Card(
        [
            dmc.Group(
                [dmc.Group(items, gap=8), link],
                justify="space-between",
                wrap="nowrap",
            ),
            dmc.Text(q_text, size="xs", c="dimmed", className="ib-mono", mt=6) if q_text else None,
            dmc.Text(
                f"score {fmt(row.get('score'))}" if row.get("score") is not None else "",
                size="xs",
                c="dimmed",
            ),
            html.Div(empty_state("Preview not shown", note), className="ib-pv-note"),
        ],
        p="sm",
        className="ib-preview-placeholder",
    )
