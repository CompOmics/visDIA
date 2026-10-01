"""The Dash application: the shell (header, navigation, stores) and the page router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dash_mantine_components as dmc
import plotly.io as pio
from dash import (
    ALL,
    ClientsideFunction,
    Dash,
    Input,
    Output,
    State,
    dcc,
    html,
    no_update,
)

from mumdia_viewer import __version__
from mumdia_viewer.data import ResultSet
from mumdia_viewer.data.fasta import Fasta

from . import browser, calibration, detail, overview, protein, qc, quant
from .icons import icon
from .state import (
    DEFAULT_THRESHOLD,
    PageContext,
    href,
    page_of,
    parse_threshold,
    query_of,
    threshold_options,
)
from .theme import MANTINE_THEME, browser_globals
from .widgets import notice_list, peptidoform

ASSETS = Path(__file__).parent / "assets"
RECENT_MAX = 8
PAGES = {
    "overview": overview,
    "identifications": browser,
    "precursor": detail,
    "protein": protein,
    "calibration": calibration,
    "qc": qc,
    "quant": quant,
}
# Navigation entries after the results: (page, label, icon).
VIEWS = (
    ("protein", "Protein", "layers"),
    ("calibration", "Calibration", "calibration"),
    ("qc", "Run QC", "spectrum"),
    ("quant", "Quant QC", "quant"),
)


# The page skeleton: Dash's default plus window.MV, the constants the browser code
# needs (theme.browser_globals), written before any script runs.
INDEX = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>window.MV = __MV__;</script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _templates() -> dict[str, Any]:
    return {s: pio.templates[f"mumdia_{s}"].to_plotly_json() for s in ("light", "dark")}


# --------------------------------------------------------------------------- header


def _brand(base: str) -> Any:
    return dcc.Link(
        dmc.Group(
            [
                dmc.ThemeIcon(
                    icon("peak", 20),
                    size=34,
                    radius="md",
                    variant="gradient",
                    gradient={"from": "indigo", "to": "violet", "deg": 135},
                ),
                html.Div(
                    [
                        dmc.Text("MuMDIA viewer", fw=700, size="md", lh=1.15),
                        dmc.Text(f"version {__version__}", size="xs", c="dimmed", lh=1.15),
                    ]
                ),
            ],
            gap=10,
            wrap="nowrap",
        ),
        href=base,
        className="mv-brand",
    )


def _result_chip(rs: ResultSet) -> Any:
    kind = f"experiment, {len(rs.runs)} runs" if rs.is_experiment else "single run"
    return dmc.Tooltip(
        dmc.Group(
            [
                icon("layers", 16),
                dmc.Text(rs.root.name, fw=600, size="sm", truncate="end", maw=280),
                dmc.Badge(kind, size="sm", color="gray"),
            ],
            gap=8,
            wrap="nowrap",
            className="mv-result-chip",
        ),
        label=str(rs.root),
        w="auto",
        multiline=False,
    )


def _header(rs: ResultSet, base: str) -> Any:
    n = len(rs.notices)
    notices_button = dmc.Indicator(
        dmc.ActionIcon(
            icon("alert", 18),
            id="notices-open",
            variant="default",
            size="lg",
            radius="xl",
            n_clicks=0,
            **{"aria-label": "Notices"},
        ),
        label=str(n),
        size=16,
        color="orange",
        disabled=n == 0,
        offset=3,
    )
    scheme_button = dmc.ActionIcon(
        [dmc.Box(icon("moon", 18), darkHidden=True), dmc.Box(icon("sun", 18), lightHidden=True)],
        id="scheme-toggle",
        variant="default",
        size="lg",
        radius="xl",
        n_clicks=0,
        **{"aria-label": "Switch between the light and the dark theme"},
    )
    threshold = dmc.Tooltip(
        dmc.Select(
            id="q-select",
            data=threshold_options(),
            value=repr(DEFAULT_THRESHOLD),
            allowDeselect=False,
            w=136,
            radius="xl",
            size="sm",
            leftSection=icon("target", 16),
            checkIconPosition="right",
            persistence=True,
            persistence_type="session",
            comboboxProps={"shadow": "md"},
        ),
        label=(
            "The q threshold of every count and table. Each count applies it to its own "
            "q column, which is named beside the count."
        ),
        position="bottom",
    )
    search = dmc.TextInput(
        id="global-search",
        placeholder="Search a peptide or protein",
        leftSection=icon("search", 16),
        rightSection=dmc.Kbd("/", size="xs"),
        w=290,
        radius="xl",
        size="sm",
        n_submit=0,
        debounce=True,
        visibleFrom="md",
    )
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Group(
                    [
                        dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                        _brand(base),
                        dmc.Divider(orientation="vertical", visibleFrom="sm", my=14),
                        html.Div(_result_chip(rs), className="mv-hide-mobile"),
                    ],
                    gap="md",
                    wrap="nowrap",
                ),
                dmc.Group(
                    [search, threshold, notices_button, scheme_button], gap="sm", wrap="nowrap"
                ),
            ],
            justify="space-between",
            h="100%",
            px="lg",
            wrap="nowrap",
        )
    )


# --------------------------------------------------------------------------- navigation


def _navbar(rs: ResultSet, base: str) -> Any:
    def link(page: str, label: str, name: str, **kwargs: Any) -> Any:
        return dmc.NavLink(
            id=f"nav-{page}",
            label=label,
            leftSection=icon(name, 18),
            href=href(base, page) if page != "precursor" else None,
            **kwargs,
        )

    keys = dmc.Stack(
        [
            dmc.Group([dmc.Kbd("/"), dmc.Text("search", size="xs", c="dimmed")], gap=6),
            dmc.Group(
                [dmc.Kbd("←"), dmc.Kbd("→"), dmc.Text("step scans", size="xs", c="dimmed")],
                gap=6,
            ),
        ],
        gap=6,
    )
    return dmc.AppShellNavbar(
        [
            dmc.AppShellSection(
                [
                    dmc.Text("Results", className="mv-nav-heading"),
                    link("overview", "Overview", "overview"),
                    link("identifications", "Identifications", "table"),
                    link("precursor", "Precursor detail", "peak", disabled=True),
                    dmc.Text("Views", className="mv-nav-heading", mt="md"),
                    *[link(page, label, name) for page, label, name in VIEWS],
                ],
                className="mv-nav",
            ),
            dmc.AppShellSection(
                dmc.ScrollArea(
                    [
                        dmc.Text("Recently viewed", className="mv-nav-heading", mt="lg"),
                        html.Div(id="nav-recent"),
                    ],
                    type="hover",
                ),
                grow=True,
                className="mv-nav",
            ),
            dmc.AppShellSection(
                [
                    dmc.Divider(mb="sm"),
                    keys,
                    dmc.Text(str(rs.root), size="xs", c="dimmed", mt="sm", className="mv-path"),
                ],
                px="xs",
            ),
        ],
        p="sm",
    )


def _recent_links(recent: list[dict[str, Any]] | None, base: str) -> Any:
    if not recent:
        return dmc.Text("Precursors you open appear here.", size="xs", c="dimmed", px="xs", mt=4)
    links = []
    for r in recent:
        where = r.get("run") or "single run"
        links.append(
            dmc.NavLink(
                label=peptidoform(r.get("peptidoform"), size="0.82em"),
                description=f"{r.get('charge')}+  {where}",
                href=href(base, "precursor", {"run": r.get("run"), "cid": r.get("cid")}),
                leftSection=dmc.Box(
                    w=8,
                    h=8,
                    style={
                        "borderRadius": "50%",
                        "background": "var(--mantine-color-orange-6)"
                        if r.get("label") == "decoy"
                        else "var(--mantine-color-indigo-6)",
                    },
                ),
                className="mv-recent",
            )
        )
    return links


# --------------------------------------------------------------------------- app


def create_app(
    rs: ResultSet,
    *,
    compare: ResultSet | None = None,
    url_base: str = "/",
    fasta: Fasta | None = None,
) -> Dash:
    """Build the app for one result set.

    ``compare`` is kept for the compare view; ``fasta`` gives the protein sequences of
    the coverage views (none: the views say how to add a FASTA).
    """
    app = Dash(
        __name__,
        url_base_pathname=url_base,
        assets_folder=str(ASSETS),
        suppress_callback_exceptions=True,
        title=f"{rs.root.name} | MuMDIA viewer",
        update_title=None,
    )
    app.index_string = INDEX.replace("__MV__", json.dumps(browser_globals()))
    state = {"rs": rs, "compare": compare, "fasta": fasta}
    app.mv_fasta = lambda: state["fasta"]  # type: ignore[attr-defined]

    def get_rs() -> ResultSet:
        return state["rs"]

    base = url_base
    notices = [(n.code, n.message) for n in rs.notices]
    shell = dmc.AppShell(
        [
            _header(rs, base),
            _navbar(rs, base),
            dmc.AppShellMain(
                dcc.Loading(
                    html.Div(id="page", className="mv-page"),
                    delay_show=350,
                    overlay_style={"visibility": "visible", "opacity": 0.55},
                    custom_spinner=dmc.Loader(type="dots", size="lg"),
                    target_components={"page": "children"},
                )
            ),
            dmc.Drawer(
                notice_list(notices) if notices else dmc.Text("No notices.", c="dimmed"),
                id="notices-drawer",
                title=dmc.Text("Notices about this result set", fw=650),
                position="right",
                size="lg",
                opened=False,
            ),
        ],
        id="shell",
        header={"height": 62},
        navbar={"width": 236, "breakpoint": "sm", "collapsed": {"mobile": True}},
        padding="lg",
    )
    app.layout = dmc.MantineProvider(
        [
            # A url.href output navigates inside the app. With refresh=True the
            # Location also wrote its stale search back after a page rewrote the
            # address with history.replaceState, and the stale address won.
            dcc.Location(id="url", refresh="callback-nav"),
            dcc.Store(id="threshold", data=DEFAULT_THRESHOLD),
            dcc.Store(id="scheme", storage_type="local"),
            dcc.Store(id="templates", data=_templates()),
            dcc.Store(id="recent", storage_type="session", data=[]),
            shell,
        ],
        id="provider",
        theme=MANTINE_THEME,
        defaultColorScheme="light",
    )

    # Colour scheme: the stored choice (or the system preference) drives Mantine, and
    # every graph swaps its Plotly template in the browser.
    app.clientside_callback(
        ClientsideFunction("mv", "toggleScheme"),
        Output("scheme", "data"),
        Input("scheme-toggle", "n_clicks"),
        State("scheme", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mv", "scheme"),
        Output("provider", "forceColorScheme"),
        Input("scheme", "data"),
    )
    app.clientside_callback(
        ClientsideFunction("mv", "retheme"),
        Output({"type": "fig", "name": ALL}, "figure", allow_duplicate=True),
        Input("scheme", "data"),
        Input({"type": "fig", "name": ALL}, "figure"),
        State("templates", "data"),
        prevent_initial_call="initial_duplicate",
    )
    app.clientside_callback(
        ClientsideFunction("mv", "threshold"),
        Output("threshold", "data"),
        Input("q-select", "value"),
    )
    app.clientside_callback(
        ClientsideFunction("mv", "openDrawer"),
        Output("notices-drawer", "opened"),
        Input("notices-open", "n_clicks"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        ClientsideFunction("mv", "burger"),
        Output("shell", "navbar"),
        Input("burger", "opened"),
        State("shell", "navbar"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("page", "children"),
        Output("recent", "data"),
        Input("url", "pathname"),
        Input("url", "search"),
        State("threshold", "data"),
        State("scheme", "data"),
        State("recent", "data"),
    )
    def route(pathname, search, t, scheme, recent):
        page = page_of(pathname, base)
        ctx = PageContext(
            rs=get_rs(),
            base=base,
            threshold=parse_threshold(t),
            scheme="dark" if scheme == "dark" else "light",
            query=query_of(search),
            compare=state["compare"],
            fasta=state["fasta"],
        )
        module = PAGES[page]
        content = module.layout(ctx)
        seen = module.recent(ctx) if hasattr(module, "recent") else None
        if not seen:
            return content, no_update
        key = (seen["run"], seen["cid"])
        kept = [r for r in (recent or []) if (r.get("run"), r.get("cid")) != key]
        return content, [seen, *kept][:RECENT_MAX]

    nav_pages = ["overview", "identifications", "precursor", *[v[0] for v in VIEWS]]

    @app.callback(
        *[Output(f"nav-{p}", "active") for p in nav_pages],
        Input("url", "pathname"),
    )
    def active(pathname):
        page = page_of(pathname, base)
        return tuple(page == p for p in nav_pages)

    @app.callback(
        Output("nav-recent", "children"),
        Output("nav-precursor", "href"),
        Output("nav-precursor", "disabled"),
        Input("recent", "data"),
    )
    def show_recent(recent):
        if not recent:
            return _recent_links(None, base), None, True
        last = recent[0]
        return (
            _recent_links(recent, base),
            href(base, "precursor", {"run": last.get("run"), "cid": last.get("cid")}),
            False,
        )

    @app.callback(
        Output("url", "href", allow_duplicate=True),
        Input("global-search", "n_submit"),
        State("global-search", "value"),
        prevent_initial_call=True,
    )
    def search(n, value):
        text = (value or "").strip()
        if not n or not text:
            return no_update
        return href(base, "identifications", {"search": text})

    for module in PAGES.values():
        module.register(app, get_rs, base)
    return app
