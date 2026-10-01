"""The Dash application: a header with the q threshold, and a router over the pages."""

from __future__ import annotations

from urllib.parse import parse_qs

from dash import Dash, Input, Output, dcc, html

from mumdia_viewer import __version__
from mumdia_viewer.data import ResultSet

from . import browser, detail, overview

DEFAULT_THRESHOLD = 0.01


def create_app(rs: ResultSet, *, compare: ResultSet | None = None, url_base: str = "/") -> Dash:
    """Build the app for one result set (``compare`` is kept for the compare view)."""
    app = Dash(
        __name__,
        url_base_pathname=url_base,
        suppress_callback_exceptions=True,
        title=f"mumdia-viewer: {rs.root.name}",
        update_title=None,
    )
    state = {"rs": rs, "compare": compare}

    def get_rs() -> ResultSet:
        return state["rs"]

    nav = {"marginRight": "16px"}
    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="threshold", data=DEFAULT_THRESHOLD),
            html.Div(
                [
                    html.Span("mumdia-viewer", style={"fontWeight": "bold", "marginRight": "18px"}),
                    dcc.Link("overview", href=url_base, style=nav),
                    dcc.Link("identifications", href=f"{url_base}identifications", style=nav),
                    html.Span("q threshold ", style={"marginLeft": "18px"}),
                    dcc.Input(
                        id="q-input",
                        type="number",
                        value=DEFAULT_THRESHOLD,
                        min=1e-6,
                        max=0.999,
                        step="any",
                        debounce=True,
                        style={"width": "90px"},
                    ),
                    html.Span(
                        f"  {rs.root}",
                        style={
                            "color": "#666",
                            "fontFamily": "monospace",
                            "marginLeft": "18px",
                            "fontSize": "0.85em",
                        },
                    ),
                    html.Span(f"viewer {__version__}", style={"float": "right", "color": "#999"}),
                ],
                style={
                    "padding": "10px 16px",
                    "borderBottom": "1px solid #ddd",
                    "background": "#fafafa",
                },
            ),
            dcc.Loading(html.Div(id="page", style={"padding": "10px 18px"}), delay_show=300),
        ],
        style={"fontFamily": "system-ui, Segoe UI, sans-serif"},
    )

    @app.callback(Output("threshold", "data"), Input("q-input", "value"))
    def set_threshold(value):
        try:
            t = float(value)
        except (TypeError, ValueError):
            return DEFAULT_THRESHOLD
        return t if 0 < t < 1 else DEFAULT_THRESHOLD

    @app.callback(
        Output("page", "children"),
        Input("url", "pathname"),
        Input("url", "search"),
        Input("threshold", "data"),
    )
    def route(pathname, search, t):
        path = (pathname or "/")[len(url_base.rstrip("/")) :].strip("/")
        query = {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}
        if path == "identifications":
            return browser.layout(get_rs())
        if path == "precursor":
            try:
                cid = int(query.get("cid", ""))
            except ValueError:
                return html.Div("no candidate id in the address")
            return detail.layout(get_rs(), url_base, query.get("run", ""), cid)
        return overview.layout(get_rs(), float(t or DEFAULT_THRESHOLD))

    browser.register(app, get_rs, url_base)
    detail.register(app, get_rs)
    return app
