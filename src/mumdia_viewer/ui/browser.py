"""Identification browser (P0 view 2): server-side filtering, sorting and paging."""

from __future__ import annotations

from dash import Input, Output, State, dash_table, dcc, html, no_update

from mumdia_viewer.data import ViewerError
from mumdia_viewer.data.tables import TableQuery, identification_table

from .components import MUTED, SECTION, fmt, precursor_href

UNIT_OPTIONS = [
    {"label": "precursors", "value": "precursor"},
    {"label": "peptides", "value": "peptide"},
    {"label": "protein groups", "value": "protein_group"},
]
Q_COLUMNS = [
    "(unit default)",
    "q_value",
    "run_psm_q",
    "precursor_q",
    "peptide_q_value",
    "pg_q_value",
    "experiment_psm_q",
    "global_q_value",
]
HIDDEN = {"source", "file_row_number", "rn", "pos"}
PAGE_SIZE = 25


def layout(ctx) -> html.Div:
    rs = ctx.rs
    runs = [{"label": "all runs", "value": ""}] + [
        {"label": r.label, "value": r.name} for r in rs.runs if r.name
    ]
    control = {"marginRight": "10px", "minWidth": "160px"}
    return html.Div(
        [
            html.H2("Identifications"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("unit", style=MUTED),
                            dcc.RadioItems(UNIT_OPTIONS, "precursor", id="tb-unit", inline=True),
                        ],
                        style=control,
                    ),
                    html.Div(
                        [
                            html.Div("q column", style=MUTED),
                            dcc.Dropdown(Q_COLUMNS, Q_COLUMNS[0], id="tb-q", clearable=False),
                        ],
                        style={**control, "minWidth": "200px"},
                    ),
                    html.Div(
                        [
                            html.Div("run", style=MUTED),
                            dcc.Dropdown(
                                runs,
                                "",
                                id="tb-run",
                                clearable=False,
                                disabled=not rs.is_experiment,
                            ),
                        ],
                        style=control,
                    ),
                    html.Div(
                        [
                            html.Div("charge", style=MUTED),
                            dcc.Dropdown(
                                [{"label": "any", "value": 0}]
                                + [{"label": str(z), "value": z} for z in range(1, 8)],
                                0,
                                id="tb-charge",
                                clearable=False,
                            ),
                        ],
                        style=control,
                    ),
                    html.Div(
                        [
                            html.Div("quant", style=MUTED),
                            dcc.Dropdown(
                                ["any", "quantified", "not_quantifiable", "not_selected"],
                                "any",
                                id="tb-quant",
                                clearable=False,
                            ),
                        ],
                        style=control,
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end"},
            ),
            html.Div(
                [
                    dcc.Input(
                        id="tb-search",
                        type="text",
                        debounce=True,
                        placeholder="peptidoform or protein contains",
                        style={"width": "260px"},
                    ),
                    dcc.Input(
                        id="tb-protein",
                        type="text",
                        debounce=True,
                        placeholder="protein contains",
                        style={"width": "200px", "marginLeft": "8px"},
                    ),
                    dcc.Input(
                        id="tb-mod",
                        type="text",
                        debounce=True,
                        placeholder="modification (e.g. Oxidation)",
                        style={"width": "220px", "marginLeft": "8px"},
                    ),
                    dcc.Checklist(
                        [{"label": " show decoys", "value": "decoys"}],
                        [],
                        id="tb-decoys",
                        inline=True,
                        style={"marginLeft": "12px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", **SECTION},
            ),
            html.Div(id="tb-description", style={**MUTED, **SECTION}),
            dash_table.DataTable(
                id="tb-table",
                page_action="custom",
                page_current=0,
                page_size=PAGE_SIZE,
                sort_action="custom",
                sort_mode="single",
                sort_by=[{"column_id": "score", "direction": "desc"}],
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "monospace",
                    "fontSize": "0.85em",
                    "textAlign": "left",
                    "maxWidth": "360px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={"fontWeight": "bold"},
            ),
            html.Div("Click a row to open its precursor detail.", style=MUTED),
        ]
    )


def _query(unit, q, run, charge, quant, search, protein, mod, decoys, t, page, sort_by):
    sort = (sort_by or [{"column_id": "score", "direction": "desc"}])[0]
    return TableQuery(
        unit=unit,
        q_column=None if q == Q_COLUMNS[0] else q,
        threshold=t,
        include_decoys="decoys" in (decoys or []),
        charge=charge or None,
        protein=protein or None,
        modification=mod or None,
        quant_status=None if quant == "any" else quant,
        search=search or None,
        run=run or None,
        sort_by=sort["column_id"],
        descending=sort["direction"] == "desc",
        offset=(page or 0) * PAGE_SIZE,
        limit=PAGE_SIZE,
    )


def register(app, get_rs, base: str) -> None:
    @app.callback(
        Output("tb-table", "data"),
        Output("tb-table", "columns"),
        Output("tb-table", "page_count"),
        Output("tb-description", "children"),
        Input("tb-unit", "value"),
        Input("tb-q", "value"),
        Input("tb-run", "value"),
        Input("tb-charge", "value"),
        Input("tb-quant", "value"),
        Input("tb-search", "value"),
        Input("tb-protein", "value"),
        Input("tb-mod", "value"),
        Input("tb-decoys", "value"),
        Input("threshold", "data"),
        Input("tb-table", "page_current"),
        Input("tb-table", "sort_by"),
    )
    def page(unit, q, run, charge, quant, search, protein, mod, decoys, t, current, sort_by):
        rs = get_rs()
        try:
            result = identification_table(
                rs,
                _query(
                    unit, q, run, charge, quant, search, protein, mod, decoys, t, current, sort_by
                ),
            )
        except (ViewerError, ValueError) as exc:
            return [], [], 1, f"cannot show this table: {exc}"
        df = result.rows
        cols = [c for c in df.columns if c not in HIDDEN]
        labels = result.column_labels or {}
        data = []
        for record in df[cols].to_dict("records"):
            row = {c: fmt(v) for c, v in record.items()}
            row["_run"] = str(record.get("run") or "")
            row["_cid"] = record.get("candidate_id")
            data.append(row)
        columns = [{"name": c, "id": c} for c in cols]
        pages = max(1, -(-result.total // PAGE_SIZE))
        description = f"{result.total:,} rows. {result.description}"
        _ = labels
        return data, columns, pages, description

    @app.callback(
        Output("url", "href"),
        Input("tb-table", "active_cell"),
        State("tb-table", "data"),
        prevent_initial_call=True,
    )
    def open_detail(cell, data):
        if not cell or not data:
            return no_update
        row = data[cell["row"]]
        if row.get("_cid") is None:
            return no_update
        return precursor_href(base, row.get("_run") or "", int(row["_cid"]))
