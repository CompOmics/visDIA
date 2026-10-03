"""The cards of the calibration page (components only; the page and its callbacks are in
:mod:`.calibration`, the figures in :mod:`.calibration_figures`).

Every number comes from :mod:`mumdia_viewer.data.calibration`: the engine's records as
written, or the viewer's rebuild, which says so (a "viewer" badge and a tooltip with
the rule).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import dash_mantine_components as dmc
import numpy as np
from dash import dcc, html

from mumdia_viewer.data import ResultSet
from mumdia_viewer.data.calibration import (
    ANCHOR_RULE,
    IN_SAMPLE,
    MASS_COLUMNS,
    AcceptedErrors,
    Anchors,
    CalRecord,
    MassCalRecord,
    RtModel,
    Scope,
)
from mumdia_viewer.data.features import EVIDENCE_FEATURES

from . import calibration_figures as cfig
from .icons import icon
from .state import href, stop_label
from .widgets import chip, fmt, fmt_q, graph, peptidoform, section, spark_bar, validation_icon

# --------------------------------------------------------------------------- formats


def num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def sec(v: Any, digits: int = 2, signed: bool = False) -> str:
    f = num(v)
    if f is None:
        return "n/a"
    return f"{f:+.{digits}f} s" if signed else f"{f:.{digits}f} s"


def ppm(v: Any, digits: int = 2, signed: bool = True) -> str:
    f = num(v)
    if f is None:
        return "n/a"
    return f"{f:+.{digits}f} ppm" if signed else f"{f:.{digits}f} ppm"


def count(v: Any) -> str:
    return "n/a" if v is None else f"{int(v):,}"


def tip(child: Any, label: Any, **kwargs: Any) -> Any:
    return dmc.Tooltip(child, label=label, **kwargs)


def viewer_badge(text: str) -> Any:
    """The mark of a viewer-derived number (the tooltip says how it was computed)."""
    return tip(
        html.Span("viewer", className="cal-derived"),
        f"Viewer-derived: {text}",
        w=340,
    )


def check_mark(equal: bool | None, text: str, size: int = 16) -> Any:
    """A check (equal to the engine's record), a cross (differs) or a dot (not checked)."""
    if equal is None:
        body = html.Span(className="cal-dot")
        colour = "gray"
    else:
        body = icon("check" if equal else "x", size - 6)
        colour = "green" if equal else "red"
    return tip(
        dmc.ThemeIcon(body, size=size, radius="xl", color=colour, variant="light"),
        text,
        w=340,
    )


def title(
    text: str,
    *,
    count: Any = None,
    count_id: str | None = None,
    help_text: Any = None,
    help_id: str | None = None,
) -> Any:
    """A card title like ``widgets.section``'s, whose count and help a callback can update."""
    parts: list[Any] = [html.Span(text)]
    if count is not None:
        parts.append(
            dmc.Badge(
                count,
                **({"id": count_id} if count_id else {}),
                size="sm",
                color="gray",
                variant="light",
                className="cal-count",
                style={"textTransform": "none"},
            )
        )
    if help_text:
        parts.append(
            dmc.Tooltip(
                html.Span(icon("info", 13), className="mv-help"),
                label=help_text,
                w=440,
                position="bottom-start",
                multiline=True,
                **({"id": help_id} if help_id else {}),
            )
        )
    return html.Span(parts, className="cal-title")


# --------------------------------------------------------------------------- header


def _run_select(rs: ResultSet, run_name: str) -> Any:
    return dmc.SegmentedControl(
        id="cal-run",
        data=[{"value": r.name, "label": r.name} for r in rs.runs],
        value=run_name,
        size="sm",
        radius="md",
        className="cal-run-select",
    )


def _band_select(scopes: Sequence[Scope], key: str) -> Any:
    return dmc.SegmentedControl(
        id="cal-band",
        data=[{"value": s.key, "label": s.key} for s in scopes],
        value=key,
        size="sm",
        radius="md",
    )


def header(
    rs: ResultSet,
    s: Scope,
    scopes: Sequence[Scope],
    anchors: Anchors,
    cal: CalRecord | None,
    model: RtModel,
) -> Any:
    """The page title, the provenance chips and the run (and band) selector."""
    if rs.is_experiment:
        index = next((i for i, r in enumerate(rs.runs) if r.name == s.run.name), 0)
        where = f"Calibration · run {s.run.name} ({index + 1} of {len(rs.runs)})"
    else:
        where = "Calibration · single run"
    if s.band is not None:
        where += f" · band {s.band.name}"
    chips: list[Any] = []
    if model.rt_predictor:
        chips.append(
            chip(
                f"RT {model.rt_predictor}",
                "teal",
                tip="model_identities.rt_predictor of the manifest: the RT model that produced "
                "the library iRT the calibration read",
            )
        )
    if cal is not None:
        status = cal.status or "not recorded"
        colour = "indigo" if status in ("loess", "linear") else "orange"
        chips.append(
            chip(
                f"fit {cal.method or 'not recorded'}",
                colour,
                tip=f"cal.json method {cal.method}; calibration_status {status}"
                + (
                    ". fallback_fixed: too few anchors for a residual window (the configured "
                    "fixed half-window)"
                    if status == "fallback_fixed"
                    else ""
                )
                + (
                    ". Fewer than two anchors: no calibration and unbounded windows"
                    if status == "insufficient_anchors_unbounded"
                    else ""
                ),
            )
        )
        if cal.w_rt_sizing:
            chips.append(
                chip(
                    f"window sizing {cal.w_rt_sizing.replace('_', ' ')}",
                    "gray",
                    tip="cal.json w_rt_sizing: in_sample sizes w_rt from the anchors' own "
                    "residuals; holdout from held-out anchors (rt_im_train.window_holdout_frac)",
                )
            )
    chips.append(
        chip(
            f"q_train {anchors.q_train:g}",
            "gray",
            tip="rt_im_train.q_train: a seed PSM anchors the fit when spectrum_q < q_train",
        )
    )
    if model.library_label:
        chips.append(
            chip(
                model.library_label.split("/")[-1],
                "gray",
                tip=f"The precursor table whose predicted_irt the calibration read: "
                f"{model.library_label}, {model.library_note}.",
                left=icon("layers", 12),
            )
        )
    agree = anchors.agrees
    if agree is not None:
        checked = [c for c in anchors.checks if c.equal is not None]
        lines = "; ".join(f"{c.name} {'equal' if c.equal else 'differs'}" for c in checked)
        chips.append(
            chip(
                "rebuild equals cal.json" if agree else "rebuild differs from cal.json",
                "green" if agree else "red",
                tip=f"The viewer rebuilt the anchors with the engine's rule and recomputed "
                f"cal.json's numbers from them and run_windows: {lines}.",
                left=icon("check" if agree else "alert", 12),
            )
        )
    left = dmc.Stack(
        [
            html.Div(
                [
                    dmc.Text(where, className="mv-eyebrow"),
                    html.Div("RT and mass calibration", className="mv-title"),
                ]
            ),
            dmc.Group(chips, gap=6),
        ],
        gap="sm",
    )
    right: list[Any] = []
    if rs.is_experiment:
        right.append(
            html.Div(
                [dmc.Text("Run", className="cal-sel-label"), _run_select(rs, s.run.name)],
                className="cal-sel",
            )
        )
    if s.band is not None and len(scopes) > 1:
        right.append(
            html.Div(
                [dmc.Text("Band", className="cal-sel-label"), _band_select(scopes, s.key)],
                className="cal-sel",
            )
        )
    return dmc.Group(
        [left, dmc.Group(right, gap="md", align="flex-end") if right else None],
        justify="space-between",
        align="flex-end",
        gap="lg",
        className="cal-hero",
    )


# --------------------------------------------------------------------------- facts


def fact(
    label: str,
    value: str,
    sub: Any,
    text: str,
    colour: str = "indigo",
    *,
    fid: str | None = None,
    derived: bool = False,
) -> Any:
    """A tile of the key numbers: label, value, one short line; the source in the tooltip."""
    head: list[Any] = [html.Span(label, className="cal-fact-label")]
    if derived:
        head.append(html.Span("viewer", className="cal-derived"))
    value_ids = {"id": f"{fid}-value"} if fid else {}
    sub_ids = {"id": f"{fid}-sub"} if fid else {}
    body = html.Div(
        [
            html.Div(head, className="cal-fact-head"),
            html.Div(value, className="cal-fact-value", **value_ids),
            html.Div(sub, className="cal-fact-sub", **sub_ids),
        ],
        className=f"cal-fact cal-fact-{colour}",
    )
    return dmc.Tooltip(
        body,
        label=text,
        w=340,
        position="bottom",
        openDelay=150,
        boxWrapperProps={"w": "100%"},
    )


def accepted_rt_text(e: AcceptedErrors) -> tuple[str, str]:
    """The median |RT error| of the accepted rows and its one-line note."""
    v = e.frame["rt_error"].to_numpy(dtype=np.float64) if e.n else np.zeros(0)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a", "no valid RT error"
    return f"{float(np.median(np.abs(v))):.2f} s", "median |error|, accepted"


def facts(
    anchors: Anchors,
    cal: CalRecord | None,
    m: MassCalRecord | None,
    e: AcceptedErrors,
) -> Any:
    """The key numbers: four of the RT fit, one of the accepted rows, three of the mass."""
    n_train = cal.n_train if cal is not None else None
    w = cal.w_rt if cal is not None else None
    width_sub = (
        f"p{round(100 * cal.p_rt)} |residual| \u00d7 {cal.multiplier:g}"
        if cal is not None and cal.p_rt is not None and cal.multiplier is not None and w
        else "unbounded windows"
        if cal is not None and w is None
        else ""
    )
    med, med_sub = accepted_rt_text(e)
    tiles = [
        fact(
            "anchors",
            count(n_train if n_train is not None else anchors.n),
            "one per peptide",
            "cal.json n_train: the confident target seed PSMs the RT fit used, the best per "
            f"base_peptide_id. The viewer's rebuild finds {anchors.n:,} ({ANCHOR_RULE}).",
        ),
        fact(
            "half-window",
            sec(w) if w is not None else "unbounded",
            width_sub,
            "cal.json w_rt: every candidate's extraction window is rt_pred_cal ± w_rt "
            "(rt_im_train.rs). In-sample sizing: the p_rt percentile of the anchors' "
            "|residual| times rt_im_train.rt_window_multiplier, at least 1 s.",
        ),
        fact(
            "|residual|",
            sec(cal.residual_abs_median_s) if cal else "n/a",
            "median, in-sample",
            "cal.json rt_residual_abs_median_s: the median |observed_rt - rt_pred_cal| over "
            f"the anchors. {IN_SAMPLE}",
        ),
        fact(
            "residual MAD",
            sec(cal.residual_mad_s) if cal else "n/a",
            f"median {sec(cal.residual_median_s, signed=True)}" if cal else "",
            "cal.json rt_residual_mad_s: the median of |r - median(r)| over the anchors' "
            f"residuals r; rt_residual_median_s is the signed median (bias). {IN_SAMPLE}",
        ),
        fact(
            "RT error",
            med,
            med_sub,
            "The median |rt_error_signed| (apex_rt - rt_pred_cal, features.parquet) of the "
            f"accepted identifications ({e.population}). Extraction searches only inside "
            "the window, so the error is truncated at the window edges.",
            "violet",
            fid="cal-fact-acc",
            derived=True,
        ),
        fact(
            "mass offset",
            ppm(m.frag_ppm_offset) if m else "n/a",
            "median deviation",
            "frag_ppm_offset of seed_psms.parquet.masscal.json: the median ppm deviation of the "
            "matched fragments of confident target seed PSMs (masscal.rs). Extract divides "
            "each observed m/z by (1 + offset \u00d7 1e-6) before matching.",
            "cyan",
        ),
        fact(
            "tolerance",
            ppm(m.frag_tol_ppm, signed=False) if m else "n/a",
            "fragments, 1.5 \u00d7 p95",
            "frag_tol_ppm of the masscal: 1.5 \u00d7 the 95th percentile of |deviation - offset|, "
            "floored at 5 ppm (masscal.rs). frag_ppm_sigma holds the same value.",
            "cyan",
        ),
        fact(
            "calibrants",
            count(m.n_dev) if m else "n/a",
            "fragment deviations",
            "n_dev of the masscal: the matched-fragment deviations the mass calibration was "
            "fitted from (fewer than 20 keep the configured search tolerance).",
            "cyan",
        ),
    ]
    return html.Div(tiles, className="cal-facts")


def key_item(label: str, value: str, text: str, kind: str, colour: str) -> Any:
    """One entry of a plot's key: a line sample, a label and a value; the meaning in the
    tooltip."""
    return tip(
        html.Span(
            [
                html.Span(className=f"cal-sw cal-sw-{kind}", style={"--cal-c": colour}),
                html.Span(label, className="cal-key-label"),
                html.Span(value, className="cal-key-value") if value else None,
            ],
            className="cal-key",
        ),
        text,
        w=320,
    )


def key_row(items: Sequence[Any], id_: str | None = None) -> Any:
    return html.Div(list(items), className="cal-keys", **({"id": id_} if id_ else {}))


# --------------------------------------------------------------------------- point bars


HINT_ANCHOR = "Hover an anchor for its peptidoform and errors; click it to open its precursor page."
HINT_ID = (
    "Hover an identification for its peptidoform and errors; click it to open its precursor page."
)


def point_hint(text: str) -> Any:
    return html.Div(
        [html.Span(icon("info", 13), className="mv-help"), html.Span(text)],
        className="cal-point-hint",
    )


def _bar(items: list[Any], link: Any) -> Any:
    return html.Div([html.Div(items, className="cal-point-items"), link], className="cal-point-row")


def _kv(label: str, value: str, text: str | None = None) -> Any:
    body = html.Span(
        [html.Span(label, className="cal-pk"), html.Span(value, className="cal-pv")],
        className="cal-pkv",
    )
    return tip(body, text, multiline=False, w="auto") if text else body


def _open(base: str, rs: ResultSet, s: Scope, cid: int) -> Any:
    run = s.run.name if rs.is_experiment else ""
    return dcc.Link(
        dmc.Button(
            "Open precursor page",
            rightSection=icon("right", 13),
            size="compact-xs",
            radius="md",
            variant="light",
        ),
        href=href(base, "precursor", {"run": run, "cid": int(cid)}),
        className="cal-open",
    )


def anchor_bar(rs: ResultSet, base: str, a: Anchors, row: int | None, threshold: float) -> Any:
    """The hovered anchor: its peptidoform, scan, RTs, residual and scored row."""
    if row is None or not 0 <= int(row) < a.n:
        return point_hint(HINT_ANCHOR)
    r = a.frame.iloc[int(row)]
    inside = bool(r["in_window"])
    items = [
        html.Span(peptidoform(str(r["peptidoform"]), size="0.92rem"), className="cal-point-pep"),
        dmc.Badge(
            f"{int(r['charge'])}+",
            size="sm",
            variant="outline",
            color="gray",
            style={"textTransform": "none"},
        ),
        _kv("scan", str(int(r["scan_index"])), "scan_index of the seed PSM"),
        _kv("observed", sec(r["observed_rt"], 1), "observed_rt of the seed PSM (seconds)"),
        _kv(
            "rt_pred_cal",
            sec(r["rt_pred_cal"], 1),
            "run_windows rt_pred_cal at the anchor's candidate",
        ),
        _kv(
            "residual", sec(r["residual"], 2, signed=True), "observed_rt - rt_pred_cal (in-sample)"
        ),
        chip(
            "in its window" if inside else "outside its window",
            "green" if inside else "yellow",
            size="sm",
            tip=f"[rt_lo, rt_hi] = [{r['rt_lo']:.1f}, {r['rt_hi']:.1f}] s",
        ),
        _kv("spectrum_q", fmt_q(r["spectrum_q"]), "spectrum_q of the seed PSM"),
    ]
    if bool(r["scored"]):
        items.append(
            html.Span(
                [
                    validation_icon(r["q"], threshold, column=a.q_column, size=16),
                    html.Span(f"{a.q_column} {fmt_q(r['q'], threshold)}", className="cal-pv"),
                ],
                className="cal-pkv",
            )
        )
        link: Any = _open(base, rs, a.scope, int(r["candidate_id"]))
    else:
        items.append(
            chip(
                "not a scored row",
                "gray",
                size="sm",
                tip="The run has no scored row of this candidate (it did not pass extraction), "
                "so it has no precursor page.",
            )
        )
        link = html.Span()
    return _bar(items, link)


def id_bar(rs: ResultSet, base: str, e: AcceptedErrors, row: int | None, column: str) -> Any:
    """The hovered accepted identification: its peptidoform, RT error and mass error."""
    if row is None or not 0 <= int(row) < e.n:
        return point_hint(HINT_ID)
    r = e.frame.iloc[int(row)]
    rel = num(r["rt_error_rel"])
    items = [
        html.Span(peptidoform(str(r["peptidoform"]), size="0.92rem"), className="cal-point-pep"),
        dmc.Badge(
            f"{int(r['charge'])}+",
            size="sm",
            variant="outline",
            color="gray",
            style={"textTransform": "none"},
        ),
        _kv("apex", sec(r["apex_rt"], 1), "apex_rt of the scored row"),
        _kv(
            "RT error", sec(r["rt_error"], 2, signed=True), "rt_error_signed: apex_rt - rt_pred_cal"
        ),
        _kv(
            "of window",
            f"{rel:+.3f}" if rel is not None else "n/a",
            "viewer-derived: rt_error_signed / the candidate's half-window",
        ),
        _kv(
            column.replace("_", " "),
            ppm(r[column]),
            f"{column} of the feature table (raw ppm)",
        ),
        html.Span(
            [
                validation_icon(r["q"], e.threshold, column=e.q_column, size=16),
                html.Span(f"{e.q_column} {fmt_q(r['q'], e.threshold)}", className="cal-pv"),
            ],
            className="cal-pkv",
        ),
    ]
    return _bar(items, _open(base, rs, e.scope, int(r["candidate_id"])))


def point_bar(id_: str, hint: str) -> Any:
    return html.Div(point_hint(hint), id=id_, className="cal-point")


# --------------------------------------------------------------------------- cards


def _switch(id_: str, data: list[tuple[str, str]], value: str) -> Any:
    return dmc.SegmentedControl(
        id=id_,
        data=[{"value": v, "label": label} for v, label in data],
        value=value,
        size="xs",
        radius="xl",
    )


def _q_chip(id_: str, text: str) -> Any:
    return dmc.Badge(
        text,
        id=id_,
        color="indigo",
        variant="light",
        size="sm",
        className="mv-kpi-q",
        style={"textTransform": "none"},
    )


def fit_card(a: Anchors, cal: CalRecord | None, scheme: str) -> Any:
    help_text = (
        f"The RT calibration anchors, rebuilt by the viewer with the engine's rule: "
        f"{ANCHOR_RULE}. Each anchor is a seed PSM (seed_psms.parquet) at its library iRT "
        f"({a.irt_source}). The line is the engine's fitted map, read from "
        f"{a.windows_source} at the anchors and at evenly spaced library rows; the band is "
        "the extraction window [rt_lo, rt_hi]. Beyond the anchors' iRT range the engine "
        "continues the curve linearly (dashed; calibrate.rs). Lower panel: the in-sample "
        "residual observed_rt - rt_pred_cal on the same iRT axis, with the half-window band. "
        "Zoom here and the RT error plots follow."
    )
    n_out = int((~a.frame["in_window"].astype(bool)).sum()) if a.n else 0
    w = cal.w_rt if cal is not None else None
    limit = cfig.residual_limit(a) if a.n else None
    n_cut = int((a.frame["residual"].abs() > limit).sum()) if a.n and limit is not None else 0
    items = [
        key_item(
            "anchors",
            f"{a.n - n_out:,}",
            "anchors whose observed RT lies in their window [rt_lo, rt_hi]",
            "dot",
            cfig.INSIDE,
        ),
        key_item(
            "outside their window",
            f"{n_out:,}",
            "anchors whose observed RT lies "
            "outside [rt_lo, rt_hi] (viewer-derived from run_windows)",
            "dot",
            cfig.OUTSIDE,
        ),
        key_item(
            "fitted map",
            "",
            "rt_pred_cal of run_windows; dashed beyond the anchors' iRT range",
            "line",
            cfig.CURVE,
        ),
        key_item(
            "window",
            f"± {w:.2f} s" if w is not None else "unbounded",
            "the extraction window [rt_lo, rt_hi] of run_windows (cal.json w_rt)",
            "band",
            "#868e96",
        ),
    ]
    if n_cut and limit is not None:
        items.append(
            key_item(
                "beyond the panel",
                f"{n_cut:,}",
                f"residuals beyond ± {limit:.1f} s, "
                "drawn at the edge of the lower panel as triangles; hover one for its "
                "residual",
                "tri",
                cfig.OUTSIDE,
            )
        )
    return section(
        "RT calibration",
        key_row(items),
        point_bar("cal-pt-fit", HINT_ANCHOR),
        graph("cal-fit", cfig.fit_figure(a, scheme)),
        count=f"{a.n:,} anchors",
        help=help_text,
        subtitle="Seed anchors against the library iRT, and their residuals",
        id="cal-fit-card",
    )


def _stat_row(label: str, value: str, check: Any, text: str) -> Any:
    return html.Div(
        [
            tip(html.Span(label, className="cal-kv-label"), text, w=340, position="top-start"),
            html.Span(value, className="cal-kv-value"),
            check if check is not None else html.Span(),
        ],
        className="cal-kv cal-kv-check",
    )


def residual_card(a: Anchors, cal: CalRecord | None, scheme: str) -> Any:
    by = {c.name: c for c in a.checks}

    def mark(name: str) -> Any:
        c = by.get(name)
        if c is None:
            return None
        if c.equal is None:
            return check_mark(None, f"Not checked: {c.note or 'no value'}.")
        rebuilt = f"{c.rebuilt:.6g}" if isinstance(c.rebuilt, float) else f"{c.rebuilt}"
        recorded = f"{c.recorded:.6g}" if isinstance(c.recorded, float) else f"{c.recorded}"
        word = "equals" if c.equal else "differs from"
        extra = f" ({c.note})" if c.note else ""
        return check_mark(
            c.equal,
            f"The viewer's value {rebuilt}{extra} {word} cal.json's {recorded}.",
        )

    n_out = int((~a.frame["in_window"].astype(bool)).sum()) if a.n else 0
    rows = [
        _stat_row(
            "anchors (n_train)",
            count(cal.n_train if cal else None),
            mark("n_train"),
            "cal.json n_train; the mark compares it with the viewer's rebuild.",
        ),
        _stat_row(
            "signed median",
            sec(cal.residual_median_s, signed=True) if cal else "n/a",
            mark("rt_residual_median_s"),
            "cal.json rt_residual_median_s: the residual bias (nearest-rank median).",
        ),
        _stat_row(
            "absolute median",
            sec(cal.residual_abs_median_s) if cal else "n/a",
            mark("rt_residual_abs_median_s"),
            "cal.json rt_residual_abs_median_s.",
        ),
        _stat_row(
            "MAD",
            sec(cal.residual_mad_s) if cal else "n/a",
            mark("rt_residual_mad_s"),
            "cal.json rt_residual_mad_s: the median of |r - median(r)|.",
        ),
        _stat_row(
            "half-window w_rt",
            sec(cal.w_rt) if cal and cal.w_rt is not None else "unbounded",
            mark("w_rt"),
            "cal.json w_rt.",
        ),
        _stat_row(
            "anchors outside their window",
            f"{n_out:,} of {a.n:,}",
            None,
            "Viewer-derived: anchors with observed_rt outside [rt_lo, rt_hi]. The window is the "
            "p_rt percentile of |residual| times the multiplier, so few anchors lie outside.",
        ),
    ]
    note = dmc.Text(IN_SAMPLE, size="xs", c="dimmed", mt=6)
    keys = []
    if cal is not None and cal.residual_median_s is not None:
        keys.append(
            key_item(
                "median",
                sec(cal.residual_median_s, signed=True),
                "cal.json rt_residual_median_s",
                "dash",
                cfig.CURVE,
            )
        )
    if cal is not None and cal.w_rt is not None:
        keys.append(
            key_item(
                "window",
                f"± {cal.w_rt:.2f} s",
                "cal.json w_rt: the half-window",
                "dashdot",
                "#868e96",
            )
        )
        p = cal.p_rt_width
        if p is not None and cal.p_rt is not None and abs(p - cal.w_rt) > 1e-9:
            keys.append(
                key_item(
                    f"p{round(100 * cal.p_rt)}",
                    f"± {p:.2f} s",
                    "w_rt / multiplier: the residual percentile behind w_rt",
                    "dotted",
                    "#868e96",
                )
            )
    return section(
        "Residuals (in-sample)",
        key_row(keys),
        graph("cal-resid", cfig.residual_figure(a, cal, scheme)),
        html.Div(rows, className="cal-kv-list"),
        note,
        count=f"{a.n:,}",
        help="The anchors' residuals observed_rt - rt_pred_cal, in seconds. The numbers are "
        "cal.json's; each mark compares one with the viewer's recomputation from the "
        "rebuilt anchors and run_windows (the engine's nearest-rank percentile). " + IN_SAMPLE,
        subtitle="observed_rt - rt_pred_cal of the anchors",
        id="cal-resid-card",
    )


def error_help(e: AcceptedErrors) -> str:
    return (
        f"The accepted identifications: {e.population}. y: rt_error_signed of the feature "
        "table, apex_rt - rt_pred_cal (seconds; positive when the peptide eluted later than "
        "predicted), or that error over the candidate's half-window (viewer-derived). "
        f"Half-window: {e.half_width_source}. The lines are the 5th, 50th and 95th "
        f"percentiles of the error in {cfig.RT_BINS} equal bins of apex RT (viewer-derived, "
        "numpy). Extraction searches only inside the window, so the error is truncated at "
        "its edges. Zoom the gradient here and the RT plots follow."
    )


def error_foot(e: AcceptedErrors) -> str:
    parts = []
    bad = e.counts.get("nan_rt_error", 0)
    if bad:
        parts.append(
            f"{bad:,} rows without a valid RT error (no feature row, or predicted_rt_raw = 0: "
            "no RT calibration) are not drawn"
        )
    parts += e.notes
    return "; ".join(parts) + ("." if parts else "")


def error_keys(e: AcceptedErrors) -> list[Any]:
    """The key of the RT error plot."""
    hw = e.frame["half_width"].to_numpy(dtype=np.float64) if e.n else np.zeros(0)
    hw = hw[np.isfinite(hw)]
    if hw.size and float(hw.min()) == float(hw.max()):
        window = f"± {float(hw[0]):.2f} s"
    elif hw.size:
        window = "per candidate"
    else:
        window = "unbounded"
    return [
        key_item("accepted", f"{e.n:,}", e.population, "dot", cfig.INSIDE),
        key_item(
            "median",
            "",
            f"the median of the error in {cfig.RT_BINS} bins of apex RT (viewer-derived)",
            "line",
            cfig.QUANTILE_LINE,
        ),
        key_item(
            "5th and 95th percentile",
            "",
            "in the same bins (viewer-derived)",
            "dotted",
            cfig.QUANTILE_LINE,
        ),
        key_item("window", window, f"the half-window: {e.half_width_source}", "dashdot", "#868e96"),
    ]


def error_card(e: AcceptedErrors, scheme: str, view: str = "seconds") -> Any:
    q_text = f"{e.q_column} ≤ {stop_label(e.threshold)}"
    return section(
        title(
            "RT error",
            count=f"{e.n:,}",
            count_id="cal-err-count",
            help_text=error_help(e),
            help_id="cal-err-help",
        ),
        key_row(error_keys(e), id_="cal-err-keys"),
        point_bar("cal-pt-err", HINT_ID),
        graph("cal-err", cfig.error_figure(e, scheme, view=view)),
        html.Div(error_foot(e), id="cal-err-foot", className="cal-foot"),
        right=dmc.Group(
            [
                _q_chip("cal-err-q", q_text),
                _switch(
                    "cal-err-view",
                    [("seconds", "Seconds"), ("fraction", "Of the window")],
                    view,
                ),
            ],
            gap=8,
            wrap="nowrap",
        ),
        subtitle="Accepted identifications: apex_rt - rt_pred_cal across the gradient",
        id="cal-err-card",
    )


def funnel_card(a: Anchors) -> Any:
    rows = cfig.funnel_rows(a.funnel)
    if not rows:
        body: Any = dmc.Text(a.error or "No seed table.", size="sm", c="dimmed")
    else:
        top = max(1, rows[0][1])
        items = []
        short = {
            "seed rows": "seed rows",
            "finite spectrum_q, score and observed_rt": "finite values",
            "spectrum_q < q_train": f"spectrum_q < {a.q_train:g}",
            "label target": "targets",
            "finite library predicted_irt": "with a library iRT",
            "anchors: best score per base_peptide_id": "best per base peptide",
        }
        for i, (step, n, removed) in enumerate(rows):
            last = i == len(rows) - 1
            items.append(
                html.Div(
                    [
                        tip(
                            html.Span(
                                short.get(step, step),
                                className="cal-fn-step" + (" cal-fn-last" if last else ""),
                            ),
                            step,
                            multiline=False,
                            w="auto",
                            position="top-start",
                        ),
                        spark_bar(
                            max(n, 1),
                            lo=1,
                            hi=top,
                            scale="log10",
                            colour="var(--mantine-color-indigo-6)"
                            if last
                            else "var(--mantine-color-gray-5)",
                            text=f"{n:,}",
                            width=48,
                            tip=f"{n:,} rows left; the bar is log10 from 1 to {top:,} seed rows",
                        ),
                        html.Span(
                            f"-{removed:,}" if i and removed else "", className="cal-fn-removed"
                        ),
                    ],
                    className="cal-fn-row",
                )
            )
        body = html.Div(items, className="cal-fn")
    return section(
        "Anchor selection",
        body,
        dmc.Text(
            f"Seed table {a.seed_source}; q_train {a.q_train:g} (rt_im_train.q_train).",
            size="xs",
            c="dimmed",
            mt=6,
        )
        if a.seed_source
        else None,
        count=f"{a.n:,}" if a.funnel else None,
        help=f"The viewer's rebuild of the anchors, step by step: {ANCHOR_RULE}. The counts are "
        "the seed rows left after each step; the last is the anchors, one per base peptide.",
        id="cal-funnel-card",
    )


def _kv_row(label: str, value: Any, text: str | None = None) -> Any:
    left: Any = html.Span(label, className="cal-kv-label")
    if text:
        left = tip(left, text, w=340, position="top-start")
    return html.Div([left, html.Span(value, className="cal-kv-value")], className="cal-kv")


def model_card(model: RtModel) -> Any:
    rows: list[Any] = [
        _kv_row(
            "RT model",
            model.rt_predictor or "not recorded",
            "model_identities.rt_predictor of the manifest",
        ),
        _kv_row(
            "iRT table",
            html.Span(model.library_label.split("/")[-1] or "not found", className="cal-mono"),
            f"{model.library_label or 'no table'}: {model.library_note}",
        ),
    ]
    if model.owner and model.library_note.startswith("the") and "reused" in model.library_note:
        rows.append(
            _kv_row(
                "adapted by",
                f"run {model.owner} (reused)",
                f"This run's calibration read {model.library_note}.",
            )
        )
    summary = model.summary or {}
    mh = model.multihead
    if mh:
        rows += [
            _kv_row(
                "model",
                str(summary.get("model") or "multi-head calibration"),
                "summary.json model: what the DeepLC worker fitted (deeplc_finetune.py)",
            ),
            _kv_row(
                "heads",
                f"{len(model.heads)} of {mh.get('heads_requested', '?')} requested",
                "multihead.heads: the LC-setup heads of the DeepLC base model that the ridge "
                "combines (deeplc MultiHeadRidgeCalibration)",
            ),
            _kv_row(
                "best head",
                str(model.best_head) if model.best_head is not None else "n/a",
                "multihead.best_head: the head that correlates best with this run's anchors",
            ),
            _kv_row(
                "ridge alpha",
                fmt(float(mh["ridge_alpha"])) if mh.get("ridge_alpha") is not None else "n/a",
                "multihead.ridge_alpha of the ridge combination",
            ),
            _kv_row(
                "worker anchors",
                count(mh.get("anchors")),
                "multihead.anchors: the reference the worker fitted on, confident target seed "
                "peptidoforms (spectrum_q ≤ q_train, one per peptidoform; deeplc_finetune.py). "
                "The LOESS anchors are chosen differently (spectrum_q < q_train, one per base "
                "peptide), so the two counts differ.",
            ),
        ]
    if summary.get("rows") is not None:
        rows.append(
            _kv_row(
                "rows predicted",
                f"{int(summary.get('repredicted') or 0):,} of {int(summary['rows']):,}",
                "summary.json repredicted of rows; retained_* rows keep their imported iRT",
            )
        )
    timings = summary.get("timings_s") if isinstance(summary.get("timings_s"), dict) else {}
    if timings:
        text = ", ".join(
            f"{k} {float(v):.0f} s"
            for k, v in timings.items()
            if isinstance(v, int | float) and k in ("fit", "predict", "featurisation")
        )
        if text:
            rows.append(_kv_row("worker time", text, "summary.json timings_s"))
    heads = None
    if model.heads:
        best = model.best_head
        chips_ = [
            html.Span(str(h), className="cal-head" + (" cal-head-best" if h == best else ""))
            for h in model.heads
        ]
        heads = dmc.Spoiler(
            html.Div(chips_, className="cal-heads"),
            maxHeight=44,
            showLabel=f"Show all {len(model.heads)} heads",
            hideLabel="Show fewer",
            className="cal-heads-spoiler",
            mt=6,
        )
    return section(
        "RT model",
        html.Div(rows, className="cal-kv-list"),
        heads,
        dmc.Text(f"Summary {model.summary_path.name}.", size="xs", c="dimmed", mt=4)
        if model.summary_path
        else dmc.Text(
            "No DeepLC summary next to the iRT table (no multi-head calibration or fine-tune).",
            size="xs",
            c="dimmed",
            mt=4,
        ),
        help="The precursor table whose predicted_irt the RT calibration read, and the model "
        "that wrote that iRT. A multi-head calibration ranks the DeepLC base model's LC-setup "
        "heads against this run's anchors and ridge-combines the best (docs/08, 4d).",
        id="cal-model-card",
    )


def masscal_card(
    m: MassCalRecord | None,
    mz_range: tuple[float, float, str] | None,
    extract_text: str,
    seed_tol: float | None,
    scheme: str,
) -> Any:
    if m is None:
        return section(
            "Fragment mass calibration",
            dmc.Text("No seed_psms.parquet.masscal.json in this scope.", size="sm", c="dimmed"),
            id="cal-masscal-card",
        )
    grid = f"{m.grid_mz.size}-node m/z grid" if m.uses_grid else "none (constant offset)"
    rows = [
        _kv_row(
            "offset", ppm(m.frag_ppm_offset), "frag_ppm_offset: the median calibrant deviation"
        ),
        _kv_row(
            "tolerance",
            ppm(m.frag_tol_ppm, signed=False),
            "frag_tol_ppm: 1.5 \u00d7 p95(|deviation - offset|), floored at 5 ppm (masscal.rs)",
        ),
        _kv_row("calibrants", count(m.n_dev), "n_dev: the matched-fragment deviations of the fit"),
        _kv_row(
            "fit passes",
            count(m.cal_passes),
            "cal_passes: 2 when search_seed.two_pass_mass_cal refitted on the first-pass inliers; "
            "0 when there were fewer than 20 calibrants",
        ),
        _kv_row(
            "residual median / MAD",
            f"{ppm(m.ppm_residual_median)} / {ppm(m.ppm_residual_mad, signed=False)}",
            "ppm_residual_median and ppm_residual_mad: the deviations after the offset "
            "correction (diagnostics, in-sample)",
        ),
        _kv_row("m/z grid", grid, "mz_cal_grid_mz and mz_cal_grid_ppm of the masscal"),
    ]
    if seed_tol is not None:
        rows.append(
            _kv_row(
                "seed search tolerance",
                ppm(seed_tol, signed=False, digits=0),
                "search_seed.fragment_tol_ppm: the tolerance of the seed search that found the "
                "calibrants (not the extraction's)",
            )
        )
    source = f"; m/z range {mz_range[2]}" if mz_range else ""
    return section(
        "Fragment mass calibration",
        graph("cal-offset", cfig.offset_figure(m, mz_range, scheme)),
        html.Div(rows, className="cal-kv-list"),
        dmc.Text(f"Extract used: {extract_text}.", size="xs", c="dimmed", mt=6),
        count=None,
        help="The fragment mass calibration of the seed search (seed_psms.parquet.masscal.json, "
        "masscal.rs): the offset extract applies to every observed m/z, against m/z, with the "
        f"tolerance band. {grid.capitalize()}{source}.",
        subtitle="The offset extract applied, against fragment m/z",
        id="cal-masscal-card",
    )


def mass_keys(m: MassCalRecord | None) -> list[Any]:
    """The key of the mass error plot: the run's offset and tolerance edges."""
    if m is None or m.frag_ppm_offset is None:
        return []
    items = [
        key_item(
            "frag_ppm_offset",
            ppm(m.frag_ppm_offset),
            "the run's fragment mass offset (masscal.json): the raw errors centre on it",
            "dash",
            cfig.MASS,
        )
    ]
    if m.frag_tol_ppm is not None:
        items.append(
            key_item(
                "tolerance",
                f"± {m.frag_tol_ppm:.2f} ppm",
                "frag_tol_ppm around the offset: the matching tolerance of extract",
                "dashdot",
                cfig.MASS,
            )
        )
    items.append(
        key_item(
            "percentiles across the gradient",
            "",
            "the median, 5th and 95th percentile in bins of apex RT (viewer-derived), "
            "in the view across the gradient",
            "line",
            cfig.QUANTILE_LINE,
        )
    )
    return items


MASS_LABELS = {"frag_mass_err_median": "Median", "signed_mean_frag_ppm": "Mean"}


def mass_help(e: AcceptedErrors, column: str) -> str:
    info = EVIDENCE_FEATURES.get(column)
    what = info.description if info is not None else column
    return (
        f"{column} of the feature table, one value per accepted identification "
        f"({e.population}): {what} Raw ppm: the values centre on the run's frag_ppm_offset, "
        "not on 0. A row with no fragment observed at the apex scan has the engine's 0 "
        "sentinel and is not drawn. Across the gradient: the values against apex RT, with "
        "the 5th, 50th and 95th percentiles in bins of apex RT (viewer-derived)."
    )


def mass_foot(e: AcceptedErrors, column: str) -> str:
    bad = e.counts.get(f"nan_{column}", 0)
    if not bad:
        return ""
    return f"{bad:,} rows without a valid value (no fragment at the apex scan) are not drawn."


def mass_card(
    e: AcceptedErrors,
    m: MassCalRecord | None,
    scheme: str,
    column: str = "frag_mass_err_median",
    view: str = "distribution",
) -> Any:
    q_text = f"{e.q_column} ≤ {stop_label(e.threshold)}"
    return section(
        title(
            "Fragment mass error",
            count=f"{e.n:,}",
            count_id="cal-mass-count",
            help_text=mass_help(e, column),
            help_id="cal-mass-help",
        ),
        key_row(mass_keys(m)),
        point_bar("cal-pt-mass", HINT_ID),
        graph("cal-mass", cfig.mass_figure(e, m, scheme, column=column, view=view)),
        html.Div(mass_foot(e, column), id="cal-mass-foot", className="cal-foot"),
        right=dmc.Group(
            [
                _q_chip("cal-mass-q", q_text),
                dmc.Tooltip(
                    _switch(
                        "cal-mass-col",
                        [(c, MASS_LABELS.get(c, c)) for c in MASS_COLUMNS],
                        column,
                    ),
                    label="Median: frag_mass_err_median; Mean: signed_mean_frag_ppm (the "
                    "fragments observed at the apex scan, each averaged over the RT window)",
                    w=300,
                ),
                _switch(
                    "cal-mass-view",
                    [("distribution", "Distribution"), ("gradient", "Across the gradient")],
                    view,
                ),
            ],
            gap=8,
            wrap="nowrap",
        ),
        subtitle="Accepted identifications, raw ppm per row",
        id="cal-mass-card",
    )


def notes_card(notes: Sequence[str]) -> Any:
    if not notes:
        return None
    rows = [
        html.Div(
            [html.Span(icon("info", 14), className="cal-note-icon"), html.Span(n)],
            className="cal-note",
        )
        for n in notes
    ]
    return section("Notes", html.Div(rows, className="cal-notes"), count=len(notes), p="sm")
