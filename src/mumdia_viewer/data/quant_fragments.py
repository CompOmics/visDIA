"""The fragments behind a precursor quantity: which fragments quant used, and their areas.

MuMDIA's ``peptide_quant`` keeps a candidate's quantity, the number of fragments used
(``n_fragments_used``) and the integration window, but not which fragments. Three sources
can say which, tried in this order for one candidate in one run:

1. **engine** (``quant_fragments.parquet``, ``quant_fragments`` v1): written by the engine
   under ``quant.fragment_detail`` (MuMDIA 0.10 and later; the desktop application sets it).
   One row per fragment of every candidate the final quant reached: ``area`` (after any
   baseline and envelope, 0 kept), ``selected``, ``weight``, ``rank``, ``background`` and the
   ``rule`` that made the quantity. It describes every rule, the cross-run step and the second
   pass. Nothing is derived; the viewer only checks that the selected fragments add up to the
   quantity.
2. **fragment_quant** (single runs only): the areas of ``fragment_quant.parquet`` (v1 or v5,
   written by the run's final quant) with the configured top-N rule applied by the viewer
   (``quant.top_n_fragments``; ``observed_area`` ranks by area, every other selection falls
   back to ``predicted`` in a single run and ranks by ``predicted_intensity``). The areas are
   the engine's, after any baseline and envelope.
3. **chromatograms**: the viewer integrates each fragment trace over the reported window
   (the engine's trapezoid on float32 retention times, the fixed scan or seconds window
   around ``integration_apex_rt``, or the walked window ``integration_lo_rt`` to
   ``integration_hi_rt``) and applies the top-N rule. Only without baseline subtraction,
   without the interference envelope, with the window centred on the apex, and, in an
   experiment, without the cross-run step, whose fitted weights and backgrounds are not
   written anywhere else.

A viewer-derived selection (2 or 3) is used only when its selected areas reproduce
``peptide_quant.quantity``; it is labelled "viewer-derived, verified". In an experiment the
run directories' ``fragment_quant.parquet`` is the cross-run step's first-pass export, which
describes an intermediate quantity, so it is never used. When no source reproduces the
quantity, the account says so and names the fix: ``quant.fragment_detail``.

Under the pattern rules (``pattern_fit``, ``pattern_profile``) the engine's quantity is a fit
of the selected fragments' samples to the library pattern, not a sum of areas: the engine
table names the fitted fragments (``weight`` NaN) and the check is their count.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .chromatograms import CandidateChromatogram, ChromatogramSource
from .discovery import ResultSet, Run
from .duck import sql_path
from .errors import ViewerError
from .quant import QuantState, quant_state

__all__ = [
    "COLUMNS",
    "NOT_REPRODUCIBLE",
    "SOURCE_CHROMATOGRAMS",
    "SOURCE_ENGINE",
    "SOURCE_FRAGMENT_QUANT",
    "SOURCE_NONE",
    "AcrossFragments",
    "QuantFragments",
    "QuantInputs",
    "fragment_areas_across",
    "quant_fragments",
    "reproduce",
]

SOURCE_ENGINE = "engine"
SOURCE_FRAGMENT_QUANT = "fragment_quant"
SOURCE_CHROMATOGRAMS = "chromatograms"
SOURCE_NONE = "none"

#: What the page says when no source reproduces the quantity.
NOT_REPRODUCIBLE = (
    "not reproducible from this run's tables; run the search with quant.fragment_detail = "
    "true to record the fragments quant used"
)

#: Columns of :attr:`QuantFragments.rows`, in order.
COLUMNS = ("fragment_name", "area", "selected", "weight", "rank", "background")

#: The rules of ``quant.fragment_selection`` whose quantity is a sum of selected areas.
TOP_N_RULES = ("observed_area", "predicted")
PATTERN_RULES = ("pattern_fit", "pattern_profile")

# Relative tolerance of the check. The engine's sums reproduce bit for bit when added in
# its order; the tolerance only covers a reader that adds them in another order.
_REL_TOL = 1e-12

_SOURCE_LABELS = {
    SOURCE_ENGINE: "engine (quant_fragments.parquet)",
    SOURCE_FRAGMENT_QUANT: "viewer-derived, verified (fragment_quant areas and the top-N rule)",
    SOURCE_CHROMATOGRAMS: "viewer-derived, verified (chromatogram areas and the top-N rule)",
    SOURCE_NONE: "not available",
}


@dataclass(frozen=True)
class QuantInputs:
    """What the account is checked against: one ``peptide_quant`` row (None when absent)."""

    state: str
    quantity: float | None
    n_fragments_used: int | None
    integration_apex_rt: float | None
    integration_lo_rt: float | None
    integration_hi_rt: float | None

    @classmethod
    def of(cls, q: QuantState | Mapping[str, Any] | None) -> QuantInputs | None:
        if q is None:
            return None
        if isinstance(q, QuantState):
            return cls(
                state=q.state,
                quantity=q.quantity,
                n_fragments_used=q.n_fragments_used,
                integration_apex_rt=q.integration_apex_rt,
                integration_lo_rt=q.integration_lo_rt,
                integration_hi_rt=q.integration_hi_rt,
            )
        n_used = _finite(q.get("n_fragments_used"))
        return cls(
            state=str(q.get("state") or ""),
            quantity=_finite(q.get("quantity")),
            n_fragments_used=None if n_used is None else int(n_used),
            integration_apex_rt=_finite(q.get("integration_apex_rt")),
            integration_lo_rt=_finite(q.get("integration_lo_rt")),
            integration_hi_rt=_finite(q.get("integration_hi_rt")),
        )


@dataclass(frozen=True)
class QuantFragments:
    """The fragments behind one candidate's quantity in one run.

    ``rows`` has :data:`COLUMNS`, one row per fragment, in chromatogram order. ``rank`` is
    1-based within the rule (0: not ranked); ``weight`` is 1 under the top-N rules, the
    fitted cross-run weight under ``cross_run`` and NaN under the pattern rules;
    ``background`` is NaN where none was subtracted or the source does not know it.
    ``reproduced`` is the quantity the selected fragments give (None under the pattern
    rules); ``verified`` says whether it equals ``quantity`` (None: nothing to check).
    """

    run: str
    candidate_id: int
    source: str
    rule: str | None
    rows: pd.DataFrame
    quantity: float | None
    reproduced: float | None
    verified: bool | None
    message: str
    tried: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        """True when there is a selection to show."""
        return self.source != SOURCE_NONE and not self.rows.empty

    @property
    def derived(self) -> bool:
        """True for a selection the viewer derived (not the engine's table)."""
        return self.source in (SOURCE_FRAGMENT_QUANT, SOURCE_CHROMATOGRAMS)

    @property
    def label(self) -> str:
        return _SOURCE_LABELS.get(self.source, self.source)

    @property
    def n_selected(self) -> int:
        return int(self.rows["selected"].sum()) if not self.rows.empty else 0

    def selected_names(self) -> list[str]:
        """The selected fragments, in rank order."""
        if self.rows.empty:
            return []
        sel = self.rows[self.rows["selected"]]
        return [str(n) for n in sel.sort_values("rank", kind="stable")["fragment_name"]]

    def by_name(self) -> dict[str, dict[str, Any]]:
        """One row per fragment name (the first, should a name repeat)."""
        out: dict[str, dict[str, Any]] = {}
        for rec in self.rows.to_dict("records"):
            out.setdefault(str(rec["fragment_name"]), rec)
        return out


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fragment_name": pd.Series([], dtype=object),
            "area": pd.Series([], dtype="float64"),
            "selected": pd.Series([], dtype=bool),
            "weight": pd.Series([], dtype="float64"),
            "rank": pd.Series([], dtype="int32"),
            "background": pd.Series([], dtype="float64"),
        }
    )


# --------------------------------------------------------------------------- the check


def reproduce(rows: pd.DataFrame, rule: str | None) -> float | None:
    """The quantity the selected fragments of ``rows`` give under ``rule``.

    The top-N rules add the selected areas in rank order, the order the engine adds them;
    ``cross_run`` adds ``weight x area`` in table order, as the engine does. None under the
    pattern rules (a fit, not a sum) and when nothing is selected.
    """
    if rows.empty or rule in PATTERN_RULES:
        return None
    sel = rows[rows["selected"]]
    if sel.empty:
        return None
    if rule != "cross_run":
        sel = sel.sort_values("rank", kind="stable")
    total = 0.0
    for w, a in zip(sel["weight"], sel["area"], strict=True):
        total += float(w) * float(a)
    return total


def _check(
    rows: pd.DataFrame, rule: str | None, q: QuantInputs | None
) -> tuple[float | None, bool | None, str]:
    """(reproduced, verified, message) of ``rows`` against the ``peptide_quant`` row."""
    quantity = q.quantity if q is not None else None
    n_sel = int(rows["selected"].sum()) if not rows.empty else 0
    if rule in PATTERN_RULES:
        if q is None or q.n_fragments_used is None:
            return None, None, "the pattern rule's quantity is a fit; nothing to check against"
        ok = n_sel == q.n_fragments_used
        return (
            None,
            ok,
            f"{n_sel} fitted fragments; peptide_quant n_fragments_used {q.n_fragments_used}"
            if ok
            else f"{n_sel} fitted fragments against n_fragments_used {q.n_fragments_used}",
        )
    rep = reproduce(rows, rule)
    if quantity is None:
        return rep, None, "peptide_quant has no quantity for this candidate"
    if rep is None:
        return None, False, "no fragment is selected, but peptide_quant has a quantity"
    exact = rep == quantity
    ok = exact or math.isclose(rep, quantity, rel_tol=_REL_TOL, abs_tol=0.0)
    if ok:
        how = "exactly" if exact else f"to {_REL_TOL:g} relative"
        return rep, True, f"the selected fragments reproduce peptide_quant.quantity {how}"
    return rep, False, f"the selected fragments give {rep:.6g}, peptide_quant {quantity:.6g}"


# --------------------------------------------------------------------------- the rules


def _rank_top_n(areas: np.ndarray, preds: np.ndarray | None, rule: str) -> list[int]:
    """The slots the engine's top-N rule ranks, in its order (``select_fragment_areas``).

    Positive finite areas only; ``observed_area`` by area, descending, ``predicted`` by the
    library intensity (finite only), then area. Python's sort is stable, as the engine's is,
    so ties keep chromatogram order.
    """
    ok = [k for k in range(areas.size) if math.isfinite(areas[k]) and areas[k] > 0.0]
    if rule == "observed_area":
        return sorted(ok, key=lambda k: -float(areas[k]))
    if preds is None:
        raise ViewerError("the predicted rule needs the fragments' predicted_intensity")
    ok = [k for k in ok if math.isfinite(float(preds[k]))]
    return sorted(ok, key=lambda k: (-float(preds[k]), -float(areas[k])))


def _top_n_rows(
    names: Sequence[str],
    areas: np.ndarray,
    preds: np.ndarray | None,
    rule: str,
    top_n: int,
) -> pd.DataFrame:
    order = _rank_top_n(areas, preds, rule)
    rank = np.zeros(len(names), dtype="int32")
    selected = np.zeros(len(names), dtype=bool)
    for j, k in enumerate(order):
        rank[k] = j + 1
        selected[k] = j < top_n
    return pd.DataFrame(
        {
            "fragment_name": list(names),
            "area": np.asarray(areas, dtype="float64"),
            "selected": selected,
            "weight": np.ones(len(names), dtype="float64"),
            "rank": rank,
            "background": np.full(len(names), np.nan),
        }
    )


@dataclass(frozen=True)
class _QuantConfig:
    """The ``quant`` settings the viewer-derived sources depend on (engine defaults)."""

    top_n_fragments: int = 3
    fragment_selection: str = "observed_area"
    bound_peak: bool = True
    fixed_scan_halfwidth: int = 3
    fixed_window_s: float = 0.0
    baseline_subtract: bool = False
    interference_envelope: bool = False
    baseline_source: str = "run"
    fixed_scan_halfwidth_scale: float = 0.0
    window_centre: str = "apex"
    recorded: bool = True

    @property
    def needs_cross_run_fit(self) -> bool:
        """The engine's ``QuantConfig::needs_cross_run_fit``: the cross-run step ran."""
        return (
            self.fragment_selection in ("cross_run", *PATTERN_RULES)
            or self.baseline_source == "experiment"
            or self.fixed_scan_halfwidth_scale > 0.0
        )

    def single_run_rule(self) -> str:
        """The rule of a quant without the cross-run step: every selection but
        ``observed_area`` falls back to ``predicted``."""
        return "observed_area" if self.fragment_selection == "observed_area" else "predicted"


def _quant_config(rs: ResultSet) -> _QuantConfig:
    cfg = rs.manifest.config_get("quant", default=None)
    if not isinstance(cfg, dict):
        return _QuantConfig(recorded=False)
    base = _QuantConfig()
    values: dict[str, Any] = {}
    for name in (
        "top_n_fragments",
        "fragment_selection",
        "bound_peak",
        "fixed_scan_halfwidth",
        "fixed_window_s",
        "baseline_subtract",
        "interference_envelope",
        "baseline_source",
        "fixed_scan_halfwidth_scale",
        "window_centre",
    ):
        if name in cfg and cfg[name] is not None:
            kind = type(getattr(base, name))
            try:
                values[name] = kind(cfg[name])
            except (TypeError, ValueError):
                continue
    return _QuantConfig(**values)


# ------------------------------------------------------------------ the engine's arithmetic


def _trapezoid(rt: np.ndarray, inten: np.ndarray) -> float:
    """The engine's ``trapezoid``: float32 differences and sums, float64 products, in order."""
    if rt.size < 2:
        return float(inten[0]) if inten.size else 0.0
    dt = (rt[1:] - rt[:-1]).astype(np.float64)
    s = (inten[:-1] + inten[1:]).astype(np.float64)
    area = 0.0
    for a, b in zip(dt.tolist(), s.tolist(), strict=True):
        area += a * b * 0.5
    return area


def _nearest_index(rt: np.ndarray, target: float) -> int:
    """The engine's ``nearest_index``: the first sample at the smallest distance."""
    if not math.isfinite(target) or rt.size == 0:
        return 0
    d = np.abs(rt.astype(np.float64) - target)
    return int(np.argmin(d))


def _fixed_window(rt: np.ndarray, apex: float, half: int, half_s: float) -> tuple[int, int] | None:
    """The engine's ``fixed_window_indices``: ``[lo, hi)`` around the sample nearest ``apex``."""
    n = rt.size
    if n == 0:
        return None
    k = _nearest_index(rt, apex)
    d = abs(float(rt[k]) - apex)
    best = math.inf if math.isnan(d) else d
    if half_s > 0.0:
        if best > half_s:
            return None
        lo = k
        while lo > 0 and (apex - float(rt[lo - 1])) <= half_s:
            lo -= 1
        hi = k + 1
        while hi < n and (float(rt[hi]) - apex) <= half_s:
            hi += 1
    else:
        lo, hi = max(k - half, 0), min(k + half + 1, n)
    if hi - lo < 2 and n >= 2:
        if hi < n:
            hi += 1
        else:
            lo = max(lo - 1, 0)
    return lo, hi


def _window_area(rt: np.ndarray, inten: np.ndarray, lo: float, hi: float) -> float:
    """The engine's ``trapezoid_window`` without the envelope: samples with ``lo <= rt <= hi``."""
    r = rt.astype(np.float64)
    keep = (r >= lo) & (r <= hi)
    if not keep.any():
        return 0.0
    return _trapezoid(rt[keep], inten[keep])


def _chromatogram_areas(
    chrom: CandidateChromatogram, cfg: _QuantConfig, q: QuantInputs
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(names, areas, predicted intensities) of the fragment rows, integrated as quant does."""
    frags = chrom.fragments()
    apex = q.integration_apex_rt
    fixed = (cfg.fixed_scan_halfwidth > 0 or cfg.fixed_window_s > 0.0) and apex is not None
    lo = q.integration_lo_rt if q.integration_lo_rt is not None else -math.inf
    hi = q.integration_hi_rt if q.integration_hi_rt is not None else math.inf
    areas = np.zeros(len(frags), dtype="float64")
    for i, t in enumerate(frags):
        rt = np.asarray(t.rt, dtype=np.float32)
        it = np.asarray(t.intensity, dtype=np.float32)
        if rt.size == 0:
            # A fragment never observed integrates to 0 under every window.
            continue
        if fixed:
            assert apex is not None
            w = _fixed_window(rt, apex, cfg.fixed_scan_halfwidth, cfg.fixed_window_s)
            areas[i] = 0.0 if w is None else _trapezoid(rt[w[0] : w[1]], it[w[0] : w[1]])
        elif cfg.bound_peak:
            areas[i] = _window_area(rt, it, lo, hi)
        else:
            areas[i] = _trapezoid(rt, it)
    preds = np.asarray([float(np.float32(t.predicted_intensity)) for t in frags], dtype="float64")
    return [t.frag_name for t in frags], areas, preds


# --------------------------------------------------------------------------- sources


def _engine_rows(rs: ResultSet, run: Run, cid: int) -> tuple[pd.DataFrame, str | None] | None:
    """The engine's rows of ``cid``; None when the run has no ``quant_fragments`` table."""
    art = run.artifact("quant_fragments")
    if art is None or not art.usable:
        return None
    art.parquet()  # checks the column contract once
    df = rs.duck.df(
        "SELECT fragment_name, area, selected, weight, rank, background, rule "
        "FROM read_parquet(?) WHERE candidate_id = ?",
        [sql_path(art.require()), int(cid)],
    )
    rule = str(df["rule"].iloc[0]) if not df.empty else None
    rows = df.drop(columns=["rule"]).reset_index(drop=True)
    if rows.empty:
        rows = _empty_rows()
    else:
        rows = rows.astype(
            {
                "area": "float64",
                "selected": bool,
                "weight": "float64",
                "rank": "int32",
                "background": "float64",
            }
        )
    return rows, rule


def _fragment_quant_areas(rs: ResultSet, run: Run, cid: int) -> dict[str, float] | None:
    art = run.artifact("fragment_quant")
    if art is None or not art.usable:
        return None
    art.parquet()
    df = rs.duck.df(
        "SELECT fragment_name, quantity FROM read_parquet(?) WHERE candidate_id = ?",
        [sql_path(art.require()), int(cid)],
    )
    out: dict[str, float] = {}
    for name, area in zip(df["fragment_name"], df["quantity"], strict=True):
        out.setdefault(str(name), float(area))
    return out


def _result(
    run: Run,
    cid: int,
    source: str,
    rule: str | None,
    rows: pd.DataFrame,
    q: QuantInputs | None,
    tried: list[str],
) -> QuantFragments:
    rep, ok, msg = _check(rows, rule, q)
    return QuantFragments(
        run=run.name,
        candidate_id=cid,
        source=source,
        rule=rule,
        rows=rows,
        quantity=q.quantity if q is not None else None,
        reproduced=rep,
        verified=ok,
        message=msg,
        tried=tuple(tried),
    )


def _none(run: Run, cid: int, q: QuantInputs | None, why: str, tried: list[str]) -> QuantFragments:
    return QuantFragments(
        run=run.name,
        candidate_id=cid,
        source=SOURCE_NONE,
        rule=None,
        rows=_empty_rows(),
        quantity=q.quantity if q is not None else None,
        reproduced=None,
        verified=None,
        message=why,
        tried=tuple(tried),
    )


def quant_fragments(
    rs: ResultSet,
    run: Run | str | int | None,
    candidate_id: int,
    *,
    chromatogram: CandidateChromatogram | None = None,
    quant: QuantState | Mapping[str, Any] | None = None,
    sources: Sequence[str] = (SOURCE_ENGINE, SOURCE_FRAGMENT_QUANT, SOURCE_CHROMATOGRAMS),
) -> QuantFragments:
    """The fragments behind the quantity of ``candidate_id`` in ``run`` (module docstring).

    ``chromatogram`` and ``quant`` (a :class:`.quant.QuantState`, or a row with its fields)
    are read when not given. ``sources`` restricts the sources tried, in order (a test or
    a reader that wants one source only).
    """
    from .quant import resolve_run

    run = resolve_run(rs, run)
    cid = int(candidate_id)
    tried: list[str] = []
    if quant is None:
        try:
            quant = quant_state(rs, run, cid)
        except ViewerError:
            quant = None
    q = QuantInputs.of(quant)

    if SOURCE_ENGINE in sources:
        found = _engine_rows(rs, run, cid)
        if found is not None:
            rows, rule = found
            if rows.empty:
                why = (
                    "the engine wrote no fragment rows for this candidate: quant did not "
                    "reach it (not selected by the quant gate, or no chromatogram rows)"
                )
                return _none(run, cid, q, why, [SOURCE_ENGINE])
            return _result(run, cid, SOURCE_ENGINE, rule, rows, q, [SOURCE_ENGINE])
        tried.append(f"{SOURCE_ENGINE}: no quant_fragments.parquet in this run")

    if q is None or q.state == "not_selected" or q.quantity is None:
        state = (
            "no peptide_quant row"
            if q is None or q.state == "not_selected"
            else ("no quantity in peptide_quant")
        )
        return _none(run, cid, q, f"{state}: nothing to derive the fragments against", tried)

    cfg = _quant_config(rs)
    if rs.is_experiment and cfg.needs_cross_run_fit:
        tried.append(
            "viewer: the cross-run step of the experiment fitted weights or backgrounds that "
            "no table records"
        )
        return _none(run, cid, q, NOT_REPRODUCIBLE, tried)
    rule = cfg.single_run_rule()
    top_n = cfg.top_n_fragments

    if chromatogram is None and (
        SOURCE_FRAGMENT_QUANT in sources or SOURCE_CHROMATOGRAMS in sources
    ):
        try:
            chromatogram = ChromatogramSource.for_run(rs, run).read(cid)
        except ViewerError as exc:
            tried.append(f"chromatograms: not readable ({exc})")
            chromatogram = None

    if SOURCE_FRAGMENT_QUANT in sources:
        if rs.is_experiment:
            tried.append(
                f"{SOURCE_FRAGMENT_QUANT}: in an experiment it is the cross-run step's "
                "first-pass export, not the reported quantity"
            )
        else:
            areas_by_name = _fragment_quant_areas(rs, run, cid)
            if areas_by_name is None:
                tried.append(f"{SOURCE_FRAGMENT_QUANT}: no fragment_quant.parquet in this run")
            else:
                if chromatogram is not None:
                    frags = chromatogram.fragments()
                    names = [t.frag_name for t in frags]
                    preds = np.asarray(
                        [float(np.float32(t.predicted_intensity)) for t in frags], dtype="float64"
                    )
                else:
                    names = list(areas_by_name)
                    preds = None
                areas = np.asarray([areas_by_name.get(n, 0.0) for n in names], dtype="float64")
                try:
                    rows = _top_n_rows(names, areas, preds, rule, top_n)
                except ViewerError as exc:
                    tried.append(f"{SOURCE_FRAGMENT_QUANT}: {exc}")
                else:
                    res = _result(run, cid, SOURCE_FRAGMENT_QUANT, rule, rows, q, tried)
                    if res.verified:
                        return res
                    tried.append(f"{SOURCE_FRAGMENT_QUANT}: {res.message}")

    if SOURCE_CHROMATOGRAMS in sources:
        reasons = []
        if cfg.baseline_subtract:
            reasons.append("quant subtracted a baseline")
        if cfg.interference_envelope:
            reasons.append("quant applied the interference envelope")
        if cfg.window_centre != "apex":
            reasons.append("the window is not centred on the apex")
        if chromatogram is None:
            reasons.append("no chromatogram rows")
        if reasons:
            tried.append(f"{SOURCE_CHROMATOGRAMS}: " + "; ".join(reasons))
        else:
            assert chromatogram is not None
            names, areas, preds = _chromatogram_areas(chromatogram, cfg, q)
            rows = _top_n_rows(names, areas, preds, rule, top_n)
            res = _result(run, cid, SOURCE_CHROMATOGRAMS, rule, rows, q, tried)
            if res.verified:
                return res
            tried.append(f"{SOURCE_CHROMATOGRAMS}: {res.message}")

    return _none(run, cid, q, NOT_REPRODUCIBLE, tried)


# --------------------------------------------------------------------------- across runs


@dataclass(frozen=True)
class AcrossFragments:
    """Fragment areas of one precursor in every run: ``area`` and ``selected`` are
    ``runs x fragments`` (NaN and False where a run has no account); ``accounts`` holds each
    run's :class:`QuantFragments` (None for a run without a scored row)."""

    runs: tuple[str, ...]
    fragments: tuple[str, ...]
    area: np.ndarray
    selected: np.ndarray
    accounts: tuple[QuantFragments | None, ...]

    @property
    def any_available(self) -> bool:
        return any(a is not None and a.available for a in self.accounts)

    @property
    def any_derived(self) -> bool:
        return any(a is not None and a.available and a.derived for a in self.accounts)


def fragment_areas_across(
    rs: ResultSet,
    runs: Sequence[Run],
    rows: Sequence[Mapping[str, Any]],
    chromatograms: Sequence[CandidateChromatogram | None],
) -> AcrossFragments:
    """One precursor's fragment areas in each run, from :func:`quant_fragments`.

    ``rows`` are the per-run rows of :class:`.across.PrecursorAcross` (``candidate_id``,
    ``scored``, ``state``, ``quantity``, ``n_fragments_used`` and the integration bounds);
    ``chromatograms`` the decoded chromatograms of :func:`.across.run_xics`, in run order.
    Fragments are in the order of the first run's chromatogram, then any others.
    """
    accounts: list[QuantFragments | None] = []
    for run, rec, chrom in zip(runs, rows, chromatograms, strict=True):
        cid = rec.get("candidate_id")
        if not rec.get("scored") or cid is None or (isinstance(cid, float) and math.isnan(cid)):
            accounts.append(None)
            continue
        try:
            accounts.append(quant_fragments(rs, run, int(cid), chromatogram=chrom, quant=rec))
        except ViewerError:
            accounts.append(None)
    order: list[str] = []
    for chrom in chromatograms:
        if chrom is not None:
            for t in chrom.fragments():
                if t.frag_name not in order:
                    order.append(t.frag_name)
    for acc in accounts:
        if acc is not None:
            for name in acc.rows["fragment_name"]:
                if str(name) not in order:
                    order.append(str(name))
    col = {n: j for j, n in enumerate(order)}
    area = np.full((len(accounts), len(order)), np.nan)
    sel = np.zeros((len(accounts), len(order)), dtype=bool)
    for i, acc in enumerate(accounts):
        if acc is None or not acc.available:
            continue
        for rec in acc.rows.to_dict("records"):
            j = col[str(rec["fragment_name"])]
            area[i, j] = float(rec["area"])
            sel[i, j] = bool(rec["selected"])
    return AcrossFragments(
        runs=tuple(r.name or "run" for r in runs),
        fragments=tuple(order),
        area=area,
        selected=sel,
        accounts=tuple(accounts),
    )
