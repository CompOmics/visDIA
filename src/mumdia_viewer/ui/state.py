"""Shared UI state: the q threshold stops and the page addresses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode

if TYPE_CHECKING:
    from mumdia_viewer.data import ResultSet

DEFAULT_THRESHOLD = 0.01

# The threshold control offers these stops. Counts are exact at every stop: they are
# counted with the engine's columns (data.counts.counts_at), never interpolated.
THRESHOLD_STOPS: tuple[float, ...] = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)

PAGES = ("overview", "identifications", "precursor")


def stop_label(t: float) -> str:
    return f"{t:.0e}".replace("e-0", "e-") if t < 0.001 else f"{t:g}"


def threshold_options() -> list[dict[str, str]]:
    return [{"value": repr(t), "label": f"q ≤ {stop_label(t)}"} for t in THRESHOLD_STOPS]


def parse_threshold(value: object, default: float = DEFAULT_THRESHOLD) -> float:
    """A threshold from the control or the address; anything else gives ``default``."""
    try:
        t = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return t if 0 < t < 1 else default


def page_of(pathname: str | None, base: str) -> str:
    path = (pathname or "/")[len(base.rstrip("/")) :].strip("/")
    return path if path in PAGES else "overview"


def query_of(search: str | None) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}


def href(base: str, page: str, query: Mapping[str, object] | None = None) -> str:
    path = "" if page == "overview" else page
    q = {k: v for k, v in (query or {}).items() if v not in (None, "")}
    return f"{base}{path}" + (f"?{urlencode(q)}" if q else "")


@dataclass(frozen=True)
class PageContext:
    """What a page needs to build its layout."""

    rs: ResultSet
    base: str
    threshold: float = DEFAULT_THRESHOLD
    scheme: str = "light"
    query: dict[str, str] = field(default_factory=dict)
    compare: ResultSet | None = None
