"""One precursor across the runs of an experiment (P2 view 9). Skeleton."""

from __future__ import annotations

from typing import Any

import dash_mantine_components as dmc

from .state import PageContext
from .widgets import section


def layout(ctx: PageContext) -> Any:
    return section("Across runs", dmc.Text("This view is being built.", c="dimmed", size="sm"))


def register(app, get_rs, base: str) -> None:
    """Callbacks of the page (app.callback and app.clientside_callback only)."""
    return None
