"""Line icons drawn for the viewer (24x24, stroke 2), rendered as CSS masks.

A mask takes the current text colour, so the icons follow the light and dark themes.
Nothing is fetched from the internet.
"""

from __future__ import annotations

from urllib.parse import quote

from dash import html

_PATHS: dict[str, str] = {
    "overview": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" '
    'height="4" rx="1.5"/><rect x="13" y="10" width="7" height="10" rx="1.5"/><rect x="4" '
    'y="13" width="7" height="7" rx="1.5"/>',
    "table": '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M3.5 '
    '14.5h17M9.5 9.5v10"/>',
    "peak": '<path d="M3 19h18"/><path d="M4 18c3 0 4-11 6-11s2.5 6 4 6 2-3 3.5-3 2 8 3.5 8"/>',
    "spectrum": '<path d="M3 20h18"/><path d="M6 20v-6M9 20V8M12 20v-9M15 20V5M18 20v-4"/>',
    "calibration": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path '
    'd="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "compare": '<path d="M8 4v16M16 4v16"/><path d="M4 8l4-4 4 4M12 16l4 4 4-4"/>',
    "quant": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 '
    '17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "alert": '<path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.3-4.3"/>',
    "external": '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 '
    '1-1-1V7a1 1 0 0 1 1-1h5"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "x": '<path d="M6 6l12 12M18 6 6 18"/>',
    "left": '<path d="m15 5-7 7 7 7"/>',
    "right": '<path d="m9 5 7 7-7 7"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle '
    'cx="12" cy="12" r="1" />',
    "bolt": '<path d="M13 3 5 13.5h6L10 21l8-10.5h-6z"/>',
    "flask": '<path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 '
    '1.3-2L14 9V3"/><path d="M7.5 15h9"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "scale": '<path d="M12 4v16M8 20h8M5 7h14"/><path d="m5 7-3 6a3 3 0 0 0 6 0zM19 7l-3 6a3 '
    '3 0 0 0 6 0z"/>',
    "keyboard": '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 10h.01M10 '
    '10h.01M14 10h.01M18 10h.01M7 14h10"/>',
}


def _data_uri(name: str) -> str:
    body = _PATHS[name]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


_URIS = {name: _data_uri(name) for name in _PATHS}


def icon(name: str, size: int = 18, *, colour: str = "currentColor") -> html.Span:
    """An icon that takes the surrounding text colour (or ``colour``)."""
    uri = _URIS[name]
    return html.Span(
        style={
            "display": "inline-block",
            "width": f"{size}px",
            "height": f"{size}px",
            "minWidth": f"{size}px",
            "backgroundColor": colour,
            "maskImage": f'url("{uri}")',
            "WebkitMaskImage": f'url("{uri}")',
            "maskRepeat": "no-repeat",
            "WebkitMaskRepeat": "no-repeat",
            "maskSize": "contain",
            "WebkitMaskSize": "contain",
            "maskPosition": "center",
            "verticalAlign": "middle",
        },
        className="mv-icon",
    )


NAMES = tuple(_PATHS)
