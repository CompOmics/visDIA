"""The visual design system: Mantine theme, colours and Plotly templates.

One place defines the palette, so the Mantine components and the Plotly figures match
in light and dark mode. The fonts are the system stack: nothing is fetched from the
internet, so the viewer looks the same offline and on a server behind SSH.
"""

from __future__ import annotations

import copy

import plotly.graph_objects as go
import plotly.io as pio

FONT = (
    "Inter, 'Segoe UI Variable', 'Segoe UI', system-ui, -apple-system, Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)
MONO = "'JetBrains Mono', 'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace"

# Semantic colours (also used in the CSS through Mantine variables).
TARGET = "#4263eb"  # indigo 7
DECOY = "#e8590c"  # orange 8
SPIKE = "#ae3ec9"  # grape 6
ACCEPT = "#2f9e44"  # green 8
WARN = "#f08c00"  # yellow 9
APEX = "#2f9e44"
ELUTION = "rgba(47, 158, 68, 0.12)"
INTEGRATION = "#7048e8"  # violet 7
PREDICTION = "#868e96"  # gray 6
WINDOW = "#adb5bd"  # gray 5
MS1 = ("#1098ad", "#3bc9db", "#99e9f2")  # cyan 7, 4, 2

# One colour per identification unit, used by the cards, the curves and the tables.
UNIT_COLOURS = {
    "psm": "#4c6ef5",  # indigo 6
    "precursor": "#7950f2",  # violet 6
    "peptide": "#12b886",  # teal 6
    "protein_group": "#e64980",  # pink 6
}
UNIT_MANTINE = {"psm": "indigo", "precursor": "violet", "peptide": "teal", "protein_group": "pink"}
# Distinct colours for runs and other categories.
SERIES = [
    "#4c6ef5",
    "#12b886",
    "#fab005",
    "#e64980",
    "#7950f2",
    "#15aabf",
    "#fd7e14",
    "#82c91e",
    "#868e96",
]

# Modifications, as in PeptideShaker: a modified residue is drawn in its modification's
# colour with a short tag; the full name is in the tooltip. A fixed modification such
# as carbamidomethylation is drawn quietly. The same table is sent to the browser
# (window.MV.mods), so tables and figures colour a modification alike.
MOD_STYLES: dict[str, tuple[str, str]] = {
    "Oxidation": ("ox", "#0ca678"),
    "Carbamidomethyl": ("cam", "#868e96"),
    "Phospho": ("ph", "#e03131"),
    "Acetyl": ("ac", "#2f9e44"),
    "Deamidated": ("de", "#c2255c"),
    "Methyl": ("me", "#ae3ec9"),
    "Dimethyl": ("me2", "#ae3ec9"),
    "Trimethyl": ("me3", "#ae3ec9"),
    "GlyGly": ("gg", "#0c8599"),
    "Gln->pyro-Glu": ("pyro", "#5c940d"),
    "Glu->pyro-Glu": ("pyro", "#5c940d"),
    "Amidated": ("am", "#f08c00"),
    "Carbamyl": ("cbm", "#e8590c"),
    "TMT6plex": ("tmt", "#495057"),
    "TMTpro": ("tmt", "#495057"),
}
# UNIMOD accessions of the names above, for peptidoforms written as [UNIMOD:35].
UNIMOD_NAMES: dict[str, str] = {
    "1": "Acetyl",
    "4": "Carbamidomethyl",
    "5": "Carbamyl",
    "7": "Deamidated",
    "21": "Phospho",
    "27": "Glu->pyro-Glu",
    "28": "Gln->pyro-Glu",
    "34": "Methyl",
    "35": "Oxidation",
    "36": "Dimethyl",
    "37": "Trimethyl",
    "121": "GlyGly",
    "737": "TMT6plex",
    "2016": "TMTpro",
}
MOD_FALLBACK = ["#7048e8", "#d6336c", "#5c940d", "#9c36b5", "#1098ad", "#e67700"]


def mod_name(text: str) -> str:
    """The modification name of a ProForma tag (``UNIMOD:35`` becomes ``Oxidation``)."""
    t = text.strip()
    head, _, tail = t.partition(":")
    if head.lower() == "unimod" and tail in UNIMOD_NAMES:
        return UNIMOD_NAMES[tail]
    return t


def mod_style(text: str) -> tuple[str, str]:
    """(short tag, colour) of a modification; an unknown one gets a stable colour.

    The fallback colour depends on the characters only (not on Python's hash seed), and
    the browser computes the same (clientside.js, mvMods.style).
    """
    name = mod_name(text)
    if name in MOD_STYLES:
        return MOD_STYLES[name]
    short = name if len(name) <= 6 else name[:3].lower()
    return short, MOD_FALLBACK[sum(ord(c) for c in name) % len(MOD_FALLBACK)]


def browser_globals() -> dict:
    """Constants the browser code needs (written into the page as window.MV)."""
    return {
        "mods": MOD_STYLES,
        "unimod": UNIMOD_NAMES,
        "modFallback": MOD_FALLBACK,
        "colours": {
            "target": TARGET,
            "decoy": DECOY,
            "spike": SPIKE,
            "accept": ACCEPT,
            "warn": WARN,
            "units": UNIT_COLOURS,
        },
    }


# Fragment ions: b in blue hues, y in red hues; each fragment gets its own shade.
B_SHADES = [
    "#1864ab",
    "#1c7ed6",
    "#339af0",
    "#4dabf7",
    "#74c0fc",
    "#3b5bdb",
    "#4c6ef5",
    "#5c7cfa",
    "#748ffc",
    "#91a7ff",
]
Y_SHADES = [
    "#c92a2a",
    "#e03131",
    "#f03e3e",
    "#fa5252",
    "#ff6b6b",
    "#d6336c",
    "#e64980",
    "#f06595",
    "#c2255c",
    "#ff8787",
]
OTHER_SHADES = ["#5f3dc4", "#7048e8", "#845ef7", "#9775fa", "#e8590c", "#f76707"]

MANTINE_THEME = {
    "primaryColor": "indigo",
    "fontFamily": FONT,
    "fontFamilyMonospace": MONO,
    "defaultRadius": "md",
    "headings": {"fontFamily": FONT, "fontWeight": "650"},
    "components": {
        "Card": {"defaultProps": {"shadow": "xs", "radius": "lg", "withBorder": True}},
        "Paper": {"defaultProps": {"radius": "lg"}},
        "Badge": {"defaultProps": {"radius": "sm", "variant": "light"}},
        "Tooltip": {"defaultProps": {"withArrow": True, "multiline": True, "w": 320}},
    },
}


def fragment_colour(name: str, ion: str | None, index: int) -> str:
    shades = B_SHADES if ion == "b" else Y_SHADES if ion == "y" else OTHER_SHADES
    return shades[index % len(shades)]


def _template(dark: bool) -> go.layout.Template:
    text = "#c1c2c5" if dark else "#343a40"
    grid = "rgba(255,255,255,0.07)" if dark else "rgba(0,0,0,0.06)"
    zero = "rgba(255,255,255,0.18)" if dark else "rgba(0,0,0,0.18)"
    paper = "rgba(0,0,0,0)"
    t = copy.deepcopy(pio.templates["plotly_dark" if dark else "plotly_white"])
    t.layout.update(
        font=dict(family=FONT, size=12, color=text),
        paper_bgcolor=paper,
        plot_bgcolor=paper,
        colorway=[TARGET, DECOY, ACCEPT, SPIKE, "#1098ad", "#f59f00", "#d6336c", "#495057"],
        hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1),
        margin=dict(l=56, r=16, t=34, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(font=dict(size=14, color=text), x=0, xanchor="left", pad=dict(l=4)),
        hovermode="closest",
        dragmode="zoom",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=text, activecolor=TARGET),
    )
    for axis in ("xaxis", "yaxis"):
        t.layout[axis].update(
            gridcolor=grid,
            zerolinecolor=zero,
            linecolor=grid,
            title=dict(font=dict(size=12)),
            tickfont=dict(size=11),
        )
    return t


pio.templates["mumdia_light"] = _template(dark=False)
pio.templates["mumdia_dark"] = _template(dark=True)


def template(scheme: str | None) -> str:
    return "mumdia_dark" if scheme == "dark" else "mumdia_light"


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "svg", "scale": 1},
}
