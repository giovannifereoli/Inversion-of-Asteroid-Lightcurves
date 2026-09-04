"""Figure styling: a validated, colourblind-safe palette and a serif math font.

Every figure in the package draws its colours from here rather than from
Matplotlib's defaults, for two reasons.

**Colour is an encoding, so it has to survive colour-vision deficiency.**  The
categorical slots below were checked with a CVD simulator rather than by eye:
the worst adjacent pair separates by ``Delta E = 9.1`` under protanopia and
``22.9`` for normal vision (OKLab x100, on the light surface), clearing the
usual thresholds of 8 and 15.  Two of the slots fall below 3:1 contrast against
the surface, so any figure using them carries a legend or direct labels -
identity is never carried by colour alone.

**Sequential data gets one hue, light to dark.**  ``chi^2`` maps and curvature
maps use a single blue ramp, never a rainbow, so that "larger" reads as "darker"
without the viewer consulting a key.

Typography defaults to Matplotlib's STIX fonts, which ship with it and give the
serif, LaTeX-like look without needing a TeX installation.  Pass ``latex=True``
to :func:`use_style` to typeset with real LaTeX where one is available.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager

__all__ = [
    "PALETTE",
    "use_style",
    "style_context",
    "sequential_cmap",
    "diverging_cmap",
    "series_colors",
]

#: Validated colour tokens.  Light and dark are separately stepped for their own
#: surface, not an automatic inversion of one another.
PALETTE: dict[str, dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#83827c",
        "grid": "#e3e2dd",
        "axis": "#a9a89f",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                       "#256abf", "#184f95", "#0d366b"],
        "diverging": ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#8f1f1e"],
        "body": "#c9b8a0",
    },
    "dark": {
        "surface": "#1a1a19",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#8f8e86",
        "grid": "#33332f",
        "axis": "#5c5b55",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "sequential": ["#0d366b", "#184f95", "#256abf", "#3987e5",
                       "#6da7ec", "#9ec5f4", "#cde2fb"],
        "diverging": ["#cde2fb", "#3987e5", "#383835", "#e34948", "#f5b0af"],
        "body": "#b3a58e",
    },
}


def series_colors(mode: str = "light") -> list[str]:
    """The categorical slots, in their fixed order.

    Assign them in order and never cycle: a ninth series should become "other",
    a facet, or a small multiple rather than a repeated hue.
    """
    return list(PALETTE[mode]["series"])  # type: ignore[arg-type]


def sequential_cmap(mode: str = "light", reverse: bool = False):
    """One-hue light-to-dark ramp for continuous magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    steps = list(PALETTE[mode]["sequential"])  # type: ignore[arg-type]
    if reverse:
        steps = steps[::-1]
    return LinearSegmentedColormap.from_list("lcinv_seq", steps)


def diverging_cmap(mode: str = "light"):
    """Two-hue ramp with a neutral midpoint, for signed quantities.

    Use it only when zero is meaningful - residuals, differences.  A neutral
    grey sits at the midpoint so "no deviation" reads as "nothing".
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "lcinv_div", list(PALETTE[mode]["diverging"])  # type: ignore[arg-type]
    )


def _rc(mode: str, latex: bool) -> dict:
    p = PALETTE[mode]
    rc = {
        # --- typography: serif body with matching maths -------------------
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "figure.titlesize": 12.5,
        # --- surfaces and ink ---------------------------------------------
        "figure.facecolor": p["surface"],
        "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"],
        "text.color": p["text_primary"],
        "axes.labelcolor": p["text_secondary"],
        "axes.titlecolor": p["text_primary"],
        "xtick.color": p["text_muted"],
        "ytick.color": p["text_muted"],
        "xtick.labelcolor": p["text_secondary"],
        "ytick.labelcolor": p["text_secondary"],
        # --- recessive chrome ---------------------------------------------
        "axes.edgecolor": p["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": p["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # --- marks ---------------------------------------------------------
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "axes.prop_cycle": _cycler(p["series"]),  # type: ignore[arg-type]
        # --- legend --------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.4,
        "legend.labelcolor": p["text_secondary"],
        # --- output --------------------------------------------------------
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
        "axes.axisbelow": True,
    }
    if latex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    return rc


def _cycler(colors):
    from cycler import cycler

    return cycler(color=list(colors))


def use_style(mode: str = "light", latex: bool = False) -> dict:
    """Apply the package style to the global Matplotlib state.

    Parameters
    ----------
    mode:
        ``"light"`` or ``"dark"``; each has its own stepped palette rather than
        one being an inversion of the other.
    latex:
        Typeset with a real LaTeX installation.  Falls back to the bundled STIX
        fonts (which already look like LaTeX) with a warning if no ``latex``
        binary is on the path, so a notebook stays runnable everywhere.

    Returns
    -------
    dict
        The colour tokens in force, so callers can pull named colours out.
    """
    import matplotlib.pyplot as plt

    if mode not in PALETTE:
        raise ValueError("mode must be 'light' or 'dark'")
    if latex and shutil.which("latex") is None:
        import warnings

        warnings.warn(
            "no LaTeX installation found; falling back to STIX fonts",
            RuntimeWarning,
            stacklevel=2,
        )
        latex = False
    plt.rcParams.update(_rc(mode, latex))
    return dict(PALETTE[mode])


@contextmanager
def style_context(mode: str = "light", latex: bool = False):
    """Apply the style for the duration of a ``with`` block."""
    import matplotlib.pyplot as plt

    if latex and shutil.which("latex") is None:
        latex = False
    with plt.rc_context(_rc(mode, latex)):
        yield dict(PALETTE[mode])
