"""
Shared style helpers for paper/figures/.

Follows the paper-plot-skills style guide (bar_paired_delta, bar_grouped_hatch,
line_confidence_band, line_training_curve) but substitutes STIX serif for the
skill's usetex=True Computer Modern setup because a system LaTeX binary is not
available in the execution environment.

All downstream scripts call `apply_paper_style()` at the top to make every
figure consistent.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colour palette — reuses the paper-plot-skills reference tones.
# ---------------------------------------------------------------------------

# Paired comparison (bar_paired_delta)
C_BASELINE = "#A8C8E8"   # light steel blue
C_METHOD = "#1B3D6E"     # deep navy
C_DELTA = "#CC2200"      # red delta annotation
C_REF_LINE = "#333333"   # baseline reference dashed line

# Grouped-hatch ablation (bar_grouped_hatch, ablation variant)
C_ABL = ["#F5C5A3", "#E8845A", "#C0392B"]   # warm light→mid→deep
# Grouped-hatch comparison (bar_grouped_hatch, comparison variant)
C_CMP = ["#C8C8C8", "#8C8C8C", "#C0392B"]   # cool light→mid→deep red
BEST_VALUE_COLOR = "#8B0000"                # dark red for best-method value label
HATCH_BEST = "//"                            # white-on-red diagonal for best

# Line confidence-band (line_confidence_band)
C_LINE_MAIN = "#3A8B3A"   # green main method
C_LINE_ALT = "#3B6BB5"    # blue alternative
C_LINE_BASE = "#999999"   # grey reference
C_REF_HLINE = "#3D78C2"   # distinct blue for reference horizontal lines

# Training-curve vertical markers (line_training_curve)
C_DYN = "#5B0DAD"
C_NODYN = "#5BBCCA"


def apply_paper_style() -> None:
    """Apply the shared rcParams used by every paper/figures/ script."""
    mpl.rcParams.update(
        {
            # Serif via STIX (math-compatible, usetex-free)
            "font.family": "serif",
            "font.serif": [
                "STIXGeneral",
                "DejaVu Serif",
                "Times New Roman",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            # Hatch in bar_grouped_hatch
            "hatch.color": "white",
            "hatch.linewidth": 1.4,
            # Default tick ticks-outward, subtle
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Figure appearance
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def style_open_axes(ax) -> None:
    """Open-axis style used by bar_grouped_hatch / line_confidence_band.

    Only left and bottom spines remain visible; top/right are hidden.
    Spine linewidth 0.9, colour #333333.
    """
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.9)
            spine.set_color("#333333")


def style_framed_axes(ax) -> None:
    """Four-sided frame style used by bar_paired_delta / line_training_curve."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#222222")


def y_grid_only(ax) -> None:
    """Y-axis dashed grid used in bar_grouped_hatch (zorder behind bars)."""
    ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.7, linestyle="--", zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
