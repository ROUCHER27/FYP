"""
Figure 3.1 — Portfolio construction pipeline (schematic).

No direct paper-plot-skills template for flow diagrams. Uses the shared
serif style and palette so the figure is visually consistent with Figs 5.1
5.5. Boxes represent the five steps of `compute_portfolio_returns`, arrows
show data flow, and the tensor shape at each stage is annotated underneath.

Inputs : none (schematic).
Output : paper/figures/fig3_1_portfolio_flow.png (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _style import (
    C_LINE_ALT,
    C_LINE_BASE,
    C_LINE_MAIN,
    apply_paper_style,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig3_1_portfolio_flow.png"

BLOCKS = [
    dict(
        text=r"Prediction vector" "\n" r"$\hat y_t = \{\hat y_{i,t}\}_{i=1}^{N}$",
        shape="Nx1  (all stocks)",
        color="#D8E3F1",
        edge="#1B3D6E",
    ),
    dict(
        text=r"Top/bottom 10% selection" "\n" r"$k = \max(1, \lfloor 0.1 N \rfloor)$",
        shape=r"long $\mathcal{L}$, short $\mathcal{S}$  (k each)",
        color="#F5C5A3",
        edge="#C0392B",
    ),
    dict(
        text=r"Within-bucket $z$-score" "\n" r"clip to $[-3,\,3]$",
        shape=r"k signed weights",
        color="#F7D9A8",
        edge="#B5651D",
    ),
    dict(
        text="Sign-consistent positive\nweights + normalise",
        shape=r"$\sum w = 1$",
        color="#E4E6AF",
        edge="#7A7A00",
    ),
    dict(
        text="Capped-simplex projection\n(max 5% per stock, ≤10 iters)",
        shape=r"$w_i \leq 0.05$, $\sum w = 1$",
        color="#C8D9B8",
        edge="#2A6A3A",
    ),
    dict(
        text=r"Long return $r_t^{\mathcal{L}}$" "\n" r"Short return $r_t^{\mathcal{S}}$" "\n"
             r"Long-short $r_t = r_t^{\mathcal{L}} - r_t^{\mathcal{S}}$",
        shape="scalar per month",
        color="#1B3D6E",
        edge="#0B1F40",
        text_color="white",
    ),
]


def draw_block(ax, x, y, w, h, body, shape_label, color, edge, text_color="#222"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2 + 0.018,
        body,
        ha="center",
        va="center",
        fontsize=10.2,
        color=text_color,
        zorder=4,
    )
    # Shape annotation below
    ax.text(
        x + w / 2,
        y - 0.035,
        shape_label,
        ha="center",
        va="top",
        fontsize=8.8,
        color="#444",
        style="italic",
        zorder=4,
    )


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.4,
            color="#333",
            zorder=5,
        )
    )
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.02,
            text,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#555",
        )


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    n = len(BLOCKS)
    # Two rows: 3 blocks top, 3 blocks bottom, arrows go right-then-down-left.
    positions = [
        (0.02, 0.63),
        (0.36, 0.63),
        (0.70, 0.63),
        (0.70, 0.15),
        (0.36, 0.15),
        (0.02, 0.15),
    ]
    box_w = 0.26
    box_h = 0.28

    for (x, y), blk in zip(positions, BLOCKS):
        draw_block(
            ax,
            x,
            y,
            box_w,
            box_h,
            blk["text"],
            blk["shape"],
            blk["color"],
            blk["edge"],
            blk.get("text_color", "#222"),
        )

    # Arrows across top row (left → right)
    for i in range(2):
        x1 = positions[i][0] + box_w
        y1 = positions[i][1] + box_h / 2
        x2 = positions[i + 1][0]
        y2 = positions[i + 1][1] + box_h / 2
        arrow(ax, x1, y1, x2, y2)

    # Arrow from block 3 (top-right) down to block 4 (bottom-right)
    x1 = positions[2][0] + box_w / 2
    y1 = positions[2][1]
    x2 = positions[3][0] + box_w / 2
    y2 = positions[3][1] + box_h
    arrow(ax, x1, y1, x2, y2)

    # Arrows across bottom row (right → left)
    for i in range(3, 5):
        x1 = positions[i][0]
        y1 = positions[i][1] + box_h / 2
        x2 = positions[i + 1][0] + box_w
        y2 = positions[i + 1][1] + box_h / 2
        arrow(ax, x1, y1, x2, y2)

    ax.text(
        0.5,
        0.98,
        "Figure 3.1 — Portfolio construction pipeline per test month",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#1B3D6E",
    )
    ax.text(
        0.5,
        0.015,
        "Same pipeline across every loss and every seed; only $\\hat y_t$ varies.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555",
        style="italic",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
