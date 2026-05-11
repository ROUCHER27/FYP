"""
Figure 3.2 — Loss-surface comparison of directional and hybrid objectives.

2×2 heatmap on the (y, ŷ) plane, range [-15%, +15%]:
  (a) GMADL signed directional score
  (b) Huber magnitude backbone
  (c) Additive hybrid A3  (λ_dir=1, λ_hub=0.1)
  (d) Multiplicative hybrid M2  (λ_dir=5)

Overlays on every panel:
  - vertical line at y=0, horizontal line at ŷ=0
  - dashed diagonal ŷ=y  labelled "calibration line"
  - text labels "Sign correct" (Q1/Q3) and "Sign wrong" (Q2/Q4)

Illustrative; closed-form components from §3.3; no training data.
Output: paper/figures/fig3_2_hybrid_loss_surfaces.png  (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import _style
from _style import apply_paper_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig3_2_hybrid_loss_surfaces.png"

A, B = 100.0, 2.0
DELTA = 0.01
LAMBDA_DIR_A3, LAMBDA_HUB_A3 = 1.0, 0.1
LAMBDA_DIR_M2 = 5.0

N = 300
LIM = 0.15
y_grid = np.linspace(-LIM, LIM, N)
yhat_grid = np.linspace(-LIM, LIM, N)
Y, YHAT = np.meshgrid(y_grid, yhat_grid)   # YHAT on y-axis (rows), Y on x-axis (cols)


import matplotlib.colors as mcolors

def _dark_center_diverging():
    """Diverging colormap: blue → dark grey → red, so centre is readable."""
    cmap_base = plt.cm.RdBu_r
    colors_base = cmap_base(np.linspace(0, 1, 256))
    # Replace the middle ~20% with dark grey (#444444)
    mid = 128
    width = 26
    grey = np.array([0.067, 0.067, 0.067, 1.0])   # #111111 — very dark centre
    for i in range(mid - width, mid + width):
        t = abs(i - mid) / width          # 0 at centre, 1 at edge
        colors_base[i] = (1 - t) * grey + t * colors_base[i]
    return mcolors.LinearSegmentedColormap.from_list("RdGrey_Bu", colors_base)

CMAP_DIV = _dark_center_diverging()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def dir_gate(Y, YHAT):
    penalty = 1.0 - sigmoid(A * Y * YHAT)
    weight = np.abs(Y) ** B
    batch_mean = weight.mean()
    return penalty * (weight / (batch_mean + 1e-8))


def huber(err):
    ae = np.abs(err)
    return np.where(ae <= DELTA, 0.5 * err**2, DELTA * (ae - 0.5 * DELTA))


def gmadl_score(Y, YHAT):
    return -(sigmoid(A * Y * YHAT) - 0.5) * np.abs(Y) ** B


def add_a3(Y, YHAT):
    return LAMBDA_DIR_A3 * dir_gate(Y, YHAT) + LAMBDA_HUB_A3 * huber(Y - YHAT)


def mul_m2(Y, YHAT):
    return (1.0 + LAMBDA_DIR_M2 * dir_gate(Y, YHAT)) * huber(Y - YHAT)


def add_overlays(ax, lim):
    # Black lines — maximum contrast on both light-grey centre and coloured extremes
    ax.axvline(0, color="black", lw=1.6, ls="-", alpha=0.85)
    ax.axhline(0, color="black", lw=1.6, ls="-", alpha=0.85)
    diag = np.linspace(-lim, lim, 2)
    ax.plot(diag, diag, color="black", lw=1.6, ls="--", alpha=0.85)
    ax.text(lim * 0.55, lim * 0.82, "calibration\nline",
            color="black", fontsize=7.5, ha="center", va="bottom", alpha=0.85)
    for (tx, ty, txt) in [
        ( lim*0.55,  lim*0.55, "Sign\ncorrect"),
        (-lim*0.55, -lim*0.55, "Sign\ncorrect"),
        (-lim*0.55,  lim*0.55, "Sign\nwrong"),
        ( lim*0.55, -lim*0.55, "Sign\nwrong"),
    ]:
        ax.text(tx, ty, txt, color="black", fontsize=7.5, ha="center", va="center",
                alpha=0.85, fontweight="bold")
    pct_fmt = mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.tick_params(labelsize=8)


def draw_panel(ax, Z, title, cmap, label, diverging=False):
    vmax = np.percentile(np.abs(Z), 97)
    if diverging:
        im = ax.imshow(Z, origin="lower", extent=[-LIM, LIM, -LIM, LIM],
                       cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(Z, origin="lower", extent=[-LIM, LIM, -LIM, LIM],
                       cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    add_overlays(ax, LIM)
    ax.set_title(title, fontsize=10.5, pad=4)
    ax.set_xlabel(r"Realised return $y$", fontsize=9.5)
    ax.set_ylabel(r"Prediction $\hat{y}$", fontsize=9.5)


def main() -> None:
    apply_paper_style()

    Z_gmadl = gmadl_score(Y, YHAT)
    Z_huber = huber(Y - YHAT)
    Z_add   = add_a3(Y, YHAT)
    Z_mul   = mul_m2(Y, YHAT)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.88,
                        hspace=0.38, wspace=0.32)

    draw_panel(axes[0, 0], Z_gmadl,
               r"(a) GMADL signed directional score",
               CMAP_DIV, "GMADL signed score", diverging=True)
    draw_panel(axes[0, 1], Z_huber,
               r"(b) Huber magnitude backbone ($\delta$=0.01)",
               "viridis", "Magnitude loss")
    draw_panel(axes[1, 0], Z_add,
               r"(c) Additive hybrid A3 ($\lambda_{\rm dir}$=1, $\lambda_{\rm hub}$=0.1)",
               "magma", "Hybrid loss")
    draw_panel(axes[1, 1], Z_mul,
               r"(d) Multiplicative hybrid M2 ($\lambda_{\rm dir}$=5)",
               "magma", "Hybrid loss")

    fig.suptitle(
        "Figure 3.2 — Loss-surface comparison of directional and hybrid objectives\n"
        r"($a$=100, $b$=2)  |  Illustrative; no training data",
        fontsize=10.5, fontweight="bold", y=1.00,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
