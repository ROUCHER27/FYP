"""
Figure 4.1 — Data coverage and heavy-tail motivation.

Paper-plot-skills style: open-axis (line_confidence_band) for the histogram,
framed (line_training_curve) for the Gantt timeline.

(a) Horizontal Gantt-style timeline: seven CSV files with their covered date
    range on a shared x-axis 1989–2025; two shaded vertical bands mark the
    training window (1990-01..1994-12) and the main test window
    (1995-01..1996-12). File name placed AT THE START of each bar (inside
    left edge); row count placed to the RIGHT of the bar, outside.
(b) Log-y histogram of raw monthly RET values in the training window
    (1990-01..1994-12 restricted to 89.12-94.csv). Vertical lines at the
    0.1% / 99.9% sample quantiles. Summary statistics box anchored in a
    lower-middle region where histogram counts are sparse.

Inputs : *.csv at the repo root (columns PERMNO/date/RET/VOL/SHROUT).
Output : paper/figures/fig4_1_data_coverage.png (dpi=300)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _style
from _style import (
    C_DYN,
    C_LINE_ALT,
    C_LINE_MAIN,
    apply_paper_style,
    style_framed_axes,
    style_open_axes,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/fig4_1_data_coverage.png"

CSV_ORDER = [
    "89.12-94.csv",
    "94-99.csv",
    "99-04.csv",
    "04-09.csv",
    "09-14.csv",
    "14.12-19.12.csv",
    "19.12-24.12.csv",
]

TRAIN_START = pd.Timestamp("1990-01-01")
TRAIN_END = pd.Timestamp("1994-12-31")
TEST_START = pd.Timestamp("1995-01-01")
TEST_END = pd.Timestamp("1996-12-31")


def load_coverage() -> pd.DataFrame:
    records = []
    for name in CSV_ORDER:
        path = ROOT / name
        d = pd.read_csv(path, usecols=["date"])
        dates = pd.to_datetime(d["date"], errors="coerce")
        records.append(
            dict(name=name, start=dates.min(), end=dates.max(), rows=len(d))
        )
    return pd.DataFrame(records)


def load_training_ret() -> pd.Series:
    d = pd.read_csv(ROOT / "89.12-94.csv", usecols=["date", "RET"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    mask = (d["date"] >= TRAIN_START) & (d["date"] <= TRAIN_END)
    ret = pd.to_numeric(d.loc[mask, "RET"], errors="coerce").dropna()
    return ret


def draw_gantt(ax, df: pd.DataFrame) -> None:
    n = len(df)
    y = np.arange(n)[::-1]  # top = first CSV

    # Training and test window shading first (behind bars).
    ax.axvspan(TRAIN_START, TRAIN_END, color="#F7CF9F", alpha=0.45, zorder=1,
               label="training 1990-01..1994-12")
    ax.axvspan(TEST_START, TEST_END, color="#EF8E6F", alpha=0.50, zorder=1,
               label="main test 1995-01..1996-12")

    for yi, (_, row) in zip(y, df.iterrows()):
        start = row["start"]
        end = row["end"]
        # Bar as a thick line
        ax.plot(
            [start, end],
            [yi, yi],
            color=C_LINE_ALT,
            linewidth=13,
            solid_capstyle="butt",
            zorder=3,
        )
        # File name label inside the left edge of the bar (white text).
        ax.annotate(
            row["name"],
            xy=(start, yi),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8.3,
            color="white",
            fontweight="bold",
            va="center",
            ha="left",
            zorder=4,
        )
        # Row count OUTSIDE the bar, to the right.
        ax.annotate(
            f"{row['rows']:,} rows",
            xy=(end, yi),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.0,
            color="#444",
            va="center",
            ha="left",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)
    ax.set_ylim(-0.8, n - 0.2)   # tight vertical extent
    ax.set_xlim(pd.Timestamp("1989-01-01"), pd.Timestamp("2028-06-01"))
    ax.set_xlabel("Calendar date", fontsize=11, fontweight="bold")
    ax.set_title("(a) CSV file coverage and experiment windows", fontsize=12, pad=6)
    style_framed_axes(ax)
    ax.tick_params(direction="out", length=4, width=0.8, labelsize=9.5)
    ax.legend(
        fontsize=9,
        loc="lower right",
        framealpha=1.0,
        edgecolor="#C8C8C8",
        fancybox=False,
    )
    ax.grid(axis="x", color="#EEE", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def draw_histogram(ax, ret: pd.Series) -> None:
    vals = ret.values
    q_lo, q_hi = np.quantile(vals, [0.001, 0.999])
    bins = np.linspace(-1.0, 2.0, 90)
    ax.hist(
        vals,
        bins=bins,
        color=C_LINE_MAIN,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.set_yscale("log")
    # Shortened quantile lines: stop at 78% of axes height so they do not
    # intersect the upper-left quantile legend or the upper-right stats box.
    ax.axvline(q_lo, color=C_DYN, linestyle="--", linewidth=1.2, ymax=0.78, zorder=4,
               label=f"0.1% quantile = {q_lo:+.3f}")
    ax.axvline(q_hi, color=C_DYN, linestyle="--", linewidth=1.2, ymax=0.78, zorder=4,
               label=f"99.9% quantile = {q_hi:+.3f}")
    ax.axvline(0, color="#777", linewidth=0.7, linestyle=":", zorder=2)

    # Stats box moved inward so its right edge does not cross the right
    # quantile line (at axes x ≈ 0.88).
    ax.text(
        0.72,
        0.96,
        f"n = {len(vals):,}\n"
        f"mean = {vals.mean():+.4f}   std = {vals.std():+.4f}\n"
        f"min = {vals.min():+.3f}   max = {vals.max():+.3f}\n"
        f"(x-axis clipped at 2.0; max bar is a pile-up of values ≥ 2)",
        transform=ax.transAxes,
        fontsize=8.3,
        va="top",
        ha="right",
        color="#222",
        bbox=dict(facecolor="white", edgecolor="#C8C8C8", boxstyle="round,pad=0.4"),
        zorder=6,
    )

    ax.set_xlim(-1.0, 2.0)
    ax.set_xlabel("Monthly RET (training window 1990-01..1994-12)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("(b) Heavy-tail distribution of raw monthly returns", fontsize=12, pad=6)
    style_open_axes(ax)
    ax.tick_params(labelsize=9.5)
    ax.legend(fontsize=9, loc="upper left", frameon=False)


def main() -> None:
    apply_paper_style()
    coverage = load_coverage()
    ret = load_training_ret()

    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(13.2, 4.6))
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.13, top=0.87, wspace=0.16)

    draw_gantt(ax_g, coverage)
    draw_histogram(ax_h, ret)

    fig.suptitle(
        "Figure 4.1 — Data coverage and heavy-tail motivation",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
