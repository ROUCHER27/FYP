#!/usr/bin/env python3
"""
将多个 (strategy, loss) 组合的 long-short 累计收益绘制在同一张图上：
- 颜色区分 loss function；
- 线型区分策略（P1/P2/P3 等）。
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_MPL_CACHE = Path("sanity_outputs/.mpl_cache")
DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINESTYLES = ["-", "--", "-.", ":"]
COLOR_OVERRIDES = {
    ("P1", "MSE"): "#2ca02c",  # green
    ("P2", "MSE"): "#d62728",  # red
    ("P2", "MedSE"): "#ff69b4",  # pink
    ("P3", "MSE"): "#9467bd",  # purple
    ("P3", "MedSE"): "#ffd700",  # yellow
}
LOSS_STYLE_OVERRIDES = {
    "MSE": "-",
    "MedSE": (0, (5, 5)),  # evenly spaced dash
}


def parse_series(value: str) -> Tuple[str, str, Path]:
    """
    输入形如 Strategy:Loss:path/to/csv
    """
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Series 参数必须形如 Strategy:Loss:path/to/csv")
    strategy, loss, path = (p.strip() for p in parts)
    if not strategy or not loss:
        raise argparse.ArgumentTypeError("Strategy 与 Loss 不能为空")
    return strategy, loss, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot long-short cumulative returns by strategy and loss."
    )
    parser.add_argument(
        "--series",
        type=parse_series,
        action="append",
        required=True,
        help="以 Strategy:Loss:path 形式提供 CSV，可多次使用。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sanity_outputs/strategy_loss_longshort.png"),
        help="输出图像路径。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Long-Short Cumulative Return by Strategy & Loss",
        help="图表标题。",
    )
    return parser.parse_args()


def compute_cumulative(returns: pd.Series) -> pd.Series:
    arr = returns.astype(float).fillna(0.0)
    return (1.0 + arr).cumprod() - 1.0


def load_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{path} 缺少 'month' 列")
    df = df.sort_values("month").reset_index(drop=True)
    df["month"] = pd.to_datetime(df["month"])
    if "cumulative_long_short_return" in df.columns:
        df["cum_long_short"] = df["cumulative_long_short_return"].astype(float)
    elif "long_short_return" in df.columns:
        df["cum_long_short"] = compute_cumulative(df["long_short_return"])
    else:
        raise ValueError(f"{path} 缺少 long_short_return 或 cumulative_long_short_return")
    return df[["month", "cum_long_short"]]


def plot_series(
    entries: List[Tuple[str, str, pd.DataFrame]], title: str, output: Path
) -> None:
    plt.figure(figsize=(12, 6))
    losses = sorted({loss for _, loss, _ in entries})
    strategies = sorted({strategy for strategy, _, _ in entries})
    cmap = plt.get_cmap("tab10")
    default_colors: Dict[str, Tuple[float, float, float]] = {}
    for idx, strategy in enumerate(strategies):
        default_colors[strategy] = cmap(idx % cmap.N)

    for strategy, loss, df in entries:
        color = COLOR_OVERRIDES.get((strategy, loss), default_colors[strategy])
        linestyle = LOSS_STYLE_OVERRIDES.get(
            loss, LINESTYLES[strategies.index(strategy) % len(LINESTYLES)]
        )
        plt.plot(
            df["month"],
            df["cum_long_short"],
            label=f"{strategy} - {loss}",
            color=color,
            linestyle=linestyle,
        )

    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Month")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved strategy-loss plot to {output}")


def main() -> None:
    args = parse_args()
    entries: List[Tuple[str, str, pd.DataFrame]] = []
    for strategy, loss, path in args.series:
        df = load_series(path)
        entries.append((strategy, loss, df))
    plot_series(entries, args.title, args.output)


if __name__ == "__main__":
    main()
