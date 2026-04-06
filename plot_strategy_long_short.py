#!/usr/bin/env python3
"""
参照参考图，将多个策略的 Long / Short / Long-Short 累计收益画在同一张图上：
- 颜色区分不同策略；
- 线型区分 Long（实线）、Short（虚线）、Long-Short（点线）。
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

DEFAULT_MPL_CACHE = Path("sanity_outputs/.mpl_cache")
DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_series(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Series 参数必须形如 Label:path/to/csv")
    label, path = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label 不能为空")
    return label, Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative long/short/long-short returns for multiple portfolios."
    )
    parser.add_argument(
        "--series",
        type=parse_series,
        action="append",
        required=True,
        help="以 label:path 形式指定每个策略的 metrics CSV，可多次使用。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sanity_outputs/portfolio_long_short.png"),
        help="输出图片路径。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Portfolio Long/Short Performance",
        help="图表标题。",
    )
    return parser.parse_args()


def compute_cumulative(values: pd.Series) -> pd.Series:
    arr = values.astype(float).fillna(0.0)
    return (1.0 + arr).cumprod() - 1.0


def load_strategy(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    required = ["month", "long_return", "short_return", "long_short_return"]
    df = pd.read_csv(path)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{label} 缺少列 '{col}'，无法画图。")
    df = df.sort_values("month").reset_index(drop=True)
    df["month"] = pd.to_datetime(df["month"])
    df["cum_long"] = compute_cumulative(df["long_return"])
    df["cum_short"] = compute_cumulative(df["short_return"])
    df["cum_long_short"] = compute_cumulative(df["long_short_return"])
    return df


def plot_strategies(curves: List[Tuple[str, pd.DataFrame]], title: str, output: Path) -> None:
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap("tab10").colors
    style_map = {
        "cum_long": ("Long", "-"),
        "cum_short": ("Short", "--"),
        "cum_long_short": ("Long-Short", ":"),
    }
    for idx, (label, df) in enumerate(curves):
        color = colors[idx % len(colors)]
        for key, (suffix, linestyle) in style_map.items():
            plt.plot(
                df["month"],
                df[key],
                label=f"{label}: {suffix}",
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
    print(f"Saved long/short comparison plot to {output}")


def main() -> None:
    args = parse_args()
    curves: List[Tuple[str, pd.DataFrame]] = []
    for label, path in args.series:
        df = load_strategy(label, path)
        curves.append((label, df))
    plot_strategies(curves, args.title, args.output)


if __name__ == "__main__":
    main()
