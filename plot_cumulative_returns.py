#!/usr/bin/env python3
"""
将多个策略的 cumulative return 曲线绘制在同一张图上，便于横向比较。
默认读取各自 metrics CSV 中的 `cumulative_long_short_return` 列。
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

DEFAULT_MPL_CACHE = Path("sanity_outputs/.mpl_cache")
DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))

import matplotlib.pyplot as plt
import pandas as pd


def parse_series(value: str) -> Tuple[str, Path]:
    """
    解析 CLI 参数，格式 label:path。
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError("Series 参数必须形如 Label:path/to/csv")
    label, path = value.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Series label 不能为空")
    path_obj = Path(path.strip())
    return label, path_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative returns for multiple portfolios."
    )
    parser.add_argument(
        "--series",
        type=parse_series,
        action="append",
        required=True,
        help="指定 label:csv_path，可多次使用构建多条曲线。",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="cumulative_long_short_return",
        help="曲线使用的列名，默认为 cumulative_long_short_return。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sanity_outputs/portfolio_cumulative_returns.png"),
        help="输出图片路径。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Portfolio Cumulative Return Comparison",
        help="图表标题。",
    )
    return parser.parse_args()


def load_curve(label: str, path: Path, column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{label} 缺少 'month' 列: {path}")
    df["month"] = pd.to_datetime(df["month"])
    if column not in df.columns:
        if column == "cumulative_long_short_return" and "long_short_return" in df.columns:
            returns = df["long_short_return"].astype(float).fillna(0.0)
            df[column] = (1.0 + returns).cumprod() - 1.0
        else:
            raise ValueError(f"{label} 缺少列 '{column}'，无法绘图。")
    df = df.sort_values("month").reset_index(drop=True)
    return df[["month", column]].rename(columns={column: label})


def plot_curves(curves: List[pd.DataFrame], output: Path, title: str) -> None:
    plt.figure(figsize=(12, 6))
    for df in curves:
        label = df.columns[1]
        plt.plot(df["month"], df[label], marker="o", label=label)
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
    print(f"Saved cumulative return plot to {output}")


def main() -> None:
    args = parse_args()
    curves = []
    for label, path in args.series:
        df = load_curve(label, path, args.column)
        curves.append(df)
    plot_curves(curves, args.output, args.title)


if __name__ == "__main__":
    main()
