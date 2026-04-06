#!/usr/bin/env python3
"""
读取 sanity_outputs 中的组合结果，计算逐月累计 Sharpe 并绘制柱状对比图。
Sharpe(i) 使用从测试开始到第 i 个月的 long-short 月度收益，符合
sqrt(12) * r.mean() / r.std(ddof=1) 的定义，不使用滚动窗口。
"""
import argparse
import os
from pathlib import Path

DEFAULT_MPL_CACHE = Path("sanity_outputs/.mpl_cache")
DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANNUAL_FACTOR = np.sqrt(12.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative monthly Sharpe for a given portfolio CSV pair."
    )
    parser.add_argument(
        "--mse-csv",
        type=Path,
        default=Path("sanity_outputs/p1_sanity_metrics_mse.csv"),
        help="MSE 实验的 metrics CSV 路径。",
    )
    parser.add_argument(
        "--medse-csv",
        type=Path,
        default=Path("sanity_outputs/p1_sanity_metrics_medse.csv"),
        help="MedSE 实验的 metrics CSV 路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sanity_outputs/portfolio_sharpe.png"),
        help="保存柱状图的路径。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Portfolio Monthly Sharpe Comparison",
        help="图形标题，可以写成 P1/P2/P3 等描述。",
    )
    return parser.parse_args()


def compute_cumulative_sharpe(returns: pd.Series) -> pd.Series:
    """Sharpe(i) = sqrt(12) * mean(r[:i]) / std(r[:i])，首个有效值需要至少两个月。"""
    sharpe = []
    for idx in range(len(returns)):
        window = returns.iloc[: idx + 1].dropna()
        if window.size < 2:
            sharpe.append(np.nan)
            continue
        std = window.std(ddof=1)
        if not np.isfinite(std) or std < 1e-12:
            sharpe.append(np.nan)
            continue
        sharpe.append(ANNUAL_FACTOR * window.mean() / std)
    return pd.Series(sharpe, index=returns.index)


def load_returns(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, usecols=["month", "long_short_return"])
    df["month"] = pd.to_datetime(df["month"])
    return df.rename(columns={"long_short_return": f"long_short_return_{suffix}"})


def plot_bars(
    months: pd.Series,
    mse_sharpe: pd.Series,
    medse_sharpe: pd.Series,
    output: Path,
    title: str,
) -> None:
    x = np.arange(len(months))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, mse_sharpe, width=width, label="MSE Sharpe", color="#4c72b0")
    plt.bar(x + width / 2, medse_sharpe, width=width, label="MedSE Sharpe", color="#dd8452")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, [m.strftime("%Y-%m") for m in months], rotation=45, ha="right")
    plt.ylabel("Annualized Sharpe (Cumulative)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved bar chart to {output}")


def main() -> None:
    args = parse_args()
    mse_df = load_returns(args.mse_csv, "mse")
    medse_df = load_returns(args.medse_csv, "medse")

    merged = (
        pd.merge(mse_df, medse_df, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise RuntimeError("Merged dataframe is empty; check input CSVs.")

    merged["sharpe_mse"] = compute_cumulative_sharpe(merged["long_short_return_mse"])
    merged["sharpe_medse"] = compute_cumulative_sharpe(merged["long_short_return_medse"])

    plot_bars(
        merged["month"],
        merged["sharpe_mse"],
        merged["sharpe_medse"],
        args.output,
        args.title,
    )


if __name__ == "__main__":
    main()
