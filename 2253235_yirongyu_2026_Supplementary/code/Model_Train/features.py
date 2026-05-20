from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """
    Configuration holder for constructing X1, X2, X3 feature sets.
    控制构建 X1/X2/X3 特征集合所需的窗口长度等参数。
    """

    lookback_months: int = 12


def rolling_group_apply(
    df: pd.DataFrame,
    value_column: str,
    window: int,
    func,
) -> pd.Series:
    """
    Apply a rolling function within each PERMNO group and align with original index.
    对每个 PERMNO 分组执行滚动计算，并将结果对齐回原始索引。
    """
    grouped = df.groupby("PERMNO")[value_column]
    result = grouped.rolling(window=window, min_periods=window).apply(func, raw=True)
    result = result.reset_index(level=0, drop=True)
    return result


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_column: str,
    date_column: str = "date",
) -> pd.Series:
    """
    Cross-sectional standardization of a factor within each date.
    在每个日期的横截面上将指标做 z-score 标准化。
    """
    grouped = df.groupby(date_column)[value_column]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (df[value_column] - mean) / std.replace(0.0, np.nan)
    return z


def build_feature_set_x1(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Build feature set X1 according to assignment formulas using panel with r and to.
    依据文档公式使用 r/to 构造 X1 特征（收益/换手的累计指标）。
    """
    df = panel.copy()
    lookbacks = [1, 3, 6, 9, 12]
    group = df.groupby("PERMNO")
    df["_r_prev"] = group["r"].shift(1)
    df["_to_prev"] = group["to"].shift(1)

    def _cum_return(values: np.ndarray) -> float:
        return float(np.prod(values + 1.0) - 1.0)

    feature_cols: List[str] = []
    for window in lookbacks:
        cr_col = f"cr_{window}m"
        co_col = f"co_{window}m"
        df[cr_col] = rolling_group_apply(df, "_r_prev", window, _cum_return)
        df[co_col] = rolling_group_apply(df, "_to_prev", window, np.sum)
        feature_cols.extend([cr_col, co_col])

    df = df.drop(columns=["_r_prev", "_to_prev"])
    df = df.dropna(subset=feature_cols)
    return df


def build_feature_set_x2(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Build feature set X2 according to assignment formulas using panel with r and to.
    依据文档公式构造 X2 特征（排除最近一期的归一化累计收益）。
    """
    df = panel.copy()
    lookbacks = [3, 6, 9, 12]
    group = df.groupby("PERMNO")
    df["_r_excl_recent"] = group["r"].shift(2)

    def _cum_return(values: np.ndarray) -> float:
        return float(np.prod(values + 1.0) - 1.0)

    feature_cols: List[str] = []
    raw_cols: List[str] = []
    for window in lookbacks:
        raw_col = f"cr_excl_{window}m"
        norm_col = f"ncr_{window}m"
        df[raw_col] = rolling_group_apply(df, "_r_excl_recent", window, _cum_return)
        df[norm_col] = cross_sectional_zscore(df, raw_col)
        raw_cols.append(raw_col)
        feature_cols.append(norm_col)

    df = df.drop(columns=["_r_excl_recent", *raw_cols])
    df = df.dropna(subset=feature_cols)
    return df


def build_feature_set_x3(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Build feature set X3 according to assignment formulas using panel with r and to.
    依据文档公式构造 X3 特征（过去 12 个月的标准化月度收益）。
    """
    df = panel.copy()
    lookback = max(config.lookback_months, 1)
    df["_r_norm"] = cross_sectional_zscore(df, "r")
    group = df.groupby("PERMNO")
    feature_cols: List[str] = []
    for lag in range(1, lookback + 1):
        col = f"nr_lag_{lag}"
        df[col] = group["_r_norm"].shift(lag)
        feature_cols.append(col)
    df = df.drop(columns=["_r_norm"])
    df = df.dropna(subset=feature_cols)
    return df


def assemble_feature_matrix(
    features: pd.DataFrame,
    target_column: str = "target_ret",
    id_column: str = "PERMNO",
    date_column: str = "date",
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Convert a feature DataFrame into numpy arrays for model training.
    将带特征的数据框打包为模型训练所需的 numpy 矩阵和标记。
    """
    feature_cols = [
        col
        for col in features.columns
        if col not in {target_column, id_column, date_column}
    ]
    x = features[feature_cols].to_numpy(dtype=np.float32)
    y = features[target_column].to_numpy(dtype=np.float32)
    ids = features[id_column]
    dates = features[date_column]
    return x, y, ids, dates


__all__ = [
    "FeatureConfig",
    "rolling_group_apply",
    "cross_sectional_zscore",
    "build_feature_set_x1",
    "build_feature_set_x2",
    "build_feature_set_x3",
    "assemble_feature_matrix",
]
