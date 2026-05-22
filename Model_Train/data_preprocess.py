from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = ("PERMNO", "date", "RET", "VOL", "SHROUT")


def load_raw_csvs(
    data_dir: Path,
    pattern: str = "*.csv",
    columns: Optional[Sequence[str]] = None,
    dtypes: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load and vertically concatenate raw CSV files.
    批量读取目录中的原始 CSV 并纵向合并。

    Parameters
    ----------
    data_dir: Path
        Directory that stores the CSV files.
    pattern: str
        Glob pattern to match against files within data_dir.
    columns: Optional[Sequence[str]]
        Optional subset of columns to read for memory efficiency.
    dtypes: Optional[Dict[str, str]]
        Optional dtype overrides forwarded to pandas.read_csv.
    """
    files = sorted(data_dir.glob(pattern))
    frames: List[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir} with pattern {pattern}")
    data = pd.concat(frames, ignore_index=True)
    return data


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """
    Ensure required columns exist before downstream operations proceed.
    检查必需列是否存在，避免后续步骤因列缺失而失败。
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def parse_dates(
    data: pd.DataFrame,
    date_column: str = "date",
    infer_format: bool = True,
) -> pd.DataFrame:
    """
    Ensure date column is in pandas datetime format and sorted by PERMNO, date.
    将日期列统一转为 pandas 的 datetime 并按 PERMNO、日期排序。
    """
    df = data.copy()
    parsed_dates = pd.to_datetime(
        df[date_column], errors="coerce", infer_datetime_format=infer_format
    )
    if parsed_dates.isna().any():
        raise ValueError("Encountered unparsable dates; please check the raw CSV files.")
    df[date_column] = parsed_dates
    df = df.sort_values(["PERMNO", date_column])
    return df


def clean_core_columns(
    data: pd.DataFrame,
    columns: Iterable[str] = ("RET", "VOL", "SHROUT"),
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Handle missing values in core columns for each PERMNO using a simple strategy.
    按证券分组对 RET/VOL/SHROUT 做缺失值处理（前/后向填充）。
    """
    df = data.copy()
    validate_columns(df, ["PERMNO", *columns])
    group = df.groupby("PERMNO", group_keys=False)
    if method == "ffill":
        df[list(columns)] = group[list(columns)].ffill()
    elif method == "bfill":
        df[list(columns)] = group[list(columns)].bfill()
    else:
        raise ValueError(f"Unsupported cleaning method: {method}")
    df = df.dropna(subset=list(columns))
    return df


def add_basic_variables(
    data: pd.DataFrame,
    ret_column: str = "RET",
    vol_column: str = "VOL",
    shrout_column: str = "SHROUT",
) -> pd.DataFrame:
    """
    Add simple return and turnover variables used as building blocks for features.
    生成基础变量 r（简单收益）和 to（换手率）供后续特征使用。
    """
    df = data.copy()
    validate_columns(df, [ret_column, vol_column, shrout_column])
    df["r"] = pd.to_numeric(df[ret_column], errors="coerce")
    vol = pd.to_numeric(df[vol_column], errors="coerce")
    shrout = pd.to_numeric(df[shrout_column], errors="coerce")
    denom = shrout * 1000.0
    denom = denom.replace(0.0, np.nan)
    df["to"] = vol / denom
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_target_return(
    data: pd.DataFrame,
    ret_column: str = "r",
    id_column: str = "PERMNO",
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Add r_{i,t+1} as the prediction target via one-period-ahead shift within each PERMNO.
    在每个 PERMNO 内向前移动一月得到预测目标 r_{i,t+1}。
    """
    df = data.copy()
    validate_columns(df, [id_column, ret_column, date_column])
    group = df.groupby(id_column, group_keys=False)
    df["target_ret"] = group[ret_column].shift(-1)
    df = df.dropna(subset=["target_ret"])
    df = df.sort_values([id_column, date_column])
    return df


def prepare_panel_data(
    data_dir: Optional[str] = None,
    pattern: str = "*.csv",
    cleaning_method: str = "ffill",
    columns: Optional[Sequence[str]] = None,
    dtypes: Optional[Dict[str, str]] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline that returns a cleaned panel with basic variables and target.
    贯穿式预处理流水线，输出包含基础变量与目标值的整洁面板数据。
    """
    if data_dir is None:
        data_dir_path = Path(__file__).resolve().parent.parent
    else:
        data_dir_path = Path(data_dir)
    requested_columns = columns if columns is not None else DEFAULT_COLUMNS
    raw = load_raw_csvs(
        data_dir_path,
        pattern=pattern,
        columns=requested_columns,
        dtypes=dtypes,
    )
    parsed = parse_dates(raw, date_column=date_column)
    cleaned = clean_core_columns(parsed, method=cleaning_method)
    enriched = add_basic_variables(cleaned)
    panel = add_target_return(enriched, date_column=date_column)
    panel = panel.dropna(subset=["r", "to", "target_ret"])
    return panel


__all__ = [
    "load_raw_csvs",
    "parse_dates",
    "clean_core_columns",
    "add_basic_variables",
    "add_target_return",
    "prepare_panel_data",
]
