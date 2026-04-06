import argparse
import ast
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def configure_matplotlib(output_dir: Path) -> None:
    """
    Some environments are headless; switch backend to Agg so plots work without a GUI.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa
    globals()["plt"] = plt

from Model_Train.data_preprocess import prepare_panel_data
from Model_Train.features import (
    FeatureConfig,
    assemble_feature_matrix,
    build_feature_set_x1,
)
from Model_Train.losses import medse_loss, mse_loss
from Model_Train.models import MLP, MLPConfig


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """
    构建公共 CLI，允许指定数据目录、时间区间、训练超参数等。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=str, default=".", help="CSV directory.")
    parser.add_argument(
        "--pattern", type=str, default="*.csv", help="Glob for CSV selection."
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        help="Lookback window for Feature Set X1.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="1990-01",
        help="Inclusive train period start (YYYY-MM).",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="1994-12",
        help="Inclusive train period end (YYYY-MM).",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="1995-01",
        help="Inclusive test period start (YYYY-MM).",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=6,
        help="Number of consecutive months for testing.",
    )
    parser.add_argument(
        "--best-config-path",
        type=str,
        default="best_hyperparameters.txt",
        help="Step 3 best config file.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Epochs for each experiment.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Training batch size."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sanity_outputs",
        help="Directory to store metrics tables.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch/random.",
    )
    return parser


def detect_device() -> torch.device:
    """
    训练前先检测 GPU/MPS，若无则退回 CPU，确保向前兼容。
    """
    if torch.cuda.is_available():
        print("Using CUDA acceleration.")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) acceleration.")
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    保证 numpy / torch / random 的伪随机数可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_best_config(path: Path, input_dim: int) -> MLPConfig:
    """
    复用 Step3 网格搜索保存的最佳 MLP 结构，避免重复调参。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Best config file not found at {path}. Run step 3 or adjust --best-config-path."
        )
    config_dict = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Config:"):
            payload = line.split("Config:", 1)[1].strip()
            config_dict = ast.literal_eval(payload)
            break
    if config_dict is None:
        raise ValueError(f"{path} does not contain a 'Config:' line.")
    hidden_dims = config_dict.get("hidden_dims")
    if hidden_dims is None:
        raise ValueError(f"'hidden_dims' missing in {path}.")
    activation = config_dict.get("activation", "relu")
    dropout = float(config_dict.get("dropout", 0.0))
    cfg_input_dim = int(config_dict.get("input_dim", input_dim))
    if cfg_input_dim != input_dim:
        print(
            f"Warning: config input_dim={cfg_input_dim} differs from features {input_dim}. Overriding."
        )
    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        activation=activation,
        dropout=dropout,
    )


def to_period(series: pd.Series) -> pd.Series:
    """
    将日期列转换为 pandas Period(M) 便于按整月筛选。
    """
    return pd.to_datetime(series).dt.to_period("M")


def build_mask(periods: pd.Series, start: str, end: str) -> pd.Series:
    """
    在给定的起止月份之间构建布尔掩码，配合静态训练窗口。
    """
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    return (periods >= start_period) & (periods <= end_period)


def month_sequence(start: str, count: int) -> List[pd.Period]:
    base = pd.Period(start, freq="M")
    return [base + i for i in range(count)]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MLPConfig,
    loss_name: str,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
) -> MLP:
    """
    单次训练流程：DataLoader -> 前向 -> 反向 -> 更新。
    loss_name 决定使用 MSE 还是 MedSE。
    """
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    if loss_name == "mse":
        criterion = lambda a, b: mse_loss(a, b, reduction="mean")
    elif loss_name == "medse":
        criterion = lambda a, b: medse_loss(a, b, reduction="median")
    else:
        raise ValueError(f"Unsupported loss for sanity check: {loss_name}")

    model.train()
    for epoch in range(max_epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(batch_y, preds)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == max_epochs:
            print(f"Epoch {epoch + 1}/{max_epochs} finished for {loss_name.upper()}.")
    return model


def predict(model: MLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    tensor = torch.from_numpy(x).float().to(device)
    with torch.no_grad():
        preds = model(tensor).cpu().numpy().reshape(-1)
    return preds


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    误差指标部分：MSE/MedSE 以及 R^2。
    - R^2 在这里衡量预测解释度，即 1 - RSS/TSS。
    """
    if y_true.size == 0:
        return {"mse": np.nan, "medse": np.nan, "r2": np.nan}
    residuals = y_true - y_pred
    mse = float(np.mean(residuals**2))
    medse = float(np.median(residuals**2))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(residuals**2) / denom)
    return {"mse": mse, "medse": medse, "r2": r2}


def compute_portfolio_returns(
    preds: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """
    Rebalance & Return 逻辑：
    - 每月按预测值排序，Top 10% 做多，Bottom 10% 做空。
    - 组合收益 = long mean - short mean。
    这就是所谓的“策略调仓（rebalance）”。
    """
    n = preds.size
    if n == 0:
        return {"long": np.nan, "short": np.nan, "long_short": np.nan}
    order = np.argsort(preds)
    k = max(1, int(np.floor(n * 0.1)))
    long_idx = order[-k:]
    short_idx = order[:k]
    long_ret = float(np.mean(targets[long_idx])) if long_idx.size else float("nan")
    short_ret = float(np.mean(targets[short_idx])) if short_idx.size else float("nan")
    long_short = (
        float(long_ret - short_ret)
        if np.isfinite(long_ret) and np.isfinite(short_ret)
        else float("nan")
    )
    return {"long": long_ret, "short": short_ret, "long_short": long_short}


def compute_long_short_stats(
    returns: pd.Series, periods_per_year: int = 12
) -> Dict[str, float]:
    """
    汇总 long-short 序列的累计收益、标准差与 Sharpe（按月收益年化）。
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "cumulative_return": float("nan"),
            "std": float("nan"),
            "sharpe": float("nan"),
        }
    cumulative_return = float(np.prod(1.0 + arr) - 1.0)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    mean = float(np.mean(arr))
    sharpe = float("nan")
    if np.isfinite(std) and std > 1e-12:
        sharpe = mean / std
        if periods_per_year and periods_per_year > 1:
            sharpe *= np.sqrt(periods_per_year)
        sharpe = float(sharpe)
    return {
        "cumulative_return": cumulative_return,
        "std": std,
        "sharpe": sharpe,
    }


def format_period(period: pd.Period) -> str:
    return period.strftime("%Y-%m")


def  plot_curves(df: pd.DataFrame, loss_name: str, output_dir: Path) -> None:
    """
    画两类图：
    1. Loss 曲线（MSE 或 MedSE vs Month）。
    2. Long-Short Return 曲线。
    """
    months = pd.to_datetime(df["month"])
    metric_col = "mse" if loss_name == "mse" else "medse"
    label = metric_col.upper()

    plt.figure(figsize=(8, 4))
    plt.plot(months, df[metric_col], marker="o", label=label)
    plt.title(f"{label} per Month")
    plt.xlabel("Month")
    plt.ylabel(label)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    loss_path = output_dir / f"{loss_name}_loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(
        months,
        df["long_short_return"],
        marker="o",
        color="purple",
        label="Long-Short Return",
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Long-Short Return per Month")
    plt.xlabel("Month")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    ret_path = output_dir / f"{loss_name}_returns_curve.png"
    plt.savefig(ret_path, dpi=200)
    plt.close()

    print(f"Loss curve saved to {loss_path}")
    print(f"Long-Short return curve saved to {ret_path}")


def run_sanity_check(loss_name: str, args: argparse.Namespace) -> None:
    """
    固定训练窗口，只训练一次模型，然后在未来 6 个月逐月预测/调仓。
    """
    set_seed(args.seed)
    device = detect_device()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    configure_matplotlib(output_dir)

    print("=== Sanity Check: Static 5Y Train / 6M Test ===")
    panel = prepare_panel_data(data_dir=args.data_dir, pattern=args.pattern)
    print(f"Loaded {len(panel)} rows from raw panel.")

    feature_cfg = FeatureConfig(lookback_months=args.lookback_months)
    df_x1 = build_feature_set_x1(panel, feature_cfg)
    x_all, y_all, ids, dates = assemble_feature_matrix(df_x1)
    if x_all.size == 0:
        raise RuntimeError("Feature matrix is empty after X1 construction.")

    date_periods = to_period(dates)
    train_mask = build_mask(date_periods, args.train_start, args.train_end).to_numpy()
    test_periods = month_sequence(args.test_start, args.test_months)
    test_mask = date_periods.isin(test_periods).to_numpy()

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    if x_train.shape[0] == 0:
        raise RuntimeError("Training window contains zero samples.")
    if test_mask.sum() == 0:
        raise RuntimeError("Testing window contains zero samples.")

    config = read_best_config(Path(args.best_config_path), input_dim=x_all.shape[1])
    print("Using MLPConfig:", config)
    ensure_output_dir(output_dir)

    model = train_model(
        x_train,
        y_train,
        config,
        loss_name=loss_name,
        device=device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )

    label = "MSE" if loss_name == "mse" else "MedSE"
    month_records: List[Dict[str, object]] = []
    print("Rebalancing monthly: Top 10% predictions -> Long, Bottom 10% -> Short.")
    for period in test_periods:
        month_mask = (date_periods == period).to_numpy()
        mask = month_mask & test_mask
        if not mask.any():
            print(f"Warning: no data for test month {format_period(period)}.")
            continue
        x_month = x_all[mask]
        y_month = y_all[mask]
        preds = predict(model, x_month, device=device)
        metrics = compute_metrics(y_month, preds)
        port = compute_portfolio_returns(preds, y_month)
        # 根据实验类型选择关注的损失（MSE 或 MedSE）
        loss_value = metrics["mse"] if loss_name == "mse" else metrics["medse"]
        month_records.append(
            {
                "month": format_period(period),
                "sample_size": int(mask.sum()),
                label.lower(): loss_value,
                "r2": metrics["r2"],
                "long_return": port["long"],
                "short_return": port["short"],
                "long_short_return": port["long_short"],
            }
        )
        print(
            f"Month {format_period(period)} | {label} {loss_value:.6f} | "
            f"R2 {metrics['r2']:.4f} | Long {port['long']:.4f} | "
            f"Short {port['short']:.4f} | Long-Short {port['long_short']:.4f}"
        )

    if not month_records:
        print(f"No monthly results recorded for {loss_name}.")
        return

    df_result = pd.DataFrame(month_records)
    long_short_returns = df_result["long_short_return"].astype(float)
    df_result["cumulative_long_short_return"] = (
        (1.0 + long_short_returns.fillna(0.0)).cumprod() - 1.0
    )
    ls_stats = compute_long_short_stats(long_short_returns)
    csv_path = Path(args.output_dir) / f"sanity_metrics_{loss_name}.csv"
    df_result.to_csv(csv_path, index=False)
    print(f"Saved metrics for {label} to {csv_path}")

    avg_key = label.lower()
    summary = {
        "loss": loss_name,
        f"avg_{avg_key}": float(df_result[avg_key].mean()),
        "avg_r2": float(df_result["r2"].mean()),  # R^2 体现解释度
        "avg_long_short": float(df_result["long_short_return"].mean()),
        "long_short_cumulative_return": ls_stats["cumulative_return"],
        "long_short_std": ls_stats["std"],
        "long_short_sharpe": ls_stats["sharpe"],
    }
    summary_path = output_dir / f"sanity_summary_{loss_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(
        "Long-Short stats | Cumulative "
        f"{ls_stats['cumulative_return']:.4f} | "
        f"Std {ls_stats['std']:.4f} | Sharpe {ls_stats['sharpe']:.4f}"
    )
    plot_curves(df_result, loss_name, output_dir)
    if loss_name == "mse":
        print("MSE 实验完成。")
    else:
        print("MedSE 实验完成。")
    print(
        "\n提示：若要比较 MSE 与 MedSE，请分别运行对应脚本（run_sanity_check_mse.py / run_sanity_check_medse.py）。"
    )
