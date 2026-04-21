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

Z_SCORE_CLIP = 3.0  # Winsorize 阈值，限制 bucket 内 z-score 极值
MAX_WEIGHT: float | None = 0.05  # Step4: 单票最大权重（5%），控制个股风险敞口


def parse_max_weight(value: str) -> float | None:
    """
    CLI helper: 允许 `--max-weight None` 表示不做 capped-simplex。
    """
    if value is None:
        return None
    value = value.strip()
    if value.lower() == "none":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --max-weight value '{value}'. Use a float or 'None'."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-weight 必须为正数或 'None'")
    return parsed


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
from Model_Train.losses import EXPERIMENT_LOSS_NAMES, get_experiment_loss_fn
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
    parser.add_argument(
        "--max-weight",
        type=parse_max_weight,
        default=MAX_WEIGHT,
        help="Step4: 最大单票权重 (控制组合集中度)，传 'None' 可关闭。",
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
    loss_name 决定本次实验使用的训练损失函数。
    """
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLP(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = get_experiment_loss_fn(loss_name)

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


def compute_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    方向性指标：
    - directional_accuracy: sign(y) == sign(ŷ) 的比例
    - sign_mismatch_large_y: |y| 高于 75 分位时的符号错误率
    """
    if y_true.size == 0:
        return {
            "directional_accuracy": float("nan"),
            "sign_mismatch_large_y": float("nan"),
        }
    sign_match = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(np.mean(sign_match))

    threshold = float(np.percentile(np.abs(y_true), 75))
    large_y_mask = np.abs(y_true) > threshold
    if large_y_mask.sum() == 0:
        sign_mismatch_large_y = float("nan")
    else:
        sign_mismatch_large_y = float(np.mean(~sign_match[large_y_mask]))

    return {
        "directional_accuracy": directional_accuracy,
        "sign_mismatch_large_y": sign_mismatch_large_y,
    }


def apply_weight_cap(
    weights: np.ndarray, max_weight: float | None, max_iter: int = 10
) -> np.ndarray:
    """
    Step4: 将初步归一化的权重进行 capped-simplex 投影，确保：
    - 所有元素非负；
    - 单票权重不超过 max_weight（例如 5%）；
    - 权重之和仍为 1。
    控制单股票风险敞口，避免极端集中。
    若 max_weight=None，则直接返回归一化权重。
    """
    if weights.size == 0:
        return weights
    capped = weights.copy().astype(np.float64)
    eps = 1e-12
    if max_weight is None:
        total = float(np.sum(capped))
        if total <= eps:
            return np.ones_like(capped) / capped.size
        return capped / total
    for _ in range(max_iter):
        over = capped > max_weight + 1e-12
        if not np.any(over):
            break
        capped[over] = max_weight
        mask_capped = capped >= max_weight - 1e-12
        capped_sum = float(np.sum(capped[mask_capped]))
        remaining = 1.0 - capped_sum
        if remaining <= eps:
            return np.ones_like(capped) / capped.size
        mask_free = ~mask_capped
        free_total = float(np.sum(capped[mask_free]))
        if free_total <= eps:
            return np.ones_like(capped) / capped.size
        scale = remaining / free_total
        capped[mask_free] *= scale
    capped = np.clip(capped, 0.0, max_weight)
    total = float(np.sum(capped))
    if total <= eps:
        return np.ones_like(capped) / capped.size
    return capped / total


def compute_portfolio_returns(
    preds: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """
    Rebalance & Return 逻辑：
    1) 每月按预测值排序，Top 10% / Bottom 10% 划分 Long / Short bucket。
    2) 在各自 bucket 内做 z-score，并 clip 到 [-Z_SCORE_CLIP, Z_SCORE_CLIP]。
    3) Long 端使用 z>0，Short 端使用 z<0 的绝对值，得到非负权重并初步归一化。
    4) 调用 apply_weight_cap 控制单票权重不超过 MAX_WEIGHT（可通过 CLI 关闭）。
    """
    n = preds.size
    if n == 0:
        return {"long": np.nan, "short": np.nan, "long_short": np.nan}
    order = np.argsort(preds)
    k = max(1, int(np.floor(n * 0.1)))
    long_idx = order[-k:]
    short_idx = order[:k]

    def _bucket_return(idx: np.ndarray, positive_side: bool) -> float:
        if idx.size == 0:
            return float("nan")
        bucket_preds = preds[idx].astype(np.float64)
        bucket_targets = targets[idx].astype(np.float64)
        mean = float(np.mean(bucket_preds))
        std = float(np.std(bucket_preds))
        if not np.isfinite(std) or std < 1e-12:
            z = np.zeros_like(bucket_preds, dtype=np.float64)
        else:
            z = (bucket_preds - mean) / std
        z = np.clip(z, -Z_SCORE_CLIP, Z_SCORE_CLIP)
        weights = np.maximum(z, 0.0) if positive_side else np.maximum(-z, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.ones_like(weights, dtype=np.float64) / weights.size
        else:
            weights = weights / weight_sum
        weights = apply_weight_cap(weights, max_weight=MAX_WEIGHT)
        return float(np.sum(weights * bucket_targets))

    long_ret = _bucket_return(long_idx, positive_side=True)
    short_ret = _bucket_return(short_idx, positive_side=False)
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


def plot_curves(df: pd.DataFrame, loss_name: str, output_dir: Path) -> None:
    """
    画两类图：
    1. Loss 曲线（固定使用 MSE vs Month）。
    2. Long-Short Return 曲线。
    """
    months = pd.to_datetime(df["month"])
    metric_col = "mse"
    label = "MSE"

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


def summarize_results(loss_name: str, df_result: pd.DataFrame) -> Dict[str, float]:
    """
    统一汇总 schema，供单实验与批量对比脚本复用。
    """
    long_short_returns = df_result["long_short_return"].astype(float)
    ls_stats = compute_long_short_stats(long_short_returns)
    return {
        "loss": loss_name,
        "avg_mse": float(df_result["mse"].mean()),
        "avg_medse": float(df_result["medse"].mean()),
        "avg_r2": float(df_result["r2"].mean()),
        "avg_directional_accuracy": float(df_result["directional_accuracy"].mean()),
        "avg_sign_mismatch_large_y": float(df_result["sign_mismatch_large_y"].mean()),
        "avg_long_short": float(df_result["long_short_return"].mean()),
        "long_short_cumulative_return": ls_stats["cumulative_return"],
        "long_short_std": ls_stats["std"],
        "long_short_sharpe": ls_stats["sharpe"],
    }


def run_sanity_check(loss_name: str, args: argparse.Namespace) -> None:
    """
    固定训练窗口，只训练一次模型，然后在未来 6 个月逐月预测/调仓。
    """
    loss_name = loss_name.lower()
    if loss_name not in EXPERIMENT_LOSS_NAMES:
        raise ValueError(
            f"Unsupported loss '{loss_name}'. Supported losses: {', '.join(EXPERIMENT_LOSS_NAMES)}"
        )
    global MAX_WEIGHT
    MAX_WEIGHT = args.max_weight
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
        directional_metrics = compute_directional_metrics(y_month, preds)
        port = compute_portfolio_returns(preds, y_month)
        month_records.append(
            {
                "month": format_period(period),
                "sample_size": int(mask.sum()),
                "mse": metrics["mse"],
                "medse": metrics["medse"],
                "r2": metrics["r2"],
                "directional_accuracy": directional_metrics["directional_accuracy"],
                "sign_mismatch_large_y": directional_metrics["sign_mismatch_large_y"],
                "long_return": port["long"],
                "short_return": port["short"],
                "long_short_return": port["long_short"],
            }
        )
        print(
            f"Month {format_period(period)} | {loss_name.upper()} | "
            f"MSE {metrics['mse']:.6f} | MedSE {metrics['medse']:.6f} | "
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
    csv_path = Path(args.output_dir) / f"sanity_metrics_{loss_name}.csv"
    df_result.to_csv(csv_path, index=False)
    print(f"Saved metrics for {loss_name.upper()} to {csv_path}")

    summary = summarize_results(loss_name, df_result)
    summary_path = output_dir / f"sanity_summary_{loss_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(
        "Long-Short stats | Cumulative "
        f"{summary['long_short_cumulative_return']:.4f} | "
        f"Std {summary['long_short_std']:.4f} | Sharpe {summary['long_short_sharpe']:.4f}"
    )
    plot_curves(df_result, loss_name, output_dir)
    print(f"{loss_name.upper()} 实验完成。")
    print(
        "\n提示：批量比较 7 个 loss 时，请使用对应 runner 或 run_all_experiments.py。"
    )
