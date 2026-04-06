import argparse
import ast
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from Model_Train.data_preprocess import prepare_panel_data
from Model_Train.features import (
    FeatureConfig,
    assemble_feature_matrix,
    build_feature_set_x1,
)
from Model_Train.models import MLPConfig
from Model_Train.train_rolling import (
    RollingConfig,
    build_window_dataloaders,
    generate_time_windows,
    get_loss_fn,
    predict_window,
    train_one_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Rolling-window training with multiple loss functions."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input CSV files.",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        help="Lookback window for Feature Set X1.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=5,
        help="Number of years in each training window.",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=6,
        help="Number of months per evaluation window.",
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=6,
        help="Step size (in months) between rolling windows.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Epochs per rolling window training run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for rolling window training.",
    )
    parser.add_argument(
        "--best-config-path",
        type=str,
        default="best_hyperparameters.txt",
        help="Text file storing the best configuration from Step 3.",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="mse,medse,madl,gmadl",
        help="Comma-separated loss names (supported: mse, medse, madl, gmadl).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="rolling_predictions.csv",
        help="Where to store concatenated monthly predictions.",
    )
    return parser.parse_args()


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA acceleration.")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) acceleration.")
        return torch.device("mps")
    return torch.device("cpu")


def read_best_config(path: Path, input_dim: int) -> MLPConfig:
    """
    Load the MLP architecture discovered during Step 3 grid search.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Best config file not found at {path}. Run step 3 or supply --best-config-path."
        )
    config_dict = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("Config:"):
            payload = line.split("Config:", 1)[1].strip()
            config_dict = ast.literal_eval(payload)
            break
    if config_dict is None:
        raise ValueError(f"File {path} does not contain a 'Config:' line.")
    hidden_dims = config_dict.get("hidden_dims")
    if hidden_dims is None:
        raise ValueError(f"Config in {path} missing 'hidden_dims'.")
    activation = config_dict.get("activation", "relu")
    dropout = config_dict.get("dropout", 0.0)
    cfg_input_dim = config_dict.get("input_dim", input_dim)
    if cfg_input_dim != input_dim:
        print(
            f"Warning: best config input_dim={cfg_input_dim} differs from current feature dim {input_dim}. Overriding to {input_dim}."
        )
    return MLPConfig(
        input_dim=input_dim,
        hidden_dims=list(hidden_dims),
        activation=activation,
        dropout=float(dropout),
    )


def parse_losses(loss_str: str) -> List[str]:
    return [token.strip().lower() for token in loss_str.split(",") if token.strip()]


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_masks(
    date_array: np.ndarray,
    window: Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
) -> Tuple[np.ndarray, np.ndarray]:
    train_start, train_end, test_start, test_end = window
    train_mask = (date_array >= train_start) & (date_array <= train_end)
    test_mask = (date_array >= test_start) & (date_array <= test_end)
    return train_mask, test_mask


def main() -> None:
    args = parse_args()
    device = detect_device()

    print("=== Step 4: Rolling-window Training ===")
    print(f"Loading data from {args.data_dir} (pattern={args.pattern})")
    panel = prepare_panel_data(data_dir=args.data_dir, pattern=args.pattern)
    print(f"Loaded {len(panel)} rows after preprocessing.")

    print("\nBuilding Feature Set X1...")
    feature_cfg = FeatureConfig(lookback_months=args.lookback_months)
    df_x1 = build_feature_set_x1(panel, feature_cfg)
    x_all, y_all, ids, dates = assemble_feature_matrix(df_x1)
    if x_all.size == 0:
        raise RuntimeError("Feature matrix is empty; cannot proceed with rolling training.")
    id_array = ids.to_numpy()
    date_series = pd.to_datetime(dates)
    date_array = date_series.to_numpy()

    print(f"Feature matrix shape: {x_all.shape}")
    best_config = read_best_config(Path(args.best_config_path), input_dim=x_all.shape[1])
    print("Using MLPConfig:", best_config)

    rolling_cfg = RollingConfig(
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
    )
    windows = generate_time_windows(date_series, rolling_cfg)
    if not windows:
        raise RuntimeError("No rolling windows were generated. Check date coverage or config.")
    print(f"Generated {len(windows)} rolling windows (step={args.step_months} months).")

    losses = parse_losses(args.losses)
    if not losses:
        raise ValueError("No loss functions specified via --losses.")
    print(f"Evaluating losses: {', '.join(losses)}")

    results: List[pd.DataFrame] = []
    for window_idx, window in enumerate(windows, 1):
        train_mask, test_mask = build_masks(date_array, window)
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())
        if train_count == 0 or test_count == 0:
            print(
                f"[Window {window_idx}] Skipping due to empty split (train={train_count}, test={test_count})."
            )
            continue
        print(
            f"[Window {window_idx}] Train {window[0].date()}–{window[1].date()} "
            f"({train_count} rows) | Test {window[2].date()}–{window[3].date()} "
            f"({test_count} rows)"
        )
        train_loader, test_loader = build_window_dataloaders(
            x_all, y_all, date_series, window, batch_size=args.batch_size, device=device
        )
        window_targets = y_all[test_mask]
        window_ids = id_array[test_mask]
        window_dates = date_array[test_mask]
        for loss_name in losses:
            loss_fn = get_loss_fn(loss_name)
            print(f"   - Training with {loss_name.upper()} loss...")
            model = train_one_window(
                best_config,
                train_loader,
                loss_fn,
                device=device,
                max_epochs=args.max_epochs,
            )
            preds = predict_window(model, test_loader, device=device).reshape(-1)
            if preds.shape[0] != test_count:
                raise RuntimeError(
                    f"Prediction count mismatch ({preds.shape[0]} vs {test_count}) "
                    f"for window {window_idx} and loss {loss_name}."
                )
            block = pd.DataFrame(
                {
                    "PERMNO": window_ids,
                    "date": window_dates,
                    "loss_name": loss_name.upper(),
                    "prediction": preds.astype(np.float32),
                    "target_ret": window_targets,
                    "window_index": window_idx,
                    "train_start": window[0],
                    "train_end": window[1],
                    "test_start": window[2],
                    "test_end": window[3],
                }
            )
            results.append(block)

    if not results:
        raise RuntimeError("No predictions were generated across all windows.")

    output_path = Path(args.output_path)
    ensure_output_dir(output_path)
    predictions_df = pd.concat(results, ignore_index=True)
    predictions_df.sort_values(["date", "PERMNO", "loss_name"], inplace=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"\nSaved rolling predictions to {output_path.resolve()}")

    summary = {
        "num_windows": len(windows),
        "losses": losses,
        "train_years": args.train_years,
        "test_months": args.test_months,
        "step_months": args.step_months,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
    }
    summary_path = output_path.with_suffix(".meta.json")
    summary_path.write_text(json.dumps(summary, default=str, indent=2))
    print(f"Wrote metadata to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
