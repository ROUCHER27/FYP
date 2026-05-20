import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from typing import Tuple
import time

# Ensure we can import from the current directory
sys.path.append(".")

from Model_Train.data_preprocess import prepare_panel_data
from Model_Train.features import (
    FeatureConfig,
    build_feature_set_x1,
    assemble_feature_matrix
)
# Import necessary classes from your module
from Model_Train.models import MLP, MLPConfig
from Model_Train.train_grid_search import build_dataloader, train_one_epoch, evaluate_mse, config_to_dict

def grid_search_with_progress(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    param_grid: dict,
    input_dim: int,
    device: torch.device,
    max_epochs: int = 20,
):
    """
    Custom grid search function with progress printing.
    """
    # 1. Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    total_configs = len(combinations)
    
    print(f"\n--- Starting Grid Search ---")
    print(f"Total configurations to test: {total_configs}")
    print(f"Max Epochs per config: {max_epochs}")
    print("-" * 60)

    best_config = None
    best_score = float("inf")
    
    start_time = time.time()

    for idx, combo in enumerate(combinations, 1):
        # Unpack combination back into a dictionary
        params = dict(zip(keys, combo))
        
        # Extract specific args
        hidden_dims = list(params.get("hidden_dims", [64, 64]))
        activation = params.get("activation", "relu")
        dropout = float(params.get("dropout", 0.0))
        lr = float(params.get("lr", 1e-3))
        batch_size = int(params.get("batch_size", 512))
        
        # Build Config
        config = MLPConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        # Prepare DataLoaders
        train_loader = build_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = build_dataloader(x_val, y_val, batch_size=batch_size, shuffle=False)
        
        # Init Model
        model = MLP(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train
        for _ in range(max_epochs):
            train_one_epoch(model, train_loader, optimizer, device)
            
        # Evaluate
        val_mse = evaluate_mse(model, val_loader, device)
        
        # Print Progress
        # Format the config for display (shorten lists for readability)
        config_str = f"Dims:{hidden_dims}, Act:{activation}, Drp:{dropout}, LR:{lr}, BS:{batch_size}"
        print(f"[{idx}/{total_configs}] MSE: {val_mse:.6f} | {config_str}")
        
        # Update Best
        if val_mse < best_score:
            best_score = val_mse
            best_config = config
            print(f"   >>> New Best Found! (Previous: {best_score:.6f})")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Grid Search Finished in {elapsed:.1f} seconds.")
    
    return best_config, best_score

def main():
    print("=== Step 3: Hyperparameter Grid Search (X1 on 89.12-94.csv) ===")
    
    # 1. Load Data
    target_pattern = "89.12-94.csv"
    data_dir = "."
    print(f"Loading data from '{target_pattern}'...")
    
    try:
        panel = prepare_panel_data(data_dir=data_dir, pattern=target_pattern)
        print(f"Loaded {len(panel)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Build Feature Set X1
    print("\nBuilding Feature Set X1...")
    config = FeatureConfig(lookback_months=12)
    try:
        df_x1 = build_feature_set_x1(panel, config)
        x_all, y_all, ids, dates = assemble_feature_matrix(df_x1)
        print(f"Feature Matrix Shape: {x_all.shape}")
        
        if x_all.shape[0] == 0:
            print("Error: No data remains after feature construction.")
            return

    except Exception as e:
        print(f"Error building features: {e}")
        return

    # 3. Split Data with validation to avoid empty partitions
    date_values = dates.to_numpy()
    unique_dates = np.unique(date_values)
    total_rows = x_all.shape[0]

    def build_row_split() -> Tuple[np.ndarray, np.ndarray]:
        """Fallback split that keeps chronological order while ensuring both sides have data."""
        if total_rows < 2:
            raise ValueError("Need at least two samples to create train/validation splits.")
        split_at = max(1, min(int(total_rows * 0.8), total_rows - 1))
        order = np.argsort(date_values)
        mask = np.zeros(total_rows, dtype=bool)
        mask[order[:split_at]] = True
        return mask, ~mask

    train_mask: np.ndarray
    val_mask: np.ndarray
    used_date_split = False

    if len(unique_dates) >= 2:
        split_idx = int(len(unique_dates) * 0.8)
        split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
        split_date = unique_dates[split_idx]
        candidate_train = date_values < split_date
        candidate_val = ~candidate_train
        if candidate_train.sum() > 0 and candidate_val.sum() > 0:
            train_mask = candidate_train
            val_mask = candidate_val
            used_date_split = True
            print(f"Splitting at {split_date} (approx 80/20 split)")
        else:
            print("Warning: Date-based split resulted in an empty partition. Falling back to row-level split.")
            train_mask, val_mask = build_row_split()
    else:
        print("Warning: Not enough unique dates for date-based split. Using row-level split instead.")
        train_mask, val_mask = build_row_split()

    if not used_date_split:
        print("Splitting with chronological row order (approx 80/20 split)")

    train_count = int(train_mask.sum())
    val_count = int(val_mask.sum())
    if train_count == 0 or val_count == 0:
        raise ValueError("Failed to create non-empty train/validation splits.")
    print(f"Train samples: {train_count}, Validation samples: {val_count}")

    x_train = torch.from_numpy(x_all[train_mask]).float()
    y_train = torch.from_numpy(y_all[train_mask]).float()
    x_val = torch.from_numpy(x_all[val_mask]).float()
    y_val = torch.from_numpy(y_all[val_mask]).float()

    # 4. Define Parameter Grid
    param_grid = {
        "hidden_dims": [[32, 16], [64, 32], [128, 64], [64, 32, 16]],
        "activation": ["relu", "tanh"],
        "dropout": [0.0, 0.2],
        "lr": [1e-3, 5e-4],
        "batch_size": [512, 1024]
    }
    
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) acceleration.")
    
    # 5. Run Search
    try:
        best_config, best_mse = grid_search_with_progress(
            x_train, y_train, x_val, y_val,
            param_grid=param_grid,
            input_dim=x_train.shape[1],
            device=device,
            max_epochs=20  # Restored to 20 as requested
        )
        
        print("\n=== Result ===")
        print(f"Best Validation MSE: {best_mse:.6f}")
        print("Best Configuration:")
        print(config_to_dict(best_config))
        
        with open("best_hyperparameters.txt", "w") as f:
            f.write(f"Best MSE: {best_mse}\n")
            f.write(f"Config: {config_to_dict(best_config)}\n")
        print("Saved to 'best_hyperparameters.txt'")
            
    except Exception as e:
        print(f"Error during grid search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
