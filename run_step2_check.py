
import sys
import pandas as pd
from pathlib import Path

# Ensure we can import from the current directory
sys.path.append(".")

from Model_Train.data_preprocess import prepare_panel_data
from Model_Train.features import (
    FeatureConfig,
    build_feature_set_x1,
    build_feature_set_x2,
    build_feature_set_x3,
    assemble_feature_matrix
)

def main():
    print("=== Step 2: Data Loading & Feature Matrix Assembly Check ===")
    
    # 1. Load and Preprocess Data
    # Assuming CSV files are in the current directory based on file list provided earlier
    data_dir = "." 
    print(f"Loading CSVs from '{data_dir}'...")
    
    try:
        # Using default pattern "*.csv" which matches 04-09.csv, etc.
        panel = prepare_panel_data(data_dir=data_dir, pattern="*.csv")
        print(f"Successfully loaded panel data.")
        print(f"Panel Shape: {panel.shape}")
        print(f"Columns: {list(panel.columns)}")
        print(f"Sample Date Range: {panel['date'].min()} to {panel['date'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    config = FeatureConfig()
    
    # 2. Build and Assemble X1
    print("\n--- Processing Feature Set X1 ---")
    try:
        df_x1 = build_feature_set_x1(panel, config)
        print(f"X1 DataFrame Shape: {df_x1.shape}")
        
        x_x1, y_x1, ids_x1, dates_x1 = assemble_feature_matrix(df_x1)
        print(f"X1 Matrix (Features): {x_x1.shape}")
        print(f"Y1 Vector (Targets):  {y_x1.shape}")
        
        # Basic sanity check
        if x_x1.shape[0] == 0:
            print("WARNING: X1 resulting matrix is empty!")
    except Exception as e:
        print(f"Error processing X1: {e}")

    # 3. Build and Assemble X2
    print("\n--- Processing Feature Set X2 ---")
    try:
        df_x2 = build_feature_set_x2(panel, config)
        print(f"X2 DataFrame Shape: {df_x2.shape}")
        
        x_x2, y_x2, ids_x2, dates_x2 = assemble_feature_matrix(df_x2)
        print(f"X2 Matrix (Features): {x_x2.shape}")
        print(f"Y2 Vector (Targets):  {y_x2.shape}")
    except Exception as e:
        print(f"Error processing X2: {e}")

    # 4. Build and Assemble X3
    print("\n--- Processing Feature Set X3 ---")
    try:
        df_x3 = build_feature_set_x3(panel, config)
        print(f"X3 DataFrame Shape: {df_x3.shape}")
        
        x_x3, y_x3, ids_x3, dates_x3 = assemble_feature_matrix(df_x3)
        print(f"X3 Matrix (Features): {x_x3.shape}")
        print(f"Y3 Vector (Targets):  {y_x3.shape}")
    except Exception as e:
        print(f"Error processing X3: {e}")

    print("\n=== Check Complete ===")

if __name__ == "__main__":
    main()
