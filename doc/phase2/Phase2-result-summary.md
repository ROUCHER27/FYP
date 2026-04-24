# Phase 1 Factual Summary

## Confirmed successful 24-month runs
- gmadl
- imadl
- dirhuber
- hybrid_add
- hybrid_mul

## Confirmed failed baseline runs
- mse
- medse

## Failure cause
- The first Colab batch pointed `--best-config-path` to `/content/drive/MyDrive/FYP/code/best_hyperparameters.txt`, which did not exist at runtime.

## Locked model config
- input_dim = 15
- hidden_dims = [64, 32, 16]
- activation = tanh
- dropout = 0.0

## Phase 1 directional-loss ranking by logged 24-month Sharpe
1. imadl
2. gmadl
3. hybrid_mul
4. dirhuber
5. hybrid_add
