# FYP: Loss-Function Research for Stock Return Prediction

This repository contains the code, data, and research notes for a final-year project on cross-sectional stock return prediction with MLP models. The current mainline work compares seven losses under one fixed sanity-check pipeline and is being prepared for GitHub + Google Colab execution.

## Project Structure

```text
FYP/
├── Model_Train/                  # Core model, preprocessing, features, and loss functions
├── run_sanity_check_*.py         # Single-loss runners for the static sanity-check pipeline
├── run_all_experiments.py        # Batch entrypoint for multi-loss experiments
├── sanity_check_signal_tilted.py # Main static train-once / rebalance-monthly workflow
├── tests/                        # Pytest regression tests for losses and runners
├── *.csv                         # Historical stock data used directly by training/eval
├── G:MADL/                       # GMADL research notebooks, figures, supporting notes
├── doc/                          # Planning notes, diagrams, supporting PDFs, Colab notes
├── best_hyperparameters.txt      # Locked MLP baseline config
└── loss_function_design___to_do_v0.2.pdf
```

## Current Experiment Baseline

- Training window: `1990-01` to `1994-12`
- Test window: starts at `1995-01`, configurable via `--test-months`
- Features: `X1`
- Model: `MLP [64, 32, 16] + ReLU + Dropout 0.2`
- Portfolio rule: monthly rebalance, top 10% long / bottom 10% short
- Supported sanity-check losses: `mse`, `medse`, `gmadl`, `imadl`, `dirhuber`, `hybrid_add`, `hybrid_mul`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

Run a single sanity check:

```bash
python run_sanity_check_mse.py --test-months 6 --max-epochs 20
```

Run the batch comparison:

```bash
python run_all_experiments.py \
  --test-months 24 \
  --max-epochs 20 \
  --best-config-path best_hyperparameters.txt
```

Run a tuned hybrid loss with runtime lambda overrides:

```bash
python run_sanity_check_hybrid_add.py \
  --output-dir sanity_outputs/hybrid_add_a4 \
  --checkpoint-dir sanity_outputs/checkpoints/hybrid_add_a4 \
  --archive-root /content/drive/MyDrive/FYP/hybrid_add_a4 \
  --best-config-path best_hyperparameters.txt \
  --loss-kwargs '{"lambda_dir": 5.0, "lambda_hub": 0.1}' \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --resume-mode auto
```

Run the dedicated hybrid lambda sweep:

```bash
python run_hybrid_lambda_sweep.py \
  --preset minimal \
  --output-root /content/FYP/lambda_sweep \
  --checkpoint-root /content/drive/MyDrive/FYP/lambda_sweep/checkpoints \
  --archive-root /content/drive/MyDrive/FYP/lambda_sweep/archive \
  --best-config-path best_hyperparameters.txt \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --skip-existing \
  --resume-mode auto
```

## Colab Workflow

Clone the repo in Colab, install dependencies, then run either a smoke test or the batch runner:

```bash
git clone https://github.com/ROUCHER27/FYP.git
cd FYP
pip install -r requirements.txt
python run_all_experiments.py \
  --test-months 2 \
  --max-epochs 2 \
  --best-config-path best_hyperparameters.txt
```

Recommended Colab practice:

- Mount Google Drive if you want to persist outputs or checkpoints.
- Treat `sanity_outputs/` as disposable generated artifacts.
- Keep the root CSV files in the repo for reproducible clone-and-run experiments unless you later move them to Drive or another data store.

## What Is Intentionally Not Committed

The repository now ignores local-only noise so the GitHub repo stays usable from Colab:

- virtual environments and caches
- Obsidian / SpecStory / editor workspace metadata
- generated `sanity_outputs/`
- screenshots and temporary local captures
- draft export PDFs not used by the experiment workflow

## Key References

- `loss_function_design___to_do_v0.2.pdf`
- `doc/plan_semester2.pdf`
- `doc/Next-step-outline.md`
- `doc/Experiment-diagrams.md`
- `G:MADL/GMADL_improve.md`
