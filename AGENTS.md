# Repository Guidelines

## Project Structure & Module Organization
- Root CSV files (e.g., `04-09.csv`) are the main input data for model training and evaluation.
- `G:MADL/` contains GMADL research notebooks, figures, and notes (primarily Phase 1 analysis and visualization).
- `Model_Train/` is reserved for Python training/evaluation scripts and experiment configuration files.
- `loss_function_design_empirical___to_do.pdf` describes the assignment phases; keep new code and reports aligned with this document.

## Build, Test, and Development Commands
- Use Python 3.10+ in a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies as needed (typical: `pip install numpy pandas matplotlib seaborn jupyter`).
- Run notebooks in `G:MADL/` via `jupyter lab` or `jupyter notebook`, executing cells top-down.
- Run training scripts from the repo root, e.g., `python Model_Train/train_gmadl.py --config configs/base.yaml` (keep CLI flags short and explicit).

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and UTF-8 encoding; comments may be in Chinese or English.
- Use `snake_case` for variables/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Name notebooks descriptively by task (e.g., `GMADL_Improvement_Test.ipynb`) and avoid spaces in filenames.
- When available, format Python code with `black` and lint with `ruff` or `flake8` before committing.

## Testing Guidelines
- Place tests under a `tests/` directory (create it if missing), mirroring the structure of `Model_Train/`.
- Name test files `test_*.py` and test functions `test_*`.
- Use `pytest` as the default test runner: `pytest -q`.
- For new loss functions, add numeric sanity tests (direction correctness, symmetry, extreme values, and gradient behaviour).

## Commit & Pull Request Guidelines
- Write concise commit messages in the form `area: summary`, e.g., `loss: add gmadl heatmap notebook`.
- Keep each commit focused on a single logical change.
- Pull requests should describe purpose, key changes, how to run code/tests, and reference related assignment phases or issues.
- Attach or link key figures (heatmaps, comparison plots) when they are central to the change.

