# Appendix B: Per-Seed Raw Results and Reproducibility

This appendix tabulates the per-seed raw Sharpes that the multi-seed grouped summaries in Chapter 5 aggregate, so that the reader can verify the mean, standard deviation, and coefficient of variation directly. It also lists the exact reproduction commands for every table in Chapter 5.

## B.1 Phase 2.2 γ refinement — per-seed annualised Sharpe

Source: `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv`. Static train `1990-01..1994-12`, test `1995-01..1996-12`, `cap05` portfolio cap. Three seeds per γ; seeds are `{42, 52, 62}`.

**Table B.1 — Phase 2.2 γ refinement per-seed annualised Sharpe.**

| Loss | seed 42 | seed 52 | seed 62 | mean | std | CV |
|---|---:|---:|---:|---:|---:|---:|
| `m2_robust_gamma03` | $-0.01987$ | $0.32631$ | $0.66380$ | $0.32341$ | $0.34184$ | $1.05698$ |
| `m2_robust_gamma05` | $0.86960$ | $0.66697$ | $0.57964$ | $0.70540$ | $0.14875$ | $0.21087$ |
| `m2_robust_gamma07` | $0.90972$ | $1.08404$ | $0.75316$ | $0.91564$ | $0.16552$ | $0.18077$ |
| `m2_robust_gamma10` | $0.45873$ | $1.58465$ | $0.96958$ | $1.00432$ | $0.56376$ | $0.56134$ |
| `m2_robust_gamma15` | $0.90214$ | $0.40845$ | $1.13823$ | $0.81627$ | $0.37239$ | $0.45621$ |

(The seed-42 `m2_robust_gamma07` row here, Sharpe $0.90972$, differs from the rounded $0.9156$ grouped-summary mean because the grouped row averages the three per-seed values. Both numbers trace to the raw CSV.)

## B.2 Phase 2.2-fix1 normalisation probe — per-seed annualised Sharpe

Source: `doc/phase2-fix/phase2.2-fix1/phase2_summary.json`.

**Table B.2 — Phase 2.2-fix1 normalisation probe per-seed Sharpe.**

| Loss | Normalised seed 1 | Normalised seed 2 | Normalised seed 3 | Normalised mean | Original mean |
|---|---:|---:|---:|---:|---:|
| `m2_robust_gamma07` | $0.5956$ | $1.4064$ | $0.7317$ | $0.9112$ | $0.9156$ |
| `m2_robust_gamma10` | $0.6254$ | $0.1181$ | $0.4780$ | $0.4072$ | $1.0043$ |
| `imadl_m2_alpha06` | $0.5628$ | $-0.8335$ | $0.2224$ | $-0.0161$ | $0.6895$ |

(The per-seed IDs are anonymised in the summary JSON. The project notes record that the normalisation probe re-uses the same three seeds as the γ refinement, but the mapping between the numeric position and `{42, 52, 62}` is not preserved in the summary file; the mean and per-seed spread remain the relevant quantities.)

## B.3 Phase 2 integrated summary — grouped row summary (reproduction)

Source (read without switching branches): `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`.

Grouped rows quoted in Chapter 5 Table 5.4 are read directly from this CSV (column-by-column, no averaging or post-processing). Per-seed raw rows for the same variants are available in the sibling `phase2_raw_runs.csv` on the same branch. Per-seed detail is elided from the main text for readability, but every seed is recoverable via:

```bash
git show phase2.2-fix:doc/phase2-fix/reports/phase2_raw_runs.csv
```

## B.4 Reproduction commands per Chapter 5 table

Tables 5.1–5.2 reproduce on `main` at commit `6c0fbde` with the repository-root data CSVs (`*.csv` matching the `--pattern`) and the shared `best_hyperparameters.txt`. Tables 5.3–5.5 are read or reproduced from the `phase2.2-fix` branch and `doc/phase2-fix/phase2.2-fix1/` artifacts as listed below.

### Table 5.1 — Baseline losses (seed 42, 24 months)

For each `{loss} in {mse, medse, madl, gmadl, imadl, hybrid_mul_m1, hybrid_mul_m2}`:

```bash
python run_sanity_check_{loss}.py \
  --data-dir . \
  --pattern '*.csv' \
  --train-start 1990-01 \
  --train-end 1994-12 \
  --test-start 1995-01 \
  --test-months 24 \
  --max-epochs 20 \
  --batch-size 1024 \
  --best-config-path best_hyperparameters.txt \
  --output-dir <output-root>/results/baseline/{loss} \
  --seed 42
```

The output CSV at `<output-root>/results/baseline/{loss}/sanity_metrics_{loss}.csv` must have `row_count = 24`, `first_month = 1995-01`, `last_month = 1996-12`. The paired summary JSON is at `sanity_summary_{loss}.json`.

### Table 5.2 — Phase 1.5 A/M variants (seed 42, 24 months)

For each variant (`A1..A5`, `M1..M4`) use the corresponding runner, e.g., for `M2`:

```bash
python run_sanity_check_hybrid_mul_m2.py \
  --data-dir . --pattern '*.csv' \
  --train-start 1990-01 --train-end 1994-12 \
  --test-start 1995-01 --test-months 24 \
  --max-epochs 20 --batch-size 1024 \
  --best-config-path best_hyperparameters.txt \
  --output-dir <output-root>/results/phase15/M2 \
  --seed 42
```

The runner-to-variant mapping is listed in `doc/final_report_all_24m_evidence/manifests/run_manifest.json`.

### Table 5.3 — Phase 2.2 γ refinement (multi-seed)

The γ refinement was run from the `phase2.2-fix` branch with three seeds and the five γ values listed in §B.1. The grouped summary is reproduced locally at:

```bash
doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv
doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv
```

The per-seed raw CSV can be aggregated into the grouped summary by:

```python
import pandas as pd
raw = pd.read_csv("doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv")
grouped = (
    raw[raw["cap_tag"] == "cap05"]
    .groupby("loss")["long_short_sharpe"]
    .agg(["mean", "std", "min", "max", "count"])
)
grouped["cv"] = grouped["std"] / grouped["mean"].abs()
```

### Tables 5.4 and 5.5 — Phase 2 integrated and normalisation

The integrated grouped summary is on branch `phase2.2-fix`:

```bash
git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv
```

The normalisation probe summaries are on `main` at:

```
doc/phase2-fix/phase2.2-fix1/phase1_summary.json
doc/phase2-fix/phase2.2-fix1/phase2_summary.json
```

## B.5 Reproducibility checklist

Before re-running any table for publication, verify:

1. Working directory is the repository root; `*.csv` data files are present.
2. `best_hyperparameters.txt` is unchanged from commit `6c0fbde` (MLP[64, 32, 16], ReLU, dropout 0.2, input_dim 15).
3. Python version ≥ 3.10; PyTorch installed; NumPy/pandas/matplotlib available for figure scripts.
4. CUDA or MPS accelerator is available if matching Phase 2 run behaviour (device auto-detection in `sanity_check_signal_tilted.detect_device`). CPU-only runs are numerically close but not bit-identical.
5. The runner CLI arguments match exactly (`--test-months 24`, `--train-start 1990-01`, `--train-end 1994-12`, `--test-start 1995-01`, `--seed 42` for single-seed tables).
6. After each run, verify the output CSV has 24 rows with first month `1995-01` and last month `1996-12`; this is automated by the `*_verification.json` writer in the evidence scripts.

## B.6 Known numerical caveats

- CUDA non-determinism. `torch.backends.cudnn.deterministic` is not forced; bit-level reproduction across hardware is not guaranteed. Grouped-summary values in Chapter 5 are reported to be stable across re-runs within the same environment, but no formal rerun artifact has been preserved.
- Floating-point noise in `compute_portfolio_returns`. The capped-simplex projection iterates up to 10 times; edge cases (multiple weights hitting the cap simultaneously) can produce microscopic differences across runs.
- `git show phase2.2-fix:...` depends on the `phase2.2-fix` branch being present in the local clone. `git fetch` if needed.
- The 6-month legacy window `1995-01..1995-06` in `sanity_outputs/` is not part of this appendix's reproduction and should not be used for any final-report table.

## B.7 Complete model input columns

**Table B.3 — Complete 15-column model input for X1.**

| # | Column | Description |
|---|---|---|
| 1 | RET | raw monthly total return for month t |
| 2 | VOL | monthly trading volume (shares) for month t |
| 3 | SHROUT | shares outstanding (thousands) for month t |
| 4 | r | numeric copy of RET |
| 5 | to | monthly turnover = VOL/(SHROUT×1000) |
| 6 | cr_1m | 1-month cumulative lagged return |
| 7 | co_1m | 1-month cumulative lagged turnover |
| 8 | cr_3m | 3-month cumulative lagged return |
| 9 | co_3m | 3-month cumulative lagged turnover |
| 10 | cr_6m | 6-month cumulative lagged return |
| 11 | co_6m | 6-month cumulative lagged turnover |
| 12 | cr_9m | 9-month cumulative lagged return |
| 13 | co_9m | 9-month cumulative lagged turnover |
| 14 | cr_12m | 12-month cumulative lagged return |
| 15 | co_12m | 12-month cumulative lagged turnover |