# Results: Source of Truth

All numbers allowed to appear in the final report are listed here with provenance. Every chapter must cite from this file; if a new number is needed, add it here first with its source artifact path and re-verify.

## 0. Shared Experimental Configuration (single-seed tables)

- Branch / commit: `main` @ `6c0fbde558b9d79208e9f21c438753be6f7bfa31`.
- Train window: `1990-01` to `1994-12` (60 monthly cross-sections).
- Main test window: `1995-01` to `1996-12` (24 monthly cross-sections, verified row count = 24).
- Seed: `42` (single-seed for baseline and Phase 1.5 tables below).
- Model: `MLPConfig(input_dim=15, hidden_dims=[64, 32, 16], activation='relu', dropout=0.2)`.
- Feature set: X1 contributes 10 engineered columns (cumulative returns + cumulative turnover at 5 horizons); the 15-column model input also includes `RET`, `VOL`, `SHROUT`, `r`, and `to` (see Appendix B Table B.3).
- Training: `max_epochs=20`, `batch_size=1024`, no retraining during test period.
- Portfolio: cross-sectional top 10% long / bottom 10% short, signal-tilted weights (z within bucket clipped to ±3), capped-simplex projection at 5% per stock.
- Evaluation metrics: monthly long-short return, cumulative return over 24 months, annualised Sharpe computed as $\text{Sharpe} = \sqrt{12} \cdot \bar r / \sigma_r$ with $\bar r, \sigma_r$ the sample mean and standard deviation of the monthly long-short portfolio return series (`compute_long_short_stats` in `sanity_check_signal_tilted.py` uses `periods_per_year=12`), and average R² across months.
- Training source: `best_hyperparameters.txt`; runner scripts under repo root `run_sanity_check_*.py`.

## 1. Baseline Losses — 24-month, seed 42

Source: `doc/final_report_all_24m_evidence/results/baseline/{loss}/sanity_summary_{loss}.json`; status in `doc/final_report_all_24m_evidence/reports/final_report_all_24m_evidence_status.csv`; commands in `doc/final_report_all_24m_evidence/manifests/baseline_{loss}_command.txt`; verification rows = 24 in every `*_verification.json`.

| Loss | Sharpe | Cumulative return | Avg R² | Avg monthly LS return | Monthly LS std |
|---|---:|---:|---:|---:|---:|
| MSE | `-0.4643` | `-0.1125` | `-102.1787` | `-0.004422` | `0.032989` |
| MedSE | `0.0932` | `0.0060` | `-2297042.2865` | `0.001214` | `0.045124` |
| MADL | `-0.3058` | `-0.0756` | `-4145108192.00` | `-0.002794` | `0.031653` |
| GMADL | `0.2025` | `0.0279` | `-7016727381.33` | `0.001429` | `0.024449` |
| IMADL | `-0.3732` | `-0.0944` | `-106.5114` | `-0.003578` | `0.033211` |
| hybrid_mul_m1 | `0.4435` | `0.0509` | `-4.7942` | `0.002215` | `0.017302` |
| hybrid_mul_m2 | `-0.0017` | `-0.0032` | `-1.0298` | `-0.000008` | `0.016096` |

Notes:
- Extremely negative R² for MedSE/MADL/GMADL indicates point-prediction scale blow-up; portfolio signal comes from cross-sectional ranking, not from calibrated predictions. Must be discussed explicitly.
- The old log heading `Static 5Y Train / 6M Test` is stale; trust the manifest / verification JSONs (24 rows).

## 2. Phase 1.5 Variants (additive A1–A5, multiplicative M1–M4) — 24-month, seed 42

Source: `doc/final_report_all_24m_evidence/results/phase15/{id}/sanity_summary_*.json`.

| Variant | Loss ID | Sharpe | Cumulative return | Avg R² |
|---|---|---:|---:|---:|
| A1 | `hybrid_add_a1` | `0.1241` | `0.0133` | `-2105.9452` |
| A2 | `hybrid_add_a2` | `0.2173` | `0.0450` | `-484.7536` |
| A3 | `hybrid_add_a3` | `0.5738` | `0.0813` | `-1383.6426` |
| A4 | `hybrid_add_a4` | `0.2311` | `0.0463` | `-11229.7887` |
| A5 | `hybrid_add_a5` | `-0.4110` | `-0.2080` | `-507162.3841` |
| M1 | `hybrid_mul_m1` | `0.4435` | `0.0509` | `-4.7942` |
| M2 | `hybrid_mul_m2` | `-0.0017` | `-0.0032` | `-1.0298` |
| M3 | `hybrid_mul_m3` | `-0.9691` | `-0.0903` | `-0.4353` |
| M4 | `hybrid_mul_m4` | `-0.3440` | `-0.0409` | `-0.0563` |

Notes:
- A3 is the best additive variant at seed 42; M1 is the best multiplicative variant at seed 42.
- This is same-window single-seed evidence only. Do not label as robustness evidence.
- M1 and M2 are identical in the Baseline and Phase 1.5 tables (same runner and config).

## 3. Phase 2.2 Gamma Refinement — 24-month, multi-seed (3 seeds per row)

Source: `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`. Each row aggregates 3 runs with portfolio cap = 5% (`cap05`).

| Loss | Runs | Sharpe mean | Sharpe std | Sharpe min / max | Cum. return mean | CV |
|---|---:|---:|---:|---:|---:|---:|
| `m2_robust_gamma03` | 3 | `0.3234` | `0.3418` | `-0.0199` / `0.6638` | `0.0818` | `1.0570` |
| `m2_robust_gamma05` | 3 | `0.7054` | `0.1488` | `0.5796` / `0.8696` | `0.2392` | `0.2109` |
| `m2_robust_gamma07` | 3 | `0.9156` | `0.1655` | `0.7532` / `1.0840` | `0.2799` | `0.1808` |
| `m2_robust_gamma10` | 3 | `1.0043` | `0.5638` | `0.4587` / `1.5847` | `0.2368` | `0.5613` |
| `m2_robust_gamma15` | 3 | `0.8163` | `0.3724` | `0.4085` / `1.1382` | `0.2277` | `0.4562` |

Headline interpretation:
- `gamma10` has the highest mean Sharpe but materially higher seed sensitivity (CV `0.5613`).
- `gamma07` is the primary recommendation: strong mean Sharpe `0.9156` with the lowest CV `0.1808` and cumulative return `0.2799`.
- `gamma15` and `gamma03` degrade on both Sharpe and CV.

## 4. Phase 2 Integrated Grouped Summary (on branch `phase2.2-fix`) — 24-month, multi-seed

Source (read without switching branches): `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`.

Selected rows (cap05, 3 runs each):

| Loss | Sharpe mean | Sharpe std | Cum. return mean | CV |
|---|---:|---:|---:|---:|
| `adaptive_lambda10` | `0.4938` | `0.7617` | `0.0618` | `1.5426` |
| `adaptive_lambda50` | `0.2763` | `0.1597` | `0.0817` | `0.5780` |
| `adaptive_lambda100` | `0.0955` | `0.0343` | `0.0021` | `0.3591` |
| `imadl_gmadl_beta03` | `-0.0345` | `0.1381` | `-0.0424` | `4.0084` |
| `imadl_gmadl_beta05` | `0.0406` | `0.4116` | `-0.0138` | `10.1328` |
| `imadl_gmadl_beta07` | `-0.0020` | `0.2747` | `-0.0409` | `139.5122` |
| `imadl_m2_alpha02` | `0.1788` | `1.1576` | `0.2225` | `6.4735` |
| `imadl_m2_alpha03` | `0.2159` | `0.2010` | `0.0588` | `0.9310` |
| `imadl_m2_alpha04` | `0.3540` | `0.0656` | `0.0962` | `0.1853` |
| `imadl_m2_alpha05` | `0.5822` | `0.3193` | `0.2465` | `0.5484` |
| `imadl_m2_alpha06` | `0.6895` | `0.1685` | `0.3042` | `0.2443` |
| `imadl_m2_alpha07` | `0.4024` | `0.2466` | `0.1036` | `0.6128` |
| `imadl_m2_alpha08` | `0.5683` | `0.4130` | `0.2071` | `0.7267` |
| `m2_robust_gamma001` | `0.6919` | `0.8258` | `0.1705` | `1.1936` |
| `m2_robust_gamma01` | `0.7470` | `0.3937` | `0.2718` | `0.5270` |
| `m2_robust_gamma10` | `1.0043` | `0.5638` | `0.2368` | `0.5613` |

Headline interpretation:
- IMADL-m2 α sweep peaks at `alpha06` with mean Sharpe `0.6895` and CV `0.2443`, the recommended stable fallback.
- `imadl_gmadl_beta{03,05,07}` are unstable; do not claim dominance.
- `adaptive_lambda*` variants under-perform Sharpe and are highly seed-sensitive.

## 5. Phase 2.2-fix1 Normalisation Probe (loss-component scaling)

Source: `doc/phase2-fix/phase2.2-fix1/phase1_summary.json` (scale ratios) and `doc/phase2-fix/phase2.2-fix1/phase2_summary.json` (normalised-vs-original Sharpes).

Scale ratios (directional vs MSE component, from diagnostics):
- `m2_robust_gamma07`: ratio ≈ 113.
- `m2_robust_gamma10`: ratio ≈ 113.
- `imadl_m2_alpha06`: ratio ≈ 34.

Normalised-vs-original Sharpe (per-seed Sharpes then averaged):

| Loss | Original mean Sharpe | Normalised mean Sharpe | Normalised per-seed Sharpes |
|---|---:|---:|---|
| `m2_robust_gamma07` | `0.9156` | `0.9112` | `0.5956, 1.4064, 0.7317` |
| `m2_robust_gamma10` | `1.0043` | `0.4072` | `0.6254, 0.1181, 0.4780` |
| `imadl_m2_alpha06` | `0.6895` | `-0.0161` | `0.5628, -0.8335, 0.2224` |

Headline interpretation:
- Component-level normalisation is NOT a universal fix.
- `gamma07` is approximately flat under normalisation.
- `gamma10` and `alpha06` degrade materially.
- Note: the diagnostics acknowledge that full per-component logging is not yet implemented and ratios are estimated. State this caveat in prose.

## 6. Final Headline Claims

Strong claims supported by evidence above:

1. Among all tested loss variants, the family of M2-robust gamma losses is the best-supported family under the reported multi-seed Sharpe/stability evidence.
2. `m2_robust_gamma07` is the primary recommended loss: mean Sharpe `0.9156`, CV `0.1808`, mean cumulative return `0.2799` over 24 months across 3 seeds.
3. `m2_robust_gamma10` is a higher-return alternative (mean Sharpe `1.0043`) but is markedly less stable (CV `0.5613`).
4. `imadl_m2_alpha06` is a robust fallback (mean Sharpe `0.6895`, CV `0.2443`, cumulative return `0.3042`).
5. Loss-component normalisation does not uniformly improve performance; it leaves gamma07 roughly unchanged and degrades gamma10 and alpha06.
6. Single-seed baseline comparison at seed 42 shows that traditional MSE/MedSE, MADL, and GMADL each fail to produce economically meaningful positive Sharpe in the seed-42 same-window baseline, while hybrid multiplicative variant M1 and additive variant A3 yield the best same-window single-seed Sharpes among baseline-style losses.

## 7. Explicitly Forbidden Claims (consistency sweep targets)

- MedSE Sharpe `2.68`, MSE Sharpe `0.37` as 24-month headlines.
- `gamma07 CV = 0.0356`, `gamma10 CV = 0.1151`.
- "Normalisation failed across all losses".
- "Phase 2 directly replicates Phase 1.5" / "exact replication".
- Treating `doc/final_report_all_24m_evidence/` as multi-seed robustness.
- Architecture: LSTM, batch size 256, 50 epochs, early stopping, 27 features, tanh.
- Single global seed set (e.g., only `{42, 123, 456}` or only `{42, 52, 62}`). Seed sets are phase-specific.
