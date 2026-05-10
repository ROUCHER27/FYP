# Empirical Window Provenance Audit

## Decision

The final report uses a controlled static training protocol:

- Training window: 1990-01 to 1994-12
- Main out-of-sample evaluation window: 1995-01 to 1996-12
- Main evaluation length: 24 months

The earlier 1995-01 to 1995-06 window is retained only as a preliminary baseline sanity check. It should not be used as the basis for final headline claims unless the table is explicitly labelled as an early 6-month check.

## Confirmed 24-Month Evidence

- Phase 2.2 gamma refinement summary exists in `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`.
- Phase 2.2 raw runs exist in `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv`.
- The Phase 2.2 grouped summary supports the final headline numbers for `m2_robust_gamma07`: mean Sharpe `0.9156`, CV `0.1808`, and mean cumulative return `0.2799`.

## Main-Branch Artifact Gap

After switching directly to `main`, the local working tree does not contain the Phase 1.5 24-month metrics directories that were present on the experiment branch. Those artifacts should be copied back from the experiment branch or regenerated before final packaging if the final report includes Phase 1.5 baseline tables.

Required baseline rerun/copy targets:

- MSE, MedSE, IMADL, GMADL, hybrid_mul_m1, hybrid_mul_m2, and hybrid_add_a4 under the 24-month window.
- Configuration: `--test-months 24 --max-epochs 20 --batch-size 1024`.
- Output should be written to a dedicated final-report output directory, not to the legacy `sanity_outputs/` directory.

## Verification Rule

Every CSV used in a final-report empirical table should pass:

- It contains 24 monthly rows for main results.
- The first month is `1995-01`.
- The last month is `1996-12`.
- The paired run spec has `train_start=1990-01`, `train_end=1994-12`, `test_start=1995-01`, and `test_months=24`.
