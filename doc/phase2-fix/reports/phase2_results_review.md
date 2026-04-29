# Phase 2 P0 Results Review

## Data Reviewed
- Source: `doc/phase2-fix/reports/phase2_raw_runs.csv`
- Source: `doc/phase2-fix/reports/phase2_grouped_summary.csv`
- Scope: 16 losses x 3 seeds x cap 0.05 = 48 completed runs

## Headline
P0 fixes worked. The corrected M2 family is now the strongest direction, and the robust M2 variant beats the original Phase 1.5 M2 target on mean Sharpe.

Top results by mean Sharpe:

| Rank | Loss | Mean Sharpe | Std | CV | Mean Cum. Return |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `m2_robust_gamma10` | 1.0043 | 0.5638 | 0.5613 | 0.2368 |
| 2 | `m2_robust_gamma01` | 0.7470 | 0.3937 | 0.5270 | 0.2718 |
| 3 | `m2_robust_gamma001` | 0.6919 | 0.8258 | 1.1936 | 0.1705 |
| 4 | `imadl_m2_alpha06` | 0.6895 | 0.1685 | 0.2443 | 0.3042 |
| 5 | `imadl_m2_alpha05` | 0.5822 | 0.3193 | 0.5484 | 0.2465 |

Note: `m2_robust_gamma10` means gamma = 1.0 in code, not gamma = 10.

## Against Expected Criteria

Minimum expectations:
- Passed: Variant 1 has multiple alpha values with Sharpe > 0.2.
- Passed: Variant 3 has gamma values with CV < 1.2.
- Passed: all 48 summaries exist.

Target expectations:
- Passed: Variant 1 has at least two alpha values with Sharpe > 0.4 (`alpha05`, `alpha06`, `alpha07`, `alpha08`).
- Passed: Variant 3 has Sharpe > 0.3 and CV < 1.0 for `gamma01` and `gamma10`.
- Not passed by strict threshold: Variant 4 did not exceed Sharpe > 0.5. `adaptive_lambda10` reached 0.4938, which is close but unstable.

Ideal expectation:
- Passed: at least one loss has Sharpe > 0.6 and CV < 0.9. Strong examples are `m2_robust_gamma10`, `m2_robust_gamma01`, and `imadl_m2_alpha06`.

## Interpretation

### Variant 3: M2 Robust Is The Main Winner
`m2_robust_gamma10` is the best mean Sharpe at 1.0043 with CV 0.5613. This is stronger than the Phase 1.5 M2 reference Sharpe of about 0.914 and much cleaner than the broken pre-fix Phase 2 behavior.

`m2_robust_gamma01` is slightly lower Sharpe but has similar stability. It also has the highest mean cumulative return among the robust M2 variants.

Recommendation: carry both `m2_robust_gamma10` and `m2_robust_gamma01` forward.

### Variant 1: IMADL + M2 Works, Best At Alpha 0.6
`imadl_m2_alpha06` is the best balance in this family: Sharpe 0.6895, std 0.1685, CV 0.2443, and mean cumulative return 0.3042. It is not the highest Sharpe overall, but it is the most stable high-performing result in this run.

Recommendation: carry `imadl_m2_alpha06` forward as the stable candidate; optionally keep `alpha05` as a nearby sensitivity point.

### Variant 2: IMADL + GMADL Fails
All IMADL+GMADL variants are near zero or negative Sharpe, with extremely bad R2 values. This suggests the two directional terms are not complementary in the current implementation and scale regime.

Recommendation: do not spend more experiment budget on this family unless there is a specific theoretical reason to revisit it after loss-scale diagnostics.

### Variant 4: Adaptive Is Not Competitive Yet
`adaptive_lambda10` is close to the 0.5 Sharpe threshold, but its CV is 1.5426 and one seed is negative. This is too unstable compared with robust M2 and alpha06.

Recommendation: keep as secondary/future work, not as the next main branch.

## Next Actions
1. Run Phase 2.1b alignment for `imadl`, `gmadl`, and `hybrid_mul` to confirm the Phase 2 runner can reproduce Phase 1.5 baselines.
2. Run loss-scale diagnostics for the representative set: `imadl_m2_alpha06`, `m2_robust_gamma01`, `m2_robust_gamma10`, and `adaptive_lambda10`.
3. Run a focused gamma refinement around the robust-M2 winner: gamma = 0.3, 0.5, 0.7, 1.0, 1.5.
4. Carry forward top candidates into seed ensemble / validation-Sharpe early stopping: `m2_robust_gamma10`, `m2_robust_gamma01`, and `imadl_m2_alpha06`.

## Caveats
- The grouped table has only 3 seeds per loss. Treat rankings as directional, not final proof.
- Many runs have very similar directional accuracy around 0.531, so the economic improvement is not coming from a broad directional-accuracy jump. It is likely coming from position sizing / return distribution effects.
- R2 remains negative for all top strategies, so the thesis narrative should stay focused on trading objective alignment rather than predictive accuracy.
