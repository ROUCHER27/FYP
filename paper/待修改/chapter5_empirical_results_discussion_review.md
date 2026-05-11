# Review: Chapter 5 Empirical Results and Discussion

Source reviewed: `paper/chapter5_empirical_results_discussion.md`.

## Summary

Chapter 5 is the strongest chapter structurally, but it contains several factual/result-interpretation errors that should be fixed before any polishing pass. Two are high priority: "full-batch training" is wrong, and the adaptive-λ CV comparison ignores `imadl_m2_alpha02`'s larger CV.

## Content and evidence issues

1. `chapter5_empirical_results_discussion.md:13` repeats the 15-dimensional X1 issue. Fix after Chapter 4 defines the exact model input.

2. `chapter5_empirical_results_discussion.md:15` says "full-batch training" while the protocol uses `batch_size=1024`. Replace with "mini-batch training with batch size 1024" or "batch size 1024."

3. `chapter5_empirical_results_discussion.md:40` mentions "earlier six-month sanity-check folklore." This phrase is informal and phase-internal. For a final report, write "earlier preliminary six-month checks."

4. `chapter5_empirical_results_discussion.md:42` says "the MLP driven by an unbounded absolute-loss family..." MADL/GMADL as defined are bounded in directional activation and weighted by realised return; the problem is scale drift under a direction/magnitude-weighted objective, not simply an "unbounded absolute-loss family." Rewrite for technical accuracy.

5. `chapter5_empirical_results_discussion.md:42` says GMADL has the second-highest Sharpe in Table 5.1. It is second among the non-hybrid baselines, but `hybrid_mul_m1` is higher and GMADL is above only MedSE among positives. Say exactly what is meant.

6. `chapter5_empirical_results_discussion.md:93` says A3's high R² magnitude is "tens or hundreds of thousands"; Table 5.2 shows A3 Avg R² is `-1,383.64`. A5 is hundreds of thousands, A4 is tens of thousands, but not A3. Correct this explanation.

7. `chapter5_empirical_results_discussion.md:122` says lower CV means the seed-42 result is more representative. CV does not specifically validate seed 42; it measures cross-seed dispersion relative to mean. Rewrite as "lower CV implies lower relative seed sensitivity."

8. `chapter5_empirical_results_discussion.md:150` says Table 5.4 reports selected grouped summaries and further γ variants but omits `m2_robust_gamma07`, the primary recommendation. That is not wrong if Table 5.3 supplies γ07, but readers may wonder why the primary row is absent from the integrated table. Add one sentence that γ07 is shown in Table 5.3 and used as the reference line in Figure 5.4.

9. `chapter5_empirical_results_discussion.md:183` says `adaptive_lambda10` has the largest CV among non-GMADL-β rows. This is false: Table 5.4 shows `imadl_m2_alpha02` has CV `6.4735`, larger than `adaptive_lambda10`'s `1.5426`. Correct the comparison.

10. `chapter5_empirical_results_discussion.md:205-211` says `gamma07` is robust to normalisation. Because the probe's scale ratios are diagnostics-estimated, use "approximately stable under the diagnostic normalisation probe" rather than a general robustness claim.

11. `chapter5_empirical_results_discussion.md:226` says the γ family "produces a mean Sharpe that exceeds the best Phase 1.5 seed-42 variant." This is true numerically for γ07 vs A3, but the sentence is framed as the "closest supportable statement." Keep it, but make explicit that it is descriptive only, not causal.

12. `chapter5_empirical_results_discussion.md:48-66` and `chapter5_empirical_results_discussion.md:97-111` still contain figure-placeholder HTML comments. Delete them.

## SciWrite issues

1. Result interpretation is generally clear, but §5.2 and §5.3 use long multi-clause paragraphs. Split the observations into shorter paragraphs or bullets.

2. Replace "Reading this row charitably" at `chapter5_empirical_results_discussion.md:142` with a more formal phrase such as "Interpreted cautiously."

3. Avoid "operator" language in `chapter5_empirical_results_discussion.md:234` unless this report is explicitly deployment-facing. "Use case" or "reader" is safer.

## Top priority revisions

1. Fix "full-batch training."
2. Correct A3 R² explanation.
3. Correct adaptive-λ CV comparison.
4. Weaken normalisation robustness wording.
5. Remove figure-placeholder comments.
