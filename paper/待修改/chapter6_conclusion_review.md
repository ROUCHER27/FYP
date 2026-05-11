# Review: Chapter 6 Conclusion

Source reviewed: `paper/chapter6_conclusion.md`.

## Summary

The conclusion answers the research questions cleanly and carries the limitations forward. The main issue is overclaiming: it occasionally compares cross-seed stability against losses that were never evaluated across seeds, and it assumes transaction costs preserve relative ordering without evidence.

## Content and evidence issues

1. `chapter6_conclusion.md:7` says MSE/MedSE produce "interpretable R² values." MSE's R² is interpretable as a scale diagnostic, but MedSE's Avg R² in Chapter 5 is about `-2,297,042`, which is not practically interpretable as prediction quality. Rewrite to avoid grouping MSE and MedSE here.

2. `chapter6_conclusion.md:7` says MSE/MedSE fail to generate a "robust positive Sharpe at seed 42." At a single seed, use "positive/economically meaningful seed-42 Sharpe," not "robust."

3. `chapter6_conclusion.md:9` says β family CV values are "single to triple digits." This is true but imprecise; use "from 4.0 to 139.5" if you want the statement to be auditable.

4. `chapter6_conclusion.md:11` says γ07 is "robust to the two different scale regimes tested." Because the probe is diagnostics-grade, soften to "approximately stable in the diagnostic normalisation probe."

5. `chapter6_conclusion.md:35` says transaction costs likely preserve relative ordering because costs apply roughly uniformly to every variant. This is unsupported: turnover can differ materially by loss function even with the same cap/bucket rule. Rewrite as an open limitation: cost-adjusted ordering is untested.

6. `chapter6_conclusion.md:47` uses "gamma10_stable_zone," which is undefined and also conflicts with γ10 being the high-return but less stable alternative. Clarify the intended ordering criterion.

7. `chapter6_conclusion.md:59` says γ07 outperforms traditional regression and pure directional losses "on cross-seed stability." Those baselines were not measured across seeds in the final evidence. It can outperform them on seed-42 Sharpe baseline comparison and outperform competitive γ values on cross-seed stability, but not both in one claim.

8. `chapter6_conclusion.md:39` repeats the X1 15-dimensional issue. Fix after Chapter 4 defines exact input columns.

## SciWrite issues

1. §6.1 is long for a conclusion. The RQ answers are useful, but each can be tightened to keep conclusion from repeating Chapter 5.

2. `chapter6_conclusion.md:61` is a strong closing paragraph. Keep it, but remove internal-file references if the final report should not read like a repository artifact.

3. The limitations are well structured. Consider ordering them by threat to validity: single window, seed depth, transaction costs, feature/architecture, scale logging.

## Top priority revisions

1. Remove "robust" from single-seed claims.
2. Delete unsupported transaction-cost relative-ordering claim.
3. Fix final sentence so cross-seed stability is only compared among multi-seed rows.
4. Clarify undefined "gamma10_stable_zone."
