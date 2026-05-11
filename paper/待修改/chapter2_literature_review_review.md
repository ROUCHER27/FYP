# Review: Chapter 2 Literature Review

Source reviewed: `paper/chapter2_literature_review.md`.

## Summary

The chapter gives a coherent bridge from asset-pricing ML to robust and trading-aware losses. The weaknesses are mostly reviewer-facing: some finance/robust-regression claims are under-cited, project-specific design discussion enters the literature review too early, and one technical statement about bounded directional losses is wrong as written.

## Content and evidence issues

1. `chapter2_literature_review.md:15-22` makes broad distributional claims about equity returns, skewness, volatility clustering, and robust regression in finance. Add citations or explicitly tie only the numerical heavy-tail facts to Chapter 4. The negative-skew claim is not established by the quoted training-panel max of `+2400%`.

2. `chapter2_literature_review.md:33-35` introduces Huber and MedSE but only Huber appears in auxiliary references. If MedSE is project-defined rather than literature-standard, state that; otherwise add a citation or define it as an implemented robust baseline.

3. `chapter2_literature_review.md:63` says sigmoid gives "smoother gradients than tanh at large arguments." Both sigmoid and tanh saturate. If the intended point is implementation smoothness versus a step-form MADL, state that instead.

4. `chapter2_literature_review.md:65` introduces IMADL as a project-code variant inside the literature review. That is acceptable only if this section is explicitly "related loss families and project variants"; otherwise move detailed IMADL implementation discussion to Chapter 3 and keep Chapter 2 focused on literature.

5. `chapter2_literature_review.md:71` says every directional-loss term is bounded and therefore robust. This is not correct as a mathematical statement for GMADL because the term is weighted by `|y|^b`; it is bounded in prediction/alignment but not bounded in realised-return magnitude. Rewrite as "bounded in the directional activation" and do not present it as full outlier robustness.

6. `chapter2_literature_review.md:99` uses "Symmetric asymmetry," which is confusing. Use "Symmetric reward and penalty" or "No asymmetric risk preference."

7. `chapter2_literature_review.md:125-127` interprets γ and α sweep results before the empirical chapter. This is useful for positioning, but the literature review should avoid sounding like a second results chapter. Keep the high-level design rationale and send numerical rankings to Chapter 5.

8. `chapter2_literature_review.md:159` says γ07 "dominates" traditional and pure directional losses. This is scoped later, but "dominates" is strong. Use "outperforms within the studied protocol" or "is best supported under the evidence gate."

9. `chapter2_literature_review.md:75-92` still contains a figure-placeholder HTML comment even though `fig2_1_loss_shapes.png` exists. Delete the comment block.

## SciWrite issues

1. Avoid "horse race" at `chapter2_literature_review.md:9` unless the school accepts informal finance jargon. "Comparative study" is safer.

2. Paragraphs `chapter2_literature_review.md:131-141` are good but long. Split the multiple-testing discussion from the internal/external validity discussion.

3. Keep terminology consistent: "absolute-loss family," "directional-loss family," and "trading-aware family" currently overlap. Define once and reuse.

## Top priority revisions

1. Fix the bounded-loss claim.
2. Add/clean citations for robust regression and return-distribution claims.
3. Remove figure-placeholder comments.
4. Soften "dominates" language.
