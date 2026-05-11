# Review: Appendix A Loss Function Definitions and Gradients

Source reviewed: `paper/appendix_A_loss_definitions.md`.

## Summary

Appendix A is valuable and should stay, but it currently overstates exactness for Phase 2 losses and has a few mathematical wording errors. The main fix is to separate implemented exact formulas from conceptual formulas.

## Content and evidence issues

1. `appendix_A_loss_definitions.md:3` says every formula matches `Model_Train/losses.py`. This is not true for `A.3.3 M2-robust γ family`, whose exact implementation lives on `phase2.2-fix` and is described conceptually. Revise the opening: "Sections A.1-A.3.2 match current-main `Model_Train/losses.py`; A.3.3 records the conceptual form used to interpret Phase 2 branch outputs."

2. `appendix_A_loss_definitions.md:48` says the Huber gradient equals `-δ` from both sides at `|e| = δ`. That is true at `e=+δ` but at `e=-δ` the derivative with respect to `ŷ` is `+δ`. Rewrite as "the one-sided derivatives agree at both boundaries."

3. `appendix_A_loss_definitions.md:77` says R² values "divulge" under GMADL. This is a typo; use "diverge."

4. `appendix_A_loss_definitions.md:95` says PyTorch does not propagate gradient through the batch mean. Since the batch mean depends only on `y_true`, not `y_pred`, this is effectively true for the prediction gradient, but the wording should be precise: "for gradients with respect to `ŷ`, the normalisation denominator is constant."

5. `appendix_A_loss_definitions.md:125` says when direction is wrong, `D` is near 1. Because `D` includes a batch-normalised `|y|^b` factor, it is not necessarily near 1 for every observation. Rewrite as "the sigmoid penalty approaches 1 and the Huber term is amplified in proportion to the normalised magnitude weight."

6. `appendix_A_loss_definitions.md:135` gives the M2-robust optimum as γ=0.07. That belongs in empirical interpretation, not formula definition. It is okay to cross-reference Chapter 5, but avoid saying "empirically tuned optimum" in a formula appendix unless you specify "within the reported protocol."

## SciWrite issues

1. The appendix is concise and mostly clear. Add a short notation table if the LaTeX pass will use it.

2. Use one spelling for "normalisation/normalization" across the final report. Current project prose mostly uses British spelling.

## Top priority revisions

1. Qualify exactness of formulas.
2. Correct Huber gradient boundary wording.
3. Fix the "divulge" typo.
4. Make the `D near 1` explanation mathematically precise.
