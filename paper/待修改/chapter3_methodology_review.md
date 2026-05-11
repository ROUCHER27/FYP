# Review: Chapter 3 Methodology

Source reviewed: `paper/chapter3_methodology.md`.

## Summary

Methodology is detailed and mostly reproducible, but it contains the report's most important technical inconsistency: the exact 15-dimensional feature matrix is not defined. The chapter also claims all loss formulas match implementation exactly, then gives only conceptual formulas for Phase 2 branch variants.

## Content and evidence issues

1. `chapter3_methodology.md:31` and `chapter3_methodology.md:151` say X1 is a 15-dimensional feature vector/matrix. The code in `Model_Train/features.py:50-74` creates 10 engineered X1 columns. The 15-dimensional input likely comes from `assemble_feature_matrix` retaining raw/base columns in addition to those 10. This must be made explicit in Chapter 3 or Chapter 4, otherwise the architecture input dimension is not reproducible from the prose.

2. `chapter3_methodology.md:39` says each loss definition "matches the implementation exactly." This is true for current-main losses in `Model_Train/losses.py`, but `chapter3_methodology.md:132-139` gives only a conceptual form for M2-robust γ and says exact implementations live on `phase2.2-fix`. Revise the global claim to distinguish exact current-main formulas from conceptual branch formulas.

3. `chapter3_methodology.md:137` says large γ approaches the Phase 1.5 M2 form. This may be true by design, but it needs a branch-code or diagnostic citation because the exact `H^γ` form is not shown.

4. `chapter3_methodology.md:145` says exact Phase 2 seed sets are recorded in manifests. Appendix B gives γ refinement seeds `{42,52,62}`, but normalisation probe seed IDs are anonymised. Keep this distinction consistent.

5. `chapter3_methodology.md:155` says Adam uses PyTorch defaults. Add the actual default learning rate (`lr=0.001`) if confirmed by the runner, because training protocol reproducibility depends on it.

6. `chapter3_methodology.md:258` says every table is reproducible from "a single branch." But Table 5.4 requires `git show phase2.2-fix:...` while baseline evidence is on `main`. Suggested wording: "from the local clone using the branch/path listed for that table."

7. `chapter3_methodology.md:260` says grouped-summary figures are stable to the fourth decimal across supervisor re-runs. This is a strong reproducibility claim. If no artifact records those reruns, weaken to "reported to be stable" or remove.

8. `chapter3_methodology.md:169-184` still contains a figure-placeholder HTML comment even though the figure exists. Delete the comment block.

## SciWrite issues

1. `chapter3_methodology.md:3` says "where the text differs from older draft material, the code is authoritative." This is useful internally but reads like process commentary. For final report tone, replace with "The description follows the implementation..."

2. The numbered training protocol is good. Consider adding a compact table of key hyperparameters before the steps for scanability.

3. Avoid repeating "Phase 2" caveats in both §3.7 and §3.8 unless one section explicitly references the other.

## Top priority revisions

1. Define the exact 15 input columns.
2. Qualify "matches implementation exactly" for Phase 2 branch variants.
3. Fix the "single branch" reproducibility wording.
4. Remove figure-placeholder comments.
