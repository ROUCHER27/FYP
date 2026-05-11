# Review: Chapter 1 Introduction

Source reviewed: `paper/chapter1_introduction.md`.

## Summary

Chapter 1 now has the right final-report structure and avoids the old phase-report framing. The main remaining problems are claim-strength drift: it says "only the loss function changes" across every run, asks RQ2 partly about single-seed families as if they were multi-seed, and relies on an unanchored "preliminary analysis" claim about GMADL limitations.

## Content and evidence issues

1. `chapter1_introduction.md:11` introduces "Preliminary analysis of the GMADL family" and lists several limitations. This needs a clear anchor to later sections, a figure, or implementation notes. Otherwise it reads like an unsupported strong claim in the introduction.

2. `chapter1_introduction.md:17-19` says the report addresses all gaps through a pipeline where "only the loss function changes." That is true for the seed-42 baseline and Phase 1.5 tables, but not across Phase 2 / integrated Phase 2 / normalisation, where γ/α/β/λ, seed sets, and branch formulas differ. Rephrase as "within each controlled comparison."

3. `chapter1_introduction.md:27` says the "strongest hybrid family" is extended into M2-robust γ. This is defensible if "strongest" means design rationale plus M-family interpretability, but Chapter 5 Table 5.2 shows A3 has higher seed-42 Sharpe than M1. Add a sentence that the choice was not based solely on seed-42 Sharpe.

4. `chapter1_introduction.md:38` asks which region among additive hybrids, multiplicative hybrids, M2-robust γ, α, β, and adaptive-λ "produces the best mean Sharpe" across three seeds. Additive A-series and multiplicative M1-M4 are not evaluated across three seeds in the final evidence. Split the RQ into single-seed design-space mapping and multi-seed Phase 2 comparison.

5. `chapter1_introduction.md:57` repeats the 15-dimensional X1 issue. Fix after Chapter 4 defines the exact input columns.

6. `chapter1_introduction.md:63` classifies the integrated Phase 2 grouped summary as "Strong." This may be acceptable within its own branch/table, but cross-comparison with baseline and Phase 1.5 should remain moderate. Clarify "strong within-table, moderate across-phase."

7. `chapter1_introduction.md:76` says the integrated sweep "rules in" γ07 and α06. "Rules in" is rhetorically strong. Suggested: "identifies" or "supports."

## SciWrite issues

1. `chapter1_introduction.md:15-17` is dense and contains several claims per paragraph. Consider splitting the literature imbalance and the three gaps into separate paragraphs.

2. The phrase "systematic empirically evaluated" at `chapter1_introduction.md:17` should be "systematically empirically evaluated" or, better, "evaluated systematically."

3. The "Non-claims" paragraph is valuable, but it is long. Convert to shorter sentences or keep as a compact bullet list.

## Top priority revisions

1. Reframe "only the loss function changes" to "within each phase/table."
2. Correct RQ2 so it does not imply A/M seed-42 families have multi-seed evidence.
3. Anchor GMADL limitation claims.
4. Fix X1 description after Chapter 4 is corrected.
