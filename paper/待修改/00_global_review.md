# Global Initial Review

Reviewer role: first-draft adversarial reviewer. Standards used: `.kiro/skills/EvoSkills/skills/paper-review/SKILL.md`, `.kiro/skills/sciwrite/SKILL.md`, `SCHEMA.md`, `doc/agent_handoff.md`, and `paper/results_source_of_truth.md`.

## Reject-first summary

This draft is not ready for final submission because several claims are stronger than the evidence gate permits, one central data/feature description is internally inconsistent, and some results commentary contains factual errors. The report is much cleaner than the old phase-style draft, but a reviewer can still attack three points: reproducibility of the exact 15-dimensional X1 input, whether three-seed evidence is being over-sold as robustness, and whether cross-phase comparisons are being treated as direct evidence despite the report's own alignment caveats.

## High-priority cross-chapter issues

1. **X1 feature dimension is not explained correctly.**
   - Affected: `abstract.md:5`, `chapter1_introduction.md:57`, `chapter3_methodology.md:31`, `chapter3_methodology.md:151`, `chapter4_data.md:88`, `chapter5_empirical_results_discussion.md:13`, `chapter6_conclusion.md:39`, `results_source_of_truth.md:12`.
   - Problem: the prose says X1 is "15-dimensional cumulative-return and cumulative-turnover" or says five horizons x two variables. That only explains 10 engineered columns. The code path `Model_Train/features.py:50-74` creates 10 X1 columns, while `assemble_feature_matrix` includes all non-target/non-id/non-date columns, which likely leaves raw/base columns (`RET`, `VOL`, `SHROUT`, `r`, `to`) and yields 15 dimensions. The report must explicitly name all 15 model inputs or change the claim.
   - Fix: add a small table in Chapter 4 listing the exact 15 columns consumed by `assemble_feature_matrix`; then update every 15-dimensional X1 phrase to match.

2. **"Robustness" is sometimes applied to single-seed evidence.**
   - Affected: `abstract.md:9`, `chapter1_introduction.md:38`, `chapter5_empirical_results_discussion.md:46`, `chapter6_conclusion.md:7`, `chapter6_conclusion.md:59`.
   - Problem: seed-42 baseline and Phase 1.5 tables cannot support robustness claims. They can support same-window, same-seed comparisons only.
   - Fix: reserve "robustness" for three-seed Phase 2 rows. Use "single-seed baseline result" or "same-window seed-42 comparison" elsewhere.

3. **Residual figure-placeholder HTML comments remain in the chapter files.**
   - Affected: `chapter2_literature_review.md:75-92`, `chapter3_methodology.md:169-184`, `chapter4_data.md:23-42`, `chapter5_empirical_results_discussion.md:48-66`, `chapter5_empirical_results_discussion.md:97-111`.
   - Problem: the figures now exist, so these draft comments should not remain in a final report source.
   - Fix: delete the placeholder comment blocks; keep the figure title, image link, and generated-by/source caption.

4. **Several "identical protocol / only loss changes" statements are too broad across phases.**
   - Affected: `abstract.md:7`, `chapter1_introduction.md:19`, `chapter3_methodology.md:7`, `chapter5_empirical_results_discussion.md:5`.
   - Problem: within a phase, loss is the isolated factor. Across Phase 1.5, Phase 2, integrated Phase 2, and normalisation, hyperparameters, seed sets, branch implementations, and formula variants differ.
   - Fix: say "within each comparison table" or "within each phase" unless explicitly discussing the evidence gate and its caveats.

5. **Bibliography and auxiliary references need final-format cleanup.**
   - Affected: `references.md`.
   - Problem: several references are placeholders or informal ("working paper / conference preprint", "Companion / extended formulation", "data as of snapshot"). Auxiliary references are named but not cited inline with bracket keys.
   - Fix: before LaTeX conversion, replace informal entries with full bibliographic details and either cite `[A1]..[A8]` consistently or merge them into the main bibliography.

## Trust scorecard

- Fairness of baseline comparison: 1.5/2. Strong within-phase design, but no transaction costs or alternative portfolio caps.
- Reproducibility details: 1.5/2. Good artifact paths and commands; X1 exact columns and Phase 2 branch dependency need tightening.
- Honest limitations: 2/2. Limitations are explicit and mostly credible.
- Failure case transparency: 1/2. Limitations are named, but failure cases are not shown empirically.
- Statistical soundness: 1/2. Three seeds are acknowledged as thin, but some prose still leans too strongly on CV.

Total: 7/10 after fixes; below 7 if X1 and overclaim issues remain.

## Top revision priorities

1. Fix the X1 15-dimensional feature explanation across all files.
2. Remove all residual figure-placeholder HTML comments.
3. Weaken single-seed and cross-phase overclaims.
4. Correct Chapter 5 factual errors flagged in its chapter review.
5. Clean bibliography quality and citation consistency before final assembly.
