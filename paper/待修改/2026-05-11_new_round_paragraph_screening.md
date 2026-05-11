# New Round Paragraph Screening - 2026-05-11

Scope: revised draft in `paper/已修改/*.md`. This is a fresh paragraph-level screen after the Claude revision.

## Global

1. **Chinese abstract is missing.**
   - Marking scheme requirement: the final report must include an English abstract followed by a Chinese translation.
   - Current file: `paper/已修改/abstract.md` has only the English abstract and keywords.
   - Fix: add a faithful Chinese abstract immediately after the English abstract. It is not marked for score, but the instructor must check correctness.

2. **Root paper files and revised files diverge.**
   - Current revised files are in `paper/已修改/`.
   - Root `paper/*.md` still includes stale prose and placeholder comments.
   - Fix: decide final source path before LaTeX conversion. If root files are the assembly source, replace them with the accepted revised files.

3. **Remove Obsidian highlight markers from headings.**
   - `paper/已修改/chapter3_methodology.md:41`
   - `paper/已修改/chapter3_methodology.md:224`
   - `paper/已修改/chapter4_data.md:105`
   - Fix: change headings like `### ==3.3.1 Regression losses==` to normal Markdown headings.

## Abstract

No old content-boundary problems remain. Add the Chinese translation as noted above.

## Chapter 1

1. `paper/已修改/chapter1_introduction.md:27`
   - Problem: "strongest hybrid family" is not fully supported because Phase 1.5 seed-42 A3 has higher Sharpe than M1.
   - Fix: describe the chosen family as "the multiplicative hybrid family selected for interpretability and for the Phase 2 robustness design", not as the strongest family.

2. `paper/已修改/chapter1_introduction.md:40`
   - Problem: "robust to loss-component normalisation" is too strong for a diagnostic probe with estimated scale ratios.
   - Fix: "approximately stable under the diagnostic loss-component normalisation probe".

## Chapter 2

1. `paper/已修改/chapter2_literature_review.md:22`
   - Problem: Huber (1964) is named but not bracket-cited, while `[A1]` exists in `references.md`.
   - Fix: cite `[A1]` or merge auxiliary references into the main bibliography.

2. `paper/已修改/chapter2_literature_review.md:130`
   - Problem: "across every run, and varies only the loss function" is too broad across phases.
   - Fix: "within each comparison table, the loss function is isolated; cross-phase comparisons differ in seed sets, branch implementations, and parameterisations."

3. `paper/已修改/chapter2_literature_review.md:142`
   - Problem: "outperforms traditional regression losses and pure directional losses on ... cross-seed stability" is unsupported because those baselines were not evaluated cross-seed.
   - Fix: split into two claims: seed-42 Sharpe vs baselines; cross-seed stability only among Phase 2 multi-seed variants.

4. `paper/已修改/chapter2_literature_review.md:108-110`
   - Problem: the literature review gives detailed project-result interpretation.
   - Fix: keep only design rationale here; leave precise alpha/gamma rankings to Chapter 5.

## Chapter 3

1. `paper/已修改/chapter3_methodology.md:3`
   - Problem: final-report tone is weakened by process prose and a repeated subject.
   - Fix: "This chapter describes the implementation used to produce every Chapter 5 result. The methodology follows `Model_Train/` and `sanity_check_signal_tilted.py` on `main` at commit `6c0fbde`."

2. `paper/已修改/chapter3_methodology.md:7`
   - Problem: "across every run; only factor varies" remains too broad.
   - Fix: lead with scoped wording: "Within each comparison table, data, features, model, training settings, portfolio construction, and metrics are fixed; the loss is varied. Cross-phase comparisons are treated separately in §3.8."

3. `paper/已修改/chapter3_methodology.md:53`
   - Problem: MedSE "full batch per optimisation step" can be misunderstood as full-dataset training, conflicting with batch size 1024.
   - Fix: "computed over each mini-batch as a median reduction; it is non-decomposable within the mini-batch."

## Chapter 4

Main old issues are fixed. Remaining polish:

1. `paper/已修改/chapter4_data.md:7`
   - Problem: "CRSP-style" remains ambiguous. If this is CRSP data, use "CRSP monthly stock file" and cite `[A4]`; if not, explain why "style" is used.

2. `paper/已修改/chapter4_data.md:155`
   - Problem: says training window contains 10,987 PERMNOs and source file covers 61 month-sections. This is now correct, but could be clearer if separated into "training partition" vs "source file".

## Chapter 5

1. `paper/已修改/chapter5_empirical_results_discussion.md:42`
   - Problem: says absolute-loss-style variants should be evaluated by "CV across seeds", but baseline absolute-loss rows are single-seed.
   - Fix: "Sharpe and cumulative return, and where multi-seed evidence exists, CV."

2. `paper/已修改/chapter5_empirical_results_discussion.md:145`
   - Problem: "deployments that weight cumulative return..." is too deployment-facing for a final report with no transaction costs.
   - Fix: "interpretations that prioritise cumulative return over volatility".

3. `paper/已修改/chapter5_empirical_results_discussion.md:173`
   - Problem: "Its signal generalises across two different scale regimes" is slightly strong for a diagnostic normalisation probe.
   - Fix: "Its signal is consistent across the original and diagnostic-normalised settings."

4. `paper/已修改/chapter5_empirical_results_discussion.md:213`
   - Problem: cross-seed stability is attributed against traditional and pure absolute losses, but those baselines are single-seed.
   - Fix: "It outperforms traditional regression and pure directional losses on seed-42 same-window Sharpe, and among the competitive Phase 2 multi-seed rows it has the best joint Sharpe-stability profile."

## Chapter 6

1. `paper/已修改/chapter6_conclusion.md:5`
   - Problem: overbroad "held fixed across every run; only loss varied".
   - Fix: scope to "within each comparison table".

2. `paper/已修改/chapter6_conclusion.md:9`
   - Problem: "dominates in multi-seed Sharpe and stability" is rhetorically stronger than needed.
   - Fix: "performs best on the joint Sharpe-stability criterion".

3. `paper/已修改/chapter6_conclusion.md:15`
   - Problem: "Robust to the component-normalisation probe" is too strong.
   - Fix: "approximately stable under the diagnostic component-normalisation probe."

4. `paper/已修改/chapter6_conclusion.md:47`
   - Problem: says "if the 3-seed Sharpe ordering ... is robust at higher seed depth". This is future-work context, but "robust" can still be misread.
   - Fix: "if the ordering persists at higher seed depth".

## Appendix A

1. `paper/已修改/appendix_A_loss_definitions.md:135`
   - Problem: "empirically tuned optimum" appears in a formula appendix without protocol scope.
   - Fix: "the empirically selected setting within the reported 24-month, 3-seed protocol is gamma = 0.07".

## Appendix B

1. `paper/已修改/appendix_B_per_seed_raw.md:47`
   - Problem: "Every row below reproduces on main" is false for Phase 2 and normalisation rows.
   - Fix: split Tables 5.1-5.2 from Tables 5.3-5.5 by branch/path.

2. `paper/已修改/appendix_B_per_seed_raw.md:137`
   - Problem: "reported to be stable across re-runs" is acceptable but still unsupported by preserved artifacts.
   - Fix: if no artifact is added, keep the current "no formal rerun artifact" caveat and avoid using this stability claim in the main chapters.
