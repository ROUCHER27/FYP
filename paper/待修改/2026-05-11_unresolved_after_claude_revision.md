# Claude Revision Status Review - 2026-05-11

Scope reviewed: `paper/已修改/*.md`, `paper/results_source_of_truth.md`, `paper/references.md`, and the previous reports in `paper/待修改/`.

Conclusion: the revision is substantially improved, but it has not completely resolved the previous `待修改` folder. I marked only fully resolved old review files with the `done_` prefix:

- `done_abstract_review.md`
- `done_chapter4_data_review.md`

The remaining old review files should stay open until the issues below are handled.

## Still-open old-review issues

### `00_global_review.md`

Status: not fully resolved.

Resolved:
- The chapter files in `已修改` now explain that X1 contributes 10 engineered columns and the full model input has 15 columns.
- Figure-placeholder HTML comments are removed from the revised chapter files.
- Most single-seed robustness language is weakened.

Still open:
- `paper/results_source_of_truth.md:12` still says "X1 (cumulative returns + cumulative turnover, 15 dimensions)", which repeats the old 10-vs-15 ambiguity.
- `paper/results_source_of_truth.md:136` still says seed-42 traditional and directional losses "fail to produce positive Sharpe that is also robust"; this is still a single-seed robustness claim.
- `paper/references.md:17-19` still has informal entries for MADL/GMADL ("working paper / conference preprint", "Companion / extended formulation"). This remains below final bibliography quality.
- `paper/已修改/chapter2_literature_review.md:130`, `paper/已修改/chapter3_methodology.md:7`, and `paper/已修改/chapter6_conclusion.md:5` still contain broad "across every run / only the loss varies" wording before later caveats. Remove the broad first claim, not just add a correction after it.

### `chapter1_introduction_review.md`

Status: mostly resolved, but not fully.

Still open:
- `paper/已修改/chapter1_introduction.md:27` still says "the strongest hybrid family" is extended into M2-robust gamma. At seed 42, A3 has higher Sharpe than M1, so "strongest" remains too strong unless defined. Suggested fix: "the multiplicative hybrid family selected for interpretability and stability diagnostics".
- `paper/已修改/chapter1_introduction.md:40` asks whether winners are "robust to loss-component normalisation". Because the normalisation probe is diagnostics-grade, use "approximately stable under the diagnostic normalisation probe".

### `chapter2_literature_review_review.md`

Status: partially resolved.

Resolved:
- The bounded-loss claim was corrected to bounded directional activation.
- Figure placeholder is gone.
- "Symmetric asymmetry" is fixed.

Still open:
- `paper/已修改/chapter2_literature_review.md:130` says the comparison fixes data/features/model/portfolio/evaluation "across every run, and varies only the loss function". This repeats the global overclaim.
- `paper/已修改/chapter2_literature_review.md:142` claims gamma07 outperforms traditional and pure directional losses on "cross-seed stability". Those baselines were not evaluated across seeds in the final evidence.
- `paper/已修改/chapter2_literature_review.md:108-110` still brings specific Phase 2 empirical rankings into the literature review. It is acceptable as positioning, but the chapter should keep numerical/empirical rankings mainly in Chapter 5.
- Citations remain uneven: Huber is named without bracket citation, while `[7]`/`[8]` are still weak bibliography entries.

### `chapter3_methodology_review.md`

Status: mostly resolved, but not fully.

Still open:
- `paper/已修改/chapter3_methodology.md:7` starts with "held fixed across every run; the only factor that varies is the loss function", then qualifies it in the next sentence. The first sentence is still too broad.
- `paper/已修改/chapter3_methodology.md:3` contains process-style and grammatical roughness: "The description is grounded ...; The description follows..." This should be cleaned for final-report tone.
- `paper/已修改/chapter3_methodology.md:41` and `paper/已修改/chapter3_methodology.md:224` contain Obsidian highlight markup (`==...==`) inside headings. Remove before final export.

### `chapter5_empirical_results_discussion_review.md`

Status: previous factual errors are largely fixed, but new/remaining claim-boundary issue exists.

Resolved:
- "full-batch" is corrected to batch size 1024 mini-batch training.
- A3 R2 explanation is corrected.
- Adaptive-lambda CV comparison is corrected.
- Figure placeholders are gone.

Still open:
- `paper/已修改/chapter5_empirical_results_discussion.md:213` says gamma07 outperforms traditional regression and pure absolute-loss variants "on long-short portfolio Sharpe and on cross-seed stability". The Sharpe comparison to baselines is seed-42/same-window; the cross-seed stability comparison is only among Phase 2 multi-seed rows. Split this sentence.
- `paper/已修改/chapter5_empirical_results_discussion.md:42` says absolute-loss-style variants should be evaluated by "Sharpe, cumulative return, CV across seeds" even though baseline absolute-loss rows have no CV. Suggested: "and, where multi-seed evidence exists, CV".

### `chapter6_conclusion_review.md`

Status: partially resolved.

Resolved:
- MedSE is no longer grouped with MSE as practically interpretable R2.
- Transaction-cost relative ordering is now stated as untested.
- `gamma10_stable_zone` is gone.

Still open:
- `paper/已修改/chapter6_conclusion.md:5` repeats "held fixed across every run; only the loss function was varied" without immediate phase scoping.
- `paper/已修改/chapter6_conclusion.md:15` says gamma07 is "Robust to the component-normalisation probe." Use "approximately stable under the diagnostic component-normalisation probe."
- `paper/已修改/chapter6_conclusion.md:9` uses "dominates" in the RQ heading. Prefer "performs best on the joint Sharpe-stability criterion".

### `appendix_A_review.md`

Status: mostly resolved, but not fully.

Still open:
- `paper/已修改/appendix_A_loss_definitions.md:135` says "the empirically tuned optimum is gamma = 0.07" inside a formula appendix. Add "within the reported 24-month, 3-seed protocol" or move this statement back to Chapter 5.

### `appendix_B_review.md`

Status: not fully resolved.

Resolved:
- Normalisation seed-ID caveat is preserved.
- The unsupported "stable to fourth decimal" claim is softened, and Appendix B notes no formal rerun artifact exists.

Still open:
- `paper/已修改/appendix_B_per_seed_raw.md:47` says "Every row below reproduces on main at commit 6c0fbde", but Tables 5.3-5.5 depend on `phase2.2-fix` or `doc/phase2-fix/phase2.2-fix1`. Change to: "Tables 5.1-5.2 reproduce on main at commit 6c0fbde; Tables 5.3-5.5 are read/reproduced from the branch/path listed below."

## Operational issue

The revised manuscript lives in `paper/已修改/`. The root files under `paper/*.md` still contain old text and old placeholder comments. If the final assembly script reads root `paper/*.md`, it will assemble the stale draft. Before final export, either copy the accepted `已修改` files into the root paper files or update the assembly workflow to read `paper/已修改/`.
