# Second Pass Review After Claude Revision - 2026-05-11

Scope reviewed: latest `paper/已修改/*.md`, `paper/results_source_of_truth.md`, and `paper/references.md`.

Verdict: the revised manuscript is now close to final-report quality. Most issues from `2026-05-11_new_round_paragraph_screening.md` are fixed: Chinese abstract is present, "strongest hybrid family" is softened, the main cross-seed-vs-single-seed claim has been split, Appendix A/B protocol wording is fixed, and Obsidian heading highlights are removed. Remaining problems are fewer but still worth fixing before final assembly.

## High-priority remaining fixes

### 1. `results_source_of_truth.md` still contains stale claims

File: `paper/results_source_of_truth.md`

Issues:
- Line 12 still says `Feature set: X1 (cumulative returns + cumulative turnover, 15 dimensions)`. This repeats the old ambiguity. It should say: "X1 contributes 10 engineered columns; the 15-column model input also includes `RET`, `VOL`, `SHROUT`, `r`, and `to`."
- Line 131 says the M2-robust gamma family "dominates in multi-seed Sharpe." This is stronger than the chapter prose. Prefer "is the best-supported family under the reported multi-seed Sharpe/stability evidence."
- Line 136 says single-seed MSE/MedSE/MADL/GMADL "fail to produce positive Sharpe that is also robust." This is still invalid because robustness cannot be tested at seed 42. Replace with "fail to produce economically meaningful positive Sharpe in the seed-42 same-window baseline."

Why this matters: Chapter 5 presents `results_source_of_truth.md` as the source of every allowed number. If the source-of-truth file contains stale claim language, a later writing/assembly pass may reintroduce rejected claims.

### 2. Bibliography is still below final-mark quality

File: `paper/references.md`

Issues:
- Line 17 still describes MADL as "Working paper / conference preprint." If no full venue exists, keep "working paper" only if this is factually true and add a URL/identifier if available.
- Line 19 still describes GMADL as "Companion / extended formulation", which reads like a placeholder rather than a bibliographic entry.
- Lines 21-23 say auxiliary references are "named in text without bracket citation", but `[A1]` and `[A4]` are now actually bracket-cited. Update the note, or better, merge auxiliary references into the same citation scheme before LaTeX conversion.
- `[A2]`, `[A3]`, `[A5]`, `[A6]`, `[A7]`, and `[A8]` appear in the bibliography but are not bracket-cited in the revised chapter prose. The MTH301 rubric explicitly rewards all bibliography items being used.

Why this matters: this is the most likely remaining low-effort score loss under the 5% "Bibliography and Citations" category.

### 3. Chapter 6 still overstates the normalisation probe in an RQ heading

File: `paper/已修改/chapter6_conclusion.md`

Issue:
- Line 11 asks: "Are the observed winners robust to loss-component normalisation?" This should match Chapter 1's improved wording: "approximately stable under the diagnostic loss-component normalisation probe."

Suggested revision:
- `**RQ3: Are the observed winners approximately stable under the diagnostic loss-component normalisation probe?**`

Why this matters: the paragraph itself is careful, but headings are high-salience claims and can be quoted by a marker.

### 4. Chinese abstract needs light polish for accuracy and naturalness

File: `paper/已修改/abstract.md`

Issues:
- Line 19: "导致模型以组合不使用的校准质量换取排名质量的损失" is awkward and can be read backwards. The English says the model sacrifices ranking quality for calibration quality the portfolio does not use.
- Line 23: "归一化探测" is understandable, but "诊断性归一化探测" better matches the English caveat.

Suggested revision:
- Replace the end of line 19 with: "导致模型牺牲组合所需的排序质量，换取组合并不直接使用的校准质量。"
- Use "诊断性归一化探测" where the English says diagnostic normalisation probe.

### 5. Final assembly source is still unresolved

Operational issue:
- Revised manuscript files are under `paper/已修改/`.
- Root `paper/*.md` still contain stale text and old placeholder-related issues.

Fix before export:
- Either copy accepted `paper/已修改/*.md` into root `paper/*.md`, or configure the final LaTeX/Markdown assembly to read from `paper/已修改/`.

Why this matters: a technically clean revised draft can still fail if the final PDF compiles the stale root files.

## Lower-priority polish

1. `paper/已修改/chapter2_literature_review.md:37` says MSE, MedSE, and absolute-loss family are evaluated under "identical protocol." This is likely intended within the baseline comparison, but "same Chapter 5 protocol" would be clearer.
2. `paper/已修改/chapter6_conclusion.md:51` says if gamma07 continues to "dominate" across feature sets. "Remain best-supported" is less rhetorically strong.
3. The abstract is now long in both English and Chinese. This is acceptable for a final report, but if page limits become tight, shorten the boundaries paragraph rather than deleting claim caveats.

## Issues now resolved

- Chinese abstract exists.
- No `FIGURE PLACEHOLDER`, `TODO`, `==...==`, `full-batch`, `gamma10_stable_zone`, `0.0356`, or `0.1151` markers remain in `paper/已修改/abstract.md`, chapter files, or appendix files.
- All expected figure image files exist under `paper/figures/`.
- Chapter 1 no longer calls the multiplicative family the strongest hybrid family.
- Chapter 2 now separates seed-42 Sharpe comparison from Phase 2 multi-seed stability.
- Chapter 3 no longer has process-style duplicate wording at the opening and no longer has highlighted headings.
- Chapter 5 now says CV only where multi-seed evidence exists and splits the final evidence claim correctly.
- Appendix B now separates main-branch reproduction from Phase 2 branch/artifact reproduction.

## Updated provisional score

Estimated current score under the MTH301 marking scheme: **77 / 100**.

Rationale: the report now satisfies the required bilingual abstract condition and has stronger claim discipline than the previous 74/100 draft. It remains below the high-first-class range because bibliography consistency, source-of-truth cleanup, final assembly source selection, and a few high-salience headings still need tightening.
