# MTH301 Marking Scheme Score Review - 2026-05-11

Source rubric: `/Users/roucher/Downloads/MTH301_FinalReports_MarkingScheme.pdf`.

Scope scored: revised draft in `paper/已修改/*.md`, plus `paper/references.md` and `paper/results_source_of_truth.md`. Engagement is not directly assessable from the manuscript, so the total below is provisional.

## Provisional score

Estimated report score: **74 / 100**.

Interpretation: currently in the low first-class / high 2:1 boundary depending on how strictly the marker penalises citation quality and residual claim-boundary issues. The manuscript has a strong structure, clear evidence gate, and substantial independent empirical work, but it still needs final claim-tightening, bibliography cleanup, Chinese abstract, and source-path consolidation.

## Rubric breakdown

| Criterion | Weight | Score | Rationale |
|---|---:|---:|---|
| Introduction | 10 | 8 | Clear motivation, objectives, research questions, and project relevance. Minor overclaim remains around "strongest hybrid family" and normalisation robustness wording. |
| Background | 10 | 7 | Relevant ML/asset-pricing/loss-function background with good project alignment. Citation quality is uneven and some empirical project-result discussion enters the literature review too early. |
| Scope and Depth | 10 | 8 | Wide and technically meaningful coverage: loss design, MLP protocol, portfolio construction, evidence gates, limitations. Depth is limited by one static test window and three-seed robustness. |
| Rigor and Correctness | 10 | 7 | Most numbers are traceable and limitations are explicit. Remaining risks: overbroad "only loss varies" phrasing, cross-seed stability comparisons against single-seed baselines, and stale `results_source_of_truth.md` claims. |
| Added Value and Own Contribution | 20 | 15 | The report adds real value through a systematic loss-function comparison, hybrid design sweep, multi-seed evaluation, and evidence map. It is empirical rather than theoretical, and robustness breadth remains limited. |
| Organization and Presentation | 10 | 8 | Six-chapter structure matches the project contract and reads logically. Remaining issues: `已修改` vs root source divergence, Obsidian `==...==` heading markers, and some path-heavy prose. |
| Clarity | 15 | 11 | Generally clear and precise, with strong tables and figure captions. Some paragraphs are still dense; several conclusion/result claims need splitting to avoid ambiguity. |
| Bibliography and Citations | 5 | 3 | Core references are relevant, but `[7]` and `[8]` remain informal, auxiliary references are not consistently cited inline, and bibliography formatting is not yet final. |
| Engagement | 10 | 6 provisional | Not assessable from the manuscript alone. I used a neutral "consistent engagement" assumption. Replace with the supervisor's actual engagement judgement if known. |

Total: **73 / 100 by arithmetic table sum**, rounded to **74 / 100** because the report's source-of-truth evidence map and empirical contribution are stronger than a typical 73-level draft. If engagement is marked higher, the total can move into the mid/high 70s; if bibliography and claim-boundary issues remain, it can fall to the high 60s.

## Marking-scheme-driven required fixes

### 1. Add Chinese abstract

Rubric requirement: "The final report must include an abstract in English, followed by its translation into Chinese."

Current issue:
- `paper/已修改/abstract.md` has no Chinese translation.

Fix:
- Add a faithful Chinese translation after the English abstract and before keywords.

### 2. Improve bibliography to secure the 4-5/5 band

Current issue:
- `paper/references.md:17-19` uses informal descriptions for MADL and GMADL.
- Auxiliary references `[A1]..[A8]` are described as "named in text without bracket citation", which risks "not all sources cited" under the rubric.

Fix:
- Replace `[7]` and `[8]` with full bibliographic records if available.
- Cite Huber, Adam, Dropout, CRSP, PyTorch/pandas/matplotlib where they are substantively used, or merge auxiliary references into a consistent bibliography style.
- Ensure every bibliography item is cited in the chapter body or remove it from the final bibliography.

### 3. Tighten rigor/correctness claims

Current issues:
- `paper/results_source_of_truth.md:136` still contains a single-seed robustness claim.
- `paper/已修改/chapter2_literature_review.md:142` and `paper/已修改/chapter5_empirical_results_discussion.md:213` blur seed-42 Sharpe comparisons with multi-seed stability comparisons.
- `paper/已修改/chapter3_methodology.md:7` and `paper/已修改/chapter6_conclusion.md:5` still overstate "only loss varies" across every run.

Fix:
- Use two-part wording throughout: seed-42 same-window comparisons for baselines/Phase 1.5; three-seed CV comparisons only for Phase 2 rows.

### 4. Improve presentation before final export

Current issues:
- `paper/已修改/chapter3_methodology.md:41`, `paper/已修改/chapter3_methodology.md:224`, and `paper/已修改/chapter4_data.md:105` contain `==...==` heading markup.
- Root `paper/*.md` remains stale while revised files are under `paper/已修改/`.

Fix:
- Remove Obsidian highlight markers from headings.
- Confirm the final assembly source path. The final PDF should not accidentally compile stale root files.

### 5. Strengthen first-class depth if time permits

Optional improvements:
- Add a compact limitations-to-validity paragraph tying static window, seed depth, no transaction costs, and feature restriction into one coherent validity framework.
- Add a short "what would falsify the recommendation" paragraph in Chapter 6 or Chapter 5. This would improve rigor because it shows the claim is bounded and testable.
- Move some project-result interpretation out of Chapter 2 and into Chapter 5, leaving the literature review more literature-driven.
