# Shared Revision Playbook (applied by every subagent to every file)

Use this alongside the relevant `paper/待修改/<file>_review.md` when revising `paper/<file>` → `paper/已修改/<file>`. Do not change numbers that appear in `paper/results_source_of_truth.md`.

## R1. X1 input dimension — authoritative description

`build_feature_set_x1(panel)` produces 10 engineered columns (5 horizons × 2 variables: cumulative return `cr_Wm`, cumulative turnover `co_Wm` for $W \in \{1,3,6,9,12\}$). `assemble_feature_matrix()` then keeps every column that is not `{PERMNO, date, target_ret}`; this drops 3 of the 18 panel columns and passes 15 columns to the MLP.

The authoritative 15 model-input columns (order as emitted by `assemble_feature_matrix`) are:

| # | Column | Description |
|---|---|---|
| 1 | `RET` | raw monthly total return for month $t$ |
| 2 | `VOL` | monthly trading volume (shares) for month $t$ |
| 3 | `SHROUT` | shares outstanding (thousands) for month $t$ |
| 4 | `r` | numeric copy of `RET` |
| 5 | `to` | monthly turnover $= \text{VOL}/(\text{SHROUT}\cdot 1000)$ |
| 6 | `cr_1m` | 1-month cumulative lagged return $\prod_{k=1}^{1}(1+r_{t-k})-1$ |
| 7 | `co_1m` | 1-month cumulative lagged turnover $\sum_{k=1}^{1}\text{to}_{t-k}$ |
| 8 | `cr_3m` | 3-month cumulative lagged return |
| 9 | `co_3m` | 3-month cumulative lagged turnover |
| 10 | `cr_6m` | 6-month cumulative lagged return |
| 11 | `co_6m` | 6-month cumulative lagged turnover |
| 12 | `cr_9m` | 9-month cumulative lagged return |
| 13 | `co_9m` | 9-month cumulative lagged turnover |
| 14 | `cr_12m` | 12-month cumulative lagged return |
| 15 | `co_12m` | 12-month cumulative lagged turnover |

Rule:
- Wherever the old text says "X1 is 15-dimensional cumulative-return and cumulative-turnover" or equivalent, correct it to "the X1 feature set contributes 10 engineered columns; the model input is 15 columns comprising these 10 X1 columns together with the 5 base panel columns (`RET`, `VOL`, `SHROUT`, `r`, `to`) that `assemble_feature_matrix` retains."
- Chapter 4 §4.4.1 gets the full 15-row table above (labelled "**Table 4.3 — 15 model-input columns consumed by `assemble_feature_matrix`.**"). Other chapters/abstract/appendix A/B cite Table 4.3 rather than repeating the full list.
- Do not rewrite the MLPConfig line `input_dim=15`; that is correct.

## R2. "Robustness" must be reserved for multi-seed rows

Single-seed (seed 42) tables cannot support robustness language. In revised prose:

| Old wording | Corrected wording |
|---|---|
| "do not produce robust positive Sharpes at the same single seed" | "do not produce economically meaningful positive Sharpes in the seed-42 same-window baseline" |
| "robust same-window baseline" | "same-window seed-42 baseline" |
| "robust multi-seed evidence" (when talking about 3-seed tables) | keep — this is legitimate |
| "robustness" applied to any Phase 1.5 / baseline row | replace with "single-seed Sharpe behaviour" or "same-window comparison" |

Multi-seed contexts (γ refinement, IMADL-m2 α sweep, normalisation probe) may use "robustness" and "seed-stability" freely with the explicit caveat "three seeds per row".

## R3. Remove all residual figure-placeholder HTML comments

The figures now exist and are embedded. Delete every `<!-- FIGURE PLACEHOLDER: ... -->` block and the "placeholder; see comment for chart spec" sentence. Keep the bold figure caption, the `![...](figures/fig_*.png)` embed, and the short paragraph that describes data source + generated-by script.

## R4. "Only the loss function varies" — scope tightening

Within a single phase table the loss is the only varying factor. Across phases (Phase 1.5 → Phase 2 γ refinement → integrated Phase 2 → normalisation probe) other factors change (seed set, λ/γ/α, IMADL formulation, branch implementation). Whenever the text now says "identical data / features / architecture / portfolio across every phase" or "only the loss varies across the report," replace with one of:

- "within each comparison table, the loss function is the only varying factor; cross-phase comparisons differ in additional dimensions (§3.8)."
- "under the same static 1995-01..1996-12 evaluation window and portfolio cap, each phase internally isolates the loss while Phase 2 also varies seed set and λ/γ/α (§3.8)."

## R5. References clean-up (Chapter 5 / references.md — out of scope here)

Reviews for `references.md` are tracked as a separate task; the per-file subagents should not invent new citations. Where prose adds a citation number, it MUST already exist in the current `references.md` (`[1]..[8]`, `[A1]..[A8]`). If a new citation is needed, mark with `[TODO-cite: ...]` and stop.

## R6. File-level "4 vs 5 phases"

Abstract (and anywhere else that counts phases) should distinguish the four comparison phases (baseline, Phase 1.5, Phase 2 γ refinement, integrated Phase 2) from the Phase 2.2-fix1 normalisation probe, which is a diagnostic follow-up to the three leading candidates rather than a fifth comparison phase. Rewrite "four phases" sections to explicitly state "four comparison phases plus one normalisation probe" or equivalent; do not silently claim five.

## R7. Other shared SciWrite fixes

- Prefer "commonly optimises" over "almost always optimises" unless backed by a survey count.
- Split any paragraph that runs longer than ~10 lines and covers more than one topic.
- Keep every numeric value in the text tied to a `results_source_of_truth.md` cell; if a revision needs a number that is not there, stop and mark `[NUM-NEEDS-VERIFY]`.

## R8. Formatting / output location

Each subagent writes to `paper/已修改/<original_basename>.md` preserving Obsidian Markdown (math in `$...$` / `$$...$$`, code fences, GitHub tables). Do not touch `paper/<original_basename>.md`. Do not run git. Finish by emitting a one-line summary `DONE: <n> sentences changed in <file>`.
