# FYP Final Report — `paper/` Workspace

Standalone final-report drafting workspace. Each chapter is a separate Markdown file; the full report is assembled by concatenation later (LaTeX conversion is out of scope for now).

## Scope

- Topic: loss-function design for cross-sectional stock-return prediction with a multi-layer perceptron (MLP) and long-short portfolio construction.
- Evidence boundary: this workspace cites only numbers that pass the evidence gate defined in `SCHEMA.md` and summarised in `paper/results_source_of_truth.md`.
- Writing language: English. Formatting: Obsidian-compatible Markdown with fenced math (`$...$` / `$$...$$`) and fenced code blocks.
- Not a phase report. Do not use "Semester 1 progress", "previous report", "update", "Phase 2.5 proves", "exact replication" style framing.

## Chapter Map

| File | Content | Primary evidence/sections |
|------|---------|---------------------------|
| `chapter1_introduction.md` | Background, motivation, research gap, research questions, objectives, contributions, scope, chapter outline | aligns with final claims in Ch5 and methodology in Ch3 |
| `chapter2_literature_review.md` | Robust regression for financial returns, heavy-tailed returns, MSE/MedSE/Huber, MADL/GMADL family, cross-sectional ML, portfolio construction | strengthen robust-loss and MADL/GMADL rationale |
| `chapter3_methodology.md` | Research design, MLP architecture, loss family (MSE/MedSE/MADL/GMADL/IMADL/hybrid add/hybrid mul/m2-robust gamma), training protocol, portfolio construction, evaluation metrics, experimental phases, reproducibility/claim boundaries | `best_hyperparameters.txt`, `Model_Train/*.py`, runner scripts, `doc/phase2.5/07_loss_implementation_details.md` |
| `chapter4_data.md` | Data source, sample construction, train/test split, feature variables (X1), preprocessing (cross-sectional z, turnover scaling, winsorisation where applied), limitations | split from old Ch3; cross-check `Model_Train/features.py`, `data_preprocess.py` |
| `chapter5_empirical_results_discussion.md` | Baseline 24m results (seed 42), Phase 1.5 variant sweep (seed 42), Phase 2 robustness across seeds, gamma refinement, IMADL α sweep fallback, normalisation check, discussion | `paper/results_source_of_truth.md` |
| `chapter6_conclusion.md` | Answers to RQs, limitations, future work, no new evidence | synthesises Ch5 and methodology claims |
| `references.md` | All bracket citations used by the report | ensures 1:1 mapping |

Supporting files:

- `paper/results_source_of_truth.md` — verified numbers and their provenance (the ONLY numeric source every chapter is allowed to cite).
- `paper/evidence_map.md` — short pointer index for subagents.
- `paper/figures/` — generated figures with their source CSV path and regeneration script recorded in captions.

## Writing Order (per `SCHEMA.md` workflow)

1. Build `results_source_of_truth.md` (done).
2. Draft Chapter 5 first; its claims drive what earlier chapters must support.
3. Draft Chapter 3 methodology against runner/config reality.
4. Draft Chapter 4 data (split from old methodology).
5. Update Chapter 1 and Chapter 2 to match final claims.
6. Write Chapter 6 conclusion last.
7. Assemble `references.md` and consistency sweep.

## Claim Strength Policy

- Strong claim: same runner, same window, verified CSVs. Use for baseline 24m table and gamma-refinement multi-seed table.
- Moderate claim: same window, different λ/γ/α or seed set. Use for cross-phase comparisons with explicit label.
- Weak/contextual: diagnostics (Phase 2.1b/Phase 2.5), normalization probe, early 6m sanity checks. Only as supporting prose, not headline tables.

## Allowed Final Headlines

Recommended (preserved across multi-seed evaluation):

- Primary recommendation: `m2_robust_gamma07`, mean Sharpe `0.9156`, CV `0.1808`, mean cumulative return `0.2799`.
- High-return alternative: `m2_robust_gamma10`, mean Sharpe `1.0043`, CV `0.5613` (explicit seed-sensitivity caveat).
- Stable fallback: `imadl_m2_alpha06`, mean Sharpe `0.6895`, CV `0.2443`, mean cumulative return `0.3042` (from `phase2.2-fix` integrated summary).
- Normalisation message: not a universal fix. `gamma07_normalized ≈ 0.9112` (~flat), `gamma10_normalized ≈ 0.4072`, `alpha06_normalized ≈ -0.0161`.

Forbidden (stale):

- MedSE Sharpe `2.68`, MSE Sharpe `0.37` as final headlines.
- `gamma07 CV=0.0356`, `gamma10 CV=0.1151`.
- LSTM architecture, batch size 256, 50 epochs, early stopping, 27 features, tanh.
- Single global seed set claim.
- "Normalisation failed across all losses" absolute statement.

## Session Continuity (ralph-loop style)

When context is near-full:

1. `git add paper/ && git commit -m "paper: <what changed this session>"` on `main`.
2. `/chat save` to persist the session.
3. `chat.disableAutoCompaction false` so the next session can resume without aggressive compaction.
4. Record the stopping point in `paper/session_log.md` (create if missing) so the next agent knows which tasks remain.
