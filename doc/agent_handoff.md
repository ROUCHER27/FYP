# Agent Handoff For Final Report Work

This repo is intended to let a new agent continue from `main` without switching branches for normal writing work.

## Required Reading Order

1. `SCHEMA.md`
2. `doc/thesis/window_provenance_audit.md`
3. `doc/thesis/final_report_requirements_gap_analysis.md`
4. `doc/thesis/thesis_revision_guide.md`
5. Current chapter drafts under `doc/thesis/`
6. Evidence files listed below

## Source-Of-Truth Evidence

- Baseline and Phase 1.5 same-window single-seed evidence: `doc/final_report_all_24m_evidence/`
- Prior MSE/MedSE 24-month cross-check: `doc/final_report_24m_baselines/`
- Phase 2.1b alignment diagnostics: `doc/phase2-fix/phase2_1b/reports/`
- Phase 2.2 gamma refinement: `doc/phase2-fix/phase2_2/gamma_refinement/reports/`
- Phase 2.2 normalization check: `doc/phase2-fix/phase2.2-fix1/`
- Phase 2.5 alignment diagnostics: `doc/phase2.5/` and `phase2.5-对齐失败完整诊断报告.md`

## Cross-Branch Evidence

The local gamma-refinement grouped summary is a 5-row gamma-only table. The `imadl_m2_alpha06` fallback result comes from the integrated Phase 2 summary on `phase2.2-fix`.

Use this without switching branches:

```bash
git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv
```

Current fallback values from that branch:

- `imadl_m2_alpha06`: mean Sharpe `0.6895`, CV `0.2443`, mean cumulative return `0.3042`.
- `m2_robust_gamma10`: mean Sharpe `1.0043`, CV `0.5613`, mean cumulative return `0.2368`.

## Writing Boundaries

- Main same-window baseline and Phase 1.5 tables are seed `42`, not multi-seed robustness evidence.
- Multi-seed robustness claims must cite Phase 2 grouped summaries.
- Do not use old 6-month sanity checks as final headline results.
- Do not reuse old thesis claims about MedSE Sharpe `2.68`, MSE Sharpe `0.37`, `gamma07` CV `0.0356`, LSTM, batch size `256`, or 50 epochs.

## Commit Discipline

Use small commits after each coherent unit:

1. Context/schema or handoff updates.
2. Draft chapter changes.
3. Figure/table generation scripts and outputs.
4. Evidence ingestion or verification updates.

Before handing off, run:

```bash
git status --short --branch
rg -n "1995-06|2.68|0.37|0.0356|0.1151|LSTM|256|50 epochs|early stopping|previous report|Semester 1 progress|exact replication" doc/thesis SCHEMA.md
```
