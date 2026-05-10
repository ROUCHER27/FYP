# Evidence Map (for chapter drafters)

All paths are relative to repo root `/Users/roucher/Documents/FYP`. Use this as a pointer index; for actual numbers cite `paper/results_source_of_truth.md`.

## Governance (always-read)

- `SCHEMA.md` — final-report contract.
- `doc/agent_handoff.md` — source-of-truth evidence map + cross-branch lookup.
- `doc/thesis/window_provenance_audit.md` — 24-month window decision.
- `doc/thesis/final_report_requirements_gap_analysis.md` — global issues.
- `doc/thesis/thesis_revision_guide.md` — chapter rewrite checklist.
- `.kiro/skills/fyp-final-report/references/{final-report-workflow,evidence-map,thesis-audit,writing-rules}.md`.

## Source-of-truth results (write-facing)

- `paper/results_source_of_truth.md` — the ONLY allowed numeric source. Cite it everywhere.

## Primary evidence (read-facing)

Baseline + Phase 1.5 single-seed 24m:
- `doc/final_report_all_24m_evidence/reports/final_report_all_24m_evidence_status.csv`
- `doc/final_report_all_24m_evidence/manifests/run_manifest.json`
- `doc/final_report_all_24m_evidence/results/*/*/sanity_summary_*.json`
- `doc/final_report_all_24m_evidence/results/*/*/sanity_metrics_*.csv`
- `doc/final_report_all_24m_evidence/manifests/*_{command.txt,status.json,verification.json}`
- `doc/final_report_all_24m_evidence/logs/*.log`
- Branch/commit: `main @ 6c0fbde`. Seed `42`. Train `1990-01..1994-12`. Test `1995-01..1996-12` (24 rows).

Phase 2.2 gamma refinement (multi-seed):
- `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_grouped_summary.csv`
- `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv`
- `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_summary_report.txt`

Phase 2 integrated summary with IMADL α sweep (on `phase2.2-fix` branch):
- `git show phase2.2-fix:doc/phase2-fix/reports/phase2_grouped_summary.csv`

Phase 2.1b alignment diagnostics:
- `doc/phase2-fix/phase2_1b/reports/phase21b_vs_phase15_grouped.csv`
- `doc/phase2-fix/phase2_1b/reports/phase21b_vs_phase15_raw.csv`

Phase 2.2-fix1 normalisation probe:
- `doc/phase2-fix/phase2.2-fix1/phase1_summary.json`
- `doc/phase2-fix/phase2.2-fix1/phase2_summary.json`

Phase 2.5 alignment diagnostics (short claim-boundary subsection only):
- `doc/phase2.5/01_config_comparison.md` … `doc/phase2.5/07_loss_implementation_details.md`
- `doc/phase2.5/executive_summary.md`
- `phase2.5-对齐失败完整诊断报告.md`

## Code / config evidence (methodology)

- `best_hyperparameters.txt`
- `Model_Train/models.py` — MLP architecture
- `Model_Train/losses.py` — MSE / MedSE / MADL / GMADL / IMADL / hybrid_add / hybrid_mul / m2-robust / adaptive
- `Model_Train/features.py` — X1 / X2 / X3
- `Model_Train/data_preprocess.py` — sample + split + preprocessing
- `sanity_check_core.py`, `sanity_check_signal_tilted.py` — portfolio construction + metrics
- Runner scripts: `run_sanity_check_{mse,medse,madl,gmadl,imadl,hybrid_mul_m[1-4],hybrid_add_a[1-5]}.py`

## Forbidden numbers / framings (will be flagged by consistency sweep)

- MedSE Sharpe `2.68`; MSE Sharpe `0.37` (as final headlines).
- `gamma07 CV=0.0356`; `gamma10 CV=0.1151`.
- LSTM, batch size 256, 50 epochs, early stopping, 27 features, tanh.
- "Previous report", "Semester 1 progress", "update", "exact replication", "Phase 2.5 proves".
- Treating `doc/final_report_all_24m_evidence/` as multi-seed evidence.
