# Session Log (ralph-loop continuity)

Track what each writing session completed so the next session can resume cleanly. Append in reverse chronological order (newest first).

## Session conventions

- Keep entries short: date, tasks completed, files touched, open follow-ups, and the next task to pick up.
- When a session ends near the context limit, before `/chat save`: commit on `main`, record the commit hash here, and list the first task the next session should run.

## 2026-05-11 — Session 3 (placeholders + abstract + appendices)

Continuation session. Added figure placeholders (with chart-type specs) for the charts that still need to be implemented manually, wrote the report's Abstract, and added two technical appendices.

### Commits added this session

1. `cabaf11` paper: add figure placeholders (Fig 2.1 loss shapes, 3.1 portfolio flow, 4.1 data timeline, 5.1 baseline, 5.2 Phase1.5) with chart specs for later implementation
2. `60c8e27` paper: add Abstract with primary/alt/fallback recommendation, scope, and keywords
3. `5c4dbd9` paper: add Appendix A (loss function definitions and pointwise gradients)
4. `a62eac3` paper: add Appendix B (per-seed raw Sharpes + reproduction commands + checklist)

### Figure placeholders (chart specs inlined as HTML comments for later implementation)

| Figure | Chapter | Chart type | Source |
|---|---|---|---|
| Fig 2.1 | Ch2 §2.4 | 4-panel didactic loss-shape curves (MSE/Huber/MedSE/MADL/GMADL/hybrid_mul) | synthetic; evaluate formulas from Ch3 §3.3 |
| Fig 3.1 | Ch3 §3.5 | flow diagram of the portfolio construction pipeline | schematic only |
| Fig 4.1 | Ch4 §4.1 | two-panel (CSV coverage timeline + log-scale training-era RET histogram) | `*.csv` at repo root |
| Fig 5.1 | Ch5 §5.2 | two-panel (baseline cumulative returns line + Sharpe bar) | `doc/final_report_all_24m_evidence/results/baseline/*` |
| Fig 5.2 | Ch5 §5.3 | grouped horizontal bar (A1–A5 + M1–M4 Sharpe) | `doc/final_report_all_24m_evidence/results/phase15/*` |

Figures 5.3 / 5.4 / 5.5 were already generated in Session 2 and embedded in Ch5.

### New markdown files

- `paper/abstract.md` (618 words) — single-page abstract summarising problem, method, findings, tiered recommendation, and scope.
- `paper/appendix_A_loss_definitions.md` (1084 words) — closed-form definitions and pointwise gradients for every loss in `Model_Train/losses.py`: MSE, MedSE, Huber (backbone), MADL, GMADL, IMADL, additive and multiplicative hybrid, M2-robust γ.
- `paper/appendix_B_per_seed_raw.md` (919 words) — per-seed raw Sharpe tables for γ refinement and the normalisation probe; exact reproduction commands for every Ch5 table; reproducibility checklist and known caveats.

### Verification

- Per-seed values in Appendix B §B.1 cross-checked to 5 decimals against `doc/phase2-fix/phase2_2/gamma_refinement/reports/phase2_raw_runs.csv`; aggregated means reproduce the grouped-summary values.
- Forbidden-term sweep on abstract + appendices passes (the single `1995-06` hit in Appendix B §B.6 is an explicit negation of using the 6-month window, consistent with SCHEMA.md §2.2.1).
- README chapter map updated to include `abstract.md`, `appendix_A_*`, `appendix_B_*`, and to document the figure placeholder list.

### Remaining optional follow-ups

- Implement Fig 2.1 / 3.1 / 4.1 / 5.1 / 5.2 per the inline chart specs.
- LaTeX conversion after the thesis template is selected (explicitly deferred by the user).

## 2026-05-11 — Session 2 (figures + audit + references expansion)

Continuation session. Added the three Chapter 5 figures, expanded `references.md` with auxiliary entries, embedded figure blocks into Chapter 5, and ran a formal audit against `SCHEMA.md` high-risk claims.

### Commits on main added this session

1. `b10946f` paper: embed figures 5.3/5.4/5.5 into Chapter 5 with full captions
2. `a45b03a` paper: add figures 5.3 (gamma refinement), 5.4 (IMADL alpha sweep), 5.5 (normalisation probe) and their scripts
3. `89b729a` paper: expand references.md with auxiliary entries (Huber, Adam, Dropout, CRSP, PyTorch, NumPy, pandas, Matplotlib)

### New paper artifacts

- `paper/figures/fig5_3_gamma_refinement.png` (γ sweep mean Sharpe ± std + CV)
- `paper/figures/fig5_4_imadl_alpha_sweep.png` (IMADL-m2 α sweep vs γ07/γ10 reference)
- `paper/figures/fig5_5_normalisation_probe.png` (original vs normalised Sharpe + per-seed dots)
- `paper/figures/plot_gamma_refinement.py`, `plot_integrated_sweep.py`, `plot_normalisation_probe.py`
- `references.md` extended with `[A1]`–`[A8]` auxiliary entries for Huber (1964), Adam (Kingma & Ba, 2014), Dropout (Srivastava et al., 2014), CRSP data source, PyTorch, NumPy, pandas, and Matplotlib.

### Formal audit against `SCHEMA.md` high-risk claims

| Check | Target | Result |
|---|---|---|
| Stale MedSE 2.68 / MSE 0.37 as headline | Chapters 1–6 | no hits |
| Stale gamma07 CV 0.0356 / gamma10 CV 0.1151 | Chapters 1–6 | no hits |
| LSTM as architecture / batch 256 / 50 epochs / 27 features / tanh as activation | Chapters 1–6 | no hits (tanh occurrences are all inside MADL/GMADL math blocks; LSTM is a single literature-background word in Ch2 §2.1) |
| Phase-report framing (Semester 1, previous report, exact replication, Phase 2.5 proves) | Chapters 1–6 | no hits |
| Absolute wording "normalisation failed across all losses" | Chapters 1–6 | no hits |
| 6-month window treated as headline | Chapters 1–6 | three hits, all explicit negation/labelling as preliminary sanity check (per SCHEMA.md §2.2.1) |
| Every Table 5.N carries window, seed set, and source path | Ch5 | Tables 5.1–5.5 all pass |
| Final recommendation is tiered with caveats | Ch5 §5.8, Ch6 §6.1 | primary / high-return alt / stable fallback all present with seed-sensitivity caveats |
| Numeric consistency for headline values | across chapters + source-of-truth | 0.9156 (14), 0.1808 (10), 0.2799 (5), 1.0043 (13), 0.5613 (9), 0.6895 (12), 0.2443 (9), 0.3042 (3), 0.9112 (5), 0.4072 (4), -0.0161 (4) |

No SCHEMA violations detected. No changes required in the chapter prose.

### Remaining optional follow-ups

- Figure polish: Chapter 5 §5.2 and §5.3 currently have only tables; adding a baseline-vs-Phase-1.5 Sharpe bar chart would help the reader but is not required by SCHEMA.
- Reviewer pass by Codex (auditor role per `AGENTS.md` skills section) when available.
- LaTeX conversion after the thesis template is selected.

## 2026-05-10 — Session 1 (complete)

**Status:** Initial full draft of all six chapters plus references, scaffolding, and source-of-truth committed to `main`. Report is internally consistent; evidence gate (`SCHEMA.md`, writing-rules) scan passes — all flagged terms are either math (`tanh` in MADL formulas), literature-context mentions (`LSTM` among ML architectures), explicit negations (`no early stopping`, "older 6-month window … not used as headline"), or meta reminder lists in `README.md` / `evidence_map.md` / `results_source_of_truth.md`.

**Commits on main (most recent first):**

1. `6a7217b` paper: draft Chapter 6 conclusion and references
2. `77d1c87` paper: draft Chapter 1 introduction
3. `f755413` paper: draft Chapter 3 methodology
4. `63ec6af` paper: correct Sharpe convention to annualised sqrt(12) in ch5 and source-of-truth
5. `fe0db7e` paper: draft Chapter 2 literature review
6. `7563c18` paper: draft Chapter 4 data
7. `a357726` paper: draft Chapter 5 empirical results and discussion
8. `5187e98` paper: scaffold final-report workspace with evidence source-of-truth

`main` is 8 commits ahead of `origin/main`. Not pushed.

**Word counts (chapters + references):**

| File | Words |
|---|---:|
| chapter1_introduction.md | 1968 |
| chapter2_literature_review.md | 3050 |
| chapter3_methodology.md | 3623 |
| chapter4_data.md | 1793 |
| chapter5_empirical_results_discussion.md | 3958 |
| chapter6_conclusion.md | 1649 |
| references.md | 379 |
| **Total** | **16420** |

**Key mid-session correction.** Commit `63ec6af` fixed the Sharpe annualisation statement in Chapter 5 §5.1 and in `results_source_of_truth.md` §0: the reported Sharpe values are annualised by $\sqrt{12}$ from monthly mean/std (verified numerically against `hybrid_mul_m1` and `medse`), not monthly-frequency as initially stated. No numeric values changed; only the description.

**Notes for future work.**

- `figures/` directory exists but is empty. Chapter 5 contains `TODO: generate from …`-style placeholders where figures would help. Figure generation is a separate follow-up.
- `references.md` is a Markdown bibliography; a LaTeX conversion pass can replace bracket numbers with `\cite{}` calls without changing content.
- Subagent tooling returned `No result` in this session; the entire draft was written directly. A later session could try running the formal `kiro_default` audit pass once that tooling is functional.

**Next session priorities (if any).**

1. Optional: auditor pass against `SCHEMA.md` / `doc/agent_handoff.md` to double-check claim strengths.
2. Optional: generate figures for Ch5 Tables 5.3 and 5.4 (γ refinement bar chart with error bars; integrated summary dot plot).
3. Optional: fill in additional bibliography entries (Huber 1964 and companion robust-regression references) once thesis template is chosen.
4. School-format/LaTeX conversion happens only after the above are done; it is out of scope for the Markdown workspace.

## 2026-05-10 — Session 1 (in progress, earlier)

- Created `paper/` skeleton: `README.md`, `evidence_map.md`, `results_source_of_truth.md`, `figures/`.
- `results_source_of_truth.md` aggregates: baseline 24m (seed 42), Phase 1.5 A/M variants (seed 42), Phase 2.2 gamma refinement multi-seed, Phase 2 integrated summary from `phase2.2-fix` branch (IMADL α sweep, adaptive λ, IMADL-GMADL β), Phase 2.2-fix1 normalisation probe.
- Next: dispatch Stage A subagents for Chapter 5 (empirical), Chapter 4 (data), Chapter 2 (literature) in parallel.
