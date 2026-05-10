# Table of Contents

Assembly order of the final report. Every entry links to the Markdown file in this workspace; a LaTeX conversion pass can replace each link with an auto-generated `\tableofcontents` entry.

## Front matter

- [Abstract](abstract.md)
- [List of Figures](list_of_figures.md)
- [List of Tables](list_of_tables.md)
- [AI-assisted Technology Disclosure](ai_use_disclosure.md) *(to be drafted)*
- [Acknowledgements](acknowledgements.md) *(to be drafted)*
- [Declaration of Authorship](declaration_of_authorship.md) *(to be drafted)*

## Chapter 1 — Introduction — [`chapter1_introduction.md`](chapter1_introduction.md)

- 1.1 Background and motivation
- 1.2 Research gap
- 1.3 Research objectives
- 1.4 Research questions
- 1.5 Scope and claim boundaries
- 1.6 Contributions and report structure

## Chapter 2 — Literature Review — [`chapter2_literature_review.md`](chapter2_literature_review.md)

- 2.1 Machine learning for cross-sectional stock-return prediction
- 2.2 Heavy-tailed returns and the case for robust losses
- 2.3 Traditional regression losses under portfolio objectives
- 2.4 Directional and trading-aware losses: MADL, GMADL, IMADL
  - *Figure 2.1 — Conceptual loss-function shape comparison*
- 2.5 Hybrid loss design and the M2-robust family
- 2.6 Validation, overfitting, and multiple testing
- 2.7 Research gap and positioning

## Chapter 3 — Methodology — [`chapter3_methodology.md`](chapter3_methodology.md)

- 3.1 Research design
- 3.2 Model architecture
- 3.3 Loss function families
  - 3.3.1 Regression losses
  - 3.3.2 Directional losses (MADL, GMADL, IMADL)
  - 3.3.3 Hybrid losses: additive and multiplicative
    - *Table 3.1 — Phase 1.5 additive A-series loss hyperparameters*
    - *Table 3.2 — Phase 1.5 multiplicative M-series loss hyperparameters*
  - 3.3.4 M2-robust γ family and related Phase 2 parameterisations
- 3.4 Training protocol
- 3.5 Portfolio construction
  - *Figure 3.1 — Portfolio construction pipeline*
- 3.6 Evaluation metrics
- 3.7 Experimental phases and evidence configuration
- 3.8 Reproducibility and claim boundaries

## Chapter 4 — Data — [`chapter4_data.md`](chapter4_data.md)

- 4.1 Data source
  - *Table 4.1 — CRSP-style monthly panel schema*
  - *Figure 4.1 — Data coverage and training-era return distribution*
- 4.2 Sample construction
- 4.3 Train/test split
  - *Table 4.2 — Static train/test split*
- 4.4 Feature variables
  - 4.4.1 X1 — cumulative return and cumulative turnover (used throughout)
  - 4.4.2 X2 — normalised momentum excluding the most recent month
  - 4.4.3 X3 — twelve lagged normalised monthly returns
- 4.5 Preprocessing
- 4.6 Data limitations

## Chapter 5 — Empirical Results and Discussion — [`chapter5_empirical_results_discussion.md`](chapter5_empirical_results_discussion.md)

- 5.1 Evaluation protocol recap
- 5.2 Baseline loss comparison (24 months, seed 42)
  - *Table 5.1 — Baseline loss comparison*
  - *Figure 5.1 — Baseline cumulative return and Sharpe*
- 5.3 Phase 1.5 variant sweep (24 months, seed 42)
  - *Table 5.2 — Phase 1.5 A/M variants*
  - *Figure 5.2 — Phase 1.5 hybrid variants (A/M)*
- 5.4 Phase 2: multi-seed robustness of the M2-robust family
  - *Table 5.3 — γ refinement grouped summary*
  - *Figure 5.3 — γ refinement multi-seed Sharpe + CV*
- 5.5 Phase 2: broader α, β, and λ sweeps (integrated summary)
  - *Table 5.4 — Phase 2 integrated summary*
  - *Figure 5.4 — IMADL-m2 α sweep vs γ reference*
- 5.6 Loss-component normalisation probe (Phase 2.2-fix1)
  - *Table 5.5 — Loss-component normalisation probe*
  - *Figure 5.5 — Normalisation probe: original vs normalised Sharpe*
- 5.7 Alignment diagnostics and claim boundaries
- 5.8 Headline findings and recommendations

## Chapter 6 — Conclusion — [`chapter6_conclusion.md`](chapter6_conclusion.md)

- 6.1 Summary of findings
- 6.2 Limitations
- 6.3 Future work
- 6.4 Closing statement

## Appendices

### Appendix A — Loss Function Definitions and Gradients — [`appendix_A_loss_definitions.md`](appendix_A_loss_definitions.md)

- A.1 Regression losses
  - A.1.1 MSE
  - A.1.2 MedSE
  - A.1.3 Huber (as magnitude backbone)
- A.2 Directional losses
  - A.2.1 MADL (differentiable)
  - A.2.2 GMADL
  - A.2.3 IMADL (rebalanced)
- A.3 Hybrid losses
  - A.3.1 Additive (A-series)
  - A.3.2 Multiplicative (M-series)
  - A.3.3 M2-robust γ family
- A.4 Notes on practical implementation

### Appendix B — Per-Seed Raw Results and Reproducibility — [`appendix_B_per_seed_raw.md`](appendix_B_per_seed_raw.md)

- B.1 Phase 2.2 γ refinement — per-seed annualised Sharpe
  - *Table B.1 — γ refinement per-seed Sharpe*
- B.2 Phase 2.2-fix1 normalisation probe — per-seed annualised Sharpe
  - *Table B.2 — Normalisation probe per-seed Sharpe*
- B.3 Phase 2 integrated summary — grouped row summary (reproduction)
- B.4 Reproduction commands per Chapter 5 table
- B.5 Reproducibility checklist
- B.6 Known numerical caveats

## Back matter

- [References](references.md)
  - Main bibliography: `[1]..[8]`
  - Auxiliary references (named in text): `[A1]..[A8]`

## Notes

- Entries marked *(to be drafted)* in Front matter are outstanding tasks from the current todo list (A1/A2/A3).
- Numbered section headings, table captions (`**Table X.Y — ...**`), and figure captions (`**Figure X.Y — ...**`) in the source Markdown map 1:1 to this table of contents.
