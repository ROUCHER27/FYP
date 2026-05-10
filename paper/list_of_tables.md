# List of Tables

Numbered tables in every chapter and appendix of the final report. Section references point to the subsection that contains each table.

## Chapter 3 — Methodology

| Table | Title | Section |
|---|---|---|
| Table 3.1 | Phase 1.5 additive A-series loss hyperparameters ($\lambda_{\mathrm{dir}}$, $\lambda_{\mathrm{hub}}$) | §3.3.3 |
| Table 3.2 | Phase 1.5 multiplicative M-series loss hyperparameters ($\lambda_{\mathrm{dir}}$) | §3.3.3 |

## Chapter 4 — Data

| Table | Title | Section |
|---|---|---|
| Table 4.1 | CRSP-style monthly panel schema: columns and source roles | §4.1 |
| Table 4.2 | Static train/test split (60 training months + 24 test months) | §4.3 |

## Chapter 5 — Empirical Results and Discussion

| Table | Title | Section |
|---|---|---|
| Table 5.1 | Baseline loss comparison (seed 42, 24 months) | §5.2 |
| Table 5.2 | Phase 1.5 additive and multiplicative variants (seed 42, 24 months) | §5.3 |
| Table 5.3 | Phase 2.2 γ refinement grouped summary (3 seeds per γ) | §5.4 |
| Table 5.4 | Phase 2 integrated summary across α, β, λ families (selected rows, 3 seeds each) | §5.5 |
| Table 5.5 | Loss-component normalisation probe: scale ratios and normalised Sharpes | §5.6 |

## Appendix B — Per-Seed Raw Results and Reproducibility

| Table | Title | Section |
|---|---|---|
| Table B.1 | Phase 2.2 γ refinement — per-seed annualised Sharpe across `{42, 52, 62}` | §B.1 |
| Table B.2 | Phase 2.2-fix1 normalisation probe — per-seed normalised Sharpes | §B.2 |

## Notes

- Every table in the list above has a bold caption header (`**Table X.Y — ...**`) in the source Markdown; a LaTeX pass can promote each to a `\begin{table}` block with the matching number.
- This list and the `list_of_figures.md` counterpart are conventional aids; they are trivially regenerable by grepping `^\*\*Table` / `^\*\*Figure` across `paper/*.md`.
