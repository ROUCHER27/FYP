# List of Figures

Numbered figures in every chapter of the final report. Section references point to the subsection that contains each figure; the regeneration column points at the Python script under `paper/figures/` that produces the PNG.

| Figure | Title | Section | Regeneration script |
|---|---|---|---|
| Figure 2.1 | Conceptual loss-function shape comparison ($y = 0.05$) | §2.4 | `paper/figures/plot_loss_shapes.py` |
| Figure 3.1 | Portfolio construction pipeline (schematic) | §3.5 | `paper/figures/plot_portfolio_flow.py` |
| Figure 4.1 | Data coverage timeline and heavy-tail `RET` histogram | §4.1 | `paper/figures/plot_data_coverage.py` |
| Figure 5.1 | Baseline cumulative long-short return + Sharpe (seed 42) | §5.2 | `paper/figures/plot_baseline_comparison.py` |
| Figure 5.2 | Phase 1.5 additive and multiplicative variants (Sharpe, seed 42) | §5.3 | `paper/figures/plot_phase15_variants.py` |
| Figure 5.3 | Phase 2.2 γ refinement multi-seed Sharpe and CV (3 seeds) | §5.4 | `paper/figures/plot_gamma_refinement.py` |
| Figure 5.4 | IMADL-m2 α sweep vs γ reference (3 seeds, integrated summary) | §5.5 | `paper/figures/plot_integrated_sweep.py` |
| Figure 5.5 | Loss-component normalisation probe (original vs normalised, 3 seeds) | §5.6 | `paper/figures/plot_normalisation_probe.py` |

## Notes

- Every figure is produced at `dpi=300` on STIX serif using the shared paper-plot style in `paper/figures/_style.py`.
- Paper-plot-skills style mapping: bar_paired_delta (Fig 5.5), bar_grouped_hatch ablation (Fig 5.3, also 5.1 right panel, 5.2), line_confidence_band Type A continuous (Fig 5.1 left panel), line_confidence_band Type B scaling (Fig 5.4), line_training_curve framed (Fig 2.1). Fig 3.1 (flow schematic) and Fig 4.1 (Gantt timeline + histogram) are custom compositions but follow the shared serif / palette conventions.
- Data provenance per figure is recorded in the figure caption embedded in the corresponding chapter and in the header docstring of each plotting script.
