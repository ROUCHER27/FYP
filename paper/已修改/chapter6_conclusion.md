# Chapter 6: Conclusion

## 6.1 Summary of findings

This report conducted a controlled, single-factor study of loss-function design for cross-sectional stock-return prediction with a multi-layer perceptron. Within each comparison table, data, features, architecture, training protocol, portfolio construction, and evaluation metrics were fixed; only the loss function was varied. Under this protocol, the empirical evidence in Chapter 5 supports the following direct answers to the three research questions posed in Chapter 1.

**RQ1: How do prediction-level and portfolio-level metrics change when only the loss function is altered?** At the single-seed baseline (Chapter 5 §5.2), prediction-level and portfolio-level metrics decouple substantially. Losses that produce extremely negative R² (the absolute-loss family) can still yield competitive portfolio Sharpes, because the portfolio depends on cross-sectional ranks rather than calibrated magnitudes. R² is therefore a scale diagnostic rather than a primary performance metric in this setting. Among baseline losses, the hybrid multiplicative variant `hybrid_mul_m1` produces the best single-seed Sharpe, motivating the hybrid-loss design direction that Phase 2 and Phase 3 extend.

**RQ2: Within the hybrid-loss design space, which combination of directional and robust components performs best on the joint Sharpe-stability criterion?** The multi-seed evidence (Chapter 5 §5.4 and §5.5, three seeds per row) identifies the M2-robust γ family as the productive region of the loss space. Within that family, `m2_robust_gamma07` achieves mean Sharpe $0.9156$ with coefficient of variation $0.1808$ and mean cumulative return $+27.99\%$ over 24 months. `m2_robust_gamma10` has a higher mean Sharpe ($1.0043$) but three times the CV, and its per-seed Sharpe range is wide ($0.4587$ to $1.5847$). The integrated sweep shows that the IMADL-m2 α family peaks at $\alpha = 0.6$ (mean Sharpe $0.6895$, CV $0.2443$), that the IMADL-GMADL β family does not produce robust positive Sharpes (CV values from 4.0 to 139.5), and that the adaptive-λ family underperforms both on Sharpe and on stability. `m2_robust_gamma07` emerges as the single best candidate on the joint Sharpe-stability criterion; `imadl_m2_alpha06` corroborates the multiplicative-hybrid direction from an independent parameterisation.

**RQ3: Are the observed winners approximately stable under the diagnostic loss-component normalisation probe?** The normalisation probe (Chapter 5 §5.6) applies component normalisation to the three strongest candidates. `m2_robust_gamma07` is approximately flat under normalisation (mean Sharpe $0.9156 \to 0.9112$, a change smaller than the per-seed dispersion). Both `m2_robust_gamma10` ($1.0043 \to 0.4072$) and `imadl_m2_alpha06` ($0.6895 \to -0.0161$) degrade materially. The operational conclusion is that component normalisation is *not* a universal fix, and that `m2_robust_gamma07` is the only one of the three candidates whose signal is approximately stable in the diagnostic normalisation probe. The probe itself acknowledges that its scale ratios are diagnostics-estimated rather than measured by a fully instrumented per-component logger; that limitation is recorded as a future-work item.

**Final recommendation.** Combining the answers to RQ1–RQ3, the report's final loss recommendation has three tiers:

- **Primary: `m2_robust_gamma07`.** Mean Sharpe $0.9156$, CV $0.1808$, mean cumulative return $+27.99\%$. Approximately stable under the diagnostic component-normalisation probe. No seed in the run produces a negative Sharpe. This is the best-supported single choice.
- **High-return alternative with explicit caveat: `m2_robust_gamma10`.** Mean Sharpe $1.0043$, CV $0.5613$. Appropriate only where the user knowingly accepts seed-sensitivity in exchange for best-case Sharpe.
- **Stable fallback: `imadl_m2_alpha06`.** Mean Sharpe $0.6895$, CV $0.2443$, mean cumulative return $+30.42\%$. Provides independent corroboration that the multiplicative-hybrid region is the productive one.

## 6.2 Limitations

Several limitations scope the findings.

**Single market and single frequency.** The study uses a CRSP-style US equity monthly panel. Generalisation to non-US markets, non-equity asset classes, or higher/lower frequencies is not tested.

**Single static evaluation window.** The main test window is `1995-01` to `1996-12` (24 months), fixed and non-rolling. This window spans a sustained bull market and does not include any major macroeconomic crisis. Regime-coverage robustness is untested. The project originally planned a rolling-window extension; it was paused when the static-window results revealed the seed-sensitivity patterns that motivated Phase 2, and the rolling-window runner has not been resumed.

**Seed depth.** Multi-seed evaluations use three seeds per row. Three seeds are sufficient to distinguish CVs at the order-of-magnitude level (as between `gamma07` CV $0.18$ and `gamma10` CV $0.56$) but are too thin to pin CV to the second decimal. A larger seed depth — ten or more per row — would give tighter confidence intervals on both mean Sharpe and CV, and would either confirm or revise the 3-seed ordering.

**Scale-ratio diagnostics are estimates.** The normalisation probe in Chapter 5 §5.6 uses scale ratios that were estimated from Phase 3 diagnostics rather than measured by a fully instrumented per-component logger. The conclusion that normalisation is not a universal fix is supported by the empirical degradation of `gamma10` and `alpha06`, but a precise quantitative accounting of the directional-vs-magnitude balance awaits a per-component logger implementation.

**Static training protocol.** The single-training-run setup measures loss-function *design* effects cleanly (RQ1–RQ3 are answered under internally valid conditions) but does not reproduce a realistic deployment schedule in which the model would be retrained periodically. Any deployment implication should be checked under a rolling-retraining design.

**Cross-phase alignment.** Phase 2 and Phase 3 runners differ in λ settings, IMADL formulations, and some implementation details. Within-phase comparisons are strong; cross-phase improvement claims are used as motivation chains rather than as direct claims.

**Gross-of-cost portfolio.** Transaction costs, financing costs, and borrow costs are not modelled. A high-turnover long-short strategy would see its empirical Sharpe reduced materially once costs are included. Whether the relative ordering across loss functions is preserved under realistic cost assumptions is untested, because turnover can differ materially by loss function even under the same portfolio cap and bucket threshold. Absolute performance levels should be interpreted accordingly, and cost-adjusted ordering is an open question for future work.

**Architecture is not ablated.** The MLP[64, 32, 16] with ReLU and dropout $0.2$ was selected on pre-test data for an earlier research question and then frozen. The report conditions on this architecture rather than arguing for it; a larger or smaller network could change the magnitude of the loss-function effects documented here.

**Feature set restriction.** All Chapter 5 numbers use the X1 feature set: the model input is 15 columns comprising 10 X1 engineered columns (5 horizons × cumulative return + cumulative turnover) together with the 5 base panel columns retained by `assemble_feature_matrix` (see Appendix B Table B.3). Alternative feature sets are implemented in the codebase but were not run under the final protocol. Loss–feature interactions are untested.

## 6.3 Future work

Five extensions would materially strengthen or broaden the report's conclusions.

**1. Rolling-window evaluation.** Re-running the Phase 2 γ sweep and the IMADL-m2 α sweep under a rolling-window retraining scheme — for example, a five-year training window with a one-year test window advanced annually — would test whether the `m2_robust_gamma07` recommendation holds across macroeconomic regimes. This is the single highest-impact extension because it simultaneously addresses the single-window and the static-training-protocol limitations.

**2. Larger seed depth.** Increasing the per-row seed count from three to ten or more would allow calibrated confidence intervals on mean Sharpe and CV. If the 3-seed Sharpe ordering `gamma07 > alpha06` on the joint Sharpe-stability criterion (mean Sharpe with lowest CV) and `gamma10 > gamma07` on mean Sharpe alone persists at higher seed depth, the report's recommendation strengthens; if not, the ordering would require revision. This extension is computationally modest and should be a short follow-up run.

**3. Per-component loss logger.** Instrumenting the directional and magnitude components of each hybrid loss to log their per-batch contributions would replace the diagnostics-estimated scale ratios used in Chapter 5 §5.6 with measured values. This would make the normalisation probe quantitatively tight and could lead to an improved variant in which the components are normalised adaptively during training.

**4. Feature-set sensitivity.** Repeating the Phase 3 γ refinement under alternative feature sets defined in the codebase would quantify how the loss-function conclusions depend on the specific X1 construction. If `m2_robust_gamma07` remains best-supported across feature sets, the conclusion generalises; if not, the result points at an interaction between feature normalisation and loss-component scaling that would itself be an interesting phenomenon.

**5. Other asset classes and frequencies.** Extending the pipeline to daily-frequency equity data, to international equity markets, or to cross-sectional currency and rates portfolios would test the external validity of the loss-design conclusions. The heavy-tail motivation for robust hybrid losses is more pronounced at higher frequencies, so a daily-frequency study is a natural first extension.

A smaller but related set of implementation extensions would also be valuable: explicit CUDA-determinism flags for exact reproduction across hardware, alignment diagnostics that produce a single binary pass/fail per loss family, and a combined report that stitches all phases into one end-to-end run script.

## 6.4 Concluding remarks

Under the static 24-month protocol studied here, `m2_robust_gamma07` is the best-supported loss recommendation — scoped to US equity monthly data, a single evaluation window, and three-seed evidence. The broader contribution is methodological: a controlled evidence protocol that isolates the loss function as a design variable and measures multi-seed stability alongside mean performance, providing a reusable template for future loss-design research.
