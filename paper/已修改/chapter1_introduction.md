# Chapter 1: Introduction

## 1.1 Background and motivation

In quantitative finance, machine-learning models have become a standard tool for predicting cross-sectional stock returns and constructing long-short portfolios. At the centre of every such model sits a frequently overlooked component: the loss function. The loss function is the training signal; it encodes what the model is actually being optimised for, and therefore which patterns the model prioritises during learning. Architecture, feature engineering, and portfolio optimisation have all received extensive research attention, but the systematic design and comparative evaluation of loss functions tailored to financial prediction tasks remains comparatively underexplored.

Most deployed return-prediction models use generic regression losses such as Mean Squared Error (MSE) or Mean Absolute Error (MAE). These losses optimise for statistical accuracy by minimising the discrepancy between predicted and realised returns. They do not explicitly account for two features of the underlying application that matter for economic performance. First, the downstream decision — a long-short portfolio — depends only on the cross-sectional *ranks* of the predictions, not on their calibrated values. Second, monthly equity returns are heavy-tailed: a small number of extreme observations dominate any scale-sensitive quantity. A quadratic loss therefore gives disproportionate training weight to what are, from the portfolio's point of view, noisy tail events.

Directional losses such as MADL and GMADL [7, 8] address part of this mismatch by rewarding correct return signs and weighting those signs by realised-return magnitude. They therefore provide a natural starting point for portfolio-oriented training. However, pure directional losses can still be unstable in heavy-tailed monthly equity panels and can drift away from useful prediction-scale diagnostics. This report therefore studies hybrid loss functions that combine directional alignment with robust magnitude control.

## 1.2 Research gap

The literature on machine learning for stock-return prediction has paid far more attention to architectures and feature sets than to the training objective itself. MSE and MAE remain common defaults even when the downstream task is ranking stocks into long-short portfolios rather than producing calibrated forecasts.

Two relevant loss-function traditions have developed largely separately. Robust regression losses reduce the effect of heavy-tailed residuals, while directional and trading-aware losses target the sign and ranking information that matters for portfolio construction. What is less well established is how these families compare under the same data, feature set, model, and portfolio rule, or whether hybrid losses can combine their advantages.

This report addresses that gap through a controlled empirical comparison of regression, robust, directional, and hybrid losses. Within each comparison table, the protocol fixes the data, X1 feature input, MLP architecture, training settings, and portfolio construction; cross-phase comparisons are treated as design evidence rather than direct causal proof.

## 1.3 Research objectives

This report has four objectives.

First, it benchmarks standard regression, robust, directional, and initial hybrid losses under a common 24-month out-of-sample protocol. This establishes the baseline setting in which the loss function is the main design variable.

Second, it develops and evaluates hybrid loss functions that combine a directional component with a robust magnitude component. The purpose is to test whether ranking alignment and residual robustness can be brought into the same training objective.

Third, it evaluates the most relevant hybrid families under multi-seed conditions, using both mean Sharpe and cross-seed stability to avoid relying on a single favourable run.

Fourth, it checks whether the recommended loss remains approximately stable under a diagnostic loss-component normalisation probe. This probe does not prove general robustness, but it helps separate a substantive loss-family effect from a simple component-scaling artefact.

The study is deliberately narrow. It fixes the equity universe, X1 feature input, MLP architecture, static 1990–1994 training window, 1995–1996 test window, and long-short portfolio construction. Architecture search, feature-set sensitivity, rolling-window validation, transaction costs, and non-US or non-monthly settings are outside the main claim boundary.

## 1.4 Research questions

The objectives above are organised around three research questions.

**RQ1: How does loss choice affect prediction-level and portfolio-level performance under a fixed evaluation protocol?** This question compares standard regression losses, robust losses, directional losses, and initial hybrid losses while holding the data, model, training window, test window, and portfolio rule fixed within each comparison table.

**RQ2: Which hybrid-loss design gives the best supported Sharpe-stability trade-off?** This question first uses a single-seed sweep to map the additive and multiplicative hybrid design space, then uses multi-seed evidence to evaluate the most relevant hybrid families. Single-seed rows are treated as design evidence, not robustness evidence.

**RQ3: Are the leading hybrid-loss candidates approximately stable under diagnostic component normalisation?** This question tests whether the observed ordering is mainly driven by the relative scale of the loss components. The result is interpreted as a diagnostic boundary check rather than a universal normalisation claim.

Chapter 5 reports the empirical evidence for these questions, and Chapter 6 synthesises the answers.

## 1.5 Contributions and report structure

The report makes four contributions.

1. **A controlled loss-function benchmark.** It compares standard regression, robust, directional, and initial hybrid losses under a common 24-month evaluation protocol.
2. **A hybrid-loss design study.** It develops additive and multiplicative hybrids that combine directional alignment with robust magnitude control, then maps their behaviour under the fixed protocol.
3. **A multi-seed robustness comparison of leading hybrid candidates.** It evaluates the strongest supported hybrid candidates using mean Sharpe, cumulative return, and cross-seed stability, rather than relying only on seed-42 performance.
4. **A bounded final recommendation.** It identifies `m2_robust_gamma07` as the best-supported primary choice under the reported evidence, with `m2_robust_gamma10` as a higher-return but less stable alternative and `imadl_m2_alpha06` as a stable fallback.

The remainder of the report is organised as follows. Chapter 2 reviews return-prediction, robust-regression, directional-loss, and hybrid-loss literature. Chapter 3 describes the methodology, including model architecture, loss definitions, training protocol, portfolio construction, evaluation metrics, phase design, and claim boundaries. Chapter 4 documents the data source, sample construction, X1 feature input, preprocessing, and data limitations. Chapter 5 presents the empirical results and discusses the final recommendation. Chapter 6 answers the research questions, states the limitations, and outlines future work.
