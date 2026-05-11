# Chapter 1: Introduction

## 1.1 Background, motivation, and research gap

Machine-learning models are now widely used for cross-sectional stock-return prediction and long-short portfolio construction. In this setting, the loss function is not just a technical training choice: it determines which prediction errors the model treats as important. Yet most return-prediction studies vary architectures and feature sets while keeping the objective at generic regression losses such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).

This default is not fully aligned with the portfolio task. A long-short portfolio depends mainly on the cross-sectional ranking of predictions, not on their calibrated point values, and monthly equity returns are heavy-tailed. A model can therefore reduce MSE while still ranking the most economically relevant names poorly, or devote excessive training weight to noisy tail observations that do not improve portfolio construction.

Existing loss-function families address parts of this problem. Robust losses reduce the influence of heavy-tailed residuals, while directional losses such as MADL and GMADL [7, 8] reward correct return signs and magnitude-weighted directional alignment. What remains less clear is how regression, robust, directional, and hybrid losses compare under the same data, feature input, model, and portfolio rule. This report addresses that gap through a controlled empirical study of loss-function design for portfolio-oriented return prediction.

## 1.2 Research objectives

This report has four objectives.
First, it benchmarks standard regression, robust, directional, and initial hybrid losses under a common 24-month out-of-sample protocol.
Second, it develops and evaluates hybrid loss functions that combine a directional component with a robust magnitude component.
Third, it evaluates the most relevant hybrid families under multi-seed conditions, using both mean Sharpe and cross-seed stability to avoid relying on a single favourable run.
Fourth, it checks whether the recommended loss remains approximately stable under a diagnostic loss-component normalisation probe.
The study is deliberately narrow. It fixes the equity universe, X1 feature input, MLP architecture, static 1990–1994 training window, 1995–1996 test window, and long-short portfolio construction. Architecture search, feature-set sensitivity, rolling-window validation, transaction costs, and non-US or non-monthly settings are outside the main claim boundary.

## 1.3 Research questions

**RQ1: How does loss choice affect prediction-level and portfolio-level performance under a fixed evaluation protocol?** This question compares regression, robust, directional, and hybrid losses while holding data, model, and portfolio rule fixed, asking whether calibration metrics (R²) and portfolio metrics (Sharpe) move together or decouple.

**RQ2: Which hybrid-loss design gives the best supported Sharpe-stability trade-off?** This question uses a single-seed sweep to map the hybrid design space, then multi-seed evidence to identify which family produces the best mean Sharpe with tolerable cross-seed variability.

**RQ3: Are the leading hybrid-loss candidates approximately stable under diagnostic component normalisation?** This question tests whether the observed Sharpe ordering is driven by the relative scale of loss components or by a genuine loss-family effect.

Chapter 5 reports the empirical evidence for these questions, and Chapter 6 synthesises the answers.

## 1.4 Contributions and report structure

The report makes four contributions.

1. **A controlled loss-function benchmark** comparing regression, robust, directional, and hybrid losses under a common 24-month protocol.
2. **A hybrid-loss design study** developing additive and multiplicative hybrids that combine directional alignment with robust magnitude control.
3. **A multi-seed robustness comparison** evaluating leading hybrid candidates using mean Sharpe, cumulative return, and cross-seed stability.
4. **A bounded final recommendation** identifying `m2_robust_gamma07` as the best-supported primary choice, with `m2_robust_gamma10` as a higher-return alternative and `imadl_m2_alpha06` as a stable fallback.

The remainder of the report is organised as follows. Chapter 2 reviews return-prediction, robust-regression, and directional-loss literature. Chapter 3 describes the methodology. Chapter 4 documents the data. Chapter 5 presents the empirical results. Chapter 6 answers the research questions, states limitations, and outlines future work.
