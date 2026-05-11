# Chapter 1: Introduction

## 1.1 Background and motivation

In quantitative finance, machine-learning models have become a standard tool for predicting cross-sectional stock returns and constructing long-short portfolios. At the centre of every such model sits a frequently overlooked component: the loss function. The loss function is the training signal; it encodes what the model is actually being optimised for, and therefore which patterns the model prioritises during learning. Architecture, feature engineering, and portfolio optimisation have all received extensive research attention, but the systematic design and comparative evaluation of loss functions tailored to financial prediction tasks remains comparatively underexplored.

Most deployed return-prediction models use generic regression losses such as Mean Squared Error (MSE) or Mean Absolute Error (MAE). These losses optimise for statistical accuracy by minimising the discrepancy between predicted and realised returns. They do not explicitly account for two features of the underlying application that matter for economic performance. First, the downstream decision — a long-short portfolio — depends only on the cross-sectional *ranks* of the predictions, not on their calibrated values; a model that is badly miscalibrated in magnitude but correctly ranked produces the same portfolio as a calibrated model with identical rankings. Second, monthly equity returns are heavy-tailed: a small number of extreme observations dominate any scale-sensitive quantity. A quadratic loss therefore gives disproportionate training weight to what are, from the portfolio's point of view, noisy tail events.

Recent work has attempted to address the ranking gap through directional losses that reward correct sign predictions weighted by realised magnitude. Michańków, Ślepaczuk, and Bielak (2024) introduced the Mean Absolute Directional Loss (MADL) and its generalised variant GMADL [7, 8]. These losses explicitly penalise directional misalignment between predicted and realised returns and weight the signal by how large the realised move was. GMADL, in particular, uses a sigmoid-based directional term combined with a magnitude weighting $|y|^b$.

Preliminary analysis of the GMADL family identifies several systematic limitations: the reward for a correct prediction and the penalty for an incorrect prediction are symmetric (no risk-aversion preference); gradients become weak when predictions approach zero (the training signal vanishes precisely where the model is uncertain); the point-prediction scale drifts away from the realised-return scale during training (calibration breaks, and any diagnostic that depends on it becomes uninformative); and the loss does not incorporate an explicit robustness mechanism to handle the heavy tails of the return distribution. These observations motivate the central design direction of this report: *hybrid* loss functions that combine a directional penalty with a robust error term, with the aim of preserving the ranking-alignment benefits of directional losses while addressing their stability and calibration limitations.

## 1.2 Research gap

The machine-learning-for-returns literature exhibits a structural imbalance. On the architectural side, a sequence of studies has moved from linear factor models (Fama and French, 1993 [2]) to non-linear and deep neural networks (Gu, Kelly, and Xiu, 2020 [1]), and more recently to transformer-style architectures; the loss function throughout this progression has remained MSE or MAE in the overwhelming majority of cases. On the loss-function side, the trading-aware family (MADL/GMADL/IMADL) and the robust-regression family (Huber, MedSE) have developed largely independently, and systematic comparisons across the two families — under identical data, features, architecture, and evaluation protocol — are rare.

Three specific gaps follow. First, ==published work== that introduces a new loss function typically demonstrates its effectiveness on a single dataset without a comprehensive same-conditions comparison against alternatives, so the marginal contribution of the loss choice is hard to isolate. Second, the interaction between loss choice and portfolio-level metrics (Sharpe, cumulative return, stability across seeds) is usually under-reported: studies report a single performance number rather than a stability distribution. Third, the specific idea of *hybridising* a directional loss with a robust error term — producing a joint objective that targets both ranking and calibration — has not been systematically empirically evaluated, nor has the robustness parameter of such a hybrid been swept under multi-seed conditions.

This report addresses all three gaps through a single controlled pipeline. Data, features, model architecture, training protocol, portfolio construction, and evaluation metrics are held fixed across every run; only the loss function changes. Within this pipeline, the report evaluates baseline regression and directional losses, two parameterised hybrid families (additive and multiplicative), and a γ-parameterised M2-robust family that forms the basis of the final recommendation.

## ==1.3 Research objectives==

The report pursues four objectives:

1. **Benchmark baseline loss families under a common 24-month evaluation protocol.** Produce a single-seed comparison of traditional regression losses (MSE, MedSE), absolute-loss / directional losses (MADL, GMADL, IMADL), and initial hybrid multiplicative variants on a fixed 24-month test window, so that subsequent hybrid-loss work has an honest baseline.
2. **Design and scan parameterised hybrid losses.** Develop additive (A-series) and multiplicative (M-series) hybrids that combine a batch-normalised directional penalty with a Huber-style magnitude term, and scan their weighting hyperparameters under the same single-seed protocol to characterise the design space.
3. **Evaluate multi-seed robustness of the strongest hybrid family.** Extend the best-performing hybrid family into a γ-parameterised M2-robust form, evaluate it across three random seeds, and measure both mean Sharpe and cross-seed stability (coefficient of variation). Locate the γ setting that balances mean performance and stability.
4. **Test whether the resulting recommendation is an artefact of loss-component scaling.** Run a loss-component normalisation probe on the top candidates to test whether the observed Sharpe ordering is driven by scale imbalance between the two loss components or by genuine loss-family differences. Record the claim boundaries that follow from the result.

Each objective maps to one or two chapters. Objective 1 is answered in Chapter 5 §5.2. Objective 2 is answered in Chapter 5 §5.3. Objective 3 is answered in Chapter 5 §5.4 and §5.5. Objective 4 is answered in Chapter 5 §5.6.

## 1.4 Research questions

The objectives above are operationalised as three research questions. All three are scoped to the static 24-month protocol described in Chapter 3.

**RQ1: How do prediction-level and portfolio-level metrics change when only the loss function is altered?** This question isolates the causal effect of loss choice by holding architecture, features, and training protocol constant. It asks whether directional and hybrid losses sacrifice calibration-quality metrics (MSE, R²) in exchange for portfolio-quality metrics (Sharpe, cumulative return), and whether such a trade-off is present, reversed, or absent for any given loss family.

**RQ2: Within the hybrid-loss design space, which combination of directional and robust components dominates in multi-seed Sharpe and stability?** This question goes beyond a single-seed comparison and asks which loss-family region — additive hybrids, multiplicative hybrids, M2-robust γ, IMADL-m2 α, IMADL-GMADL β, or adaptive-λ — produces the best mean Sharpe with tolerable coefficient of variation across three seeds.

**RQ3: Are the observed winners robust to loss-component normalisation?** This question addresses a specific confound: that the apparent Sharpe advantage of a hybrid loss might reflect relative scaling of its two components rather than a genuine improvement. The question is answered empirically by running the top candidates with and without explicit component normalisation and comparing their mean Sharpes across seeds.

Chapter 5 gives the evidence-backed answer to each RQ; Chapter 6 synthesises the answers into the report's final conclusions.

## 1.5 Scope and claim boundaries

The scope of this report is deliberately narrow so that the causal attribution in RQ1 is clean. Readers should read the findings within the following boundaries.

**What is inside the scope.**
- Loss-function design and comparison within a fixed MLP architecture.
- Single equity market (US-listed equities from a CRSP-style monthly panel).
- A static training window `1990-01..1994-12` (60 monthly cross-sections) and a static out-of-sample test window `1995-01..1996-12` (24 monthly cross-sections).
- Long-short portfolio construction with top/bottom 10% buckets, signal-tilted within-bucket weights, and a capped-simplex projection at 5% per stock.
- Per-month metrics (MSE, MedSE, R²), portfolio metrics (monthly and cumulative long-short return, $\sqrt{12}$-annualised Sharpe), and cross-seed stability (CV) for multi-seed phases.

**What is outside the scope.**
- Architecture search: the MLP[64, 32, 16] with ReLU and dropout $0.2$ is fixed via pre-test grid search and not tuned during the study.
- Feature engineering: all Chapter 5 evidence uses the 15-dimensional X1 feature set; X2 and X3 are defined in the code but not carried into the final tables.
- Rolling-window evaluation: the rolling-window extension was paused in the project schedule once the static-window results revealed seed-sensitivity patterns worth investigating.
- Transaction costs, financing costs, and borrow costs: portfolio returns are reported gross of frictions.
- Non-equity asset classes, non-US markets, and non-monthly frequencies.

**Claim strength taxonomy.** Chapter 3 §3.8 formalises the claim boundaries in detail, and every empirical table in Chapter 5 is labelled by its claim strength. Three levels are used throughout:
- *Strong.* Same runner, same window, verified CSVs: the single-seed baseline and Phase 1.5 tables, the multi-seed γ refinement, and the integrated Phase 2 grouped summary.
- *Moderate.* Same window but different hyperparameters or seed sets across the compared rows: cross-phase comparisons (e.g., Phase 1.5 M1 vs Phase 2 γ07) are used as motivation chains, not as direct-improvement claims.
- *Weak / contextual.* Alignment diagnostics (Phase 2.1b, Phase 2.5), the normalisation probe with estimated scale ratios (Phase 2.2-fix1), and legacy 6-month sanity checks. These are cited for scope-setting, not as headlines.

**Non-claims.** The report does *not* claim that (i) MLP[64, 32, 16] is the optimal architecture for loss-design research, (ii) the M2-robust γ family is optimal across market regimes, (iii) loss-component normalisation universally fails, (iv) single-seed results indicate robustness, or (v) Phase 2 exactly replicates Phase 1.5 at a formula level. Each of these is explicitly avoided throughout Chapters 3 and 5.

## 1.6 Contributions and report structure

The primary contributions of this report are:

1. **A same-conditions 24-month baseline.** A single-seed comparison of seven baseline loss families (MSE, MedSE, MADL, GMADL, IMADL, and two hybrid multiplicative variants) under an identical runner, window, feature set, model, batch size, and portfolio construction (Chapter 5 §5.2).
2. **A parameterised hybrid-loss sweep.** Nine Phase 1.5 variants (A1–A5 and M1–M4) with explicit $\lambda_{\mathrm{dir}}$ and $\lambda_{\mathrm{hub}}$ settings, evaluated under the same protocol as the baseline (Chapter 5 §5.3).
3. **A multi-seed γ refinement.** Five γ values of the M2-robust family evaluated across three seeds, with explicit cross-seed mean, standard deviation, minimum, maximum, and coefficient of variation (Chapter 5 §5.4 and Table 5.3).
4. **An integrated sweep across alternative hybrid parameterisations.** IMADL-m2 α, IMADL-GMADL β, and adaptive-λ families evaluated under the same multi-seed protocol, producing a grouped summary that rules in `m2_robust_gamma07` and `imadl_m2_alpha06` as the self-consistent peaks (Chapter 5 §5.5 and Table 5.4).
5. **A loss-component normalisation probe.** A three-candidate test of whether scale imbalance between the directional and magnitude components drives the observed Sharpe ordering; the result is that `m2_robust_gamma07` is approximately scale-robust while `gamma10` and `alpha06` degrade under normalisation (Chapter 5 §5.6).
6. **A final recommendation with explicit tiers and caveats.** Primary choice: `m2_robust_gamma07` (mean Sharpe $0.9156$, CV $0.1808$, cumulative return $0.2799$ over 24 months across three seeds). High-return alternative with seed-sensitivity caveat: `m2_robust_gamma10`. Stable fallback: `imadl_m2_alpha06` (Chapter 5 §5.8, reiterated in Chapter 6).
7. **A documented evidence gate.** Every number in the report is traced to a verifiable source artifact (CSV, JSON, or grouped summary), and a formal claim-boundary taxonomy governs what can and cannot be said across phases (Chapter 3 §3.8).

The remainder of the report is organised as follows. Chapter 2 reviews the literature on cross-sectional return prediction, robust regression, directional/trading-aware losses, and hybrid designs, and positions the research gap. Chapter 3 describes the methodology, including model architecture, loss-function definitions, training protocol, portfolio construction, evaluation metrics, phase design, and claim-boundary taxonomy. Chapter 4 documents the data pipeline, feature variables, preprocessing, and data limitations. Chapter 5 reports the empirical results across the four phases and the normalisation probe, and synthesises the findings into the final recommendation. Chapter 6 revisits the research questions, states limitations that follow from Chapter 5's evidence boundaries, and outlines directions for future work.
