# Chapter 1: Introduction

## 1.1 Background and Motivation

In the domain of quantitative finance, machine learning models have increasingly been deployed to predict cross-sectional stock returns and construct profitable trading strategies. At the heart of these models lies a critical but often overlooked component: the loss function. The loss function serves as the "reward signal" that guides the learning process, determining what patterns the model prioritizes during training. While substantial research has focused on model architectures, feature engineering, and portfolio optimization techniques, the systematic design and evaluation of loss functions tailored to financial prediction tasks remains relatively underexplored.

Traditional approaches to return prediction typically employ generic regression loss functions such as Mean Squared Error (MSE) or Mean Absolute Error (MAE). These functions optimize for statistical accuracy by minimizing the discrepancy between predicted and actual returns. However, they do not explicitly account for the directional nature of trading decisions or the economic value of predictions. A model trained with MSE may achieve low prediction error while failing to capture the directional signals that drive portfolio profitability.

Recent work has attempted to bridge this gap by introducing directional loss functions. Michańków et al. (2024) proposed the Mean Absolute Directional Loss (MADL) and its generalized variant, Generalized MADL (GMADL), which explicitly penalize directional misalignment between predicted and actual returns [7, 8]. These loss functions aim to align model training more closely with the ultimate goal of generating profitable trading signals. GMADL, in particular, uses a sigmoid-based directional term combined with a magnitude weighting to reward correct directional predictions proportionally to the size of the true return.

Despite these theoretical advances, preliminary analysis conducted during Semester 1 of this project revealed systematic limitations in GMADL's formulation. Specifically, the loss function exhibits symmetry in its treatment of gains and losses, fails to provide strong learning signals when predictions approach zero, and does not explicitly incorporate robustness to outliers—a critical concern in financial data characterized by heavy-tailed return distributions. These observations motivate the need for hybrid loss functions that combine directional penalties with robust error terms, potentially offering superior performance across both prediction accuracy and portfolio profitability.

## 1.2 Research Gap

The existing literature on machine learning for financial prediction exhibits a notable imbalance. While considerable effort has been devoted to developing sophisticated model architectures—ranging from deep neural networks to ensemble methods—and engineering predictive features from market data, the loss function is typically treated as a fixed component selected from a standard menu of options. This approach overlooks the potential for loss function design to serve as a powerful lever for improving model performance.

Most studies that do address loss functions focus on model architecture or feature selection rather than conducting systematic comparisons of different loss formulations under controlled conditions. For instance, Gu et al. (2020) demonstrated that deep learning models can extract complex patterns from financial data, but their analysis primarily compared model architectures while holding the loss function constant [1]. Similarly, research on factor models and return prediction often emphasizes the choice of predictive variables rather than the optimization objective.

The GMADL framework represents a significant step toward trading-goal-oriented loss design, but it has known limitations that have not been systematically addressed. The Semester 1 analysis identified three key issues: (1) the symmetry problem, where the magnitude of reward for correct predictions equals the magnitude of penalty for incorrect predictions, failing to reflect risk aversion principles; (2) weak gradients near zero predictions, which provide insufficient learning signals in critical regions; and (3) lack of explicit robustness mechanisms to handle outliers in return distributions.

Furthermore, there is a lack of empirical research that isolates the impact of loss function choice on both statistical prediction metrics (such as R² and MSE) and economic performance metrics (such as Sharpe ratio and cumulative returns). Most studies conflate multiple design choices, making it difficult to attribute performance differences to specific components. This project addresses this gap by conducting a controlled experimental comparison where only the loss function varies while the model architecture, features, and evaluation protocol remain fixed.

## 1.3 Research Objectives

This thesis pursues four primary objectives:

1. **Systematically diagnose the limitations of GMADL**: Through theoretical analysis and empirical testing, identify the specific scenarios and conditions under which GMADL underperforms or exhibits undesirable behavior. This includes examining its gradient properties, symmetry characteristics, and sensitivity to outliers.

2. **Design hybrid loss functions**: Develop novel loss formulations that combine directional penalty terms with robust error components. These hybrid losses aim to preserve the directional alignment benefits of GMADL while addressing its limitations through explicit robustness mechanisms and asymmetric treatment of gains and losses.

3. **Evaluate loss function impact on prediction and portfolio performance**: Conduct a comprehensive empirical comparison of baseline loss functions (MSE, MedSE, MADL, GMADL) and hybrid variants across multiple evaluation dimensions. Assess both statistical metrics (out-of-sample R², MSE, Median Squared Error) and economic metrics (Sharpe ratio, cumulative returns, volatility) to understand the full spectrum of loss function impact.

4. **Quantify robustness-performance tradeoffs**: Through multi-seed testing and coefficient of variation analysis, measure the stability of different loss functions across random initializations. This addresses the practical concern of whether performance gains are consistent or merely artifacts of favorable random seeds.

## 1.4 Research Questions

This research is guided by three central questions:

1. **How do prediction metrics change when only the loss function is altered?** By holding the model architecture, features, and training protocol constant, we can isolate the causal effect of loss function choice on statistical prediction accuracy. This question examines whether directional losses sacrifice prediction accuracy for economic performance, or whether they can achieve both simultaneously.

2. **What systematic differences in portfolio performance arise from different loss functions?** Beyond prediction accuracy, we investigate how loss function choice translates into portfolio-level outcomes. This includes analyzing cumulative returns, risk-adjusted returns (Sharpe ratios), and volatility patterns across different portfolio construction methods (equal-weighted, signal-weighted, and capped strategies).

3. **Can hybrid losses outperform single-component losses?** We test the hypothesis that combining directional penalties with robust error terms yields superior performance compared to using either component in isolation. This question addresses whether the added complexity of hybrid formulations is justified by measurable performance improvements.

## 1.5 Scope and Limitations

This study focuses specifically on loss function design within a fixed neural network architecture for cross-sectional stock return prediction. Several important aspects of the broader problem are deliberately excluded to maintain experimental control and clarity:

**Focus Areas:**
- Loss function formulation and comparison
- Impact on prediction accuracy and portfolio performance
- Robustness across random seeds
- Three portfolio construction strategies (equal-weighted, signal-weighted, capped)

**Excluded from Scope:**
- Model architecture search: The neural network structure (3-layer MLP with [64, 32, 16] hidden units) is fixed based on preliminary grid search to avoid confounding architecture effects with loss function effects.
- Feature engineering: The feature sets (cumulative momentum, denoised standardized momentum, raw monthly series) are predetermined based on established literature.
- Portfolio optimization: We use simple long-short strategies rather than mean-variance optimization or other sophisticated portfolio construction methods.
- Transaction costs: The analysis assumes frictionless trading, which may overstate realized returns in practice.
- Market microstructure: Liquidity constraints, market impact, and execution considerations are not modeled.

**Temporal Scope:**
- Training period: January 1990 to December 1994 (5 years)
- Main out-of-sample test period: January 1995 to December 1996 (24 months)
- Early baseline checks used January 1995 to June 1995 (6 months) only as preliminary sanity checks
- This represents a static sanity check rather than a full rolling-window backtest

**Asset Class:**
- US equities only, sourced from CRSP monthly database
- No international markets, fixed income, or alternative assets

**Limitations:**
The main empirical comparisons use a 24-month static out-of-sample window, which is longer than the early 6-month baseline checks but still limited relative to a full rolling-window backtest across multiple regimes. The static training protocol (train once, predict through the test window) does not reflect realistic deployment where models would be retrained periodically. Results should be interpreted as controlled evidence on loss-function design rather than production-ready strategy performance. The exclusion of transaction costs likely overstates net returns, particularly for high-turnover strategies.

## 1.6 Thesis Structure

The remainder of this thesis is organized as follows:

**Chapter 2: Literature Review** surveys the relevant academic literature across four domains: algorithmic investment strategies, testing frameworks and validation protocols, loss functions in machine learning, and loss functions specifically designed for financial prediction. This chapter positions the current research within the broader context of quantitative finance and machine learning.

**Chapter 3: Data and Methodology** provides a detailed description of the dataset, preprocessing, feature engineering, model architecture, loss function formulations, training protocol, portfolio construction methods, and evaluation metrics. This chapter ensures reproducibility and transparency of the research process.

**Chapter 4: Loss Function Design** presents the development of novel hybrid loss formulations (M1, M2, and M2_robust) that combine directional penalties with robust error terms, including theoretical motivation and implementation details.

**Chapter 5: Empirical Results and Discussion** reports the empirical findings across all experimental phases, including baseline comparisons, lambda sweep analysis, hybrid variant testing, gamma refinement, and normalization checks. Results are presented for both statistical and economic metrics, with each table tied to its evaluation window.

**Chapter 6: Conclusion** synthesizes the key findings, discusses their implications for loss function design in financial machine learning, acknowledges limitations, and proposes directions for future research.
