# Chapter 2: Literature Review

## 2.1 Algorithmic Investment Strategies

Algorithmic investment strategies leverage computational methods to identify profitable trading opportunities in financial markets. The field has evolved from simple rule-based systems to sophisticated machine learning models capable of processing high-dimensional data and extracting complex patterns.

**Quantitative Trading Foundations**

Traditional quantitative trading relies on factor models that explain cross-sectional variation in stock returns through observable characteristics. Fama and French (1993) established the three-factor model incorporating market risk, size, and value factors, which has been extended to include momentum, profitability, and investment factors [2]. These models provide a theoretical foundation for understanding return predictability but typically employ linear relationships and hand-crafted factors.

**Machine Learning in Finance**

The application of machine learning to financial prediction has accelerated dramatically in recent years. Gu et al. (2020) demonstrated that deep neural networks can outperform traditional linear models by capturing non-linear interactions among predictive variables [1]. Their comprehensive study compared various machine learning architectures—including random forests, gradient boosted trees, and deep neural networks—on a dataset of 94 stock characteristics. The key finding was that model complexity matters: deeper networks with more parameters consistently achieved better out-of-sample prediction accuracy.

However, this literature largely focuses on model architecture rather than the optimization objective. Most studies employ standard regression losses (MSE or MAE) without questioning whether these objectives align with the ultimate goal of generating profitable trading signals. This represents a significant gap that motivates the current research.

**Cross-Sectional Stock Return Prediction**

Cross-sectional prediction aims to forecast relative returns across stocks at a given point in time, rather than predicting absolute return levels or time-series dynamics. This approach is particularly relevant for long-short equity strategies that profit from correctly ranking stocks rather than timing market movements.

Daniel and Moskowitz (2016) analyzed momentum-based strategies and identified key features that predict cross-sectional returns, including cumulative returns over various horizons and turnover measures [3]. Their work emphasizes the importance of feature engineering and the distinction between short-term reversal effects and medium-term momentum. This research informed the feature set design in the current study, particularly the use of cumulative momentum and turnover variables.

## 2.2 Testing Framework and Validation

Rigorous testing and validation protocols are essential in financial machine learning to avoid overfitting and ensure that observed performance reflects genuine predictive ability rather than data mining artifacts.

**Backtest Overfitting**

Lopez de Prado (2014) and Bailey et al. (2014) documented the pervasive problem of backtest overfitting in quantitative finance [4, 5]. When researchers test multiple strategies on the same dataset and report only the best-performing variant, the reported performance is upward-biased and unlikely to persist out-of-sample. This "multiple testing problem" is exacerbated by the ease of running thousands of backtests with modern computational tools.

The authors propose several solutions: (1) adjusting performance metrics for the number of trials conducted, (2) using combinatorially symmetric cross-validation to estimate overfitting, and (3) requiring out-of-sample validation on truly held-out data. These principles inform the experimental design of this thesis, which employs a strict train-test split with no parameter tuning on the test set.

**Multiple Testing Problem**

Harvey et al. (2016) extended this analysis to the broader academic literature, arguing that many published factor anomalies are likely false discoveries resulting from multiple testing without appropriate corrections [6]. They recommend using higher t-statistic thresholds (3.0 instead of 2.0) to account for the large number of factors tested across all published studies.

While this thesis does not claim to discover new factors, the multiple testing concern applies to loss function comparison. Testing seven baseline losses and multiple hybrid variants creates a multiple comparison problem. To address this, we report results for all tested configurations rather than cherry-picking the best performer, and we use multi-seed testing to assess robustness.

**Validation Protocols**

The distinction between in-sample fitting, out-of-sample validation, and true holdout testing is critical. In-sample performance measures how well a model fits the training data but provides no information about generalization. Out-of-sample validation on a separate time period offers a more realistic assessment, but if this validation set is used for hyperparameter tuning, it becomes part of the training process.

This study employs a controlled static evaluation protocol: train once on 1990-1994 and evaluate out-of-sample without retraining or parameter updates. Early baseline checks used a shorter six-month 1995 window, while the main empirical comparisons use the 24-month window from 1995-01 to 1996-12. This conservative approach prioritizes experimental control over realistic deployment simulation. The architecture was determined via grid search on an earlier period to avoid look-ahead bias.

## 2.3 Loss Functions in Machine Learning

Loss functions are fundamental to supervised learning, defining the optimization objective that guides model training. The choice of loss function has profound implications for what patterns the model learns and how it generalizes to new data.

**Traditional Regression Losses**

The most common regression loss is Mean Squared Error (MSE):

```math
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

MSE penalizes large errors heavily due to the quadratic term, making it sensitive to outliers. It is convex and differentiable everywhere, which facilitates optimization via gradient descent. Under the assumption of Gaussian noise, MSE corresponds to maximum likelihood estimation.

Mean Absolute Error (MAE) provides an alternative:

```math
L_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
```

MAE is more robust to outliers than MSE because it penalizes errors linearly rather than quadratically. However, it is not differentiable at zero, which can complicate optimization. MAE corresponds to maximum likelihood under Laplacian noise assumptions.

**Huber Loss**

Huber loss combines the best properties of MSE and MAE:

```math
L_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
```

For small errors (below threshold δ), Huber loss behaves like MSE, providing smooth gradients. For large errors, it behaves like MAE, limiting the influence of outliers. The threshold δ is a hyperparameter that controls the transition point. Huber loss is convex and differentiable everywhere, making it attractive for optimization while maintaining robustness.

**Properties and Tradeoffs**

Key properties that distinguish loss functions include:

1. **Convexity**: Convex losses guarantee that gradient descent converges to the global minimum. Non-convex losses may have multiple local minima, complicating optimization.

2. **Robustness**: Robust losses limit the influence of outliers, preventing a few extreme observations from dominating the training process. This is particularly important in financial data with heavy-tailed return distributions.

3. **Computational Efficiency**: Some losses (e.g., quantile regression) require iterative optimization procedures that are computationally expensive. Losses with simple closed-form gradients enable faster training.

4. **Differentiability**: Smooth, differentiable losses facilitate gradient-based optimization. Non-differentiable points (e.g., the kink in MAE at zero) can slow convergence.

The choice among these properties involves tradeoffs. MSE offers computational simplicity and strong theoretical foundations but lacks robustness. MAE provides robustness but sacrifices differentiability. Huber loss attempts to balance these concerns but introduces an additional hyperparameter.

**Impact on Model Training and Generalization**

The loss function shapes what the model learns during training. MSE-trained models minimize expected squared error, which corresponds to learning the conditional mean E[y|x]. MAE-trained models learn the conditional median, which may differ substantially in skewed distributions. This distinction has important implications for financial prediction, where return distributions are typically non-Gaussian with heavy tails and skewness.

Furthermore, the loss function affects generalization through its interaction with model capacity and regularization. Robust losses can act as implicit regularizers by limiting the influence of outliers, potentially improving out-of-sample performance even if in-sample fit is worse.

## 2.4 Loss Functions in Financial Prediction

While traditional regression losses optimize for statistical accuracy, they do not explicitly account for the economic value of predictions or the directional nature of trading decisions. Recent research has begun to address this gap by developing loss functions tailored to financial applications.

**Directional Losses: MADL and GMADL**

Michańków et al. (2024) introduced the Mean Absolute Directional Loss (MADL) to align model training with trading objectives [7]. MADL penalizes directional misalignment between predicted and actual returns:

```math
L_{MADL} = -\tanh(a \cdot y \cdot \hat{y}) \times |y|
```

where y is the true return, ŷ is the predicted return, and a is a scaling parameter (typically 25). The tanh function provides a smooth transition between reward (negative loss) for correct directional predictions and penalty (positive loss) for incorrect predictions. The magnitude term |y| weights the loss by the size of the true return, reflecting the economic principle that correctly predicting large moves is more valuable than correctly predicting small moves.

The authors extended this to Generalized MADL (GMADL):

```math
L_{GMADL} = -[\sigma(a \cdot y \cdot \hat{y}) - 0.5] \times |y|^b
```

where σ is the sigmoid function, a = 100, and b = 2. The sigmoid term provides smoother gradients than tanh, and the squared magnitude term |y|² further emphasizes large returns.

These loss functions represent a significant conceptual advance by explicitly optimizing for directional correctness rather than magnitude accuracy. However, as documented in Chapter 4, they have limitations including symmetry issues and weak gradients near zero predictions.

**Ranking Losses**

An alternative approach treats return prediction as a ranking problem rather than a regression problem. Ranking losses aim to correctly order stocks by expected return rather than predicting exact return values.

LambdaRank and ListNet are examples from information retrieval that have been adapted to financial applications. These methods optimize ranking metrics like Normalized Discounted Cumulative Gain (NDCG) directly. However, they typically require pairwise or listwise comparisons, which are computationally expensive for large cross-sections of stocks.

**Economic Loss Functions**

Some researchers have proposed loss functions that directly incorporate portfolio-level objectives. For example, one could define loss as the negative Sharpe ratio of the resulting portfolio. However, this approach faces significant challenges: the mapping from predictions to portfolio returns is non-differentiable (due to discrete ranking and selection), and the loss depends on the entire cross-section rather than individual predictions, complicating batch-based training.

**Gap in the Literature**

Despite these advances, there remains a lack of systematic comparison of different loss functions under controlled conditions. Most studies introduce a new loss function and demonstrate its effectiveness on a specific dataset, but they rarely compare against a comprehensive set of alternatives or isolate the loss function effect from other design choices.

Furthermore, the interaction between loss function choice and portfolio construction method is underexplored. A loss function that performs well with equal-weighted portfolios may behave differently with signal-weighted or optimized portfolios. This thesis addresses these gaps through controlled experimentation across multiple loss functions and portfolio strategies.

## 2.5 The GMADL Framework

Given its central role in this research, GMADL warrants detailed examination. The loss function is defined as:

```math
L_i = -[\sigma(a \cdot y_i \cdot \hat{y}_i) - 0.5] \times |y_i|^b
```

**Mathematical Formulation**

The sigmoid term σ(a · y · ŷ) maps the product of true and predicted returns to the interval (0, 1). When y and ŷ have the same sign (correct direction), the product is positive, and the sigmoid output exceeds 0.5, resulting in negative loss (reward). When they have opposite signs (incorrect direction), the product is negative, the sigmoid output is below 0.5, and the loss is positive (penalty).

The parameter a controls the steepness of the sigmoid transition. With a = 100, the sigmoid approaches a step function, providing sharp distinction between correct and incorrect predictions. The subtraction of 0.5 centers the sigmoid output, ensuring that the loss is zero when the prediction is uninformative (ŷ = 0 or y · ŷ = 0).

The magnitude term |y|^b weights the loss by the size of the true return. With b = 2, large returns receive quadratically more weight than small returns. This reflects the economic intuition that correctly predicting a 10% move is more valuable than correctly predicting a 1% move.

**Theoretical Advantages**

GMADL offers several theoretical advantages over traditional regression losses:

1. **Directional Alignment**: By explicitly rewarding correct directional predictions, GMADL aligns the training objective with the goal of generating profitable trading signals.

2. **Economic Weighting**: The magnitude term ensures that the model prioritizes predictions that matter most for portfolio performance—those involving large price movements.

3. **Smooth Gradients**: The sigmoid function is differentiable everywhere, facilitating gradient-based optimization. Unlike step functions or hard thresholds, the sigmoid provides informative gradients even when predictions are far from optimal.

4. **Bounded Loss**: The loss is bounded between -|y|^b and +|y|^b, preventing any single observation from dominating the training process through extreme loss values.

**Known Limitations**

Despite these advantages, preliminary analysis revealed several limitations:

1. **Symmetry Problem**: The magnitude of reward for correct predictions equals the magnitude of penalty for incorrect predictions. This fails to reflect the risk aversion principle that losses hurt more than equivalent gains. In trading, avoiding large losses is often more important than capturing large gains.

2. **Weak Gradients Near Zero**: When the prediction ŷ approaches zero, the sigmoid term approaches 0.5, and the gradient becomes very small. This provides weak learning signals precisely when the model is most uncertain, potentially slowing convergence.

3. **Lack of Explicit Robustness**: While the bounded nature of the loss provides some implicit robustness, GMADL does not explicitly incorporate mechanisms to handle outliers in the return distribution. Extreme returns can still exert disproportionate influence through the |y|^b term.

4. **Ignoring Prediction Precision**: The loss depends on the sign of ŷ but not on its magnitude relative to y. A prediction of +0.01 and +1.0 receive the same reward if the true return is +0.1, even though the latter is far less accurate. This may fail to incentivize precise magnitude estimation.

These limitations motivate the development of hybrid loss functions that preserve GMADL's directional focus while addressing its weaknesses through explicit robustness mechanisms and asymmetric treatment of gains and losses.

## 2.6 Robust Loss Functions

Robustness to outliers is a critical concern in financial machine learning due to the heavy-tailed nature of return distributions. Extreme events—market crashes, earnings surprises, merger announcements—occur more frequently than Gaussian models predict, and a few outliers can severely distort model training if not handled appropriately.

**Huber Loss Revisited**

As discussed in Section 2.3, Huber loss provides a principled approach to robustness by combining quadratic penalties for small errors with linear penalties for large errors. In the financial context, this means that extreme returns (which may reflect idiosyncratic events rather than predictable patterns) receive bounded influence on the training process.

The threshold parameter δ determines the transition point between quadratic and linear regimes. Smaller δ values provide more aggressive outlier resistance but may sacrifice efficiency if the data is truly Gaussian. Larger δ values behave more like MSE, offering less protection against outliers. Typical choices range from 0.5 to 2.0 times the standard deviation of the residuals.

**Median-Based Losses**

Median Squared Error (MedSE) replaces the mean with the median:

```math
L_{MedSE} = \text{median}[(y_i - \hat{y}_i)^2]
```

The median is inherently robust to outliers because it depends only on the middle observation, not on extreme values. Up to 50% of the data can be arbitrarily corrupted without affecting the median. This makes MedSE particularly attractive for financial data with frequent outliers.

However, median-based losses present computational challenges. Unlike mean-based losses, the median does not decompose across observations, requiring the entire batch to be processed simultaneously. Furthermore, the median is not differentiable in the traditional sense, though subgradient methods can be employed.

**Outlier Resistance in Financial Data**

Financial return distributions exhibit several characteristics that challenge standard loss functions:

1. **Heavy Tails**: Returns have fatter tails than the Gaussian distribution, with extreme events occurring more frequently than predicted by normal models.

2. **Skewness**: Return distributions are often negatively skewed, with large negative returns (crashes) more common than large positive returns of equivalent magnitude.

3. **Time-Varying Volatility**: Return volatility clusters over time, with periods of high volatility following market stress. This heteroskedasticity violates the constant-variance assumption underlying MSE.

4. **Idiosyncratic Shocks**: Individual stocks experience company-specific events (earnings surprises, management changes, regulatory actions) that generate large returns unrelated to predictable factors.

Robust loss functions help models focus on the predictable component of returns rather than fitting noise from these idiosyncratic shocks. This can improve out-of-sample performance even if in-sample fit (as measured by MSE) is worse.

## 2.7 Research Positioning

This thesis contributes to the literature on loss function design for financial machine learning by conducting a systematic, controlled comparison of multiple loss formulations. Unlike prior work that introduces a single new loss function and demonstrates its effectiveness, this research evaluates a comprehensive set of alternatives under identical conditions.

**Systematic Loss Function Comparison**

The experimental design isolates the loss function effect by holding constant all other aspects of the modeling pipeline: data, features, architecture, training protocol, and evaluation metrics. This allows causal attribution of performance differences to the loss function rather than confounding factors. The comparison includes:

- Traditional regression losses (MSE, MedSE)
- Directional losses (MADL, GMADL, IMADL)
- Novel hybrid losses (M1, M2, M2_robust)

Each loss is evaluated across multiple dimensions: statistical prediction accuracy (R², MSE, MedSE), economic performance (Sharpe ratio, cumulative returns), and robustness (coefficient of variation across seeds).

**Controlled Experimental Design**

The static sanity check protocol--train once on 1990-1994 and evaluate on a fixed out-of-sample window--provides a clean experimental setting that prioritizes internal validity over external validity. The main comparisons use 1995-01 to 1996-12, while shorter 1995-only results are treated as preliminary baseline checks. This design still limits generalizability to broader market regimes, but it ensures that observed differences reflect genuine loss function effects rather than artifacts of data snooping or parameter tuning.

The multi-seed testing protocol addresses the concern that performance differences might be due to favorable random initializations rather than systematic advantages of particular loss functions. Because different phases used different seed sets, the exact seeds are reported with the corresponding empirical tables. By computing coefficient of variation across seeds, we quantify the stability of each loss function's performance.

**Focus on Loss Function Impact, Not Architecture or Features**

This research deliberately avoids the temptation to optimize model architecture or engineer new features. The architecture (3-layer MLP with [64, 32, 16] hidden units) was determined via grid search on an earlier period and then fixed for all experiments. The features (cumulative momentum, denoised standardized momentum, raw monthly series) are based on established literature and remain constant across all loss functions.

This focus on loss function impact distinguishes this thesis from the broader literature on financial machine learning, which typically emphasizes architecture search and feature engineering while treating the loss function as a fixed component. By inverting this emphasis, we shed light on an underexplored but potentially high-impact design choice.

The findings from this research have practical implications for practitioners developing quantitative trading strategies. If hybrid loss functions demonstrate superior performance, they can be readily incorporated into existing modeling pipelines without requiring changes to data infrastructure, feature engineering, or model architecture. This represents a low-cost, high-impact intervention that could improve the profitability of machine learning-based trading strategies.
