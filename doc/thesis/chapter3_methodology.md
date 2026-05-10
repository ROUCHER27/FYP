# Chapter 3: Methodology

## 3.1 Overview

This study employs a controlled experimental design to systematically evaluate the impact of different loss functions on neural network-based cross-sectional stock return prediction. The core principle is to hold all aspects of the modeling pipeline constant—data, features, architecture, training protocol—while varying only the loss function. This isolation allows causal attribution of performance differences to the loss function rather than confounding factors.

The research design follows a static sanity check protocol: train once on historical data (1990-1994), then predict out-of-sample without any parameter updates or retraining. Early baseline checks used a short 6-month window (1995-01 to 1995-06), but the main Phase 1.5-2.2 empirical comparisons use a 24-month window (1995-01 to 1996-12). This conservative approach prioritizes experimental control and internal validity over realistic deployment simulation. While a rolling-window backtest with periodic retraining would better reflect production usage, the static protocol eliminates concerns about data snooping and ensures that all loss functions are evaluated under identical conditions within each reported table.

The experimental workflow proceeds in five phases:

1. **Phase 1**: Baseline comparison of seven loss functions (MSE, MedSE, MADL, GMADL, IMADL, M1, M2) with single seed
2. **Phase 1.5**: Lambda sweep for M1/M2 hybrid losses across four λ_dir values with three seeds
3. **Phase 2**: Hybrid variant testing with four configurations and four hyperparameter settings, three seeds each
4. **Phase 2.2**: Gamma refinement for M2_robust across five gamma values, three seeds each
5. **Phase 2.2-fix1**: Normalization test comparing three losses with three seeds

This chapter provides detailed documentation of each component to ensure reproducibility.

## 3.2 Data Description and Preprocessing

**Data Source**

Raw data is sourced from the Center for Research in Security Prices (CRSP) monthly stock database via Wharton Research Data Services (WRDS). CRSP provides comprehensive coverage of US-listed equities with high-quality data on prices, returns, volume, and shares outstanding. The database is widely used in academic finance research and is considered the gold standard for US equity data.

**Sample Period**

The full sample spans January 1990 to December 2025, providing 35 years of monthly observations. This period includes multiple market regimes: the dot-com boom and bust (1995-2002), the financial crisis (2007-2009), the post-crisis recovery (2010-2019), the COVID-19 pandemic (2020), and the subsequent recovery. For the controlled static experiments reported in the main empirical tables, the first five years (1990-1994) are used for training, with the subsequent 24 months (1995-01 to 1996-12) used for out-of-sample testing. The shorter 1995-01 to 1995-06 window is retained only as an early sanity-check reference.

**Variables**

Three primary variables are extracted from CRSP:

1. **Monthly Return (RET)**: The holding period return for stock i in month t, calculated as (P_t - P_{t-1} + D_t) / P_{t-1}, where P_t is the price at the end of month t and D_t is any dividends paid during month t. Returns are expressed as decimals (e.g., 0.05 for 5%).

2. **Volume (VOL)**: The total number of shares traded during month t. This is used to construct turnover measures.

3. **Shares Outstanding (SHROUT)**: The number of shares outstanding at the end of month t, measured in thousands. This is used to calculate turnover ratios.

**Preprocessing Steps**

The raw CRSP data undergoes several preprocessing steps to ensure data quality and consistency:

1. **Missing Value Handling**: Observations with missing values in any of the three key variables (RET, VOL, SHROUT) are excluded from the analysis. For stocks with intermittent missing data, forward fill is applied for up to one month, after which the observation is dropped. This balances the need to retain data with the requirement for complete feature vectors.

2. **Outlier Filtering**: Extreme returns that likely reflect data errors rather than genuine price movements are filtered. Specifically, returns below -0.95 (95% loss) or above 10.0 (1000% gain) are flagged and excluded. This affects less than 0.1% of observations but prevents obvious data errors from distorting model training.

3. **Delisting Returns**: CRSP provides delisting returns for stocks that exit the database due to mergers, bankruptcies, or other corporate actions. These are incorporated into the return series to avoid survivorship bias.

4. **No Feature Scaling**: Unlike many machine learning applications, the raw features are not standardized or normalized at the individual stock level. Instead, cross-sectional standardization (z-scores computed across all stocks in a given month) is applied only to specific feature sets (X² and X³) as described in Section 3.3. This preserves the economic interpretation of momentum and turnover measures.

**Train-Test Split**

The data is divided into non-overlapping train and test periods:

- **Training Window**: January 1990 to December 1994 (60 months, 5 years)
- **Main Test Window**: January 1995 to December 1996 (24 months)
- **Early Sanity-Check Window**: January 1995 to June 1995 (6 months), used only for preliminary baseline checks

This split ensures that the test period is strictly out-of-sample with no information leakage from future data. The 5-year training window provides sufficient data for neural network training (typically 3,000-5,000 stock-month observations per month, totaling 180,000-300,000 observations depending on the number of listed stocks).

The model architecture was determined via grid search on an even earlier period (1989-1994) to avoid look-ahead bias. This ensures that no aspect of the model design—architecture, hyperparameters, or features—was tuned on the test period.

## 3.3 Feature Engineering

Feature engineering transforms raw price and volume data into predictive signals. Based on the literature (Daniel and Moskowitz 2016; Gu et al. 2020), three complementary feature sets are constructed, each capturing different aspects of price momentum and trading activity.

### Feature Set 1 (X¹): Cumulative Momentum and Turnover

This feature set captures basic price momentum and trading activity over multiple horizons. For each stock i in month t, we compute cumulative returns and cumulative turnover over m ∈ {1, 3, 6, 9, 12} months:

```math
cr_{i,m} = \prod_{j=t-m}^{t} (1 + r_{ij}) - 1
```

```math
co_{i,m} = \sum_{j=t-m}^{t} to_{ij}
```

where r_{ij} is the simple return for stock i in month j, and to_{ij} is the turnover ratio (volume divided by shares outstanding) in month j.

**Rationale**: Cumulative returns capture momentum effects documented extensively in the literature. Stocks with strong recent performance tend to continue outperforming in the near term (momentum effect), while very short-term returns may exhibit reversal. By including multiple horizons (1, 3, 6, 9, 12 months), we allow the model to learn which momentum windows are most predictive.

Cumulative turnover measures trading activity, which is associated with liquidity and information flow. High turnover may indicate increased investor attention or information arrival, both of which can predict future returns. The cumulative measure smooths out monthly noise in volume data.

**Dimensionality**: This feature set produces 10 features per stock-month (5 momentum + 5 turnover).

### Feature Set 2 (X²): Denoised Standardized Momentum

This feature set refines the momentum calculation by excluding the most recent month and applying cross-sectional standardization:

```math
cr^*_{i,m} = \prod_{j=t-m}^{t-1} (1 + r_{ij}) - 1
```

The cumulative return is computed from t-m to t-1 (excluding month t), then z-score standardized cross-sectionally:

```math
z_{i,m} = \frac{cr^*_{i,m} - \mu_m}{\sigma_m}
```

where μ_m and σ_m are the cross-sectional mean and standard deviation of cr*_{i,m} across all stocks in month t.

**Rationale**: Excluding the most recent month (t-1 to t) avoids short-term reversal effects documented by Jegadeesh (1990). Returns over the past month tend to reverse, while returns over longer horizons (2-12 months) exhibit momentum. By excluding the most recent month, we isolate the momentum component from the reversal component.

Cross-sectional standardization removes time-varying market-wide effects. In months when all stocks rise (bull markets), raw momentum measures are uniformly high, providing little cross-sectional discrimination. Z-scores focus on relative performance, identifying stocks that outperformed or underperformed their peers.

**Dimensionality**: This feature set produces 5 features per stock-month (one z-score for each of m ∈ {1, 3, 6, 9, 12}).

### Feature Set 3 (X³): Raw Monthly Series

This feature set provides the neural network with raw monthly returns from t-1 to t-12, allowing it to extract non-linear time dependencies:

```math
X^3_i = [r_{i,t-1}, r_{i,t-2}, \ldots, r_{i,t-12}]
```

Each monthly return is z-score standardized cross-sectionally before being fed to the network.

**Rationale**: While X¹ and X² provide hand-crafted momentum features based on domain knowledge, X³ allows the neural network to discover its own temporal patterns. The network may learn non-linear combinations of past returns, interactions between different lags, or time-varying relationships that are not captured by simple cumulative measures.

This approach follows Gu et al. (2020), who demonstrated that deep networks can extract complex patterns from raw features that outperform hand-crafted transformations. By including both hand-crafted (X¹, X²) and raw (X³) features, we combine domain expertise with the network's ability to discover novel patterns.

**Dimensionality**: This feature set produces 12 features per stock-month (one for each of the past 12 months).

### Combined Feature Vector

The three feature sets are concatenated to form the final input vector for each stock-month observation:

```math
X_i = [X^1_i, X^2_i, X^3_i]
```

**Total Dimensionality**: 10 + 5 + 12 = 27 features per observation.

This combined representation provides the neural network with multiple views of the same underlying data: cumulative measures (X¹), denoised relative measures (X²), and raw temporal sequences (X³). The network can learn to weight these different representations based on their predictive value.

## 3.4 Model Architecture

A Multilayer Perceptron (MLP) architecture is employed for all experiments. The architecture was determined via grid search on the period 1989-1994 (prior to the training window) to avoid look-ahead bias. Once selected, the architecture remains fixed across all loss function experiments to ensure fair comparison.

**Architecture Specification**

- **Input Layer**: 27 features (as described in Section 3.3)
- **Hidden Layer 1**: 64 units, Tanh activation, Dropout (p=0.2)
- **Hidden Layer 2**: 32 units, Tanh activation, Dropout (p=0.2)
- **Hidden Layer 3**: 16 units, Tanh activation, Dropout (p=0.2)
- **Output Layer**: 1 unit, Linear activation (predicting next month's return)

**Total Parameters**: 3,665 trainable parameters
- Layer 1: 27 × 64 + 64 = 1,792
- Layer 2: 64 × 32 + 32 = 2,080
- Layer 3: 32 × 16 + 16 = 528
- Output: 16 × 1 + 1 = 17

**Activation Function**: Tanh is used for all hidden layers. Tanh maps inputs to the range (-1, 1) and is zero-centered, which can facilitate training compared to sigmoid activation. It provides non-linearity necessary for the network to learn complex patterns while maintaining smooth gradients.

**Regularization**: Dropout with probability p=0.2 is applied after each hidden layer. Dropout randomly sets 20% of activations to zero during training, forcing the network to learn redundant representations and preventing co-adaptation of features. This improves generalization by reducing overfitting.

**Optimizer**: Adam optimizer with learning rate 1e-3 (0.001). Adam combines the benefits of momentum-based optimization with adaptive learning rates for each parameter. The default learning rate of 1e-3 is used without tuning, as preliminary experiments showed it provides stable convergence across different loss functions.

**Batch Size**: 1,024 observations per batch. This batch size balances computational efficiency with gradient estimate quality. Larger batches provide more stable gradient estimates but require more memory and may converge to sharper minima. Smaller batches introduce more noise but can help escape poor local minima.

**Training Duration**: Maximum 20 epochs. Early stopping is not employed; all models train for the full 20 epochs. This ensures that all loss functions receive equal training time and that differences in convergence speed do not confound the comparison.

**Rationale for Architecture Choice**

The relatively shallow architecture (3 hidden layers) with moderate width (64-32-16) was chosen based on several considerations:

1. **Sample Size**: With approximately 180,000-300,000 training observations, a network with 3,665 parameters provides a reasonable parameter-to-data ratio (roughly 1:50 to 1:80), reducing overfitting risk.

2. **Interpretability**: Shallower networks are easier to analyze and debug than very deep architectures. Given that the focus is on loss function comparison rather than achieving state-of-the-art prediction accuracy, a simpler architecture is preferable.

3. **Computational Efficiency**: The chosen architecture trains quickly (minutes rather than hours), enabling the extensive experimental program (103 total runs across all phases).

4. **Prior Validation**: Grid search on 1989-1994 data tested architectures ranging from 2 to 5 hidden layers and widths from 32 to 128 units. The [64, 32, 16] configuration achieved the best validation performance without excessive overfitting.

## 3.5 Loss Functions

This section documents all loss functions tested in the experimental program. Each loss function is presented with its mathematical formulation, key properties, and implementation details.

### Baseline Losses

**1. Mean Squared Error (MSE)**

```math
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

The standard regression loss that penalizes squared errors. Sensitive to outliers due to quadratic penalty.

**2. Median Squared Error (MedSE)**

```math
L_{MedSE} = \text{median}[(y_i - \hat{y}_i)^2]
```

Robust alternative to MSE that uses the median instead of mean. Resistant to outliers but requires processing entire batch simultaneously.

**3. Mean Absolute Directional Loss (MADL)**

```math
L_{MADL} = -\tanh(25 \cdot y \cdot \hat{y}) \times |y|
```

Directional loss that rewards correct sign predictions weighted by true return magnitude. The tanh function provides smooth transition between reward and penalty.

**4. Generalized MADL (GMADL)**

```math
L_{GMADL} = -[\sigma(100 \cdot y \cdot \hat{y}) - 0.5] \times |y|^2
```

where σ is the sigmoid function. Extends MADL with sigmoid activation and squared magnitude weighting.

**5. Improved MADL (IMADL)**

```math
L_{IMADL} = -\tanh(25 \cdot y \cdot \hat{y}) \times |y| + \alpha \cdot (y - \hat{y})^2
```

Combines MADL directional term with MSE magnitude term (α = 0.1). Attempts to balance directional correctness with magnitude accuracy.

### Hybrid Losses

**6. M1 (Multiplicative Hybrid)**

```math
L_{M1} = (1 + \lambda_{dir} \cdot \text{dir\_term}) \times \text{huber\_term}
```

where:
```math
\text{dir\_term} = -[\sigma(100 \cdot y \cdot \hat{y}) - 0.5]
```

```math
\text{huber\_term} = \begin{cases}
0.5 \cdot (y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - 0.5 \cdot \delta^2 & \text{otherwise}
\end{cases}
```

Multiplicative combination of directional penalty and Huber loss. The parameter λ_dir controls the strength of directional influence (tested values: 0.5, 1.0, 2.0, 5.0).

**7. M2 (Multiplicative Hybrid, Variant)**

Same formulation as M1 but with different default λ_dir value. In practice, M1 and M2 differ only in their hyperparameter settings during Phase 1.5 lambda sweep.

**8. M2_robust (Enhanced Robustness)**

```math
L_{M2\_robust} = (1 + \lambda_{dir} \cdot \text{dir\_term}) \times \text{huber\_term}(\gamma)
```

where the Huber threshold δ is replaced with a learnable or tuned parameter γ. Additional robustness enhancements include:
- Adaptive threshold based on residual distribution
- Gradient clipping to prevent extreme updates
- Outlier detection and down-weighting

The gamma parameter controls the transition point between quadratic and linear regimes (tested values: 0.5, 1.0, 1.5, 2.0, 2.5).

### Implementation Details

All loss functions are implemented in PyTorch and support batch-wise computation for efficient training. Numerical stability is ensured through:
- Clipping of sigmoid/tanh inputs to prevent overflow
- Addition of small epsilon (1e-8) in denominators to prevent division by zero
- Gradient clipping at norm 1.0 to prevent exploding gradients

## 3.6 Training Protocol

The training protocol is designed to ensure fair comparison across all loss functions by maintaining identical conditions except for the loss function itself.

**Static Sanity Check**

The core protocol is a static sanity check: train once on the training window (1990-1994), then predict on the fixed out-of-sample test window without any parameter updates or retraining. The main empirical comparisons use 1995-01 to 1996-12; early MSE/MedSE baseline checks used 1995-01 to 1995-06 only as preliminary diagnostics. This differs from realistic deployment in two ways:

1. **No Retraining**: In production, models would be retrained periodically (e.g., monthly or quarterly) as new data arrives. The static protocol trains once and uses those fixed parameters for all test period predictions.

2. **No Parameter Tuning**: Hyperparameters (learning rate, batch size, dropout rate) are fixed in advance and not tuned on the test set. This prevents data snooping but may result in suboptimal performance for some loss functions.

The static protocol prioritizes experimental control over realism. By eliminating retraining and parameter tuning, we ensure that all loss functions are evaluated under identical conditions and that performance differences reflect genuine loss function effects rather than artifacts of the training schedule or hyperparameter choices.

**Training Procedure**

For each experimental run:

1. **Initialization**: Network weights are initialized using PyTorch's default initialization (Kaiming uniform for linear layers). The random seed is set to ensure reproducibility (seeds: 42, 123, 456 for multi-seed experiments).

2. **Data Loading**: Training data (1990-1994) is loaded and shuffled. Batches of size 1,024 are created.

3. **Training Loop**: For each of 20 epochs:
   - Iterate through all training batches
   - Forward pass: compute predictions ŷ
   - Loss computation: evaluate loss function L(y, ŷ)
   - Backward pass: compute gradients via backpropagation
   - Parameter update: apply Adam optimizer step
   - No validation set evaluation during training

4. **Final Model**: After 20 epochs, the model parameters are frozen.

5. **Test Prediction**: For each month in the reported test window:
   - Load features for all stocks in that month
   - Forward pass through frozen network to generate predictions
   - Store predictions for portfolio construction

**Deterministic Settings**

To ensure reproducibility, the following deterministic settings are enforced:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings eliminate sources of randomness in network initialization, data shuffling, and GPU operations. Combined with fixed random seeds, this ensures that each experimental run is exactly reproducible.

**Multi-Seed Testing**

To assess robustness, most experiments are repeated with three different random seeds: 42, 123, and 456. This tests whether performance differences are consistent across different random initializations or merely artifacts of favorable seeds.

For each loss function, we compute:
- Mean performance across seeds
- Standard deviation across seeds
- Coefficient of variation (CV = σ/μ) as a measure of stability

Lower CV indicates more stable, reliable performance.

## 3.7 Portfolio Construction

Predictions from the neural network are translated into portfolio positions using three different strategies. Each strategy represents a different approach to weighting stocks based on predicted returns.

### P1: Equal Weighted Long-Short

**Construction**:
1. Rank all stocks by predicted return ŷ
2. Long: Top 10% (equal weights of 1/N_long each)
3. Short: Bottom 10% (equal weights of -1/N_short each)
4. Rebalance monthly

**Rationale**: This is the simplest strategy, treating all stocks in the long (short) bucket identically. It tests whether the model can correctly identify the direction of relative performance without requiring precise magnitude estimates. Equal weighting also provides natural diversification within each bucket.

**Portfolio Return**:
```math
R_t^{P1} = \frac{1}{N_{long}} \sum_{i \in \text{Long}} r_{i,t} - \frac{1}{N_{short}} \sum_{i \in \text{Short}} r_{i,t}
```

### P2: Signal Weighted Long-Short

**Construction**:
1. Rank all stocks by predicted return ŷ
2. Long: Top 10%, weighted by |ŷ_i| / Σ|ŷ_j| within bucket
3. Short: Bottom 10%, weighted by |ŷ_i| / Σ|ŷ_j| within bucket
4. Rebalance monthly

**Rationale**: This strategy uses the magnitude of predictions to determine position sizes. Stocks with stronger predicted returns receive larger weights, allowing the model to express confidence through position sizing. This tests whether the model's magnitude estimates contain useful information beyond directional signals.

**Portfolio Return**:
```math
R_t^{P2} = \sum_{i \in \text{Long}} w_i \cdot r_{i,t} - \sum_{i \in \text{Short}} w_i \cdot r_{i,t}
```

where:
```math
w_i = \frac{|\hat{y}_i|}{\sum_{j \in \text{bucket}} |\hat{y}_j|}
```

### P3: Capped Signal Weighted

**Construction**:
1. Same as P2, but individual stock weights are capped at 5%
2. If a stock's weight exceeds 5%, the excess is redistributed proportionally to other stocks in the bucket
3. Rebalance monthly

**Rationale**: This strategy addresses concentration risk in P2. If the model assigns very high predicted returns to a few stocks, P2 may result in concentrated positions vulnerable to idiosyncratic risk. The 5% cap limits exposure to any single stock while still allowing signal-based weighting.

**Portfolio Return**: Same formula as P2, but with capped weights:
```math
w_i^{cap} = \min(w_i, 0.05)
```

followed by renormalization to ensure weights sum to 1 within each bucket.

### Rebalancing and Transaction Costs

All three strategies rebalance monthly at the end of each month. Transaction costs are not modeled in the current analysis, which represents a limitation. In practice, P2 and P3 may have higher turnover than P1 due to time-varying signal strengths, and the net returns after costs could differ from the gross returns reported here.

## 3.8 Evaluation Metrics

Performance is evaluated across two dimensions: statistical prediction accuracy and economic portfolio performance.

### Statistical Metrics

**Out-of-Sample R² (R²_oos)**

```math
R^2_{oos} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

where ȳ is the mean of actual returns in the test set. R²_oos measures the proportion of return variance explained by the model. Unlike in-sample R², out-of-sample R² can be negative if the model performs worse than a naive mean prediction.

**Mean Squared Error (MSE)**

```math
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

Standard measure of prediction accuracy. Lower values indicate better fit.

**Median Squared Error (MedSE)**

```math
MedSE = \text{median}[(y_i - \hat{y}_i)^2]
```

Robust alternative to MSE that is less sensitive to outliers.

### Economic Metrics

**Cumulative Return**

```math
R_{cum} = \prod_{t=1}^{T} (1 + R_t) - 1
```

Total return over the test period, compounded monthly.

**Annualized Return**

```math
R_{ann} = (1 + R_{cum})^{12/T} - 1
```

where T is the number of months in the reported test period. For the main empirical tables, T = 24. For early baseline sanity checks, T = 6 and the table is explicitly labelled as preliminary. Annualized return facilitates comparison with other strategies and benchmarks.

**Sharpe Ratio**

```math
SR = \frac{R_p - R_f}{\sigma_p}
```

where R_p is the mean portfolio return, R_f is the risk-free rate (assumed to be 0 for simplicity), and σ_p is the standard deviation of portfolio returns. Sharpe ratio measures risk-adjusted performance, with higher values indicating better return per unit of risk.

**Volatility (Standard Deviation)**

```math
\sigma_p = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (R_t - \bar{R})^2}
```

Annualized by multiplying by √12. Measures the variability of portfolio returns.

### Robustness Metrics

**Coefficient of Variation (CV)**

```math
CV = \frac{\sigma}{\mu}
```

Computed across the three random seeds for each loss function. Lower CV indicates more stable performance that is less dependent on random initialization.

## 3.9 Experimental Design

The experimental program consists of five phases, each designed to answer specific research questions.

### Phase 1: Baseline Comparison

**Objective**: Compare seven loss functions (MSE, MedSE, MADL, GMADL, IMADL, M1, M2) under identical conditions.

**Design**: 7 losses × 1 seed (42) = 7 runs

**Evaluation**: Statistical metrics (R², MSE, MedSE) and economic metrics (Sharpe, returns) for all three portfolio strategies (P1, P2, P3).

### Phase 1.5: Lambda Sweep

**Objective**: Determine optimal directional weighting parameter λ_dir for hybrid losses M1 and M2.

**Design**: 2 losses (M1, M2) × 4 λ_dir values (0.5, 1.0, 2.0, 5.0) × 3 seeds (42, 123, 456) = 24 runs

**Evaluation**: Performance across λ_dir values to identify optimal balance between directional and magnitude components.

### Phase 2: Hybrid Variant Testing

**Objective**: Test robustness enhancements and alternative formulations of hybrid losses.

**Design**: 4 variants × 4 hyperparameter settings × 3 seeds = 48 runs

**Evaluation**: Compare enhanced robustness mechanisms against baseline hybrid losses.

### Phase 2.2: Gamma Refinement

**Objective**: Fine-tune the Huber threshold parameter γ for M2_robust.

**Design**: 5 gamma values (0.5, 1.0, 1.5, 2.0, 2.5) × 3 seeds = 15 runs

**Evaluation**: Identify optimal transition point between quadratic and linear regimes.

### Phase 2.2-fix1: Normalization Test

**Objective**: Test impact of feature normalization on loss function performance.

**Design**: 3 losses × 3 seeds = 9 runs

**Evaluation**: Compare performance with and without normalization.

**Total Experimental Runs**: 7 + 24 + 48 + 15 + 9 = 103 runs

This comprehensive experimental design ensures that loss function effects are thoroughly characterized across multiple dimensions: different hyperparameter settings, random seeds, and portfolio construction methods. The large number of runs provides statistical power to detect meaningful performance differences while the multi-seed protocol guards against spurious findings due to favorable random initializations.
