# Phase 2.5 Investigation: Data Preprocessing Comparison

**Investigation Date:** 2026-05-05  
**Investigator:** Research Agent  
**Purpose:** Compare data preprocessing between Phase 1.5 and Phase 2 to identify potential sources of systematic deviations

---

## Executive Summary

**Finding:** Data preprocessing is **IDENTICAL** between Phase 1.5 and Phase 2. Both phases use the same:
- Data loading pipeline (`prepare_panel_data`)
- Feature engineering functions (`build_feature_set_x1`)
- Train/test window parameters
- Feature scaling and normalization

**Conclusion:** The systematic deviations observed in Phase 2.1b (+17.6% IMADL, +62.4% GMADL, -28.1% M2) **cannot be attributed to data preprocessing differences**. The root cause must lie elsewhere (loss implementation, training dynamics, or evaluation methodology).

---

## 1. Data Loading Pipeline

### 1.1 Common Data Preprocessing Module

Both Phase 1.5 and Phase 2 use the **same data preprocessing module**:

**File:** `/Users/roucher/Documents/FYP/Model_Train/data_preprocess.py`

**Pipeline Function:** `prepare_panel_data()`

```python
def prepare_panel_data(
    data_dir: Optional[str] = None,
    pattern: str = "*.csv",
    cleaning_method: str = "ffill",
    columns: Optional[Sequence[str]] = None,
    dtypes: Optional[Dict[str, str]] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline that returns a cleaned panel 
    with basic variables and target.
    """
```

**Pipeline Steps:**
1. **Load raw CSVs** - `load_raw_csvs()` - Batch load and concatenate CSV files
2. **Parse dates** - `parse_dates()` - Convert date column to pandas datetime, sort by PERMNO and date
3. **Clean core columns** - `clean_core_columns()` - Handle missing values in RET/VOL/SHROUT using forward fill
4. **Add basic variables** - `add_basic_variables()` - Compute `r` (simple return) and `to` (turnover)
5. **Add target return** - `add_target_return()` - Shift return by 1 period to create `target_ret`
6. **Drop missing values** - Final cleanup of rows with NaN in `r`, `to`, or `target_ret`

### 1.2 Data Loading Verification

**Phase 1.5 (sanity_check_core.py):**
```python
from Model_Train.data_preprocess import prepare_panel_data

panel = prepare_panel_data(data_dir=args.data_dir, pattern=args.pattern)
```

**Phase 2 (sanity_check_signal_tilted.py):**
```python
from Model_Train.data_preprocess import prepare_panel_data

panel = prepare_panel_data(data_dir=args.data_dir, pattern=args.pattern)
```

**Verification:** ✅ **IDENTICAL** - Both phases import and call the same function with the same default parameters.

---

## 2. Feature Engineering

### 2.1 Feature Set X1 Construction

Both phases use **Feature Set X1** exclusively:

**File:** `/Users/roucher/Documents/FYP/Model_Train/features.py`

**Function:** `build_feature_set_x1(panel, config)`

**Features Generated:**
- **Cumulative Returns (cr):** 1m, 3m, 6m, 9m, 12m lookback windows
- **Cumulative Turnover (co):** 1m, 3m, 6m, 9m, 12m lookback windows
- **Total Features:** 10 features (5 cr + 5 co)

**Feature Computation:**
```python
def build_feature_set_x1(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    lookbacks = [1, 3, 6, 9, 12]
    group = df.groupby("PERMNO")
    df["_r_prev"] = group["r"].shift(1)  # Lag by 1 month
    df["_to_prev"] = group["to"].shift(1)
    
    for window in lookbacks:
        # Cumulative return: prod(1 + r) - 1
        df[f"cr_{window}m"] = rolling_group_apply(df, "_r_prev", window, _cum_return)
        # Cumulative turnover: sum(to)
        df[f"co_{window}m"] = rolling_group_apply(df, "_to_prev", window, np.sum)
```

**Key Properties:**
- **Lagged features:** All features use `shift(1)` to avoid look-ahead bias
- **Rolling windows:** Computed within each PERMNO group
- **No normalization:** X1 features are NOT cross-sectionally normalized (unlike X2)
- **Missing value handling:** Rows with NaN in any feature are dropped

### 2.2 Feature Engineering Verification

**Phase 1.5:**
```python
from Model_Train.features import FeatureConfig, build_feature_set_x1

feature_cfg = FeatureConfig(lookback_months=args.lookback_months)
df_x1 = build_feature_set_x1(panel, feature_cfg)
```

**Phase 2:**
```python
from Model_Train.features import FeatureConfig, build_feature_set_x1

feature_cfg = FeatureConfig(lookback_months=args.lookback_months)
df_x1 = build_feature_set_x1(panel, feature_cfg)
```

**Verification:** ✅ **IDENTICAL** - Both phases use the same feature engineering pipeline.

---

## 3. Train/Test Window Configuration

### 3.1 Default Parameters

**Phase 1.5 (sanity_check_core.py):**
```python
parser.add_argument("--train-start", default="1990-01")
parser.add_argument("--train-end", default="1994-12")
parser.add_argument("--test-start", default="1995-01")
parser.add_argument("--test-months", type=int, default=6)
parser.add_argument("--lookback-months", type=int, default=12)
```

**Phase 2 (sanity_check_signal_tilted.py):**
```python
parser.add_argument("--train-start", default="1990-01")
parser.add_argument("--train-end", default="1994-12")
parser.add_argument("--test-start", default="1995-01")
parser.add_argument("--test-months", type=int, default=6)
parser.add_argument("--lookback-months", type=int, default=12)
```

**Verification:** ✅ **IDENTICAL** - Same default train/test windows.

### 3.2 Phase 2.1b Alignment Configuration

**Phase 2.1b (run_phase2_1b_alignment.py):**
```python
parser.add_argument("--train-start", default="1990-01")
parser.add_argument("--train-end", default="1994-12")
parser.add_argument("--test-start", default="1995-01")
parser.add_argument("--test-months", type=int, default=24)  # Extended to 24 months
parser.add_argument("--lookback-months", type=int, default=12)
```

**Key Difference:** Phase 2.1b uses **24-month test window** instead of 6 months.

**Impact Analysis:**
- **Phase 1.5 Robustness Test:** Used 24-month test window (1995-01 to 1996-12)
- **Phase 2.1b Alignment:** Uses 24-month test window (1995-01 to 1996-12)
- **Verification:** ✅ **CONSISTENT** - Both use 24-month test windows for robustness testing

### 3.3 Train/Test Split Implementation

**Both phases use identical splitting logic:**

```python
def to_period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M")

def build_mask(periods: pd.Series, start: str, end: str) -> pd.Series:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    return (periods >= start_period) & (periods <= end_period)

date_periods = to_period(dates)
train_mask = build_mask(date_periods, args.train_start, args.train_end)
test_periods = month_sequence(args.test_start, args.test_months)
test_mask = date_periods.isin(test_periods)
```

**Verification:** ✅ **IDENTICAL** - Same date filtering and masking logic.

---

## 4. Data Filtering and Cleaning

### 4.1 Missing Value Handling

**Common Pipeline (data_preprocess.py):**

```python
def clean_core_columns(data, columns=("RET", "VOL", "SHROUT"), method="ffill"):
    """Handle missing values using forward fill within each PERMNO group."""
    group = df.groupby("PERMNO", group_keys=False)
    df[list(columns)] = group[list(columns)].ffill()
    df = df.dropna(subset=list(columns))
    return df
```

**Strategy:**
- **Forward fill** within each PERMNO group
- **Drop rows** with remaining NaN values after forward fill
- **Applied to:** RET, VOL, SHROUT columns

**Verification:** ✅ **IDENTICAL** - Both phases use the same missing value handling.

### 4.2 Outlier Filtering

**Phase 1.5:** No explicit outlier filtering in data preprocessing

**Phase 2:** No explicit outlier filtering in data preprocessing

**Note:** Phase 2 introduces **Z-score clipping** in portfolio construction (not data preprocessing):
```python
Z_SCORE_CLIP = 3.0  # Winsorize threshold in bucket z-score
```

**Impact:** This affects portfolio weights, not the training data itself.

**Verification:** ✅ **CONSISTENT** - No outlier filtering in data preprocessing for either phase.

### 4.3 Stock Universe Selection

**Both phases use the same stock universe:**
- **Source:** All stocks in the provided CSV files matching the pattern
- **Filtering:** Only by data availability (non-missing RET, VOL, SHROUT after forward fill)
- **No explicit filters:** No market cap filters, no exchange filters, no sector filters

**Verification:** ✅ **IDENTICAL** - Same stock universe selection.

---

## 5. Target Variable Construction

### 5.1 Target Return Computation

**Common Implementation (data_preprocess.py):**

```python
def add_target_return(data, ret_column="r", id_column="PERMNO", date_column="date"):
    """Add r_{i,t+1} as prediction target via one-period-ahead shift."""
    group = df.groupby(id_column, group_keys=False)
    df["target_ret"] = group[ret_column].shift(-1)  # Shift forward by 1 month
    df = df.dropna(subset=["target_ret"])
    return df
```

**Target Variable:**
- **Definition:** Next month's return `r_{i,t+1}`
- **Computation:** Forward shift by 1 period within each PERMNO
- **No transformations:** Raw returns, no log transforms, no winsorization

**Verification:** ✅ **IDENTICAL** - Same target variable construction.

### 5.2 Return Calculation

**Common Implementation (data_preprocess.py):**

```python
def add_basic_variables(data, ret_column="RET", vol_column="VOL", shrout_column="SHROUT"):
    """Add simple return and turnover variables."""
    df["r"] = pd.to_numeric(df[ret_column], errors="coerce")
    vol = pd.to_numeric(df[vol_column], errors="coerce")
    shrout = pd.to_numeric(df[shrout_column], errors="coerce")
    df["to"] = vol / (shrout * 1000.0)  # Turnover = volume / (shares outstanding * 1000)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
```

**Return Definition:**
- **Simple return:** `r = RET` (no log transform)
- **Turnover:** `to = VOL / (SHROUT * 1000)`
- **Infinity handling:** Replace inf/-inf with NaN

**Verification:** ✅ **IDENTICAL** - Same return and turnover calculations.

---

## 6. Feature Scaling and Normalization

### 6.1 X1 Feature Scaling

**Feature Set X1 (used by both phases):**
- **NO cross-sectional normalization**
- **NO z-score standardization**
- **Raw cumulative returns and turnover values**

**Rationale:** X1 features preserve the magnitude information, which is important for return prediction.

**Verification:** ✅ **IDENTICAL** - No feature scaling in X1 for either phase.

### 6.2 Alternative Feature Sets (Not Used)

**X2 Features (NOT used in Phase 1.5 or Phase 2):**
- Uses cross-sectional z-score normalization
- Excludes most recent month
- Function: `cross_sectional_zscore()`

**X3 Features (NOT used in Phase 1.5 or Phase 2):**
- Uses cross-sectional z-score normalization
- Individual monthly returns
- Function: `cross_sectional_zscore()`

**Verification:** ✅ **CONSISTENT** - Neither phase uses X2 or X3 features.

---

## 7. Data Preprocessing Comparison Summary

| Component | Phase 1.5 | Phase 2 | Status |
|-----------|-----------|---------|--------|
| **Data Loading** | `prepare_panel_data()` | `prepare_panel_data()` | ✅ IDENTICAL |
| **Feature Engineering** | `build_feature_set_x1()` | `build_feature_set_x1()` | ✅ IDENTICAL |
| **Feature Set** | X1 (10 features) | X1 (10 features) | ✅ IDENTICAL |
| **Lookback Window** | 12 months | 12 months | ✅ IDENTICAL |
| **Train Window** | 1990-01 to 1994-12 | 1990-01 to 1994-12 | ✅ IDENTICAL |
| **Test Window** | 1995-01 to 1996-12 (24m) | 1995-01 to 1996-12 (24m) | ✅ IDENTICAL |
| **Missing Value Handling** | Forward fill + drop | Forward fill + drop | ✅ IDENTICAL |
| **Outlier Filtering** | None | None | ✅ IDENTICAL |
| **Stock Universe** | All available stocks | All available stocks | ✅ IDENTICAL |
| **Target Variable** | `r_{i,t+1}` (next month return) | `r_{i,t+1}` (next month return) | ✅ IDENTICAL |
| **Return Calculation** | Simple return (no log) | Simple return (no log) | ✅ IDENTICAL |
| **Feature Scaling** | None (raw X1) | None (raw X1) | ✅ IDENTICAL |
| **Train/Test Split** | Period-based masking | Period-based masking | ✅ IDENTICAL |

---

## 8. Hypothesis: Could Data Differences Cause Deviations?

### 8.1 Observed Deviations (Phase 2.1b vs Phase 1.5)

| Loss | Phase 1.5 Target | Phase 2.1b Observed | Deviation |
|------|------------------|---------------------|-----------|
| **IMADL** | 0.464 | 0.546 | **+17.6%** |
| **GMADL** | 0.307 | 0.499 | **+62.4%** |
| **M2** | 0.914 | 0.657 | **-28.1%** |

### 8.2 Data Preprocessing as Root Cause?

**Analysis:**

1. **Identical Preprocessing Pipeline:** Both phases use the exact same data loading, feature engineering, and train/test splitting code.

2. **Identical Feature Sets:** Both use X1 features (10 features: 5 cumulative returns + 5 cumulative turnover).

3. **Identical Train/Test Windows:** Both use 1990-01 to 1994-12 for training and 1995-01 to 1996-12 for testing.

4. **Identical Data Filtering:** Both use forward fill for missing values and drop rows with remaining NaN.

5. **Identical Target Variable:** Both predict next month's simple return `r_{i,t+1}`.

**Conclusion:** ❌ **Data preprocessing differences CANNOT explain the deviations.**

### 8.3 Alternative Hypotheses

Since data preprocessing is identical, the deviations must be caused by:

1. **Loss Implementation Differences:**
   - Phase 2 introduced new loss variants (IMADL_M2_alpha, IMADL_GMADL_beta, etc.)
   - Potential bugs or numerical instabilities in loss computation
   - **Next Investigation:** Compare loss implementations between phases

2. **Training Dynamics:**
   - Different random seeds affecting weight initialization
   - Different optimization trajectories
   - Gradient clipping or learning rate differences
   - **Next Investigation:** Compare training hyperparameters and optimization settings

3. **Evaluation Methodology:**
   - Portfolio construction differences (weight capping, rebalancing)
   - Sharpe ratio calculation differences
   - **Next Investigation:** Compare evaluation pipelines

4. **Model Architecture:**
   - Different MLP configurations
   - Different activation functions or dropout rates
   - **Next Investigation:** Verify MLP architecture consistency

---

## 9. Recommendations

### 9.1 Immediate Actions

1. **✅ Data Preprocessing:** Verified as identical - no further investigation needed.

2. **🔍 Loss Implementation:** Compare loss function implementations between Phase 1.5 and Phase 2.
   - Check for numerical stability issues
   - Verify gradient computation
   - Compare loss values during training

3. **🔍 Training Dynamics:** Compare training hyperparameters and optimization settings.
   - Learning rate, batch size, epochs
   - Weight initialization seeds
   - Gradient clipping thresholds

4. **🔍 Evaluation Pipeline:** Compare portfolio construction and Sharpe calculation.
   - Weight capping implementation
   - Rebalancing logic
   - Sharpe ratio formula

### 9.2 Next Investigation Priority

**Priority 1:** Loss Implementation Comparison (Document 05)
- Most likely source of systematic deviations
- IMADL and GMADL both show positive deviations
- M2 shows negative deviation

**Priority 2:** Training Dynamics Analysis (Document 06)
- Could explain seed-specific variations
- Less likely to cause systematic deviations across all seeds

**Priority 3:** Evaluation Methodology (Document 07)
- Could affect Sharpe calculation
- Less likely given that Phase 1.5 results are well-established

---

## 10. Conclusion

**Finding:** Data preprocessing is **completely identical** between Phase 1.5 and Phase 2. Both phases use:
- The same data loading pipeline (`prepare_panel_data`)
- The same feature engineering functions (`build_feature_set_x1`)
- The same train/test window parameters (1990-01 to 1994-12 train, 1995-01 to 1996-12 test)
- The same feature set (X1: 10 features)
- The same target variable (next month simple return)
- The same missing value handling (forward fill + drop)
- No feature scaling or normalization

**Implication:** The systematic deviations observed in Phase 2.1b (+17.6% IMADL, +62.4% GMADL, -28.1% M2) **cannot be attributed to data preprocessing differences**.

**Next Step:** Investigate loss implementation differences between Phase 1.5 and Phase 2 (Document 05).

---

**Document Status:** ✅ Complete  
**Next Document:** `05_loss_implementation.md` - Compare loss function implementations
