# Test Window Analysis: Phase 1.5 vs Phase 2.1b

**Date**: 2026-05-05  
**Status**: HYPOTHESIS REJECTED  
**Investigator**: Research Agent

---

## Executive Summary

**Hypothesis**: Different test window lengths between Phase 1.5 and Phase 2.1b caused the 15-62% Sharpe ratio deviations.

**Finding**: **HYPOTHESIS REJECTED**. Both phases used **identical 24-month test windows** (1995-01 to 1996-12).

**Conclusion**: Test window length differences **cannot explain** the observed deviations:
- IMADL: +17.6% (0.464 → 0.546)
- GMADL: +62.4% (0.307 → 0.499)

The root cause must lie elsewhere (data preprocessing, loss implementations, or training dynamics).

---

## Test Window Configuration

### Phase 1.5 (codex/phase15-colab-drive)

**Source**: `run_phase15_robustness.py` lines 303-309

```python
parser.add_argument(
    "--test-start",
    type=str,
    default="1995-01",
    help="Inclusive test period start (YYYY-MM).",
)
parser.add_argument(
    "--test-months",
    type=int,
    default=24,  # ← 24 MONTHS
    help="Number of consecutive test months.",
)
```

**Test Window**: 
- Start: 1995-01
- Duration: 24 months
- End: 1996-12
- Total months: 24

### Phase 2.1b (phase2.2-fix branch)

**Source**: `run_phase2_1b_alignment.py` line 28

```python
parser.add_argument("--test-months", type=int, default=24)
```

**Test Window**:
- Start: 1995-01 (line 27)
- Duration: 24 months
- End: 1996-12
- Total months: 24

### Core Implementation (Both Phases)

**Source**: `sanity_check_core.py` line 71

```python
parser.add_argument(
    "--test-months",
    type=int,
    default=6,  # ← Core default is 6, but OVERRIDDEN by runners
    help="Number of consecutive months for testing.",
)
```

**Note**: The core default of 6 months is **overridden** by both Phase 1.5 and Phase 2.1b runners, which explicitly pass `--test-months 24`.

---

## Verification Evidence

### 1. Phase 1.5 Runner Default

```bash
$ git show codex/phase15-colab-drive:run_phase15_robustness.py | grep -A 5 "test-months"
    "--test-months",
    type=int,
    default=24,
    help="Number of consecutive test months.",
```

### 2. Phase 2.1b Runner Default

```bash
$ grep -A 2 "test-months" run_phase2_1b_alignment.py
    parser.add_argument("--test-months", type=int, default=24)
```

### 3. Command Line Invocation

Both runners pass `--test-months` to `sanity_check_core.py`:

**Phase 1.5**:
```python
command = [
    sys.executable,
    str(resolve_runner_path(run.loss_name)),
    "--test-start", test_start,
    "--test-months", str(test_months),  # ← Passes 24
    ...
]
```

**Phase 2.1b**:
```python
# Inherits from run_phase2_robustness.py
args.test_months = 24  # Default from argument parser
```

---

## Market Conditions Analysis

Since both phases tested on the **same 24-month period** (1995-01 to 1996-12), market conditions are identical.

### S&P 500 Performance (1995-1996)

| Period | Market Regime | Characteristics |
|--------|--------------|-----------------|
| 1995 Q1-Q2 | Bull market recovery | Post-1994 correction rebound |
| 1995 Q3-Q4 | Strong rally | Tech sector growth |
| 1996 Q1-Q2 | Continued growth | Low volatility |
| 1996 Q3-Q4 | "Irrational exuberance" | Greenspan speech (Dec 1996) |

**Overall**: Strong bull market with low volatility, favorable for momentum strategies.

**Implication**: Since both phases tested on the same period, market conditions **cannot explain** performance differences.

---

## Confusion Source: Core Default vs Runner Override

The confusion likely arose from:

1. **Core default**: `sanity_check_core.py` has `default=6` for `--test-months`
2. **Runner override**: Both `run_phase15_robustness.py` and `run_phase2_1b_alignment.py` override this with `default=24`
3. **Documentation ambiguity**: Phase 1.5 Results Summary (line 139) mentions "24-month test window" but doesn't clarify this was the actual configuration

### Phase 1.5 Results Summary Quote

From `doc/phase1.5/Phase1.5-Results-Summary.md` line 139:

> **High Priority:**
> 1. **Extend test period**: Current 24-month test window may be insufficient to evaluate robustness

This confirms Phase 1.5 used a **24-month test window**, not 6 months.

---

## Impact on Sharpe Ratio Calculation

### Statistical Properties

**24-month test window**:
- Sample size: 24 data points
- Degrees of freedom: 23
- Standard error: σ / √24 ≈ 0.204σ
- 95% confidence interval: ±1.96 × 0.204σ ≈ ±0.40σ

**Theoretical 6-month test window** (for comparison):
- Sample size: 6 data points
- Degrees of freedom: 5
- Standard error: σ / √6 ≈ 0.408σ
- 95% confidence interval: ±1.96 × 0.408σ ≈ ±0.80σ

**Conclusion**: A 24-month window provides **2x better precision** than a 6-month window, but both phases used 24 months, so this is irrelevant.

---

## Hypothesis Evaluation

### Original Hypothesis

> Phase 1.5 may have tested on **24 months** (1995-01 to 1996-12)  
> Phase 2.1b may have tested on **6 months** (1995-01 to 1995-06)  
> Different market conditions in different periods can easily cause 15-60% Sharpe deviations

### Findings

| Aspect | Phase 1.5 | Phase 2.1b | Match? |
|--------|-----------|------------|--------|
| Test start | 1995-01 | 1995-01 | ✅ |
| Test months | 24 | 24 | ✅ |
| Test end | 1996-12 | 1996-12 | ✅ |
| Market conditions | Bull market, low vol | Bull market, low vol | ✅ |

**Verdict**: **HYPOTHESIS REJECTED**. Test windows are identical.

---

## Remaining Hypotheses

Since test window differences are ruled out, the 15-62% deviations must be caused by:

### 1. Data Preprocessing Differences (HIGH PROBABILITY)

**Evidence**:
- IMADL and GMADL loss functions are identical between phases
- Yet IMADL shows +17.6% deviation, GMADL shows +62.4% deviation
- This suggests input data differences, not loss function differences

**Next Steps**:
- Compare `Model_Train/data_preprocess.py` between branches
- Compare `Model_Train/features.py` between branches
- Verify CSV data files are byte-identical (checksums)
- Check for floating-point precision differences

### 2. Loss Implementation Differences (CONFIRMED FOR M2)

**Evidence**:
- M2 (hybrid_mul) used λ_dir=5.0 in Phase 1.5
- M2 (hybrid_mul) used λ_dir=1.0 in Phase 2.1b
- This explains the -28.1% M2 deviation

**Status**: Already documented in `01_config_comparison.md`

### 3. Model Architecture Differences (MEDIUM PROBABILITY)

**Evidence**:
- Both phases use `best_hyperparameters.txt`
- But the config file may differ between phases
- PyTorch version differences may affect initialization

**Next Steps**:
- Compare `best_hyperparameters.txt` files
- Compare `Model_Train/models.py` between branches
- Check PyTorch versions

### 4. Random Seed Handling (MEDIUM PROBABILITY)

**Evidence**:
- Despite identical `set_seed()` functions, differences may arise from:
  - PyTorch version differences (CUDA/MPS backend changes)
  - NumPy version differences
  - DataLoader shuffling differences

**Next Steps**:
- Check PyTorch/NumPy versions in Phase 1.5 vs Phase 2.1b environments
- Verify DataLoader shuffle behavior
- Compare first-batch predictions

---

## Recommendations

### Priority 1: Data Preprocessing Investigation (CRITICAL)

The 62.4% GMADL deviation is **too large** to be explained by random seed differences or minor implementation changes. This strongly suggests:

1. **Different CSV data files** between Phase 1.5 and Phase 2.1b
2. **Different feature engineering** (normalization, scaling, outlier handling)
3. **Different data loading order** (despite seed setting)

**Action**: Compare data preprocessing pipelines line-by-line.

### Priority 2: Environment Verification (HIGH)

Document exact versions used in each phase:

| Component | Phase 1.5 | Phase 2.1b |
|-----------|-----------|------------|
| Python | ? | ? |
| PyTorch | ? | ? |
| NumPy | ? | ? |
| Pandas | ? | ? |
| CUDA/MPS | ? | ? |

**Action**: Extract version information from Colab notebooks or requirements.txt.

### Priority 3: Checkpoint Comparison (MEDIUM)

If Phase 1.5 checkpoints are available:

1. Load Phase 1.5 checkpoint
2. Evaluate on Phase 2.1b test data
3. Compare predictions to Phase 2.1b model
4. Identify where divergence occurs (data vs model)

**Action**: Locate Phase 1.5 checkpoints in Google Drive.

---

## Conclusion

**Test window length is NOT the root cause** of the Phase 2.1b alignment failure. Both phases used identical 24-month test windows (1995-01 to 1996-12).

The investigation must now focus on:
1. **Data preprocessing differences** (most likely cause of IMADL/GMADL deviations)
2. **Loss implementation differences** (confirmed cause of M2 deviation)
3. **Environment/version differences** (possible contributing factor)

**Next Document**: `06_data_preprocessing_comparison.md` - Line-by-line comparison of data preprocessing pipelines.

---

**Document Version**: v1.0  
**Created**: 2026-05-05  
**Author**: Research Agent  
**Status**: Hypothesis Rejected - Investigation Continues
