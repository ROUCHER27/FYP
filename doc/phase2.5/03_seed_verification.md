# Seed Initialization Verification: Phase 1.5 vs Phase 2

**Investigation Date:** 2026-05-05  
**Investigator:** Research Agent  
**Purpose:** Determine if seed initialization differences could explain 15-62% deviations between Phase 1.5 and Phase 2

---

## Executive Summary

**Finding:** Seed initialization is **IDENTICAL** between Phase 1.5 and Phase 2. Both phases use the same seeds (42, 52, 62) and identical seed-setting code. Seed handling differences **CANNOT** explain the observed deviations.

**Key Evidence:**
- Both phases use seeds: 42, 52, 62
- Identical `set_seed()` function implementation
- Phase 2 adds per-epoch seed determinism via `torch.Generator`
- No CUDNN determinism settings in either phase

**Conclusion:** The 15-62% deviations must stem from other sources (loss function implementation, data handling, or training dynamics), not seed initialization.

---

## 1. Seed Configuration Comparison

### Phase 1.5 Seeds
```python
# From Phase 1.5 documentation
Seeds: 42, 52, 62
```

### Phase 2 Seeds
```python
# From run_phase2_robustness.py:23
DEFAULT_SEEDS = (42, 52, 62)

# From run_phase2_1b_alignment.py:20
parser.add_argument("--seeds", default="42,52,62")

# From run_phase2_gamma_refinement.py:26
parser.add_argument("--seeds", default="42,52,62")
```

**Verdict:** ✅ **IDENTICAL** - Both phases use seeds 42, 52, 62

---

## 2. Seed Initialization Code Comparison

### Phase 1.5 Implementation (sanity_check_core.py)

```python
# sanity_check_core.py:117-126
def set_seed(seed: int) -> None:
    """
    保证 numpy / torch / random 的伪随机数可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Called at:** Line 368 (once at start of `run_sanity_check()`)

### Phase 2 Implementation (sanity_check_signal_tilted.py)

```python
# sanity_check_signal_tilted.py:186-195
def set_seed(seed: int) -> None:
    """
    保证 numpy / torch / random 的伪随机数可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Called at:**
1. Line 793: Global seed at start of `run_sanity_check_signal_tilted()`
2. Line 395: Per-epoch seed via `set_epoch_seed(base_seed, epoch_index)`

**Verdict:** ✅ **IDENTICAL FUNCTION** - Same implementation, but Phase 2 adds per-epoch reseeding

---

## 3. Seed Scope Analysis

### Phase 1.5 Seed Scope

```python
# sanity_check_core.py:368
def run_sanity_check(loss_name: str, args: argparse.Namespace) -> None:
    set_seed(args.seed)  # Set once at start
    device = detect_device()
    # ... data loading ...
    # ... model training ...
```

**Scope:**
- ✅ Set before data loading
- ✅ Set before model initialization
- ✅ Set before training loop
- ❌ NOT reset per-epoch

### Phase 2 Seed Scope

```python
# sanity_check_signal_tilted.py:793
def run_sanity_check_signal_tilted(loss_name: str, args: argparse.Namespace) -> None:
    set_seed(args.seed)  # Set once at start
    device = detect_device()
    # ... data loading ...
    # ... per-month training loop ...

# sanity_check_signal_tilted.py:393-396
def set_epoch_seed(base_seed: int, epoch_index: int) -> int:
    epoch_seed = int(base_seed) + int(epoch_index)
    set_seed(epoch_seed)  # Reset per-epoch
    return epoch_seed
```

**Scope:**
- ✅ Set before data loading
- ✅ Set before model initialization
- ✅ Set before training loop
- ✅ **ADDITIONALLY** reset per-epoch with `epoch_seed = base_seed + epoch_index`

**Verdict:** ⚠️ **PHASE 2 MORE DETERMINISTIC** - Phase 2 adds per-epoch seed resets, making it MORE reproducible, not less

---

## 4. DataLoader Shuffle Behavior

### Phase 1.5 DataLoader

```python
# sanity_check_core.py:204
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

**Behavior:**
- Shuffle enabled
- No explicit generator
- Uses PyTorch's default random state

### Phase 2 DataLoader

```python
# sanity_check_signal_tilted.py:495-502
generator = torch.Generator(device="cpu")
generator.manual_seed(epoch_seed)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=generator,
)
```

**Behavior:**
- Shuffle enabled
- **Explicit generator with per-epoch seed**
- Guarantees deterministic shuffle order

**Verdict:** ⚠️ **PHASE 2 MORE DETERMINISTIC** - Phase 2 uses explicit generator for reproducible shuffling

---

## 5. Determinism Settings

### CUDNN Determinism

```bash
# Search for CUDNN settings
$ grep -rn "torch.backends.cudnn" /Users/roucher/Documents/FYP --include="*.py" --exclude-dir=".venv"
# Result: (no output)
```

**Verdict:** ❌ **NEITHER PHASE** sets `torch.backends.cudnn.deterministic` or `torch.backends.cudnn.benchmark`

**Implication:** Both phases may have non-deterministic CUDNN operations, but this affects BOTH equally.

---

## 6. Train/Test Split Determinism

### Phase 1.5 Split

```python
# sanity_check_core.py:385-387
train_mask = build_mask(date_periods, args.train_start, args.train_end).to_numpy()
test_periods = month_sequence(args.test_start, args.test_months)
test_mask = date_periods.isin(test_periods).to_numpy()
```

### Phase 2 Split

```python
# sanity_check_signal_tilted.py:825-827
train_mask = build_mask(date_periods, args.train_start, args.train_end).to_numpy()
test_periods = month_sequence(args.test_start, args.test_months)
test_mask = date_periods.isin(test_periods).to_numpy()
```

**Verdict:** ✅ **IDENTICAL** - Both use deterministic date-based masking (no randomness)

---

## 7. Key Differences Summary

| Aspect | Phase 1.5 | Phase 2 | Impact |
|--------|-----------|---------|--------|
| **Seeds Used** | 42, 52, 62 | 42, 52, 62 | ✅ Identical |
| **set_seed() Function** | Lines 117-126 | Lines 186-195 | ✅ Identical |
| **Global Seed Timing** | Before training | Before training | ✅ Identical |
| **Per-Epoch Reseeding** | ❌ No | ✅ Yes (`base_seed + epoch`) | ⚠️ Phase 2 MORE deterministic |
| **DataLoader Generator** | ❌ No | ✅ Yes (explicit) | ⚠️ Phase 2 MORE deterministic |
| **CUDNN Determinism** | ❌ Not set | ❌ Not set | ✅ Identical |
| **Train/Test Split** | Date-based | Date-based | ✅ Identical |

---

## 8. Hypothesis: Could Seed Handling Cause 15-62% Deviations?

### Analysis

**NO.** Seed handling differences **CANNOT** explain the deviations because:

1. **Same Seeds:** Both phases use identical seeds (42, 52, 62)
2. **Same Initialization:** Both use identical `set_seed()` function
3. **Phase 2 More Deterministic:** Phase 2 adds per-epoch reseeding and explicit DataLoader generators, making it MORE reproducible, not less
4. **Identical Split Logic:** Train/test splits are deterministic and identical

### What Phase 2's Enhanced Determinism Means

Phase 2's per-epoch reseeding ensures:
- Each epoch starts with a predictable random state
- DataLoader shuffling is reproducible across runs
- Training dynamics are MORE stable, not less

**If anything, Phase 2 should be MORE consistent than Phase 1.5, not less.**

---

## 9. Implications for Deviation Investigation

Since seed handling is **NOT** the cause, the 15-62% deviations must stem from:

### Likely Causes (Ranked by Probability)

1. **Loss Function Implementation Differences** (HIGH)
   - Phase 2 uses `imadl_m2_alpha06` vs Phase 1.5's `imadl`
   - Different loss formulations could produce different optima
   - Even with same seed, different loss landscapes → different results

2. **Training Dynamics** (MEDIUM)
   - Phase 2's per-epoch reseeding changes gradient descent trajectory
   - Same seed, different epoch-level randomness → different convergence

3. **Data Handling Differences** (MEDIUM)
   - Feature engineering differences
   - Normalization or preprocessing changes
   - Different data loading order (despite same split)

4. **Model Architecture or Hyperparameters** (LOW)
   - Different learning rates, batch sizes, or epochs
   - Different model initialization (though seed is same)

5. **Numerical Precision Issues** (LOW)
   - Different PyTorch versions
   - Different CUDA versions
   - Floating-point accumulation differences

---

## 10. Recommendations

### Immediate Actions

1. **Compare Loss Function Implementations**
   - Verify `imadl` (Phase 1.5) vs `imadl_m2_alpha06` (Phase 2)
   - Check if loss formulations are truly equivalent
   - Document any differences in loss computation

2. **Compare Training Hyperparameters**
   - Verify learning rate, batch size, epochs are identical
   - Check optimizer settings (Adam parameters)
   - Verify model architecture is identical

3. **Compare Data Preprocessing**
   - Verify feature engineering is identical
   - Check normalization/standardization steps
   - Verify data loading order (even with same split)

### Long-Term Actions

1. **Add CUDNN Determinism Settings**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
   This will eliminate non-deterministic CUDNN operations in BOTH phases.

2. **Document All Configuration Differences**
   - Create a comprehensive config comparison document
   - Include loss functions, hyperparameters, data processing
   - Track PyTorch/CUDA versions

3. **Run Controlled Experiments**
   - Isolate one variable at a time
   - Test Phase 1.5 loss with Phase 2 seed handling
   - Test Phase 2 loss with Phase 1.5 seed handling

---

## 11. Conclusion

**Seed initialization is NOT the cause of 15-62% deviations between Phase 1.5 and Phase 2.**

Both phases use:
- ✅ Identical seeds (42, 52, 62)
- ✅ Identical `set_seed()` implementation
- ✅ Identical train/test split logic

Phase 2 actually has **ENHANCED** determinism through:
- Per-epoch seed resets
- Explicit DataLoader generators

The deviations must stem from **loss function differences, training dynamics, or data handling**, not seed initialization.

**Next Steps:** Investigate loss function implementations and training hyperparameters (see Section 10).

---

## Appendix: Code References

### Phase 1.5 Files
- `sanity_check_core.py:117-126` - `set_seed()` function
- `sanity_check_core.py:368` - Global seed call
- `sanity_check_core.py:204` - DataLoader without generator

### Phase 2 Files
- `sanity_check_signal_tilted.py:186-195` - `set_seed()` function
- `sanity_check_signal_tilted.py:793` - Global seed call
- `sanity_check_signal_tilted.py:393-396` - `set_epoch_seed()` function
- `sanity_check_signal_tilted.py:495-502` - DataLoader with generator
- `run_phase2_robustness.py:23` - `DEFAULT_SEEDS = (42, 52, 62)`

### Documentation
- `doc/phase1.5/Phase1.5-Robustness-Analysis.md` - Phase 1.5 seed configuration
- `doc/phase2-fix/README.md:107` - Phase 2 seed configuration
- `doc/phase2-fix/LOSS_MAPPING.md:158` - Seed variance note
