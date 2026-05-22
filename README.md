# FYP — Loss Function Design for Cross-Sectional Stock Return Prediction

Final-year research project on whether a **hybrid loss** that combines a directional-accuracy term with a **robust magnitude** term can outperform traditional regression losses (MSE, MedSE) and pure directional losses (MADL, GMADL, IMADL) when the downstream task is a long-short portfolio.

> **Author:** Yirong Yu (2253235) · **Year:** 2025/26  
> **Final report workspace:** [`paper/`](paper/) · **Marker bundle:** [`2253235_yirongyu_2026_Supplementary/`](2253235_yirongyu_2026_Supplementary/)

---

## 1. Research Question

Cross-sectional return prediction is almost always trained with MSE, but the downstream decision (a long-short portfolio) depends on the **cross-sectional rank** of predictions, not their calibrated magnitudes. Under heavy-tailed monthly returns this mismatch wastes training capacity. We ask:

> Conditional on a fixed prediction pipeline, can loss-function design alone produce a more robust portfolio signal?

---

## 2. Experimental Protocol (the single source of truth)

All headline numbers in the report are produced under an identical protocol:

| Setting | Value |
|---|---|
| Training window | 1990-01 → 1994-12 (60 monthly cross-sections, static) |
| **Main test window** | **1995-01 → 1996-12 (24 months)** |
| Feature set | X1 — 15 dims: cumulative return + cumulative turnover at {1, 3, 6, 9, 12} months |
| Model | MLP `[64, 32, 16]` + ReLU + dropout 0.2 |
| Optimiser | Adam, batch size 1024, 20 epochs |
| Portfolio | Top/Bottom 10% signal-tilted long-short, 5% per-name cap, monthly rebalance |
| Metric | Annualised Sharpe (√12 · monthly mean/std), 24-m cumulative return, multi-seed CV |

> Earlier 6-month sanity checks are **preliminary only** and never used as final headlines (see `SCHEMA.md` §2.4).

---

## 3. Experimental Phases

The repository implements four phases under the protocol above. Each phase has its own runners and evidence CSVs; see [`2253235_yirongyu_2026_Supplementary/CODE_INDEX_BY_PHASE.md`](2253235_yirongyu_2026_Supplementary/CODE_INDEX_BY_PHASE.md) for the full phase ↔ chapter ↔ table ↔ runner ↔ CSV mapping.

| Phase | Scope | Seeds | Runner family |
|---|---|---|---|
| **Phase 1** | 7 baseline losses: `mse`, `medse`, `madl`, `gmadl`, `imadl`, `hybrid_mul_m1`, `hybrid_mul_m2` | 42 | `run_sanity_check_<loss>.py` |
| **Phase 2** | 9 hybrid variants: additive A1–A5, multiplicative M1–M4 | 42 | `run_sanity_check_hybrid_{add,mul}_*.py` |
| **Phase 3a** | γ-refinement of M2-robust: `L = L_M2 + γ · Var(ŷ)`, γ ∈ {0.3, 0.5, 0.7, 1.0, 1.5} | {42, 52, 62} | `run_phase2_gamma_refinement.py` |
| **Phase 3b** | Integrated sweeps: IMADL-m2 α ∈ {0.2..0.8}, IMADL-GMADL β ∈ {0.3, 0.5, 0.7}, adaptive λ ∈ {1, 5, 10}, plus fine-γ {0.01, 0.1} | {42, 52, 62} | `run_phase2_robustness.py` |
| **Phase 4** | Loss-component **normalisation probe** on the three leading candidates | {42, 52, 62} | `run_sanity_check_*_normalized.py` + `notebooks/phase2_loss_component_analysis.ipynb` |

The Phase 3/4 multi-seed runners live on the `phase3-4` branch (their original development line); `main` carries the Phase 1/2 runners and mirrored evidence CSVs.

---

## 4. Headline Results

Multi-seed (3-seed) results from Phase 3a/3b. Numbers are mean across seeds.

| Loss | Mean Sharpe | CV | Mean cum. return | Role |
|---|---:|---:|---:|---|
| **`m2_robust_gamma07`** | **0.9156** | **0.1808** | **+27.99%** | Primary recommendation — best Sharpe/stability balance; no negative-Sharpe seed |
| `m2_robust_gamma10` | 1.0043 | 0.5613 | — | High-return alternative; explicit seed-sensitivity caveat |
| `imadl_m2_alpha06` | 0.6895 | 0.2443 | +30.42% | Stable fallback from an independent parameterisation |

**Phase 4 normalisation probe** — not a universal fix:

- `gamma07_normalized` ≈ 0.9112 (essentially flat → γ07 is **not** a scale artefact)
- `gamma10_normalized` ≈ 0.4072 (substantial degradation → scale-sensitive)
- `alpha06_normalized` ≈ −0.0161 (collapses → fallback fails the probe)

**Calibration vs ranking dissociation.** Several losses (notably the absolute/directional family) record very negative point-prediction R² while still producing competitive long-short Sharpes — consistent with the portfolio trading ranks, not calibrated values. The report makes this dissociation explicit rather than hiding it.

---

## 5. Repository Layout

```
FYP/
├── Model_Train/                       # Core training modules
│   ├── models.py                      # MLP architecture
│   ├── losses.py                      # All loss families (MSE, MedSE, MADL, GMADL, IMADL, hybrids)
│   ├── features.py                    # Feature engineering (X1, X2, X3)
│   ├── data_preprocess.py             # Data loading and cross-sectional preprocessing
│   ├── train_grid_search.py           # Hyperparameter grid search
│   └── train_rolling.py               # Rolling-window training (future work)
├── sanity_check_signal_tilted.py      # Shared evaluation pipeline (build_arg_parser, run_sanity_check)
├── run_sanity_check_*.py              # Phase 1 / Phase 2 single-seed runners
├── run_step3_grid_search.py           # Hyperparameter search runner
├── run_step4_rolling.py               # Rolling-window runner (future work)
├── plot_*.py                          # Figure-generation scripts
├── *.csv                              # Monthly US equity panels (Dec 1989 – Dec 2024)
├── doc/
│   ├── thesis/                        # Original thesis drafts
│   ├── final_report_24m_baselines/    # Verified 24m baseline evidence (mse, medse)
│   ├── final_report_all_24m_evidence/ # Verified 24m evidence — 7 baselines + 9 hybrids
│   ├── phase2-fix/                    # Phase 3/4 multi-seed evidence (grouped + per-seed CSVs)
│   ├── phase2.5/                      # Phase 2.5 alignment diagnostics
│   ├── agent_handoff.md               # Cross-branch evidence map
│   └── 2253235_YirongYu_2025.pdf      # Interim report
├── paper/                             # Final-report drafting workspace (Markdown chapters)
│   ├── abstract.md, chapter1..6_*.md, appendix_A/B_*.md, references.md
│   ├── results_source_of_truth.md     # The only file that may be cited for numbers
│   ├── evidence_map.md                # Pointer index for sub-agents
│   ├── figures/                       # Generated paper-quality figures
│   └── latex/                         # LaTeX source (final typeset)
├── 2253235_yirongyu_2026_Supplementary/   # Marker submission bundle
│   ├── code/, data/, latex/, colab_runs/
│   ├── CODE_INDEX_BY_PHASE.md         # Phase ↔ chapter ↔ runner ↔ CSV mapping
│   └── README.md
├── scripts/                           # Batch shell scripts for Colab/local runs
├── tests/                             # Pytest suite (numeric sanity for losses)
├── SCHEMA.md                          # Highest-level evidence and claim-boundary contract
├── AGENTS.md, CLAUDE.md               # Agent guidelines
└── README.md                          # (this file)
```

---

## 6. Loss Function Families

Implementations live in `Model_Train/losses.py`; closed-form definitions are in [`paper/appendix_A_loss_definitions.md`](paper/appendix_A_loss_definitions.md).

| Family | Members | Idea |
|---|---|---|
| **Regression** | `mse`, `medse` | Standard / robust calibration; ignores direction |
| **Directional** | `madl`, `gmadl`, `imadl` | Reward sign-agreement of (ŷ, y); ignores magnitude scale |
| **Additive hybrid** | `hybrid_add_a1..a5` | `λ_dir · Dir(ŷ, y) + λ_hub · Huber(ŷ, y)` |
| **Multiplicative hybrid** | `hybrid_mul_m1..m4` | Huber backbone gated by a directional multiplier `D(ŷ, y)` |
| **M2-robust (γ family)** | `m2_robust_gamma{0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5}` | M2 + variance penalty `γ · Var(ŷ)` to control prediction spread |
| **Integrated sweeps** | `imadl_m2_alpha*`, `imadl_gmadl_beta*`, `adaptive_lambda*` | Independent parameterisations used to triangulate the productive design region |

---

## 7. Quick Start

### Prerequisites
- Python 3.10+
- PyTorch, pandas, numpy, matplotlib, seaborn

### Install
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch pandas numpy matplotlib seaborn
```

### Run a Phase 1 baseline (24-month protocol)
```bash
python run_sanity_check_mse.py \
  --data-dir . --pattern '*.csv' \
  --train-start 1990-01 --train-end 1994-12 \
  --test-start 1995-01 --test-months 24 \
  --max-epochs 20 --batch-size 1024 --seed 42
```
Replace `mse` with any of `medse`, `madl`, `gmadl`, `imadl`, `hybrid_mul_m1`, `hybrid_mul_m2`.

### Run a Phase 2 hybrid variant
```bash
python run_sanity_check_hybrid_add_a3.py     # additive, λ_dir=1.0, λ_hub=0.1
python run_sanity_check_hybrid_mul_m1.py     # multiplicative, λ_dir=2.0
```

### Run Phase 3/4 multi-seed batches (requires `phase3-4` branch)
```bash
python run_phase2_gamma_refinement.py        # 5 γ × 3 seeds = 15 runs
python run_phase2_robustness.py              # 16 losses × 3 seeds = 48 runs
```

### View results
```bash
ls sanity_outputs/                           # per-run CSVs / figures
ls doc/final_report_all_24m_evidence/        # verified Phase 1 + Phase 2 evidence
ls doc/phase2-fix/                           # Phase 3/4 multi-seed evidence
```

---

## 8. Reproducing Figures and Tables

Every figure and table in the final report is regenerated from a CSV recorded in [`paper/results_source_of_truth.md`](paper/results_source_of_truth.md). The phase ↔ table ↔ runner ↔ CSV map is in [`2253235_yirongyu_2026_Supplementary/CODE_INDEX_BY_PHASE.md`](2253235_yirongyu_2026_Supplementary/CODE_INDEX_BY_PHASE.md). The Colab notebooks under [`2253235_yirongyu_2026_Supplementary/colab_runs/`](2253235_yirongyu_2026_Supplementary/colab_runs/) preserve the original cloud-execution paths.

---

## 9. Limitations and Future Work

1. **Single static window.** All headlines come from the 24-month static test window; rolling-window extension is implemented (`run_step4_rolling.py`, `Model_Train/train_rolling.py`) but not part of the final headlines.
2. **Three-seed CV.** Sufficient to separate stable vs unstable candidates at the order-of-magnitude level; not sufficient to pin CV to the second decimal.
3. **Diagnostics-estimated normalisation scales.** A per-component scale logger is the priority instrumentation upgrade.
4. **Single feature set.** X2 (z-scored momentum) and X3 (lagged monthly returns) are implemented in `Model_Train/features.py` but not part of the headline evaluation.

---

## 10. Project Documents

- [`SCHEMA.md`](SCHEMA.md) — evidence-gate and claim-boundary contract (read first).
- [`doc/agent_handoff.md`](doc/agent_handoff.md) — cross-branch evidence map.
- [`paper/results_source_of_truth.md`](paper/results_source_of_truth.md) — the only numeric source the report may cite.
- [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md) — agent / contributor guidelines.

---

## License

Academic research project. Contact the author for usage permissions.
