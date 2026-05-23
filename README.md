# FYP: Loss-Function Design for Cross-Sectional Stock Return Prediction

This branch (`phase3-4`) contains the **Phase 3a / Phase 3b / Phase 4** multi-seed experiment code. Phase 1 / Phase 2 single-seed code lives on the `main` branch under `2253235_yirongyu_2026_Supplementary/code/`.

---

## Directory Structure

```
ROUCHER27/FYP repository
│
├─── main branch ──────────────────────────────────────────────────────────────
│
│   2253235_yirongyu_2026_Supplementary/
│   ├── README.md                          # Supplementary entry point
│   ├── CODE_INDEX_BY_PHASE.md             # Per-phase index: code, runners, CLI, evidence paths
│   ├── code/                              # Phase 1 / Phase 2 source code (single-seed, relu + dropout 0.2)
│   │   ├── Model_Train/                   # Core modules (losses.py has 16 entries)
│   │   ├── sanity_check_signal_tilted.py  # Shared pipeline
│   │   ├── run_sanity_check_mse.py        # Phase 1 · MSE baseline
│   │   ├── run_sanity_check_medse.py      # Phase 1 · MedSE baseline
│   │   ├── run_sanity_check_madl.py       # Phase 1 · MADL baseline
│   │   ├── run_sanity_check_gmadl.py      # Phase 1 · GMADL baseline
│   │   ├── run_sanity_check_imadl.py      # Phase 1 · IMADL baseline
│   │   ├── run_sanity_check_hybrid_add_a[1-5].py   # Phase 2 · Additive A1–A5
│   │   ├── run_sanity_check_hybrid_mul_m[1-4].py   # Phase 2 · Multiplicative M1–M4
│   │   └── scripts/                       # Shell batch drivers
│   ├── colab_runs/                        # Colab notebooks for Phase 1–4 reproduction
│   ├── data/                              # Raw CSV datasets (1989-12 – 2024-12)
│   └── latex/                             # LaTeX source for the final report
│
├─── phase3-4 branch (this branch) ────────────────────────────────────────────
│
│   (repository root)                      # Multi-seed runners (tanh + dropout 0.0)
│   ├── Model_Train/                                             # Same layout; losses.py extended to 31 entries
│   ├── sanity_check_signal_tilted.py                            # Shared pipeline (+ Phase 4 normalised-loss path)
│   ├── run_sanity_check_m2_robust_gamma03.py                    # Phase 3a · γ = 0.3
│   ├── run_sanity_check_m2_robust_gamma05.py                    # Phase 3a · γ = 0.5
│   ├── run_sanity_check_m2_robust_gamma07.py                    # Phase 3a · γ = 0.7 (recommended)
│   ├── run_sanity_check_m2_robust_gamma10.py                    # Phase 3a · γ = 1.0
│   ├── run_sanity_check_m2_robust_gamma15.py                    # Phase 3a · γ = 1.5
│   ├── run_sanity_check_imadl_m2_alpha{02..08}.py               # Phase 3b · α sweep (7 values)
│   ├── run_sanity_check_imadl_gmadl_beta{03,05,07}.py           # Phase 3b · β sweep (3 values)
│   ├── run_sanity_check_adaptive_lambda{10,50,100}.py           # Phase 3b · adaptive λ sweep (3 values)
│   ├── run_sanity_check_m2_robust_gamma{001,01}.py              # Phase 3b · fine γ (γ → 0 limit)
│   ├── run_sanity_check_m2_robust_gamma07_normalized.py         # Phase 4 · normalisation probe
│   ├── run_sanity_check_m2_robust_gamma10_normalized.py         # Phase 4 · normalisation probe
│   ├── run_sanity_check_imadl_m2_alpha06_normalized.py          # Phase 4 · normalisation probe
│   ├── run_phase2_gamma_refinement.py                           # Phase 3a batch orchestrator (5 × 3 seeds)
│   ├── run_phase2_robustness.py                                 # Phase 3b + Phase 4 batch orchestrator
│   ├── run_loss_scale_diagnostics.py                            # Phase 4 · per-batch component scale diagnostics
│   ├── analyze_loss_scales.py                                   # Phase 4 · diagnostics aggregator
│   ├── notebooks/
│   │   └── phase2_loss_component_analysis.ipynb                 # Phase 4 · interactive analysis & figures
│   ├── doc/phase2-fix/                                          # Evidence CSV/JSON (grouped summaries, per-seed raw)
│   └── *.csv                                                    # Same raw data files as on main
│
└───────────────────────────────────────────────────────────────────────────────
```

> **Model configuration**: This branch uses `tanh + dropout 0.0` (vs. `relu + dropout 0.2` on `main`). Within each phase all loss rows share the same config — only the loss function varies. See `CODE_INDEX_BY_PHASE.md` on `main` for detailed per-loss CLI templates and evidence paths.

---

## How to Run (Phase 3 / Phase 4)

### Prerequisites

- Python 3.10+
- PyTorch, pandas, numpy, matplotlib

### Phase 3a · γ refinement (5 values × 3 seeds)

```bash
# Single γ value
python run_sanity_check_m2_robust_gamma07.py \
  --data-dir . --pattern '*.csv' \
  --seeds 42,52,62 --caps 0.05 \
  --train-start 1990-01 --train-end 1994-12 \
  --test-start 1995-01 --test-months 24 \
  --max-epochs 20 --batch-size 1024

# Full 5 × 3 grid
python run_phase2_gamma_refinement.py --data-dir . --output-dir results/gamma
```

### Phase 3b · α / β / λ sweeps (16 losses × 3 seeds)

```bash
python run_phase2_robustness.py --data-dir . --output-dir results/phase3b
```

### Phase 4 · normalisation probe (3 rows)

```bash
python run_sanity_check_m2_robust_gamma07_normalized.py [same CLI flags as Phase 3a]
python run_sanity_check_m2_robust_gamma10_normalized.py [same CLI flags]
python run_sanity_check_imadl_m2_alpha06_normalized.py  [same CLI flags]
```

### Loss-scale diagnostics

```bash
python run_loss_scale_diagnostics.py --data-dir . --output-root results/diagnostics
```

---

## Evidence Artefacts

Results produced by the above runners are stored under `doc/phase2-fix/`:

| Path | Content |
|---|---|
| `doc/phase2-fix/reports/phase2_grouped_summary.csv` | Phase 3a + 3b grouped summary (3-seed averages) |
| `doc/phase2-fix/reports/phase2_raw_runs.csv` | Per-seed raw metrics |
| `doc/phase2-fix/phase2.2-fix/phase1_summary.json` | Phase 4 normalisation probe (normalised results) |
| `doc/phase2-fix/phase2.2-fix/phase2_summary.json` | Phase 4 normalisation probe (original vs normalised) |
| `doc/phase2-fix/phase2.2-fix/LOSS_COMPONENT_ANALYSIS_RESULTS.md` | Phase 4 scale-ratio analysis |

These artefacts are also mirrored on the `main` branch under `doc/phase2-fix/phase2_2/` and `doc/phase2-fix/phase2.2-fix1/` for read-only reference.

---

## Relationship to `main`

- **`main`**: final delivery branch — paper LaTeX, deck, supplementary package, Phase 1/2 code, Phase 3/4 evidence mirror
- **`phase3-4`** (this branch): Phase 3/4 code origin — multi-seed runners, orchestrators, normalisation probe, diagnostics

For the full Phase ↔ code ↔ paper mapping, see `main:2253235_yirongyu_2026_Supplementary/CODE_INDEX_BY_PHASE.md`.
