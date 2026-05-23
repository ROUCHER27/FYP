# Supplementary Materials

**Student:** Yirong Yu (2253235)  
**Academic Year:** 2025/26

---

## Where to Start

`CODE_INDEX_BY_PHASE.md` is the entry point: it maps each Phase (1–4) reported in the deck and Chapter 5 (§5.2–§5.6) to the corresponding runner script, CLI parameters, source branch, and evidence CSV/JSON paths. Read it first, then drill into `code/`, `colab_runs/`, or `latex/` as needed.

---

## Directory Structure

```
ROUCHER27/FYP repository
│
├─── main branch ──────────────────────────────────────────────────────────────
│
│   2253235_yirongyu_2026_Supplementary/
│   ├── README.md                          # This file
│   ├── CODE_INDEX_BY_PHASE.md             # Per-phase index: code, runners, CLI, evidence paths
│   ├── code/                              # Phase 1 / Phase 2 source code (single-seed runners)
│   │   ├── Model_Train/                   # Core training modules
│   │   │   ├── models.py                  # MLP architecture
│   │   │   ├── losses.py                  # Loss functions (16 entries: MSE / MedSE / MADL / GMADL / IMADL / hybrid A/M)
│   │   │   ├── features.py                # Feature engineering (X1)
│   │   │   ├── data_preprocess.py         # Data loading and preprocessing
│   │   │   ├── train_grid_search.py       # Hyperparameter grid search (frozen → best_hyperparameters.txt)
│   │   │   ├── train_rolling.py           # Rolling-window training helper (future-work scaffold)
│   │   │   └── __init__.py
│   │   ├── sanity_check_signal_tilted.py  # Shared pipeline (data → MLP → long-short → metrics + figures)
│   │   ├── run_sanity_check_mse.py        # Phase 1 · MSE baseline
│   │   ├── run_sanity_check_medse.py      # Phase 1 · MedSE baseline
│   │   ├── run_sanity_check_madl.py       # Phase 1 · MADL baseline
│   │   ├── run_sanity_check_gmadl.py      # Phase 1 · GMADL baseline
│   │   ├── run_sanity_check_imadl.py      # Phase 1 · IMADL baseline
│   │   ├── run_sanity_check_hybrid_add_a[1-5].py   # Phase 2 · Additive variants A1–A5
│   │   ├── run_sanity_check_hybrid_mul_m[1-4].py   # Phase 2 · Multiplicative variants M1–M4
│   │   ├── run_step3_grid_search.py       # Phase 0 · Hyperparameter search runner
│   │   └── scripts/
│   │       ├── run_final_report_24m_baselines_colab.sh
│   │       └── run_final_report_all_24m_evidence_colab.sh
│   ├── colab_runs/                        # Colab notebooks + guides for Phase 1–4 reproduction history
│   ├── data/                              # Raw datasets (US equity monthly panel, 1989-12 – 2024-12)
│   │   ├── 89.12-94.csv
│   │   ├── 94-99.csv
│   │   ├── 99-04.csv
│   │   ├── 04-09.csv
│   │   ├── 09-14.csv
│   │   ├── 14.12-19.12.csv
│   │   └── 19.12-24.12.csv
│   └── latex/                             # LaTeX source for the final report
│       ├── main.tex
│       ├── chapter[1-6]_*.tex
│       ├── appendix_A_loss_definitions.tex
│       ├── appendix_B_per_seed_raw.tex
│       ├── references.bib
│       └── figures/
│
├─── phase3-4 branch (git checkout phase3-4) ──────────────────────────────────
│
│   (repository root)
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
│   └── notebooks/
│       └── phase2_loss_component_analysis.ipynb                 # Phase 4 · interactive analysis & figures
│
└───────────────────────────────────────────────────────────────────────────────
```

> **Note on model configuration**: Phase 1 / Phase 2 on `main` uses `relu + dropout 0.2`; Phase 3 / Phase 4 on `phase3-4` uses `tanh + dropout 0.0`. Within each phase all loss rows share the same config — only the loss function varies. See `CODE_INDEX_BY_PHASE.md` §2.3–§2.5 for detailed CLI templates, hyperparameters, and evidence CSV paths. Phase 3/4 evidence CSV/JSON is also mirrored on `main` under `doc/phase2-fix/` for read-only reference.

---

## How to Run

### Prerequisites

- Python 3.10+
- PyTorch, pandas, numpy, matplotlib

### Phase 1 / Phase 2 (single seed, on `main`)

From repository root, with the seven CSV data files alongside:

```bash
cd 2253235_yirongyu_2026_Supplementary/code/

# Phase 1 baseline (one of seven loss families)
python run_sanity_check_mse.py \
  --data-dir <repo-root> --pattern '*.csv' \
  --train-start 1990-01 --train-end 1994-12 \
  --test-start 1995-01 --test-months 24 \
  --max-epochs 20 --batch-size 1024 --seed 42

# Phase 2 hybrid variant (A3 or M1 shown; A1–A5 / M1–M4 follow the same template)
python run_sanity_check_hybrid_mul_m1.py [same CLI flags]
python run_sanity_check_hybrid_add_a3.py [same CLI flags]
```

CLI flags are uniform across all Phase 1 / Phase 2 wrappers; only the runner filename and the implicit loss id change.

### Phase 3 / Phase 4 (multi seed, on `phase3-4`)

These rows require checking out the `phase3-4` branch and using the same CSV data files at the repository root. CLI flags are identical to Phase 1/2 except for `--seeds 42,52,62 --caps 0.05`.

```bash
git checkout phase3-4

# Phase 3a γ refinement (one of five γ values)
python run_sanity_check_m2_robust_gamma07.py \
  --data-dir <repo-root> --pattern '*.csv' \
  --seeds 42,52,62 --caps 0.05 \
  --train-start 1990-01 --train-end 1994-12 \
  --test-start 1995-01 --test-months 24 \
  --max-epochs 20 --batch-size 1024
# or run the full 5 × 3 grid via the orchestrator:
python run_phase2_gamma_refinement.py --data-dir <repo-root> --output-dir <output-root>

# Phase 4 normalisation probe (three rows)
python run_sanity_check_m2_robust_gamma07_normalized.py [same CLI flags]
python run_sanity_check_m2_robust_gamma10_normalized.py [same CLI flags]
python run_sanity_check_imadl_m2_alpha06_normalized.py [same CLI flags]
```

### Compile LaTeX

```bash
cd latex/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Data Description

Each CSV file contains monthly US equity panel data with columns including stock identifiers, volume (`VOL`), returns (`RET`), shares outstanding (`SHROUT`), monthly return (`r`), and turnover (`to`). The training window uses `1990-01` to `1994-12`; the test window uses `1995-01` to `1996-12` (24 months static out-of-sample).
