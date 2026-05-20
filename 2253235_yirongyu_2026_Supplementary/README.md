# Supplementary Materials

**Student:** Yirong Yu (2253235)  
**Title:** Loss Function Design for Neural Network-Based Cross-Sectional Stock Return Prediction  
**Academic Year:** 2025/26

---

## Directory Structure

```
2253235_yirongyu_2026_Supplementary/
├── code/                          # Source code
│   ├── Model_Train/               # Core training modules
│   │   ├── models.py             # MLP architecture
│   │   ├── losses.py             # All loss functions (MSE, MedSE, MADL, GMADL, IMADL, Hybrids)
│   │   ├── features.py           # Feature engineering (X1, X2, X3)
│   │   ├── data_preprocess.py    # Data loading and preprocessing
│   │   ├── train_grid_search.py  # Hyperparameter grid search
│   │   ├── train_rolling.py      # Rolling window training
│   │   └── __init__.py
│   ├── sanity_check_signal_tilted.py   # Core evaluation pipeline
│   ├── run_sanity_check_mse.py         # Phase 1: MSE baseline
│   ├── run_sanity_check_medse.py       # Phase 1: MedSE baseline
│   ├── run_sanity_check_madl.py        # Phase 1: MADL baseline
│   ├── run_sanity_check_gmadl.py       # Phase 1: GMADL baseline
│   ├── run_sanity_check_imadl.py       # Phase 1: IMADL baseline
│   ├── run_sanity_check_hybrid_add_a[1-5].py  # Phase 1.5: Additive hybrids
│   ├── run_sanity_check_hybrid_mul_m[1-4].py  # Phase 1.5: Multiplicative hybrids
│   ├── run_step3_grid_search.py        # Hyperparameter search runner
│   └── scripts/                        # Shell scripts for batch experiments
│       ├── run_final_report_all_24m_evidence_colab.sh
│       └── run_final_report_24m_baselines_colab.sh
├── data/                          # Raw datasets (US equity monthly panel)
│   ├── 89.12-94.csv              # Dec 1989 – Dec 1994
│   ├── 94-99.csv                 # 1994 – 1999
│   ├── 99-04.csv                 # 1999 – 2004
│   ├── 04-09.csv                 # 2004 – 2009
│   ├── 09-14.csv                 # 2009 – 2014
│   ├── 14.12-19.12.csv           # Dec 2014 – Dec 2019
│   └── 19.12-24.12.csv           # Dec 2019 – Dec 2024
├── latex/                         # LaTeX source for the final report
│   ├── main.tex                  # Main document
│   ├── chapter[1-6]_*.tex        # Chapter source files
│   ├── abstract.tex
│   ├── appendix_A_loss_definitions.tex
│   ├── appendix_B_per_seed_raw.tex
│   ├── references.bib
│   └── figures/                  # All figures used in the report
└── README.md                     # This file
```

## How to Run

### Prerequisites
- Python 3.10+
- PyTorch, pandas, numpy, matplotlib

### Run baseline experiments
```bash
cd code/
python run_sanity_check_mse.py      # MSE loss
python run_sanity_check_medse.py    # MedSE loss
python run_sanity_check_madl.py     # MADL loss
```

### Run hybrid experiments
```bash
python run_sanity_check_hybrid_mul_m1.py   # Multiplicative hybrid M1
python run_sanity_check_hybrid_add_a3.py   # Additive hybrid A3
```

### Compile LaTeX
```bash
cd latex/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Data Description

Each CSV file contains monthly US equity panel data with columns including stock identifiers, volume (VOL), returns (RET), shares outstanding (SHROUT), monthly return (r), and turnover (to). The training window uses 1990-01 to 1994-12; the test window uses 1995-01 to 1996-12.
