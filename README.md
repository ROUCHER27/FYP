# FYP: Neural Network-Based Stock Return Prediction

A machine learning research project comparing different loss functions (MSE vs MedSE) for cross-sectional stock return prediction using multi-layer perceptrons.

## 📋 Project Overview

This project investigates whether robust loss functions (Median Squared Error) can outperform traditional MSE in predicting stock returns and constructing long-short portfolios. The research follows a rigorous experimental design with static train/test splits to validate model performance before scaling to rolling window backtests.

## 🏗️ Project Structure

```
FYP/
├── Model_Train/              # Core training modules
│   ├── models.py            # MLP architecture
│   ├── losses.py            # Loss functions (MSE, MedSE, MADL, GMADL)
│   ├── features.py          # Feature engineering (X1, X2, X3)
│   ├── data_preprocess.py   # Data loading and preprocessing
│   ├── train_grid_search.py # Hyperparameter search
│   └── train_rolling.py     # Rolling window training (paused)
├── sanity_check_signal_tilted.py  # Core sanity check logic
├── run_sanity_check_mse.py        # MSE experiment runner
├── run_sanity_check_medse.py      # MedSE experiment runner
├── plot_*.py                      # Visualization scripts
├── sanity_outputs/                # Experiment results
├── *.csv                          # Historical stock data
└── G:MADL/                        # GMADL research notebooks
```

## 🔬 Experimental Design

### Training Setup
- **Training Window**: 1990-01 to 1994-12 (5 years, static)
- **Testing Window**: 1995-01 to 1995-06 (6 months)
- **Feature Set**: X1 (cumulative returns + turnover, 15 dimensions)
- **Model Architecture**: MLP [64, 32, 16] + ReLU + Dropout 0.2
- **Training**: Single training run, no retraining during test period

### Portfolio Strategy
1. **Stock Selection**: Top 10% long, Bottom 10% short
2. **Signal Weighting**:
   - Z-score predictions within each bucket
   - Clip to [-3, 3] range
   - Long: use z>0, Short: use |z|<0
3. **Risk Control**: Max 5% weight per stock (capped-simplex projection)

## 📊 Key Results

### MSE Model
- Cumulative Return: -7.2%
- Sharpe Ratio: -1.46
- Average R²: -113.81 (poor predictive power)

### MedSE Model
- Cumulative Return: +16.6%
- Sharpe Ratio: 3.23 (excellent risk-adjusted return)
- Average R²: -1,714,490 (extremely negative)

**Note**: The negative R² values indicate poor point prediction accuracy, but the MedSE model achieves strong portfolio performance through ranking information. This paradox requires further investigation to rule out data leakage or overfitting.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch
- pandas, numpy, matplotlib

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch pandas numpy matplotlib seaborn
```

### Run Experiments
```bash
# MSE experiment
python run_sanity_check_mse.py

# MedSE experiment
python run_sanity_check_medse.py

# View results
ls sanity_outputs/
```

### Hyperparameter Search (Optional)
```bash
python run_step3_grid_search.py
```

## 📈 Feature Sets

### X1: Momentum & Turnover
- Cumulative returns: 1, 3, 6, 9, 12 months
- Cumulative turnover: 1, 3, 6, 9, 12 months

### X2: Normalized Momentum (excluding recent month)
- Cross-sectional z-scored cumulative returns: 3, 6, 9, 12 months

### X3: Monthly Return Series
- 12 lagged monthly returns (cross-sectionally normalized)

## 🔧 Configuration

Best hyperparameters (from grid search):
```python
{
  'input_dim': 15,
  'hidden_dims': [64, 32, 16],
  'activation': 'relu',
  'dropout': 0.2
}
```

## 📝 Research Notes

- `Train_logic1.md`: Professor feedback and experimental design rationale
- `AGENTS.md`: Repository guidelines and coding standards
- `G:MADL/`: GMADL loss function research and visualizations

## ⚠️ Known Issues

1. **Extreme negative R²**: Requires investigation for data leakage or feature construction bugs
2. **Limited test period**: Only 6 months; needs extension to 1-2 years
3. **Single feature set**: X2 and X3 not yet tested

## 🎯 Next Steps

1. Investigate R² anomaly and validate data pipeline
2. Extend test period to 12-24 months
3. Compare X1, X2, X3 feature sets
4. Test MADL and GMADL loss functions
5. Resume rolling window experiments (Step 4)

## 📚 References

- Loss function design: `loss_function_design_empirical___to_do.pdf`
- GMADL research: `G:MADL/GMADL_improve.md`

## 📄 License

Academic research project - contact author for usage permissions.

## 👤 Author

ROUCHER27 - Final Year Project 2025
