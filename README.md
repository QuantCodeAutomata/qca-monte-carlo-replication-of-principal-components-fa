# Monte Carlo Replication of Principal-Components Factor Recovery and Forecast Efficiency

This repository replicates the synthetic-data experiments and empirical forecasting exercises from research on approximate dynamic factor models using principal components analysis.

## Overview

The repository implements two main experiments:

1. **Experiment 1 (Monte Carlo Simulation)**: Replicates Table 1 from the paper, validating that principal-components estimates recover the latent static factor space and produce forecasts close to infeasible true-factor forecasts across various simulation designs.

2. **Experiment 2 (Empirical Forecasting)**: Replicates Table 2 from the paper, producing 12-month-ahead forecasts of U.S. industrial production using principal-components diffusion indexes in a simulated real-time expanding-window setting.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_generation.py      # Synthetic data generation for Monte Carlo
│   ├── pca_estimation.py       # Principal components factor extraction
│   ├── factor_selection.py     # Bai-Ng and regression-based criteria
│   ├── evaluation_metrics.py   # Factor recovery and forecast closeness metrics
│   ├── monte_carlo.py          # Monte Carlo simulation runner
│   ├── empirical_forecasting.py # Real-time forecasting framework
│   ├── data_loader.py          # Data loading and preprocessing
│   └── visualization.py        # Plotting and reporting
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_pca_estimation.py
│   ├── test_factor_selection.py
│   ├── test_evaluation_metrics.py
│   ├── test_monte_carlo.py
│   └── test_empirical_forecasting.py
├── results/
│   └── RESULTS.md              # Experiment results and metrics
├── data/                       # Downloaded empirical data
├── requirements.txt
└── run_experiments.py          # Main script to run all experiments

```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:

```bash
python run_experiments.py
```

Run individual experiments:

```bash
# Monte Carlo simulation (Experiment 1)
python -c "from src.monte_carlo import run_monte_carlo_experiment; run_monte_carlo_experiment()"

# Empirical forecasting (Experiment 2)
python -c "from src.empirical_forecasting import run_empirical_experiment; run_empirical_experiment()"
```

Run tests:

```bash
pytest tests/ -v
```

## Methodology

### Experiment 1: Monte Carlo Simulation

Generates synthetic panel data from a dynamic factor model with:
- Optional serial correlation in idiosyncratic errors
- Weak cross-sectional dependence
- Conditional heteroskedasticity (GARCH)
- Irrelevant predictors
- Dynamic factors with lags
- Slowly time-varying loadings

For each design:
1. Generate N×T panel data following the DGP specification
2. Extract principal components for candidate factor counts
3. Select optimal factor count using multiple criteria (Bai-Ng IC_p1/p2/p3, AIC/BIC)
4. Evaluate factor space recovery (R²)
5. Evaluate forecast closeness to infeasible true-factor forecast (S²)
6. Average over 2,000 Monte Carlo repetitions

### Experiment 2: Empirical Forecasting

Uses monthly U.S. macroeconomic panel (149 variables, 1959:1-1998:12):
1. Expanding-window real-time forecasting (1970:1-1997:12)
2. Extract principal components at each forecast origin
3. Forecast 12-month industrial production growth
4. Compare out-of-sample MSE against AR and other benchmarks
5. Report relative MSE values

## Results

All results are saved in `results/RESULTS.md` with:
- Monte Carlo averages for Table 1 replication
- Out-of-sample forecast evaluation for Table 2 replication
- Visualizations and diagnostic plots

## References

This implementation strictly follows the methodology described in the original paper on approximate dynamic factor models and principal components analysis for forecasting.
