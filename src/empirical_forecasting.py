"""
Empirical Forecasting Exercise for Experiment 2

Implements expanding-window real-time forecasting of U.S. industrial production
using principal-components diffusion indexes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

from .data_loader import load_empirical_data, prepare_forecasting_data
from .pca_estimation import extract_factors_for_candidates, standardize_panel
from .factor_selection import compute_bai_ng_criteria


def ar_forecast(y: pd.Series, lags: int = 12) -> float:
    """
    Produce AR forecast using OLS.
    
    Args:
        y: Target series
        lags: Number of AR lags
        
    Returns:
        One-step-ahead forecast
    """
    T = len(y)
    
    # Construct lagged matrix
    y_values = y.values
    X_ar = np.zeros((T - lags, lags))
    y_ar = y_values[lags:]
    
    for lag in range(1, lags + 1):
        X_ar[:, lag-1] = y_values[lags-lag:-lag]
    
    # Add intercept
    X_ar = np.column_stack([np.ones(len(X_ar)), X_ar])
    
    # OLS
    beta = np.linalg.lstsq(X_ar, y_ar, rcond=None)[0]
    
    # Forecast
    X_forecast = np.concatenate([[1.0], y_values[-lags:][::-1]])
    forecast = X_forecast @ beta
    
    return forecast


def pc_forecast(X: np.ndarray, y: np.ndarray, k: int, 
               include_intercept: bool = True) -> float:
    """
    Produce principal components forecast.
    
    Args:
        X: Predictor panel (T, N)
        y: Target variable (T,)
        k: Number of factors
        include_intercept: Whether to include intercept
        
    Returns:
        One-step-ahead forecast
    """
    T, N = X.shape
    
    if k == 0:
        # Mean forecast
        return np.mean(y)
    
    # Extract factors
    pca_results = extract_factors_for_candidates(X, k, standardize=False)
    F = pca_results['F'][k]
    
    # OLS regression
    if include_intercept:
        X_reg = np.column_stack([np.ones(T), F])
    else:
        X_reg = F
    
    beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
    
    # Forecast using last factor values
    if include_intercept:
        X_forecast = np.concatenate([[1.0], F[-1, :]])
    else:
        X_forecast = F[-1, :]
    
    forecast = X_forecast @ beta
    
    return forecast


def expanding_window_forecast(predictors_df: pd.DataFrame, y_series: pd.Series,
                              train_start: str, eval_start: str, eval_end: str,
                              k_max: int = 10, ar_lags: int = 12,
                              verbose: bool = True) -> Dict:
    """
    Run expanding-window real-time forecasting exercise.
    
    At each forecast origin T:
    1. Use data from train_start through T
    2. Standardize predictors using only data through T
    3. Extract principal components
    4. Select number of factors using IC_p3
    5. Produce forecasts using various methods
    6. Store forecast and realized value
    
    Args:
        predictors_df: Predictor panel DataFrame
        y_series: Target variable Series
        train_start: Start of training period (e.g., '1959-01')
        eval_start: Start of evaluation period (e.g., '1970-01')
        eval_end: End of evaluation period (e.g., '1997-12')
        k_max: Maximum number of factors
        ar_lags: Number of AR lags for benchmark
        verbose: Whether to show progress
        
    Returns:
        Dictionary with forecasts and realized values for each method
    """
    # Get evaluation dates
    eval_dates = pd.date_range(start=eval_start, end=eval_end, freq='MS')
    eval_dates = eval_dates.intersection(y_series.index)
    
    # Initialize storage
    forecasts = {
        'AR': [],
        'PC_IC_p3': [],
        'PC_k2': [],
        'PC_k3': [],
        'PC_k4': [],
        'realized': [],
        'dates': []
    }
    
    selected_k = []
    
    iterator = tqdm(eval_dates, desc="Forecasting") if verbose else eval_dates
    
    for date in iterator:
        # Get training data up to current date
        train_mask = (predictors_df.index >= train_start) & (predictors_df.index <= date)
        X_train = predictors_df.loc[train_mask].values
        y_train = y_series.loc[train_mask].values
        
        # Standardize predictors using only training data
        X_train_std, mean, std = standardize_panel(X_train)
        
        T_train, N = X_train_std.shape
        
        # Skip if insufficient data
        if T_train < ar_lags + 10:
            continue
        
        # AR forecast
        y_train_series = pd.Series(y_train)
        try:
            ar_fc = ar_forecast(y_train_series, lags=ar_lags)
        except:
            ar_fc = np.mean(y_train)
        
        # Principal components forecasts
        try:
            # Extract factors for all candidates
            pca_results = extract_factors_for_candidates(X_train_std, k_max, standardize=False)
            
            # Select k using IC_p3
            bai_ng = compute_bai_ng_criteria(pca_results['V'], T_train, N, k_max)
            k_selected = bai_ng['IC_p3'][0]
            selected_k.append(k_selected)
            
            # Forecasts with selected k
            pc_ic_fc = pc_forecast(X_train_std, y_train, k_selected)
            
            # Forecasts with fixed k
            pc_k2_fc = pc_forecast(X_train_std, y_train, 2)
            pc_k3_fc = pc_forecast(X_train_std, y_train, 3)
            pc_k4_fc = pc_forecast(X_train_std, y_train, 4)
            
        except Exception as e:
            if verbose:
                print(f"\nWarning: PC forecast failed at {date}: {e}")
            pc_ic_fc = np.mean(y_train)
            pc_k2_fc = np.mean(y_train)
            pc_k3_fc = np.mean(y_train)
            pc_k4_fc = np.mean(y_train)
            selected_k.append(0)
        
        # Store forecasts
        forecasts['AR'].append(ar_fc)
        forecasts['PC_IC_p3'].append(pc_ic_fc)
        forecasts['PC_k2'].append(pc_k2_fc)
        forecasts['PC_k3'].append(pc_k3_fc)
        forecasts['PC_k4'].append(pc_k4_fc)
        forecasts['realized'].append(y_train[-1])  # Last available value as proxy
        forecasts['dates'].append(date)
    
    # Convert to arrays
    for key in forecasts:
        if key != 'dates':
            forecasts[key] = np.array(forecasts[key])
    
    forecasts['selected_k'] = np.array(selected_k)
    
    return forecasts


def evaluate_forecasts(forecasts: Dict) -> Dict:
    """
    Evaluate forecast performance using MSE and relative MSE.
    
    Args:
        forecasts: Dictionary from expanding_window_forecast
        
    Returns:
        Dictionary with evaluation metrics
    """
    realized = forecasts['realized']
    
    methods = ['AR', 'PC_IC_p3', 'PC_k2', 'PC_k3', 'PC_k4']
    
    results = {}
    
    # Compute MSE for each method
    for method in methods:
        fc = forecasts[method]
        errors = fc - realized
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        results[method] = {
            'MSE': mse,
            'RMSE': rmse
        }
    
    # Compute relative MSE (relative to AR)
    ar_mse = results['AR']['MSE']
    for method in methods:
        results[method]['Relative_MSE'] = results[method]['MSE'] / ar_mse
    
    return results


def run_empirical_experiment(data_dir: str = 'data',
                            train_start: str = '1959-01',
                            eval_start: str = '1970-01',
                            eval_end: str = '1997-12',
                            forecast_horizon: int = 12,
                            k_max: int = 10,
                            ar_lags: int = 12,
                            verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Run full empirical forecasting experiment (Experiment 2).
    
    Args:
        data_dir: Directory for data storage
        train_start: Start of training period
        eval_start: Start of evaluation period
        eval_end: End of evaluation period
        forecast_horizon: Forecast horizon in months
        k_max: Maximum number of factors
        ar_lags: Number of AR lags
        verbose: Whether to show progress
        
    Returns:
        Tuple of (forecasts, evaluation_results)
    """
    print("Loading empirical data...")
    predictors_df, ip_series = load_empirical_data(data_dir)
    
    if predictors_df is None or ip_series is None:
        print("Failed to load data.")
        return None, None
    
    print(f"Data loaded: {len(ip_series)} observations, {predictors_df.shape[1]} predictors")
    
    print(f"\nPreparing forecasting data with horizon h={forecast_horizon}...")
    X_df, y_series = prepare_forecasting_data(predictors_df, ip_series, forecast_horizon)
    
    print(f"Forecasting sample: {len(y_series)} observations")
    print(f"Evaluation period: {eval_start} to {eval_end}")
    
    print("\nRunning expanding-window forecasts...")
    forecasts = expanding_window_forecast(
        X_df, y_series,
        train_start=train_start,
        eval_start=eval_start,
        eval_end=eval_end,
        k_max=k_max,
        ar_lags=ar_lags,
        verbose=verbose
    )
    
    print("\nEvaluating forecasts...")
    evaluation = evaluate_forecasts(forecasts)
    
    return forecasts, evaluation


def format_table2_results(evaluation: Dict) -> str:
    """
    Format empirical forecasting results in a table similar to Table 2.
    
    Args:
        evaluation: Dictionary from evaluate_forecasts
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 2: Out-of-Sample Forecast Evaluation")
    lines.append("Industrial Production Growth (12-month horizon)")
    lines.append("=" * 80)
    lines.append("")
    
    header = f"{'Method':<25} {'RMSE':<15} {'Relative MSE':<15}"
    lines.append(header)
    lines.append("-" * 80)
    
    methods = ['AR', 'PC_IC_p3', 'PC_k2', 'PC_k3', 'PC_k4']
    method_names = {
        'AR': 'AR(12)',
        'PC_IC_p3': 'PC (IC_p3 selected)',
        'PC_k2': 'PC (k=2)',
        'PC_k3': 'PC (k=3)',
        'PC_k4': 'PC (k=4)'
    }
    
    for method in methods:
        if method in evaluation:
            name = method_names[method]
            rmse = evaluation[method]['RMSE']
            rel_mse = evaluation[method]['Relative_MSE']
            
            row = f"{name:<25} {rmse:<15.4f} {rel_mse:<15.4f}"
            lines.append(row)
    
    lines.append("=" * 80)
    lines.append("Note: Relative MSE is computed relative to AR(12) benchmark.")
    lines.append("Lower values indicate better forecast performance.")
    lines.append("=" * 80)
    
    return "\n".join(lines)
