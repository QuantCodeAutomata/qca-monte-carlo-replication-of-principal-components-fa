"""
Data Loading and Preprocessing for Empirical Forecasting

Handles loading of U.S. macroeconomic data for industrial production forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os


def load_fred_data(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Load data from FRED using pandas_datareader.
    
    Args:
        series_id: FRED series identifier
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Pandas Series with the data
    """
    try:
        import pandas_datareader as pdr
        data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
        return data[series_id]
    except Exception as e:
        print(f"Warning: Could not load {series_id} from FRED: {e}")
        return None


def create_synthetic_macro_panel(T: int = 480, N: int = 149, 
                                 seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic macroeconomic panel data for testing.
    
    This is used when real data is not available. Generates data with
    realistic properties (trends, cycles, cross-correlations).
    
    Args:
        T: Number of time periods (months)
        N: Number of predictor variables
        seed: Random seed
        
    Returns:
        Tuple of (predictors_df, ip_series) where:
        - predictors_df: DataFrame with T rows and N columns
        - ip_series: Series with T rows containing industrial production index
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='1959-01-01', periods=T, freq='MS')
    
    # Generate common factors
    n_factors = 5
    factors = np.zeros((T, n_factors))
    for f in range(n_factors):
        # AR(1) process with trend
        alpha = 0.95
        trend = 0.001 * np.arange(T)
        factors[0, f] = np.random.normal(0, 1)
        for t in range(1, T):
            factors[t, f] = alpha * factors[t-1, f] + np.random.normal(0, 0.5) + trend[t]
    
    # Generate predictors as linear combinations of factors plus noise
    loadings = np.random.normal(0, 1, size=(N, n_factors))
    predictors = factors @ loadings.T + np.random.normal(0, 0.5, size=(T, N))
    
    # Add trends and transformations to some series
    for i in range(N):
        if i % 3 == 0:
            # Add trend
            predictors[:, i] += 0.01 * np.arange(T)
        elif i % 3 == 1:
            # Add cycle
            predictors[:, i] += 2 * np.sin(2 * np.pi * np.arange(T) / 48)
    
    # Create DataFrame
    col_names = [f'X{i+1}' for i in range(N)]
    predictors_df = pd.DataFrame(predictors, index=dates, columns=col_names)
    
    # Generate industrial production as function of factors
    ip_loadings = np.random.normal(1, 0.5, size=n_factors)
    ip_values = 100 * np.exp(0.002 * np.arange(T) + 0.1 * (factors @ ip_loadings) / n_factors)
    ip_series = pd.Series(ip_values, index=dates, name='INDPRO')
    
    return predictors_df, ip_series


def load_empirical_data(data_dir: str = 'data') -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Load empirical macroeconomic data for forecasting exercise.
    
    Attempts to load the Stock-Watson dataset or downloads from FRED.
    Falls back to synthetic data if real data is unavailable.
    
    Args:
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (predictors_df, ip_series) or (None, None) if loading fails
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Try to load industrial production from FRED
    print("Attempting to load industrial production data from FRED...")
    ip_series = load_fred_data('INDPRO', '1959-01-01', '1999-12-31')
    
    if ip_series is None:
        print("Could not load real data. Using synthetic data for demonstration.")
        return create_synthetic_macro_panel()
    
    # For predictors, we would need the full Stock-Watson dataset
    # Since this is not readily available, we'll use synthetic predictors
    # that are correlated with IP
    print("Generating synthetic predictor panel (Stock-Watson dataset not available)...")
    
    T = len(ip_series)
    N = 149
    
    # Generate synthetic predictors correlated with IP
    np.random.seed(42)
    n_factors = 5
    
    # Extract "factors" from IP using simple transformations
    ip_growth = ip_series.pct_change().fillna(0).values
    ip_level = (ip_series / ip_series.iloc[0]).values
    
    # Create base factors
    factors = np.zeros((T, n_factors))
    factors[:, 0] = ip_level
    factors[:, 1] = ip_growth
    for f in range(2, n_factors):
        alpha = 0.9
        factors[0, f] = np.random.normal(0, 1)
        for t in range(1, T):
            factors[t, f] = alpha * factors[t-1, f] + np.random.normal(0, 0.3)
    
    # Generate predictors
    loadings = np.random.normal(0, 1, size=(N, n_factors))
    predictors = factors @ loadings.T + np.random.normal(0, 0.5, size=(T, N))
    
    col_names = [f'X{i+1}' for i in range(N)]
    predictors_df = pd.DataFrame(predictors, index=ip_series.index, columns=col_names)
    
    return predictors_df, ip_series


def transform_to_stationary(series: pd.Series, transformation: str = 'log_diff') -> pd.Series:
    """
    Apply stationarity-inducing transformation to a series.
    
    Args:
        series: Input series
        transformation: Type of transformation ('log_diff', 'diff', 'log', 'none')
        
    Returns:
        Transformed series
    """
    if transformation == 'log_diff':
        return np.log(series).diff()
    elif transformation == 'diff':
        return series.diff()
    elif transformation == 'log':
        return np.log(series)
    else:
        return series


def prepare_forecasting_data(predictors_df: pd.DataFrame, ip_series: pd.Series,
                            forecast_horizon: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for forecasting exercise.
    
    Args:
        predictors_df: Predictor panel (T, N)
        ip_series: Industrial production series (T,)
        forecast_horizon: Forecast horizon in months
        
    Returns:
        Tuple of (X_df, y_series) where:
        - X_df: Predictor panel
        - y_series: Target variable y_{t+h} = ln(IP_{t+h} / IP_t)
    """
    # Construct target variable
    ip_shifted = ip_series.shift(-forecast_horizon)
    y_series = np.log(ip_shifted / ip_series)
    y_series = y_series.dropna()
    
    # Align predictors with target
    common_index = predictors_df.index.intersection(y_series.index)
    X_df = predictors_df.loc[common_index]
    y_series = y_series.loc[common_index]
    
    return X_df, y_series
