"""
Evaluation Metrics for Factor Models

Implements metrics for evaluating:
- Factor space recovery: R²_{F_hat, F}
- Forecast closeness: S²_{y_hat, y_tilde}
"""

import numpy as np
from typing import Tuple


def compute_factor_recovery_r_squared(F_hat: np.ndarray, F_true: np.ndarray) -> float:
    """
    Compute the factor space recovery statistic R²_{F_hat, F}.
    
    R²_{F_hat, F} = tr(F_hat' P_F F_hat) / tr(F_hat' F_hat)
    
    where P_F = F (F'F)^{-1} F' is the projection matrix onto the true factor space.
    
    This statistic measures how well the estimated factor space spans the true
    factor space. It is invariant to sign flips and orthogonal rotations.
    
    Args:
        F_hat: Estimated factors of shape (T, k_hat)
        F_true: True factors of shape (T, r)
        
    Returns:
        R² statistic (scalar in [0, 1])
    """
    T = F_true.shape[0]
    
    if F_hat.shape[1] == 0:
        return 0.0
    
    # Compute projection matrix P_F = F (F'F)^{-1} F'
    FtF = F_true.T @ F_true
    
    # Check for singularity
    if np.linalg.cond(FtF) > 1e10:
        # Use pseudo-inverse for numerical stability
        FtF_inv = np.linalg.pinv(FtF)
    else:
        FtF_inv = np.linalg.inv(FtF)
    
    P_F = F_true @ FtF_inv @ F_true.T
    
    # Compute R² = tr(F_hat' P_F F_hat) / tr(F_hat' F_hat)
    numerator = np.trace(F_hat.T @ P_F @ F_hat)
    denominator = np.trace(F_hat.T @ F_hat)
    
    if denominator == 0:
        return 0.0
    
    R_squared = numerator / denominator
    
    # Clamp to [0, 1] to handle numerical errors
    R_squared = np.clip(R_squared, 0.0, 1.0)
    
    return R_squared


def forecast_with_factors(y: np.ndarray, F: np.ndarray, 
                         include_intercept: bool = True) -> Tuple[float, np.ndarray]:
    """
    Produce one-step-ahead forecast using OLS regression on factors.
    
    Estimates y_{t+1} = beta_0 + beta' F_t + error over t = 0, ..., T-2
    and forecasts y_{T} using F_{T-1}.
    
    Args:
        y: Target variable of shape (T+1,) where y[t] corresponds to y_{t+1}
        F: Factor matrix of shape (T, k)
        include_intercept: Whether to include intercept
        
    Returns:
        Tuple of (forecast, coefficients) where:
        - forecast: One-step-ahead forecast y_hat_{T}
        - coefficients: Estimated regression coefficients
    """
    T = F.shape[0]
    k = F.shape[1]
    
    # Estimation sample: y_{t+1} on F_t for t = 0, ..., T-2
    T_est = T - 1
    y_est = y[1:T]  # y_{t+1} for t = 0, ..., T-2
    F_est = F[:T_est, :]  # F_t for t = 0, ..., T-2
    
    if k == 0:
        # No factors: use mean forecast
        if include_intercept:
            forecast = np.mean(y_est)
            coefficients = np.array([forecast])
        else:
            forecast = 0.0
            coefficients = np.array([])
    else:
        # OLS regression
        if include_intercept:
            X_reg = np.column_stack([np.ones(T_est), F_est])
        else:
            X_reg = F_est
        
        # Solve OLS
        coefficients = np.linalg.lstsq(X_reg, y_est, rcond=None)[0]
        
        # Forecast using F_{T-1}
        F_T_minus_1 = F[T-1, :]
        if include_intercept:
            X_forecast = np.concatenate([[1.0], F_T_minus_1])
        else:
            X_forecast = F_T_minus_1
        
        forecast = X_forecast @ coefficients
    
    return forecast, coefficients


def compute_forecast_closeness_s_squared(y_hat: float, y_tilde: float) -> float:
    """
    Compute the forecast closeness statistic S²_{y_hat, y_tilde}.
    
    S²_{y_hat, y_tilde} = 1 - ((y_hat - y_tilde)² / y_hat²)
    
    This measures how close the feasible forecast (based on estimated factors)
    is to the infeasible forecast (based on true factors).
    
    Args:
        y_hat: Feasible forecast based on estimated factors
        y_tilde: Infeasible forecast based on true factors
        
    Returns:
        S² statistic (can be negative if forecasts are very different)
    """
    if y_hat == 0:
        # Avoid division by zero
        if y_tilde == 0:
            return 1.0
        else:
            return -np.inf
    
    S_squared = 1.0 - ((y_hat - y_tilde)**2 / y_hat**2)
    
    return S_squared


def evaluate_monte_carlo_repetition(X: np.ndarray, y: np.ndarray, F_true: np.ndarray,
                                    F_hat_dict: dict, k_selected: dict,
                                    include_intercept: bool = True) -> dict:
    """
    Evaluate a single Monte Carlo repetition.
    
    For each factor selection method:
    1. Compute factor recovery R²
    2. Compute feasible forecast y_hat
    3. Compute infeasible forecast y_tilde (using true factors)
    4. Compute forecast closeness S²
    
    Args:
        X: Panel data matrix (T, N)
        y: Target variable (T+1,)
        F_true: True static factors (T, r_static)
        F_hat_dict: Dictionary mapping j -> estimated factors (T, j)
        k_selected: Dictionary mapping criterion -> selected k
        include_intercept: Whether to include intercept in forecasting
        
    Returns:
        Dictionary with results for each selection method
    """
    results = {}
    
    # Compute infeasible forecast using true factors
    y_tilde, _ = forecast_with_factors(y, F_true, include_intercept)
    
    # Evaluate each selection method
    for method, k in k_selected.items():
        F_hat = F_hat_dict[k]
        
        # Factor recovery
        R_squared = compute_factor_recovery_r_squared(F_hat, F_true)
        
        # Feasible forecast
        y_hat, _ = forecast_with_factors(y, F_hat, include_intercept)
        
        # Forecast closeness
        S_squared = compute_forecast_closeness_s_squared(y_hat, y_tilde)
        
        results[method] = {
            'k_selected': k,
            'R_squared': R_squared,
            'y_hat': y_hat,
            'y_tilde': y_tilde,
            'S_squared': S_squared
        }
    
    return results


def aggregate_monte_carlo_results(all_results: list) -> dict:
    """
    Aggregate results across Monte Carlo repetitions.
    
    Computes mean and standard error for each metric and selection method.
    
    Args:
        all_results: List of result dictionaries from each repetition
        
    Returns:
        Dictionary with aggregated statistics
    """
    n_reps = len(all_results)
    
    # Extract methods from first repetition
    methods = list(all_results[0].keys())
    
    aggregated = {}
    
    for method in methods:
        # Collect values across repetitions
        k_selected = [rep[method]['k_selected'] for rep in all_results]
        R_squared = [rep[method]['R_squared'] for rep in all_results]
        S_squared = [rep[method]['S_squared'] for rep in all_results]
        
        # Compute statistics
        aggregated[method] = {
            'k_selected_mean': np.mean(k_selected),
            'k_selected_std': np.std(k_selected, ddof=1),
            'R_squared_mean': np.mean(R_squared),
            'R_squared_se': np.std(R_squared, ddof=1) / np.sqrt(n_reps),
            'S_squared_mean': np.mean(S_squared),
            'S_squared_se': np.std(S_squared, ddof=1) / np.sqrt(n_reps),
            'n_reps': n_reps
        }
    
    return aggregated
