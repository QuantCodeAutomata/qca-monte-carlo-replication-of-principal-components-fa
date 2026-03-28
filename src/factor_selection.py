"""
Factor Number Selection Criteria

Implements multiple criteria for selecting the number of factors:
- Bai-Ng information criteria: IC_p1, IC_p2, IC_p3
- Regression-based AIC and BIC
"""

import numpy as np
from typing import Dict, Tuple


def compute_bai_ng_criteria(V_dict: Dict[int, float], T: int, N: int, k_max: int) -> Dict[str, Tuple[int, Dict[int, float]]]:
    """
    Compute Bai-Ng information criteria for factor number selection.
    
    IC_p(j) = ln(V_j) + j * g(T, N)
    
    where:
    - g_1 = ((N + T) / (NT)) * ln(NT / (N + T))
    - g_2 = ((N + T) / (NT)) * ln(C_NT²)
    - g_3 = ln(C_NT²) / C_NT²
    - C_NT² = min(N, T)
    
    Args:
        V_dict: Dictionary mapping j -> reconstruction error V_j
        T: Time series length
        N: Number of cross-sectional units
        k_max: Maximum number of factors considered
        
    Returns:
        Dictionary with keys 'IC_p1', 'IC_p2', 'IC_p3', each containing:
        - Tuple of (selected_k, IC_values_dict)
    """
    C_NT_sq = min(N, T)
    
    # Penalty functions
    g_1 = ((N + T) / (N * T)) * np.log((N * T) / (N + T))
    g_2 = ((N + T) / (N * T)) * np.log(C_NT_sq)
    g_3 = np.log(C_NT_sq) / C_NT_sq
    
    # Compute IC for each criterion
    IC_p1_values = {}
    IC_p2_values = {}
    IC_p3_values = {}
    
    for j in range(k_max + 1):
        if V_dict[j] > 0:
            IC_p1_values[j] = np.log(V_dict[j]) + j * g_1
            IC_p2_values[j] = np.log(V_dict[j]) + j * g_2
            IC_p3_values[j] = np.log(V_dict[j]) + j * g_3
        else:
            # If V_j = 0 (perfect fit), set IC to -inf (will be selected)
            IC_p1_values[j] = -np.inf
            IC_p2_values[j] = -np.inf
            IC_p3_values[j] = -np.inf
    
    # Select k minimizing each criterion
    k_IC_p1 = min(IC_p1_values.keys(), key=lambda j: IC_p1_values[j])
    k_IC_p2 = min(IC_p2_values.keys(), key=lambda j: IC_p2_values[j])
    k_IC_p3 = min(IC_p3_values.keys(), key=lambda j: IC_p3_values[j])
    
    return {
        'IC_p1': (k_IC_p1, IC_p1_values),
        'IC_p2': (k_IC_p2, IC_p2_values),
        'IC_p3': (k_IC_p3, IC_p3_values)
    }


def compute_regression_criteria(y: np.ndarray, F_dict: Dict[int, np.ndarray], 
                                k_max: int, include_intercept: bool = True) -> Dict[str, Tuple[int, Dict[int, float]]]:
    """
    Compute AIC and BIC for forecasting regression with different factor counts.
    
    For each j, estimate y_{t+1} = beta_0 + beta' F_t + error by OLS and compute:
    - AIC = T * ln(RSS/T) + 2 * (j + 1)
    - BIC = T * ln(RSS/T) + (j + 1) * ln(T)
    
    where RSS is the residual sum of squares and (j + 1) accounts for intercept.
    
    Args:
        y: Target variable of shape (T+1,) where y[t] corresponds to y_{t+1}
        F_dict: Dictionary mapping j -> factor matrix (T, j)
        k_max: Maximum number of factors
        include_intercept: Whether to include intercept in regression
        
    Returns:
        Dictionary with keys 'AIC', 'BIC', each containing:
        - Tuple of (selected_k, criterion_values_dict)
    """
    T = len(y) - 1  # Usable sample size for forecasting
    
    AIC_values = {}
    BIC_values = {}
    
    for j in range(k_max + 1):
        F_j = F_dict[j]
        
        # Construct regression sample: y_{t+1} on F_t for t = 0, ..., T-1
        # But we need to ensure we have T-1 observations for estimation
        # (reserve last observation for out-of-sample forecast)
        T_est = T - 1
        y_est = y[1:T]  # y_{t+1} for t = 0, ..., T-2
        F_est = F_j[:T_est, :]  # F_t for t = 0, ..., T-2
        
        if j == 0:
            # No factors: intercept-only or mean model
            if include_intercept:
                y_pred = np.mean(y_est)
                residuals = y_est - y_pred
                n_params = 1
            else:
                residuals = y_est
                n_params = 0
        else:
            # OLS regression
            if include_intercept:
                X_reg = np.column_stack([np.ones(T_est), F_est])
                n_params = j + 1
            else:
                X_reg = F_est
                n_params = j
            
            # Solve OLS
            beta_hat = np.linalg.lstsq(X_reg, y_est, rcond=None)[0]
            y_pred = X_reg @ beta_hat
            residuals = y_est - y_pred
        
        # Compute RSS
        RSS = np.sum(residuals**2)
        
        # Avoid log(0)
        if RSS == 0:
            RSS = 1e-10
        
        # Compute AIC and BIC
        AIC_values[j] = T_est * np.log(RSS / T_est) + 2 * n_params
        BIC_values[j] = T_est * np.log(RSS / T_est) + n_params * np.log(T_est)
    
    # Select k minimizing each criterion
    k_AIC = min(AIC_values.keys(), key=lambda j: AIC_values[j])
    k_BIC = min(BIC_values.keys(), key=lambda j: BIC_values[j])
    
    return {
        'AIC': (k_AIC, AIC_values),
        'BIC': (k_BIC, BIC_values)
    }


def select_factors(V_dict: Dict[int, float], y: np.ndarray, F_dict: Dict[int, np.ndarray],
                  T: int, N: int, k_max: int, r_true: int,
                  include_intercept: bool = True) -> Dict[str, int]:
    """
    Select number of factors using all available criteria.
    
    Args:
        V_dict: Dictionary mapping j -> reconstruction error
        y: Target variable
        F_dict: Dictionary mapping j -> factor matrix
        T: Time series length
        N: Number of cross-sectional units
        k_max: Maximum number of factors
        r_true: True number of static factors (for benchmark)
        include_intercept: Whether to include intercept in forecasting regression
        
    Returns:
        Dictionary mapping criterion name -> selected number of factors
    """
    # Bai-Ng criteria
    bai_ng = compute_bai_ng_criteria(V_dict, T, N, k_max)
    
    # Regression criteria
    reg_criteria = compute_regression_criteria(y, F_dict, k_max, include_intercept)
    
    # Compile results
    selected = {
        'true': r_true,
        'IC_p1': bai_ng['IC_p1'][0],
        'IC_p2': bai_ng['IC_p2'][0],
        'IC_p3': bai_ng['IC_p3'][0],
        'AIC': reg_criteria['AIC'][0],
        'BIC': reg_criteria['BIC'][0]
    }
    
    return selected
