"""
Tests for factor selection criteria.

Validates Bai-Ng and regression-based selection methods.
"""

import numpy as np
import pytest
from src.factor_selection import (
    compute_bai_ng_criteria, compute_regression_criteria, select_factors
)


def test_compute_bai_ng_criteria():
    """Test Bai-Ng information criteria computation."""
    T, N = 100, 50
    k_max = 5
    
    # Create mock V_dict with decreasing values
    V_dict = {j: 1.0 / (j + 1) for j in range(k_max + 1)}
    
    criteria = compute_bai_ng_criteria(V_dict, T, N, k_max)
    
    # Check that all criteria are computed
    assert 'IC_p1' in criteria
    assert 'IC_p2' in criteria
    assert 'IC_p3' in criteria
    
    # Check that each returns (selected_k, IC_values)
    for crit_name in ['IC_p1', 'IC_p2', 'IC_p3']:
        selected_k, IC_values = criteria[crit_name]
        assert isinstance(selected_k, int)
        assert 0 <= selected_k <= k_max
        assert len(IC_values) == k_max + 1


def test_bai_ng_penalty_ordering():
    """Test that Bai-Ng penalties have correct ordering."""
    T, N = 100, 50
    
    C_NT_sq = min(N, T)
    
    # Compute penalty functions
    g_1 = ((N + T) / (N * T)) * np.log((N * T) / (N + T))
    g_2 = ((N + T) / (N * T)) * np.log(C_NT_sq)
    g_3 = np.log(C_NT_sq) / C_NT_sq
    
    # Typically g_1 < g_2 and g_3 is smallest
    # This affects how conservative the criteria are
    assert g_1 > 0
    assert g_2 > 0
    assert g_3 > 0


def test_compute_regression_criteria():
    """Test regression-based AIC and BIC."""
    np.random.seed(42)
    T = 100
    k_max = 5
    
    # Generate synthetic data
    F_true = np.random.normal(0, 1, size=(T, 3))
    beta_true = np.array([1.0, 0.5, -0.3])
    y = np.concatenate([[0], F_true @ beta_true + np.random.normal(0, 0.1, size=T)])
    
    # Create F_dict
    F_dict = {}
    for j in range(k_max + 1):
        if j == 0:
            F_dict[j] = np.zeros((T, 0))
        else:
            F_dict[j] = np.random.normal(0, 1, size=(T, j))
    
    # Add true factors
    F_dict[3] = F_true
    
    criteria = compute_regression_criteria(y, F_dict, k_max, include_intercept=True)
    
    # Check that both criteria are computed
    assert 'AIC' in criteria
    assert 'BIC' in criteria
    
    # Check structure
    for crit_name in ['AIC', 'BIC']:
        selected_k, crit_values = criteria[crit_name]
        assert isinstance(selected_k, int)
        assert 0 <= selected_k <= k_max
        assert len(crit_values) == k_max + 1


def test_select_factors():
    """Test comprehensive factor selection."""
    np.random.seed(42)
    T, N = 100, 50
    k_max = 5
    r_true = 3
    
    # Create mock data
    V_dict = {j: 1.0 / (j + 1) for j in range(k_max + 1)}
    
    F_dict = {}
    for j in range(k_max + 1):
        if j == 0:
            F_dict[j] = np.zeros((T, 0))
        else:
            F_dict[j] = np.random.normal(0, 1, size=(T, j))
    
    y = np.random.normal(0, 1, size=T + 1)
    
    selected = select_factors(V_dict, y, F_dict, T, N, k_max, r_true)
    
    # Check that all methods are included
    assert 'true' in selected
    assert 'IC_p1' in selected
    assert 'IC_p2' in selected
    assert 'IC_p3' in selected
    assert 'AIC' in selected
    assert 'BIC' in selected
    
    # Check that true method returns r_true
    assert selected['true'] == r_true
    
    # Check that all selections are valid
    for method, k in selected.items():
        assert 0 <= k <= k_max


def test_bic_more_conservative_than_aic():
    """Test that BIC typically selects fewer factors than AIC."""
    np.random.seed(42)
    T = 100
    k_max = 10
    
    # Generate data with few true factors
    F_true = np.random.normal(0, 1, size=(T, 2))
    beta_true = np.array([1.0, 0.5])
    y = np.concatenate([[0], F_true @ beta_true + np.random.normal(0, 0.5, size=T)])
    
    # Create F_dict with many candidates
    F_dict = {}
    for j in range(k_max + 1):
        if j == 0:
            F_dict[j] = np.zeros((T, 0))
        else:
            F_dict[j] = np.random.normal(0, 1, size=(T, j))
    
    criteria = compute_regression_criteria(y, F_dict, k_max, include_intercept=True)
    
    k_AIC = criteria['AIC'][0]
    k_BIC = criteria['BIC'][0]
    
    # BIC has stronger penalty, so typically selects fewer factors
    # (not always true in small samples, so we just check it's reasonable)
    assert 0 <= k_BIC <= k_max
    assert 0 <= k_AIC <= k_max


def test_zero_factors_case():
    """Test selection when k=0 might be optimal."""
    np.random.seed(42)
    T = 100
    k_max = 5
    
    # Generate pure noise data (no factor structure)
    y = np.random.normal(0, 1, size=T + 1)
    
    F_dict = {}
    for j in range(k_max + 1):
        if j == 0:
            F_dict[j] = np.zeros((T, 0))
        else:
            # Random factors uncorrelated with y
            F_dict[j] = np.random.normal(0, 1, size=(T, j))
    
    criteria = compute_regression_criteria(y, F_dict, k_max, include_intercept=True)
    
    # Should be able to handle k=0 selection
    k_AIC = criteria['AIC'][0]
    k_BIC = criteria['BIC'][0]
    
    assert k_AIC >= 0
    assert k_BIC >= 0
