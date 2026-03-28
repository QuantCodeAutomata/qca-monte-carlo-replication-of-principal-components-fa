"""
Tests for evaluation metrics.

Validates R² factor recovery and S² forecast closeness metrics.
"""

import numpy as np
import pytest
from src.evaluation_metrics import (
    compute_factor_recovery_r_squared, forecast_with_factors,
    compute_forecast_closeness_s_squared, evaluate_monte_carlo_repetition,
    aggregate_monte_carlo_results
)


def test_compute_factor_recovery_r_squared_perfect():
    """Test R² with perfect factor recovery."""
    np.random.seed(42)
    T, r = 100, 3
    
    F_true = np.random.normal(0, 1, size=(T, r))
    
    # Perfect recovery
    R_sq = compute_factor_recovery_r_squared(F_true, F_true)
    assert np.isclose(R_sq, 1.0, atol=1e-10)


def test_compute_factor_recovery_r_squared_rotation():
    """Test R² is invariant to rotation."""
    np.random.seed(42)
    T, r = 100, 3
    
    F_true = np.random.normal(0, 1, size=(T, r))
    
    # Random orthogonal rotation
    Q = np.linalg.qr(np.random.normal(0, 1, size=(r, r)))[0]
    F_rotated = F_true @ Q
    
    R_sq = compute_factor_recovery_r_squared(F_rotated, F_true)
    assert np.isclose(R_sq, 1.0, atol=1e-6)


def test_compute_factor_recovery_r_squared_partial():
    """Test R² with partial factor recovery."""
    np.random.seed(42)
    T, r = 100, 5
    
    # Create orthogonal factors to ensure partial recovery
    F_true_raw = np.random.normal(0, 1, size=(T, r))
    F_true, _ = np.linalg.qr(F_true_raw)
    F_true = F_true * np.sqrt(T)  # Scale to have reasonable norm
    
    # Recover only first 3 factors
    F_partial = F_true[:, :3]
    
    R_sq = compute_factor_recovery_r_squared(F_partial, F_true)
    
    # Should be between 0 and 1
    assert 0 < R_sq <= 1.0
    
    # Should be roughly 3/5 if factors are orthogonal
    # (actual value depends on factor correlations)
    assert R_sq > 0.3


def test_compute_factor_recovery_r_squared_zero():
    """Test R² with no factors."""
    np.random.seed(42)
    T, r = 100, 3
    
    F_true = np.random.normal(0, 1, size=(T, r))
    F_empty = np.zeros((T, 0))
    
    R_sq = compute_factor_recovery_r_squared(F_empty, F_true)
    assert R_sq == 0.0


def test_forecast_with_factors():
    """Test forecasting with factors."""
    np.random.seed(42)
    T, k = 100, 3
    
    # Generate data
    F = np.random.normal(0, 1, size=(T, k))
    beta_true = np.array([1.0, 0.5, -0.3])
    y = np.concatenate([[0], F @ beta_true + np.random.normal(0, 0.1, size=T)])
    
    # Forecast
    forecast, coefficients = forecast_with_factors(y, F, include_intercept=True)
    
    # Check that forecast is a scalar
    assert isinstance(forecast, (float, np.floating))
    
    # Check that coefficients have correct length (intercept + k factors)
    assert len(coefficients) == k + 1


def test_forecast_with_zero_factors():
    """Test forecasting with no factors (mean forecast)."""
    np.random.seed(42)
    T = 100
    
    y = np.random.normal(5, 1, size=T + 1)
    F = np.zeros((T, 0))
    
    forecast, coefficients = forecast_with_factors(y, F, include_intercept=True)
    
    # Should return mean of y (excluding first element)
    expected_mean = np.mean(y[1:T])
    assert np.isclose(forecast, expected_mean, atol=1e-6)


def test_compute_forecast_closeness_s_squared_perfect():
    """Test S² with perfect forecast match."""
    y_hat = 0.5
    y_tilde = 0.5
    
    S_sq = compute_forecast_closeness_s_squared(y_hat, y_tilde)
    assert np.isclose(S_sq, 1.0)


def test_compute_forecast_closeness_s_squared_close():
    """Test S² with close forecasts."""
    y_hat = 1.0
    y_tilde = 1.1
    
    S_sq = compute_forecast_closeness_s_squared(y_hat, y_tilde)
    
    # S² = 1 - (0.1² / 1.0²) = 1 - 0.01 = 0.99
    assert np.isclose(S_sq, 0.99)


def test_compute_forecast_closeness_s_squared_far():
    """Test S² with distant forecasts."""
    y_hat = 1.0
    y_tilde = 2.0
    
    S_sq = compute_forecast_closeness_s_squared(y_hat, y_tilde)
    
    # S² = 1 - (1.0² / 1.0²) = 0
    assert np.isclose(S_sq, 0.0)


def test_compute_forecast_closeness_s_squared_negative():
    """Test S² can be negative when forecasts are very different."""
    y_hat = 1.0
    y_tilde = 3.0
    
    S_sq = compute_forecast_closeness_s_squared(y_hat, y_tilde)
    
    # S² = 1 - (2.0² / 1.0²) = 1 - 4 = -3
    assert S_sq < 0


def test_evaluate_monte_carlo_repetition():
    """Test evaluation of a single Monte Carlo repetition."""
    np.random.seed(42)
    T, N, r = 100, 50, 3
    
    # Generate synthetic data
    F_true = np.random.normal(0, 1, size=(T, r))
    Lambda_true = np.random.normal(0, 1, size=(N, r))
    X = F_true @ Lambda_true.T + np.random.normal(0, 0.1, size=(T, N))
    
    beta_true = np.ones(r)
    y = np.concatenate([[0], F_true @ beta_true + np.random.normal(0, 0.1, size=T)])
    
    # Create F_hat_dict
    F_hat_dict = {
        0: np.zeros((T, 0)),
        1: F_true[:, :1],
        2: F_true[:, :2],
        3: F_true,  # Perfect recovery
    }
    
    k_selected = {
        'true': 3,
        'IC_p3': 3,
        'AIC': 2,
    }
    
    results = evaluate_monte_carlo_repetition(X, y, F_true, F_hat_dict, k_selected)
    
    # Check that all methods are evaluated
    assert 'true' in results
    assert 'IC_p3' in results
    assert 'AIC' in results
    
    # Check structure of results
    for method in results:
        assert 'k_selected' in results[method]
        assert 'R_squared' in results[method]
        assert 'y_hat' in results[method]
        assert 'y_tilde' in results[method]
        assert 'S_squared' in results[method]
    
    # Check that true method has perfect R²
    assert results['true']['R_squared'] > 0.99


def test_aggregate_monte_carlo_results():
    """Test aggregation of Monte Carlo results."""
    np.random.seed(42)
    n_reps = 10
    
    # Create mock results
    all_results = []
    for rep in range(n_reps):
        rep_result = {
            'true': {
                'k_selected': 3,
                'R_squared': 0.95 + 0.01 * np.random.randn(),
                'S_squared': 0.90 + 0.01 * np.random.randn(),
            },
            'IC_p3': {
                'k_selected': 3 + np.random.randint(-1, 2),
                'R_squared': 0.93 + 0.02 * np.random.randn(),
                'S_squared': 0.88 + 0.02 * np.random.randn(),
            }
        }
        all_results.append(rep_result)
    
    aggregated = aggregate_monte_carlo_results(all_results)
    
    # Check that both methods are aggregated
    assert 'true' in aggregated
    assert 'IC_p3' in aggregated
    
    # Check structure
    for method in aggregated:
        assert 'k_selected_mean' in aggregated[method]
        assert 'k_selected_std' in aggregated[method]
        assert 'R_squared_mean' in aggregated[method]
        assert 'R_squared_se' in aggregated[method]
        assert 'S_squared_mean' in aggregated[method]
        assert 'S_squared_se' in aggregated[method]
        assert 'n_reps' in aggregated[method]
    
    # Check that n_reps is correct
    assert aggregated['true']['n_reps'] == n_reps
    
    # Check that means are reasonable
    assert 0.9 < aggregated['true']['R_squared_mean'] < 1.0
    assert 0.8 < aggregated['true']['S_squared_mean'] < 1.0
