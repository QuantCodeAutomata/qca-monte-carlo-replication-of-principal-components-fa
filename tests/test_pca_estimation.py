"""
Tests for PCA estimation module.

Validates principal components extraction and factor recovery metrics.
"""

import numpy as np
import pytest
from src.pca_estimation import (
    extract_principal_components, compute_reconstruction_error,
    extract_factors_for_candidates, standardize_panel,
    compute_factor_recovery_statistic
)


def test_extract_principal_components():
    """Test basic PCA extraction."""
    np.random.seed(42)
    T, N = 100, 50
    k_max = 5
    
    # Generate data with known factor structure
    F_true = np.random.normal(0, 1, size=(T, k_max))
    Lambda_true = np.random.normal(0, 1, size=(N, k_max))
    X = F_true @ Lambda_true.T + np.random.normal(0, 0.1, size=(T, N))
    
    F_hat, Lambda_hat, eigenvalues = extract_principal_components(X, k_max)
    
    # Check shapes
    assert F_hat.shape == (T, k_max)
    assert Lambda_hat.shape == (N, k_max)
    assert eigenvalues.shape == (k_max,)
    
    # Check normalization: Lambda' Lambda / N = I
    Lambda_norm = Lambda_hat.T @ Lambda_hat / N
    np.testing.assert_array_almost_equal(Lambda_norm, np.eye(k_max), decimal=10)
    
    # Check that eigenvalues are in descending order
    assert np.all(np.diff(eigenvalues) <= 0)


def test_compute_reconstruction_error():
    """Test reconstruction error computation."""
    np.random.seed(42)
    T, N, k = 100, 50, 3
    
    F = np.random.normal(0, 1, size=(T, k))
    Lambda = np.random.normal(0, 1, size=(N, k))
    X = F @ Lambda.T
    
    # Perfect reconstruction should have zero error
    V = compute_reconstruction_error(X, F, Lambda)
    assert V < 1e-10
    
    # With noise, error should be positive
    X_noisy = X + np.random.normal(0, 0.5, size=(T, N))
    V_noisy = compute_reconstruction_error(X_noisy, F, Lambda)
    assert V_noisy > 0


def test_extract_factors_for_candidates():
    """Test extraction for multiple candidate dimensions."""
    np.random.seed(42)
    T, N = 100, 50
    k_max = 5
    
    # Generate data
    F_true = np.random.normal(0, 1, size=(T, 3))
    Lambda_true = np.random.normal(0, 1, size=(N, 3))
    X = F_true @ Lambda_true.T + np.random.normal(0, 0.1, size=(T, N))
    
    results = extract_factors_for_candidates(X, k_max)
    
    # Check that all required keys are present
    assert 'F_all' in results
    assert 'Lambda_all' in results
    assert 'eigenvalues' in results
    assert 'V' in results
    assert 'F' in results
    assert 'Lambda' in results
    
    # Check that V is computed for all candidates
    for j in range(k_max + 1):
        assert j in results['V']
        assert j in results['F']
        assert j in results['Lambda']
    
    # Check that V decreases with more factors
    V_values = [results['V'][j] for j in range(k_max + 1)]
    assert np.all(np.diff(V_values) <= 0)


def test_standardize_panel():
    """Test panel standardization."""
    np.random.seed(42)
    T, N = 100, 50
    
    X = np.random.normal(5, 2, size=(T, N))
    
    X_std, mean, std = standardize_panel(X)
    
    # Check shapes
    assert X_std.shape == X.shape
    assert mean.shape == (N,)
    assert std.shape == (N,)
    
    # Check that standardized data has mean ~0 and std ~1
    assert np.allclose(X_std.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_std.std(axis=0, ddof=1), 1, atol=1e-10)
    
    # Test with pre-computed mean and std
    X_std2, _, _ = standardize_panel(X, mean=mean, std=std)
    np.testing.assert_array_almost_equal(X_std, X_std2)


def test_compute_factor_recovery_statistic():
    """Test factor recovery R² statistic."""
    np.random.seed(42)
    T = 100
    r = 3
    
    # Perfect recovery: F_hat = F_true
    F_true = np.random.normal(0, 1, size=(T, r))
    R_sq = compute_factor_recovery_statistic(F_true, F_true)
    assert np.isclose(R_sq, 1.0, atol=1e-10)
    
    # Rotated factors: should still have R² = 1
    Q = np.linalg.qr(np.random.normal(0, 1, size=(r, r)))[0]
    F_rotated = F_true @ Q
    R_sq_rotated = compute_factor_recovery_statistic(F_rotated, F_true)
    assert np.isclose(R_sq_rotated, 1.0, atol=1e-6)
    
    # No factors: R² = 0
    F_empty = np.zeros((T, 0))
    R_sq_empty = compute_factor_recovery_statistic(F_empty, F_true)
    assert R_sq_empty == 0.0
    
    # Partial recovery
    F_partial = F_true[:, :2]  # Only first 2 factors
    R_sq_partial = compute_factor_recovery_statistic(F_partial, F_true)
    assert 0 < R_sq_partial < 1.0


def test_pca_sign_invariance():
    """Test that PCA results are consistent (up to sign)."""
    np.random.seed(42)
    T, N = 100, 50
    k = 3
    
    F_true = np.random.normal(0, 1, size=(T, k))
    Lambda_true = np.random.normal(0, 1, size=(N, k))
    X = F_true @ Lambda_true.T + np.random.normal(0, 0.1, size=(T, N))
    
    # Extract factors twice
    F_hat1, Lambda_hat1, _ = extract_principal_components(X, k)
    F_hat2, Lambda_hat2, _ = extract_principal_components(X, k)
    
    # Results should be identical (same seed)
    np.testing.assert_array_almost_equal(F_hat1, F_hat2)
    np.testing.assert_array_almost_equal(Lambda_hat1, Lambda_hat2)


def test_pca_with_zero_factors():
    """Test PCA extraction with k=0."""
    np.random.seed(42)
    T, N = 100, 50
    X = np.random.normal(0, 1, size=(T, N))
    
    results = extract_factors_for_candidates(X, k_max=5)
    
    # Check k=0 case
    assert results['F'][0].shape == (T, 0)
    assert results['Lambda'][0].shape == (N, 0)
    assert results['V'][0] > 0  # Should equal variance of X


def test_pca_numerical_stability():
    """Test PCA with ill-conditioned data."""
    np.random.seed(42)
    T, N = 100, 50
    k = 3
    
    # Create data with very different scales
    F = np.random.normal(0, 1, size=(T, k))
    Lambda = np.random.normal(0, 1, size=(N, k))
    Lambda[:, 0] *= 1000  # Large scale for first factor
    Lambda[:, 2] *= 0.001  # Small scale for third factor
    
    X = F @ Lambda.T + np.random.normal(0, 0.1, size=(T, N))
    
    # Should still work without errors
    F_hat, Lambda_hat, eigenvalues = extract_principal_components(X, k)
    
    assert F_hat.shape == (T, k)
    assert Lambda_hat.shape == (N, k)
    assert not np.any(np.isnan(F_hat))
    assert not np.any(np.isnan(Lambda_hat))
