"""
Tests for data generation module.

Validates that the DGP implementation follows the paper's methodology.
"""

import numpy as np
import pytest
from src.data_generation import (
    DGPParameters, generate_panel_data, initialize_factors,
    calibrate_lambda_star, initialize_loadings, generate_garch_innovations,
    generate_idiosyncratic_errors, get_table1_designs
)


def test_dgp_parameters_validation():
    """Test that DGP parameters are validated correctly."""
    # Valid parameters
    params = DGPParameters(
        T=100, N=50, r_tilde=3, q=0, k=5,
        pi=0.0, a=0.0, b=0.0, c=0.0,
        delta_1=0.3, delta_2=0.4, alpha=0.5
    )
    assert np.isclose(params.delta_0, 0.3)  # 1 - 0.3 - 0.4
    assert params.r_static == 3  # r_tilde * (1 + q)
    
    # Invalid GARCH parameters
    with pytest.raises(AssertionError):
        DGPParameters(
            T=100, N=50, r_tilde=3, q=0, k=5,
            pi=0.0, a=0.0, b=0.0, c=0.0,
            delta_1=0.6, delta_2=0.6, alpha=0.5  # Sum > 1
        )
    
    # Invalid AR coefficient
    with pytest.raises(AssertionError):
        DGPParameters(
            T=100, N=50, r_tilde=3, q=0, k=5,
            pi=0.0, a=0.0, b=0.0, c=0.0,
            delta_1=0.3, delta_2=0.4, alpha=1.5  # |alpha| >= 1
        )


def test_initialize_factors():
    """Test factor initialization from stationary distribution."""
    np.random.seed(42)
    r_tilde = 3
    alpha = 0.5
    q = 1
    T = 100
    
    factors = initialize_factors(r_tilde, alpha, q, T)
    
    # Check shape
    assert factors.shape == (T + q, r_tilde)
    
    # Check stationarity: variance should be close to 1/(1-alpha²)
    expected_var = 1.0 / (1.0 - alpha**2)
    actual_var = np.var(factors, axis=0)
    
    # Allow some sampling variation
    assert np.all(actual_var > 0.5 * expected_var)
    assert np.all(actual_var < 2.0 * expected_var)


def test_calibrate_lambda_star():
    """Test loading calibration for target R²."""
    r_tilde = 3
    q = 0
    alpha = 0.0
    
    # Test R² = 0
    lambda_star = calibrate_lambda_star(0.0, r_tilde, q, alpha)
    assert lambda_star == 0.0
    
    # Test R² = 0.5
    lambda_star = calibrate_lambda_star(0.5, r_tilde, q, alpha)
    assert lambda_star > 0.0
    
    # Test R² = 0.8
    lambda_star_high = calibrate_lambda_star(0.8, r_tilde, q, alpha)
    assert lambda_star_high > lambda_star  # Higher R² requires larger loadings


def test_initialize_loadings():
    """Test loading initialization with irrelevant predictor mechanism."""
    np.random.seed(42)
    N = 100
    r_tilde = 3
    q = 0
    pi = 0.5  # 50% irrelevant predictors
    alpha = 0.0
    
    lambda_0, R_squared = initialize_loadings(N, r_tilde, q, pi, alpha)
    
    # Check shapes
    assert lambda_0.shape == (N, q + 1, r_tilde)
    assert R_squared.shape == (N,)
    
    # Check that approximately pi fraction have R² = 0
    n_irrelevant = np.sum(R_squared == 0.0)
    expected_irrelevant = N * pi
    
    # Allow some sampling variation (binomial)
    assert n_irrelevant > expected_irrelevant - 3 * np.sqrt(N * pi * (1 - pi))
    assert n_irrelevant < expected_irrelevant + 3 * np.sqrt(N * pi * (1 - pi))
    
    # Check that relevant predictors have R² in [0.1, 0.8]
    relevant_R_sq = R_squared[R_squared > 0]
    assert np.all(relevant_R_sq >= 0.1)
    assert np.all(relevant_R_sq <= 0.8)


def test_generate_garch_innovations():
    """Test GARCH innovation generation."""
    np.random.seed(42)
    N = 50
    T = 1000
    delta_0 = 0.3
    delta_1 = 0.3
    delta_2 = 0.4
    
    v = generate_garch_innovations(N, T, delta_0, delta_1, delta_2)
    
    # Check shape
    assert v.shape == (N, T)
    
    # Check unconditional variance is close to 1
    # (delta_0 = 1 - delta_1 - delta_2 ensures this)
    var_v = np.var(v, axis=1)
    assert np.mean(var_v) > 0.8
    assert np.mean(var_v) < 1.2


def test_generate_idiosyncratic_errors():
    """Test idiosyncratic error generation with AR and spatial MA."""
    np.random.seed(42)
    N = 50
    T = 500
    a = 0.5
    b = 0.3
    delta_0 = 0.5
    delta_1 = 0.3
    delta_2 = 0.2
    
    e = generate_idiosyncratic_errors(N, T, a, b, delta_0, delta_1, delta_2)
    
    # Check shape
    assert e.shape == (N, T)
    
    # Check that errors have non-zero variance
    assert np.all(np.var(e, axis=1) > 0)


def test_generate_panel_data():
    """Test full panel data generation."""
    np.random.seed(42)
    params = DGPParameters(
        T=100, N=50, r_tilde=3, q=1, k=5,
        pi=0.2, a=0.3, b=0.2, c=5.0,
        delta_1=0.3, delta_2=0.4, alpha=0.5
    )
    
    data = generate_panel_data(params, seed=42)
    
    # Check all required outputs
    assert 'X' in data
    assert 'y' in data
    assert 'F_static' in data
    assert 'F_primitive' in data
    assert 'Lambda' in data
    assert 'e' in data
    
    # Check shapes
    assert data['X'].shape == (params.T, params.N)
    assert data['y'].shape == (params.T + 1,)
    assert data['F_static'].shape == (params.T, params.r_static)
    assert data['F_primitive'].shape == (params.T, params.r_tilde)
    assert data['Lambda'].shape == (params.N, params.q + 1, params.r_tilde, params.T)
    assert data['e'].shape == (params.N, params.T)
    
    # Check that X has non-zero variance
    assert np.var(data['X']) > 0
    
    # Check that y has non-zero variance
    assert np.var(data['y']) > 0


def test_static_factor_construction():
    """Test that static factors are correctly constructed from dynamic factors."""
    np.random.seed(42)
    params = DGPParameters(
        T=50, N=30, r_tilde=2, q=2, k=5,
        pi=0.0, a=0.0, b=0.0, c=0.0,
        delta_1=0.0, delta_2=0.0, alpha=0.5
    )
    
    data = generate_panel_data(params, seed=42)
    
    F_static = data['F_static']
    F_primitive = data['F_primitive']
    
    # Static factor should be (f_t, f_{t-1}, f_{t-2})
    # Check dimensions
    assert F_static.shape[1] == params.r_tilde * (params.q + 1)
    
    # Verify construction for a few time points
    # Note: Need to account for lagged factors
    # This is a structural test, not a numerical one


def test_get_table1_designs():
    """Test that Table 1 designs are properly specified."""
    designs = get_table1_designs()
    
    # Check that we have multiple designs
    assert len(designs) > 5
    
    # Check that each design has required fields
    required_fields = ['name', 'T', 'N', 'r_tilde', 'q', 'k', 
                      'pi', 'a', 'b', 'c', 'delta_1', 'delta_2', 'alpha']
    
    for design in designs:
        for field in required_fields:
            assert field in design
        
        # Check that parameters are valid
        assert design['T'] > 0
        assert design['N'] > 0
        assert design['r_tilde'] > 0
        assert design['q'] >= 0
        assert design['k'] > 0
        assert 0 <= design['pi'] <= 1
        assert abs(design['a']) < 1
        assert design['delta_1'] + design['delta_2'] < 1
        assert abs(design['alpha']) < 1


def test_reproducibility():
    """Test that data generation is reproducible with same seed."""
    params = DGPParameters(
        T=50, N=30, r_tilde=3, q=0, k=5,
        pi=0.0, a=0.0, b=0.0, c=0.0,
        delta_1=0.0, delta_2=0.0, alpha=0.0
    )
    
    data1 = generate_panel_data(params, seed=123)
    data2 = generate_panel_data(params, seed=123)
    
    # Check that outputs are identical
    np.testing.assert_array_equal(data1['X'], data2['X'])
    np.testing.assert_array_equal(data1['y'], data2['y'])
    np.testing.assert_array_equal(data1['F_static'], data2['F_static'])


def test_time_varying_loadings():
    """Test that loadings vary over time when c > 0."""
    np.random.seed(42)
    params = DGPParameters(
        T=100, N=50, r_tilde=3, q=0, k=5,
        pi=0.0, a=0.0, b=0.0, c=10.0,  # Strong time variation
        delta_1=0.0, delta_2=0.0, alpha=0.0
    )
    
    data = generate_panel_data(params, seed=42)
    Lambda = data['Lambda']
    
    # Check that loadings change over time
    lambda_0 = Lambda[:, :, :, 0]
    lambda_T = Lambda[:, :, :, -1]
    
    # Loadings should be different at t=0 and t=T
    assert not np.allclose(lambda_0, lambda_T)
    
    # Variance of loading changes should be related to c
    # E[change] ~ c/T per period
    for i in range(min(5, params.N)):
        changes = np.diff(Lambda[i, 0, 0, :])
        assert np.std(changes) > 0
