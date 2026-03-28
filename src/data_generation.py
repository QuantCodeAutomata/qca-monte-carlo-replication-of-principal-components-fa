"""
Synthetic Data Generation for Monte Carlo Experiments

Implements the dynamic factor model data generating process (DGP) with:
- Dynamic factors with AR structure
- Time-varying factor loadings
- Serial correlation in idiosyncratic errors
- Spatial cross-sectional dependence
- Conditional heteroskedasticity (GARCH)
- Irrelevant predictors mechanism
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DGPParameters:
    """
    Parameters for the dynamic factor model data generating process.
    
    Attributes:
        T: Time series length
        N: Number of cross-sectional units
        r_tilde: Dimension of primitive factor vector
        q: Number of dynamic lags in measurement equation
        k: Maximum number of factors to extract
        pi: Probability of irrelevant predictor (R²=0)
        a: AR coefficient for idiosyncratic errors
        b: Spatial MA coefficient for cross-sectional dependence
        c: Time-variation coefficient for loadings
        delta_1: GARCH AR coefficient
        delta_2: GARCH MA coefficient
        alpha: AR coefficient for primitive factors
    """
    T: int
    N: int
    r_tilde: int
    q: int
    k: int
    pi: float
    a: float
    b: float
    c: float
    delta_1: float
    delta_2: float
    alpha: float
    
    def __post_init__(self):
        """Validate parameters and compute derived quantities."""
        assert self.delta_1 + self.delta_2 < 1, "GARCH parameters must satisfy delta_1 + delta_2 < 1"
        assert abs(self.alpha) < 1, "Factor AR coefficient must satisfy |alpha| < 1"
        assert abs(self.a) < 1, "Idiosyncratic AR coefficient must satisfy |a| < 1"
        assert 0 <= self.pi <= 1, "Irrelevant predictor probability must be in [0,1]"
        
    @property
    def delta_0(self) -> float:
        """GARCH intercept ensuring unconditional variance = 1."""
        return 1.0 - self.delta_1 - self.delta_2
    
    @property
    def r_static(self) -> int:
        """Dimension of static factor representation."""
        return self.r_tilde * (1 + self.q)


def initialize_factors(r_tilde: int, alpha: float, q: int, T: int) -> np.ndarray:
    """
    Initialize primitive factors from stationary distribution.
    
    Args:
        r_tilde: Dimension of primitive factor vector
        alpha: AR coefficient (applied componentwise)
        q: Number of lags needed
        T: Time series length
        
    Returns:
        Array of shape (T+q, r_tilde) containing factors from t=-q to t=T-1
    """
    # Stationary variance for AR(1) with unit innovation variance
    if abs(alpha) < 1:
        stationary_var = 1.0 / (1.0 - alpha**2)
    else:
        stationary_var = 1.0
    
    # Initialize pre-sample factors from stationary distribution
    f = np.zeros((T + q, r_tilde))
    f[:q] = np.random.normal(0, np.sqrt(stationary_var), size=(q, r_tilde))
    
    # Generate factors via AR(1) recursion
    for t in range(q, T + q):
        u_t = np.random.normal(0, 1, size=r_tilde)
        f[t] = alpha * f[t-1] + u_t
    
    return f


def calibrate_lambda_star(R_squared: float, r_tilde: int, q: int, alpha: float) -> float:
    """
    Calibrate loading scale factor to achieve target R² at t=0.
    
    The target R² is defined as:
    R² = Var(sum_{j=0}^q lambda_{ij0}' f_{-j}) / Var(x_{i0})
    
    where Var(x_{i0}) = Var(factor component) + Var(e_{i0})
    
    Args:
        R_squared: Target fraction of variance explained by factors
        r_tilde: Dimension of primitive factor vector
        q: Number of dynamic lags
        alpha: AR coefficient for factors
        
    Returns:
        Scaling factor lambda_star
    """
    if R_squared == 0:
        return 0.0
    
    # Variance of primitive factor (stationary)
    if abs(alpha) < 1:
        var_f = 1.0 / (1.0 - alpha**2)
    else:
        var_f = 1.0
    
    # Expected squared norm of raw loading vector (iid N(0,1))
    # E[||lambda_tilde||²] = r_tilde for each lag
    expected_loading_sq_norm = r_tilde
    
    # Variance of factor component: sum over q+1 lags
    # Assuming independence across lags (valid for stationary factors)
    var_factor_component = (q + 1) * expected_loading_sq_norm * var_f
    
    # Variance of idiosyncratic error (stationary, unconditional variance = 1)
    var_e = 1.0
    
    # Solve for lambda_star:
    # R² = lambda_star² * var_factor_component / (lambda_star² * var_factor_component + var_e)
    # R² * (lambda_star² * var_factor_component + var_e) = lambda_star² * var_factor_component
    # R² * var_e = lambda_star² * var_factor_component * (1 - R²)
    # lambda_star² = R² * var_e / (var_factor_component * (1 - R²))
    
    lambda_star_squared = R_squared * var_e / (var_factor_component * (1.0 - R_squared))
    lambda_star = np.sqrt(lambda_star_squared)
    
    return lambda_star


def initialize_loadings(N: int, r_tilde: int, q: int, pi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize factor loadings using irrelevant-predictor mechanism.
    
    Args:
        N: Number of cross-sectional units
        r_tilde: Dimension of primitive factor vector
        q: Number of dynamic lags
        pi: Probability of irrelevant predictor
        alpha: AR coefficient for factors (needed for calibration)
        
    Returns:
        Tuple of (lambda_0, R_squared) where:
        - lambda_0: Array of shape (N, q+1, r_tilde) containing initial loadings
        - R_squared: Array of shape (N,) containing target R² values
    """
    lambda_0 = np.zeros((N, q + 1, r_tilde))
    R_squared = np.zeros(N)
    
    for i in range(N):
        # Draw R² from irrelevant-predictor mechanism
        if np.random.rand() < pi:
            R_squared[i] = 0.0
        else:
            R_squared[i] = np.random.uniform(0.1, 0.8)
        
        # Calibrate loading scale
        lambda_star = calibrate_lambda_star(R_squared[i], r_tilde, q, alpha)
        
        # Draw raw loadings
        for j in range(q + 1):
            lambda_tilde = np.random.normal(0, 1, size=r_tilde)
            lambda_0[i, j, :] = lambda_star * lambda_tilde
    
    return lambda_0, R_squared


def generate_garch_innovations(N: int, T: int, delta_0: float, delta_1: float, delta_2: float) -> np.ndarray:
    """
    Generate innovations with GARCH(1,1) conditional heteroskedasticity.
    
    The process is:
    v_it = sigma_it * eta_it
    sigma_it² = delta_0 + delta_1 * sigma_{i,t-1}² + delta_2 * v_{i,t-1}²
    
    where eta_it ~ iid N(0,1) and delta_0 = 1 - delta_1 - delta_2 ensures
    unconditional variance = 1.
    
    Args:
        N: Number of cross-sectional units
        T: Time series length
        delta_0: GARCH intercept
        delta_1: GARCH AR coefficient
        delta_2: GARCH MA coefficient
        
    Returns:
        Array of shape (N, T) containing GARCH innovations
    """
    v = np.zeros((N, T))
    sigma_sq = np.ones((N, T))  # Initialize at unconditional variance
    
    for i in range(N):
        for t in range(T):
            if t == 0:
                # Initialize at unconditional variance
                sigma_sq[i, t] = 1.0
            else:
                sigma_sq[i, t] = delta_0 + delta_1 * sigma_sq[i, t-1] + delta_2 * v[i, t-1]**2
            
            eta_it = np.random.normal(0, 1)
            v[i, t] = np.sqrt(sigma_sq[i, t]) * eta_it
    
    return v


def generate_idiosyncratic_errors(N: int, T: int, a: float, b: float, 
                                   delta_0: float, delta_1: float, delta_2: float) -> np.ndarray:
    """
    Generate idiosyncratic errors with serial and spatial dependence.
    
    The process is:
    (1 - aL) e_it = (1 + b²) v_it + b v_{i+1,t} + b v_{i-1,t}
    
    where v_it follows GARCH(1,1) with unconditional variance = 1.
    Boundary condition: v_{0,t} = v_{N+1,t} = 0.
    
    Args:
        N: Number of cross-sectional units
        T: Time series length
        a: AR coefficient for idiosyncratic errors
        b: Spatial MA coefficient
        delta_0: GARCH intercept
        delta_1: GARCH AR coefficient
        delta_2: GARCH MA coefficient
        
    Returns:
        Array of shape (N, T) containing idiosyncratic errors
    """
    # Generate GARCH innovations
    v = generate_garch_innovations(N, T, delta_0, delta_1, delta_2)
    
    # Construct spatial MA component
    m = np.zeros((N, T))
    for i in range(N):
        m[i] = (1 + b**2) * v[i]
        if i > 0:
            m[i] += b * v[i-1]
        if i < N - 1:
            m[i] += b * v[i+1]
    
    # Generate AR errors
    e = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            if t == 0:
                # Initialize from stationary distribution or zero
                e[i, t] = m[i, t] / (1 - a) if abs(a) < 1 else m[i, t]
            else:
                e[i, t] = a * e[i, t-1] + m[i, t]
    
    return e


def generate_panel_data(params: DGPParameters, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic panel data from dynamic factor model.
    
    The model is:
    x_it = sum_{j=0}^q lambda_ijt' f_{t-j} + e_it
    y_{t+1} = sum_{j=0}^q iota' f_{t-j} + epsilon_{t+1}
    
    where:
    - f_t = alpha * f_{t-1} + u_t (primitive factors)
    - lambda_ijt = lambda_{ij,t-1} + (c/T) * zeta_ijt (time-varying loadings)
    - e_it follows AR with spatial MA and GARCH innovations
    
    Args:
        params: DGP parameters
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'X': Panel data matrix (T, N)
        - 'y': Target variable (T+1,)
        - 'F_static': True static factors (T, r_static)
        - 'F_primitive': True primitive factors (T, r_tilde)
        - 'Lambda': True loadings (N, q+1, r_tilde, T)
        - 'e': Idiosyncratic errors (N, T)
    """
    if seed is not None:
        np.random.seed(seed)
    
    T, N, r_tilde, q = params.T, params.N, params.r_tilde, params.q
    r_static = params.r_static
    
    # Initialize primitive factors (includes pre-sample lags)
    f_all = initialize_factors(r_tilde, params.alpha, q, T)
    f_primitive = f_all[q:]  # Shape (T, r_tilde), indexed from t=0 to T-1
    
    # Initialize loadings
    lambda_0, R_squared = initialize_loadings(N, r_tilde, q, params.pi, params.alpha)
    
    # Evolve time-varying loadings
    Lambda = np.zeros((N, q + 1, r_tilde, T))
    for i in range(N):
        for j in range(q + 1):
            Lambda[i, j, :, 0] = lambda_0[i, j, :]
            for t in range(1, T):
                zeta_ijt = np.random.normal(0, 1, size=r_tilde)
                Lambda[i, j, :, t] = Lambda[i, j, :, t-1] + (params.c / T) * zeta_ijt
    
    # Generate idiosyncratic errors
    e = generate_idiosyncratic_errors(N, T, params.a, params.b, 
                                      params.delta_0, params.delta_1, params.delta_2)
    
    # Construct panel data X
    X = np.zeros((T, N))
    for t in range(T):
        for i in range(N):
            factor_component = 0.0
            for j in range(q + 1):
                # f_{t-j} is at index t-j in f_all (which starts at index q for t=0)
                f_lag = f_all[q + t - j]  # This is f_{t-j}
                factor_component += Lambda[i, j, :, t] @ f_lag
            X[t, i] = factor_component + e[i, t]
    
    # Construct static factor matrix F_static
    F_static = np.zeros((T, r_static))
    for t in range(T):
        for j in range(q + 1):
            f_lag = f_all[q + t - j]
            F_static[t, j*r_tilde:(j+1)*r_tilde] = f_lag
    
    # Generate target variable y
    iota = np.ones(r_tilde)
    y = np.zeros(T + 1)
    for t in range(T):
        factor_component = 0.0
        for j in range(q + 1):
            if t - j >= 0:
                f_lag = f_all[q + t - j]
                factor_component += iota @ f_lag
        epsilon_t = np.random.normal(0, 1)
        y[t] = factor_component + epsilon_t
    # Generate y_{T+1}
    factor_component = 0.0
    for j in range(q + 1):
        f_lag = f_all[q + T - 1 - j]
        factor_component += iota @ f_lag
    epsilon_T1 = np.random.normal(0, 1)
    y[T] = factor_component + epsilon_T1
    
    return {
        'X': X,
        'y': y,
        'F_static': F_static,
        'F_primitive': f_primitive,
        'Lambda': Lambda,
        'e': e,
        'R_squared': R_squared
    }


def get_table1_designs() -> list[Dict]:
    """
    Return the simulation designs corresponding to Table 1 from the paper.
    
    These designs test various aspects:
    - Simple large-sample cases
    - Small-sample cases
    - Irrelevant predictors
    - Dependent errors (serial and spatial)
    - Dynamic factors
    - Time-varying loadings
    - Combined complications
    
    Returns:
        List of design dictionaries with parameter specifications
    """
    designs = []
    
    # Design 1: Simple large-sample baseline (static factors)
    designs.append({
        'name': 'Simple_T100_N250_r5',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 2: Small T
    designs.append({
        'name': 'Small_T25_N250_r5',
        'T': 25, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 3: Small N
    designs.append({
        'name': 'Small_T100_N50_r5',
        'T': 100, 'N': 50, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 4: Irrelevant predictors
    designs.append({
        'name': 'Irrelevant_pi0.5',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.5, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 5: Serial correlation in errors
    designs.append({
        'name': 'Serial_a0.5',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.5, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 6: Spatial dependence
    designs.append({
        'name': 'Spatial_b0.3',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.3, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 7: GARCH heteroskedasticity
    designs.append({
        'name': 'GARCH_d1_0.3_d2_0.4',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.3, 'delta_2': 0.4, 'alpha': 0.0
    })
    
    # Design 8: Dynamic factors (q=1)
    designs.append({
        'name': 'Dynamic_q1_alpha0.5',
        'T': 100, 'N': 250, 'r_tilde': 3, 'q': 1, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.5
    })
    
    # Design 9: Time-varying loadings (moderate)
    designs.append({
        'name': 'TimeVarying_c5',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 5.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 10: Time-varying loadings (strong)
    designs.append({
        'name': 'TimeVarying_c10',
        'T': 100, 'N': 250, 'r_tilde': 5, 'q': 0, 'k': 10,
        'pi': 0.0, 'a': 0.0, 'b': 0.0, 'c': 10.0,
        'delta_1': 0.0, 'delta_2': 0.0, 'alpha': 0.0
    })
    
    # Design 11: Combined complications
    designs.append({
        'name': 'Combined_all',
        'T': 100, 'N': 250, 'r_tilde': 3, 'q': 1, 'k': 10,
        'pi': 0.3, 'a': 0.3, 'b': 0.2, 'c': 5.0,
        'delta_1': 0.2, 'delta_2': 0.3, 'alpha': 0.5
    })
    
    return designs
