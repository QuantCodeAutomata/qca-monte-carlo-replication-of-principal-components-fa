"""
Principal Components Analysis for Factor Extraction

Implements PCA-based factor estimation following the paper's methodology:
- Minimizes V(F, Lambda) = (NT)^{-1} sum_i sum_t (x_it - lambda_i' F_t)^2
- Uses SVD for numerical stability
- Consistent normalization: Lambda' Lambda / N = I, F_hat = X Lambda / N
"""

import numpy as np
from typing import Tuple, Optional


def extract_principal_components(X: np.ndarray, k_max: int, 
                                  standardize: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract principal components from panel data matrix.
    
    Uses SVD decomposition for numerical stability. The normalization follows:
    Lambda' Lambda / N = I and F_hat = X Lambda / N.
    
    Args:
        X: Data matrix of shape (T, N) where T is time periods and N is cross-section
        k_max: Maximum number of factors to extract
        standardize: Whether to standardize X before PCA (default: False per paper)
        
    Returns:
        Tuple of (F_hat, Lambda_hat, eigenvalues) where:
        - F_hat: Estimated factors of shape (T, k_max)
        - Lambda_hat: Estimated loadings of shape (N, k_max)
        - eigenvalues: Array of shape (k_max,) containing eigenvalues
    """
    T, N = X.shape
    
    # Optionally standardize (typically not done in Monte Carlo per paper)
    if standardize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0, ddof=1)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        X_centered = (X - X_mean) / X_std
    else:
        X_centered = X.copy()
    
    # Compute X'X / (NT) for eigendecomposition
    # Use SVD for numerical stability: X = U S V'
    # Then X'X = V S² V'
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Eigenvalues of X'X / (NT)
    eigenvalues = (s**2) / (N * T)
    
    # Extract top k_max components
    k_max = min(k_max, len(eigenvalues))
    
    # Loadings: eigenvectors of X'X normalized so Lambda' Lambda / N = I
    # V contains eigenvectors as columns
    V = Vt.T  # Shape (N, min(T,N))
    Lambda_hat = V[:, :k_max] * np.sqrt(N)  # Normalization
    
    # Factors: F_hat = X Lambda / N
    F_hat = X_centered @ Lambda_hat / N
    
    return F_hat, Lambda_hat, eigenvalues[:k_max]


def compute_reconstruction_error(X: np.ndarray, F: np.ndarray, Lambda: np.ndarray) -> float:
    """
    Compute the reconstruction error V(F, Lambda).
    
    V(F, Lambda) = (NT)^{-1} sum_i sum_t (x_it - lambda_i' F_t)^2
    
    Args:
        X: Data matrix of shape (T, N)
        F: Factor matrix of shape (T, k)
        Lambda: Loading matrix of shape (N, k)
        
    Returns:
        Reconstruction error (scalar)
    """
    T, N = X.shape
    X_hat = F @ Lambda.T
    residuals = X - X_hat
    V = np.sum(residuals**2) / (N * T)
    return V


def extract_factors_for_candidates(X: np.ndarray, k_max: int, 
                                   standardize: bool = False) -> dict:
    """
    Extract factors for all candidate dimensions j = 0, 1, ..., k_max.
    
    Args:
        X: Data matrix of shape (T, N)
        k_max: Maximum number of factors
        standardize: Whether to standardize X before PCA
        
    Returns:
        Dictionary with keys:
        - 'F_all': Full factor matrix (T, k_max)
        - 'Lambda_all': Full loading matrix (N, k_max)
        - 'eigenvalues': Eigenvalues (k_max,)
        - 'V': Dictionary mapping j -> reconstruction error for j factors
        - 'F': Dictionary mapping j -> factor matrix (T, j) for j factors
        - 'Lambda': Dictionary mapping j -> loading matrix (N, j) for j factors
    """
    T, N = X.shape
    
    # Extract all k_max factors
    F_all, Lambda_all, eigenvalues = extract_principal_components(X, k_max, standardize)
    
    # Compute reconstruction errors for each candidate j
    V_dict = {}
    F_dict = {}
    Lambda_dict = {}
    
    for j in range(k_max + 1):
        if j == 0:
            # No factors: V_0 = variance of X
            V_dict[j] = np.sum(X**2) / (N * T)
            F_dict[j] = np.zeros((T, 0))
            Lambda_dict[j] = np.zeros((N, 0))
        else:
            F_j = F_all[:, :j]
            Lambda_j = Lambda_all[:, :j]
            V_dict[j] = compute_reconstruction_error(X, F_j, Lambda_j)
            F_dict[j] = F_j
            Lambda_dict[j] = Lambda_j
    
    return {
        'F_all': F_all,
        'Lambda_all': Lambda_all,
        'eigenvalues': eigenvalues,
        'V': V_dict,
        'F': F_dict,
        'Lambda': Lambda_dict
    }


def standardize_panel(X: np.ndarray, mean: Optional[np.ndarray] = None, 
                     std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize panel data by subtracting mean and dividing by standard deviation.
    
    This is used in empirical applications where each predictor series is standardized
    using only information available up to the current forecast origin.
    
    Args:
        X: Data matrix of shape (T, N)
        mean: Optional pre-computed means of shape (N,). If None, computed from X.
        std: Optional pre-computed standard deviations of shape (N,). If None, computed from X.
        
    Returns:
        Tuple of (X_standardized, mean, std)
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0  # Avoid division by zero
    
    X_standardized = (X - mean) / std
    
    return X_standardized, mean, std


def compute_factor_recovery_statistic(F_hat: np.ndarray, F_true: np.ndarray) -> float:
    """
    Compute the factor space recovery statistic R²_{F_hat, F}.
    
    R²_{F_hat, F} = tr(F_hat' P_F F_hat) / tr(F_hat' F_hat)
    
    where P_F = F (F'F)^{-1} F' is the projection matrix onto the true factor space.
    
    This statistic is invariant to sign flips and measures how well the estimated
    factor space spans the true factor space.
    
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
    FtF_inv = np.linalg.inv(FtF)
    P_F = F_true @ FtF_inv @ F_true.T
    
    # Compute R² = tr(F_hat' P_F F_hat) / tr(F_hat' F_hat)
    numerator = np.trace(F_hat.T @ P_F @ F_hat)
    denominator = np.trace(F_hat.T @ F_hat)
    
    if denominator == 0:
        return 0.0
    
    R_squared = numerator / denominator
    
    return R_squared
