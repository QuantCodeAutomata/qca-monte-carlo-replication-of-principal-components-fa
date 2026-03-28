"""
Monte Carlo Simulation Runner for Experiment 1

Runs Monte Carlo simulations to replicate Table 1 from the paper.
For each design, performs 2000 repetitions and evaluates:
- Factor space recovery (R²)
- Forecast closeness to infeasible true-factor forecast (S²)
- Performance across multiple factor selection criteria
"""

import numpy as np
from typing import Dict, List
import time
from tqdm import tqdm

from .data_generation import DGPParameters, generate_panel_data, get_table1_designs
from .pca_estimation import extract_factors_for_candidates
from .factor_selection import select_factors
from .evaluation_metrics import evaluate_monte_carlo_repetition, aggregate_monte_carlo_results


def run_single_repetition(params: DGPParameters, seed: int, 
                         include_intercept: bool = True) -> Dict:
    """
    Run a single Monte Carlo repetition.
    
    Args:
        params: DGP parameters
        seed: Random seed for this repetition
        include_intercept: Whether to include intercept in forecasting
        
    Returns:
        Dictionary with evaluation results for all selection methods
    """
    # Generate data
    data = generate_panel_data(params, seed=seed)
    X = data['X']
    y = data['y']
    F_true = data['F_static']
    
    # Extract principal components for all candidate dimensions
    pca_results = extract_factors_for_candidates(X, params.k, standardize=False)
    
    # Select number of factors using all criteria
    k_selected = select_factors(
        V_dict=pca_results['V'],
        y=y,
        F_dict=pca_results['F'],
        T=params.T,
        N=params.N,
        k_max=params.k,
        r_true=params.r_static,
        include_intercept=include_intercept
    )
    
    # Evaluate this repetition
    results = evaluate_monte_carlo_repetition(
        X=X,
        y=y,
        F_true=F_true,
        F_hat_dict=pca_results['F'],
        k_selected=k_selected,
        include_intercept=include_intercept
    )
    
    return results


def run_design(design: Dict, n_reps: int = 2000, 
              include_intercept: bool = True,
              verbose: bool = True) -> Dict:
    """
    Run Monte Carlo simulation for a single design.
    
    Args:
        design: Design specification dictionary
        n_reps: Number of Monte Carlo repetitions
        include_intercept: Whether to include intercept in forecasting
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with aggregated results
    """
    # Create DGP parameters
    params = DGPParameters(
        T=design['T'],
        N=design['N'],
        r_tilde=design['r_tilde'],
        q=design['q'],
        k=design['k'],
        pi=design['pi'],
        a=design['a'],
        b=design['b'],
        c=design['c'],
        delta_1=design['delta_1'],
        delta_2=design['delta_2'],
        alpha=design['alpha']
    )
    
    if verbose:
        print(f"\nRunning design: {design['name']}")
        print(f"Parameters: T={params.T}, N={params.N}, r_tilde={params.r_tilde}, "
              f"q={params.q}, r_static={params.r_static}")
    
    # Run repetitions
    all_results = []
    
    iterator = tqdm(range(n_reps), desc=design['name']) if verbose else range(n_reps)
    
    for rep in iterator:
        try:
            result = run_single_repetition(params, seed=rep, include_intercept=include_intercept)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"\nWarning: Repetition {rep} failed with error: {e}")
            continue
    
    # Aggregate results
    aggregated = aggregate_monte_carlo_results(all_results)
    
    # Add design info
    aggregated['design'] = design
    aggregated['params'] = params
    
    return aggregated


def run_monte_carlo_experiment(n_reps: int = 2000, 
                               designs: List[Dict] = None,
                               include_intercept: bool = True,
                               verbose: bool = True) -> Dict[str, Dict]:
    """
    Run full Monte Carlo experiment for all designs.
    
    Args:
        n_reps: Number of Monte Carlo repetitions per design
        designs: List of design specifications. If None, uses Table 1 designs.
        include_intercept: Whether to include intercept in forecasting
        verbose: Whether to show progress
        
    Returns:
        Dictionary mapping design name -> aggregated results
    """
    if designs is None:
        designs = get_table1_designs()
    
    print(f"Running Monte Carlo experiment with {n_reps} repetitions per design")
    print(f"Total designs: {len(designs)}")
    
    results = {}
    start_time = time.time()
    
    for design in designs:
        design_start = time.time()
        
        result = run_design(design, n_reps=n_reps, 
                          include_intercept=include_intercept,
                          verbose=verbose)
        
        results[design['name']] = result
        
        design_time = time.time() - design_start
        if verbose:
            print(f"Design completed in {design_time:.1f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    return results


def format_table1_results(results: Dict[str, Dict]) -> str:
    """
    Format Monte Carlo results in a table similar to Table 1 from the paper.
    
    Args:
        results: Dictionary of aggregated results from run_monte_carlo_experiment
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 1: Monte Carlo Results - Factor Recovery and Forecast Efficiency")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Design':<25} {'Method':<10} {'k_sel':<8} {'R²':<10} {'S²':<10}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Results for each design
    for design_name, result in results.items():
        design = result['design']
        
        # Show key parameters
        param_str = f"T={design['T']}, N={design['N']}, r={result['params'].r_static}"
        
        first_row = True
        for method in ['true', 'IC_p1', 'IC_p2', 'IC_p3', 'AIC', 'BIC']:
            if method in result:
                method_result = result[method]
                
                design_col = f"{design_name}" if first_row else ""
                k_sel = f"{method_result['k_selected_mean']:.2f}"
                R_sq = f"{method_result['R_squared_mean']:.4f}"
                S_sq = f"{method_result['S_squared_mean']:.4f}"
                
                row = f"{design_col:<25} {method:<10} {k_sel:<8} {R_sq:<10} {S_sq:<10}"
                lines.append(row)
                
                first_row = False
        
        lines.append("")
    
    lines.append("=" * 120)
    lines.append(f"Note: Results based on {result['true']['n_reps']} Monte Carlo repetitions per design.")
    lines.append("R² = Factor space recovery statistic")
    lines.append("S² = Forecast closeness to infeasible true-factor forecast")
    lines.append("k_sel = Average selected number of factors")
    lines.append("=" * 120)
    
    return "\n".join(lines)
