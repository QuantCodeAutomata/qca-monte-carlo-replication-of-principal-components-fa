"""
Main script to run all experiments and generate results.

Executes:
1. Monte Carlo simulation (Experiment 1)
2. Empirical forecasting (Experiment 2)
3. Visualization and reporting
"""

import os
import sys
import time
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.monte_carlo import run_monte_carlo_experiment, format_table1_results
from src.empirical_forecasting import run_empirical_experiment, format_table2_results
from src.visualization import (
    plot_monte_carlo_results, plot_empirical_results, save_results_markdown
)


def run_experiment_1(n_reps: int = 100, verbose: bool = True):
    """
    Run Monte Carlo simulation experiment.
    
    Args:
        n_reps: Number of Monte Carlo repetitions (default: 100 for speed, use 2000 for full replication)
        verbose: Whether to show progress
        
    Returns:
        Tuple of (results, formatted_table)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Monte Carlo Simulation")
    print("="*80)
    
    start_time = time.time()
    
    # Run Monte Carlo experiment
    results = run_monte_carlo_experiment(
        n_reps=n_reps,
        designs=None,  # Use default Table 1 designs
        include_intercept=True,
        verbose=verbose
    )
    
    # Format results
    table = format_table1_results(results)
    
    elapsed = time.time() - start_time
    print(f"\nExperiment 1 completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("\n" + table)
    
    return results, table


def run_experiment_2(verbose: bool = True):
    """
    Run empirical forecasting experiment.
    
    Args:
        verbose: Whether to show progress
        
    Returns:
        Tuple of (forecasts, evaluation, formatted_table)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Empirical Forecasting")
    print("="*80)
    
    start_time = time.time()
    
    # Run empirical experiment
    forecasts, evaluation = run_empirical_experiment(
        data_dir='data',
        train_start='1959-01',
        eval_start='1970-01',
        eval_end='1997-12',
        forecast_horizon=12,
        k_max=10,
        ar_lags=12,
        verbose=verbose
    )
    
    if forecasts is None or evaluation is None:
        print("Experiment 2 failed to complete.")
        return None, None, None
    
    # Format results
    table = format_table2_results(evaluation)
    
    elapsed = time.time() - start_time
    print(f"\nExperiment 2 completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("\n" + table)
    
    return forecasts, evaluation, table


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run principal components factor model experiments'
    )
    parser.add_argument(
        '--exp1-reps', type=int, default=100,
        help='Number of Monte Carlo repetitions for Experiment 1 (default: 100, use 2000 for full replication)'
    )
    parser.add_argument(
        '--skip-exp1', action='store_true',
        help='Skip Experiment 1 (Monte Carlo)'
    )
    parser.add_argument(
        '--skip-exp2', action='store_true',
        help='Skip Experiment 2 (Empirical forecasting)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    print("\n" + "="*80)
    print("PRINCIPAL COMPONENTS FACTOR MODEL REPLICATION")
    print("="*80)
    print("\nThis script replicates two experiments:")
    print("1. Monte Carlo simulation validating factor recovery and forecast efficiency")
    print("2. Empirical forecasting of U.S. industrial production")
    print("\nResults will be saved to the 'results/' directory.")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Run experiments
    mc_results = None
    mc_table = None
    emp_forecasts = None
    emp_evaluation = None
    emp_table = None
    
    if not args.skip_exp1:
        mc_results, mc_table = run_experiment_1(n_reps=args.exp1_reps, verbose=verbose)
    else:
        print("\nSkipping Experiment 1 (Monte Carlo)")
    
    if not args.skip_exp2:
        emp_forecasts, emp_evaluation, emp_table = run_experiment_2(verbose=verbose)
    else:
        print("\nSkipping Experiment 2 (Empirical forecasting)")
    
    # Generate visualizations
    if not args.no_plots:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        if mc_results is not None:
            print("\nCreating Monte Carlo plots...")
            plot_monte_carlo_results(mc_results, output_dir='results')
        
        if emp_forecasts is not None and emp_evaluation is not None:
            print("Creating empirical forecasting plots...")
            plot_empirical_results(emp_forecasts, emp_evaluation, output_dir='results')
    
    # Save results to markdown
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    save_results_markdown(
        monte_carlo_results=mc_results if mc_results else {},
        empirical_evaluation=emp_evaluation if emp_evaluation else {},
        monte_carlo_table=mc_table if mc_table else "Not run",
        empirical_table=emp_table if emp_table else "Not run",
        output_file='results/RESULTS.md'
    )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/RESULTS.md (main results document)")
    if not args.no_plots:
        print("  - results/*.png (visualizations)")
    print("\nTo view results:")
    print("  cat results/RESULTS.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
