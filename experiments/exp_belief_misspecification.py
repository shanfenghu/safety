# experiments/exp_belief_misspecification.py

import pandas as pd
import numpy as np
from itertools import product

import config
from src.pipeline import run
from src.agents.rational import RationalDeveloperAgent
from src.contracts.optimal import calculate_optimal_contract


def main():
    """Defines and runs the belief misspecification experiment."""
    print("--- Starting Belief Misspecification Experiment ---")
    
    # Set to 100 for the final, high-quality results
    iterations = config.ITERATIONS * 10

    nu_range = np.round(np.arange(0.1, 1.0, 0.1), 1)
    nu_grid = list(product(nu_range, nu_range))
    print(f"  - Testing {len(nu_grid)} combinations of (Assumed Nu, True Nu)...")

    all_results = []

    for i, (nu_assumed, nu_true) in enumerate(nu_grid):
        print(f"\nRunning scenario {i+1}/{len(nu_grid)}: Assumed Nu = {nu_assumed}, True Nu = {nu_true}")

        # --- A. Calculate the misspecified optimal contract ---
        params_assumed = config.BASELINE_PARAMS.copy()
        params_assumed['nu'] = nu_assumed
        misspecified_contract_menu, optimal_efforts = calculate_optimal_contract(params_assumed)
        
        misspecified_solution_vector = optimal_efforts['solver_solution']

        # --- B. Run simulations with the true environment parameters ---
        params_true = config.BASELINE_PARAMS.copy()
        params_true['nu'] = nu_true
        params_true['developer_class'] = RationalDeveloperAgent
        
        # Scenario 1: Optimal contract under misspecified beliefs
        params_optimal_misspecified = params_true.copy()
        params_optimal_misspecified['contract_type'] = 'pre_calculated_optimal'
        params_optimal_misspecified['pre_calculated_menu'] = misspecified_contract_menu
        df_optimal, _ = run(params_optimal_misspecified, iterations=iterations)
        
        # Scenario 2: The "perfect information" benchmark for this true nu
        params_benchmark = params_true.copy()
        params_benchmark['contract_type'] = 'optimal'
        params_benchmark['initial_guess'] = misspecified_solution_vector
        
        benchmark_contract_menu, _ = calculate_optimal_contract(params_benchmark)
        params_benchmark_run = params_true.copy()
        params_benchmark_run['contract_type'] = 'pre_calculated_optimal'
        params_benchmark_run['pre_calculated_menu'] = benchmark_contract_menu
        df_benchmark, _ = run(params_benchmark_run, iterations=iterations)
        
        # Scenario 3: The heuristic hybrid contract (control)
        params_hybrid = params_true.copy()
        params_hybrid['contract_type'] = 'hybrid'
        df_hybrid, _ = run(params_hybrid, iterations=iterations)

        # --- C. Collect and store results for this grid cell ---
        welfare_optimal = df_optimal['SocialWelfare'].mean()
        welfare_benchmark = df_benchmark['SocialWelfare'].mean()
        welfare_hybrid = df_hybrid['SocialWelfare'].mean()
        
        all_results.append({
            'nu_assumed': nu_assumed,
            'nu_true': nu_true,
            'welfare_optimal_misspecified': welfare_optimal,
            'welfare_benchmark': welfare_benchmark,
            'welfare_hybrid': welfare_hybrid
        })

    # --- 3. Process and Save Final Results ---
    print("\nStep 3: Processing and saving final results...")
    results_df = pd.DataFrame(all_results)
    
    results_df['welfare_loss_optimal'] = results_df['welfare_benchmark'] - results_df['welfare_optimal_misspecified']
    results_df['welfare_loss_hybrid'] = results_df['welfare_benchmark'] - results_df['welfare_hybrid']
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "belief_misspecification.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Successfully saved results to '{output_path}'")

    print("\n--- Results Snippet (Welfare Loss for Optimal Contract) ---")
    loss_pivot = results_df.pivot(index='nu_true', columns='nu_assumed', values='welfare_loss_optimal')
    print(loss_pivot.round(2))
    
    print("\n--- Belief Misspecification Experiment Complete ---")


if __name__ == "__main__":
    main()