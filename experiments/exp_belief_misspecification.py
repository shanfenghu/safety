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
    
    iterations = config.ITERATIONS * 10
    nu_range = np.round(np.arange(0.1, 1.0, 0.1), 1)

    # --- Step 1: Pre-calculate the true, unbiased benchmarks for each possible nu_true ---
    print("\nStep 1: Pre-calculating true optimal benchmarks...")
    benchmarks = {}
    for nu_val in nu_range:
        print(f"  - Calculating benchmark for nu_true = {nu_val}")
        params = config.BASELINE_PARAMS.copy()
        params['nu'] = nu_val
        params['contract_type'] = 'optimal'
        
        params['developer_class'] = RationalDeveloperAgent
        
        # Run simulation with the truly optimal contract for this nu
        df_benchmark, _ = run(params, iterations=iterations)
        benchmarks[nu_val] = df_benchmark['SocialWelfare'].mean()
    
    print("--- Benchmarks Calculated ---")
    print(benchmarks)

    # --- Step 2: Run the main experiment grid ---
    nu_grid = list(product(nu_range, nu_range))
    print(f"\nStep 2: Testing {len(nu_grid)} combinations of (Assumed Nu, True Nu)...")

    all_results = []

    for i, (nu_assumed, nu_true) in enumerate(nu_grid):
        print(f"\nRunning scenario {i+1}/{len(nu_grid)}: Assumed Nu = {nu_assumed}, True Nu = {nu_true}")

        # --- A. Calculate and simulate the misspecified optimal contract ---
        params_assumed = config.BASELINE_PARAMS.copy()
        params_assumed['nu'] = nu_assumed
        misspecified_contract_menu, _ = calculate_optimal_contract(params_assumed)
        
        params_true = config.BASELINE_PARAMS.copy()
        params_true['nu'] = nu_true
        params_true['developer_class'] = RationalDeveloperAgent
        
        params_optimal_misspecified = params_true.copy()
        params_optimal_misspecified['contract_type'] = 'pre_calculated_optimal'
        params_optimal_misspecified['pre_calculated_menu'] = misspecified_contract_menu
        df_optimal, _ = run(params_optimal_misspecified, iterations=iterations)
        
        # --- B. Simulate the heuristic hybrid contract (control) ---
        params_hybrid = params_true.copy()
        params_hybrid['contract_type'] = 'hybrid'
        df_hybrid, _ = run(params_hybrid, iterations=iterations)

        # --- C. Collect results ---
        welfare_optimal_misspecified = df_optimal['SocialWelfare'].mean()
        welfare_hybrid = df_hybrid['SocialWelfare'].mean()
        
        # Use the pre-calculated, unbiased benchmark
        welfare_benchmark = benchmarks[nu_true]
        
        # On the diagonal, the loss is definitionally zero.
        if nu_assumed == nu_true:
            welfare_loss_optimal = 0.0
            # We still compare the hybrid to the true benchmark
            welfare_loss_hybrid = welfare_benchmark - welfare_hybrid
        else:
            welfare_loss_optimal = welfare_benchmark - welfare_optimal_misspecified
            welfare_loss_hybrid = welfare_benchmark - welfare_hybrid

        all_results.append({
            'nu_assumed': nu_assumed,
            'nu_true': nu_true,
            'welfare_optimal_misspecified': welfare_optimal_misspecified,
            'welfare_benchmark': welfare_benchmark,
            'welfare_hybrid': welfare_hybrid,
            'welfare_loss_optimal': welfare_loss_optimal,
            'welfare_loss_hybrid': welfare_loss_hybrid
        })

    # --- Step 3: Process and Save Final Results ---
    print("\nStep 3: Processing and saving final results...")
    results_df = pd.DataFrame(all_results)
    
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