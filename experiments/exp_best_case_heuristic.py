# experiments/exp_best_case_heuristic.py

import pandas as pd
import numpy as np
from itertools import product

import config
from src.pipeline import run
from src.agents.rational import RationalDeveloperAgent


def main():
    """
    Finds the 'best-case' heuristic by performing a grid search over hybrid 
    contract bonus parameters and compares it to the optimal contract.
    """
    print("--- Starting Best-Case Heuristic Experiment ---")
    
    # Use fewer iterations for a quicker grid search, but increase for final results
    iterations = config.ITERATIONS * 10

    # --- 1. Define the search grid for heuristic bonuses ---
    bonus_range = np.arange(25, 251, 25)  # From 25 to 250 in steps of 25
    bonus_grid = list(product(bonus_range, bonus_range))
    print(f"Grid search over {len(bonus_grid)} combinations of (safety_bonus, perf_bonus)...")

    heuristic_results = []

    for i, (safety_bonus, perf_bonus) in enumerate(bonus_grid):
        print(f"  - Running hybrid combo {i+1}/{len(bonus_grid)}: safety={safety_bonus}, perf={perf_bonus}")
        
        params = config.BASELINE_PARAMS.copy()
        params['contract_type'] = 'hybrid'
        params['developer_class'] = RationalDeveloperAgent
        params['safety_bonus'] = float(safety_bonus)
        params['performance_bonus'] = float(perf_bonus)
        
        df_hybrid, _ = run(params, iterations=iterations)
        
        heuristic_results.append({
            'safety_bonus': safety_bonus,
            'performance_bonus': perf_bonus,
            'mean_social_welfare': df_hybrid['SocialWelfare'].mean()
        })

    # --- 2. Find the best-performing heuristic from the grid search ---
    results_df = pd.DataFrame(heuristic_results)
    best_heuristic = results_df.loc[results_df['mean_social_welfare'].idxmax()]
    
    print("\n--- Grid Search Complete ---")
    print("Best performing heuristic found:")
    print(best_heuristic)

    # --- 3. Run the theoretically optimal contract for comparison ---
    print("\nRunning optimal contract for benchmark comparison...")
    params_optimal = config.BASELINE_PARAMS.copy()
    params_optimal['contract_type'] = 'optimal'
    params_optimal['developer_class'] = RationalDeveloperAgent
    df_optimal, _ = run(params_optimal, iterations=iterations)
    optimal_welfare = df_optimal['SocialWelfare'].mean()
    
    print(f"Optimal Contract Mean Social Welfare: {optimal_welfare:.2f}")

    # --- 4. Save the comparison and the full grid search results ---
    # Save the full grid for heatmap plotting
    grid_output_path = config.RESULTS_DIR / "best_case_heuristic_grid.csv"
    results_df.to_csv(grid_output_path, index=False)
    print(f"\nSaved full grid search results to '{grid_output_path}'")
    
    # Save the final comparison data for a bar chart
    comparison_data = {
        'Contract': ['Optimal', 'Best-Case Heuristic'],
        'MeanSocialWelfare': [optimal_welfare, best_heuristic['mean_social_welfare']]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_output_path = config.RESULTS_DIR / "best_case_heuristic_comparison.csv"
    comparison_df.to_csv(comparison_output_path, index=False)
    print(f"Saved final comparison results to '{comparison_output_path}'")

    print("\n--- Best-Case Heuristic Experiment Complete ---")


if __name__ == "__main__":
    main()