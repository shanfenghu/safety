# experiments/exp_stakes_sensitivity.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import config
from src.pipeline import run
from src.contracts.optimal import calculate_optimal_contract

def main():
    """
    Runs a sensitivity analysis on the environmental stakes (delta_W) to test
    how the severity of a disaster affects contract performance.
    """
    print("--- Starting Environmental Stakes (delta_W) Sensitivity Experiment ---")

    iterations = config.ITERATIONS * 10
    # Use the same sweep range as the comparative statics for consistency
    delta_w_sweep = np.linspace(100, 5000, 50)
    base_params = config.BASELINE_PARAMS.copy()
    base_params.update(config.AGENT_CONFIGS['rational'])

    # --- Phase 1: Deterministic Solver with Warm Starts for the Optimal Contract ---
    print("\n--- Phase 1: Solving for optimal contracts sequentially ---")
    optimal_parameter_sets = []
    last_solution = None

    for delta_w in tqdm(delta_w_sweep, desc="Solving for optimal contracts"):
        # a. Create params for this specific solver run
        params = base_params.copy()
        params['delta_W'] = delta_w
        if last_solution is not None:
            params['initial_guess'] = last_solution
            
        # b. Solve for the optimal contract ONCE
        contract_menu, optimal_efforts = calculate_optimal_contract(params)
        
        # c. Prepare the parameter set for the simulation phase
        sim_params = params.copy()
        sim_params['contract_type'] = 'pre_calculated_optimal'
        sim_params['pre_calculated_menu'] = contract_menu
        optimal_parameter_sets.append(sim_params)
        
        # d. Update the warm start for the next iteration
        last_solution = optimal_efforts.get('solver_solution')

    # --- Phase 2: Stochastic Simulation with All Contract Types ---
    print("\n--- Phase 2: Running stochastic simulations for all contracts ---")
    all_results = []

    # Run simulations for the pre-calculated optimal contracts
    print("  - Running simulations for Optimal contract...")
    df_optimal, _ = run(parameters=optimal_parameter_sets, iterations=iterations)
    all_results.append(df_optimal)

    # Run simulations for the heuristic contracts
    heuristic_contracts = ['fine', 'hybrid', 'performance']
    for contract_type in heuristic_contracts:
        print(f"  - Running simulations for {contract_type} contract...")
        heuristic_parameter_sets = []
        for delta_w in delta_w_sweep:
            params = base_params.copy()
            params['delta_W'] = delta_w
            params['contract_type'] = contract_type
            heuristic_parameter_sets.append(params)
        
        df_heuristic, _ = run(parameters=heuristic_parameter_sets, iterations=iterations)
        all_results.append(df_heuristic)

    # --- 3. Process and Save Final Results ---
    print("\n--- Experiment Complete ---")
    results_df = pd.concat(all_results)
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "stakes_sensitivity.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved results to '{output_path}'")

    print("\n--- Results Snippet ---")
    # Aggregate and pivot the results for a quick view
    final_welfare = results_df.groupby(['delta_W', 'contract_type'])['SocialWelfare'].mean().unstack()
    print(final_welfare.round(2).head()) # Print the first 5 rows
    
    print("\n--- Stakes Sensitivity Experiment Complete ---")


if __name__ == "__main__":
    main()