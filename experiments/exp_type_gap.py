# experiments/exp_type_gap.py (Final Fixed Version)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import config
from src.pipeline import run
from src.contracts.optimal import calculate_optimal_contract

def main():
    """
    Runs a sensitivity analysis on the 'type gap' (theta_H) using a
    two-phase approach with a warm-start chain for stability.
    """
    print("--- Starting Type Gap Sensitivity Experiment (Two-Phase Fix) ---")

    iterations = config.ITERATIONS * 10
    theta_h_range = np.round(np.arange(1.5, 4.1, 0.1), 1)
    base_params = config.BASELINE_PARAMS.copy()
    base_params.update(config.AGENT_CONFIGS['rational'])

    # --- Phase 1: Deterministic Solver with Warm Starts for the Optimal Contract ---
    print("\n--- Phase 1: Solving for optimal contracts sequentially ---")
    optimal_parameter_sets = []
    last_solution = None

    for theta_h in tqdm(theta_h_range, desc="Solving for optimal contracts"):
        # a. Create params for this specific solver run
        params = base_params.copy()
        params['theta_H'] = theta_h
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
        for theta_h in theta_h_range:
            params = base_params.copy()
            params['theta_H'] = theta_h
            params['contract_type'] = contract_type
            heuristic_parameter_sets.append(params)
        
        df_heuristic, _ = run(parameters=heuristic_parameter_sets, iterations=iterations)
        all_results.append(df_heuristic)

    # --- 3. Process and Save Final Results ---
    print("\n--- Experiment Complete ---")
    results_df = pd.concat(all_results)
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "type_gap_sensitivity.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved results to '{output_path}'")

    print("\n--- Results Snippet ---")
    # We need to aggregate the results as the 'run' function returns per-step data
    final_welfare = results_df.groupby(['theta_H', 'contract_type'])['SocialWelfare'].mean().unstack()
    print(final_welfare.round(2))
    
    print("\n--- Type Gap Sensitivity Experiment Complete ---")


if __name__ == "__main__":
    main()