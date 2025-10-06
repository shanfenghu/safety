# experiments/exp_comparative_statics.py

"""
Runs the comparative statics analysis using a two-phase approach.

Phase 1: A deterministic "warm start" loop to find the stable, optimal effort
         levels and corresponding contracts for each value of the swept
         parameter ('delta_W'). This avoids numerical instability.

Phase 2: A stochastic batch run that executes a large number of simulations
         for each of the pre-calculated optimal contracts to gather robust
         statistics for plotting error bars.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
from src.pipeline import run
from src.contracts.optimal import calculate_optimal_contract

def main():
    """Defines and runs the two-phase comparative statics experiment."""
    print("--- Starting Comparative Statics Experiment (Two-Phase) ---")

    # --- 1. Setup the Parameter Sweep ---
    delta_w_sweep = np.linspace(100, 5000, 50)
    base_params = config.BASELINE_PARAMS.copy()
    base_params.update(config.AGENT_CONFIGS['rational'])
    
    # --- Phase 1: Deterministic Solver with Warm Starts ---
    print("\n--- Phase 1: Solving for optimal contracts sequentially ---")
    parameter_sets_for_pipeline = []
    last_solution = None

    for delta_w in tqdm(delta_w_sweep, desc="Solving for optimal contracts"):
        # a. Create params for this specific solver run
        params = base_params.copy()
        params['delta_W'] = delta_w
        if last_solution is not None:
            params['initial_guess'] = last_solution
            
        # b. Solve for the optimal contract ONCE for this delta_w value
        contract_menu, optimal_efforts = calculate_optimal_contract(params)
        
        # c. Prepare the parameter set for the simulation phase. We use the
        #    special 'pre_calculated_optimal' hook in the model to bypass the solver.
        sim_params = params.copy()
        sim_params['contract_type'] = 'pre_calculated_optimal'
        sim_params['pre_calculated_menu'] = contract_menu
        
        parameter_sets_for_pipeline.append(sim_params)
        
        # d. Update the warm start for the next iteration
        last_solution = optimal_efforts.get('solver_solution')

    # --- Phase 2: Stochastic Simulation with Pre-calculated Contracts ---
    print("\n--- Phase 2: Running stochastic simulations for statistics ---")
    iterations = config.ITERATIONS * 10
    print(f"Running {iterations} iterations for each of the {len(delta_w_sweep)} solved points.")
    
    # We now call our robust pipeline with the list of fully defined parameter sets.
    # The pipeline will handle the iterations and data aggregation.
    results_df, _ = run(
        parameters=parameter_sets_for_pipeline,
        iterations=iterations,
        number_processes=config.NUM_PROCESSES
    )
    
    # --- 3. Save the Results ---
    print("\nStep 3: Saving raw results to disk...")
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "comparative_statics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Successfully saved results to '{output_path}'")

    print("\n--- Comparative Statics Experiment Complete ---")


if __name__ == "__main__":
    main()