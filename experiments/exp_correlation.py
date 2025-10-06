# experiments/exp_correlation.py

"""
Runs the advanced experiment on the impact of correlated efforts.

This script tests the robustness of the different contract types to the
model's orthogonality assumption. It runs the simulation for all four
contract types under three different scenarios for effort correlation:
1. Orthogonal (k=0): The baseline assumption.
2. Positive (k>0): Performance effort contributes to safety.
3. Negative (k<0): Performance effort detracts from safety.

The script generates the data needed for Figure 7 in the paper's appendix.
"""

import pandas as pd

# Import the centralized configuration and the experiment pipeline
import config
from src.pipeline import run


def validate_and_summarize_results(df: pd.DataFrame):
    """
    Performs post-experiment validation and prints a summary of results.

    Args:
        df: The DataFrame containing the raw experimental results.
    """
    print("\n--- Correlated Efforts Experiment: Summary ---")

    # --- 1. Group data and calculate mean disaster frequency ---
    # We group by both correlation and contract type for a full picture.
    summary = df.groupby(['correlation_k', 'contract_type']).agg(
        disaster_freq=('DisasterOccurred', 'mean')
    ).reset_index()

    # --- 2. Print Summary Table ---
    # Pivot the table for a more readable, wide-format summary.
    summary_pivot = summary.pivot(
        index='correlation_k', 
        columns='contract_type', 
        values='disaster_freq'
    )
    # Format as percentages
    summary_pivot = summary_pivot.applymap(lambda x: f"{x:.2%}")
    
    print("\n" + "="*70)
    print("      CORRELATED EFFORTS: DISASTER FREQUENCY SUMMARY")
    print("="*70)
    print(summary_pivot)
    print("="*70 + "\n")


def main():
    """Defines and runs the correlated efforts experiment."""
    print("--- Starting Correlated Efforts Robustness Experiment ---")

    # --- 1. Define Experimental Conditions ---
    print("Step 1: Defining parameter set for the experiment...")
    
    # Start with the baseline theoretical parameters
    params = config.BASELINE_PARAMS.copy()
    
    # This experiment uses the rational agent
    params.update(config.AGENT_CONFIGS['rational'])
    
    # Define the two parameters that will be varied in a grid
    params['contract_type'] = list(config.CONTRACT_CONFIGS.keys())
    # Ensure correlation_k is a list of floats, not dicts
    try:
        params['correlation_k'] = [float(v) for v in config.CORRELATION_SCENARIOS.values()]
    except Exception:
        # Fallback if values are dicts like {'correlation_k': 0.5}
        params['correlation_k'] = [float(v['correlation_k']) for v in config.CORRELATION_SCENARIOS.values()]
    
    num_contracts = len(params['contract_type'])
    num_correlations = len(params['correlation_k'])
    total_configs = num_contracts * num_correlations
    
    print(f"  - Variable parameter 'contract_type' set to: {params['contract_type']}")
    print(f"  - Variable parameter 'correlation_k' set to: {params['correlation_k']}")
    
    # --- 2. Execute the Simulation Pipeline ---
    print("\nStep 2: Executing simulation pipeline...")
    print(f"Running {config.ITERATIONS} iterations for each of the {total_configs} configurations.")
    
    results_df, _ = run(
        parameters=params,
        iterations=config.ITERATIONS * 10,
        number_processes=config.NUM_PROCESSES
    )
    
    # --- 3. Save the Results ---
    print("\nStep 3: Saving raw results to disk...")
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "correlation_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"Successfully saved results to '{output_path}'")
    
    # --- 4. Post-Experiment Summary ---
    validate_and_summarize_results(results_df)

    print("\n--- Correlated Efforts Experiment Complete ---")


if __name__ == "__main__":
    main()
