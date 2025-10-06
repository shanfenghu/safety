# experiments/exp_learning_robustness.py

"""
Runs the learning agent experiment to test for robustness.

This script executes the main comparison with the `LearningDeveloperAgent` to
test whether the incentive structures of the contracts are strong enough to
guide a boundedly rational agent towards the desired behavior.

The model is run for many steps to allow the agent's Q-learning algorithm
to converge. The script saves the full, step-by-step data, which is needed
to generate the learning curves and final policy heatmaps.
"""

import pandas as pd

# Import the centralized configuration and the experiment pipeline
import config
from src.pipeline import run


def validate_and_summarize_results(df: pd.DataFrame, final_phase_fraction: float = 0.1):
    """
    Performs post-experiment validation and prints a summary of the agent's
    final, learned performance.

    Args:
        df: The DataFrame containing the step-by-step experimental results.
        final_phase_fraction: The fraction of final steps to analyze for
                              converged performance (e.g., last 10%).
    """
    print("\n--- Learning Experiment: Validation and Summary ---")

    # --- 1. Sanity Checks ---
    print("Step 1: Running validation checks...")
    assert not df.empty, "Results DataFrame is empty."
    
    # Check for the presence of step-level data
    assert 'Step' in df.columns, "DataFrame is missing 'Step' column for learning analysis."
    print("  - Sanity checks passed.")

    # --- 2. Analyze Converged Performance ---
    print(f"\nStep 2: Analyzing final {final_phase_fraction:.0%} of learning period...")
    
    max_steps = df['Step'].max()
    final_phase_start_step = max_steps * (1 - final_phase_fraction)
    
    # Filter for the data from the final phase of the simulation
    final_phase_df = df[df['Step'] >= final_phase_start_step]
    
    # Group by contract type and calculate final performance metrics
    summary = final_phase_df.groupby('contract_type').agg(
        mean_welfare=('SocialWelfare', 'mean'),
        disaster_freq=('DisasterOccurred', 'mean'),
        mean_es=('ChosenEs', 'mean'),
        mean_ep=('ChosenEp', 'mean')
    ).reset_index()

    # --- 3. Print Summary Table ---
    summary_to_print = summary.copy()
    summary_to_print['disaster_freq'] = summary_to_print['disaster_freq'].apply(lambda x: f"{x:.2%}")
    summary_to_print = summary_to_print.round(3)
    summary_to_print.set_index('contract_type', inplace=True)
    
    print("\n" + "="*60)
    print("        LEARNING EXPERIMENT: CONVERGED PERFORMANCE")
    print("="*60)
    print(summary_to_print)
    print("="*60 + "\n")

def main():
    """Defines and runs the learning agent experiment."""
    print("--- Starting Learning Agent Robustness Experiment ---")

    # --- 1. Define Experimental Conditions ---
    print("Step 1: Defining parameter set for the experiment...")
    
    params = config.BASELINE_PARAMS.copy()
    
    # Use the 'learning' agent configuration from the config file
    params.update(config.AGENT_CONFIGS['learning'])
    
    # We will vary the contract type across the four main scenarios
    params['contract_type'] = list(config.CONTRACT_CONFIGS.keys())
    
    # Set the number of steps for the learning process
    params['max_steps'] = 5000
    
    print(f"  - Agent type set to: {params['developer_class'].__name__}")
    print(f"  - Simulation will run for {params['max_steps']} steps.")
    
    # --- 2. Execute the Simulation Pipeline ---
    print("\nStep 2: Executing simulation pipeline...")
    # For a learning experiment, each "iteration" is a full learning history.
    # We run fewer iterations as each one is computationally intensive.
    iterations = config.ITERATIONS * 10
    print(f"Running {iterations} full learning histories for each of the {len(params['contract_type'])} configurations.")
    
    results_df, learned_q_df = run(
        parameters=params,
        iterations=iterations,
        number_processes=config.NUM_PROCESSES
    )
    
    # --- 3. Save the Results ---
    print("\nStep 3: Saving raw, step-by-step results and learned Q-tables to disk...")
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "learning_robustness.csv"
    qtable_path = config.RESULTS_DIR / "learning_qtables.csv"
    results_df.to_csv(output_path, index=False)
    learned_q_df.to_csv(qtable_path, index=False)
    
    print(f"Successfully saved results to '{output_path}' and learned Q tables to '{qtable_path}'")
    
    # --- 4. Post-Experiment Validation and Summary ---
    validate_and_summarize_results(results_df)

    print("\n--- Learning Agent Experiment Complete ---")


if __name__ == "__main__":
    main()
