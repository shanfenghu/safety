# experiments/exp_main_comparison.py

"""
Runs the main experimental comparison for the paper.

This script executes the core experiment, which compares the performance of the
four main contract types. After running the simulations, it performs automated
validation checks and prints a summary of the results to the console.
"""

import pandas as pd

# Import the centralized configuration and the experiment pipeline
import config
from src.pipeline import run


def validate_and_summarize_results(df: pd.DataFrame):
    """
    Performs post-experiment validation checks and prints a results summary.

    Args:
        df: The DataFrame containing the raw experimental results.
    """
    print("\n--- Post-Experiment Validation and Summary ---")

    # --- 1. Generate Summary Statistics ---
    print("Step 1: Generating results summary...")
    
    # Group data for analysis
    summary = df.groupby('contract_type').agg(
        mean_welfare=('SocialWelfare', 'mean'),
        std_welfare=('SocialWelfare', 'std'),
        disaster_freq=('DisasterOccurred', 'mean'),
        mean_es=('ChosenEs', 'mean'),
        std_es=('ChosenEs', 'std'),
        mean_ep=('ChosenEp', 'mean'),
        std_ep=('ChosenEp', 'std')
    ).reset_index()

    # --- 2. Print Full Summary Table ---
    # We print the summary first to aid in debugging.
    summary_to_print = summary.copy()
    summary_to_print['disaster_freq'] = summary_to_print['disaster_freq'].apply(lambda x: f"{x:.2%}")
    summary_to_print = summary_to_print.round(3)
    summary_to_print.set_index('contract_type', inplace=True)
    
    print("\n" + "="*80)
    print("                 MAIN EXPERIMENT: FULL SUMMARY OF RESULTS")
    print("="*80)
    print(summary_to_print)
    print("="*80 + "\n")

    # --- 3. Automated Validation Checks (as warnings for debugging) ---
    print("Step 2: Running validation checks (as warnings)...")
    
    # Sanity Checks
    assert not df.empty, "Results DataFrame is empty."
    print("  - Sanity check: DataFrame is not empty. Passed.")
    
    # Theoretical Consistency Checks
    try:
        if not summary['mean_welfare'].idxmax() == summary[summary['contract_type'] == 'optimal'].index[0]:
            best_contract = summary.loc[summary['mean_welfare'].idxmax()]['contract_type']
            print(f"  - WARNING: Optimal contract did not yield the highest mean social welfare. Best was '{best_contract}'.")
        else:
            print("  - Validation: Optimal contract yielded the highest mean social welfare. Passed.")
            
        if not summary['disaster_freq'].idxmax() == summary[summary['contract_type'] == 'performance'].index[0]:
            worst_contract = summary.loc[summary['disaster_freq'].idxmax()]['contract_type']
            print(f"  - WARNING: Performance contract did not yield the highest disaster frequency. Worst was '{worst_contract}'.")
        else:
            print("  - Validation: Performance contract yielded the highest disaster frequency. Passed.")

    except IndexError:
        print("  - ERROR: Could not perform all validation checks. One or more contract types may be missing from results.")


def main():
    """Defines and runs the main experimental comparison."""
    print("--- Starting Main Experimental Comparison ---")

    # --- 1. Define Experimental Conditions ---
    print("Step 1: Defining parameter set for the experiment...")
    
    params = config.BASELINE_PARAMS.copy()
    params.update(config.AGENT_CONFIGS['rational'])
    params['contract_type'] = list(config.CONTRACT_CONFIGS.keys())
    
    # --- 2. Execute the Simulation Pipeline ---
    print("\nStep 2: Executing simulation pipeline...")
    
    results_df, _ = run(
        parameters=params,
        iterations=config.ITERATIONS * 10,
        number_processes=config.NUM_PROCESSES
    )
    
    # --- 3. Print Granular Log for First Few Runs ---
    print("\n--- Granular Log of First 5 Runs per Contract Type ---")
    for contract_type in results_df['contract_type'].unique():
        print(f"\nContract: {contract_type}")
        print("-" * 30)
        subset = results_df[results_df['contract_type'] == contract_type].head(5)
        print(subset[['RunId', 'Theta', 'ChosenEp', 'ChosenEs', 'SocialWelfare']].round(2))
    
    # --- 4. Save the Results ---
    print("\nStep 3: Saving raw results to disk...")
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "main_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Successfully saved results to '{output_path}'")
    
    # --- 5. Post-Experiment Validation ---
    validate_and_summarize_results(results_df)

    print("\n--- Main Experimental Comparison Complete ---")


if __name__ == "__main__":
    main()

