# experiments/exp_risk_aversion.py

"""
Runs the risk aversion robustness experiment.

This script compares contract types across two agent types
(risk-neutral vs risk-averse) and saves the raw results to CSV,
then prints a compact summary table for quick validation.
"""

import pandas as pd

# Import the centralized configuration and the experiment pipeline
import config
from src.pipeline import run
from src.agents.rational import RationalDeveloperAgent
from src.agents.risk_averse import RiskAverseDeveloperAgent

def validate_and_summarize_results(df: pd.DataFrame):
    """Prints a pivoted summary comparing agents and contracts."""
    print("\n--- Risk Aversion Experiment: Summary ---")

    assert not df.empty, "Results DataFrame is empty."

    # Group by agent type (class name) and contract type
    df = df.copy()
    
    # Use the class name for easier reading in the output
    df['developer_class'] = df['developer_class'].apply(lambda c: c.__name__)

    summary = df.groupby(['developer_class', 'contract_type']).agg(
        mean_welfare=('SocialWelfare', 'mean'),
        mean_es=('ChosenEs', 'mean')
    ).reset_index()

    # Create two pivot tables for clear comparison
    welfare_pivot = summary.pivot(index='contract_type', columns='developer_class', values='mean_welfare')
    es_pivot = summary.pivot(index='contract_type', columns='developer_class', values='mean_es')
    
    # Define a consistent order for the summary table
    contract_order = ['optimal', 'optimal_ra', 'fine', 'hybrid', 'performance']
    welfare_pivot = welfare_pivot.reindex(contract_order)
    es_pivot = es_pivot.reindex(contract_order)

    print("\n--- Mean Social Welfare ---")
    print(welfare_pivot.round(2))
    print("\n--- Mean Safety Effort (e_s) ---")
    print(es_pivot.round(3))
    print("-" * 30)


def main():
    """Defines and runs the risk aversion experiment."""
    
    # 1. Define Experimental Conditions
    print("Step 1: Defining parameter set for the experiment...")

    params = config.BASELINE_PARAMS.copy()

    # Two agent types to compare
    params['developer_class'] = [RationalDeveloperAgent, RiskAverseDeveloperAgent]

    # Add the new 'optimal_ra' contract to the list of contracts to be tested
    params['contract_type'] = list(config.CONTRACT_CONFIGS.keys()) + ['optimal_ra']

    print(f"  - developer_class set to: {[c.__name__ for c in params['developer_class']]}")
    print(f"  - contract_type set to: {params['contract_type']}")

    # 2. Execute the Simulation Pipeline
    print("\nStep 2: Executing simulation pipeline...")
    # The `run` function in pipeline.py will now correctly call the augmented solver
    # based on the (contract_type, developer_class) combination.
    results_df, _ = run(
        parameters=params,
        iterations=config.ITERATIONS * 10,
        number_processes=config.NUM_PROCESSES
    )

    # 3. Save Results
    print("\nStep 3: Saving raw results to disk...")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / "risk_aversion_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Successfully saved results to '{output_path}'")
    
    # 4. Post-Experiment Validation and Summary
    validate_and_summarize_results(results_df)

    print("\n--- Risk Aversion Experiment Complete ---")


if __name__ == "__main__":
    main()