# src/pipeline.py

"""
A custom, transparent pipeline for running simulations.

This module provides a custom `run` function that replaces Mesa's `batch_run`.
It manually handles parameter expansion and data aggregation, offering greater
transparency and debuggability.
"""

import pandas as pd
from itertools import product
from typing import Dict, List, Optional, Any

# tqdm is used for progress bars. It is an optional dependency.
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.model import SafetyModel


def _expand_parameters(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expands a Mesa-style parameters dictionary into a list of all unique
    parameter combinations.

    This is an internal helper function.

    Args:
        parameters: A dictionary where keys are parameter names and values are
                    either single values (fixed) or lists of values (variable).

    Returns:
        A list of dictionaries, where each dictionary is a unique parameter set.
    """
    # Separate variable parameters from fixed ones
    variable_params = {k: v for k, v in parameters.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in parameters.items() if not isinstance(v, list)}

    # Handle the edge case where there are no variable parameters
    if not variable_params:
        return [fixed_params]

    # Use itertools.product to create all combinations of variable parameters
    keys, values = zip(*variable_params.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Add the fixed parameters to each combination
    parameter_sets = []
    for combo in param_combinations:
        parameter_sets.append({**fixed_params, **combo})
        
    return parameter_sets


def run(
    parameters: Dict[str, Any],
    iterations: int,
    number_processes: Optional[int] = 1
) -> pd.DataFrame:
    """
    Executes a batch run of the SafetyModel using a custom loop and returns the results.

    This function replaces `mesa.batch_run` to provide a more transparent and
    debuggable experimental pipeline. Note: this implementation is single-threaded
    and does not support parallel processing.

    Args:
        parameters: A dictionary mapping parameter names to either a single
                    value (fixed) or a list of values (variable).
        iterations: The number of times to run the simulation for each
                    unique combination of variable parameters.
        number_processes: This argument is kept for API compatibility but is
                          not used. A warning will be issued if it's > 1.

    Returns:
        A pandas DataFrame containing the collected data from all simulation runs.
    """
    # 1. Input Validation and Warnings
    if not isinstance(parameters, dict):
        raise TypeError("`parameters` must be a dictionary.")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("`iterations` must be a positive integer.")
    if number_processes is not None and number_processes > 1:
        print("Warning: Custom pipeline does not support multiprocessing. "
              "Running in a single process.")

    # 2. Manually expand parameter sets
    parameter_sets = _expand_parameters(parameters)
    total_runs = len(parameter_sets) * iterations
    print(f"Starting custom batch run: {len(parameter_sets)} configurations, "
          f"{iterations} iterations each. Total runs: {total_runs}")

    # 3. Main Simulation Loop
    all_run_data = []
    run_id_counter = 0

    # Use tqdm for a progress bar if it's available
    run_iterator = product(parameter_sets, range(iterations))
    if HAS_TQDM:
        run_iterator = tqdm(run_iterator, total=total_runs)

    for params, i in run_iterator:
        # a. Instantiate the model with the specific parameters for this run
        model = SafetyModel(params=params)
        
        # b. Run the model for one step (as it's a one-shot game)
        model.step()
        
        # c. Collect data for this run
        # Get the last row of the model and agent dataframes
        model_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
        agent_data = model.datacollector.get_agent_vars_dataframe().iloc[-1]
        
        # d. Combine all data and add metadata
        run_data = {**model_data, **agent_data}
        run_data['RunId'] = run_id_counter
        run_data['iteration'] = i
        
        # Add the original parameters to the row for easy grouping later
        run_data.update(params)
        
        all_run_data.append(run_data)
        run_id_counter += 1

    # 4. Final Data Aggregation
    results_df = pd.DataFrame(all_run_data)
    print(f"\nBatch run complete. Collected {len(results_df)} total runs.")
    
    return results_df

