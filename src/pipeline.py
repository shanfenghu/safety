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
    iterations: int,
    parameters: Optional[Dict[str, Any]] = None,
    parameter_sets: Optional[List[Dict[str, Any]]] = None,
    number_processes: Optional[int] = 1
) -> pd.DataFrame:
    """
    Executes a batch run of the SafetyModel using a custom loop.

    This function can be called in two ways:
    1. With a `parameters` dictionary (for automatic expansion).
    2. With an explicit `parameter_sets` list (for pre-calculated configurations).
    """
    # 1. Determine the final list of parameter sets to run
    if parameter_sets is not None:
        final_parameter_sets = parameter_sets
    elif parameters is not None:
        final_parameter_sets = _expand_parameters(parameters)
    else:
        raise ValueError("Either 'parameters' or 'parameter_sets' must be provided.")

    total_runs = len(final_parameter_sets) * iterations
    print(f"Starting custom batch run: {len(final_parameter_sets)} configurations, "
          f"{iterations} iterations each. Total runs: {total_runs}")

    # 2. Main Simulation Loop
    all_run_data = []
    run_id_counter = 0

    run_iterator = product(final_parameter_sets, range(iterations))
    if HAS_TQDM:
        run_iterator = tqdm(run_iterator, total=total_runs)

    for params, i in run_iterator:
        model = SafetyModel(params=params)
        model.step()
        
        model_data = model.datacollector.get_model_vars_dataframe().iloc[-1]
        agent_data = model.datacollector.get_agent_vars_dataframe().iloc[-1]
        
        run_data = {**model_data, **agent_data, 'RunId': run_id_counter, 'iteration': i, **params}
        all_run_data.append(run_data)
        run_id_counter += 1

    # 3. Final Data Aggregation
    results_df = pd.DataFrame(all_run_data)
    print(f"\nBatch run complete. Collected {len(results_df)} total runs.")
    
    return results_df

