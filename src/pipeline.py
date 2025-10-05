# src/pipeline.py

"""
A custom, transparent pipeline for running simulations.

This module provides a custom `run` function that handles parameter expansion
and data aggregation. It is designed to be flexible, supporting both single-shot
and multi-step (e.g., learning) simulations.
"""

import pandas as pd
from itertools import product
from typing import Dict, List, Optional, Any

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
    """
    variable_params = {k: v for k, v in parameters.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in parameters.items() if not isinstance(v, list)}

    if not variable_params:
        return [fixed_params]

    keys, values = zip(*variable_params.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return [{**fixed_params, **combo} for combo in param_combinations]


def run(
    parameters: Dict[str, Any],
    iterations: int,
    number_processes: Optional[int] = 1
) -> pd.DataFrame:
    """
    Executes a batch run of the SafetyModel using a custom loop and returns the results.
    """
    if not isinstance(parameters, dict):
        raise TypeError("`parameters` must be a dictionary.")
    if number_processes is not None and number_processes > 1:
        print("Warning: Custom pipeline does not support multiprocessing. Running in a single process.")

    parameter_sets = _expand_parameters(parameters)
    total_runs = len(parameter_sets) * iterations
    print(f"Starting custom batch run: {len(parameter_sets)} configurations, "
          f"{iterations} iterations each. Total runs: {total_runs}")

    all_run_data = []
    all_qtables = []
    run_id_counter = 0

    run_iterator = product(parameter_sets, range(iterations))
    if HAS_TQDM:
        run_iterator = tqdm(run_iterator, total=total_runs)

    for params, i in run_iterator:
        model = SafetyModel(params=params)
        
        # Run the model for the specified number of steps
        max_steps = params.get('max_steps', 1)
        for _ in range(max_steps):
            if not model.running:
                break
            model.step()
        
        # Collect data for this run
        run_model_data = model.datacollector.get_model_vars_dataframe()
        run_agent_data = model.datacollector.get_agent_vars_dataframe()
        
        # Always use reset_index() to ensure 'Step' and 'AgentID' become columns
        run_agent_data.reset_index(inplace=True)
        run_model_data.reset_index(inplace=True)

        # Filter to developer-only rows to avoid missing reporter fields on other agents
        if isinstance(run_agent_data, pd.DataFrame) and 'AgentID' in run_agent_data.columns:
            run_agent_data = run_agent_data[run_agent_data['AgentID'] == model.developer.unique_id]

        # Robustly ensure the step column is named 'Step' for both frames
        # Mesa may produce an unnamed index -> reset_index will call it 'index'
        for df in (run_agent_data, run_model_data):
            if 'Step' not in df.columns:
                if 'index' in df.columns:
                    df.rename(columns={'index': 'Step'}, inplace=True)
                elif 'step' in df.columns:
                    df.rename(columns={'step': 'Step'}, inplace=True)
        
        # Normalize dtype of 'Step' and drop duplicate index/column ambiguity
        if 'Step' in run_agent_data.columns:
            run_agent_data['Step'] = pd.to_numeric(run_agent_data['Step'], errors='coerce').astype('Int64')
        if 'Step' in run_model_data.columns:
            run_model_data['Step'] = pd.to_numeric(run_model_data['Step'], errors='coerce').astype('Int64')
        # If Step also exists as an index level (rare after reset_index), drop index to avoid ambiguity
        if isinstance(run_agent_data.index, pd.MultiIndex) or 'Step' in getattr(run_agent_data.index, 'names', []):
            run_agent_data = run_agent_data.reset_index(drop=True)
        if isinstance(run_model_data.index, pd.MultiIndex) or 'Step' in getattr(run_model_data.index, 'names', []):
            run_model_data = run_model_data.reset_index(drop=True)
        
        # Add metadata
        run_agent_data['RunId'] = run_id_counter
        run_agent_data['iteration'] = i
        for param_key, param_val in params.items():
            # Avoid adding list parameters to the final dataframe
            if not isinstance(param_val, list):
                run_agent_data[param_key] = param_val
        
        # Merge model-level and agent-level data based on the 'Step'
        full_run_data = pd.merge(run_agent_data, run_model_data, on='Step')
        # If merge unexpectedly produces zero rows (e.g., off-by-one step indexing), try alignment fallbacks
        if full_run_data.empty and not run_agent_data.empty and not run_model_data.empty and 'Step' in run_agent_data.columns and 'Step' in run_model_data.columns:
            # Try shifting agent steps by -1 then +1 and pick the merge with more rows
            best_merge = full_run_data
            best_len = 0
            for shift in (-1, 1):
                shifted = run_agent_data.copy()
                shifted['Step'] = shifted['Step'] + shift
                candidate = pd.merge(shifted, run_model_data, on='Step')
                if len(candidate) > best_len:
                    best_len = len(candidate)
                    best_merge = candidate
            if best_len > 0:
                full_run_data = best_merge
        
        all_run_data.append(full_run_data)

        # Capture the learned Q-table at the end of the run (one row per action)
        if hasattr(model.developer, 'q_table') and hasattr(model.developer, 'action_space'):
            q_rows = []
            for action_index, (ep, es) in enumerate(model.developer.action_space):
                q_rows.append({
                    'RunId': run_id_counter,
                    'iteration': i,
                    'contract_type': params.get('contract_type', None),
                    'ActionIndex': action_index,
                    'Ep': ep,
                    'Es': es,
                    'QValue': float(model.developer.q_table[action_index])
                })
            all_qtables.append(pd.DataFrame(q_rows))
        run_id_counter += 1

    # Final Data Aggregation
    results_df = pd.concat(all_run_data, ignore_index=True)
    learned_q_df = pd.concat(all_qtables, ignore_index=True) if all_qtables else pd.DataFrame()
    print(f"\nBatch run complete. Collected {len(results_df)} total data points and {len(learned_q_df)} learned Q rows.")
    
    # Return both the main results and the learned Q-tables
    return results_df, learned_q_df