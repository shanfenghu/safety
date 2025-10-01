# src/pipeline.py

"""
A reusable pipeline for running Mesa batch simulations.

This module abstracts the common logic for setting up and executing a Mesa
batch run, processing the results into a pandas DataFrame, and providing
user-friendly feedback like progress bars.
"""

import mesa
import pandas as pd
from typing import List, Dict, Optional

# tqdm is used for progress bars. It is an optional dependency.
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import the model class that the pipeline will run
from src.model import SafetyModel


def run(parameter_sets: List[Dict], iterations: int, number_processes: Optional[int] = 1) -> pd.DataFrame:
    """
    Executes a batch run of the SafetyModel and returns the results.

    This function serves as a standardized, robust pipeline for all experiments.

    Args:
        parameter_sets: A list of parameter dictionaries. Each dictionary
                        defines a specific configuration of the SafetyModel to run.
        iterations: The number of times to run the simulation for each
                    unique parameter set.
        number_processes: The number of processes to use for parallel execution.
                          Defaults to 1 (no parallelization). Set to None to use
                          all available CPU cores.

    Returns:
        A pandas DataFrame containing the collected data from all simulation runs.
    """
    # 1. Input Validation
    if not isinstance(parameter_sets, list) or not parameter_sets:
        raise ValueError("`parameter_sets` must be a non-empty list of dictionaries.")
    if not all(isinstance(p, dict) for p in parameter_sets):
        raise TypeError("All items in `parameter_sets` must be dictionaries.")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("`iterations` must be a positive integer.")
    if number_processes is not None and (not isinstance(number_processes, int) or number_processes < 1):
        raise ValueError("`number_processes` must be a positive integer or None.")

    # 2. Execute the Batch Run using Mesa
    print(f"Starting batch run: {len(parameter_sets)} configurations, "
          f"{iterations} iterations each...")

    raw_results = mesa.batch_run(
        model_cls=SafetyModel,
        parameters=parameter_sets,
        iterations=iterations,
        number_processes=number_processes,
        data_collection_period=-1, # Only collect data at the end of each run
        display_progress=HAS_TQDM
    )

    # 3. Process and Return Results
    results_df = pd.DataFrame(raw_results)
    print(f"Batch run complete. Collected {len(results_df)} total runs.")
    
    return results_df