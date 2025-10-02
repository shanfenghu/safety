# src/pipeline.py

"""
A reusable pipeline for running Mesa batch simulations.

This module abstracts the common logic for setting up and executing a Mesa
batch run, processing the results into a pandas DataFrame, and providing
user-friendly feedback like progress bars. It includes a wrapper to ensure
compatibility between the model's constructor and Mesa's batch runner.
"""

import mesa
import pandas as pd
from typing import Dict, Any, Optional

# tqdm is used for progress bars. It is an optional dependency.
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import the original model class that the pipeline will run
from src.model import SafetyModel as OriginalSafetyModel

class _SafetyModelWrapper(OriginalSafetyModel):
    """
    An internal wrapper to make our SafetyModel compatible with mesa.batch_run.
    
    Mesa's batch_run unpacks parameters as keyword arguments, but our model's
    __init__ expects a single dictionary. This wrapper class bridges that gap
    by accepting keyword arguments and passing them as a single dictionary to
    the parent class constructor.
    """
    def __init__(self, **kwargs):
        """
        Initializes the wrapper.
        
        Args:
            **kwargs: Keyword arguments that will be collected into a single
                      dictionary and passed to the parent model's constructor.
        """
        super().__init__(params=kwargs)


def run(
    parameters: Dict[str, Any],
    iterations: int,
    number_processes: Optional[int] = 1
) -> pd.DataFrame:
    """
    Executes a batch run of the SafetyModel and returns the results.

    This function serves as a standardized, robust pipeline for all experiments.

    Args:
        parameters: A dictionary mapping parameter names to either a single
                    value (for fixed parameters) or a list of values (for
                    variable parameters to be iterated over).
        iterations: The number of times to run the simulation for each
                    unique combination of variable parameters.
        number_processes: The number of processes to use for parallel execution.
                          Defaults to 1 (no parallelization). Set to None to use
                          all available CPU cores.

    Returns:
        A pandas DataFrame containing the collected data from all simulation runs.
    """
    # 1. Input Validation
    if not isinstance(parameters, dict):
        raise TypeError("`parameters` must be a dictionary.")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("`iterations` must be a positive integer.")
    if number_processes is not None and (not isinstance(number_processes, int) or number_processes < 1):
        raise ValueError("`number_processes` must be a positive integer or None.")

    # 2. Execute the Batch Run using Mesa
    print(f"Starting batch run with {iterations} iterations per parameter combination...")

    raw_results = mesa.batch_run(
        model_cls=_SafetyModelWrapper, # Use the wrapper to ensure compatibility
        parameters=parameters,
        iterations=iterations,
        number_processes=number_processes,
        data_collection_period=-1, # Only collect data at the end of each run
        display_progress=HAS_TQDM
    )

    # 3. Process and Return Results
    results_df = pd.DataFrame(raw_results)
    print(f"Batch run complete. Collected {len(results_df)} total runs.")
    
    return results_df