# src/simulation_utils.py

"""
Provides utility functions for the simulation's core mechanics.

This module contains the mathematical functions that define the stochastic
outcomes of the agent's actions, such as the probability of achieving a
high performance signal or a good safety outcome.
"""

import numpy as np

def prob_high_performance_signal(e_p: float) -> float:
    """
    Calculates the probability of a high performance signal given performance effort.

    This function represents f(e_p) from the theoretical model. It is a
    monotonically increasing function with diminishing returns.

    Args:
        e_p: The performance effort exerted by the agent.

    Returns:
        The probability (between 0.0 and 1.0) of achieving a high performance signal.
    """
    # Parameter validation
    assert e_p >= 0, "Performance effort (e_p) cannot be negative."

    # Functional form: 1 - e^(-e_p)
    return 1.0 - np.exp(-e_p)


def prob_good_safety_outcome(e_s: float) -> float:
    """
    Calculates the probability of a good safety outcome given safety effort.

    This function represents 1 - p(e_s) from the theoretical model. It is a
    monotonically increasing function with diminishing returns.

    Args:
        e_s: The safety effort exerted by the agent.

    Returns:
        The probability (between 0.0 and 1.0) of achieving a good safety outcome.
    """
    # Parameter validation
    assert e_s >= 0, "Safety effort (e_s) cannot be negative."

    # Functional form: 1 - e^(-e_s)
    prob_bad = np.exp(-e_s)
    return 1.0 - prob_bad