# src/contracts.py

"""
Defines the structure and creation functions for regulatory contracts.

This module provides functions to generate the payment schedules for the different
contract types used in the simulation:
- The theoretically Optimal Contract (C_OPT)
- The Naive Fine heuristic (C_FINE)
- The Performance-Based heuristic (C_PERF)
- The Hybrid heuristic (C_HYB)

All contracts adhere to the Limited Liability constraint (payments >= 0).
"""

from typing import Dict

# A Contract is represented as a nested dictionary: {outcome_q: {outcome_pi: payment}}
# e.g., contract['G']['H'] is the payment for a Good safety outcome and High performance.
Contract = Dict[str, Dict[str, float]]


def create_naive_fine_contract(safety_bonus: float) -> Contract:
    """
    Creates a contract that only rewards a good safety outcome.

    This represents a simple liability rule, where a "fine" for a bad outcome
    is modeled as the absence of a large bonus for a good outcome.

    Args:
        safety_bonus: The payment received if the safety outcome is 'G'.

    Returns:
        A contract dictionary representing the payment schedule.
    """
    # Parameter validation
    assert safety_bonus >= 0, "Safety bonus must be non-negative."

    contract: Contract = {
        'G': {'H': safety_bonus, 'L': safety_bonus}, # Payment is `safety_bonus` if q='G'
        'B': {'H': 0.0, 'L': 0.0}                    # Payment is 0 if q='B'
    }
    return contract


def create_performance_contract(performance_bonus: float) -> Contract:
    """
    Creates a contract that only rewards a high performance signal.

    This represents a naive regulator who incentivizes what is easy to measure.

    Args:
        performance_bonus: The payment received if the performance signal is 'H'.

    Returns:
        A contract dictionary representing the payment schedule.
    """
    # Parameter validation
    assert performance_bonus >= 0, "Performance bonus must be non-negative."

    contract: Contract = {
        'G': {'H': performance_bonus, 'L': 0.0},
        'B': {'H': performance_bonus, 'L': 0.0}
    }
    return contract


def create_hybrid_contract(safety_bonus: float, performance_bonus: float) -> Contract:
    """
    Creates a contract that combines a safety bonus and a performance bonus.

    This represents a more sophisticated but still theoretically naive heuristic that
    rewards both observable outcomes.

    Args:
        safety_bonus: The base payment for a good safety outcome.
        performance_bonus: The additional payment for a high performance signal.

    Returns:
        A contract dictionary representing the payment schedule.
    """
    # Parameter validation
    assert safety_bonus >= 0, "Safety bonus must be non-negative."
    assert performance_bonus >= 0, "Performance bonus must be non-negative."

    contract: Contract = {
        'G': {'H': safety_bonus + performance_bonus, 'L': safety_bonus},
        'B': {'H': performance_bonus,               'L': 0.0}
    }
    return contract


def calculate_optimal_contract(params: dict) -> Dict[str, Contract]:
    """
    Calculates the menu for the theoretically optimal second-best contract.

    This function solves the principal's problem as defined in the paper's
    theoretical analysis to find the set of payments that optimally balances
    incentives, rent extraction, and participation under all constraints.

    NOTE: This is a placeholder implementation. The actual implementation
    requires solving the system of equations from the theoretical proofs.

    Args:
        params: A dictionary of model parameters (e.g., thetas, nu, delta_W).

    Returns:
        A dictionary containing two contracts, one for the high-quality type
        and one for the low-quality type.
    """
    # Parameter validation
    required_keys = ['theta_L', 'theta_H', 'nu', 'delta_W']
    for key in required_keys:
        assert key in params, f"Required parameter '{key}' is missing."

    # --- TODO: Implement the full analytical solution from the theory ---
    # The code here would take the parameters and solve the FOCs from the
    # appendix to derive the optimal effort levels and then the required
    # payment spreads to implement them, respecting all binding constraints
    # (IR-L, IC-H, and Limited Liability).
    #
    # For now, we return pre-calculated, illustrative values.
    
    # Illustrative payments for the High-Quality Type (θ_H)
    contract_H: Contract = {
        'G': {'H': 30.0, 'L': 15.0},
        'B': {'H': 10.0, 'L': 0.0}
    }

    # Illustrative payments for the Low-Quality Type (θ_L)
    contract_L: Contract = {
        'G': {'H': 20.0, 'L': 10.0},
        'B': {'H': 5.0,  'L': 0.0}
    }

    return {'H': contract_H, 'L': contract_L}