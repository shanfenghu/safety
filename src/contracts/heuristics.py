# src/contracts/heuristics.py

"""
Defines the simple, heuristic contract structures for the simulation.

These functions create contracts that represent intuitive but theoretically
sub-optimal regulatory approaches, such as simple fines or performance-based
rewards. They serve as benchmarks against which the optimal contract is compared.
"""

from src.types import Contract


def create_naive_fine_contract(safety_bonus: float) -> Contract:
    """
    Creates a contract that only rewards a good safety outcome.

    This represents a simple liability rule, where a "fine" for a bad outcome
    is modeled as the absence of a large bonus for a good outcome, respecting
    the limited liability (t >= 0) constraint.

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