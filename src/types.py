# src/types.py

"""
Defines shared, project-wide type aliases for static analysis.

This module serves as the single source of truth for custom data structures,
preventing circular dependencies between other modules.
"""

from typing import Dict

# A Contract is represented as a nested dictionary mapping the safety outcome (str)
# and the performance outcome (str) to a payment (float).
#
# Example: contract['G']['H'] is the payment for a Good safety outcome
#          and a High performance signal.
Contract = Dict[str, Dict[str, float]]