# src/agents/regulator.py

"""
Defines the RegulatorAgent class for the simulation.
"""

import mesa
from typing import Dict

from src.types import Contract


class RegulatorAgent(mesa.Agent):
    """
    An agent representing the public regulator (the Principal).

    In this model, the regulator is a passive agent whose main role is to
    hold the contract menu that is offered to the developer. The logic for
    creating the menu and managing the game is handled by the `SafetyModel`.
    """
    def __init__(self, unique_id: int, model: mesa.Model, contract_menu: Dict[str, Contract]):
        """
        Creates a new Regulator agent.
        
        Args:
            unique_id: The agent's unique identifier.
            model: The model instance the agent belongs to.
            contract_menu: The menu of contracts to offer to the developer types.
        """
        super().__init__(unique_id, model)
        self.contract_menu = contract_menu

    def step(self):
        """
        The regulator's action in a step. This agent is passive during the
        main simulation step, so this method does nothing.
        """
        pass