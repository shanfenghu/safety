# src/agents/base_developer.py

"""
Defines the BaseDeveloperAgent, an abstract base class for all developer types.
"""

import mesa
from abc import ABC, abstractmethod

from src.types import Contract


class BaseDeveloperAgent(mesa.Agent, ABC):
    """
    An abstract base class for an AI developer agent.

    This class provides the common interface and shared attributes for all
    developer agent types. Subclasses must implement the `step` and `learn`
    methods.
    """

    def __init__(self, unique_id: int, model: mesa.Model, theta: float):
        """
        Creates a new BaseDeveloperAgent.

        Args:
            unique_id: The agent's unique identifier.
            model: The model instance the agent belongs to.
            theta: The agent's private type (AI quality).
        """
        super().__init__(unique_id, model)

        # Parameter validation
        assert theta > 0, "Theta must be positive."
        
        # --- Agent's Private Attributes ---
        self.theta = theta
        
        # --- State variables to be recorded by the DataCollector ---
        self.contract_offer: Contract = {}
        self.chosen_ep: float = 0.0
        self.chosen_es: float = 0.0
        self.payoff: float = 0.0

    @abstractmethod
    def step(self):
        """
        The agent's main decision-making method for a single simulation step.
        
        Subclasses must implement this method to define their behavior for
        choosing effort levels.
        """
        raise NotImplementedError("Subclasses must implement the 'step' method.")

    @abstractmethod
    def learn(self, reward: float, action_index: int):
        """
        The agent's learning method, called after a step is resolved.

        Subclasses that are not learning agents can simply pass.
        
        Args:
            reward: The payoff received from the last action.
            action_index: The index of the action that was taken.
        """
        raise NotImplementedError("Subclasses must implement the 'learn' method.")