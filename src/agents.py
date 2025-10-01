# src/agents.py

"""
Defines the Agent classes for the environmental AI safety simulation.

This module contains:
- RegulatorAgent: A simple agent representing the principal.
- DeveloperAgent: The main agent representing the self-interested AI developer.
  This agent solves an optimization problem to choose its efforts.
"""
import mesa
import numpy as np
from scipy.optimize import minimize
from typing import Dict

# Import the Contract type definition from our contracts module
from src.contracts import Contract


# --- Helper functions for probability calculations ---
# These functions represent p(e_s) and f(e_p) from the model.

def prob_good_safety_outcome(e_s: float) -> float:
    """
    Calculates Pr(q='G' | e_s) = 1 - p(e_s).
    
    This function is type-independent to isolate the effect of theta on cost.
    """
    prob_bad = np.exp(-e_s)
    return 1.0 - prob_bad

def prob_high_performance_signal(e_p: float) -> float:
    """
    Calculates Pr(π='H' | e_p) = f(e_p).
    
    A simple functional form where effort increases the probability of a
    high signal, with diminishing returns.
    """
    return 1.0 - np.exp(-e_p)


class RegulatorAgent(mesa.Agent):
    """An agent representing the public regulator (Principal)."""
    
    def __init__(self, unique_id: int, model: mesa.Model, contract_menu: Dict[str, Contract]):
        """
        Create a new Regulator agent.
        
        Args:
            unique_id: The agent's unique identifier.
            model: The model instance the agent belongs to.
            contract_menu: The menu of contracts to offer.
        """
        super().__init__(unique_id, model)
        self.contract_menu = contract_menu

    def step(self):
        """The regulator's action in a step (passive in this model)."""
        pass


class DeveloperAgent(mesa.Agent):
    """An agent representing the AI developer (Agent)."""

    def __init__(self, unique_id: int, model: mesa.Model, theta: float, behavior_mode: str = 'rational'):
        """
        Create a new Developer agent.

        Args:
            unique_id: The agent's unique identifier.
            model: The model instance the agent belongs to.
            theta: The agent's private type (AI quality).
            behavior_mode: The decision-making mode ('rational' or 'learning').
        """
        super().__init__(unique_id, model)
        
        # --- Parameter Validation ---
        assert theta > 0, "Theta must be positive."
        assert behavior_mode in ['rational', 'learning'], "Behavior mode is not recognized."
        
        # --- Agent's Private Attributes ---
        self.theta = theta
        self.behavior_mode = behavior_mode

        # --- State variables to be recorded by the DataCollector ---
        self.contract_offer: Contract = {}
        self.chosen_ep: float = 0.0
        self.chosen_es: float = 0.0
        self.expected_utility: float = 0.0
        self.payoff: float = 0.0

    def _calculate_expected_utility(self, efforts: np.ndarray, contract: Contract, theta: float) -> float:
        """Calculates the agent's expected utility for a given effort pair."""
        e_p, e_s = efforts
        
        # Cost of effort (original version, type-independent performance cost)
        cost = (e_p**2 / 2.0) + (e_s**2 / (2.0 * theta))
        
        # Probabilities of outcomes
        p_pi_H = prob_high_performance_signal(e_p)
        p_q_G = prob_good_safety_outcome(e_s)

        # Expected payment from the contract lottery
        expected_payment = (
            p_q_G * p_pi_H * contract['G']['H'] +          # Good safety, High perf
            p_q_G * (1 - p_pi_H) * contract['G']['L'] +    # Good safety, Low perf
            (1 - p_q_G) * p_pi_H * contract['B']['H'] +    # Bad safety, High perf
            (1 - p_q_G) * (1 - p_pi_H) * contract['B']['L']  # Bad safety, Low perf
        )
        
        return expected_payment - cost

    def _choose_optimal_efforts(self):
        """Solves the agent's optimization problem to find the best efforts."""
        
        # The function to minimize is the *negative* of the expected utility
        objective_func = lambda e: -self._calculate_expected_utility(e, self.contract_offer, self.theta)
        
        # Initial guess for the optimizer
        initial_guess = np.array([1.0, 1.0])
        
        # Efforts must be non-negative
        bounds = [(0, None), (0, None)]
        
        # Run the optimization
        result = minimize(objective_func, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.chosen_ep, self.chosen_es = result.x
            self.expected_utility = -result.fun
        else:
            # If optimizer fails, default to zero effort
            self.chosen_ep, self.chosen_es = 0.0, 0.0
            self.expected_utility = -objective_func(np.array([0.0, 0.0]))


    def step(self):
        """
        Executes the developer's decision-making process for one step.
        
        The model's step() function will first call a method to set the
        `self.contract_offer` before this agent's step is activated.
        """
        if self.behavior_mode == 'rational':
            self._choose_optimal_efforts()
        
        elif self.behavior_mode == 'learning':
            # --- TODO: Implement Q-learning or other reinforcement learning logic ---
            # The agent would select an action from its action space,
            # observe the payoff, and update its Q-table.
            self._choose_optimal_efforts() # Placeholder for learning behavior