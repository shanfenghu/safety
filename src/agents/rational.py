# src/agents/rational.py

"""
Defines the RationalDeveloperAgent class for the simulation.
"""

import numpy as np
from scipy.optimize import minimize

from src.agents.base_developer import BaseDeveloperAgent
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal
from src.types import Contract


class RationalDeveloperAgent(BaseDeveloperAgent):
    """
    An agent that makes decisions by maximizing its expected utility.

    This agent has perfect knowledge of the contract and the probability
    distributions of outcomes. It uses a numerical optimizer to find the
    effort levels that maximize its payoff.
    """

    def _calculate_expected_utility(self, efforts: np.ndarray) -> float:
        """
        Calculates the agent's expected utility for a given effort pair.
        
        This is the objective function for the agent's optimization problem.

        Args:
            efforts: A NumPy array containing [e_p, e_s].

        Returns:
            The calculated expected utility.
        """
        e_p, e_s = efforts
        
        # Cost of effort using the agent's private type
        cost = (e_p**2 / 2.0) + (e_s**2 / (2.0 * self.theta))
        
        # Probabilities of outcomes based on effort
        p_pi_H = prob_high_performance_signal(e_p)
        p_q_G = prob_good_safety_outcome(e_s)

        # Expected payment from the contract lottery
        expected_payment = (
            p_q_G * p_pi_H * self.contract_offer['G']['H'] +
            p_q_G * (1 - p_pi_H) * self.contract_offer['G']['L'] +
            (1 - p_q_G) * p_pi_H * self.contract_offer['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * self.contract_offer['B']['L']
        )
        
        return expected_payment - cost

    def _choose_optimal_efforts(self):
        """Solves the agent's optimization problem to find the best efforts."""
        
        # The function for the optimizer to minimize is the *negative* of the expected utility
        objective_func = lambda e: -self._calculate_expected_utility(e)
        
        # Initial guess for the optimizer
        initial_guess = np.array([1.0, 1.0])
        
        # Efforts must be non-negative
        bounds = [(0, None), (0, None)]
        
        # Run the optimization
        result = minimize(
            fun=objective_func, 
            x0=initial_guess, 
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        if result.success:
            # If successful, store the chosen efforts
            self.chosen_ep, self.chosen_es = result.x
        else:
            # If the optimizer fails for any reason, default to zero effort
            self.chosen_ep, self.chosen_es = 0.0, 0.0

    def step(self):
        """
        Executes the rational agent's decision-making process for one step.
        """
        self._choose_optimal_efforts()

    def learn(self, reward: float, action_index: int):
        """
        The learning method for the agent. The rational agent does not learn,
        so this method does nothing.
        """
        pass