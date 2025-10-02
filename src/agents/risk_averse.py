# src/agents/risk_averse.py

"""
Defines the RiskAverseDeveloperAgent class for the simulation.
"""

import numpy as np
from scipy.optimize import minimize

from src.agents.base_developer import BaseDeveloperAgent
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal
from src.types import Contract


class RiskAverseDeveloperAgent(BaseDeveloperAgent):
    """
    An agent that is risk-averse with respect to monetary payoffs.

    This agent's utility for money is concave, modeled here using a square
    root function (U(payment) = sqrt(payment)). This means it dislikes the
    uncertainty of outcome-contingent contracts more than a risk-neutral agent.
    It still chooses efforts to maximize its expected utility.
    """

    def _calculate_expected_utility(self, efforts: np.ndarray) -> float:
        """
        Calculates the agent's expected utility for a given effort pair,
        applying a square root utility function to the monetary payoffs.
        """
        e_p, e_s = efforts
        
        # Cost of effort is a direct disutility, not transformed by the utility function.
        cost = (e_p**2 / 2.0) + (e_s**2 / (2.0 * self.theta))
        
        # Probabilities of outcomes based on effort
        p_pi_H = prob_high_performance_signal(e_p)
        p_q_G = prob_good_safety_outcome(e_s)

        # Expected utility of payment, applying sqrt() to each potential payment
        expected_utility_of_payment = (
            p_q_G * p_pi_H * np.sqrt(self.contract_offer['G']['H']) +
            p_q_G * (1 - p_pi_H) * np.sqrt(self.contract_offer['G']['L']) +
            (1 - p_q_G) * p_pi_H * np.sqrt(self.contract_offer['B']['H']) +
            (1 - p_q_G) * (1 - p_pi_H) * np.sqrt(self.contract_offer['B']['L'])
        )
        
        return expected_utility_of_payment - cost

    def _choose_optimal_efforts(self):
        """Solves the agent's optimization problem to find the best efforts."""
        
        # The function for the optimizer to minimize is the *negative* of the expected utility
        objective_func = lambda e: -self._calculate_expected_utility(e)
        
        initial_guess = np.array([1.0, 1.0])
        bounds = [(0, None), (0, None)]
        
        result = minimize(
            fun=objective_func, 
            x0=initial_guess, 
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        if result.success:
            self.chosen_ep, self.chosen_es = result.x
        else:
            self.chosen_ep, self.chosen_es = 0.0, 0.0

    def step(self):
        """
        Executes the risk-averse agent's decision-making process for one step.
        """
        self._choose_optimal_efforts()

    def learn(self, reward: float):
        """
        The learning method. This agent does not learn from experience.
        """
        pass