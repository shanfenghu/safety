# tests/agents/test_risk_averse.py (Corrected)

import unittest
import mesa
import numpy as np

from src.agents.rational import RationalDeveloperAgent
from src.agents.risk_averse import RiskAverseDeveloperAgent
from src.contracts.heuristics import create_naive_fine_contract, create_hybrid_contract

class DummyModel(mesa.Model):
    """A dummy model for testing purposes."""
    pass

class TestRiskAverseDeveloperAgent(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel()
        self.theta = 2.0

    # test_initialization and test_risk_averse_utility_calculation are correct and unchanged...
    def test_initialization(self):
        agent = RiskAverseDeveloperAgent(unique_id=1, model=self.model, theta=self.theta)
        self.assertEqual(agent.theta, self.theta)

    def test_risk_averse_utility_calculation(self):
        agent = RiskAverseDeveloperAgent(unique_id=1, model=self.model, theta=self.theta)
        efforts = (1.0, 2.0)
        agent.contract_offer = create_hybrid_contract(safety_bonus=100.0, performance_bonus=50.0)
        from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal
        e_p, e_s = efforts; cost = (e_p**2/2.0) + (e_s**2/(2.0*self.theta))
        p_pi_H = prob_high_performance_signal(e_p); p_q_G = prob_good_safety_outcome(e_s)
        c = agent.contract_offer
        expected_utility_of_payment = (p_q_G*p_pi_H*np.sqrt(c['G']['H']) + p_q_G*(1-p_pi_H)*np.sqrt(c['G']['L']) + 
                                     (1-p_q_G)*p_pi_H*np.sqrt(c['B']['H']) + (1-p_q_G)*(1-p_pi_H)*np.sqrt(c['B']['L']))
        expected_total_utility = expected_utility_of_payment - cost
        calculated_utility = agent._calculate_expected_utility(efforts)
        self.assertAlmostEqual(calculated_utility, expected_total_utility, places=5)

    def test_behavioral_difference_from_rational(self):
        """
        Tests that the risk-averse agent behaves differently from a rational
        agent when faced with a high-variance ("gain-only") contract.
        """
        # --- Arrange ---
        rational_agent = RationalDeveloperAgent(unique_id=1, model=self.model, theta=self.theta)
        risk_averse_agent = RiskAverseDeveloperAgent(unique_id=2, model=self.model, theta=self.theta)
        risky_contract = create_naive_fine_contract(safety_bonus=200.0)
        rational_agent.contract_offer = risky_contract
        risk_averse_agent.contract_offer = risky_contract

        # --- Act ---
        rational_agent.step()
        risk_averse_agent.step()

        # --- Assert ---
        # With a concave utility function (sqrt), the large bonus has diminished
        # marginal utility for the risk-averse agent. They are therefore LESS
        # motivated to exert costly effort to obtain it compared to a rational agent.
        self.assertLess(
            risk_averse_agent.chosen_es, rational_agent.chosen_es,
            "Risk-averse agent should choose less safety effort for this gain-only contract."
        )
        
        # Sanity check: both should still choose more safety than performance effort
        self.assertGreater(rational_agent.chosen_es, rational_agent.chosen_ep)
        self.assertGreater(risk_averse_agent.chosen_es, risk_averse_agent.chosen_ep)


if __name__ == '__main__':
    unittest.main()