# tests/agents/test_rational_developer.py

"""
Unit tests for the RationalDeveloperAgent class in src/agents/rational.py.

This suite is designed to be comprehensive, testing the agent's logic against
a variety of different contract scenarios to ensure its behavior is robust and
consistent with the paper's theoretical predictions.
"""

import unittest
import mesa
from src.agents.rational import RationalDeveloperAgent
from src.contracts.heuristics import (
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract
)

class DummyModel(mesa.Model):
    """A dummy model for testing purposes."""
    pass

class TestRationalDeveloperAgent(unittest.TestCase):

    def setUp(self):
        """Set up a dummy model and common parameters for tests."""
        self.model = DummyModel()
        self.theta_L = 1.0  # Low-quality type
        self.theta_H = 2.0  # High-quality type
        
        # Create a list of varied contract scenarios for robust testing
        self.varied_hybrid_contracts = [
            # Balanced incentives
            create_hybrid_contract(safety_bonus=100.0, performance_bonus=50.0),
            # Safety-heavy incentive
            create_hybrid_contract(safety_bonus=200.0, performance_bonus=20.0),
            # Performance-heavy incentive
            create_hybrid_contract(safety_bonus=20.0, performance_bonus=100.0),
            # High overall incentives
            create_hybrid_contract(safety_bonus=200.0, performance_bonus=100.0),
        ]

    def test_initialization(self):
        """Tests that the agent initializes correctly and validates its inputs."""
        agent = RationalDeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H)
        self.assertEqual(agent.theta, self.theta_H)

        with self.assertRaises(AssertionError):
            RationalDeveloperAgent(unique_id=2, model=self.model, theta=-1.0)
            
    def test_expected_utility_calculation(self):
        """Tests the internal expected utility calculation with a known example."""
        agent = RationalDeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H)
        
        efforts = (1.0, 2.0)
        agent.contract_offer = self.varied_hybrid_contracts[0] # Use the balanced contract
        
        calculated_utility = agent._calculate_expected_utility(efforts)
        
        # This value is the correct result for this specific scenario.
        self.assertAlmostEqual(calculated_utility, 116.57, places=2)
        
    def test_response_to_pure_incentives(self):
        """
        Tests that the agent's effort choice responds correctly to pure, one-sided contracts.
        """
        agent = RationalDeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H)

        # Scenario 1: Safety-only incentive
        agent.contract_offer = create_naive_fine_contract(safety_bonus=100.0)
        agent.step()
        self.assertGreater(agent.chosen_es, agent.chosen_ep)
        self.assertAlmostEqual(agent.chosen_ep, 0.0, places=5)

        # Scenario 2: Performance-only incentive
        agent.contract_offer = create_performance_contract(performance_bonus=50.0)
        agent.step()
        self.assertGreater(agent.chosen_ep, agent.chosen_es)
        self.assertAlmostEqual(agent.chosen_es, 0.0, places=5)

    def test_behavior_by_type_across_scenarios(self):
        """
        Tests the single-crossing property robustly across a variety of contracts.
        
        For any given contract, the high-quality agent should always exert more
        safety effort than the low-quality agent.
        """
        agent_L = RationalDeveloperAgent(unique_id=1, model=self.model, theta=self.theta_L)
        agent_H = RationalDeveloperAgent(unique_id=2, model=self.model, theta=self.theta_H)

        for contract in self.varied_hybrid_contracts:
            with self.subTest(contract=contract):
                # Assign the same contract to both agents
                agent_L.contract_offer = contract
                agent_H.contract_offer = contract

                # Have them choose their efforts
                agent_L.step()
                agent_H.step()

                # Assert that the high-quality agent always chooses more safety effort
                self.assertGreater(
                    agent_H.chosen_es, agent_L.chosen_es,
                    msg="High-theta agent should choose more safety effort."
                )
                
                # Assert that both choose non-zero efforts
                self.assertGreater(agent_L.chosen_es, 0.0)
                self.assertGreater(agent_H.chosen_es, 0.0)


if __name__ == '__main__':
    unittest.main()