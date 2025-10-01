# tests/test_agents.py

"""
Unit tests for the Agent classes in src/agents.py.

This suite focuses on the DeveloperAgent, ensuring it:
1.  Initializes correctly and validates its parameters.
2.  Correctly calculates its expected utility for a given contract and effort.
3.  Responds rationally to different incentive structures (e.g., performance-only vs. safety-only).
4.  Demonstrates behavior consistent with the single-crossing property (i.e., higher-quality
    types exert more safety effort).
"""

import unittest
import mesa
from src.agents import DeveloperAgent, RegulatorAgent
from src.contracts import (
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract
)

# A mock Mesa model is needed to instantiate agents for testing
class DummyModel(mesa.Model):
    """A dummy model for testing purposes."""
    pass


class TestDeveloperAgent(unittest.TestCase):

    def setUp(self):
        """
        Set up a dummy model and common parameters for tests.
        This method is run before each test.
        """
        self.model = DummyModel()
        self.theta_L = 1.0  # Low-quality type
        self.theta_H = 2.0  # High-quality type

    def test_initialization(self):
        """
        Tests that the DeveloperAgent initializes correctly and validates its inputs.
        """
        # Test 1: Correct initialization
        agent = DeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H, behavior_mode='rational')
        self.assertEqual(agent.theta, self.theta_H)
        self.assertEqual(agent.behavior_mode, 'rational')

        # Test 2: Invalid theta
        with self.assertRaises(AssertionError, msg="Should fail for non-positive theta"):
            DeveloperAgent(unique_id=2, model=self.model, theta=-1.0)
            
        # Test 3: Invalid behavior_mode
        with self.assertRaises(AssertionError, msg="Should fail for unrecognized behavior_mode"):
            DeveloperAgent(unique_id=3, model=self.model, theta=1.0, behavior_mode='invalid_mode')

    def test_calculate_expected_utility(self):
        """
        Tests the internal expected utility calculation with a known example.
        This isolates the calculation logic from the optimization logic.
        """
        agent = DeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H)
        
        # Define a known scenario
        efforts = (1.0, 2.0) # e_p = 1, e_s = 2
        contract = create_hybrid_contract(safety_bonus=100.0, performance_bonus=50.0)
        
        calculated_utility = agent._calculate_expected_utility(efforts, contract, agent.theta)
        
        # This value is the correct, manually recalculated result based on the final functions.
        self.assertAlmostEqual(calculated_utility, 116.57, places=2, msg="Expected utility calculation is incorrect.")
        
    def test_effort_choice_responds_to_incentives(self):
        """
        Tests that the agent's effort choice rationally responds to the contract structure.
        """
        agent = DeveloperAgent(unique_id=1, model=self.model, theta=self.theta_H)

        # Scenario 1: Safety-only incentive
        agent.contract_offer = create_naive_fine_contract(safety_bonus=100.0)
        agent.step()
        self.assertGreater(agent.chosen_es, agent.chosen_ep, "Agent should choose more safety effort for a safety-only contract.")
        self.assertAlmostEqual(agent.chosen_ep, 0.0, places=5, msg="Agent should choose near-zero performance effort for a safety-only contract.")

        # Scenario 2: Performance-only incentive
        agent.contract_offer = create_performance_contract(performance_bonus=50.0)
        agent.step()
        self.assertGreater(agent.chosen_ep, agent.chosen_es, "Agent should choose more performance effort for a performance-only contract.")
        self.assertAlmostEqual(agent.chosen_es, 0.0, places=5, msg="Agent should choose near-zero safety effort for a performance-only contract.")

    def test_effort_choice_depends_on_type(self):
        """
        Tests that a higher-quality agent exerts more safety effort, all else equal.
        This is a critical test of the model's core single-crossing property.
        """
        # Create two agents, identical except for their type
        agent_L = DeveloperAgent(unique_id=1, model=self.model, theta=self.theta_L)
        agent_H = DeveloperAgent(unique_id=2, model=self.model, theta=self.theta_H)

        # Give both agents the same, non-trivial contract
        contract = create_hybrid_contract(safety_bonus=100.0, performance_bonus=50.0)
        agent_L.contract_offer = contract
        agent_H.contract_offer = contract

        # Have them choose their efforts
        agent_L.step()
        agent_H.step()

        # --- Assertions ---
        # The high-quality agent has a lower marginal cost of safety, so they should choose more.
        self.assertGreater(agent_H.chosen_es, agent_L.chosen_es, "High-theta agent should choose more safety effort.")
        
        # Ensure that both chose non-trivial efforts
        self.assertGreater(agent_L.chosen_es, 0.0)
        self.assertGreater(agent_H.chosen_es, 0.0)


if __name__ == '__main__':
    unittest.main()