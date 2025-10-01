# tests/test_model.py

"""
Unit tests for the SafetyModel class in src/model.py.

This suite ensures that the model:
1.  Initializes correctly with the specified agents and parameters.
2.  Assigns developer types stochastically according to the prior `nu`.
3.  Assigns the correct contract based on the experimental condition and agent type.
4.  Executes a single step of the game logic correctly.
5.  Collects data accurately using the DataCollector.
"""

import unittest
from unittest.mock import patch
import pandas as pd

from src.model import SafetyModel
from src.agents import DeveloperAgent, RegulatorAgent
from src.contracts import calculate_optimal_contract, create_naive_fine_contract

class TestSafetyModel(unittest.TestCase):

    def setUp(self):
        """Set up a base parameter dictionary for all tests."""
        self.base_params = {
            'nu': 0.5,
            'theta_L': 1.0,
            'theta_H': 2.0,
            'delta_W': 1000.0,
            'contract_type': 'optimal',
            'safety_bonus': 100.0,
            'performance_bonus': 50.0
        }

    def test_model_initialization(self):
        """Tests that the model initializes with the correct agents and setup."""
        model = SafetyModel(self.base_params)
        
        self.assertEqual(model.schedule.get_agent_count(), 2, "Model should have 2 agents.")
        
        agent_types = [type(agent) for agent in model.schedule.agents]
        self.assertIn(RegulatorAgent, agent_types)
        self.assertIn(DeveloperAgent, agent_types)

        # Check that the developer's theta is one of the valid types
        self.assertIn(model.developer.theta, [self.base_params['theta_L'], self.base_params['theta_H']])
        
        # Check that the datacollector has the initial state
        self.assertEqual(len(model.datacollector.get_model_vars_dataframe()), 1)

    def test_stochastic_type_assignment(self):
        """Tests that developer types are assigned according to the prior `nu`."""
        # Test with nu = 1.0 (always high type)
        params_H = self.base_params.copy()
        params_H['nu'] = 1.0
        model_H = SafetyModel(params_H)
        self.assertEqual(model_H.developer.theta, params_H['theta_H'])

        # Test with nu = 0.0 (always low type)
        params_L = self.base_params.copy()
        params_L['nu'] = 0.0
        model_L = SafetyModel(params_L)
        self.assertEqual(model_L.developer.theta, params_L['theta_L'])

        # Statistical test for nu = 0.5
        n_runs = 1000
        high_type_count = 0
        for _ in range(n_runs):
            model = SafetyModel(self.base_params)
            if model.developer.theta == self.base_params['theta_H']:
                high_type_count += 1
        
        proportion = high_type_count / n_runs
        # Check if the proportion is within a reasonable range of 0.5
        self.assertGreater(proportion, 0.4)
        self.assertLess(proportion, 0.6)

    def test_contract_assignment(self):
        """Tests that the correct contract is assigned to the developer."""
        # Test optimal contract assignment for a high type
        params_opt = self.base_params.copy()
        params_opt['nu'] = 1.0  # Force high type
        params_opt['contract_type'] = 'optimal'
        model_opt = SafetyModel(params_opt)
        
        expected_menu = calculate_optimal_contract(params_opt)
        self.assertEqual(model_opt.developer.contract_offer, expected_menu['H'])

        # Test heuristic contract assignment
        params_fine = self.base_params.copy()
        params_fine['contract_type'] = 'fine'
        model_fine = SafetyModel(params_fine)
        
        expected_contract = create_naive_fine_contract(params_fine['safety_bonus'])
        self.assertEqual(model_fine.developer.contract_offer, expected_contract)
    
    @patch('numpy.random.random')
    def test_step_logic_and_payoffs(self, mock_random):
        """
        Tests the logic of a single model step by controlling for stochasticity.
        We mock the random number generator to force specific outcomes.
        """
        # --- Arrange ---
        # Force a high-type developer
        params = self.base_params.copy()
        params['nu'] = 1.0
        model = SafetyModel(params)
        
        # Manually set the developer's choice to known values to isolate model logic
        model.developer.chosen_ep = 1.0
        model.developer.chosen_es = 2.0
        
        # Mock the random outcomes:
        # 1st call for performance signal (p_pi_H > 0.4 -> High)
        # 2nd call for safety outcome (p_q_G > 0.4 -> Good)
        mock_random.side_effect = [0.4, 0.4]
        
        # Expected payment for (q='G', pi='H') from the placeholder optimal contract
        expected_payment = model.developer.contract_offer['G']['H']
        
        # --- Act ---
        model.step()
        
        # --- Assert ---
        self.assertFalse(model.disaster_occurred)
        self.assertEqual(model.developer.payoff, expected_payment, "Payoff was not calculated correctly.")
        
        # Expected welfare = W_G - payment = 0 - 30.0 = -30.0
        self.assertEqual(model.social_welfare, -expected_payment)
        
        # Check that data was collected for the step
        self.assertEqual(len(model.datacollector.get_model_vars_dataframe()), 2)

    def test_datacollector_output(self):
        """Tests that the DataCollector is set up correctly and produces a valid DataFrame."""
        model = SafetyModel(self.base_params)
        model.step()

        agent_df = model.datacollector.get_agent_vars_dataframe()
        self.assertIsInstance(agent_df, pd.DataFrame)
        
        # Check for expected columns
        expected_agent_cols = ["Theta", "ChosenEp", "ChosenEs", "Payoff"]
        for col in expected_agent_cols:
            self.assertIn(col, agent_df.columns)
            
        # Check that the data for the specific developer agent is present
        self.assertIn(1, agent_df.index.get_level_values('AgentID'))


if __name__ == '__main__':
    unittest.main()