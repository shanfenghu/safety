# tests/test_model.py

import unittest
from unittest.mock import patch
import pandas as pd
import mesa

from src.model import SafetyModel
from src.agents.rational import RationalDeveloperAgent
from src.agents.learning import LearningDeveloperAgent
from src.contracts.heuristics import *
from src.contracts.optimal import calculate_optimal_contract

class TestSafetyModel(unittest.TestCase):
    def setUp(self):
        """Set up base parameters for reuse in tests."""
        self.base_params = {
            'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0,
            'safety_bonus': 100.0, 'performance_bonus': 50.0
        }
        self.learning_params = {
            'action_space': [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            'alpha': 0.1, 'epsilon': 0.1
        }
    
    # --- The first four tests are unchanged and passing ---
    def test_initialization_with_rational_agent(self):
        params = self.base_params.copy(); params['developer_class'] = RationalDeveloperAgent
        model = SafetyModel(params)
        self.assertIsInstance(model.developer, RationalDeveloperAgent)

    def test_initialization_with_learning_agent(self):
        params = self.base_params.copy(); params['developer_class'] = LearningDeveloperAgent
        params['developer_params'] = self.learning_params
        model = SafetyModel(params)
        self.assertIsInstance(model.developer, LearningDeveloperAgent)

    def test_contract_assignment_scenarios(self):
        scenarios = ['optimal', 'fine', 'performance', 'hybrid']
        for ctype in scenarios:
            with self.subTest(contract_type=ctype):
                params = self.base_params.copy(); params['contract_type'] = ctype
                params['developer_class'] = RationalDeveloperAgent; params['nu'] = 1.0
                model = SafetyModel(params)
                if ctype == 'optimal':
                    expected, _ = calculate_optimal_contract(params)
                    self.assertDictEqual(model.developer.contract_offer, expected['H'])
                elif ctype == 'fine':
                    expected_contract = create_naive_fine_contract(params['safety_bonus'])
                    self.assertDictEqual(model.developer.contract_offer, expected_contract)
                elif ctype == 'performance':
                    expected_contract = create_performance_contract(params['performance_bonus'])
                    self.assertDictEqual(model.developer.contract_offer, expected_contract)
                elif ctype == 'hybrid':
                    expected_contract = create_hybrid_contract(params['safety_bonus'], params['performance_bonus'])
                    self.assertDictEqual(model.developer.contract_offer, expected_contract)

    def test_step_logic_rational_agent(self):
        params = self.base_params.copy(); params['developer_class'] = RationalDeveloperAgent
        params['nu'] = 1.0; model = SafetyModel(params)
        with patch.object(model.random, 'random', side_effect=[0.1, 0.1]):
            with patch.object(model.developer, 'learn') as mock_learn:
                model.step()
                mock_learn.assert_called_once()
                self.assertFalse(model.disaster_occurred)
                expected_payment = model.developer.contract_offer['G']['H']
                self.assertAlmostEqual(model.developer.payoff, expected_payment, places=2)

    def test_step_logic_learning_agent(self):
        """Tests that the model correctly calls the learn() method for a learning agent."""
        params = self.base_params.copy()
        # MODIFIED: Use a simple, known contract for this test
        params['contract_type'] = 'hybrid'
        params['developer_class'] = LearningDeveloperAgent
        params['developer_params'] = self.learning_params
        model = SafetyModel(params)

        with patch.object(model.random, 'random') as mock_rand, \
             patch.object(model.developer, 'learn') as mock_learn:
            
            # This side_effect forces pi='H' and q='B' (disaster with high performance).
            mock_rand.side_effect = [0.5, 0.1, 0.9]
            
            model.step()
            
            # For a hybrid contract, the payment for ('B', 'H') is the performance_bonus.
            expected_payment = self.base_params['performance_bonus']
            self.assertEqual(model.developer.payoff, expected_payment)
            
            # Assert that the learn method was called exactly once with the correct reward.
            mock_learn.assert_called_once_with(reward=expected_payment)


if __name__ == '__main__':
    unittest.main()