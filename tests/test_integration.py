# tests/test_integration.py

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import mesa

from src.model import SafetyModel
from src.pipeline import run
from src.agents.rational import RationalDeveloperAgent
from src.agents.learning import LearningDeveloperAgent
from src.agents.risk_averse import RiskAverseDeveloperAgent
from src.contracts.optimal import calculate_optimal_contract
from src.contracts.heuristics import *

class DummyModel(mesa.Model):
    """A dummy model for testing agent logic in isolation."""
    def __init__(self):
        super().__init__() # Initializes the parent class
        self.random = np.random.default_rng()

class TestIntegration(unittest.TestCase):

    def setUp(self):
        """Set up varied parameter sets for robust testing."""
        self.base_params = {
            'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0,
            'safety_bonus': 100.0, 'performance_bonus': 50.0
        }
        self.learning_params = {
            'action_space': [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            'alpha': 0.1, 'epsilon': 0.1
        }
        # A larger set for the main solver test
        self.varied_param_sets = [
            self.base_params,
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 5000.0, 'safety_bonus': 100.0, 'performance_bonus': 50.0},
        ]

    def test_optimal_contract_implementation_varied(self):
        """
        Tests that the RationalDeveloperAgent correctly responds to the optimal
        contract across different parameter scenarios. THIS TEST IS EXPECTED TO FAIL.
        """
        for params in self.varied_param_sets:
            with self.subTest(params=params):
                contract_menu, theoretical_efforts = calculate_optimal_contract(params)
                
                low_type_agent = RationalDeveloperAgent(unique_id=1, model=DummyModel(), theta=params['theta_L'])
                low_type_agent.contract_offer = contract_menu['L']
                low_type_agent.step()
                
                # Relaxed precision to account for numerical noise between solvers
                self.assertAlmostEqual(low_type_agent.chosen_es, theoretical_efforts['L']['es'], places=3)
                self.assertAlmostEqual(low_type_agent.chosen_ep, theoretical_efforts['L']['ep'], places=3)

                high_type_agent = RationalDeveloperAgent(unique_id=2, model=DummyModel(), theta=params['theta_H'])
                high_type_agent.contract_offer = contract_menu['H']
                high_type_agent.step()
                self.assertAlmostEqual(high_type_agent.chosen_es, theoretical_efforts['H']['es'], places=3)
                self.assertAlmostEqual(high_type_agent.chosen_ep, theoretical_efforts['H']['ep'], places=3)

    def test_heuristic_contract_behaviors(self):
        """
        Tests the full model's end-to-end behavior with simple heuristic contracts.
        """
        params = self.base_params.copy()
        params['developer_class'] = RationalDeveloperAgent

        with self.subTest(contract='performance'):
            perf_params = params.copy(); perf_params['contract_type'] = 'performance'
            model_perf = SafetyModel(perf_params); model_perf.step()
            self.assertAlmostEqual(model_perf.developer.chosen_es, 0.0, places=4)
        
        with self.subTest(contract='fine'):
            fine_params = params.copy(); fine_params['contract_type'] = 'fine'
            model_fine = SafetyModel(fine_params); model_fine.step()
            self.assertAlmostEqual(model_fine.developer.chosen_ep, 0.0, places=4)

    def test_agent_class_integration(self):
        """
        Smoke test to ensure the SafetyModel can initialize and run with all agent types.
        """
        agent_classes = [RationalDeveloperAgent, LearningDeveloperAgent, RiskAverseDeveloperAgent]
        for agent_class in agent_classes:
            with self.subTest(agent_class=agent_class.__name__):
                params = self.base_params.copy()
                params['developer_class'] = agent_class
                params['developer_params'] = self.learning_params if agent_class == LearningDeveloperAgent else {}
                model = SafetyModel(params); model.step()

    def test_learning_agent_hook(self):
        """Verifies the model correctly calls the learn() method on a learning agent."""
        params = self.base_params.copy()
        params['contract_type'] = 'hybrid'
        params['developer_class'] = LearningDeveloperAgent
        params['developer_params'] = self.learning_params
        model = SafetyModel(params)
        with patch.object(model.developer, 'learn') as mock_learn:
            model.step()
            mock_learn.assert_called_once()

    def test_pipeline_end_to_end(self):
        """A top-level smoke test for the entire experimental pipeline."""
        # Define parameters explicitly for this test to ensure isolation
        params_for_run = {
            "nu": [0.2, 0.8], # Variable parameter
            "theta_L": 1.0,
            "theta_H": 2.0,
            "delta_W": 1000.0,
            "safety_bonus": 100.0,
            "performance_bonus": 50.0,
            "developer_class": RationalDeveloperAgent
        }
        
        results_df = run(parameters=params_for_run, iterations=1)
        self.assertIsInstance(results_df, pd.DataFrame)
        # The correct way to check the number of runs is by the number of unique RunId's
        self.assertEqual(results_df['RunId'].nunique(), 2)

if __name__ == '__main__':
    unittest.main()