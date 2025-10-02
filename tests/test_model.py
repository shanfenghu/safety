# tests/test_model.py

import unittest
from unittest.mock import patch
import pandas as pd
import mesa

# Imports from our refactored codebase
from src.model import SafetyModel
from src.agents.rational import RationalDeveloperAgent
from src.agents.learning import LearningDeveloperAgent
from src.agents.risk_averse import RiskAverseDeveloperAgent
from src.contracts.heuristics import *
from src.contracts.optimal import calculate_optimal_contract

class TestSafetyModel(unittest.TestCase):

    def setUp(self):
        self.base_params = {
            'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0,
            'safety_bonus': 100.0, 'performance_bonus': 50.0
        }
        self.learning_params = {
            'action_space': [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            'alpha': 0.1, 'epsilon': 0.1
        }

    def test_initialization_scenarios(self):
        """Tests that the model correctly instantiates all planned developer agent types."""
        agent_scenarios = [
            {'name': 'Rational', 'class': RationalDeveloperAgent, 'params': {}},
            {'name': 'Learning', 'class': LearningDeveloperAgent, 'params': self.learning_params},
            {'name': 'RiskAverse', 'class': RiskAverseDeveloperAgent, 'params': {}},
        ]
        
        for scenario in agent_scenarios:
            with self.subTest(agent_type=scenario['name']):
                params = self.base_params.copy(); params['developer_class'] = scenario['class']
                params['developer_params'] = scenario['params']; model = SafetyModel(params)
                self.assertEqual(model.schedule.get_agent_count(), 2)
                self.assertIsInstance(model.developer, scenario['class'])

    def test_contract_assignment_scenarios(self):
        """Tests that the correct contract is assigned for all contract types."""
        contract_scenarios = ['optimal', 'fine', 'performance', 'hybrid']
        
        for ctype in contract_scenarios:
            with self.subTest(contract_type=ctype):
                params = self.base_params.copy(); params['contract_type'] = ctype
                params['developer_class'] = RationalDeveloperAgent; params['nu'] = 1.0
                model = SafetyModel(params)
                if ctype == 'optimal':
                    expected, _ = calculate_optimal_contract(params)
                    expected_contract = expected['H']
                elif ctype == 'fine':
                    expected_contract = create_naive_fine_contract(params['safety_bonus'])
                elif ctype == 'performance':
                    expected_contract = create_performance_contract(params['performance_bonus'])
                elif ctype == 'hybrid':
                    expected_contract = create_hybrid_contract(params['safety_bonus'], params['performance_bonus'])
                self.assertDictEqual(model.developer.contract_offer, expected_contract)

    def test_stochastic_type_assignment(self):
        """Tests that developer types are assigned stochastically according to the prior `nu`."""
        n_runs = 1000; high_type_count = 0
        params = self.base_params.copy(); params['developer_class'] = RationalDeveloperAgent
        for _ in range(n_runs):
            model = SafetyModel(params)
            if model.developer.theta == self.base_params['theta_H']:
                high_type_count += 1
        proportion = high_type_count / n_runs
        self.assertGreater(proportion, 0.4); self.assertLess(proportion, 0.6)

    def test_correlation_feature_logic(self):
        """Tests that the correlation_k parameter correctly alters safety outcome logic."""
        correlation_scenarios = [0.0, 0.5, -0.5]
        
        for k in correlation_scenarios:
            with self.subTest(correlation_k=k):
                params = self.base_params.copy()
                params['developer_class'] = RationalDeveloperAgent
                params['correlation_k'] = k
                model = SafetyModel(params)
                
                model.developer.chosen_ep = 1.0
                model.developer.chosen_es = 2.0
                
                with patch('src.model.prob_good_safety_outcome') as mock_prob_func, \
                     patch.object(model.developer, 'step') as mock_agent_step:
                    
                    mock_prob_func.return_value = 0.8
                    model.step()
                    
                    # The patch on agent.step prevents it from running its optimizer,
                    # which is the goal. The scheduler still calls the (mocked) step method.
                    mock_agent_step.assert_called_once()
                    
                    # Verify the function was called with the correct 'effective' safety effort
                    expected_effective_es = 2.0 + k * 1.0
                    mock_prob_func.assert_called_once()
                    self.assertAlmostEqual(mock_prob_func.call_args[0][0], max(0, expected_effective_es))

    def test_step_logic_rational_agent(self):
        """Tests the step logic for a rational agent, controlling stochasticity."""
        params = self.base_params.copy(); params['developer_class'] = RationalDeveloperAgent
        params['nu'] = 1.0; model = SafetyModel(params)
        
        with patch.object(model.random, 'random', side_effect=[0.1, 0.1]):
            model.step()
            self.assertFalse(model.disaster_occurred)
            expected_payment = model.developer.contract_offer['G']['H']
            self.assertAlmostEqual(model.developer.payoff, expected_payment, places=2)

    def test_step_logic_learning_agent_hook(self):
        """Tests that the model correctly calls the learn() method for a learning agent."""
        params = self.base_params.copy(); params['contract_type'] = 'hybrid'
        params['developer_class'] = LearningDeveloperAgent
        params['developer_params'] = self.learning_params; model = SafetyModel(params)
        
        with patch.object(model.random, 'random', side_effect=[0.5, 0.1, 0.9]), \
             patch.object(model.developer, 'learn') as mock_learn:
            
            model.step()
            expected_payment = model.developer.contract_offer['B']['H']
            mock_learn.assert_called_once_with(reward=expected_payment)

    def test_datacollector_output(self):
        """Tests that the DataCollector is set up correctly and produces a valid DataFrame."""
        params = self.base_params.copy(); params['developer_class'] = RationalDeveloperAgent
        model = SafetyModel(params); model.step()
        agent_df = model.datacollector.get_agent_vars_dataframe()
        self.assertIsInstance(agent_df, pd.DataFrame)
        expected_cols = ["Theta", "ChosenEp", "ChosenEs", "Payoff"]
        for col in expected_cols: self.assertIn(col, agent_df.columns)

if __name__ == '__main__':
    unittest.main()