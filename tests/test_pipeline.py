# tests/test_pipeline.py

"""
Comprehensive unit tests for the custom experiment pipeline in src/pipeline.py.

This suite is broken into multiple classes to test each component of the pipeline
in isolation and with a wide variety of scenarios, ensuring the entire
experimental workflow is robust and correct.
"""

import unittest
import pandas as pd
from unittest.mock import patch

from src.pipeline import run, _expand_parameters
from src.agents.rational import RationalDeveloperAgent

class TestExpandParameters(unittest.TestCase):
    """
    A dedicated test class for the _expand_parameters helper function.
    """

    def test_one_variable_parameter(self):
        """Tests expansion with a single variable parameter."""
        params = {"nu": [0.2, 0.8], "theta_L": 1.0}
        expanded = _expand_parameters(params)
        self.assertEqual(len(expanded), 2)
        self.assertIn({'nu': 0.2, 'theta_L': 1.0}, expanded)
        self.assertIn({'nu': 0.8, 'theta_L': 1.0}, expanded)

    def test_multiple_variable_parameters(self):
        """Tests expansion with a grid of two variable parameters."""
        params = {"nu": [0.2, 0.8], "contract_type": ["fine", "hybrid"], "theta_L": 1.0}
        expanded = _expand_parameters(params)
        self.assertEqual(len(expanded), 4)
        self.assertIn({'nu': 0.2, 'contract_type': 'fine', 'theta_L': 1.0}, expanded)
        self.assertIn({'nu': 0.8, 'contract_type': 'hybrid', 'theta_L': 1.0}, expanded)

    def test_no_variable_parameters(self):
        """Tests the edge case where all parameters are fixed."""
        params = {"theta_L": 1.0, "theta_H": 2.0}
        expanded = _expand_parameters(params)
        self.assertEqual(len(expanded), 1)
        self.assertIn(params, expanded)

    def test_parameter_values_are_objects(self):
        """Tests expansion when a parameter value is a class object."""
        params = {"developer_class": [RationalDeveloperAgent], "theta_L": 1.0}
        expanded = _expand_parameters(params)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]['developer_class'], RationalDeveloperAgent)

    def test_empty_list_parameter(self):
        """Tests the edge case where a variable parameter list is empty."""
        params = {"nu": [], "theta_L": 1.0}
        expanded = _expand_parameters(params)
        # itertools.product with an empty list results in zero combinations
        self.assertEqual(len(expanded), 0)


class TestRunFunction(unittest.TestCase):
    """
    A dedicated test class for the main `run` function, using mocking.
    """
    def setUp(self):
        """
        Sets up a complete base parameter dictionary required by the SafetyModel.
        """
        self.base_params = {
            'nu': 0.5,
            'theta_L': 1.0,
            'theta_H': 2.0,
            'delta_W': 1000.0,
            'safety_bonus': 100.0,
            'performance_bonus': 50.0,
            'developer_class': RationalDeveloperAgent,
            'contract_type': 'hybrid'
        }

    def test_input_validation(self):
        """Tests that the run function raises errors for invalid inputs."""
        with self.assertRaises(TypeError):
            run(parameters=[], iterations=1)
        with self.assertRaises(ValueError):
            run(parameters=self.base_params, iterations=0)

    @patch('builtins.print')
    def test_warning_for_multiprocessing(self, mock_print):
        """Tests that a warning is printed when number_processes > 1."""
        run(parameters=self.base_params, iterations=1, number_processes=4)
        mock_print.assert_any_call("Warning: Custom pipeline does not support multiprocessing. "
                                   "Running in a single process.")

    @patch('src.pipeline.SafetyModel')
    def test_run_count_and_instantiation(self, MockSafetyModel):
        """
        Verifies that the model is instantiated the correct number of times.
        """
        params = self.base_params.copy()
        params['nu'] = [0.2, 0.8] # 2 combinations
        iterations = 5
        total_runs = 2 * iterations
        
        run(parameters=params, iterations=iterations)
        
        self.assertEqual(MockSafetyModel.call_count, total_runs)

    @patch('src.pipeline.SafetyModel')
    def test_data_aggregation_structure(self, MockSafetyModel):
        """
        Verifies that the final DataFrame has the correct structure and columns.
        """
        # Arrange: Configure the mock model's datacollector
        mock_instance = MockSafetyModel.return_value
        mock_instance.datacollector.get_model_vars_dataframe.return_value = pd.DataFrame([{'SocialWelfare': -10}])
        mock_instance.datacollector.get_agent_vars_dataframe.return_value = pd.DataFrame([{'Theta': 1.0, 'Payoff': 50}])

        # Act
        results_df = run(parameters=self.base_params, iterations=1)
        
        # Assert
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 1)
        
        expected_cols = [
            'SocialWelfare', 'Theta', 'Payoff', # from datacollector
            'RunId', 'iteration',              # metadata
            'developer_class', 'theta_L'       # from original params
        ]
        for col in expected_cols:
            self.assertIn(col, results_df.columns)

    @patch('src.pipeline.SafetyModel')
    def test_data_aggregation_content(self, MockSafetyModel):
        """
        Verifies that the content of the DataFrame for a single run is correct.
        """
        # Arrange
        mock_instance = MockSafetyModel.return_value
        mock_instance.datacollector.get_model_vars_dataframe.return_value = pd.DataFrame([{'SocialWelfare': -10}])
        mock_instance.datacollector.get_agent_vars_dataframe.return_value = pd.DataFrame([{'Theta': 1.0, 'Payoff': 50}])
        params = self.base_params.copy()
        
        # Act
        results_df = run(parameters=params, iterations=1)
        
        # Assert
        self.assertEqual(results_df.loc[0, 'SocialWelfare'], -10)
        self.assertEqual(results_df.loc[0, 'Payoff'], 50)
        self.assertEqual(results_df.loc[0, 'RunId'], 0)
        self.assertEqual(results_df.loc[0, 'nu'], 0.5)

if __name__ == '__main__':
    unittest.main()

