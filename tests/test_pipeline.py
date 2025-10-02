# tests/test_pipeline.py

"""
Unit tests for the experiment pipeline in src/pipeline.py.
"""

import unittest
from unittest.mock import patch
import pandas as pd
import mesa

from src.pipeline import run, _SafetyModelWrapper
from src.agents.rational import RationalDeveloperAgent

class TestPipeline(unittest.TestCase):

    def setUp(self):
        """Set up parameter configurations for reuse in tests."""
        self.parameters = {
            "nu": [0.2, 0.8], # Variable parameter
            "theta_L": 1.0,   # Fixed parameter
            "theta_H": 2.0,   # Fixed parameter
            "delta_W": 1000.0,
            "safety_bonus": 100.0,
            "performance_bonus": 50.0,
            "developer_class": RationalDeveloperAgent
        }
        self.iterations = 10

    def test_input_validation(self):
        """
        Tests that the run function raises errors for various invalid inputs.
        """
        with self.assertRaises(TypeError, msg="Should fail if parameters is not a dict"):
            run(parameters=[], iterations=self.iterations)
        
        with self.assertRaises(ValueError, msg="Should fail for non-positive iterations"):
            run(parameters=self.parameters, iterations=0)

    @patch('src.pipeline.mesa.batch_run')
    def test_batch_run_called_correctly(self, mock_batch_run):
        """
        Tests that our pipeline function calls mesa.batch_run with the correct arguments,
        including the internal wrapper class.
        """
        run(parameters=self.parameters, iterations=self.iterations)
        
        mock_batch_run.assert_called_once()
        mock_batch_run.assert_called_with(
            model_cls=_SafetyModelWrapper, # Crucially, it should call the wrapper
            parameters=self.parameters,
            iterations=self.iterations,
            number_processes=1,
            data_collection_period=-1,
            display_progress=True
        )

    @patch('src.pipeline.mesa.batch_run')
    def test_result_processing(self, mock_batch_run):
        """
        Tests that the function correctly processes the raw output from batch_run.
        """
        # Arrange: Configure the mock to return fake data
        sample_raw_results = [{'RunId': 0, 'nu': 0.2}, {'RunId': 1, 'nu': 0.8}]
        mock_batch_run.return_value = sample_raw_results
        
        # Act
        results_df = run(parameters=self.parameters, iterations=1)
        
        # Assert
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), len(sample_raw_results))
        self.assertIn('nu', results_df.columns)

    @patch('src.pipeline.mesa.batch_run')
    def test_parallelization_argument(self, mock_batch_run):
        """
        Tests that the number_processes argument is passed correctly.
        """
        run(parameters=self.parameters, iterations=self.iterations, number_processes=4)
        call_args = mock_batch_run.call_args.kwargs
        self.assertEqual(call_args['number_processes'], 4)


if __name__ == '__main__':
    unittest.main()