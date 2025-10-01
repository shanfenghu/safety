# tests/test_pipeline.py

"""
Unit tests for the experiment pipeline in src/pipeline.py.

This suite ensures that the `run` function:
1.  Properly validates its inputs and raises errors for invalid configurations.
2.  Calls the underlying `mesa.batch_run` with the correct, expected arguments.
3.  Correctly processes the raw output from `batch_run` into a pandas DataFrame.
4.  Handles optional arguments like parallelization correctly.
"""

import unittest
from unittest.mock import patch
import pandas as pd

from src.pipeline import run
from src.model import SafetyModel


class TestPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a base parameter configuration for reuse in tests."""
        self.base_params = {
            'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0,
            'contract_type': 'optimal', 'safety_bonus': 100.0, 'performance_bonus': 50.0
        }
        self.parameter_sets = [self.base_params]
        self.iterations = 10

    def test_input_validation(self):
        """
        Tests that the run function raises errors for invalid inputs.
        """
        with self.assertRaises(ValueError, msg="Should fail for empty parameter_sets"):
            run(parameter_sets=[], iterations=self.iterations)
        
        with self.assertRaises(TypeError, msg="Should fail if parameter_sets contains non-dicts"):
            run(parameter_sets=[self.base_params, "not_a_dict"], iterations=self.iterations)

        with self.assertRaises(ValueError, msg="Should fail for non-positive iterations"):
            run(parameter_sets=self.parameter_sets, iterations=0)
            
        with self.assertRaises(ValueError, msg="Should fail for non-positive number_processes"):
            run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=0)

    @patch('src.pipeline.mesa.batch_run')
    def test_batch_run_called_correctly(self, mock_batch_run):
        """
        Tests that our pipeline function calls mesa.batch_run with the correct arguments.
        """
        # --- Act ---
        run(parameter_sets=self.parameter_sets, iterations=self.iterations)
        
        # --- Assert ---
        # Check that the mocked batch_run was called exactly once
        mock_batch_run.assert_called_once()
        
        # Check that it was called with the arguments we expect
        mock_batch_run.assert_called_with(
            model_cls=SafetyModel,
            parameters=self.parameter_sets,
            iterations=self.iterations,
            number_processes=1, # The default value
            data_collection_period=-1,
            display_progress=True # Assuming tqdm is installed
        )

    @patch('src.pipeline.mesa.batch_run')
    def test_result_processing_and_return_type(self, mock_batch_run):
        """
        Tests that the function correctly processes the raw output from batch_run.
        """
        # --- Arrange ---
        # Create a fake raw output that mesa.batch_run would produce
        sample_raw_results = [
            {'RunId': 0, 'iteration': 0, 'Step': 1, 'Theta': 2.0, 'Payoff': 30.0},
            {'RunId': 1, 'iteration': 0, 'Step': 1, 'Theta': 1.0, 'Payoff': 20.0},
        ]
        # Configure the mock to return this fake data
        mock_batch_run.return_value = sample_raw_results
        
        # --- Act ---
        results_df = run(parameter_sets=self.parameter_sets, iterations=self.iterations)
        
        # --- Assert ---
        # 1. The return type should be a pandas DataFrame
        self.assertIsInstance(results_df, pd.DataFrame)
        
        # 2. The DataFrame should have the correct number of rows
        self.assertEqual(len(results_df), len(sample_raw_results))
        
        # 3. The DataFrame columns should match the keys from the raw results
        self.assertListEqual(list(results_df.columns), list(sample_raw_results[0].keys()))
        
        # 4. A value should be correct
        self.assertEqual(results_df.loc[0, 'Payoff'], 30.0)

    @patch('src.pipeline.mesa.batch_run')
    def test_parallelization_argument(self, mock_batch_run):
        """
        Tests that the number_processes argument is passed correctly to mesa.batch_run.
        """
        # Test with a specific number of processes
        run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=4)
        # The 'kwargs' of the call object contains the arguments
        call_args = mock_batch_run.call_args.kwargs
        self.assertEqual(call_args['number_processes'], 4)

        # Test with None (for all available cores)
        run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=None)
        call_args = mock_batch_run.call_args.kwargs
        self.assertEqual(call_args['number_processes'], None)


if __name__ == '__main__':
    unittest.main()