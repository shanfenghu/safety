# tests/test_pipeline.py

"""
Unit tests for the experiment pipeline in src/pipeline.py.

This suite uses mocking to test the pipeline's logic in isolation, ensuring it:
1.  Properly validates its inputs.
2.  Calls the underlying `mesa.batch_run` with the correct arguments.
3.  Correctly processes the raw output into a pandas DataFrame.
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
        self.parameter_sets = [{
            'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0,
        }]
        self.iterations = 10

    def test_input_validation(self):
        """
        Tests that the run function raises errors for various invalid inputs.
        """
        with self.assertRaises(ValueError, msg="Should fail for empty parameter_sets"):
            run(parameter_sets=[], iterations=self.iterations)
        
        with self.assertRaises(TypeError, msg="Should fail if parameter_sets contains non-dicts"):
            run(parameter_sets=[self.parameter_sets[0], "not_a_dict"], iterations=self.iterations)

        with self.assertRaises(ValueError, msg="Should fail for non-positive iterations"):
            run(parameter_sets=self.parameter_sets, iterations=0)
            
        with self.assertRaises(ValueError, msg="Should fail for non-positive number_processes"):
            run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=-1)

    @patch('src.pipeline.mesa.batch_run')
    def test_batch_run_called_correctly(self, mock_batch_run):
        """
        Tests that our pipeline function calls mesa.batch_run with the correct arguments.
        """
        # --- Act ---
        run(parameter_sets=self.parameter_sets, iterations=self.iterations)
        
        # --- Assert ---
        mock_batch_run.assert_called_once()
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
        Tests that the function correctly processes the raw output from batch_run
        into a pandas DataFrame.
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
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), len(sample_raw_results))
        self.assertListEqual(list(results_df.columns), list(sample_raw_results[0].keys()))
        self.assertEqual(results_df.loc[0, 'Payoff'], 30.0)

    @patch('src.pipeline.mesa.batch_run')
    def test_parallelization_argument(self, mock_batch_run):
        """
        Tests that the number_processes argument is passed correctly to mesa.batch_run.
        """
        # Test with a specific number of processes
        run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=4)
        call_args = mock_batch_run.call_args.kwargs
        self.assertEqual(call_args['number_processes'], 4)

        # Test with None (for all available cores)
        run(parameter_sets=self.parameter_sets, iterations=self.iterations, number_processes=None)
        call_args = mock_batch_run.call_args.kwargs
        self.assertEqual(call_args['number_processes'], None)


if __name__ == '__main__':
    unittest.main()