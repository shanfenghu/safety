# tests/test_simulation_utils.py

"""
Unit tests for the utility functions in src/simulation_utils.py.

This suite ensures that the probability functions are mathematically correct,
adhere to the properties of probabilities (i.e., values are in [0, 1]),
and handle invalid inputs gracefully.
"""

import unittest
from src.simulation_utils import (
    prob_high_performance_signal,
    prob_good_safety_outcome
)

class TestSimulationUtils(unittest.TestCase):

    def test_prob_high_performance_signal(self):
        """
        Tests the prob_high_performance_signal function from every aspect.
        """
        # Test 1: Boundary condition at zero effort
        self.assertEqual(
            prob_high_performance_signal(0.0), 0.0,
            "Probability should be 0 for zero effort."
        )

        # Test 2: Behavior with very high effort (should approach 1)
        self.assertAlmostEqual(
            prob_high_performance_signal(100.0), 1.0,
            "Probability should approach 1 for very high effort."
        )

        # Test 3: Monotonicity (more effort should mean higher probability)
        self.assertGreater(
            prob_high_performance_signal(2.0), prob_high_performance_signal(1.0),
            "Function should be monotonically increasing."
        )

        # Test 4: Range validation (output must be a valid probability)
        prob = prob_high_performance_signal(1.5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # Test 5: Input validation (should fail for negative effort)
        with self.assertRaises(AssertionError, msg="Should fail for negative effort"):
            prob_high_performance_signal(-1.0)
            
    def test_prob_good_safety_outcome(self):
        """
        Tests the prob_good_safety_outcome function from every aspect.
        """
        # Test 1: Boundary condition at zero effort
        self.assertEqual(
            prob_good_safety_outcome(0.0), 0.0,
            "Probability should be 0 for zero effort."
        )

        # Test 2: Behavior with very high effort (should approach 1)
        self.assertAlmostEqual(
            prob_good_safety_outcome(100.0), 1.0,
            "Probability should approach 1 for very high effort."
        )

        # Test 3: Monotonicity (more effort should mean higher probability)
        self.assertGreater(
            prob_good_safety_outcome(2.0), prob_good_safety_outcome(1.0),
            "Function should be monotonically increasing."
        )

        # Test 4: Range validation (output must be a valid probability)
        prob = prob_good_safety_outcome(1.5)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # Test 5: Input validation (should fail for negative effort)
        with self.assertRaises(AssertionError, msg="Should fail for negative effort"):
            prob_good_safety_outcome(-1.0)


if __name__ == '__main__':
    unittest.main()