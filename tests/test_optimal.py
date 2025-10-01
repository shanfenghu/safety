# tests/contracts/test_optimal.py

"""
Pure unit tests for the optimal contract solver in src/contracts/optimal.py.

This suite tests the solver's internal components in isolation, without any
dependency on the agent classes, to ensure its mathematical correctness.
Tests are run against multiple, varied parameter sets for robustness.
"""

import unittest
import numpy as np

from src.contracts.optimal import _solve_for_optimal_efforts, _solve_for_payments
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal

class TestOptimalContractSolver(unittest.TestCase):

    def setUp(self):
        """Set up multiple, varied parameter configurations for robust testing."""
        self.param_sets = [
            # 1. Baseline Scenario
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0},

            # 2. High Stakes Scenario
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 5000.0},

            # 3. Low Type Separation Scenario
            {'nu': 0.5, 'theta_L': 1.5, 'theta_H': 2.0, 'delta_W': 1000.0},

            # 4. High Prior for Good Type Scenario
            {'nu': 0.8, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0},
            
            # 5. Low Stakes Scenario (Disaster is not very costly)
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 100.0},
            
            # 6. High Type Separation (Types are easy to distinguish)
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 5.0, 'delta_W': 1000.0},
            
            # 7. Low Prior for Good Type (Good type is rare)
            {'nu': 0.2, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0},
            
            # 8. Extreme High Stakes
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 10000.0},
            
            # 9. Combined: Low Separation & High Prior (Theoretically challenging)
            {'nu': 0.9, 'theta_L': 1.9, 'theta_H': 2.0, 'delta_W': 1000.0},
            
            # 10. Combined: High Stakes & Low Prior
            {'nu': 0.1, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 5000.0},
        ]

    def test_solve_for_optimal_efforts(self):
        """
        Unit test for the _solve_for_optimal_efforts helper function.
        Verifies that the solved efforts satisfy the theoretical FOCs.
        """
        for params in self.param_sets:
            with self.subTest(params=params):
                # --- Act ---
                optimal_efforts = _solve_for_optimal_efforts(params)

                # --- Assert ---
                # 1. Sanity checks
                self.assertGreater(optimal_efforts['H']['es'], optimal_efforts['L']['es'])
                self.assertTrue(all(e >= 0 for eff in optimal_efforts.values() for e in eff.values()))

                # 2. Verify FOCs are satisfied (i.e., the function found the root)
                e_sH, e_sL = optimal_efforts['H']['es'], optimal_efforts['L']['es']
                
                foc_H_result = np.exp(-e_sH) * params['delta_W'] - e_sH / params['theta_H']
                self.assertAlmostEqual(foc_H_result, 0.0, places=6, msg="FOC for high-type not satisfied.")

                foc_L_result = (1 - params['nu']) * (np.exp(-e_sL) * params['delta_W'] - e_sL / params['theta_L']) \
                               - params['nu'] * (e_sL / params['theta_L'] - e_sL / params['theta_H'])
                self.assertAlmostEqual(foc_L_result, 0.0, places=6, msg="FOC for low-type not satisfied.")

    def test_solve_for_payments(self):
        """
        Unit test for the _solve_for_payments helper function.
        Verifies that the generated payments satisfy all binding constraints.
        """
        for params in self.param_sets:
            with self.subTest(params=params):
                # --- Arrange ---
                # First, get the efforts that these payments are supposed to implement
                optimal_efforts = _solve_for_optimal_efforts(params)
                e_pL, e_sL = optimal_efforts['L']['ep'], optimal_efforts['L']['es']
                e_pH, e_sH = optimal_efforts['H']['ep'], optimal_efforts['H']['es']

                # --- Act ---
                contract_menu = _solve_for_payments(optimal_efforts, params)

                # --- Assert ---
                # 1. Verify Limited Liability for all 8 payments
                for contract in contract_menu.values():
                    for payment_dict in contract.values():
                        for payment in payment_dict.values():
                            self.assertGreaterEqual(payment, -1e-9, msg="Payment violates LL.") # Use tolerance for float precision

                # 2. Verify binding IR-L constraint (Utility for low type should be 0)
                utility_L = self._calculate_utility_manually(
                    efforts=(e_pL, e_sL), contract=contract_menu['L'], theta=params['theta_L']
                )
                self.assertAlmostEqual(utility_L, 0.0, places=5, msg="Low-type IR constraint not binding.")

                # 3. Verify binding IC-H constraint (High type's utility should equal mimicking utility)
                utility_H_actual = self._calculate_utility_manually(
                    efforts=(e_pH, e_sH), contract=contract_menu['H'], theta=params['theta_H']
                )
                utility_H_mimic = self._calculate_utility_manually(
                    efforts=(e_pL, e_sL), contract=contract_menu['L'], theta=params['theta_H'] # Note: theta_H is used
                )
                self.assertAlmostEqual(utility_H_actual, utility_H_mimic, places=5, msg="High-type IC constraint not binding.")

    def _calculate_utility_manually(self, efforts, contract, theta):
        """Helper to calculate expected utility without an Agent instance."""
        e_p, e_s = efforts
        cost = (e_p**2 / 2) + (e_s**2 / (2 * theta))
        p_pi_H = prob_high_performance_signal(e_p)
        p_q_G = prob_good_safety_outcome(e_s)
        
        expected_payment = (
            p_q_G * p_pi_H * contract['G']['H'] +
            p_q_G * (1 - p_pi_H) * contract['G']['L'] +
            (1 - p_q_G) * p_pi_H * contract['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * contract['B']['L']
        )
        return expected_payment - cost

if __name__ == '__main__':
    unittest.main()