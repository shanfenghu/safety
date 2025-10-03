# tests/contracts/test_optimal.py

"""
Pure unit tests for the optimal contract solver in src/contracts/optimal.py.

This suite is designed to be a comprehensive and robust validation of the
PIPS (Projected Incentive-Preserving Shift) algorithm. It verifies that the
solver's outputs correctly satisfy all theoretical properties and constraints
across a wide variety of parameter scenarios.
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
            # 2. High Stakes Scenario (likely to be non-implementable)
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 5000.0},
            # 3. Low Type Separation
            {'nu': 0.5, 'theta_L': 1.8, 'theta_H': 2.0, 'delta_W': 1000.0},
            # 4. High Prior for Good Type
            {'nu': 0.8, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0},
            # 5. Low Stakes Scenario (likely to be implementable)
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 100.0},
            # 6. High Type Separation
            {'nu': 0.5, 'theta_L': 1.0, 'theta_H': 5.0, 'delta_W': 1000.0},
            # 7. Low Prior for Good Type
            {'nu': 0.2, 'theta_L': 1.0, 'theta_H': 2.0, 'delta_W': 1000.0},
        ]

    def test_solve_for_optimal_efforts(self):
        """
        Unit test for the _solve_for_optimal_efforts helper function.
        Verifies that the solved efforts satisfy the theoretical FOCs.
        """
        for params in self.param_sets:
            with self.subTest(params=params):
                optimal_efforts = _solve_for_optimal_efforts(params)

                # Sanity checks
                self.assertGreater(optimal_efforts['H']['es'], optimal_efforts['L']['es'])
                self.assertTrue(all(e >= 0 for eff in optimal_efforts.values() for e in eff.values()))

                # Verify FOCs are satisfied
                e_sH, e_sL = optimal_efforts['H']['es'], optimal_efforts['L']['es']
                foc_H = np.exp(-e_sH) * params['delta_W'] - e_sH / params['theta_H']
                foc_L = ((1 - params['nu']) * (np.exp(-e_sL) * params['delta_W'] - e_sL / params['theta_L']) -
                         params['nu'] * (e_sL / params['theta_L'] - e_sL / params['theta_H']))
                self.assertAlmostEqual(foc_H, 0.0, places=3)
                self.assertAlmostEqual(foc_L, 0.0, places=3)

    def test_pips_algorithm_properties(self):
        """
        Unit test for the _solve_for_payments function (PIPS algorithm).
        Verifies that the generated contract satisfies all key theoretical properties.
        """
        for params in self.param_sets:
            with self.subTest(params=params):
                # --- Arrange ---
                optimal_efforts = _solve_for_optimal_efforts(params)
                e_pL, e_sL = optimal_efforts['L']['ep'], optimal_efforts['L']['es']
                
                # --- Act ---
                contract_menu = _solve_for_payments(optimal_efforts, params)

                # --- Assert ---
                # Property 1: Non-Negativity (Limited Liability)
                for contract in contract_menu.values():
                    for payment in contract_menu['L'].values():
                        self.assertGreaterEqual(min(payment.values()), -1e-9)

                # Property 2: Incentive Preservation (Theorem 5.2)
                # We verify this for the low-type agent's contract.
                contract_L = contract_menu['L']
                
                # Calculate the theoretical spreads the contract *should* have
                target_delta_t_q = (e_sL / params['theta_L']) / np.exp(-e_sL) if e_sL > 0 else 0
                target_delta_t_pi = e_pL / np.exp(-e_pL) if e_pL > 0 else 0
                
                # Calculate the actual spreads the final contract *does* have
                actual_delta_t_q = (contract_L['G']['H'] + contract_L['G']['L'])/2 - (contract_L['B']['H'] + contract_L['B']['L'])/2
                actual_delta_t_pi = (contract_L['G']['H'] + contract_L['B']['H'])/2 - (contract_L['G']['L'] + contract_L['B']['L'])/2
                
                self.assertAlmostEqual(actual_delta_t_q, target_delta_t_q, places=5)
                self.assertAlmostEqual(actual_delta_t_pi, target_delta_t_pi, places=5)

                # Property 3: Utility Constraint (Theorem 5.3)
                # The agent's utility must be greater than or equal to their target (0 for low-type).
                # It will be > 0 in the non-implementable "next-best" case.
                utility_L = self._calculate_utility_manually(
                    efforts=(e_pL, e_sL), contract=contract_L, theta=params['theta_L']
                )
                self.assertGreaterEqual(utility_L, -1e-9, msg="Low-type agent's utility cannot be negative.")

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
