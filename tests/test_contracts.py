# tests/test_contracts.py

"""
Unit tests for the contract generation functions in src/contracts.py.

This suite ensures that each contract function:
1.  Returns the correct payment structure for valid inputs.
2.  Raises errors for invalid inputs (e.g., negative bonuses).
3.  Adheres to the model's fundamental constraints, such as Limited Liability.
"""

import unittest
from src.contracts import (
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract,
    calculate_optimal_contract
)

class TestContracts(unittest.TestCase):

    def test_naive_fine_contract(self):
        """
        Tests the create_naive_fine_contract function for correctness and validation.
        """
        # --- Test 1: Correctness with valid input ---
        bonus = 100.0
        contract = create_naive_fine_contract(safety_bonus=bonus)
        
        # A good safety outcome ('G') should yield the bonus, regardless of performance
        self.assertEqual(contract['G']['H'], bonus)
        self.assertEqual(contract['G']['L'], bonus)
        
        # A bad safety outcome ('B') should yield zero payment
        self.assertEqual(contract['B']['H'], 0.0)
        self.assertEqual(contract['B']['L'], 0.0)

        # --- Test 2: Parameter validation with invalid input ---
        with self.assertRaises(AssertionError, msg="Should fail for negative bonus"):
            create_naive_fine_contract(safety_bonus=-50.0)

    def test_performance_contract(self):
        """
        Tests the create_performance_contract function for correctness and validation.
        """
        # --- Test 1: Correctness with valid input ---
        bonus = 50.0
        contract = create_performance_contract(performance_bonus=bonus)
        
        # A high performance signal ('H') should yield the bonus, regardless of safety
        self.assertEqual(contract['G']['H'], bonus)
        self.assertEqual(contract['B']['H'], bonus)
        
        # A low performance signal ('L') should yield zero payment
        self.assertEqual(contract['G']['L'], 0.0)
        self.assertEqual(contract['B']['L'], 0.0)

        # --- Test 2: Parameter validation with invalid input ---
        with self.assertRaises(AssertionError, msg="Should fail for negative bonus"):
            create_performance_contract(performance_bonus=-50.0)

    def test_hybrid_contract(self):
        """
        Tests the create_hybrid_contract function for correctness and validation.
        """
        # --- Test 1: Correctness with valid inputs ---
        safety_bonus = 100.0
        performance_bonus = 50.0
        contract = create_hybrid_contract(safety_bonus=safety_bonus, performance_bonus=performance_bonus)
        
        # Check all four possible outcomes
        self.assertEqual(contract['G']['H'], safety_bonus + performance_bonus)
        self.assertEqual(contract['G']['L'], safety_bonus)
        self.assertEqual(contract['B']['H'], performance_bonus)
        self.assertEqual(contract['B']['L'], 0.0)

        # --- Test 2: Parameter validation with invalid inputs ---
        with self.assertRaises(AssertionError, msg="Should fail for negative safety_bonus"):
            create_hybrid_contract(safety_bonus=-100.0, performance_bonus=50.0)
        with self.assertRaises(AssertionError, msg="Should fail for negative performance_bonus"):
            create_hybrid_contract(safety_bonus=100.0, performance_bonus=-50.0)

    def test_optimal_contract_placeholder(self):
        """
        Tests the placeholder implementation of calculate_optimal_contract.
        """
        # --- Test 1: Correct structure and placeholder values ---
        dummy_params = {'theta_L': 1.0, 'theta_H': 2.0, 'nu': 0.5, 'delta_W': 1000}
        contract_menu = calculate_optimal_contract(params=dummy_params)

        # Check that it returns a menu for both types
        self.assertIn('H', contract_menu)
        self.assertIn('L', contract_menu)

        # Check that the returned contracts have the correct structure
        self.assertIsInstance(contract_menu['H'], dict)
        self.assertIn('G', contract_menu['L'])
        self.assertIn('H', contract_menu['L']['B'])
        
        # --- Test 2: Parameter validation ---
        incomplete_params = {'theta_L': 1.0, 'nu': 0.5} # Missing delta_W
        with self.assertRaises(AssertionError, msg="Should fail if params are missing keys"):
            calculate_optimal_contract(params=incomplete_params)

    def test_limited_liability_for_all_contracts(self):
        """
        Ensures all generated contracts respect the t >= 0 constraint.
        """
        # Create one of each contract type
        contracts_to_test = [
            create_naive_fine_contract(100.0),
            create_performance_contract(50.0),
            create_hybrid_contract(100.0, 50.0)
        ]
        # Add the optimal contracts from the menu
        dummy_params = {'theta_L': 1.0, 'theta_H': 2.0, 'nu': 0.5, 'delta_W': 1000}
        optimal_menu = calculate_optimal_contract(params=dummy_params)
        contracts_to_test.extend(optimal_menu.values())

        # Iterate through every payment in every contract and check it's non-negative
        for contract in contracts_to_test:
            for q_outcome in contract.values():
                for payment in q_outcome.values():
                    self.assertGreaterEqual(payment, 0.0, msg=f"Payment {payment} violates limited liability.")


if __name__ == '__main__':
    unittest.main()