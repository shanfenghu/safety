# tests/contracts/test_heuristics.py

"""
Unit tests for the heuristic contract functions in src/contracts/heuristics.py.
"""

import unittest
from src.contracts.heuristics import (
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract
)

class TestHeuristicContracts(unittest.TestCase):

    def test_naive_fine_contract(self):
        """
        Tests the create_naive_fine_contract function for correctness and validation.
        """
        # Test 1: Correctness with a valid input
        bonus = 100.0
        contract = create_naive_fine_contract(safety_bonus=bonus)
        
        self.assertEqual(contract['G']['H'], bonus)
        self.assertEqual(contract['G']['L'], bonus)
        self.assertEqual(contract['B']['H'], 0.0)
        self.assertEqual(contract['B']['L'], 0.0)

        # Test 2: Input validation for negative values
        with self.assertRaises(AssertionError):
            create_naive_fine_contract(safety_bonus=-50.0)

    def test_performance_contract(self):
        """
        Tests the create_performance_contract function for correctness and validation.
        """
        # Test 1: Correctness with a valid input
        bonus = 50.0
        contract = create_performance_contract(performance_bonus=bonus)
        
        self.assertEqual(contract['G']['H'], bonus)
        self.assertEqual(contract['B']['H'], bonus)
        self.assertEqual(contract['G']['L'], 0.0)
        self.assertEqual(contract['B']['L'], 0.0)

        # Test 2: Input validation for negative values
        with self.assertRaises(AssertionError):
            create_performance_contract(performance_bonus=-50.0)

    def test_hybrid_contract(self):
        """
        Tests the create_hybrid_contract function for correctness and validation.
        """
        # Test 1: Correctness with valid inputs
        safety_bonus = 100.0
        performance_bonus = 50.0
        contract = create_hybrid_contract(
            safety_bonus=safety_bonus, 
            performance_bonus=performance_bonus
        )
        
        # Check all four possible outcomes
        self.assertEqual(contract['G']['H'], safety_bonus + performance_bonus)
        self.assertEqual(contract['G']['L'], safety_bonus)
        self.assertEqual(contract['B']['H'], performance_bonus)
        self.assertEqual(contract['B']['L'], 0.0)

        # Test 2: Input validation for negative values
        with self.assertRaises(AssertionError):
            create_hybrid_contract(safety_bonus=-100.0, performance_bonus=50.0)
        with self.assertRaises(AssertionError):
            create_hybrid_contract(safety_bonus=100.0, performance_bonus=-50.0)


if __name__ == '__main__':
    unittest.main()