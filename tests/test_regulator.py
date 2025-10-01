# tests/agents/test_regulator.py

"""
Unit tests for the RegulatorAgent class in src/agents/regulator.py.
"""

import unittest
import mesa

from src.agents.regulator import RegulatorAgent

# A mock Mesa model is needed to instantiate agents for testing
class DummyModel(mesa.Model):
    """A dummy model for testing purposes."""
    pass

class TestRegulatorAgent(unittest.TestCase):

    def test_initialization(self):
        """
        Tests that the RegulatorAgent initializes correctly and stores its attributes.
        """
        # --- Arrange ---
        # Create dummy objects required for instantiation
        model = DummyModel()
        dummy_menu = {
            'H': {'G': {'H': 1.0, 'L': 1.0}, 'B': {'H': 0.0, 'L': 0.0}},
            'L': {'G': {'H': 1.0, 'L': 1.0}, 'B': {'H': 0.0, 'L': 0.0}},
        }
        
        # --- Act ---
        agent = RegulatorAgent(unique_id=0, model=model, contract_menu=dummy_menu)
        
        # --- Assert ---
        # Check that all attributes were set correctly
        self.assertEqual(agent.unique_id, 0)
        self.assertIs(agent.model, model)
        self.assertDictEqual(agent.contract_menu, dummy_menu)

        # Verify the step method can be called without error
        try:
            agent.step()
        except Exception as e:
            self.fail(f"agent.step() raised an unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()