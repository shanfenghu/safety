# tests/agents/test_learning.py

"""
Unit tests for the LearningDeveloperAgent class in src/agents/learning.py.
"""

import unittest
import mesa
import numpy as np

from src.agents.learning import LearningDeveloperAgent
from src.contracts.heuristics import create_performance_contract
from src.simulation_utils import prob_high_performance_signal

class DummyModel(mesa.Model):
    """A dummy model for testing that includes a random number generator."""
    def __init__(self):
        super().__init__()
        self.random = np.random.default_rng()

class TestLearningDeveloperAgent(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for the learning agent tests."""
        self.model = DummyModel()
        self.theta = 1.0
        self.action_space = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0)]
        self.learning_params = {'alpha': 0.1, 'epsilon': 0.1}

    def test_initialization(self):
        """Tests that the learning agent initializes correctly."""
        agent = LearningDeveloperAgent(
            unique_id=1, model=self.model, theta=self.theta,
            action_space=self.action_space, **self.learning_params
        )
        self.assertEqual(agent.alpha, self.learning_params['alpha'])
        self.assertEqual(agent.epsilon, self.learning_params['epsilon'])
        self.assertEqual(agent.q_table.shape, (len(self.action_space),))
        self.assertTrue(np.all(agent.q_table == 0))

    def test_learn_method_update_rule(self):
        """
        Unit tests the Q-table update rule in isolation to verify the math.
        """
        agent = LearningDeveloperAgent(
            unique_id=1, model=self.model, theta=self.theta,
            action_space=self.action_space, alpha=0.1, epsilon=0.1
        )
        action_index_to_test = 2
        agent.last_action_index = action_index_to_test
        
        # First update from zero
        reward1 = 10.0
        expected_q1 = 0.0 + 0.1 * (reward1 - 0.0) # = 1.0
        agent.learn(reward=reward1)
        self.assertAlmostEqual(agent.q_table[action_index_to_test], expected_q1)

        # Second update from a non-zero value
        reward2 = 5.0
        expected_q2 = expected_q1 + 0.1 * (reward2 - expected_q1) # = 1.0 + 0.1 * 4.0 = 1.4
        agent.learn(reward=reward2)
        self.assertAlmostEqual(agent.q_table[action_index_to_test], expected_q2)

    def test_action_selection_exploit(self):
        """
        Unit tests the 'exploit' part of the epsilon-greedy policy.
        """
        agent = LearningDeveloperAgent(
            unique_id=1, model=self.model, theta=self.theta,
            action_space=self.action_space, alpha=0.1, epsilon=0.0 # Epsilon = 0 means pure exploitation
        )
        # Manually set the Q-table to have a clear best action
        agent.q_table = np.array([10.0, 20.0, 50.0, 30.0])
        best_action_index = 2
        
        agent.step()
        
        self.assertEqual(agent.last_action_index, best_action_index)

    def test_full_learning_process_statistical(self):
        """
        A statistical test to verify that the agent learns the best action over time.
        """
        # --- Arrange ---
        # Create a simple environment with a clearly optimal action
        action_space = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)] # e_p levels
        optimal_action_index = 2 # The action with the highest e_p

        # A performance contract makes higher e_p more profitable
        contract = create_performance_contract(performance_bonus=100.0)
        
        agent = LearningDeveloperAgent(
            unique_id=1, model=self.model, theta=self.theta,
            action_space=action_space, alpha=0.1, epsilon=0.1
        )
        agent.contract_offer = contract

        # --- Act ---
        # Run a training loop for N steps
        num_steps = 1000
        for _ in range(num_steps):
            # 1. Agent chooses an action
            agent.step()
            
            # 2. We calculate the reward for that action
            e_p, e_s = agent.action_space[agent.last_action_index]
            cost = (e_p**2 / 2.0) + (e_s**2 / (2.0 * agent.theta))
            expected_payment = prob_high_performance_signal(e_p) * contract['G']['H']
            reward = expected_payment - cost
            
            # 3. Agent learns from the reward
            agent.learn(reward=reward)

        # --- Assert ---
        # After many steps, the Q-value for the optimal action should be the highest.
        self.assertEqual(
            np.argmax(agent.q_table), 
            optimal_action_index,
            "Agent failed to learn the optimal action over time."
        )

if __name__ == '__main__':
    unittest.main()