# src/agents/learning.py

"""
Defines the LearningDeveloperAgent class for the simulation.
"""

from typing import List, Tuple
import mesa
import numpy as np

from src.agents.base_developer import BaseDeveloperAgent


class LearningDeveloperAgent(BaseDeveloperAgent):
    """
    An agent that uses Q-learning to choose its effort levels.

    This agent represents a boundedly rational developer who does not know the
    underlying model. It learns which action to take through trial-and-error,
    updating its Q-table based on the payoffs it receives.
    """

    def __init__(self, unique_id: int, model: mesa.Model, theta: float,
                 action_space: List[Tuple[float, float]], alpha: float, epsilon: float):
        """
        Creates a new LearningDeveloperAgent.

        Args:
            unique_id: The agent's unique identifier.
            model: The model instance the agent belongs to.
            theta: The agent's private type (AI quality).
            action_space: A list of (e_p, e_s) tuples representing the discrete actions.
            alpha: The learning rate for the Q-learning algorithm.
            epsilon: The exploration rate for the epsilon-greedy policy.
        """
        super().__init__(unique_id, model, theta)

        # --- Parameter Validation ---
        assert 0 <= alpha <= 1, "Learning rate alpha must be in [0, 1]."
        assert 0 <= epsilon <= 1, "Exploration rate epsilon must be in [0, 1]."
        
        # --- Learning-Specific Attributes ---
        self.action_space = action_space
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize the Q-table with zeros for each action
        self.q_table = np.zeros(len(self.action_space))
        
        # Keep track of the last action taken to update the correct Q-value
        self.last_action_index: int = None


    def step(self):
        """
        Chooses an action (effort pair) using an epsilon-greedy policy.
        """
        # --- Epsilon-Greedy Action Selection ---
        if self.model.random.random() < self.epsilon:
            # Explore: choose a random action
            action_index = self.model.random.integers(len(self.action_space))
        else:
            # Exploit: choose the best-known action, breaking ties randomly
            best_q_value = self.q_table.max()
            best_actions = np.flatnonzero(self.q_table == best_q_value)
            action_index = self.model.random.choice(best_actions)
            
        # Store the chosen action index for the learning step
        self.last_action_index = action_index
        
        # Set the chosen efforts based on the selected action
        self.chosen_ep, self.chosen_es = self.action_space[action_index]

    def learn(self, reward: float):
        """
        Updates the Q-table based on the reward received.

        This method implements a simple Q-learning update rule where the
        discount factor (gamma) is 0, as each simulation step is an
        independent game.

        Args:
            reward: The payoff received from the last action.
        """
        if self.last_action_index is None:
            return # Cannot learn if no action has been taken
        
        # Q-learning update rule: Q(a) <- Q(a) + alpha * (reward - Q(a))
        old_q_value = self.q_table[self.last_action_index]
        self.q_table[self.last_action_index] = old_q_value + self.alpha * (reward - old_q_value)