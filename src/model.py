# src/model.py

"""
Defines the Mesa Model for the environmental AI safety simulation.

This module contains the `SafetyModel` class, which sets up the simulation,
manages the agents, controls the step-by-step execution of the game, and
collects all relevant data for analysis.
"""

import mesa
import numpy as np
from typing import Dict

from src.agents import RegulatorAgent, DeveloperAgent, prob_high_performance_signal, prob_good_safety_outcome
from src.contracts import (
    Contract,
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract,
    calculate_optimal_contract
)


class SafetyModel(mesa.Model):
    """The main model for the AI safety regulation game."""

    def __init__(self, params: dict):
        """
        Create a new SafetyModel.

        Args:
            params: A dictionary of parameters for the simulation run.
                    Expected keys: 'nu', 'theta_L', 'theta_H', 'delta_W',
                    'contract_type', 'safety_bonus', 'performance_bonus'.
        """
        super().__init__()
        self.params = params
        self.running = True

        # --- Setup Agents and Scheduler ---
        self.schedule = mesa.time.BaseScheduler(self)

        # Create the Regulator agent
        self.regulator = RegulatorAgent(unique_id=0, model=self, contract_menu={})
        self.schedule.add(self.regulator)

        # Create the Developer agent with a stochastically assigned type
        is_high_type = self.random.random() < self.params['nu']
        dev_type = 'H' if is_high_type else 'L'
        theta = self.params['theta_H'] if is_high_type else self.params['theta_L']
        
        self.developer = DeveloperAgent(unique_id=1, model=self, theta=theta)
        self.schedule.add(self.developer)
        
        # --- Contract Selection Logic ---
        # The regulator "offers" the contract menu appropriate for the experiment
        contract_type = self.params.get('contract_type', 'optimal')
        
        if contract_type == 'optimal':
            # For the optimal contract, the agent gets the menu and chooses
            # (or is assigned, per the Revelation Principle) the correct one.
            self.regulator.contract_menu = calculate_optimal_contract(self.params)
            self.developer.contract_offer = self.regulator.contract_menu[dev_type]
        else:
            # For heuristic contracts, the menu is simple (both types get same contract)
            if contract_type == 'fine':
                heuristic_contract = create_naive_fine_contract(self.params['safety_bonus'])
            elif contract_type == 'performance':
                heuristic_contract = create_performance_contract(self.params['performance_bonus'])
            elif contract_type == 'hybrid':
                heuristic_contract = create_hybrid_contract(self.params['safety_bonus'], self.params['performance_bonus'])
            else:
                raise ValueError(f"Unknown contract type: {contract_type}")
            
            self.regulator.contract_menu = {'H': heuristic_contract, 'L': heuristic_contract}
            self.developer.contract_offer = heuristic_contract

        # --- State variables for data collection ---
        self.disaster_occurred = False
        self.social_welfare = 0.0

        # --- Setup DataCollector ---
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "DisasterOccurred": "disaster_occurred",
                "SocialWelfare": "social_welfare",
                "ContractType": lambda m: m.params.get('contract_type'),
            },
            agent_reporters={
                "Theta": "theta",
                "ChosenEp": "chosen_ep",
                "ChosenEs": "chosen_es",
                "Payoff": "payoff"
            }
        )
        self.datacollector.collect(self) # Collect initial state

    def step(self):
        """
        Executes one round of the game.
        """
        # --- 1. Agent Action ---
        # The scheduler activates the developer's step() method, where they
        # observe their contract and choose their optimal efforts.
        self.schedule.step()

        # --- 2. Realize Outcomes ---
        # Determine outcomes based on the developer's chosen efforts
        e_p = self.developer.chosen_ep
        e_s = self.developer.chosen_es

        # Performance signal outcome
        p_pi_H = prob_high_performance_signal(e_p)
        pi_outcome = 'H' if self.random.random() < p_pi_H else 'L'

        # Safety outcome
        p_q_G = prob_good_safety_outcome(e_s)
        q_outcome = 'G' if self.random.random() < p_q_G else 'B'

        self.disaster_occurred = (q_outcome == 'B')

        # --- 3. Calculate Payoffs ---
        # Look up the payment from the developer's contract
        payment = self.developer.contract_offer[q_outcome][pi_outcome]
        self.developer.payoff = payment

        # Calculate social welfare for the round
        welfare = 0 if q_outcome == 'G' else -self.params['delta_W']
        self.social_welfare = welfare - payment

        # --- 4. Collect Data ---
        self.datacollector.collect(self)