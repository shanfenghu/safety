# src/model.py

"""
Defines the Mesa Model for the environmental AI safety simulation.

This module contains the `SafetyModel` class, which sets up the simulation,
manages the agents, controls the step-by-step execution of the game, and
collects all relevant data for analysis.
"""

import mesa
from typing import Dict, Type

# Updated imports to reflect the new modular structure
from src.agents.base_developer import BaseDeveloperAgent
from src.agents.regulator import RegulatorAgent
from src.simulation_utils import prob_high_performance_signal, prob_good_safety_outcome
from src.contracts.heuristics import (
    create_naive_fine_contract,
    create_performance_contract,
    create_hybrid_contract
)
from src.contracts.optimal import calculate_optimal_contract


class SafetyModel(mesa.Model):
    """The main model for the AI safety regulation game."""

    def __init__(self, params: Dict):
        """
        Create a new SafetyModel.

        Args:
            params: A dictionary of parameters for the simulation run. It must
                    contain a 'developer_class' key specifying which agent to use.
        """
        super().__init__()
        self.params = params
        self.running = True
        self.max_steps = self.params.get('max_steps', 1) # Default to 1 for our single-shot experiments
        self.correlation_k = self.params.get('correlation_k', 0.0)

        # --- Setup Agents and Scheduler ---
        self.schedule = mesa.time.BaseScheduler(self)

        # Create the Regulator agent (its class is fixed)
        self.regulator = RegulatorAgent(unique_id=0, model=self, contract_menu={})
        self.schedule.add(self.regulator)

        # Determine the developer's private type stochastically
        is_high_type = self.random.random() < self.params['nu']
        dev_type_key = 'H' if is_high_type else 'L'
        theta = self.params['theta_H'] if is_high_type else self.params['theta_L']
        
        # Create the Developer agent using the class provided in the parameters
        # This makes the model flexible to handle any developer type
        developer_class: Type[BaseDeveloperAgent] = self.params['developer_class']
        developer_params = self.params.get('developer_params', {})
        self.developer = developer_class(
            unique_id=1, 
            model=self, 
            theta=theta, 
            **developer_params
        )
        self.schedule.add(self.developer)
        
        # --- Contract Selection Logic ---
        # The regulator "offers" the contract menu appropriate for the experiment
        contract_type = self.params.get('contract_type', 'optimal')
        
        if contract_type == 'pre_calculated_optimal':
            # This special type allows experiments to bypass the slow solver by
            # using a contract menu that has been passed in directly.
            menu = self.params['pre_calculated_menu']
            self.regulator.contract_menu = menu
            self.developer.contract_offer = self.regulator.contract_menu[dev_type_key]
        elif contract_type == 'optimal':
            # Call the solver for the risk-neutral case
            menu, _ = calculate_optimal_contract(self.params, risk_averse=False)
            self.regulator.contract_menu = menu
            self.developer.contract_offer = self.regulator.contract_menu[dev_type_key]
        elif contract_type == 'optimal_ra':
            # Call the SAME solver but for the risk-averse case
            menu, _ = calculate_optimal_contract(self.params, risk_averse=True)
            self.regulator.contract_menu = menu
            self.developer.contract_offer = self.regulator.contract_menu[dev_type_key]
        else:
            if contract_type == 'fine':
                c = create_naive_fine_contract(self.params['safety_bonus'])
            elif contract_type == 'performance':
                c = create_performance_contract(self.params['performance_bonus'])
            elif contract_type == 'hybrid':
                c = create_hybrid_contract(self.params['safety_bonus'], self.params['performance_bonus'])
            else:
                raise ValueError(f"Unknown contract type: {contract_type}")
            
            self.regulator.contract_menu = {'H': c, 'L': c}
            self.developer.contract_offer = c

        # --- State variables for data collection ---
        self.disaster_occurred = False
        self.social_welfare = 0.0

        # --- Setup DataCollector ---
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "DisasterOccurred": "disaster_occurred",
                "SocialWelfare": "social_welfare",
            },
            agent_reporters={
                "Theta": "theta",
                "ChosenEp": "chosen_ep",
                "ChosenEs": "chosen_es",
                "Payoff": "payoff"
            }
        )
        # Note: Initial state is collected at the end of __init__ automatically by Mesa's batch runner

    def step(self):
        """Executes one round of the game."""
        # 1. Agents choose their actions (only the developer has a complex step method)
        self.schedule.step()

        # 2. Realize outcomes based on the developer's chosen efforts
        e_p, e_s = self.developer.chosen_ep, self.developer.chosen_es
        pi_outcome = 'H' if self.random.random() < prob_high_performance_signal(e_p) else 'L'
        # q_outcome = 'G' if self.random.random() < prob_good_safety_outcome(e_s) else 'B'
        effective_es = e_s + self.correlation_k * e_p
        q_outcome = 'G' if self.random.random() < prob_good_safety_outcome(max(0, effective_es)) else 'B'
        self.disaster_occurred = (q_outcome == 'B')

        # 3. Calculate and assign payoffs
        payment = self.developer.contract_offer[q_outcome][pi_outcome]
        self.developer.payoff = payment

        welfare = 0 if q_outcome == 'G' else -self.params['delta_W']
        self.social_welfare = welfare - payment

        # 4. Allow learning agents to update their policy
        self.developer.learn(reward=self.developer.payoff)
        
        # 5. Collect data for this step
        self.datacollector.collect(self)

        # 6. Check for stopping condition in multi-step runs
        if self.schedule.steps >= self.max_steps:
            self.running = False