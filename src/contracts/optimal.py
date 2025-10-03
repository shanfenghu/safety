# src/contracts/optimal.py

"""
Defines the solver for the second-best optimal contract.

This module implements a nested optimization to find the true optimal contract.
The outer layer finds the optimal effort levels by maximizing the Principal's
true utility, while the inner layer (the PIPS algorithm) calculates the
cost of implementing those efforts under all constraints.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize

from src.types import Contract
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal


def _calculate_mimicking_utility(
    contract_L: Contract,
    mimic_theta: float,
    true_theta: float
) -> float:
    """
    Calculates the maximum utility a high-quality agent can get by taking
    the low-quality agent's contract.

    This is an internal helper function that simulates the "mimicking" scenario
    by running an optimization from the high-quality agent's perspective.

    Args:
        contract_L: The contract intended for the low-quality agent.
        mimic_theta: The theta of the agent choosing the efforts (i.e., theta_H).
        true_theta: The theta used in the cost function for the efforts chosen (i.e., theta_H)

    Returns:
        The maximum achievable utility for the mimicking agent.
    """

    # Define the utility function from the mimicking agent's perspective
    def utility_func(efforts: np.ndarray) -> float:
        e_p, e_s = efforts
        cost = (e_p**2 / 2.0) + (e_s**2 / (2.0 * mimic_theta))
        
        p_pi_H = prob_high_performance_signal(e_p)
        p_q_G = prob_good_safety_outcome(e_s)
        
        expected_payment = (
            p_q_G * p_pi_H * contract_L['G']['H'] +
            p_q_G * (1 - p_pi_H) * contract_L['G']['L'] +
            (1 - p_q_G) * p_pi_H * contract_L['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * contract_L['B']['L']
        )
        return expected_payment - cost

    # The optimizer minimizes the *negative* utility
    objective_func = lambda e: -utility_func(e)
    
    # Run the optimization to find the best efforts the mimic can choose
    result = minimize(
        fun=objective_func,
        x0=np.array([0.1, 1.0]),
        bounds=[(0, None), (0, None)],
        method='L-BFGS-B'
    )
    
    # The maximum utility is the negative of the minimized value
    return -result.fun if result.success else 0.0


def _solve_for_payments(optimal_efforts: Dict, params: Dict) -> Dict[str, Contract]:
    """
    Calculates the 8 payment values for the contract menu using the PIPS algorithm.
    This version correctly calculates the true information rent.
    
    This is an internal helper function.
    """
    # Unpack parameters
    theta_L, theta_H = params['theta_L'], params['theta_H']
    e_pL, e_sL = optimal_efforts['L']['ep'], optimal_efforts['L']['es']
    
    def get_contract_for_type(ep: float, es: float, theta: float, target_utility: float) -> Contract:
        """
        Implements the PIPS algorithm for a single agent type.
        """
        delta_t_q = (es / theta) / np.exp(-es) if es > 0 else 0
        delta_t_pi = ep / np.exp(-ep) if ep > 0 else 0
        base_contract = {
            'G': {'H': delta_t_q + delta_t_pi, 'L': delta_t_q},
            'B': {'H': delta_t_pi,              'L': 0.0}
        }
        cost = (ep**2 / 2.0) + (es**2 / (2.0 * theta))
        p_q_G = prob_good_safety_outcome(es)
        p_pi_H = prob_high_performance_signal(ep)
        expected_base_payment = (
            p_q_G * p_pi_H * base_contract['G']['H'] +
            p_q_G * (1 - p_pi_H) * base_contract['G']['L'] +
            (1 - p_q_G) * p_pi_H * base_contract['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * base_contract['B']['L']
        )
        base_utility = expected_base_payment - cost
        k = target_utility - base_utility
        t_shifted = {
            q: {p: payment + k for p, payment in p_dict.items()}
            for q, p_dict in base_contract.items()
        }
        min_payment = min(p for outcomes in t_shifted.values() for p in outcomes.values())
        if min_payment < 0:
            shift = -min_payment
            for q_key in t_shifted:
                for p_key in t_shifted[q_key]:
                    t_shifted[q_key][p_key] += shift
        return t_shifted

    # --- Stage 1: Calculate the Low-Quality Type's Contract ---
    # The target utility is 0 because the IR-L constraint is binding.
    contract_L = get_contract_for_type(e_pL, e_sL, theta_L, target_utility=0.0)

    # --- Stage 2: Calculate the True Information Rent ---
    # We simulate what a high-type agent would do with the low-type contract
    # to find their true mimicking utility.
    true_information_rent = _calculate_mimicking_utility(
        contract_L=contract_L,
        mimic_theta=theta_H, # The agent's type
        true_theta=theta_H   # The agent's true cost function
    )
    
    # --- Stage 3: Calculate the High-Quality Type's Contract ---
    # The target utility is now the true, correctly calculated information rent.
    e_pH, e_sH = optimal_efforts['H']['ep'], optimal_efforts['H']['es']
    contract_H = get_contract_for_type(e_pH, e_sH, theta_H, target_utility=true_information_rent)
    
    return {'H': contract_H, 'L': contract_L}


def _solve_for_optimal_efforts(params: Dict) -> Dict:
    """
    Solves for the true optimal second-best effort levels by directly
    maximizing the Principal's objective function.
    
    This is an internal helper function that uses a nested optimization.
    """
    # Unpack parameters for clarity
    nu, delta_W = params['nu'], params['delta_W']

    # --- Define the Principal's Objective Function ---
    # This function calculates the Principal's total expected utility for a
    # given set of four effort levels.
    def principal_objective(efforts: np.ndarray) -> float:
        e_pH, e_pL, e_sH, e_sL = efforts
        
        # Create a temporary effort dictionary to pass to the payment solver
        temp_efforts = {
            'H': {'ep': e_pH, 'es': e_sH},
            'L': {'ep': e_pL, 'es': e_sL}
        }
        
        # --- Inner Optimization: Find the cost of the contract ---
        # For these efforts, find the best possible non-negative contract
        contract_menu = _solve_for_payments(temp_efforts, params)

        # --- Calculate the Principal's Expected Utility ---
        # Expected Social Welfare (avoids disaster)
        welfare_H = (1 - prob_good_safety_outcome(e_sH)) * -delta_W
        welfare_L = (1 - prob_good_safety_outcome(e_sL)) * -delta_W
        expected_welfare = nu * welfare_H + (1 - nu) * welfare_L

        # Expected Payments (cost of the contracts)
        p_pi_H_H = prob_high_performance_signal(e_pH); p_q_G_H = prob_good_safety_outcome(e_sH)
        p_pi_H_L = prob_high_performance_signal(e_pL); p_q_G_L = prob_good_safety_outcome(e_sL)
        
        c_H = contract_menu['H']
        exp_payment_H = (p_q_G_H * p_pi_H_H * c_H['G']['H'] + p_q_G_H * (1-p_pi_H_H) * c_H['G']['L'] +
                         (1-p_q_G_H) * p_pi_H_H * c_H['B']['H'] + (1-p_q_G_H) * (1-p_pi_H_H) * c_H['B']['L'])
        
        c_L = contract_menu['L']
        exp_payment_L = (p_q_G_L * p_pi_H_L * c_L['G']['H'] + p_q_G_L * (1-p_pi_H_L) * c_L['G']['L'] +
                         (1-p_q_G_L) * p_pi_H_L * c_L['B']['H'] + (1-p_q_G_L) * (1-p_pi_H_L) * c_L['B']['L'])

        expected_payments = nu * exp_payment_H + (1 - nu) * exp_payment_L
        
        # The optimizer's goal is to maximize this utility, so we return its negative
        return -(expected_welfare - expected_payments)

    # --- Run the Outer Optimization ---
    # Use the provided initial guess if available, otherwise use a default
    initial_guess = params.get('initial_guess', np.array([0.15, 0.10, 5.0, 4.0]))
    bounds = [(0, None), (0, None), (0, None), (0, None)]
    
    result = minimize(
        fun=principal_objective,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    e_pH_opt, e_pL_opt, e_sH_opt, e_sL_opt = result.x if result.success else initial_guess

    return {
        'H': {'ep': e_pH_opt, 'es': e_sH_opt},
        'L': {'ep': e_pL_opt, 'es': e_sL_opt},
        'solver_solution': result.x # Also return the raw solution vector for the warm start
    }


def calculate_optimal_contract(params: Dict) -> Tuple[Dict[str, Contract], Dict]:
    """
    Calculates the menu for the theoretically optimal second-best contract.

    This function serves as the public interface for the optimal contract solver.
    It orchestrates the process of first solving for the true optimal effort
    levels, and then solving for the payments required to implement them.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A tuple containing:
        - The contract menu (a dict with keys 'H' and 'L').
        - The dictionary of the optimal effort levels solved for.
    """
    optimal_efforts = _solve_for_optimal_efforts(params)
    contract_menu = _solve_for_payments(optimal_efforts, params)
    return contract_menu, optimal_efforts