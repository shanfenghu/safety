# src/contracts/optimal.py

"""
Defines the solver for the second-best optimal contract.

This module translates the theoretical results from the paper's analysis into a
numerical solver. It implements the PIPS (Projected Incentive-Preserving Shift)
algorithm to find a robust, non-negative, and incentive-compatible contract menu.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize

from src.types import Contract
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal


def _solve_for_optimal_efforts(params: Dict) -> Dict:
    """
    Solves for the optimal second-best effort levels by directly maximizing
    the Principal's objective function from the theoretical proofs.
    
    This is an internal helper function.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A dictionary containing the optimal efforts for high and low types.
    """
    # Unpack parameters for clarity
    theta_L, theta_H, nu, delta_W = params['theta_L'], params['theta_H'], params['nu'], params['delta_W']

    # Define the Principal's objective function to be maximized.
    # This is the expected social surplus minus the expected information rent.
    def principal_objective(efforts: np.ndarray) -> float:
        e_pH, e_pL, e_sH, e_sL = efforts

        # Calculate social surplus for each type
        # Note: We subtract cost of welfare, which is W_G - W_B, so we add W_B back for total welfare
        # For simplicity in optimization, we can treat W_G as 0, making W_B = -delta_W.
        # Expected welfare is (1-p)*W_G + p*W_B = -p*delta_W.
        surplus_H = - (1 - prob_good_safety_outcome(e_sH)) * delta_W - (e_pH**2 / 2.0) - (e_sH**2 / (2.0 * theta_H))
        surplus_L = - (1 - prob_good_safety_outcome(e_sL)) * delta_W - (e_pL**2 / 2.0) - (e_sL**2 / (2.0 * theta_L))

        # Calculate the information rent paid to the high-type agent
        rent = (e_sL**2 / 2.0) * (1/theta_L - 1/theta_H)

        # The objective is the expected total surplus minus the expected rent
        total_welfare = nu * surplus_H + (1 - nu) * surplus_L - nu * rent
        
        # We use a minimizer, so we return the negative of the welfare
        return -total_welfare

    # --- Run the optimization ---
    initial_guess = np.array([0.15, 0.10, 5.0, 4.0]) # [e_pH, e_pL, e_sH, e_sL]
    bounds = [(0, None), (0, None), (0, None), (0, None)]
    
    result = minimize(
        fun=principal_objective,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    e_pH_opt, e_pL_opt, e_sH_opt, e_sL_opt = result.x if result.success else (0,0,0,0)

    return {
        'H': {'ep': e_pH_opt, 'es': e_sH_opt},
        'L': {'ep': e_pL_opt, 'es': e_sL_opt}
    }


def _solve_for_payments(optimal_efforts: Dict, params: Dict) -> Dict[str, Contract]:
    """
    Calculates the 8 payment values for the contract menu using the PIPS algorithm.
    
    This is an internal helper function.
    """
    # Unpack parameters
    theta_L, theta_H = params['theta_L'], params['theta_H']
    e_pL, e_sL = optimal_efforts['L']['ep'], optimal_efforts['L']['es']
    
    def get_contract_for_type(ep: float, es: float, theta: float, target_utility: float) -> Contract:
        """
        Implements the PIPS algorithm for a single agent type.
        """
        
        # 1. Calculate required payment spreads from the agent's FOCs
        delta_t_q = (es / theta) / np.exp(-es) if es > 0 else 0
        delta_t_pi = ep / np.exp(-ep) if ep > 0 else 0

        # 2. Construct a base contract with these spreads
        base_contract = {
            'G': {'H': delta_t_q + delta_t_pi, 'L': delta_t_q},
            'B': {'H': delta_t_pi,              'L': 0.0}
        }

        # 3. Calculate the expected utility this base contract would provide
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

        # 4. Calculate the additive constant 'k' to shift utility to the target
        k = target_utility - base_utility
        
        # 5. Apply the shift to all payments
        t_shifted: Contract = {
            q: {p: payment + k for p, payment in p_dict.items()}
            for q, p_dict in base_contract.items()
        }

        # 6. Project to non-negative space
        min_payment = min(p for outcomes in t_shifted.values() for p in outcomes.values())
        if min_payment < 0:
            shift = -min_payment
            for q_key in t_shifted:
                for p_key in t_shifted[q_key]:
                    t_shifted[q_key][p_key] += shift
        
        return t_shifted

    # --- Calculate Contract for Low-Quality Type ---
    contract_L = get_contract_for_type(e_pL, e_sL, theta_L, target_utility=0.0)

    # --- Calculate Contract for High-Quality Type ---
    rent = (e_sL**2 / 2) * (1/theta_L - 1/theta_H)
    e_pH, e_sH = optimal_efforts['H']['ep'], optimal_efforts['H']['es']
    contract_H = get_contract_for_type(e_pH, e_sH, theta_H, target_utility=rent)
    
    return {'H': contract_H, 'L': contract_L}


def calculate_optimal_contract(params: Dict) -> Tuple[Dict[str, Contract], Dict]:
    """
    Public interface for the optimal contract solver.
    """
    optimal_efforts = _solve_for_optimal_efforts(params)
    contract_menu = _solve_for_payments(optimal_efforts, params)
    return contract_menu, optimal_efforts

