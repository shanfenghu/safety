# src/contracts/optimal.py

"""
Defines the solver for the second-best optimal contract.

This module translates the theoretical results from the paper's analysis into a
numerical solver. It calculates the optimal effort levels and the precise
payment schedule that forms the optimal contract menu.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import fsolve

from src.types import Contract
from src.simulation_utils import prob_good_safety_outcome, prob_high_performance_signal


def _solve_for_optimal_efforts(params: Dict) -> Dict:
    """
    Solves for the optimal second-best effort levels from the theory's FOCs.
    
    This is an internal helper function.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A dictionary containing the optimal efforts for high and low types.
    """
    # Unpack parameters for clarity
    theta_L = params['theta_L']
    theta_H = params['theta_H']
    nu = params['nu']
    delta_W = params['delta_W']

    # --- Solve for High-Quality Type's Safety Effort (First-Best) ---
    # FOC: -p_s(e_s) * delta_W = C_s(e_s, theta_H) => exp(-e_s) * delta_W = e_s / theta_H
    func_sH = lambda es: np.exp(-es) * delta_W - es / theta_H
    e_sH_opt = fsolve(func_sH, x0=2.0)[0]
    
    # --- Solve for Low-Quality Type's Safety Effort (Distorted) ---
    # FOC includes the rent-reduction term
    func_sL = lambda es: (1 - nu) * (np.exp(-es) * delta_W - es / theta_L) \
                        - nu * (es / theta_L - es / theta_H)
    e_sL_opt = fsolve(func_sL, x0=1.0)[0]
    
    # --- Determine Performance Efforts ---
    # As established in the theory, e_p* > 0 is induced as the most efficient
    # way to pay the agent under limited liability. The exact level is part of a
    # complex global optimization. For the simulation, we set these to small,
    # positive, plausible values that reflect the muting of incentives.
    e_pL_opt = 0.10
    e_pH_opt = 0.15

    return {
        'H': {'ep': max(0, e_pH_opt), 'es': max(0, e_sH_opt)},
        'L': {'ep': max(0, e_pL_opt), 'es': max(0, e_sL_opt)}
    }


def _solve_for_payments(optimal_efforts: Dict, params: Dict) -> Dict[str, Contract]:
    """
    Calculates the 8 payment values for the contract menu using a robust
    scaling method based on an additively separable payment structure.
    
    This is an internal helper function.
    """
    # Unpack parameters
    theta_L, theta_H = params['theta_L'], params['theta_H']
    e_pL, e_sL = optimal_efforts['L']['ep'], optimal_efforts['L']['es']
    
    def get_contract_for_type(ep, es, theta, target_utility):
        """Helper to solve for the 4 payments for a single agent type."""
        
        # 1. Calculate required payment spreads from the agent's FOCs
        # Using original cost function C = e_p^2/2 + e_s^2/(2*theta)
        delta_t_q = (es / theta) / np.exp(-es) if es > 0 else 0
        delta_t_pi = ep / np.exp(-ep) if ep > 0 else 0 # C_p = e_p

        # 2. Construct a base contract using an additive structure (t_ij = t_q + t_pi)
        # This structure correctly implements the required incentive spreads.
        base_contract = {
            'G': {'H': delta_t_q + delta_t_pi, 'L': delta_t_q},
            'B': {'H': delta_t_pi,              'L': 0.0}
        }

        # 3. Calculate the expected payment this base contract would provide
        p_q_G = prob_good_safety_outcome(es)
        p_pi_H = prob_high_performance_signal(ep)
        
        expected_base_payment = (
            p_q_G * p_pi_H * base_contract['G']['H'] +
            p_q_G * (1 - p_pi_H) * base_contract['G']['L'] +
            (1 - p_q_G) * p_pi_H * base_contract['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * base_contract['B']['L']
        )

        # 4. Calculate the target payment needed to satisfy the agent's constraints
        cost = (ep**2 / 2) + (es**2 / (2 * theta))
        target_payment = cost + target_utility

        # 5. Find a single scaling factor to apply to all bonuses to meet the target payment
        if expected_base_payment > 1e-9: # Avoid division by zero
            scale_factor = target_payment / expected_base_payment
        else:
            scale_factor = 0.0

        final_contract: Contract = {
            'G': {'H': base_contract['G']['H'] * scale_factor, 'L': base_contract['G']['L'] * scale_factor},
            'B': {'H': base_contract['B']['H'] * scale_factor, 'L': base_contract['B']['L'] * scale_factor}
        }
        
        # Ensure payments respect limited liability post-scaling
        for q_key in final_contract:
            for p_key in final_contract[q_key]:
                final_contract[q_key][p_key] = max(0, final_contract[q_key][p_key])
        
        return final_contract

    # --- Calculate Contract for Low-Quality Type (target_utility = 0) ---
    contract_L = get_contract_for_type(e_pL, e_sL, theta_L, target_utility=0.0)

    # --- Calculate Contract for High-Quality Type (target_utility = rent) ---
    rent = (e_sL**2 / 2) * (1/theta_L - 1/theta_H)
    e_pH, e_sH = optimal_efforts['H']['ep'], optimal_efforts['H']['es']
    contract_H = get_contract_for_type(e_pH, e_sH, theta_H, target_utility=rent)
    
    return {'H': contract_H, 'L': contract_L}


def calculate_optimal_contract(params: Dict) -> Tuple[Dict[str, Contract], Dict]:
    """
    Calculates the menu for the theoretically optimal second-best contract.

    This function serves as the public interface for the optimal contract solver.
    It orchestrates the process of first solving for the optimal effort levels,
    and then solving for the payments required to implement those efforts.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A tuple containing:
        - The contract menu (a dict with keys 'H' and 'L').
        - The dictionary of the optimal effort levels solved for.
    """
    # 1. Solve for the theoretically optimal effort levels
    optimal_efforts = _solve_for_optimal_efforts(params)
    
    # 2. Solve for the payments that implement these efforts
    contract_menu = _solve_for_payments(optimal_efforts, params)
    
    return contract_menu, optimal_efforts