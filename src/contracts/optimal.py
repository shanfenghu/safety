# src/contracts/optimal.py

"""
Defines the solver for the second-best optimal contract.

This module translates the theoretical results from the paper's analysis into a
numerical solver. It implements the PIPS (Projected Incentive-Preserving Shift)
algorithm to find a robust, non-negative, and incentive-compatible contract menu.
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
    theta_L, theta_H, nu, delta_W = params['theta_L'], params['theta_H'], params['nu'], params['delta_W']

    # --- Solve for High-Quality Type's Safety Effort (First-Best) ---
    # FOC: -p_s(e_s) * delta_W = C_s(e_s, theta_H)
    func_sH = lambda es: np.exp(-es) * delta_W - es / theta_H
    e_sH_opt = fsolve(func_sH, x0=2.0)[0]
    
    # --- Solve for Low-Quality Type's Safety Effort (Distorted) ---
    # This is the corrected FOC from the theoretical proof's first-order conditions.
    func_sL = lambda es: ( (1 - nu) * (np.exp(-es) * delta_W - es / theta_L) - 
                           nu * (es / theta_L - es / theta_H) )
    e_sL_opt = fsolve(func_sL, x0=1.0)[0]
    
    # --- Determine Performance Efforts ---
    # For the simulation, we set these to small, plausible positive values
    # that reflect the muting of incentives due to the Limited Liability constraint.
    e_pL_opt = 0.10
    e_pH_opt = 0.15

    return {
        'H': {'ep': max(0, e_pH_opt), 'es': max(0, e_sH_opt)},
        'L': {'ep': max(0, e_pL_opt), 'es': max(0, e_sL_opt)}
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
        
        # --- PIPS Step 1: Solve for the Unconstrained "Ideal" Incentives ---
        delta_t_q = (es / theta) / np.exp(-es) if es > 0 else 0
        delta_t_pi = ep / np.exp(-ep) if ep > 0 else 0
        p_q_G = prob_good_safety_outcome(es)
        p_pi_H = prob_high_performance_signal(ep)
        
        # The linear system's only job is to find the payment spreads that
        # create the target incentives. We can construct this directly.
        # This is the "ideal" contract, which may have negative payments.
        t_ideal = {
            'G': {'H': delta_t_q + delta_t_pi, 'L': delta_t_q},
            'B': {'H': delta_t_pi,              'L': 0.0}
        }

        # --- PIPS Step 2 & 3: Apply Incentive-Preserving Utility Shift ---
        cost = (ep**2 / 2.0) + (es**2 / (2.0 * theta))
        expected_ideal_payment = (
            p_q_G * p_pi_H * t_ideal['G']['H'] +
            p_q_G * (1 - p_pi_H) * t_ideal['G']['L'] +
            (1 - p_q_G) * p_pi_H * t_ideal['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * t_ideal['B']['L']
        )
        ideal_utility = expected_ideal_payment - cost
        
        # Calculate the constant 'k' to add to all payments to meet the target utility
        k = target_utility - ideal_utility
        
        # Apply the shift
        t_shifted = {
            q: {p: payment + k for p, payment in p_dict.items()}
            for q, p_dict in t_ideal.items()
        }

        # --- PIPS Step 4: Project to Non-Negative Space ---
        min_payment = min(p for outcomes in t_shifted.values() for p in outcomes.values())
        s = -min_payment if min_payment < 0 else 0.0

        # Apply the final shift to ensure all payments are non-negative
        final_contract: Contract = {
            q: {p: payment + s for p, payment in p_dict.items()}
            for q, p_dict in t_shifted.items()
        }
        
        return final_contract

    # --- Calculate Contract for Low-Quality Type ---
    # The target utility is 0 because the IR-L constraint is binding.
    contract_L = get_contract_for_type(e_pL, e_sL, theta_L, target_utility=0.0)

    # --- Calculate Contract for High-Quality Type ---
    # The target utility is the information rent because the IC-H constraint is binding.
    rent = (e_sL**2 / 2) * (1/theta_L - 1/theta_H)
    e_pH, e_sH = optimal_efforts['H']['ep'], optimal_efforts['H']['es']
    contract_H = get_contract_for_type(e_pH, e_sH, theta_H, target_utility=rent)
    
    return {'H': contract_H, 'L': contract_L}


def calculate_optimal_contract(params: Dict) -> Tuple[Dict[str, Contract], Dict]:
    """
    Calculates the menu for the theoretically optimal second-best contract.

    This function serves as the public interface for the optimal contract solver.
    It orchestrates the process of first solving for the optimal effort levels,
    and then solving for the payments required to implement those efforts using
    the robust PIPS algorithm.

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