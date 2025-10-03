# experiments/config.py

"""
Central configuration file for all experiments.

This file defines all parameters, settings, and configurations used in the
simulation runs and for plotting the results. Centralizing these settings
ensures consistency and reproducibility across all experiments.
"""

from pathlib import Path

# --- Import Agent Classes for Configuration ---
from src.agents.rational import RationalDeveloperAgent
from src.agents.learning import LearningDeveloperAgent
from src.agents.risk_averse import RiskAverseDeveloperAgent


# -----------------------------------------------------------------------------
# 1. Global Simulation Settings
# -----------------------------------------------------------------------------
# Number of times to run each unique parameter combination. A higher number
# yields smoother data and more reliable error bars.
ITERATIONS = 100

# Number of CPU cores to use for parallel execution of experiments.
# Set to None to use all available cores.
NUM_PROCESSES = 1

# A global random seed for the main experiments to ensure that the set of
# runs is reproducible. Individual runs within the batch will still be
# stochastic, but the overall experiment can be replicated.
SEED = 42


# -----------------------------------------------------------------------------
# 2. Core Theoretical Parameters
# -----------------------------------------------------------------------------
# This dictionary contains the default values for the variables from our
# theoretical model. It serves as the "base" configuration that individual
# experiments will modify.
BASELINE_PARAMS = {
    'nu': 0.5,           # Prior probability of a high-quality type
    'theta_L': 1.0,      # Quality parameter for the low type
    'theta_H': 2.0,      # Quality parameter for the high type
    'delta_W': 1000.0,   # Social cost of a disaster (W_G - W_B)
    
    # Default bonus values for the simple heuristic contracts
    'safety_bonus': 150.0,
    'performance_bonus': 75.0,
}


# -----------------------------------------------------------------------------
# 3. Agent Configurations
# -----------------------------------------------------------------------------
# This dictionary defines reusable configurations for each developer agent type.
# It allows experimental scripts to easily swap between different agent behaviors.
AGENT_CONFIGS = {
    'rational': {
        'developer_class': RationalDeveloperAgent,
        'developer_params': {} # Rational agent has no extra params
    },
    'learning': {
        'developer_class': LearningDeveloperAgent,
        'developer_params': {
            # A 3x3 grid of discrete effort choices for the Q-learning agent
            'action_space': [
                (0.0, 0.0), (0.0, 2.0), (0.0, 4.0),
                (1.5, 0.0), (1.5, 2.0), (1.5, 4.0),
                (3.0, 0.0), (3.0, 2.0), (3.0, 4.0),
            ],
            'alpha': 0.1,    # Learning rate
            'epsilon': 0.1   # Exploration rate
        }
    },
    'risk_averse': {
        'developer_class': RiskAverseDeveloperAgent,
        'developer_params': {} # Risk-averse agent has no extra params
    }
}


# -----------------------------------------------------------------------------
# 4. Contract Configurations
# -----------------------------------------------------------------------------
# Defines the parameter sets for each of the four main contract types to be tested.
CONTRACT_CONFIGS = {
    'optimal': {'contract_type': 'optimal'},
    'fine': {'contract_type': 'fine'},
    'performance': {'contract_type': 'performance'},
    'hybrid': {'contract_type': 'hybrid'}
}


# -----------------------------------------------------------------------------
# 5. Advanced Experiment Parameters
# -----------------------------------------------------------------------------
# Specific settings for the advanced robustness check experiments.
CORRELATION_SCENARIOS = {
    'Orthogonal': {'correlation_k': 0.0},
    'Positive': {'correlation_k': 0.5},
    'Negative': {'correlation_k': -0.5},
}


# -----------------------------------------------------------------------------
# 6. Plotting and Output Configuration
# -----------------------------------------------------------------------------
# Centralizes paths and style settings for consistent, professional visuals.
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")

# A consistent color palette for plotting different contract types.
CONTRACT_COLORS = {
    'optimal': 'royalblue',
    'fine': 'firebrick',
    'performance': 'goldenrod',
    'hybrid': 'seagreen'
}

# Font settings for professional, publication-quality figures that match
# the LaTeX document style.
FONT_SETTINGS = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "font.size": 14
}
