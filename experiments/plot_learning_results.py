# experiments/plot_learning_results.py

"""
Generates the learning agent results figure for the paper's appendix.

This script loads the step-by-step data from the learning experiment and
creates a two-part figure:
- Part (a): Learning curves showing the average chosen safety effort over time.
- Part (b): A 2x2 grid of heatmaps visualizing the final, optimal Q-values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

import config, plot_utils
from src.types import Contract
from src.contracts.heuristics import *
from src.contracts.optimal import calculate_optimal_contract
from src.simulation_utils import prob_high_performance_signal, prob_good_safety_outcome


def calculate_q_value_heatmap(contract: Contract, agent_params: dict) -> np.ndarray:
    """
    Calculates the theoretical expected reward (optimal Q-value) for each action.
    """
    action_space = config.AGENT_CONFIGS['learning']['developer_params']['action_space']
    theta = agent_params['theta']
    
    q_values = []
    for ep, es in action_space:
        cost = (ep**2 / 2.0) + (es**2 / (2.0 * theta))
        p_pi_H = prob_high_performance_signal(ep)
        p_q_G = prob_good_safety_outcome(es)
        
        expected_payment = (
            p_q_G * p_pi_H * contract['G']['H'] +
            p_q_G * (1 - p_pi_H) * contract['G']['L'] +
            (1 - p_q_G) * p_pi_H * contract['B']['H'] +
            (1 - p_q_G) * (1 - p_pi_H) * contract['B']['L']
        )
        q_values.append(expected_payment - cost)
    
    # Reshape into a 3x3 grid for the heatmap
    return np.array(q_values).reshape(3, 3)


def main():
    """
    Loads data, creates, and saves the learning agent results plot.
    """
    print("--- Generating Learning Agent Results Plot ---")

    # 1. Setup plot style
    plot_utils.setup_plot_style()

    # 2. Load the experimental data
    print("Loading data from 'results/learning_robustness.csv'...")
    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "learning_robustness.csv")
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_learning_robustness.py' first.")
        return

    # 3. Create the multi-panel figure
    print("Creating the plot...")
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1])

    # --- Panel (a): Learning Curves ---
    ax1 = fig.add_subplot(gs[0, :])
    
    print("Pre-aggregating and smoothing data for learning curves...")
    # Pre-aggregate data to calculate the mean of both effort types at each step
    learning_curves_data = results_df.groupby(['Step', 'contract_type'])[['ChosenEs', 'ChosenEp', 'SocialWelfare']].mean().reset_index()

    # Calculate a rolling average for both effort types to smooth the curves
    window_size = 500
    learning_curves_data['SmoothedEs'] = learning_curves_data.groupby('contract_type')['ChosenEs'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    learning_curves_data['SmoothedEp'] = learning_curves_data.groupby('contract_type')['ChosenEp'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    
    # Calculate final converged welfare for legend
    final_phase_start_step = learning_curves_data['Step'].max() * 0.9
    converged_welfare = learning_curves_data[learning_curves_data['Step'] >= final_phase_start_step].groupby('contract_type')['SocialWelfare'].mean()

    # Create a second y-axis for performance effort
    ax2 = ax1.twinx()

    contract_order = ['optimal', 'fine', 'hybrid', 'performance']
    for contract_name in contract_order:
        subset = learning_curves_data[learning_curves_data['contract_type'] == contract_name]
        welfare = converged_welfare.get(contract_name, 0.0)
        color = config.CONTRACT_COLORS.get(contract_name, 'gray')
        
        # Plot Safety Effort on the left axis
        ax1.plot(subset['Step'], subset['SmoothedEs'], color=color, linestyle='-', 
                 label=f"{contract_name.capitalize()} (Welfare: {welfare:.1f})", linewidth=3)
        
        # Plot Performance Effort on the right axis
        ax2.plot(subset['Step'], subset['SmoothedEp'], color=color, linestyle='--', linewidth=3)

    ax1.set_title("(a) Learning Curves: Average Chosen Efforts Over Time", weight='bold')
    ax1.set_xlabel("Learning Step")
    ax1.set_ylabel("Mean Safety Effort ($e_s$)", color='black')
    ax2.set_ylabel("Mean Performance Effort ($e_p$)", color='black')
    ax1.legend(title="Contract & Final Welfare")
    
    # --- Panel (b): Final Policy Heatmaps ---
    print("Calculating learned Q-value heatmaps...")
    heatmap_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                    fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    action_space = config.AGENT_CONFIGS['learning']['developer_params']['action_space']
    ep_labels = sorted(list(set(e[0] for e in action_space)))
    es_labels = sorted(list(set(e[1] for e in action_space)))

    # Load learned Q-tables and aggregate to the 3x3 grid by action (Ep, Es)
    try:
        learned_q_df = pd.read_csv(config.RESULTS_DIR / "learning_qtables.csv")
    except FileNotFoundError:
        print("Error: Learned Q tables file not found. Please re-run the experiment.")
        return

    # Average learned Q across runs/iterations for each contract and action
    learned_q_mean = learned_q_df.groupby(['contract_type', 'Ep', 'Es'])['QValue'].mean().reset_index()

    # Build heatmap matrices for each contract in the specified order
    heatmap_data_by_contract = {}
    global_min = np.inf
    global_max = -np.inf
    for name in contract_order:
        sub = learned_q_mean[learned_q_mean['contract_type'] == name]
        # Map action space order into a 3x3 matrix matching ep_labels/es_labels order
        grid = np.full((len(es_labels), len(ep_labels)), np.nan)
        for _, row in sub.iterrows():
            i = es_labels.index(row['Es'])
            j = ep_labels.index(row['Ep'])
            grid[i, j] = row['QValue']
        heatmap_data_by_contract[name] = grid
        if np.isfinite(grid).any():
            global_min = min(global_min, np.nanmin(grid))
            global_max = max(global_max, np.nanmax(grid))

    for ax, name in zip(heatmap_axes, contract_order):
        heatmap_data = heatmap_data_by_contract[name]
        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
            cbar=False,
            xticklabels=ep_labels,
            yticklabels=es_labels,
            annot_kws={"size":18}
        )
        ax.set_title(f"{name.capitalize()} Contract", weight="bold")
        ax.set_xlabel("Performance Effort ($e_p$)")
        ax.set_ylabel("Safety Effort ($e_s$)")
        ax.invert_yaxis()

    # Add a single shared colorbar to the right of the heatmap grid
    norm = mpl.colors.Normalize(vmin=float(global_min), vmax=float(global_max))
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Explicitly set clim to ensure the colorbar lower bound reflects negatives when present
    sm.set_clim(vmin=float(global_min), vmax=float(global_max))
    cbar = fig.colorbar(sm, ax=heatmap_axes, orientation='vertical', fraction=0.025, pad=0.08)
    cbar.set_label("Q-value", rotation=90)
    # Optional: force ticks to span the full range
    try:
        ticks = np.linspace(float(global_min), float(global_max), 6)
        cbar.set_ticks(ticks)
    except Exception:
        pass

    fig.text(0.5, 0.565, "(b) Heatmaps of Learned Q-Values", ha='center', va='center', weight='bold', fontsize=20)

    # 4. Final layout and save
    # Reserve right margin for the shared colorbar to avoid overlap
    plt.tight_layout(rect=[0, 0, 0.94, 0.96], h_pad=1.5, w_pad=1.5)
    plot_utils.save_figure(fig, "learning_robustness.pdf")
    
    print("\n--- Plot generation complete. ---")


if __name__ == "__main__":
    main()

