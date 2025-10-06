# experiments/plot_belief_misspecification.py

"""
Creates a multi-panel heatmap figure for the Belief Misspecification experiment.

Panels:
- (a) Heatmap of welfare loss for the Optimal contract.
- (b) Heatmap of welfare loss for the Hybrid (heuristic) contract.

Reads: results/belief_misspecification.csv
Outputs: figures/belief_misspecification.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
import plot_utils


def main():
    """Generates and saves the belief misspecification heatmap plot."""
    print('--- Generating Belief Misspecification Results Plot ---')
    plot_utils.setup_plot_style()

    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(config.RESULTS_DIR / 'belief_misspecification.csv')
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_belief_misspecification.py' first.")
        return

    # Create the pivot tables needed for the heatmaps
    pivot_optimal = df.pivot(
        index='nu_true', 
        columns='nu_assumed', 
        values='welfare_loss_optimal'
    )
    pivot_hybrid = df.pivot(
        index='nu_true', 
        columns='nu_assumed', 
        values='welfare_loss_hybrid'
    )

    # --- 2. Create Figure and Axes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # --- 3. Plot Panel (a): Optimal Contract ---
    sns.heatmap(
        pivot_optimal,
        ax=ax1,
        annot=True,          # Annotate cells with numerical values
        fmt=".1f",           # Format annotations to one decimal place
        cmap='coolwarm',     # Use a diverging colormap
        center=0,            # Center the colormap at zero
        linewidths=.5,
        cbar_kws={'label': 'Welfare Loss (Benchmark - Misspecified)'}
    )
    ax1.set_title('(a) Welfare Loss for Optimal Contract', fontsize=20, pad=15)
    ax1.set_xlabel(r"Regulator's Assumed Belief ($\nu_{assumed}$)", fontsize=20)
    ax1.set_ylabel(r'True State of the World ($\nu_{true}$)', fontsize=20)
    ax1.invert_yaxis() # Puts (0.1, 0.1) at the bottom-left for intuitive reading

    # --- 4. Plot Panel (b): Hybrid Contract ---
    sns.heatmap(
        pivot_hybrid,
        ax=ax2,
        annot=True,
        fmt=".0f", # Format as integer since these losses are large
        cmap='Reds', # A sequential colormap is better here as there are no negative losses
        linewidths=.5,
        cbar_kws={'label': 'Welfare Loss (Benchmark - Hybrid)'}
    )
    ax2.set_title('(b) Welfare Loss for Hybrid Contract', fontsize=20, pad=15)
    ax2.set_xlabel(r"Regulator's Assumed Belief ($\nu_{assumed}$)", fontsize=20)
    ax2.set_ylabel('') # Remove redundant y-axis label

    # --- 5. Final Touches and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.FIGURES_DIR / "belief_misspecification.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved plot to '{output_path}'")


if __name__ == "__main__":
    main()