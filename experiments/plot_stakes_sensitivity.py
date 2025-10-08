# experiments/plot_stakes_sensitivity.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import config
from plot_utils import setup_plot_style

def main():
    """
    Plots the results of the environmental stakes (delta_W) sensitivity analysis,
    including an inset to zoom in on the top-performing contracts.
    """
    print("--- Plotting Stakes Sensitivity Results (with Inset) ---")
    setup_plot_style()

    # --- 1. Load Data ---
    data_path = config.RESULTS_DIR / "stakes_sensitivity.csv"
    
    if not data_path.exists():
        print(f"Error: Data file not found at '{data_path}'. Please run 'exp_stakes_sensitivity.py' first.")
        return

    df = pd.read_csv(data_path)
    
    # Rename for clarity in the legend
    df['contract_type'] = df['contract_type'].replace({'pre_calculated_optimal': 'optimal'})

    # --- 2. Create the Main Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x='delta_W',
        y='SocialWelfare',
        hue='contract_type',
        style='contract_type',
        palette=config.CONTRACT_COLORS,
        markers=False,
        dashes=True,
        ax=ax
    )

    # --- 3. Create and Customize the Inset Plot ---
    
    # Create an inset axes in the bottom left
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad=3)
    
    # Filter data to exclude the 'performance' contract for the zoom
    inset_df = df[df['contract_type'] != 'performance']
    
    # Plot the filtered data on the inset axes
    sns.lineplot(
        data=inset_df,
        x='delta_W',
        y='SocialWelfare',
        hue='contract_type',
        style='contract_type',
        markers=False,
        dashes=True,
        ax=ax_inset,
        legend=False # Hide the legend in the inset to avoid clutter
    )
    
    # Set the zoom level for the inset
    ax_inset.set_title('Zoom on Top Contracts')
    ax_inset.set_xlabel('') # Remove x-label from inset
    ax_inset.set_ylabel('') # Remove y-label from inset
    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Optional: Draw lines to indicate the zoomed area
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # --- 4. Customize the Main Plot ---
    ax.set_title('Contract Performance vs. Environmental Stakes (ΔW)', fontweight='bold')
    ax.set_xlabel('Environmental Stakes (Cost of Disaster, ΔW)')
    ax.set_ylabel('Mean Social Welfare')
    ax.legend(title='Contract Type', loc="lower right")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 5. Finalize and Save Plot ---
    plt.tight_layout()
    output_path = config.FIGURES_DIR / "stakes_sensitivity.pdf"
    plt.savefig(output_path)
    
    print(f"\nSuccessfully saved plot to '{output_path}'")
    print("--- Plotting Complete ---")


if __name__ == "__main__":
    main()