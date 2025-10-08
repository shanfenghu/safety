# experiments/plot_type_gap.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import config
from plot_utils import setup_plot_style

def main():
    """
    Plots the results of the type gap sensitivity analysis, including an 
    inset plot that zooms in exclusively on the optimal contract's performance.
    """
    print("--- Plotting Type Gap Sensitivity Results (with Optimal-Only Inset) ---")
    setup_plot_style()

    # --- 1. Load Data ---
    data_path = config.RESULTS_DIR / "type_gap_sensitivity.csv"
    
    if not data_path.exists():
        print(f"Error: Data file not found at '{data_path}'. Please run 'exp_type_gap.py' first.")
        return

    df = pd.read_csv(data_path)
    
    # Rename for clarity in the legend
    df['contract_type'] = df['contract_type'].replace({'pre_calculated_optimal': 'Optimal'})

    # --- 2. Create the Main Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x='theta_H',
        y='SocialWelfare',
        hue='contract_type',
        style='contract_type',
        markers=True,
        dashes=True,
        ax=ax
    )

    # --- 3. Create and Customize the Inset Plot (Optimal Contract Only) ---
    
    # Create an inset axes in the bottom left corner
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad=3)
    
    # --- MODIFICATION: Filter data to *only* include the 'Optimal' contract ---
    inset_df = df[df['contract_type'] == 'Optimal']
    
    # Plot the filtered data on the inset axes
    sns.lineplot(
        data=inset_df,
        x='theta_H',
        y='SocialWelfare',
        hue='contract_type', # Still use hue to get the correct color
        style='contract_type',
        markers=True,
        dashes=False, # Use a solid line in the inset for clarity
        ax=ax_inset,
        legend=False # Hide the legend in the inset
    )
    
    # Set a tighter zoom level for the inset
    ax_inset.set_ylim(-110, -70)
    ax_inset.set_title('Zoom on Optimal Contract')
    ax_inset.set_xlabel('') # Remove x-label from inset
    ax_inset.set_ylabel('') # Remove y-label from inset
    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Indicate the zoomed area on the main plot
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    # --- 4. Customize the Main Plot ---
    ax.set_title('Contract Performance vs. Type Gap (Severity of Adverse Selection)', loc='left', fontweight='bold')
    ax.set_xlabel('High-Quality Type Parameter ($θ_H$)')
    ax.set_ylabel('Mean Social Welfare')
    ax.legend(title='Contract Type')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 5. Finalize and Save Plot ---
    plt.tight_layout()
    output_path = config.FIGURES_DIR / "type_gap_sensitivity.pdf"
    plt.savefig(output_path)
    
    print(f"\nSuccessfully saved plot to '{output_path}'")
    print("--- Plotting Complete ---")


if __name__ == "__main__":
    main()