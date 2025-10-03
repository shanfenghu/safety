# experiments/plot_effort_tradeoff.py

"""
Generates the effort trade-off scatter plot (Figure 5) for the paper.

This script loads the data from the main comparison experiment and creates a
scatter plot to visualize the mean chosen effort pairs (e_p, e_s) for each of the
four contract types, separated by the developer's agent type (theta).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the centralized configuration and plotting utilities
import config, plot_utils

def main():
    """
    Loads data, creates, and saves the effort trade-off scatter plot.
    """
    print("--- Generating Effort Trade-off Plot (Figure 5) ---")

    # 1. Setup plot style
    plot_utils.setup_plot_style()

    # 2. Load the experimental data
    print("Loading data from 'results/main_comparison.csv'...")
    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "main_comparison.csv")
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_main_comparison.py' first.")
        return

    # 3. Aggregate data to find the mean effort for each contract AND agent type
    # This gives us 8 distinct points to plot (4 contracts x 2 thetas)
    mean_efforts_by_type = results_df.groupby(['contract_type', 'Theta'])[['ChosenEp', 'ChosenEs']].mean().reset_index()
    print("Mean effort choices by agent type:\n", mean_efforts_by_type)

    # 4. Create the scatter plot
    print("Creating the plot...")
    fig, ax = plt.subplots(figsize=(11, 9))

    # Use seaborn's scatterplot for easy coloring and styling
    sns.scatterplot(
        data=mean_efforts_by_type,
        x="ChosenEp",
        y="ChosenEs",
        hue="contract_type",
        style="Theta", # Use different markers for theta_L and theta_H
        palette=config.CONTRACT_COLORS,
        s=300, # Large, clear markers
        edgecolor='black',
        linewidth=1.5,
        ax=ax
    )

    # 5. Set titles and labels for clarity
    ax.set_title("Mean Developer Effort Choices by Contract and Type", weight='bold')
    ax.set_xlabel("Performance Effort ($e_p$)")
    ax.set_ylabel("Safety Effort ($e_s$)")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    # The legend will have entries for both hue and style. We can customize it for clarity.
    # For now, the default seaborn legend is quite good.
    plt.legend(title='Contract & Type', title_fontsize='20')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # 6. Save the figure
    plot_utils.save_figure(fig, "effort_tradeoff.pdf")
    
    print("\n--- Plot generation complete. ---")


if __name__ == "__main__":
    main()