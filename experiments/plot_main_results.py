# experiments/plot_main_results.py

"""
Generates the main results figure (Figure 4) for the paper.

This script loads the aggregated data from the main comparison experiment,
calculates the mean and standard error for key metrics, and plots them
as a two-panel bar chart:
- Panel (a): Mean Social Welfare
- Panel (b): Disaster Frequency

The plot includes error bars to represent statistical uncertainty and is styled
for publication quality using the plot_utils module.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import the centralized configuration and plotting utilities
import config, plot_utils


def main():
    """
    Loads data, creates, and saves the main results plot.
    """
    print("--- Generating Main Results Plot (Figure 4) ---")

    # 1. Setup plot style for a professional, consistent look
    plot_utils.setup_plot_style()

    # 2. Load the experimental data
    print("Loading data from 'results/main_comparison.csv'...")
    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "main_comparison.csv")
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_main_comparison.py' first.")
        return

    # 3. Aggregate data to calculate mean and standard error
    print("Aggregating data to calculate mean and standard error...")
    summary = results_df.groupby('contract_type').agg(
        # Calculate mean for the bar height
        mean_welfare=('SocialWelfare', 'mean'),
        mean_disaster_freq=('DisasterOccurred', 'mean'),
        # Calculate Standard Error of the Mean for the error bars
        sem_welfare=('SocialWelfare', 'sem'),
        sem_disaster_freq=('DisasterOccurred', 'sem')
    )

    # Reorder the contracts for a logical presentation in the plot
    contract_order = ['optimal', 'fine', 'hybrid', 'performance']
    summary = summary.reindex(contract_order)
    print("Aggregated data:\n", summary)

    # 4. Create the two-panel plot
    print("Creating the plot...")
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=(14, 6), 
        sharex=True  # Both subplots will share the same x-axis labels
    )

    # --- Panel (a): Mean Social Welfare ---
    ax1.bar(
        summary.index,
        summary['mean_welfare'],
        yerr=summary['sem_welfare'], # Add error bars
        capsize=5, # Add caps to the error bars for clarity
        color=[config.CONTRACT_COLORS.get(c, 'gray') for c in summary.index]
    )
    ax1.set_ylabel("Mean Social Welfare")
    ax1.set_title("(a) Social Welfare Comparison", weight="bold")
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add a zero line for reference

    # --- Panel (b): Disaster Frequency ---
    ax2.bar(
        summary.index,
        summary['mean_disaster_freq'],
        yerr=summary['sem_disaster_freq'],
        capsize=5,
        color=[config.CONTRACT_COLORS.get(c, 'gray') for c in summary.index]
    )
    ax2.set_ylabel("Disaster Frequency")
    ax2.set_title("(b) Safety Outcome Comparison", weight="bold")
    # Format the y-axis to show percentages
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_ylim(bottom=0)

    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=15, ha='right')
    
    # Final layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 5. Save the figure using the utility function
    plot_utils.save_figure(fig, "main_comparison.pdf")
    
    print("\n--- Plot generation complete. ---")


if __name__ == "__main__":
    main()
