# experiments/plot_correlation.py

"""
Generates the effort correlation results figure (Figure 7) for the paper.

This script loads the data from the correlation experiment and creates a
grouped bar chart. The chart visualizes the disaster frequency for each of the
four contract types under the three different effort correlation scenarios
(Negative, Orthogonal, and Positive).

This figure provides a powerful demonstration of the robustness of the optimal
contract compared to the fragility of the naive heuristic contracts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib.gridspec as gridspec

# Import the centralized configuration and plotting utilities
import config, plot_utils


def main():
    """
    Loads data, creates, and saves the correlation results plot.
    """
    print("--- Generating Effort Correlation Plot (Figure 7) ---")

    # 1. Setup plot style for a professional, consistent look
    plot_utils.setup_plot_style()

    # 2. Load the experimental data
    print("Loading data from 'results/correlation_results.csv'...")
    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "correlation_results.csv")
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_correlation.py' first.")
        return

    # 3. Create the grouped bar chart
    print("Creating the plot...")
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    # Consistent x order across panels based on config scenarios
    x_order = [
        config.CORRELATION_SCENARIOS['Negative']['correlation_k'],
        config.CORRELATION_SCENARIOS['Orthogonal']['correlation_k'],
        config.CORRELATION_SCENARIOS['Positive']['correlation_k']
    ]

    # Panel (a): Mean Social Welfare
    sns.barplot(
        data=results_df,
        x="correlation_k",
        y="SocialWelfare",
        hue="contract_type",
        hue_order=["optimal", "fine", "hybrid", "performance"],
        order=x_order,
        palette=config.CONTRACT_COLORS,
        ax=ax_top,
        errorbar=("ci", 95),
        errwidth=1.5,
        capsize=0.15
    )
    ax_top.set_title("(a) Mean Social Welfare by Effort Correlation", weight='bold')
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Mean Social Welfare")
    ax_top.set_xticklabels([
        f"Negative (k={config.CORRELATION_SCENARIOS['Negative']['correlation_k']})",
        f"Orthogonal (k={config.CORRELATION_SCENARIOS['Orthogonal']['correlation_k']})",
        f"Positive (k={config.CORRELATION_SCENARIOS['Positive']['correlation_k']})"
    ])
    # Remove duplicate legend from the top panel; keep only bottom legend
    if ax_top.get_legend() is not None:
        ax_top.get_legend().remove()

    # Panel (b): Disaster Frequency
    sns.barplot(
        data=results_df,
        x="correlation_k",
        y="DisasterOccurred",
        hue="contract_type",
        hue_order=["optimal", "fine", "hybrid", "performance"],
        order=x_order,
        palette=config.CONTRACT_COLORS,
        ax=ax,
        errorbar=("ci", 95),
        errwidth=1.5,
        capsize=0.15
    )
    ax.set_title("(b) Disaster Frequency by Effort Correlation", weight='bold')
    ax.set_xlabel("Effort Correlation (k)")
    ax.set_ylabel("Disaster Frequency")

    # Set custom labels for the x-axis ticks
    x_tick_labels = [
        f"Negative (k={config.CORRELATION_SCENARIOS['Negative']['correlation_k']})",
        f"Orthogonal (k={config.CORRELATION_SCENARIOS['Orthogonal']['correlation_k']})",
        f"Positive (k={config.CORRELATION_SCENARIOS['Positive']['correlation_k']})"
    ]
    ax.set_xticklabels(x_tick_labels)

    # Format the y-axis to show percentages
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05) # Set y-limit to 105% to give space for 100% bar

    # Improve the legend
    ax.legend(title="Contract Type", fontsize=20)

    # Final layout adjustment
    plt.tight_layout()

    # 5. Save the figure using the utility function
    plot_utils.save_figure(fig, "correlation.pdf")
    
    print("\n--- Plot generation complete. ---")


if __name__ == "__main__":
    main()
