# experiments/plot_comparative_statics.py

"""
Generates the comparative statics plot (Figure 6) for the paper.

This script loads the data from the comparative statics experiment, which
sweeps the 'delta_W' parameter (the cost of a disaster). It then plots the
mean safety effort (e_s) and mean performance effort (e_p) as a function of
this parameter, including shaded confidence intervals.

The resulting plot provides empirical validation for Theorem 5 and illustrates
the sophisticated trade-offs the optimal contract makes as stakes increase.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import the centralized configuration and plotting utilities
import config, plot_utils


def main():
    """
    Loads data, creates, and saves the comparative statics plot.
    """
    print("--- Generating Comparative Statics Plot (Figure 6) ---")

    # 1. Setup plot style for a professional, consistent look
    plot_utils.setup_plot_style()

    # 2. Load the experimental data
    print("Loading data from 'results/comparative_statics.csv'...")
    try:
        results_df = pd.read_csv(config.RESULTS_DIR / "comparative_statics.csv")
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_comparative_statics.py' first.")
        return

    # 3. Aggregate data to calculate mean and standard error for plotting trends
    print("Aggregating data to calculate mean and standard error...")
    summary = results_df.groupby('delta_W').agg(
        # Calculate mean for the trend line
        mean_es=('ChosenEs', 'mean'),
        mean_ep=('ChosenEp', 'mean'),
        # Calculate Standard Error of the Mean for the confidence interval
        sem_es=('ChosenEs', 'sem'),
        sem_ep=('ChosenEp', 'sem')
    ).reset_index()
    print("Aggregated data trends:\n", summary.head())

    # 4. Create the plot with two y-axes
    print("Creating the plot...")
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # --- Plot Safety Effort (e_s) on the left y-axis ---
    # Plot the mean trend line
    ax1.plot(summary['delta_W'], summary['mean_es'], color='royalblue', label='Mean Safety Effort ($e_s$)')
    # Add the shaded confidence interval (mean +/- 1 SEM)
    ax1.fill_between(
        summary['delta_W'],
        summary['mean_es'] - summary['sem_es'],
        summary['mean_es'] + summary['sem_es'],
        color='royalblue',
        alpha=0.2
    )
    ax1.set_xlabel("Environmental Stakes ($ΔW$)")
    ax1.set_ylabel("Mean Safety Effort ($e_s$)", color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(bottom=0)

    # --- Plot Performance Effort (e_p) on the right y-axis ---
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(summary['delta_W'], summary['mean_ep'], color='firebrick', linestyle='--', label='Mean Performance Effort ($e_p$)')
    # Add its confidence interval
    ax2.fill_between(
        summary['delta_W'],
        summary['mean_ep'] - summary['sem_ep'],
        summary['mean_ep'] + summary['sem_ep'],
        color='firebrick',
        alpha=0.2
    )
    ax2.set_ylabel("Mean Performance Effort ($e_p$)", color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.set_ylim(bottom=0)

    # 5. Set titles and legend
    ax1.set_title("Optimal Effort Choices vs. Environmental Stakes", weight='bold')
    # Combine legends from both axes into one
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    plt.tight_layout()

    # 6. Save the figure using the utility function
    plot_utils.save_figure(fig, "comparative_statics.pdf")
    
    print("\n--- Plot generation complete. ---")


if __name__ == "__main__":
    main()
