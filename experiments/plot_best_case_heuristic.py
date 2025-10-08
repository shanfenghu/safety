# experiments/plot_best_case_heuristic.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config
from plot_utils import setup_plot_style

def main():
    """
    Plots the results of the best-case heuristic experiment, creating a 
    two-panel figure with a heatmap and a comparison bar chart.
    """
    print("--- Plotting Best-Case Heuristic Results ---")
    setup_plot_style()

    # --- 1. Load Data ---
    grid_path = config.RESULTS_DIR / "best_case_heuristic_grid.csv"
    comparison_path = config.RESULTS_DIR / "best_case_heuristic_comparison.csv"
    
    if not grid_path.exists() or not comparison_path.exists():
        print(f"Error: Data files not found. Please run 'exp_best_case_heuristic.py' first.")
        return

    grid_df = pd.read_csv(grid_path)
    comparison_df = pd.read_csv(comparison_path)

    # --- 2. Create Figure and Axes ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 3. Panel (a): Heatmap of the Grid Search ---
    ax1 = axes[0]
    pivot_df = grid_df.pivot(
        index='safety_bonus', 
        columns='performance_bonus', 
        values='mean_social_welfare'
    )
    
    res = sns.heatmap(pivot_df, ax=ax1, cmap='viridis', annot=True, fmt=".1f", cbar_kws={'label': 'Mean Social Welfare'}, annot_kws={"size": 10})
    ax1.set_title('(a) Grid Search for Best-Case Heuristic', fontweight='bold')
    ax1.set_xlabel('Performance Bonus')
    ax1.set_ylabel('Safety Bonus')
    ax1.invert_yaxis() # To have the origin at the bottom-left
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 15)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 15)


    # Highlight the best-performing cell
    best_heuristic = grid_df.loc[grid_df['mean_social_welfare'].idxmax()]
    best_perf_bonus = best_heuristic['performance_bonus']
    best_safety_bonus = best_heuristic['safety_bonus']
    
    # Find the index of the best cell to draw a rectangle
    y_idx = pivot_df.index.get_loc(best_safety_bonus)
    x_idx = pivot_df.columns.get_loc(best_perf_bonus)
    
    ax1.add_patch(plt.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor='red', lw=3))


    # --- 4. Panel (b): Comparison Bar Chart ---
    ax2 = axes[1]
    sns.barplot(
        x='Contract', 
        y='MeanSocialWelfare', 
        data=comparison_df, 
        ax=ax2,
        palette=[config.CONTRACT_COLORS["optimal"], config.CONTRACT_COLORS["hybrid"]] # Blue for Optimal, Green for Heuristic
    )
    ax2.set_title('(b) Optimal vs. Best-Case Heuristic', fontweight='bold')
    ax2.set_ylabel('Mean Social Welfare')
    ax2.set_xlabel('')
    
    # Add labels on top of the bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f', padding=3, fontsize=20)

    # Set y-axis limits to be consistent
    min_welfare = comparison_df['MeanSocialWelfare'].min()
    ax2.set_ylim(min_welfare * 1.1, 0)

    # --- 5. Finalize and Save Plot ---
    plt.tight_layout(pad=1.5)
    output_path = config.FIGURES_DIR / "best_case_heuristic.pdf"
    plt.savefig(output_path)
    
    print(f"\nSuccessfully saved plot to '{output_path}'")
    print("--- Plotting Complete ---")


if __name__ == "__main__":
    main()