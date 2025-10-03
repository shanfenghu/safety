# experiments/plot_utils.py

"""
Provides utility functions for creating and saving standardized plots.

This module centralizes plotting styles and file-saving logic to ensure all
figures in the paper have a consistent, publication-quality appearance.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import the centralized configuration for paths and styles
import config


def setup_plot_style():
    """
    Sets up the global Matplotlib style for all plots.

    This function applies a professional 'seaborn' style and configures
    the font to match the LaTeX document, ensuring visual consistency. It should
    be called once at the beginning of each plotting script.
    """
    # Apply a professional and clean plot style (disable LaTeX to avoid dependency)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Ensure Matplotlib does not try to use LaTeX even if other styles tweak it
    plt.rcParams['text.usetex'] = False

    # Update rcParams with the font settings defined in the config file
    plt.rcParams.update(config.FONT_SETTINGS)
    
    # Further refine plot elements for readability in a paper
    plt.rcParams.update({
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20,
    })
    print("Plot style setup complete.")


def save_figure(fig: Figure, filename: str):
    """
    Saves a Matplotlib figure to the designated figures directory.

    This function handles the creation of the output directory and saves the
    figure in a high-quality PDF format suitable for publication.

    Args:
        fig: The Matplotlib Figure object to be saved.
        filename: The name of the output file (e.g., "figure4_main_results.pdf").
    """
    # Ensure the output directory exists
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Construct the full, platform-independent file path
    full_path = config.FIGURES_DIR / filename
    
    # Save the figure with high-quality settings
    fig.savefig(
        full_path,
        format='pdf',        # Use vector graphics for scalability
        bbox_inches='tight', # Crop whitespace around the figure
        dpi=300              # Ensure high resolution for any rasterized elements
    )
    
    print(f"Figure saved to '{full_path}'")
