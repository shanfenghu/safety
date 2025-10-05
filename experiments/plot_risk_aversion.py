# experiments/plot_risk_aversion.py

"""
Creates a multi-panel figure for the Risk Aversion experiment results.

Panels:
- (a) Slopegraph of mean social welfare: Rational -> Risk-averse per contract
- (b) Slopegraph of mean safety effort e_s: Rational -> Risk-averse per contract
- (c) Decomposition bars: expected payment vs expected disaster loss magnitudes

Reads: results/risk_aversion_results.csv
Outputs: figures/risk_aversion.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import config, plot_utils


def _prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize class names for grouping/labels
    df = df.copy()
    if df['developer_class'].dtype == 'O':
        df['developer_class'] = df['developer_class'].astype(str)

    grouped = df.groupby(['developer_class', 'contract_type']).agg(
        mean_welfare=('SocialWelfare', 'mean'),
        mean_es=('ChosenEs', 'mean'),
        mean_ep=('ChosenEp', 'mean'),
        disaster_freq=('DisasterOccurred', 'mean'),
        expected_payment=('Payoff', 'mean')
    ).reset_index()

    # Order contracts consistently
    contract_order = ['optimal', 'optimal_ra', 'fine', 'hybrid', 'performance']
    grouped['contract_type'] = pd.Categorical(grouped['contract_type'], categories=contract_order, ordered=True)
    grouped = grouped.sort_values(['contract_type', 'developer_class'])
    return grouped


def _plot_slope(ax, data: pd.DataFrame, y_col: str, title: str, ylabel: str):
    contracts = data['contract_type'].cat.categories if hasattr(data['contract_type'], 'cat') else sorted(data['contract_type'].unique())
    x_positions = [0, 1]
    handles = []
    labels = []
    for c in contracts:
        sub = data[data['contract_type'] == c]
        if sub.empty:
            continue
        # Expect exactly two rows: Rational and Risk-averse
        # Sort to ensure consistent order: Rational first, then RiskAverse
        sub = sub.copy()
        sub['x'] = sub['developer_class'].apply(lambda s: 0 if 'Rational' in s else 1)
        sub = sub.sort_values('x')
        y_vals = sub[y_col].values
        color = config.CONTRACT_COLORS.get(c, 'gray')
        line, = ax.plot(x_positions, y_vals, marker='o', color=color, linewidth=2.5)
        handles.append(line)
        labels.append(c.capitalize() if c != 'optimal_ra' else 'Optimal-RA')
    ax.set_title(title, weight='bold')
    ax.set_xticks(x_positions, ['Rational', 'Risk-averse'])
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    # Add legend for contracts
    ax.legend(handles=handles, labels=labels, title='Contract Type', fontsize=20)


def _plot_decomposition(ax, data: pd.DataFrame):
    # Build a tidy frame with magnitudes of components (positive for display)
    delta_W = config.BASELINE_PARAMS['delta_W']
    df = data.copy()
    df['disaster_loss_mag'] = df['disaster_freq'] * delta_W
    df['payment_mag'] = df['expected_payment'].clip(lower=0).abs()

    # Bars grouped by contract, hue by developer_class, stacked components
    contract_order = ['optimal', 'optimal_ra', 'fine', 'hybrid', 'performance']
    dev_order = ['RationalDeveloperAgent', 'RiskAverseDeveloperAgent']
    # Ensure readable labels for developer class
    df['dev_label'] = df['developer_class'].apply(lambda s: 'Rational' if 'Rational' in s else 'Risk-averse')

    # Compute positions
    x_positions = np.arange(len(contract_order))
    width = 0.4
    offsets = {-1: -width/2, 1: width/2}

    # Track tops for annotation
    tops_by_contract = {c: {"left": 0.0, "right": 0.0} for c in contract_order}
    for i, (dev_filter, side) in enumerate(zip(dev_order, [-1, 1])):
        sub = df[df['developer_class'].str.contains('Rational' if dev_filter.startswith('Rational') else 'RiskAverse')]
        sub = sub.set_index('contract_type').reindex(contract_order)
        x = x_positions + offsets[side]
        # Stacked bars: payments (bottom) + disaster loss (top)
        bar1 = ax.bar(x, sub['payment_mag'], width=width, label=f"{ 'Rational' if side==-1 else 'Risk-averse' } – Payment", color='#7aa6c2', alpha=0.9)
        bar2 = ax.bar(x, sub['disaster_loss_mag'], width=width, bottom=sub['payment_mag'], label=f"{ 'Rational' if side==-1 else 'Risk-averse' } – Disaster", color='#c27a7a', alpha=0.9)
        # Store tops for annotation
        for idx, c in enumerate(contract_order):
            total = (sub.loc[c, 'payment_mag'] if pd.notnull(sub.loc[c, 'payment_mag']) else 0) + (sub.loc[c, 'disaster_loss_mag'] if pd.notnull(sub.loc[c, 'disaster_loss_mag']) else 0)
            if side == -1:
                tops_by_contract[c]["left"] = total
            else:
                tops_by_contract[c]["right"] = total
            # Add disaster frequency label for this bar
            try:
                freq = float(sub.loc[c, 'disaster_freq'])
                if np.isfinite(freq):
                    ax.text(x[idx], total - 0.01 * max(1.0, np.nanmax(df[['payment_mag','disaster_loss_mag']].sum(axis=1))), f"{freq:.1%}", ha='center', va='bottom', fontsize=12)
            except Exception:
                pass

    ax.set_title('(c) Components: Expected Payment + Disaster Loss (magnitudes)', weight='bold')
    ax.set_xticks(x_positions, [c.capitalize() if c!='optimal_ra' else 'Optimal-RA' for c in contract_order])
    ax.set_ylabel('Magnitude')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    # Single legend with fewer entries: map to two patches
    handles, labels = ax.get_legend_handles_labels()
    # Reduce to two unique component labels using the last two
    ax.legend(handles=[handles[-2], handles[-1]], labels=['Payment', 'Disaster loss'], title='Components')

    # Annotate meaning of each bar in the pair
    y_pad = 0.02 * max(1.0, np.nanmax(df[['payment_mag','disaster_loss_mag']].sum(axis=1)))
    for i, c in enumerate(contract_order):
        x_left = x_positions[i] + offsets[-1]
        x_right = x_positions[i] + offsets[1]
        ax.text(x_left, tops_by_contract[c]["left"] + y_pad, 'Rational', ha='center', va='bottom', fontsize=12)
        ax.text(x_right, tops_by_contract[c]["right"] + y_pad, 'Risk-averse', ha='center', va='bottom', fontsize=12)


def main():
    print('--- Generating Risk Aversion Results Plot ---')
    plot_utils.setup_plot_style()

    # Load data
    try:
        df = pd.read_csv(config.RESULTS_DIR / 'risk_aversion_results.csv')
    except FileNotFoundError:
        print("Error: Results file not found. Please run 'exp_risk_aversion.py' first.")
        return

    summary = _prepare_summary(df)

    # Create figure
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) Welfare slopegraph
    _plot_slope(ax_a, summary, y_col='mean_welfare', title='(a) Mean Social Welfare', ylabel='Mean Social Welfare')

    # (b) Safety effort slopegraph
    _plot_slope(ax_b, summary, y_col='mean_es', title='(b) Mean Safety Effort $e_s$', ylabel='Mean $e_s$')

    # (c) Decomposition bars
    _plot_decomposition(ax_c, summary)

    plt.tight_layout()
    plot_utils.save_figure(fig, 'risk_aversion.pdf')
    print('\n--- Plot generation complete. ---')


if __name__ == '__main__':
    main()


