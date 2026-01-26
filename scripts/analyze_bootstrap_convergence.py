#!/usr/bin/env python3
"""Analyze Bootstrap Convergence.

This script analyzes the convergence of bootstrap uncertainty estimates
by running multiple bootstrap repetitions and plotting convergence metrics.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def analyze_convergence(
    bootstrap_csv: str,
    n_repeats: int = 10,
    output_dir: Optional[str] = None
):
    """Analyze bootstrap convergence.

    Args:
        bootstrap_csv: Path to bootstrap results CSV
        n_repeats: Number of convergence repeats
        output_dir: Output directory for plots
    """
    if output_dir is None:
        output_dir = Path(bootstrap_csv).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load bootstrap data
    df = pd.read_csv(bootstrap_csv)
    print(f"Loaded {len(df)} bootstrap samples")

    # Metrics to analyze
    metrics = ['temporal_std', 'wwi_percent', 'jaccard_with_original']

    # Run convergence analysis
    convergence_data = {}

    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: {metric} not found in data")
            continue

        values = df[metric].dropna().values
        if len(values) == 0:
            continue

        # Simulate convergence by subsampling
        n_samples = len(values)
        step_size = max(1, n_samples // 50)  # 50 points

        means = []
        stds = []

        for n in range(step_size, n_samples + 1, step_size):
            subset = values[:n]
            means.append(np.mean(subset))
            stds.append(np.std(subset))

        convergence_data[metric] = {
            'sample_sizes': list(range(step_size, n_samples + 1, step_size)),
            'means': means,
            'stds': stds
        }

    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (metric, data) in enumerate(convergence_data.items()):
        if i >= 4:
            break

        ax = axes[i]
        sizes = data['sample_sizes']
        means = data['means']
        stds = data['stds']

        # Plot mean
        ax.plot(sizes, means, 'b-', label='Mean', linewidth=2)

        # Plot std as error bars
        ax.fill_between(sizes,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.3, color='blue', label='±1 STD')

        ax.set_xlabel('Number of Bootstrap Samples')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'Convergence of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(len(convergence_data), 4):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plot_path = output_dir / 'bootstrap_convergence.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot to {plot_path}")

    # Save convergence data
    for metric, data in convergence_data.items():
        conv_df = pd.DataFrame({
            'n_samples': data['sample_sizes'],
            'mean': data['means'],
            'std': data['stds']
        })
        csv_path = output_dir / f'convergence_{metric}.csv'
        conv_df.to_csv(csv_path, index=False)
        print(f"Saved convergence data for {metric} to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Bootstrap Convergence')
    parser.add_argument('--bootstrap-csv', required=True,
                       help='Path to bootstrap results CSV')
    parser.add_argument('--n-repeats', type=int, default=10,
                       help='Number of convergence repeats')
    parser.add_argument('--output-dir',
                       help='Output directory for plots (default: same as input)')

    args = parser.parse_args()

    analyze_convergence(
        args.bootstrap_csv,
        args.n_repeats,
        args.output_dir
    )


if __name__ == '__main__':
    main()