#!/usr/bin/env python3
"""Compare Samplers on KDR100.

This script compares different sampling methods on KDR100 dataset
by evaluating their performance on selection tasks.
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

ROOT = Path(__file__).resolve().parents[1]


def load_selection_results(selection_json: str) -> Dict:
    """Load selection results from JSON."""
    with open(selection_json, 'r') as f:
        return json.load(f)


def compare_samplers(
    selection_json: str,
    output_dir: str = 'outputs'
):
    """Compare different samplers based on selection results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    data = load_selection_results(selection_json)
    print(f"Loaded selection results with {len(data)} entries")

    # Extract sampler information
    samplers = {}
    for key, value in data.items():
        if isinstance(value, dict) and 'sampler' in value:
            sampler = value['sampler']
            if sampler not in samplers:
                samplers[sampler] = []
            samplers[sampler].append(value)

    if not samplers:
        print("No sampler data found in JSON")
        return

    print(f"Found {len(samplers)} different samplers: {list(samplers.keys())}")

    # Compare metrics
    metrics_to_compare = ['temporal_std', 'wwi_percent', 'jaccard_with_original',
                         'selection_size', 'compute_time']

    results = []
    for sampler, runs in samplers.items():
        for run in runs:
            result = {'sampler': sampler}
            for metric in metrics_to_compare:
                if metric in run:
                    result[metric] = run[metric]
                else:
                    result[metric] = None
            results.append(result)

    results_df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / 'sampler_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved comparison results to {csv_path}")

    # Plot comparisons
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_compare):
        if i >= 6:
            break

        ax = axes[i]
        valid_data = results_df.dropna(subset=[metric])

        if valid_data.empty:
            ax.text(0.5, 0.5, f'No data for {metric}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Box plot
        sampler_groups = [valid_data[valid_data['sampler'] == s][metric].values
                         for s in valid_data['sampler'].unique()]

        if sampler_groups:
            ax.boxplot(sampler_groups, labels=valid_data['sampler'].unique())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for i in range(len(metrics_to_compare), 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plot_path = output_dir / 'sampler_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plots to {plot_path}")

    # Print summary statistics
    print("\nSampler Comparison Summary:")
    summary = results_df.groupby('sampler').agg({
        'temporal_std': ['mean', 'std', 'count'],
        'wwi_percent': ['mean', 'std'],
        'jaccard_with_original': ['mean', 'std'],
        'selection_size': ['mean', 'std'],
        'compute_time': ['mean', 'std']
    }).round(4)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='Compare Samplers on KDR100')
    parser.add_argument('--selection-json', required=True,
                       help='Path to selection results JSON file')

    args = parser.parse_args()

    compare_samplers(args.selection_json)


if __name__ == '__main__':
    main()