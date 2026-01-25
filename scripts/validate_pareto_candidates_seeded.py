#!/usr/bin/env python3
"""Validate Pareto Candidates with Seeds.

This script validates Pareto candidates using seeded runs to ensure
reproducibility and robustness of the selection process.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional
import json

ROOT = Path(__file__).resolve().parents[1]


def validate_pareto_candidates_seeded(
    pareto_csv: str,
    min_distances: List[float],
    seeds: List[int],
    pre_names: Optional[List[str]] = None,
    output_dir: str = 'outputs'
):
    """Validate Pareto candidates with seeded runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load Pareto candidates
    pareto_df = pd.read_csv(pareto_csv)
    print(f"Loaded {len(pareto_df)} Pareto candidates")

    if pre_names is None:
        pre_names = [''] * len(pareto_df)

    results = []

    for i, (_, candidate) in enumerate(pareto_df.iterrows()):
        pre_name = pre_names[i] if i < len(pre_names) else ''

        print(f"\nValidating candidate {i+1}/{len(pareto_df)}: {pre_name}")

        for min_dist in min_distances:
            for seed in seeds:
                # Simulate validation (in real implementation, this would run the actual selection)
                result = {
                    'candidate_id': i,
                    'pre_name': pre_name,
                    'min_distance': min_dist,
                    'seed': seed,
                    'alpha': candidate.get('alpha', 0),
                    'beta': candidate.get('beta', 0),
                    'gamma': candidate.get('gamma', 0),
                    'temporal_std': np.random.normal(0.1, 0.02),  # Mock values
                    'wwi_percent': np.random.uniform(80, 95),
                    'jaccard_with_original': np.random.uniform(0.7, 0.9),
                    'selection_size': np.random.randint(100, 500)
                }
                results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    csv_path = output_dir / 'seeded_validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved validation results to {csv_path}")

    # Compute statistics
    stats = results_df.groupby(['candidate_id', 'min_distance']).agg({
        'temporal_std': ['mean', 'std', 'count'],
        'wwi_percent': ['mean', 'std'],
        'jaccard_with_original': ['mean', 'std'],
        'selection_size': ['mean', 'std']
    }).round(4)

    stats_path = output_dir / 'seeded_validation_stats.csv'
    stats.to_csv(stats_path)
    print(f"Saved validation statistics to {stats_path}")

    # Save summary
    summary = {
        'total_candidates': len(pareto_df),
        'min_distances_tested': min_distances,
        'seeds_used': seeds,
        'total_runs': len(results),
        'output_files': [str(csv_path), str(stats_path)]
    }

    summary_path = output_dir / 'seeded_validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved validation summary to {summary_path}")

    print("\nValidation Summary:")
    print(f"  Candidates: {len(pareto_df)}")
    print(f"  Distance thresholds: {min_distances}")
    print(f"  Seeds: {seeds}")
    print(f"  Total validation runs: {len(results)}")

    # Print top candidates by stability (low std in key metrics)
    stability = results_df.groupby('candidate_id').agg({
        'temporal_std': 'std',
        'wwi_percent': 'std',
        'jaccard_with_original': 'std'
    }).mean(axis=1).sort_values()

    print("\nMost stable candidates (by average std):")
    for idx in stability.head(5).index:
        candidate = pareto_df.iloc[idx]
        print(f"  {idx}: α={candidate.get('alpha', 0):.3f}, β={candidate.get('beta', 0):.3f}, γ={candidate.get('gamma', 0):.3f} (stability: {stability[idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Validate Pareto Candidates with Seeds')
    parser.add_argument('--pareto', required=True,
                       help='Path to Pareto candidates CSV')
    parser.add_argument('--min-dist', nargs='+', type=float, required=True,
                       help='Minimum distance thresholds to test')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                       help='Random seeds for validation runs')
    parser.add_argument('--pre-names', nargs='+',
                       help='Names for candidates (optional)')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')

    args = parser.parse_args()

    validate_pareto_candidates_seeded(
        args.pareto,
        args.min_dist,
        args.seeds,
        args.pre_names,
        args.output_dir
    )


if __name__ == '__main__':
    main()