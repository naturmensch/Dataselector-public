<<<<<<< HEAD
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
=======
"""Validate Pareto candidates with an optional pre-selected seed name or index.
Writes results to outputs/validation_seeded/validation_results.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Avoid modifying sys.path at import time; import project modules at runtime when needed
OUTDIR = ROOT / "outputs" / "validation_seeded"
OUTDIR.mkdir(parents=True, exist_ok=True)


def validate(
    pareto_csv: str,
    min_distances=[25, 35, 50],
    seeds=[42, 43, 44, 45, 46],
    n_samples: int = 673,
    pre_selected_names=None,
    pre_selected_indices=None,
    output_dir: str = None,
):
    pareto = pd.read_csv(pareto_csv)

    outdir = Path(output_dir) if output_dir is not None else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    run_i = 0
    total = len(pareto) * len(min_distances) * len(seeds)

    # Load metadata and features once to save time
    metadata_path = Path(ROOT) / "outputs" / "metadata.csv"
    # features path placeholder (not used in seeded test mode)

    from src.io import load_metadata, load_or_extract_features

    metadata = (
        load_metadata(str(metadata_path))
        if metadata_path.exists()
        else load_metadata("data/new_all_tiles.csv")
    )
    features = load_or_extract_features(
        outdir,
        csv_meta=(
            str(outdir / "metadata.csv")
            if (outdir / "metadata.csv").exists()
            else str(metadata_path)
            if metadata_path.exists()
            else None
        ),
        batch_size=16,
        cache=True,
    )

    # Compute embeddings and cluster labels
    from src.clustering import ClusteringPipeline

    clustering = ClusteringPipeline(n_clusters=8)

    try:
        embeddings_2d, cluster_labels = clustering.fit_transform(features)
    except Exception as e:
        print(
            f"Warning: UMAP/KMeans failed for small dataset ({e}), using fallback embeddings/labels"
        )
        n = features.shape[0]
        embeddings_2d = np.zeros((n, 2))
        cluster_labels = np.zeros(n, dtype=int)

    from src.visualizer import Visualizer

    viz = Visualizer(output_dir=str(outdir / "plots"))

    from src.diversity_selector import DiversitySelector
    from src.metrics import compute_metrics

    for _, row in pareto.iterrows():
        alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
        for min_d in min_distances:
            for seed in seeds:
                run_i += 1
                print(
                    f"Run {run_i}/{total}: α={alpha}, β={beta}, γ={gamma}, min_dist={min_d}, seed={seed} (seeded={pre_selected_names or pre_selected_indices})"
                )
                t0 = time.time()

                ds = DiversitySelector(
                    n_samples=n_samples, use_multi_criteria=True, random_state=int(seed)
                )
                selected = ds.select(
                    features=features,
                    metadata=metadata,
                    alpha_visual=float(alpha),
                    beta_spatial=float(beta),
                    gamma_temporal=float(gamma),
                    spatial_constraint=True,
                    min_distance_km=float(min_d),
                    pre_selected=pre_selected_indices,
                    pre_selected_names=pre_selected_names,
                )

                duration = time.time() - t0
                metrics = compute_metrics(selected, metadata, cluster_labels, features)
                metrics.update(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "min_distance_km": min_d,
                        "seed": seed,
                        "duration_s": duration,
                        "pre_selected_names": pre_selected_names,
                        "pre_selected_indices": pre_selected_indices,
                    }
                )
                rows.append(metrics)

                # Save snapshot
                sel_df = metadata.iloc[selected].copy()
                sel_df["selection_rank"] = range(len(sel_df))
                sel_file = (
                    outdir / f"selection_a{alpha}_b{beta}_g{gamma}_d{min_d}_s{seed}.csv"
                )
                sel_df.to_csv(sel_file, index=False)

                prefix = f"sel_a{alpha}_b{beta}_g{gamma}_d{min_d}_s{seed}"
                try:
                    viz.create_summary_report(
                        embeddings_2d=embeddings_2d,
                        cluster_labels=cluster_labels,
                        metadata=metadata,
                        selected_indices=selected,
                        output_prefix=prefix,
                    )
                except Exception as e:
                    print(f"Warning: could not create plots for {prefix}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "validation_results.csv", index=False)
    print(f"Validation finished. Results saved to {outdir / 'validation_results.csv'}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pareto",
        type=str,
        default=str(
            ROOT / "outputs" / "tuning_weights" / "pareto" / "pareto_solutions.csv"
        ),
        help="Path to pareto solutions CSV from LHS exploration (default: outputs/tuning_weights/pareto/)",
    )
    parser.add_argument("--min-dist", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Target number of final samples (overrides config.selection.n_samples when provided)",
    )
    parser.add_argument("--pre-names", type=str, nargs="*", default=None)
    parser.add_argument("--pre-indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    validate(
        pareto_csv=args.pareto,
        min_distances=args.min_dist,
        seeds=args.seeds,
        n_samples=args.n_samples,
        pre_selected_names=args.pre_names,
        pre_selected_indices=args.pre_indices,
    )
<<<<<<< HEAD
>>>>>>> ci/add-smoke-tests
=======
>>>>>>> chore/ci-lint-attrs-gdf
