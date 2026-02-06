"""Validation workflows for Pareto solutions and selection candidates.

Provides robust multi-seed, multi-constraint validation of Pareto-optimal
hyperparameter configurations identified during exploration/optimization phases.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def validate_pareto_candidates(
    pareto_csv: str | Path,
    min_distances: List[float] = None,
    seeds: List[int] = None,
    n_samples: int = 673,
    output_dir: str | Path = None,
) -> pd.DataFrame:
    """Validate Pareto-optimal candidates via min_distance sweep + bootstrapping.

    For each Pareto solution, runs selections across multiple `min_distance_km`
    values and multiple random seeds. Generates comprehensive validation metrics,
    selection snapshots, and visualizations.

    Args:
        pareto_csv: Path to Pareto solutions CSV (α, β, γ columns)
        min_distances: List of min_distance_km values to test (default: [25, 35, 50])
        seeds: Random seeds for bootstrapping (default: [42, 43, 44, 45, 46])
        n_samples: Target sample size (default: 673)
        output_dir: Output directory for results (default: outputs/validation/)

    Returns:
        DataFrame with validation results (metrics × configurations × seeds)

    Example:
        >>> results = validate_pareto_candidates(
        ...     pareto_csv="outputs/tuning_weights/pareto/pareto_solutions.csv",
        ...     min_distances=[25, 35, 50],
        ...     seeds=[42, 43, 44],
        ...     n_samples=673
        ... )
        >>> print(f"Tested {len(results)} configurations")
    """
    # Lazy imports (avoid heavy dependencies at module load)
    from dataselector.analysis.metrics import compute_metrics
    from dataselector.analysis.visualizer import Visualizer
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector

    # Default parameters
    if min_distances is None:
        min_distances = [25, 35, 50]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    # Setup paths
    pareto_csv = Path(pareto_csv)
    if not pareto_csv.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {pareto_csv}")

    root = Path.cwd()
    if output_dir is None:
        output_dir = root / "outputs" / "validation"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Pareto solutions
    pareto = pd.read_csv(pareto_csv)
    if not {"alpha", "beta", "gamma"}.issubset(pareto.columns):
        raise ValueError(
            f"Pareto CSV must contain alpha, beta, gamma columns. Found: {pareto.columns.tolist()}"
        )

    # Load data once (reuse across all validation runs)
    metadata_path_default = (
        root / "outputs" / "metadata.csv"
    )
    metadata_path = (
        output_dir / "metadata.csv"
        if (output_dir / "metadata.csv").exists()
        else metadata_path_default
    )

    metadata = load_metadata(
        str(metadata_path) if metadata_path.exists() else "data/new_all_tiles.csv"
    )

    features = load_or_extract_features(
        output_dir,
        csv_meta=str(metadata_path) if metadata_path.exists() else None,
        batch_size=16,
        cache=True,
    )

    # Compute embeddings and cluster labels (consistent with main pipeline)
    clustering = ClusteringPipeline(n_clusters=8)

    try:
        embeddings_2d, cluster_labels = clustering.fit_transform(features)
    except Exception as e:
        # Fallback for extremely small datasets (test mode)
        print(f"Warning: UMAP/KMeans failed ({e}), using fallback embeddings/labels")
        n = features.shape[0]
        embeddings_2d = np.zeros((n, 2))
        cluster_labels = np.zeros(n, dtype=int)

    # Setup visualizer
    viz = Visualizer(output_dir=str(output_dir / "plots"))

    # Run validation sweep
    rows = []
    run_i = 0
    total = len(pareto) * len(min_distances) * len(seeds)

    for _, row in pareto.iterrows():
        alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]

        for min_d in min_distances:
            for seed in seeds:
                run_i += 1
                print(
                    f"Run {run_i}/{total}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, "
                    f"min_dist={min_d}km, seed={seed}"
                )

                t0 = time.time()

                # Run selection with current configuration
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
                )

                duration = time.time() - t0

                # Compute metrics
                metrics = compute_metrics(selected, metadata, cluster_labels, features)
                metrics.update(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "min_distance_km": min_d,
                        "seed": seed,
                        "duration_s": duration,
                    }
                )
                rows.append(metrics)

                # Save selection snapshot
                sel_df = metadata.iloc[selected].copy()
                sel_df["selection_rank"] = range(len(sel_df))
                sel_file = (
                    output_dir
                    / f"selection_a{alpha}_b{beta}_g{gamma}_d{min_d}_s{seed}.csv"
                )
                sel_df.to_csv(sel_file, index=False)

                # Generate visualizations
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

    # Save validation results
    df = pd.DataFrame(rows)
    results_path = output_dir / "validation_results.csv"
    df.to_csv(results_path, index=False)

    print(f"\n✓ Validation finished. Results: {results_path}")
    print(f"  - {len(pareto)} Pareto candidates")
    print(f"  - {len(min_distances)} min_distance values: {min_distances}")
    print(f"  - {len(seeds)} seeds: {seeds}")
    print(f"  - {total} total configurations validated")

    return df
