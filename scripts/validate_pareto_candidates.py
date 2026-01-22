# ruff: noqa: E402
"""Validierung der Pareto-Punkte mittels min_distance Sweep und Bootstrapping.

Für jede Pareto-Lösung werden Runs für unterschiedliche `min_distance_km`
und mehrere Seeds ausgeführt. Ergebnisse werden in `outputs/validation/` gespeichert.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Avoid modifying sys.path at import time; import project modules at runtime when needed
from src.diversity_selector import DiversitySelector
from src.metrics import compute_metrics

OUTDIR = ROOT / "outputs" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)


def validate(
    pareto_csv: str,
    min_distances=[25, 35, 50],
    seeds=[42, 43, 44, 45, 46],
    n_samples: int = 673,
    output_dir: str = None,
):
    pareto = pd.read_csv(pareto_csv)

    outdir = Path(output_dir) if output_dir is not None else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    run_i = 0
    total = len(pareto) * len(min_distances) * len(seeds)

    # Load metadata and features once to save time (from provided outdir if present)
    metadata_path = Path(ROOT) / "outputs" / "metadata.csv"
    _features_path = Path(ROOT) / "outputs" / "features.npy"

    # Allow tests to override by placing files in outdir
    if (outdir / "metadata.csv").exists():
        metadata_path = outdir / "metadata.csv"
    if (outdir / "features.npy").exists():
        # features path intentionally not used in this test-mode helper
        pass

    from src.io import load_metadata, load_or_extract_features

    metadata = (
        pd.read_csv(metadata_path)
        if metadata_path.exists()
        else load_metadata("data/new_all_tiles.csv")
    )
    features = load_or_extract_features(
        outdir,
        csv_meta=(
            str(outdir / "metadata.csv")
            if (outdir / "metadata.csv").exists()
            else str(metadata_path) if metadata_path.exists() else None
        ),
        batch_size=16,
        cache=True,
    )

    # Compute embeddings and cluster labels using ClusteringPipeline (consistent with main pipeline)
    from src.clustering import ClusteringPipeline

    clustering = ClusteringPipeline(n_clusters=8)

    # UMAP can fail on extremely small datasets (test mode). Fallback to trivial embeddings/clusters.
    try:
        embeddings_2d, cluster_labels = clustering.fit_transform(features)
    except Exception as e:
        print(
            f"Warning: UMAP/KMeans failed for small dataset ({e}), using fallback embeddings/labels"
        )
        n = features.shape[0]
        embeddings_2d = np.zeros((n, 2))
        cluster_labels = np.zeros(n, dtype=int)

    # Optional visualizer for maps
    from src.visualizer import Visualizer

    viz = Visualizer(output_dir=str(outdir / "plots"))

    for _, row in pareto.iterrows():
        alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
        for min_d in min_distances:
            for seed in seeds:
                run_i += 1
                print(
                    f"Run {run_i}/{total}: α={alpha}, β={beta}, γ={gamma}, min_dist={min_d}, seed={seed}"
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
                    }
                )
                rows.append(metrics)

                # Save selection snapshot
                sel_df = metadata.iloc[selected].copy()
                sel_df["selection_rank"] = range(len(sel_df))
                sel_file = (
                    outdir / f"selection_a{alpha}_b{beta}_g{gamma}_d{min_d}_s{seed}.csv"
                )
                sel_df.to_csv(sel_file, index=False)

                # Generate maps and plots for this selection
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
    )
    parser.add_argument("--min-dist", type=int, nargs="+", default=[25, 35, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-samples", type=int, default=673)
    args = parser.parse_args()

    validate(
        pareto_csv=args.pareto,
        min_distances=args.min_dist,
        seeds=args.seeds,
        n_samples=args.n_samples,
    )
