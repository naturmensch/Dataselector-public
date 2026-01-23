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
            else str(metadata_path) if metadata_path.exists() else None
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
