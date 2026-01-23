# ruff: noqa: E402
"""
Parameter sweep for selection optimization with optional subset sampling.
Saves results to outputs/optimization_results.csv (full) or to outputs/optimization_results_subset.csv when using subset.
"""

import argparse
import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Project-level imports are deferred or lazy-initialized to prevent heavy
# IO and side-effects at module import time. Use `_init_data()` to populate
# `features_full` and `metadata_full` before running `run_grid()`.

# Configurable parameter grid
min_distance_vals = [40.0, 45.0, 50.0]
temporal_weights = [0.2, 0.35, 0.5]
# expanded a bit to explore
default_n_clusters = [8, 10, 12]

OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

# Lazy placeholders
features_full = None
metadata_full = None


def _init_data():
    """Initialize features_full and metadata_full (called on-demand)."""
    global features_full, metadata_full
    if features_full is not None and metadata_full is not None:
        return

    from src.io import load_metadata, load_or_extract_features
    from src.clustering import ClusteringPipeline

    features_full = load_or_extract_features(
        OUT,
        csv_meta=str(OUT / "metadata.csv") if (OUT / "metadata.csv").exists() else None,
        batch_size=16,
        cache=True,
    )
    # Load metadata via loader so any projected coordinates are attached
    metadata_full = (
        load_metadata(str(OUT / "metadata.csv"))
        if (OUT / "metadata.csv").exists()
        else load_metadata("data/new_all_tiles.csv")
    )


def run_grid(
    subset: int = None,
    sample_method: str = "first",
    random_seed: int = 0,
    out_file: str = None,
    n_samples: int = 40,
    n_clusters_grid: list = None,
):
    """Run the parameter grid; if subset is provided, operate on that subset only."""
    if subset is None:
        features = features_full
        metadata = metadata_full
    else:
        n = min(subset, len(features_full))
        if sample_method == "first":
            idxs = list(range(n))
        elif sample_method == "random":
            rng = random.Random(random_seed)
            idxs = rng.sample(range(len(features_full)), n)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        features = features_full[idxs]
        metadata = metadata_full.iloc[idxs].reset_index(drop=True)
        # preserve projected coords if available
        if getattr(metadata_full, "gdf_metric", None) is not None:
            metadata.gdf_metric = metadata_full.gdf_metric.iloc[idxs].reset_index(drop=True)

    results = []

    clusters_to_test = n_clusters_grid if n_clusters_grid else default_n_clusters

    for min_d, tw, nc in itertools.product(
        min_distance_vals, temporal_weights, clusters_to_test
    ):
        run_start = time.perf_counter()
        entry = {
            "min_distance_km": float(min_d),
            "temporal_weight": float(tw),
            "n_clusters": int(nc),
            "subset_n": len(features),
        }

        try:
            # Clustering
            cl = ClusteringPipeline(n_clusters=nc)
            emb, labels = cl.fit_transform(features)
            print(
                f"  [DEBUG] Clustering done: {nc} clusters -> unique labels: {len(np.unique(labels))}"
            )

            # Selection
            sel = DiversitySelector(n_samples=n_samples)
            print(
                f"  [DEBUG] Calling select with: temporal_weight={tw}, min_distance_km={min_d}"
            )
            selected_idx = sel.select(
                features,
                metadata=metadata,
                temporal_weight=tw,
                spatial_constraint=True,
                min_distance_km=min_d,
            )
            print(f"  [DEBUG] Selected {len(selected_idx)} indices")

            # Metrics
            n_selected = len(selected_idx)
            clusters_covered = (
                int(len(np.unique(labels[selected_idx]))) if n_selected > 0 else 0
            )
            diversity_score = (
                float(sel.get_coverage_statistics(features, labels)["diversity_score"])
                if n_selected > 0
                else 0.0
            )

            # temporal std
            years = metadata.iloc[selected_idx]["year"].dropna().values
            temporal_std = float(np.nanstd(years)) if len(years) > 0 else float("nan")

            # mean pairwise distance
            from math import atan2, cos, radians, sin, sqrt

            def hav(lat1, lon1, lat2, lon2):
                R = 6371
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                return R * c

            # Prefer projected coordinates (_proj_x/_proj_y) when available on metadata
            pairwise = []
            idxs = list(selected_idx)
            use_metric = getattr(metadata, "gdf_metric", None) is not None
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    if use_metric:
                        a = metadata.gdf_metric.loc[idxs[i], ["_proj_x", "_proj_y"]].values.astype(float)
                        b = metadata.gdf_metric.loc[idxs[j], ["_proj_x", "_proj_y"]].values.astype(float)
                        pairwise.append(float((((a - b) ** 2).sum()) ** 0.5 / 1000.0))
                    else:
                        r1 = metadata.iloc[idxs[i]]
                        r2 = metadata.iloc[idxs[j]]
                        pairwise.append(hav(r1["N"], r1["left"], r2["N"], r2["left"]))
            mean_pairwise = float(np.mean(pairwise)) if pairwise else float("nan")

            entry.update(
                {
                    "n_selected": int(n_selected),
                    "clusters_covered": clusters_covered,
                    "diversity_score": diversity_score,
                    "temporal_std": temporal_std,
                    "mean_pairwise_km": mean_pairwise,
                    "success": True,
                }
            )
        except Exception as e:
            entry.update({"success": False, "error": str(e)})

        entry["run_time_s"] = time.perf_counter() - run_start
        print(
            f"Run: min_d={min_d}, tw={tw}, nc={nc}, subset={len(features)} -> success={entry['success']}, time={entry['run_time_s']:.2f}s"
        )
        results.append(entry)

    out_df = pd.DataFrame(results)
    if out_file is None:
        out_file = OUT / (
            f"optimization_results_subset_{len(features)}.csv"
            if subset
            else "optimization_results.csv"
        )
    out_df.to_csv(out_file, index=False)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize selection parameters (grid search)"
    )
    parser.add_argument(
        "--subset", type=int, default=None, help="Use a subset of data for quick runs"
    )
    parser.add_argument(
        "--sample-method",
        choices=["first", "random"],
        default="first",
        help="How to select subset",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file path")
    parser.add_argument(
        "--n-samples", type=int, default=40, help="Number of samples to select"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        default=None,
        help="List of n_clusters to test (default: 8 10 12)",
    )
    args = parser.parse_args()

    df = run_grid(
        subset=args.subset,
        sample_method=args.sample_method,
        random_seed=args.seed,
        out_file=Path(args.out) if args.out else None,
        n_samples=args.n_samples,
        n_clusters_grid=args.n_clusters,
    )
    print("\nDone. Results saved.")
