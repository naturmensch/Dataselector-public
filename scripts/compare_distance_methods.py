#!/usr/bin/env python3
<<<<<<< HEAD
"""Compare Distance Methods.

This script compares different distance methods for clustering and selection
in the context of dataset selection.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Callable
import time

ROOT = Path(__file__).resolve().parents[1]


def euclidean_distance(X: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix."""
    return euclidean_distances(X)


def manhattan_distance(X: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance matrix."""
    return manhattan_distances(X)


def cosine_distance(X: np.ndarray) -> np.ndarray:
    """Compute Cosine distance matrix."""
    return cosine_distances(X)


def mahalanobis_distance(X: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance matrix."""
    # Use identity matrix as covariance (simplified)
    VI = np.linalg.inv(np.cov(X.T) + np.eye(X.shape[1]) * 1e-6)
    return np.array([[np.sqrt((x - y).T @ VI @ (x - y)) for y in X] for x in X])


def run_distance_comparison(
    features_csv: str,
    methods: List[str] = None,
    n_subsets: int = 5,
    subset_sizes: List[int] = None,
    output_dir: str = 'outputs'
):
    """Run distance method comparison."""
    if methods is None:
        methods = ['euclidean', 'manhattan', 'cosine', 'mahalanobis']

    if subset_sizes is None:
        subset_sizes = [50, 100, 200, 500]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load features
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} samples with {df.shape[1]} features")

    # Assume features are in columns (exclude any non-numeric)
    feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    X = df[feature_cols].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Using {X_scaled.shape[1]} numeric features")

    # Define distance functions
    distance_funcs = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'cosine': cosine_distance,
        'mahalanobis': mahalanobis_distance
    }

    results = []

    for method in methods:
        if method not in distance_funcs:
            print(f"Warning: Unknown method {method}, skipping")
            continue

        print(f"Computing {method} distances...")
        start_time = time.time()

        try:
            dist_matrix = distance_funcs[method](X_scaled)
            compute_time = time.time() - start_time

            # Compute statistics
            dist_flat = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            stats_result = {
                'method': method,
                'compute_time': compute_time,
                'mean_distance': np.mean(dist_flat),
                'std_distance': np.std(dist_flat),
                'min_distance': np.min(dist_flat),
                'max_distance': np.max(dist_flat),
                'median_distance': np.median(dist_flat)
            }

            # Test on random subsets
            rng = np.random.RandomState(42)
            for subset_size in subset_sizes:
                if subset_size >= len(X_scaled):
                    continue

                for i in range(n_subsets):
                    indices = rng.choice(len(X_scaled), subset_size, replace=False)
                    subset_dist = dist_matrix[np.ix_(indices, indices)]
                    subset_flat = subset_dist[np.triu_indices_from(subset_dist, k=1)]

                    results.append({
                        **stats_result,
                        'subset_size': subset_size,
                        'subset_id': i,
                        'subset_mean': np.mean(subset_flat),
                        'subset_std': np.std(subset_flat)
                    })

        except Exception as e:
            print(f"Error computing {method}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    csv_path = output_dir / 'distance_methods_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved comparison results to {csv_path}")

    # Plot results
    if not results_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Compute time
        methods_list = results_df['method'].unique()
        times = [results_df[results_df['method'] == m]['compute_time'].iloc[0] for m in methods_list]
        axes[0, 0].bar(methods_list, times)
        axes[0, 0].set_ylabel('Compute Time (seconds)')
        axes[0, 0].set_title('Distance Computation Time')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Mean distance by method
        mean_by_method = results_df.groupby('method')['mean_distance'].first()
        axes[0, 1].bar(mean_by_method.index, mean_by_method.values)
        axes[0, 1].set_ylabel('Mean Distance')
        axes[0, 1].set_title('Mean Distance by Method')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Distance distribution by subset size
        for method in methods_list:
            subset_data = results_df[results_df['method'] == method]
            if not subset_data.empty:
                axes[1, 0].scatter(subset_data['subset_size'], subset_data['subset_mean'],
                                 label=method, alpha=0.7)

        axes[1, 0].set_xlabel('Subset Size')
        axes[1, 0].set_ylabel('Mean Distance')
        axes[1, 0].set_title('Distance vs Subset Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Standard deviation
        std_by_method = results_df.groupby('method')['std_distance'].first()
        axes[1, 1].bar(std_by_method.index, std_by_method.values)
        axes[1, 1].set_ylabel('Distance Std Dev')
        axes[1, 1].set_title('Distance Variability by Method')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = output_dir / 'distance_methods_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plots to {plot_path}")

        # Print summary
        print("\nDistance Methods Summary:")
        summary = results_df.groupby('method').agg({
            'compute_time': 'first',
            'mean_distance': 'first',
            'std_distance': 'first',
            'min_distance': 'first',
            'max_distance': 'first'
        }).round(4)
        print(summary)


def main():
    parser = argparse.ArgumentParser(description='Compare Distance Methods')
    parser.add_argument('--features-csv', required=True,
                       help='Path to features CSV file')
    parser.add_argument('--methods', nargs='+',
                       default=['euclidean', 'manhattan', 'cosine', 'mahalanobis'],
                       help='Distance methods to compare')
    parser.add_argument('--n-subsets', type=int, default=5,
                       help='Number of random subsets per size')
    parser.add_argument('--subset-sizes', nargs='+', type=int,
                       default=[50, 100, 200, 500],
                       help='Subset sizes to test')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')

    args = parser.parse_args()

    run_distance_comparison(
        args.features_csv,
        args.methods,
        args.n_subsets,
        args.subset_sizes,
        args.output_dir
    )


if __name__ == '__main__':
    main()
=======
"""Compare legacy Haversine distances with GeoPandas UTM metric distances.

Generates a JSON report and simple plots saved to outputs/.
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point

from src.spatial_facility_location import haversine_distance

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "new_all_tiles.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_haversine_matrix(df: pd.DataFrame) -> np.ndarray:
    lats = df["N"].to_numpy()
    lons = df["left"].to_numpy()
    n = len(lats)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
            mat[i, j] = mat[j, i] = d
    return mat


def compute_utm_matrix(df: pd.DataFrame, target_crs: str = "EPSG:25832") -> np.ndarray:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["left"].to_numpy(), df["N"].to_numpy())],
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(target_crs)
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    n = len(xs)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            m = (dx * dx + dy * dy) ** 0.5
            mat[i, j] = mat[j, i] = m / 1000.0
    return mat


def summarize_and_plot(hav: np.ndarray, utm: np.ndarray, df: pd.DataFrame):
    assert hav.shape == utm.shape
    n = hav.shape[0]
    # take upper triangle i<j
    iu1 = np.triu_indices(n, k=1)
    hav_vals = hav[iu1]
    utm_vals = utm[iu1]
    abs_diff = np.abs(hav_vals - utm_vals)
    rel_diff = abs_diff / np.maximum(1e-6, utm_vals)

    report = {
        "pairs": int(len(hav_vals)),
        "hav_min_km": float(hav_vals.min()),
        "hav_median_km": float(np.median(hav_vals)),
        "hav_max_km": float(hav_vals.max()),
        "utm_min_km": float(utm_vals.min()),
        "utm_median_km": float(np.median(utm_vals)),
        "utm_max_km": float(utm_vals.max()),
        "abs_diff_median_km": float(np.median(abs_diff)),
        "abs_diff_max_km": float(abs_diff.max()),
        "rel_diff_median": float(np.median(rel_diff)),
    }

    # Per-tile statistics (median/mean/max absolute diff to all others)
    per_tile = []
    # build full absolute-difference matrix
    full_abs = np.abs(hav - utm)
    for i in range(n):
        # exclude diagonal
        vals = np.delete(full_abs[i, :], i)
        per_tile.append(
            {
                "index": int(i),
                "sheet": df.iloc[i].shortName
                if "shortName" in df.columns
                else df.index[i],
                "median_abs_diff_km": float(np.median(vals)),
                "mean_abs_diff_km": float(np.mean(vals)),
                "max_abs_diff_km": float(np.max(vals)),
            }
        )

    # save per-tile stats
    per_tile_path = OUT_DIR / "distance_comparison_per_tile.json"
    with per_tile_path.open("w") as fh:
        json.dump({"per_tile": per_tile}, fh, indent=2)

    # add top outliers to report
    per_tile_sorted = sorted(
        per_tile, key=lambda x: x["median_abs_diff_km"], reverse=True
    )
    report["top_10_median_abs_diff"] = per_tile_sorted[:10]

    # Scatter plot Haversine vs UTM
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=utm_vals, y=hav_vals, alpha=0.2)
    plt.plot([0, utm_vals.max()], [0, utm_vals.max()], color="k", ls="--")
    plt.xlabel("UTM distance (km)")
    plt.ylabel("Haversine distance (km)")
    plt.title("Haversine vs UTM distances")
    plt.tight_layout()
    scatter_path = OUT_DIR / "distance_comparison_scatter.png"
    plt.savefig(scatter_path)
    plt.close()

    # Histogram of absolute differences
    plt.figure(figsize=(6, 4))
    sns.histplot(abs_diff, bins=100)
    plt.xlabel("Absolute difference (km)")
    plt.title("Distribution of |Haversine - UTM|")
    plt.tight_layout()
    hist_path = OUT_DIR / "distance_comparison_hist.png"
    plt.savefig(hist_path)
    plt.close()

    # Heatmap / spatial plot: color each tile by median_abs_diff
    try:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[
                Point(xy) for xy in zip(df["left"].to_numpy(), df["N"].to_numpy())
            ],
            crs="EPSG:4326",
        )
        per_median = np.array([p["median_abs_diff_km"] for p in per_tile])
        gdf["median_abs_diff_km"] = per_median
        gdf = gdf.to_crs("EPSG:3857")

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        gdf.plot(
            column="median_abs_diff_km", cmap="YlOrRd", legend=True, markersize=8, ax=ax
        )
        plt.title("Median |Haversine - UTM| per Tile (km)")
        heatmap_path = OUT_DIR / "distance_comparison_heatmap.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
    except Exception as e:
        print("Warning: could not produce spatial heatmap:", e)
        heatmap_path = None

    # Save small report (including top outliers)
    out_json = OUT_DIR / "distance_comparison_report.json"
    with out_json.open("w") as fh:
        json.dump(report, fh, indent=2)

    print(f"Report written to: {out_json}")
    print(f"Scatter: {scatter_path}, Hist: {hist_path}, Heatmap: {heatmap_path}")

    return report


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} rows from {DATA_CSV}")

    hav = compute_haversine_matrix(df)
    utm = compute_utm_matrix(df)

    report = summarize_and_plot(hav, utm, df)
    print(report)


if __name__ == "__main__":
    main()
>>>>>>> chore/ci-lint-attrs-gdf
