#!/usr/bin/env python3
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
