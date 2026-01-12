"""
Vergleichstest: Legacy Post-Filter vs Constraint-integrierte Optimierung

Testet beide Methoden auf Subset (N=50) und Full (N=673) und vergleicht:
- Anzahl selektierter Samples
- Temporal diversity (std)
- Spatial distribution (mean pairwise distance)
- Cluster coverage
- Laufzeit
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector

# Load cached features
OUT = Path("outputs")
features = np.load(OUT / "features.npy")
metadata = pd.read_csv(OUT / "metadata.csv")

print("=" * 80)
print("CONSTRAINT-INTEGRATION TEST: Legacy vs Scientific")
print("=" * 80)

# Test configurations
configs = [
    {"name": "Subset N=50", "n": 50, "n_clusters": 8},
    {"name": "Full N=673", "n": None, "n_clusters": 8},
]

parameters = [
    {"min_distance_km": 40.0, "temporal_weight": 0.2},
    {"min_distance_km": 40.0, "temporal_weight": 0.5},
    {"min_distance_km": 50.0, "temporal_weight": 0.2},
]

results = []

for config in configs:
    print(f"\n{'=' * 80}")
    print(f"CONFIG: {config['name']}")
    print("=" * 80)

    # Subset if needed
    if config["n"] is not None:
        feat = features[: config["n"]]
        meta = metadata.iloc[: config["n"]].reset_index(drop=True)
    else:
        feat = features
        meta = metadata

    # Clustering (once per config)
    cl = ClusteringPipeline(n_clusters=config["n_clusters"])
    emb, labels = cl.fit_transform(feat)

    for params in parameters:
        print(
            f"\n--- Parameters: min_d={params['min_distance_km']}km, tw={params['temporal_weight']} ---"
        )

        for method in ["legacy", "constraint_integrated"]:
            use_integration = method == "constraint_integrated"

            selector = DiversitySelector(
                n_samples=34, use_constraint_integration=use_integration
            )

            t0 = time.perf_counter()
            selected = selector.select(
                feat,
                meta,
                temporal_weight=params["temporal_weight"],
                spatial_constraint=True,
                min_distance_km=params["min_distance_km"],
            )
            runtime = time.perf_counter() - t0

            # Metrics
            n_selected = len(selected)
            clusters_covered = len(np.unique(labels[selected]))

            # Temporal
            years = meta.iloc[selected]["year"].dropna().values
            temporal_std = float(np.std(years)) if len(years) > 0 else np.nan

            # Spatial
            from src.spatial_facility_location import haversine_distance

            pairwise = []
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    r1 = meta.iloc[selected[i]]
                    r2 = meta.iloc[selected[j]]
                    pairwise.append(
                        haversine_distance(r1["N"], r1["left"], r2["N"], r2["left"])
                    )
            mean_pairwise = float(np.mean(pairwise)) if pairwise else np.nan
            min_pairwise = float(np.min(pairwise)) if pairwise else np.nan

            print(
                f"  [{method:20s}] n={n_selected:2d}, clusters={clusters_covered}/{config['n_clusters']}, "
                f"temporal_std={temporal_std:.2f}, mean_dist={mean_pairwise:.1f}km, "
                f"min_dist={min_pairwise:.1f}km, time={runtime:.2f}s"
            )

            results.append(
                {
                    "config": config["name"],
                    "n_samples": len(feat),
                    "method": method,
                    "min_distance_km": params["min_distance_km"],
                    "temporal_weight": params["temporal_weight"],
                    "n_selected": n_selected,
                    "clusters_covered": clusters_covered,
                    "temporal_std": temporal_std,
                    "mean_pairwise_km": mean_pairwise,
                    "min_pairwise_km": min_pairwise,
                    "runtime_s": runtime,
                }
            )

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT / "method_comparison.csv", index=False)

print(f"\n{'=' * 80}")
print("COMPARISON SUMMARY")
print("=" * 80)

for config_name in df["config"].unique():
    print(f"\n{config_name}:")
    subset = df[df["config"] == config_name]

    # Group by parameters and compare methods
    for (min_d, tw), group in subset.groupby(["min_distance_km", "temporal_weight"]):
        print(f"\n  min_d={min_d}km, tw={tw}:")
        legacy = group[group["method"] == "legacy"].iloc[0]
        integrated = group[group["method"] == "constraint_integrated"].iloc[0]

        print(
            f"    Legacy:      n={legacy['n_selected']:2d}, temporal_std={legacy['temporal_std']:.2f}, mean_dist={legacy['mean_pairwise_km']:.1f}km"
        )
        print(
            f"    Integrated:  n={integrated['n_selected']:2d}, temporal_std={integrated['temporal_std']:.2f}, mean_dist={integrated['mean_pairwise_km']:.1f}km"
        )

        # Differences
        delta_n = integrated["n_selected"] - legacy["n_selected"]
        delta_temporal = integrated["temporal_std"] - legacy["temporal_std"]
        delta_spatial = integrated["mean_pairwise_km"] - legacy["mean_pairwise_km"]

        print(
            f"    Δ:           n={delta_n:+3d}, temporal_std={delta_temporal:+.2f}, mean_dist={delta_spatial:+.1f}km"
        )

print(f"\n{'=' * 80}")
print(f"Results saved to: {OUT / 'method_comparison.csv'}")
print("=" * 80)
