"""
Multi-Criteria Temporal Sensitivity Test

Testet ob explizite gamma_temporal Gewichtung messbare Effekte hat.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector

OUT = Path("outputs")
features = np.load(OUT / "features.npy")
metadata = pd.read_csv(OUT / "metadata.csv")

print("=" * 80)
print("MULTI-CRITERIA TEMPORAL SENSITIVITY TEST")
print("=" * 80)

# Test on full dataset
feat = features
meta = metadata

# Clustering
cl = ClusteringPipeline(n_clusters=8)
emb, labels = cl.fit_transform(feat)

# Test various temporal weights
gamma_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

results = []

print(f"\nTesting gamma_temporal (temporal weight): {gamma_values}")
print("Fixed: min_distance_km=40.0, alpha_visual varies to maintain sum=1.0\n")

for gamma in gamma_values:
    # Adjust alpha and beta to maintain sum=1
    beta = 0.15  # Fixed spatial weight
    alpha = 1.0 - beta - gamma  # Visual weight adjusted

    selector = DiversitySelector(n_samples=40, use_multi_criteria=True)

    t0 = time.perf_counter()
    selected = selector.select(
        feat,
        meta,
        spatial_constraint=True,
        min_distance_km=40.0,
        alpha_visual=alpha,
        beta_spatial=beta,
        gamma_temporal=gamma,
    )
    runtime = time.perf_counter() - t0

    # Metrics
    n_sel = len(selected)
    years = meta.iloc[selected]["year"].dropna().values
    temporal_std = float(np.std(years)) if len(years) > 0 else np.nan
    temporal_range = float(np.max(years) - np.min(years)) if len(years) > 0 else np.nan

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
        f"γ={gamma:.2f} (α={alpha:.2f}, β={beta:.2f}): n={n_sel:2d}, "
        f"temp_std={temporal_std:5.2f}, temp_range={temporal_range:4.0f}, "
        f"mean_dist={mean_pairwise:6.1f}km, min_dist={min_pairwise:5.1f}km, t={runtime:.2f}s"
    )

    results.append(
        {
            "gamma_temporal": gamma,
            "alpha_visual": alpha,
            "beta_spatial": beta,
            "n_selected": n_sel,
            "temporal_std": temporal_std,
            "temporal_range": temporal_range,
            "mean_pairwise_km": mean_pairwise,
            "min_pairwise_km": min_pairwise,
            "runtime_s": runtime,
        }
    )

# Analysis
df = pd.DataFrame(results)
df.to_csv(OUT / "multi_criteria_temporal_test.csv", index=False)

print(f"\n{'=' * 80}")
print("ANALYSIS")
print("=" * 80)
print(
    f"Correlation gamma_temporal ↔ temporal_std: r={df['gamma_temporal'].corr(df['temporal_std']):.3f}"
)
print(
    f"Temporal STD range: {df['temporal_std'].min():.2f} - {df['temporal_std'].max():.2f} (Δ={df['temporal_std'].max()-df['temporal_std'].min():.2f})"
)
print(
    f"Temporal range: {df['temporal_range'].min():.0f} - {df['temporal_range'].max():.0f} (Δ={df['temporal_range'].max()-df['temporal_range'].min():.0f})"
)

if abs(df["gamma_temporal"].corr(df["temporal_std"])) > 0.5:
    print("\n✅ SUCCESS: Temporal weight hat signifikanten Effekt!")
    print("   Multi-Criteria Methode funktioniert wie erwartet.")
else:
    print("\n⚠️  Temporal weight zeigt schwachen/keinen Effekt")
    print("   Mögliche Ursache: Spatial constraint zu dominant")

print(f"\nResults saved to: {OUT / 'multi_criteria_temporal_test.csv'}")
