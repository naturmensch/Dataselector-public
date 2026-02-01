# ruff: noqa: E402
"""
Multi-Criteria Performance-Test: n_samples=673 (Full Dataset)

Testet:
- n_samples: 673 (volle Coverage)
- alpha_visual: 0.60
- beta_spatial: 0.15
- gamma_temporal: 0.25
- min_distance_km: 0.0 (keine harte Filterung)

Ergebnisse:
- Laufzeit
- Verteilung (WWI-Anteil, Temporal STD, Spatial Mean)
- Cluster Coverage
"""

import time
from pathlib import Path

import numpy as np

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector
from src.spatial_facility_location import haversine_distance

OUT = Path("outputs")
from src.io import load_metadata, load_or_extract_features

csv_meta = OUT / "metadata.csv"
csv_meta = str(csv_meta) if csv_meta.exists() else None
features = load_or_extract_features(
    out_dir=OUT, csv_meta=csv_meta, batch_size=16, cache=False
)
metadata = load_metadata(csv_meta if csv_meta is not None else "data/new_all_tiles.csv")

print("=" * 80)
print("MULTI-CRITERIA PERFORMANCE TEST: n_samples=673 (Full Dataset)")
print("=" * 80)
print(f"Dataset: {len(features)} Tiles")

# Clustering
cl = ClusteringPipeline(n_clusters=8)
emb, labels = cl.fit_transform(features)

selector = DiversitySelector(n_samples=673, use_multi_criteria=True)

start = time.perf_counter()
selected = selector.select(
    features,
    metadata,
    spatial_constraint=True,
    min_distance_km=0.0,  # spatial constraint nur als Kriterium, nicht als Filter
    alpha_visual=0.60,
    beta_spatial=0.15,
    gamma_temporal=0.25,
)
runtime = time.perf_counter() - start

n_sel = len(selected)
clusters_covered = len(np.unique(labels[selected]))
years = metadata.iloc[selected]["year"].dropna().values
temporal_std = float(np.std(years)) if len(years) > 0 else np.nan
temporal_range = float(np.max(years) - np.min(years)) if len(years) > 0 else np.nan
temporal_mean = float(np.mean(years)) if len(years) > 0 else np.nan
wwi_years = (years >= 1906) & (years <= 1918)
wwi_fraction = float(np.sum(wwi_years) / len(years)) if len(years) > 0 else np.nan

pairwise = []
for i in range(len(selected)):
    for j in range(i + 1, len(selected)):
        r1 = metadata.iloc[selected[i]]
        r2 = metadata.iloc[selected[j]]
        pairwise.append(haversine_distance(r1["N"], r1["left"], r2["N"], r2["left"]))
mean_pairwise = float(np.mean(pairwise)) if pairwise else np.nan
min_pairwise = float(np.min(pairwise)) if pairwise else np.nan
std_pairwise = float(np.std(pairwise)) if pairwise else np.nan

print("\nErgebnisse:")
print(f"  n_selected: {n_sel} (erwartet: 673)")
print(f"  Cluster Coverage: {clusters_covered}/8")
print(f"  Temporal STD: {temporal_std:.2f}")
print(f"  Temporal Range: {temporal_range:.0f} Jahre")
print(f"  WWI-Anteil: {wwi_fraction*100:.1f}%")
print(f"  Spatial Mean Distance: {mean_pairwise:.1f}km")
print(f"  Min Distance: {min_pairwise:.1f}km")
print(f"  Runtime: {runtime:.2f}s")

# Save selection
metadata.iloc[selected].to_csv(OUT / "multi_criteria_full_selection.csv", index=False)

print("\nSelection saved to: outputs/multi_criteria_full_selection.csv")
