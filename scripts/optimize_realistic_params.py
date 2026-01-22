"""
Optimierter Parameter-Test basierend auf Dataset-Erkenntnissen.

Testet:
- n_samples: [40, 60, 80] (aktuell nur 11% des Potentials)
- gamma_temporal: [0.10, 0.15, 0.20, 0.25] (gegen WWI-Clustering)
- min_distance_km: [35, 40] (konservativ vs. moderat)

Fixed: alpha + beta + gamma = 1.0, beta_spatial = 0.15
"""

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")


def main() -> int:
    from src.clustering import ClusteringPipeline
    from src.diversity_selector import DiversitySelector
    from src.spatial_facility_location import haversine_distance
    from src.io import load_metadata, load_or_extract_features

    features = load_or_extract_features(
        OUT,
        csv_meta=str(OUT / "metadata.csv") if (OUT / "metadata.csv").exists() else None,
        batch_size=16,
        cache=True,
    )
    metadata = (
        pd.read_csv(OUT / "metadata.csv")
        if (OUT / "metadata.csv").exists()
        else load_metadata("data/new_all_tiles.csv")
    )

    print("=" * 80)
    print("REALITÄTS-ANGEPASSTE PARAMETER-OPTIMIERUNG")
    print("=" * 80)
    print(f"\nDataset: {len(features)} Tiles")
    print("Temporale Konzentration: 42.8% in WWI-Ära (1906-1918)")
    print("Theoretisches Maximum: ~364 Tiles bei min_d=40km")
    print("Aktuell selektiert: 40 (11% des Potentials)")
    print()

    # Clustering (einmal für alle)
    print("Führe Clustering durch...")
    cl = ClusteringPipeline(n_clusters=8)
    emb, labels = cl.fit_transform(features)
    print("✓ Clustering abgeschlossen\n")

    # Parameter-Grid
    n_samples_vals = [40, 60, 80]
    gamma_temporal_vals = [0.10, 0.15, 0.20, 0.25]
    min_distance_vals = [35.0, 40.0]
    beta_spatial = 0.15  # Fixed

    results = []

    total_runs = len(n_samples_vals) * len(gamma_temporal_vals) * len(min_distance_vals)
    run_num = 0

    print(f"Starte {total_runs} Parameter-Kombinationen...\n")
    print("-" * 80)

    for n_samp, gamma, min_d in itertools.product(
        n_samples_vals, gamma_temporal_vals, min_distance_vals
    ):
        run_num += 1
        alpha = 1.0 - beta_spatial - gamma

        print(
            f"[{run_num:2d}/{total_runs}] n={n_samp}, γ={gamma:.2f}, min_d={min_d:.0f}km ",
            end="",
            flush=True,
        )

        selector = DiversitySelector(n_samples=n_samp, use_multi_criteria=True)

        t0 = time.perf_counter()
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_d,
            alpha_visual=alpha,
            beta_spatial=beta_spatial,
            gamma_temporal=gamma,
        )
        runtime = time.perf_counter() - t0

        # Metrics
        n_sel = len(selected)
        clusters_covered = len(np.unique(labels[selected]))

        # Temporal
        years = metadata.iloc[selected]["year"].dropna().values
        temporal_std = float(np.std(years)) if len(years) > 0 else np.nan
        temporal_range = float(np.max(years) - np.min(years)) if len(years) > 0 else np.nan
        temporal_mean = float(np.mean(years)) if len(years) > 0 else np.nan

        # WWI concentration check
        wwi_years = (years >= 1906) & (years <= 1918)
        wwi_fraction = float(np.sum(wwi_years) / len(years)) if len(years) > 0 else np.nan

        # Spatial
        pairwise = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                r1 = metadata.iloc[selected[i]]
                r2 = metadata.iloc[selected[j]]
                pairwise.append(
                    haversine_distance(r1["N"], r1["left"], r2["N"], r2["left"])
                )

        mean_pairwise = float(np.mean(pairwise)) if pairwise else np.nan
        min_pairwise = float(np.min(pairwise)) if pairwise else np.nan
        std_pairwise = float(np.std(pairwise)) if pairwise else np.nan

        print(
            f"→ n={n_sel:2d}, clusters={clusters_covered}/8, temp_std={temporal_std:5.2f}, "
            f"WWI%={100*wwi_fraction:4.1f}%, mean_dist={mean_pairwise:5.1f}km, t={runtime:.2f}s"
        )

        results.append(
            {
                "n_samples_target": n_samp,
                "gamma_temporal": gamma,
                "alpha_visual": alpha,
                "beta_spatial": beta_spatial,
                "min_distance_km": min_d,
                "n_selected": n_sel,
                "clusters_covered": clusters_covered,
                "temporal_std": temporal_std,
                "temporal_range": temporal_range,
                "temporal_mean": temporal_mean,
                "wwi_fraction": wwi_fraction,
                "mean_pairwise_km": mean_pairwise,
                "min_pairwise_km": min_pairwise,
                "std_pairwise_km": std_pairwise,
                "runtime_s": runtime,
            }
        )

    print("-" * 80)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUT / "optimized_parameters.csv", index=False)

    # (Remaining reporting/analysis kept as before)
    print(f"\n{'=' * 80}")
    print("ANALYSE DER ERGEBNISSE")
    print("=" * 80)

    # Best by different criteria
    print("\n1. BESTE KONFIGURATION NACH KRITERIEN:\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# Clustering (einmal für alle)
print("Führe Clustering durch...")
cl = ClusteringPipeline(n_clusters=8)
emb, labels = cl.fit_transform(features)
print("✓ Clustering abgeschlossen\n")

# Parameter-Grid
n_samples_vals = [40, 60, 80]
gamma_temporal_vals = [0.10, 0.15, 0.20, 0.25]
min_distance_vals = [35.0, 40.0]
beta_spatial = 0.15  # Fixed

results = []

total_runs = len(n_samples_vals) * len(gamma_temporal_vals) * len(min_distance_vals)
run_num = 0

print(f"Starte {total_runs} Parameter-Kombinationen...\n")
print("-" * 80)

for n_samp, gamma, min_d in itertools.product(
    n_samples_vals, gamma_temporal_vals, min_distance_vals
):
    run_num += 1
    alpha = 1.0 - beta_spatial - gamma

    print(
        f"[{run_num:2d}/{total_runs}] n={n_samp}, γ={gamma:.2f}, min_d={min_d:.0f}km ",
        end="",
        flush=True,
    )

    selector = DiversitySelector(n_samples=n_samp, use_multi_criteria=True)

    t0 = time.perf_counter()
    selected = selector.select(
        features,
        metadata,
        spatial_constraint=True,
        min_distance_km=min_d,
        alpha_visual=alpha,
        beta_spatial=beta_spatial,
        gamma_temporal=gamma,
    )
    runtime = time.perf_counter() - t0

    # Metrics
    n_sel = len(selected)
    clusters_covered = len(np.unique(labels[selected]))

    # Temporal
    years = metadata.iloc[selected]["year"].dropna().values
    temporal_std = float(np.std(years)) if len(years) > 0 else np.nan
    temporal_range = float(np.max(years) - np.min(years)) if len(years) > 0 else np.nan
    temporal_mean = float(np.mean(years)) if len(years) > 0 else np.nan

    # WWI concentration check
    wwi_years = (years >= 1906) & (years <= 1918)
    wwi_fraction = float(np.sum(wwi_years) / len(years)) if len(years) > 0 else np.nan

    # Spatial
    pairwise = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            r1 = metadata.iloc[selected[i]]
            r2 = metadata.iloc[selected[j]]
            pairwise.append(
                haversine_distance(r1["N"], r1["left"], r2["N"], r2["left"])
            )

    mean_pairwise = float(np.mean(pairwise)) if pairwise else np.nan
    min_pairwise = float(np.min(pairwise)) if pairwise else np.nan
    std_pairwise = float(np.std(pairwise)) if pairwise else np.nan

    print(
        f"→ n={n_sel:2d}, clusters={clusters_covered}/8, temp_std={temporal_std:5.2f}, "
        f"WWI%={100*wwi_fraction:4.1f}%, mean_dist={mean_pairwise:5.1f}km, t={runtime:.2f}s"
    )

    results.append(
        {
            "n_samples_target": n_samp,
            "gamma_temporal": gamma,
            "alpha_visual": alpha,
            "beta_spatial": beta_spatial,
            "min_distance_km": min_d,
            "n_selected": n_sel,
            "clusters_covered": clusters_covered,
            "temporal_std": temporal_std,
            "temporal_range": temporal_range,
            "temporal_mean": temporal_mean,
            "wwi_fraction": wwi_fraction,
            "mean_pairwise_km": mean_pairwise,
            "min_pairwise_km": min_pairwise,
            "std_pairwise_km": std_pairwise,
            "runtime_s": runtime,
        }
    )

print("-" * 80)

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT / "optimized_parameters.csv", index=False)

print(f"\n{'=' * 80}")
print("ANALYSE DER ERGEBNISSE")
print("=" * 80)

# Best by different criteria
print("\n1. BESTE KONFIGURATION NACH KRITERIEN:\n")

# Lowest WWI concentration (best temporal diversity)
best_temporal = df.loc[df["wwi_fraction"].idxmin()]
print(f"Niedrigste WWI-Konzentration ({best_temporal['wwi_fraction']*100:.1f}%):")
print(
    f"  → n={int(best_temporal['n_samples_target'])}, γ={best_temporal['gamma_temporal']:.2f}, "
    f"min_d={best_temporal['min_distance_km']:.0f}km"
)
print(
    f"  → Temporal: STD={best_temporal['temporal_std']:.2f}, Range={best_temporal['temporal_range']:.0f} Jahre"
)
print(
    f"  → Spatial: Mean={best_temporal['mean_pairwise_km']:.1f}km, Min={best_temporal['min_pairwise_km']:.1f}km"
)
print()

# Highest temporal diversity (std)
best_temporal_std = df.loc[df["temporal_std"].idxmax()]
print(f"Höchste temporale Streuung (STD={best_temporal_std['temporal_std']:.2f}):")
print(
    f"  → n={int(best_temporal_std['n_samples_target'])}, γ={best_temporal_std['gamma_temporal']:.2f}, "
    f"min_d={best_temporal_std['min_distance_km']:.0f}km"
)
print(f"  → WWI-Konzentration: {best_temporal_std['wwi_fraction']*100:.1f}%")
print()

# Maximum samples (spatial coverage)
best_coverage = df.loc[df["n_selected"].idxmax()]
print(f"Maximale Sample-Anzahl (n={int(best_coverage['n_selected'])}):")
print(
    f"  → γ={best_coverage['gamma_temporal']:.2f}, min_d={best_coverage['min_distance_km']:.0f}km"
)
print(
    f"  → Temporal: STD={best_coverage['temporal_std']:.2f}, WWI={best_coverage['wwi_fraction']*100:.1f}%"
)
print(f"  → Spatial: Mean={best_coverage['mean_pairwise_km']:.1f}km")
print()

# Balanced (composite score)
df["temp_diversity_score"] = (df["temporal_std"] - df["temporal_std"].min()) / (
    df["temporal_std"].max() - df["temporal_std"].min()
)
df["wwi_score"] = 1 - df["wwi_fraction"]  # Lower WWI = better
df["coverage_score"] = (df["n_selected"] - df["n_selected"].min()) / (
    df["n_selected"].max() - df["n_selected"].min()
)
df["composite_score"] = (
    0.4 * df["temp_diversity_score"]
    + 0.3 * df["wwi_score"]
    + 0.3 * df["coverage_score"]
)

best_balanced = df.loc[df["composite_score"].idxmax()]
print(f"Beste Balance (Composite Score={best_balanced['composite_score']:.3f}):")
print(
    f"  → n={int(best_balanced['n_samples_target'])}, γ={best_balanced['gamma_temporal']:.2f}, "
    f"min_d={best_balanced['min_distance_km']:.0f}km"
)
print(
    f"  → Selected: {int(best_balanced['n_selected'])}, WWI={best_balanced['wwi_fraction']*100:.1f}%"
)
print(
    f"  → Temporal: STD={best_balanced['temporal_std']:.2f}, Range={best_balanced['temporal_range']:.0f} Jahre"
)
print(f"  → Spatial: Mean={best_balanced['mean_pairwise_km']:.1f}km")

print(f"\n{'=' * 80}")
print("2. PARAMETER-EFFEKTE:\n")

print("Effekt von gamma_temporal auf WWI-Konzentration:")
for gamma in sorted(df["gamma_temporal"].unique()):
    subset = df[df["gamma_temporal"] == gamma]
    mean_wwi = subset["wwi_fraction"].mean()
    mean_std = subset["temporal_std"].mean()
    print(
        f"  γ={gamma:.2f}: WWI={mean_wwi*100:4.1f}% (avg), Temp_STD={mean_std:5.2f} (avg)"
    )

print("\nEffekt von n_samples auf Coverage:")
for n in sorted(df["n_samples_target"].unique()):
    subset = df[df["n_samples_target"] == n]
    success_rate = subset["n_selected"].mean() / n
    mean_dist = subset["mean_pairwise_km"].mean()
    print(
        f"  n={n:2d}: Success={success_rate*100:5.1f}% (avg {subset['n_selected'].mean():.1f} selected), "
        f"Mean_Dist={mean_dist:.1f}km"
    )

print("\nEffekt von min_distance:")
for min_d in sorted(df["min_distance_km"].unique()):
    subset = df[df["min_distance_km"] == min_d]
    mean_n = subset["n_selected"].mean()
    mean_min_dist = subset["min_pairwise_km"].mean()
    print(
        f"  {min_d:.0f}km: Avg_Selected={mean_n:.1f}, Avg_Min_Dist={mean_min_dist:.1f}km"
    )

print(f"\n{'=' * 80}")
print("EMPFEHLUNG")
print("=" * 80)
print(
    "\nBasierend auf Composite Score (40% Temporal Diversity + 30% WWI-Reduktion + 30% Coverage):\n"
)
print(f"n_samples: {int(best_balanced['n_samples_target'])}")
print(f"gamma_temporal: {best_balanced['gamma_temporal']:.2f}")
print(f"alpha_visual: {best_balanced['alpha_visual']:.2f}")
print(f"beta_spatial: {best_balanced['beta_spatial']:.2f}")
print(f"min_distance_km: {best_balanced['min_distance_km']:.0f}")
print(
    f"\nErwartet: {int(best_balanced['n_selected'])} Samples, WWI-Anteil {best_balanced['wwi_fraction']*100:.1f}%, "
    f"Temporal STD {best_balanced['temporal_std']:.2f}"
)

print(f"\nResults saved to: {OUT / 'optimized_parameters.csv'}")
