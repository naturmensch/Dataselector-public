"""
Grid search über alpha_visual, beta_spatial, gamma_temporal für Multi-Criteria Modus.
Speichert für jede Kombination Metriken: Temporal STD, Cluster Coverage, Spatial Mean Dist, WWI-Anteil.

Usage:
    ./scripts/exec_in_env.sh --env dataselector -- PYTHONPATH=. python scripts/tune_weights_and_run.py

"""

from pathlib import Path
from scripts.common import data_path

import numpy as np

from src.metadata_processor import MetadataProcessor

# Config
ROOT = Path(__file__).resolve().parents[1]
DATA_META = data_path("new_all_tiles.csv")
OUTPUT_DIR = ROOT / "outputs" / "tuning_weights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n_samples = 673
# Sweep ranges chosen around recommended adjustments
# Weight combinations must sum to 1.0 (explicit, no implicit normalization)
# Systematischer Grid für wissenschaftliche Pareto-Analyse
weight_combinations = [
    # Visual-dominant (70%)
    (0.7, 0.15, 0.15),
    (0.7, 0.20, 0.10),
    (0.7, 0.10, 0.20),
    (0.7, 0.25, 0.05),
    (0.7, 0.05, 0.25),
    # Balanced (60%)
    (0.6, 0.20, 0.20),
    (0.6, 0.25, 0.15),
    (0.6, 0.15, 0.25),
    (0.6, 0.30, 0.10),
    (0.6, 0.10, 0.30),
    # Visual reduced (50%)
    (0.5, 0.25, 0.25),
    (0.5, 0.30, 0.20),
    (0.5, 0.20, 0.30),
    (0.5, 0.35, 0.15),
    (0.5, 0.15, 0.35),
]

# Fixed clustering configuration (same as used in perf tests)
n_clusters = 8


# Helpers
def compute_metrics(selected_idx, metadata, cluster_labels, features):
    selected = metadata.iloc[selected_idx]
    temporal_std = float(selected["year"].std())
    temporal_range = int(selected["year"].max() - selected["year"].min())
    wwi_frac = float((selected["year"].between(1914, 1918)).mean() * 100)
    # spatial mean distance: pairwise mean distance

    mp = MetadataProcessor("")
    coords = selected[["N", "left"]].values
    if len(coords) > 1:
        dists = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dists.append(
                    mp.calculate_spatial_distance(
                        coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                    )
                )
        spatial_mean = float(np.mean(dists))
        spatial_min = float(np.min(dists))
    else:
        spatial_mean = 0.0
        spatial_min = 0.0

    clusters_covered = int(len(np.unique(cluster_labels[selected_idx])))

    return {
        "n_selected": len(selected_idx),
        "temporal_std": temporal_std,
        "temporal_range": temporal_range,
        "wwi_percent": wwi_frac,
        "spatial_mean_km": spatial_mean,
        "spatial_min_km": spatial_min,
        "clusters_covered": clusters_covered,
    }


def main():
    from src.experiments import ExperimentRunner
    from src.pareto import (
        compute_pareto_front,
        export_pareto_report,
        visualize_pareto_front,
    )

    runner = ExperimentRunner(output_dir=str(OUTPUT_DIR))

    # Run full sweep (no early stopping for Pareto analysis)
    results = runner.run_weight_sweep(
        csv_meta=str(DATA_META),
        n_samples=n_samples,
        weight_combinations=weight_combinations,
        n_clusters=n_clusters,
        batch_size=16,
        min_distance_km=30.0,  # Kompromiss: realistische constraint, aber nicht zu restriktiv
        patience=None,  # Kein Early-Stopping für Pareto
    )

    # Compute Pareto front
    print("\n" + "=" * 70)
    print("Computing Pareto-Front...")
    print("=" * 70)

    pareto_front = compute_pareto_front(results)

    print(
        f"\nPareto-Front: {len(pareto_front)} von {len(results)} Lösungen sind Pareto-optimal"
    )

    # Visualize
    visualize_pareto_front(results, pareto_front, output_dir=str(OUTPUT_DIR / "pareto"))

    # Export report
    export_pareto_report(
        pareto_front, output_path=str(OUTPUT_DIR / "pareto" / "pareto_solutions.csv")
    )


if __name__ == "__main__":
    main()
