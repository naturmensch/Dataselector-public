"""
Coarse Grid Sweep: Weights x MinDistance
Führt systematische Suche durch und erstellt Pareto-Plots.
"""

import itertools
from pathlib import Path

import pandas as pd

from src.experiments import ExperimentRunner
from src.pareto import (
    compute_pareto_front,
    export_pareto_report,
    visualize_pareto_front,
)

# Setup
ROOT = Path(__file__).resolve().parents[1]
DATA_META = ROOT / "data" / "new_all_tiles.csv"
OUTPUT_DIR = ROOT / "outputs" / "coarse_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Grid definieren (konfigurierbar)
import yaml  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "config" / "pipeline_config.yaml"))

betas = [0.05, 0.10, 0.15]
gammas = [0.20, 0.25, 0.30]
# Empfohlene Coarse-Grid: unter/um/über Optimum
min_distances = [20.0, 35.0, 50.0]

# Gewichte generieren (alpha = 1 - beta - gamma)
weight_combos = []
for b, g in itertools.product(betas, gammas):
    a = round(1.0 - b - g, 2)
    if a > 0.01:  # Nur valide Kombinationen
        weight_combos.append((a, b, g))

print(
    f"Grid-Größe: {len(weight_combos)} Gewichtungen x {len(min_distances)} Distanzen = {len(weight_combos)*len(min_distances)} Runs"
)

# Lese defaults aus config
n_clusters_cfg = cfg.get("clustering", {}).get("n_clusters", 8)
n_samples_cfg = cfg.get("selection", {}).get("n_samples", 34)
batch_size_cfg = cfg.get("feature_extraction", {}).get("batch_size", 8)

# CLI args
import argparse  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--min-distances",
    type=str,
    default=None,
    help="Comma-separated min distances to test",
)
parser.add_argument(
    "--max-runs",
    type=int,
    default=None,
    help="Limit number of runs per distance for smoke testing",
)
args = parser.parse_args()
if args.min_distances:
    min_distances = [float(x) for x in args.min_distances.split(",")]

# 2. Sweep ausführen
all_results = []
feasibility_summary = []
runner = ExperimentRunner(output_dir=str(OUTPUT_DIR / "runs"))

for min_dist in min_distances:
    print(f"\n--- Starte Sweep für min_distance = {min_dist} km ---")

    # Nutze ExperimentRunner für Batch-Verarbeitung der Gewichte
    df = runner.run_weight_sweep(
        csv_meta=str(DATA_META),
        n_samples=34,  # Zielgröße für Masterarbeit
        weight_combinations=weight_combos,
        n_clusters=8,
        min_distance_km=min_dist,
        patience=None,  # Kein Early-Stopping, wir wollen alle Daten für Pareto
    )

    # Füge Distanz-Info hinzu (falls nicht im df) und sammle Ergebnisse
    df["min_distance_km"] = min_dist
    all_results.append(df)

    # Feasibility-Check: markiere Runs mit zu wenigen Auswahlen
    infeasible_mask = df["n_selected"] < (0.9 * n_samples_cfg)
    infeasible_count = int(infeasible_mask.sum())
    total_runs = len(df)
    median_selected = int(df["n_selected"].median())
    feasibility_row = {
        "min_distance_km": min_dist,
        "total_runs": total_runs,
        "infeasible_count": infeasible_count,
        "infeasible_pct": (
            infeasible_count / total_runs * 100.0 if total_runs > 0 else 0.0
        ),
        "median_n_selected": median_selected,
    }
    feasibility_summary.append(feasibility_row)

# 3. Aggregation & Pareto
full_df = pd.concat(all_results, ignore_index=True)
full_csv = OUTPUT_DIR / "coarse_sweep_results.csv"
full_df.to_csv(full_csv, index=False)
print(f"\nGesamtergebnisse gespeichert: {full_csv}")

# Feasibility summary
if feasibility_summary:
    fs_df = pd.DataFrame(feasibility_summary)
    fs_path = OUTPUT_DIR / "feasibility_summary.csv"
    fs_df.to_csv(fs_path, index=False)
    print(f"Feasibility summary geschrieben: {fs_path}")

# Pareto-Front berechnen (feasible-only)
print("Berechne globale Pareto-Front (feasible runs only)...")
feasible_mask = full_df["n_selected"] >= (0.9 * n_samples_cfg)
n_infeasible = (~feasible_mask).sum()
if n_infeasible > 0:
    print(f"Info: {n_infeasible} infeasible runs removed from Pareto computation.")
feasible_df = full_df[feasible_mask].reset_index(drop=True)
pareto_front = compute_pareto_front(feasible_df)
export_pareto_report(pareto_front, output_path=str(OUTPUT_DIR / "pareto_solutions.csv"))
# also save filtered results for reproducibility
feasible_df.to_csv(OUTPUT_DIR / "coarse_sweep_results_feasible.csv", index=False)
print(
    f"Feasible-only results saved: {OUTPUT_DIR / 'coarse_sweep_results_feasible.csv'}"
)

# Visualisierung
print("Erstelle Plots...")
visualize_pareto_front(full_df, pareto_front, output_dir=str(OUTPUT_DIR / "plots"))

print("\n✅ Coarse Sweep abgeschlossen!")
