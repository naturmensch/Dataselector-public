"""
Fine Grid Sweep: denser search around Pareto region.
Saves results to outputs/fine_sweep
"""
import itertools
from pathlib import Path
import pandas as pd

from src.experiments import ExperimentRunner
from src.pareto import compute_pareto_front, visualize_pareto_front, export_pareto_report

ROOT = Path(__file__).resolve().parents[1]
DATA_META = ROOT / "data" / "new_all_tiles.csv"
OUTPUT_DIR = ROOT / "outputs" / "fine_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
cfg = yaml.safe_load(open(ROOT / "config" / "pipeline_config.yaml"))

alphas = [0.55, 0.60, 0.65, 0.70, 0.75]
betas = [0.05, 0.10, 0.15, 0.20]
# Focused fine grid around empirical optimum
min_distances = [30.0, 35.0, 40.0, 45.0, 50.0]

# Generate valid weight combinations where gamma = 1 - alpha - beta > 0.01
weight_combos = []
for a, b in itertools.product(alphas, betas):
    g = round(1.0 - a - b, 3)
    if g > 0.01:
        weight_combos.append((a, b, g))

print(f"Fine grid size: {len(weight_combos)} weight combos x {len(min_distances)} distances = {len(weight_combos)*len(min_distances)} runs")

n_clusters_cfg = cfg.get('clustering', {}).get('n_clusters', 8)
n_samples_cfg = cfg.get('selection', {}).get('n_samples', 34)
batch_size_cfg = cfg.get('feature_extraction', {}).get('batch_size', 8)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--min-distances', type=str, default=None,
                    help='Comma-separated list of min distances to test (overrides defaults)')
parser.add_argument('--max-runs', type=int, default=None, help='Limit number of runs per distance for smoke testing')
args = parser.parse_args()

if args.min_distances:
    min_distances = [float(x) for x in args.min_distances.split(',')]

max_runs = args.max_runs
runner = ExperimentRunner(output_dir=str(OUTPUT_DIR / "runs"))
all_results = []

for min_dist in min_distances:
    print(f"\n--- Fine sweep for min_distance = {min_dist} km ---")
    df = runner.run_weight_sweep(
        csv_meta=str(DATA_META),
        n_samples=n_samples_cfg,
        weight_combinations=weight_combos,
        n_clusters=n_clusters_cfg,
        batch_size=batch_size_cfg,
        min_distance_km=min_dist,
        patience=None,
        max_runs=max_runs,
    )
    df["min_distance_km"] = min_dist
    all_results.append(df)

    # Feasibility-check per distance
    infeasible_mask = df["n_selected"] < (0.9 * n_samples_cfg)
    infeasible_count = int(infeasible_mask.sum())
    total_runs = len(df)
    median_selected = int(df["n_selected"].median())
    # append to summary list (create if not present)
    try:
        feasibility_summary.append({
            "min_distance_km": min_dist,
            "total_runs": total_runs,
            "infeasible_count": infeasible_count,
            "infeasible_pct": infeasible_count / total_runs * 100.0 if total_runs > 0 else 0.0,
            "median_n_selected": median_selected,
        })
    except NameError:
        feasibility_summary = []
        feasibility_summary.append({
            "min_distance_km": min_dist,
            "total_runs": total_runs,
            "infeasible_count": infeasible_count,
            "infeasible_pct": infeasible_count / total_runs * 100.0 if total_runs > 0 else 0.0,
            "median_n_selected": median_selected,
        })

full_df = pd.concat(all_results, ignore_index=True)
full_csv = OUTPUT_DIR / "fine_sweep_results.csv"
full_df.to_csv(full_csv, index=False)
print(f"Fine sweep results saved: {full_csv}")

# write feasibility summary
if 'feasibility_summary' in locals():
    import pandas as _pd
    fs_df = _pd.DataFrame(feasibility_summary)
    fs_path = OUTPUT_DIR / "feasibility_summary.csv"
    fs_df.to_csv(fs_path, index=False)
    print(f"Feasibility summary written: {fs_path}")

# Pareto-Front berechnen (feasible-only)
feasible_mask = full_df["n_selected"] >= (0.9 * n_samples_cfg)
n_infeasible = (~feasible_mask).sum()
if n_infeasible > 0:
    print(f"Info: {n_infeasible} infeasible runs removed from Pareto computation.")
feasible_df = full_df[feasible_mask].reset_index(drop=True)
pareto_front = compute_pareto_front(feasible_df)
export_pareto_report(pareto_front, output_path=str(OUTPUT_DIR / "pareto_solutions.csv"))
# also save filtered results for reproducibility
feasible_df.to_csv(OUTPUT_DIR / "fine_sweep_results_feasible.csv", index=False)
visualize_pareto_front(feasible_df, pareto_front, output_dir=str(OUTPUT_DIR / "plots"))
print("Fine sweep + Pareto finished")
