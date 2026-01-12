# ruff: noqa: E402
"""Final selection runner: runs selection with given weights and min_distance and produces outputs.

Usage:
    PYTHONPATH=. python scripts/final_selection.py
"""

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector
from src.metrics import compute_metrics
from src.visualizer import Visualizer

OUT = ROOT / "outputs" / "final_selection"
OUT.mkdir(parents=True, exist_ok=True)

# Parameters (default read from config)
import yaml

config_path = ROOT / "config" / "pipeline_config.yaml"
if config_path.exists():
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

n_samples = cfg.get("selection", {}).get("n_samples", 34)
alpha = cfg.get("selection", {}).get("alpha_visual", 0.7)
beta = cfg.get("selection", {}).get("beta_spatial", 0.05)
gamma = cfg.get("selection", {}).get("gamma_temporal", 0.25)
min_distance_km = cfg.get("selection", {}).get("min_distance_km", 50.0)
spatial_penalty_weight = cfg.get("selection", {}).get("spatial_penalty_weight", 0.1)
seed = cfg.get("selection", {}).get("random_state", 42)

# Load data (cached or extract on-demand)
from src.io import load_metadata, load_or_extract_features

OUT_ROOT = ROOT / "outputs"
features = load_or_extract_features(
    OUT_ROOT,
    csv_meta=(
        str(OUT_ROOT / "metadata.csv") if (OUT_ROOT / "metadata.csv").exists() else None
    ),
    batch_size=16,
    cache=True,
)
metadata = (
    pd.read_csv(OUT_ROOT / "metadata.csv")
    if (OUT_ROOT / "metadata.csv").exists()
    else load_metadata(str(ROOT / "data" / "new_all_tiles.csv"))
)

# Clustering
clustering = ClusteringPipeline(n_clusters=8)
embeddings_2d, cluster_labels = clustering.fit_transform(features)

# Selection
selector = DiversitySelector(
    n_samples=n_samples, use_multi_criteria=True, random_state=seed
)
print(
    f"Running final selection: n_samples={n_samples}, α={alpha}, β={beta}, γ={gamma}, min_dist={min_distance_km}"
)
start = time.time()
selected_idx = selector.select(
    features=features,
    metadata=metadata,
    alpha_visual=alpha,
    beta_spatial=beta,
    gamma_temporal=gamma,
    spatial_penalty_weight=spatial_penalty_weight,
    spatial_constraint=True,
    min_distance_km=min_distance_km,
)
duration = time.time() - start

# Export selection
sel_df = metadata.iloc[selected_idx].copy()
sel_df["selection_rank"] = range(len(sel_df))
sel_csv = (
    OUT
    / f"final_selection_n{n_samples}_a{alpha}_b{beta}_g{gamma}_d{int(min_distance_km)}.csv"
)
sel_df.to_csv(sel_csv, index=False)
print(f"Selection saved: {sel_csv}")

# Metrics
metrics = compute_metrics(selected_idx, metadata, cluster_labels, features)
metrics.update(
    {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "min_distance_km": min_distance_km,
        "n_requested": n_samples,
        "n_selected": len(selected_idx),
        "duration_s": duration,
    }
)

# Visualizations
viz = Visualizer(output_dir=str(OUT))
viz.create_summary_report(
    embeddings_2d=embeddings_2d,
    cluster_labels=cluster_labels,
    metadata=metadata,
    selected_indices=selected_idx,
    output_prefix=f"final_n{n_samples}",
)

# Report
report = OUT / "final_selection_report.md"
with open(report, "w") as f:
    f.write("# Final Selection Report\n")
    f.write(f"Generated: {pd.Timestamp.now()}\n\n")
    f.write("## Parameters\n")
    f.write(f"- n_requested: {n_samples}\n")
    f.write(f"- α={alpha}, β={beta}, γ={gamma}\n")
    f.write(f"- min_distance_km: {min_distance_km}\n")
    f.write(f"- seed: {seed}\n\n")
    f.write("## Metrics\n")
    for k, v in metrics.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n")
    f.write(f"Selection CSV: {sel_csv}\n")
    f.write(f'Plots: {OUT / ("final_n%g"%n_samples)}\n')

print("Final report written:", report)
print("Done.")
