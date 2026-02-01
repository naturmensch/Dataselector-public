# ruff: noqa: E402
"""Final selection runner: runs selection with given weights and min_distance and produces outputs.

Usage:
    ./scripts/exec_in_env.sh --env dataselector -- PYTHONPATH=. python scripts/final_selection.py
"""

import time
from pathlib import Path
try:
    from scripts.common import data_path
except Exception:
    # Allow invocation as a script (not as a package) by ensuring project root is on sys.path
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.common import data_path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

# STARTUP ENV VALIDATION
try:
    from src.compat import validate_environment_full
    if "--skip-env-check" not in __import__("sys").argv:
        validate_environment_full()
except Exception as e:
    print(f"\n❌ STARTUP VALIDATION FAILED:\n{e}\n", file=__import__("sys").stderr)
    print("Fix: ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages 'numpy==1.26.4 numba==0.63.1' --yes -- python scripts/final_selection.py", file=__import__("sys").stderr)
    __import__("sys").exit(1)

from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector
from src.metrics import compute_metrics
from src.visualizer import Visualizer

OUT = ROOT / "outputs" / "final_selection"
OUT.mkdir(parents=True, exist_ok=True)

# Parameters (default read from config)
import yaml
import argparse

# Parse CLI args first
parser = argparse.ArgumentParser()
parser.add_argument('--use-bootstrap-best', action='store_true', help='Use bootstrap-best config from outputs/pipeline_config.bootstrap.yaml')
parser.add_argument('--n-samples', type=int, default=None, help='Override n_samples')
parser.add_argument('--min-distance-km', type=float, default=None, help='Override min_distance_km')
args = parser.parse_args()

# Select config source
if args.use_bootstrap_best:
    bootstrap_cfg = ROOT / "outputs" / "pipeline_config.bootstrap.yaml"
    if bootstrap_cfg.exists():
        config_path = bootstrap_cfg
        print(f"Using Bootstrap-best config: {config_path}")
    else:
        print(f"Warning: Bootstrap config not found at {bootstrap_cfg}, falling back to default")
        config_path = ROOT / "config" / "pipeline_config.yaml"
else:
    config_path = ROOT / "config" / "pipeline_config.yaml"

if config_path.exists():
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

n_samples = args.n_samples if args.n_samples else cfg.get("selection", {}).get("n_samples", 34)
alpha = cfg.get("selection", {}).get("alpha_visual", 0.7)
beta = cfg.get("selection", {}).get("beta_spatial", 0.05)
gamma = cfg.get("selection", {}).get("gamma_temporal", 0.25)
min_distance_km = args.min_distance_km if args.min_distance_km else cfg.get("selection", {}).get("min_distance_km", 50.0)
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
    else load_metadata(str(data_path("new_all_tiles.csv")))
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
# Optional: pre-selected names/indices from config (e.g., ['KDR_146'] or ['Hamburg'])
pre_selected_names = cfg.get("selection", {}).get("pre_selected_names", None)
pre_selected_indices = cfg.get("selection", {}).get("pre_selected_indices", None)
if pre_selected_names is not None:
    print(f"Using pre-selected names: {pre_selected_names}")
if pre_selected_indices is not None:
    print(f"Using pre-selected indices: {pre_selected_indices}")

selected_idx = selector.select(
    features=features,
    metadata=metadata,
    alpha_visual=alpha,
    beta_spatial=beta,
    gamma_temporal=gamma,
    spatial_constraint=True,
    min_distance_km=min_distance_km,
    pre_selected=pre_selected_indices,
    pre_selected_names=pre_selected_names,
)

# Add pre-selected info to metrics/report
metrics_extra = {
    "pre_selected_names": pre_selected_names,
    "pre_selected_indices": pre_selected_indices,
}

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
# include extra info (preselection)
metrics.update(metrics_extra)


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
