"""Compare selection with and without pre-selected seeds (e.g. 'Hamburg').
Produces a small CSV and Markdown report in outputs/seed_benchmark.
"""
from pathlib import Path
from scripts.common import data_path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.io import load_or_extract_features, load_metadata
from src.diversity_selector import DiversitySelector
from src.metrics import compute_metrics

OUT = ROOT / "outputs" / "seed_benchmark"
OUT.mkdir(parents=True, exist_ok=True)

# Config (consistent with final_selection defaults)
n_samples = 34
alpha = 0.7
beta = 0.05
gamma = 0.25
min_distance_km = 50.0

# Load cached features & metadata
features = load_or_extract_features(OUT, csv_meta=str(data_path("new_all_tiles.csv")), batch_size=16, cache=True)
metadata = load_metadata(str(data_path("new_all_tiles.csv")))

cluster_labels = None
# compute cluster labels using existing pipeline to be consistent
from src.clustering import ClusteringPipeline
clustering = ClusteringPipeline(n_clusters=8)
_, cluster_labels = clustering.fit_transform(features)

results = []

# Two scenarios: baseline (no seed) and seeded (Hamburg)
scenarios = [
    ("no_seed", None, None),
    ("seed_Hamburg_name", ["Hamburg"], None),
]

for tag, pre_names, pre_idxs in scenarios:
    ds = DiversitySelector(n_samples=n_samples, use_multi_criteria=True, random_state=42)
    selected = ds.select(
        features=features,
        metadata=metadata,
        alpha_visual=alpha,
        beta_spatial=beta,
        gamma_temporal=gamma,
        spatial_constraint=True,
        min_distance_km=min_distance_km,
        pre_selected=pre_idxs,
        pre_selected_names=pre_names,
    )

    metrics = compute_metrics(selected, metadata, cluster_labels, features)
    metrics.update({
        "scenario": tag,
        "pre_selected_names": pre_names,
        "pre_selected_indices": pre_idxs,
        "n_selected": len(selected),
    })

    # Also save the selection CSV snapshot
    sel_df = metadata.iloc[selected].copy()
    sel_df["selection_rank"] = range(len(sel_df))
    sel_out = OUT / f"selection_{tag}.csv"
    sel_df.to_csv(sel_out, index=False)

    results.append(metrics)

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT / "seed_vs_unseed_metrics.csv", index=False)

# Write small Markdown summary
md = OUT / "seed_vs_unseed_report.md"
with open(md, "w") as f:
    f.write("# Seed vs No-Seed Selection Benchmark\n\n")
    f.write("This short report compares baseline selection and selection seeded with 'Hamburg'.\n\n")
    try:
        f.write(df.to_markdown(index=False))
    except Exception:
        # tabulate may be missing in some envs; fall back to CSV-style table
        f.write("\n" + df.to_string(index=False) + "\n")
    f.write("\n\nSelections saved in this folder for inspection.\n")

print("Done. Results:", OUT)
