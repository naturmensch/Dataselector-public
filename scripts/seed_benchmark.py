# ruff: noqa: E402
"""
Seed and parallelism benchmark for UMAP on cached features.
Strategy:
 - Try utopian (non-deterministic) setting first: umap_random_state=None, umap_n_jobs=-1
 - If successful, record time and recommend this (fastest)
 - Otherwise, test a set of seeds (42, 0, 1, 123, 999, 2026) and pick fastest seed
 - Save results to outputs/seed_benchmark_results.csv and print recommendation

This script intentionally uses only a subset (first N) for speed.
"""

import csv
import time
from pathlib import Path

from src.clustering import ClusteringPipeline
from src.io import load_metadata, load_or_extract_features

OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

# Ensure features/metadata exists or extract on-the-fly
csv_meta = OUT / "metadata.csv"
csv_meta = str(csv_meta) if csv_meta.exists() else None
features = load_or_extract_features(
    out_dir=OUT, csv_meta=csv_meta, batch_size=16, cache=True
)
from scripts.common import data_path

metadata = load_metadata(csv_meta if csv_meta is not None else str(data_path("new_all_tiles.csv")))

# subset size for quick timing
SUBSET_N = min(200, len(features))
feat_sub = features[:SUBSET_N]

results = []

# test utopian (fast) setting
print("Testing utopian (non-deterministic, n_jobs=-1) setting...")
try:
    t0 = time.perf_counter()
    cl = ClusteringPipeline(n_clusters=8, umap_random_state=None, umap_n_jobs=-1)
    emb = cl.fit_transform(feat_sub)[0]
    t = time.perf_counter() - t0
    print(f"UTOPIAN success: {t:.3f}s")
    results.append({"mode": "utopian", "seed": None, "n_jobs": -1, "time_s": t})
except Exception as e:
    print("UTOPIAN failed:", e)

# if utopian succeeded and is fast enough we may stop; but we still measure seeds to compare
seed_list = [42, 0, 1, 123, 999, 2026]
for s in seed_list:
    print(f"Testing seed={s} (deterministic, single-thread) ...")
    try:
        t0 = time.perf_counter()
        cl = ClusteringPipeline(n_clusters=8, umap_random_state=int(s), umap_n_jobs=1)
        emb = cl.fit_transform(feat_sub)[0]
        t = time.perf_counter() - t0
        print(f"  seed {s} success: {t:.3f}s")
        results.append({"mode": "seeded", "seed": int(s), "n_jobs": 1, "time_s": t})
    except Exception as e:
        print(f"  seed {s} failed:", e)

# Save results
out_csv = OUT / "seed_benchmark_results.csv"
keys = ["mode", "seed", "n_jobs", "time_s"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print("\nBenchmark results saved to", out_csv)

# Decide recommended configuration: fastest overall
if results:
    best = min(results, key=lambda x: x["time_s"])
    print("\nBest config:")
    print(best)
    # Recommend but do not auto-change config file; instead persist recommendation
    rec_file = OUT / "seed_benchmark_recommendation.txt"
    with open(rec_file, "w") as f:
        f.write("Best config (fastest):\n")
        f.write(str(best) + "\n")
    print("Recommendation written to", rec_file)
else:
    print("No successful runs recorded.")
