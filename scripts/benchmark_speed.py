# ruff: noqa: E402
"""
Benchmark script to measure runtime for different UMAP and feature extraction settings.
Saves results to outputs/benchmark_results.csv
"""

import itertools
import time
from pathlib import Path

import pandas as pd
import torch
import umap

from src.feature_extractor import FeatureExtractor

OUT = Path("outputs")
OUT.mkdir(exist_ok=True, parents=True)

# Load features and metadata (cached or on-demand)
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

results = []

# UMAP settings to test
random_states = [42, None]
n_jobs_options = [1, -1]

print("Benchmarking UMAP configurations...")
for rs, nj in itertools.product(random_states, n_jobs_options):
    cfg = {"umap_random_state": rs, "umap_n_jobs": nj}
    desc = f"random_state={rs}, n_jobs={nj}"
    print("Testing", desc)

    t0 = time.time()
    reducer = umap.UMAP(n_components=2, random_state=rs, n_jobs=nj, metric="cosine")
    emb = reducer.fit_transform(features)
    t = time.time() - t0

    results.append(
        {
            "test_type": "umap",
            "description": desc,
            "time_s": t,
            "n_samples": features.shape[0],
        }
    )

# Feature extraction settings — use a small subset to estimate
print("\nBenchmarking Feature Extraction (subset)...")
image_dir = Path("data/images")
# pick up to 50 image filenames from metadata
image_filenames = metadata["image_filename"].dropna().tolist()
subset = image_filenames[:50]

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
batch_sizes = [4, 8, 16]

for device, bs in itertools.product(devices, batch_sizes):
    desc = f"device={device}, batch_size={bs}"
    print("Testing", desc)
    extractor = FeatureExtractor(device=device)

    t0 = time.time()
    feats = extractor.extract_features_batch(subset, image_dir, batch_size=bs)
    t = time.time() - t0

    results.append(
        {
            "test_type": "feature_extraction",
            "description": desc,
            "time_s": t,
            "n_samples": len(subset),
        }
    )

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "benchmark_results.csv", index=False)
print("\nBenchmark complete. Results saved to outputs/benchmark_results.csv")
print(res_df)
