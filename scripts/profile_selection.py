"""Profiling script for the DiversitySelector pipeline.

Usage:
    python scripts/profile_selection.py

Saves profiling results to `outputs/` as both `.prof` (binary) and `.txt` (human readable).
If `outputs/features.npy` and `outputs/metadata.csv` exist they will be used; otherwise a synthetic dataset is generated.
"""

import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so local `src` package can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diversity_selector import DiversitySelector

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_or_create_data(n=2000, dim=512):
    features_path = OUT_DIR / "features.npy"
    metadata_path = OUT_DIR / "metadata.csv"

    if features_path.exists() and metadata_path.exists():
        print("Loading existing features/metadata from outputs/ ...")
        features = np.load(features_path)
        metadata = pd.read_csv(metadata_path)
    else:
        print("No existing data found — generating synthetic dataset for profiling.")
        rng = np.random.RandomState(123)
        features = rng.randn(n, dim).astype("float32")
        metadata = pd.DataFrame(
            {
                "N": np.random.uniform(48, 55, n),
                "left": np.random.uniform(6, 15, n),
                "year": np.random.randint(1880, 1945, n),
            }
        )

    return features, metadata


def profile_mode(
    mode_name: str,
    selector: DiversitySelector,
    features: np.ndarray,
    metadata: pd.DataFrame,
    **kwargs,
):
    prof = cProfile.Profile()
    print(f"Profiling mode: {mode_name}")
    t0 = time.time()
    prof.enable()

    # Run selection once (the thing we care about)
    selector.select(features, metadata, **kwargs)

    prof.disable()
    elapsed = time.time() - t0

    # Save profiler data
    prof_file = OUT_DIR / f"profile_{mode_name}.prof"
    txt_file = OUT_DIR / f"profile_{mode_name}.txt"

    ps = pstats.Stats(prof, stream=open(txt_file, "w"))
    ps.strip_dirs().sort_stats("cumulative").print_stats(50)

    # Also save binary .prof using dump_stats
    prof.dump_stats(str(prof_file))

    print(f"Mode {mode_name} finished in {elapsed:.2f}s — stats written to {txt_file}")
    return elapsed


def main():
    features, metadata = load_or_create_data()

    results = {}

    # LEGACY mode (FacilityLocation + optional spatial filter)
    selector_legacy = DiversitySelector(n_samples=34, use_multi_criteria=False)
    print("Warm-up (legacy) to allow JIT compilation...")
    selector_legacy.select(
        features, metadata, spatial_constraint=True, min_distance_km=50.0
    )
    results["legacy"] = profile_mode(
        "legacy",
        selector_legacy,
        features,
        metadata,
        spatial_constraint=True,
        min_distance_km=50.0,
    )

    # CONSTRAINT-INTEGRATED mode
    selector_constraint = DiversitySelector(
        n_samples=34, use_multi_criteria=False, use_constraint_integration=True
    )
    print("Warm-up (constraint_integrated) to allow JIT compilation...")
    selector_constraint.select(
        features, metadata, spatial_constraint=True, min_distance_km=50.0
    )
    results["constraint_integrated"] = profile_mode(
        "constraint_integrated",
        selector_constraint,
        features,
        metadata,
        spatial_constraint=True,
        min_distance_km=50.0,
    )

    # MULTI-CRITERIA mode
    if "year" in metadata.columns:
        selector_multi = DiversitySelector(n_samples=34, use_multi_criteria=True)
        print("Warm-up (multi_criteria) to allow JIT compilation...")
        selector_multi.select(
            features, metadata, alpha_visual=0.7, beta_spatial=0.15, gamma_temporal=0.15
        )
        results["multi_criteria"] = profile_mode(
            "multi_criteria",
            selector_multi,
            features,
            metadata,
            alpha_visual=0.7,
            beta_spatial=0.15,
            gamma_temporal=0.15,
        )

    # Save timing summary
    summary = OUT_DIR / "profile_summary.csv"
    pd.DataFrame.from_dict(results, orient="index", columns=["time_s"]).to_csv(summary)
    print("Profiling complete. Summary saved to", summary)


if __name__ == "__main__":
    main()
