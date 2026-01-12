"""Run a batch of diverse selection runs and validate outputs.

Produces:
- outputs/experiments_20_runs.csv
- outputs/selection_run_<i>.csv for each run
- prints summary of validity (violations, incomplete selections)

Usage:
    python scripts/run_diverse_experiments.py --n-runs 20 --n-samples 100
"""

import argparse
import os

# Ensure local src is importable when run from project root
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diversity_selector import DiversitySelector
from src.spatial_facility_location import haversine_distance

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
EXPDIR = OUT / "experiments"
EXPDIR.mkdir(exist_ok=True)


def run_experiments(n_runs: int = 20, n_samples: int = 100, seed: int = 42):
    rng = np.random.RandomState(seed)

    features = np.load(OUT / "features.npy")
    metadata = pd.read_csv(OUT / "metadata.csv")

    n_candidates = features.shape[0]
    print(
        f"Loaded {n_candidates} candidates; running {n_runs} experiments (n_samples={n_samples})"
    )

    rows = []

    for i in range(n_runs):
        # Sample diverse parameters
        a = rng.rand()
        b = rng.rand()
        c = rng.rand()
        total = a + b + c
        alpha = a / total
        beta = b / total
        gamma = c / total

        min_dist = int(rng.uniform(10, 150))  # km
        use_lazy = bool(rng.choice([0, 1]))
        seed_run = int(rng.randint(0, 2**31 - 1))

        selector = DiversitySelector(
            n_samples=n_samples,
            use_multi_criteria=True,
            use_lazy_greedy=use_lazy,
            random_state=seed_run,
        )

        t0 = time.time()
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_dist,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
        )
        duration = time.time() - t0

        n_selected = len(selected)
        diversity = (
            selector._calculate_diversity_score(features[selected])
            if n_selected > 0
            else 0.0
        )
        spatial_spread = (
            metadata.loc[selected, ["N", "left"]].std().mean()
            if n_selected > 0
            else 0.0
        )

        # Check violations
        violations = 0
        for idx_i in range(n_selected):
            for idx_j in range(idx_i + 1, n_selected):
                id1 = int(selected[idx_i])
                id2 = int(selected[idx_j])
                d = haversine_distance(
                    metadata.loc[id1, "N"],
                    metadata.loc[id1, "left"],
                    metadata.loc[id2, "N"],
                    metadata.loc[id2, "left"],
                )
                if d < min_dist - 1e-6:
                    violations += 1

        # Save selection
        sel_df = metadata.iloc[selected].copy()
        sel_df["selection_rank"] = range(len(sel_df))
        sel_file = EXPDIR / f"selection_run_{i:02d}.csv"
        sel_df.to_csv(sel_file, index=False)

        rows.append(
            {
                "run": i,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "min_distance_km": min_dist,
                "use_lazy_greedy": use_lazy,
                "seed": seed_run,
                "n_selected": n_selected,
                "diversity": diversity,
                "spatial_spread_deg": spatial_spread,
                "violations": violations,
                "duration_s": duration,
                "selection_file": str(sel_file),
            }
        )

        print(
            f"Run {i+1}/{n_runs}: selected={n_selected}, violations={violations}, diversity={diversity:.3f}, spread={spatial_spread:.3f}, t={duration:.2f}s"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"experiments_{n_runs}_runs_{n_samples}_samples.csv", index=False)
    print(
        "\nSummary saved to", OUT / f"experiments_{n_runs}_runs_{n_samples}_samples.csv"
    )

    # Basic summary
    print("\n--- Summary statistics ---")
    print("Avg diversity:", df["diversity"].mean())
    print("Avg spatial_spread_deg:", df["spatial_spread_deg"].mean())
    print("Total violations across runs:", df["violations"].sum())
    print(
        "Runs with incomplete selection (<n_samples):",
        (df["n_selected"] < n_samples).sum(),
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiments(n_runs=args.n_runs, n_samples=args.n_samples, seed=args.seed)
