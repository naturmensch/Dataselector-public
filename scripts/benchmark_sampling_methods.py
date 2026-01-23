"""Benchmark sampling methods: LHS vs Sobol (space-filling quality).

Produces CSV summary and a simple boxplot comparing min pairwise distances as a
proxy for space-filling quality across multiple trials and sample sizes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib and sampling strategy placeholders set at runtime
plt = None
ss = None

# Defer matplotlib backend and project imports to runtime to avoid import-time side-effects

OUT = Path("outputs") / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


def _ensure_runtime_deps():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Import project sampling strategies at runtime and expose as module global
    from src import sampling_strategies as ss

    globals()["plt"] = plt
    globals()["ss"] = ss


# Ensure the plotting and project imports are available when running as script


# Ensure the plotting and project imports are available when running as script


def min_pairwise_distance(samples: np.ndarray) -> float:
    # compute pairwise distances (Euclidean) and return minimum non-zero distance
    if samples.shape[0] < 2:
        return 0.0
    dists = np.sqrt(((samples[:, None, :] - samples[None, :, :]) ** 2).sum(axis=-1))
    # mask diagonal
    dists = dists + np.eye(samples.shape[0]) * dists.max() * 10
    return float(dists.min())


def benchmark_space_filling(sample_sizes, n_trials=10, dim=3, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for n in sample_sizes:
        for trial in range(n_trials):
            s = rng.randint(0, 2**30)
            # LHS (projected to simplex to mimic weight sampling)
            try:
                lhs = ss.sample_unit_hypercube_lhs(n, dim, seed=s)
                lhs_norm = lhs / lhs.sum(axis=1)[:, None]
                lhs_min = min_pairwise_distance(lhs_norm)
            except Exception:
                lhs_min = float("nan")

            # Sobol (projected to simplex)
            try:
                sob = ss.sample_unit_hypercube_sobol(n, dim, seed=s)
                sob_norm = sob / sob.sum(axis=1)[:, None]
                sob_min = min_pairwise_distance(sob_norm)
            except Exception:
                sob_min = float("nan")

            rows.append(
                {"method": "lhs", "n": n, "trial": trial, "min_pairwise_dist": lhs_min}
            )
            rows.append(
                {
                    "method": "sobol",
                    "n": n,
                    "trial": trial,
                    "min_pairwise_dist": sob_min,
                }
            )

    df = pd.DataFrame(rows)
    return df


def main():
    _ensure_runtime_deps()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="List of sample sizes to benchmark",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Trials per sample size"
    )
    parser.add_argument(
        "--dim", type=int, default=3, help="Dimensionality (weights simplex -> dim=3)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-prefix", type=str, default=str(OUT / "sampling_benchmark")
    )
    args = parser.parse_args()

    df = benchmark_space_filling(
        args.sample_sizes, n_trials=args.n_trials, dim=args.dim, seed=args.seed
    )

    csv_path = Path(f"{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote benchmark CSV: {csv_path}")

    # Plot: boxplot of min_pairwise_dist per method and sample size
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = []
    data = []
    for n in sorted(df["n"].unique()):
        for method in ["lhs", "sobol"]:
            vals = (
                df[(df["n"] == n) & (df["method"] == method)]["min_pairwise_dist"]
                .dropna()
                .values
            )
            data.append(vals)
            labels.append(f"{method}\n n={n}")

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Min. pairwise distance (unit simplex)")
    ax.set_title("Sampling space-filling comparison (LHS vs Sobol)")
    plt.tight_layout()
    fig_path = Path(f"{args.out_prefix}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote benchmark plot: {fig_path}")


if __name__ == "__main__":
    main()
