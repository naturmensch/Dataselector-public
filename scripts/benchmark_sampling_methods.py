<<<<<<< HEAD
#!/usr/bin/env python3
"""Benchmark Sampling Methods.

This script benchmarks different sampling methods (LHS, Sobol, Random)
for hyperparameter optimization in the context of dataset selection.
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import stats

# Import sampling libraries
try:
    from pyDOE import lhs
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False

try:
    import sobol_seq
    HAS_SOBOL = True
except ImportError:
    HAS_SOBOL = False

ROOT = Path(__file__).resolve().parents[1]


def random_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate random samples."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n_samples, n_dims))


def lhs_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube samples."""
    if not HAS_PYDOE:
        raise ImportError("pyDOE required for LHS sampling")
    return lhs(n_dims, samples=n_samples, criterion='maximin', random_state=seed)


def sobol_sampling(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """Generate Sobol sequence samples."""
    if not HAS_SOBOL:
        raise ImportError("sobol-seq required for Sobol sampling")
    # Sobol requires power of 2
    n_power2 = 2 ** int(np.ceil(np.log2(n_samples)))
    seq = sobol_seq.i4_sobol_generate(n_dims, n_power2, seed)
    return seq[:n_samples]


def benchmark_sampling_method(
    method_name: str,
    sampling_func,
    n_samples: int,
    n_dims: int,
    n_repeats: int = 10
) -> Dict:
    """Benchmark a sampling method."""
    times = []
    samples_list = []

    for seed in range(n_repeats):
        start_time = time.time()
        try:
            samples = sampling_func(n_samples, n_dims, seed=seed)
            end_time = time.time()
            times.append(end_time - start_time)
            samples_list.append(samples)
        except Exception as e:
            print(f"Error with {method_name}, seed {seed}: {e}")
            times.append(float('nan'))
            samples_list.append(None)

    # Compute quality metrics
    valid_samples = [s for s in samples_list if s is not None]
    if not valid_samples:
        return {
            'method': method_name,
            'mean_time': float('nan'),
            'std_time': float('nan'),
            'discrepancy': float('nan'),
            'min_distance': float('nan')
        }

    # Simple discrepancy measure (L2 between samples and uniform grid)
    avg_samples = np.mean(valid_samples, axis=0)
    uniform_grid = np.linspace(0, 1, n_samples + 1)[1:]  # Remove 0
    uniform_grid = np.tile(uniform_grid.reshape(-1, 1), (1, n_dims))

    discrepancies = []
    for samples in valid_samples:
        # Sort samples by first dimension for comparison
        sorted_samples = samples[np.argsort(samples[:, 0])]
        disc = np.mean(np.abs(sorted_samples - uniform_grid))
        discrepancies.append(disc)

    # Minimum distance between samples
    min_distances = []
    for samples in valid_samples:
        distances = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = np.linalg.norm(samples[i] - samples[j])
                distances.append(dist)
        if distances:
            min_distances.append(np.min(distances))

    return {
        'method': method_name,
        'mean_time': np.nanmean(times),
        'std_time': np.nanstd(times),
        'discrepancy': np.mean(discrepancies),
        'min_distance': np.mean(min_distances) if min_distances else float('nan')
    }


def run_benchmark(
    n_samples_list: List[int],
    n_dims: int = 4,
    n_repeats: int = 5,
    output_dir: str = 'outputs'
):
    """Run sampling method benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    methods = [
        ('Random', random_sampling),
    ]

    if HAS_PYDOE:
        methods.append(('LHS', lhs_sampling))
    else:
        print("Warning: pyDOE not available, skipping LHS")

    if HAS_SOBOL:
        methods.append(('Sobol', sobol_sampling))
    else:
        print("Warning: sobol-seq not available, skipping Sobol")

    results = []

    for n_samples in n_samples_list:
        print(f"Benchmarking with {n_samples} samples, {n_dims} dimensions...")
        for method_name, sampling_func in methods:
            result = benchmark_sampling_method(
                method_name, sampling_func, n_samples, n_dims, n_repeats
            )
            result['n_samples'] = n_samples
            result['n_dims'] = n_dims
            results.append(result)

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / 'sampling_benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved benchmark results to {csv_path}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Time vs sample size
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axes[0, 0].errorbar(subset['n_samples'], subset['mean_time'],
                           yerr=subset['std_time'], label=method, capsize=3)

    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Sampling Time')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Discrepancy vs sample size
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axes[0, 1].plot(subset['n_samples'], subset['discrepancy'],
                       'o-', label=method)

    axes[0, 1].set_xlabel('Number of Samples')
    axes[0, 1].set_ylabel('Discrepancy')
    axes[0, 1].set_title('Sampling Quality (Discrepancy)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Min distance vs sample size
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axes[1, 0].plot(subset['n_samples'], subset['min_distance'],
                       'o-', label=method)

    axes[1, 0].set_xlabel('Number of Samples')
    axes[1, 0].set_ylabel('Min Distance')
    axes[1, 0].set_title('Minimum Distance Between Samples')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Quality vs time
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axes[1, 1].scatter(subset['mean_time'], subset['discrepancy'],
                          label=method, s=50)

    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Discrepancy')
    axes[1, 1].set_title('Quality vs Speed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'sampling_benchmark_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved benchmark plots to {plot_path}")

    # Print summary
    print("\nBenchmark Summary:")
    print(df.groupby('method').agg({
        'mean_time': ['mean', 'std'],
        'discrepancy': ['mean', 'std'],
        'min_distance': ['mean', 'std']
    }).round(4))


def main():
    parser = argparse.ArgumentParser(description='Benchmark Sampling Methods')
    parser.add_argument('--n-samples', nargs='+', type=int, default=[50, 100, 200, 500],
                       help='Number of samples to test')
    parser.add_argument('--n-dims', type=int, default=4,
                       help='Number of dimensions')
    parser.add_argument('--n-repeats', type=int, default=5,
                       help='Number of repeats per method')
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory')

    args = parser.parse_args()

    run_benchmark(
        args.n_samples,
        args.n_dims,
        args.n_repeats,
        args.output_dir
    )


if __name__ == '__main__':
    main()
=======
"""Benchmark sampling methods: LHS vs Sobol (space-filling quality).

Produces CSV summary and a simple boxplot comparing min pairwise distances as a
proxy for space-filling quality across multiple trials and sample sizes.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import sampling_strategies as ss

OUT = Path("outputs") / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


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
            s = rng.randint(0, 2 ** 30)
            # LHS (projected to simplex to mimic weight sampling)
            try:
                lhs = ss.sample_unit_hypercube_lhs(n, dim, seed=s)
                lhs_norm = lhs / lhs.sum(axis=1)[:, None]
                lhs_min = min_pairwise_distance(lhs_norm)
            except Exception:
                lhs_min = float('nan')

            # Sobol (projected to simplex)
            try:
                sob = ss.sample_unit_hypercube_sobol(n, dim, seed=s)
                sob_norm = sob / sob.sum(axis=1)[:, None]
                sob_min = min_pairwise_distance(sob_norm)
            except Exception:
                sob_min = float('nan')

            rows.append({'method': 'lhs', 'n': n, 'trial': trial, 'min_pairwise_dist': lhs_min})
            rows.append({'method': 'sobol', 'n': n, 'trial': trial, 'min_pairwise_dist': sob_min})

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[20, 50, 100], help='List of sample sizes to benchmark')
    parser.add_argument('--n-trials', type=int, default=20, help='Trials per sample size')
    parser.add_argument('--dim', type=int, default=3, help='Dimensionality (weights simplex -> dim=3)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-prefix', type=str, default=str(OUT / 'sampling_benchmark'))
    args = parser.parse_args()

    df = benchmark_space_filling(args.sample_sizes, n_trials=args.n_trials, dim=args.dim, seed=args.seed)

    csv_path = Path(f"{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote benchmark CSV: {csv_path}")

    # Plot: boxplot of min_pairwise_dist per method and sample size
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = []
    data = []
    for n in sorted(df['n'].unique()):
        for method in ['lhs', 'sobol']:
            vals = df[(df['n'] == n) & (df['method'] == method)]['min_pairwise_dist'].dropna().values
            data.append(vals)
            labels.append(f"{method}\n n={n}")

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel('Min. pairwise distance (unit simplex)')
    ax.set_title('Sampling space-filling comparison (LHS vs Sobol)')
    plt.tight_layout()
    fig_path = Path(f"{args.out_prefix}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Wrote benchmark plot: {fig_path}")


if __name__ == '__main__':
    main()
>>>>>>> ci/add-smoke-tests
