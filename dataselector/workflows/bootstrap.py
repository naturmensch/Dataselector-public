"""Bootstrap uncertainty quantification workflows.

Provides functions for bootstrap-based robustness analysis:
- bootstrap_selection: Core resampling logic for final selection
- bootstrap_candidate: Core resampling logic for Pareto candidates
- run_bootstrap_final: High-level orchestration for final selection
- run_bootstrap_pareto: High-level orchestration for Pareto candidates
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import trange

from dataselector.cli_decorators import cli_command


def _get_repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).resolve().parents[2]


def jaccard(a, b) -> float:
    """Jaccard similarity between two sets.

    Shared utility function used by all bootstrap scripts.

    Args:
        a: First set (or list)
        b: Second set (or list)

    Returns:
        Jaccard similarity coefficient [0, 1]
    """
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def bootstrap_selection(
    alpha: float,
    beta: float,
    gamma: float,
    min_distance_km: float,
    n_samples: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    original_selection: list,
    cluster_labels_full: np.ndarray,
    n_boot: int = 500,
    random_seed: int = 42,
    pre_selected_names: Optional[list] = None,
    pre_selected_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Perform bootstrap resampling to assess selection stability.

    Args:
        alpha: Visual diversity weight
        beta: Spatial diversity weight
        gamma: Temporal diversity weight
        min_distance_km: Minimum distance constraint
        n_samples: Number of samples to select
        features: Feature matrix (N x D)
        metadata: Metadata DataFrame
        original_selection: Original selection indices
        cluster_labels_full: Cluster labels for all samples
        n_boot: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        pre_selected_names: Optional pre-selected tile names
        pre_selected_indices: Optional pre-selected tile indices

    Returns:
        DataFrame with columns: iteration, n_samples, jaccard_with_original,
        clusters_covered, temporal_std, spatial_mean_km, wwi_percent, etc.
    """
    from dataselector.analysis.metrics import compute_metrics
    from dataselector.selection.diversity_selector import DiversitySelector

    rng = np.random.default_rng(random_seed)
    N = features.shape[0]
    results = []

    ds = DiversitySelector(
        n_samples=n_samples,
        clustering_method="kmeans",
        n_clusters=8,
    )

    for i in trange(n_boot, desc="Bootstrap iterations"):
        # Resample with replacement
        sample_idx = rng.integers(0, N, size=N)
        boot_features = features[sample_idx]
        boot_meta = metadata.iloc[sample_idx].reset_index(drop=True)

        selected_boot = ds.select(
            features=boot_features,
            metadata=boot_meta,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_distance_km),
            pre_selected=pre_selected_indices,
            pre_selected_names=pre_selected_names,
        )

        # Map back to original indices
        mapped = np.unique(sample_idx[selected_boot]).tolist()

        # Compute metrics on original data
        metrics = compute_metrics(mapped, metadata, cluster_labels_full, features)
        metrics["jaccard_with_original"] = jaccard(mapped, original_selection)
        metrics["bootstrap_iteration"] = i
        metrics["n_samples"] = len(mapped)
        results.append(metrics)

    return pd.DataFrame(results)


def summarize_bootstrap(df_boot: pd.DataFrame, original_metrics: dict) -> pd.Series:
    """Compute summary statistics (mean, std, CI) for bootstrap results.

    Args:
        df_boot: Bootstrap results DataFrame
        original_metrics: Original selection metrics

    Returns:
        Series with summary statistics (_mean, _std, _ci_lower, _ci_upper, _original)
    """
    summary = {}
    # Key metrics to summarize
    metrics = [
        "n_selected",
        "clusters_covered",
        "temporal_std",
        "spatial_mean_km",
        "wwi_percent",
        "jaccard_with_original",
    ]

    for m in metrics:
        if m in df_boot.columns:
            vals = df_boot[m].dropna()
            if len(vals) > 0:
                summary[f"{m}_mean"] = vals.mean()
                summary[f"{m}_std"] = vals.std()
                summary[f"{m}_ci_lower"] = vals.quantile(0.025)
                summary[f"{m}_ci_upper"] = vals.quantile(0.975)
                summary[f"{m}_original"] = original_metrics.get(m, np.nan)

    return pd.Series(summary)


def bootstrap_candidate(
    alpha: float,
    beta: float,
    gamma: float,
    min_d: float,
    features: np.ndarray,
    metadata: pd.DataFrame,
    original_selection: list,
    cluster_labels_full: np.ndarray,
    n_boot: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Perform bootstrap resampling for Pareto candidate.

    Args:
        alpha: Visual diversity weight
        beta: Spatial diversity weight
        gamma: Temporal diversity weight
        min_d: Minimum distance constraint
        features: Feature matrix
        metadata: Metadata DataFrame
        original_selection: Original selection indices
        cluster_labels_full: Cluster labels
        n_boot: Number of bootstrap iterations
        random_seed: Random seed

    Returns:
        DataFrame with bootstrap statistics
    """
    from dataselector.analysis.metrics import compute_metrics
    from dataselector.data.io import attach_metric_gdf, get_metric_gdf
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector

    rng = np.random.default_rng(random_seed)
    N = features.shape[0]
    results = []

    ds = DiversitySelector(n_samples=300, clustering_method="kmeans", n_clusters=8)

    for i in range(n_boot):
        sample_idx = rng.integers(0, N, size=N)
        boot_features = features[sample_idx]
        boot_meta = metadata.iloc[sample_idx].reset_index(drop=True)

        # Preserve projected coords in the bootstrap sample if present
        gdf_metric = get_metric_gdf(metadata)
        if gdf_metric is not None:
            attach_metric_gdf(
                boot_meta, gdf_metric.iloc[sample_idx].reset_index(drop=True)
            )

        # clustering on boot features (not used for metrics -- metrics computed on original mapping)
        clustering = ClusteringPipeline(n_clusters=8)
        try:
            _embeddings, _cluster_labels_boot = clustering.fit_transform(boot_features)
        except Exception:
            # if clustering fails, continue with metrics computed on original labels
            pass

        selected_boot = ds.select(
            features=boot_features,
            metadata=boot_meta,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
        )

        # Map selected indices back to original indices
        mapped = np.unique(sample_idx[selected_boot]).tolist()

        # compute metrics relative to original data using full clustering labels
        metrics = compute_metrics(mapped, metadata, cluster_labels_full, features)
        # compute jaccard with original_selection
        metrics["jaccard_with_original"] = jaccard(mapped, original_selection)
        metrics["bootstrap_i"] = i
        results.append(metrics)

    return pd.DataFrame(results)


def run_bootstrap_final(
    run_dir: str | Path,
    n_boot: int = 500,
    seed: int = 42,
) -> int:
    """High-level orchestration: Bootstrap UQ for final Optuna selection.

    This replaces the main() function from bootstrap_final_selection.py.

    Args:
        run_dir: Path to run directory
        n_boot: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Exit code (0 on success)
    """
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector

    ROOT = _get_repo_root()
    run_dir = Path(run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Load best trial configuration
    best_trial_file = run_dir / "results" / "best_trial.json"
    if not best_trial_file.exists():
        print(f"Error: best_trial.json not found in {run_dir}")
        return 1

    with open(best_trial_file) as f:
        best_trial = json.load(f)

    # Load best selection config
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        sel_config = config.get("selection", {})
    else:
        # Fallback: extract from best_trial
        sel_config = {}
        total = best_trial["a"] + best_trial["b"] + best_trial["c"]
        sel_config["alpha_visual"] = best_trial["a"] / total
        sel_config["beta_spatial"] = best_trial["b"] / total
        sel_config["gamma_temporal"] = best_trial["c"] / total
        sel_config["min_distance_km"] = best_trial["min_distance_km"]
        sel_config["n_samples"] = best_trial["n_samples"]
        sel_config["pre_selected_names"] = best_trial.get("pre_selected_names")
        sel_config["pre_selected_indices"] = best_trial.get("pre_selected_indices")

    print(f"\n{'='*60}")
    print("Bootstrap UQ for Final Selection")
    print(f"{'='*60}")
    print(f"Run: {run_dir.name}")
    print(f"Min distance: {sel_config['min_distance_km']} km")
    print(f"n_samples: {sel_config['n_samples']}")
    print(f"Bootstrap iterations: {n_boot}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading metadata and features...")
    metadata_path = ROOT / "outputs" / "metadata.csv"
    if not metadata_path.exists():
        metadata = load_metadata(str(ROOT / "data" / "new_all_tiles.csv"))
    else:
        metadata = load_metadata(str(metadata_path))

    features = load_or_extract_features(
        ROOT / "outputs",
        csv_meta=str(metadata_path) if metadata_path.exists() else None,
        cache=True,
    )

    # Full clustering for metrics
    print("Computing cluster labels...")
    clustering = ClusteringPipeline(n_clusters=8)
    try:
        _, cluster_labels_full = clustering.fit_transform(features)
    except Exception as e:
        print(f"Warning: Clustering failed ({e}), using dummy labels")
        cluster_labels_full = np.zeros(features.shape[0], dtype=int)

    # Compute original selection
    ds = DiversitySelector(
        n_samples=sel_config["n_samples"],
        clustering_method="kmeans",
        n_clusters=8,
    )
    original_selection = ds.select(
        features=features,
        metadata=metadata,
        alpha_visual=sel_config["alpha_visual"],
        beta_spatial=sel_config["beta_spatial"],
        gamma_temporal=sel_config["gamma_temporal"],
        spatial_constraint=True,
        min_distance_km=sel_config["min_distance_km"],
        pre_selected=sel_config.get("pre_selected_indices"),
        pre_selected_names=sel_config.get("pre_selected_names"),
    )

    from dataselector.analysis.metrics import compute_metrics

    original_metrics = compute_metrics(
        original_selection, metadata, cluster_labels_full, features
    )

    print(f"Original selection: {len(original_selection)} samples")
    print(f"  Clusters: {original_metrics['clusters_covered']}")
    print(f"  Temporal std: {original_metrics['temporal_std']:.2f}")
    print(f"  Spatial mean: {original_metrics['spatial_mean_km']:.2f} km\n")

    # Run bootstrap
    df_boot = bootstrap_selection(
        alpha=sel_config["alpha_visual"],
        beta=sel_config["beta_spatial"],
        gamma=sel_config["gamma_temporal"],
        min_distance_km=sel_config["min_distance_km"],
        n_samples=sel_config["n_samples"],
        features=features,
        metadata=metadata,
        original_selection=original_selection,
        cluster_labels_full=cluster_labels_full,
        n_boot=n_boot,
        random_seed=seed,
        pre_selected_names=sel_config.get("pre_selected_names"),
        pre_selected_indices=sel_config.get("pre_selected_indices"),
    )

    # Save full results
    results_file = run_dir / "results" / "bootstrap_final_selection_full.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    df_boot.to_csv(results_file, index=False)
    print(f"\n✓ Saved full bootstrap results: {results_file}")

    # Compute and save summary
    summary = summarize_bootstrap(df_boot, original_metrics)
    summary_df = summary.to_frame().T
    summary_file = run_dir / "results" / "bootstrap_final_selection_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("Bootstrap Summary (95% CI)")
    print(f"{'='*60}")
    for key in [
        "n_selected",
        "clusters_covered",
        "temporal_std",
        "spatial_mean_km",
        "jaccard_with_original",
    ]:
        if f"{key}_mean" in summary.index:
            mean = summary[f"{key}_mean"]
            std = summary[f"{key}_std"]
            ci_low = summary[f"{key}_ci_lower"]
            ci_up = summary[f"{key}_ci_upper"]
            orig = summary.get(f"{key}_original", np.nan)
            print(
                f"{key:25s}: {mean:7.2f} ± {std:6.2f}  [{ci_low:7.2f}, {ci_up:7.2f}]  (orig: {orig:.2f})"
            )
    print(f"{'='*60}\n")

    return 0


def run_bootstrap_pareto(
    pareto_csv: str | Path,
    n_boot: int = 200,
    output_csv: Optional[str | Path] = None,
    random_seed: int = 42,
    uq_method: str = "bootstrap",
) -> int:
    """High-level orchestration: Bootstrap UQ for Pareto candidates.

    This replaces the main() function from bootstrap_pareto_candidates.py.

    Args:
        pareto_csv: Path to Pareto candidates CSV
        n_boot: Number of bootstrap iterations
        output_csv: Optional output CSV path
        random_seed: Random seed
        uq_method: UQ method ('bootstrap' or 'ensemble')

    Returns:
        Exit code (0 on success)
    """
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector

    ROOT = _get_repo_root()
    pareto = pd.read_csv(pareto_csv)

    # load full metadata and features
    metadata_path = ROOT / "outputs" / "metadata.csv"
    if not metadata_path.exists():
        metadata = load_metadata(str(ROOT / "data" / "new_all_tiles.csv"))
    else:
        metadata = load_metadata(str(metadata_path))

    features = load_or_extract_features(
        ROOT / "outputs",
        csv_meta=str(metadata_path) if metadata_path.exists() else None,
        cache=True,
    )

    # full clustering (for cluster labels)
    clustering = ClusteringPipeline(n_clusters=8)
    try:
        embeddings_full, cluster_labels_full = clustering.fit_transform(features)
    except Exception:
        cluster_labels_full = np.zeros(features.shape[0], dtype=int)

    all_boot = []
    summary_rows = []

    ds = DiversitySelector(n_samples=300, clustering_method="kmeans", n_clusters=8)

    for idx, row in pareto.iterrows():
        alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
        min_d = row["min_distance_km"]

        # compute original selection on full dataset
        selected = ds.select(
            features=features,
            metadata=metadata,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
        )
        original_sel = list(selected)

        # Bootstrap
        df_boot = bootstrap_candidate(
            alpha,
            beta,
            gamma,
            min_d,
            features,
            metadata,
            original_sel,
            cluster_labels_full,
            n_boot=n_boot,
            random_seed=random_seed,
        )
        df_boot["alpha"] = alpha
        df_boot["beta"] = beta
        df_boot["gamma"] = gamma
        df_boot["min_distance_km"] = min_d
        df_boot["n_selected"] = int(row["n_selected"])
        df_boot["pareto_idx"] = idx
        all_boot.append(df_boot)

        # summary
        summary = {
            "pareto_idx": idx,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "min_distance_km": min_d,
            "n_selected": int(row["n_selected"]),
            "temporal_std_mean": df_boot["temporal_std"].mean(),
            "temporal_std_std": df_boot["temporal_std"].std(),
            "wwi_percent_mean": df_boot["wwi_percent"].mean(),
            "wwi_percent_std": df_boot["wwi_percent"].std(),
            "jaccard_mean": df_boot["jaccard_with_original"].mean(),
            "jaccard_std": df_boot["jaccard_with_original"].std(),
            "method": "bootstrap",
        }
        summary_rows.append(summary)

    if len(all_boot) > 0:
        df_all = pd.concat(all_boot, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    df_summary = pd.DataFrame(summary_rows)

    # Save results
    outdir = (
        Path(output_csv).parent
        if output_csv is not None
        else ROOT / "outputs" / "fine_sweep"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        # Default legacy path
        default_dir = ROOT / "outputs" / "fine_sweep"
        if not df_all.empty:
            df_all.to_csv(default_dir / "bootstrap_results_full.csv", index=False)
        df_summary.to_csv(default_dir / "bootstrap_summary.csv", index=False)
    else:
        if not df_all.empty:
            df_all.to_csv(output_csv, index=False)
        df_summary.to_csv(
            Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv"),
            index=False,
        )

    print("Bootstrap finished. Results saved.")
    return 0


@cli_command(
    "bootstrap-final",
    help="Bootstrap UQ for final Optuna selection",
    args={
        "run_dir": {
            "type": str,
            "required": True,
            "help": "Path to run directory",
        },
        "n_boot": {
            "type": int,
            "default": 500,
            "help": "Number of bootstrap iterations",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Random seed",
        },
    },
)
def bootstrap_final_cli(run_dir: str, n_boot: int = 500, seed: int = 42) -> int:
    """CLI entry point for bootstrap final selection."""
    return run_bootstrap_final(run_dir=run_dir, n_boot=n_boot, seed=seed)


@cli_command(
    "bootstrap-pareto",
    help="Bootstrap UQ for Pareto candidates",
    args={
        "pareto_csv": {
            "type": str,
            "required": True,
            "help": "Path to Pareto candidates CSV",
        },
        "n_boot": {
            "type": int,
            "default": 200,
            "help": "Number of bootstrap iterations",
        },
        "output_csv": {
            "type": str,
            "default": None,
            "help": "Optional output CSV path",
        },
        "random_seed": {
            "type": int,
            "default": 42,
            "help": "Random seed",
        },
        "uq_method": {
            "type": str,
            "choices": ["bootstrap", "ensemble"],
            "default": "bootstrap",
            "help": "UQ method",
        },
    },
)
def bootstrap_pareto_cli(
    pareto_csv: str,
    n_boot: int = 200,
    output_csv: str = None,
    random_seed: int = 42,
    uq_method: str = "bootstrap",
) -> int:
    """CLI entry point for bootstrap pareto candidates."""
    return run_bootstrap_pareto(
        pareto_csv=pareto_csv,
        n_boot=n_boot,
        output_csv=output_csv,
        random_seed=random_seed,
        uq_method=uq_method,
    )


# Legacy support: for backwards compatibility with old CLI structure
def main(argv=None) -> int:
    """Legacy wrapper for old subcommand-based interface.

    This is kept for backwards compatibility only.
    New code should use bootstrap_final_cli() or bootstrap_pareto_cli() directly.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="dataselector bootstrap",
        description="Bootstrap uncertainty quantification workflows",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # bootstrap final
    final_parser = subparsers.add_parser(
        "final", help="Bootstrap UQ for final Optuna selection"
    )
    final_parser.add_argument("--run-dir", required=True, help="Path to run directory")
    final_parser.add_argument(
        "--n-boot", type=int, default=500, help="Number of bootstrap iterations"
    )
    final_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # bootstrap pareto
    pareto_parser = subparsers.add_parser(
        "pareto", help="Bootstrap UQ for Pareto candidates"
    )
    pareto_parser.add_argument(
        "--pareto-csv", required=True, help="Path to Pareto candidates CSV"
    )
    pareto_parser.add_argument(
        "--n-boot", type=int, default=200, help="Number of bootstrap iterations"
    )
    pareto_parser.add_argument("--output-csv", help="Optional output CSV path")
    pareto_parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed"
    )
    pareto_parser.add_argument(
        "--uq-method",
        choices=["bootstrap", "ensemble"],
        default="bootstrap",
        help="UQ method (bootstrap only for now)",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "final":
        return run_bootstrap_final(
            run_dir=args.run_dir,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    elif args.subcommand == "pareto":
        return run_bootstrap_pareto(
            pareto_csv=args.pareto_csv,
            n_boot=args.n_boot,
            output_csv=args.output_csv,
            random_seed=args.random_seed,
            uq_method=args.uq_method,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
