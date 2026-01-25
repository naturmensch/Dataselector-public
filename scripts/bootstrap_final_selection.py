#!/usr/bin/env python3
"""Bootstrap uncertainty quantification for final Optuna selection.
This extends bootstrap analysis to the actual final selection produced by Optuna,
providing confidence intervals and stability metrics for the thesis.
Usage:
    python scripts/bootstrap_final_selection.py --run-dir outputs/runs/<run> --n-boot 500          """

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.common import DATA_DIR, data_path
except Exception:
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.common import DATA_DIR, data_path

ROOT = Path(__file__).resolve().parents[1]

# STARTUP ENV VALIDATION
try:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.compat import validate_environment_full
    if "--skip-env-check" not in sys.argv:
        validate_environment_full()
except Exception as e:
    print(f"\n❌ STARTUP VALIDATION FAILED:\n{e}\n", file=sys.stderr)
    print("Fix: ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages 'numpy==1.26.4 numba==0.63.1' --yes -- python scripts/bootstrap_final_selection.py", file=sys.stderr)
    sys.exit(1)

# Note: Project imports (src.*) are deferred into `main` or helper functions to make
# this module import-safe for tests and linters (avoid import-time side-effects).


def jaccard(a, b):
    """Jaccard similarity between two sets."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def bootstrap_selection(
    alpha,
    beta,
    gamma,
    min_distance_km,
    n_samples,
    features,
    metadata,
    original_selection,
    cluster_labels_full,
    n_boot=500,
    random_seed=42,
    pre_selected_names=None,
    pre_selected_indices=None,
):
    """Perform bootstrap resampling to assess selection stability.

    Returns DataFrame with metrics for each bootstrap iteration.
    """
    # Local imports to keep module import-safe
    from tqdm import trange
    from src.diversity_selector import DiversitySelector
    from src.metrics import compute_metrics

    rng = np.random.default_rng(random_seed)
    N = features.shape[0]
    results = []

    for i in trange(n_boot, desc="Bootstrap iterations"):
        # Resample with replacement
        sample_idx = rng.integers(0, N, size=N)
        boot_features = features[sample_idx]
        boot_meta = metadata.iloc[sample_idx].reset_index(drop=True)
        # Run selection on bootstrap sample
        ds = DiversitySelector(
            n_samples=n_samples, use_multi_criteria=True, random_state=int(1000 + i)
        )
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


def summarize_bootstrap(df_boot, original_metrics):
    """Compute summary statistics (mean, std, CI) for bootstrap results."""
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
            summary[f"{m}_mean"] = df_boot[m].mean()
            summary[f"{m}_std"] = df_boot[m].std()
            summary[f"{m}_ci_lower"] = df_boot[m].quantile(0.025)
            summary[f"{m}_ci_upper"] = df_boot[m].quantile(0.975)
            summary[f"{m}_original"] = original_metrics.get(m, np.nan)
    return pd.Series(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap UQ for final Optuna selection"
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--n-boot", type=int, default=500, help="Number of bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Smoke-mode: operate on small synthetic data if real data missing")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Load best trial configuration
    best_trial_file = run_dir / "results" / "best_trial.json"
    if not best_trial_file.exists():
        print(f"Error: best_trial.json not found in {run_dir}")
        return 1

    import json

    with open(best_trial_file) as f:
        best_trial = json.load(f)

    # Load best selection config
    config_file = run_dir / "config" / "config_best_selection.yaml"
    if config_file.exists():
        import yaml

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
    print(
        f"Config: α={sel_config['alpha_visual']:.3f}, β={sel_config['beta_spatial']:.3f}, γ={sel_config['gamma_temporal']:.3f}"
    )
    print(f"Min distance: {sel_config['min_distance_km']} km")
    print(f"n_samples: {sel_config['n_samples']}")
    print(f"Bootstrap iterations: {args.n_boot}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading metadata and features...")
    metadata_path = ROOT / "outputs" / "metadata.csv"

    # Local imports to keep module import-safe
    from src.io import load_metadata, load_or_extract_features
    from src.clustering import ClusteringPipeline
    from src.diversity_selector import DiversitySelector
    from src.metrics import compute_metrics

    try:
        if not metadata_path.exists():
            metadata = load_metadata(str(data_path("new_all_tiles.csv")))
        else:
            metadata = pd.read_csv(metadata_path)

        features = load_or_extract_features(
            ROOT / "outputs",
            csv_meta=str(metadata_path) if metadata_path.exists() else None,
            cache=True,
        )
    except FileNotFoundError as e:
        # In smoke mode, try to collect test data automatically
        if args.smoke:
            print(f"Warning: {e} — attempting to collect test data for smoke mode")
            import subprocess
            try:
                result = subprocess.run([
                    "bash", "tests/scripts/collect_test_subset.sh",
                    "--n-images", "5",
                    "--datasets", "hamburg", "kdr100"
                ], capture_output=True, text=True, cwd=ROOT)
                if result.returncode == 0:
                    print("Test data collected successfully, creating data symlink...")
                    # Create symlink from data/ to tests/test_data/ if not exists
                    data_dir = ROOT / "data"
                    test_data_dir = ROOT / "tests" / "test_data"
                    if not data_dir.exists():
                        data_dir.symlink_to(test_data_dir, target_is_directory=True)
                    # Retry loading data
                    features = load_or_extract_features(
                        ROOT / "outputs",
                        csv_meta=str(metadata_path) if metadata_path.exists() else None,
                        cache=True,
                    )
                else:
                    raise FileNotFoundError(f"Failed to collect test data: {result.stderr}")
            except Exception as collect_e:
                raise FileNotFoundError(f"Real data required for smoke mode and auto-collection failed: {e}; {collect_e}") from e
        else:
            raise


    # Full clustering for metrics
    print("Computing cluster labels...")
    clustering = ClusteringPipeline(n_clusters=8)
    try:
        _, cluster_labels_full = clustering.fit_transform(features)
    except Exception as e:
        print(f"Warning: Clustering failed ({e}), using dummy labels")
        cluster_labels_full = np.zeros(features.shape[0], dtype=int)
    # Compute original selection
    print("Computing original selection...")
    ds = DiversitySelector(
        n_samples=sel_config["n_samples"], use_multi_criteria=True, random_state=42
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
    original_metrics = compute_metrics(original_selection, metadata, cluster_labels_full, features)

    print(f"Original selection: {len(original_selection)} samples")
    print(f"  Clusters: {original_metrics['clusters_covered']}")
    print(f"  Temporal std: {original_metrics['temporal_std']:.2f}")
    print(f"  Spatial mean: {original_metrics['spatial_mean_km']:.2f} km\n")
    # Run bootstrap
    print(f"Running {args.n_boot} bootstrap iterations...\n")
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
        n_boot=args.n_boot,
        random_seed=args.seed,
        pre_selected_names=sel_config.get("pre_selected_names"),
        pre_selected_indices=sel_config.get("pre_selected_indices"),
    )

    # Save full results
    results_file = run_dir / "results" / "bootstrap_final_selection_full.csv"
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


if __name__ == "__main__":
    sys.exit(main())