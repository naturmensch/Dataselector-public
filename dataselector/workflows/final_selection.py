"""Final selection workflow with bootstrap-best config support."""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command

logger = logging.getLogger(__name__)


def run_final_selection(
    n_samples: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    min_distance_km: Optional[float] = None,
    use_bootstrap_best: bool = False,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run final selection with given parameters or load from bootstrap-best config.

    Args:
        n_samples: Number of samples to select (if None, load from config)
        alpha: Visual diversity weight (if None, load from config)
        beta: Spatial diversity weight (if None, load from config)
        gamma: Temporal diversity weight (if None, load from config)
        min_distance_km: Minimum distance constraint in km (if None, load from config)
        use_bootstrap_best: Load parameters from bootstrap-best config
        seed: Random seed for reproducibility
        output_dir: Output directory (defaults to outputs/final_selection)
        config_path: Path to config file (optional override)
        metadata_path: Path to metadata CSV (optional override)
        features_path: Path to features NPY (optional override)

    Returns:
        Tuple of (selected_metadata_df, metrics_dict)
    """
    # Lazy imports to avoid heavy dependencies at import time
    import yaml

    from dataselector.analysis.metrics import compute_metrics
    from dataselector.analysis.visualizer import Visualizer
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata
    from dataselector.selection.diversity_selector import DiversitySelector

    # Set up output directory
    if output_dir is None:
        output_dir = Path("outputs/final_selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config (bootstrap or default)
    if config_path is None:
        if use_bootstrap_best:
            config_path = Path("outputs/pipeline_config.bootstrap.yaml")
            if not config_path.exists():
                print(
                    f"Warning: Bootstrap config not found at {config_path}, falling back to default"
                )
                config_path = Path("config/pipeline_config.yaml")
            else:
                print(f"Using bootstrap-best config: {config_path}")
        else:
            config_path = Path("config/pipeline_config.yaml")

    print(f"Loading config from: {config_path}")

    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        cfg = {}

    # Extract parameters (CLI overrides config)
    n_samples = n_samples or cfg.get("selection", {}).get("n_samples", 34)
    alpha = (
        alpha
        if alpha is not None
        else cfg.get("selection", {}).get("alpha_visual", 0.7)
    )
    beta = (
        beta if beta is not None else cfg.get("selection", {}).get("beta_spatial", 0.05)
    )
    gamma = (
        gamma
        if gamma is not None
        else cfg.get("selection", {}).get("gamma_temporal", 0.25)
    )

    # Production policy: only canonical metadata source is allowed.
    metadata_path = assert_canonical_metadata(
        metadata_path,
        context="final-selection",
    )

    if min_distance_km is None:
        cfg_distance = cfg.get("selection", {}).get("min_distance_km")
        if cfg_distance is not None:
            min_distance_km = float(cfg_distance)
        else:
            from dataselector.pipeline.pipeline_utils import compute_min_distance_km

            min_distance_km = compute_min_distance_km(str(metadata_path))

    print("\nFinal selection parameters:")
    print(f"  n_samples: {n_samples}")
    print(f"  α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
    print(f"  min_distance_km: {min_distance_km}")
    print(f"  seed: {seed}")

    # Load metadata and features
    metadata = load_metadata(str(metadata_path))
    print(f"Loaded {len(metadata)} metadata records from {metadata_path}")

    # Load features (with caching and on-demand extraction)
    if features_path is None:
        print("Loading or extracting features (cached)...")
        features = load_or_extract_features(
            out_dir=Path("outputs"),
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=True,
            enforce_canonical=True,
        )
    else:
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        features = np.load(features_path)

    print(f"Loaded features: shape={features.shape}")

    if len(features) != len(metadata):
        raise ValueError(
            f"Features ({len(features)}) and metadata ({len(metadata)}) length mismatch"
        )

    # Load cluster labels (optional, for metrics)
    cluster_path = Path("data/cluster_labels.npy")
    if cluster_path.exists():
        cluster_labels = np.load(cluster_path)
        print(f"Loaded cluster labels: {len(cluster_labels)} samples")
    else:
        print("No cluster labels found, using dummy labels for metrics")
        cluster_labels = np.zeros(len(metadata), dtype=int)

    # Load 2D embeddings for visualization
    embeddings_2d_path = Path("data/embeddings_2d.npy")
    if embeddings_2d_path.exists():
        embeddings_2d = np.load(embeddings_2d_path)
        print(f"Loaded 2D embeddings: shape={embeddings_2d.shape}")
    else:
        print("No 2D embeddings found, skipping visualizations")
        embeddings_2d = None

    # Initialize selector
    selector = DiversitySelector(
        n_samples=n_samples, random_state=seed, use_multi_criteria=True
    )

    # Pre-selected names/indices (optional)
    pre_selected_names = cfg.get("selection", {}).get("pre_selected_names", None)
    pre_selected_indices = cfg.get("selection", {}).get("pre_selected_indices", None)

    if pre_selected_names is not None:
        print(f"Using pre-selected names: {pre_selected_names}")
    if pre_selected_indices is not None:
        print(f"Using pre-selected indices: {pre_selected_indices}")

    # Run selection
    print("\nRunning selection...")
    start_time = time.time()

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

    duration = time.time() - start_time
    print(f"Selection completed in {duration:.2f}s")
    print(f"Selected {len(selected_idx)} samples")
    shortfall = max(0, int(n_samples) - int(len(selected_idx)))
    if shortfall > 0:
        print(
            "⚠️  Hard-cut shortfall in final-selection: "
            f"requested={n_samples}, selected={len(selected_idx)}"
        )

    # Export selection
    sel_df = metadata.iloc[selected_idx].copy()
    sel_df["selection_rank"] = range(len(sel_df))

    sel_csv = (
        output_dir
        / f"final_selection_n{n_samples}_a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}_d{int(min_distance_km)}.csv"
    )
    sel_df.to_csv(sel_csv, index=False)
    print(f"Selection saved: {sel_csv}")

    # Compute metrics
    metrics = compute_metrics(selected_idx, metadata, cluster_labels, features)
    metrics.update(
        {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "min_distance_km": min_distance_km,
            "n_requested": n_samples,
            "n_selected": len(selected_idx),
            "selection_shortfall": shortfall,
            "hardcut_target_met": shortfall == 0,
            "duration_s": duration,
            "seed": seed,
            "pre_selected_names": pre_selected_names,
            "pre_selected_indices": pre_selected_indices,
        }
    )

    # Create visualizations (if 2D embeddings available)
    if embeddings_2d is not None:
        print("Creating visualizations...")
        viz = Visualizer(output_dir=str(output_dir))
        viz.create_summary_report(
            embeddings_2d=embeddings_2d,
            cluster_labels=cluster_labels,
            metadata=metadata,
            selected_indices=selected_idx,
            output_prefix=f"final_n{n_samples}",
        )

    # Write report
    report_path = output_dir / "final_selection_report.md"
    with open(report_path, "w") as f:
        f.write("# Final Selection Report\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("## Parameters\n")
        f.write(f"- n_requested: {n_samples}\n")
        f.write(f"- α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\n")
        f.write(f"- min_distance_km: {min_distance_km}\n")
        f.write(f"- seed: {seed}\n")
        if pre_selected_names:
            f.write(f"- pre_selected_names: {pre_selected_names}\n")
        if pre_selected_indices:
            f.write(f"- pre_selected_indices: {pre_selected_indices}\n")
        f.write("\n## Metrics\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"- {k}: {v:.4f}\n")
            else:
                f.write(f"- {k}: {v}\n")
        f.write("\n## Outputs\n")
        f.write(f"- Selection CSV: {sel_csv}\n")
        if embeddings_2d is not None:
            f.write(f"- Plots: {output_dir / f'final_n{n_samples}'}\n")

    print(f"Report saved: {report_path}")
    print("Done.")

    return sel_df, metrics


@cli_command(
    "final-selection",
    help="Run final selection with given parameters or bootstrap-best config",
    args={
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Number of samples to select (default: from config)",
        },
        "alpha": {
            "type": float,
            "default": None,
            "help": "Visual diversity weight α (default: from config)",
        },
        "beta": {
            "type": float,
            "default": None,
            "help": "Spatial diversity weight β (default: from config)",
        },
        "gamma": {
            "type": float,
            "default": None,
            "help": "Temporal diversity weight γ (default: from config)",
        },
        "min_distance_km": {
            "type": float,
            "default": None,
            "help": "Minimum distance constraint in km (default: from config)",
        },
        "use_bootstrap_best": {
            "type": bool,
            "action": "store_true",
            "help": "Load parameters from bootstrap-best config",
        },
        "seed": {
            "type": int,
            "default": 42,
            "help": "Random seed for reproducibility",
        },
        "output_dir": {
            "type": str,
            "default": None,
            "help": "Output directory (default: outputs/final_selection)",
        },
        "config_path": {
            "type": str,
            "default": None,
            "help": "Path to config file (optional override)",
        },
        "metadata_path": {
            "type": str,
            "default": None,
            "help": "Path to metadata CSV (optional override)",
        },
        "features_path": {
            "type": str,
            "default": None,
            "help": "Path to features NPY (optional override)",
        },
    },
)
def main(
    n_samples: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    min_distance_km: Optional[float] = None,
    use_bootstrap_best: bool = False,
    seed: int = 42,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    features_path: Optional[str] = None,
):
    """CLI entry point for final selection workflow."""
    # Convert str paths to Path objects
    output_dir_path = Path(output_dir) if output_dir else None
    config_path_obj = Path(config_path) if config_path else None
    metadata_path_obj = Path(metadata_path) if metadata_path else None
    features_path_obj = Path(features_path) if features_path else None

    # Call the workflow function
    sel_df, metrics = run_final_selection(
        n_samples=n_samples,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        min_distance_km=min_distance_km,
        use_bootstrap_best=use_bootstrap_best,
        seed=seed,
        output_dir=output_dir_path,
        config_path=config_path_obj,
        metadata_path=metadata_path_obj,
        features_path=features_path_obj,
    )

    print("\n✅ Final selection complete!")
    print(f"Selected {len(sel_df)} tiles")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run final selection workflow")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples")
    parser.add_argument(
        "--alpha", type=float, default=None, help="Visual diversity weight"
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Spatial diversity weight"
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="Temporal diversity weight"
    )
    parser.add_argument(
        "--min-distance-km",
        type=float,
        default=None,
        help="Min distance constraint (km)",
    )
    parser.add_argument(
        "--use-bootstrap-best", action="store_true", help="Use bootstrap-best config"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--config-path", type=str, default=None, help="Config file path"
    )
    parser.add_argument(
        "--metadata-path", type=str, default=None, help="Metadata CSV path"
    )
    parser.add_argument(
        "--features-path", type=str, default=None, help="Features NPY path"
    )

    args = parser.parse_args()
    main(
        n_samples=args.n_samples,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        min_distance_km=args.min_distance_km,
        use_bootstrap_best=args.use_bootstrap_best,
        seed=args.seed,
        output_dir=args.output_dir,
        config_path=args.config_path,
        metadata_path=args.metadata_path,
        features_path=args.features_path,
    )
