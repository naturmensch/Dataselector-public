#!/usr/bin/env python3
"""
Tune Weights Workflow — LHS/Sobol Exploration for Multi-Criteria Selection

Phase 1: Exploration via Latin Hypercube Sampling or Sobol sequences
Generates weight combinations to explore Pareto front and trade-offs.

Migration from: scripts/tune_weights_and_run.py
Author: Phase 5 Migration
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_META = ROOT / "data" / "new_all_tiles.csv"
OUTPUT_DIR = ROOT / "outputs" / "tuning_weights"


def generate_weights(n_points: int = 50, seed: int = 42, sampler: str = "lhs"):
    """
    Generate weight combinations on the simplex using the specified sampler.

    Parameters
    ----------
    n_points : int
        Number of samples
    seed : int
        Random seed
    sampler : str
        'lhs' or 'sobol'

    Returns
    -------
    list
        List of (alpha, beta, gamma) tuples summing to 1.0
    """
    from dataselector.workflows.sampling_strategies import (
        sample_weights_on_simplex_lhs,
        sample_weights_on_simplex_sobol,
    )

    if sampler.lower() == "sobol":
        return sample_weights_on_simplex_sobol(n_points, dim=3, seed=seed)
    elif sampler.lower() == "lhs":
        return sample_weights_on_simplex_lhs(n_points, dim=3, seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")


def run_exploration(
    n_samples: int = 50,
    selection_n_samples: int | None = None,
    sampler: str = "lhs",
    objective_authority: str | None = None,
    seed: int = 42,
    metadata_path: Path | str | None = None,
    min_distance_km: float | None = None,
    n_clusters: int | None = None,
    batch_size: int | None = None,
    pre_names: list | None = None,
    pre_indices: list | None = None,
    output_dir: Path | None = None,
) -> tuple[list, Path]:
    """
    Run LHS/Sobol exploration sweep and compute Pareto front.

    Parameters
    ----------
    n_samples : int
        Number of weight combinations (LHS/Sobol points)
    selection_n_samples : int | None
        Number of tiles to select per weight combination. If None, resolve from
        config (`selection.n_samples`) or autoscale artifact; otherwise fail-fast.
    sampler : str
        'lhs' or 'sobol'
    objective_authority : str | None
        Selection objective mode for sweep ranking.
    seed : int
        Random seed
    metadata_path : Path | str | None
        Path to metadata CSV for computing min_distance_km (required, no hardcoded fallback)
    min_distance_km : float | None
        Explicit min distance policy (km). If None, load from pipeline config and
        only fall back to data-driven computation when config has no value.
    n_clusters : int | None
        Clustering count used by ExperimentRunner sweep. If None, resolve from config.
    batch_size : int | None
        Feature extraction batch size used by ExperimentRunner sweep. If None, resolve
        from config.
    pre_names : list | None
        Pre-selected tile names
    pre_indices : list | None
        Pre-selected indices
    output_dir : Path | None
        Output directory

    Returns
    -------
    tuple[list, Path]
        (pareto_front, report_csv_path)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline integration
    import os

    import yaml

    em = None
    exp_dir = os.environ.get("EXPERIMENT_RUN_DIR")
    if exp_dir:
        from dataselector.pipeline.experiment_manager import ExperimentManager

        em = ExperimentManager.from_existing(exp_dir)
        em.log("Attached to pipeline run (exploration stage)")
        em.save_config(
            "exploration",
            {
                "lhs_points": n_samples,
                "selection_n_samples": selection_n_samples,
                "sampler": sampler,
                "seed": seed,
            },
        )

    from dataselector.pipeline.experiments import ExperimentRunner
    from dataselector.pipeline.pipeline_utils import compute_min_distance_km
    from dataselector.selection.pareto import (
        compute_pareto_front,
        export_pareto_report,
        visualize_pareto_front,
    )
    from dataselector.workflows._selection_target import resolve_selection_n_samples

    # Compute min_distance_km from data (no fallback)
    if metadata_path is None:
        raise ValueError(
            "metadata_path is required for computing min_distance_km. "
            "No hardcoded fallback is provided (long-term solution)."
        )

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    min_distance: float | None = None
    min_distance_source = "explicit"
    if min_distance_km is not None:
        min_distance = float(min_distance_km)
    else:
        # Thesis policy default comes from config, not from implicit recomputation.
        min_distance_source = "config"
        try:
            import yaml

            cfg_path = ROOT / "config" / "pipeline_config.yaml"
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            cfg_dist = cfg.get("selection", {}).get("min_distance_km")
            if cfg_dist is not None:
                min_distance = float(cfg_dist)
            else:
                min_distance_source = "data-driven fallback"
        except Exception:
            min_distance_source = "data-driven fallback"

    if min_distance is None:
        min_distance = compute_min_distance_km(str(metadata_path))

    cfg: dict = {}
    cfg_path = ROOT / "config" / "pipeline_config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if n_clusters is None:
        cfg_clusters = cfg.get("clustering", {}).get("n_clusters")
        if cfg_clusters is None:
            raise ValueError(
                "n_clusters unresolved for exploration. Set explicit n_clusters or "
                "provide clustering.n_clusters in config/pipeline_config.yaml."
            )
        n_clusters = int(cfg_clusters)

    if batch_size is None:
        cfg_batch = cfg.get("feature_extraction", {}).get("batch_size")
        if cfg_batch is None:
            cfg_batch = cfg.get("data", {}).get("batch_size")
        if cfg_batch is None:
            raise ValueError(
                "batch_size unresolved for exploration. Set explicit batch_size or "
                "provide feature_extraction.batch_size in config/pipeline_config.yaml."
            )
        batch_size = int(cfg_batch)

    if objective_authority is None:
        objective_authority = (
            str(
                cfg.get("selection", {}).get(
                    "objective_authority", "unified_normalized"
                )
            )
            .strip()
            .lower()
        )

    # Resolve selection target size (separate from LHS point count).
    selection_target, selection_target_source = resolve_selection_n_samples(
        selection_n_samples,
        context="tune_weights.run_exploration",
        root=ROOT,
        experiment_run_dir=exp_dir,
    )

    runner = ExperimentRunner(
        output_dir=str(output_dir),
        # Share feature cache across runs to avoid repeated heavy extraction.
        feature_cache_dir=ROOT / "outputs",
    )

    # Generate weight combinations
    weight_combinations = generate_weights(
        n_points=n_samples, seed=seed, sampler=sampler
    )

    print("\n" + "=" * 70)
    print("PHASE 1: EXPLORATION (LHS SWEEP)")
    print("=" * 70)
    print(
        f"Weight combinations: {len(weight_combinations)} ({sampler.upper()}-samples)"
    )
    print(f"Selection target: {selection_target} samples ({selection_target_source})")
    print(f"Min Distance Constraint: {min_distance} km ({min_distance_source})")
    print(f"n_clusters: {n_clusters}")
    print(f"batch_size: {batch_size}")
    print(f"objective_authority: {objective_authority}")
    print(f"Seed: {seed}")
    print("=" * 70 + "\n")

    # Run sweep
    results = runner.run_weight_sweep(
        csv_meta=str(metadata_path),
        n_samples=selection_target,
        weight_combinations=weight_combinations,
        n_clusters=int(n_clusters),
        batch_size=int(batch_size),
        min_distance_km=min_distance,
        objective_authority=objective_authority,
        patience=None,
        pre_selected=pre_indices,
        pre_selected_names=pre_names,
    )

    # Compute Pareto front
    print("\n" + "=" * 70)
    print("COMPUTING PARETO-FRONT (Exploration Phase)...")
    print("=" * 70)

    pareto_front = compute_pareto_front(results)
    print(
        f"✅ Pareto-Front: {len(pareto_front)} of {len(results)} "
        "solutions are Pareto-optimal\n"
    )

    # Visualizations
    if em is not None:
        viz_dir = em.get_path("artifacts") / "pareto"
        viz_dir.mkdir(parents=True, exist_ok=True)
    else:
        viz_dir = output_dir / "pareto"
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating visualizations in {viz_dir}...")
    visualize_pareto_front(results, pareto_front, output_dir=str(viz_dir))

    # Export report
    report_path = viz_dir / "pareto_solutions.csv"
    export_pareto_report(pareto_front, output_path=str(report_path))

    # Save to ExperimentManager
    if em is not None:
        try:
            import pandas as pd

            em.save_results("pareto_solutions", pd.read_csv(report_path), format="csv")
            em.mark_stage_complete(
                "exploration",
                summary={
                    "pareto_count": len(pareto_front),
                    "lhs_points": n_samples,
                    "selection_n_samples": selection_target,
                },
            )
        except Exception as e:
            print(f"Warning: could not save to experiment manager: {e}")

    print("\n✅ Phase 1 COMPLETE")
    print(f"📊 Plots: {viz_dir}")
    print(f"📋 CSV:   {report_path}")

    return pareto_front, report_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for exploration sweep."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Exploration with LHS/Sobol sweep"
    )
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument(
        "--selection-n-samples",
        type=int,
        default=None,
        help=(
            "Selection target per run. Resolution: explicit > config "
            "selection.n_samples > autoscale artifact > fail-fast."
        ),
    )
    parser.add_argument(
        "--sampler", choices=["lhs", "sobol"], required=False, default=None
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective-authority",
        choices=["unified_normalized", "legacy_lexicographic"],
        default=None,
        help="Selection objective authority used for best-combination ranking",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        required=False,
        help="Path to metadata CSV (required)",
    )
    parser.add_argument("--pre-names", type=str, nargs="*", default=None)
    parser.add_argument("--pre-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--min-distance-km",
        type=float,
        required=False,
        default=None,
        help="Optional min distance policy override in km",
    )

    args = parser.parse_args(argv)

    # metadata_path defaults to DATA_META if not provided
    metadata_path = args.metadata_path or DATA_META

    try:
        run_exploration(
            n_samples=args.n_samples,
            selection_n_samples=args.selection_n_samples,
            sampler=args.sampler,
            objective_authority=args.objective_authority,
            seed=args.seed,
            metadata_path=metadata_path,
            min_distance_km=args.min_distance_km,
            pre_names=args.pre_names,
            pre_indices=args.pre_indices,
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
