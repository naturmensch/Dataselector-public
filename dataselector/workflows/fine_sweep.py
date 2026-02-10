#!/usr/bin/env python3
"""
Fine Sweep Workflow — Dense Grid Search Around Pareto Region

Phase 2: Fine-grained parameter sweep
Tests combinations of weights and minimum distances to refine Pareto front.

Migration from: scripts/run_fine_sweep.py
Author: Phase 5 Migration
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_META = ROOT / "data" / "new_all_tiles.csv"


def default_weight_combinations():
    """Return default fine-grid weight combinations."""
    import itertools

    alphas = [0.55, 0.60, 0.65, 0.70, 0.75]
    betas = [0.05, 0.10, 0.15, 0.20]

    weight_combos = []
    for a, b in itertools.product(alphas, betas):
        g = round(1.0 - a - b, 3)
        if g > 0.01:
            weight_combos.append((a, b, g))
    return weight_combos


def run_fine_sweep(
    min_distances: list[float] | None = None,
    weight_combos: list | None = None,
    max_runs: int | None = None,
    pre_names: list | None = None,
    pre_indices: list | None = None,
    output_dir: Path | None = None,
) -> tuple[object, list]:
    """
    Run fine-grained parameter sweep.

    Parameters
    ----------
    min_distances : list[float] | None
        List of min_distance values to test
    weight_combos : list | None
        Weight combinations (default: fine grid)
    max_runs : int | None
        Limit runs per distance
    pre_names : list | None
        Pre-selected tile names
    pre_indices : list | None
        Pre-selected indices
    output_dir : Path | None
        Output directory

    Returns
    -------
    tuple[pd.DataFrame, list]
        (full_results_df, pareto_front)
    """
    import os

    import pandas as pd
    import yaml

    from dataselector.pipeline.experiments import ExperimentRunner
    from dataselector.selection.pareto import (
        compute_pareto_front,
        export_pareto_report,
        visualize_pareto_front,
    )

    if output_dir is None:
        output_dir = ROOT / "outputs" / "fine_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    if weight_combos is None:
        weight_combos = default_weight_combinations()

    if min_distances is None:
        min_distances = [30.0, 35.0, 40.0, 45.0, 50.0]

    print(
        f"Fine grid size: {len(weight_combos)} weight combos × {len(min_distances)} distances = {len(weight_combos)*len(min_distances)} runs"
    )

    # Load config
    cfg = yaml.safe_load(open(ROOT / "config" / "pipeline_config.yaml"))
    n_clusters_cfg = cfg.get("clustering", {}).get("n_clusters", 8)
    n_samples_cfg = cfg.get("selection", {}).get("n_samples")
    batch_size_cfg = cfg.get("feature_extraction", {}).get("batch_size", 8)

    # Attach to pipeline
    em = None
    exp_dir = os.environ.get("EXPERIMENT_RUN_DIR")
    if exp_dir:
        from dataselector.pipeline.experiment_manager import ExperimentManager

        em = ExperimentManager.from_existing(exp_dir)
        output_dir = em.get_path("results") / "fine_sweep"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try to infer n_samples from upstream configs
        try:
            run_cfg_path = em.get_path("config") / "config_run.yaml"
            if run_cfg_path.exists():
                run_cfg = yaml.safe_load(open(run_cfg_path)) or {}
                n_lhs = run_cfg.get("n_lhs")
                if n_lhs:
                    n_samples_cfg = int(n_lhs)
                    print(f"ℹ️ Using n_samples from run config (n_lhs): {n_samples_cfg}")

            if n_samples_cfg is None:
                expl_cfg_path = em.get_path("config") / "config_exploration.yaml"
                if expl_cfg_path.exists():
                    expl_cfg = yaml.safe_load(open(expl_cfg_path)) or {}
                    n_expl = expl_cfg.get("n_samples")
                    if n_expl:
                        n_samples_cfg = int(n_expl)
                        print(
                            f"ℹ️ Using n_samples from exploration config: {n_samples_cfg}"
                        )
        except Exception as e:
            print(f"⚠️ Error reading run configs: {e}")

        # Adaptive fallback
        if n_samples_cfg is None:
            try:
                data_path = Path(DATA_META)
                n_tiles = None
                if data_path.exists():
                    try:
                        n_tiles = len(pd.read_csv(data_path))
                    except Exception:
                        pass
                from dataselector.pipeline.pipeline_utils import (
                    compute_adaptive_n_initial,
                )

                n_samples_cfg = compute_adaptive_n_initial(
                    n_dimensions=3, n_tiles=n_tiles, strategy="modern"
                )
                print(f"⚠ Using adaptive n_samples={n_samples_cfg}")
            except Exception:
                n_samples_cfg = 34
                print(f"⚠ Fallback n_samples={n_samples_cfg}")

    runner = ExperimentRunner(output_dir=str(output_dir / "runs"))

    all_results = []
    feasibility_summary = []

    for min_dist in min_distances:
        print(f"\n--- Fine sweep for min_distance = {min_dist} km ---")
        df = runner.run_weight_sweep(
            csv_meta=str(DATA_META),
            n_samples=n_samples_cfg,
            weight_combinations=weight_combos,
            n_clusters=n_clusters_cfg,
            batch_size=batch_size_cfg,
            min_distance_km=min_dist,
            patience=None,
            max_runs=max_runs,
            pre_selected=pre_indices,
            pre_selected_names=pre_names,
        )
        df["min_distance_km"] = min_dist
        all_results.append(df)

        # Feasibility check
        infeasible_mask = df["n_selected"] < (0.9 * n_samples_cfg)
        infeasible_count = int(infeasible_mask.sum())
        total_runs = len(df)
        median_selected = int(df["n_selected"].median())

        feasibility_summary.append(
            {
                "min_distance_km": min_dist,
                "total_runs": total_runs,
                "infeasible_count": infeasible_count,
                "infeasible_pct": (
                    infeasible_count / total_runs * 100.0 if total_runs > 0 else 0.0
                ),
                "median_n_selected": median_selected,
            }
        )

    full_df = pd.concat(all_results, ignore_index=True)
    full_csv = output_dir / "fine_sweep_results.csv"
    full_df.to_csv(full_csv, index=False)
    print(f"Fine sweep results saved: {full_csv}")

    # Save to ExperimentManager
    if em is not None:
        try:
            em.save_results("fine_sweep", full_df, format="csv")
            em.mark_stage_complete("fine_sweep", summary={"n_runs": len(full_df)})
        except Exception as e:
            print(f"Warning: could not save to experiment manager: {e}")

    # Feasibility summary
    fs_df = pd.DataFrame(feasibility_summary)
    fs_path = output_dir / "feasibility_summary.csv"
    fs_df.to_csv(fs_path, index=False)
    print(f"Feasibility summary written: {fs_path}")

    # Pareto front (feasible only)
    feasible_mask = full_df["n_selected"] >= (0.9 * n_samples_cfg)
    n_infeasible = (~feasible_mask).sum()
    if n_infeasible > 0:
        print(f"Info: {n_infeasible} infeasible runs removed from Pareto computation.")

    feasible_df = full_df[feasible_mask].reset_index(drop=True)
    pareto_front = compute_pareto_front(feasible_df)

    export_pareto_report(
        pareto_front, output_path=str(output_dir / "pareto_solutions.csv")
    )
    feasible_df.to_csv(output_dir / "fine_sweep_results_feasible.csv", index=False)
    visualize_pareto_front(
        feasible_df, pareto_front, output_dir=str(output_dir / "plots")
    )

    print("Fine sweep + Pareto finished")

    return full_df, pareto_front


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for fine sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-distances",
        type=str,
        default=None,
        help="Comma-separated list of min distances",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--pre-names", type=str, nargs="*", default=None)
    parser.add_argument("--pre-indices", type=int, nargs="*", default=None)

    args = parser.parse_args(argv)

    min_distances = None
    if args.min_distances:
        min_distances = [float(x) for x in args.min_distances.split(",")]

    try:
        run_fine_sweep(
            min_distances=min_distances,
            max_runs=args.max_runs,
            pre_names=args.pre_names,
            pre_indices=args.pre_indices,
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
