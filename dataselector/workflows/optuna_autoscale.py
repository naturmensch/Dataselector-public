#!/usr/bin/env python3
"""
Optuna Autoscale Workflow — Staged Progressive Refinement with Early Stopping

Performs multi-stage Optuna optimization with adaptive bounds narrowing:
- Progressively increases n_samples (e.g., 50 → 100 → 300 → full)
- Narrows search ranges around best parameters after each stage
- Stops early if parameters converge (delta < tolerance) for N consecutive stages

Migration from: scripts/optuna_autoscale.py
Author: Phase 5 Migration
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dataselector.cli_decorators import cli_command
from dataselector.data.spatial_schema import (
    normalize_spatial_schema,
)
from dataselector.data.spatial_schema import spatial_spread as compute_spatial_spread
from dataselector.workflows.objective_scoring import (
    compute_baselines,
    normalized_objective,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("outputs")


def _select_best_production_trial(study):
    """Return best non-diagnostic trial, fallback to study.best_trial."""
    production_trials = [
        t
        for t in study.trials
        if t.value is not None and not bool(t.user_attrs.get("full_coverage_mode", False))
    ]
    if production_trials:
        return max(production_trials, key=lambda t: float(t.value)), True
    return study.best_trial, False


def load_or_create_data(
    out_dir: Path,
    n: int | None = None,
    dim: int = 256,
    seed: int = 123,
    require_metadata: bool = False,
    config_path: str | None = None,
    cache_mode: str = "read_write",
    strict_real_data: bool = False,
):
    """Load features and metadata, or create synthetic data for testing."""
    from dataselector.data.io import load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata

    features_path = out_dir / "features.npy"
    metadata_path = assert_canonical_metadata(
        None,
        context="optuna-autoscale",
    )

    if require_metadata:
        if not metadata_path.exists():
            raise FileNotFoundError(
                "optuna-autoscale requires canonical metadata file at "
                f"'{metadata_path}'."
            )
        features = load_or_extract_features(
            out_dir=out_dir,
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=True,
            cache_mode=cache_mode,
            config_path=config_path,
            enforce_canonical=True,
        )
        from dataselector.data.io import load_metadata

        metadata = load_metadata(str(metadata_path))
    elif features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=out_dir,
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=True,
            cache_mode=cache_mode,
            config_path=config_path,
            enforce_canonical=True,
        )
        from dataselector.data.io import load_metadata

        metadata = load_metadata(str(metadata_path))
    else:
        if strict_real_data:
            raise FileNotFoundError(
                "Strict real-data mode forbids synthetic fallback in autoscale."
            )
        rng = np.random.RandomState(seed)
        if n is None:
            n = 673
        features = rng.randn(n, dim).astype("float32")
        center_x = rng.uniform(450000, 650000, n)
        center_y = rng.uniform(5800000, 6100000, n)
        half_w = rng.uniform(40, 80, n)
        half_h = rng.uniform(40, 80, n)
        metadata = pd.DataFrame(
            {
                "ul_x": center_x - half_w,
                "ul_y": center_y + half_h,
                "lr_x": center_x + half_w,
                "lr_y": center_y - half_h,
                "year": rng.randint(1880, 1945, n),
            }
        )

    return features, metadata


def clamp(v, lo, hi):
    """Clamp value between bounds."""
    return max(lo, min(hi, v))


def make_objective(
    features,
    metadata,
    n_samples,
    min_distance_bounds,
    pre_selected_names=None,
    pre_selected_indices=None,
    score_weights: tuple[float, float] = (0.5, 0.5),
    infeasible_penalty: float = 0.1,
):
    """Create Optuna objective function for given stage."""
    baseline_diversity, baseline_spread = compute_baselines(
        features=features,
        metadata=metadata,
        metric="euclidean",
    )

    def objective(trial):
        # Lazy import to avoid module-level side effects

        from dataselector.selection.diversity_selector import DiversitySelector

        a = trial.suggest_float("a", *min_distance_bounds["a"])
        b = trial.suggest_float("b", *min_distance_bounds["b"])
        c = trial.suggest_float("c", *min_distance_bounds["c"])
        total = a + b + c
        alpha = a / total
        beta = b / total
        gamma = c / total

        full_coverage_mode = int(n_samples) >= int(len(features))
        if full_coverage_mode:
            # In full-coverage stages, any positive min_distance can make the target
            # cardinality infeasible by construction. Treat as diagnostic stage and
            # disable spatial distance enforcement in the objective.
            min_dist = 0
        else:
            min_dist = trial.suggest_int(
                "min_distance_km", *min_distance_bounds["min_distance_km"]
            )

        selector = DiversitySelector(n_samples=n_samples, use_multi_criteria=True)
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_dist,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
            pre_selected=pre_selected_indices,
            pre_selected_names=pre_selected_names,
        )

        n_selected = len(selected)
        if n_selected == 0:
            trial.set_user_attr("infeasible", True)
            trial.set_user_attr("feasibility_ratio", 0.0)
            return 0.0

        diversity = selector._calculate_diversity_score(features[selected])
        spatial_meta = normalize_spatial_schema(
            metadata, require_bounds=True, copy=True
        )
        spread = compute_spatial_spread(spatial_meta, selected)
        objective_score = normalized_objective(
            diversity=float(diversity),
            spread=float(spread),
            baseline_diversity=baseline_diversity,
            baseline_spread=baseline_spread,
            n_selected=int(n_selected),
            target_n=int(n_samples),
            weight_diversity=float(score_weights[0]),
            weight_spread=float(score_weights[1]),
            infeasible_penalty=float(infeasible_penalty),
        )

        trial.set_user_attr("alpha", float(alpha))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("gamma", float(gamma))
        trial.set_user_attr("min_distance_km", int(min_dist))
        trial.set_user_attr("n_selected", int(n_selected))
        trial.set_user_attr("selection_backend", str(selector.selection_backend))
        trial.set_user_attr("diversity", float(diversity))
        trial.set_user_attr("spatial_spread", float(spread))
        trial.set_user_attr("diversity_norm", float(objective_score.diversity_norm))
        trial.set_user_attr("spatial_spread_norm", float(objective_score.spread_norm))
        trial.set_user_attr("objective_score_raw", float(objective_score.raw_score))
        trial.set_user_attr("infeasible", bool(objective_score.infeasible))
        trial.set_user_attr("feasibility_ratio", float(objective_score.feasibility_ratio))
        trial.set_user_attr("n_samples", int(n_samples))
        trial.set_user_attr("full_coverage_mode", bool(full_coverage_mode))
        trial.set_user_attr("diagnostic_only", bool(full_coverage_mode))

        return float(objective_score.score)

    return objective


def run_autoscale(
    n_trials_per_stage: list[int],
    stages_samples: list[int],
    features: np.ndarray,
    metadata: pd.DataFrame,
    patience: int = 2,
    tol: dict | None = None,
    pre_selected_names: list | None = None,
    pre_selected_indices: np.ndarray | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Run multi-stage autoscale optimization.

    Parameters
    ----------
    n_trials_per_stage : list[int]
        Trials per stage
    stages_samples : list[int]
        n_samples for each stage
    features : np.ndarray
        Feature embeddings
    metadata : pd.DataFrame
        Tile metadata
    patience : int
        Stop if converged for N consecutive stages
    tol : dict | None
        Convergence tolerances
    pre_selected_names : list | None
        Pre-selected tile names
    pre_selected_indices : np.ndarray | None
        Pre-selected indices
    out_dir : Path | None
        Output directory

    Returns
    -------
    tuple[Path, Path]
        (summary_csv, best_json)
    """
    import optuna

    if out_dir is None:
        out_dir = OUT
    out_dir.mkdir(exist_ok=True)

    # Initialize global search bounds
    bounds = {
        "a": (0.01, 1.0),
        "b": (0.01, 1.0),
        "c": (0.01, 1.0),
        "min_distance_km": (0, 60),
    }

    study = optuna.create_study(direction="maximize")

    history = []
    best_prev = None
    no_change = 0

    for stage_idx, n_samples in enumerate(stages_samples):
        full_coverage_stage = int(n_samples) >= int(len(features))
        print(
            f"\n=== Stage {stage_idx+1}/{len(stages_samples)}: n_samples={n_samples} | trials={n_trials_per_stage[stage_idx]} ==="
        )
        if full_coverage_stage:
            print(
                "Stage uses full coverage mode (n_samples == candidate count): "
                "forcing min_distance_km=0 for objective feasibility."
            )
        objective = make_objective(
            features,
            metadata,
            n_samples,
            bounds,
            pre_selected_names=pre_selected_names,
            pre_selected_indices=pre_selected_indices,
        )

        start_trial_count = len(study.trials)
        study.optimize(objective, n_trials=n_trials_per_stage[stage_idx])
        end_trial_count = len(study.trials)

        stage_trials = study.trials[start_trial_count:end_trial_count]

        # Plot per-stage diagnostics
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            date = datetime.now().strftime("%Y%m%d")

            # Objective history
            values = [t.value for t in stage_trials if t.value is not None]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(len(values)), values, marker="o")
                ax.set_xlabel("trial (stage)")
                ax.set_ylabel("value")
                ax.set_title(f"Stage {stage_idx+1} objective")
                out = (
                    out_dir / f"optuna_autoscale_stage{stage_idx+1}_history_{date}.png"
                )
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)

            # min_distance distribution
            md_vals = [
                t.user_attrs.get("min_distance_km")
                for t in stage_trials
                if "min_distance_km" in t.user_attrs
            ]
            if md_vals:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(md_vals, bins=10, kde=False, ax=ax)
                ax.set_title(f"Stage {stage_idx+1} min_distance distribution")
                out = (
                    out_dir
                    / f"optuna_autoscale_stage{stage_idx+1}_min_distance_{date}.png"
                )
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)

            # alpha vs beta scatter
            alf = [
                t.user_attrs.get("alpha")
                for t in stage_trials
                if "alpha" in t.user_attrs
            ]
            bet = [
                t.user_attrs.get("beta") for t in stage_trials if "beta" in t.user_attrs
            ]
            if alf and bet:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=alf, y=bet, ax=ax)
                ax.set_xlabel("alpha")
                ax.set_ylabel("beta")
                ax.set_title(f"Stage {stage_idx+1} alpha vs beta")
                out = (
                    out_dir
                    / f"optuna_autoscale_stage{stage_idx+1}_alpha_beta_{date}.png"
                )
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)

            print(f"Wrote stage plots for stage {stage_idx+1}")
        except Exception as e:
            print("Warning: could not create stage plots:", e)

        best = study.best_trial
        best_params = {
            "value": best.value,
            "alpha": best.user_attrs.get("alpha"),
            "beta": best.user_attrs.get("beta"),
            "gamma": best.user_attrs.get("gamma"),
            "min_distance_km": best.user_attrs.get("min_distance_km"),
        }

        print("Stage best:", best_params)

        hist = {
            "stage": stage_idx,
            "n_samples": n_samples,
            "trials": n_trials_per_stage[stage_idx],
            **best_params,
        }
        history.append(hist)

        # Check convergence
        if best_prev is not None:
            dalpha = abs(best_prev["alpha"] - best_params["alpha"])
            dbeta = abs(best_prev["beta"] - best_params["beta"])
            dmin = abs(best_prev["min_distance_km"] - best_params["min_distance_km"])
            dval = abs(best_prev["value"] - best_params["value"])

            print(
                f"deltas: dalpha={dalpha:.4f}, dbeta={dbeta:.4f}, dmin={dmin:.2f}, dval={dval:.4f}"
            )

            tol = tol or {
                "alpha": 0.02,
                "beta": 0.02,
                "min_distance_km": 1.0,
                "value": 1e-3,
            }

            if (
                dalpha < tol["alpha"]
                and dbeta < tol["beta"]
                and dmin < tol["min_distance_km"]
            ):
                no_change += 1
                print(f"No significant change detected ({no_change}/{patience})")
            else:
                no_change = 0

            if no_change >= patience:
                print("Converged: stopping auto-scale optimization")
                break

        best_prev = best_params

        # Narrow bounds around best for next stage
        alpha = best_params["alpha"]
        beta = best_params["beta"]
        gamma = best_params["gamma"]

        for key, val in [("a", alpha), ("b", beta), ("c", gamma)]:
            lo = clamp(val - 0.2, 0.01, 1.0)
            hi = clamp(val + 0.2, 0.01, 1.0)
            bounds[key] = (lo, hi)

        if full_coverage_stage:
            print(
                "Skipping min_distance bounds update from full-coverage stage "
                "(diagnostic stage)."
            )
        else:
            md = best_params["min_distance_km"]
            if md is None:
                md = 28
            bounds["min_distance_km"] = (max(0, int(md - 10)), int(md + 10))

        print("New bounds for next stage:", bounds)

    # Persist history
    date = datetime.now().strftime("%Y%m%d")
    summary_df = pd.DataFrame(history)
    summary_file = out_dir / f"optuna_autoscale_summary_{date}.csv"
    summary_df.to_csv(summary_file, index=False)

    # Create markdown report
    report_md = out_dir / f"optuna_autoscale_report_{date}.md"
    lines = [f"# Optuna Auto-scale Report ({date})", "", "## Stages summary", ""]
    try:
        lines.append(summary_df.to_markdown(index=False))
    except Exception:
        lines.append(
            f"(could not render markdown table; saved CSV summary: {summary_file.name})"
        )

    lines.append("")
    lines.append("## Stage plots")
    for f in sorted(out_dir.glob(f"optuna_autoscale_stage*_{date}.png")):
        lines.append(f"- {f.name}")

    report_md.write_text("\n".join(lines))
    print("Report written to", report_md)

    # Save best trial from production stages only (exclude diagnostic full-coverage).
    best_overall, best_from_production = _select_best_production_trial(study)

    out_best = out_dir / f"optuna_autoscale_best_{date}.json"
    with out_best.open("w") as fh:
        json.dump(
            {
                "value": best_overall.value,
                "params": best_overall.params,
                "user_attrs": best_overall.user_attrs,
                "best_from_production_stage": bool(best_from_production),
            },
            fh,
            indent=2,
        )

    # Latest best file
    latest = out_dir / "optuna_autoscale_best_latest.json"
    with latest.open("w") as fh:
        json.dump(
            {
                "value": best_overall.value,
                "params": best_overall.params,
                "user_attrs": best_overall.user_attrs,
                "best_from_production_stage": bool(best_from_production),
            },
            fh,
            indent=2,
        )

    # Extract n_samples
    best_n_samples = best_overall.user_attrs.get("n_samples")
    sel_file = out_dir / "optuna_autoscale_selected_n_samples.txt"
    if best_n_samples is not None:
        sel_file.write_text(str(int(best_n_samples)))
        print(f"BEST_N_SAMPLES: {int(best_n_samples)}")
    else:
        try:
            hist_df = pd.read_csv(summary_file)
            idx = hist_df["value"].idxmax()
            inferred = int(hist_df.loc[idx, "n_samples"])
            sel_file.write_text(str(inferred))
            print(f"BEST_N_SAMPLES (inferred): {inferred}")
        except Exception:
            print("WARNING: could not determine best n_samples")

    # Save study
    try:
        import joblib

        joblib.dump(study, out_dir / f"optuna_autoscale_study_{date}.pkl")
    except Exception:
        print("joblib not available; not saving study object")

    print("Auto-scale optimization finished. Summary:", summary_file)
    return summary_file, out_best


def run_optuna_autoscale_workflow(
    *,
    n_trials: list[int] | None = None,
    stages: list[str] | None = None,
    n_candidates: int | None = None,
    dim: int = 256,
    seed: int = 42,
    patience: int = 2,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
    output_dir: str = "outputs",
    config_path: str | None = None,
    cache_mode: str = "read_write",
    strict_real_data: bool = True,
) -> int:
    """Run autoscale workflow and persist artifacts under output_dir."""
    trials = n_trials or [20, 40, 80, 160]
    stage_values = stages or ["50", "100", "300", "full"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, metadata = load_or_create_data(
        out_dir=out_dir,
        n=n_candidates,
        dim=dim,
        seed=seed,
        require_metadata=True,
        config_path=config_path,
        cache_mode=cache_mode,
        strict_real_data=strict_real_data,
    )

    actual_n = len(features)
    parsed_stages: list[int] = []
    for stage in stage_values:
        if stage == "full":
            parsed_stages.append(actual_n)
        else:
            parsed_stages.append(int(stage))

    if len(trials) != len(parsed_stages):
        if len(trials) == 1:
            n_trials_per_stage = [trials[0]] * len(parsed_stages)
        else:
            raise SystemExit("Provide n-trials per stage or a single value.")
    else:
        n_trials_per_stage = trials

    run_autoscale(
        n_trials_per_stage,
        parsed_stages,
        features,
        metadata,
        patience=patience,
        pre_selected_names=pre_names,
        pre_selected_indices=np.array(pre_indices) if pre_indices else None,
        out_dir=out_dir,
    )

    return 0


@cli_command(
    "optuna-autoscale",
    help="Run staged Optuna autoscale and write compute artifacts",
    args={
        "n_trials": {
            "type": int,
            "nargs": "+",
            "default": [20, 40, 80, 160],
            "help": "Trials per stage, or one value reused for all stages",
        },
        "stages": {
            "type": str,
            "nargs": "+",
            "default": ["50", "100", "300", "full"],
            "help": "Stage sample counts (e.g. 50 100 300 full)",
        },
        "n_candidates": {
            "type": int,
            "default": None,
            "help": "Candidate count override",
        },
        "dim": {"type": int, "default": 256, "help": "Feature dimension"},
        "seed": {"type": int, "default": 42, "help": "Random seed"},
        "patience": {"type": int, "default": 2, "help": "Early stop patience"},
        "pre_names": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Optional pre-selected tile names",
        },
        "pre_indices": {
            "type": int,
            "nargs": "*",
            "default": None,
            "help": "Optional pre-selected tile indices",
        },
        "output_dir": {
            "type": str,
            "default": "outputs",
            "help": "Output directory for autoscale artifacts",
        },
        "config_path": {
            "type": str,
            "default": "config/pipeline_config.yaml",
            "help": "Feature config path propagated to extraction",
        },
        "cache_mode": {
            "type": str,
            "default": "read_write",
            "choices": ["off", "read_only", "write_only", "read_write"],
        },
        "strict_real_data": {"type": bool, "default": True},
    },
)
def cli_optuna_autoscale(
    n_trials: list[int] | None = None,
    stages: list[str] | None = None,
    n_candidates: int | None = None,
    dim: int = 256,
    seed: int = 42,
    patience: int = 2,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
    output_dir: str = "outputs",
    config_path: str = "config/pipeline_config.yaml",
    cache_mode: str = "read_write",
    strict_real_data: bool = True,
) -> int:
    return run_optuna_autoscale_workflow(
        n_trials=n_trials,
        stages=stages,
        n_candidates=n_candidates,
        dim=dim,
        seed=seed,
        patience=patience,
        pre_names=pre_names,
        pre_indices=pre_indices,
        output_dir=output_dir,
        config_path=config_path,
        cache_mode=cache_mode,
        strict_real_data=strict_real_data,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Optuna autoscale."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, nargs="+", default=[20, 40, 80, 160])
    parser.add_argument("--stages", nargs="+", default=["50", "100", "300", "full"])
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--pre-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional pre-selected tile names",
    )
    parser.add_argument(
        "--pre-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional pre-selected tile indices",
    )

    args = parser.parse_args(argv)

    features, metadata = load_or_create_data(
        out_dir=OUT,
        n=args.n_candidates,
        dim=args.dim,
        seed=args.seed,
        require_metadata=True,
    )

    actual_n = len(features)

    stages = []
    for s in args.stages:
        if s == "full":
            stages.append(actual_n)
        else:
            stages.append(int(s))

    if len(args.n_trials) != len(stages):
        if len(args.n_trials) == 1:
            n_trials_per_stage = [args.n_trials[0]] * len(stages)
        else:
            raise SystemExit("Provide n-trials per stage or a single value.")
    else:
        n_trials_per_stage = args.n_trials

    run_autoscale(
        n_trials_per_stage,
        stages,
        features,
        metadata,
        patience=args.patience,
        pre_selected_names=args.pre_names,
        pre_selected_indices=args.pre_indices,
        out_dir=OUT,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
