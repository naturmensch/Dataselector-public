#!/usr/bin/env python3
"""
Optuna Optimization Workflow — Hyperparameter Tuning for Multi-Criteria Selection

Performs Optuna-based hyperparameter optimization for:
- Visual diversity weight (alpha)
- Spatial diversity weight (beta)
- Temporal diversity weight (gamma)
- Minimum distance constraint
- Number of samples (optional range)

Migration from: scripts/optuna_optimize.py
Author: Phase 5 Migration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import optuna


def get_optuna_sampler(sampler_name: str = "tpe", seed: int = 42, **sampler_kwargs):
    """
    Factory function to get an Optuna sampler by name.

    Parameters
    ----------
    sampler_name : str
        Sampler type: 'tpe', 'cmaes', 'qmc', 'qmc-sobol', 'qmc-halton'
    seed : int
        Random seed
    **sampler_kwargs
        Additional sampler-specific kwargs

    Returns
    -------
    optuna.samplers.BaseSampler
        Configured Optuna sampler
    """
    import optuna

    sampler_name = sampler_name.lower() if sampler_name else "tpe"

    if sampler_name == "qmc" or sampler_name.startswith("qmc"):
        qmc_type = "sobol"
        if "-" in sampler_name:
            qmc_type = sampler_name.split("-", 1)[1]

        # Try with qmc_type first
        try:
            return optuna.samplers.QMCSampler(
                seed=seed, qmc_type=qmc_type, **sampler_kwargs
            )
        except TypeError:
            # Fallback: try without qmc_type if older API
            try:
                return optuna.samplers.QMCSampler(seed=seed, **sampler_kwargs)
            except TypeError:
                # If QMC fails completely, fall back to TPE
                return optuna.samplers.TPESampler(seed=seed, **sampler_kwargs)
    elif sampler_name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, **sampler_kwargs)
    else:
        # Default to TPE for 'tpe' or unknown
        return optuna.samplers.TPESampler(seed=seed, **sampler_kwargs)


def load_or_create_data(
    out_dir: Path,
    n: int = 500,
    dim: int = 512,
    seed: int = 123,
    require_metadata: bool = False,
    metadata_path: str | Path | None = None,
    feature_cache_dir: Path | None = None,
):
    """
    Load features and metadata, or create synthetic data for testing.

    Parameters
    ----------
    out_dir : Path
        Output directory
    n : int
        Number of samples
    dim : int
        Feature dimensionality
    seed : int
        Random seed

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Features array and metadata DataFrame
    """
    import numpy as np
    import pandas as pd

    from dataselector.data.io import load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata

    features_path = out_dir / "features.npy"
    cache_dir = feature_cache_dir if feature_cache_dir is not None else out_dir
    metadata_path = assert_canonical_metadata(
        metadata_path,
        context="optuna-optimize",
    )

    if require_metadata:
        if not metadata_path.exists():
            raise FileNotFoundError(
                "optuna-optimize requires canonical metadata file at "
                f"'{metadata_path}'."
            )
        features = load_or_extract_features(
            out_dir=cache_dir,
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=True,
            enforce_canonical=True,
        )
        from dataselector.data.io import load_metadata

        metadata = load_metadata(str(metadata_path))
    elif features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=cache_dir,
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=True,
            enforce_canonical=True,
        )
        from dataselector.data.io import load_metadata

        metadata = load_metadata(str(metadata_path))
    else:
        rng = np.random.RandomState(seed)
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


def objective_factory(
    features,
    metadata,
    n_samples: int,
    min_distance_bounds: tuple[int, int] | None = None,
    n_samples_range: tuple[int, int] | None = None,
    constrain_bounds: dict[str, float] | None = None,
    pre_selected_names: list[str] | None = None,
    pre_selected_indices: list[int] | None = None,
):
    """
    Create Optuna objective function for hyperparameter optimization.

    Parameters
    ----------
    features : np.ndarray
        Feature embeddings
    metadata : pd.DataFrame
        Tile metadata
    n_samples : int
        Fixed number of samples (if n_samples_range is None)
    min_distance_bounds : tuple[int, int] | None
        (min, max) bounds for min_distance_km optimization
    n_samples_range : tuple[int, int] | None
        Optional range for n_samples optimization
    constrain_bounds : dict | None
        Optional bounds constraints for parameters

    Returns
    -------
    callable
        Optuna objective function
    """

    baseline_diversity, baseline_spread = compute_baselines(
        features=features,
        metadata=metadata,
        metric="euclidean",
    )

    # Lazy import to avoid sklearn dependency at module import time
    def objective(trial: "optuna.trial.Trial"):
        try:
            from dataselector.selection.diversity_selector import DiversitySelector

            # Allow Optuna to explore n_samples if a range is provided
            n_samp = (
                trial.suggest_int("n_samples", n_samples_range[0], n_samples_range[1])
                if n_samples_range is not None
                else n_samples
            )

            # Determine bounds for a, b, c
            if constrain_bounds:
                a_bounds = (
                    constrain_bounds.get("a_min", 0.01),
                    constrain_bounds.get("a_max", 1.0),
                )
                b_bounds = (
                    constrain_bounds.get("b_min", 0.01),
                    constrain_bounds.get("b_max", 1.0),
                )
                c_bounds = (
                    constrain_bounds.get("c_min", 0.01),
                    constrain_bounds.get("c_max", 1.0),
                )
                min_dist_bnds = (
                    constrain_bounds.get("min_dist_min", 0),
                    constrain_bounds.get("min_dist_max", 60),
                )
            else:
                a_bounds = (0.01, 1.0)
                b_bounds = (0.01, 1.0)
                c_bounds = (0.01, 1.0)
                min_dist_bnds = min_distance_bounds if min_distance_bounds else (0, 60)

            # Sample raw weights and normalize (ensures sum=1 and non-negative)
            a = trial.suggest_float("a", *a_bounds)
            b = trial.suggest_float("b", *b_bounds)
            c = trial.suggest_float("c", *c_bounds)
            total = a + b + c
            alpha = a / total
            beta = b / total
            gamma = c / total

            # Use bounds from computed min_distance_km
            min_dist = trial.suggest_int("min_distance_km", *min_dist_bnds)

            selector = DiversitySelector(n_samples=n_samp, use_multi_criteria=True)
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

            # Compute metrics
            n_selected = len(selected)
            if n_selected == 0:
                trial.set_user_attr("selection_shortfall", int(n_samp))
                trial.set_user_attr("hardcut_target_met", False)
                return 0.0
            shortfall = max(0, int(n_samp) - int(n_selected))
            if shortfall > 0:
                print(
                    "⚠️  Optuna hard-cut shortfall: "
                    f"requested n_samples={n_samp}, selected={n_selected}, "
                    f"trial={getattr(trial, 'number', 'N/A')}, min_distance_km={min_dist}"
                )

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
                target_n=int(n_samp),
                weight_diversity=0.5,
                weight_spread=0.5,
                infeasible_penalty=0.1,
            )

            # Log intermediate values
            trial.set_user_attr("alpha", float(alpha))
            trial.set_user_attr("beta", float(beta))
            trial.set_user_attr("gamma", float(gamma))
            trial.set_user_attr("min_distance_km", int(min_dist))
            trial.set_user_attr("n_selected", int(n_selected))
            trial.set_user_attr("n_samples", int(n_samp))
            trial.set_user_attr("selection_shortfall", int(shortfall))
            trial.set_user_attr("hardcut_target_met", shortfall == 0)
            trial.set_user_attr("selection_backend", str(selector.selection_backend))
            trial.set_user_attr("diversity", float(diversity))
            trial.set_user_attr("spatial_spread", float(spread))
            trial.set_user_attr("diversity_norm", float(objective_score.diversity_norm))
            trial.set_user_attr(
                "spatial_spread_norm", float(objective_score.spread_norm)
            )
            trial.set_user_attr("objective_score_raw", float(objective_score.raw_score))
            trial.set_user_attr("infeasible", bool(objective_score.infeasible))
            trial.set_user_attr(
                "feasibility_ratio", float(objective_score.feasibility_ratio)
            )

            return float(objective_score.score)
        except Exception as e:
            import traceback

            print(f"[ERROR] Exception in trial {getattr(trial, 'number', 'N/A')}: {e}")
            traceback.print_exc()
            trial.set_user_attr("error", str(e))
            return 0.0

    return objective


def run_optuna(
    n_trials: int = 50,
    n_candidates: int = 500,
    dim: int = 512,
    n_samples: int = 34,
    n_samples_range: tuple[int, int] | None = None,
    metadata_path: Path | str | None = None,
    seed: int = 42,
    study_name: str = "kdr100_opt",
    sampler_name: str = "tpe",
    constrain_bounds: dict[str, float] | None = None,
    exp_name: str | None = None,
    checkpoint_every: int = 0,
    out_dir: Path | None = None,
    feature_cache_dir: Path | None = None,
    study_db: str | None = None,
    pre_selected_names: list[str] | None = None,
    pre_selected_indices: list[int] | None = None,
) -> "optuna.Study":
    """
    Run Optuna optimization for multi-criteria hyperparameters.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    n_candidates : int
        Total candidate pool size
    dim : int
        Feature dimensionality
    n_samples : int
        Fixed number of samples (if n_samples_range is None)
    n_samples_range : tuple[int, int] | None
        Optional range for n_samples optimization
    metadata_path : Path | str | None
        Path to metadata CSV for computing min_distance_km bounds (required, no fallback)
    seed : int
        Random seed
    study_name : str
        Optuna study name
    sampler_name : str
        Sampler type (tpe, cmaes, qmc)
    constrain_bounds : dict | None
        Optional parameter bounds constraints
    exp_name : str | None
        Experiment name for run-specific outputs
    checkpoint_every : int
        Save checkpoint every N trials (0 = disabled)
    out_dir : Path | None
        Output directory (default: outputs/)
    feature_cache_dir : Path | None
        Shared directory for feature-cache files across repeated runs.
        If None, uses `out_dir`.
    study_db : str | None
        SQLite DB path for persistent storage
    pre_selected_names : list[str] | None
        Optional pre-selected tile names to enforce during selection
    pre_selected_indices : list[int] | None
        Optional pre-selected tile indices to enforce during selection

    Returns
    -------
    optuna.Study
        Completed Optuna study object
    """
    import optuna

    from dataselector.data.metadata_source import assert_canonical_metadata
    from dataselector.pipeline.pipeline_utils import compute_min_distance_km

    if out_dir is None:
        out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    if feature_cache_dir is not None:
        feature_cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute min_distance_km bounds (no fallback)
    if metadata_path is None:
        raise ValueError(
            "metadata_path is required for computing min_distance_km bounds. "
            "No hardcoded fallback is provided (long-term solution)."
        )

    metadata_path = assert_canonical_metadata(
        metadata_path,
        context="optuna-optimize",
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    min_dist_computed = compute_min_distance_km(str(metadata_path))
    # Bounds: ±20 km around computed value
    min_distance_bounds = (
        max(10, int(min_dist_computed - 20)),
        min(100, int(min_dist_computed + 20)),
    )

    features, metadata = load_or_create_data(
        out_dir,
        n=n_candidates,
        dim=dim,
        seed=seed,
        require_metadata=True,
        metadata_path=metadata_path,
        feature_cache_dir=feature_cache_dir,
    )

    sampler = get_optuna_sampler(sampler_name, seed=seed)

    # If a sqlite DB path was provided, use it as persistent storage
    if study_db:
        try:
            Path(study_db).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        storage = f"sqlite:///{study_db}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize", study_name=study_name, sampler=sampler
        )

    objective = objective_factory(
        features,
        metadata,
        n_samples=n_samples,
        min_distance_bounds=min_distance_bounds,
        n_samples_range=n_samples_range,
        constrain_bounds=constrain_bounds,
        pre_selected_names=pre_selected_names,
        pre_selected_indices=pre_selected_indices,
    )

    # Setup optional checkpoint callback
    callbacks = []
    if checkpoint_every and checkpoint_every > 0:

        def _optuna_checkpoint_callback(study_obj, trial_obj):
            try:
                if (trial_obj.number + 1) % checkpoint_every == 0:
                    try:
                        import joblib

                        joblib.dump(
                            study_obj,
                            out_dir
                            / f"optuna_study_checkpoint_{trial_obj.number+1}.pkl",
                        )
                    except Exception:
                        try:
                            import pickle

                            with open(
                                out_dir
                                / f"optuna_study_checkpoint_{trial_obj.number+1}.pkl",
                                "wb",
                            ) as f:
                                pickle.dump(study_obj, f)
                        except Exception as e:
                            print(f"[WARN] Failed to save study checkpoint: {e}")

                    df = study_obj.trials_dataframe()
                    df.to_csv(
                        out_dir / f"optuna_results_checkpoint_{trial_obj.number+1}.csv",
                        index=False,
                    )
                    print(
                        f"[INFO] Saved Optuna checkpoint at trial {trial_obj.number+1}"
                    )
            except Exception as e:
                print(f"[WARN] Exception in checkpoint callback: {e}")

        callbacks = [_optuna_checkpoint_callback]

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    # Save results
    print(f"OUT_DIR is {out_dir}")
    try:
        results_df = study.trials_dataframe()
        results_df.to_csv(out_dir / "optuna_results.csv", index=False)
        print(
            f"Optuna optimization finished. Results saved to {out_dir / 'optuna_results.csv'}"
        )
    except Exception as e:
        print(f"Failed to save results: {e}")

    # If an experiment name was provided, also save per-run outputs
    if exp_name:
        run_results_dir = out_dir / "runs" / exp_name / "results"
        run_results_dir.mkdir(parents=True, exist_ok=True)
        trials_csv = run_results_dir / "trials.csv"
        results_df.to_csv(trials_csv, index=False)
        print(f"Saved run trials to {trials_csv}")

    # Save study object
    try:
        import joblib

        joblib.dump(study, out_dir / "optuna_study.pkl")
    except Exception:
        print("joblib not available: only saving trials dataframe")

    return study


@cli_command(
    "optuna-optimize",
    help="Optuna hyperparameter optimization for multi-criteria selection",
    args={
        "n_trials": {
            "type": int,
            "default": 20,
            "help": "Number of optimization trials",
        },
        "n_candidates": {
            "type": int,
            "default": 500,
            "help": "Number of candidate tiles",
        },
        "dim": {"type": int, "default": 256, "help": "Feature dimension"},
        "n_samples": {
            "type": int,
            "default": 34,
            "help": "Number of samples for selection",
        },
        "smoke": {
            "type": bool,
            "action": "store_true",
            "help": "Run in smoke mode with reduced trials/candidates",
        },
        "workspace": {
            "type": str,
            "default": None,
            "help": "Alternate workspace path for outputs/data",
        },
        "n_samples_min": {
            "type": int,
            "default": None,
            "help": "Min samples for range (overrides n-samples if set)",
        },
        "n_samples_max": {
            "type": int,
            "default": None,
            "help": "Max samples for range (ignored if n-samples-min not set)",
        },
        "min_distance_km": {
            "type": float,
            "default": None,
            "help": (
                "Deprecated no-op. min_distance bounds are derived from "
                "metadata_path + constraints."
            ),
        },
        "seed": {"type": int, "default": 42, "help": "Random seed"},
        "checkpoint_every": {
            "type": int,
            "default": 0,
            "help": "Save Optuna study every N trials (0 disables)",
        },
        "sampler": {
            "type": str,
            "default": "tpe",
            "help": "Optuna sampler (qmc, tpe, cmaes)",
        },
        "exp_name": {"type": str, "default": None, "help": "Experiment name"},
        "use_study_db": {
            "type": bool,
            "action": "store_true",
            "help": "Create/use default outputs/optuna_study.db",
        },
        "study_db": {
            "type": str,
            "default": None,
            "help": "Path to SQLite DB file for Optuna storage",
        },
        "metadata_path": {
            "type": str,
            "default": "data/new_all_tiles.csv",
            "help": "Path to metadata CSV used for min_distance_km computation",
        },
        "constrain_a_min": {
            "type": float,
            "default": None,
            "help": "Constrain alpha min",
        },
        "constrain_a_max": {
            "type": float,
            "default": None,
            "help": "Constrain alpha max",
        },
        "constrain_b_min": {
            "type": float,
            "default": None,
            "help": "Constrain beta min",
        },
        "constrain_b_max": {
            "type": float,
            "default": None,
            "help": "Constrain beta max",
        },
        "constrain_c_min": {
            "type": float,
            "default": None,
            "help": "Constrain gamma min",
        },
        "constrain_c_max": {
            "type": float,
            "default": None,
            "help": "Constrain gamma max",
        },
        "constrain_min_dist_min": {
            "type": int,
            "default": None,
            "help": "Constrain min_distance lower bound",
        },
        "constrain_min_dist_max": {
            "type": int,
            "default": None,
            "help": "Constrain min_distance upper bound",
        },
        "pre_names": {
            "type": str,
            "nargs": "*",
            "default": None,
            "help": "Pre-selected tile names",
        },
        "pre_indices": {
            "type": int,
            "nargs": "*",
            "default": None,
            "help": "Pre-selected tile indices",
        },
        "hamburg": {
            "type": bool,
            "action": "store_true",
            "help": "Add Hamburg to pre-selected names",
        },
    },
)
def main(
    n_trials: int = 20,
    n_candidates: int = 500,
    dim: int = 256,
    n_samples: int = 34,
    smoke: bool = False,
    workspace: str | None = None,
    n_samples_min: int | None = None,
    n_samples_max: int | None = None,
    min_distance_km: float | None = None,
    seed: int = 42,
    checkpoint_every: int = 0,
    sampler: str = "tpe",
    exp_name: str | None = None,
    use_study_db: bool = False,
    study_db: str | None = None,
    metadata_path: str = "data/new_all_tiles.csv",
    constrain_a_min: float | None = None,
    constrain_a_max: float | None = None,
    constrain_b_min: float | None = None,
    constrain_b_max: float | None = None,
    constrain_c_min: float | None = None,
    constrain_c_max: float | None = None,
    constrain_min_dist_min: int | None = None,
    constrain_min_dist_max: int | None = None,
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
    hamburg: bool = False,
) -> int:
    """CLI entry point for Optuna optimization."""
    if min_distance_km is not None:
        print(
            "[WARN] --min-distance-km is deprecated in optuna-optimize and is ignored. "
            "Bounds are derived from metadata_path (and optional constraint args)."
        )

    # Apply smoke-mode overrides
    if smoke:
        n_trials = min(3, n_trials)
        n_candidates = min(50, n_candidates)
        checkpoint_every = checkpoint_every or 1

    # Workspace override
    if workspace:
        print(f"Setting DATASELECTOR_WORKSPACE to {workspace}")
        os.environ.setdefault("DATASELECTOR_WORKSPACE", workspace)

    pre_names_list = list(pre_names) if pre_names is not None else []
    if hamburg:
        pre_names_list.append("Hamburg")
    pre_names_list = list(dict.fromkeys(pre_names_list))
    pre_indices_list = (
        list(dict.fromkeys(int(idx) for idx in pre_indices))
        if pre_indices is not None
        else []
    )

    # Set output directory
    if os.environ.get("DATASELECTOR_WORKSPACE"):
        out_dir = Path(os.environ.get("DATASELECTOR_WORKSPACE")) / "outputs"
    else:
        out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Build n_samples range
    n_samples_range = None
    if n_samples_min is not None and n_samples_max is not None:
        n_samples_range = (n_samples_min, n_samples_max)

    # Build constraint bounds dict
    constrain_bounds = None
    if any(
        [
            constrain_a_min,
            constrain_a_max,
            constrain_b_min,
            constrain_b_max,
            constrain_c_min,
            constrain_c_max,
            constrain_min_dist_min,
            constrain_min_dist_max,
        ]
    ):
        constrain_bounds = {
            "a_min": constrain_a_min or 0.01,
            "a_max": constrain_a_max or 1.0,
            "b_min": constrain_b_min or 0.01,
            "b_max": constrain_b_max or 1.0,
            "c_min": constrain_c_min or 0.01,
            "c_max": constrain_c_max or 1.0,
            "min_dist_min": constrain_min_dist_min or 0,
            "min_dist_max": constrain_min_dist_max or 60,
        }

    # Determine SQLite DB path
    study_db_path = None
    if use_study_db:
        study_db_path = str(out_dir / "optuna_study.db")
    if study_db:
        study_db_path = str(study_db)

    run_optuna(
        n_trials=n_trials,
        n_candidates=n_candidates,
        dim=dim,
        n_samples=n_samples,
        seed=seed,
        n_samples_range=n_samples_range,
        sampler_name=sampler,
        constrain_bounds=constrain_bounds,
        exp_name=exp_name,
        checkpoint_every=checkpoint_every,
        out_dir=out_dir,
        study_db=study_db_path,
        metadata_path=metadata_path,
        pre_selected_names=pre_names_list if pre_names_list else None,
        pre_selected_indices=pre_indices_list if pre_indices_list else None,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
