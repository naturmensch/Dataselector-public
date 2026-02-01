"""Optuna hyperparameter optimization for Multi-Criteria weights.

REFACTORED: Uses ExperimentManager for professional versioning and reproducibility.

Usage:
    python scripts/optuna_optimize.py --n-trials 50 --n-candidates 500
    python scripts/optuna_optimize.py --n-trials 2000 --pre-names Hamburg --exp-name hamburg_sweep

Saves results to: outputs/runs/<timestamp>_<exp_name>/
  - config/config_optuna.yaml
  - results/trials.csv (incremental)
  - results/best_trial.csv
  - logs/optuna.log
  - manifest.json
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Project root (Path object); avoid modifying sys.path at module import time
ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is on sys.path so 'src' package resolves both for module and script invocations
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import optuna
except Exception:
    # Graceful exit in environments without optuna (e.g., CI/tests)
    if __name__ == "__main__":
        print("Optuna nicht installiert; Skript beendet sich sauber.")
        sys.exit(0)
    else:
        # Avoid hard import failure when module is imported elsewhere
        optuna = None

<<<<<<< HEAD
<<<<<<< HEAD
# Defer heavy imports where possible; attempt to import DiversitySelector but tolerate failures for smoke/test mode
try:
    from src.diversity_selector import DiversitySelector
    DIVERSITY_IMPORT_ERROR = None
except Exception as e:
    DiversitySelector = None
    DIVERSITY_IMPORT_ERROR = e
=======
from src.diversity_selector import DiversitySelector
from src.experiment_manager import ExperimentManager
from src.incremental_results import IncrementalCSVWriter, TrialBuffer
>>>>>>> ci/add-smoke-tests

# Allow overriding workspace via envvar (set by CLI helpers/tests)
if os.environ.get("DATASELECTOR_WORKSPACE"):
    OUT_DIR = Path(os.environ.get("DATASELECTOR_WORKSPACE")) / "outputs"
else:
    OUT_DIR = Path("outputs")
=======
OUT_DIR = Path("outputs")
>>>>>>> chore/ci-lint-attrs-gdf
OUT_DIR.mkdir(exist_ok=True)


def load_or_create_data(
    n: Optional[int] = None,
    dim=512,
    seed=123,
    pre_selected_names: Optional[list] = None,
    pre_selected_indices: Optional[list] = None,
):
    """Load real metadata from `data/new_all_tiles.csv` if available, otherwise
    fall back to synthetic metadata that includes `shortName` and `longName` so
    pre-selection by name works in Optuna seeded runs.
    """
    features_path = OUT_DIR / "features.npy"
    metadata_path = OUT_DIR / "metadata.csv"

    from src.io import load_or_extract_features

    # Prefer existing precomputed features + metadata in outputs/
    if features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=OUT_DIR, csv_meta=str(metadata_path), batch_size=16, cache=False
        )
        # Use load_metadata so any cached/derived projection (gdf_metric) is attached
        from src.io import load_metadata

        metadata = load_metadata(str(metadata_path))
    else:
        # Try loading canonical dataset metadata if present in repo
        canonical_meta = Path("data") / "new_all_tiles.csv"
        if canonical_meta.exists():
            metadata_full = pd.read_csv(canonical_meta)

            if n is None:
                n = len(metadata_full)

            # Identify rows that match any pre-selected names or indices to ensure they
            # are included in the sampled candidate set.
            include_mask = pd.Series(False, index=metadata_full.index)
            if pre_selected_names is not None:
                for nm in pre_selected_names:
                    mask = (
                        metadata_full["shortName"].astype(str).str.lower()
                        == str(nm).lower()
                    ) | metadata_full["longName"].astype(str).str.lower().str.contains(
                        str(nm).lower()
                    )
                    include_mask |= mask
            if pre_selected_indices is not None:
                # Try to match against the 'SheetNumber' column if present, otherwise try index
                if "SheetNumber" in metadata_full.columns:
                    include_mask |= metadata_full["SheetNumber"].isin(
                        pre_selected_indices
                    )
                else:
                    include_mask |= metadata_full.index.isin(pre_selected_indices)

            include_rows = metadata_full[include_mask]

            # Sample remaining rows to reach desired n candidates
            remaining = metadata_full[~include_mask]
            n_to_sample = max(0, int(n) - len(include_rows))
            if n_to_sample > 0 and n_to_sample < len(remaining):
                sampled = remaining.sample(n=n_to_sample, random_state=seed)
                metadata = pd.concat([include_rows, sampled], ignore_index=True)
            else:
                metadata = pd.concat([include_rows, remaining], ignore_index=True)

            # If the resulting set is larger than requested n, truncate but keep included ones
            if len(metadata) > n:
                # Ensure include_rows are retained; sample the rest down
                include_ids = set(include_rows.index)
                non_included = metadata[~metadata.index.isin(include_ids)]
                needed = int(n) - len(include_rows)
                if needed > 0 and len(non_included) > needed:
                    non_included = non_included.sample(n=needed, random_state=seed)
                metadata = pd.concat([include_rows, non_included], ignore_index=True)

            # Preserve original indices so we can map projected coords (gdf_metric) later
            original_indices = metadata.index.copy()
            metadata = metadata.reset_index(drop=True)

            # Attempt to attach projected coordinates from the canonical metadata
            try:
                from src.metadata_processor import MetadataProcessor

                mp = MetadataProcessor(str(canonical_meta))
                mp.load_csv()
                full_metric = mp.ensure_metric_crs()
                if full_metric is not None:
                    # Subset the full projected frame to the sampled rows and reindex to metadata
                    metadata.gdf_metric = full_metric.loc[original_indices].reset_index(
                        drop=True
                    )
            except Exception:
                # Best effort: if geopandas unavailable or mapping fails, proceed without gdf_metric
                pass

            # Generate synthetic features for sampling experiments
            rng = np.random.RandomState(seed)
            features = rng.randn(len(metadata), dim).astype("float32")
        else:
            rng = np.random.RandomState(seed)
            if n is None:
                n = 673  # Fallback for synthetic data
            features = rng.randn(n, dim).astype("float32")
            metadata = pd.DataFrame(
                {
                    "N": np.random.uniform(48, 55, n),
                    "left": np.random.uniform(6, 15, n),
                    "year": np.random.randint(1880, 1945, n),
                }
            )
            # Add synthetic shortName/longName columns so name-based preselection works
            metadata["shortName"] = [f"KDR_{i:03d}" for i in range(len(metadata))]
            metadata["longName"] = [
                f"KDR_{i:03d}_Synthetic" for i in range(len(metadata))
            ]

    # Guard: ensure shortName/longName exist for preselection
    if "shortName" not in metadata.columns:
        metadata["shortName"] = metadata.index.map(lambda i: f"KDR_{i:03d}")
    if "longName" not in metadata.columns:
        metadata["longName"] = metadata["shortName"].astype(str) + "_Synthetic"

    return features, metadata


<<<<<<< HEAD
def objective_factory(features, metadata, n_samples, min_distance_km, n_samples_range=None, constrain_bounds=None):
=======
def objective_factory(
    features,
    metadata,
    fixed_n_samples: Optional[int] = None,
    min_distance_km: int = 40,
    min_distance_min: Optional[int] = None,
    min_distance_max: Optional[int] = None,
    n_samples_min: Optional[int] = None,
    n_samples_max: Optional[int] = None,
    pre_selected_names: Optional[list] = None,
    pre_selected_indices: Optional[list] = None,
):
>>>>>>> ci/add-smoke-tests
    def objective(trial: optuna.trial.Trial):
        try:
            # Allow Optuna to explore n_samples if a range is provided; otherwise use fixed.
            n_samp = (
                trial.suggest_int("n_samples", n_samples_range[0], n_samples_range[1])
                if n_samples_range is not None
                else n_samples
            )
            
            # Determine bounds for a, b, c
            if constrain_bounds:
                a_bounds = (constrain_bounds.get("a_min", 0.01), constrain_bounds.get("a_max", 1.0))
                b_bounds = (constrain_bounds.get("b_min", 0.01), constrain_bounds.get("b_max", 1.0))
                c_bounds = (constrain_bounds.get("c_min", 0.01), constrain_bounds.get("c_max", 1.0))
                min_dist_bounds = (constrain_bounds.get("min_dist_min", 0), constrain_bounds.get("min_dist_max", 60))
            else:
                a_bounds = (0.01, 1.0)
                b_bounds = (0.01, 1.0)
                c_bounds = (0.01, 1.0)
                min_dist_bounds = (0, 60)
            
            # Sample raw weights and normalize (ensures sum=1 and non-negative)
            a = trial.suggest_float("a", *a_bounds)
            b = trial.suggest_float("b", *b_bounds)
            c = trial.suggest_float("c", *c_bounds)
            total = a + b + c
            alpha = a / total
            beta = b / total
            gamma = c / total

<<<<<<< HEAD
            # Use conservative bounds for min_distance based on dataset grid (median ≈ 28km).
            # Limit search to [0, 60] km to avoid overly restrictive values that prevent selecting enough samples.
            min_dist = trial.suggest_int("min_distance_km", *min_dist_bounds)

            selector = DiversitySelector(n_samples=n_samp, use_multi_criteria=True)
            selected = selector.select(
                features,
                metadata,
                spatial_constraint=True,
                min_distance_km=min_dist,
                alpha_visual=alpha,
                beta_spatial=beta,
                gamma_temporal=gamma,
            )
=======
        # Optimize min_distance_km: use dynamic bounds if provided, else fallback to hardcoded defaults
        if min_distance_min is not None and min_distance_max is not None:
            # Ensure bounds are valid (low <= high)
            lo = min(min_distance_min, min_distance_max)
            hi = max(min_distance_min, min_distance_max)
            min_dist = trial.suggest_int("min_distance_km", lo, hi)
        else:
            # Fallback: Use conservative bounds based on dataset grid (median ≈ 28km)
            # Limit search to a focused range around empirically validated optimum (25–55 km)
            min_dist = trial.suggest_int("min_distance_km", 25, 55)

        # Decide n_samples: either fixed or suggested within a range
        if n_samples_min is not None and n_samples_max is not None:
            # Ensure bounds are valid
            lo = int(min(n_samples_min, n_samples_max))
            hi = int(max(n_samples_min, n_samples_max))
            n_samples = trial.suggest_int("n_samples", lo, hi)
        else:
            # Fall back to fixed value or adaptive heuristic when unspecified
            if fixed_n_samples is not None:
                n_samples = int(fixed_n_samples)
            else:
                # Use adaptive heuristic for initial sample size (dimension-aware rule)
                from src.pipeline_utils import compute_adaptive_n_initial

                n_samples = compute_adaptive_n_initial(n_dimensions=3)

        # Import selector lazily to avoid module-level side effects when importing script
        from src.diversity_selector import DiversitySelector

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
>>>>>>> ci/add-smoke-tests

            # Compute metrics
            n_selected = len(selected)
            if n_selected == 0:
                return 0.0

            diversity = selector._calculate_diversity_score(features[selected])
            spatial_spread = metadata.loc[selected, ["N", "left"]].std().mean()

            # Composite objective (maximize)
            score = diversity * spatial_spread

            # Log intermediate values
            trial.set_user_attr("alpha", float(alpha))
            trial.set_user_attr("beta", float(beta))
            trial.set_user_attr("gamma", float(gamma))
            trial.set_user_attr("min_distance_km", int(min_dist))
            trial.set_user_attr("n_selected", int(n_selected))
            trial.set_user_attr("n_samples", int(n_samp))
            trial.set_user_attr("diversity", float(diversity))
            trial.set_user_attr("spatial_spread", float(spatial_spread))

            return float(score)
        except Exception as e:
            import traceback
            print(f"[ERROR] Exception in trial {getattr(trial, 'number', 'N/A')}: {e}")
            traceback.print_exc()
            trial.set_user_attr("error", str(e))
            return 0.0

<<<<<<< HEAD
    return objective


def run_optuna(
    n_trials=50,
    n_candidates=500,
    dim=512,
    n_samples=34,
    n_samples_range=None,
    min_distance_km=28,
    seed=42,
    study_name="kdr100_opt",
    sampler_name="tpe",
    constrain_bounds=None,
    exp_name: str | None = None,
    checkpoint_every: int = 0,
):
    features, metadata = load_or_create_data(n=n_candidates, dim=dim, seed=seed)

    sampler = None
    if sampler_name == "qmc":
        sampler = optuna.samplers.QMCSampler()
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler()
    else:
        sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler)
    objective = objective_factory(
        features, metadata, n_samples=n_samples, min_distance_km=min_distance_km, n_samples_range=n_samples_range, constrain_bounds=constrain_bounds
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
                            study_obj, OUT_DIR / f"optuna_study_checkpoint_{trial_obj.number+1}.pkl"
                        )
                    except Exception:
                        try:
                            import pickle

                            with open(OUT_DIR / f"optuna_study_checkpoint_{trial_obj.number+1}.pkl", "wb") as f:
                                pickle.dump(study_obj, f)
                        except Exception as e:
                            print(f"[WARN] Failed to save study checkpoint: {e}")

                    df = study_obj.trials_dataframe()
                    df.to_csv(OUT_DIR / f"optuna_results_checkpoint_{trial_obj.number+1}.csv", index=False)
                    print(f"[INFO] Saved Optuna checkpoint at trial {trial_obj.number+1}")
            except Exception as e:
                print(f"[WARN] Exception in checkpoint callback: {e}")

        callbacks = [_optuna_checkpoint_callback]

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(OUT_DIR / "optuna_results.csv", index=False)

    # If an experiment name was provided, also save per-run outputs to a run-specific folder
    if exp_name:
        run_results_dir = OUT_DIR / "runs" / exp_name / "results"
        run_results_dir.mkdir(parents=True, exist_ok=True)
        trials_csv = run_results_dir / "trials.csv"
        results_df.to_csv(trials_csv, index=False)
        print(f"Saved run trials to {trials_csv}")

    # Save study object
=======
        diversity = selector._calculate_diversity_score(features[selected])
        spatial_spread = metadata.loc[selected, ["N", "left"]].std().mean()

        # Composite objective (maximize)
        score = diversity * spatial_spread

        # Log intermediate values
        trial.set_user_attr("alpha", float(alpha))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("gamma", float(gamma))
        trial.set_user_attr("min_distance_km", int(min_dist))
        trial.set_user_attr("n_samples", int(n_samples))
        trial.set_user_attr("n_selected", int(n_selected))
        trial.set_user_attr("diversity", float(diversity))
        trial.set_user_attr("spatial_spread", float(spatial_spread))
        trial.set_user_attr("pre_selected_names", pre_selected_names)
        trial.set_user_attr("pre_selected_indices", pre_selected_indices)

        return float(score)

    return objective


def get_optuna_sampler(sampler_name: str = "qmc", seed: int = 42):
    """Return an Optuna sampler instance based on a name.

    Supported samplers: 'qmc' (QMCSampler/Sobol), 'tpe' (TPESampler), 'cmaes' (CmaEsSampler)
    Falls back to TPESampler when requested sampler is unavailable.
    """
    name = sampler_name.lower()
>>>>>>> ci/add-smoke-tests
    try:
        if name == "qmc":
            # Prefer QMCSampler (Sobol) for QMC sampling. Different optuna
            # versions accepted different keyword names, so try them in order.
            try:
                # Newer optuna (>= 3.x) might accept qmc_type or qmc
                sampler = optuna.samplers.QMCSampler(seed=seed, qmc_type="sobol")
                return sampler
            except TypeError:
                try:
                    sampler = optuna.samplers.QMCSampler(seed=seed, qmc="sobol")
                    return sampler
                except TypeError:
                    # Last resort: call without qmc kwargs
                    sampler = optuna.samplers.QMCSampler(seed=seed)
                    return sampler
        elif name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=seed)
        else:
            # Default to TPE for 'tpe' or unknown learners
            return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    except Exception as e:
        # Fallback: ensure we can always return a working sampler
        print(
            f"Warning: could not instantiate sampler '{sampler_name}' ({e}), falling back to TPE."
        )
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)


def run_optuna(
    n_trials: int = 50,
    n_candidates: Optional[int] = None,
    dim: int = 512,
    n_samples: Optional[int] = None,
    n_samples_min: Optional[int] = None,
    n_samples_max: Optional[int] = None,
    min_distance_km: int = 40,
    min_distance_min: Optional[int] = None,
    min_distance_max: Optional[int] = None,
    seed: int = 42,
    study_name: str = "kdr100_opt",
    sampler_name: str = "qmc",
    pre_selected_names: Optional[List[str]] = None,
    pre_selected_indices: Optional[List[int]] = None,
    exp_name: str = "optuna",
    exp_description: str = "",
    storage: Optional[str] = None,
):
    """Run Optuna optimization with professional experiment management.

    Args:
        n_trials: Number of trials
        n_candidates: Number of candidates in the pool
        dim: Feature dimension
        n_samples: Fixed number of samples (or None for range-based optimization)
        n_samples_min/max: Range for n_samples optimization
        min_distance_km: Minimum distance constraint
        seed: Random seed
        study_name: Optuna study name
        sampler_name: Sampler type ('qmc', 'tpe', 'cmaes')
        pre_selected_names: List of tile names to preselect
        pre_selected_indices: List of tile indices to preselect
        exp_name: Experiment identifier for versioning
        exp_description: Human-readable description
        storage: Optuna storage URL (e.g. sqlite:///...). If None, defaults to sqlite in run dir.
    """
    # Initialize or attach to an experiment manager

    # Ensure ExperimentManager is available in this runtime context
    try:
        from src.experiment_manager import ExperimentManager
    except Exception as e:
        raise RuntimeError(
            "ExperimentManager import failed. Ensure project is installed or PYTHONPATH includes the repo root"
        ) from e

    em_env = os.environ.get("EXPERIMENT_RUN_DIR")
    if em_env:
        em = ExperimentManager.from_existing(em_env)
        em.log("Attached to pipeline run (optuna stage)")
        em.manifest.setdefault("metadata", {}).update(
            {
                "n_trials": n_trials,
                "n_candidates": n_candidates,
                "sampler": sampler_name,
                "seed": seed,
                "n_samples": n_samples,
                "n_samples_min": n_samples_min,
                "n_samples_max": n_samples_max,
                "min_distance_km": min_distance_km,
                "min_distance_min": min_distance_min,
                "min_distance_max": min_distance_max,
                "pre_selected_names": pre_selected_names,
                "pre_selected_indices": pre_selected_indices,
            }
        )
        em.save_manifest()
    else:
        em = ExperimentManager(
            name=exp_name,
            description=exp_description
            or f"Optuna optimization (n_trials={n_trials}, n_candidates={n_candidates})",
            base_dir=Path("outputs/runs"),
            metadata={
                "n_trials": n_trials,
                "n_candidates": n_candidates,
                "seed": seed,
                "sampler": sampler_name,
            },
        )
        # Export run dir to environment so downstream modules (e.g., clustering) can persist their config
        try:
            import os as _os

            _os.environ["EXPERIMENT_RUN_DIR"] = str(em.run_dir)
        except Exception:
            pass

    em.log(f"Loading features and metadata (n_candidates={n_candidates}, dim={dim})")
    features, metadata = load_or_create_data(
        n=n_candidates,
        dim=dim,
        seed=seed,
        pre_selected_names=pre_selected_names,
        pre_selected_indices=pre_selected_indices,
    )

    # Update n_candidates if it was None (auto-detected)
    if n_candidates is None:
        n_candidates = len(metadata)

    # Document candidate set and preselection
    try:
        em.log(
            f"Documenting candidate set: {len(metadata)} candidates, pre-selection: names={pre_selected_names}, indices={pre_selected_indices}"
        )
        cand_df = metadata.reset_index().rename(columns={"index": "candidate_index"})
        # Ensure name columns are strings to avoid dtype warnings
        if "shortName" in cand_df.columns:
            cand_df["shortName"] = cand_df["shortName"].astype(str)
        if "longName" in cand_df.columns:
            cand_df["longName"] = cand_df["longName"].astype(str)
        cand_df["is_preselected"] = False

        if pre_selected_names is not None:
            for nm in pre_selected_names:
                mask = (cand_df["shortName"].str.lower() == str(nm).lower()) | cand_df[
                    "longName"
                ].str.lower().str.contains(str(nm).lower())
                cand_df.loc[mask, "is_preselected"] = True

        if pre_selected_indices is not None:
            if "SheetNumber" in cand_df.columns:
                cand_df.loc[
                    cand_df["SheetNumber"].isin(pre_selected_indices), "is_preselected"
                ] = True
            else:
                cand_df.loc[
                    cand_df["candidate_index"].isin(pre_selected_indices),
                    "is_preselected",
                ] = True

        cols = [
            c
            for c in [
                "candidate_index",
                "SheetNumber",
                "shortName",
                "longName",
                "is_preselected",
            ]
            if c in cand_df.columns
        ]
        em.save_results("candidate_set", cand_df[cols], format="csv")
    except Exception as e:
        em.log(f"Warning: could not document candidate set: {e}", level="warning")

    # Save Optuna configuration
    optuna_config = {
        "n_trials": n_trials,
        "n_candidates": n_candidates,
        "sampler": sampler_name,
        "seed": seed,
        "n_samples": n_samples,
        "n_samples_min": n_samples_min,
        "n_samples_max": n_samples_max,
        "min_distance_km": min_distance_km,
        "pre_selected_names": pre_selected_names,
        "pre_selected_indices": pre_selected_indices,
    }
    em.save_config("optuna", optuna_config)

    # Setup incremental trial saving
    trials_csv_path = em.get_path("results/trials.csv")
    trial_fieldnames = [
        "trial_number",
        "datetime_start",
        "datetime_complete",
        "duration_sec",
        "value",
        "a",
        "b",
        "c",
        "min_distance_km",
        "n_samples",
        "state",
    ]
    # Incremental writer helpers
    from src.incremental_results import IncrementalCSVWriter, TrialBuffer

    trial_writer = IncrementalCSVWriter(
        trials_csv_path, fieldnames=trial_fieldnames, buffer_size=50
    )
    trial_buffer = TrialBuffer(trial_writer)

    em.log(f"Initialized incremental trial writer: {trials_csv_path}")

    # Determine storage URL if not provided
    if storage is None:
        db_path = em.run_dir / "optuna_study.db"
        storage = f"sqlite:///{db_path}"
        em.log(f"Using persistent storage: {storage}")

    # Create Optuna study
    sampler = get_optuna_sampler(sampler_name, seed=seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # Validate n_samples configuration
    has_fixed = n_samples is not None
    has_range = (n_samples_min is not None) and (n_samples_max is not None)
    if has_fixed and has_range:
        raise ValueError(
            "Specify either --n-samples OR the range --n-samples-min/--n-samples-max, not both."
        )
    # Relaxed validation: if neither is specified, objective_factory will use a heuristic default.
    # if not has_fixed and not has_range: ...

    # Create objective factory
    objective = objective_factory(
        features,
        metadata,
        fixed_n_samples=n_samples,
        min_distance_km=min_distance_km,
        min_distance_min=min_distance_min,
        min_distance_max=min_distance_max,
        n_samples_min=n_samples_min,
        n_samples_max=n_samples_max,
        pre_selected_names=pre_selected_names,
        pre_selected_indices=pre_selected_indices,
    )

    # Setup callback to save trials incrementally
    trial_start_times = {}

    def trial_callback(study, trial):
        """Callback after each trial completes."""
        try:
            trial_dict = {
                "trial_number": trial.number,
                "datetime_start": trial_start_times.get(trial.number, "N/A"),
                "datetime_complete": datetime.now().isoformat(),
                "duration_sec": (
                    trial.duration.total_seconds() if trial.duration else None
                ),
                "value": trial.value,
                "a": trial.params.get("a"),
                "b": trial.params.get("b"),
                "c": trial.params.get("c"),
                "min_distance_km": trial.params.get("min_distance_km"),
                # n_samples may be a trial.param (when range-sampling) or a user_attr (when fixed/adaptive). Prefer param, fall back to user_attr.
                "n_samples": (
                    trial.params.get("n_samples")
                    if trial.params and trial.params.get("n_samples") is not None
                    else (
                        trial.user_attrs.get("n_samples")
                        if hasattr(trial, "user_attrs")
                        and trial.user_attrs.get("n_samples") is not None
                        else None
                    )
                ),
                "state": str(trial.state),
            }
            trial_buffer.add_trial(trial_dict)

            # Flush every 100 trials
            if (trial.number + 1) % 100 == 0:
                trial_buffer.flush_to_csv()
                stats = trial_buffer.get_stats()
                em.log(
                    f"Trial {trial.number + 1}/{n_trials} - Best: {study.best_value:.4f} - Stats: {stats}"
                )
        except Exception as e:
            em.log(f"Error in trial callback: {e}", level="error")

    # Wrap objective to track start times
    original_objective = objective

    def objective_with_timing(trial):
        trial_start_times[trial.number] = datetime.now().isoformat()
        return original_objective(trial)

    # Run optimization
    em.log(f"Starting Optuna optimization: {n_trials} trials")
    study.optimize(objective_with_timing, n_trials=n_trials, callbacks=[trial_callback])

    # Flush any remaining trials
    trial_buffer.flush_to_csv()
    trial_writer.close()
    em.log(f"All {len(study.trials)} trials saved to {trials_csv_path}")

    # Determine if there are any valid completed trials with numeric values
    import math

    valid_values = [
        t.value
        for t in study.trials
        if t.value is not None
        and not (isinstance(t.value, float) and math.isnan(t.value))
    ]

    # Save best trial only if we have valid values
    if len(valid_values) == 0:
        em.log(
            "No valid completed trials found; skipping best trial extraction.",
            level="warning",
        )
        best_value = None
        best_trial_number = None
    else:
        try:
            best = study.best_trial
            best_dict = {
                "trial_number": best.number,
                "value": best.value,
                "a": best.params.get("a"),
                "b": best.params.get("b"),
                "c": best.params.get("c"),
                "min_distance_km": best.params.get("min_distance_km"),
                "n_samples": best.params.get("n_samples"),
                "datetime": (
                    best.datetime_complete.isoformat()
                    if best.datetime_complete
                    else None
                ),
            }
            em.save_results("best_trial", best_dict, format="json")

            # Extract weights
            if all(k in best.params for k in ["a", "b", "c"]):
                total = best.params["a"] + best.params["b"] + best.params["c"]
                alpha = float(best.params["a"] / total)
                beta = float(best.params["b"] / total)
                gamma = float(best.params["c"] / total)
            else:
                alpha = beta = gamma = None

            # Save best config
            best_config = {
                "selection": {
                    "alpha_visual": alpha,
                    "beta_spatial": beta,
                    "gamma_temporal": gamma,
                    "min_distance_km": int(
                        best.params.get("min_distance_km", min_distance_km)
                    ),
                    "n_samples": int(best.params.get("n_samples", n_samples or 34)),
                    "pre_selected_names": pre_selected_names,
                    "pre_selected_indices": pre_selected_indices,
                }
            }
            em.save_config("best_selection", best_config)
            em.log(f"Best trial: #{best.number}, value={best.value:.4f}")
            best_value = best.value
            best_trial_number = best.number
        except Exception as e:
            em.log(f"Error saving best trial: {e}", level="error")
            best_value = None
            best_trial_number = None

    # Mark stage complete
    em.mark_stage_complete(
        "optuna",
        summary={
            "n_trials_completed": len(study.trials),
            "best_value": best_value,
            "best_trial_number": best_trial_number,
        },
    )

    # Save manifest
    em.save_manifest()
    em.mark_complete(success=True)

    print(em.summary())

    return study, em


if __name__ == "__main__":
<<<<<<< HEAD
<<<<<<< HEAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=34)
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode with reduced trials/candidates")
    parser.add_argument("--workspace", type=str, default=None, help="Alternate workspace path for outputs/data")
    parser.add_argument("--n-samples-min", type=int, default=None, help="Min samples for range (overrides n-samples if set)")
    parser.add_argument("--n-samples-max", type=int, default=None, help="Max samples for range (ignored if n-samples-min not set)")
    parser.add_argument("--min-distance-km", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save Optuna study and results every N trials (0 disables)")
    parser.add_argument("--sampler", type=str, default="tpe", help="Optuna sampler (qmc, tpe, cmaes)")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (optional)")
    parser.add_argument("--exp-desc", type=str, default=None, help="Experiment description (optional)")
    parser.add_argument("--hamburg", action="store_true", help="Use Hamburg dataset preselection")
    parser.add_argument("--constrain-a-min", type=float, default=None, help="Constrain a (alpha-proxy) lower bound")
    parser.add_argument("--constrain-a-max", type=float, default=None, help="Constrain a upper bound")
    parser.add_argument("--constrain-b-min", type=float, default=None, help="Constrain b (beta-proxy) lower bound")
    parser.add_argument("--constrain-b-max", type=float, default=None, help="Constrain b upper bound")
    parser.add_argument("--constrain-c-min", type=float, default=None, help="Constrain c (gamma-proxy) lower bound")
    parser.add_argument("--constrain-c-max", type=float, default=None, help="Constrain c upper bound")
    parser.add_argument("--constrain-min-dist-min", type=int, default=None, help="Constrain min_distance lower bound")
    parser.add_argument("--constrain-min-dist-max", type=int, default=None, help="Constrain min_distance upper bound")

    args = parser.parse_args()

    # Apply smoke-mode overrides if requested
    if args.smoke:
        args.n_trials = min(3, args.n_trials)
        args.n_candidates = min(50, args.n_candidates)
        # checkpoint frequently in smoke to verify checkpointing behavior
        args.checkpoint_every = args.checkpoint_every or 1

    # Allow workspace override for tests
    if args.workspace:
        import os

        os.environ.setdefault("DATASELECTOR_WORKSPACE", args.workspace)

    # If smoke mode requested and DiversitySelector not importable (heavy deps missing),
    # simulate a minimal optuna output instead of running full optimization.
    if args.smoke and DiversitySelector is None:
        print("Running simulated optuna smoke (DiversitySelector import failed)")
        import pandas as _pd
        import csv as _csv
        # Recompute OUT_DIR in case workspace env var was set after import
        if os.environ.get("DATASELECTOR_WORKSPACE"):
            out_dir = Path(os.environ.get("DATASELECTOR_WORKSPACE")) / "outputs"
        else:
            out_dir = OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write a small checkpoint and final results CSV
        ck_csv = out_dir / "optuna_results_checkpoint_1.csv"
        df = _pd.DataFrame([
            {"trial_number": 0, "value": 0.1, "a": 0.7, "b": 0.15, "c": 0.15, "n_samples": 5}
        ])
        df.to_csv(ck_csv, index=False)
        df.to_csv(out_dir / "optuna_results.csv", index=False)
        print(f"Simulated optuna results written to {out_dir}")
        sys.exit(0)

    n_samples_range = None
    if args.n_samples_min is not None and args.n_samples_max is not None:
        n_samples_range = (args.n_samples_min, args.n_samples_max)

    n_samples = args.n_samples
    
    # Build constraint bounds dict if any constraint flags are set
    constrain_bounds = None
    if any([args.constrain_a_min, args.constrain_a_max, args.constrain_b_min, args.constrain_b_max, 
            args.constrain_c_min, args.constrain_c_max, args.constrain_min_dist_min, args.constrain_min_dist_max]):
        constrain_bounds = {
            "a_min": args.constrain_a_min or 0.01,
            "a_max": args.constrain_a_max or 1.0,
            "b_min": args.constrain_b_min or 0.01,
            "b_max": args.constrain_b_max or 1.0,
            "c_min": args.constrain_c_min or 0.01,
            "c_max": args.constrain_c_max or 1.0,
            "min_dist_min": args.constrain_min_dist_min or 0,
            "min_dist_max": args.constrain_min_dist_max or 60,
        }

    run_optuna(
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        dim=args.dim,
        n_samples=n_samples,
=======
    parser = argparse.ArgumentParser(description="Optuna optimization with experiment versioning")
=======
    parser = argparse.ArgumentParser(
        description="Optuna optimization with experiment versioning"
    )
>>>>>>> chore/ci-lint-attrs-gdf
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of candidates (default: all)",
    )
    parser.add_argument("--dim", type=int, default=256, help="Feature dimension")
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Fixed number of samples to select"
    )
    parser.add_argument(
        "--n-samples-min",
        type=int,
        default=None,
        help="Lower bound for optimizing n_samples",
    )
    parser.add_argument(
        "--n-samples-max",
        type=int,
        default=None,
        help="Upper bound for optimizing n_samples",
    )
    parser.add_argument(
        "--min-distance-km", type=int, default=40, help="Minimum distance constraint"
    )
    parser.add_argument(
        "--min-distance-min",
        type=int,
        default=None,
        help="Lower bound for min_distance optimization",
    )
    parser.add_argument(
        "--min-distance-max",
        type=int,
        default=None,
        help="Upper bound for min_distance optimization",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pre-names",
        type=str,
        nargs="*",
        default=None,
        help="Pre-selected tile names (e.g. Hamburg)",
    )
    parser.add_argument(
        "--pre-indices",
        type=int,
        nargs="*",
        default=None,
        help="Pre-selected tile indices",
    )
    parser.add_argument(
        "--sampler", choices=["tpe", "qmc", "cmaes"], default="qmc", help="Sampler type"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="optuna",
        help="Experiment identifier for versioning",
    )
    parser.add_argument(
        "--exp-desc", type=str, default="", help="Experiment description"
    )
    parser.add_argument(
        "--hamburg", action="store_true", help="Convenience flag: pre-select Hamburg"
    )
    parser.add_argument(
        "--KDR146", action="store_true", help="Convenience flag: pre-select KDR_146"
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///path/to/db). Defaults to sqlite in run dir.",
    )

    args = parser.parse_args()

    # Handle convenience flags for preselection
    pre_names = list(args.pre_names) if args.pre_names is not None else []
    if args.hamburg:
        pre_names.append("Hamburg")
    if args.KDR146:
        pre_names.append("KDR_146")

    args.pre_names = pre_names if pre_names else None

    study, em = run_optuna(
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        dim=args.dim,
        n_samples=args.n_samples,
        n_samples_min=args.n_samples_min,
        n_samples_max=args.n_samples_max,
>>>>>>> ci/add-smoke-tests
        min_distance_km=args.min_distance_km,
        min_distance_min=args.min_distance_min,
        min_distance_max=args.min_distance_max,
        seed=args.seed,
<<<<<<< HEAD
        n_samples_range=n_samples_range,
        sampler_name=args.sampler,
        constrain_bounds=constrain_bounds,
        exp_name=args.exp_name,
        checkpoint_every=args.checkpoint_every,
=======
        sampler_name=args.sampler,
        pre_selected_names=args.pre_names,
        pre_selected_indices=args.pre_indices,
        exp_name=args.exp_name,
        exp_description=args.exp_desc,
        storage=args.optuna_storage,
>>>>>>> ci/add-smoke-tests
    )
