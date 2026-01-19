# ruff: noqa: E402
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
from typing import Optional, List
import os
import sys
from pathlib import Path
from datetime import datetime

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

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

from src.diversity_selector import DiversitySelector
from src.experiment_manager import ExperimentManager
from src.incremental_results import IncrementalCSVWriter, TrialBuffer

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def load_or_create_data(n: Optional[int] = None, dim=512, seed=123, pre_selected_names: Optional[list] = None, pre_selected_indices: Optional[list] = None):
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
        metadata = pd.read_csv(metadata_path)
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
                        metadata_full["shortName"].astype(str).str.lower() == str(nm).lower()
                    ) | metadata_full["longName"].astype(str).str.lower().str.contains(
                        str(nm).lower()
                    )
                    include_mask |= mask
            if pre_selected_indices is not None:
                # Try to match against the 'SheetNumber' column if present, otherwise try index
                if "SheetNumber" in metadata_full.columns:
                    include_mask |= metadata_full["SheetNumber"].isin(pre_selected_indices)
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

            metadata = metadata.reset_index(drop=True)

            # Generate synthetic features for sampling experiments
            rng = np.random.RandomState(seed)
            features = rng.randn(len(metadata), dim).astype("float32")
        else:
            rng = np.random.RandomState(seed)
            if n is None: n = 673  # Fallback for synthetic data
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
            metadata["longName"] = [f"KDR_{i:03d}_Synthetic" for i in range(len(metadata))]

    # Guard: ensure shortName/longName exist for preselection
    if "shortName" not in metadata.columns:
        metadata["shortName"] = metadata.index.map(lambda i: f"KDR_{i:03d}")
    if "longName" not in metadata.columns:
        metadata["longName"] = metadata["shortName"].astype(str) + "_Synthetic"

    return features, metadata


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
    def objective(trial: optuna.trial.Trial):
        # Sample raw weights and normalize (ensures sum=1 and non-negative)
        a = trial.suggest_float("a", 0.01, 1.0)
        b = trial.suggest_float("b", 0.01, 1.0)
        c = trial.suggest_float("c", 0.01, 1.0)
        total = a + b + c
        alpha = a / total
        beta = b / total
        gamma = c / total

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
        trial.set_user_attr("n_samples", int(n_samples))
        trial.set_user_attr("n_selected", int(n_selected))
        trial.set_user_attr("diversity", float(diversity))
        trial.set_user_attr("spatial_spread", float(spatial_spread))
        trial.set_user_attr("pre_selected_names", pre_selected_names)
        trial.set_user_attr("pre_selected_indices", pre_selected_indices)

        return float(score)

    return objective


def get_optuna_sampler(sampler_name: str = 'qmc', seed: int = 42):
    """Return an Optuna sampler instance based on a name.

    Supported samplers: 'qmc' (QMCSampler/Sobol), 'tpe' (TPESampler), 'cmaes' (CmaEsSampler)
    Falls back to TPESampler when requested sampler is unavailable.
    """
    name = sampler_name.lower()
    try:
        if name == 'qmc':
            # Prefer QMCSampler (Sobol) for QMC sampling. Different optuna
            # versions accepted different keyword names, so try them in order.
            try:
                # Newer optuna (>= 3.x) might accept qmc_type or qmc
                sampler = optuna.samplers.QMCSampler(seed=seed, qmc_type='sobol')
                return sampler
            except TypeError:
                try:
                    sampler = optuna.samplers.QMCSampler(seed=seed, qmc='sobol')
                    return sampler
                except TypeError:
                    # Last resort: call without qmc kwargs
                    sampler = optuna.samplers.QMCSampler(seed=seed)
                    return sampler
        elif name == 'cmaes':
            return optuna.samplers.CmaEsSampler(seed=seed)
        else:
            # Default to TPE for 'tpe' or unknown learners
            return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    except Exception as e:
        # Fallback: ensure we can always return a working sampler
        print(f"Warning: could not instantiate sampler '{sampler_name}' ({e}), falling back to TPE.")
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
    sampler_name: str = 'qmc',
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
    import os
    em_env = os.environ.get('EXPERIMENT_RUN_DIR')
    if em_env:
        em = ExperimentManager.from_existing(em_env)
        em.log("Attached to pipeline run (optuna stage)")
        em.manifest.setdefault('metadata', {}).update({
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
        })
        em.save_manifest()
    else:
        em = ExperimentManager(
            name=exp_name,
            description=exp_description or f"Optuna optimization (n_trials={n_trials}, n_candidates={n_candidates})",
            base_dir=Path("outputs/runs"),
            metadata={
                "n_trials": n_trials,
                "n_candidates": n_candidates,
                "seed": seed,
                "sampler": sampler_name,
            }
        )
    
    em.log(f"Loading features and metadata (n_candidates={n_candidates}, dim={dim})")
    features, metadata = load_or_create_data(
        n=n_candidates, dim=dim, seed=seed,
        pre_selected_names=pre_selected_names,
        pre_selected_indices=pre_selected_indices
    )
    
    # Update n_candidates if it was None (auto-detected)
    if n_candidates is None:
        n_candidates = len(metadata)
    
    # Document candidate set and preselection
    try:
        em.log(f"Documenting candidate set: {len(metadata)} candidates, pre-selection: names={pre_selected_names}, indices={pre_selected_indices}")
        cand_df = metadata.reset_index().rename(columns={"index": "candidate_index"})
        # Ensure name columns are strings to avoid dtype warnings
        if 'shortName' in cand_df.columns:
            cand_df['shortName'] = cand_df['shortName'].astype(str)
        if 'longName' in cand_df.columns:
            cand_df['longName'] = cand_df['longName'].astype(str)
        cand_df["is_preselected"] = False
        
        if pre_selected_names is not None:
            for nm in pre_selected_names:
                mask = (
                    cand_df["shortName"].str.lower() == str(nm).lower()
                ) | cand_df["longName"].str.lower().str.contains(str(nm).lower())
                cand_df.loc[mask, "is_preselected"] = True
        
        if pre_selected_indices is not None:
            if "SheetNumber" in cand_df.columns:
                cand_df.loc[cand_df["SheetNumber"].isin(pre_selected_indices), "is_preselected"] = True
            else:
                cand_df.loc[cand_df["candidate_index"].isin(pre_selected_indices), "is_preselected"] = True
        
        cols = [c for c in ["candidate_index", "SheetNumber", "shortName", "longName", "is_preselected"] if c in cand_df.columns]
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
        "trial_number", "datetime_start", "datetime_complete", "duration_sec",
        "value", "a", "b", "c", "min_distance_km", "n_samples", "state"
    ]
    trial_writer = IncrementalCSVWriter(trials_csv_path, fieldnames=trial_fieldnames, buffer_size=50)
    trial_buffer = TrialBuffer(trial_writer)
    
    em.log(f"Initialized incremental trial writer: {trials_csv_path}")
    
    # Determine storage URL if not provided
    if storage is None:
        db_path = em.run_dir / "optuna_study.db"
        storage = f"sqlite:///{db_path}"
        em.log(f"Using persistent storage: {storage}")

    # Create Optuna study
    sampler = get_optuna_sampler(sampler_name, seed=seed)
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    
    # Validate n_samples configuration
    has_fixed = n_samples is not None
    has_range = (n_samples_min is not None) and (n_samples_max is not None)
    if has_fixed and has_range:
        raise ValueError("Specify either --n-samples OR the range --n-samples-min/--n-samples-max, not both.")
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
                "duration_sec": trial.duration.total_seconds() if trial.duration else None,
                "value": trial.value,
                "a": trial.params.get("a"),
                "b": trial.params.get("b"),
                "c": trial.params.get("c"),
                "min_distance_km": trial.params.get("min_distance_km"),
                "n_samples": trial.params.get("n_samples"),
                "state": str(trial.state),
            }
            trial_buffer.add_trial(trial_dict)
            
            # Flush every 100 trials
            if (trial.number + 1) % 100 == 0:
                trial_buffer.flush_to_csv()
                stats = trial_buffer.get_stats()
                em.log(f"Trial {trial.number + 1}/{n_trials} - Best: {study.best_value:.4f} - Stats: {stats}")
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
    
    # Save best trial
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
            "datetime": best.datetime_complete.isoformat() if best.datetime_complete else None,
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
                "min_distance_km": int(best.params.get("min_distance_km", min_distance_km)),
                "n_samples": int(best.params.get("n_samples", n_samples or 34)),
                "pre_selected_names": pre_selected_names,
                "pre_selected_indices": pre_selected_indices,
            }
        }
        em.save_config("best_selection", best_config)
        em.log(f"Best trial: #{best.number}, value={best.value:.4f}")
    except Exception as e:
        em.log(f"Error saving best trial: {e}", level="error")
    
    # Mark stage complete
    em.mark_stage_complete(
        "optuna",
        summary={
            "n_trials_completed": len(study.trials),
            "best_value": study.best_value,
            "best_trial_number": study.best_trial.number if study.best_trial else None,
        }
    )
    
    # Save manifest
    em.save_manifest()
    em.mark_complete(success=True)
    
    print(em.summary())
    
    return study, em


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna optimization with experiment versioning")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--n-candidates", type=int, default=None, help="Number of candidates (default: all)")
    parser.add_argument("--dim", type=int, default=256, help="Feature dimension")
    parser.add_argument("--n-samples", type=int, default=None, help="Fixed number of samples to select")
    parser.add_argument("--n-samples-min", type=int, default=None, help="Lower bound for optimizing n_samples")
    parser.add_argument("--n-samples-max", type=int, default=None, help="Upper bound for optimizing n_samples")
    parser.add_argument("--min-distance-km", type=int, default=40, help="Minimum distance constraint")
    parser.add_argument("--min-distance-min", type=int, default=None, help="Lower bound for min_distance optimization")
    parser.add_argument("--min-distance-max", type=int, default=None, help="Upper bound for min_distance optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pre-names", type=str, nargs="*", default=None, help="Pre-selected tile names (e.g. Hamburg)")
    parser.add_argument("--pre-indices", type=int, nargs="*", default=None, help="Pre-selected tile indices")
    parser.add_argument("--sampler", choices=["tpe", "qmc", "cmaes"], default="qmc", help="Sampler type")
    parser.add_argument("--exp-name", type=str, default="optuna", help="Experiment identifier for versioning")
    parser.add_argument("--exp-desc", type=str, default="", help="Experiment description")
    parser.add_argument("--hamburg", action="store_true", help="Convenience flag: pre-select Hamburg")
    parser.add_argument("--KDR146", action="store_true", help="Convenience flag: pre-select KDR_146")
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///path/to/db). Defaults to sqlite in run dir.")

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
        min_distance_km=args.min_distance_km,
        min_distance_min=args.min_distance_min,
        min_distance_max=args.min_distance_max,
        seed=args.seed,
        sampler_name=args.sampler,
        pre_selected_names=args.pre_names,
        pre_selected_indices=args.pre_indices,
        exp_name=args.exp_name,
        exp_description=args.exp_desc,
        storage=args.optuna_storage,
    )
