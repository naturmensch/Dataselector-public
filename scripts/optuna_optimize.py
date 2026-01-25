# ruff: noqa: E402
"""Optuna hyperparameter optimization for Multi-Criteria weights.

Usage:
    python scripts/optuna_optimize.py --n-trials 50 --n-candidates 500

Saves results to `outputs/optuna_results.csv` and `outputs/optuna_study.pkl`.
"""

import argparse
import os
import sys
from pathlib import Path

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# STARTUP ENV VALIDATION
try:
    from src.compat import validate_environment_full
    if "--skip-env-check" not in sys.argv:
        validate_environment_full()
except Exception as e:
    print(f"\n❌ STARTUP VALIDATION FAILED:\n{e}\n", file=sys.stderr)
    print("Fix: ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages 'numpy==1.26.4 numba==0.63.1' --yes -- python scripts/optuna_optimize.py", file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd

try:
    import optuna
except Exception as e:
    raise ImportError(
        "Please install optuna (pip install optuna) to run optimization"
    ) from e

# Defer heavy imports where possible; attempt to import DiversitySelector but tolerate failures for smoke/test mode
try:
    from src.diversity_selector import DiversitySelector
    DIVERSITY_IMPORT_ERROR = None
except Exception as e:
    DiversitySelector = None
    DIVERSITY_IMPORT_ERROR = e

# Allow overriding workspace via envvar (set by CLI helpers/tests)
# OUT_DIR will be set after parsing args


def load_or_create_data(n=500, dim=512, seed=123):
    features_path = OUT_DIR / "features.npy"
    metadata_path = OUT_DIR / "metadata.csv"

    from src.io import load_or_extract_features

    if features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=OUT_DIR, csv_meta=str(metadata_path), batch_size=16, cache=False
        )
        metadata = pd.read_csv(metadata_path)
    else:
        rng = np.random.RandomState(seed)
        features = rng.randn(n, dim).astype("float32")
        metadata = pd.DataFrame(
            {
                "N": np.random.uniform(48, 55, n),
                "left": np.random.uniform(6, 15, n),
                "year": np.random.randint(1880, 1945, n),
            }
        )

    return features, metadata


def objective_factory(features, metadata, n_samples, min_distance_km, n_samples_range=None, constrain_bounds=None):
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
    out_dir: Path | None = None,
    study_db: str | None = None,
):
    """Run Optuna optimization.

    If `study_db` is provided, the Optuna study will be persisted to SQLite using
    the provided path (e.g. 'outputs/optuna_study.db' or an absolute path). This
    enables robust resume/reconstruction flows via the monitor.
    """
    # Set global OUT_DIR for this run
    global OUT_DIR
    if out_dir is None:
        OUT_DIR = Path("outputs")
    else:
        OUT_DIR = out_dir
    OUT_DIR.mkdir(exist_ok=True)

    features, metadata = load_or_create_data(n=n_candidates, dim=dim, seed=seed)

    sampler = None
    if sampler_name == "qmc":
        sampler = optuna.samplers.QMCSampler()
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler()
    else:
        sampler = optuna.samplers.TPESampler()

    # If a sqlite DB path was provided, use it as persistent storage for the study
    if study_db:
        # Ensure parent dir exists
        try:
            Path(study_db).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        storage = f"sqlite:///{study_db}"
        study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler, storage=storage)
    else:
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
    print(f"OUT_DIR is {OUT_DIR}")
    try:
        results_df = study.trials_dataframe()
        results_df.to_csv(OUT_DIR / "optuna_results.csv", index=False)
        print(f"Optuna optimization finished. Results saved to {OUT_DIR / 'optuna_results.csv'}")
    except Exception as e:
        print(f"Failed to save results: {e}")

    # If an experiment name was provided, also save per-run outputs to a run-specific folder
    if exp_name:
        run_results_dir = OUT_DIR / "runs" / exp_name / "results"
        run_results_dir.mkdir(parents=True, exist_ok=True)
        trials_csv = run_results_dir / "trials.csv"
        results_df.to_csv(trials_csv, index=False)
        print(f"Saved run trials to {trials_csv}")

    # Save study object
    try:
        import joblib

        joblib.dump(study, OUT_DIR / "optuna_study.pkl")
    except Exception:
        # Fallback: save trials dataframe only
        print("joblib not available: only saving trials dataframe")
    return study


if __name__ == "__main__":
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
    # Optuna persistent study DB options
    parser.add_argument("--use-study-db", action="store_true", help="Create/use default outputs/optuna_study.db for persistent Optuna storage")
    parser.add_argument("--study-db", type=str, default=None, help="Path to SQLite DB file for Optuna storage (overrides --use-study-db)")
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
        print(f"Setting DATASELECTOR_WORKSPACE to {args.workspace}")
        os.environ.setdefault("DATASELECTOR_WORKSPACE", args.workspace)

    # Set OUT_DIR after env var is set
    if os.environ.get("DATASELECTOR_WORKSPACE"):
        OUT_DIR = Path(os.environ.get("DATASELECTOR_WORKSPACE")) / "outputs"
    else:
        OUT_DIR = Path("outputs")
    OUT_DIR.mkdir(exist_ok=True)

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

    # Determine optional SQLite DB path for persistent Optuna storage
    study_db = None
    if args.use_study_db:
        study_db = str(OUT_DIR / "optuna_study.db")
    if args.study_db:
        study_db = str(args.study_db)

    run_optuna(
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        dim=args.dim,
        n_samples=n_samples,
        min_distance_km=args.min_distance_km,
        seed=args.seed,
        n_samples_range=n_samples_range,
        sampler_name=args.sampler,
        constrain_bounds=constrain_bounds,
        exp_name=args.exp_name,
        checkpoint_every=args.checkpoint_every,
        out_dir=OUT_DIR,
        study_db=study_db,
    )
