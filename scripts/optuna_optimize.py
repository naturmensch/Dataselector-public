"""Optuna hyperparameter optimization for Multi-Criteria weights.

Usage:
    python scripts/optuna_optimize.py --n-trials 50 --n-candidates 500

Saves results to `outputs/optuna_results.csv` and `outputs/optuna_study.pkl`.
"""
import os
import sys
import argparse
from pathlib import Path

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

try:
    import optuna
except Exception as e:
    raise ImportError("Please install optuna (pip install optuna) to run optimization") from e

from src.diversity_selector import DiversitySelector

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)


def load_or_create_data(n=500, dim=512, seed=123):
    features_path = OUT_DIR / 'features.npy'
    metadata_path = OUT_DIR / 'metadata.csv'

    if features_path.exists() and metadata_path.exists():
        features = np.load(features_path)
        metadata = pd.read_csv(metadata_path)
    else:
        rng = np.random.RandomState(seed)
        features = rng.randn(n, dim).astype('float32')
        metadata = pd.DataFrame({
            'N': np.random.uniform(48, 55, n),
            'left': np.random.uniform(6, 15, n),
            'year': np.random.randint(1880, 1945, n)
        })

    return features, metadata


def objective_factory(features, metadata, n_samples, min_distance_km):
    def objective(trial: optuna.trial.Trial):
        # Sample raw weights and normalize (ensures sum=1 and non-negative)
        a = trial.suggest_float('a', 0.01, 1.0)
        b = trial.suggest_float('b', 0.01, 1.0)
        c = trial.suggest_float('c', 0.01, 1.0)
        total = a + b + c
        alpha = a / total
        beta = b / total
        gamma = c / total

        # Use conservative bounds for min_distance based on dataset grid (median ≈ 28km).
        # Limit search to [0, 60] km to avoid overly restrictive values that prevent selecting enough samples.
        min_dist = trial.suggest_int('min_distance_km', 0, 60)

        selector = DiversitySelector(n_samples=n_samples, use_multi_criteria=True)
        selected = selector.select(
            features,
            metadata,
            spatial_constraint=True,
            min_distance_km=min_dist,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma
        )

        # Compute metrics
        n_selected = len(selected)
        if n_selected == 0:
            return 0.0

        diversity = selector._calculate_diversity_score(features[selected])
        spatial_spread = metadata.loc[selected, ['N', 'left']].std().mean()

        # Composite objective (maximize)
        score = diversity * spatial_spread

        # Log intermediate values
        trial.set_user_attr('alpha', float(alpha))
        trial.set_user_attr('beta', float(beta))
        trial.set_user_attr('gamma', float(gamma))
        trial.set_user_attr('min_distance_km', int(min_dist))
        trial.set_user_attr('n_selected', int(n_selected))
        trial.set_user_attr('diversity', float(diversity))
        trial.set_user_attr('spatial_spread', float(spatial_spread))

        return float(score)

    return objective


def run_optuna(n_trials=50, n_candidates=500, dim=512, n_samples=34, min_distance_km=28, seed=42, study_name='kdr100_opt'):
    features, metadata = load_or_create_data(n=n_candidates, dim=dim, seed=seed)

    study = optuna.create_study(direction='maximize', study_name=study_name)
    objective = objective_factory(features, metadata, n_samples=n_samples, min_distance_km=min_distance_km)

    study.optimize(objective, n_trials=n_trials)

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(OUT_DIR / 'optuna_results.csv', index=False)

    # Save study object
    try:
        import joblib
        joblib.dump(study, OUT_DIR / 'optuna_study.pkl')
    except Exception:
        # Fallback: save trials dataframe only
        print('joblib not available: only saving trials dataframe')

    print('Optuna optimization finished. Results saved to outputs/optuna_results.csv')
    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--n-candidates', type=int, default=500)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=34)
    parser.add_argument('--min-distance-km', type=int, default=28)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    run_optuna(
        n_trials=args.n_trials,
        n_candidates=args.n_candidates,
        dim=args.dim,
        n_samples=args.n_samples,
        min_distance_km=args.min_distance_km,
        seed=args.seed
    )
