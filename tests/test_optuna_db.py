import sqlite3
from pathlib import Path

import numpy as np

from scripts.optuna_optimize import run_optuna, DIVERSITY_IMPORT_ERROR, DiversitySelector


class DummySelector:
    def __init__(self, n_samples, use_multi_criteria=True):
        self.n_samples = n_samples

    def select(self, features, metadata, spatial_constraint, min_distance_km, alpha_visual, beta_spatial, gamma_temporal):
        n = min(self.n_samples, len(features))
        return list(range(n))

    def _calculate_diversity_score(self, selected_features):
        # simple proxy for diversity
        return float(np.mean(np.var(selected_features, axis=0)))


def test_optuna_creates_sqlite_db(tmp_path, monkeypatch):
    # If real DiversitySelector is available, we still prefer to inject a dummy for speed
    monkeypatch.setattr("scripts.optuna_optimize.DiversitySelector", DummySelector, raising=False)

    out_dir = tmp_path / "outputs"
    study_db = tmp_path / "optuna_study.db"

    # Provide synthetic features/metadata to avoid heavy imports (umap/numba)
    features = np.random.RandomState(1).randn(20, 16).astype("float32")
    import pandas as _pd

    metadata = _pd.DataFrame({
        "N": np.random.uniform(48, 55, 20),
        "left": np.random.uniform(6, 15, 20),
        "year": np.random.randint(1880, 1945, 20),
    })

    monkeypatch.setattr("scripts.optuna_optimize.load_or_create_data", lambda n, dim, seed: (features, metadata), raising=False)

    study = run_optuna(
        n_trials=2,
        n_candidates=20,
        dim=16,
        n_samples=5,
        min_distance_km=5,
        seed=1,
        sampler_name="tpe",
        checkpoint_every=0,
        out_dir=out_dir,
        study_db=str(study_db),
    )

    assert study_db.exists(), f"Expected sqlite DB at {study_db}"

    # Basic integrity check using sqlite pragma
    conn = sqlite3.connect(str(study_db))
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check;")
    rows = cur.fetchall()
    conn.close()
    assert rows and rows[0][0] == "ok"
