from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from dataselector.workflows.optuna_optimize import run_optuna


def test_optuna_creates_sqlite_db(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    study_db = tmp_path / "optuna_study.db"
    metadata_csv = tmp_path / "metadata.csv"

    metadata = pd.DataFrame(
        {
            "ul_x": np.linspace(6.0, 15.0, 20) - 0.05,
            "ul_y": np.linspace(48.0, 55.0, 20) + 0.05,
            "lr_x": np.linspace(6.0, 15.0, 20) + 0.05,
            "lr_y": np.linspace(48.0, 55.0, 20) - 0.05,
            "year": np.arange(1900, 1920),
        }
    )
    metadata.to_csv(metadata_csv, index=False)

    features = np.random.RandomState(1).randn(20, 16).astype("float32")

    monkeypatch.setattr(
        "dataselector.workflows.optuna_optimize.load_or_create_data",
        lambda out_dir, n, dim, seed: (features, metadata),
        raising=True,
    )

    def fake_objective_factory(*args, **kwargs):
        def _objective(_trial):
            return 1.0

        return _objective

    monkeypatch.setattr(
        "dataselector.workflows.optuna_optimize.objective_factory",
        fake_objective_factory,
        raising=True,
    )

    run_optuna(
        n_trials=2,
        n_candidates=20,
        dim=16,
        n_samples=5,
        metadata_path=metadata_csv,
        seed=1,
        sampler_name="tpe",
        checkpoint_every=0,
        out_dir=out_dir,
        study_db=str(study_db),
    )

    assert study_db.exists(), f"Expected sqlite DB at {study_db}"

    conn = sqlite3.connect(str(study_db))
    cur = conn.cursor()
    cur.execute("PRAGMA integrity_check;")
    rows = cur.fetchall()
    conn.close()
    assert rows and rows[0][0] == "ok"
