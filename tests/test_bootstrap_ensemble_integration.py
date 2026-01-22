import pytest

import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

pytestmark = pytest.mark.integration


def test_bootstrap_main_ensemble_mode(tmp_path, monkeypatch):
    # Skip if optional dependencies are missing at runtime
    pytest.importorskip("torch")
    pytest.importorskip("numba", exc_type=ImportError)

    # Dynamically load the script module to avoid module-level imports after pytest.skip
    ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "bootstrap_pareto_candidates", ROOT / "scripts" / "bootstrap_pareto_candidates.py"
    )
    bpc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bpc)

    # Create tiny pareto csv
    df = pd.DataFrame(
        {
            "alpha": [0.6],
            "beta": [0.2],
            "gamma": [0.2],
            "min_distance_km": [28],
            "n_selected": [10],
        }
    )
    p = tmp_path / "pareto.csv"
    df.to_csv(p, index=False)

    # Monkeypatch data loaders to return tiny synthetic data
    def fake_load_metadata(path):
        n = 50
        rng = np.random.RandomState(2)
        return pd.DataFrame(
            {
                "N": rng.uniform(40, 60, n),
                "left": rng.uniform(6, 15, n),
                "year": rng.randint(1880, 1945, n),
                "lat": rng.uniform(48.0, 54.0, n),
                "lon": rng.uniform(7.0, 15.0, n),
            }
        )

    def fake_load_or_extract_features(out_dir, csv_meta=None, cache=True):
        return np.random.RandomState(1).randn(50, 16)

    monkeypatch.setattr(bpc, "load_metadata", fake_load_metadata)
    monkeypatch.setattr(bpc, "load_or_extract_features", fake_load_or_extract_features)

    # Monkeypatch clustering pipeline to be trivial
    class DummyClustering:
        def __init__(self, n_clusters=8):
            pass

        def fit_transform(self, X):
            return X, np.zeros(X.shape[0], dtype=int)

    monkeypatch.setattr(bpc, "ClusteringPipeline", DummyClustering)

    out = tmp_path / "out.csv"

    # Run main in ensemble mode with small training budget
    bpc.main(
        str(p),
        n_boot=20,
        output_csv=str(out),
        random_seed=1,
        uq_method="ensemble",
        n_ensemble_models=2,
        ensemble_epochs=2,
    )

    # Check that summary file was written
    summary = pd.read_csv(str(out).replace(".csv", "_summary.csv"))
    assert "temporal_std_mean" in summary.columns
    assert "jaccard_mean" in summary.columns
    assert summary.shape[0] == 1
