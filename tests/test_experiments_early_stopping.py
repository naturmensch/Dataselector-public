import numpy as np
import pandas as pd

from dataselector.pipeline.experiments import ExperimentRunner


def _fake_meta():
    return pd.DataFrame(
        {
            "ul_x": [9.95, 10.95, 11.95],
            "ul_y": [50.05, 51.05, 52.05],
            "lr_x": [10.05, 11.05, 12.05],
            "lr_y": [49.95, 50.95, 51.95],
            "year": [1900, 1914, 1918],
            "image_path": ["a", "b", "c"],
        }
    )


def test_early_stopping(monkeypatch, tmp_path):
    runner = ExperimentRunner(output_dir=str(tmp_path))
    canonical_csv = tmp_path / "data" / "new_all_tiles.csv"
    canonical_csv.parent.mkdir(parents=True, exist_ok=True)
    canonical_csv.write_text("id\n1\n", encoding="utf-8")

    # Monkeypatch loading and feature extraction to be lightweight
    monkeypatch.setattr(
        "dataselector.pipeline.experiments.load_metadata", lambda p: _fake_meta()
    )
    monkeypatch.setattr(
        "dataselector.pipeline.experiments.extract_features",
        lambda meta, batch_size=16: np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )

    # Monkeypatch DiversitySelector.select to always return the same selection (no improvements)
    class DummyDS:
        def __init__(self, *args, **kwargs):
            pass

        def select(self, **kwargs):
            return [0, 1]

        def export_selection(self, metadata, out_csv):
            df = metadata.iloc[[0, 1]]
            df.to_csv(out_csv, index=False)

    monkeypatch.setattr("dataselector.pipeline.experiments.DiversitySelector", DummyDS)
    monkeypatch.setattr(
        "dataselector.pipeline.experiments.compute_metrics",
        lambda *args, **kwargs: {
            "clusters_covered": 1,
            "temporal_std": 0.0,
            "spatial_mean_km": 0.0,
            "spatial_min_km": 0.0,
            "n_selected": 2,
            "temporal_range": 0,
            "wwi_percent": 0.0,
        },
    )

    weight_combinations = [(0.7, 0.15, 0.15)] * 10

    df = runner.run_weight_sweep(
        csv_meta=str(canonical_csv),
        n_samples=2,
        weight_combinations=weight_combinations,
        n_clusters=2,
        batch_size=1,
        min_distance_km=0.0,
        patience=2,
        max_runs=10,
    )

    # Expect that we stopped early: results length should be <= patience+1
    assert len(df) <= 3
