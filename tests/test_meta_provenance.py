import json

import numpy as np
import pandas as pd

from src import experiments as experiments_module


def test_meta_provenance_includes_versions(tmp_path, monkeypatch):
    # Create minimal metadata CSV
    meta_df = pd.DataFrame(
        {
            "longName": ["a", "b"],
            "N": [0.0, 1.0],
            "left": [0.0, 1.0],
            "year": [1900, 1901],
            "image_path": ["a.png", "b.png"],
        }
    )
    csv_path = tmp_path / "meta.csv"
    meta_df.to_csv(csv_path, index=False)

    # Provide dummy features to avoid heavy extraction
    dummy_feats = np.ones((2, 8))
    monkeypatch.setattr(
        experiments_module,
        "extract_features",
        lambda meta, batch_size=None: dummy_feats,
    )

    # Make selection deterministic and non-empty by patching DiversitySelector.select
    import dataselector.selection.diversity_selector as ds_mod

    def _fake_select(self, **kwargs):
        self.selected_indices = np.array([0], dtype=int)
        return self.selected_indices

    monkeypatch.setattr(ds_mod.DiversitySelector, "select", _fake_select)

    out_dir = tmp_path / "out"
    runner = experiments_module.ExperimentRunner(output_dir=str(out_dir))

    runner.run_weight_sweep(
        csv_meta=str(csv_path),
        n_samples=1,
        n_clusters=1,
        weight_combinations=[(0.7, 0.15, 0.15)],
        min_distance_km=0.0,
        max_runs=1,
        patience=None,
    )

    meta_json_path = out_dir / "meta.json"
    assert meta_json_path.exists(), "meta.json should be created"

    meta = json.loads(meta_json_path.read_text())

    # Assert provenance fields are present
    assert "git_commit" in meta
    assert "python_version" in meta
    assert "numpy_version" in meta
    assert "torch_version" in meta
    assert "pip_packages" in meta

    # Basic sanity checks
    assert isinstance(meta["python_version"], (str, type(None)))
    assert isinstance(meta["numpy_version"], (str, type(None)))
    assert isinstance(meta["torch_version"], (str, type(None)))
    assert isinstance(meta["pip_packages"], (dict, type(None)))
