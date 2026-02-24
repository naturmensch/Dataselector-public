import numpy as np
import pandas as pd
import pytest

from dataselector.data import io as io_mod
from dataselector.pipeline import experiments as experiments_mod


@pytest.mark.integration
def test_cache_migration_and_load(tmp_path, monkeypatch):
    """Legacy features.npy is ignored; immutable hash cache is built from extraction."""
    out = tmp_path / "outputs"
    out.mkdir()

    legacy_feats = np.arange(12).reshape(3, 4)
    np.save(out / "features.npy", legacy_feats)

    csv = tmp_path / "data.csv"
    csv.write_text("id\n1\n2\n3\n", encoding="utf-8")

    meta = pd.DataFrame({"image_path": ["a.png", "b.png", "c.png"]})
    extracted = np.full((3, 4), 7.0, dtype=np.float32)
    monkeypatch.setattr(io_mod, "load_metadata", lambda _csv: meta)
    monkeypatch.setattr(
        io_mod,
        "_extract_features_with_provenance",
        lambda _metadata, **_: (extracted, {}),
    )

    loaded = io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        batch_size=4,
        cache=True,
        cache_scope="run_local",
    )
    assert np.array_equal(loaded, extracted)

    immutable_cache_files = list(out.glob("*/features.npy"))
    assert immutable_cache_files
    # Legacy file remains untouched; cache contract no longer migrates it.
    assert (out / "features.npy").exists()


@pytest.mark.integration
def test_feature_cache_validation_reextracts(tmp_path, monkeypatch):
    """Stale legacy cache is ignored and a valid immutable hash cache is written."""
    out = tmp_path / "outputs"
    out.mkdir()
    np.save(out / "features.npy", np.zeros((2, 16), dtype=np.float32))

    csv = out / "metadata.csv"
    csv.write_text(
        "longName,shortName,ul_x,ul_y,lr_x,lr_y,image_path,image_filename,year\n"
        + "\n".join(
            [
                f"A,i,{499950 + i*100},{5900050 + i*100},{500050 + i*100},{5899950 + i*100},a.png,a.png,190{i}"
                for i in range(4)
            ]
        ),
        encoding="utf-8",
    )

    meta = pd.DataFrame({"image_path": ["a.png", "b.png", "c.png", "d.png"]})
    monkeypatch.setattr(io_mod, "load_metadata", lambda _csv: meta)
    monkeypatch.setattr(
        io_mod,
        "_extract_features_with_provenance",
        lambda metadata, **_: (np.zeros((len(metadata), 16), dtype=np.float32), {}),
    )

    feats = io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        batch_size=4,
        cache=True,
        cache_scope="run_local",
    )
    assert feats.shape == (4, 16)

    assert list(out.glob("*/features.npy"))


@pytest.mark.integration
def test_pipeline_smoke_small(tmp_path, monkeypatch):
    """Small smoke run for ExperimentRunner with lightweight stubs."""
    root = tmp_path / "repo"
    root.mkdir()
    data_dir = root / "data"
    outputs = root / "outputs"
    data_dir.mkdir()
    (data_dir / "images").mkdir()
    outputs.mkdir()

    csv_meta = data_dir / "new_all_tiles.csv"
    rows = ["longName,shortName,ul_x,ul_y,lr_x,lr_y,image_path,image_filename,year"]
    for i in range(4):
        img = data_dir / "images" / f"KDR_{i:03d}.png"
        img.write_bytes(b"png")
        cx = 500000.0 + i * 100.0
        cy = 5900000.0 + i * 100.0
        rows.append(
            f"KDR_{i:03d}.png,KDR_{i:03d},{cx-50},{cy+50},{cx+50},{cy-50},{img},{img.name},{1890+i}"
        )
    csv_meta.write_text("\n".join(rows), encoding="utf-8")
    metadata = pd.read_csv(csv_meta)

    monkeypatch.setattr(io_mod, "load_metadata", lambda _csv: metadata.copy())
    monkeypatch.setattr(
        io_mod,
        "_extract_features_with_provenance",
        lambda _metadata, **_: (
            np.zeros((len(_metadata), 16), dtype=np.float32),
            {},
        ),
    )

    io_mod.load_or_extract_features(
        out_dir=outputs,
        csv_meta=str(csv_meta),
        batch_size=4,
        cache=True,
        cache_scope="run_local",
    )
    assert list(outputs.glob("*/features.npy"))

    monkeypatch.setattr(experiments_mod, "load_metadata", lambda _csv: metadata.copy())
    monkeypatch.setattr(
        experiments_mod,
        "extract_features",
        lambda _metadata, batch_size=16: np.zeros(
            (len(_metadata), 16), dtype=np.float32
        ),
    )

    class FakeKMeans:
        def __init__(self, n_clusters=8, random_state=None):
            self.n_clusters = n_clusters
            self.random_state = random_state

        def fit_predict(self, X):
            n = X.shape[0]
            return np.arange(n) % max(1, self.n_clusters)

    class FakeDS:
        def __init__(self, n_samples=5, **kwargs):
            self.n_samples = n_samples

        def select(self, features, metadata, *args, **kwargs):
            return list(range(min(self.n_samples, len(features))))

        def export_selection(self, metadata, out_file):
            return pd.DataFrame(
                {"selection_rank": list(range(min(self.n_samples, len(metadata))))}
            )

    monkeypatch.setattr(experiments_mod, "KMeans", FakeKMeans)
    monkeypatch.setattr(experiments_mod, "DiversitySelector", FakeDS)
    monkeypatch.setattr(
        experiments_mod,
        "compute_metrics",
        lambda selected_idx, metadata, cluster_labels, features: {
            "n_selected": len(selected_idx),
            "temporal_std": 0.0,
            "spatial_mean_km": 0.0,
            "clusters_covered": int(len(set(cluster_labels[selected_idx]))),
        },
    )

    runner = experiments_mod.ExperimentRunner(output_dir=str(outputs / "tuning_fast"))
    runner.run_weight_sweep(
        csv_meta=str(csv_meta),
        n_samples=3,
        weight_combinations=[(0.7, 0.2, 0.1), (0.6, 0.3, 0.1)],
        n_clusters=2,
        batch_size=4,
        min_distance_km=0.0,
        patience=1,
        max_runs=2,
    )

    assert (outputs / "tuning_fast" / "tuning_results.csv").exists()
    assert (outputs / "tuning_fast" / "meta.json").exists()
