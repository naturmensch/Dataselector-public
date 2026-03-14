from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataselector.data import io as data_io
from dataselector.data import metadata_source
from dataselector.workflows.validation import validate_pareto_candidates


@pytest.mark.slow
def _make_pareto_csv(tmp_path):
    df = pd.DataFrame(
        [
            {"alpha": 0.7, "beta": 0.15, "gamma": 0.15},
            {"alpha": 0.5, "beta": 0.35, "gamma": 0.15},
        ]
    )
    p = tmp_path / "pareto.csv"
    df.to_csv(p, index=False)
    # create fake outputs for features and metadata in a temp outdir
    out = tmp_path / "outputs"
    out.mkdir()
    np.save(out / "features.npy", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    pd.DataFrame(
        {
            "longName": ["a.png", "b.png", "c.png"],
            "ul_x": [9.9, 10.9, 11.9],
            "ul_y": [50.1, 51.1, 52.1],
            "lr_x": [10.1, 11.1, 12.1],
            "lr_y": [49.9, 50.9, 51.9],
            "year": [1900, 1914, 1918],
            "image_path": ["a", "b", "c"],
        }
    ).to_csv(out / "metadata.csv", index=False)

    # Canonical production metadata source used by validation workflow.
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "longName": ["a.png", "b.png", "c.png"],
            "ul_x": [9.9, 10.9, 11.9],
            "ul_y": [50.1, 51.1, 52.1],
            "lr_x": [10.1, 11.1, 12.1],
            "lr_y": [49.9, 50.9, 51.9],
            "year": [1900, 1914, 1918],
            "image_path": ["a", "b", "c"],
        }
    ).to_csv(data_dir / "new_all_tiles.csv", index=False)
    return str(p), str(out)


def _patch_validation_data(monkeypatch):
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    metadata = pd.DataFrame(
        {
            "longName": ["a.png", "b.png", "c.png"],
            "shortName": ["KDR_146", "KDR_002", "KDR_003"],
            "city": ["Hamburg", "B", "C"],
            "ul_x": [9.9, 10.9, 11.9],
            "ul_y": [50.1, 51.1, 52.1],
            "lr_x": [10.1, 11.1, 12.1],
            "lr_y": [49.9, 50.9, 51.9],
            "year": [1900, 1914, 1918],
            "image_path": ["a", "b", "c"],
        }
    )
    monkeypatch.setattr(data_io, "load_or_extract_features", lambda *a, **k: features)
    monkeypatch.setattr(data_io, "load_metadata", lambda *a, **k: metadata.copy())
    monkeypatch.setattr(
        metadata_source,
        "assert_canonical_metadata",
        lambda *a, **k: "data/new_all_tiles.csv",
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:k >= N for N \\* N square matrix.*")
def test_validate_small(tmp_path, monkeypatch):
    pareto, outdir = _make_pareto_csv(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_validation_data(monkeypatch)
    # Run validation with small params to be quick and point to temp outdir
    df = validate_pareto_candidates(
        pareto,
        min_distances=[10],
        seeds=[1, 2],
        n_samples=2,
        output_dir=outdir,
        pre_selected_names=["Hamburg"],
        pre_selected_indices=[1],
        replicate_mode="seed_replay",
    )
    assert "n_selected" in df.columns
    assert "pre_selected_names" in df.columns
    assert "pre_selected_indices" in df.columns
    assert len(df) == 4
    assert set(df["pre_selected_names"].astype(str)) == {"['Hamburg']"}
    assert set(df["pre_selected_indices"].astype(str)) == {"[1]"}

    # Check that plots were generated
    plots_dir = Path(outdir) / "plots" / "sel_a0.7_b0.15_g0.15_d10.0_s1"
    assert plots_dir.exists()
    assert (plots_dir / "spatial_distribution.png").exists()
    assert (Path(outdir) / "validation_results_seed_replay.csv").exists()
    assert (Path(outdir) / "validation_summary_stats.csv").exists()
    assert (Path(outdir) / "validation_method_contract.md").exists()


@pytest.mark.slow
def test_validate_requires_resolved_selection_target(tmp_path, monkeypatch):
    """Validation must not silently default n_samples to full metadata length."""
    pareto, outdir = _make_pareto_csv(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_validation_data(monkeypatch)
    # Explicitly keep config unset/null for selection target resolution.
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: null\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="could not resolve selection target n_samples"
    ):
        validate_pareto_candidates(
            pareto,
            min_distances=[10],
            seeds=[1],
            n_samples=None,
            output_dir=outdir,
            replicate_mode="seed_replay",
        )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:k >= N for N \\* N square matrix.*")
def test_validate_bootstrap_mode_writes_expected_outputs(tmp_path, monkeypatch):
    pareto, outdir = _make_pareto_csv(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_validation_data(monkeypatch)
    df = validate_pareto_candidates(
        pareto,
        min_distances=[10],
        seeds=[1, 2],
        n_samples=2,
        output_dir=outdir,
        replicate_mode="bootstrap_candidates",
        n_bootstrap=6,
        bootstrap_sample_frac=1.0,
    )
    assert len(df) == 12  # 2 pareto x 1 dist x 6 bootstrap replicates
    assert set(df["replicate_mode"]) == {"bootstrap_candidates"}
    assert (Path(outdir) / "validation_results_bootstrap.csv").exists()
    assert (Path(outdir) / "validation_summary_stats.csv").exists()
