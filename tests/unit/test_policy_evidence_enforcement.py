"""Policy/computation enforcement for critical thesis parameters."""

from __future__ import annotations

from pathlib import Path

import pytest



def _config_text_without_computed_artifact() -> str:
    return (
        "feature_extraction:\n"
        "  model: dinov2\n"
        "  batch_size: 8\n"
        "  crop_size: [2048, 2048]\n"
        "  device: auto\n"
        "clustering:\n"
        "  n_clusters: 8\n"
        "  umap_components: 2\n"
        "  umap_n_neighbors: 15\n"
        "  umap_min_dist: 0.1\n"
        "  umap_random_state: 42\n"
        "  umap_n_jobs: 1\n"
        "selection:\n"
        "  n_samples: 24\n"
        "  min_distance_km: 28.5\n"
        "  metric: euclidean\n"
        "  alpha_visual: 0.4\n"
        "  beta_spatial: 0.3\n"
        "  gamma_temporal: 0.3\n"
        "  spatial_constraint: true\n"
        "  use_multi_criteria: true\n"
        "  use_constraint_integration: false\n"
        "  random_state: 42\n"
        "  optuna_sampler: tpe\n"
        "  exploration_sampler: lhs\n"
    )


def test_compute_params_requires_artifact_for_critical_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "pipeline_config.yaml").write_text(
        _config_text_without_computed_artifact(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="requires computed artifact"):
        run_thesis_pipeline(
            n_lhs=5,
            n_trials=2,
            compute_params=True,
            no_auto_continue=True,
            skip_exploration=True,
            skip_optimization=True,
            skip_validation=True,
            dry_run=True,
            output_dir=tmp_path / "outputs",
            strict_scientific=True,
        )
