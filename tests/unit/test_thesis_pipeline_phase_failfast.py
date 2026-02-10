"""Fail-fast semantics for thesis pipeline phases."""

from __future__ import annotations

import json
from pathlib import Path



def _config_text() -> str:
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


def test_strict_scientific_blocks_later_phases_after_phase1_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataselector.workflows import thesis_pipeline as mod

    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "pipeline_config.yaml").write_text(
        _config_text(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    calls = {"optuna": 0, "validation": 0}

    def _boom_exploration(**_kwargs):
        raise RuntimeError("phase1 boom")

    def _fake_optuna(**_kwargs):
        calls["optuna"] += 1
        return object()

    def _fake_validation(**_kwargs):
        calls["validation"] += 1

    monkeypatch.setattr("dataselector.workflows.tune_weights.run_exploration", _boom_exploration)
    monkeypatch.setattr("dataselector.workflows.optuna_optimize.run_optuna", _fake_optuna)
    monkeypatch.setattr(
        "dataselector.workflows.validation.validate_pareto_candidates",
        _fake_validation,
    )
    monkeypatch.setattr(
        "dataselector.workflows.generate_reports.generate_thesis_final_report",
        lambda **_kwargs: None,
    )

    out_dir = tmp_path / "outputs"
    ok = mod.run_thesis_pipeline(
        n_lhs=4,
        n_trials=2,
        skip_exploration=False,
        skip_optimization=False,
        skip_validation=False,
        strict_scientific=True,
        dry_run=False,
        output_dir=out_dir,
    )

    assert ok is False
    assert calls["optuna"] == 0
    assert calls["validation"] == 0

    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    phase_status = meta["extra"]["phase_status"]
    assert phase_status["phase1_exploration"] == "failed"
    assert phase_status["phase2_optimization"] == "skipped_due_to_prior_failure"
    assert phase_status["phase3_validation"] == "skipped_due_to_prior_failure"
