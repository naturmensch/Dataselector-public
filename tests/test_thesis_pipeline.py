"""Tests for thesis pipeline workflow."""

import json
from pathlib import Path

import pytest


def _minimal_resolved_config(
    *,
    n_samples: int | None,
    min_distance_km: float = 28.5,
    optuna_sampler: str = "tpe",
    exploration_sampler: str = "lhs",
) -> str:
    n_samples_literal = "null" if n_samples is None else str(int(n_samples))
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
        "  umap_random_state: 42\n"
        "  umap_n_jobs: 1\n"
        "selection:\n"
        f"  n_samples: {n_samples_literal}\n"
        f"  min_distance_km: {min_distance_km}\n"
        "  metric: euclidean\n"
        "  alpha_visual: 0.4\n"
        "  beta_spatial: 0.3\n"
        "  gamma_temporal: 0.3\n"
        "  spatial_constraint: true\n"
        "  use_multi_criteria: true\n"
        "  use_constraint_integration: false\n"
        "  random_state: 42\n"
        f"  optuna_sampler: {optuna_sampler}\n"
        f"  exploration_sampler: {exploration_sampler}\n"
    )


def test_thesis_pipeline_importable():
    """Test that thesis_pipeline module can be imported."""
    from dataselector.workflows import thesis_pipeline

    assert hasattr(thesis_pipeline, "run_thesis_pipeline")


def test_thesis_pipeline_signature():
    """Test run_thesis_pipeline function signature."""
    import inspect

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    sig = inspect.signature(run_thesis_pipeline)
    params = list(sig.parameters.keys())

    expected_params = [
        "n_lhs",
        "n_samples",
        "n_trials",
        "skip_exploration",
        "skip_optimization",
        "skip_validation",
        "dry_run",
        "output_dir",
        "execution_profile",
        "seed",
        "pre_names",
        "pre_indices",
        "hamburg",
        "validation_seeds",
        "validation_min_distances",
    ]

    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


def test_run_thesis_pipeline_dry_run_skip_validation(tmp_path):
    """Regression guard for CLI dry-run path without validation imports."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    success = run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_validation=True,
        dry_run=True,
        output_dir=tmp_path / "outputs",
        execution_profile="thesis_repro",
        seed=7,
    )

    assert success is True
    metadata_path = tmp_path / "outputs" / "run_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution_profile"] == "thesis_repro"
    assert metadata["seed"] == 7


def test_run_thesis_pipeline_fails_without_resolvable_n_samples(tmp_path, monkeypatch):
    """Thesis path must fail fast if n_samples cannot be resolved from any source."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (ws / "config" / "pipeline_config.yaml").write_text(
        _minimal_resolved_config(n_samples=None),
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)

    with pytest.raises(
        ValueError, match="could not resolve selection target n_samples"
    ):
        run_thesis_pipeline(
            n_lhs=5,
            n_samples=None,
            n_trials=2,
            skip_validation=True,
            dry_run=True,
            output_dir=ws / "outputs",
            execution_profile="thesis_repro",
            seed=7,
        )


def test_run_thesis_pipeline_passes_metadata_path_to_stages(tmp_path, monkeypatch):
    """Non-dry-run must pass metadata_path to exploration and Optuna workflows."""
    from dataselector.workflows import thesis_pipeline as mod

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (ws / "config" / "pipeline_config.yaml").write_text(
        _minimal_resolved_config(n_samples=34),
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)

    calls: dict[str, str] = {}

    def fake_exploration(**kwargs):
        calls["exploration_metadata_path"] = str(kwargs["metadata_path"])
        calls["exploration_selection_n_samples"] = kwargs["selection_n_samples"]
        return [], ws / "dummy.csv"

    def fake_optuna(**kwargs):
        calls["optuna_metadata_path"] = str(kwargs["metadata_path"])
        calls["optuna_n_samples"] = kwargs["n_samples"]
        return object()

    monkeypatch.setattr(
        "dataselector.workflows.tune_weights.run_exploration", fake_exploration
    )
    monkeypatch.setattr(
        "dataselector.workflows.optuna_optimize.run_optuna", fake_optuna
    )
    monkeypatch.setattr(
        "dataselector.workflows.generate_reports.generate_thesis_final_report",
        lambda **_kwargs: None,
    )
    success = mod.run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_validation=True,
        dry_run=False,
        output_dir=ws / "outputs",
        seed=11,
    )

    assert success is True
    assert calls["exploration_metadata_path"].endswith("data/new_all_tiles.csv")
    assert calls["optuna_metadata_path"].endswith("data/new_all_tiles.csv")
    assert calls["exploration_selection_n_samples"] == 34
    assert calls["optuna_n_samples"] == 34


def test_run_thesis_pipeline_phase4_single_run_report(tmp_path):
    """Phase 4 summary must work for single-run output_dir contract."""
    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    success = run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_exploration=True,
        skip_optimization=True,
        skip_validation=True,
        dry_run=False,
        output_dir=tmp_path / "outputs",
        execution_profile="thesis_repro",
        seed=42,
    )

    assert success is True
    assert (tmp_path / "outputs" / "THESIS_PIPELINE_REPORT.md").exists()


def test_run_thesis_pipeline_fails_without_resolvable_optuna_sampler(
    tmp_path, monkeypatch
):
    """Canonical thesis path must not silently fall back to implicit sampler defaults."""
    from dataselector.workflows import thesis_pipeline as mod

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    # Deliberately omit selection.optuna_sampler and sampler artifacts.
    (ws / "config" / "pipeline_config.yaml").write_text(
        _minimal_resolved_config(n_samples=24).replace("  optuna_sampler: tpe\n", ""),
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)

    with pytest.raises(ValueError, match="Optuna sampler unresolved"):
        mod.run_thesis_pipeline(
            n_lhs=5,
            n_trials=2,
            skip_exploration=True,
            skip_optimization=False,
            skip_validation=True,
            dry_run=False,
            output_dir=ws / "outputs",
            seed=11,
        )


def test_run_thesis_pipeline_preselection_forwarding_and_dedup(tmp_path, monkeypatch):
    """Hamburg shortcut and preselection must be deduped and forwarded end-to-end."""
    from dataselector.workflows import thesis_pipeline as mod

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (ws / "config" / "pipeline_config.yaml").write_text(
        _minimal_resolved_config(n_samples=34),
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)

    calls = {}

    def fake_exploration(**kwargs):
        calls["exploration"] = kwargs
        return [], ws / "dummy.csv"

    def fake_optuna(**kwargs):
        calls["optuna"] = kwargs
        return object()

    def fake_validate(**kwargs):
        calls["validation"] = kwargs
        return None

    monkeypatch.setattr(
        "dataselector.workflows.tune_weights.run_exploration", fake_exploration
    )
    monkeypatch.setattr(
        "dataselector.workflows.optuna_optimize.run_optuna", fake_optuna
    )
    monkeypatch.setattr(
        "dataselector.workflows.validation.validate_pareto_candidates", fake_validate
    )
    monkeypatch.setattr(
        "dataselector.workflows.generate_reports.generate_thesis_final_report",
        lambda **_kwargs: None,
    )
    pareto_dir = ws / "outputs" / "tuning_weights" / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    (pareto_dir / "pareto_solutions.csv").write_text(
        "alpha,beta,gamma\n0.7,0.2,0.1\n",
        encoding="utf-8",
    )

    success = mod.run_thesis_pipeline(
        n_lhs=5,
        n_samples=37,
        n_trials=2,
        skip_exploration=False,
        skip_optimization=False,
        skip_validation=False,
        dry_run=False,
        output_dir=ws / "outputs",
        seed=11,
        pre_names=["Kiel", "Hamburg", "Kiel"],
        pre_indices=[3, 3, 1],
        hamburg=True,
        validation_seeds=[42, 99],
        validation_min_distances=[28.5, 35.0],
    )

    assert success is True
    assert calls["exploration"]["pre_names"] == ["Kiel", "Hamburg"]
    assert calls["exploration"]["pre_indices"] == [3, 1]
    assert calls["exploration"]["selection_n_samples"] == 37
    assert calls["optuna"]["pre_selected_names"] == ["Kiel", "Hamburg"]
    assert calls["optuna"]["pre_selected_indices"] == [3, 1]
    assert calls["optuna"]["n_samples"] == 37
    assert calls["validation"]["pre_selected_names"] == ["Kiel", "Hamburg"]
    assert calls["validation"]["pre_selected_indices"] == [3, 1]
    assert calls["validation"]["n_samples"] == 37
    assert calls["validation"]["seeds"] == [42, 99]
    assert calls["validation"]["min_distances"] == [28.5, 35.0]

    run_meta = json.loads((ws / "outputs" / "run_metadata.json").read_text("utf-8"))
    assert run_meta["extra"]["pre_selected_names"] == ["Kiel", "Hamburg"]
    assert run_meta["extra"]["pre_selected_indices"] == [3, 1]
    assert run_meta["extra"]["hamburg_shortcut"] is True
    assert run_meta["extra"]["n_samples"] == 37
    assert run_meta["extra"]["validation_seeds"] == [42, 99]
    assert run_meta["extra"]["validation_min_distances"] == [28.5, 35.0]


def test_run_thesis_pipeline_writes_central_parameter_provenance(tmp_path, monkeypatch):
    """Central thesis path must snapshot critical parameters + provenance pre-run."""
    from dataselector.workflows import thesis_pipeline as mod

    ws = tmp_path
    (ws / "data").mkdir(parents=True, exist_ok=True)
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )
    (ws / "config" / "pipeline_config.yaml").write_text(
        _minimal_resolved_config(n_samples=24),
        encoding="utf-8",
    )
    monkeypatch.chdir(ws)
    monkeypatch.setattr(
        "dataselector.workflows.generate_reports.generate_thesis_final_report",
        lambda **_kwargs: None,
    )

    success = mod.run_thesis_pipeline(
        n_lhs=5,
        n_trials=2,
        skip_exploration=True,
        skip_optimization=True,
        skip_validation=True,
        dry_run=True,
        output_dir=ws / "outputs",
        seed=11,
        snapshot_config=True,
    )

    assert success is True
    run_meta = json.loads((ws / "outputs" / "run_metadata.json").read_text("utf-8"))
    snapshot_path = run_meta["extra"]["resolved_snapshot_path"]
    assert snapshot_path is not None

    assert run_meta["extra"]["resolved_snapshot_sha256"] is not None

    import yaml

    resolved = yaml.safe_load(Path(snapshot_path).read_text(encoding="utf-8"))
    params = resolved["parameters"]
    assert "selection" in params and "_provenance" in params["selection"]
    assert "clustering" in params and "_provenance" in params["clustering"]
    assert "feature_extraction" in params and "_provenance" in params["feature_extraction"]
    assert "alpha_visual" in params["selection"]["_provenance"]
    assert "n_clusters" in params["clustering"]["_provenance"]
    assert "batch_size" in params["feature_extraction"]["_provenance"]


@pytest.mark.skipif(
    True, reason="Requires full pipeline setup (data, features, config)"
)
def test_run_thesis_pipeline_integration():
    """Integration test for run_thesis_pipeline (skipped in CI)."""
    import tempfile
    from pathlib import Path

    from dataselector.workflows.thesis_pipeline import run_thesis_pipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "outputs"

        success = run_thesis_pipeline(
            n_lhs=10,
            n_trials=5,
            dry_run=True,
            output_dir=output_dir,
        )

        assert success is True


def test_cli_integration():
    """Test CLI integration via subprocess (smoke test)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "thesis-pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )

    # CLI might not exist yet, so just check it doesn't crash
    assert result.returncode in (0, 1, 2)
    out = f"{result.stdout}\n{result.stderr}"
    assert "--pre-names" in out
    assert "--pre-indices" in out
    assert "--hamburg" in out
    assert "--n-samples" in out
    assert "--validation-seeds" in out
    assert "--validation-min-distances" in out
