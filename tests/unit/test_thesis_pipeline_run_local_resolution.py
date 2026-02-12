from __future__ import annotations

from pathlib import Path

from dataselector.workflows import thesis_pipeline as mod


def test_computed_selection_values_ignore_global_outputs_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs" / "runs" / "local_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global legacy artifact should not be consumed.
    legacy_dir = tmp_path / "outputs"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "optuna_autoscale_best_latest.json").write_text(
        '{"params":{"a":1.0,"b":1.0,"c":1.0,"min_distance_km":12},"user_attrs":{"n_samples":33}}',
        encoding="utf-8",
    )

    values, method, source_file, source_hash = mod._resolve_computed_selection_values(
        compute_params=True,
        output_dir=output_dir,
    )
    assert values == {}
    assert method is None
    assert source_file is None
    assert source_hash is None


def test_sampler_resolution_ignores_global_outputs_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs" / "runs" / "local_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global legacy artifact should not be consumed.
    legacy_dir = tmp_path / "outputs"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "selected_sampler.json").write_text(
        '{"selected_sampler":"tpe"}',
        encoding="utf-8",
    )

    sampler, source, source_path = mod._resolve_optuna_sampler(
        config={},
        output_dir=output_dir,
        n_trials=10,
        n_samples=24,
        validation_seeds=[42],
        compute_params=False,
        dry_run=False,
    )

    assert sampler is None
    assert source == "unresolved"
    assert source_path is None
