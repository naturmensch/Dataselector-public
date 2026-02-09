"""Regression tests for tune_weights exploration workflow semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _write_metadata_csv(path: Path, n_rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "ul_x": list(range(1, n_rows + 1)),
            "ul_y": list(range(10, n_rows + 10)),
            "lr_x": list(range(2, n_rows + 2)),
            "lr_y": list(range(9, n_rows + 9)),
            "year": [1900] * n_rows,
        }
    )
    df.to_csv(path, index=False)
    return path


def _install_exploration_mocks(monkeypatch, *, captured: dict) -> None:
    from dataselector.workflows import tune_weights as mod

    class DummyRunner:
        def __init__(self, *args, **kwargs):
            return None

        def run_weight_sweep(self, **kwargs):
            captured["run_weight_sweep_kwargs"] = kwargs
            return pd.DataFrame(
                [
                    {
                        "alpha": 0.4,
                        "beta": 0.3,
                        "gamma": 0.3,
                        "clusters_covered": 8,
                        "temporal_std": 10.0,
                        "temporal_range": 40.0,
                        "spatial_mean_km": 500.0,
                        "n_selected": kwargs["n_samples"],
                    }
                ]
            )

    monkeypatch.setattr(
        "dataselector.pipeline.experiments.ExperimentRunner", DummyRunner
    )
    monkeypatch.setattr(
        "dataselector.selection.pareto.compute_pareto_front", lambda df: df
    )
    monkeypatch.setattr(
        "dataselector.selection.pareto.visualize_pareto_front", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "dataselector.selection.pareto.export_pareto_report",
        lambda df, output_path: df.to_csv(output_path, index=False),
    )
    monkeypatch.setattr(
        mod,
        "generate_weights",
        lambda n_points, seed, sampler: [(0.4, 0.3, 0.3)] * n_points,
    )


def test_run_exploration_separates_lhs_points_from_selection_target(
    tmp_path: Path, monkeypatch
):
    """`n_samples` controls weight combinations, not selection target size."""
    from dataselector.workflows import tune_weights as mod

    captured: dict = {}
    _install_exploration_mocks(monkeypatch, captured=captured)

    repo_root = tmp_path / "repo"
    cfg_dir = repo_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: 21\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "ROOT", repo_root)

    metadata_csv = _write_metadata_csv(tmp_path / "metadata.csv", n_rows=123)
    out_dir = tmp_path / "out"

    mod.run_exploration(
        n_samples=7,  # LHS points
        sampler="lhs",
        seed=42,
        metadata_path=metadata_csv,
        min_distance_km=40.0,
        output_dir=out_dir,
    )

    kwargs = captured["run_weight_sweep_kwargs"]
    assert len(kwargs["weight_combinations"]) == 7
    assert kwargs["n_samples"] == 21


def test_run_exploration_honors_explicit_selection_target(tmp_path: Path, monkeypatch):
    """Explicit `selection_n_samples` must override fallback/config logic."""
    from dataselector.workflows import tune_weights as mod

    captured: dict = {}
    _install_exploration_mocks(monkeypatch, captured=captured)

    repo_root = tmp_path / "repo"
    cfg_dir = repo_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "ROOT", repo_root)

    metadata_csv = _write_metadata_csv(tmp_path / "metadata.csv", n_rows=80)
    out_dir = tmp_path / "out"

    mod.run_exploration(
        n_samples=5,
        selection_n_samples=12,
        sampler="lhs",
        seed=123,
        metadata_path=metadata_csv,
        min_distance_km=40.0,
        output_dir=out_dir,
    )

    kwargs = captured["run_weight_sweep_kwargs"]
    assert len(kwargs["weight_combinations"]) == 5
    assert kwargs["n_samples"] == 12


def test_run_exploration_fails_without_selection_target_source(
    tmp_path: Path, monkeypatch
):
    """No implicit numeric fallback is allowed when no source resolves n_samples."""
    from dataselector.workflows import tune_weights as mod

    captured: dict = {}
    _install_exploration_mocks(monkeypatch, captured=captured)

    repo_root = tmp_path / "repo"
    cfg_dir = repo_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "ROOT", repo_root)

    metadata_csv = _write_metadata_csv(tmp_path / "metadata.csv", n_rows=50)

    with pytest.raises(
        ValueError, match="could not resolve selection target n_samples"
    ):
        mod.run_exploration(
            n_samples=4,
            sampler="lhs",
            seed=1,
            metadata_path=metadata_csv,
            min_distance_km=28.5,
            output_dir=tmp_path / "out",
        )
