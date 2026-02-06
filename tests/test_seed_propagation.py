from pathlib import Path

import pandas as pd
import pytest

import dataselector.pipeline.pipeline_utils as pipeline_utils
import dataselector.workflows.adaptive_pipeline as adaptive_pipeline
import dataselector.workflows.bootstrap as bootstrap

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_run_adaptive_pipeline_seed_propagation(tmp_path, monkeypatch):
    # Isolate workflow root
    monkeypatch.setattr(adaptive_pipeline, "ROOT", tmp_path)

    # Minimal metadata CSV (missing N/left is fine: workflow falls back to default distance)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "new_all_tiles.csv"
    pd.DataFrame({"longName": ["A.png", "B.png"], "year": [1900, 1910]}).to_csv(
        csv_path, index=False
    )

    # Provide a pareto file expected by skip-fine path
    pareto_dir = tmp_path / "outputs" / "experiments" / "fine_sweep"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "alpha": [0.3, 0.4],
            "beta": [0.3, 0.3],
            "gamma": [0.4, 0.3],
            "min_distance_km": [10, 20],
        }
    ).to_csv(pareto_dir / "pareto_solutions.csv", index=False)

    # Keep test focused on seed propagation; stub expensive phase functions.
    monkeypatch.setattr(
        pipeline_utils, "compute_fine_search_bounds", lambda *_args, **_kwargs: [10, 20]
    )
    monkeypatch.setattr(
        bootstrap, "run_bootstrap_pareto", lambda *_args, **_kwargs: 0
    )

    run_dir = adaptive_pipeline.run_adaptive_pipeline(
        experiment_name="seed_propagation_test",
        csv_path=csv_path,
        n_lhs=2,
        n_trials=2,
        n_boot=1,
        sampler="lhs",
        seed=12345,
        skip_exploration=True,
        skip_fine=True,
        skip_optuna=True,
        skip_bootstrap_injection=True,
        dry_run=True,
    )

    cfg = Path(run_dir) / "config" / "config_run.yaml"
    assert cfg.exists(), f"Config missing: {cfg}"
    assert "seed: 12345" in cfg.read_text()
