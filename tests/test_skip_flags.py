from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def run_pipe():
    pytest.importorskip("numba", exc_type=ImportError)
    from dataselector.workflows import adaptive_pipeline

    return adaptive_pipeline


def _prepare_pipeline_fs(repo_root: Path) -> None:
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "SheetNumber": [1, 2],
            "shortName": ["S1", "S2"],
            "longName": ["tile_1", "tile_2"],
            "city": ["test_city", "test_city"],
            "year": [1910, 1911],
            "filename": ["tile_1.tif", "tile_2.tif"],
            "image_path": ["tile_1.png", "tile_2.png"],
            "ul_x": [500000.0, 500200.0],
            "ul_y": [5900000.0, 5899800.0],
            "lr_x": [500100.0, 500300.0],
            "lr_y": [5899900.0, 5899700.0],
            "width_px": [1000, 1000],
            "height_px": [1000, 1000],
            "pixel_width": [0.1, 0.1],
            "pixel_height": [-0.1, -0.1],
            "data_quality": ["ok", "ok"],
        }
    ).to_csv(data_dir / "new_all_tiles.csv", index=False)

    fine_dir = repo_root / "outputs" / "experiments" / "fine_sweep"
    fine_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "alpha": [0.6],
            "beta": [0.2],
            "gamma": [0.2],
            "min_distance_km": [34],
        }
    ).to_csv(fine_dir / "pareto_solutions.csv", index=False)


def test_skip_optuna_and_skip_flags_dryrun(tmp_path, monkeypatch, capsys, run_pipe):
    from dataselector.workflows import bootstrap

    _prepare_pipeline_fs(tmp_path)
    monkeypatch.setattr(run_pipe, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "run_bootstrap_pareto", lambda *args, **kwargs: None)

    # Ensure skip-optuna is honored by adaptive-pipeline workflow
    assert (
        run_pipe.main(
            dry_run=True,
            skip_optuna=True,
            skip_exploration=True,
            skip_fine=True,
            skip_bootstrap_injection=True,
            sampler="lhs",
            n_lhs=2,
            n_trials=2,
            n_boot=1,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Skipping Optuna stage" in out, "skip-optuna not honored"

    # Ensure skip-exploration prevents exploration cmd invocation
    assert (
        run_pipe.main(
            dry_run=True,
            skip_optuna=True,
            skip_exploration=True,
            skip_fine=True,
            skip_bootstrap_injection=True,
            sampler="lhs",
            n_lhs=2,
            n_trials=2,
            n_boot=1,
        )
        == 0
    )
    out2 = capsys.readouterr().out
    assert "Exploration SKIPPED" in out2, "--skip-exploration not honored"

    # Ensure skip-fine prevents fine sweep execution
    assert (
        run_pipe.main(
            dry_run=True,
            skip_optuna=True,
            skip_exploration=True,
            skip_fine=True,
            skip_bootstrap_injection=True,
            sampler="lhs",
            n_lhs=2,
            n_trials=2,
            n_boot=1,
        )
        == 0
    )
    out3 = capsys.readouterr().out
    assert "Fine Sweep SKIPPED" in out3, "--skip-fine not honored"
