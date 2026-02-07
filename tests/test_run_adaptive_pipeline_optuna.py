from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def run_pipe():
    pytest.importorskip("numba", exc_type=ImportError)
    from dataselector.workflows import adaptive_pipeline

    return adaptive_pipeline


def _write_minimal_metadata(repo_root: Path) -> None:
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


def _write_minimal_fine_pareto(repo_root: Path) -> None:
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


def _prepare_pipeline_fs(tmp_path: Path) -> None:
    _write_minimal_metadata(tmp_path)
    _write_minimal_fine_pareto(tmp_path)


def _raise_optuna_error(**kwargs):
    raise Exception("subprocess fail")


def test_optuna_failure_aborts(monkeypatch, run_pipe, tmp_path):
    from dataselector.workflows import bootstrap
    from dataselector.workflows import optuna_optimize

    _prepare_pipeline_fs(tmp_path)
    monkeypatch.setattr(run_pipe, "ROOT", tmp_path)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "optuna" else None,
    )
    monkeypatch.setattr(
        bootstrap, "run_bootstrap_pareto", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(optuna_optimize, "run_optuna", _raise_optuna_error)
    with pytest.raises(SystemExit) as excinfo:
        run_pipe.main(
            sampler="lhs",
            n_lhs=2,
            skip_exploration=True,
            skip_fine=True,
        )
    assert excinfo.value.code == 1


def test_optuna_continue_on_failure(monkeypatch, run_pipe, tmp_path):
    from dataselector.workflows import bootstrap
    from dataselector.workflows import optuna_optimize

    _prepare_pipeline_fs(tmp_path)
    monkeypatch.setattr(run_pipe, "ROOT", tmp_path)
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "optuna" else None,
    )
    monkeypatch.setattr(
        bootstrap, "run_bootstrap_pareto", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(optuna_optimize, "run_optuna", _raise_optuna_error)
    assert (
        run_pipe.main(
            sampler="lhs",
            n_lhs=2,
            skip_exploration=True,
            skip_fine=True,
            continue_on_analysis_failure=True,
        )
        == 0
    )
