"""Tests for the sampler comparison script."""

import pickle
from pathlib import Path

import pytest

from unittest.mock import MagicMock, patch

from tests._helpers.load_script import load_script


@pytest.fixture(scope="module")
def compare_samplers():
    ROOT = Path(__file__).resolve().parents[1]
    return load_script(ROOT / "scripts" / "compare_samplers.py", module_name="scripts.compare_samplers_test")


def test_run_single_sampler_picklable(compare_samplers):
    """Ensure the worker function can be pickled for multiprocessing."""
    try:
        pickle.dumps(compare_samplers.run_single_sampler)
    except (pickle.PicklingError, AttributeError) as e:
        pytest.fail(f"run_single_sampler is not picklable: {e}")


def test_compare_samplers_importable(compare_samplers):
    """Ensure the module can be imported without side effects."""
    assert hasattr(compare_samplers, "run_sampler_comparison")


def test_run_single_sampler_validates_output(tmp_path, monkeypatch, compare_samplers):
    """Test that run_single_sampler returns None if trials.csv is missing/empty."""
    # Mock ROOT to point to tmp_path
    monkeypatch.setattr(compare_samplers, "ROOT", tmp_path)

    # Mock subprocess.run to succeed
    with (
        patch("subprocess.run") as mock_run,
        patch("scripts.compare_samplers.time.sleep"),
    ):
        mock_run.return_value.returncode = 0

        # Case 1: No run dir found
        assert compare_samplers.run_single_sampler("qmc", 1, 1, 1, None, "desc") is None

        # Setup fake run dir
        run_dir = tmp_path / "outputs" / "runs" / "qmc_1trials"
        run_dir.mkdir(parents=True)
        results_dir = run_dir / "results"
        results_dir.mkdir()

        # Case 2: Run dir exists, but no trials.csv
        assert compare_samplers.run_single_sampler("qmc", 1, 1, 1, None, "desc") is None

        # Case 3: trials.csv exists but empty
        (results_dir / "trials.csv").touch()
        assert compare_samplers.run_single_sampler("qmc", 1, 1, 1, None, "desc") is None

        # Case 4: Valid
        (results_dir / "trials.csv").write_text("header\n1,0.5")
        assert compare_samplers.run_single_sampler("qmc", 1, 1, 1, None, "desc") == str(
            run_dir
        )


def test_parallel_execution_smoke(monkeypatch, tmp_path, compare_samplers):
    """Smoke test verifying that multiprocessing Pool is utilized."""
    mock_pool = MagicMock()
    mock_context = MagicMock()
    mock_context.Pool.return_value = mock_pool
    mock_pool.__enter__.return_value = mock_pool
    # Return empty list to skip analysis part, just testing the call structure
    mock_pool.starmap.return_value = []

    monkeypatch.setattr(
        compare_samplers.multiprocessing, "get_context", lambda x: mock_context
    )
    monkeypatch.setattr(
        compare_samplers.multiprocessing, "get_all_start_methods", lambda: ["fork"]
    )
    monkeypatch.setattr(compare_samplers, "ROOT", tmp_path)

    compare_samplers.run_sampler_comparison(samplers=["s1"], n_trials=1)

    assert mock_context.Pool.called
    assert mock_pool.starmap.called


def test_parallel_execution_cpu_count_fallback(monkeypatch, tmp_path, compare_samplers):
    """Test that run_sampler_comparison handles cpu_count failure gracefully."""
    mock_pool = MagicMock()
    mock_context = MagicMock()
    mock_context.Pool.return_value = mock_pool
    mock_pool.__enter__.return_value = mock_pool
    mock_pool.starmap.return_value = []

    # Simulate cpu_count failure
    mock_context.cpu_count.side_effect = NotImplementedError

    monkeypatch.setattr(
        compare_samplers.multiprocessing, "get_context", lambda x: mock_context
    )
    monkeypatch.setattr(
        compare_samplers.multiprocessing, "get_all_start_methods", lambda: ["fork"]
    )
    monkeypatch.setattr(compare_samplers, "ROOT", tmp_path)

    # Should not raise and default to 1 process (or min(len, 1))
    compare_samplers.run_sampler_comparison(samplers=["s1", "s2"], n_trials=1)

    # Verify Pool was created with processes=1
    assert mock_context.Pool.call_args[1]["processes"] == 1
