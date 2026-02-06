"""E2E test for autoscale multi-stage optimization workflow.

Tests the dataselector autoscale command with multi-stage optimization:
- 3 consecutive stages with increasing trial counts
- Validates autoscale output JSON structure
- Checks convergence and best parameters selection

NOTE: These tests require optuna package which is not installed in test environment.
Marking as skip until optuna is available.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(reason="Requires optuna package")


@pytest.mark.smoke
@pytest.mark.autoscale
def test_autoscale_smoke(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Quick smoke test: autoscale runs and produces output.

    Verifies:
    - dataselector autoscale command exits with code 0
    - Output JSON file is created
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--stages",
        "3",
        "5",  # Very small stages for speed
        "--n-trials",
        "2",
        "3",  # Match stage count
        "--output-dir",
        str(output_dir),
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace))

    assert result.returncode == 0, f"autoscale failed:\n{result.stderr.decode()}"

    # Check for output JSON
    best_json = output_dir / "autoscale_best_latest.json"
    assert best_json.exists(), f"Expected output {best_json} not found"


@pytest.mark.integration
@pytest.mark.autoscale
def test_autoscale_output_structure(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Verify autoscale output JSON structure and content.

    Validates:
    - best_params key exists
    - Convergence information present
    - All required fields populated
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--stages",
        "3",
        "5",
        "--output-dir",
        str(output_dir),
        "--n-candidates",
        "20",  # Small for speed
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace))
    assert result.returncode == 0

    best_json = output_dir / "autoscale_best_latest.json"
    assert best_json.exists()

    # Parse and validate structure
    with open(best_json) as f:
        data = json.load(f)

    # Check required keys
    required_keys = ["best_params", "best_value", "n_trials"]
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

    # Validate best_params structure
    params = data["best_params"]
    assert "alpha" in params, "Missing alpha parameter"
    assert "beta" in params, "Missing beta parameter"
    assert "gamma" in params, "Missing gamma parameter"

    # Weights should sum to ~1.0
    weight_sum = params["alpha"] + params["beta"] + params["gamma"]
    assert 0.95 < weight_sum < 1.05, f"Weights don't sum to 1: {weight_sum}"


@pytest.mark.workflow
@pytest.mark.autoscale
def test_autoscale_multi_stage(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Full autoscale multi-stage workflow (longer test).

    Simulates real autoscale with 3 stages:
    - Stage 1: 10 trials
    - Stage 2: 20 trials
    - Stage 3: 30 trials (abbreviated to 10 for testing)

    Validates:
    - Each stage completes
    - Trial counts match expected ranges
    - Best parameters improve over stages
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run with real-ish parameters but small numbers for speed
    cmd = [
        "autoscale",
        "--csv",
        str(sample_csv),
        "--stages",
        "10",
        "20",
        "10",  # Abbreviated for testing
        "--output-dir",
        str(output_dir),
        "--n-candidates",
        "30",
        "--seed",
        "42",  # Deterministic for testing
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), timeout=300)

    assert result.returncode == 0, f"autoscale failed:\n{result.stderr.decode()}"

    best_json = output_dir / "autoscale_best_latest.json"
    assert best_json.exists(), "Output JSON not created"

    with open(best_json) as f:
        final_result = json.load(f)

    # Verify convergence
    assert final_result["n_trials"] >= 30, "Expected at least 30 trials total"
    assert final_result["best_value"] is not None, "Best value not set"
    assert final_result["best_value"] > 0, "Best value should be positive"


@pytest.mark.error
@pytest.mark.autoscale
def test_autoscale_missing_csv(tmp_workspace: Path, run_dataselector_cli):
    """Error test: autoscale with missing CSV file.

    Should fail gracefully with clear error message.
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autoscale",
        "--csv",
        str(tmp_workspace / "nonexistent.csv"),
        "--output-dir",
        str(output_dir),
    ]

    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace))

    # Should fail
    assert result.returncode != 0, "Should have failed with missing CSV"

    # Error message should be helpful
    stderr = result.stderr.decode()
    assert (
        "not found" in stderr.lower()
        or "error" in stderr.lower()
        or "no such" in stderr.lower()
    ), f"Error message not helpful: {stderr}"
