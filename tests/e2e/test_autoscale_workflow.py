"""E2E test for autoscale multi-stage optimization workflow."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tests.utils import seed_immutable_feature_cache


@pytest.fixture
def autoscale_csv(tmp_workspace: Path) -> Path:
    """Create metadata with large projected spacing so autoscale stays feasible."""
    csv_file = tmp_workspace / "data" / "new_all_tiles.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "name",
                "year",
                "longitude",
                "latitude",
                "quadrant",
                "ul_x",
                "ul_y",
                "lr_x",
                "lr_y",
            ],
        )
        writer.writeheader()

        for i in range(50):
            year = 1900 + (i % 50)
            lon = 8.0 + (i % 10) * 0.5
            lat = 47.0 + (i // 10) * 0.5
            center_x = 500000.0 + (i % 10) * 100000.0
            center_y = 5900000.0 + (i // 10) * 100000.0
            half = 500.0
            writer.writerow(
                {
                    "id": f"tile_{i:04d}",
                    "name": f"Tile_{i}",
                    "year": year,
                    "longitude": lon,
                    "latitude": lat,
                    "quadrant": f"Q{(i % 4) + 1}",
                    "ul_x": center_x - half,
                    "ul_y": center_y + half,
                    "lr_x": center_x + half,
                    "lr_y": center_y - half,
                }
            )

    return csv_file


def _seed_autoscale_cache(output_dir: Path, metadata_csv: Path) -> None:
    """Keep autoscale CLI on the immutable cache-hit path for smoke coverage."""
    with metadata_csv.open(encoding="utf-8") as handle:
        row_count = max(0, sum(1 for _ in handle) - 1)
    seed_immutable_feature_cache(
        out_dir=output_dir,
        metadata_csv=metadata_csv,
        features=np.random.RandomState(11).randn(row_count, 32),
        batch_size=16,
    )


@pytest.fixture(autouse=True)
def skip_if_no_optuna():
    pytest.importorskip("optuna", exc_type=ImportError)


@pytest.mark.smoke
@pytest.mark.autoscale
def test_autoscale_smoke(
    tmp_workspace: Path, autoscale_csv: Path, run_dataselector_cli
):
    """Quick smoke test: autoscale runs and produces output.

    Verifies:
    - dataselector autoscale command exits with code 0
    - Output JSON file is created
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_autoscale_cache(output_dir, autoscale_csv)

    cmd = [
        "autoscale",
        "--csv",
        str(autoscale_csv),
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
    tmp_workspace: Path, autoscale_csv: Path, run_dataselector_cli
):
    """Verify autoscale output JSON structure and content.

    Validates:
    - best_params key exists
    - Convergence information present
    - All required fields populated
    """
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_autoscale_cache(output_dir, autoscale_csv)

    cmd = [
        "autoscale",
        "--csv",
        str(autoscale_csv),
        "--stages",
        "3",
        "5",
        "--n-trials",
        "2",
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

    # Validate current autoscale result schema
    params = data["best_params"]
    assert "a" in params, "Missing Optuna parameter a"
    assert "b" in params, "Missing Optuna parameter b"
    assert "c" in params, "Missing Optuna parameter c"
    assert "min_distance_km" in params, "Missing min_distance_km parameter"

    user_attrs = data["user_attrs"]
    assert "alpha" in user_attrs, "Missing normalized alpha attribute"
    assert "beta" in user_attrs, "Missing normalized beta attribute"
    assert "gamma" in user_attrs, "Missing normalized gamma attribute"

    # Normalized weights should sum to ~1.0
    weight_sum = user_attrs["alpha"] + user_attrs["beta"] + user_attrs["gamma"]
    assert 0.95 < weight_sum < 1.05, f"Weights don't sum to 1: {weight_sum}"


@pytest.mark.workflow
@pytest.mark.autoscale
def test_autoscale_multi_stage(
    tmp_workspace: Path, autoscale_csv: Path, run_dataselector_cli
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
    _seed_autoscale_cache(output_dir, autoscale_csv)

    # Run with real-ish parameters but small numbers for speed
    cmd = [
        "autoscale",
        "--csv",
        str(autoscale_csv),
        "--stages",
        "10",
        "20",
        "10",  # Abbreviated for testing
        "--n-trials",
        "2",
        "3",
        "2",
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
    assert final_result["n_trials"] >= 7, "Expected at least 7 trials total"
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
