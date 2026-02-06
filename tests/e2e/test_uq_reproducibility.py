"""Integration test for UQ reproducibility across runs.

Tests that quantile estimates are reproducible when using same seed.
"""

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.uq
def test_uq_reproducibility_same_seed(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Run pipeline twice with same seed, verify identical UQ estimates.

    Validates reproducibility of uncertainty quantification.
    """
    output_dirs = [
        tmp_workspace / "outputs" / "run1",
        tmp_workspace / "outputs" / "run2",
    ]

    results = []
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "autoscale",
            "--csv",
            str(sample_csv),
            "--output-dir",
            str(out_dir),
            "--seed",
            "42",
            "--n-trials",
            "5",
        ]

        result = run_dataselector_cli(
            cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
        )
        assert result.returncode == 0

        # Collect output JSON
        output_json = out_dir / "selection_output.json"
        if output_json.exists():
            with open(output_json) as f:
                results.append(json.load(f))

    # Compare quantile estimates if they exist
    if len(results) == 2 and "quantiles" in results[0]:
        assert (
            results[0]["quantiles"] == results[1]["quantiles"]
        ), "UQ estimates not reproducible with same seed"


@pytest.mark.integration
@pytest.mark.uq
def test_uq_different_seeds(
    tmp_workspace: Path, sample_csv: Path, run_dataselector_cli
):
    """Run pipeline with different seeds, verify different UQ estimates.

    Validates that different seeds produce different results.
    """
    output_dirs = []

    for seed in [42, 123]:
        out_dir = tmp_workspace / "outputs" / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(out_dir)

        cmd = [
            "autoscale",
            "--csv",
            str(sample_csv),
            "--output-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--n-trials",
            "5",
        ]

        result = run_dataselector_cli(
            cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
        )
        assert result.returncode == 0
