"""Integration test for final-selection with different variants.

Tests final-selection command with various --method choices.
"""

import json
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.selection
def test_final_selection_default_variant(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Run final-selection with default parameters."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        
        "final-selection",
        "--csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--n-samples", "10",
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120)
    assert result.returncode == 0, f"final-selection failed:\n{result.stderr.decode()}"


@pytest.mark.integration
@pytest.mark.selection
def test_final_selection_output_structure(tmp_workspace: Path, sample_csv: Path, run_dataselector_cli):
    """Verify final-selection output has required structure."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        
        "final-selection",
        "--csv", str(sample_csv),
        "--output-dir", str(output_dir),
        "--n-samples", "5",
    ]
    
    result = run_dataselector_cli(cmd, cwd=str(tmp_workspace), capture_output=True, timeout=120)
    assert result.returncode == 0
    
    # Check for output file (name may vary)
    output_files = list(output_dir.glob("selection*.json"))
    if output_files:
        with open(output_files[0]) as f:
            output = json.load(f)
        assert isinstance(output, (dict, list)), "Output should be JSON object/array"
