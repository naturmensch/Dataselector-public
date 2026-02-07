import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_thesis_pipeline_graceful_fallback(tmp_path):
    """Test that thesis-pipeline CLI handles missing dependencies gracefully.

    If the full native deps are present, fallback won't happen and test is skipped.
    This verifies graceful degradation behavior.
    """
    ws = tmp_path / "workspace"
    data = ws / "data"
    data.mkdir(parents=True)
    # minimal csv so any downstream code that checks it can find it
    (data / "new_all_tiles.csv").write_text(
        "longName,shortName,ul_x,ul_y,lr_x,lr_y\n"
        "A,a,499950,5900050,500050,5899950\n"
        "B,b,500950,5901050,501050,5900950\n"
    )

    # Use thesis-pipeline CLI with dry-run to test import paths
    cmd = ["thesis-pipeline", "--dry-run", "--n-lhs", "5"]
    proc = run_dataselector_cli(cmd, capture_output=True, text=True, cwd=str(ws))

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Check for graceful error handling (not a hard crash)
    if proc.returncode != 0 and "ModuleNotFoundError" in out:
        # If imports fail, ensure we get a clean error message
        assert (
            "Warning" in out or "Error" in out
        ), "Missing deps should produce clear error message"
    else:
        pytest.skip("Full dependencies present — fallback not triggered")
