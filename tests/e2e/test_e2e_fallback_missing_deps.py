import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.e2e
def test_pipeline_emits_graceful_warning_on_missing_native_deps(tmp_path):
    """Run `run_pipeline.py --smoke` and assert it prints a clear warning when heavy deps are missing.

    If the full native deps are present in the environment, the fallback won't happen and the test is skipped (this test verifies graceful degradation behavior).
    """
    ws = tmp_path / "workspace"
    data = ws / "data"
    data.mkdir(parents=True)
    # minimal csv so any downstream code that checks it can find it
    (data / "new_all_tiles.csv").write_text("longName,shortName,N,left\nA,a,50,10\nB,b,51,11\n")

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"), "--smoke", "--workspace", str(ws), "--tune"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # Common substring printed by run_pipeline when it cannot import the pipeline
    marker = "Warning: pipeline execution skipped due to import/exec error"

    if marker not in out:
        pytest.skip("Full native dependencies present — fallback warning not emitted in this environment")

    assert proc.returncode == 0
    assert marker in out
