import subprocess
import sys


def test_adaptive_pipeline_help_available():
    result = subprocess.run(
        [sys.executable, "-m", "dataselector", "adaptive-pipeline", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--n-trials" in out
    assert "--sampler" in out
