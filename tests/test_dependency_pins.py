import subprocess
import sys
from pathlib import Path


def test_check_env_executes_successfully():
    repo_root = Path(__file__).resolve().parents[1]
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "dataselector",
            "check-env",
            "dataselector",
            "tests",
            "Makefile",
            ".github/workflows",
        ],
        cwd=repo_root,
    )
    assert rc in (0, 2)
