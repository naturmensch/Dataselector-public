import subprocess
import sys
from pathlib import Path


def test_check_dependency_pins_executes_successfully():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "check_dependency_pins.py"
    assert script.exists()
    rc = subprocess.call([sys.executable, str(script)])
    # Should succeed since we already updated pins in repo
    assert rc == 0
