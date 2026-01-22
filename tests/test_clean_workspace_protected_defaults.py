from pathlib import Path
from tests._helpers.load_script import load_script
ROOT = Path(__file__).resolve().parents[1]
cp = load_script(ROOT / "scripts" / "check_protected.py", module_name="scripts.check_protected_test")


def test_dry_run_shows_protected(capsys):
    cp.main(["--list"])
    captured = capsys.readouterr()
    out = captured.out

    assert "data/images" in out
    assert "data/archive" in out
    assert "outputs/final_selection" in out
