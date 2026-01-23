from pathlib import Path

from tests._helpers.load_script import load_script


def test_run_fine_sweep_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "run_fine_sweep.py",
        module_name="scripts.run_fine_sweep_test",
    )
    assert hasattr(mod, "main")
