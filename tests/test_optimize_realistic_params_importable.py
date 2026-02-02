from pathlib import Path

from tests._helpers.load_script import load_script


def test_optimize_realistic_params_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "optimize_realistic_params.py",
        module_name="scripts.optimize_realistic_params_test",
    )
    assert hasattr(mod, "main")
