from pathlib import Path

from tests._helpers.load_script import load_script


def test_optuna_optimize_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "optuna_optimize.py",
        module_name="scripts.optuna_optimize_test",
    )
    assert hasattr(mod, "run_optuna")
