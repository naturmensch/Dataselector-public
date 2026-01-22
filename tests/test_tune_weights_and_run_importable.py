from pathlib import Path
from tests._helpers.load_script import load_script


def test_tune_weights_and_run_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(ROOT / "scripts" / "tune_weights_and_run.py", module_name="scripts.tune_weights_and_run_test")
    assert hasattr(mod, "generate_weights")
