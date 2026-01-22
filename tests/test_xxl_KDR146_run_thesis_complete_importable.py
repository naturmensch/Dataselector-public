from pathlib import Path
from tests._helpers.load_script import load_script


def test_xxl_thesis_orchestrator_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(ROOT / "scripts" / "xxl_KDR146_run_thesis_complete.py", module_name="scripts.xxl_KDR146_run_thesis_complete_test")
    assert hasattr(mod, "log")
