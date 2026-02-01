from pathlib import Path

from tests._helpers.load_script import load_script


def test_xxl_limited_instrumented_run_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "xxl_limited_instrumented_run.py",
        module_name="scripts.xxl_limited_instrumented_run_test",
    )
    assert hasattr(mod, "main")
