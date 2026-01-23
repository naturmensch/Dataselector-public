from pathlib import Path

from tests._helpers.load_script import load_script


def test_compare_samplers_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "compare_samplers.py",
        module_name="scripts.compare_samplers_test",
    )
    assert hasattr(mod, "run_single_sampler")
