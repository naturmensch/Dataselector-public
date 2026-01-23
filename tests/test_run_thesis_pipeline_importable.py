from pathlib import Path

from tests._helpers.load_script import load_script


def test_run_thesis_pipeline_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "run_thesis_pipeline.py",
        module_name="scripts.run_thesis_pipeline_test",
    )
    assert hasattr(mod, "main")
