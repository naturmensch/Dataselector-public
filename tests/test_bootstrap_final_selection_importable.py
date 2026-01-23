from pathlib import Path

from tests._helpers.load_script import load_script


def test_bootstrap_final_selection_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "bootstrap_final_selection.py",
        module_name="scripts.bootstrap_final_selection_test",
    )
    assert hasattr(mod, "main")
