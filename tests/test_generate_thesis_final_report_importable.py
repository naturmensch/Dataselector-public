from pathlib import Path
from tests._helpers.load_script import load_script


def test_generate_thesis_final_report_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "generate_thesis_final_report.py",
        module_name="scripts.generate_thesis_final_report_test",
    )
    assert hasattr(mod, "main")
