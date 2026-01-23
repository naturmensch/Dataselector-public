from pathlib import Path

from tests._helpers.load_script import load_script


def test_validate_pareto_candidates_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "validate_pareto_candidates.py",
        module_name="scripts.validate_pareto_candidates_test",
    )
    assert hasattr(mod, "validate")
