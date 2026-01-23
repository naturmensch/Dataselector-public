from pathlib import Path

from tests._helpers.load_script import load_script


def test_monitor_live_updates_importable():
    ROOT = Path(__file__).resolve().parents[1]
    mod = load_script(
        ROOT / "scripts" / "monitor_live_updates.py",
        module_name="scripts.monitor_live_updates_test",
    )
    assert hasattr(mod, "main")
