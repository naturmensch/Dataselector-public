import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / 'outputs' / 'selected_sampler.json'


def test_monitor_uses_selected_sampler(tmp_path, monkeypatch):
    # Prepare selected_sampler artifact
    SEL.parent.mkdir(parents=True, exist_ok=True)
    data = {"best": "cmaes", "metric": "mean_best_value", "score": 123.4}
    SEL.write_text(json.dumps(data))

    env = dict(**{k: v for k, v in {'PYTHONPATH': str(ROOT)}.items()})

    # Run monitor in child-dry-run mode so it doesn't actually launch heavy workloads
    cmd = [sys.executable, str(ROOT / 'scripts' / 'xxl_full_run_monitor.py'), '--child-dry-run', '--no-new-session']
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = res.stdout + res.stderr
    assert '--sampler cmaes' in out or "Using selected sampler from artifact: cmaes" in out

    # Clean up
    try:
        SEL.unlink()
    except Exception:
        pass
