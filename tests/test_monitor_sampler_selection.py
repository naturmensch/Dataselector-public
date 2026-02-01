import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "outputs" / "selected_sampler.json"


def test_monitor_uses_selected_sampler(tmp_path, monkeypatch):
    # Prepare selected_sampler artifact
    SEL.parent.mkdir(parents=True, exist_ok=True)
    data = {"best": "cmaes", "metric": "mean_best_value", "score": 123.4}
    SEL.write_text(json.dumps(data))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # Run monitor in child-dry-run mode so it doesn't actually launch heavy workloads
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "xxl_full_run_monitor.py"),
        "--child-dry-run",
        "--no-new-session",
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = res.stdout + res.stderr
    if not (
        "--sampler cmaes" in out or "Using selected sampler from artifact: cmaes" in out
    ):
        # Fallback: check any monitor logs under outputs for the message
        found = False
        for p in (ROOT / "outputs").glob("**/*.log"):
            try:
                if (
                    "Using selected sampler for Optuna from artifact: cmaes"
                    in p.read_text(errors="ignore")
                    or "--optuna-sampler cmaes" in p.read_text(errors="ignore")
                ):
                    found = True
                    break
            except Exception:
                continue
        assert found, f"Selected sampler not detected in stdout or logs. stdout: {out}"

    # Clean up
    try:
        SEL.unlink()
    except Exception:
        pass
