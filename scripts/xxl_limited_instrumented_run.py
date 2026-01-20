#!/usr/bin/env python3
"""Instrumented limited run for XXL thesis pipeline

This script runs a fast, deterministic, instrumented end-to-end validation of
`scripts/xxl_KDR146_run_thesis_complete.py` by simulating the heavy
sub-steps (calls to `run_adaptive_pipeline.py`) and creating minimal
`outputs/runs/.../results/trials.csv` files so the statistics and summary
extraction code can be exercised quickly.

Usage:
    PYTHONPATH=. python scripts/xxl_limited_instrumented_run.py

It intentionally monkeypatches `run_cmd_with_retry` and `run_cmd` to avoid
long-running computations while preserving the orchestration logic.
"""

import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import functions to test
# We need to insert project root into sys.path above; importing the script
# here therefore violates the 'imports at top' rule (E402) but is intentional.
# ruff: noqa: E402
from scripts.xxl_KDR146_run_thesis_complete import (
    phase_1_xxl_hamburg,
    phase_2_reproducibility,
    phase_3_final_statistics,
    phase_4_thesis_summary,
    run_cmd,
    run_cmd_with_retry,
)

# Instrumentation: create simulated runs
SIM_OUTPUT_DIR = ROOT / "outputs" / "runs"
SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG = []


def make_fake_run(
    seed: int, kind: str = "hamburg", is_xxl_final: bool = False, n_trials: int = 10
):
    """Create a fake run directory with a trials.csv with deterministic values."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"{ts}.{kind}_cmaes_{n_trials}trials_s{seed}"
    if is_xxl_final:
        tag = f"{ts}.{kind}_xxl_final_s{seed}"
    run_dir = SIM_OUTPUT_DIR / tag
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    # Create deterministic trials
    trials = []
    base = 0.5 + seed / 1000.0
    for t in range(1, n_trials + 1):
        val = base + (t / n_trials) * 0.5  # increasing value
        trials.append(
            {
                "trial_number": t,
                "state": "TrialState.COMPLETE",
                "value": val,
                "a": 0.1 * t,
                "b": 0.2 * t,
                "c": 0.3 * t,
                "min_distance_km": 1.0,
                "n_samples": 5,
            }
        )
    df = pd.DataFrame(trials)
    df.to_csv(run_dir / "results" / "trials.csv", index=False)
    LOG.append(f"Simulated run created: {run_dir}")
    return run_dir


# Monkeypatch the heavy command runner to instead simulate runs
_original_run_cmd = run_cmd
_original_run_cmd_with_retry = run_cmd_with_retry


def _sim_run_cmd(cmd: str, cwd=None, fail_ok: bool = False) -> int:
    """Simulate shell commands: look for `run_adaptive_pipeline.py` calls and
    create fake runs; otherwise, just log."""
    LOG.append(f"SIM_RUN_CMD: {cmd}")
    # detect seed in command
    seed = None
    import re

    m = re.search(r"--seed\s+(\d+)", cmd)
    if m:
        seed = int(m.group(1))
    # check if __exp-name includes "thesis_xxl_hamburg_final" -> make xxl final
    if "thesis_xxl_hamburg_final" in cmd:
        make_fake_run(seed=seed or 42, is_xxl_final=True, kind="hamburg", n_trials=10)
        return 0
    if "thesis_hamburg_reproducibility" in cmd:
        make_fake_run(seed=seed or 43, kind="hamburg", n_trials=10)
        return 0
    # default: create a small generic run
    if "run_adaptive_pipeline.py" in cmd:
        make_fake_run(seed=seed or random.randint(40, 99), kind="hamburg", n_trials=5)
        return 0
    LOG.append(f"No simulation rule matched for cmd: {cmd}")
    return 0


def _sim_run_cmd_with_retry(
    cmd: str, retries: int = 2, delay: int = 5, cwd=None, fail_ok: bool = False
) -> int:
    LOG.append(f"SIM_RUN_CMD_WITH_RETRY: {cmd} (retries={retries})")
    return _sim_run_cmd(cmd, cwd=cwd, fail_ok=fail_ok)


# Apply monkeypatch
import scripts.xxl_KDR146_run_thesis_complete as xxl_mod  # noqa: E402

xxl_mod.run_cmd = _sim_run_cmd
xxl_mod.run_cmd_with_retry = _sim_run_cmd_with_retry


def main() -> int:
    LOG.append("Starting instrumented limited run")
    start = time.time()

    # Phase 1: small n_trials and candidates
    ok = phase_1_xxl_hamburg(n_trials=10, n_candidates=5, pass_params=True)
    LOG.append(f"Phase 1 returned: {ok}")
    if not ok:
        LOG.append("Phase 1 failed")
        return 1

    # Phase 2: reproducibility seeds
    ok = phase_2_reproducibility(
        seeds=[43, 44], n_trials=10, n_candidates=5, pass_params=True
    )
    LOG.append(f"Phase 2 returned: {ok}")
    if not ok:
        LOG.append("Phase 2 failed")
        return 1

    # Phase 3: statistics extraction should find the xxl final run created above
    ok = phase_3_final_statistics()
    LOG.append(f"Phase 3 returned: {ok}")
    if not ok:
        LOG.append("Phase 3 failed")
        return 1

    # Phase 4: summary generation
    ok = phase_4_thesis_summary()
    LOG.append(f"Phase 4 returned: {ok}")
    if not ok:
        LOG.append("Phase 4 failed")
        return 1

    elapsed = time.time() - start
    LOG.append(f"Instrumented run complete in {elapsed:.2f}s")

    # Dump LOG to outputs for inspection
    out_file = ROOT / "outputs" / "INSTRUMENTED_LIMITED_RUN_LOG.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(LOG))
    print("\n".join(LOG))
    print(f"Wrote log to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
