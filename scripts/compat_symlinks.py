#!/usr/bin/env python3
"""Create backward-compatible symlinks to latest run artifacts.

Creates:
- outputs/optuna_results.csv -> outputs/runs/<latest>/results/trials.csv
- outputs/best_trial.json -> outputs/runs/<latest>/results/best_trial.json (if exists)

This is a convenience tool for legacy analysis scripts.
"""

import sys
from pathlib import Path


def main():
    root = Path("outputs")
    runs = root / "runs"
    if not runs.exists():
        print("No runs/ directory found; nothing to link.")
        return 0

    candidates = sorted([d for d in runs.iterdir() if d.is_dir()])
    if not candidates:
        print("No run directories found.")
        return 0

    latest = candidates[-1]
    src_trials = latest / "results" / "trials.csv"
    src_best = latest / "results" / "best_trial.json"

    if src_trials.exists():
        target = root / "optuna_results.csv"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(src_trials)
        print(f"Linked {target} -> {src_trials}")

    if src_best.exists():
        target = root / "best_trial.json"
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(src_best)
        print(f"Linked {target} -> {src_best}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
