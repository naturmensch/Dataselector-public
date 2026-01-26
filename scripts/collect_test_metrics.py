#!/usr/bin/env python3
"""Run a curated test set and store durations + junitxml into docs/test_metrics/<timestamp>/

Usage:
    ./scripts/exec_in_env.sh --env dataselector -- python scripts/collect_test_metrics.py --tests tests/integration --junit
"""
from pathlib import Path
import subprocess
import datetime
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "test_metrics"

parser = argparse.ArgumentParser()
./scripts/exec_in_env.sh --env dataselector -- parser.add_argument("--tests", default="tests", help="pytest path or pattern")
parser.add_argument("--junit", action="store_true", help="also produce junitxml")
./scripts/exec_in_env.sh --env dataselector -- parser.add_argument("--extra-args", default="", help="extra args to pass to pytest")
args = parser.parse_args()

stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
outdir = OUT / stamp
outdir.mkdir(parents=True, exist_ok=True)

import sys
./scripts/exec_in_env.sh --env dataselector -- cmd = [sys.executable, "-m", "pytest", args.tests, "-q", "--durations=10", "-rA"]
if args.junit:
    cmd.append(f"--junitxml={outdir / 'junit.xml'}")
if args.extra_args:
    cmd.extend(args.extra_args.split())

print("Running:", " ".join(cmd))
res = subprocess.run(cmd)

# Save a short summary
summary = outdir / "summary.txt"
with summary.open("w") as f:
    f.write(f"returncode: {res.returncode}\n")

print(f"Results stored in {outdir}")

if res.returncode != 0:
    raise SystemExit(res.returncode)
