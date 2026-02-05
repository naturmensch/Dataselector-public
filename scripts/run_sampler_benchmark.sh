#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper to run multi-seed sampler benchmark on Hamburg and KDR100
python -m dataselector compare-samplers multi-seed --samplers qmc tpe cmaes --seeds 42 43 44 45 46 --n-trials 500 --datasets hamburg kdr100
