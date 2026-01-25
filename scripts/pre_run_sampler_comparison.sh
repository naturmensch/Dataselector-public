#!/usr/bin/env bash
# Pre-run hook: execute sampler comparison

cd <dataselector-repo>
mamba run -n dataselector python scripts/compare_samplers_multi_seed.py \
  --samplers qmc tpe cmaes \
  --seeds 42 43 \
  --n-trials 50 \
  --datasets hamburg
