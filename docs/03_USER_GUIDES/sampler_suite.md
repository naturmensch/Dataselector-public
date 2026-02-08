# Historical/Legacy (Not Normative)

This page is kept for historical notes and may reference script-era usage.

Use the authoritative replacements:

1. `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`
2. `docs/03_USER_GUIDES/PIPELINES.md`

# Sampler Suite (QMC vs TPE vs CMA-ES)

This page explains how to run the sampler suite for multi-seed comparisons:

- Script: `scripts/run_thesis_sampler_suite.py`
- Typical run: 10 seeds × 3 samplers × 1000 trials (adjust with `--n-trials` and `--seeds`)
- Output: `outputs/selected_sampler.json` with recommended sampler and per-seed metadata

(Include recommended commands, interpretation and plotting notes.)
