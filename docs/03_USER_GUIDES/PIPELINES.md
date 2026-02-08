# Pipelines Overview (Authoritative)

This page summarizes the current thesis-relevant pipelines.

Primary runbook:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## Pipeline Map

1. `benchmark-sampling`
   - Compare exploration methods and persist exploration plan.
2. `thesis-sampler-suite`
   - Multi-seed sampler comparison and best-sampler selection.
3. `thesis-pipeline`
   - Exploration -> Optuna -> validation in one workflow.
4. `xxl`
   - Phase-based long-run orchestration with monitoring artifacts.
5. `adaptive-auto`
   - Thin orchestrator: autoscale handoff + adaptive pipeline.

## Canonical Commands

```bash
python -m dataselector benchmark-sampling --help
python -m dataselector thesis-sampler-suite --help
python -m dataselector thesis-pipeline --help
python -m dataselector xxl --help
python -m dataselector adaptive-auto --help
```

## Typical Thesis Sequence

```bash
python -m dataselector thesis-sampler-suite --autoscale
python -m dataselector xxl
```

For deterministic annotation qualification use:

```bash
python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --dry-run --output-dir outputs/thesis_preflight
python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --output-dir outputs/thesis_run_A
python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --output-dir outputs/thesis_run_B
```

## Notes

1. `real_images` stays local-only and requires `DATASELECTOR_IMAGE_DIR`.
2. Full E2E remains opt-in via `RUN_FULL_INTEGRATION=1`.
3. Legacy script-era command examples are intentionally excluded from this page.

