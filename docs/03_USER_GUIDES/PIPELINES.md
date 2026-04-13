# Pipelines Overview (Authoritative)

This page summarizes the retained thesis-relevant pipelines.

Primary runbook:

- `docs/03_USER_GUIDES/THESIS_PIPELINE_HOWTO.md`

## Pipeline Map

1. `thesis-orchestrate`
   - Canonical end-to-end thesis orchestration with snapshotting and run metadata.
2. `thesis-pipeline`
   - Canonical direct pipeline execution from validated parameters.
3. `thesis-sampler-suite`
   - Supplementary sampler comparison workflow.
4. `benchmark-sampling`
   - Supplementary exploration benchmark workflow.
5. `adaptive-auto`
   - Supplementary orchestrator for autoscale handoff + adaptive pipeline.

## Canonical Commands

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate --help
micromamba run -n dataselector python -m dataselector thesis-pipeline --help
micromamba run -n dataselector python -m dataselector thesis-sampler-suite --help
micromamba run -n dataselector python -m dataselector generate-monitor --help
```

## Typical Thesis Sequence

```bash
micromamba run -n dataselector python -m dataselector thesis-orchestrate \
  --config config/pipeline_config.yaml \
  --output-dir outputs/runs/<run_id>

micromamba run -n dataselector python -m dataselector thesis-pipeline \
  --use-params outputs/runs/<run_id>/final_config.yaml
```

For deterministic annotation qualification use:

```bash
micromamba run -n dataselector python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --dry-run --output-dir outputs/runs/thesis_preflight
micromamba run -n dataselector python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --output-dir outputs/runs/thesis_run_A
micromamba run -n dataselector python -m dataselector thesis-pipeline --execution-profile thesis_repro --seed 42 --output-dir outputs/runs/thesis_run_B
```

## Notes

1. `real_images` stays local-only and requires `DATASELECTOR_IMAGE_DIR`.
2. `generate-monitor` summarizes canonical thesis run artifacts under `outputs/runs/`.
3. The former `xxl` / `xxl-monitor` surface is archived and not part of the active pipeline contract.
