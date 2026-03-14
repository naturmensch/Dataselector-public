# Phase4H Automation Runner

> Historical closeout automation.
> This folder is preserved for traceability and replay of the Phase4H
> closeout, but it is **not** the canonical thesis-v2 runtime path.
> For current release-grade execution, use
> `python -m dataselector thesis-orchestrate` or
> `python -m dataselector thesis-pipeline`.

This folder contains resumable wave scripts for the Phase4H scientific closeout.

## Entry Point

```bash
scripts/phase4h/run_all.sh
```

Resume from a specific wave:

```bash
scripts/phase4h/run_all.sh --resume-from wave2_distance
```

Force re-run for a stamped wave:

```bash
scripts/phase4h/run_all.sh --force-wave wave2_distance
```

## Waves

1. `wave1_city` (`run_wave_1_city.sh`)
2. `wave2_distance` (`run_wave_2_distance.sh`)
3. `wave3_nsamples` (`run_wave_3_nsamples.sh`)
4. `wave4_docs_gates` (`run_wave_4_docs_gates.sh`)
5. `wave5_golive_policy24` (`run_wave_5_golive_policy24.sh`)
6. `wave6_docs_finalize` (`run_wave_6_docs_finalize.sh`)
7. `wave7_final_gates` (`run_wave_7_final_gates.sh`)

## Status / Logs / Stamps

1. Logs: `outputs/phase4h/logs/`
2. Stamps: `outputs/phase4h/.stamps/`
3. Plan status updates are appended to:
   - `docs/status/phase4h_scientific_completion_plan_2026-02-09.md`

## Runtime

Run Phase4H through the env wrapper using `DATASELECTOR_ENV_NAME` (defaults to `dataselector`):

```bash
scripts/exec_in_env.sh --env "${DATASELECTOR_ENV_NAME:-dataselector}" -- \
   scripts/phase4h/run_all.sh
```

Resume from a wave with the same launcher:

```bash
scripts/exec_in_env.sh --env "${DATASELECTOR_ENV_NAME:-dataselector}" -- \
   scripts/phase4h/run_all.sh --resume-from wave2_distance
```

Create the env on first run if needed:

```bash
scripts/exec_in_env.sh --env "${DATASELECTOR_ENV_NAME:-dataselector}" --create --yes -- \
   scripts/phase4h/run_all.sh
```

## Manual Black Step

`wave7_final_gates` intentionally pauses before completion and asks for manual
`black --check`. Resume with:

```bash
PHASE4H_BLACK_CONFIRMED=1 scripts/phase4h/run_all.sh --resume-from wave7_final_gates
```

## Repro Helpers

For standalone evidence reruns:

1. `scripts/reproduce_min_distance_decision.sh`
2. `scripts/reproduce_n_samples_decision.sh`
