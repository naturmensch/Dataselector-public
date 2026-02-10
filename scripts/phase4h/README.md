# Phase4H Automation Runner

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

The runner uses:

```bash
/opt/miniconda3/envs/dataselector/bin/python
```

Override with:

```bash
PYTHON_BIN=/path/to/python scripts/phase4h/run_all.sh
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
