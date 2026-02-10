#!/usr/bin/env bash
# Orchestrator for a full / intensive experiment run
# Usage examples:
#   ./scripts/run_full_experiment.sh                # runs all steps with defaults
#   ./scripts/run_full_experiment.sh --n-trials 200 --n-boot 200
#   ./scripts/run_full_experiment.sh --skip-optuna --yes

set -euo pipefail
IFS=$'\n\t'

# Defaults
N_TRIALS=200
N_CANDIDATES=500
DIM=256
N_SAMPLES=34
MIN_DISTANCE_KM=28
N_BOOT=200
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="outputs/runs/run_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

SKIP_COARSE=0
SKIP_FINE=0
SKIP_OPTUNA=0
SKIP_BOOTSTRAP=0
SKIP_FINAL=0
USE_OPTUNA_BEST=0
USE_OPTUNA_INJECT=0
FINAL_WITH_OPTUNA_CONFIG=0
ASSUME_YES=0

function usage() {
  cat <<EOF
Usage: $0 [options]
Options:
  --skip-coarse         Skip coarse grid sweep
  --skip-fine           Skip fine grid sweep
  --skip-optuna         Skip Optuna optimization
  --skip-bootstrap      Skip bootstrap robustness analysis
  --skip-final          Skip final selection run
  --use-optuna-best      After running Optuna, extract best trial and write config to experiment folder
  --inject-optuna        Inject best Optuna trial directly into 'config/pipeline_config.yaml' (backup created)
  --final-with-optuna-config Run final selection temporarily using the generated Optuna config
  --n-trials N           Optuna trials (default: ${N_TRIALS})
  --n-candidates N       Optuna candidates (default: ${N_CANDIDATES})
  --n-boot N             Bootstrap resamples (default: ${N_BOOT})
  --yes                  Non-interactive (assume yes)
  -h, --help             Show this help

Example:
  $0 --n-trials 300 --n-boot 300
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-coarse) SKIP_COARSE=1; shift ;;
    --skip-fine) SKIP_FINE=1; shift ;;
    --skip-optuna) SKIP_OPTUNA=1; shift ;;
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --skip-final) SKIP_FINAL=1; shift ;;
    --use-optuna-best) USE_OPTUNA_BEST=1; shift ;;
    --inject-optuna) USE_OPTUNA_INJECT=1; shift ;;
    --final-with-optuna-config) FINAL_WITH_OPTUNA_CONFIG=1; shift ;;
    --n-trials) N_TRIALS="$2"; shift 2 ;;
    --n-candidates) N_CANDIDATES="$2"; shift 2 ;;
    --n-boot) N_BOOT="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --min-distance-km) MIN_DISTANCE_KM="$2"; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# Summary
echo "Experiment run: ${TIMESTAMP}"
echo "Output dir: ${OUT_DIR}"
echo "Steps: "
[[ $SKIP_COARSE -eq 0 ]] && echo "  - coarse sweep"
[[ $SKIP_FINE -eq 0 ]] && echo "  - fine sweep"
[[ $SKIP_OPTUNA -eq 0 ]] && echo "  - optuna (n_trials=${N_TRIALS})"
[[ $SKIP_BOOTSTRAP -eq 0 ]] && echo "  - bootstrap (n_boot=${N_BOOT})"
[[ $SKIP_FINAL -eq 0 ]] && echo "  - final selection"

if [[ $ASSUME_YES -eq 0 ]]; then
  read -p "Proceed with the run? [y/N] " RESP
  if [[ "${RESP,,}" != "y" ]]; then
    echo "Aborting by user request."; exit 0
  fi
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Detect wrapper presence and prefer using it to enforce environment
WRAPPER=""
if [ -f "${ROOT}/scripts/exec_in_env.sh" ]; then
  WRAPPER="${ROOT}/scripts/exec_in_env.sh --env dataselector --"
fi

# Helper to run a command with timestamped logging
# NOTE: We join all args into a shell command string and run via 'bash -lc' so
# constructs like 'PYTHONPATH=. python script.py' work correctly.
run_step() {
  local label="$1"; shift
  local cmd_str="$*"
  local logfile="${OUT_DIR}/${label}.log"
  echo "\n=== [$label] Starting: $(date -u +%FT%TZ) ==="
  echo "> ${cmd_str}" | tee -a "$logfile"

  # If wrapper available, prefix the command so it runs inside the dataselector env
  if [ -n "${WRAPPER}" ]; then
    prefixed_cmd="${WRAPPER} ${cmd_str}"
  else
    prefixed_cmd="${cmd_str}"
  fi

  if bash -lc "$prefixed_cmd" >>"$logfile" 2>&1; then
    echo "[${label}] Completed: $(date -u +%FT%TZ)" | tee -a "$logfile"
  else
    echo "[${label}] FAILED (see ${logfile})" | tee -a "$logfile"
    exit 1
  fi
}

# 1) Coarse sweep
if [[ $SKIP_COARSE -eq 0 ]]; then
  run_step "coarse_sweep" PYTHONPATH=. python scripts/run_coarse_sweep.py
fi

# 2) Fine sweep
if [[ $SKIP_FINE -eq 0 ]]; then
  run_step "fine_sweep" PYTHONPATH=. python scripts/run_fine_sweep.py
fi

# 3) Optuna optimization
if [[ $SKIP_OPTUNA -eq 0 ]]; then
  run_step "optuna" PYTHONPATH=. python scripts/optuna_optimize.py --n-trials ${N_TRIALS} --n-candidates ${N_CANDIDATES} --dim ${DIM} --n-samples ${N_SAMPLES} --min-distance-km ${MIN_DISTANCE_KM}
  # copy results into experiment folder
  cp -v outputs/optuna_results.csv "${OUT_DIR}/" || true
  cp -v outputs/optuna_study.pkl "${OUT_DIR}/" 2>/dev/null || true
fi

# Optionally apply the best Optuna trial to the pipeline config
if [[ $USE_OPTUNA_BEST -eq 1 ]]; then
  OPTUNA_CSV="outputs/optuna_results.csv"
  if [[ -f "$OPTUNA_CSV" ]]; then
    OPTUNA_CONFIG_OUT="${OUT_DIR}/pipeline_config.optuna.yaml"
    if [[ $USE_OPTUNA_INJECT -eq 1 ]]; then
      # Inject directly into the repo config (creates backup)
      run_step "apply_optuna_best_inject" PYTHONPATH=. python scripts/apply_optuna_best.py --optuna-csv "$OPTUNA_CSV" --inject
      # copy backup to experiment folder for provenance
      BAK_SRC="config/pipeline_config.yaml.optuna_bak"
      if [[ -f "$BAK_SRC" ]]; then
        cp -v "$BAK_SRC" "${OUT_DIR}/" || true
      fi
      APPLIED_CONFIG="config/pipeline_config.yaml"
      echo "Optuna best trial injected into config/pipeline_config.yaml (backup saved)."
    else
      # Write a separate config file in the experiment folder (safer)
      run_step "apply_optuna_best_write" PYTHONPATH=. python scripts/apply_optuna_best.py --optuna-csv "$OPTUNA_CSV" --write-config "$OPTUNA_CONFIG_OUT"
      APPLIED_CONFIG="$OPTUNA_CONFIG_OUT"
      echo "Optuna best config written to: ${OPTUNA_CONFIG_OUT}"
    fi

    # Copy the applied config into the experiment folder for provenance
    if [[ -n "${APPLIED_CONFIG:-}" && -f "${APPLIED_CONFIG}" ]]; then
      cp -v "${APPLIED_CONFIG}" "${OUT_DIR}/" || true
      echo "Copied applied config to ${OUT_DIR}/$(basename ${APPLIED_CONFIG})"
    fi
  else
    echo "Optuna results not found ($OPTUNA_CSV). Skipping config injection.";
  fi
fi

# 4) Bootstrap robustness analysis (requires fine_sweep/pareto)
if [[ $SKIP_BOOTSTRAP -eq 0 ]]; then
  PARETO="outputs/fine_sweep/pareto_solutions.csv"
  if [[ ! -f "$PARETO" ]]; then
    echo "Pareto file not found at ${PARETO}. Skipping bootstrap (or run fine sweep first)."; exit 1
  fi
  run_step "bootstrap" PYTHONPATH=. python scripts/bootstrap_pareto_candidates.py --pareto ${PARETO} --n-boot ${N_BOOT} --out ${OUT_DIR}/bootstrap_results.csv --seed 42
  cp -v outputs/fine_sweep/bootstrap_summary.csv "${OUT_DIR}/" 2>/dev/null || true
fi

# 5) Final selection run (use recommended config or pick best found config manually)
if [[ $SKIP_FINAL -eq 0 ]]; then
  if [[ $FINAL_WITH_OPTUNA_CONFIG -eq 1 ]]; then
    # Make sure we have an applied config path
    if [[ -z "${APPLIED_CONFIG:-}" ]]; then
      echo "No Optuna-generated config available to run final with."
      exit 1
    fi
    # Backup original config and replace temporarily
    cp -v config/pipeline_config.yaml config/pipeline_config.yaml.optuna_bak || true
    echo "Using Optuna config for final run: ${APPLIED_CONFIG} -> config/pipeline_config.yaml"
    cp -v "${APPLIED_CONFIG}" config/pipeline_config.yaml
    run_step "final_selection_optuna" PYTHONPATH=. python scripts/final_selection.py
    # After run, restore original config
    if [[ -f config/pipeline_config.yaml.optuna_bak ]]; then
      echo "Restoring original config from config/pipeline_config.yaml.optuna_bak"
      cp -v config/pipeline_config.yaml.optuna_bak config/pipeline_config.yaml
    fi
  else
    run_step "final_selection" PYTHONPATH=. python scripts/final_selection.py
  fi
  cp -v outputs/final_selection/* "${OUT_DIR}/" 2>/dev/null || true
fi

# If we injected Optuna best (explicit injection), restore original config (backup was created beside config)
if [[ $USE_OPTUNA_INJECT -eq 1 ]]; then
  BAK_FILE=config/pipeline_config.yaml.optuna_bak
  if [[ -f "$BAK_FILE" ]]; then
    echo "Restoring original config from $BAK_FILE"
    cp -v "$BAK_FILE" config/pipeline_config.yaml
    echo "Original config restored."
  fi
fi

echo "\nAll requested steps finished. Experiment outputs are under: ${OUT_DIR}"

echo "Done."
