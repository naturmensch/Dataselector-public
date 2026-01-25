#!/usr/bin/env bash
################################################################################
# Complete Thesis Production Pipeline Orchestrator
# 
# Automatisiert alle 3 Schritte der wissenschaftlichen Datenselektions-Pipeline:
# 1. Multi-Seed Sampler Suite (10 Seeds, 1000 Trials)
# 2. XXL Pipeline mit Konvergenzanalyse (4 Phasen)
# 3. Bootstrap Uncertainty Quantification (500 Resamples)
#
# Alle Parameter werden automatisch berechnet - keine Konfiguration nötig!
#
# Usage:
#   bash scripts/run_complete_thesis_pipeline.sh
#   # oder mit Custom Environment:
#   bash scripts/run_complete_thesis_pipeline.sh --env my-env
#
# Logs:
#   - outputs/sampler_suite_run_scientific.log (Sampler Suite)
#   - outputs/XXL_FULL_RUN.log (XXL Pipeline)
#   - outputs/bootstrap_run.log (Bootstrap)
################################################################################

set -euo pipefail

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASELECTOR_ENV="dataselector"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS_DIR="${ROOT}/outputs"
LOGS_DIR="${OUTPUTS_DIR}"
# Allow overriding data directory for tests
DATA_DIR="${DATA_DIR:-${ROOT}/data}"

# Create outputs directory
mkdir -p "${OUTPUTS_DIR}"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "${LOGS_DIR}/thesis_pipeline.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "${LOGS_DIR}/thesis_pipeline.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOGS_DIR}/thesis_pipeline.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "${LOGS_DIR}/thesis_pipeline.log"
}

# Check environment
check_environment() {
    log_info "Prüfe Umgebung..."

    # Prefer canonical wrapper if available (skip if SKIP_EXEC_IN_ENV=1)
    if [ "${SKIP_EXEC_IN_ENV:-0}" != "1" ] && [ -f "${ROOT}/scripts/exec_in_env.sh" ]; then
        RUNNER="${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} --"
        # Ensure env exists (create if missing) non-interactively
        echo "Ensuring environment '${DATASELECTOR_ENV}' exists (using exec_in_env.sh)"
        ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} --create --yes -- true || {
            log_error "Konnte Environment '${DATASELECTOR_ENV}' nicht erstellen!"
            exit 1
        }
        log_success "Environment '${DATASELECTOR_ENV}' aktiv (via exec_in_env.sh)"
        return 0
    fi

    # Fallback to checking mamba/conda directly
    if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
        log_error "Weder mamba noch conda gefunden! Bitte installieren Sie eine der beiden oder nutzen Sie '; ./scripts/exec_in_env.sh'"
        exit 1
    fi

    # Try to activate environment
    if command -v mamba &> /dev/null; then
        RUNNER="mamba run -n ${DATASELECTOR_ENV}"
    else
        RUNNER="conda run -n ${DATASELECTOR_ENV}"
    fi

    # Quick test
    ./scripts/exec_in_env.sh --env dataselector -- if ! ${RUNNER} python --version &>/dev/null; then
        log_error "Kann Environment '${DATASELECTOR_ENV}' nicht aktivieren!"
        log_info "Verfügbare Environments:"
        if command -v mamba &> /dev/null; then
            mamba env list
        else
            conda env list
        fi
        exit 1
    fi

    log_success "Environment '${DATASELECTOR_ENV}' aktiv"
}

# Step 1: Sampler Suite
step_1_sampler_suite() {
    log_info "==================================================================="
    log_info "SCHRITT 1: Thesis Sampler Suite starten"
    log_info "Samples: 10 Seeds [42-51] × 3 Samplers × 1000 Trials"
    log_info "Datasets: Hamburg + KDR100"
    log_info "Erwartete Dauer: 8-12 Stunden"
    log_info "==================================================================="
    
    start_time=$(date +%s)
    
    # Run sampler suite WITHOUT timeout constraints
    timeout_val="" # No timeout
    ./scripts/exec_in_env.sh --env dataselector -- ${RUNNER} python scripts/run_thesis_sampler_suite.py \
        --seeds 42 43 44 45 46 47 48 49 50 51 \
        --n-trials 1000 \
        --datasets hamburg kdr100 \
        --samplers qmc tpe cmaes \
        2>&1 | tee "${LOGS_DIR}/sampler_suite_run_scientific.log"
    
    if [ $? -ne 0 ]; then
        log_error "Sampler Suite fehlgeschlagen!"
        exit 1
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_success "Sampler Suite abgeschlossen ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
    
    # Verify selected_sampler.json exists
    if [ ! -f "${OUTPUTS_DIR}/selected_sampler.json" ]; then
        log_error "selected_sampler.json nicht gefunden - Sampler Suite möglicherweise fehlgeschlagen!"
        exit 1
    fi
    
    log_success "✓ selected_sampler.json erfolgreich generiert"
}

# Step 2: XXL Pipeline with Autoscale Results
step_2_xxl_pipeline() {
    log_info "==================================================================="
    log_info "SCHRITT 2: XXL Pipeline (Phasen 0-5) mit Autoscale-Ergebnissen"
    log_info "Nutzt: Best Sampler + Optimierte Hyperparams + Bootstrap UQ"
    log_info "Phase 0: Konvergenzanalyse (Hamburg, n_samples aus Autoscale)"
    log_info "Phase 1-4: KDR100 Optimierung (alle 676 Kacheln)"
    log_info "Phase 5: Bootstrap UQ (500 Resamples)"
    log_info "Erwartete Dauer: 3.5-6 Stunden (mit Bootstrap)"
    log_info "==================================================================="
    
    start_time=$(date +%s)
    
    # Read best_sampler and autoscale results from Schritt 1
    if [ -f "${OUTPUTS_DIR}/selected_sampler.json" ]; then
        BEST_SAMPLER=$(grep -o '"best":"[^"]*' "${OUTPUTS_DIR}/selected_sampler.json" | cut -d'"' -f4)
        log_info "Best Sampler (from suite): $BEST_SAMPLER"
    else
        BEST_SAMPLER="tpe"
        log_warning "No best sampler found, defaulting to TPE"
    fi
    
    if [ -f "${OUTPUTS_DIR}/optuna_autoscale_selected_n_samples.txt" ]; then
        AUTOSCALE_N_SAMPLES=$(cat "${OUTPUTS_DIR}/optuna_autoscale_selected_n_samples.txt")
        log_info "Autoscale n_samples: $AUTOSCALE_N_SAMPLES"
    fi
    
    # If a small test config is provided (e.g., via monitor tests), run XXL in smoke mode
    if [ -n "${CONFIG_PATH:-}" ]; then
        log_info "CONFIG_PATH set -> running XXL in smoke mode for faster, deterministic test runs"
        ./scripts/exec_in_env.sh --env dataselector -- ${RUNNER} python scripts/xxl_KDR146_run_thesis_complete_modern.py --best-sampler "$BEST_SAMPLER" --smoke 2>&1 | tee "${LOGS_DIR}/XXL_FULL_RUN.log"
    else
        ./scripts/exec_in_env.sh --env dataselector -- ${RUNNER} python scripts/xxl_KDR146_run_thesis_complete_modern.py \
            --best-sampler "$BEST_SAMPLER" \
            2>&1 | tee "${LOGS_DIR}/XXL_FULL_RUN.log"
    fi
    
    if [ $? -ne 0 ]; then
        log_error "XXL Pipeline fehlgeschlagen!"
        exit 1
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_success "XXL Pipeline mit Bootstrap abgeschlossen ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
}

# Old Step 3 (Bootstrap) is now removed - integrated into Phase 5


# Add CLI argument parsing for dry-run
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1; shift ;;
        --env)
            DATASELECTOR_ENV="$2"; shift 2 ;;
        *)
            # pass through other args
            shift ;;
    esac
done

preflight_check() {
    log_info "Running preflight checks..."

    # Ensure wrapper exists
    if [ ! -f "${ROOT}/scripts/exec_in_env.sh" ]; then
        log_error "Missing exec_in_env.sh wrapper at ${ROOT}/scripts/exec_in_env.sh"
        return 1
    fi

    # Check required Python packages inside environment
    PKGS=(optuna pandas numpy matplotlib seaborn)
    MISSING=()
    for p in "${PKGS[@]}"; do
        ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- python - <<PY
try:
    import ${p}
    print('OK')
except Exception as e:
    raise SystemExit(1)
PY
        if [ $? -ne 0 ]; then
            MISSING+=("${p}")
        fi
    done

    if [ ${#MISSING[@]} -ne 0 ]; then
        log_warning "Missing Python packages in environment: ${MISSING[*]}"
        log_info "You can install them with: ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- pip install ${MISSING[*]}"
    else
        log_success "All required Python packages present in environment"
    fi

    # Check presence of important files
    if [ ! -f "${DATA_DIR}/new_all_tiles.csv" ]; then
        log_warning "data/new_all_tiles.csv not found - attempting to regenerate from image sidecars..."
        if [ -f "${ROOT}/scripts/build_new_all_tiles.py" ]; then
            # Prefer running inside canonical env wrapper if available
            if [ -f "${ROOT}/scripts/exec_in_env.sh" ]; then
                ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- python "${ROOT}/scripts/build_new_all_tiles.py" --image-dir "${DATA_DIR}/images" || {
                    log_warning "Regeneration of new_all_tiles.csv failed - scripts will fall back to defaults"
                }
                
                # Also ensure we prefer running subsequent helper scripts via RUNNER when available
            else
                if [ -n "${RUNNER:-}" ]; then
                    ./scripts/exec_in_env.sh --env dataselector -- ${RUNNER} python "${ROOT}/scripts/build_new_all_tiles.py" --image-dir "${DATA_DIR}/images" || {
                        log_warning "Regeneration of new_all_tiles.csv failed - scripts will fall back to defaults"
                    }
                else
                    ./scripts/exec_in_env.sh --env dataselector -- python "${ROOT}/scripts/build_new_all_tiles.py" --image-dir "${DATA_DIR}/images" || {
                        log_warning "Regeneration of new_all_tiles.csv failed - scripts will fall back to defaults"
                    }
                fi
            fi
        else
            log_warning "No generator script found (scripts/build_new_all_tiles.py) - scripts will fall back to defaults"
        fi
    fi

    if [ ! -f "${ROOT}/scripts/optuna_autoscale.py" ] || [ ! -f "${ROOT}/scripts/compare_samplers_multi_seed.py" ] || [ ! -f "${ROOT}/scripts/xxl_KDR146_run_thesis_complete_modern.py" ]; then
        log_error "One or more pipeline scripts are missing (optuna_autoscale.py, compare_samplers_multi_seed.py, xxl_KDR146_run_thesis_complete_modern.py)"
        return 1
    fi

    log_success "Preflight checks passed (with warnings if any)"
    return 0
}

# Dry-run target that performs minimal executions of all components
dry_run() {
    log_info "Starting DRY RUN: running minimal autoscale, sampler suite and XXL stub"

    preflight_check || { log_error "Preflight failed - aborting dry run"; exit 1; }

    # Minimal autoscale
    log_info "Running minimal Autoscale (small stages)"
    ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- python scripts/optuna_autoscale.py --n-candidates 50 --stages 40 80 --n-trials 2 2 --patience 1 2>&1 | tee "${LOGS_DIR}/dry_autoscale.log"

    # Minimal sampler suite (no autoscale, short trials)
    log_info "Running minimal Sampler Suite (1 seed × 5 trials)"
    ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- python scripts/run_thesis_sampler_suite.py --seeds 42 --n-trials 5 --n-trials-full 5 --datasets hamburg --samplers qmc tpe cmaes --no-autoscale --n-candidates 50 2>&1 | tee "${LOGS_DIR}/dry_sampler_suite.log"

    # XXL stub (reads selected_sampler.json if present)
    BEST="tpe"
    if [ -f "${OUTPUTS_DIR}/selected_sampler.json" ]; then
        BEST=$(grep -o '"best":"[^\"]*' "${OUTPUTS_DIR}/selected_sampler.json" | cut -d'"' -f4 || echo "tpe")
    fi
    log_info "Running XXL stub with best sampler: ${BEST}"
    ${ROOT}/scripts/exec_in_env.sh --env ${DATASELECTOR_ENV} -- python scripts/xxl_KDR146_run_thesis_complete_modern.py --best-sampler ${BEST} 2>&1 | tee "${LOGS_DIR}/dry_xxl.log"

    log_success "Dry run complete. Check logs: ${LOGS_DIR}/dry_*.log"
}

# Main orchestration
main() {
    log_info "🚀 COMPLETE THESIS PIPELINE ORCHESTRATOR"
    log_info "Starte automatisierte Production Pipeline..."
    log_info ""

    check_environment

    if [ "$DRY_RUN" -eq 1 ]; then
        dry_run
        exit 0
    fi

    echo ""

    # Run all 2 steps (Bootstrap now integrated into Step 2)
    OVERALL_START=$(date +%s)

    step_1_sampler_suite
    echo ""

    step_2_xxl_pipeline
    echo ""

    OVERALL_END=$(date +%s)
    OVERALL_DURATION=$((OVERALL_END - OVERALL_START))

    log_info "==================================================================="
    log_success "✅ COMPLETE THESIS PIPELINE (INKL. BOOTSTRAP) ABGESCHLOSSEN!"
    log_info "==================================================================="
    log_info "Gesamtdauer: $(printf '%02d:%02d:%02d' $((OVERALL_DURATION/3600)) $((OVERALL_DURATION%3600/60)) $((OVERALL_DURATION%60)))"
    log_info ""
    log_info "Outputs:"
    log_info "  - Sampler Suite: ${OUTPUTS_DIR}/runs/sampler_thesis_suite_*/"
    log_info "  - XXL Pipeline + Bootstrap: ${OUTPUTS_DIR}/runs/thesis_xxl_hamburg_final/"
    log_info ""
    log_info "Thesis-Ready Artifacts:"
    log_info "  - ${OUTPUTS_DIR}/THESIS_FINAL_SELECTION_XXL.json"
    log_info "  - ${OUTPUTS_DIR}/THESIS_XXL_SUMMARY.md"
    log_info "  - ${OUTPUTS_DIR}/runs/thesis_xxl_hamburg_final/results/bootstrap_final_selection_summary.csv"
    log_info ""
    log_info "Logs:"
    log_info "  - ${LOGS_DIR}/thesis_pipeline.log (Überblick)"
    log_info "  - ${LOGS_DIR}/sampler_suite_run_scientific.log (Schritt 1)"
    log_info "  - ${LOGS_DIR}/XXL_FULL_RUN.log (Schritt 2: Phase 0-5)"
    log_info ""
    log_info "📊 Bereit für Thesis-Integration!"
}

# Run main
main "$@"
