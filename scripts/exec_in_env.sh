#!/usr/bin/env bash
# exec_in_env.sh - canonical execution wrapper
# Usage: scripts/exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--yes] -- <command> [args...]

set -euo pipefail

ENV_NAME="dataselector"
ACTION="none"
FORCE_RECREATE=0
ASSUME_YES=0
SHOW_HELP=0
ENSURE_PACKAGES=""

print_help() {
    cat <<'HELP'
Usage: exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--ensure-packages "pkg=1.2 pkg2"] [--yes] -- <command> [args...]

Options:
  --env NAME                Name of the conda/mamba environment (default: dataselector)
  --create                  Create the environment if missing (uses environment.yml)
  --update                  Update environment if present
  --force-recreate          Recreate environment from scratch (destructive)
  --ensure-packages "..."   Ensure specific packages are installed in the env (conda/mamba)
  --yes                     Assume 'yes' for prompts (non-interactive)
  --help                    Show this help

Behaviour:
  Uses mamba run -n NAME -- <cmd> for execution. Requires mamba to be installed.

Examples:
  ./scripts/exec_in_env.sh --env dataselector --create --ensure-packages "numpy==1.26.4 numba==0.63.1" --yes -- python scripts/run_adaptive_pipeline.py --yes
HELP
}

# parse args
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            ENV_NAME="$2"; shift 2;
            ;;
        --create)
            ACTION="create"; shift;
            ;;
        --update)
            ACTION="update"; shift;
            ;;
        --force-recreate)
            ACTION="force-recreate"; shift;
            ;;
        --yes)
            ASSUME_YES=1; shift;
            ;;
        --help|-h)
            SHOW_HELP=1; shift;
            ;;
        --ensure-packages)
            ENSURE_PACKAGES="$2"; shift 2;
            ;;
        --)
            shift; POSITIONAL+=("$@"); break
            ;;
        *)
            echo "Unknown arg: $1" >&2; print_help; exit 2
            ;;
    esac
done

if [ $SHOW_HELP -eq 1 ]; then
    print_help
    exit 0
fi

if [ ${#POSITIONAL[@]} -eq 0 ]; then
    echo "Error: No command provided to run inside environment." >&2
    print_help
    exit 2
fi

# helper functions
cmd_exists() { command -v "$1" >/dev/null 2>&1; }
confirm() {
    if [ $ASSUME_YES -eq 1 ]; then
        return 0
    fi
    read -r -p "$1 [y/N]: " ans
    case "$ans" in
        [Yy]*) return 0;;
        *) return 1;;
    esac
}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_YML="${REPO_ROOT}/environment.yml"
REQ_TXT="${REPO_ROOT}/requirements.txt"

# check for environment existence (conda env list)
env_exists_conda() {
    # Check both mamba and conda env lists (some systems have mismatched outputs)
    found=1
    if cmd_exists mamba; then
        if mamba env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\""; then
            found=0
        fi
    fi
    if cmd_exists conda; then
        if conda env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\""; then
            found=0
        fi
    fi
    # As a last resort, look for env dirs matching the name under standard envs path
    if [ $found -ne 0 ]; then
        # Look under CONDA_PREFIX envs directory
        PREFIXS=("/opt/miniconda3/envs" "$HOME/.local/conda/envs" "/usr/local/conda/envs")
        for p in "${PREFIXS[@]}"; do
            if [ -d "$p/${ENV_NAME}" ]; then
                found=0; break
            fi
        done
    fi
    if [ $found -eq 0 ]; then
        return 0
    fi
    return 1
}

venv_exists() {
    if [ -d "${REPO_ROOT}/.venv" ] || [ -d "${REPO_ROOT}/venv" ]; then
        return 0
    fi
    return 1
}

# If ENSURE_PACKAGES not set, use validated default pins for reproducibility
if [ -z "${ENSURE_PACKAGES}" ]; then
    ENSURE_PACKAGES="numpy==1.26.4 numba==0.63.1"
fi

print_environment_fix_hint() {
    local ENV_NAME="$1"
    cat >&2 <<EOF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ ENVIRONMENT SETUP REQUIRED

The environment '${ENV_NAME}' must have exact, validated versions:

  numpy==1.26.4
  numba==0.63.1
  umap-learn==0.5.11 (recommended)
  apricot-select==0.6.1 (recommended)

Fix:

  ./scripts/exec_in_env.sh --env ${ENV_NAME} --create \
    --ensure-packages "numpy==1.26.4 numba==0.63.1 umap-learn==0.5.11 apricot-select==0.6.1" --yes -- python scripts/check_env.py

Or:

  mamba env create -f environment.yml
  conda activate ${ENV_NAME}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF
}

validate_environment_before_run() {
    local ENV_NAME="$1"
    echo "Validating environment '${ENV_NAME}'..."

    # Try running the Python-based validation script in the target env
    if cmd_exists mamba; then
        if mamba run -n "${ENV_NAME}" python scripts/check_env.py; then
            return 0
        fi
    elif cmd_exists conda; then
        if conda run -n "${ENV_NAME}" python scripts/check_env.py; then
            return 0
        fi
    fi

    echo "Environment validation failed — attempting auto-repair with ENSURE_PACKAGES: ${ENSURE_PACKAGES}" >&2
    ensure_packages "${ENSURE_PACKAGES}" || {
        echo "Auto-repair failed (ensure_packages)." >&2
        return 1
    }

    # Retry validation
    if cmd_exists mamba; then
        if mamba run -n "${ENV_NAME}" python scripts/check_env.py; then
            return 0
        fi
    elif cmd_exists conda; then
        if conda run -n "${ENV_NAME}" python scripts/check_env.py; then
            return 0
        fi
    fi

    echo "Environment validation failed after attempted repair." >&2
    return 1
}

create_conda_env() {
    if cmd_exists mamba; then
        echo "Creating conda env ${ENV_NAME} with mamba..."
        if [ -f "${ENV_YML}" ]; then
            mamba env create -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y}
        else
            echo "ERROR: environment.yml not found at ${ENV_YML}" >&2; return 2
        fi
    else
        echo "ERROR: mamba not available. Please install mamba." >&2; return 2
    fi
}

update_conda_env() {
    if cmd_exists mamba; then
        echo "Updating conda env ${ENV_NAME} with mamba..."
        if [ -f "${ENV_YML}" ]; then
            mamba env update -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y}
        else
            echo "ERROR: environment.yml not found at ${ENV_YML}" >&2; return 2
        fi
    else
        echo "ERROR: mamba not available. Please install mamba." >&2; return 2
    fi
}

remove_conda_env() {
    if cmd_exists mamba; then
        echo "Removing conda env ${ENV_NAME} with mamba..."
        mamba env remove -n "${ENV_NAME}" ${ASSUME_YES:+-y}
    else
        echo "ERROR: mamba not available. Please install mamba." >&2; return 2
    fi
}

# Ensure packages installed into the env (mamba only)
ensure_packages() {
    PKG_STR="$1"
    echo "Ensuring packages in env ${ENV_NAME}: ${PKG_STR}"
    if cmd_exists mamba; then
        echo "Installing with mamba: mamba install -n ${ENV_NAME} ${PKG_STR}"
        mamba install -n "${ENV_NAME}" ${PKG_STR} ${ASSUME_YES:+-y}
        return $?
    else
        echo "ERROR: mamba not available. Please install mamba." >&2
        return 2
    fi
}

# Try to resolve environment name if exact name missing (fuzzy match)
resolve_env_name() {
    if env_exists_conda; then
        return 0
    fi
    PREFIXS=("/opt/miniconda3/envs" "$HOME/.local/conda/envs" "/usr/local/conda/envs")
    for p in "${PREFIXS[@]}"; do
        for d in "$p/${ENV_NAME}"*; do
            if [ -d "$d" ]; then
                CAND=$(basename "$d")
                echo "Warning: Environment '${ENV_NAME}' not found; using '${CAND}' instead"
                ENV_NAME="$CAND"
                return 0
            fi
        done
    done
    return 1
}

# Decide on env existence
CONDAXISTS=1
if resolve_env_name && env_exists_conda; then
    echo "Detected conda-style env: ${ENV_NAME}"
else
    CONDAXISTS=0
fi

# Handle create/update/force-recreate
if [ "$ACTION" = "create" ]; then
    if [ $CONDAXISTS -eq 1 ]; then
        echo "Environment ${ENV_NAME} already exists. Running update instead of create..."
        update_conda_env
    else
        create_conda_env
    fi
elif [ "$ACTION" = "update" ]; then
    if [ $CONDAXISTS -eq 1 ]; then
        update_conda_env
    else
        echo "Environment ${ENV_NAME} does not exist; use --create to create it." >&2
        exit 2
    fi
elif [ "$ACTION" = "force-recreate" ]; then
    if [ $CONDAXISTS -eq 1 ]; then
        if confirm "Force recreate environment ${ENV_NAME}? This will remove and re-create (destructive)."; then
            remove_conda_env
            create_conda_env
        else
            echo "Aborted by user."; exit 1
        fi
    else
        create_conda_env
    fi
fi

    # Run the command
    # Strict mode: the requested Conda environment must exist and a runner must be available.
    if env_exists_conda; then
        # Build a single safe command string to pass through 'bash -lc' to avoid word-splitting issues
        CMD_STR=""
        for _a in "${POSITIONAL[@]}"; do
            # Use printf %q to properly escape each token for the shell
            CMD_STR="$CMD_STR $(printf '%q' "${_a}")"
        done

        # If requested, ensure additional packages are installed into the existing environment
        if [ -n "${ENSURE_PACKAGES}" ]; then
            ensure_packages "${ENSURE_PACKAGES}" || {
                echo "ERROR: ensure_packages failed" >&2
                exit 2
            }
        fi

        # Run with conda run (fallback to mamba run)
        if cmd_exists conda; then
            if EXEC_IN_ENV=1 conda run -n "${ENV_NAME}" $CMD_STR; then
                exit 0
            else
                echo "ERROR: 'conda run' failed" >&2
                exit 2
            fi
        elif cmd_exists mamba; then
            if EXEC_IN_ENV=1 mamba run -n "${ENV_NAME}" bash -lc "$CMD_STR"; then
                exit 0
            else
                echo "ERROR: 'mamba run' failed" >&2
                exit 2
            fi
        else
            echo "ERROR: neither conda nor mamba available. Please install conda or mamba." >&2
            exit 2
        fi
    else
        echo "ERROR: Conda environment '${ENV_NAME}' not found. Use --create to create it or specify a different --env." >&2
        exit 2
    fi
