#!/usr/bin/env bash
# exec_in_env.sh - canonical execution wrapper
# Usage: scripts/exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--yes] -- <command> [args...]

set -euo pipefail

ENV_NAME="dataselector"
ACTION="none"
FORCE_RECREATE=0
ASSUME_YES=0
SHOW_HELP=0

print_help() {
    cat <<'HELP'
Usage: exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--yes] -- <command> [args...]

Options:
  --env NAME                Name of the conda/mamba environment (default: dataselector)
  --create                  Create the environment if missing (uses environment.yml)
  --update                  Update environment if present
  --force-recreate          Recreate environment from scratch (destructive)
  --yes                     Assume 'yes' for prompts (non-interactive)
  --help                    Show this help

Behaviour:
  1. Prefer mamba run -n NAME -- <cmd>, then conda run -n NAME -- <cmd>
  2. If neither available, prefer an existing .venv/venv in repo root and activate it
  3. If env missing and --create, try mamba/conda create, else fall back to .venv + pip

Examples:
  ./scripts/exec_in_env.sh --env dataselector --create -- python scripts/run_adaptive_pipeline.py --yes
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
    if cmd_exists mamba || cmd_exists conda; then
        if cmd_exists mamba; then
            mamba env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\"" && return 0 || return 1
        else
            conda env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\"" && return 0 || return 1
        fi
    fi
    return 1
}

venv_exists() {
    if [ -d "${REPO_ROOT}/.venv" ] || [ -d "${REPO_ROOT}/venv" ]; then
        return 0
    fi
    return 1
}

create_conda_env() {
    if cmd_exists mamba; then
        echo "Creating conda env ${ENV_NAME} with mamba..."
        if [ -f "${ENV_YML}" ]; then
            mamba env create -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y} || {
                echo "[WARNING] mamba konnte Environment nicht erstellen, versuche conda mit flexible channel priority..."
                if cmd_exists conda; then
                    CONDA_CHANNEL_PRIORITY=flexible conda env create -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y}
                else
                    echo "No conda available; falling back to venv (.venv) creation..."
                    python -m venv "${REPO_ROOT}/.venv"
                    source "${REPO_ROOT}/.venv/bin/activate"
                    if [ -f "${REQ_TXT}" ]; then
                        pip install -r "${REQ_TXT}"
                    fi
                    deactivate
                fi
            }
        else
            echo "ERROR: environment.yml not found at ${ENV_YML}" >&2; return 2
        fi
    elif cmd_exists conda; then
        echo "Creating conda env ${ENV_NAME} with conda (flexible priority)..."
        if [ -f "${ENV_YML}" ]; then
            CONDA_CHANNEL_PRIORITY=flexible conda env create -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y}
        else
            echo "ERROR: environment.yml not found at ${ENV_YML}" >&2; return 2
        fi
    else
        echo "No conda/mamba available; falling back to venv (.venv) creation..."
        python -m venv "${REPO_ROOT}/.venv"
        source "${REPO_ROOT}/.venv/bin/activate"
        if [ -f "${REQ_TXT}" ]; then
            pip install -r "${REQ_TXT}"
        fi
        deactivate
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
    elif cmd_exists conda; then
        echo "Updating conda env ${ENV_NAME} with conda..."
        if [ -f "${ENV_YML}" ]; then
            conda env update -n "${ENV_NAME}" -f "${ENV_YML}" ${ASSUME_YES:+-y}
        else
            echo "ERROR: environment.yml not found at ${ENV_YML}" >&2; return 2
        fi
    else
        echo "No conda/mamba available; cannot update conda env." >&2; return 2
    fi
}

remove_conda_env() {
    if cmd_exists mamba; then
        echo "Removing conda env ${ENV_NAME} with mamba..."
        mamba env remove -n "${ENV_NAME}" ${ASSUME_YES:+-y}
    elif cmd_exists conda; then
        echo "Removing conda env ${ENV_NAME} with conda..."
        conda env remove -n "${ENV_NAME}" ${ASSUME_YES:+-y}
    else
        echo "No conda/mamba available; cannot remove conda env." >&2; return 2
    fi
}

# Decide on env existence
CONDAXISTS=1
if env_exists_conda; then
    echo "Detected conda-style env: ${ENV_NAME}"
else
    CONDAXISTS=0
fi

# Handle create/update/force-recreate
if [ "$ACTION" = "create" ]; then
    if [ $CONDAXISTS -eq 1 ]; then
        echo "Environment ${ENV_NAME} already exists. Use --update or --force-recreate if you want to change it."
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
    # Wenn Conda-Umgebung existiert, IMMER conda run verwenden (ignoriere .venv)
    if env_exists_conda; then
        if cmd_exists mamba; then
            echo "Executing via: mamba run -n ${ENV_NAME} -- ${POSITIONAL[*]}"
            mamba run -n "${ENV_NAME}" -- "${POSITIONAL[@]}"
            exit $?
        elif cmd_exists conda; then
            echo "Executing via: conda run -n ${ENV_NAME} -- ${POSITIONAL[*]}"
            conda run -n "${ENV_NAME}" -- "${POSITIONAL[@]}"
            exit $?
        fi
    fi

    # Fallback: nur wenn KEINE Conda-Umgebung existiert
    if venv_exists; then
        VENV_DIR="${REPO_ROOT}/.venv"
        if [ ! -d "${VENV_DIR}" ]; then
            VENV_DIR="${REPO_ROOT}/venv"
        fi
        echo "Activating venv at ${VENV_DIR} and running command"
        # shellcheck disable=SC1090
        source "${VENV_DIR}/bin/activate"
        "${POSITIONAL[@]}"
        RET=$?
        deactivate
        exit $RET
    fi

    # Last resort: run with current Python environment
    echo "No suitable env runner found; running command with current interpreter (ensure correct env is active)"
    "${POSITIONAL[@]}"
