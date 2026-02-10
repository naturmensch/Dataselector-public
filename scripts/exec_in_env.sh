#!/usr/bin/env bash
# exec_in_env.sh - compatibility environment wrapper
# Usage: scripts/exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--yes] -- <command> [args...]

set -euo pipefail

ENV_NAME="dataselector"
ACTION="none"
ASSUME_YES=0
SHOW_HELP=0
ENSURE_PACKAGES=""
YES_ARGS=()

print_help() {
	cat <<'HELP'
Usage: exec_in_env.sh --env <name> [--create|--update|--force-recreate] [--ensure-packages "pkg=1.2 pkg2"] [--yes] -- <command> [args...]

Options:
	--env NAME                Name of the micromamba/conda env (default: dataselector)
	--create                  Create the environment if missing (uses environment.yml)
	--update                  Update environment if present
	--force-recreate          Recreate environment from scratch (destructive)
	--ensure-packages "..."   Ensure specific packages are installed in the env
	--yes                     Assume 'yes' for prompts (non-interactive)
	--help                    Show this help

Behavior:
	Compatibility layer for runtime commands.
	Canonical runtime policy is: micromamba run -n <env> <command>
	This script prefers micromamba, then mamba, then conda. Falls back to .venv if present.

Examples:
	./scripts/exec_in_env.sh --env dataselector --create --yes -- python scripts/run_adaptive_pipeline.py --yes
HELP
}

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

if [[ $SHOW_HELP -eq 1 ]]; then
	print_help
	exit 0
fi

if [[ ${#POSITIONAL[@]} -eq 0 ]]; then
	echo "Error: No command provided to run inside environment." >&2
	print_help
	exit 2
fi

if [[ $ASSUME_YES -eq 1 ]]; then
	YES_ARGS=(-y)
fi

CMD_STR=""
for token in "${POSITIONAL[@]}"; do
	CMD_STR+=" $(printf '%q' "${token}")"
done

cmd_exists() { command -v "$1" >/dev/null 2>&1; }

confirm() {
	if [[ $ASSUME_YES -eq 1 ]]; then
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
VENV_DIR="${REPO_ROOT}/.venv"

pick_runner() {
	if cmd_exists micromamba; then
		echo "micromamba"; return 0
	fi
	if cmd_exists mamba; then
		echo "mamba"; return 0
	fi
	if cmd_exists conda; then
		echo "conda"; return 0
	fi
	echo ""
}

env_exists() {
	local runner="$1"
	case "$runner" in
		micromamba)
			if micromamba env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\""; then
				return 0
			fi
			local root_prefix
			root_prefix="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
			[[ -d "${root_prefix}/envs/${ENV_NAME}" ]]
			;;
		mamba)
			if mamba env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\""; then
				return 0
			fi
			;;
		conda)
			if conda env list --json 2>/dev/null | grep -q "\"${ENV_NAME}\""; then
				return 0
			fi
			;;
	esac
	return 1
}

create_env() {
	local runner="$1"
	if [[ ! -f "${ENV_YML}" ]]; then
		echo "ERROR: environment.yml not found at ${ENV_YML}" >&2
		return 2
	fi
	case "$runner" in
		micromamba) micromamba create -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		mamba) mamba env create -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		conda) conda env create -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		*) echo "ERROR: no runner available" >&2; return 2 ;;
	esac
}

update_env() {
	local runner="$1"
	if [[ ! -f "${ENV_YML}" ]]; then
		echo "ERROR: environment.yml not found at ${ENV_YML}" >&2
		return 2
	fi
	case "$runner" in
		micromamba) micromamba env update -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		mamba) mamba env update -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		conda) conda env update -n "${ENV_NAME}" -f "${ENV_YML}" "${YES_ARGS[@]}" ;;
		*) echo "ERROR: no runner available" >&2; return 2 ;;
	esac
}

remove_env() {
	local runner="$1"
	case "$runner" in
		micromamba) micromamba env remove -n "${ENV_NAME}" "${YES_ARGS[@]}" ;;
		mamba) mamba env remove -n "${ENV_NAME}" "${YES_ARGS[@]}" ;;
		conda) conda env remove -n "${ENV_NAME}" "${YES_ARGS[@]}" ;;
		*) echo "ERROR: no runner available" >&2; return 2 ;;
	esac
}

ensure_packages() {
	local runner="$1"
	local pkg_str="$2"
	if [[ -z "${pkg_str}" ]]; then
		return 0
	fi
	case "$runner" in
		micromamba) micromamba install -n "${ENV_NAME}" ${pkg_str} "${YES_ARGS[@]}" ;;
		mamba) mamba install -n "${ENV_NAME}" ${pkg_str} "${YES_ARGS[@]}" ;;
		conda) conda install -n "${ENV_NAME}" ${pkg_str} "${YES_ARGS[@]}" ;;
		*) echo "ERROR: no runner available" >&2; return 2 ;;
	esac
}

run_in_env() {
	local runner="$1"
	local cmd_str="$2"
	case "$runner" in
		micromamba)
			# micromamba syntax compatibility:
			# prefer current form without explicit `--`, fallback to legacy `--` form only
			# when parser returns argument error.
			EXEC_IN_ENV=1 micromamba run -n "${ENV_NAME}" bash -lc "${cmd_str}" && return 0
			local rc=$?
			if [[ ${rc} -eq 109 ]]; then
				EXEC_IN_ENV=1 micromamba run -n "${ENV_NAME}" bash -lc "${cmd_str}"
				return $?
			fi
			return ${rc}
			;;
		mamba) EXEC_IN_ENV=1 mamba run -n "${ENV_NAME}" -- bash -lc "${cmd_str}" ;;
		conda) EXEC_IN_ENV=1 conda run -n "${ENV_NAME}" -- bash -lc "${cmd_str}" ;;
		*) return 2 ;;
	esac
}

RUNNER="$(pick_runner)"

if [[ "$ACTION" != "none" && -z "$RUNNER" ]]; then
	echo "ERROR: no micromamba/mamba/conda found for env management" >&2
	exit 2
fi

if [[ "$ACTION" == "create" ]]; then
	if env_exists "$RUNNER"; then
		update_env "$RUNNER"
	else
		create_env "$RUNNER"
	fi
elif [[ "$ACTION" == "update" ]]; then
	if env_exists "$RUNNER"; then
		update_env "$RUNNER"
	else
		echo "Environment '${ENV_NAME}' does not exist; use --create to create it." >&2
		exit 2
	fi
elif [[ "$ACTION" == "force-recreate" ]]; then
	if env_exists "$RUNNER"; then
		if confirm "Force recreate environment ${ENV_NAME}? This will remove and re-create (destructive)."; then
			remove_env "$RUNNER"
			create_env "$RUNNER"
		else
			echo "Aborted by user."; exit 1
		fi
	else
		create_env "$RUNNER"
	fi
fi

if [[ -n "$RUNNER" ]] && env_exists "$RUNNER"; then
	ensure_packages "$RUNNER" "$ENSURE_PACKAGES"
	run_in_env "$RUNNER" "$CMD_STR"
	exit $?
fi

if [[ -d "${VENV_DIR}" ]]; then
	bash -lc "source '${VENV_DIR}/bin/activate' && ${CMD_STR}"
	exit $?
fi

echo "ERROR: No suitable env runner found and .venv missing. Install micromamba or conda." >&2
exit 2
