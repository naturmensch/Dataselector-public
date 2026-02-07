#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYBIN="/opt/miniconda3/envs/dataselector/bin/python"
RUNNER_NAME="$(basename "$0")"

MODE="dry-run"
WAVE=""
LIMIT="20"
COMMAND=""

ALLOWED_NONBLOCKING_WORKFLOWS=(
  ".github/workflows/docs-link-check.yml"
  ".github/workflows/verify-env-usage.yml"
  ".github/workflows/verify-archive.yml"
  ".github/workflows/geo-integration.yml"
  ".github/workflows/smoke-tests.yml"
  ".github/workflows/regenerate-lockfile.yml"
)

report_dir=""

usage() {
  cat <<EOF
Usage:
  ${RUNNER_NAME} <command> [options]

Commands:
  preflight                 Verify auth/env/git safety constraints
  collect-stability         Export workflow run stability report
  gates                     Run mandatory Phase 4 test/lint safety gates
  verify-safety             Fail on unsafe branch changes
  full                      Run preflight + collect-stability + verify-safety + gates

Options:
  --wave <c|b|d|generic>    Wave identifier for artifact path (required for collect/full)
  --limit <N>               Workflow run sample size for collect-stability (default: 20)
  --dry-run                 Default mode; no mutating actions
  --apply                   Explicitly allow mutating actions (reserved)
  -h, --help                Show this message

Examples:
  ${RUNNER_NAME} preflight
  ${RUNNER_NAME} collect-stability --wave c --limit 20
  ${RUNNER_NAME} full --wave c
EOF
}

log() {
  printf '[phase4-runner] %s\n' "$*"
}

die() {
  printf '[phase4-runner][error] %s\n' "$*" >&2
  exit 1
}

parse_args() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  COMMAND="$1"
  shift

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --wave)
        WAVE="${2:-}"
        shift 2
        ;;
      --limit)
        LIMIT="${2:-}"
        shift 2
        ;;
      --dry-run)
        MODE="dry-run"
        shift
        ;;
      --apply)
        MODE="apply"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

prepare_report_dir() {
  local wave="${WAVE:-generic}"
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  report_dir="${REPO_ROOT}/artifacts/phase4/${wave}/${ts}"
  mkdir -p "${report_dir}"
  log "Artifacts: ${report_dir}"
}

cmd_preflight() {
  cd "${REPO_ROOT}"

  log "Checking gh auth"
  gh auth status >/dev/null

  log "Checking authoritative python env"
  [[ -x "${PYBIN}" ]] || die "Missing authoritative python: ${PYBIN}"

  log "Checking clean tracked tree"
  git diff --quiet || die "Tracked unstaged changes present. Commit/stash before running."
  git diff --cached --quiet || die "Staged but uncommitted changes present."

  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"
  [[ "${branch}" == "main" || "${branch}" == phase4/* ]] || \
    die "Run only on main or phase4/* branch. Current branch: ${branch}"

  log "Preflight passed"
}

cmd_collect_stability() {
  [[ -n "${WAVE}" ]] || die "--wave is required for collect-stability"
  [[ "${LIMIT}" =~ ^[0-9]+$ ]] || die "--limit must be integer"

  prepare_report_dir

  cd "${REPO_ROOT}"
  local report_md="${report_dir}/workflow_stability.md"
  local report_json="${report_dir}/workflow_stability.jsonl"
  : >"${report_json}"

  local workflows=(
    "Geo Integration Tests"
    "Docs Link Check"
    "Verify Env Usage"
    "Verify Archive References"
    "Smoke tests"
    "Regenerate Lockfile"
  )

  {
    echo "# Workflow Stability Report"
    echo
    echo "- Wave: ${WAVE}"
    echo "- Sample size per workflow: ${LIMIT}"
    echo "- Generated: $(date -Iseconds)"
    echo
    echo "| Workflow | completed | success | failure | cancelled | unknown |"
    echo "|---|---:|---:|---:|---:|---:|"
  } >"${report_md}"

  for wf in "${workflows[@]}"; do
    local raw out
    raw="$(gh run list --workflow "${wf}" --limit "${LIMIT}" --json conclusion,status,headBranch,createdAt)"
    printf '{"workflow":%q,"runs":%s}\n' "${wf}" "${raw}" >>"${report_json}"

    out="$("${PYBIN}" - <<'PY' "${raw}"
import json
import sys

runs = json.loads(sys.argv[1])
completed = sum(1 for r in runs if r.get("status") == "completed")
success = sum(1 for r in runs if r.get("conclusion") == "success")
failure = sum(1 for r in runs if r.get("conclusion") == "failure")
cancelled = sum(1 for r in runs if r.get("conclusion") == "cancelled")
unknown = len(runs) - success - failure - cancelled
print(f"{completed}\t{success}\t{failure}\t{cancelled}\t{unknown}")
PY
)"
    IFS=$'\t' read -r completed success failure cancelled unknown <<<"${out}"
    printf '| %s | %s | %s | %s | %s | %s |\n' \
      "${wf}" "${completed}" "${success}" "${failure}" "${cancelled}" "${unknown}" \
      >>"${report_md}"
  done

  log "Wrote ${report_md}"
  log "Wrote ${report_json}"
}

cmd_verify_safety() {
  cd "${REPO_ROOT}"
  log "Verifying safety boundaries"

  local changed
  changed="$(git diff --name-only origin/main...HEAD || true)"
  if [[ -n "${changed}" ]] && echo "${changed}" | rg -q '^dataselector/'; then
    die "Product code changes detected under dataselector/ in Phase 4 wave."
  fi

  local added_coe
  added_coe="$(git diff --unified=0 origin/main...HEAD | rg '^\+.*continue-on-error:\s*true' || true)"
  if [[ -n "${added_coe}" ]]; then
    die "New continue-on-error: true additions detected:\n${added_coe}"
  fi

  local broad_skips
  broad_skips="$(git diff --unified=0 origin/main...HEAD | rg '^\+.*(@pytest\.mark\.(skip|xfail)|pytest\.skip\()' || true)"
  if [[ -n "${broad_skips}" ]]; then
    die "New broad skip/xfail additions detected:\n${broad_skips}"
  fi

  log "Safety verification passed"
}

cmd_gates() {
  cd "${REPO_ROOT}"
  log "Running mandatory Phase 4 gates"
  "${PYBIN}" -m pytest -q tests/unit/test_no_legacy_script_references.py
  "${PYBIN}" -m pytest -q tests -k "not real_images"
  "${PYBIN}" -m pytest -q tests/unit/test_ci_nonblocking_allowlist.py
  /opt/miniconda3/envs/dataselector/bin/ruff check .
  /opt/miniconda3/envs/dataselector/bin/black --check .
  /opt/miniconda3/envs/dataselector/bin/isort --check-only .
}

cmd_full() {
  [[ -n "${WAVE}" ]] || die "--wave is required for full"
  cmd_preflight
  cmd_collect_stability
  cmd_verify_safety
  cmd_gates
}

main() {
  parse_args "$@"

  case "${COMMAND}" in
    preflight)
      cmd_preflight
      ;;
    collect-stability)
      cmd_collect_stability
      ;;
    verify-safety)
      cmd_verify_safety
      ;;
    gates)
      cmd_gates
      ;;
    full)
      cmd_full
      ;;
    *)
      die "Unknown command: ${COMMAND}"
      ;;
  esac

  log "Done (${MODE})"
}

main "$@"
