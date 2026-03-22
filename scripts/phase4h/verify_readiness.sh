#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

WAVE="wave8_readiness_verify"
LOG_FILE="$(log_file_for "${WAVE}")"
SNAPSHOT_PATH="${PHASE4H_OUT}/readiness_snapshot_$(date -u +%Y%m%dT%H%M%SZ).txt"

append_status "${WAVE}" "running" "Running final readiness verification gates"

{
  echo "[${WAVE}] started $(ts_utc)"
  echo "[${WAVE}] log=${LOG_FILE}"
  cd "${REPO_ROOT}"
  print_phase4h_context "${WAVE}"

  "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

import pandas as pd
import yaml

from dataselector.pipeline.pipeline_utils import compute_min_distance_km

root = Path.cwd()
csv_path = root / "data/new_all_tiles.csv"
cfg_path = root / "config/pipeline_config.yaml"

cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
sel = cfg.get("selection", {})

df = pd.read_csv(csv_path)
summary = {
    "rows": int(len(df)),
    "city_non_empty": int(df["city"].fillna("").astype(str).str.strip().ne("").sum()),
    "city_source_non_empty": int(df["city_source"].fillna("").astype(str).str.strip().ne("").sum()),
    "hamburg_rows": int((df["city"].astype(str).str.lower() == "hamburg").sum()),
    "kiel_rows": int((df["city"].astype(str).str.lower() == "kiel").sum()),
    "selection_n_samples": sel.get("n_samples"),
    "selection_min_distance_km": sel.get("min_distance_km"),
    "selection_validation_seeds": sel.get("validation_seeds"),
    "min_distance_geom_ref": compute_min_distance_km(str(csv_path)),
    "artifacts": {
        "min_distance_pre_registration": (root / "docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_PRE_REGISTRATION_2026-02-09.md").exists(),
        "min_distance_decision": (root / "docs/06_REFERENCE/thesis_decision_evidence/MIN_DISTANCE_DECISION_2026-02-09.md").exists(),
        "n_samples_pre_registration": (root / "docs/06_REFERENCE/thesis_decision_evidence/N_SAMPLES_PRE_REGISTRATION_2026-02-09.md").exists(),
        "n_samples_decision": (root / "docs/06_REFERENCE/thesis_decision_evidence/N_SAMPLES_DECISION_2026-02-09.md").exists(),
        "policy24_golive": (root / "docs/06_REFERENCE/thesis_decision_evidence/GO_LIVE_EVIDENCE_POLICY24_20260210T000328Z.md").exists(),
    },
}
print(json.dumps(summary, indent=2, ensure_ascii=False))

assert summary["rows"] == 676, summary
assert summary["city_non_empty"] == 676, summary
assert summary["city_source_non_empty"] == 676, summary
assert summary["hamburg_rows"] >= 1, summary
assert summary["kiel_rows"] >= 1, summary
assert summary["selection_n_samples"] == 24, summary
assert float(summary["selection_min_distance_km"]) == 28.5, summary
assert summary["min_distance_geom_ref"] == 45.0, summary
assert all(summary["artifacts"].values()), summary
PY

  "${PYTHON_BIN}" -m pytest -q \
    tests/unit/test_city_contract.py \
    tests/test_build_new_all_tiles.py \
    tests/unit/test_metadata_source_policy.py \
    tests/unit/test_canonical_source_contract.py \
    tests/test_preselection.py \
    tests/test_spatial_constraint.py \
    tests/test_bootstrap_module.py \
    tests/test_generate_reports_diagnostics.py \
    tests/unit/test_runtime_pass_allowlist.py \
    tests/test_adaptive_pipeline.py \
    tests/test_run_adaptive_pipeline_optuna.py \
    tests/test_metadata_processor.py -rs

  RUN_FULL_INTEGRATION=1 "${PYTHON_BIN}" -m pytest -q \
    tests/e2e/test_thesis_complete_e2e.py::test_thesis_pipeline_smoke -rs

  "${PYTHON_BIN}" -m ruff check .
  "${PYTHON_BIN}" -m isort --check-only .

  {
    echo "ready_verified_at=$(ts_utc)"
    echo "commit=$(git rev-parse HEAD)"
    echo "base_head=$(git rev-parse --short HEAD)"
    echo "python=${PYTHON_BIN}"
    echo "black_manual_command=${PYTHON_BIN} -m black --check --fast --no-cache dataselector tests scripts docs"
  } > "${SNAPSHOT_PATH}"

  echo "[${WAVE}] snapshot=${SNAPSHOT_PATH}"
  echo "[${WAVE}] pre-black verification finished $(ts_utc)"
} | tee "${LOG_FILE}"

if [[ "${PHASE4H_BLACK_CONFIRMED:-0}" != "1" ]]; then
  append_status "${WAVE}" "paused" "Readiness checks passed; waiting for manual black confirmation"
  cat <<MSG
[phase4h] ${WAVE} paused at manual black step.
Run this command manually in your terminal:
  ${PYTHON_BIN} -m black --check --fast --no-cache dataselector tests scripts docs

Then re-run with:
  PHASE4H_BLACK_CONFIRMED=1 scripts/phase4h/verify_readiness.sh
MSG
  exit 3
fi

append_status "${WAVE}" "completed" "Readiness checks complete (including manual black confirmation)"
echo "[phase4h] ${WAVE} done"
