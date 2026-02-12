from __future__ import annotations

from pathlib import Path


def test_double_run_wrapper_stays_helper_only() -> None:
    script = Path("scripts/run_thesis_orchestrate_double.sh").read_text(encoding="utf-8")

    assert "compute_effective_tile_count" not in script
    assert "N_SAMPLES_MODE" not in script
    assert "python - <<'PY'" not in script
    assert 'BUILD_SPLITS="${BUILD_SPLITS:-false}"' in script
