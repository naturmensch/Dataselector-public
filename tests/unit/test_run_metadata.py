from __future__ import annotations

import json
from pathlib import Path

from dataselector.runtime.run_metadata import write_run_metadata


def test_write_run_metadata_contains_required_fields(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("alpha: 0.3\n", encoding="utf-8")

    out = write_run_metadata(
        output_dir=tmp_path,
        execution_profile="thesis_repro",
        seed=42,
        command=["python", "-m", "dataselector", "thesis-pipeline"],
        config_path=cfg,
        runtime_state={"thread_env": {"OMP_NUM_THREADS": "1"}},
        extra={"workflow": "thesis-pipeline"},
    )

    assert out == tmp_path / "run_metadata.json"
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["execution_profile"] == "thesis_repro"
    assert payload["seed"] == 42
    assert payload["command"][2] == "dataselector"
    assert payload["runtime_state"]["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert payload["config"]["exists"] is True
    assert payload["config"]["sha256"]
