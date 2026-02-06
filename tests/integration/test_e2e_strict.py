import json
import os
from pathlib import Path

import dataselector.workflows.xxl as mod
from tests.helpers.create_minimal_autoscale import create_minimal_autoscale


def test_e2e_requires_autoscale_and_runs(tmp_path, monkeypatch):
    # Write minimal autoscale artifacts and selected_sampler
    create_minimal_autoscale(tmp_path, n_samples=40)
    out = tmp_path / "outputs"
    (out / "selected_sampler.json").write_text(json.dumps({"best": "tpe"}))

    # Run orchestrator in smoke mode so heavy steps are simulated
    rc = mod.main(best_sampler="tpe", smoke=True, output_dir=str(out))
    assert rc == 0
    assert (out / "thesis_finalization_summary.json").exists()
