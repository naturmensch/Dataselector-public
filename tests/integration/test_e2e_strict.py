import json
import os
import sys
from pathlib import Path

from scripts import xxl_KDR146_run_thesis_complete_modern as mod
from tests.helpers.create_minimal_autoscale import create_minimal_autoscale


def test_e2e_requires_autoscale_and_runs(tmp_path, monkeypatch):
    # Setup isolated project root
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    # Ensure pytest env var not set for the script (we want non-smoke behavior)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    # Write minimal autoscale artifacts and selected_sampler
    create_minimal_autoscale(tmp_path, n_samples=40)
    out = tmp_path / "outputs"
    (out / "selected_sampler.json").write_text(json.dumps({"best": "tpe"}))

    # Run orchestrator in dry-run so heavy steps are simulated
    monkeypatch.setattr(
        sys, "argv", ["xxl", "--best-sampler", "tpe", "--dry-run", "--skip-env-check"]
    )
    rc = mod.main()
    assert rc == 0
    assert (tmp_path / "outputs" / "thesis_finalization_summary.json").exists()
