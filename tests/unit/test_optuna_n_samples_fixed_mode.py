from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataselector.workflows import optuna_autoscale as mod


def test_workflow_uses_fixed_mode_from_config(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(
        "selection:\n"
        "  autoscale_n_samples_mode: fixed\n"
        "  autoscale_n_samples_fixed: 34\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    captured: dict[str, object] = {}

    def fake_load_or_create_data(**kwargs):
        features = np.zeros((676, 4), dtype=np.float32)
        metadata = pd.DataFrame(
            {
                "ul_x": [0.0] * 676,
                "ul_y": [1.0] * 676,
                "lr_x": [2.0] * 676,
                "lr_y": [0.0] * 676,
                "year": [1900] * 676,
            }
        )
        return features, metadata

    def fake_run_autoscale(
        n_trials_per_stage,
        stages_samples,
        features,
        metadata,
        **kwargs,
    ):
        captured["trials"] = list(n_trials_per_stage)
        captured["stages"] = list(stages_samples)
        captured["policy"] = dict(kwargs.get("n_samples_policy") or {})
        (Path(kwargs["out_dir"]) / "optuna_autoscale_best_latest.json").write_text(
            json.dumps({"value": 1.0, "params": {}, "user_attrs": {"n_samples": 34}}),
            encoding="utf-8",
        )
        (Path(kwargs["out_dir"]) / "optuna_autoscale_selected_n_samples.txt").write_text(
            "34",
            encoding="utf-8",
        )
        return Path(kwargs["out_dir"]) / "dummy.csv", Path(kwargs["out_dir"]) / "dummy.json"

    monkeypatch.setattr(mod, "load_or_create_data", fake_load_or_create_data)
    monkeypatch.setattr(mod, "run_autoscale", fake_run_autoscale)
    monkeypatch.setattr(mod, "_load_min_distance_policy", lambda _cfg: (1, 60, True))

    rc = mod.run_optuna_autoscale_workflow(
        output_dir=str(out_dir),
        config_path=str(cfg),
    )
    assert rc == 0
    assert captured["stages"] == [34]
    assert captured["trials"] == [30]
    assert captured["policy"]["mode"] == "fixed"
    assert captured["policy"]["stages_resolved"] == [34]
