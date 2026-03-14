from __future__ import annotations

from pathlib import Path

from dataselector.workflows import thesis_orchestrate as mod


def _write_minimal_inputs(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "config" / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: 24\n",
        encoding="utf-8",
    )
    (root / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )


def test_orchestrator_passes_no_stage_trial_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "_require_torch", lambda: None)

    def _fake_autoscale(**kwargs):
        captured.update(kwargs)
        return 0

    def _fake_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", _fake_autoscale)
    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(tmp_path / "outputs" / "runs" / "orch"),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )
    assert rc == 0
    assert captured["n_trials"] is None
    assert captured["stages"] is None
