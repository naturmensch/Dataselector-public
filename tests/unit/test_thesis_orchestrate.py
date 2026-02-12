"""Tests for the thesis-orchestrate trigger-all command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest



def _write_minimal_inputs(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "config" / "pipeline_config.yaml").write_text(
        "selection:\n  n_samples: 24\n",
        encoding="utf-8",
    )
    (root / "config" / "tile_exclusion_policy.yaml").write_text(
        "rules: []\n",
        encoding="utf-8",
    )
    (root / "config" / "spatial_split_policy.yaml").write_text(
        "split:\n"
        "  ratios:\n"
        "    train: 0.7\n"
        "    val: 0.15\n"
        "    test: 0.15\n"
        "leakage:\n"
        "  calibration:\n"
        "    min_pairs_per_bin: 1\n",
        encoding="utf-8",
    )
    (root / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )


def test_thesis_orchestrate_precompute_only_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    calls: list[dict] = []

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)

    def _fake_run_thesis_pipeline(**kwargs):
        calls.append(kwargs)
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    assert len(calls) == 1
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["orchestrator_mode"] == "precompute_only"


def test_thesis_orchestrate_preserves_pipeline_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        # Simulate metadata already written by thesis-pipeline.
        (out_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-02-11T23:00:00Z",
                    "command": ["python", "-m", "dataselector", "thesis-pipeline"],
                    "runtime_state": {"profile": "thesis_repro"},
                    "extra": {"pipeline_marker": "kept", "n_trials": 100},
                }
            ),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_preserve"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["pipeline_marker"] == "kept"
    assert meta["extra"]["pipeline_metadata_preserved"] is True
    assert (
        meta["extra"]["pipeline_metadata_snapshot"]["command"]
        == ["python", "-m", "dataselector", "thesis-pipeline"]
    )


def test_thesis_orchestrate_reconciles_runtime_state_conservatively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)
    monkeypatch.setattr(
        mod,
        "activate_repro_mode",
        lambda **_: {
            "profile": "thesis_repro",
            "seed": 42,
            "repro_degraded": False,
            "parallelism_degraded": False,
            "repro_warnings": [],
            "thread_env": {"OMP_NUM_THREADS": "1"},
        },
    )

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        # Simulate pipeline runtime snapshot that reports stricter degradation.
        (out_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-02-11T23:00:00Z",
                    "command": ["python", "-m", "dataselector", "thesis-pipeline"],
                    "runtime_state": {
                        "profile": "thesis_repro",
                        "seed": 42,
                        "repro_degraded": True,
                        "parallelism_degraded": True,
                        "repro_warnings": [
                            "set_num_interop_threads_failed:cannot set number of interop threads after parallel work has started",
                            "set_num_interop_threads_failed:cannot set number of interop threads after parallel work has started",
                        ],
                        "thread_env": {"OMP_NUM_THREADS": "1"},
                        "torch": {"available": True, "num_threads": 1},
                    },
                    "extra": {"pipeline_marker": "kept", "n_trials": 100},
                }
            ),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_reconcile"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["runtime_state"]["repro_degraded"] is True
    assert meta["runtime_state"]["parallelism_degraded"] is True
    assert meta["runtime_state"]["repro_warnings"] == [
        "set_num_interop_threads_failed:cannot set number of interop threads after parallel work has started"
    ]
    assert meta["extra"]["runtime_state_reconciled"] is True
    assert meta["extra"]["runtime_state_sources"] == [
        "orchestrator",
        "pipeline_snapshot",
    ]
    # Snapshot remains raw for traceability.
    assert (
        meta["extra"]["pipeline_metadata_snapshot"]["runtime_state"]["repro_degraded"]
        is True
    )


def test_thesis_orchestrate_validation_errors_fail_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: ["hash mismatch"])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_error"
    with pytest.raises(RuntimeError, match="Scientific contract validation failed"):
        mod.run_thesis_orchestrate(
            config="config/pipeline_config.yaml",
            output_dir=str(out_dir),
            precompute_only=True,
            run_after_precompute=False,
            force=False,
            build_splits="false",
        )


def test_thesis_orchestrate_force_requires_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="force-override-reason"):
        mod.run_thesis_orchestrate(
            config="config/pipeline_config.yaml",
            output_dir=str(tmp_path / "outputs" / "runs" / "orch_force"),
            precompute_only=True,
            run_after_precompute=False,
            build_splits="false",
            force=True,
            force_override_reason=None,
        )


def test_thesis_orchestrate_passes_config_and_cache_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    observed: dict[str, dict] = {}
    monkeypatch.setattr(mod, "_require_torch", lambda: None)

    def _fake_autoscale(**kwargs):
        observed["autoscale"] = kwargs
        return 0

    def _fake_run_thesis_pipeline(**kwargs):
        observed.setdefault("pipeline_calls", {})
        observed["pipeline_calls"] = kwargs
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", _fake_autoscale)
    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(tmp_path / "outputs" / "runs" / "orch_cfg"),
        precompute_only=True,
        run_after_precompute=False,
        cache_mode="write_only",
        build_splits="false",
    )

    assert rc == 0
    assert observed["autoscale"]["config_path"] == "config/pipeline_config.yaml"
    assert observed["autoscale"]["cache_mode"] == "write_only"
    assert str(observed["pipeline_calls"]["config_path"]).endswith(
        "config/pipeline_config.yaml"
    )
    assert observed["pipeline_calls"]["cache_mode"] == "write_only"


def test_thesis_orchestrate_fails_for_non_empty_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "outputs" / "runs" / "existing_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "already_here.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        mod.run_thesis_orchestrate(
            config="config/pipeline_config.yaml",
            output_dir=str(out_dir),
            precompute_only=True,
            run_after_precompute=False,
            build_splits="false",
        )


def test_thesis_orchestrate_writes_artifact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config_20260212T000000Z.yaml").write_text(
            "parameters: {}\n",
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_manifest"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    manifest_path = out_dir / "manifest" / "artifact_hashes.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["files"]["run_metadata.json"]["exists"] is True
    assert payload["files"]["run_metadata.json"]["sha256"]
    assert payload["files"]["final_config_20260212T000000Z.yaml"]["exists"] is True
