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
        (
            "ul_x,ul_y,lr_x,lr_y,year,source_crs,crs_source,"
            "crs_provenance,crs_explicit\n"
            "1,2,3,4,1900,EPSG:4326,sidecar_xml,explicit_sidecar_xml,true\n"
        ),
        encoding="utf-8",
    )


def test_thesis_orchestrate_precompute_only_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert calls[0]["tile_exclusion_policy"] == Path(
        "config/tile_exclusion_policy.yaml"
    )
    assert calls[0]["apply_tile_exclusion"] is True
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["extra"]["orchestrator_mode"] == "precompute_only"


def test_thesis_orchestrate_passes_phase5_flags_only_to_final_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    calls: list[dict[str, object]] = []

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

    out_dir = tmp_path / "outputs" / "runs" / "orch_phase5"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=False,
        run_after_precompute=True,
        build_splits="false",
        build_handoffs=True,
        patches_per_tile=3,
        patch_include_case="true",
        handoff_root="handoff_custom",
    )

    assert rc == 0
    assert len(calls) == 2
    assert calls[0]["no_auto_continue"] is True
    assert calls[0]["build_handoffs"] is False
    assert calls[1]["build_handoffs"] is True
    assert calls[1]["patches_per_tile"] == 3
    assert calls[1]["patch_include_case"] is True
    assert calls[1]["handoff_root"] == "handoff_custom"


def test_thesis_orchestrate_writes_tile_exclusion_metadata_without_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)
    monkeypatch.setattr(
        mod,
        "_write_year_scope_audit",
        lambda **_: {
            "year_scope_audit_path": "outputs/runs/ignored/data_quality/year_scope_audit.csv",
            "year_scope_before_n": 676,
            "year_scope_after_n": 675,
            "year_scope_before_max": 1985,
            "year_scope_after_max": 1985,
            "tile_exclusions_applied": True,
            "tile_exclusions_count": 1,
            "tile_excluded_shortnames": ["KDR_155b"],
            "tile_flagged_count": 2,
            "tile_flagged_shortnames": ["KDR_039", "KDR_521"],
            "tile_flagged_classes": ["temporal_scope_outlier"],
            "tile_flagged_caveats": [
                {"shortName": "KDR_039"},
                {"shortName": "KDR_521"},
            ],
            "tile_exclusion_policy_sha256": "abc123",
            "effective_tile_count": 675,
        },
    )

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_tile_meta"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    extra = meta["extra"]
    assert extra["tile_exclusions_applied"] is True
    assert extra["tile_exclusions_count"] == 1
    assert extra["tile_excluded_shortnames"] == ["KDR_155b"]
    assert extra["tile_flagged_count"] == 2
    assert extra["tile_flagged_shortnames"] == ["KDR_039", "KDR_521"]
    assert extra["tile_exclusion_policy_sha256"] == "abc123"
    assert extra["effective_tile_count"] == 675


def test_thesis_orchestrate_keeps_year_scope_and_tile_exclusion_metadata_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(mod, "_require_torch", lambda: None)
    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", lambda **_: 0)
    monkeypatch.setattr(
        mod,
        "_write_year_scope_audit",
        lambda **_: {
            "year_scope_audit_path": "outputs/runs/ignored/data_quality/year_scope_audit.csv",
            "year_scope_before_n": 676,
            "year_scope_after_n": 675,
            "year_scope_before_max": 1985,
            "year_scope_after_max": 1985,
            "tile_exclusions_applied": True,
            "tile_exclusions_count": 1,
            "tile_excluded_shortnames": ["KDR_155b"],
            "tile_flagged_count": 2,
            "tile_flagged_shortnames": ["KDR_039", "KDR_521"],
            "tile_flagged_classes": ["temporal_scope_outlier"],
            "tile_flagged_caveats": [
                {"shortName": "KDR_039"},
                {"shortName": "KDR_521"},
            ],
            "tile_exclusion_policy_sha256": "abc123",
            "effective_tile_count": 675,
        },
    )

    def _fake_run_thesis_pipeline(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    out_dir = tmp_path / "outputs" / "runs" / "orch_year_scope_tile_consistency"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="false",
    )

    assert rc == 0
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    extra = meta["extra"]
    assert extra["year_scope_before_n"] == 676
    assert extra["year_scope_after_n"] == 675
    assert extra["tile_exclusions_count"] == 1
    assert (
        extra["year_scope_before_n"] - extra["year_scope_after_n"]
        == extra["tile_exclusions_count"]
    )
    assert extra["year_scope_after_n"] == extra["effective_tile_count"]


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
    assert meta["extra"]["pipeline_metadata_snapshot"]["command"] == [
        "python",
        "-m",
        "dataselector",
        "thesis-pipeline",
    ]


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


def test_thesis_orchestrate_default_n_trials_is_370(tmp_path: Path) -> None:
    """Default `n_trials` for orchestrator should be 370 (policy change)."""
    import inspect

    from dataselector.workflows.thesis_orchestrate import (
        cli_thesis_orchestrate,
        run_thesis_orchestrate,
    )

    assert (
        inspect.signature(run_thesis_orchestrate).parameters["n_trials"].default == 370
    )
    assert (
        inspect.signature(cli_thesis_orchestrate).parameters["n_trials"].default == 370
    )


@pytest.mark.parametrize(
    "anchor_env, expected_pre_names",
    [
        (None, None),  # Case 1: Standard, no env var
        ("Hamburg", ["Hamburg"]),  # Case 2: With Anchor-Tile Env-Var
    ],
)
def test_thesis_orchestrate_passes_arguments_correctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor_env: str | None,
    expected_pre_names: list[str] | None,
) -> None:
    from dataselector.workflows import thesis_orchestrate as mod

    _write_minimal_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Optional: Set Environment Variable for this test run
    if anchor_env:
        monkeypatch.setenv("DATASELECTOR_ANCHOR_TILE", anchor_env)

    observed: dict[str, dict] = {}
    monkeypatch.setattr(mod, "_require_torch", lambda: None)

    # Mock sub-workflows to intercept arguments
    def _fake_autoscale(**kwargs):
        observed["autoscale"] = kwargs
        return 0

    def _fake_run_thesis_pipeline(**kwargs):
        # Collect all calls in a list
        calls = observed.setdefault("pipeline_calls", [])
        calls.append(kwargs)

        # Create dummy outputs so orchestrator continues
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_config.yaml").write_text("parameters: {}\n", encoding="utf-8")
        return True

    monkeypatch.setattr(mod, "run_optuna_autoscale_workflow", _fake_autoscale)
    monkeypatch.setattr(mod, "run_thesis_pipeline", _fake_run_thesis_pipeline)

    # Mocks for validation (Boilerplate)
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])

    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(tmp_path / "outputs" / "runs" / "orch_args"),
        precompute_only=True,
        run_after_precompute=False,
        cache_mode="write_only",
        build_splits="false",
    )

    assert rc == 0

    # 1. Existing Checks (Config & Cache Mode)
    assert observed["autoscale"]["config_path"] == "config/pipeline_config.yaml"
    assert observed["autoscale"]["cache_mode"] == "write_only"

    # 2. New Check: Was pre_names passed correctly?
    assert observed["autoscale"].get("pre_names") == expected_pre_names

    # Check pipeline calls as well
    assert len(observed["pipeline_calls"]) > 0
    for call_kwargs in observed["pipeline_calls"]:
        assert call_kwargs.get("pre_names") == expected_pre_names


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
