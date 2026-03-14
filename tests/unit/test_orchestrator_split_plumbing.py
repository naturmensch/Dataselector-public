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
        "split:\n  ratios:\n    train: 0.7\n    val: 0.15\n    test: 0.15\n",
        encoding="utf-8",
    )
    (root / "data" / "new_all_tiles.csv").write_text(
        "ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n",
        encoding="utf-8",
    )


def test_orchestrator_writes_split_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])
    monkeypatch.setattr(mod, "_write_year_scope_audit", lambda **_: {})
    monkeypatch.setattr(
        mod,
        "_write_crs_provenance_audit",
        lambda **_: {
            "crs_provenance_audit_path": "outputs/runs/x/data_quality/crs_provenance_audit.csv",
            "crs_provenance_status": "explicit_uniform",
            "crs_strict_ready": True,
            "crs_explicit_tile_count": 1,
            "crs_missing_explicit_count": 0,
            "crs_heuristic_fallback_count": 0,
            "crs_consistency_issue_count": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_leakage_safe_splits",
        lambda **_: {
            "d_leak_km": 12.0,
            "split_manifest_path": "outputs/runs/x/splits/split_manifest.json",
            "leakage_audit_path": "outputs/runs/x/splits/leakage_audit.csv",
            "leakage_violations_count": 0,
        },
    )

    out_dir = tmp_path / "outputs" / "runs" / "orch_split"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
        build_splits="true",
    )
    assert rc == 0
    payload = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    extra = payload["extra"]
    assert extra["d_leak_km"] == 12.0
    assert extra["leakage_violations_count"] == 0


def test_orchestrator_default_does_not_build_splits(
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
    monkeypatch.setattr(mod, "validate_snapshot_file", lambda _p: [])
    monkeypatch.setattr(mod, "load_snapshot", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "load_parameter_contract", lambda _p: {"parameters": {}})
    monkeypatch.setattr(mod, "validate_snapshot_against_contract", lambda **_: [])
    monkeypatch.setattr(mod, "_write_year_scope_audit", lambda **_: {})
    monkeypatch.setattr(
        mod,
        "_write_crs_provenance_audit",
        lambda **_: {
            "crs_provenance_audit_path": "outputs/runs/x/data_quality/crs_provenance_audit.csv",
            "crs_provenance_status": "explicit_uniform",
            "crs_strict_ready": True,
            "crs_explicit_tile_count": 1,
            "crs_missing_explicit_count": 0,
            "crs_heuristic_fallback_count": 0,
            "crs_consistency_issue_count": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_leakage_safe_splits",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("split build should stay off by default")
        ),
    )

    out_dir = tmp_path / "outputs" / "runs" / "orch_no_split_default"
    rc = mod.run_thesis_orchestrate(
        config="config/pipeline_config.yaml",
        output_dir=str(out_dir),
        precompute_only=True,
        run_after_precompute=False,
    )
    assert rc == 0
    payload = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "split_manifest_path" not in payload["extra"]
