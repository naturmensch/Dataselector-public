"""Tests for scientific parameter resolution contract validation."""

from __future__ import annotations

from pathlib import Path

from dataselector.runtime.parameter_contract import (
    load_parameter_contract,
    validate_snapshot_against_contract,
)


def test_parameter_resolution_contract_file_loads() -> None:
    contract_path = Path("config/parameter_resolution_contract.yaml")
    contract = load_parameter_contract(contract_path)
    params = contract.get("parameters", {})
    assert "selection.alpha_visual" in params
    assert "selection.optuna_sampler" in params
    assert "clustering.umap_min_dist" in params
    assert "feature_extraction.pooling" in params
    assert "feature_extraction.model_variant" in params


def test_validate_snapshot_against_contract_success(tmp_path: Path) -> None:
    # Keep contract minimal and self-contained for deterministic test behavior.
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "parameter_resolution/optuna_autoscale_best_latest.json",
            },
            "selection.exploration_sampler": {
                "allowed_methods": ["mapped_from_optuna_sampler"],
                "required_evidence": "resolved_optuna_sampler",
            },
        }
    }

    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.5,
                "exploration_sampler": "lhs",
                "_provenance": {
                    "alpha_visual": {
                        "method": "computed_autoscale_artifact",
                        "source_file": str(
                            tmp_path
                            / "parameter_resolution"
                            / "optuna_autoscale_best_latest.json"
                        ),
                    },
                    "exploration_sampler": {
                        "method": "mapped_from_optuna_sampler",
                        "compute_args": {"resolved_optuna_sampler": "tpe"},
                    },
                },
            }
        }
    }
    evidence_path = (
        tmp_path / "parameter_resolution" / "optuna_autoscale_best_latest.json"
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("{}", encoding="utf-8")

    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=tmp_path,
        repo_root=tmp_path,
    )
    assert errors == []


def test_validate_snapshot_against_contract_missing_provenance_fails(
    tmp_path: Path,
) -> None:
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "parameter_resolution/optuna_autoscale_best_latest.json",
            }
        }
    }

    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.5,
                "_provenance": {},
            }
        }
    }

    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=tmp_path,
        repo_root=tmp_path,
    )
    assert any("missing provenance entry" in err.lower() for err in errors)


def test_validate_snapshot_against_contract_missing_evidence_fails(
    tmp_path: Path,
) -> None:
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "parameter_resolution/optuna_autoscale_best_latest.json",
            }
        }
    }
    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.4,
                "_provenance": {
                    "alpha_visual": {
                        "method": "computed_autoscale_artifact",
                        "source_file": str(tmp_path / "different.json"),
                    }
                },
            }
        }
    }

    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=tmp_path,
        repo_root=tmp_path,
    )
    assert any("requires evidence" in err.lower() for err in errors)


def test_validate_snapshot_against_contract_missing_source_file_fails(
    tmp_path: Path,
) -> None:
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "parameter_resolution/optuna_autoscale_best_latest.json",
            }
        }
    }
    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.4,
                "_provenance": {
                    "alpha_visual": {
                        "method": "computed_autoscale_artifact",
                        "source_file": str(
                            tmp_path
                            / "parameter_resolution"
                            / "optuna_autoscale_best_latest.json"
                        ),
                    }
                },
            }
        }
    }

    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=tmp_path,
        repo_root=tmp_path,
    )
    assert any("does not exist" in err.lower() for err in errors)


def test_validate_snapshot_against_contract_run_relative_scope(tmp_path: Path) -> None:
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "parameter_resolution/optuna_autoscale_best_latest.json",
            }
        }
    }
    run_dir = tmp_path / "run"
    repo_dir = tmp_path / "repo"
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "parameter_resolution").mkdir(parents=True, exist_ok=True)
    (run_dir / "parameter_resolution" / "optuna_autoscale_best_latest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.4,
                "_provenance": {
                    "alpha_visual": {
                        "method": "computed_autoscale_artifact",
                        "source_file": str(
                            run_dir
                            / "parameter_resolution"
                            / "optuna_autoscale_best_latest.json"
                        ),
                    }
                },
            }
        }
    }
    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=run_dir,
        repo_root=repo_dir,
        evidence_scope="run_dir",
    )
    assert errors == []


def test_validate_snapshot_against_contract_repo_prefix(tmp_path: Path) -> None:
    contract = {
        "parameters": {
            "selection.alpha_visual": {
                "allowed_methods": ["computed_autoscale_artifact"],
                "required_evidence": "repo:docs/PARAMETER_POLICY_LEDGER.md",
            }
        }
    }
    run_dir = tmp_path / "run"
    repo_dir = tmp_path / "repo"
    run_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs" / "PARAMETER_POLICY_LEDGER.md").write_text(
        "ok",
        encoding="utf-8",
    )
    snapshot = {
        "parameters": {
            "selection": {
                "alpha_visual": 0.4,
                "_provenance": {
                    "alpha_visual": {
                        "method": "computed_autoscale_artifact",
                        "source_file": str(
                            repo_dir / "docs" / "PARAMETER_POLICY_LEDGER.md"
                        ),
                    }
                },
            }
        }
    }
    errors = validate_snapshot_against_contract(
        snapshot=snapshot,
        contract=contract,
        run_root=run_dir,
        repo_root=repo_dir,
        evidence_scope="run_dir",
    )
    assert errors == []
