from __future__ import annotations

from pathlib import Path

from dataselector.workflows.optuna_autoscale import (
    _derive_corridor_stages,
    _load_n_samples_policy,
)


def test_load_n_samples_policy_defaults_to_corridor(tmp_path: Path) -> None:
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text("selection:\n  n_samples: 24\n", encoding="utf-8")

    policy = _load_n_samples_policy(str(cfg))
    assert policy["mode"] == "corridor"
    assert policy["corridor_min_pct"] == 0.04
    assert policy["corridor_target_pct"] == 0.05
    assert policy["corridor_max_pct"] == 0.08
    assert policy["corridor_min_abs"] == 24
    assert policy["corridor_max_abs"] == 96
    assert policy["plateau_delta"] == 0.02


def test_derive_corridor_stages_is_deterministic_and_clamped() -> None:
    policy = {
        "corridor_min_pct": 0.04,
        "corridor_target_pct": 0.05,
        "corridor_max_pct": 0.08,
        "corridor_step": 1,
        "corridor_min_abs": 24,
        "corridor_max_abs": 96,
    }
    stages = _derive_corridor_stages(676, policy)
    assert stages == list(range(27, 55))
