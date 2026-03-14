from __future__ import annotations

from pathlib import Path

from dataselector.workflows.optuna_autoscale import _load_min_distance_policy


def test_load_min_distance_policy_uses_selection_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(
        "selection:\n" "  min_distance_km: 28.5\n",
        encoding="utf-8",
    )
    floor, ceiling, global_search = _load_min_distance_policy(str(cfg))
    assert floor == 28
    assert ceiling == 60
    assert global_search is True


def test_load_min_distance_policy_respects_explicit_autoscale_fields(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "pipeline_config.yaml"
    cfg.write_text(
        "selection:\n"
        "  min_distance_km: 28.5\n"
        "  autoscale_min_distance_floor_km: 12\n"
        "  autoscale_min_distance_ceiling_km: 40\n"
        "  autoscale_min_distance_global_search: false\n",
        encoding="utf-8",
    )
    floor, ceiling, global_search = _load_min_distance_policy(str(cfg))
    assert floor == 12
    assert ceiling == 40
    assert global_search is False
