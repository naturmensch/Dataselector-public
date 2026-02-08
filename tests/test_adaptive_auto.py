from __future__ import annotations

from pathlib import Path

import dataselector.workflows.adaptive_auto as mod


def test_adaptive_auto_uses_explicit_n_samples_without_autoscale(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "tiles.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")

    calls = {"autoscale": 0, "adaptive": 0}

    def fake_autoscale_main(**kwargs):
        calls["autoscale"] += 1
        return 0

    def fake_adaptive_main(**kwargs):
        calls["adaptive"] += 1
        assert kwargs["n_samples"] == 12
        assert kwargs["sampler"] == "lhs"
        return 0

    monkeypatch.setattr(mod, "autoscale_main", fake_autoscale_main)
    monkeypatch.setattr(mod, "adaptive_pipeline_main", fake_adaptive_main)

    rc = mod.run_adaptive_auto(
        csv=str(csv_path),
        output_dir=str(tmp_path / "outputs"),
        n_samples=12,
        sampler="lhs",
        dry_run=False,
    )

    assert rc == 0
    assert calls["autoscale"] == 0
    assert calls["adaptive"] == 1


def test_adaptive_auto_handoff_from_autoscale(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "tiles.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")
    out_dir = tmp_path / "outputs"

    def fake_autoscale_main(**kwargs):
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["output_dir"]) / "autoscale_selected_n_samples.txt").write_text(
            "34", encoding="utf-8"
        )
        return 0

    captured = {}

    def fake_adaptive_main(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(mod, "autoscale_main", fake_autoscale_main)
    monkeypatch.setattr(mod, "adaptive_pipeline_main", fake_adaptive_main)

    rc = mod.run_adaptive_auto(
        csv=str(csv_path),
        output_dir=str(out_dir),
        n_samples=None,
        sampler="sobol",
        dry_run=False,
    )

    assert rc == 0
    assert captured["n_samples"] == 34
    assert captured["sampler"] == "sobol"


def test_adaptive_auto_errors_when_autoscale_does_not_emit_n_samples(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "tiles.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")

    monkeypatch.setattr(mod, "autoscale_main", lambda **_kwargs: 0)
    monkeypatch.setattr(mod, "adaptive_pipeline_main", lambda **_kwargs: 0)

    rc = mod.run_adaptive_auto(
        csv=str(csv_path),
        output_dir=str(tmp_path / "outputs"),
        n_samples=None,
    )
    assert rc == 1


def test_adaptive_auto_dry_run_skips_execution(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "tiles.csv"
    csv_path.write_text("ul_x,ul_y,lr_x,lr_y,year\n1,2,3,4,1900\n", encoding="utf-8")

    calls = {"autoscale": 0, "adaptive": 0}

    monkeypatch.setattr(
        mod, "autoscale_main", lambda **_kwargs: calls.__setitem__("autoscale", 1) or 0
    )
    monkeypatch.setattr(
        mod,
        "adaptive_pipeline_main",
        lambda **_kwargs: calls.__setitem__("adaptive", 1) or 0,
    )

    rc = mod.run_adaptive_auto(
        csv=str(csv_path),
        output_dir=str(tmp_path / "outputs"),
        n_samples=None,
        dry_run=True,
    )
    assert rc == 0
    assert calls["autoscale"] == 0
    assert calls["adaptive"] == 0
