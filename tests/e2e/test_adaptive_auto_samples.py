"""E2E smoke tests for adaptive-auto command."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.adaptive
def test_adaptive_auto_help(run_dataselector_cli):
    result = run_dataselector_cli(["adaptive-auto", "--help"], capture_output=True)
    assert result.returncode == 0
    text = result.stdout.decode().lower()
    assert "--csv" in text
    assert "--n-samples" in text
    assert "--sampler" in text


@pytest.mark.integration
@pytest.mark.adaptive
def test_adaptive_auto_smoke_with_explicit_n_samples(
    tmp_workspace: Path, run_dataselector_cli
):
    """CI-safe smoke path: explicit n-samples avoids autoscale dependency in this test."""
    output_dir = tmp_workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_workspace / "data" / "tiles.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "\n".join(
            [
                "ul_x,ul_y,lr_x,lr_y,year,tile_name,image_path",
                "500000,5900100,500080,5900020,1900,tile_a,/tmp/a.png",
                "500100,5900200,500180,5900120,1901,tile_b,/tmp/b.png",
                "500200,5900300,500280,5900220,1902,tile_c,/tmp/c.png",
            ]
        ),
        encoding="utf-8",
    )

    cmd = [
        "adaptive-auto",
        "--csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--n-samples",
        "8",
        "--sampler",
        "lhs",
        "--dry-run",
    ]
    result = run_dataselector_cli(
        cmd, cwd=str(tmp_workspace), capture_output=True, timeout=300
    )
    assert result.returncode == 0, result.stderr.decode()
