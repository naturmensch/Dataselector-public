from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataselector.data import io as io_mod


def _write_config(path: Path, *, pooling: str) -> None:
    path.write_text(
        "\n".join(
            [
                "feature_extraction:",
                "  model: dinov2",
                "  input_size: 392",
                "  crop_size: [2048, 2048]",
                f"  pooling: {pooling}",
                "  model_variant: dinov2_vits14",
                "  dinov2_repo: facebookresearch/dinov2",
                "  dinov2_ref: main",
            ]
        ),
        encoding="utf-8",
    )


def test_feature_cache_identity_prevents_reuse_on_pooling_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "new_all_tiles.csv"
    csv.write_text("id\n1\n2\n", encoding="utf-8")

    cfg1 = tmp_path / "config_a.yaml"
    cfg2 = tmp_path / "config_b.yaml"
    _write_config(cfg1, pooling="cls")
    _write_config(cfg2, pooling="mean")

    monkeypatch.setattr(
        io_mod,
        "load_metadata",
        lambda _csv: pd.DataFrame({"image_path": ["a.png", "b.png"]}),
    )
    calls = {"count": 0}

    def _fake_extract(_meta, *, batch_size=16, config_path=None, resolved_feature_config=None):
        calls["count"] += 1
        feats = np.full((2, 8), fill_value=float(calls["count"]), dtype=np.float32)
        return feats, {}

    monkeypatch.setattr(io_mod, "_extract_features_with_provenance", _fake_extract)

    out = tmp_path / "outputs"
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        config_path=str(cfg1),
        cache=True,
    )
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        config_path=str(cfg2),
        cache=True,
    )
    # Repeating config1 should hit cache and avoid a third extract.
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        config_path=str(cfg1),
        cache=True,
    )

    assert calls["count"] == 2
    assert len(list(out.glob("features-*.npy"))) >= 2
