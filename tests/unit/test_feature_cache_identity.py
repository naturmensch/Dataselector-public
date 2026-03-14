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

    def _fake_extract(
        _meta, *, batch_size=16, config_path=None, resolved_feature_config=None
    ):
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
    assert len(list(out.glob("*/features.npy"))) >= 2


def test_feature_cache_identity_separates_filtered_metadata_basis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "new_all_tiles.csv"
    csv.write_text("id\n1\n2\n3\n", encoding="utf-8")

    policy_path = tmp_path / "tile_policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "rules:",
                "  - id: exclude_kdr_155b",
                "    action: exclude_from_candidate_pool",
                "    match:",
                "      shortName: [KDR_155b]",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_load_metadata(
        _csv,
        *,
        resolve_images=True,
        tile_exclusion_policy=None,
        apply_tile_exclusion=None,
    ):
        if apply_tile_exclusion:
            df = pd.DataFrame(
                {
                    "shortName": ["KDR_155", "KDR_200"],
                    "image_path": ["a.png", "b.png"],
                }
            )
            df.attrs["tile_exclusions_applied"] = True
            df.attrs["tile_exclusions_count"] = 1
            df.attrs["tile_excluded_shortnames"] = ["KDR_155b"]
            return df
        return pd.DataFrame(
            {
                "shortName": ["KDR_155", "KDR_155b", "KDR_200"],
                "image_path": ["a.png", "b.png", "c.png"],
            }
        )

    monkeypatch.setattr(io_mod, "load_metadata", _fake_load_metadata)
    calls = {"count": 0}

    def _fake_extract(
        _meta, *, batch_size=16, config_path=None, resolved_feature_config=None
    ):
        calls["count"] += 1
        return (
            np.full((len(_meta), 4), fill_value=float(len(_meta)), dtype=np.float32),
            {},
        )

    monkeypatch.setattr(io_mod, "_extract_features_with_provenance", _fake_extract)

    out = tmp_path / "outputs"
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        cache=True,
        tile_exclusion_policy=str(policy_path),
        apply_tile_exclusion=False,
    )
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        cache=True,
        tile_exclusion_policy=str(policy_path),
        apply_tile_exclusion=True,
    )
    io_mod.load_or_extract_features(
        out_dir=out,
        csv_meta=str(csv),
        cache_mode="read_write",
        cache=True,
        tile_exclusion_policy=str(policy_path),
        apply_tile_exclusion=True,
    )

    assert calls["count"] == 2
    assert len(list(out.glob("*/features.npy"))) >= 2
