from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataselector.data import io as io_mod


def _prepare_canonical_csv(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "new_all_tiles.csv"
    csv.write_text("id\n1\n2\n", encoding="utf-8")
    return csv


def test_cache_mode_read_only_requires_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    csv = _prepare_canonical_csv(tmp_path)

    monkeypatch.setattr(
        io_mod,
        "load_metadata",
        lambda _csv: pd.DataFrame({"image_path": ["a.png", "b.png"]}),
    )

    with pytest.raises(FileNotFoundError, match="read_only mode"):
        io_mod.load_or_extract_features(
            out_dir=tmp_path / "outputs",
            csv_meta=str(csv),
            cache_mode="read_only",
            cache=True,
        )


def test_cache_mode_off_does_not_write_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    csv = _prepare_canonical_csv(tmp_path)
    monkeypatch.setattr(
        io_mod,
        "load_metadata",
        lambda _csv: pd.DataFrame({"image_path": ["a.png", "b.png"]}),
    )
    monkeypatch.setattr(
        io_mod,
        "_extract_features_with_provenance",
        lambda _meta, **_: (np.zeros((2, 4), dtype=np.float32), {}),
    )

    feats = io_mod.load_or_extract_features(
        out_dir=tmp_path / "outputs",
        csv_meta=str(csv),
        cache_mode="off",
        cache=True,
    )
    assert feats.shape == (2, 4)
    assert list((tmp_path / "outputs").glob("features-*.npy")) == []


def test_cache_mode_write_only_forces_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    csv = _prepare_canonical_csv(tmp_path)
    monkeypatch.setattr(
        io_mod,
        "load_metadata",
        lambda _csv: pd.DataFrame({"image_path": ["a.png", "b.png"]}),
    )
    calls = {"count": 0}

    def _fake_extract(_meta, **_):
        calls["count"] += 1
        return np.full((2, 4), fill_value=float(calls["count"]), dtype=np.float32), {}

    monkeypatch.setattr(io_mod, "_extract_features_with_provenance", _fake_extract)
    out_dir = tmp_path / "outputs"

    io_mod.load_or_extract_features(
        out_dir=out_dir,
        csv_meta=str(csv),
        cache_mode="read_write",
        cache=True,
    )
    io_mod.load_or_extract_features(
        out_dir=out_dir,
        csv_meta=str(csv),
        cache_mode="write_only",
        cache=True,
    )

    assert calls["count"] == 2
