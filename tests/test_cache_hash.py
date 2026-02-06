import json
import os
import tempfile
from pathlib import Path

import numpy as np

from dataselector.pipeline.cache import (
    atomic_write_features_with_meta,
    compute_meta_hash,
    create_meta_info,
    features_path_for_hash,
    load_features_by_hash,
    meta_path_for_hash,
)

# Load src/cache.py directly to avoid top-level imports in package __init__ during tests




def test_compute_meta_hash_deterministic(tmp_path: Path):
    csv = tmp_path / "meta.csv"
    csv.write_text("id,x\n1,10\n2,20\n")
    h1 = compute_meta_hash(str(csv), params={"batch_size": 16})
    h2 = compute_meta_hash(str(csv), params={"batch_size": 16})
    assert isinstance(h1, str) and len(h1) == 64
    assert h1 == h2


def test_atomic_write_and_load(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    feats = np.random.RandomState(0).randn(5, 16)

    csv = tmp_path / "meta.csv"
    csv.write_text("id\n1\n2\n3\n4\n5\n")
    meta_hash = compute_meta_hash(str(csv), params={"batch_size": 4})
    meta_info = create_meta_info(str(csv), params={"batch_size": 4})

    atomic_write_features_with_meta(out, feats, meta_hash, meta_info)

    fpath = features_path_for_hash(out, meta_hash)
    mpath = meta_path_for_hash(out, meta_hash)
    assert fpath.exists()
    assert mpath.exists()

    loaded = load_features_by_hash(out, meta_hash)
    assert np.array_equal(loaded, feats)

    # Check meta file contents
    data = json.loads(mpath.read_text())
    assert data.get("metadata_csv")


def test_migration_script(tmp_path: Path, monkeypatch):
    # Create legacy features.npy and small CSV
    out = tmp_path / "outputs"
    out.mkdir()
    feats = np.arange(12).reshape(3, 4)
    np.save(out / "features.npy", feats)

    csv = tmp_path / "data.csv"
    csv.write_text("id\n1\n2\n3\n")

    # Run migration function directly
    from scripts.migrate_feature_cache_to_hash import migrate
    code = migrate(out, csv, dry_run=False)
    assert code == 0

    # There should be no legacy features.npy
    assert not (out / "features.npy").exists()
    # There should be exactly one features-*.npy
    matches = list(out.glob("features-*.npy"))
    assert len(matches) == 1

    # And backup exists
    backups = list((out / "backups").glob("features_legacy_backup_*.npy"))
    assert backups
