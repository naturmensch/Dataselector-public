import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def compute_meta_hash(csv_path: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Compute a SHA256 hash over the binary contents of the CSV and optional params.

    The result is a hex string (64 chars)."""
    h = hashlib.sha256()
    # Hash CSV bytes (read in binary)
    with open(csv_path, "rb") as fh:
        # read in chunks to avoid huge memory usage
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)

    if params:
        params_bytes = json.dumps(params, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        h.update(params_bytes)

    return h.hexdigest()


def features_path_for_hash(out_dir: str | Path, meta_hash: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"features-{meta_hash}.npy"


def meta_path_for_hash(out_dir: str | Path, meta_hash: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"features-{meta_hash}.meta.json"


def atomic_write_features_with_meta(
    out_dir: str | Path, feats: np.ndarray, meta_hash: str, meta_info: Dict[str, Any]
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = features_path_for_hash(out_dir, meta_hash)
    meta_target = meta_path_for_hash(out_dir, meta_hash)

    pid = os.getpid()
    ts = int(time.time())
    tmp_features = out_dir / f"{target.name}.tmp-{pid}-{ts}.npy"
    tmp_meta = out_dir / f"{meta_target.name}.tmp-{pid}-{ts}.json"

    # Write tmp files (ensure explicit extensions so numpy doesn't append .npy again)
    np.save(str(tmp_features), feats)
    with open(tmp_meta, "w", encoding="utf-8") as fh:
        json.dump(meta_info, fh, sort_keys=True, indent=2)

    # Atomic replace
    os.replace(str(tmp_features), str(target))
    os.replace(str(tmp_meta), str(meta_target))


def find_cache_by_hash(out_dir: str | Path, meta_hash: str) -> Optional[Path]:
    f = features_path_for_hash(out_dir, meta_hash)
    m = meta_path_for_hash(out_dir, meta_hash)
    if f.exists() and m.exists():
        return f
    return None


def load_features_by_hash(out_dir: str | Path, meta_hash: str) -> Optional[np.ndarray]:
    path = find_cache_by_hash(out_dir, meta_hash)
    if path is None:
        return None
    return np.load(path)


def load_meta_by_hash(out_dir: str | Path, meta_hash: str) -> Optional[Dict[str, Any]]:
    meta_path = meta_path_for_hash(out_dir, meta_hash)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_all_feature_caches(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    return sorted(out_dir.glob("features-*.npy"))


def create_meta_info(
    csv_path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    feature_identity: Optional[Dict[str, Any]] = None,
    model_provenance: Optional[Dict[str, Any]] = None,
    config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "metadata_csv": str(Path(csv_path).resolve()),
        "params": params or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if feature_identity is not None:
        payload["feature_identity"] = feature_identity
    if model_provenance is not None:
        payload["model_provenance"] = model_provenance
    if config_sha256:
        payload["config_sha256"] = config_sha256
    return payload
