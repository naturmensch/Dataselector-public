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


def cache_dir_for_hash(cache_root: str | Path, meta_hash: str) -> Path:
    return Path(cache_root) / str(meta_hash)


def features_path_for_hash(cache_root: str | Path, meta_hash: str) -> Path:
    return cache_dir_for_hash(cache_root, meta_hash) / "features.npy"


def meta_path_for_hash(cache_root: str | Path, meta_hash: str) -> Path:
    return cache_dir_for_hash(cache_root, meta_hash) / "meta.json"


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_features_with_meta(
    cache_root: str | Path, feats: np.ndarray, meta_hash: str, meta_info: Dict[str, Any]
) -> None:
    """Write-once cache write.

    - Existing key + identical meta => no-op (reuse immutable object).
    - Existing key + different meta => hard error.
    - Partial object presence => hard error.
    """
    cache_dir = cache_dir_for_hash(cache_root, meta_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = features_path_for_hash(cache_root, meta_hash)
    meta_target = meta_path_for_hash(cache_root, meta_hash)
    lock_path = cache_dir / ".lock"

    pid = os.getpid()
    ts = int(time.time())
    canonical_meta = _canonical_json(meta_info)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        features_exists = target.exists()
        meta_exists = meta_target.exists()
        if features_exists or meta_exists:
            if not (features_exists and meta_exists):
                raise RuntimeError(
                    f"Incomplete cache object for key {meta_hash}: expected both features.npy and meta.json."
                )
            existing_meta = _load_json(meta_target)
            if _canonical_json(existing_meta) != canonical_meta:
                raise RuntimeError(
                    f"Immutable cache conflict for key {meta_hash}: existing meta differs from current provenance."
                )
            return

        tmp_features = cache_dir / f"features.tmp-{pid}-{ts}.npy"
        tmp_meta = cache_dir / f"meta.tmp-{pid}-{ts}.json"
        np.save(str(tmp_features), feats)
        tmp_meta.write_text(
            json.dumps(meta_info, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_features), str(target))
        os.replace(str(tmp_meta), str(meta_target))
    except FileExistsError:
        # Another process is writing this key. Wait briefly, then validate immutable object.
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if not lock_path.exists():
                break
            time.sleep(0.2)
        features_exists = target.exists()
        meta_exists = meta_target.exists()
        if not (features_exists and meta_exists):
            raise RuntimeError(
                f"Cache lock timeout for key {meta_hash}: writer did not produce a complete immutable object."
            )
        existing_meta = _load_json(meta_target)
        if _canonical_json(existing_meta) != canonical_meta:
            raise RuntimeError(
                f"Immutable cache conflict for key {meta_hash}: existing meta differs from current provenance."
            )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                # Best-effort cleanup: do not escalate errors from unlinking a
                # transient lockfile. Failing here should not invalidate the
                # correctness of the immutable cache object written above and
                # is therefore intentionally non-fatal.
                pass


def find_cache_by_hash(cache_root: str | Path, meta_hash: str) -> Optional[Path]:
    f = features_path_for_hash(cache_root, meta_hash)
    m = meta_path_for_hash(cache_root, meta_hash)
    if f.exists() and m.exists():
        return f
    return None


def load_features_by_hash(
    cache_root: str | Path, meta_hash: str
) -> Optional[np.ndarray]:
    path = find_cache_by_hash(cache_root, meta_hash)
    if path is None:
        return None
    return np.load(path)


def load_meta_by_hash(
    cache_root: str | Path, meta_hash: str
) -> Optional[Dict[str, Any]]:
    meta_path = meta_path_for_hash(cache_root, meta_hash)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_all_feature_caches(cache_root: str | Path) -> list[Path]:
    cache_root = Path(cache_root)
    return sorted(cache_root.glob("*/features.npy"))


def create_meta_info(
    csv_path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    feature_identity: Optional[Dict[str, Any]] = None,
    model_provenance: Optional[Dict[str, Any]] = None,
    config_sha256: Optional[str] = None,
    metadata_basis: Optional[Dict[str, Any]] = None,
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
    if metadata_basis is not None:
        payload["metadata_basis"] = metadata_basis
    return payload
