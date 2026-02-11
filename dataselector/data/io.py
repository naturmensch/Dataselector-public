import os
import time
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dataselector.data.metadata_processor import MetadataProcessor
from dataselector.data.metadata_source import (
    CANONICAL_METADATA_RELATIVE_PATH,
    assert_canonical_metadata,
    canonical_metadata_path,
)
from dataselector.runtime.parameter_snapshot import compute_file_sha256

PREPROCESS_PIPELINE_ID = "historical_grayscale_autocontrast_rgb_v1"
CACHE_MODES = {"off", "read_only", "write_only", "read_write"}


def load_metadata(
    csv_path: str | Path,
    image_dir: str | Path | None = None,
    resolve_images: bool = True,
    strict_image_resolution: bool = False,
    strict_metric_crs: bool | None = None,
    metric_epsg: int | None = None,
) -> pd.DataFrame:
    mp = MetadataProcessor(str(csv_path))
    df = mp.load_csv()
    df = mp.add_temporal_metadata()

    if resolve_images:
        resolved_image_dir = Path(
            image_dir
            if image_dir is not None
            else os.getenv("DATASELECTOR_IMAGE_DIR", "data/images")
        )
        has_image_path = "image_path" in df.columns
        has_any_path = (
            has_image_path
            and df["image_path"].notna().any()
            and (df["image_path"].astype(str).str.strip() != "").any()
        )

        # Resolve paths only when needed or when strict mode is explicitly requested.
        if strict_image_resolution or not has_any_path:
            resolve_sig = signature(mp.resolve_image_paths)
            kwargs = {}
            if "prefer_shortname" in resolve_sig.parameters:
                kwargs["prefer_shortname"] = True
            if "strict" in resolve_sig.parameters:
                kwargs["strict"] = strict_image_resolution
            df = mp.resolve_image_paths(resolved_image_dir, **kwargs)
        elif "image_filename" not in df.columns:
            df["image_filename"] = df["image_path"].apply(
                lambda p: Path(str(p)).name if pd.notna(p) and str(p).strip() else None
            )

    # Ensure metric CRS (UTM) is available for precise spatial calculations
    # Attach into DataFrame.attrs to avoid fragile attribute access and to persist through copies
    strict_crs_flag = (
        bool(strict_metric_crs)
        if strict_metric_crs is not None
        else os.getenv("DATASELECTOR_STRICT_CRS", "0") == "1"
    )
    target_epsg = (
        int(metric_epsg)
        if metric_epsg is not None
        else int(os.getenv("DATASELECTOR_METRIC_EPSG", "25832"))
    )
    gdf_metric = mp.ensure_metric_crs(target_epsg=target_epsg, strict=strict_crs_flag)
    attach_metric_gdf(df, gdf_metric)
    if hasattr(df, "attrs"):
        df.attrs["source_crs"] = mp.source_crs
        df.attrs["metric_crs"] = mp.metric_crs
        df.attrs["transform_applied"] = bool(mp.transform_applied)
    if strict_crs_flag and gdf_metric is None:
        raise RuntimeError(
            "Strict CRS mode requires metric reprojection, but no metric coordinates were produced."
        )

    # ensure placeholder for missing images
    if "image_path" not in df.columns:
        df["image_path"] = None
    df["image_path"] = df["image_path"].fillna("missing_placeholder.png")
    return df


# Helper utilities for robustly attaching/reading metric GeoDataFrame
def attach_metric_gdf(df, gdf_metric):
    """Attach the metric GeoDataFrame to pandas DataFrame metadata in a safe way.

    Stores the gdf in df.attrs['gdf_metric'] so it survives common pandas operations.
    """
    # Preferred storage path for pandas DataFrame.
    if hasattr(df, "attrs"):
        try:
            df.attrs["gdf_metric"] = gdf_metric
            return
        except Exception:
            # Fall through to attribute fallback.
            ...

    # Backward-compatible fallback for custom containers.
    try:
        setattr(df, "gdf_metric", gdf_metric)
    except Exception:
        return


def get_metric_gdf(df):
    """Retrieve attached metric GeoDataFrame if present (attrs or attribute)."""
    if hasattr(df, "attrs"):
        try:
            if "gdf_metric" in df.attrs:
                return df.attrs.get("gdf_metric")
        except Exception:
            return getattr(df, "gdf_metric", None)
    # Backward-compatible fallback
    return getattr(df, "gdf_metric", None)


def extract_features(metadata: pd.DataFrame, batch_size: int = 16) -> np.ndarray:
    # Backward-compatible wrapper: use default config discovery.
    features, _ = _extract_features_with_provenance(
        metadata,
        batch_size=batch_size,
        config_path=None,
        resolved_feature_config=None,
    )
    return features


def _normalize_cache_mode(cache_mode: str) -> str:
    normalized = str(cache_mode).strip().lower()
    if normalized not in CACHE_MODES:
        raise ValueError(
            f"Unknown cache_mode '{cache_mode}'. Expected one of: {sorted(CACHE_MODES)}"
        )
    return normalized


def _resolve_feature_config(
    *,
    config_path: str | Path | None,
    resolved_feature_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], Path | None]:
    defaults: dict[str, Any] = {
        "model": "dinov2",
        "input_size": 392,
        "crop_size": [2048, 2048],
        "pooling": "cls",
        "model_variant": "dinov2_vits14",
        "dinov2_repo": "facebookresearch/dinov2",
        "dinov2_ref": "main",
        "device": "auto",
        "preprocess_pipeline_id": PREPROCESS_PIPELINE_ID,
    }

    if isinstance(resolved_feature_config, dict):
        merged = dict(defaults)
        merged.update(resolved_feature_config)
        return merged, None

    # Config path precedence: explicit arg > env-var > canonical default.
    path: Path | None
    if config_path is not None:
        path = Path(config_path)
    else:
        env_cfg = os.getenv("DATASELECTOR_ACTIVE_CONFIG")
        path = Path(env_cfg) if env_cfg else Path("config/pipeline_config.yaml")

    if path is not None and path.exists():
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            feat_cfg = payload.get("feature_extraction", {})
            merged = dict(defaults)
            if isinstance(feat_cfg, dict):
                merged.update(feat_cfg)
            return merged, path
        except Exception as exc:
            print(
                f"Warnung: Konnte Feature-Config nicht lesen ({exc}), nutze Defaults"
            )
    return dict(defaults), path if path is not None and path.exists() else None


def build_feature_identity(
    *,
    feature_cfg: dict[str, Any],
    batch_size: int,
    config_sha256: str | None,
) -> dict[str, Any]:
    model = str(feature_cfg.get("model", "dinov2")).strip().lower()
    input_size = int(feature_cfg.get("input_size", 392 if model == "dinov2" else 224))
    # Keep identity aligned with FeatureExtractor behavior.
    if model == "dinov2" and input_size == 224:
        input_size = 392

    raw_crop = feature_cfg.get("crop_size", [2048, 2048])
    if isinstance(raw_crop, (list, tuple)) and len(raw_crop) == 2:
        crop_size = [int(raw_crop[0]), int(raw_crop[1])]
    else:
        crop_size = [int(input_size), int(input_size)]

    identity: dict[str, Any] = {
        "model_name": model,
        "model_variant": str(feature_cfg.get("model_variant", "dinov2_vits14")).strip(),
        "dinov2_repo": str(feature_cfg.get("dinov2_repo", "facebookresearch/dinov2")).strip(),
        "dinov2_ref": str(feature_cfg.get("dinov2_ref", "main")).strip(),
        "pooling": str(feature_cfg.get("pooling", "cls")).strip().lower(),
        "input_size": int(input_size),
        "crop_size": crop_size,
        "preprocess_pipeline_id": str(
            feature_cfg.get("preprocess_pipeline_id", PREPROCESS_PIPELINE_ID)
        ).strip(),
        "batch_size": int(batch_size),
    }
    if config_sha256:
        identity["config_sha256"] = config_sha256
    return identity


def _extract_features_with_provenance(
    metadata: pd.DataFrame,
    *,
    batch_size: int = 16,
    config_path: str | Path | None,
    resolved_feature_config: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Lazy import: keep basic I/O paths independent from torch.
    from dataselector.features.feature_extractor import FeatureExtractor

    missing_paths = []
    for raw_path in metadata["image_path"].tolist():
        if pd.isna(raw_path) or str(raw_path).strip() == "":
            missing_paths.append(str(raw_path))
            continue
        p = Path(str(raw_path))
        if not p.is_absolute():
            p = Path(".") / p
        if not p.exists():
            missing_paths.append(str(raw_path))

    if missing_paths:
        sample = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            "Feature extraction requires real images. "
            f"Missing image files (sample): {sample}"
        )

    feature_cfg, cfg_path = _resolve_feature_config(
        config_path=config_path,
        resolved_feature_config=resolved_feature_config,
    )
    model_name = str(feature_cfg.get("model", "dinov2")).strip().lower()
    input_size = int(
        feature_cfg.get("input_size", 392 if model_name == "dinov2" else 224)
    )
    raw_crop = feature_cfg.get("crop_size", [2048, 2048])
    if isinstance(raw_crop, (list, tuple)) and len(raw_crop) == 2:
        crop_size = (int(raw_crop[0]), int(raw_crop[1]))
    else:
        crop_size = (2048, 2048)
    pooling = str(feature_cfg.get("pooling", "cls")).strip().lower()
    model_variant = str(feature_cfg.get("model_variant", "dinov2_vits14")).strip()
    dinov2_repo = str(feature_cfg.get("dinov2_repo", "facebookresearch/dinov2")).strip()
    dinov2_ref = str(feature_cfg.get("dinov2_ref", "main")).strip()
    cfg_device = str(feature_cfg.get("device", "auto")).strip().lower()
    device = None if cfg_device == "auto" else cfg_device

    fe = FeatureExtractor(
        model_name=model_name,
        input_size=input_size,
        default_crop_size=crop_size,
        pooling=pooling,
        model_variant=model_variant,
        dinov2_repo=dinov2_repo,
        dinov2_ref=dinov2_ref,
        device=device,
    )
    image_paths = metadata["image_path"].tolist()
    features = fe.extract_features_batch(
        image_paths=image_paths,
        data_dir=Path("."),
        batch_size=batch_size,
        crop_size=crop_size,
    )
    model_provenance = fe.get_model_provenance()
    config_sha256 = compute_file_sha256(cfg_path) if cfg_path and cfg_path.exists() else None
    feature_identity = build_feature_identity(
        feature_cfg=feature_cfg,
        batch_size=batch_size,
        config_sha256=config_sha256,
    )
    # Align identity with effective model provenance values (e.g. input_size override).
    if "input_size" in model_provenance:
        feature_identity["input_size"] = int(model_provenance["input_size"])
    return features, {
        "feature_identity": feature_identity,
        "model_provenance": model_provenance,
        "config_sha256": config_sha256,
        "config_path": str(cfg_path) if cfg_path else None,
    }


def load_or_extract_features(
    out_dir: str | Path = "outputs",
    csv_meta: str | None = None,
    batch_size: int = 16,
    cache: bool = True,
    enforce_canonical: bool = False,
    cache_mode: str = "read_write",
    config_path: str | Path | None = None,
    resolved_feature_config: dict[str, Any] | None = None,
    strict_cache_identity: bool = False,
    force_extract: bool = False,
) -> np.ndarray:
    """Load features from a hash-identified cache or extract and create a new cache."""
    from dataselector.pipeline.cache import (
        atomic_write_features_with_meta,
        compute_meta_hash,
        create_meta_info,
        features_path_for_hash,
        load_features_by_hash,
        load_meta_by_hash,
    )

    out_dir = Path(out_dir)

    # Environment fallback for orchestration-driven runs.
    if cache_mode == "read_write":
        cache_mode = os.getenv("DATASELECTOR_FEATURE_CACHE_MODE", cache_mode)
    cache_mode = _normalize_cache_mode(cache_mode)
    if cache is False and cache_mode == "read_write":
        # Backward-compatible behavior: no cache writes when cache=False.
        # Explicitly avoid strict read-only hard-fail on cache miss.
        cache_mode = "off"

    # Resolve metadata CSV path (production default is canonical source only).
    csv_meta = _resolve_feature_metadata_csv(
        csv_meta,
        context="load_or_extract_features",
        enforce_canonical=enforce_canonical,
    )

    # Compute deterministic hash for this metadata + extraction params
    feature_cfg, cfg_path = _resolve_feature_config(
        config_path=config_path,
        resolved_feature_config=resolved_feature_config,
    )
    config_sha256 = compute_file_sha256(cfg_path) if cfg_path and cfg_path.exists() else None
    feature_identity = build_feature_identity(
        feature_cfg=feature_cfg,
        batch_size=batch_size,
        config_sha256=config_sha256,
    )
    params = {"batch_size": batch_size, "feature_identity": feature_identity}
    try:
        meta_hash = compute_meta_hash(csv_meta, params=params)
    except Exception:
        # If we cannot compute a hash (e.g., missing CSV), fall back to legacy behavior
        print(
            "[WARN] Could not compute metadata hash; falling back to legacy cache behavior."
        )
        return load_or_extract_features_legacy(
            out_dir=out_dir,
            csv_meta=csv_meta,
            batch_size=batch_size,
            cache=cache,
            enforce_canonical=enforce_canonical,
        )

    # Try to find existing hash-named cache
    cached = None
    if cache_mode in {"read_only", "read_write"} and not force_extract:
        cached = load_features_by_hash(out_dir, meta_hash)

    if cached is not None:
        # Basic shape validation
        meta = load_metadata(csv_meta)
        if cached.shape[0] != len(meta):
            msg = (
                "[WARN] Cache for hash {} has {} rows but metadata has {}.".format(
                    meta_hash, cached.shape[0], len(meta)
                )
            )
            if cache_mode == "read_only":
                raise ValueError(msg + " read_only mode forbids re-extraction.")
            print(msg + " Re-extracting.")
        else:
            cached_meta = load_meta_by_hash(out_dir, meta_hash) or {}
            cached_identity = cached_meta.get("feature_identity")
            if strict_cache_identity and not isinstance(cached_identity, dict):
                msg = (
                    f"[WARN] Cache meta for hash {meta_hash[:8]}... misses feature_identity"
                )
                if cache_mode == "read_only":
                    raise ValueError(msg + " strict read_only mode forbids fallback.")
                print(msg + " Re-extracting with current identity.")
            elif isinstance(cached_identity, dict) and cached_identity != feature_identity:
                msg = (
                    f"[WARN] Feature identity mismatch for cache hash {meta_hash[:8]}..."
                )
                if strict_cache_identity and cache_mode == "read_only":
                    raise ValueError(msg + " strict read_only mode forbids fallback.")
                print(msg + " Re-extracting with current identity.")
            else:
                print(
                    f"[INFO] ✓ Feature cache hit (hash: {meta_hash[:8]}..., shape: {cached.shape})"
                )
                return cached

    if cache_mode == "read_only":
        raise FileNotFoundError(
            "Feature cache miss in read_only mode for identity hash "
            f"{meta_hash[:8]}... (csv={csv_meta})."
        )

    if cache_mode == "off":
        print("[INFO] cache_mode=off -> extracting features without cache read/write")
        meta = load_metadata(csv_meta)
        feats, _ = _extract_features_with_provenance(
            meta,
            batch_size=batch_size,
            config_path=config_path,
            resolved_feature_config=resolved_feature_config,
        )
        return feats

    # In strict scientific mode, do not silently migrate/remove legacy caches.
    legacy = out_dir / "features.npy"
    if legacy.exists() and strict_cache_identity and not force_extract:
        raise RuntimeError(
            "Legacy cache file features.npy detected. "
            "Strict scientific mode forbids silent migration. "
            "Delete it manually or rerun with force_extract=True."
        )

    # If not found, check for legacy features.npy and attempt safe migration
    if legacy.exists() and cache_mode == "read_write":
        try:
            legacy_feats = np.load(legacy)
            meta = load_metadata(csv_meta)
            if legacy_feats.shape[0] == len(meta):
                # Safe to migrate
                meta_info = create_meta_info(
                    csv_meta,
                    params=params,
                    feature_identity=feature_identity,
                    config_sha256=config_sha256,
                )
                atomic_write_features_with_meta(
                    out_dir, legacy_feats, meta_hash, meta_info
                )
                # preserve legacy file as a timestamped backup
                backup_dir = out_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                backup_path = backup_dir / f"features_legacy_backup_{ts}.npy"
                legacy.rename(backup_path)
                print(
                    f"[INFO] Migrated legacy features.npy to features-{meta_hash}.npy and backed up legacy file."
                )
                return legacy_feats
            else:
                print(
                    "[WARN] Legacy features.npy row count ({}) does not match metadata ({}) - ignoring legacy cache.".format(
                        legacy_feats.shape[0], len(meta)
                    )
                )
        except Exception:
            print(
                "[WARN] Could not migrate legacy features.npy (read error); ignoring it."
            )

    # No usable cache found: extract and create a new hash-named cache
    print(
        f"[INFO] Feature cache miss (hash: {meta_hash[:8]}...) - extracting features with batch_size={batch_size}..."
    )
    meta = load_metadata(csv_meta)
    feats, provenance = _extract_features_with_provenance(
        meta,
        batch_size=batch_size,
        config_path=config_path,
        resolved_feature_config=resolved_feature_config,
    )
    provenance.setdefault("feature_identity", feature_identity)
    provenance.setdefault("config_sha256", config_sha256)

    if cache and cache_mode in {"write_only", "read_write"}:
        meta_info = create_meta_info(
            csv_meta,
            params=params,
            feature_identity=provenance.get("feature_identity"),
            model_provenance=provenance.get("model_provenance"),
            config_sha256=provenance.get("config_sha256"),
        )
        try:
            atomic_write_features_with_meta(out_dir, feats, meta_hash, meta_info)
        except Exception:
            # Fallback: try to write legacy file to not lose work
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "features.npy", feats)

    return feats


# Preserve the original legacy behavior as a small helper for safety fallback
def load_or_extract_features_legacy(
    out_dir: str | Path = "outputs",
    csv_meta: str | None = None,
    batch_size: int = 16,
    cache: bool = True,
    enforce_canonical: bool = False,
) -> np.ndarray:
    # Original algorithm preserved for falling back in rare error cases
    out_dir = Path(out_dir)
    features_path = out_dir / "features.npy"

    if features_path.exists():
        feats = np.load(features_path)

        # Determine metadata source for validation.
        csv_meta = _resolve_feature_metadata_csv(
            csv_meta,
            context="load_or_extract_features_legacy",
            enforce_canonical=enforce_canonical,
        )

        # Load metadata and validate shapes: cached features must match metadata rows
        try:
            meta = load_metadata(csv_meta)
            if feats.shape[0] != len(meta):
                print(
                    "[WARN] Cached features.npy rows ({}) != metadata rows ({}). "
                    "Removing stale cache and re-extracting features.".format(
                        feats.shape[0], len(meta)
                    )
                )
                try:
                    features_path.unlink()
                except Exception:
                    pass
                # fall through to extraction below
            else:
                return feats
        except Exception:
            # If metadata cannot be loaded for validation, conservatively re-extract
            print(
                "[WARN] Could not validate feature cache against metadata; re-extracting."
            )
            try:
                features_path.unlink()
            except Exception:
                pass

    # Determine metadata source.
    csv_meta = _resolve_feature_metadata_csv(
        csv_meta,
        context="load_or_extract_features_legacy",
        enforce_canonical=enforce_canonical,
    )

    meta = load_metadata(csv_meta)
    feats, _ = _extract_features_with_provenance(
        meta,
        batch_size=batch_size,
        config_path=None,
        resolved_feature_config=None,
    )

    if cache:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(features_path, feats)

    return feats


def save_selection(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_feature_metadata_csv(
    csv_meta: str | Path | None,
    *,
    context: str,
    enforce_canonical: bool = False,
) -> str:
    """Resolve metadata CSV for feature extraction/caching.

    Productive default is always the canonical source file.
    """
    if enforce_canonical:
        path = assert_canonical_metadata(
            csv_meta,
            context=context,
        )
    elif csv_meta is None:
        path = canonical_metadata_path()
    else:
        path = Path(csv_meta)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{context}: metadata CSV not found at '{path}'. "
            f"Expected canonical source '{CANONICAL_METADATA_RELATIVE_PATH.as_posix()}' "
            "for productive runs."
        )
    return str(path)
