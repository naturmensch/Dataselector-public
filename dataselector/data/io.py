import os
import time
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dataselector.data.metadata_processor import MetadataProcessor
from dataselector.data.metadata_source import (
    CANONICAL_METADATA_RELATIVE_PATH,
    assert_canonical_metadata,
    canonical_metadata_path,
)


def load_metadata(
    csv_path: str | Path,
    image_dir: str | Path | None = None,
    resolve_images: bool = True,
    strict_image_resolution: bool = False,
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
    gdf_metric = mp.ensure_metric_crs()
    attach_metric_gdf(df, gdf_metric)

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

    # Lade Modell-Konfiguration aus pipeline_config.yaml
    config_path = Path("config/pipeline_config.yaml")
    model_name = "dinov2"  # Fallback auf neuen Standard
    input_size = 392
    crop_size = (2048, 2048)
    pooling = "cls"
    model_variant = "dinov2_vits14"
    dinov2_repo = "facebookresearch/dinov2"
    dinov2_ref = "main"
    device = None
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                feat_cfg = cfg.get("feature_extraction", {})
                model_name = feat_cfg.get("model", "dinov2")
                input_size = int(feat_cfg.get("input_size", input_size))
                raw_crop = feat_cfg.get("crop_size", list(crop_size))
                if isinstance(raw_crop, (list, tuple)) and len(raw_crop) == 2:
                    crop_size = (int(raw_crop[0]), int(raw_crop[1]))
                pooling = str(feat_cfg.get("pooling", pooling)).strip().lower()
                model_variant = str(feat_cfg.get("model_variant", model_variant)).strip()
                dinov2_repo = str(feat_cfg.get("dinov2_repo", dinov2_repo)).strip()
                dinov2_ref = str(feat_cfg.get("dinov2_ref", dinov2_ref)).strip()
                cfg_device = str(feat_cfg.get("device", "auto")).strip().lower()
                device = None if cfg_device == "auto" else cfg_device
        except Exception as e:
            print(
                f"Warnung: Konnte Config nicht lesen ({e}), nutze Default: {model_name}"
            )

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
    return features


def load_or_extract_features(
    out_dir: str | Path = "outputs",
    csv_meta: str | None = None,
    batch_size: int = 16,
    cache: bool = True,
    enforce_canonical: bool = False,
) -> np.ndarray:
    """Load features from a hash-identified cache or extract and create a new cache."""
    from dataselector.pipeline.cache import (
        atomic_write_features_with_meta,
        compute_meta_hash,
        create_meta_info,
        features_path_for_hash,
        load_features_by_hash,
    )

    out_dir = Path(out_dir)

    # Resolve metadata CSV path (production default is canonical source only).
    csv_meta = _resolve_feature_metadata_csv(
        csv_meta,
        context="load_or_extract_features",
        enforce_canonical=enforce_canonical,
    )

    # Compute deterministic hash for this metadata + extraction params
    params = {"batch_size": batch_size}
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
    cached = load_features_by_hash(out_dir, meta_hash)
    if cached is not None:
        # Basic shape validation
        meta = load_metadata(csv_meta)
        if cached.shape[0] != len(meta):
            print(
                "[WARN] Cache for hash {} has {} rows but metadata has {}; removing cache and re-extracting.".format(
                    meta_hash, cached.shape[0], len(meta)
                )
            )
            try:
                fpath = features_path_for_hash(out_dir, meta_hash)
                fpath.unlink()
                mpath = out_dir / f"features-{meta_hash}.meta.json"
                if mpath.exists():
                    mpath.unlink()
            except Exception:
                pass
        else:
            print(
                f"[INFO] ✓ Feature cache hit (hash: {meta_hash[:8]}..., shape: {cached.shape})"
            )
            return cached

    # If not found, check for legacy features.npy and attempt safe migration
    legacy = out_dir / "features.npy"
    if legacy.exists():
        try:
            legacy_feats = np.load(legacy)
            meta = load_metadata(csv_meta)
            if legacy_feats.shape[0] == len(meta):
                # Safe to migrate
                meta_info = create_meta_info(csv_meta, params=params)
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
    feats = extract_features(meta, batch_size=batch_size)

    if cache:
        meta_info = create_meta_info(csv_meta, params=params)
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
    feats = extract_features(meta, batch_size=batch_size)

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
