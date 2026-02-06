import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dataselector.data.metadata_processor import MetadataProcessor
from dataselector.features.feature_extractor import FeatureExtractor


def load_metadata(csv_path: str) -> pd.DataFrame:
    mp = MetadataProcessor(str(csv_path))
    df = mp.load_csv()
    df = mp.add_temporal_metadata()
    df = mp.resolve_image_paths("data/images")

    # Ensure metric CRS (UTM) is available for precise spatial calculations
    # Attach into DataFrame.attrs to avoid fragile attribute access and to persist through copies
    gdf_metric = mp.ensure_metric_crs()
    attach_metric_gdf(df, gdf_metric)

    # ensure placeholder for missing images
    df["image_path"] = df["image_path"].fillna("missing_placeholder.png")
    return df


# Helper utilities for robustly attaching/reading metric GeoDataFrame
def attach_metric_gdf(df, gdf_metric):
    """Attach the metric GeoDataFrame to pandas DataFrame metadata in a safe way.

    Stores the gdf in df.attrs['gdf_metric'] so it survives common pandas operations.
    """
    try:
        if isinstance(df, (type.__class__, object)):
            # Accept arbitrary containers --- for pandas DataFrame use .attrs
            try:
                df.attrs["gdf_metric"] = gdf_metric
            except Exception:
                # Fallback: set attribute for backward compatibility
                setattr(df, "gdf_metric", gdf_metric)
        else:
            setattr(df, "gdf_metric", gdf_metric)
    except Exception:
        # Last resort: set attribute
        setattr(df, "gdf_metric", gdf_metric)


def get_metric_gdf(df):
    """Retrieve attached metric GeoDataFrame if present (attrs or attribute)."""
    try:
        if hasattr(df, "attrs") and "gdf_metric" in df.attrs:
            return df.attrs.get("gdf_metric")
    except Exception:
        pass
    # Backward-compatible fallback
    return getattr(df, "gdf_metric", None)


def extract_features(metadata: pd.DataFrame, batch_size: int = 16) -> np.ndarray:
    # Lade Modell-Konfiguration aus pipeline_config.yaml
    config_path = Path("config/pipeline_config.yaml")
    model_name = "dinov2"  # Fallback auf neuen Standard
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                model_name = cfg.get("feature_extraction", {}).get("model", "dinov2")
        except Exception as e:
            print(
                f"Warnung: Konnte Config nicht lesen ({e}), nutze Default: {model_name}"
            )

    fe = FeatureExtractor(model_name=model_name)
    image_paths = metadata["image_path"].tolist()
    features = fe.extract_features_batch(
        image_paths=image_paths, data_dir=Path("."), batch_size=batch_size
    )
    return features


def load_or_extract_features(
    out_dir: str | Path = "outputs",
    csv_meta: str | None = None,
    batch_size: int = 16,
    cache: bool = True,
) -> np.ndarray:
    """Load features from a hash-identified cache or extract and create a new cache."""
    from dataselector.pipeline.cache import (
        atomic_write_features_with_meta,
        compute_meta_hash,
        create_meta_info,
        features_path_for_hash,
        find_cache_by_hash,
        load_features_by_hash,
    )

    out_dir = Path(out_dir)

    # Resolve metadata CSV path
    if csv_meta is None:
        candidate = out_dir / "metadata.csv"
        csv_meta = str(candidate) if candidate.exists() else "data/new_all_tiles.csv"

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
            out_dir=out_dir, csv_meta=csv_meta, batch_size=batch_size, cache=cache
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
) -> np.ndarray:
    # Original algorithm preserved for falling back in rare error cases
    out_dir = Path(out_dir)
    features_path = out_dir / "features.npy"

    if features_path.exists():
        feats = np.load(features_path)

        # Determine metadata source for validation
        if csv_meta is None:
            candidate = out_dir / "metadata.csv"
            if candidate.exists():
                csv_meta = str(candidate)
            else:
                csv_meta = "data/new_all_tiles.csv"

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

    # Determine metadata source
    if csv_meta is None:
        candidate = out_dir / "metadata.csv"
        if candidate.exists():
            csv_meta = str(candidate)
        else:
            # Fallback to project CSV
            csv_meta = "data/new_all_tiles.csv"

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
