import os
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
FEATURE_CACHE_SCOPES = {"run_local", "global_shared"}
DEFAULT_FEATURE_CACHE_ROOT = Path("outputs/cache/features")


def load_metadata(
    csv_path: str | Path,
    image_dir: str | Path | None = None,
    resolve_images: bool = True,
    strict_image_resolution: bool = False,
    strict_metric_crs: bool | None = None,
    metric_epsg: int | None = None,
    tile_exclusion_policy: str | Path | None = None,
    apply_tile_exclusion: bool | None = None,
) -> pd.DataFrame:
    mp = MetadataProcessor(str(csv_path))
    df = mp.load_csv()
    df = mp.add_temporal_metadata()

    policy_path = (
        Path(tile_exclusion_policy)
        if tile_exclusion_policy is not None
        else (
            Path(os.getenv("DATASELECTOR_TILE_EXCLUSION_POLICY"))
            if os.getenv("DATASELECTOR_TILE_EXCLUSION_POLICY")
            else None
        )
    )
    apply_policy = (
        bool(apply_tile_exclusion)
        if apply_tile_exclusion is not None
        else os.getenv("DATASELECTOR_APPLY_TILE_EXCLUSION", "0") == "1"
    )
    if apply_policy and policy_path is not None:
        from dataselector.data.tile_policy import (
            apply_tile_exclusion_policy,
            load_tile_exclusion_policy,
        )

        policy_payload, resolved_policy_path = load_tile_exclusion_policy(policy_path)
        policy_result = apply_tile_exclusion_policy(
            df,
            policy=policy_payload,
            policy_path=resolved_policy_path,
        )
        df = policy_result.dataframe
        mp.df = df
        # Keep CRS logic stable after row filtering by using the pandas/pyproj path.
        mp.gdf = None
        if hasattr(df, "attrs"):
            df.attrs["tile_exclusions_applied"] = bool(policy_result.applied)
            df.attrs["tile_exclusion_policy_sha256"] = policy_result.policy_sha256
            df.attrs["tile_exclusions_count"] = int(policy_result.excluded_count)
            df.attrs["tile_excluded_shortnames"] = list(policy_result.excluded_shortnames)
            df.attrs["effective_tile_count"] = int(len(df))

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
        if "effective_tile_count" not in df.attrs:
            df.attrs["effective_tile_count"] = int(len(df))
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
            # NOTE: Best-effort attachment only — if pandas.attrs is read-only or
            # mutated in unexpected environments we intentionally do not fail
            # the pipeline. This `pass` keeps the system robust across pandas
            # versions and container runtimes while preserving metric data when
            # possible.
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


def _resolve_feature_cache_settings(
    *,
    config_path: str | Path | None,
    cache_scope: str | None,
    cache_root: str | Path | None,
) -> tuple[str, Path]:
    """Resolve cache scope/root with explicit overrides winning over config defaults."""
    scope = "run_local"
    root = DEFAULT_FEATURE_CACHE_ROOT

    cfg_path: Path | None = Path(config_path) if config_path is not None else None
    if cfg_path is not None and cfg_path.exists():
        try:
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            section = payload.get("feature_cache", {}) if isinstance(payload, dict) else {}
            if isinstance(section, dict):
                configured_scope = section.get("scope")
                configured_root = section.get("root")
                if configured_scope:
                    scope = str(configured_scope).strip().lower()
                if configured_root:
                    root = Path(str(configured_root))
        except Exception:
            pass

    env_scope = os.getenv("DATASELECTOR_FEATURE_CACHE_SCOPE")
    env_root = os.getenv("DATASELECTOR_FEATURE_CACHE_ROOT")
    if env_scope:
        scope = env_scope.strip().lower()
    if env_root:
        root = Path(env_root)

    if cache_scope is not None:
        scope = str(cache_scope).strip().lower()
    if cache_root is not None:
        root = Path(cache_root)

    if scope not in FEATURE_CACHE_SCOPES:
        raise ValueError(
            f"Unknown feature cache scope '{scope}'. Expected one of: {sorted(FEATURE_CACHE_SCOPES)}"
        )
    return scope, root


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
    cache_scope: str | None = None,
    cache_root: str | Path | None = None,
) -> np.ndarray:
    """Load features from cache or extract and store with immutable provenance."""
    from dataselector.pipeline.cache import (
        atomic_write_features_with_meta,
        compute_meta_hash,
        create_meta_info,
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

    feature_cfg, cfg_path = _resolve_feature_config(
        config_path=config_path,
        resolved_feature_config=resolved_feature_config,
    )
    effective_scope, effective_cache_root = _resolve_feature_cache_settings(
        config_path=str(cfg_path) if cfg_path else config_path,
        cache_scope=cache_scope,
        cache_root=cache_root,
    )
    cache_dir = (
        effective_cache_root if effective_scope == "global_shared" else Path(out_dir)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve metadata CSV path (production default is canonical source only).
    csv_meta = _resolve_feature_metadata_csv(
        csv_meta,
        context="load_or_extract_features",
        enforce_canonical=enforce_canonical,
    )

    # Compute deterministic hash for this metadata + extraction params
    config_sha256 = compute_file_sha256(cfg_path) if cfg_path and cfg_path.exists() else None
    feature_identity = build_feature_identity(
        feature_cfg=feature_cfg,
        batch_size=batch_size,
        config_sha256=config_sha256,
    )
    params = {"batch_size": batch_size, "feature_identity": feature_identity}
    meta_hash = compute_meta_hash(csv_meta, params=params)

    # Try to find existing hash-named cache
    cached = None
    if cache_mode in {"read_only", "read_write"} and not force_extract:
        cached = load_features_by_hash(cache_dir, meta_hash)

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
            cached_meta = load_meta_by_hash(cache_dir, meta_hash)
            if not isinstance(cached_meta, dict):
                raise RuntimeError(
                    f"Corrupt immutable cache for hash {meta_hash[:8]}...: missing or unreadable meta.json"
                )
            cached_identity = cached_meta.get("feature_identity")
            if strict_cache_identity and not isinstance(cached_identity, dict):
                msg = (
                    f"[WARN] Cache meta for hash {meta_hash[:8]}... misses feature_identity"
                )
                if cache_mode == "read_only":
                    raise ValueError(msg + " strict read_only mode forbids fallback.")
                print(msg + " Re-extracting with current identity.")
            elif isinstance(cached_identity, dict) and cached_identity != feature_identity:
                raise RuntimeError(
                    f"Immutable cache conflict for hash {meta_hash[:8]}...: "
                    "feature_identity in meta does not match current feature identity."
                )
            else:
                print(
                    f"[INFO] ✓ Feature cache hit (scope={effective_scope}, hash={meta_hash[:8]}..., shape={cached.shape})"
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

    # No usable cache found: extract and create (or verify) immutable hash-named cache.
    print(
        f"[INFO] Feature cache miss (scope={effective_scope}, hash={meta_hash[:8]}...) - extracting features with batch_size={batch_size}..."
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
        meta_info["cache_scope"] = effective_scope
        meta_info["cache_root"] = str(cache_dir.resolve())
        atomic_write_features_with_meta(cache_dir, feats, meta_hash, meta_info)

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
