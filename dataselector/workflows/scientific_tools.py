"""CLI-first scientific helper workflows.

This module centralizes scientific analysis helpers that were previously
implemented as top-level scripts. Top-level scripts are kept as thin wrappers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataselector.cli_decorators import cli_command

logger = logging.getLogger(__name__)


def _load_common_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    from dataselector.pipeline.validation_config import get_required, load_config

    cfg = load_config(config_path)
    output_dir = Path(get_required(cfg, ["output.dir", "pipeline.output_dir"], "output dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return cfg, output_dir


def _extract_weight_triplet(config: dict[str, Any]) -> tuple[float, float, float]:
    from dataselector.pipeline.validation_config import get_required

    alpha = float(
        get_required(
            config,
            ["selection.alpha_visual", "selection.weights.alpha"],
            "selection alpha",
        )
    )
    beta = float(
        get_required(
            config,
            ["selection.beta_spatial", "selection.weights.beta"],
            "selection beta",
        )
    )
    gamma = float(
        get_required(
            config,
            ["selection.gamma_temporal", "selection.weights.gamma"],
            "selection gamma",
        )
    )
    return alpha, beta, gamma


def _extract_sweep_base_params(config: dict[str, Any]) -> dict[str, Any]:
    from dataselector.pipeline.validation_config import get_required

    return {
        "csv_meta": get_required(
            config,
            ["data.csv_path", "data.metadata_path", "data.csv_meta"],
            "data CSV path",
        ),
        "n_samples": int(get_required(config, ["selection.n_samples"], "selection n_samples")),
        "batch_size": int(
            get_required(
                config,
                ["feature_extraction.batch_size", "data.batch_size"],
                "feature batch_size",
            )
        ),
        "n_clusters": int(get_required(config, ["clustering.n_clusters"], "clustering n_clusters")),
        "min_distance_km": float(
            get_required(
                config,
                ["selection.min_distance_km", "selection.spatial_constraint.min_distance_km"],
                "selection min_distance_km",
            )
        ),
    }


def _generate_sensitivity_variations(
    base_weights: dict[str, float],
    variation_percent: float,
) -> list[tuple[str, tuple[float, float, float]]]:
    variations: list[tuple[str, tuple[float, float, float]]] = []
    variations.append(
        (
            "baseline",
            (
                float(base_weights["alpha"]),
                float(base_weights["beta"]),
                float(base_weights["gamma"]),
            ),
        )
    )

    for weight_name in ("alpha", "beta", "gamma"):
        for offset in (
            -variation_percent,
            -variation_percent / 2.0,
            variation_percent / 2.0,
            variation_percent,
        ):
            candidate = dict(base_weights)
            candidate[weight_name] = max(
                0.0,
                min(1.0, base_weights[weight_name] * (1.0 + offset / 100.0)),
            )
            total = float(sum(candidate.values()))
            if total <= 0:
                continue
            for key in candidate:
                candidate[key] = float(candidate[key] / total)
            variations.append(
                (
                    f"{weight_name}_{offset:+.0f}%",
                    (
                        float(candidate["alpha"]),
                        float(candidate["beta"]),
                        float(candidate["gamma"]),
                    ),
                )
            )
    return variations


def run_sensitivity_sweep(config_path: str | Path, variation_percent: float) -> dict[str, Any]:
    import pandas as pd

    from dataselector.pipeline.experiments import ExperimentRunner

    config, output_dir = _load_common_config(config_path)
    params = _extract_sweep_base_params(config)
    alpha, beta, gamma = _extract_weight_triplet(config)
    base_weights = {"alpha": alpha, "beta": beta, "gamma": gamma}
    variations = _generate_sensitivity_variations(base_weights, variation_percent)

    runner = ExperimentRunner(output_dir=str(output_dir), feature_cache_dir=str(output_dir))
    rows: list[dict[str, Any]] = []
    for label, (w_alpha, w_beta, w_gamma) in variations:
        logger.info(
            "Running sensitivity variation %s (a=%.4f b=%.4f g=%.4f)",
            label,
            w_alpha,
            w_beta,
            w_gamma,
        )
        try:
            result_df = runner.run_weight_sweep(
                csv_meta=str(params["csv_meta"]),
                n_samples=int(params["n_samples"]),
                weight_combinations=[(w_alpha, w_beta, w_gamma)],
                n_clusters=int(params["n_clusters"]),
                batch_size=int(params["batch_size"]),
                min_distance_km=float(params["min_distance_km"]),
                max_runs=1,
            )
            if result_df.empty:
                rows.append(
                    {
                        "variation": label,
                        "alpha": w_alpha,
                        "beta": w_beta,
                        "gamma": w_gamma,
                        "score": None,
                        "n_selected": None,
                    }
                )
                continue
            row = result_df.iloc[0]
            rows.append(
                {
                    "variation": label,
                    "alpha": w_alpha,
                    "beta": w_beta,
                    "gamma": w_gamma,
                    "score": float(row.get("score", 0.0)),
                    "n_selected": int(row.get("n_selected", params["n_samples"])),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.error("Sensitivity variation failed (%s): %s", label, exc)
            rows.append(
                {
                    "variation": label,
                    "alpha": w_alpha,
                    "beta": w_beta,
                    "gamma": w_gamma,
                    "score": None,
                    "n_selected": None,
                    "error": str(exc),
                }
            )

    result_df = pd.DataFrame(rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"sensitivity_results_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)
    metadata_path = output_dir / f"sensitivity_metadata_{timestamp}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(config_path).resolve()),
                "variation_percent": variation_percent,
                "base_weights": base_weights,
                "n_variations": len(variations),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Sensitivity sweep written: %s", csv_path)
    return {
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir),
        "results_csv": str(csv_path),
        "metadata_json": str(metadata_path),
    }


def run_ablation_study(config_path: str | Path) -> dict[str, Any]:
    import pandas as pd

    from dataselector.pipeline.experiments import ExperimentRunner

    config, output_dir = _load_common_config(config_path)
    params = _extract_sweep_base_params(config)
    trial_weights = _extract_weight_triplet(config)
    scenarios = [
        ("visual_only", (1.0, 0.0, 0.0)),
        ("spatial_only", (0.0, 1.0, 0.0)),
        ("temporal_only", (0.0, 0.0, 1.0)),
        ("visual+spatial", (0.5, 0.5, 0.0)),
        ("visual+temporal", (0.5, 0.0, 0.5)),
        ("spatial+temporal", (0.0, 0.5, 0.5)),
        ("visual+spatial+temporal", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        ("trial_config", trial_weights),
    ]

    runner = ExperimentRunner(output_dir=str(output_dir), feature_cache_dir=str(output_dir))
    rows: list[dict[str, Any]] = []
    for name, (alpha, beta, gamma) in scenarios:
        logger.info("Running ablation scenario %s", name)
        try:
            result_df = runner.run_weight_sweep(
                csv_meta=str(params["csv_meta"]),
                n_samples=int(params["n_samples"]),
                weight_combinations=[(alpha, beta, gamma)],
                n_clusters=int(params["n_clusters"]),
                batch_size=int(params["batch_size"]),
                min_distance_km=float(params["min_distance_km"]),
                max_runs=1,
            )
            if result_df.empty:
                rows.append(
                    {
                        "scenario": name,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "score": None,
                        "n_selected": None,
                    }
                )
                continue
            row = result_df.iloc[0]
            rows.append(
                {
                    "scenario": name,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "score": float(row.get("score", 0.0)),
                    "n_selected": int(row.get("n_selected", params["n_samples"])),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.error("Ablation scenario failed (%s): %s", name, exc)
            rows.append(
                {
                    "scenario": name,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "score": None,
                    "n_selected": None,
                    "error": str(exc),
                }
            )

    result_df = pd.DataFrame(rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"ablation_study_results_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)
    json_path = output_dir / f"ablation_study_results_{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(config_path).resolve()),
                "rows": rows,
                "trial_weights": {
                    "alpha": trial_weights[0],
                    "beta": trial_weights[1],
                    "gamma": trial_weights[2],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Ablation study written: %s", csv_path)
    return {
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir),
        "results_csv": str(csv_path),
        "results_json": str(json_path),
    }


def _extract_features_for_backbones(
    *,
    csv_meta: str,
    data_dir: Path,
    batch_size: int,
    crop_size: tuple[int, int],
    device: str | None,
    dinov2_input_size: int,
    resnet_input_size: int,
    dinov2_pooling: str,
    dinov2_model_variant: str,
    dinov2_repo: str,
    dinov2_ref: str,
) -> tuple[np.ndarray, np.ndarray]:
    from dataselector.features.feature_extractor import FeatureExtractor
    from dataselector.pipeline.experiments import load_metadata

    metadata = load_metadata(csv_meta)
    image_paths = metadata["longName"].tolist()
    dinov2_extractor = FeatureExtractor(
        model_name="dinov2",
        input_size=dinov2_input_size,
        device=device,
        default_crop_size=crop_size,
        pooling=dinov2_pooling,
        model_variant=dinov2_model_variant,
        dinov2_repo=dinov2_repo,
        dinov2_ref=dinov2_ref,
    )
    dinov2_features = dinov2_extractor.extract_features_batch(
        image_paths,
        data_dir=data_dir,
        batch_size=batch_size,
        crop_size=crop_size,
    )
    resnet_extractor = FeatureExtractor(
        model_name="resnet50",
        input_size=resnet_input_size,
        device=device,
        default_crop_size=crop_size,
    )
    resnet_features = resnet_extractor.extract_features_batch(
        image_paths,
        data_dir=data_dir,
        batch_size=batch_size,
        crop_size=crop_size,
    )
    return dinov2_features, resnet_features


def run_backbone_comparison(config_path: str | Path) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from dataselector.pipeline.validation_config import get_required
    from dataselector.selection.clustering import ClusteringPipeline

    config, output_dir = _load_common_config(config_path)
    csv_meta = get_required(
        config,
        ["data.csv_path", "data.metadata_path", "data.csv_meta"],
        "data CSV path",
    )
    data_dir = Path(get_required(config, ["data.image_dir"], "data image_dir"))
    n_clusters = int(get_required(config, ["clustering.n_clusters"], "clustering n_clusters"))
    batch_size = int(
        get_required(
            config,
            ["feature_extraction.batch_size", "data.batch_size"],
            "feature batch_size",
        )
    )
    crop_size = tuple(int(x) for x in get_required(config, ["feature_extraction.crop_size"], "crop_size"))
    device = get_required(config, ["feature_extraction.device"], "feature device")
    if device == "auto":
        device = None
    dinov2_input_size = int(
        get_required(config, ["feature_extraction.input_size"], "dinov2 input size")
    )
    resnet_input_size = int(
        get_required(config, ["feature_extraction.resnet_input_size"], "resnet input size")
    )
    dinov2_pooling = str(
        get_required(config, ["feature_extraction.pooling"], "dinov2 pooling")
    ).strip().lower()
    dinov2_model_variant = str(
        get_required(config, ["feature_extraction.model_variant"], "dinov2 model variant")
    ).strip()
    dinov2_repo = str(
        get_required(config, ["feature_extraction.dinov2_repo"], "dinov2 repo")
    ).strip()
    dinov2_ref = str(
        get_required(config, ["feature_extraction.dinov2_ref"], "dinov2 ref")
    ).strip()

    dinov2_feats, resnet_feats = _extract_features_for_backbones(
        csv_meta=str(csv_meta),
        data_dir=data_dir,
        batch_size=batch_size,
        crop_size=(crop_size[0], crop_size[1]),
        device=device,
        dinov2_input_size=dinov2_input_size,
        resnet_input_size=resnet_input_size,
        dinov2_pooling=dinov2_pooling,
        dinov2_model_variant=dinov2_model_variant,
        dinov2_repo=dinov2_repo,
        dinov2_ref=dinov2_ref,
    )

    clustering_dino = ClusteringPipeline(n_clusters=n_clusters, random_state=42)
    dino_emb, dino_labels = clustering_dino.fit_transform(dinov2_feats)
    clustering_resnet = ClusteringPipeline(n_clusters=n_clusters, random_state=42)
    resnet_emb, resnet_labels = clustering_resnet.fit_transform(resnet_feats)

    metrics = {
        "silhouette_dinov2": float(silhouette_score(dino_emb, dino_labels)),
        "silhouette_resnet50": float(silhouette_score(resnet_emb, resnet_labels)),
        "adjusted_rand_index": float(adjusted_rand_score(dino_labels, resnet_labels)),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"backbone_comparison_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(config_path).resolve()),
                "metrics": metrics,
                "dinov2_shape": list(dinov2_feats.shape),
                "resnet50_shape": list(resnet_feats.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Backbone comparison written: %s", out_path)
    return {"config_path": str(Path(config_path).resolve()), "results_json": str(out_path)}


def run_validate_kmeans(config_path: str | Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_samples,
        silhouette_score,
    )

    from dataselector.pipeline.experiments import load_or_extract_features
    from dataselector.pipeline.validation_config import get_required
    from dataselector.selection.clustering import ClusteringPipeline

    config, output_dir = _load_common_config(config_path)
    csv_meta = get_required(
        config,
        ["data.csv_path", "data.metadata_path", "data.csv_meta"],
        "data CSV path",
    )
    batch_size = int(
        get_required(
            config,
            ["feature_extraction.batch_size", "data.batch_size"],
            "feature batch_size",
        )
    )
    n_clusters = int(get_required(config, ["clustering.n_clusters"], "clustering n_clusters"))
    umap_components = int(get_required(config, ["clustering.umap_components"], "clustering umap_components"))
    umap_neighbors = int(get_required(config, ["clustering.umap_n_neighbors"], "clustering umap_n_neighbors"))
    random_state = int(
        get_required(config, ["clustering.umap_random_state", "selection.random_state"], "random_state")
    )

    features = load_or_extract_features(
        out_dir=output_dir,
        csv_meta=str(csv_meta),
        batch_size=batch_size,
        cache=True,
    )
    clusterer = ClusteringPipeline(
        n_clusters=n_clusters,
        umap_n_components=umap_components,
        random_state=random_state,
        umap_random_state=random_state,
        umap_n_neighbors=umap_neighbors,
    )
    embeddings, labels = clusterer.fit_transform(features)
    silhouette_avg = float(silhouette_score(embeddings, labels))
    silhouette_per = silhouette_samples(embeddings, labels)
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    metrics = {
        "silhouette_score": silhouette_avg,
        "silhouette_per_cluster": {
            int(i): float(np.mean(silhouette_per[labels == i])) for i in range(n_clusters)
        },
        "davies_bouldin_index": float(davies_bouldin_score(embeddings, labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(embeddings, labels)),
        "kmeans_inertia": float(clusterer.kmeans.inertia_),
        "cluster_sizes": {int(i): int(cluster_sizes[i]) for i in range(n_clusters)},
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"clustering_validation_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(config_path).resolve()),
                "metrics": metrics,
                "data_shapes": {
                    "features": list(features.shape),
                    "embeddings": list(embeddings.shape),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("KMeans validation written: %s", out_path)
    return {"config_path": str(Path(config_path).resolve()), "results_json": str(out_path)}


def _continuity_score(
    features_high: Any,
    embeddings_low: Any,
    n_neighbors: int,
) -> float:
    import numpy as np
    from sklearn.metrics import pairwise_distances

    n_samples = features_high.shape[0]
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be < number of samples")

    dist_high = pairwise_distances(features_high)
    dist_low = pairwise_distances(embeddings_low)
    high_neighbors = np.argsort(dist_high, axis=1)[:, 1 : n_neighbors + 1]
    low_ranks = np.argsort(np.argsort(dist_low, axis=1), axis=1)

    penalty_sum = 0.0
    for i in range(n_samples):
        for j in high_neighbors[i]:
            rank = low_ranks[i, j]
            if rank > n_neighbors:
                penalty_sum += rank - n_neighbors

    denom = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    return 1.0 - (2.0 / denom) * penalty_sum


def run_validate_umap(config_path: str | Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.manifold import trustworthiness
    from sklearn.preprocessing import StandardScaler

    from dataselector.pipeline.experiments import load_or_extract_features
    from dataselector.pipeline.validation_config import get_required

    config, output_dir = _load_common_config(config_path)
    csv_meta = get_required(
        config,
        ["data.csv_path", "data.metadata_path", "data.csv_meta"],
        "data CSV path",
    )
    batch_size = int(
        get_required(
            config,
            ["feature_extraction.batch_size", "data.batch_size"],
            "feature batch_size",
        )
    )
    n_components = int(get_required(config, ["clustering.umap_components"], "umap components"))
    n_neighbors = int(get_required(config, ["clustering.umap_n_neighbors"], "umap neighbors"))
    random_state = int(get_required(config, ["clustering.umap_random_state"], "umap random state"))
    min_dist = float(get_required(config, ["clustering.umap_min_dist"], "umap min_dist"))
    metric = str(get_required(config, ["clustering.umap_metric"], "umap metric"))
    n_jobs = int(get_required(config, ["clustering.umap_n_jobs"], "umap n_jobs"))
    max_samples = int(
        get_required(config, ["validation.umap_max_samples"], "validation umap_max_samples")
    )

    features = load_or_extract_features(
        out_dir=output_dir,
        csv_meta=str(csv_meta),
        batch_size=batch_size,
        cache=True,
    )
    scaled = StandardScaler().fit_transform(features)
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            min_dist=min_dist,
            metric=metric,
            n_jobs=n_jobs,
        )
        embeddings = reducer.fit_transform(scaled)
        method = "umap"
    except ImportError:  # pragma: no cover - environment dependent
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=random_state)
        embeddings = reducer.fit_transform(scaled)
        method = "pca_fallback"

    if scaled.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(scaled.shape[0], size=max_samples, replace=False)
        scaled_eval = scaled[idx]
        emb_eval = embeddings[idx]
    else:
        scaled_eval = scaled
        emb_eval = embeddings

    trust = float(trustworthiness(scaled_eval, emb_eval, n_neighbors=n_neighbors))
    cont = float(_continuity_score(scaled_eval, emb_eval, n_neighbors=n_neighbors))
    metrics = {
        "trustworthiness": trust,
        "continuity": cont,
        "mean_preservation": float((trust + cont) / 2.0),
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"umap_validation_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(config_path).resolve()),
                "method": method,
                "metrics": metrics,
                "hyperparameters": {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "random_state": random_state,
                    "min_dist": min_dist,
                    "metric": metric,
                    "n_jobs": n_jobs,
                },
                "input_shape": list(features.shape),
                "embedding_shape": list(embeddings.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("UMAP validation written: %s", out_path)
    return {"config_path": str(Path(config_path).resolve()), "results_json": str(out_path)}


def run_snapshot_config(
    *,
    config_path: str | Path,
    output_dir: str | Path = "outputs/runs",
    provenance_json: str | Path | None = None,
    notes: str = "",
    name: str = "final_config",
) -> dict[str, Any]:
    import yaml

    from dataselector.runtime.parameter_snapshot import build_snapshot, write_snapshot

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle) or {}
    provenance = {}
    if provenance_json:
        with Path(provenance_json).open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
    snapshot = build_snapshot(
        parameters=parameters,
        provenance=provenance,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path.resolve()),
        },
        notes=notes,
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(output_dir) / f"{name}_{ts}.yaml"
    write_snapshot(snapshot, out_path)
    logger.info("Snapshot written: %s", out_path)
    return {
        "snapshot_path": str(out_path),
        "parameters_hash": snapshot["hashes"]["parameters_hash"],
    }


@cli_command(
    "sensitivity-sweep",
    help="Run hyperparameter sensitivity analysis",
    args={
        "config": {"type": str, "required": True, "help": "Path to config YAML"},
        "variation_percent": {
            "type": float,
            "required": True,
            "help": "Variation percentage (e.g. 20)",
        },
    },
)
def cli_sensitivity_sweep(config: str, variation_percent: float) -> int:
    run_sensitivity_sweep(config_path=config, variation_percent=variation_percent)
    return 0


@cli_command(
    "ablation-study",
    help="Run diversity ablation study",
    args={"config": {"type": str, "required": True, "help": "Path to config YAML"}},
)
def cli_ablation_study(config: str) -> int:
    run_ablation_study(config_path=config)
    return 0


@cli_command(
    "compare-backbones",
    help="Compare DINOv2 and ResNet50 feature backbones",
    args={"config": {"type": str, "required": True, "help": "Path to config YAML"}},
)
def cli_compare_backbones(config: str) -> int:
    run_backbone_comparison(config_path=config)
    return 0


@cli_command(
    "validate-kmeans",
    help="Validate clustering metrics for configured KMeans/UMAP pipeline",
    args={"config": {"type": str, "required": True, "help": "Path to config YAML"}},
)
def cli_validate_kmeans(config: str) -> int:
    run_validate_kmeans(config_path=config)
    return 0


@cli_command(
    "validate-umap",
    help="Validate UMAP topology preservation metrics",
    args={"config": {"type": str, "required": True, "help": "Path to config YAML"}},
)
def cli_validate_umap(config: str) -> int:
    run_validate_umap(config_path=config)
    return 0


@cli_command(
    "snapshot-config",
    help="Create final config snapshot with provenance/hash metadata",
    args={
        "config": {"type": str, "required": True, "help": "Path to config YAML"},
        "output_dir": {
            "type": str,
            "default": "outputs/runs",
            "help": "Output directory for snapshot",
        },
        "provenance_json": {
            "type": str,
            "default": None,
            "help": "Optional provenance JSON path",
        },
        "notes": {"type": str, "default": "", "help": "Optional notes"},
        "name": {
            "type": str,
            "default": "final_config",
            "help": "Snapshot filename prefix",
        },
    },
)
def cli_snapshot_config(
    config: str,
    output_dir: str = "outputs/runs",
    provenance_json: str | None = None,
    notes: str = "",
    name: str = "final_config",
) -> int:
    run_snapshot_config(
        config_path=config,
        output_dir=output_dir,
        provenance_json=provenance_json,
        notes=notes,
        name=name,
    )
    return 0
