import itertools
from datetime import timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from dataselector.analysis.metrics import compute_metrics

# Monkeypatch-friendly local wrappers for metadata/features I/O
from dataselector.data.io import ensure_output_dir
from dataselector.data.io import extract_features as _io_extract_features
from dataselector.data.io import load_metadata as _io_load_metadata
from dataselector.data.io import save_selection
from dataselector.data.metadata_source import (
    CANONICAL_METADATA_RELATIVE_PATH,
    canonical_metadata_path,
)
from dataselector.data.spatial_schema import normalize_spatial_schema
from dataselector.data.spatial_schema import spatial_spread as compute_spatial_spread
from dataselector.selection.diversity_selector import DiversitySelector
from dataselector.workflows.objective_scoring import (
    compute_baselines,
    normalized_objective,
)


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Local wrapper around src.io.load_metadata to allow tests to monkeypatch
    src.experiments.load_metadata without touching src.io.
    """
    return _io_load_metadata(csv_path)


def extract_features(metadata: pd.DataFrame, batch_size: int = 16):
    """Local wrapper around src.io.extract_features to allow tests to monkeypatch
    src.experiments.extract_features.
    """
    return _io_extract_features(metadata, batch_size=batch_size)


def load_or_extract_features(
    out_dir: str | Path = "outputs",
    csv_meta: str | None = None,
    batch_size: int = 16,
    cache: bool = True,
):
    """Load features from cache or extract using local wrappers.

    Replicates the logic from src.io.load_or_extract_features but uses the
    local load_metadata/extract_features functions so tests can monkeypatch
    without performing real I/O.
    """
    import numpy as np

    out_dir = Path(out_dir)
    features_path = out_dir / "features.npy"

    if features_path.exists():
        return np.load(features_path)

    # Determine metadata source (canonical by default).
    if csv_meta is None:
        csv_meta = str(canonical_metadata_path())

    csv_path = Path(csv_meta)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            "pipeline.experiments.load_or_extract_features: metadata CSV not found at "
            f"'{csv_path}'. Expected canonical source "
            f"'{CANONICAL_METADATA_RELATIVE_PATH.as_posix()}' for productive runs."
        )
    csv_meta = str(csv_path)

    meta = load_metadata(csv_meta)
    feats = extract_features(meta, batch_size=batch_size)

    if cache:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(features_path, feats)

    return feats


class ExperimentRunner:
    def __init__(
        self,
        output_dir: str = "outputs/tuning_weights",
        feature_cache_dir: str | Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        ensure_output_dir(self.output_dir)
        self.feature_cache_dir = (
            Path(feature_cache_dir)
            if feature_cache_dir is not None
            else self.output_dir
        )
        ensure_output_dir(self.feature_cache_dir)

    def run_weight_sweep(
        self,
        csv_meta: str,
        n_samples: int,
        weight_combinations: List[Tuple[float, float, float]] = None,
        alpha_vals: List[float] = None,
        beta_vals: List[float] = None,
        gamma_vals: List[float] = None,
        n_clusters: int = 8,
        batch_size: int = 16,
        min_distance_km: float | None = None,
        patience: int = 5,
        max_runs: int = None,
        score_fn=None,
        objective_authority: str = "unified_normalized",
        objective_weight_diversity: float = 0.5,
        objective_weight_spread: float = 0.5,
        objective_infeasible_penalty: float = 0.1,
        pre_selected: list = None,
        pre_selected_names: list = None,
    ) -> pd.DataFrame:
        meta = load_metadata(csv_meta)
        # Load cached features when available to avoid repeated expensive extraction.
        # For shared cache directories, use the hash-based I/O cache implementation.
        if self.feature_cache_dir != self.output_dir:
            from dataselector.data.io import (
                load_or_extract_features as io_load_or_extract_features,
            )

            features = io_load_or_extract_features(
                out_dir=self.feature_cache_dir,
                csv_meta=csv_meta,
                batch_size=batch_size,
                cache=True,
            )
        else:
            # Keep monkeypatch-friendly local behavior for unit/integration tests.
            features = load_or_extract_features(
                out_dir=self.feature_cache_dir,
                csv_meta=csv_meta,
                batch_size=batch_size,
                cache=True,
            )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Resolve min-distance policy only once per sweep.
        resolved_min_distance = (
            float(min_distance_km) if min_distance_km is not None else None
        )
        if resolved_min_distance is None:
            try:
                import yaml

                cfg_path = Path("config/pipeline_config.yaml")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg_dist = cfg.get("selection", {}).get("min_distance_km")
                if cfg_dist is not None:
                    resolved_min_distance = float(cfg_dist)
            except Exception:
                resolved_min_distance = None

        if resolved_min_distance is None:
            from dataselector.pipeline.pipeline_utils import compute_min_distance_km

            resolved_min_distance = float(compute_min_distance_km(csv_meta))

        results = []

        # Build iterable of weight combinations (backwards compatible)
        if weight_combinations is not None:
            combos = list(weight_combinations)
        else:
            alpha_vals = alpha_vals or []
            beta_vals = beta_vals or []
            gamma_vals = gamma_vals or []
            combos = list(itertools.product(alpha_vals, beta_vals, gamma_vals))

        total_runs = len(combos)
        run_i = 0

        objective_mode = str(objective_authority).strip().lower()
        if objective_mode not in {"unified_normalized", "legacy_lexicographic"}:
            raise ValueError(
                "objective_authority must be unified_normalized|legacy_lexicographic "
                f"(got {objective_authority!r})"
            )

        baseline_diversity, baseline_spread = compute_baselines(
            features=features,
            metadata=meta,
            metric="euclidean",
        )
        spatial_meta = normalize_spatial_schema(meta, require_bounds=True, copy=True)

        def _selection_diversity(selected_indices: np.ndarray) -> float:
            if len(selected_indices) <= 1:
                return 0.0
            return float(np.mean(pdist(features[selected_indices], metric="euclidean")))

        def _selection_spread(selected_indices: np.ndarray) -> float:
            if len(selected_indices) == 0:
                return 0.0
            return float(compute_spatial_spread(spatial_meta, selected_indices))

        if score_fn is None:
            if objective_mode == "unified_normalized":

                def score_fn(m):
                    return float(m.get("objective_score", float("-inf")))

            else:

                def score_fn(m):
                    return (
                        m["clusters_covered"],
                        m["temporal_std"],
                        m["spatial_mean_km"],
                    )

        best_score = None
        best_metrics = None
        no_improve = 0

        for alpha, beta, gamma in combos:
            if max_runs is not None and run_i >= max_runs:
                print(f"Reached max_runs={max_runs}, stopping.")
                break

            run_i += 1
            print(
                f"Running {run_i}/{total_runs}: alpha={alpha}, beta={beta}, gamma={gamma}"
            )

            ds = DiversitySelector(
                n_samples=n_samples, use_multi_criteria=True, random_state=42
            )

            selected_idx = ds.select(
                features=features,
                metadata=meta,
                alpha_visual=alpha,
                beta_spatial=beta,
                gamma_temporal=gamma,
                spatial_constraint=True,
                min_distance_km=resolved_min_distance,
                pre_selected=pre_selected,
                pre_selected_names=pre_selected_names,
            )

            metrics = compute_metrics(selected_idx, meta, cluster_labels, features)
            metrics.update({"alpha": alpha, "beta": beta, "gamma": gamma})

            sel_idx_arr = (
                np.asarray(selected_idx, dtype=int)
                if getattr(selected_idx, "__len__", None) is not None
                and len(selected_idx) > 0
                else np.array([], dtype=int)
            )
            diversity = _selection_diversity(sel_idx_arr)
            spread = _selection_spread(sel_idx_arr)
            objective_score = normalized_objective(
                diversity=float(diversity),
                spread=float(spread),
                baseline_diversity=float(baseline_diversity),
                baseline_spread=float(baseline_spread),
                n_selected=int(len(sel_idx_arr)),
                target_n=int(n_samples),
                weight_diversity=float(objective_weight_diversity),
                weight_spread=float(objective_weight_spread),
                infeasible_penalty=float(objective_infeasible_penalty),
            )
            metrics.update(
                {
                    "objective_authority": objective_mode,
                    "objective_score": float(objective_score.score),
                    "objective_score_raw": float(objective_score.raw_score),
                    "diversity_norm": float(objective_score.diversity_norm),
                    "spatial_spread_norm": float(objective_score.spread_norm),
                    "objective_infeasible": bool(objective_score.infeasible),
                    "objective_feasibility_ratio": float(
                        objective_score.feasibility_ratio
                    ),
                }
            )
            results.append(metrics)

            out_csv = self.output_dir / f"selection_a{alpha}_b{beta}_g{gamma}.csv"
            exported = ds.export_selection(meta, out_csv)
            # export_selection may either write the CSV itself and return None,
            # or return a DataFrame — handle both cases
            if exported is not None:
                save_selection(exported, str(out_csv))

            # Early-stopping logic
            current_score = score_fn(metrics)
            if best_score is None or current_score > best_score:
                best_score = current_score
                best_metrics = metrics
                no_improve = 0
                if objective_mode == "unified_normalized":
                    print(
                        "  [New best] objective_score={:.6f} (raw={:.6f}, feasible={})".format(
                            float(metrics["objective_score"]),
                            float(metrics["objective_score_raw"]),
                            not bool(metrics["objective_infeasible"]),
                        )
                    )
                else:
                    print(
                        f"  [New best] score={best_score} (clusters={metrics['clusters_covered']}, temporal_std={metrics['temporal_std']:.2f})"
                    )
            else:
                no_improve += 1
                print(
                    f"  [No improve] {no_improve}/{patience} trials without improvement"
                )

            if patience is not None and no_improve >= patience:
                print(
                    f"Early stopping triggered (no improvement in last {patience} trials)."
                )
                break

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "tuning_results.csv", index=False)

        # Schreibe Meta-Informationen für Caching / Reproduzierbarkeit
        try:
            import hashlib
            import json
            import subprocess
            from datetime import datetime

            def _file_hash(path: str) -> str:
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    while True:
                        chunk = fh.read(8192)
                        if not chunk:
                            break
                        h.update(chunk)
                return h.hexdigest()

            # Get git commit hash for provenance
            git_commit = None
            try:
                git_commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode()
                    .strip()
                )
            except Exception as e_git:
                print(f"Warning: could not get git commit: {e_git}")

            # Collect runtime/package versions for provenance
            python_version = None
            numpy_version = None
            torch_version = None
            try:
                import platform

                python_version = platform.python_version()
            except Exception as e_plat:
                print(f"Warning: could not get python version: {e_plat}")

            try:
                import numpy as _np

                numpy_version = _np.__version__
            except Exception as e_np:
                print(f"Warning: could not get numpy version: {e_np}")

            try:
                import torch as _torch

                torch_version = _torch.__version__
            except Exception:
                # Torch optional — set to None if unavailable
                torch_version = None

            # Collect pip packages of interest (subset) for reproducibility
            pip_packages = None
            try:
                import subprocess as _sub
                import sys as _sys

                pip_freeze = (
                    _sub.check_output([_sys.executable, "-m", "pip", "freeze"])
                    .decode()
                    .splitlines()
                )
                interesting = {
                    "torch",
                    "numpy",
                    "scikit-learn",
                    "umap-learn",
                    "optuna",
                    "pandas",
                }
                pkgs = {}
                for line in pip_freeze:
                    if "==" in line:
                        name, _, ver = line.partition("==")
                        if name.lower() in interesting:
                            pkgs[name] = ver
                pip_packages = pkgs
            except Exception as e_pip:
                print(f"Warning: could not get pip freeze: {e_pip}")

            meta = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "csv_meta": str(csv_meta),
                "csv_meta_hash": (
                    _file_hash(str(csv_meta)) if Path(str(csv_meta)).exists() else None
                ),
                "git_commit": git_commit,
                "python_version": python_version,
                "numpy_version": numpy_version,
                "torch_version": torch_version,
                "pip_packages": pip_packages,
                "n_combinations": len(combos),
                "objective_authority": objective_mode,
                "objective_weight_diversity": float(objective_weight_diversity),
                "objective_weight_spread": float(objective_weight_spread),
                "objective_infeasible_penalty": float(objective_infeasible_penalty),
                "best_metrics": best_metrics,
                "pre_selected": pre_selected,
                "pre_selected_names": pre_selected_names,
            }

            meta_path = self.output_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Wrote tuning meta: {meta_path}")
        except Exception as e:
            print(f"Warning: could not write meta info: {e}")

        print(
            f"Tuning finished. Results saved to {self.output_dir / 'tuning_results.csv'}"
        )

        if best_metrics is not None:
            if objective_mode == "unified_normalized":
                print(
                    "Best metrics: alpha={}, beta={}, gamma={}, objective_score={:.6f}, feasible={}".format(
                        best_metrics["alpha"],
                        best_metrics["beta"],
                        best_metrics["gamma"],
                        float(best_metrics["objective_score"]),
                        not bool(best_metrics["objective_infeasible"]),
                    )
                )
            else:
                print(
                    f"Best metrics: alpha={best_metrics['alpha']}, beta={best_metrics['beta']}, gamma={best_metrics['gamma']}, clusters={best_metrics['clusters_covered']}, temporal_std={best_metrics['temporal_std']:.2f}"
                )

        return df
