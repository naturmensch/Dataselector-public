"""Validation workflows for Pareto solutions and selection candidates.

Provides robust multi-seed, multi-constraint validation of Pareto-optimal
hyperparameter configurations identified during exploration/optimization phases.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd


def _resolve_preselected_indices(
    metadata: pd.DataFrame,
    *,
    pre_selected_names: list[str] | None,
    pre_selected_indices: list[int] | None,
) -> list[int]:
    """Resolve pre-selected names/indices against metadata once."""
    if pre_selected_indices:
        seen: set[int] = set()
        out: list[int] = []
        for raw in pre_selected_indices:
            idx = int(raw)
            if idx < 0 or idx >= len(metadata):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    if not pre_selected_names:
        return []

    alias_map = {"hamburg": ["KDR_146"]}
    resolved: list[int] = []
    for nm in pre_selected_names:
        text = str(nm).strip()
        if not text:
            continue
        terms = [text]
        terms.extend(alias_map.get(text.lower(), []))
        mask = pd.Series(False, index=metadata.index)
        for term in terms:
            term_lower = str(term).strip().lower()
            if not term_lower:
                continue
            if "longName" in metadata.columns:
                mask = mask | metadata["longName"].astype(str).str.lower().str.contains(
                    term_lower
                )
            if "shortName" in metadata.columns:
                mask = mask | (
                    metadata["shortName"].astype(str).str.lower() == term_lower
                )
            if "city" in metadata.columns:
                mask = mask | (metadata["city"].astype(str).str.lower() == term_lower)
        resolved.extend(int(i) for i in mask[mask].index.tolist())
    # Deduplicate while preserving order.
    return list(dict.fromkeys(int(i) for i in resolved if 0 <= int(i) < len(metadata)))


def _pairwise_jaccard_mean_min(selection_sets: list[set[int]]) -> tuple[float, float]:
    if len(selection_sets) <= 1:
        return 1.0, 1.0
    vals: list[float] = []
    for i in range(len(selection_sets)):
        for j in range(i + 1, len(selection_sets)):
            a = selection_sets[i]
            b = selection_sets[j]
            union = a | b
            if not union:
                vals.append(1.0)
            else:
                vals.append(float(len(a & b) / len(union)))
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.min())


def _summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["replicate_mode", "alpha", "beta", "gamma", "min_distance_km"]
    metric_cols = [
        "n_selected",
        "clusters_covered",
        "spatial_mean_km",
        "spatial_min_km",
        "temporal_std",
        "wwi_percent",
        "selection_shortfall",
        "hardcut_target_met",
    ]
    for key, sub in df.groupby(group_cols):
        key_map = dict(zip(group_cols, key))
        # Stability over selected index sets for this configuration
        set_list = []
        for payload in sub["selected_indices_json"].tolist():
            try:
                arr = json.loads(payload)
            except Exception:
                arr = []
            set_list.append(set(int(i) for i in arr))
        j_mean, j_min = _pairwise_jaccard_mean_min(set_list)
        rows.append(
            {
                **key_map,
                "metric": "selection_jaccard_mean",
                "count": int(len(sub)),
                "mean": float(j_mean),
                "std": 0.0,
                "median": float(j_mean),
                "ci95_lo": float(j_mean),
                "ci95_hi": float(j_mean),
            }
        )
        rows.append(
            {
                **key_map,
                "metric": "selection_jaccard_min",
                "count": int(len(sub)),
                "mean": float(j_min),
                "std": 0.0,
                "median": float(j_min),
                "ci95_lo": float(j_min),
                "ci95_hi": float(j_min),
            }
        )
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().astype(float)
            if vals.empty:
                continue
            rows.append(
                {
                    **key_map,
                    "metric": metric,
                    "count": int(vals.size),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "median": float(vals.median()),
                    "ci95_lo": float(vals.quantile(0.025)),
                    "ci95_hi": float(vals.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def validate_pareto_candidates(
    pareto_csv: str | Path,
    min_distances: List[float] = None,
    seeds: List[int] = None,
    n_samples: int | None = None,
    n_clusters: int | None = None,
    batch_size: int | None = None,
    umap_n_components: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_random_state: int | None = None,
    umap_n_jobs: int | None = None,
    output_dir: str | Path = None,
    feature_cache_dir: str | Path | None = None,
    pre_selected_names: list[str] | None = None,
    pre_selected_indices: list[int] | None = None,
    replicate_mode: str = "bootstrap_candidates",
    n_bootstrap: int = 200,
    bootstrap_sample_frac: float = 1.0,
) -> pd.DataFrame:
    """Validate Pareto-optimal candidates via min-distance sweep + replicate UQ.

    For each Pareto solution, runs selections across multiple `min_distance_km`
    values and replicate draws. Generates comprehensive validation metrics,
    selection snapshots, and summary statistics.

    Args:
        pareto_csv: Path to Pareto solutions CSV (α, β, γ columns)
        min_distances: List of min_distance_km values to test (default: [25, 35, 50])
        seeds: Seed panel for deterministic replay or bootstrap RNG initialization
        n_samples: Target sample size (resolved via explicit/config/autoscale)
        n_clusters: Clustering count for validation embeddings (default: 8)
        batch_size: Feature extraction batch size (default: 16)
        umap_n_components: UMAP component count (default: 2)
        umap_n_neighbors: UMAP neighbors (default: adaptive behavior in clustering)
        umap_random_state: UMAP random state (default: 42)
        umap_n_jobs: UMAP parallel jobs when non-deterministic (default: 1)
        output_dir: Output directory for results (default: outputs/validation/)
        feature_cache_dir: Optional shared feature-cache directory across runs
        pre_selected_names: Optional pre-selected tile names enforced in selection
        pre_selected_indices: Optional pre-selected tile indices enforced in selection
        replicate_mode: `seed_replay` or `bootstrap_candidates`
        n_bootstrap: Number of bootstrap replicates (for bootstrap mode)
        bootstrap_sample_frac: Candidate resampling fraction in bootstrap mode

    Returns:
        DataFrame with validation results for requested replicate mode

    Example:
        >>> results = validate_pareto_candidates(
        ...     pareto_csv="outputs/tuning_weights/pareto/pareto_solutions.csv",
        ...     min_distances=[25, 35, 50],
        ...     seeds=[42, 43, 44],
        ...     n_samples=34
        ... )
        >>> print(f"Tested {len(results)} configurations")
    """
    # Lazy imports (avoid heavy dependencies at module load)
    from dataselector.analysis.metrics import compute_metrics
    from dataselector.analysis.visualizer import Visualizer
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector
    from dataselector.workflows._selection_target import resolve_selection_n_samples

    # Default parameters
    if min_distances is None:
        min_distances = [25, 35, 50]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    replicate_mode = str(replicate_mode).strip().lower()
    if replicate_mode not in {"seed_replay", "bootstrap_candidates"}:
        raise ValueError(
            "replicate_mode must be one of {'seed_replay', 'bootstrap_candidates'}"
        )
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if float(bootstrap_sample_frac) <= 0.0:
        raise ValueError("bootstrap_sample_frac must be > 0")
    if n_clusters is None:
        n_clusters = 8
    if batch_size is None:
        batch_size = 16
    if umap_n_components is None:
        umap_n_components = 2
    if umap_random_state is None:
        umap_random_state = 42
    if umap_n_jobs is None:
        umap_n_jobs = 1

    # Setup paths
    pareto_csv = Path(pareto_csv)
    if not pareto_csv.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {pareto_csv}")

    root = Path.cwd()
    if output_dir is None:
        output_dir = root / "outputs" / "validation"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(feature_cache_dir) if feature_cache_dir is not None else output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load Pareto solutions
    pareto = pd.read_csv(pareto_csv)
    if not {"alpha", "beta", "gamma"}.issubset(pareto.columns):
        raise ValueError(
            f"Pareto CSV must contain alpha, beta, gamma columns. Found: {pareto.columns.tolist()}"
        )

    # Load data once (reuse across all validation runs)
    metadata_path = assert_canonical_metadata(
        None,
        context="validation",
        root=root,
    )
    metadata = load_metadata(str(metadata_path))
    resolved_n_samples, n_samples_source = resolve_selection_n_samples(
        n_samples,
        context="validation.validate_pareto_candidates",
        root=root,
        experiment_run_dir=output_dir,
    )
    print(
        "Validation selection target: {} samples ({})".format(
            resolved_n_samples,
            n_samples_source,
        )
    )

    features = load_or_extract_features(
        cache_dir,
        csv_meta=str(metadata_path),
        batch_size=int(batch_size),
        cache=True,
        enforce_canonical=True,
    )

    # Compute embeddings and cluster labels (consistent with main pipeline)
    clustering = ClusteringPipeline(
        n_clusters=int(n_clusters),
        umap_n_components=int(umap_n_components),
        random_state=int(seeds[0]) if len(seeds) > 0 else 42,
        umap_random_state=int(umap_random_state),
        umap_n_jobs=int(umap_n_jobs),
        umap_n_neighbors=(
            int(umap_n_neighbors) if umap_n_neighbors is not None else None
        ),
    )

    try:
        embeddings_2d, cluster_labels = clustering.fit_transform(features)
    except Exception as e:
        # Fallback for extremely small datasets (test mode)
        print(f"Warning: UMAP/KMeans failed ({e}), using fallback embeddings/labels")
        n = features.shape[0]
        embeddings_2d = np.zeros((n, 2))
        cluster_labels = np.zeros(n, dtype=int)

    # Setup visualizer
    viz = Visualizer(output_dir=str(output_dir / "plots"))

    preselected_global = _resolve_preselected_indices(
        metadata,
        pre_selected_names=pre_selected_names,
        pre_selected_indices=pre_selected_indices,
    )
    if preselected_global:
        print(f"Resolved global pre-selected indices: {preselected_global}")

    def _save_selection_snapshot(
        *,
        selected_orig_idx: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        min_d: float,
        replicate_suffix: str,
        replicate_seed: int,
        mode: str,
    ) -> None:
        sel_df = metadata.iloc[selected_orig_idx].copy()
        sel_df["selection_rank"] = range(len(sel_df))
        sel_file = (
            output_dir
            / f"selection_a{alpha}_b{beta}_g{gamma}_d{min_d}_{replicate_suffix}.csv"
        )
        sel_df.to_csv(sel_file, index=False)

        # Bootstrap mode intentionally skips per-replicate plots (can be hundreds).
        if mode != "seed_replay":
            return
        prefix = f"sel_a{alpha}_b{beta}_g{gamma}_d{min_d}_{replicate_suffix}"
        try:
            viz.create_summary_report(
                embeddings_2d=embeddings_2d,
                cluster_labels=cluster_labels,
                metadata=metadata,
                selected_indices=selected_orig_idx,
                output_prefix=prefix,
            )
        except Exception as e:
            print(f"Warning: could not create plots for {prefix}: {e}")

    def _run_one_selection(
        *,
        alpha: float,
        beta: float,
        gamma: float,
        min_d: float,
        replicate_seed: int,
        mode: str,
        replicate_id: int,
        sampled_indices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if sampled_indices is None:
            local_features = features
            local_metadata = metadata
            local_preselected = preselected_global
        else:
            local_features = features[sampled_indices]
            local_metadata = metadata.iloc[sampled_indices].reset_index(drop=True)
            local_preselected = []
            if preselected_global:
                for global_idx in preselected_global:
                    hits = np.where(sampled_indices == int(global_idx))[0]
                    if hits.size > 0:
                        local_preselected.append(int(hits[0]))
                local_preselected = list(dict.fromkeys(local_preselected))

        selector = DiversitySelector(
            n_samples=resolved_n_samples,
            use_multi_criteria=True,
            random_state=int(replicate_seed),
        )
        selected_local = selector.select(
            features=local_features,
            metadata=local_metadata,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
            pre_selected=local_preselected if local_preselected else None,
            pre_selected_names=None,
        )
        selected_local_idx = np.asarray(selected_local, dtype=int)
        if sampled_indices is None:
            selected_orig_idx = selected_local_idx
        else:
            selected_orig_idx = sampled_indices[selected_local_idx]
            # Keep deterministic order while deduplicating duplicates from bootstrap sampling.
            selected_orig_idx = np.asarray(
                list(
                    dict.fromkeys(
                        int(i)
                        for i in selected_orig_idx.tolist()
                        if 0 <= int(i) < len(metadata)
                    )
                ),
                dtype=int,
            )
        selected_count = len(selected_orig_idx)
        shortfall = max(0, int(resolved_n_samples) - selected_count)
        if shortfall > 0:
            print(
                "⚠️  Validation hard-cut shortfall: "
                f"requested n_samples={resolved_n_samples}, selected={selected_count} "
                f"(alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, "
                f"min_distance_km={min_d}, replicate_seed={replicate_seed}, mode={mode})"
            )

        metrics = compute_metrics(selected_orig_idx, metadata, cluster_labels, features)
        metrics.update(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "min_distance_km": min_d,
                "seed": int(replicate_seed),
                "replicate_id": int(replicate_id),
                "replicate_mode": mode,
                "pre_selected_names": pre_selected_names,
                "pre_selected_indices": pre_selected_indices,
                "requested_n_samples": int(resolved_n_samples),
                "n_samples_source": n_samples_source,
                "selection_shortfall": shortfall,
                "hardcut_target_met": shortfall == 0,
                "bootstrap_sample_frac": float(bootstrap_sample_frac),
                "n_bootstrap": int(n_bootstrap),
                "selected_indices_json": json.dumps(
                    [int(i) for i in selected_orig_idx.tolist()]
                ),
            }
        )

        suffix = (
            f"s{int(replicate_seed)}" if mode == "seed_replay" else f"b{replicate_id}"
        )
        _save_selection_snapshot(
            selected_orig_idx=selected_orig_idx,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_d=min_d,
            replicate_suffix=suffix,
            replicate_seed=int(replicate_seed),
            mode=mode,
        )
        return metrics

    # Run validation sweep
    rows = []
    run_i = 0
    if replicate_mode == "seed_replay":
        total = len(pareto) * len(min_distances) * len(seeds)
        for _, row in pareto.iterrows():
            alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
            for min_d in min_distances:
                for seed in seeds:
                    run_i += 1
                    print(
                        f"Run {run_i}/{total}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, "
                        f"min_dist={min_d}km, seed={seed}, mode=seed_replay"
                    )
                    t0 = time.time()
                    metrics = _run_one_selection(
                        alpha=float(alpha),
                        beta=float(beta),
                        gamma=float(gamma),
                        min_d=float(min_d),
                        replicate_seed=int(seed),
                        mode="seed_replay",
                        replicate_id=int(seed),
                        sampled_indices=None,
                    )
                    metrics["duration_s"] = float(time.time() - t0)
                    rows.append(metrics)
    else:
        rng = np.random.default_rng(int(seeds[0]) if len(seeds) > 0 else 42)
        n_candidates = int(features.shape[0])
        sample_size = max(
            1,
            min(n_candidates, int(round(float(bootstrap_sample_frac) * n_candidates))),
        )
        total = len(pareto) * len(min_distances) * int(n_bootstrap)
        for _, row in pareto.iterrows():
            alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
            for min_d in min_distances:
                for rep in range(int(n_bootstrap)):
                    run_i += 1
                    replicate_seed = int(rng.integers(0, 2**31 - 1))
                    sampled_indices = rng.choice(
                        n_candidates,
                        size=sample_size,
                        replace=True,
                    ).astype(int)
                    print(
                        f"Run {run_i}/{total}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, "
                        f"min_dist={min_d}km, bootstrap_rep={rep}, mode=bootstrap_candidates"
                    )
                    t0 = time.time()
                    metrics = _run_one_selection(
                        alpha=float(alpha),
                        beta=float(beta),
                        gamma=float(gamma),
                        min_d=float(min_d),
                        replicate_seed=replicate_seed,
                        mode="bootstrap_candidates",
                        replicate_id=int(rep),
                        sampled_indices=sampled_indices,
                    )
                    metrics["duration_s"] = float(time.time() - t0)
                    rows.append(metrics)

    # Save validation results
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Validation did not produce any rows.")

    seed_replay_df = df[df["replicate_mode"] == "seed_replay"].copy()
    bootstrap_df = df[df["replicate_mode"] == "bootstrap_candidates"].copy()
    if not seed_replay_df.empty:
        seed_replay_df.to_csv(
            output_dir / "validation_results_seed_replay.csv", index=False
        )
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(
            output_dir / "validation_results_bootstrap.csv", index=False
        )

    # Backward-compatible canonical path: points to the requested primary mode.
    primary_df = (
        bootstrap_df if replicate_mode == "bootstrap_candidates" else seed_replay_df
    )
    if primary_df.empty:
        primary_df = df
    results_path = output_dir / "validation_results.csv"
    primary_df.to_csv(results_path, index=False)

    summary_df = _summary_stats(df)
    summary_path = output_dir / "validation_summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)

    method_contract_path = output_dir / "validation_method_contract.md"
    with method_contract_path.open("w", encoding="utf-8") as f:
        f.write("# Validation Method Contract\n\n")
        f.write(f"- Requested replicate mode: `{replicate_mode}`\n")
        f.write(
            "- Inference mode policy: `bootstrap_candidates` is inferential; "
            "`seed_replay` is deterministic replay/stability only.\n"
        )
        f.write(f"- Seeds panel (input): `{[int(s) for s in seeds]}`\n")
        f.write(f"- n_bootstrap: `{int(n_bootstrap)}`\n")
        f.write(f"- bootstrap_sample_frac: `{float(bootstrap_sample_frac)}`\n")
        f.write("\n## Outputs\n\n")
        f.write("- `validation_results.csv` (primary mode)\n")
        f.write("- `validation_results_bootstrap.csv` (if bootstrap rows exist)\n")
        f.write("- `validation_results_seed_replay.csv` (if seed replay rows exist)\n")
        f.write("- `validation_summary_stats.csv`\n")

    print(f"\n✓ Validation finished. Results: {results_path}")
    print(f"  - {len(pareto)} Pareto candidates")
    print(f"  - {len(min_distances)} min_distance values: {min_distances}")
    print(f"  - replicate_mode: {replicate_mode}")
    print(f"  - seeds (input panel): {seeds}")
    print(f"  - n_bootstrap: {int(n_bootstrap)}")
    print(f"  - bootstrap_sample_frac: {float(bootstrap_sample_frac)}")
    print(f"  - {total} total configurations validated")
    print(f"  - summary: {summary_path}")
    print(f"  - method contract: {method_contract_path}")

    return primary_df.reset_index(drop=True)
