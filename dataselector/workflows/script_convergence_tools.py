"""CLI-first convergence tools migrated from legacy top-level scripts.

This module hosts scientific/analysis logic that used to live in scripts/* so that
scripts can remain thin compatibility wrappers.
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataselector.cli_decorators import cli_command


@dataclass
class MinDistanceRunResult:
    distance: float
    seed: int
    n_requested: int
    n_selected: int
    shortfall: int
    hardcut_target_met: bool
    clusters_covered: int
    temporal_std: float
    temporal_range: float
    spatial_mean_km: float
    spatial_min_km: float
    wwi_percent: float
    selected_indices: list[int]


def _pairwise_jaccard(sets: list[set[int]]) -> tuple[float, float]:
    import numpy as np

    if len(sets) <= 1:
        return 1.0, 1.0

    vals: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                vals.append(1.0)
                continue
            vals.append(len(a & b) / len(union))

    return float(np.mean(vals)), float(np.min(vals))


def _resolve_min_distance_defaults(config_path: Path) -> dict[str, float | int]:
    import yaml

    defaults: dict[str, float | int] = {
        "n_samples": 34,
        "n_clusters": 8,
        "alpha": 0.40,
        "beta": 0.30,
        "gamma": 0.30,
    }

    if not config_path.exists():
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        return defaults

    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    clu = (
        cfg.get("clustering", {}) if isinstance(cfg.get("clustering", {}), dict) else {}
    )

    n_samples = sel.get("n_samples")
    if isinstance(n_samples, int) and n_samples > 0:
        defaults["n_samples"] = int(n_samples)

    defaults["n_clusters"] = int(clu.get("n_clusters", defaults["n_clusters"]))
    defaults["alpha"] = float(sel.get("alpha_visual", defaults["alpha"]))
    defaults["beta"] = float(sel.get("beta_spatial", defaults["beta"]))
    defaults["gamma"] = float(sel.get("gamma_temporal", defaults["gamma"]))
    return defaults


def _run_single_min_distance(
    *,
    features: Any,
    metadata: Any,
    distance: float,
    seed: int,
    n_samples: int,
    n_clusters: int,
    alpha: float,
    beta: float,
    gamma: float,
    pre_selected_names: list[str] | None,
    pre_selected_indices: list[int] | None,
) -> MinDistanceRunResult:
    import numpy as np
    from sklearn.cluster import KMeans

    from dataselector.analysis.metrics import compute_metrics
    from dataselector.selection.diversity_selector import DiversitySelector

    k = max(2, min(int(n_clusters), int(features.shape[0])))
    kmeans = KMeans(n_clusters=k, random_state=int(seed), n_init=10)
    cluster_labels = kmeans.fit_predict(features)

    selector = DiversitySelector(
        n_samples=int(n_samples),
        use_multi_criteria=True,
        random_state=int(seed),
    )
    selected = selector.select(
        features=features,
        metadata=metadata,
        alpha_visual=float(alpha),
        beta_spatial=float(beta),
        gamma_temporal=float(gamma),
        spatial_constraint=True,
        min_distance_km=float(distance),
        pre_selected_names=pre_selected_names,
        pre_selected=pre_selected_indices,
    )

    metrics = compute_metrics(selected, metadata, cluster_labels, features)
    selected_list = np.asarray(selected, dtype=int).tolist()
    n_selected = len(selected_list)
    shortfall = max(0, int(n_samples) - n_selected)

    return MinDistanceRunResult(
        distance=float(distance),
        seed=int(seed),
        n_requested=int(n_samples),
        n_selected=n_selected,
        shortfall=shortfall,
        hardcut_target_met=shortfall == 0,
        clusters_covered=int(metrics.get("clusters_covered", 0)),
        temporal_std=float(metrics.get("temporal_std", 0.0)),
        temporal_range=float(metrics.get("temporal_range", 0.0)),
        spatial_mean_km=float(metrics.get("spatial_mean_km", 0.0)),
        spatial_min_km=float(metrics.get("spatial_min_km", 0.0)),
        wwi_percent=float(metrics.get("wwi_percent", 0.0)),
        selected_indices=selected_list,
    )


def _summarize_min_distance(df: Any) -> Any:
    import pandas as pd

    rows: list[dict[str, float | int | str]] = []

    for distance, sub in df.groupby("distance"):
        sets = [set(json.loads(x)) for x in sub["selected_indices_json"].tolist()]
        mean_j, min_j = _pairwise_jaccard(sets)

        rows.append(
            {
                "distance": float(distance),
                "runs": int(len(sub)),
                "target_met_rate": float(sub["hardcut_target_met"].mean()),
                "mean_n_selected": float(sub["n_selected"].mean()),
                "std_n_selected": float(sub["n_selected"].std(ddof=0)),
                "mean_shortfall": float(sub["shortfall"].mean()),
                "shortfall_rate": float((sub["shortfall"] > 0).mean()),
                "mean_clusters_covered": float(sub["clusters_covered"].mean()),
                "mean_temporal_std": float(sub["temporal_std"].mean()),
                "mean_wwi_percent": float(sub["wwi_percent"].mean()),
                "mean_spatial_mean_km": float(sub["spatial_mean_km"].mean()),
                "mean_spatial_min_km": float(sub["spatial_min_km"].mean()),
                "stability_jaccard_mean": mean_j,
                "stability_jaccard_min": min_j,
            }
        )

    out = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    out["decision_score"] = (
        out["target_met_rate"] * 100.0
        + out["stability_jaccard_mean"] * 10.0
        + out["mean_clusters_covered"] * 0.5
    )

    viable = out[out["shortfall_rate"] < 0.10].copy()
    pool = viable if len(viable) > 0 else out
    pool = pool.sort_values(
        [
            "target_met_rate",
            "stability_jaccard_mean",
            "mean_clusters_covered",
            "decision_score",
        ],
        ascending=[False, False, False, False],
    )

    best = pool.iloc[0]
    tie_mask = (
        (pool["target_met_rate"] >= float(best["target_met_rate"]) - 0.01)
        & (
            pool["stability_jaccard_mean"]
            >= float(best["stability_jaccard_mean"]) - 0.02
        )
        & (pool["mean_clusters_covered"] >= float(best["mean_clusters_covered"]) - 0.2)
        & (pool["decision_score"] >= float(best["decision_score"]) - 0.05)
    )
    tied = pool[tie_mask].copy()
    recommended = float(
        tied.sort_values("distance", ascending=True).iloc[0]["distance"]
    )

    rationale = (
        "feasibility/stability/coverage rule with near-tie preference for smaller "
        "distance (higher downstream combination flexibility)"
    )
    out["recommended_distance"] = recommended
    out["recommendation_rationale"] = rationale
    return out


def run_compare_min_distance_policies(
    *,
    metadata_path: str,
    distances: list[float],
    seeds: list[int],
    n_samples: int | None,
    config_path: str,
    output_dir: str,
    pre_names: list[str] | None,
    pre_indices: list[int] | None,
) -> dict[str, str | float]:
    import pandas as pd

    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata

    canonical_path = assert_canonical_metadata(
        metadata_path,
        context="compare_min_distance_policies",
        root=Path.cwd(),
    )

    defaults = _resolve_min_distance_defaults(Path(config_path))
    resolved_n_samples = (
        int(n_samples) if n_samples is not None else int(defaults["n_samples"])
    )
    n_clusters = int(defaults["n_clusters"])
    alpha = float(defaults["alpha"])
    beta = float(defaults["beta"])
    gamma = float(defaults["gamma"])

    metadata = load_metadata(str(canonical_path))
    features = load_or_extract_features(
        out_dir="outputs/runs",
        csv_meta=str(canonical_path),
        batch_size=16,
        cache=True,
        enforce_canonical=True,
    )

    records: list[dict[str, object]] = []
    for distance in distances:
        for seed in seeds:
            rr = _run_single_min_distance(
                features=features,
                metadata=metadata,
                distance=float(distance),
                seed=int(seed),
                n_samples=resolved_n_samples,
                n_clusters=n_clusters,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                pre_selected_names=pre_names,
                pre_selected_indices=pre_indices,
            )
            records.append(
                {
                    "distance": rr.distance,
                    "seed": rr.seed,
                    "n_requested": rr.n_requested,
                    "n_selected": rr.n_selected,
                    "shortfall": rr.shortfall,
                    "hardcut_target_met": rr.hardcut_target_met,
                    "clusters_covered": rr.clusters_covered,
                    "temporal_std": rr.temporal_std,
                    "temporal_range": rr.temporal_range,
                    "spatial_mean_km": rr.spatial_mean_km,
                    "spatial_min_km": rr.spatial_min_km,
                    "wwi_percent": rr.wwi_percent,
                    "selected_indices_json": json.dumps(rr.selected_indices),
                }
            )

    raw_df = pd.DataFrame(records)
    summary_df = _summarize_min_distance(raw_df)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_csv = out_dir / f"min_distance_policy_runs_{ts}.csv"
    summary_csv = out_dir / f"min_distance_policy_summary_{ts}.csv"
    summary_md = out_dir / f"min_distance_policy_summary_{ts}.md"

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    recommended = float(summary_df["recommended_distance"].iloc[0])
    rationale = str(summary_df["recommendation_rationale"].iloc[0])

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Min Distance Policy Comparison\n\n")
        f.write(f"- UTC timestamp: `{ts}`\n")
        f.write(f"- Metadata: `{canonical_path}`\n")
        f.write(f"- Rows: `{len(metadata)}`\n")
        f.write(f"- n_samples: `{resolved_n_samples}`\n")
        f.write(f"- seeds: `{seeds}`\n")
        f.write(f"- distances: `{distances}`\n")
        f.write(f"- weights: alpha={alpha}, beta={beta}, gamma={gamma}\n")
        if pre_names:
            f.write(f"- pre_selected_names: `{pre_names}`\n")
        if pre_indices:
            f.write(f"- pre_selected_indices: `{pre_indices}`\n")
        f.write("\n## Summary\n\n")
        try:
            f.write(summary_df.to_markdown(index=False))
        except Exception:
            f.write("```text\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n```\n")
        f.write("\n\n## Recommendation\n\n")
        f.write(f"- Recommended `min_distance_km`: **{recommended}**\n")
        f.write(f"- Rationale: {rationale}\n")
        f.write(
            "- Rule applied: prefer low-shortfall and stable candidates; in near-ties choose smaller distance.\n"
        )

    return {
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "recommended_min_distance_km": recommended,
    }


def run_compare_seed_vs_unseeded(
    *,
    config_path: str | None,
    output_dir: str | None,
    seeds: list[int] | None = None,
    n_samples: int | None = None,
    alpha_visual: float | None = None,
    beta_spatial: float | None = None,
    gamma_temporal: float | None = None,
    min_distance_km: float | None = None,
    report_label: str | None = None,
    run_production_quick: bool = False,
    production_seeded_run: str | None = None,
    production_unseeded_run: str | None = None,
) -> dict[str, str]:
    from dataselector.workflows.compare_samplers import (
        compare_production_runs_quick_delta,
        compare_seeded_vs_unseeded,
    )

    default_seeded = "outputs/runs/thesis_orchestrate_full_20260213T151106Z_B"
    default_unseeded = "outputs/runs/thesis_orchestrate_full_20260213T141421Z_B"

    if output_dir:
        root_out = Path(output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root_out = Path("outputs/runs") / f"anchor_evidence_{ts}"
    root_out.mkdir(parents=True, exist_ok=True)

    isolated_out = root_out / "isolated"
    isolated = compare_seeded_vs_unseeded(
        config_path=Path(config_path) if config_path else None,
        output_dir=isolated_out,
        seeds=seeds,
        n_samples=n_samples,
        alpha_visual=alpha_visual,
        beta_spatial=beta_spatial,
        gamma_temporal=gamma_temporal,
        min_distance_km=min_distance_km,
        report_label=report_label,
    )
    result: dict[str, str] = {
        "output_dir": str(root_out),
        "isolated_output_dir": str(isolated),
    }

    if run_production_quick:
        seeded_dir = Path(production_seeded_run or default_seeded)
        unseeded_dir = Path(production_unseeded_run or default_unseeded)
        production_out = root_out / "production_quick"
        compare_production_runs_quick_delta(
            seeded_run_dir=seeded_dir,
            unseeded_run_dir=unseeded_dir,
            output_dir=production_out,
        )
        result["production_quick_output_dir"] = str(production_out)
        result["production_seeded_run"] = str(seeded_dir)
        result["production_unseeded_run"] = str(unseeded_dir)

    return result


def run_seed_benchmark(
    *, seeds: list[int] | None, output_dir: str | None, subset_n: int
) -> dict[str, str]:
    from dataselector.workflows.compare_samplers import benchmark_seed

    out_csv = benchmark_seed(
        seeds=seeds,
        output_dir=Path(output_dir) if output_dir else None,
        subset_n=int(subset_n),
    )
    return {"results_csv": str(out_csv)}


def _load_or_create_profile_data(out_dir: Path, n: int, dim: int) -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    from dataselector.data.io import load_or_extract_features

    features_path = out_dir / "features.npy"
    metadata_path = out_dir / "metadata.csv"

    if features_path.exists() and metadata_path.exists():
        features = load_or_extract_features(
            out_dir=out_dir,
            csv_meta=str(metadata_path),
            batch_size=16,
            cache=False,
        )
        metadata = pd.read_csv(metadata_path)
    else:
        rng = np.random.RandomState(123)
        features = rng.randn(n, dim).astype("float32")
        metadata = pd.DataFrame(
            {
                "N": np.random.uniform(48, 55, n),
                "left": np.random.uniform(6, 15, n),
                "year": np.random.randint(1880, 1945, n),
            }
        )

    return features, metadata


def _profile_mode(
    *,
    mode_name: str,
    selector: Any,
    features: Any,
    metadata: Any,
    out_dir: Path,
    select_kwargs: dict[str, Any],
) -> float:
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    selector.select(features, metadata, **select_kwargs)
    prof.disable()
    elapsed = time.time() - t0

    prof_file = out_dir / f"profile_{mode_name}.prof"
    txt_file = out_dir / f"profile_{mode_name}.txt"

    with txt_file.open("w", encoding="utf-8") as handle:
        pstats.Stats(prof, stream=handle).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(50)
    prof.dump_stats(str(prof_file))
    return elapsed


def run_profile_selection(
    *,
    output_dir: str,
    n_samples: int,
    min_distance_km: float,
    synthetic_n: int,
    synthetic_dim: int,
) -> dict[str, str]:
    import pandas as pd

    from dataselector.selection.diversity_selector import DiversitySelector

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features, metadata = _load_or_create_profile_data(
        out_dir, synthetic_n, synthetic_dim
    )

    results: dict[str, float] = {}

    selector_legacy = DiversitySelector(n_samples=n_samples, use_multi_criteria=False)
    selector_legacy.select(
        features, metadata, spatial_constraint=True, min_distance_km=min_distance_km
    )
    results["legacy"] = _profile_mode(
        mode_name="legacy",
        selector=selector_legacy,
        features=features,
        metadata=metadata,
        out_dir=out_dir,
        select_kwargs={
            "spatial_constraint": True,
            "min_distance_km": min_distance_km,
        },
    )

    selector_constraint = DiversitySelector(
        n_samples=n_samples,
        use_multi_criteria=False,
        use_constraint_integration=True,
    )
    selector_constraint.select(
        features, metadata, spatial_constraint=True, min_distance_km=min_distance_km
    )
    results["constraint_integrated"] = _profile_mode(
        mode_name="constraint_integrated",
        selector=selector_constraint,
        features=features,
        metadata=metadata,
        out_dir=out_dir,
        select_kwargs={
            "spatial_constraint": True,
            "min_distance_km": min_distance_km,
        },
    )

    if "year" in metadata.columns:
        selector_multi = DiversitySelector(n_samples=n_samples, use_multi_criteria=True)
        selector_multi.select(
            features,
            metadata,
            alpha_visual=0.7,
            beta_spatial=0.15,
            gamma_temporal=0.15,
        )
        results["multi_criteria"] = _profile_mode(
            mode_name="multi_criteria",
            selector=selector_multi,
            features=features,
            metadata=metadata,
            out_dir=out_dir,
            select_kwargs={
                "alpha_visual": 0.7,
                "beta_spatial": 0.15,
                "gamma_temporal": 0.15,
            },
        )

    summary = out_dir / "profile_summary.csv"
    pd.DataFrame.from_dict(results, orient="index", columns=["time_s"]).to_csv(summary)
    return {"summary_csv": str(summary)}


def run_temporal_sensitivity_test(
    *,
    output_dir: str,
    csv_meta: str,
    batch_size: int,
    n_clusters: int,
    n_samples: int,
    min_distance_km: float,
) -> dict[str, str]:
    import numpy as np
    import pandas as pd

    from dataselector.data.io import (
        get_metric_gdf,
        load_metadata,
        load_or_extract_features,
    )
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector
    from dataselector.selection.spatial_facility_location import haversine_distance

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = load_or_extract_features(
        out_dir=out_dir,
        csv_meta=csv_meta,
        batch_size=int(batch_size),
        cache=True,
        enforce_canonical=False,
    )
    metadata = load_metadata(csv_meta)

    temporal_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    all_rows: list[dict[str, Any]] = []

    for name, subset_n in [("Subset N=50", 50), ("Full", None)]:
        if subset_n:
            feat = features[:subset_n]
            meta = metadata.iloc[:subset_n].reset_index(drop=True)
            if getattr(metadata, "gdf_metric", None) is not None:
                meta.gdf_metric = metadata.gdf_metric.iloc[:subset_n].reset_index(
                    drop=True
                )
        else:
            feat = features
            meta = metadata

        cl = ClusteringPipeline(n_clusters=n_clusters)
        _emb, _labels = cl.fit_transform(feat)

        for tw in temporal_weights:
            selector = DiversitySelector(
                n_samples=n_samples,
                use_constraint_integration=True,
            )
            selected = selector.select(
                feat,
                meta,
                temporal_weight=tw,
                spatial_constraint=True,
                min_distance_km=min_distance_km,
            )

            years = meta.iloc[selected]["year"].dropna().values
            temporal_std = float(np.std(years)) if len(years) > 0 else float("nan")
            temporal_range = (
                float(np.max(years) - np.min(years)) if len(years) > 0 else float("nan")
            )

            use_metric = get_metric_gdf(meta) is not None
            pairwise: list[float] = []
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if use_metric:
                        gdf = get_metric_gdf(meta)
                        a = gdf.loc[selected[i], ["_proj_x", "_proj_y"]].values.astype(
                            float
                        )
                        b = gdf.loc[selected[j], ["_proj_x", "_proj_y"]].values.astype(
                            float
                        )
                        pairwise.append(float((((a - b) ** 2).sum()) ** 0.5 / 1000.0))
                    else:
                        r1 = meta.iloc[selected[i]]
                        r2 = meta.iloc[selected[j]]
                        pairwise.append(
                            haversine_distance(r1["N"], r1["left"], r2["N"], r2["left"])
                        )
            mean_pairwise = float(np.mean(pairwise)) if pairwise else float("nan")

            all_rows.append(
                {
                    "dataset": name,
                    "temporal_weight": tw,
                    "n_selected": len(selected),
                    "temporal_std": temporal_std,
                    "temporal_range": temporal_range,
                    "mean_pairwise_km": mean_pairwise,
                }
            )

    df = pd.DataFrame(all_rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = out_dir / f"temporal_sensitivity_{ts}.csv"
    out_md = out_dir / f"temporal_sensitivity_{ts}.md"
    df.to_csv(out_csv, index=False)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Temporal Sensitivity Test\n\n")
        f.write(f"- Metadata: `{csv_meta}`\n")
        f.write(f"- n_samples: `{n_samples}`\n")
        f.write(f"- min_distance_km: `{min_distance_km}`\n")
        f.write("\n## Results\n\n")
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write("```text\n")
            f.write(df.to_string(index=False))
            f.write("\n```\n")

    return {
        "results_csv": str(out_csv),
        "results_md": str(out_md),
    }


@cli_command(
    "compare-min-distance-policies",
    help="Compare candidate min_distance_km policy values on canonical metadata",
    args={
        "metadata_path": {
            "type": str,
            "default": "data/new_all_tiles.csv",
            "help": "Canonical metadata CSV",
        },
        "distances": {
            "nargs": "+",
            "type": float,
            "default": [28.5, 40.0, 45.0],
            "help": "Candidate min_distance_km values",
        },
        "seeds": {
            "nargs": "+",
            "type": int,
            "default": [42, 43, 44, 45, 46],
            "help": "Seeds for replicated runs",
        },
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Override n_samples (default from config)",
        },
        "config_path": {
            "type": str,
            "default": "config/pipeline_config.yaml",
            "help": "Config path for defaults",
        },
        "output_dir": {
            "type": str,
            "default": "reports_2026-02-09",
            "help": "Directory for result CSV/MD artifacts",
        },
        "pre_names": {
            "nargs": "*",
            "type": str,
            "default": None,
            "help": "Optional pre-selected names",
        },
        "pre_indices": {
            "nargs": "*",
            "type": int,
            "default": None,
            "help": "Optional pre-selected indices",
        },
    },
)
def cli_compare_min_distance_policies(
    metadata_path: str = "data/new_all_tiles.csv",
    distances: list[float] | None = None,
    seeds: list[int] | None = None,
    n_samples: int | None = None,
    config_path: str = "config/pipeline_config.yaml",
    output_dir: str = "reports_2026-02-09",
    pre_names: list[str] | None = None,
    pre_indices: list[int] | None = None,
) -> int:
    result = run_compare_min_distance_policies(
        metadata_path=metadata_path,
        distances=distances or [28.5, 40.0, 45.0],
        seeds=seeds or [42, 43, 44, 45, 46],
        n_samples=n_samples,
        config_path=config_path,
        output_dir=output_dir,
        pre_names=pre_names,
        pre_indices=pre_indices,
    )
    print(json.dumps(result, indent=2))
    return 0


@cli_command(
    "compare-seed-vs-unseeded",
    help="Compare baseline vs Hamburg-seeded selection evidence",
    args={
        "config_path": {
            "type": str,
            "default": None,
            "help": "Optional pipeline config path",
        },
        "output_dir": {
            "type": str,
            "default": None,
            "help": "Optional output directory",
        },
        "seeds": {
            "nargs": "+",
            "type": int,
            "default": None,
            "help": "Optional seed panel (default: 42 43 44 45 46)",
        },
        "n_samples": {
            "type": int,
            "default": None,
            "help": "Override fixed n_samples for isolated comparison",
        },
        "alpha_visual": {
            "type": float,
            "default": None,
            "help": "Override alpha_visual for isolated comparison",
        },
        "beta_spatial": {
            "type": float,
            "default": None,
            "help": "Override beta_spatial for isolated comparison",
        },
        "gamma_temporal": {
            "type": float,
            "default": None,
            "help": "Override gamma_temporal for isolated comparison",
        },
        "min_distance_km": {
            "type": float,
            "default": None,
            "help": "Override min_distance_km for isolated comparison",
        },
        "report_label": {
            "type": str,
            "default": None,
            "help": "Optional report title for isolated markdown report",
        },
        "run_production_quick": {
            "type": bool,
            "action": "store_true",
            "help": "Also build production quick-delta from two existing run dirs",
        },
        "production_seeded_run": {
            "type": str,
            "default": "outputs/runs/thesis_orchestrate_full_20260213T151106Z_B",
            "help": "Seeded full-run directory for production quick-delta",
        },
        "production_unseeded_run": {
            "type": str,
            "default": "outputs/runs/thesis_orchestrate_full_20260213T141421Z_B",
            "help": "Unseeded full-run directory for production quick-delta",
        },
    },
)
def cli_compare_seed_vs_unseeded(
    config_path: str | None = None,
    output_dir: str | None = None,
    seeds: list[int] | None = None,
    n_samples: int | None = None,
    alpha_visual: float | None = None,
    beta_spatial: float | None = None,
    gamma_temporal: float | None = None,
    min_distance_km: float | None = None,
    report_label: str | None = None,
    run_production_quick: bool = False,
    production_seeded_run: (
        str | None
    ) = "outputs/runs/thesis_orchestrate_full_20260213T151106Z_B",
    production_unseeded_run: (
        str | None
    ) = "outputs/runs/thesis_orchestrate_full_20260213T141421Z_B",
) -> int:
    result = run_compare_seed_vs_unseeded(
        config_path=config_path,
        output_dir=output_dir,
        seeds=seeds,
        n_samples=n_samples,
        alpha_visual=alpha_visual,
        beta_spatial=beta_spatial,
        gamma_temporal=gamma_temporal,
        min_distance_km=min_distance_km,
        report_label=report_label,
        run_production_quick=run_production_quick,
        production_seeded_run=production_seeded_run,
        production_unseeded_run=production_unseeded_run,
    )
    print(json.dumps(result, indent=2))
    return 0


@cli_command(
    "seed-benchmark",
    help="Benchmark seeded deterministic mode against baseline",
    args={
        "seeds": {
            "nargs": "+",
            "type": int,
            "default": None,
            "help": "Optional list of seeds",
        },
        "output_dir": {
            "type": str,
            "default": None,
            "help": "Optional output directory",
        },
        "subset_n": {
            "type": int,
            "default": 200,
            "help": "Feature subset size used for timing benchmark",
        },
    },
)
def cli_seed_benchmark(
    seeds: list[int] | None = None,
    output_dir: str | None = None,
    subset_n: int = 200,
) -> int:
    result = run_seed_benchmark(seeds=seeds, output_dir=output_dir, subset_n=subset_n)
    print(json.dumps(result, indent=2))
    return 0


@cli_command(
    "profile-selection",
    help="Profile selection modes and emit .prof/.txt artifacts",
    args={
        "output_dir": {
            "type": str,
            "default": "outputs/runs",
            "help": "Output directory for profiling artifacts",
        },
        "n_samples": {
            "type": int,
            "default": 34,
            "help": "Sample size for profiling selectors",
        },
        "min_distance_km": {
            "type": float,
            "default": 50.0,
            "help": "Spatial minimum distance",
        },
        "synthetic_n": {
            "type": int,
            "default": 2000,
            "help": "Synthetic dataset size when no cached data exists",
        },
        "synthetic_dim": {
            "type": int,
            "default": 512,
            "help": "Synthetic feature dimension",
        },
    },
)
def cli_profile_selection(
    output_dir: str = "outputs/runs",
    n_samples: int = 34,
    min_distance_km: float = 50.0,
    synthetic_n: int = 2000,
    synthetic_dim: int = 512,
) -> int:
    result = run_profile_selection(
        output_dir=output_dir,
        n_samples=n_samples,
        min_distance_km=min_distance_km,
        synthetic_n=synthetic_n,
        synthetic_dim=synthetic_dim,
    )
    print(json.dumps(result, indent=2))
    return 0


@cli_command(
    "temporal-sensitivity-test",
    help="Run temporal-weight sensitivity test with constraint integration",
    args={
        "output_dir": {
            "type": str,
            "default": "outputs/runs",
            "help": "Output directory",
        },
        "csv_meta": {
            "type": str,
            "default": "data/new_all_tiles.csv",
            "help": "Metadata CSV path",
        },
        "batch_size": {
            "type": int,
            "default": 16,
            "help": "Feature extraction batch size",
        },
        "n_clusters": {
            "type": int,
            "default": 8,
            "help": "Clustering k",
        },
        "n_samples": {
            "type": int,
            "default": 40,
            "help": "Number of samples",
        },
        "min_distance_km": {
            "type": float,
            "default": 40.0,
            "help": "Spatial minimum distance",
        },
    },
)
def cli_temporal_sensitivity_test(
    output_dir: str = "outputs/runs",
    csv_meta: str = "data/new_all_tiles.csv",
    batch_size: int = 16,
    n_clusters: int = 8,
    n_samples: int = 40,
    min_distance_km: float = 40.0,
) -> int:
    result = run_temporal_sensitivity_test(
        output_dir=output_dir,
        csv_meta=csv_meta,
        batch_size=batch_size,
        n_clusters=n_clusters,
        n_samples=n_samples,
        min_distance_km=min_distance_km,
    )
    print(json.dumps(result, indent=2))
    return 0
