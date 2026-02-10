#!/usr/bin/env python3
"""Compare candidate min-distance policies on canonical metadata.

This script evaluates candidate values (e.g. 28.5, 40.0, 45.0) on an identical
code path and summarizes:
- hardcut target-met rate
- selection shortfall
- diversity/spread proxies
- stability across seeds (pairwise Jaccard overlap)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


@dataclass
class RunResult:
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


def _resolve_defaults(config_path: Path) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "n_samples": 34,
        "n_clusters": 8,
        "alpha": 0.40,
        "beta": 0.30,
        "gamma": 0.30,
    }

    if not config_path.exists():
        return defaults

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sel = cfg.get("selection", {}) if isinstance(cfg, dict) else {}
    clu = cfg.get("clustering", {}) if isinstance(cfg, dict) else {}

    n_samples = sel.get("n_samples")
    if isinstance(n_samples, int) and n_samples > 0:
        defaults["n_samples"] = int(n_samples)

    defaults["n_clusters"] = int(clu.get("n_clusters", defaults["n_clusters"]))
    defaults["alpha"] = float(sel.get("alpha_visual", defaults["alpha"]))
    defaults["beta"] = float(sel.get("beta_spatial", defaults["beta"]))
    defaults["gamma"] = float(sel.get("gamma_temporal", defaults["gamma"]))
    return defaults


def _run_single(
    *,
    features: np.ndarray,
    metadata: pd.DataFrame,
    distance: float,
    seed: int,
    n_samples: int,
    n_clusters: int,
    alpha: float,
    beta: float,
    gamma: float,
    pre_selected_names: list[str] | None,
    pre_selected_indices: list[int] | None,
) -> RunResult:
    from sklearn.cluster import KMeans

    from dataselector.analysis.metrics import compute_metrics
    from dataselector.selection.diversity_selector import DiversitySelector

    k = max(2, min(int(n_clusters), features.shape[0]))
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

    return RunResult(
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


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for distance, sub in df.groupby("distance"):
        sets = [set(json.loads(x)) for x in sub["selected_indices_json"].tolist()]
        mean_j, min_j = _pairwise_jaccard(sets)

        row = {
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
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    out["decision_score"] = (
        out["target_met_rate"] * 100.0
        + out["stability_jaccard_mean"] * 10.0
        + out["mean_clusters_covered"] * 0.5
    )

    # Rule-based recommendation:
    # 1) keep candidates with low shortfall
    # 2) among them optimize feasibility+stability+coverage
    # 3) if near tie, prefer smaller distance for more downstream combination space
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-path",
        default="data/new_all_tiles.csv",
        help="Canonical metadata CSV",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        default=[28.5, 40.0, 45.0],
        help="Candidate min_distance_km values",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seeds for replicated runs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override n_samples (default from config, fallback 34)",
    )
    parser.add_argument(
        "--config-path",
        default="config/pipeline_config.yaml",
        help="Config path for defaults",
    )
    parser.add_argument(
        "--output-dir",
        default="reports_2026-02-09",
        help="Directory for result CSV/MD artifacts",
    )
    parser.add_argument(
        "--pre-names",
        nargs="*",
        default=None,
        help="Optional pre-selected names for all runs",
    )
    parser.add_argument(
        "--pre-indices",
        nargs="*",
        type=int,
        default=None,
        help="Optional pre-selected indices for all runs",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.data.metadata_source import assert_canonical_metadata

    args = parse_args(argv)

    canonical_path = assert_canonical_metadata(
        args.metadata_path,
        context="compare_min_distance_policies",
        root=Path.cwd(),
    )

    defaults = _resolve_defaults(Path(args.config_path))
    n_samples = (
        int(args.n_samples)
        if args.n_samples is not None
        else int(defaults["n_samples"])
    )
    n_clusters = int(defaults["n_clusters"])
    alpha = float(defaults["alpha"])
    beta = float(defaults["beta"])
    gamma = float(defaults["gamma"])

    metadata = load_metadata(str(canonical_path))
    features = load_or_extract_features(
        out_dir="outputs",
        csv_meta=str(canonical_path),
        batch_size=16,
        cache=True,
        enforce_canonical=True,
    )

    pre_names = args.pre_names if args.pre_names else None
    pre_indices = args.pre_indices if args.pre_indices else None

    records: list[dict[str, object]] = []
    for distance in args.distances:
        for seed in args.seeds:
            rr = _run_single(
                features=features,
                metadata=metadata,
                distance=float(distance),
                seed=int(seed),
                n_samples=int(n_samples),
                n_clusters=int(n_clusters),
                alpha=float(alpha),
                beta=float(beta),
                gamma=float(gamma),
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
    summary_df = _summarize(raw_df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    raw_csv = out_dir / f"min_distance_policy_runs_{ts}.csv"
    summary_csv = out_dir / f"min_distance_policy_summary_{ts}.csv"
    summary_md = out_dir / f"min_distance_policy_summary_{ts}.md"

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    recommended = float(summary_df["recommended_distance"].iloc[0])
    rationale = str(summary_df["recommendation_rationale"].iloc[0])

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Min Distance Policy Comparison\n\n")
        f.write(f"- UTC timestamp: `{ts}`\n")
        f.write(f"- Metadata: `{canonical_path}`\n")
        f.write(f"- Rows: `{len(metadata)}`\n")
        f.write(f"- n_samples: `{n_samples}`\n")
        f.write(f"- seeds: `{args.seeds}`\n")
        f.write(f"- distances: `{args.distances}`\n")
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

    print("\n=== Min Distance Policy Comparison Complete ===")
    print(f"raw_runs: {raw_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_md: {summary_md}")
    print(f"recommended_min_distance_km: {recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
