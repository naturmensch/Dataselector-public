"""Compare selection with and without pre-selected seeds (e.g. 'Hamburg').
Produces a small CSV and Markdown report in outputs/seed_benchmark.
"""

from pathlib import Path
import pandas as pd


def main() -> int:
    ROOT = Path(__file__).resolve().parents[1]

    # Config (read from pipeline config for consistency)
    import yaml

    cfg = yaml.safe_load(open(ROOT / "config" / "pipeline_config.yaml"))

    n_samples = cfg.get("selection", {}).get("n_samples", 34)
    alpha = cfg.get("selection", {}).get("alpha_visual", 0.7)
    beta = cfg.get("selection", {}).get("beta_spatial", 0.05)
    gamma = cfg.get("selection", {}).get("gamma_temporal", 0.25)
    min_distance_km = cfg.get("selection", {}).get("min_distance_km", 40.0)
    batch_size = cfg.get("feature_extraction", {}).get("batch_size", 8)

    OUT = ROOT / "outputs" / "seed_benchmark"
    OUT.mkdir(parents=True, exist_ok=True)

    # Runtime imports to avoid heavy deps at import-time
    from dataselector.selection.clustering import ClusteringPipeline
    from dataselector.selection.diversity_selector import DiversitySelector
    from dataselector.data.io import load_metadata, load_or_extract_features
    from dataselector.analysis.metrics import compute_metrics

    # Load cached features & metadata
    features = load_or_extract_features(
        OUT,
        csv_meta=str(ROOT / "data" / "new_all_tiles.csv"),
        batch_size=batch_size,
        cache=True,
    )
    metadata = load_metadata(str(ROOT / "data" / "new_all_tiles.csv"))

    # compute cluster labels using existing pipeline to be consistent
    n_clusters_cfg = cfg.get("clustering", {}).get("n_clusters", 8)
    clustering = ClusteringPipeline(n_clusters=n_clusters_cfg)
    _, cluster_labels = clustering.fit_transform(features)

    results = []

    # Two scenarios: baseline (no seed) and seeded (Hamburg)
    scenarios = [
        ("no_seed", None, None),
        ("seed_Hamburg_name", ["Hamburg"], None),
    ]

    for tag, pre_names, pre_idxs in scenarios:
        ds = DiversitySelector(
            n_samples=n_samples, use_multi_criteria=True, random_state=42
        )
        selected = ds.select(
            features=features,
            metadata=metadata,
            alpha_visual=alpha,
            beta_spatial=beta,
            gamma_temporal=gamma,
            spatial_constraint=True,
            min_distance_km=min_distance_km,
            pre_selected=pre_idxs,
            pre_selected_names=pre_names,
        )

        metrics = compute_metrics(selected, metadata, cluster_labels, features)
        metrics.update(
            {
                "scenario": tag,
                "pre_selected_names": pre_names,
                "pre_selected_indices": pre_idxs,
                "n_selected": len(selected),
            }
        )

        # Also save the selection CSV snapshot
        sel_df = metadata.iloc[selected].copy()
        sel_df["selection_rank"] = range(len(sel_df))
        sel_out = OUT / f"selection_{tag}.csv"
        sel_df.to_csv(sel_out, index=False)

        results.append(metrics)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUT / "seed_vs_unseed_metrics.csv", index=False)

    # Write small Markdown summary
    md = OUT / "seed_vs_unseed_report.md"
    with open(md, "w") as f:
        f.write("# Seed vs No-Seed Selection Benchmark\n\n")
        f.write(
            "This short report compares baseline selection and selection seeded with 'Hamburg'.\n\n"
        )
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            # tabulate may be missing in some envs; fall back to CSV-style table
            f.write("\n" + df.to_string(index=False) + "\n")
        f.write("\n\nSelections saved in this folder for inspection.\n")

    print("Done. Results:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
