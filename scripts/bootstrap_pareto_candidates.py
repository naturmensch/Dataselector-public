"""Bootstrap-based uncertainty estimates for Pareto candidates.

Usage example:


def jaccard(a, b):
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def bootstrap_candidate(
    alpha,
    beta,
    gamma,
    min_d,
    features,
    metadata,
    original_selection,
    cluster_labels_full,
    n_boot=200,
    random_seed=42,
):
    # Local imports to keep module import-safe
    from tqdm import trange
    from src.clustering import ClusteringPipeline
    from src.diversity_selector import DiversitySelector
    from src.metrics import compute_metrics

    rng = np.random.default_rng(random_seed)
    N = features.shape[0]
    results = []

    for i in range(n_boot):
        sample_idx = rng.integers(0, N, size=N)
        boot_features = features[sample_idx]
        boot_meta = metadata.iloc[sample_idx].reset_index(drop=True)
        # Preserve projected coords in the bootstrap sample if present
        from src.io import attach_metric_gdf, get_metric_gdf

        gdf_metric = get_metric_gdf(metadata)
        if gdf_metric is not None:
            attach_metric_gdf(
                boot_meta, gdf_metric.iloc[sample_idx].reset_index(drop=True)
            )

        # clustering on boot features (not used for metrics -- metrics computed on original mapping)
        clustering = ClusteringPipeline(n_clusters=8)
        try:
            _embeddings, _cluster_labels_boot = clustering.fit_transform(boot_features)
        except Exception:
            # if clustering fails, continue with metrics computed on original labels
            pass

        selected_boot = ds.select(
            features=boot_features,
            metadata=boot_meta,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
        )

        # Map selected indices back to original indices
        mapped = np.unique(sample_idx[selected_boot]).tolist()

        # compute metrics relative to original data using full clustering labels
        metrics = compute_metrics(mapped, metadata, cluster_labels_full, features)
        # compute jaccard with original_selection
        metrics["jaccard_with_original"] = jaccard(mapped, original_selection)
        metrics["bootstrap_i"] = i
        results.append(metrics)

    return pd.DataFrame(results)


def main(
    pareto_csv,
    n_boot=200,
    output_csv=None,
    random_seed=42,
    uq_method: str = "bootstrap",
    n_ensemble_models: int = 5,
    ensemble_epochs: int = 50,
    pre_selected_names=None,
    pre_selected_indices=None,
):
    # Local imports to keep module import-safe, but prefer module-level hooks if tests patched them
    if load_metadata is None or load_or_extract_features is None:
        from src.io import load_metadata as _load_metadata_fn
        from src.io import load_or_extract_features as _load_or_extract_features_fn
    else:
        _load_metadata_fn = load_metadata
        _load_or_extract_features_fn = load_or_extract_features
    from src.clustering import ClusteringPipeline
    from src.diversity_selector import DiversitySelector

    pareto = pd.read_csv(pareto_csv)

    # load full metadata and features
    metadata = (
        _load_metadata_fn(str(Path(ROOT) / "data" / "new_all_tiles.csv"))
        if (Path(ROOT) / "outputs" / "metadata.csv").exists() is False
        else _load_metadata_fn(str(Path(ROOT) / "outputs" / "metadata.csv"))
    )
    features = _load_or_extract_features_fn(
        Path(ROOT) / "outputs",
        csv_meta=(
            str(Path(ROOT) / "outputs" / "metadata.csv")
            if (Path(ROOT) / "outputs" / "metadata.csv").exists()
            else None
        ),
        cache=True,
    )

    # full clustering (for cluster labels)
    clustering = ClusteringPipeline(n_clusters=8)
    try:
        embeddings_full, cluster_labels_full = clustering.fit_transform(features)
    except Exception:
        cluster_labels_full = np.zeros(features.shape[0], dtype=int)
    all_boot = []
    summary_rows = []

    # option: ensemble-based UQ (faster inference after modest training)

    for idx, row in pareto.iterrows():
        alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
        min_d = row["min_distance_km"]

        # compute original selection on full dataset
        selected = ds.select(
            features=features,
            metadata=metadata,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
            pre_selected=pre_selected_indices,
            pre_selected_names=pre_selected_names,
        )
        original_sel = list(selected)

        if use_ensemble:
            # Train small ensemble on a reduced number of bootstrap resamples
            n_boot_train = min(50, max(10, int(n_boot // 4)))
            df_boot_train = bootstrap_candidate(
                alpha,
                beta,
                gamma,
                min_d,
                features,
                metadata,
                original_sel,
                cluster_labels_full,
                n_boot=n_boot_train,
                random_seed=random_seed,
            )
            # attach hyperparameter columns so ensemble training can use them
            df_boot_train["alpha"] = alpha
            df_boot_train["beta"] = beta
            df_boot_train["gamma"] = gamma
            df_boot_train["min_distance_km"] = min_d

            # Fit ensembles for each target metric
            try:
                from scripts.uncertainty_quantification import (
                    fit_ensemble_on_bootstrap_df,
                    predict_with_uncertainty,
                )

                input_cols = ["alpha", "beta", "gamma", "min_distance_km"]
                targets = ["temporal_std", "wwi_percent", "jaccard_with_original"]

                preds = {}
                X_query = [[float(alpha), float(beta), float(gamma), float(min_d)]]
                for t in targets:
                    if t not in df_boot_train.columns:
                        preds[t] = (float("nan"), float("nan"))
                        continue
                    models = fit_ensemble_on_bootstrap_df(
                        df_boot_train,
                        input_cols=input_cols,
                        target_col=t,
                        n_models=n_ensemble_models,
                        epochs=ensemble_epochs,
                    )
                    mean, std = predict_with_uncertainty(models, np.array(X_query))
                    preds[t] = (float(mean[0]), float(std[0]))

                summary = {
                    "pareto_idx": idx,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "min_distance_km": min_d,
                    "n_selected": int(row["n_selected"]),
                    "temporal_std_mean": preds["temporal_std"][0],
                    "temporal_std_std": preds["temporal_std"][1],
                    "wwi_percent_mean": preds["wwi_percent"][0],
                    "wwi_percent_std": preds["wwi_percent"][1],
                    "jaccard_mean": preds["jaccard_with_original"][0],
                    "jaccard_std": preds["jaccard_with_original"][1],
                    "method": "ensemble",
                }

                # For ensemble mode we do not record per-resample rows, only the summary.
            except Exception as e:
                print(
                    f"Ensemble UQ failed ({e}), falling back to standard bootstrap for this candidate"
                )
                df_boot = bootstrap_candidate(
                    alpha,
                    beta,
                    gamma,
                    min_d,
                    features,
                    metadata,
                    original_sel,
                    cluster_labels_full,
                    n_boot=n_boot,
                    random_seed=random_seed,
                )
                df_boot["alpha"] = alpha
                df_boot["beta"] = beta
                df_boot["gamma"] = gamma
                df_boot["min_distance_km"] = min_d
                df_boot["n_selected"] = int(row["n_selected"])
                df_boot["pareto_idx"] = idx
                all_boot.append(df_boot)

                summary = {
                    "pareto_idx": idx,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "min_distance_km": min_d,
                    "n_selected": int(row["n_selected"]),
                    "temporal_std_mean": df_boot["temporal_std"].mean(),
                    "temporal_std_std": df_boot["temporal_std"].std(),
                    "wwi_percent_mean": df_boot["wwi_percent"].mean(),
                    "wwi_percent_std": df_boot["wwi_percent"].std(),
                    "jaccard_mean": df_boot["jaccard_with_original"].mean(),
                    "jaccard_std": df_boot["jaccard_with_original"].std(),
                    "method": "bootstrap",
                }
        else:
            # Bootstrap
            all_boot.append(df_boot)

            # summary
            summary = {
                "pareto_idx": idx,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "min_distance_km": min_d,
                "n_selected": int(row["n_selected"]),
                "temporal_std_mean": df_boot["temporal_std"].mean(),
                "temporal_std_std": df_boot["temporal_std"].std(),
                "wwi_percent_mean": df_boot["wwi_percent"].mean(),
                "wwi_percent_std": df_boot["wwi_percent"].std(),
                "jaccard_mean": df_boot["jaccard_with_original"].mean(),
                "jaccard_std": df_boot["jaccard_with_original"].std(),
                "method": "bootstrap",
            }

        summary_rows.append(summary)

    if len(all_boot) > 0:
        df_all = pd.concat(all_boot, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    df_summary = pd.DataFrame(summary_rows)

    outdir = (
        Path(output_csv).parent
        if output_csv is not None
        else Path(ROOT) / "outputs" / "fine_sweep"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        # Default legacy path
        default_dir = Path(ROOT) / "outputs" / "fine_sweep"
        if not df_all.empty:
            df_all.to_csv(default_dir / "bootstrap_results_full.csv", index=False)
        df_summary.to_csv(default_dir / "bootstrap_summary.csv", index=False)
    else:
        if not df_all.empty:
            df_all.to_csv(output_csv, index=False)
        df_summary.to_csv(
            Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv"),
            index=False,
        )

    # If attached to an ExperimentManager, persist into the run directory
    import os

    exp_dir = os.environ.get("EXPERIMENT_RUN_DIR")
    if exp_dir:
        try:
            from src.experiment_manager import ExperimentManager

            em = ExperimentManager.from_existing(exp_dir)
            if not df_all.empty:
                em.save_results("bootstrap_results_full", df_all, format="csv")
            em.save_results("bootstrap_summary", df_summary, format="csv")
            em.mark_stage_complete(
                "bootstrap", summary={"n_candidates": len(df_summary)}
            )
        except Exception as e:
            print(
                f"Warning: could not save bootstrap results to experiment manager: {e}"
            )

    print("Bootstrap finished. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pareto",
        type=str,
        default=str(Path(ROOT) / "outputs" / "fine_sweep" / "pareto_solutions.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(ROOT) / "outputs" / "fine_sweep" / "bootstrap_results.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uq-method",
        choices=["bootstrap", "ensemble"],
        default="bootstrap",
        help="Uncertainty quantification method to use",
    )
    parser.add_argument(
        "--n-ensemble-models",
        type=int,
        default=5,
        help="Number of ensemble members when using ensemble UQ",
    )
    parser.add_argument(
        "--ensemble-epochs",
        type=int,
        default=50,
        help="Epochs per ensemble member when training ensemble UQ",
    )
    parser.add_argument(
        "--pre-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional pre-selected tile names (e.g. Hamburg)",
    )
    parser.add_argument(
        "--pre-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional pre-selected tile indices",
    )
    args = parser.parse_args()

