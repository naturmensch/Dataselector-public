"""Bootstrap-based uncertainty estimates for Pareto candidates.

Usage example:
  PYTHONPATH=. python scripts/bootstrap_pareto_candidates.py --pareto outputs/fine_sweep/pareto_solutions.csv --n-boot 200 --out outputs/fine_sweep/bootstrap_results.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diversity_selector import DiversitySelector
from src.metrics import compute_metrics
from src.clustering import ClusteringPipeline
from src.io import load_metadata, load_or_extract_features


def jaccard(a, b):
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def bootstrap_candidate(alpha, beta, gamma, min_d, features, metadata, original_selection, cluster_labels_full, n_boot=200, random_seed=42):
    rng = np.random.default_rng(random_seed)
    N = features.shape[0]
    results = []

    for i in range(n_boot):
        sample_idx = rng.integers(0, N, size=N)
        boot_features = features[sample_idx]
        boot_meta = metadata.iloc[sample_idx].reset_index(drop=True)

        # clustering on boot features (not used for metrics -- metrics computed on original mapping)
        clustering = ClusteringPipeline(n_clusters=8)
        try:
            embeddings, cluster_labels_boot = clustering.fit_transform(boot_features)
        except Exception:
            cluster_labels_boot = np.zeros(N, dtype=int)

        ds = DiversitySelector(n_samples=len(original_selection), use_multi_criteria=True, random_state=int(1000 + i))
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
        metrics['jaccard_with_original'] = jaccard(mapped, original_selection)
        metrics['bootstrap_i'] = i
        results.append(metrics)

    return pd.DataFrame(results)


def main(pareto_csv, n_boot=200, output_csv=None, random_seed=42):
    pareto = pd.read_csv(pareto_csv)

    # load full metadata and features
    metadata = load_metadata(str(Path(ROOT) / 'data' / 'new_all_tiles.csv')) if (Path(ROOT) / 'outputs' / 'metadata.csv').exists() is False else pd.read_csv(Path(ROOT) / 'outputs' / 'metadata.csv')
    features = load_or_extract_features(Path(ROOT) / 'outputs', csv_meta=str(Path(ROOT) / 'outputs' / 'metadata.csv') if (Path(ROOT) / 'outputs' / 'metadata.csv').exists() else None, cache=True)

    # full clustering (for cluster labels)
    clustering = ClusteringPipeline(n_clusters=8)
    try:
        embeddings_full, cluster_labels_full = clustering.fit_transform(features)
    except Exception:
        cluster_labels_full = np.zeros(features.shape[0], dtype=int)

    all_boot = []
    summary_rows = []

    for idx, row in pareto.iterrows():
        alpha, beta, gamma = row['alpha'], row['beta'], row['gamma']
        min_d = row['min_distance_km']

        # compute original selection on full dataset
        ds = DiversitySelector(n_samples=int(row['n_selected']), use_multi_criteria=True, random_state=42)
        selected = ds.select(
            features=features,
            metadata=metadata,
            alpha_visual=float(alpha),
            beta_spatial=float(beta),
            gamma_temporal=float(gamma),
            spatial_constraint=True,
            min_distance_km=float(min_d),
        )
        original_sel = list(selected)

        # Bootstrap
        df_boot = bootstrap_candidate(alpha, beta, gamma, min_d, features, metadata, original_sel, cluster_labels_full, n_boot=n_boot, random_seed=random_seed)
        df_boot['alpha'] = alpha
        df_boot['beta'] = beta
        df_boot['gamma'] = gamma
        df_boot['min_distance_km'] = min_d
        df_boot['n_selected'] = int(row['n_selected'])
        df_boot['pareto_idx'] = idx
        all_boot.append(df_boot)

        # summary
        summary = {
            'pareto_idx': idx,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'min_distance_km': min_d,
            'n_selected': int(row['n_selected']),
            'temporal_std_mean': df_boot['temporal_std'].mean(),
            'temporal_std_std': df_boot['temporal_std'].std(),
            'wwi_percent_mean': df_boot['wwi_percent'].mean(),
            'wwi_percent_std': df_boot['wwi_percent'].std(),
            'jaccard_mean': df_boot['jaccard_with_original'].mean(),
            'jaccard_std': df_boot['jaccard_with_original'].std(),
        }
        summary_rows.append(summary)

    df_all = pd.concat(all_boot, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    outdir = Path(output_csv).parent if output_csv is not None else Path(ROOT) / 'outputs' / 'fine_sweep'
    outdir.mkdir(parents=True, exist_ok=True)
    if output_csv is None:
        df_all.to_csv(Path(ROOT) / 'outputs' / 'fine_sweep' / 'bootstrap_results_full.csv', index=False)
        df_summary.to_csv(Path(ROOT) / 'outputs' / 'fine_sweep' / 'bootstrap_summary.csv', index=False)
    else:
        df_all.to_csv(output_csv, index=False)
        df_summary.to_csv(Path(output_csv).with_name(Path(output_csv).stem + '_summary.csv'), index=False)

    print('Bootstrap finished. Results saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pareto', type=str, default=str(Path(ROOT) / 'outputs' / 'fine_sweep' / 'pareto_solutions.csv'))
    parser.add_argument('--n-boot', type=int, default=200)
    parser.add_argument('--out', type=str, default=str(Path(ROOT) / 'outputs' / 'fine_sweep' / 'bootstrap_results.csv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args.pareto, n_boot=args.n_boot, output_csv=args.out, random_seed=args.seed)
