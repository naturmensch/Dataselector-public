"""
Quick greedy benchmark for fastest working configuration on this machine.
- Warm-up
- Per-step timings (extraction / umap / clustering / selection)
- Greedy configs (stop at first success)
- Writes CSV + JSON summary in outputs/
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import pandas as pd
import numpy as np

from src.feature_extractor import FeatureExtractor
from src.clustering import ClusteringPipeline
from src.diversity_selector import DiversitySelector

OUT = Path('outputs')
OUT.mkdir(exist_ok=True, parents=True)


def filter_existing(image_names, image_dir: Path):
    existing = []
    for n in image_names:
        p = Path(n) if isinstance(n, (str, Path)) else Path(n)
        if not p.is_absolute():
            p = image_dir / p
        if p.exists():
            existing.append(p.name)
    return existing


def run_benchmark(subset_size:int=50, stop_on_success:bool=True):
    features_cache = OUT / 'features.npy'
    meta_cache = OUT / 'metadata.csv'
    image_dir = Path('data/images')

    # Prefer cached features if available
    if features_cache.exists() and meta_cache.exists():
        print("Using cached features + metadata for quick test")
        t0 = time.perf_counter()
        features = np.load(features_cache)
        metadata = pd.read_csv(meta_cache)
        n = min(subset_size, len(features))
        feat_sub = features[:n]
        meta_sub = metadata.iloc[:n].reset_index(drop=True)

        timings = {}
        timings['start'] = 0.0
        # timing UMAP + clustering + selection
        t0 = time.perf_counter()
        cl = ClusteringPipeline(n_clusters=8)
        emb, labels = cl.fit_transform(feat_sub)
        timings['umap_and_kmeans_s'] = time.perf_counter() - t0

        t1 = time.perf_counter()
        selector = DiversitySelector(n_samples=10)
        sel = selector.select(feat_sub, metadata=meta_sub, temporal_weight=0.2, spatial_constraint=False)
        timings['selection_s'] = time.perf_counter() - t1
        timings['total_s'] = sum([timings.get('umap_and_kmeans_s', 0.0), timings.get('selection_s', 0.0)])
        print("Cached flow timings:", timings)
        # Save quick summary
        summary = {'mode':'cached','subset_n':n,'timings':timings, 'success': True}
        (OUT / 'quick_benchmark_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    # else: prepare image list from metadata
    meta_path = Path('data/new_all_tiles.csv')
    if not meta_path.exists():
        raise SystemExit("Metadata not found: data/new_all_tiles.csv")

    metadata = pd.read_csv(meta_path)
    if 'image_filename' in metadata.columns and metadata['image_filename'].notna().any():
        candidates = metadata['image_filename'].dropna().tolist()
    elif 'shortName' in metadata.columns and metadata['shortName'].notna().any():
        candidates = metadata['shortName'].dropna().tolist()
    else:
        candidates = metadata['longName'].dropna().tolist()

    candidates = filter_existing(candidates, image_dir)
    if not candidates:
        raise SystemExit("Keine existierenden Bilddateien im Subset gefunden. Prüfe image paths.")

    subset = candidates[:subset_size]
    print(f"Using {len(subset)} images from {len(candidates)} available for benchmark")

    # configs from fastest to slowest
    configs = [
        {'crop_size': (512, 512), 'batch_size': 32},
        {'crop_size': (512, 512), 'batch_size': 16},
        {'crop_size': (1024, 1024), 'batch_size': 16},
        {'crop_size': (2048, 2048), 'batch_size': 8},
    ]

    results = []
    for cfg in configs:
        print("\nTrying config:", cfg)
        extractor = FeatureExtractor(model_name='resnet50', input_size=224, default_crop_size=cfg['crop_size'])
        # Warm-up: small dry-run (avoid counting initialization overhead)
        warm = subset[:min(5, len(subset))]
        try:
            extractor.extract_features_batch(warm, image_dir, batch_size=min(cfg['batch_size'], len(warm)), crop_size=cfg['crop_size'])
        except Exception as e:
            print("Warm-up failed, continuing to next config:", e)

        try:
            t_start = time.perf_counter()
            feats = extractor.extract_features_batch(subset, image_dir, batch_size=cfg['batch_size'], crop_size=cfg['crop_size'])
            t_ex = time.perf_counter() - t_start

            t_umap0 = time.perf_counter()
            cl = ClusteringPipeline(n_clusters=8)
            emb, labels = cl.fit_transform(feats)
            t_umap = time.perf_counter() - t_umap0

            t_sel0 = time.perf_counter()
            selector = DiversitySelector(n_samples=10)
            sel = selector.select(feats, metadata=metadata.iloc[:len(subset)].reset_index(drop=True), temporal_weight=0.2, spatial_constraint=False)
            t_sel = time.perf_counter() - t_sel0

            total = time.perf_counter() - t_start
            print(f"Config success: extraction={t_ex:.2f}s umap={t_umap:.2f}s selection={t_sel:.2f}s total={total:.2f}s")
            res = {
                'config': cfg,
                'extraction_s': t_ex,
                'umap_s': t_umap,
                'selection_s': t_sel,
                'total_s': total,
                'selected_indices': sel,
                'success': True
            }
            results.append(res)
            # persist subset features for inspection
            np.save(OUT / f'quick_features_{cfg["crop_size"][0]}_{cfg["batch_size"]}.npy', feats)

            # write CSV summary (append)
            df = pd.DataFrame(results)
            df.to_csv(OUT / 'quick_benchmark_results.csv', index=False)
            (OUT / 'quick_benchmark_summary.json').write_text(json.dumps({
                'best': res,
                'all': results
            }, ensure_ascii=False, indent=2))

            if stop_on_success:
                print("Stopping on first successful (fastest) config.")
                return res

        except Exception as e:
            print("Config failed:", cfg, e)
            traceback.print_exc()
            results.append({'config': cfg, 'success': False, 'error': str(e)})

    # none succeeded
    (OUT / 'quick_benchmark_summary.json').write_text(json.dumps({'best': None, 'all': results}, ensure_ascii=False, indent=2))
    return {'best': None, 'all': results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=50, help='Subset size for quick run')
    parser.add_argument('--no-stop', dest='stop', action='store_false', help='Do not stop on first success')
    args = parser.parse_args()
    run_benchmark(subset_size=args.subset, stop_on_success=args.stop)
