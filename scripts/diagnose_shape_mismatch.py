#!/usr/bin/env python3
"""
Diagnose script: prüft Alignment zwischen Metadata CSV und gecachten Features sowie Candidate‑CSVs in runs.
Nur lesende Operationen - kein Schreiben/Aktualisieren von Caches.

Usage:

"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import glob
import sys

import numpy as np
import pandas as pd


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument(
        "--features", required=False, help="Path to features.npy (optional). If omitted, script searches common outputs paths"
    )
    parser.add_argument(
        "--check-runs", action="store_true", help="Scan runs for candidate_set sizes"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: metadata CSV not found: {csv_path}")
        sys.exit(2)

    print(f"Metadata CSV: {csv_path}")
    meta_hash = sha256_of_file(csv_path)
    print(f"  sha256: {meta_hash}")

    meta = pd.read_csv(csv_path)
    n_meta = len(meta)
    print(f"  rows (candidates): {n_meta}")

    # Basic metadata health checks
    print("\nMetadata checks:")
    missing_coords = meta[['N','left']].isna().any(axis=1).sum()
    print(f"  rows with missing coord (N or left): {missing_coords}")
    n_missing_image = meta['image_path'].isna().sum() if 'image_path' in meta.columns else None
    print(f"  rows with missing image_path: {n_missing_image}")
    dup_short = meta['shortName'].duplicated().sum() if 'shortName' in meta.columns else None
    print(f"  duplicate shortName count: {dup_short}")
    dup_filename = meta['image_filename'].duplicated().sum() if 'image_filename' in meta.columns else None
    print(f"  duplicate image_filename count: {dup_filename}")
    n_nan_years = meta['year'].isna().sum() if 'year' in meta.columns else None
    print(f"  rows with NaN year: {n_nan_years}")

    # Look for an obvious features cache
    candidate_feature_paths = []
    if args.features:
        candidate_feature_paths.append(Path(args.features))
    # common locations
    candidate_feature_paths.extend([
        Path('outputs') / 'features.npy',
        Path('outputs') / 'tuning_weights' / 'features.npy',
        Path('outputs') / 'fine_sweep' / 'runs' / 'features.npy',
    ])

    found = []
    for p in candidate_feature_paths:
        if p.exists():
            found.append(p)

    # Also search for any outputs/*/features*.npy within outputs
    for p in glob.glob('outputs/**/*features*.npy', recursive=True):
        p = Path(p)
        if p not in found:
            found.append(p)

    print('\nFeature caches found:')
    if not found:
        print('  (none found under outputs/*.npy)')
    else:
        for p in sorted(found):
            try:
                arr = np.load(p, mmap_mode='r')
                print(f"  {p} -> shape={arr.shape}, dtype={arr.dtype}")
            except Exception as e:
                print(f"  {p} -> could not load: {e}")

    # Compare primary features.npy (outputs/features.npy) to metadata length
    primary = Path('outputs') / 'features.npy'
    if primary.exists():
        try:
            arr = np.load(primary, mmap_mode='r')
            print('\nPrimary cache: outputs/features.npy')
            print(f"  shape={arr.shape}")
            if arr.shape[0] != n_meta:
                print('  >>> MISMATCH: features rows != metadata rows')
            else:
                print('  OK: features rows == metadata rows')
        except Exception as e:
            print(f"  Could not load primary cache: {e}")
    else:
        print('\nPrimary cache outputs/features.npy not present')

    # Search runs' candidate_set.csv files and compare their lengths
    if args.check_runs:
        print('\nScanning outputs/runs/*/results/candidate_set.csv ...')
        run_paths = glob.glob('outputs/runs/*/results/candidate_set.csv')
        if not run_paths:
            print('  (no candidate_set.csv files found)')
        else:
            discrepancies = []
            for p in sorted(run_paths):
                try:
                    df = pd.read_csv(p)
                    n = len(df)
                except Exception as e:
                    print(f"  {p} -> could not read: {e}")
                    continue
                # look for a manifest nearby
                manifest = Path(Path(p).parents[2]) / 'manifest.json'
                manifest_n = None
                if manifest.exists():
                    try:
                        m = json.load(open(manifest))
                        manifest_n = m.get('n_candidates')
                    except Exception:
                        manifest_n = None
                if n != n_meta:
                    discrepancies.append((p, n, manifest_n))
                print(f"  {p} -> rows={n}, manifest_n_candidates={manifest_n}")

            if discrepancies:
                print('\nRuns with candidate counts != metadata rows:')
                for p,n,man in discrepancies:
                    print(f"  {p} -> {n} (manifest: {man})")
            else:
                print('\nAll runs match current metadata row count')

    # Check for features meta info file
    fm = Path('outputs') / 'features.meta.json'
    if fm.exists():
        try:
            j = json.load(open(fm))
            print('\nFound outputs/features.meta.json:')
            print(json.dumps(j, indent=2))
        except Exception as e:
            print(f"Could not read features.meta.json: {e}")
    else:
        print('\nNo outputs/features.meta.json found')

    print('\nDiagnosis complete.')


if __name__ == '__main__':
    main()
