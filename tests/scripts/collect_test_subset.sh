#!/usr/bin/env bash
set -euo pipefail

# Collect a minimal test data subset for e2e tests into tests/test_data/
# Usage: bash tests/scripts/collect_test_subset.sh --n-images 5 [--datasets hamburg kiel] [--random] [--out-dir DIR]

N_IMAGES=5
DATASETS=("hamburg" "kdr100")  # Default datasets
RANDOM_MODE=0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
OUT_DIR="${ROOT}/tests/test_data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-images)
      N_IMAGES="$2"; shift 2;;
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATASETS+=("$1")
        shift
      done;;
    --random)
      RANDOM_MODE=1; shift;;
    --out-dir)
      OUT_DIR="$2"; shift 2;;
    --force)
      FORCE=1; shift;;
    *) shift;;
  esac
done

mkdir -p "${OUT_DIR}"

python3 - <<PY
import os, sys, shutil, random, csv
from pathlib import Path
from PIL import Image

root = Path(os.environ.get('ROOT', '${ROOT}'))
out = Path('${OUT_DIR}')
n_images = int('${N_IMAGES}')
datasets = "${DATASETS[@]}".split()
random_mode = int('${RANDOM_MODE}')

print('Output dir:', out)
print(f'Datasets: {datasets}, Random mode: {random_mode}, N images per dataset: {n_images}')

# Read the image data CSV
csv_path = root / 'data' / 'new_all_tiles.csv'
if csv_path.exists():
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        all_images = list(reader)
    print(f'Loaded {len(all_images)} image entries from {csv_path}')
else:
    print(f'Warning: {csv_path} not found, falling back to synthetic data')
    all_images = []

all_candidates = []
for img in all_images:
    img_path = Path(img['image_path'])
    if img_path.exists():
        # Determine dataset from path or assume
        ds = 'kdr100'  # Default, or parse from path
        all_candidates.append((img_path, ds, img))
    else:
        print(f'Image {img_path} not found, skipping')

if not all_candidates:
    print('No valid images found, generating synthetic PNGs for all datasets')
    for ds in datasets:
        ds_dir = out / ds
        img_dir = ds_dir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            fname = img_dir / f'{ds}_tile_{i+1:03d}.png'
            im = Image.new('RGB', (64,64), (int(i*30)%255, int(i*60)%255, int(i*90)%255))
            im.save(fname)
else:
    if random_mode:
        # Random selection across all
        selected = random.sample(all_candidates, min(n_images * len(datasets), len(all_candidates)))
        for img_path, ds, img in selected:
            ds_dir = out / ds
            img_dir = ds_dir / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            dst = img_dir / img['image_filename']
            shutil.copy2(img_path, dst)
        print(f'Randomly selected {len(selected)} images')
    else:
        # Per-dataset, but since ds is fixed, group by ds
        ds_groups = {}
        for img_path, ds, img in all_candidates:
            ds_groups.setdefault(ds, []).append((img_path, img))
        
        for ds in datasets:
            candidates = ds_groups.get(ds, [])
            if candidates:
                selected = candidates[:n_images]
                ds_dir = out / ds
                img_dir = ds_dir / 'images'
                img_dir.mkdir(parents=True, exist_ok=True)
                for img_path, img in selected:
                    dst = img_dir / img['image_filename']
                    shutil.copy2(img_path, dst)
                print(f'Copied {len(selected)} images for {ds}')
            else:
                print(f'No images for {ds}, generating synthetic')
                ds_dir = out / ds
                img_dir = ds_dir / 'images'
                img_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n_images):
                    fname = img_dir / f'{ds}_tile_{i+1:03d}.png'
                    im = Image.new('RGB', (64,64), (int(i*30)%255, int(i*60)%255, int(i*90)%255))
                    im.save(fname)

# Generate metadata CSVs using the copied images
for ds in datasets:
    ds_dir = out / ds
    img_dir = ds_dir / 'images'
    meta_file = ds_dir / 'metadata.csv'
    if img_dir.exists():
        images = list(img_dir.iterdir())
        if images:
            lines = ["longName,shortName,N,left,image_path,image_filename,year"]
            for img_file in images:
                # Use data from CSV if available, else dummy
                img_name = img_file.name
                # Find matching entry
                entry = next((img for img_path, _, img in all_candidates if img['image_filename'] == img_name), None)
                if entry:
                    lines.append(f"{entry['longName']},{entry['shortName']},{entry['id']},{entry['left']},{img_file},{img_name},{entry['year']}")
                else:
                    lines.append(f"{img_name},{img_name.split('.')[0]},1.0,1.0,{img_file},{img_name},1900")
            meta_file.write_text('\n'.join(lines))

# Create new_all_tiles.csv at out level, using selected images
all_tiles_file = out / 'new_all_tiles.csv'
lines = ['tile_id,year,lon,lat']
tile_id = 1
for ds in datasets:
    ds_dir = out / ds
    img_dir = ds_dir / 'images'
    if img_dir.exists():
        for img_file in img_dir.iterdir():
            lines.append(f'tile_{tile_id},1900,{9.0 + tile_id*0.1},{53.0 + tile_id*0.1}')
            tile_id += 1
all_tiles_file.write_text('\n'.join(lines))

print('Test data subset created successfully')
PY

echo 'Done.'
