#!/usr/bin/env python3
"""
Extract best selection from KDR100_sample run and prepare for sampler comparison.

Usage:
    python scripts/extract_and_compare_samplers.py --run-id 20260116_T214724_adaptive_full
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def extract_best_selection(run_id: str) -> dict:
    """Extract best trial and its selected tiles from run."""
    run_dir = Path(f"outputs/runs/{run_id}")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    
    # Read best trial
    best_trial_file = run_dir / "results" / "best_trial.json"
    if not best_trial_file.exists():
        raise FileNotFoundError(f"best_trial.json not found: {best_trial_file}")
    
    with open(best_trial_file) as f:
        best_trial = json.load(f)
    
    # Try to read best_selection.csv if it exists
    selection_file = run_dir / "results" / "best_selection.csv"
    selected_tiles = []
    
    if selection_file.exists():
        df = pd.read_csv(selection_file)
        selected_tiles = df['tile_id'].tolist() if 'tile_id' in df.columns else df['SheetNumber'].tolist()
    
    return {
        'run_id': run_id,
        'trial_number': best_trial.get('trial_number', best_trial.get('number')),
        'fitness': best_trial.get('fitness', best_trial.get('value')),
        'params': best_trial.get('params', {}),
        'n_selected_tiles': len(selected_tiles),
        'selected_tiles': selected_tiles,
        'best_trial_json': best_trial,
        'timestamp': datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract best selection and prepare sampler comparison")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., 20260116_T214724_adaptive_full)")
    parser.add_argument("--wait", action="store_true", help="Wait for run to complete before extracting")
    parser.add_argument("--output", default="outputs/kdr100_best_selection_info.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    print(f"[INFO] Extracting best selection from {args.run_id}...")
    
    try:
        selection_info = extract_best_selection(args.run_id)
        print(f"[✓] Best Trial #{selection_info['trial_number']}")
        print(f"    Fitness: {selection_info['fitness']:.4f}")
        print(f"    Selected Tiles: {selection_info['n_selected_tiles']}")
        print(f"    Parameters: {selection_info['params']}")
        
        # Save to JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(selection_info, f, indent=2)
        
        print(f"[✓] Saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
