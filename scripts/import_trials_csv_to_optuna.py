#!/usr/bin/env python3
"""Import trials from a CSV file into an Optuna storage (SQLite).

Useful for migrating legacy runs or restoring state from CSV backups.

Usage:
    python scripts/import_trials_csv_to_optuna.py --csv outputs/runs/.../results/trials.csv --storage sqlite:///outputs/runs/.../optuna_study.db
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import optuna
from optuna.trial import TrialState, create_trial

def main():
    parser = argparse.ArgumentParser(description="Import trials CSV to Optuna storage")
    parser.add_argument("--csv", required=True, help="Path to trials.csv")
    parser.add_argument("--storage", required=True, help="Storage URL (e.g. sqlite:///study.db)")
    parser.add_argument("--study-name", default="kdr100_opt", help="Study name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Connecting to storage {args.storage} (study: {args.study_name})...")
    study = optuna.create_study(storage=args.storage, study_name=args.study_name, load_if_exists=True, direction="maximize")

    print(f"Importing {len(df)} trials...")
    
    # Identify parameter columns
    # In trials.csv, params are usually 'a', 'b', 'c', 'min_distance_km', 'n_samples'
    param_cols = ['a', 'b', 'c', 'min_distance_km', 'n_samples']
    
    count = 0
    for i, row in df.iterrows():
        # State
        state_val = row.get('state', 'COMPLETE')
        if isinstance(state_val, str):
            state_val = state_val.replace('TrialState.', '')
            try:
                state = getattr(TrialState, state_val)
            except AttributeError:
                state = TrialState.COMPLETE
        else:
            state = TrialState.COMPLETE

        # Value
        value = row.get('value')
        if pd.isna(value):
            value = None
        else:
            value = float(value)

        # Params
        params = {}
        for p in param_cols:
            if p in row and pd.notna(row[p]):
                val = row[p]
                if p in ['n_samples', 'min_distance_km']:
                    params[p] = int(float(val))
                else:
                    params[p] = float(val)
        
        trial = create_trial(state=state, value=value, params=params)
        study.add_trial(trial)
        count += 1

    print(f"Import complete. Added {count} trials.")

if __name__ == "__main__":
    main()