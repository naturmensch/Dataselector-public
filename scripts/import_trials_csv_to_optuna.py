#!/usr/bin/env python3
"""Import trials from a CSV file into an Optuna storage (SQLite).

Useful for migrating legacy runs or restoring state from CSV backups.

Usage:
    python scripts/import_trials_csv_to_optuna.py --csv outputs/runs/.../results/trials.csv --storage sqlite:///outputs/runs/.../optuna_study.db
"""

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState, create_trial


def main():
    parser = argparse.ArgumentParser(description="Import trials CSV to Optuna storage")
    parser.add_argument("--csv", required=True, help="Path to trials.csv")
    parser.add_argument(
        "--storage", required=True, help="Storage URL (e.g. sqlite:///study.db)"
    )
    parser.add_argument("--study-name", default="kdr100_opt", help="Study name")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Connecting to storage {args.storage} (study: {args.study_name})...")
    study = optuna.create_study(
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
        direction="maximize",
    )

    print(f"Importing {len(df)} trials...")

    # Identify parameter columns dynamically from CSV (exclude known metadata cols)
    reserved = {"number", "value", "state"}
    param_cols = [c for c in df.columns if c not in reserved]

    # Build distributions for each param based on observed CSV values so Optuna accepts the trials
    distributions = {}
    for p in param_cols:
        col = df[p].dropna()
        if col.empty:
            # fallback to a generic float distribution
            distributions[p] = optuna.distributions.FloatDistribution(0.0, 1.0)
            continue
        # integer-like columns -> IntDistribution
        if pd.api.types.is_integer_dtype(col) or all(
            float(x).is_integer() for x in col
        ):
            lo = int(col.min())
            hi = int(col.max())
            if lo == hi:
                hi = lo + 10
            distributions[p] = optuna.distributions.IntDistribution(
                low=lo, high=max(hi, lo + 1)
            )
        else:
            lo = float(col.min())
            hi = float(col.max())
            if lo == hi:
                lo = max(0.0, lo - 0.1)
                hi = lo + 0.2
            distributions[p] = optuna.distributions.FloatDistribution(low=lo, high=hi)

    count = 0
    for i, row in df.iterrows():
        # State
        state_val = row.get("state", "COMPLETE")
        if isinstance(state_val, str):
            state_val = state_val.replace("TrialState.", "")
            try:
                state = getattr(TrialState, state_val)
            except AttributeError:
                state = TrialState.COMPLETE
        else:
            state = TrialState.COMPLETE

        # Value
        value = row.get("value")
        if pd.isna(value):
            value = None
        else:
            value = float(value)

        # Params
        params = {}
        for p in param_cols:
            if p in row and pd.notna(row[p]):
                val = row[p]
                if p in ["n_samples", "min_distance_km"]:
                    params[p] = int(float(val))
                else:
                    params[p] = float(val)

        # Create trial with inferred parameter distributions so Optuna validation succeeds
        trial = create_trial(
            state=state,
            value=value,
            params=params,
            distributions={
                k: distributions[k] for k in params.keys() if k in distributions
            },
        )
        study.add_trial(trial)
        count += 1

    print(f"Import complete. Added {count} trials.")


if __name__ == "__main__":
    main()
