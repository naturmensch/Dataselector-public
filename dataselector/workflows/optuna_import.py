"""Import trials from CSV into Optuna storage."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from dataselector.cli_decorators import cli_command

logger = logging.getLogger(__name__)


def import_trials_from_csv(
    csv_path: Path,
    storage: str,
    study_name: str = "kdr100_opt",
    direction: str = "maximize",
) -> int:
    """
    Import trials from CSV file into Optuna study storage.

    Args:
        csv_path: Path to trials.csv file
        storage: Optuna storage URL (e.g., sqlite:///study.db)
        study_name: Name of the Optuna study
        direction: Optimization direction ('maximize' or 'minimize')

    Returns:
        Number of trials imported

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ImportError: If optuna is not available
    """
    # Lazy import to avoid hard dependency
    try:
        import optuna
        from optuna.trial import TrialState, create_trial
    except ImportError as e:
        raise ImportError(
            "optuna is required for CSV import. Install with: pip install optuna"
        ) from e

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading trials from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} trials in CSV")

    print(f"Connecting to storage {storage} (study: {study_name})...")
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        direction=direction,
    )

    print(f"Importing {len(df)} trials...")

    # Identify parameter columns (exclude metadata)
    reserved = {"number", "value", "state"}
    param_cols = [c for c in df.columns if c not in reserved]

    # Build distributions for each parameter based on observed CSV values
    distributions = {}
    for p in param_cols:
        col = df[p].dropna()
        if len(col) == 0:
            continue

        # Infer distribution type from values
        if col.dtype in ("int64", "int32"):
            distributions[p] = optuna.distributions.IntDistribution(
                low=int(col.min()), high=int(col.max())
            )
        elif col.dtype in ("float64", "float32"):
            distributions[p] = optuna.distributions.FloatDistribution(
                low=float(col.min()), high=float(col.max())
            )
        else:
            # Categorical for strings/other types
            unique_vals = col.unique().tolist()
            distributions[p] = optuna.distributions.CategoricalDistribution(
                choices=unique_vals
            )

    print(f"Detected {len(distributions)} parameters: {list(distributions.keys())}")

    # Import each trial
    imported_count = 0
    for idx, row in df.iterrows():
        # Extract parameters (skip NaN values)
        params = {p: row[p] for p in param_cols if pd.notna(row[p])}

        # Determine trial state
        if "state" in df.columns and pd.notna(row["state"]):
            state_str = str(row["state"]).upper()
            try:
                state = TrialState[state_str]
            except KeyError:
                state = TrialState.COMPLETE
        else:
            state = TrialState.COMPLETE

        # Get objective value
        value = row.get("value", None)
        if pd.isna(value):
            value = None

        # Create trial
        try:
            trial = create_trial(
                state=state,
                params=params,
                distributions=distributions,
                values=[value] if value is not None else [],
            )
            study.add_trial(trial)
            imported_count += 1
        except Exception as e:
            print(f"Warning: Failed to import trial {idx}: {e}")
            continue

    print(f"Successfully imported {imported_count}/{len(df)} trials")
    return imported_count


@cli_command(
    "optuna-import",
    help="Import Optuna trials from CSV into storage",
    args={
        "csv": {"type": str, "required": True, "help": "Path to trials CSV"},
        "storage": {
            "type": str,
            "required": True,
            "help": "Optuna storage URL (e.g. sqlite:///study.db)",
        },
        "study_name": {
            "type": str,
            "default": "kdr100_opt",
            "help": "Study name",
        },
        "direction": {
            "type": str,
            "default": "maximize",
            "help": "Optimization direction",
        },
    },
)
def main(
    csv: str,
    storage: str,
    study_name: str = "kdr100_opt",
    direction: str = "maximize",
) -> int:
    import_trials_from_csv(
        csv_path=Path(csv),
        storage=storage,
        study_name=study_name,
        direction=direction,
    )
    return 0
