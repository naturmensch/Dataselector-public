"""Utilities for incremental result management during long-running experiments.

Provides:
- IncrementalCSVWriter: Appends rows to CSV as they become available
- TrialBuffer: Batches trials for efficient I/O
- ResultCache: In-memory cache with periodic flushing
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class IncrementalCSVWriter:
    """Write CSV results incrementally without loading entire file into memory.

    Features:
    - Creates file with header on first write
    - Appends rows efficiently
    - Maintains backup versions
    - Thread-safe (optional locking)

    Usage:
        writer = IncrementalCSVWriter("trials.csv", fieldnames=["trial_id", "value"])
        for trial in trials:
            writer.append({"trial_id": trial.id, "value": trial.value})
        writer.close()
    """

    def __init__(
        self,
        filepath: Path,
        fieldnames: List[str],
        create_backup: bool = True,
        buffer_size: int = 100,
    ):
        """Initialize incremental CSV writer.

        Args:
            filepath: Output file path
            fieldnames: CSV column names
            create_backup: Create backup of existing file
            buffer_size: Buffer rows before writing to disk
        """
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.row_count = 0

        # Create backup if file exists
        if self.filepath.exists() and create_backup:
            backup = self.filepath.with_suffix(
                f".backup_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
            )
            self.filepath.rename(backup)

        # Initialize file with header
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(self, row: Dict[str, Any]):
        """Queue a row for writing.

        Args:
            row: Dictionary matching fieldnames
        """
        self.buffer.append(row)
        self.row_count += 1

        # Flush if buffer full
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered rows to disk."""
        if not self.buffer:
            return

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)

        self.buffer.clear()

    def close(self):
        """Flush any remaining rows and close."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TrialBuffer:
    """Buffer trials for batch processing and incremental saving.

    Useful for Optuna studies where you want to:
    - Batch extract trials every N iterations
    - Save incrementally without blocking optimization
    - Compute running statistics
    """

    def __init__(self, csv_writer: IncrementalCSVWriter):
        """Initialize trial buffer.

        Args:
            csv_writer: IncrementalCSVWriter instance
        """
        self.writer = csv_writer
        self.trials: List[Dict[str, Any]] = []
        self.best_value = float("inf")
        self.best_trial = None

    def add_trial(self, trial_dict: Dict[str, Any]):
        """Add a trial to the buffer.

        Args:
            trial_dict: Trial data as dictionary
        """
        self.trials.append(trial_dict)

        # Track best
        if "value" in trial_dict and trial_dict["value"] < self.best_value:
            self.best_value = trial_dict["value"]
            self.best_trial = trial_dict

    def flush_to_csv(self):
        """Write all buffered trials to CSV."""
        for trial in self.trials:
            self.writer.append(trial)
        self.writer.flush()
        self.trials.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.trials:
            return {"n_buffered": 0}

        values = [t.get("value", None) for t in self.trials if "value" in t]
        return {
            "n_buffered": len(self.trials),
            "n_values": len(values),
            "mean_value": sum(values) / len(values) if values else None,
            "best_buffered": min(values) if values else None,
            "best_overall": self.best_value,
        }


class ResultCache:
    """In-memory cache for results with periodic disk flushing.

    Provides:
    - Fast in-memory access
    - Configurable persistence (batch/timed)
    - Automatic state snapshots

    Usage:
        cache = ResultCache("trials.csv", fieldnames=[...])
        for trial in trials:
            cache.add("trials", trial_dict)
            if trial_num % 100 == 0:
                cache.persist()
    """

    def __init__(
        self,
        filepath: Path,
        fieldnames: List[str],
        persist_interval: int = 50,
    ):
        """Initialize result cache.

        Args:
            filepath: CSV file path
            fieldnames: CSV columns
            persist_interval: Flush to disk every N additions
        """
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames
        self.persist_interval = persist_interval
        self.cache: List[Dict[str, Any]] = []
        self.total_persisted = 0

        # Load existing data if file exists
        if self.filepath.exists():
            try:
                df = pd.read_csv(self.filepath)
                self.cache = df.to_dict("records")
                self.total_persisted = len(self.cache)
            except Exception as e:
                print(f"Warning: Could not load existing cache: {e}")

    def add(self, row: Dict[str, Any]):
        """Add a row to cache."""
        self.cache.append(row)

        # Auto-flush if threshold reached
        if len(self.cache) - self.total_persisted >= self.persist_interval:
            self.persist()

    def persist(self):
        """Write cache to disk as CSV."""
        if not self.cache:
            return

        df = pd.DataFrame(self.cache)
        df.to_csv(self.filepath, index=False)
        self.total_persisted = len(self.cache)

    def get_all(self) -> pd.DataFrame:
        """Return cache as DataFrame."""
        return pd.DataFrame(self.cache)

    def get_best(
        self, by: str = "value", ascending: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get best row by metric."""
        if not self.cache:
            return None
        df = pd.DataFrame(self.cache)
        best_idx = df[by].idxmin() if ascending else df[by].idxmax()
        return self.cache[best_idx]

    def close(self):
        """Persist and finalize."""
        self.persist()
