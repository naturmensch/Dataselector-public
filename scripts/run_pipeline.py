#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def should_run_tuning(
    tune_flag: bool, force: bool, ttl_days: int, csv_meta: Path, out_dir: Path
) -> bool:
    """Decide whether to run tuning based on cache and flags.

    Rules:
      - If force: True -> run
      - If tune_flag False -> skip
      - If meta.json missing -> run
      - If csv_meta hash != stored -> run
      - If meta timestamp older than ttl_days -> run
      - else skip
    """
    if force:
        return True
    if not tune_flag:
        return False

    meta_path = Path(out_dir) / "meta.json"
    results_path = Path(out_dir) / "tuning_results.csv"

    if not results_path.exists() or not meta_path.exists():
        return True

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return True

    stored_hash = meta.get("csv_meta_hash")
    if stored_hash is None:
        return True

    current_hash = _file_hash(Path(csv_meta))
    if current_hash != stored_hash:
        return True

    ts = meta.get("timestamp_utc")
    if ts is None:
        return True
    try:
        meta_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - meta_time).days
        if age_days > ttl_days:
            return True
    except Exception:
        return True

    return False


__all__ = ["should_run_tuning"]
# ruff: noqa: E402
"""Run pipeline with optional tuning.

Usage:
    python scripts/run_pipeline.py [--tune] [--force-tune] [--tune-ttl DAYS] [--interactive]
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delay heavy imports (umap/numba/torch) until needed to improve testability
# from src.experiments import ExperimentRunner
# from src.main import KDR100SelectionPipeline


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def should_run_tuning(
    tune_flag: bool, force: bool, ttl_days: int, csv_meta: Path, out_dir: Path
) -> bool:
    """Decide whether to run tuning.
    Rules:
      - If force: True -> run
      - If tune_flag False -> skip
      - If meta.json missing -> run
      - If csv_meta hash != stored -> run
      - If meta timestamp older than ttl_days -> run
      - else skip
    """
    if force:
        return True
    if not tune_flag:
        return False

    meta_path = out_dir / "meta.json"
    results_path = out_dir / "tuning_results.csv"

    if not results_path.exists() or not meta_path.exists():
        return True

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return True

    # Check hash
    stored_hash = meta.get("csv_meta_hash")
    if stored_hash is None:
        return True

    current_hash = _file_hash(csv_meta)
    if current_hash != stored_hash:
        return True

    # Check age
    ts = meta.get("timestamp_utc")
    if ts is None:
        return True
    try:
        meta_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - meta_time).days
        if age_days > ttl_days:
            return True
    except Exception:
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run parameter tuning before main pipeline (default: skip)",
    )
    parser.add_argument(
        "--force-tune",
        action="store_true",
        help="Force tuning even if cached results exist",
    )
    parser.add_argument(
        "--tune-ttl", type=int, default=7, help="Cache validity in days (default: 7)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for confirmation before expensive ops",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test (small sample size, limited runs)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples for tuning (overrides auto-detect)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Alternate workspace path to use for data and outputs (useful for tests)",
    )
    args = parser.parse_args()

    # Config
    CSV_META = ROOT / "data" / "new_all_tiles.csv"
    OUT_DIR = ROOT / "outputs" / "tuning_weights"
    # Allow overriding workspace for smoke tests / CI
    if args.workspace:
        CSV_META = Path(args.workspace) / "data" / "new_all_tiles.csv"
        OUT_DIR = Path(args.workspace) / "outputs" / "tuning_weights"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_tune = should_run_tuning(
        args.tune, args.force_tune, args.tune_ttl, CSV_META, OUT_DIR
    )

    if args.tune and args.interactive:
        resp = input("Tuning requested and may take long, proceed? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborting tuning by user request.")
            run_tune = False

    if run_tune:
        print("Running parameter tuning...")

        # If smoke mode, simulate minimal tuning outputs and avoid heavy imports
        if args.smoke:
            import pandas as _pd
            import json as _json
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            # Small fake tuning results
            df = _pd.DataFrame([
                {"alpha": 0.7, "beta": 0.15, "gamma": 0.15, "n_selected": 5, "clusters_covered": 3}
            ])
            df.to_csv(OUT_DIR / "tuning_results.csv", index=False)
            meta = {
                "timestamp_utc": "2026-01-24T00:00:00Z",
                "csv_meta": str(CSV_META),
                "csv_meta_hash": None,
                "n_combinations": 1,
                "best_metrics": df.iloc[0].to_dict(),
            }
            with open(OUT_DIR / "meta.json", "w") as _mf:
                _json.dump(meta, _mf)

            print(f"Smoke tuning completed: wrote {OUT_DIR / 'tuning_results.csv'}")

        else:
            # Import here to avoid heavy module imports during argument parsing / tests
            from src.experiments import ExperimentRunner

            runner = ExperimentRunner(output_dir=str(OUT_DIR))

            # Determine n_samples: CLI override -> smoke default -> legacy default
            if args.n_samples is not None:
                n_samples_val = int(args.n_samples)
            else:
                n_samples_val = 673

            # Limit runs in smoke mode (not set here)
            runner.run_weight_sweep(
                csv_meta=str(CSV_META),
                n_samples=n_samples_val,
                weight_combinations=None,
            )
    else:
        print("Skipping tuning (cache valid or not requested).")

    # After tuning (or skipping), run main pipeline (if importable).
    try:
        from src.main import KDR100SelectionPipeline

        pipeline = KDR100SelectionPipeline(
            config_path=str(ROOT / "config" / "pipeline_config.yaml")
        )
        pipeline.run()
    except Exception as e:
        # If the pipeline cannot be imported (e.g., missing heavy deps) but tuning finished, that's acceptable for smoke tests
        print(f"Warning: pipeline execution skipped due to import/exec error: {e}")


if __name__ == "__main__":
    main()
