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

from src.experiments import ExperimentRunner
from src.main import KDR100SelectionPipeline


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
    args = parser.parse_args()

    # Config
    CSV_META = ROOT / "data" / "new_all_tiles.csv"
    OUT_DIR = ROOT / "outputs" / "tuning_weights"
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
        runner = ExperimentRunner(output_dir=str(OUT_DIR))
        runner.run_weight_sweep(
            csv_meta=str(CSV_META), n_samples=673, weight_combinations=None
        )
    else:
        print("Skipping tuning (cache valid or not requested).")

    # After tuning (or skipping), run main pipeline
    pipeline = KDR100SelectionPipeline(
        config_path=str(ROOT / "config" / "pipeline_config.yaml")
    )
    pipeline.run()


if __name__ == "__main__":
    main()
