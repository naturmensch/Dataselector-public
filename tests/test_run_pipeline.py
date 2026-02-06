import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
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
    """Decide whether to run tuning (legacy logic from run_pipeline.py).

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


def _write_meta(path: Path, csv_hash: str, days_ago: int = 0):
    meta = {
        "timestamp_utc": (
            datetime.now(timezone.utc) - timedelta(days=days_ago)
        ).isoformat(),
        "csv_meta_hash": csv_hash,
    }
    with open(path, "w") as f:
        json.dump(meta, f)


def _write_file(path: Path, content: bytes):
    with open(path, "wb") as f:
        f.write(content)


def test_should_run_tuning_logic(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_bytes(b"hello")
    out = tmp_path / "out"
    out.mkdir()

    # 1) No meta -> run
    assert should_run_tuning(True, False, 7, csv, out) is True

    # 2) meta present, same hash, recent -> don't run
    meta_path = out / "meta.json"
    results_path = out / "tuning_results.csv"
    # create a dummy results file so the function doesn't early-exit on missing file
    results_path.write_text("run,alpha\n")

    # override meta json to have matching hash
    _write_meta(meta_path, _compute_hash(csv), days_ago=1)

    assert should_run_tuning(True, False, 7, csv, out) is False

    # 3) force -> run
    assert should_run_tuning(True, True, 7, csv, out) is True

    # 4) old meta -> run
    _write_meta(meta_path, _compute_hash(csv), days_ago=10)
    assert should_run_tuning(True, False, 7, csv, out) is True

    # 5) tune flag false -> don't run
    assert should_run_tuning(False, False, 7, csv, out) is False


def _compute_hash(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
