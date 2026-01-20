import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import importlib.util

pytest.importorskip("numba", exc_type=ImportError)
pytestmark = pytest.mark.integration




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
    # Load the module dynamically after environment skip checks
    ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_pipeline", ROOT / "scripts" / "run_pipeline.py"
    )
    run_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_pipeline)
    should_run_tuning = run_pipeline.should_run_tuning

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
