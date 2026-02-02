import json
import time
from pathlib import Path

from dataselector.pipeline.experiment_manager import ExperimentManager


def test_heartbeat_writes_manifest(tmp_path: Path):
    em = ExperimentManager(
        name="hbtest",
        description="heartbeat test",
        base_dir=tmp_path,
        capture_provenance=False,
    )
    # start a fast heartbeat
    em.start_heartbeat(interval_seconds=1)
    # wait to allow at least one heartbeat
    time.sleep(2.5)
    manifest = em.run_dir / "manifest.json"
    assert manifest.exists(), "manifest should be created by heartbeat"
    with open(manifest, "r") as f:
        data = json.load(f)
    assert data["experiment"]["status"] in ("running", "initialized")
    # stop heartbeat and finalize
    em.stop_heartbeat()
    em.mark_complete(success=True, summary="heartbeat test complete")
    with open(manifest, "r") as f:
        data2 = json.load(f)
    assert data2["experiment"]["status"] == "complete"
    assert "completion_time" in data2["experiment"]
