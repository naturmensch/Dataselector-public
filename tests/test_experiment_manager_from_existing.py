import json

import pandas as pd
import pytest

from dataselector.pipeline.experiment_manager import ExperimentManager

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def skip_if_no_numba():
    pytest.importorskip("numba", exc_type=ImportError)


def test_from_existing_loads_manifest(tmp_path):
    # Arrange: create a fake run directory with a manifest
    run_dir = tmp_path / "20260101_T000000_testrun"
    run_dir.mkdir(parents=True)
    (run_dir / "config").mkdir()
    (run_dir / "results").mkdir()
    manifest = {
        "experiment": {
            "name": "testrun",
            "description": "unit test",
            "run_id": run_dir.name,
            "timestamp_utc": "20260101_T000000",
            "status": "attached",
            "stages": {},
        },
        "provenance": {"python_version": "3.11.0"},
        "metadata": {"seed": 123},
        "results": {},
        "artifacts": [],
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Act
    em = ExperimentManager.from_existing(run_dir)

    # Assert
    assert em.run_dir.resolve() == run_dir.resolve()
    assert em.manifest["experiment"]["run_id"] == run_dir.name
    assert em.manifest["metadata"]["seed"] == 123


def test_from_existing_can_save_results(tmp_path):
    _run_dir = tmp_path / "20260101_T000001_testrun2"
    em_orig = ExperimentManager(
        name="testrun2", description="original", base_dir=tmp_path
    )
    # Ensure a manifest exists
    em_orig.save_manifest()

    # Attach to it
    em = ExperimentManager.from_existing(em_orig.run_dir)
    df = pd.DataFrame({"a": [1, 2, 3]})
    em.save_results("sample_df", df, format="csv")

    out_path = em.run_dir / "results" / "sample_df.csv"
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert list(loaded["a"]) == [1, 2, 3]
