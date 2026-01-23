from src.experiment_manager import ExperimentManager


def test_experiment_manager_no_provenance(tmp_path):
    # Use a temporary base dir to avoid touching repository outputs/
    em = ExperimentManager(name="test_em", base_dir=tmp_path, capture_provenance=False)

    # On init with capture_provenance=False, manifest should contain an empty provenance dict
    assert em.manifest["provenance"] == {}

    # Calling capture_provenance should populate the manifest with provenance info
    prov = em.capture_provenance()
    assert isinstance(prov, dict)
    assert "packages" in prov
    assert em.manifest["provenance"] == prov
