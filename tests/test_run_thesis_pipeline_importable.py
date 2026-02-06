import dataselector.workflows.thesis_pipeline as mod


def test_run_thesis_pipeline_importable():
    assert hasattr(mod, "main")
