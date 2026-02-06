import dataselector.workflows.adaptive_pipeline as mod


def test_run_adaptive_pipeline_importable():
    assert hasattr(mod, "main")
