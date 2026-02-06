import dataselector.workflows.xxl as mod


def test_xxl_limited_instrumented_run_importable():
    assert hasattr(mod, "main")
