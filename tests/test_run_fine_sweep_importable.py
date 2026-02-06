import dataselector.workflows.fine_sweep as mod


def test_run_fine_sweep_importable():
    assert hasattr(mod, "main")
