import dataselector.workflows.tune_weights as mod


def test_tune_weights_and_run_importable():
    assert hasattr(mod, "generate_weights")
