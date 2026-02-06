import dataselector.workflows.optuna_optimize as mod


def test_optuna_optimize_importable():
    assert hasattr(mod, "run_optuna")
