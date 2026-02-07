import importlib

import pytest


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("dataselector.workflows.final_selection", "main"),
        ("dataselector.workflows.adaptive_pipeline", "main"),
        ("dataselector.workflows.fine_sweep", "main"),
        ("dataselector.workflows.thesis_pipeline", "main"),
        ("dataselector.workflows.optuna_optimize", "run_optuna"),
        ("dataselector.workflows.tune_weights", "generate_weights"),
        ("dataselector.workflows.autoscale", "main"),
        ("dataselector.workflows.xxl", "main"),
    ],
)
def test_workflow_module_import_sanity(module_name: str, symbol: str):
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name} missing expected symbol '{symbol}'"
