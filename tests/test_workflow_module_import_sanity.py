import importlib
import inspect

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
        ("dataselector.workflows.validation", "validate_pareto_candidates"),
        ("dataselector.workflows.generate_reports", "generate_monitor_report"),
        ("dataselector.workflows.generate_reports", "generate_thesis_final_report"),
        ("dataselector.workflows.compare_samplers", "compare_multi_seed"),
        ("dataselector.workflows.compare_samplers", "run_single_optuna"),
        ("dataselector.workflows.compare_samplers", "compare_seeded_vs_unseeded"),
        ("dataselector.workflows.compare_samplers", "benchmark_seed"),
        ("dataselector.workflows.bootstrap", "run_bootstrap_pareto"),
        ("dataselector.workflows.bootstrap", "run_bootstrap_final"),
        ("dataselector.workflows.bootstrap", "bootstrap_candidate"),
        ("dataselector.workflows.bootstrap", "bootstrap_selection"),
        ("dataselector.workflows.bootstrap", "summarize_bootstrap"),
        ("dataselector.workflows.bootstrap", "jaccard"),
        ("dataselector.workflows.width_calibration", "sync_width_calibration_source"),
        ("dataselector.workflows.width_calibration", "prepare_width_calibration"),
        ("dataselector.workflows.width_calibration", "measure_width_calibration"),
        ("dataselector.workflows.width_calibration", "summarize_width_calibration"),
        (
            "dataselector.workflows.width_calibration",
            "audit_width_calibration_sensitivity",
        ),
        (
            "dataselector.workflows.width_calibration",
            "build_width_calibration_roads_source",
        ),
        (
            "dataselector.workflows.width_calibration",
            "render_width_calibration_debug_masks",
        ),
    ],
)
def test_workflow_module_import_sanity(module_name: str, symbol: str):
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name} missing expected symbol '{symbol}'"


@pytest.mark.fast
@pytest.mark.unit
def test_bootstrap_workflow_signatures():
    bootstrap = importlib.import_module("dataselector.workflows.bootstrap")

    pareto_sig = inspect.signature(bootstrap.run_bootstrap_pareto)
    assert {"pareto_csv", "n_boot", "output_csv", "random_seed"} <= set(
        pareto_sig.parameters
    )

    final_sig = inspect.signature(bootstrap.run_bootstrap_final)
    assert {"run_dir", "n_boot", "seed"} <= set(final_sig.parameters)
