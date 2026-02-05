def test_compare_samplers_importable():
    from dataselector.workflows import compare_samplers

    assert hasattr(compare_samplers, "compare_multi_seed")
    assert hasattr(compare_samplers, "run_single_optuna")
    assert hasattr(compare_samplers, "compare_seeded_vs_unseeded")
    assert hasattr(compare_samplers, "benchmark_seed")
