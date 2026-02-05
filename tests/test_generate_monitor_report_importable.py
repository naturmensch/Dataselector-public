def test_generate_monitor_report_importable():
    from dataselector.workflows import generate_reports

    assert hasattr(generate_reports, "generate_monitor_report")
