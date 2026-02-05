def test_generate_thesis_final_report_importable():
    from dataselector.workflows import generate_reports

    assert hasattr(generate_reports, "generate_thesis_final_report")
