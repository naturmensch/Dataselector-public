import os
import subprocess
import sys
from pathlib import Path

OUT = Path("outputs")


def test_generate_reports_creates_report(tmp_path):
    cmd = [sys.executable, "-m", "dataselector", "generate-monitor"]
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == 0, f"Script failed: {result.stdout}"

    monitor_reports = OUT / "monitor_reports"
    assert monitor_reports.exists(), "monitor_reports directory not created"
    report = monitor_reports / "monitor_report.md"
    assert report.exists(), f"Report {report} not created"

    # At least one timestamped report should exist
    dated_reports = list(monitor_reports.glob("monitor_report_*.md"))
    assert dated_reports
