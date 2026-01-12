import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os

OUT = Path('outputs')

def test_generate_reports_creates_report(tmp_path):
    cmd = [sys.executable, 'scripts/generate_reports.py']
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, f"Script failed: {result.stdout}"

    date = datetime.now().strftime('%Y%m%d')
    report = OUT / f'report_{date}.md'
    assert report.exists(), f"Report {report} not created"

    # At least one file with date suffix should exist (report or png)
    files_with_date = list(OUT.glob(f"*_{date}.*"))
    assert len(files_with_date) >= 1
