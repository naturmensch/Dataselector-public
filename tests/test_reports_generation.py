import json
import os
import subprocess
import sys
from pathlib import Path

from dataselector.workflows import generate_reports


def _write_thesis_run(run_dir: Path, *, command: str = "thesis-orchestrate") -> None:
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "monitor").mkdir(parents=True, exist_ok=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-22T12:00:00+00:00",
                "command": ["python", "-m", "dataselector", command],
                "execution_profile": "thesis_repro",
                "extra": {
                    "n_samples": 30,
                    "n_trials": 370,
                    "cache_mode": "read_write",
                    "parameter_source": "snapshot:final_config.yaml",
                    "phase_status": {
                        "phase1_exploration": "success",
                        "phase2_optimization": "success",
                        "phase3_validation": "success",
                        "phase4_summary": "success",
                    },
                    "resolution_only": False,
                    "dry_run": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "THESIS_PIPELINE_REPORT.md").write_text(
        "# Thesis Pipeline Summary Report\n\n## Validation\n\n- OK\n",
        encoding="utf-8",
    )
    (run_dir / "logs" / "status.log").write_text(
        "phase1_exploration=success\nphase4_summary=success\n",
        encoding="utf-8",
    )
    (run_dir / "monitor" / "summary.json").write_text(
        json.dumps({"status": "ok", "phases": 4}, indent=2),
        encoding="utf-8",
    )


def test_generate_monitor_cli_creates_report_for_explicit_run_dir(tmp_path):
    run_dir = tmp_path / "outputs" / "runs" / "thesis_orchestrate_20260322T120000Z"
    _write_thesis_run(run_dir)

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dataselector",
            "generate-monitor",
            "--run-dir",
            str(run_dir),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stdout}"

    monitor_reports = run_dir / "monitor_reports"
    assert monitor_reports.exists(), "monitor_reports directory not created"
    report = monitor_reports / "monitor_report.md"
    assert report.exists(), f"Report {report} not created"
    dated_reports = list(monitor_reports.glob("monitor_report_*.md"))
    assert dated_reports
    assert "Thesis Run" in report.read_text(encoding="utf-8")


def test_generate_monitor_autodiscovers_latest_canonical_thesis_run(
    tmp_path, monkeypatch
):
    run_a = tmp_path / "outputs" / "runs" / "thesis_orchestrate_20260321T120000Z"
    run_b = tmp_path / "outputs" / "runs" / "thesis_pipeline_20260322T120000Z"
    skipped = tmp_path / "outputs" / "runs" / "thesis_debug_resolution_only"

    _write_thesis_run(run_a, command="thesis-orchestrate")
    _write_thesis_run(run_b, command="thesis-pipeline")
    _write_thesis_run(skipped, command="thesis-pipeline")

    skipped_metadata = json.loads((skipped / "run_metadata.json").read_text())
    skipped_metadata["extra"]["resolution_only"] = True
    (skipped / "run_metadata.json").write_text(
        json.dumps(skipped_metadata, indent=2), encoding="utf-8"
    )

    monkeypatch.setattr(generate_reports, "_get_repo_root", lambda: tmp_path)
    report_path = generate_reports.generate_monitor_report()

    assert report_path.parent == run_b / "monitor_reports"
    assert report_path.exists()
    assert "thesis-pipeline" in report_path.read_text(encoding="utf-8")
