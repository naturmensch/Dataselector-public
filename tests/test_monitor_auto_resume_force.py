import sys

import pytest

import scripts.xxl_full_run_monitor as monitor


def test_auto_resume_force_success(monkeypatch, tmp_path):
    called = {}

    def fake_resume(run_selector, active_log, force=False, dry_run=False):
        called["args"] = (run_selector, str(active_log), force, dry_run)
        return {"ok": True}

    monkeypatch.setattr(monitor, "_resume_run", fake_resume)
    # point ROOT to temp dir to avoid touching real outputs
    monkeypatch.setattr(monitor, "ROOT", tmp_path)

    # Provide minimal args
    monkeypatch.setattr(sys, "argv", ["xxl_full_run_monitor.py", "--auto-resume-force"])

    with pytest.raises(SystemExit) as se:
        monitor.main()
    assert se.value.code == 0
    assert "args" in called
    assert called["args"][0] == "last"
    assert called["args"][2] is True


def test_auto_resume_force_failure(monkeypatch, tmp_path):
    called = {}

    def fake_resume(run_selector, active_log, force=False, dry_run=False):
        called["args"] = (run_selector, str(active_log), force, dry_run)
        return {"ok": False, "reason": "no_db"}

    monkeypatch.setattr(monitor, "_resume_run", fake_resume)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["xxl_full_run_monitor.py", "--auto-resume-force"])

    with pytest.raises(SystemExit) as se:
        monitor.main()
    assert se.value.code == 1
    assert "args" in called
    assert called["args"][2] is True
