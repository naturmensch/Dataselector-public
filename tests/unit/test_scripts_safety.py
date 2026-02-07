from __future__ import annotations

import subprocess
from pathlib import Path

from dataselector.tools import check, clean


def test_get_protected_paths_env(monkeypatch):
    monkeypatch.setenv("PROTECTED_PATHS", "data/secret, outputs/custom")
    paths = check.get_protected_paths()
    assert "data/images" in paths
    assert "data/secret" in paths
    assert "outputs/custom" in paths


def test_default_protected_contains_expected():
    paths = check.get_protected_paths()
    expected = [
        "data/images",
        "data/archive",
        "models",
        "outputs/final_selection",
        "outputs/kdr100_selection",
    ]
    for item in expected:
        assert item in paths


def test_offending_files():
    protected = ["data/images", "outputs/special"]
    staged = [
        "data/images/0001.png",
        "README.md",
        "dataselector/foo.py",
        "outputs/special/report.csv",
    ]
    offenders = check.offending_files(staged, protected)
    assert "data/images/0001.png" in offenders
    assert "outputs/special/report.csv" in offenders


def test_cli_staged_override():
    offenders = check.offending_files(["data/nope.png"], ["data/images"])
    assert offenders == []


def test_check_all_tracked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "images" / "sample.png").write_text("x")
    (tmp_path / "README.md").write_text("hi")

    subprocess.check_call(["git", "init"], cwd=str(tmp_path))
    subprocess.check_call(["git", "add", "."], cwd=str(tmp_path))
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=str(tmp_path),
    )

    monkeypatch.setenv("PROTECTED_PATHS", "data/images")
    tracked = check.git_tracked_files()
    offenders = check.offending_files(tracked, check.get_protected_paths())
    assert any("data/images" in f for f in offenders)


def test_is_protected():
    assert clean.is_protected(clean.ROOT / "data" / "images")
    assert not clean.is_protected(clean.ROOT / "outputs" / "validation")


def test_clean_workspace_dry_run_no_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "ROOT", tmp_path)
    (tmp_path / "outputs" / "validation").mkdir(parents=True)
    rc = clean.clean_workspace(delete_outputs=True, yes=False)
    assert rc == 0
    assert (tmp_path / "outputs" / "validation").exists()


def test_clean_workspace_delete_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "ROOT", tmp_path)
    (tmp_path / "outputs" / "validation").mkdir(parents=True)
    (tmp_path / "data" / "images").mkdir(parents=True)
    rc = clean.clean_workspace(delete_outputs=True, yes=True)
    assert rc == 0
    assert not (tmp_path / "outputs" / "validation").exists()
    assert (tmp_path / "data" / "images").exists()
