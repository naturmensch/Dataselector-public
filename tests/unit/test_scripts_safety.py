import os
import subprocess
import sys
from pathlib import Path
import pytest
from tests.utils import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_get_protected_paths_env(monkeypatch):
    cp = load_module_from_path("cp", REPO_ROOT / "scripts" / "check_protected.py")
    monkeypatch.setenv("PROTECTED_PATHS", "data/secret, outputs/custom")
    p = cp.get_protected_paths()
    assert "data/images" in p
    assert "data/secret" in p
    assert "outputs/custom" in p


def test_default_protected_contains_expected():
    cp = load_module_from_path("cp2", REPO_ROOT / "scripts" / "check_protected.py")
    p = cp.get_protected_paths()
    expected = [
        "data/images",
        "data/archive",
        "models",
        "outputs/final_selection",
        "outputs/kdr100_selection",
    ]
    for e in expected:
        assert e in p, f"Expected default protected path {e} to be present"


def test_offending_files():
    cp = load_module_from_path("cp3", REPO_ROOT / "scripts" / "check_protected.py")
    prot = ["data/images", "outputs/special"]
    staged = [
        "data/images/0001.png",
        "README.md",
        "src/foo.py",
        "outputs/special/report.csv",
    ]
    off = cp.offending_files(staged, prot)
    assert "data/images/0001.png" in off
    assert "outputs/special/report.csv" in off


def test_cli_staged_override():
    cp = load_module_from_path("cp4", REPO_ROOT / "scripts" / "check_protected.py")
    prot = ["data/images"]
    staged = ["data/nope.png"]
    off = cp.offending_files(staged, prot)
    assert off == []


def test_check_all_tracked(tmp_path, monkeypatch):
    cp = load_module_from_path("cp5", REPO_ROOT / "scripts" / "check_protected.py")
    # Setup temporary git repo
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "images" / "sample.png").write_text("x")
    (tmp_path / "README.md").write_text("hi")

    subprocess.check_call(["git", "init"], cwd=str(tmp_path))
    subprocess.check_call(["git", "add", "."], cwd=str(tmp_path))
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=str(tmp_path))

    monkeypatch.setenv("PROTECTED_PATHS", "data/images")

    staged = cp.git_tracked_files()
    offenders = cp.offending_files(staged, cp.get_protected_paths())
    assert any("data/images" in f for f in offenders)


# clean_workspace related tests

def test_dry_run_shows_protected(tmp_path, capsys, monkeypatch):
    cw = load_module_from_path("cw", REPO_ROOT / "scripts" / "clean_workspace.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "archive").mkdir(parents=True)
    (tmp_path / "outputs" / "final_selection").mkdir(parents=True)

    sys_argv = sys.argv
    try:
        sys.argv = ["clean_workspace.py", "--dry-run"]
        cw.main()
        captured = capsys.readouterr()
        assert "data/images" in captured.out
        assert "data/archive" in captured.out
        assert "outputs/final_selection" in captured.out
        assert "PROTECTED" in captured.out
    finally:
        sys.argv = sys_argv


def test_images_are_protected(tmp_path, monkeypatch, capsys):
    cw = load_module_from_path("cw2", REPO_ROOT / "scripts" / "clean_workspace.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "outputs" / "validation").mkdir(parents=True)
    (tmp_path / ".venv").mkdir(parents=True)

    monkeypatch.setattr(
        cw,
        "CANDIDATES",
        {
            "outputs/validation": "outputs/validation",
            "data/images": "data/images",
            ".venv": ".venv",
        },
    )

    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    sys_argv = sys.argv
    try:
        sys.argv = ["clean_workspace.py", "--dry-run"]
        cw.main()
        captured = capsys.readouterr()
        assert "data/images" in captured.out
        assert "PROTECTED" in captured.out
    finally:
        sys.argv = sys_argv


def test_delete_outputs_skips_protected(tmp_path, monkeypatch):
    cw = load_module_from_path("cw3", REPO_ROOT / "scripts" / "clean_workspace.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "outputs" / "validation").mkdir(parents=True)

    monkeypatch.setattr(
        cw,
        "CANDIDATES",
        {
            "outputs/validation": "outputs/validation",
            "data/images": "data/images",
        },
    )

    sys_argv = sys.argv
    try:
        sys.argv = ["clean_workspace.py", "--delete-outputs"]
        cw.main()
        assert (tmp_path / "data" / "images").exists()
        assert not (tmp_path / "outputs" / "validation").exists()
    finally:
        sys.argv = sys_argv
