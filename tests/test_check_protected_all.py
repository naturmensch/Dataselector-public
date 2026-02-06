import dataselector.tools.check as cp


def test_check_all_tracked(tmp_path, monkeypatch):
    # Setup temporary git repo
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    # Create a tracked file inside images so it appears in `git ls-files`
    (tmp_path / "data" / "images" / "sample.png").write_text("x")
    (tmp_path / "README.md").write_text("hi")

    # init git, add files
    import subprocess

    subprocess.check_call(["git", "init"])
    subprocess.check_call(["git", "add", "."])

    # override PROTECTED to a small set for test
    monkeypatch.setenv("PROTECTED_PATHS", "data/images")

    # Run check_protected with --all, expect non-zero exit code when offending files exist
    staged = cp.git_tracked_files()
    offenders = cp.offending_files(staged, cp.get_protected_paths())
    assert any("data/images" in f for f in offenders)
