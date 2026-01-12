import sys

import scripts.clean_workspace as cw


def test_dry_run_shows_protected(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Create some of the default protected folders
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "archive").mkdir(parents=True)
    (tmp_path / "outputs" / "final_selection").mkdir(parents=True)

    sys.argv = ["clean_workspace.py", "--dry-run"]
    cw.main()
    captured = capsys.readouterr()

    assert "data/images" in captured.out
    assert "data/archive" in captured.out
    assert "outputs/final_selection" in captured.out
    assert "PROTECTED" in captured.out
