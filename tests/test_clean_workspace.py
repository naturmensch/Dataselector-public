import sys

import scripts.clean_workspace as cw


def test_images_are_protected(tmp_path, monkeypatch, capsys):
    # Create fake candidates in a temporary workspace
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "outputs" / "validation").mkdir(parents=True)
    (tmp_path / ".venv").mkdir(parents=True)

    # Override CANDIDATES to point to our temp dirs
    monkeypatch.setattr(
        cw,
        "CANDIDATES",
        {
            "outputs/validation": "outputs/validation",
            "data/images": "data/images",
            ".venv": ".venv",
        },
    )

    # Run dry-run and capture output
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    sys.argv = ["clean_workspace.py", "--dry-run"]
    cw.main()
    captured = capsys.readouterr()

    assert "data/images" in captured.out
    assert "PROTECTED" in captured.out


def test_delete_outputs_skips_protected(tmp_path, monkeypatch):
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

    # Perform delete outputs
    sys.argv = ["clean_workspace.py", "--delete-outputs"]
    cw.main()

    # images should still exist, outputs/validation should be removed
    assert (tmp_path / "data" / "images").exists()
    assert not (tmp_path / "outputs" / "validation").exists()
