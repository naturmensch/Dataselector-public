from pathlib import Path

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
aw = load_script(
    ROOT / "scripts" / "archive_workspace.py",
    module_name="scripts.archive_workspace_test",
)
cp = load_script(
    ROOT / "scripts" / "check_protected.py", module_name="scripts.check_protected_test"
)


def test_images_are_protected():
    staged = [
        "data/images/a.jpg",
        "outputs/validation/run.txt",
    ]
    protected = cp.get_protected_paths()
    offenders = cp.offending_files(staged, protected)

    assert "data/images/a.jpg" in offenders
    assert "outputs/validation/run.txt" not in offenders


def test_delete_outputs_skips_protected(tmp_path, monkeypatch):
    monkeypatch.setattr(aw, "ROOT", tmp_path)

    images = tmp_path / "data" / "images"
    out_validation = tmp_path / "outputs" / "validation"
    images.mkdir(parents=True, exist_ok=True)
    out_validation.mkdir(parents=True, exist_ok=True)

    assert aw.is_whitelisted(images) is True
    assert aw.is_whitelisted(out_validation) is False
