import json
from pathlib import Path

from tests._helpers.load_script import load_script

ROOT = Path(__file__).resolve().parents[1]
aw = load_script(
    ROOT / "scripts" / "archive_workspace.py",
    module_name="scripts.archive_workspace_test",
)


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_archive_outputs_with_manifest(tmp_path, monkeypatch):
    root = tmp_path
    outputs = root / "outputs"
    keep = outputs / "keep" / "a.txt"
    final_sel = outputs / "final_selection" / "final.csv"
    tw_run = outputs / "tuning_weights" / "run.json"
    touch(keep)
    touch(final_sel)
    touch(tw_run)

    monkeypatch.setattr(aw, "ROOT", root)
    archive_dir = root / "archive_local"
    monkeypatch.setattr(aw, "ARCHIVE_DIR", archive_dir)

    monkeypatch.setattr(
        aw,
        "WHITELIST_PATTERNS",
        set(aw.WHITELIST_PATTERNS) | {"outputs/tuning_weights", "tuning_weights"},
    )

    cat = aw.ArchiveCategory("old_outputs", "Test outputs")
    cat.add_directory(outputs)

    archived_count = aw.archive_category(cat, dry_run=False)

    assert archived_count >= 1
    assert archive_dir.exists()

    subdirs = sorted([p for p in archive_dir.iterdir() if p.is_dir()])
    assert subdirs, "expected an archive subdir"
    latest = subdirs[-1]
    manifest = latest / "old_outputs_manifest.json"
    assert manifest.exists()

    data = json.loads(manifest.read_text())
    archived_files = {f["path"] for f in data["files"]}

    assert "outputs/keep/a.txt" in archived_files
    assert not any("final_selection" in p for p in archived_files)
    assert not any("tuning_weights" in p for p in archived_files)
