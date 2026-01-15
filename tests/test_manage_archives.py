import tarfile
from pathlib import Path
import tempfile
import os

from scripts.manage_archives import archive_outputs, list_archives, restore_archive


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_archive_with_exclude(tmp_path):
    # prepare fake outputs
    out = tmp_path / "outputs"
    (out / "keep" / "a.txt").parent.mkdir(parents=True)
    (out / "keep" / "a.txt").write_text("keep")
    (out / "final_selection" / "final.csv").parent.mkdir(parents=True)
    (out / "final_selection" / "final.csv").write_text("final")
    (out / "tuning_weights" / "run.json").parent.mkdir(parents=True)
    (out / "tuning_weights" / "run.json").write_text("run")

    dst = tmp_path / "archive"
    # exclude final_selection and tuning_weights
    archive_path = archive_outputs(out, dst, exclude=["final_selection/*", "tuning_weights/*"]) 

    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
        # ensure excluded not present
        assert not any("final_selection" in n for n in names)
        assert not any("tuning_weights" in n for n in names)
        # ensure keep present
        assert any("keep/a.txt" in n for n in names)


def test_list_and_restore(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "x.txt").write_text("x")
    dst = tmp_path / "archive"
    archive_path = archive_outputs(out, dst)
    listed = list_archives(dst)
    assert archive_path in listed
    # restore into new dir
    target = tmp_path / "restore"
    restore_archive(archive_path, target)
    assert (target / "outputs" / "x.txt").exists()
