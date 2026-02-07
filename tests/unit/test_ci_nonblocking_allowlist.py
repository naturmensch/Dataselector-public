import re
from pathlib import Path

ALLOWED_NONBLOCKING_WORKFLOWS = {
    ".github/workflows/docs-link-check.yml",
    ".github/workflows/verify-env-usage.yml",
    ".github/workflows/verify-archive.yml",
    ".github/workflows/smoke-tests.yml",
    ".github/workflows/regenerate-lockfile.yml",
}


def test_nonblocking_workflow_allowlist_is_enforced():
    repo_root = Path(__file__).resolve().parents[2]
    workflow_dir = repo_root / ".github" / "workflows"
    pattern = re.compile(r"^\s*continue-on-error:\s*true\s*$", re.MULTILINE)

    offenders = []
    for wf in sorted(workflow_dir.glob("*.yml")):
        if pattern.search(wf.read_text()):
            rel = str(wf.relative_to(repo_root)).replace("\\", "/")
            if rel not in ALLOWED_NONBLOCKING_WORKFLOWS:
                offenders.append(rel)

    assert not offenders, (
        "Found continue-on-error in workflows outside allowlist: "
        f"{', '.join(offenders)}"
    )
