import dataselector.tools.check as cp


def test_get_protected_paths_env(monkeypatch):
    monkeypatch.setenv("PROTECTED_PATHS", "data/secret, outputs/custom")
    p = cp.get_protected_paths()
    assert "data/images" in p
    assert "data/secret" in p
    assert "outputs/custom" in p


def test_default_protected_contains_expected():
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
    prot = ["data/images"]
    staged = ["data/nope.png"]
    off = cp.offending_files(staged, prot)
    assert off == []
