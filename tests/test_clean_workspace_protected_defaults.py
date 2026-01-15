import scripts.check_protected as cp


def test_dry_run_shows_protected(capsys):
    cp.main(["--list"])
    captured = capsys.readouterr()
    out = captured.out

    assert "data/images" in out
    assert "data/archive" in out
    assert "outputs/final_selection" in out
