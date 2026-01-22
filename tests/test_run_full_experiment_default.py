import subprocess


def test_run_full_experiment_shows_adaptive_default():
    result = subprocess.run(
        ["bash", "scripts/run_full_experiment.sh", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "--adaptive" in out
    assert "RECOMMENDED (DEFAULT)" in out or "DEFAULT" in out
