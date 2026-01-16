import os
import signal
import time
import subprocess


def test_run_full_experiment_detach(tmp_path):
    out_dir = os.path.join(os.getcwd(), "outputs/experiments")
    # ensure outputs dir exists
    os.makedirs(out_dir, exist_ok=True)
    # Run short detached experiment
    cmd = [
        "./scripts/run_full_experiment.sh",
        "--adaptive",
        "--detach",
        "--n-trials",
        "1",
        "--n-candidates",
        "5",
        "--n-boot",
        "1",
        "--yes",
    ]
    proc = subprocess.run(cmd, check=True)

    # Find the most recent pidfile
    pidfiles = sorted([p for p in os.listdir(out_dir) if p.endswith(".pid")])
    assert pidfiles, "No pidfile created"
    latest = pidfiles[-1]
    pidfile = os.path.join(out_dir, latest)
    with open(pidfile) as f:
        pid = int(f.read().strip())

    # process should exist
    assert pid > 0
    time.sleep(1)
    assert os.path.exists(pidfile)

    # cleanup: kill process
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(0.5)
    # remove pidfile for cleanup
    try:
        os.remove(pidfile)
    except OSError:
        pass

    # also check a log file exists
    logfiles = sorted([p for p in os.listdir(out_dir) if p.endswith(".log")])
    assert logfiles, "No log file created"
