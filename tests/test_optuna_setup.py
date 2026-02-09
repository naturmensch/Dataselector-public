import os
import subprocess
import sys

import numpy as np
import pandas as pd


def test_optuna_command_runs(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = data_dir / "new_all_tiles.csv"
    n_rows = 20
    metadata = pd.DataFrame(
        {
            "ul_x": np.linspace(500000.0, 500950.0, n_rows),
            "ul_y": np.linspace(5901000.0, 5901950.0, n_rows),
            "lr_x": np.linspace(500050.0, 501000.0, n_rows),
            "lr_y": np.linspace(5900950.0, 5901900.0, n_rows),
            "year": np.arange(1900, 1900 + n_rows),
            "image_path": [f"dummy_{i}.png" for i in range(n_rows)],
            "image_filename": [f"dummy_{i}.png" for i in range(n_rows)],
        }
    )
    metadata.to_csv(metadata_csv, index=False)

    # Provide legacy cache with matching row count so smoke CLI avoids real extraction.
    np.save(out_dir / "features.npy", np.random.RandomState(7).randn(n_rows, 32))

    # Run package command with smoke settings in isolated canonical workspace.
    cmd = [
        sys.executable,
        "-m",
        "dataselector",
        "optuna-optimize",
        "--smoke",
        "--n-trials",
        "1",
        "--n-candidates",
        "20",
        "--dim",
        "32",
        "--n-samples",
        "5",
    ]

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )

    # Check return code and output file
    assert result.returncode == 0, f"optuna command failed: {result.stdout}"
    assert (out_dir / "optuna_results.csv").exists()
